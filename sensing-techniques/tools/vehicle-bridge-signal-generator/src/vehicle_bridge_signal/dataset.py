"""HDF5 datasets: generation (optionally parallel), reading and leak-free splits.

File layout (schema 1.x)
------------------------
::

    /time                       (T,)            float64  [s]   output time vector (t = 0 at window start)
    /measured                   (N, T)          float32  [m/s^2] smartphone signal (network input)
    /components/<key>           (N, T)          float32  [m/s^2] bridge_mode_1..K, bridge_higher_modes,
                                                                 roughness, engine, wheel, noise
    /contact/<key>              (N, A, T)       float32  [m/s^2] contact-point acceleration per axle
                                                                 (same keys; 'total' instead of 'noise')
    /bridge/midspan_displacement (N, T)         float32  [m]
    /bridge/midspan_acceleration (N, T)         float32  [m/s^2]
    /on_bridge_mask             (N, T)          uint8    1 while any axle is on the bridge
    /on_bridge_mask_axles       (N, A, T)       uint8    per axle
    /metadata/<field>           (N,) or (N, M)  per-sample table (NaN-padded for per-mode arrays)

Root attributes: ``schema_version``, ``config_yaml`` (full resolved configuration), ``base_seed``,
``git_commit``, ``git_dirty``, ``package_version``, ``created_utc``, ``component_keys``,
``contact_keys``, ``fs_out``, ``n_samples_out``, ``n_axles``, ... Every array is chunked one
sample per chunk so random access by sample is cheap.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as _dt
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from .config import TOOL_ROOT, ConfigError, config_to_yaml, distribution_paths, get_path, load_config, yaml_load
from .sampling import SamplePlan, plan_dataset, scenario_for
from .simulate import component_keys, contact_keys, simulate

__all__ = ["SCHEMA_VERSION", "generate_dataset", "load_dataset", "VBIDataset", "group_split", "regenerate_sample",
           "git_info"]

SCHEMA_VERSION = "1.0.0"
_N_AXLES = {"sprung_mass": 1, "quarter_car": 1, "half_car": 2}


# --------------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------------
def git_info() -> tuple[str, bool]:
    """(commit hash, dirty flag) of the repository that contains the tool, or ('unknown', False)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TOOL_ROOT, capture_output=True, text=True,
                                timeout=10).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=TOOL_ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip()
        return (commit or "unknown"), bool(status)
    except (OSError, subprocess.SubprocessError):
        return "unknown", False


# --------------------------------------------------------------------------------------------
# One sample (runs inside worker processes)
# --------------------------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(cfg: dict, base_seed: int, sampled_paths: list[str]) -> None:
    _WORKER.update(cfg=cfg, base_seed=base_seed, sampled_paths=sampled_paths)


def _run_one(plan: SamplePlan) -> dict:
    cfg, base_seed, sampled = _WORKER["cfg"], _WORKER["base_seed"], _WORKER["sampled_paths"]
    try:
        scenario, info = scenario_for(cfg, plan, base_seed)
        res = simulate(scenario, plan.sample_seed)
    except Exception as exc:  # report which sample failed so it can be reproduced
        raise RuntimeError(f"Sample {plan.index} (bridge_id={plan.bridge_id}, vehicle_id={plan.vehicle_id}, "
                           f"sample_seed={plan.sample_seed}) failed: {type(exc).__name__}: {exc}") from exc
    meta = dict(info)
    for path in sampled:
        meta[f"param.{path}"] = get_path(scenario, path)
    for key, value in res.metadata.items():
        meta[key] = "; ".join(value) if key == "warnings" else value
    return {
        "index": plan.index,
        "measured": res.measured.astype(np.float32),
        "components": {k: v.astype(np.float32) for k, v in res.components.items()},
        "contact": {k: v.astype(np.float32) for k, v in res.contact.items()},
        "midspan_displacement": res.bridge_midspan_displacement.astype(np.float32),
        "midspan_acceleration": res.bridge_midspan_acceleration.astype(np.float32),
        "mask": res.on_bridge.astype(np.uint8),
        "mask_axles": res.on_bridge_axles.astype(np.uint8),
        "metadata": meta,
    }


# --------------------------------------------------------------------------------------------
# Metadata table
# --------------------------------------------------------------------------------------------
def _to_column(values: list[Any], width: int | None = None):
    """Convert a list of per-sample values (with ``None`` for missing) into an array for HDF5.

    List values become a 2-D NaN-padded array of ``width`` columns (or the longest list).
    """
    present = [v for v in values if v is not None]
    if not present:
        return np.full(len(values), np.nan)
    first = present[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        width = max([len(v) for v in present] + [width or 0])
        if all(isinstance(x, str) for v in present for x in v):
            return np.array(["; ".join(map(str, v)) if v is not None else "" for v in values], dtype=object)
        out = np.full((len(values), width), np.nan)
        for i, v in enumerate(values):
            if v is not None and len(v):
                out[i, :len(v)] = np.asarray(v, float)
        return out
    if isinstance(first, (bool, np.bool_)) and all(isinstance(v, (bool, np.bool_)) for v in present):
        if len(present) == len(values):
            return np.array(values, dtype=bool)
        return np.array([np.nan if v is None else float(v) for v in values])
    if isinstance(first, str) or any(isinstance(v, str) for v in present):
        return np.array(["" if v is None else str(v) for v in values], dtype=object)
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in present) and len(present) == len(values):
        return np.array(values, dtype=np.int64)
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def _write_metadata(group: h5py.Group, rows: list[dict], widths: dict[str, int] | None = None) -> None:
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    for key in keys:
        col = _to_column([row.get(key) for row in rows], (widths or {}).get(key))
        if col.dtype == object:
            group.create_dataset(key, data=col.astype(str).astype(object), dtype=h5py.string_dtype())
        else:
            group.create_dataset(key, data=col)


# --------------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------------
def _check_output_path(out_path: Path, overwrite: bool, allow_inside_tool: bool) -> Path:
    out_path = Path(out_path).expanduser().resolve()
    if out_path.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError(f"The output file should end in .h5 or .hdf5, got '{out_path.name}'")
    if not allow_inside_tool:
        try:
            out_path.relative_to(TOOL_ROOT)
            inside = True
        except ValueError:
            inside = False
        if inside:
            raise ValueError(f"Datasets are not stored inside the tool folder ({TOOL_ROOT}). Save them in the data "
                             "folder of the project that uses them (or pass allow_inside_tool=True).")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists (use overwrite=True / --overwrite)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _compression(dcfg: dict) -> dict:
    comp = dcfg["compression"]
    if comp == "none":
        return {}
    if comp == "gzip":
        return {"compression": "gzip", "compression_opts": int(dcfg["compression_level"]), "shuffle": True}
    return {"compression": "lzf", "shuffle": True}


def _iter_results(cfg: dict, plans: list[SamplePlan], base_seed: int, workers: int, sampled: list[str]):
    if workers <= 1:
        _init_worker(cfg, base_seed, sampled)
        for plan in plans:
            yield _run_one(plan)
        return
    ctx = mp.get_context("spawn")
    chunksize = max(1, min(16, len(plans) // (4 * workers)))
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                                initargs=(cfg, base_seed, sampled)) as pool:
        yield from pool.map(_run_one, plans, chunksize=chunksize)


def generate_dataset(config, n_samples: int | None, out_path, seed: int = 0, workers: int = 1, *, overwrite: bool = False,
                     progress: bool = True, allow_inside_tool: bool = False, command: str | None = None) -> Path:
    """Generate ``n_samples`` random crossings and write them to an HDF5 file.

    Parameters
    ----------
    config
        Preset name, YAML path or mapping (distributions are sampled, fixed values stay fixed).
    n_samples
        Number of crossings (``None``: ``dataset.n_samples`` of the configuration).
    out_path
        Destination ``.h5`` file. It is written as ``<out_path>.partial`` and renamed at the end, so an
        interrupted run never leaves a file that looks complete.
    seed
        Base seed; sample ``i`` uses seeds derived from it (see :mod:`vehicle_bridge_signal.sampling`).
    workers
        Number of processes (0 = all CPUs). The result does not depend on it.
    """
    cfg = load_config(config)
    if n_samples is None:
        n_samples = cfg["dataset"]["n_samples"]
        if n_samples is None:
            raise ValueError("n_samples not given and the configuration has no dataset.n_samples")
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    workers = int(workers)
    if workers < 0:
        raise ValueError("workers must be >= 0")
    if workers == 0:
        workers = os.cpu_count() or 1
    out_path = _check_output_path(Path(out_path), overwrite, allow_inside_tool)
    plans = plan_dataset(cfg, n_samples, int(seed))
    sampled = distribution_paths(cfg, relevant_only=True)

    K = int(cfg["bridge"]["n_label_modes"])
    T = int(cfg["simulation"]["n_samples_out"])
    fs_out = float(cfg["simulation"]["fs_out"])
    n_axles = _N_AXLES[cfg["vehicle"]["model"]]
    dcfg = cfg["dataset"]
    comp = _compression(dcfg)
    ckeys, xkeys = component_keys(K), contact_keys(K)
    tmp = out_path.with_name(out_path.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    commit, dirty = git_info()
    from . import __version__

    rows: list[dict] = [None] * n_samples  # type: ignore[list-item]
    t_start = time.time()
    f = h5py.File(tmp, "w")
    try:
        f.attrs.update({
            "schema_version": SCHEMA_VERSION, "generator": "vehicle_bridge_signal", "package_version": __version__,
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit, "git_dirty": dirty, "config_yaml": config_to_yaml(cfg), "base_seed": int(seed),
            "n_samples": n_samples, "fs_out": fs_out, "n_samples_out": T, "n_axles": n_axles, "n_label_modes": K,
            "vehicle_model": cfg["vehicle"]["model"], "interaction": cfg["simulation"]["interaction"],
            "amplitude_mode": cfg["simulation"]["amplitude_mode"], "component_keys": json.dumps(ckeys),
            "contact_keys": json.dumps(xkeys), "sampled_parameters": json.dumps(sampled),
            "command": command or "", "python_version": platform.python_version(),
            "numpy_version": np.__version__, "h5py_version": h5py.__version__,
            "units": json.dumps({"time": "s", "acceleration": "m/s^2", "displacement": "m"}),
        })
        f.create_dataset("time", data=np.arange(T) / fs_out).attrs["units"] = "s"

        def make(name, shape, dtype="float32", units="m/s^2"):
            ds = f.create_dataset(name, shape=shape, dtype=dtype, chunks=(1,) + shape[1:], **comp)
            ds.attrs["units"] = units
            return ds

        d_meas = make("measured", (n_samples, T))
        d_comp = {k: make(f"components/{k}", (n_samples, T)) for k in ckeys}
        d_cont = {k: make(f"contact/{k}", (n_samples, n_axles, T)) for k in xkeys} if dcfg["store_contact"] else {}
        d_bri = {}
        if dcfg["store_bridge_response"]:
            d_bri = {"midspan_displacement": make("bridge/midspan_displacement", (n_samples, T), units="m"),
                     "midspan_acceleration": make("bridge/midspan_acceleration", (n_samples, T))}
        d_mask = make("on_bridge_mask", (n_samples, T), "uint8", "-")
        d_mask_a = make("on_bridge_mask_axles", (n_samples, n_axles, T), "uint8", "-")

        last = 0.0
        for done, res in enumerate(_iter_results(cfg, plans, int(seed), workers, sampled), start=1):
            i = res["index"]
            d_meas[i] = res["measured"]
            for k, ds in d_comp.items():
                ds[i] = res["components"][k]
            for k, ds in d_cont.items():
                ds[i] = res["contact"][k]
            for k, ds in d_bri.items():
                ds[i] = res[k]
            d_mask[i] = res["mask"]
            d_mask_a[i] = res["mask_axles"]
            rows[i] = res["metadata"]
            now = time.time()
            if progress and (now - last > 2.0 or done == n_samples):
                rate = done / max(now - t_start, 1e-9)
                eta = (n_samples - done) / rate if rate > 0 else float("nan")
                print(f"  {done}/{n_samples} samples ({100 * done / n_samples:.0f} %), {rate:.1f} samples/s, "
                      f"ETA {eta:.0f} s", file=sys.stderr, flush=True)
                last = now
        n_modes_max = int(cfg["bridge"]["max_modes"])
        n_dof = {"sprung_mass": 1, "quarter_car": 2, "half_car": 4}[cfg["vehicle"]["model"]]
        widths = {"bridge_frequencies": n_modes_max, "bridge_damping_ratios": n_modes_max,
                  "param.bridge.damping_ratio": n_modes_max, "vehicle_frequencies": n_dof,
                  "vehicle_damping_ratios": n_dof, "engine_orders": 3}
        _write_metadata(f.create_group("metadata"), rows, widths)
        f.attrs["generation_seconds"] = round(time.time() - t_start, 2)
        f.attrs["workers"] = workers
        f.close()
        os.replace(tmp, out_path)
    except BaseException:
        f.close()
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return out_path


# --------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------
class VBIDataset:
    """Lazy reader. Arrays stay on disk until indexed (``ds.measured[0:32]``).

    Use as a context manager or call :meth:`close`.
    """

    def __init__(self, path, mode: str = "r"):
        self.path = Path(path)
        self.file = h5py.File(self.path, mode)
        version = str(self.file.attrs.get("schema_version", "0"))
        major = int(version.split(".")[0])
        if major != int(SCHEMA_VERSION.split(".")[0]):
            self.file.close()
            raise ValueError(f"{path}: schema {version} is not compatible with this reader ({SCHEMA_VERSION})")
        if version != SCHEMA_VERSION:
            warnings.warn(f"{path}: schema {version}, reader {SCHEMA_VERSION}", stacklevel=2)
        self.schema_version = version
        self.component_keys: list[str] = json.loads(self.file.attrs["component_keys"])
        self.contact_keys: list[str] = json.loads(self.file.attrs["contact_keys"])
        self._metadata: dict | None = None

    # ---- context management ----
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.file.id.valid:
            self.file.close()

    # ---- properties ----
    def __len__(self) -> int:
        return int(self.file["measured"].shape[0])

    @property
    def attrs(self) -> dict:
        return dict(self.file.attrs)

    @property
    def config(self) -> dict:
        import yaml
        return yaml_load(self.file.attrs["config_yaml"])

    @property
    def fs(self) -> float:
        return float(self.file.attrs["fs_out"])

    @property
    def time(self) -> np.ndarray:
        return self.file["time"][()]

    @property
    def measured(self) -> h5py.Dataset:
        return self.file["measured"]

    def component(self, key: str) -> h5py.Dataset:
        return self.file[f"components/{key}"]

    def contact(self, key: str) -> h5py.Dataset:
        if "contact" not in self.file:
            raise KeyError("This dataset was generated with store_contact: false")
        return self.file[f"contact/{key}"]

    @property
    def on_bridge_mask(self) -> h5py.Dataset:
        return self.file["on_bridge_mask"]

    @property
    def metadata(self) -> dict[str, np.ndarray]:
        """Per-sample table as a dict of arrays (strings decoded)."""
        if self._metadata is None:
            out = {}
            for k, ds in self.file["metadata"].items():
                arr = ds[()]
                if arr.dtype.kind in ("O", "S"):
                    arr = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr], dtype=object)
                out[k] = arr
            self._metadata = out
        return self._metadata

    def metadata_frame(self):
        """Metadata as a pandas DataFrame (2-D fields expanded as ``name_1, name_2, ...``)."""
        import pandas as pd
        cols = {}
        for k, v in self.metadata.items():
            if v.ndim == 1:
                cols[k] = v
            else:
                for j in range(v.shape[1]):
                    cols[f"{k}_{j + 1}"] = v[:, j]
        return pd.DataFrame(cols)

    def sample(self, i: int) -> dict:
        """All arrays of sample ``i`` (float32) plus its metadata row."""
        out = {"time": self.time, "measured": self.measured[i],
               "components": {k: self.component(k)[i] for k in self.component_keys},
               "on_bridge_mask": self.on_bridge_mask[i].astype(bool)}
        if "contact" in self.file:
            out["contact"] = {k: self.contact(k)[i] for k in self.contact_keys}
        if "bridge" in self.file:
            out["bridge"] = {k: ds[i] for k, ds in self.file["bridge"].items()}
        out["metadata"] = {k: v[i] for k, v in self.metadata.items()}
        return out

    def stack_components(self, indices=None, keys: Iterable[str] | None = None) -> np.ndarray:
        """Array (n, n_keys, T) of the requested components (default: all but ``noise``)."""
        keys = [k for k in self.component_keys if k != "noise"] if keys is None else list(keys)
        idx = slice(None) if indices is None else np.sort(np.asarray(indices))
        return np.stack([self.component(k)[idx] for k in keys], axis=1)

    def __repr__(self) -> str:
        return (f"VBIDataset('{self.path}', n={len(self)}, T={self.file.attrs['n_samples_out']}, "
                f"fs={self.fs} Hz, components={self.component_keys})")


def load_dataset(path) -> VBIDataset:
    """Open a dataset written by :func:`generate_dataset` (lazy; see :class:`VBIDataset`)."""
    return VBIDataset(path)


def regenerate_sample(dataset, index: int, return_internal: bool = False):
    """Re-simulate sample ``index`` of a dataset in float64 (same scenario and seed)."""
    ds = dataset if isinstance(dataset, VBIDataset) else VBIDataset(dataset)
    try:
        cfg = ds.config
        base_seed = int(ds.file.attrs["base_seed"])
        meta = ds.metadata
        plan = SamplePlan(int(index), int(meta["bridge_id"][index]), int(meta["vehicle_id"][index]),
                          int(meta["partition"][index]), int(meta["sample_seed"][index]))
        scenario, _ = scenario_for(cfg, plan, base_seed)
        return simulate(scenario, plan.sample_seed, return_internal=return_internal)
    finally:
        if ds is not dataset:
            ds.close()


# --------------------------------------------------------------------------------------------
# Leak-free splits (optional helper)
# --------------------------------------------------------------------------------------------
def group_split(source, fractions=(0.7, 0.15, 0.15), names=("train", "val", "test"),
                by: tuple[str, ...] = ("bridge_id", "vehicle_id"), seed: int = 0, method: str = "auto") -> dict:
    """Split sample indices so that no bridge (and/or vehicle) appears in two splits.

    method
        ``partitions``: use the ``partition`` column written when the dataset was generated with
        ``dataset.partitions`` (nothing is discarded). ``groups``: assign every group id to a split at
        random with the requested fractions; with two keys a sample is kept only when its bridge and
        its vehicle fall in the same split (about sum(f^2) of the samples survive).
        ``auto``: ``partitions`` if available with the same number of splits, else ``groups``.

    Returns a dict ``{name: indices, ..., 'discarded': indices}``.
    """
    if isinstance(source, (str, os.PathLike)):
        with VBIDataset(source) as ds:
            meta = ds.metadata
    else:
        meta = source.metadata if isinstance(source, VBIDataset) else source
    fractions = np.asarray(fractions, float)
    if fractions.ndim != 1 or len(fractions) != len(names) or np.any(fractions < 0) or fractions.sum() <= 0:
        raise ValueError("fractions must be non-negative, one per split name")
    fractions = fractions / fractions.sum()
    n = len(next(iter(meta.values())))
    part = meta.get("partition")
    n_parts = len(np.unique(part)) if part is not None else 1
    if method == "auto":
        method = "partitions" if part is not None and n_parts == len(names) else "groups"
    if method == "partitions":
        if part is None or n_parts != len(names):
            raise ValueError("The dataset has no partitions matching the requested splits")
        out = {name: np.flatnonzero(part == g) for g, name in enumerate(names)}
        out["discarded"] = np.array([], int)
        return out
    if method != "groups":
        raise ValueError("method must be 'auto', 'partitions' or 'groups'")
    rng = np.random.default_rng(seed)
    assignment = np.zeros((len(by), n), int)
    for r, key in enumerate(by):
        ids = np.asarray(meta[key])
        uniq, counts = np.unique(ids, return_counts=True)
        order = rng.permutation(uniq.size)
        target = np.cumsum(fractions) * n
        split_of = {}
        acc = 0
        for o in order:
            s = int(np.searchsorted(target, acc + 0.5 * counts[o]))
            split_of[uniq[o]] = min(s, len(names) - 1)
            acc += counts[o]
        assignment[r] = [split_of[i] for i in ids]
    same = np.all(assignment == assignment[0], axis=0)
    out = {name: np.flatnonzero(same & (assignment[0] == g)) for g, name in enumerate(names)}
    out["discarded"] = np.flatnonzero(~same)
    return out
