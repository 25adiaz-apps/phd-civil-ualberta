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
    /profile/<key>              (N, A, T)       float32  under each wheel (dataset.store_profile): position [m],
                                                         road_displacement [m], road_acceleration [m/s^2],
                                                         bridge_displacement [m], bridge_acceleration [m/s^2]
    /segment                    (N, T)          uint8    0 before, 1 on, 2 after the bridge (store_profile)
    /metadata/<field>           (N,) or (N, M)  per-sample table (NaN-padded for per-mode arrays)

Layouts (``dataset.layout``, root attribute ``layout``)
    fixed   every sample has T = n_samples_out samples (shapes above).
    padded  T = pad_to_samples; each sample is zero-padded after its ``/length`` samples and
            ``/valid_mask`` (N, T) marks the real ones.
    ragged  no padding: the time axis of every signal is the concatenation of all samples, so
            ``/measured`` is (sum of lengths,) and ``/profile/<key>`` is (A, sum of lengths);
            sample i spans ``offsets[i]:offsets[i + 1]`` (``/offsets`` has N + 1 entries).
    :meth:`VBIDataset.signal` and :meth:`VBIDataset.sample` return the unpadded arrays of one sample
    for any layout. :func:`derive_dataset` crops the road before/after the bridge and changes the
    layout without simulating again.

Root attributes: ``schema_version``, ``config_yaml`` (full resolved configuration), ``base_seed``,
``git_commit``, ``git_dirty``, ``package_version``, ``created_utc``, ``component_keys``,
``contact_keys``, ``fs_out``, ``n_samples_out``, ``n_axles``, ... Every array is chunked one
sample per chunk so random access by sample is cheap.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as _dt
import json
import math
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

from .config import TOOL_ROOT, config_to_yaml, distribution_paths, get_path, load_config, yaml_load
from .sampling import SamplePlan, plan_dataset, scenario_for
from .simulate import PROFILE_UNITS, component_keys, contact_keys, simulate

__all__ = ["SCHEMA_VERSION", "generate_dataset", "derive_dataset", "load_dataset", "VBIDataset", "group_split",
           "regenerate_sample", "git_info"]

SCHEMA_VERSION = "1.1.0"   # 1.1: layouts padded/ragged, profile, segment, signals attribute
_RAGGED_CHUNK = 1 << 16
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
    _WORKER.update(cfg=cfg, base_seed=base_seed, sampled_paths=sampled_paths, specs=_signal_specs(cfg))


def _signal_specs(cfg: dict) -> dict[str, dict]:
    """Per-sample signals written for a configuration: {hdf5 name: {axles, dtype, units}}."""
    dcfg = cfg["dataset"]
    K = int(cfg["bridge"]["n_label_modes"])
    specs = {"measured": dict(axles=False, dtype="float32", units="m/s^2")}
    for k in component_keys(K):
        specs[f"components/{k}"] = dict(axles=False, dtype="float32", units="m/s^2")
    if dcfg["store_contact"]:
        for k in contact_keys(K):
            specs[f"contact/{k}"] = dict(axles=True, dtype="float32", units="m/s^2")
    if dcfg["store_bridge_response"]:
        specs["bridge/midspan_displacement"] = dict(axles=False, dtype="float32", units="m")
        specs["bridge/midspan_acceleration"] = dict(axles=False, dtype="float32", units="m/s^2")
    specs["on_bridge_mask"] = dict(axles=False, dtype="uint8", units="-")
    specs["on_bridge_mask_axles"] = dict(axles=True, dtype="uint8", units="-")
    if dcfg["store_profile"]:
        for k, units in PROFILE_UNITS.items():
            specs[f"profile/{k}"] = dict(axles=True, dtype="float32", units=units)
        specs["segment"] = dict(axles=False, dtype="uint8", units="-")
    return specs


def _signals_of(res, specs: dict) -> dict[str, np.ndarray]:
    arrays = {"measured": res.measured, "on_bridge_mask": res.on_bridge, "on_bridge_mask_axles": res.on_bridge_axles,
              "bridge/midspan_displacement": res.bridge_midspan_displacement,
              "bridge/midspan_acceleration": res.bridge_midspan_acceleration, "segment": res.segment}
    arrays.update({f"components/{k}": v for k, v in res.components.items()})
    arrays.update({f"contact/{k}": v for k, v in res.contact.items()})
    arrays.update({f"profile/{k}": v for k, v in res.profile.items()})
    return {name: np.asarray(arrays[name]).astype(spec["dtype"]) for name, spec in specs.items()}


def _run_one(plan: SamplePlan) -> dict:
    cfg, base_seed, sampled = _WORKER["cfg"], _WORKER["base_seed"], _WORKER["sampled_paths"]
    dcfg = cfg["dataset"]
    resample = dcfg["wheel_lift_off"] == "resample"
    max_redo = int(dcfg["max_lift_off_resamples"]) if resample else 0
    try:
        for redo in range(max_redo + 1):
            scenario, info = scenario_for(cfg, plan, base_seed, redo)
            res = simulate(scenario, info["sample_seed"])
            if not (resample and res.metadata["wheel_lift_off"]):
                break
        else:
            raise RuntimeError(f"wheel lift-off in all {max_redo + 1} draws; raise dataset.max_lift_off_resamples or "
                               "use dataset.wheel_lift_off: keep")
        n_out = res.measured.size
        if dcfg["layout"] == "padded" and n_out > int(dcfg["pad_to_samples"]):
            raise ValueError(f"the crossing needs {n_out} samples, more than dataset.pad_to_samples = "
                             f"{dcfg['pad_to_samples']}")
    except Exception as exc:  # report which sample failed so it can be reproduced
        raise RuntimeError(f"Sample {plan.index} (bridge_id={plan.bridge_id}, vehicle_id={plan.vehicle_id}, "
                           f"sample_seed={plan.sample_seed}) failed: {type(exc).__name__}: {exc}") from exc
    meta = dict(info)
    for path in sampled:
        meta[f"param.{path}"] = get_path(scenario, path)
    for key, value in res.metadata.items():
        meta[key] = "; ".join(value) if key == "warnings" else value
    return {"index": plan.index, "length": int(n_out), "signals": _signals_of(res, _WORKER["specs"]),
            "metadata": meta}


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


class _Writer:
    """Writes per-sample signals in the fixed, padded or ragged layout."""

    def __init__(self, f: h5py.File, layout: str, n: int, T: int | None, n_axles: int, comp: dict):
        self.f, self.layout, self.n, self.T, self.A, self.comp = f, layout, n, T, n_axles, comp
        self.ds: dict[str, h5py.Dataset] = {}
        self.lengths = np.zeros(n, np.int64)
        self.offset = 0
        self.next = 0
        self.valid = None
        if layout != "ragged":
            f.create_dataset("time", data=np.arange(T) / float(f.attrs["fs_out"])).attrs["units"] = "s"
        if layout == "padded":
            self.valid = f.create_dataset("valid_mask", shape=(n, T), dtype="uint8", chunks=(1, T), **comp)
            self.valid.attrs["units"] = "-"

    def create(self, name: str, axles: bool, dtype: str, units: str) -> None:
        lead = (self.A,) if axles else ()
        if self.layout == "ragged":
            ds = self.f.create_dataset(name, shape=lead + (0,), maxshape=lead + (None,), dtype=dtype,
                                       chunks=lead + (_RAGGED_CHUNK,), **self.comp)
        else:
            shape = (self.n,) + lead + (self.T,)
            ds = self.f.create_dataset(name, shape=shape, dtype=dtype, chunks=(1,) + shape[1:], **self.comp)
        ds.attrs["units"] = units
        self.ds[name] = ds

    def put(self, i: int, signals: dict[str, np.ndarray], length: int) -> None:
        if self.layout == "ragged":
            if i != self.next:
                raise RuntimeError("the ragged layout writes the samples in order")
            end = self.offset + length
            for name, ds in self.ds.items():
                ds.resize(end, axis=ds.ndim - 1)
                ds[..., self.offset:end] = signals[name]
            self.offset = end
            self.next += 1
        else:
            if length > self.T or (self.layout == "fixed" and length != self.T):
                raise ValueError(f"sample {i} has {length} samples; the {self.layout} layout holds {self.T}")
            for name, ds in self.ds.items():
                ds[i, ..., :length] = signals[name]
            if self.valid is not None:
                self.valid[i, :length] = 1
        self.lengths[i] = length

    def finish(self) -> None:
        if self.layout != "fixed":
            self.f.create_dataset("length", data=self.lengths).attrs["units"] = "samples"
        if self.layout == "ragged":
            self.f.create_dataset("offsets", data=np.concatenate([[0], np.cumsum(self.lengths)]).astype(np.int64))


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
    fs_out = float(cfg["simulation"]["fs_out"])
    n_axles = _N_AXLES[cfg["vehicle"]["model"]]
    dcfg = cfg["dataset"]
    layout = dcfg["layout"]
    T = {"fixed": cfg["simulation"]["n_samples_out"], "padded": dcfg["pad_to_samples"], "ragged": None}[layout]
    T = None if T is None else int(T)
    comp = _compression(dcfg)
    ckeys, xkeys = component_keys(K), contact_keys(K)
    specs = _signal_specs(cfg)
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
            "n_samples": n_samples, "fs_out": fs_out, "n_samples_out": -1 if T is None else T, "layout": layout,
            "signals": json.dumps(specs), "n_axles": n_axles, "n_label_modes": K,
            "vehicle_model": cfg["vehicle"]["model"], "interaction": cfg["simulation"]["interaction"],
            "amplitude_mode": cfg["simulation"]["amplitude_mode"], "component_keys": json.dumps(ckeys),
            "contact_keys": json.dumps(xkeys), "sampled_parameters": json.dumps(sampled),
            "command": command or "", "python_version": platform.python_version(),
            "numpy_version": np.__version__, "h5py_version": h5py.__version__,
            "units": json.dumps({"time": "s", "acceleration": "m/s^2", "displacement": "m"}),
        })
        writer = _Writer(f, layout, n_samples, T, n_axles, comp)
        for name, spec in specs.items():
            writer.create(name, **spec)

        last = 0.0
        for done, res in enumerate(_iter_results(cfg, plans, int(seed), workers, sampled), start=1):
            i = res["index"]
            writer.put(i, res["signals"], res["length"])
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
        writer.finish()
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
        self.layout = str(self.file.attrs.get("layout", "fixed"))
        self.component_keys: list[str] = json.loads(self.file.attrs["component_keys"])
        self.contact_keys: list[str] = json.loads(self.file.attrs["contact_keys"])
        self._metadata: dict | None = None
        self._offsets = self.file["offsets"][()] if self.layout == "ragged" else None
        self._lengths = self.file["length"][()] if "length" in self.file else None

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
        return int(self.file.attrs["n_samples"])

    @property
    def lengths(self) -> np.ndarray:
        """Number of real (unpadded) samples of every crossing."""
        if self._lengths is None:
            return np.full(len(self), int(self.file.attrs["n_samples_out"]), np.int64)
        return self._lengths

    def signal(self, name: str, i: int) -> np.ndarray:
        """Unpadded array of sample ``i`` for any layout, e.g. ``signal('profile/road_displacement', 3)``.

        Shape (T_i,) or (n_axles, T_i).
        """
        ds = self.file[name]
        i = int(i)
        if not 0 <= i < len(self):
            raise IndexError(f"sample {i} out of range (n={len(self)})")
        if self.layout == "ragged":
            return ds[..., int(self._offsets[i]):int(self._offsets[i + 1])]
        if self.layout == "padded":
            return ds[i, ..., :int(self._lengths[i])]
        return ds[i]

    @property
    def signal_names(self) -> list[str]:
        if "signals" in self.file.attrs:
            return list(json.loads(self.file.attrs["signals"]))
        names = []
        self.file.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) and not n.startswith(
            ("metadata/", "time")) else None)
        return names

    @property
    def attrs(self) -> dict:
        return dict(self.file.attrs)

    @property
    def config(self) -> dict:
        return yaml_load(self.file.attrs["config_yaml"])

    @property
    def fs(self) -> float:
        return float(self.file.attrs["fs_out"])

    @property
    def time(self) -> np.ndarray:
        """Common time vector (fixed and padded layouts); ragged samples use ``arange(length) / fs``."""
        if "time" not in self.file:
            raise ValueError("A ragged dataset has no common time vector; use sample(i)['time']")
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
        """All arrays of sample ``i`` (float32, unpadded) plus its metadata row."""
        n = int(self.lengths[i])
        out = {"time": np.arange(n) / self.fs, "measured": self.signal("measured", i),
               "components": {k: self.signal(f"components/{k}", i) for k in self.component_keys},
               "on_bridge_mask": self.signal("on_bridge_mask", i).astype(bool)}
        for group in ("contact", "bridge", "profile"):
            if group in self.file:
                out[group] = {k: self.signal(f"{group}/{k}", i) for k in self.file[group]}
        if "segment" in self.file:
            out["segment"] = self.signal("segment", i)
        out["metadata"] = {k: v[i] for k, v in self.metadata.items()}
        return out

    def stack_components(self, indices=None, keys: Iterable[str] | None = None) -> np.ndarray:
        """Array (n, n_keys, T) of the requested components (default: all but ``noise``).

        Fixed and padded layouts only (padded samples keep their zero padding).
        """
        if self.layout == "ragged":
            raise ValueError("stack_components needs a fixed or padded layout; use signal(name, i)")
        keys = [k for k in self.component_keys if k != "noise"] if keys is None else list(keys)
        idx = slice(None) if indices is None else np.sort(np.asarray(indices))
        return np.stack([self.component(k)[idx] for k in keys], axis=1)

    def __repr__(self) -> str:
        T = self.file.attrs["n_samples_out"]
        if self.layout != "fixed":
            T = f"{int(self.lengths.min())}-{int(self.lengths.max())} ({self.layout}"
            T += f", padded to {self.file.attrs['n_samples_out']})" if self.layout == "padded" else ")"
        return (f"VBIDataset('{self.path}', n={len(self)}, T={T}, "
                f"fs={self.fs} Hz, components={self.component_keys})")


def load_dataset(path) -> VBIDataset:
    """Open a dataset written by :func:`generate_dataset` (lazy; see :class:`VBIDataset`)."""
    return VBIDataset(path)


def regenerate_sample(dataset, index: int, return_internal: bool = False):
    """Re-simulate sample ``index`` of a dataset in float64 (same scenario and seed).

    For a dataset made by :func:`derive_dataset` the result is the full source crossing; the stored
    sample is ``[crop_start_index : crop_start_index + length]`` of it.
    """
    ds = dataset if isinstance(dataset, VBIDataset) else VBIDataset(dataset)
    try:
        cfg = ds.config
        base_seed = int(ds.file.attrs["base_seed"])
        meta = ds.metadata
        plan = SamplePlan(int(index), int(meta["bridge_id"][index]), int(meta["vehicle_id"][index]),
                          int(meta["partition"][index]), int(meta["sample_seed"][index]))
        redo = int(meta["lift_off_resamples"][index]) if "lift_off_resamples" in meta else 0
        scenario, info = scenario_for(cfg, plan, base_seed, redo)
        return simulate(scenario, info["sample_seed"], return_internal=return_internal)
    finally:
        if ds is not dataset:
            ds.close()


# --------------------------------------------------------------------------------------------
# Derived datasets: crop the road before/after the bridge and/or change the layout
# --------------------------------------------------------------------------------------------
def _crop_bounds(meta: dict, lengths: np.ndarray, fs: float, approach_m, exit_m) -> tuple[np.ndarray, np.ndarray]:
    """First index and length of the cropped window of every sample."""
    n = lengths.size
    start = np.zeros(n, np.int64)
    end = lengths.astype(np.int64) - 1
    speed = np.asarray(meta["speed"], float)
    for i in range(n):
        if approach_m is not None:
            t0 = float(meta["entry_time"][i]) - float(approach_m) / speed[i]
            start[i] = int(math.ceil(t0 * fs - 1e-6))
            if start[i] < 0:
                raise ValueError(f"sample {i} has only {float(meta['entry_time'][i]) * speed[i]:.2f} m of road before "
                                 f"the bridge (asked for {approach_m} m)")
        if exit_m is not None:
            t1 = float(meta["exit_time"][i]) + float(exit_m) / speed[i]
            end[i] = int(math.floor(t1 * fs + 1e-6))
            if end[i] > lengths[i] - 1:
                have = ((lengths[i] - 1) / fs - float(meta["exit_time"][i])) * speed[i]
                raise ValueError(f"sample {i} has only {have:.2f} m of road after the bridge (asked for {exit_m} m)")
    return start, end - start + 1


def derive_dataset(source, out_path, approach_length_m: float | None = None, exit_length_m: float | None = None,
                   layout: str | None = None, pad_to_samples: int | None = None, *, overwrite: bool = False,
                   allow_inside_tool: bool = False, progress: bool = True, command: str | None = None) -> Path:
    """New dataset with the same crossings, cropped and/or in another layout (no new simulation).

    approach_length_m, exit_length_m
        Road kept before the first axle enters / after the last axle leaves the bridge (``None``:
        keep everything). The source must contain at least that much road.
    layout, pad_to_samples
        ``padded`` (needs ``pad_to_samples``) or ``ragged``; ``None`` keeps the source layout.

    Every signal of a crossing is sliced with the same indices, so the cropped crossings are the
    same crossings (same vehicle, speed, bridge vibration, road and noise). The metadata keeps the
    source values except entry_time, exit_time, n_samples_out, window_s and the road lengths, and
    gains ``crop_start_index`` (first source sample kept).
    """
    out_path = _check_output_path(Path(out_path), overwrite, allow_inside_tool)
    src = source if isinstance(source, VBIDataset) else VBIDataset(source)
    tmp = out_path.with_name(out_path.name + ".partial")
    try:
        if "signals" not in src.file.attrs:
            raise ValueError("derive_dataset needs a dataset written with schema >= 1.1")
        layout = layout or src.layout
        if layout not in ("padded", "ragged"):
            raise ValueError("a derived dataset uses the padded or ragged layout")
        if layout == "padded":
            if pad_to_samples is None:
                pad_to_samples = int(src.file.attrs["n_samples_out"]) if src.layout == "padded" else None
            if pad_to_samples is None or int(pad_to_samples) < 1:
                raise ValueError("the padded layout needs pad_to_samples")
            pad_to_samples = int(pad_to_samples)
        specs = json.loads(src.file.attrs["signals"])
        meta = src.metadata
        fs = src.fs
        n = len(src)
        start, new_len = _crop_bounds(meta, src.lengths, fs, approach_length_m, exit_length_m)
        if layout == "padded" and new_len.max() > pad_to_samples:
            raise ValueError(f"the longest cropped crossing has {int(new_len.max())} samples, more than "
                             f"pad_to_samples = {pad_to_samples}")
        cfg = src.config
        comp = _compression(cfg["dataset"])
        if tmp.exists():
            tmp.unlink()
        commit, dirty = git_info()
        t_start = time.time()
        with h5py.File(tmp, "w") as f:
            f.attrs.update(dict(src.file.attrs))
            f.attrs.update({
                "layout": layout, "n_samples_out": pad_to_samples if layout == "padded" else -1,
                "derived_from": str(src.path.name), "derived_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                    timespec="seconds"),
                "derived_git_commit": commit, "derived_git_dirty": dirty, "command": command or "",
                "crop_approach_length_m": float("nan") if approach_length_m is None else float(approach_length_m),
                "crop_exit_length_m": float("nan") if exit_length_m is None else float(exit_length_m),
            })
            writer = _Writer(f, layout, n, pad_to_samples, int(src.file.attrs["n_axles"]), comp)
            for name, spec in specs.items():
                writer.create(name, **spec)
            last = 0.0
            for i in range(n):
                j0, m = int(start[i]), int(new_len[i])
                writer.put(i, {name: src.signal(name, i)[..., j0:j0 + m] for name in specs}, m)
                now = time.time()
                if progress and (now - last > 2.0 or i == n - 1):
                    print(f"  {i + 1}/{n} samples", file=sys.stderr, flush=True)
                    last = now
            src.file.copy(src.file["metadata"], f, name="metadata")
            g = f["metadata"]
            shift = start / fs
            prev = np.asarray(meta["crop_start_index"], np.int64) if "crop_start_index" in meta else np.zeros(n, np.int64)
            updates = {
                "entry_time": np.asarray(meta["entry_time"], float) - shift,
                "exit_time": np.asarray(meta["exit_time"], float) - shift,
                "n_samples_out": new_len.astype(np.int64), "window_s": new_len / fs,
                "crop_start_index": prev + start,
            }
            if approach_length_m is not None:
                updates["approach_length_m"] = np.full(n, float(approach_length_m))
            if exit_length_m is not None:
                updates["exit_length_m"] = np.full(n, float(exit_length_m))
            for key, value in updates.items():
                if key in g:
                    del g[key]
                g.create_dataset(key, data=value)
            writer.finish()
            f.attrs["generation_seconds"] = round(time.time() - t_start, 2)
        os.replace(tmp, out_path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    finally:
        if src is not source:
            src.close()
    return out_path


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
