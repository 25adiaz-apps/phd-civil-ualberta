"""Random sampling of scenarios for datasets.

Reproducibility contract
------------------------
* Everything derives from one base seed through ``np.random.SeedSequence(base_seed, spawn_key=...)``
  (the same children that ``SeedSequence(base_seed).spawn`` produces):
  ``(0, b)`` bridge ``b``, ``(1, v)`` vehicle ``v``, ``(2, i)`` crossing/sample ``i``,
  ``(3,)`` assignment of bridges and vehicles to samples.
* Each parameter uses its own generator, seeded with its owner's entropy plus a hash of the
  parameter path, so adding or changing one parameter does not change the draws of the others.
* A sample depends only on (config, base_seed, its plan entry), never on the order of execution, so
  datasets are identical for any number of worker processes.

Parameter scopes
----------------
* bridge   : ``bridge.*`` (and ``roughness.*`` if ``roughness.scope: bridge``)
* vehicle  : ``vehicle.*``, engine type and order amplitudes, wheel, phone mount and sensor
* crossing : everything else (speed, rpm, engine level, roughness, ambient, SNR targets, ...)

Several crossings share a bridge (``bridge_id``) or a vehicle (``vehicle_id``) so that consumers can
split train/validation/test by bridge and by vehicle without leakage.
"""
from __future__ import annotations

import copy
import math
import zlib
from dataclasses import dataclass

import numpy as np

from .config import ConfigError, distribution_paths, get_path, is_distribution, load_config, set_path
from .roughness import class_bounds
from .simulate import CrossingDoesNotFitError, build_bridge, crossing_window
from .vehicle import VehicleInfeasibleError, build_vehicle

__all__ = ["SamplePlan", "plan_dataset", "scenario_for", "sample_scenario", "sample_value", "parameter_scope"]

VEHICLE_SCOPE_PREFIXES = ("vehicle.", "engine.n_cylinders", "engine.firing_amplitude", "engine.double_firing_amplitude",
                          "engine.half_order_amplitude", "wheel.diameter_m", "wheel.eccentricity_m", "mount.", "sensor.")


@dataclass(frozen=True)
class SamplePlan:
    index: int
    bridge_id: int
    vehicle_id: int
    partition: int
    sample_seed: int


def sample_value(spec, rng: np.random.Generator):
    """Draw one value from a distribution specification (or return a fixed value)."""
    if not is_distribution(spec):
        return copy.deepcopy(spec)
    d = spec["dist"]
    if d == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    if d == "loguniform":
        return float(math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"]))))
    if d == "normal":
        lo, hi = spec.get("low"), spec.get("high")
        for _ in range(100_000):
            x = float(rng.normal(spec["mean"], spec["std"]))
            if (lo is None or x >= lo) and (hi is None or x <= hi):
                return x
        raise ConfigError(f"Could not draw from the truncated normal {spec}; check mean/std/bounds")
    if d == "choice":
        values = spec["values"]
        w = spec.get("weights")
        p = None if w is None else np.asarray(w, float) / np.sum(w)
        return copy.deepcopy(values[int(rng.choice(len(values), p=p))])
    raise ConfigError(f"Unknown distribution {d}")


def parameter_scope(path: str, cfg: dict) -> str:
    if path.startswith("bridge."):
        return "bridge"
    if path.startswith("roughness."):
        return "bridge" if cfg["roughness"]["scope"] == "bridge" else "crossing"
    if any(path == p or path.startswith(p) for p in VEHICLE_SCOPE_PREFIXES):
        return "vehicle"
    return "crossing"


def _words(base_seed: int, *key: int) -> list[int]:
    return [int(w) for w in np.random.SeedSequence(base_seed, spawn_key=key).generate_state(4)]


def _rng(words: list[int], path: str, attempt: int = 0) -> np.random.Generator:
    return np.random.default_rng(words + [zlib.crc32(path.encode("utf-8")), attempt])


def _sample_paths(cfg: dict, src: dict, paths: list[str], words: list[int], attempt: int) -> None:
    max_modes = int(cfg["bridge"]["max_modes"])
    for path in paths:
        spec = get_path(src, path)
        rng = _rng(words, path, attempt)
        if path == "bridge.damping_ratio":  # independent draw for every mode
            value = [sample_value(spec, rng) for _ in range(max_modes)]
        else:
            value = sample_value(spec, rng)
        set_path(cfg, path, value)


def _apply_scope(cfg: dict, src: dict, scope: str, words: list[int], attempt: int = 0,
                 restrict: str | None = None) -> None:
    paths = [p for p in distribution_paths(src) if parameter_scope(p, src) == scope]
    if restrict is not None:
        paths = [p for p in paths if p.startswith(restrict)]
    _sample_paths(cfg, src, paths, words, attempt)
    if scope == ("bridge" if src["roughness"]["scope"] == "bridge" else "crossing") and restrict is None:
        r = cfg["roughness"]
        if src["roughness"]["gd_n0"] is None and isinstance(r["class"], str):
            # Random mode: G_d(n0) log-uniform inside the limits of the drawn class.
            lo, hi = class_bounds(r["class"])
            r["gd_n0"] = float(math.exp(_rng(words, "roughness.gd_n0", attempt).uniform(math.log(lo), math.log(hi))))
        if scope == "bridge" and src["roughness"]["seed"] is None:
            r["seed"] = int(np.random.SeedSequence(words).generate_state(1, np.uint32)[0])


def _allocate(total: int, fractions: np.ndarray, what: str, at_least_one: bool = True) -> np.ndarray:
    """Split ``total`` items in integer counts proportional to ``fractions`` (largest remainder).

    With ``at_least_one`` every group gets one item (bridge and vehicle pools); samples may leave a
    partition empty (e.g. a one-sample preview).
    """
    if at_least_one and total < fractions.size:
        raise ConfigError(f"{what} ({total}) must be at least the number of partitions ({fractions.size})")
    counts = np.floor(fractions * total).astype(int)
    if at_least_one:
        counts = np.maximum(1, counts)
    while counts.sum() > total:
        counts[np.argmax(counts)] -= 1
    remainder = fractions * total - counts
    for k in np.argsort(-remainder)[: total - counts.sum()]:
        counts[k] += 1
    return counts


def _balanced(rng: np.random.Generator, n_items: int, n_groups: int) -> np.ndarray:
    """Assign ``n_items`` to ``n_groups`` with counts differing by at most one, random order."""
    return rng.permutation(np.arange(n_items) % n_groups)


def plan_dataset(config, n_samples: int, base_seed: int = 0) -> list[SamplePlan]:
    """Assign a bridge, a vehicle, a partition and a seed to every sample."""
    cfg = load_config(config)
    if int(n_samples) < 1:
        raise ValueError("n_samples must be >= 1")
    n = int(n_samples)
    dcfg = cfg["dataset"]
    n_b = int(dcfg["n_bridges"] or n)
    n_v = int(dcfg["n_vehicles"] or n)
    rng = np.random.default_rng(np.random.SeedSequence(base_seed, spawn_key=(3,)))
    fractions = dcfg["partitions"]
    if fractions is None:
        groups = [np.arange(n)]
        b_split, v_split = [n_b], [n_v]
        partition = np.zeros(n, int)
    else:
        f = np.asarray(fractions, float) / np.sum(fractions)
        counts = _allocate(n, f, "n_samples", at_least_one=False)
        partition = rng.permutation(np.repeat(np.arange(f.size), counts))
        groups = [np.flatnonzero(partition == g) for g in range(f.size)]
        b_split = _allocate(n_b, f, "dataset.n_bridges").tolist()
        v_split = _allocate(n_v, f, "dataset.n_vehicles").tolist()
    bridge_id = np.zeros(n, int)
    vehicle_id = np.zeros(n, int)
    b_off = v_off = 0
    for g, idx in enumerate(groups):
        if idx.size:
            bridge_id[idx] = b_off + _balanced(rng, idx.size, min(b_split[g], idx.size))
            vehicle_id[idx] = v_off + _balanced(rng, idx.size, min(v_split[g], idx.size))
        b_off += b_split[g]
        v_off += v_split[g]
    plans = []
    for i in range(n):
        seed = int(np.random.SeedSequence(base_seed, spawn_key=(2, i)).generate_state(1, np.uint64)[0] >> 1)
        plans.append(SamplePlan(i, int(bridge_id[i]), int(vehicle_id[i]), int(partition[i]), seed))
    return plans


def scenario_for(config, plan: SamplePlan, base_seed: int = 0) -> tuple[dict, dict]:
    """Fixed scenario for one plan entry. Returns ``(scenario_config, sampling_info)``."""
    src = load_config(config)
    cfg = copy.deepcopy(src)
    fs_out = float(src["simulation"]["fs_out"])
    info = {"sample_index": plan.index, "bridge_id": plan.bridge_id, "vehicle_id": plan.vehicle_id,
            "partition": plan.partition, "sample_seed": plan.sample_seed}

    bw = _words(base_seed, 0, plan.bridge_id)
    _apply_scope(cfg, src, "bridge", bw)
    bridge, _ = build_bridge(cfg["bridge"], fs_out)

    vw = _words(base_seed, 1, plan.vehicle_id)
    _apply_scope(cfg, src, "vehicle", vw)
    model = src["vehicle"]["model"]
    max_tries = int(src["vehicle"][model].get("max_resamples", 1))
    for attempt in range(max_tries):
        try:
            if attempt:
                _apply_scope(cfg, src, "vehicle", vw, attempt, restrict=f"vehicle.{model}.")
            vehicle = build_vehicle(cfg["vehicle"])
            info["vehicle_resamples"] = attempt
            break
        except VehicleInfeasibleError as exc:
            last = exc
    else:
        raise ConfigError(f"No feasible vehicle after {max_tries} draws of vehicle.{model}; last error: {last}")

    cw = _words(base_seed, 2, plan.index)
    _apply_scope(cfg, src, "crossing", cw)
    sim = src["simulation"]
    window = int(sim["n_samples_out"]) / fs_out
    ccfg = src["crossing"]
    max_tries = int(ccfg["max_speed_resamples"])
    for attempt in range(max_tries):
        if attempt:
            if not is_distribution(ccfg["speed"]):
                break
            _sample_paths(cfg, src, ["crossing.speed"], cw, attempt)
        speed = float(cfg["crossing"]["speed"])
        try:
            crossing_window(window, (bridge.span + vehicle.wheelbase) / speed, float(cfg["crossing"]["approach_margin_s"]),
                            float(cfg["crossing"]["exit_margin_s"]))
            info["speed_resamples"] = attempt
            break
        except CrossingDoesNotFitError:
            continue
    else:
        attempt = max_tries
    if "speed_resamples" not in info:
        raise CrossingDoesNotFitError(
            f"Sample {plan.index}: no speed in the configured range lets the crossing of a {bridge.span:.1f} m bridge "
            f"fit in the {window:.2f} s window after {attempt + 1} draws; widen the speed range, shorten the span, "
            "reduce the margins or increase simulation.n_samples_out")
    return cfg, info


def sample_scenario(config, seed: int = 0) -> tuple[dict, int]:
    """Draw one fixed scenario from a (possibly random) configuration.

    Returns ``(scenario_config, sample_seed)``; ``simulate(scenario_config, sample_seed)`` then
    simulates it. Equivalent to sample 0 of a one-sample dataset with base seed ``seed``.
    """
    plan = plan_dataset(config, 1, seed)[0]
    cfg, _ = scenario_for(config, plan, seed)
    return cfg, plan.sample_seed
