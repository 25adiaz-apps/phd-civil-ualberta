"""Configuration loading, merging and validation.

A configuration is a nested mapping (normally read from YAML). Every configuration is
deep-merged on top of ``defaults.yaml`` so presets only list what they change. Values can be
fixed scalars or distribution specifications (``{dist: uniform, low: .., high: ..}``); see
``defaults.yaml`` for the syntax. Validation happens at load time so that a typo or an invalid
range fails immediately instead of at sample 7000 of a dataset.
"""
from __future__ import annotations

import copy
import difflib
import functools
import math
import os
import re
import warnings
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

import yaml

__all__ = [
    "ConfigError",
    "ConfigWarning",
    "load_config",
    "load_preset",
    "list_presets",
    "find_preset",
    "with_overrides",
    "parse_override",
    "config_to_yaml",
    "default_config",
    "is_distribution",
    "get_path",
    "set_path",
    "iter_leaves",
    "distribution_paths",
    "require_fixed",
    "PRESETS_DIR",
    "TOOL_ROOT",
]

TOOL_ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = Path(os.environ.get("VBSIG_PRESETS_DIR", TOOL_ROOT / "presets"))

DIST_TYPES = ("uniform", "loguniform", "normal", "choice")
_DIST_FIELDS = {
    "uniform": ({"low", "high"}, set()),
    "loguniform": ({"low", "high"}, set()),
    "normal": ({"mean", "std"}, {"low", "high"}),
    "choice": ({"values"}, {"weights"}),
}
ROUGHNESS_CLASSES = ("A", "B", "C", "D", "E", "F", "G", "H")
TOP_LEVEL_FREE_KEYS = ("name", "description", "extends")

# Keys that define array shapes or the solver itself: they cannot be distributions.
STRUCTURAL_PREFIXES = (
    "simulation.",
    "crossing.placement",
    "crossing.max_speed_resamples",
    "bridge.f_max",
    "bridge.max_modes",
    "bridge.n_label_modes",
    "bridge.nominal_f1_table",
    "vehicle.model",
    "vehicle.sprung_mass.max_resamples",
    "vehicle.quarter_car.max_resamples",
    "vehicle.half_car.max_resamples",
    "roughness.scope",
    "roughness.n0",
    "snr_control.max_iterations",
    "dataset.",
)


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads YAML 1.2 floats such as ``3.3e9`` or ``1e-4`` as numbers.

    PyYAML follows YAML 1.1, where a float needs a dot and a signed exponent (``3.3e+9``), so
    ``3.3e9`` would silently become a string.
    """


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
                |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                |[-+]?\.(?:inf|Inf|INF)
                |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."),
)


def yaml_load(text: str) -> Any:
    """Parse YAML with :class:`_Loader` (safe, YAML 1.2 floats)."""
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - _Loader derives from SafeLoader


class ConfigError(ValueError):
    """Invalid configuration (unknown key, wrong type, impossible range...)."""


class ConfigWarning(UserWarning):
    """Suspicious but valid configuration."""


# --------------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------------
def is_distribution(value: Any) -> bool:
    """True if ``value`` is a distribution specification (a mapping with a ``dist`` key)."""
    return isinstance(value, Mapping) and "dist" in value


def _is_branch(value: Any) -> bool:
    return isinstance(value, Mapping) and not is_distribution(value)


def get_path(cfg: Mapping, path: str, default: Any = KeyError) -> Any:
    """Return ``cfg['a']['b']`` for ``path='a.b'``."""
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            if default is KeyError:
                raise KeyError(path)
            return default
        node = node[part]
    return node


def set_path(cfg: dict, path: str, value: Any) -> None:
    """Set ``cfg['a']['b'] = value`` for ``path='a.b'`` (in place)."""
    parts = path.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def iter_leaves(cfg: Mapping, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every leaf; distributions count as leaves."""
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if _is_branch(value):
            yield from iter_leaves(value, path + ".")
        else:
            yield path, value


def distribution_paths(cfg: Mapping, relevant_only: bool = False) -> list[str]:
    """Dotted paths of the parameters that are still distributions.

    With ``relevant_only`` the parameters that the simulation never reads (inactive vehicle models,
    disabled disturbances, f1_factor when f1 is given, ...) are left out.
    """
    paths = [p for p, v in iter_leaves(cfg) if is_distribution(v)]
    if relevant_only:
        skip = irrelevant_prefixes(cfg)
        paths = [p for p in paths if not any(p == s.rstrip(".") or p.startswith(s) for s in skip)]
    return paths


def irrelevant_prefixes(cfg: Mapping) -> list[str]:
    """Paths (or path prefixes ending with '.') that the simulation does not read for this config.

    A switch that is itself still a distribution keeps everything under it relevant.
    """
    def fixed(path):
        return get_path(cfg, path, None)

    out = ["dataset.", "name", "description"]
    model = fixed("vehicle.model")
    out += [f"vehicle.{m}." for m in ("sprung_mass", "quarter_car", "half_car") if m != model]
    for section in ("roughness", "engine", "wheel", "ambient", "mount"):
        if fixed(f"{section}.enabled") is False:
            out.append(f"{section}.")
    for sub in ("white_noise", "pink_noise", "bias", "drift", "anti_alias", "jitter", "quantization"):
        if fixed(f"sensor.{sub}.enabled") is False:
            out.append(f"sensor.{sub}.")
    if fixed("roughness.gd_n0") is not None and not is_distribution(fixed("roughness.gd_n0")):
        out.append("roughness.class")
    if fixed("roughness.smoothing.enabled") is False:
        out.append("roughness.smoothing.length_m")
    if fixed("bridge.flexural_rigidity") is not None:
        out += ["bridge.f1", "bridge.f1_factor"]
    elif fixed("bridge.f1") is not None:
        out.append("bridge.f1_factor")
    if fixed("crossing.placement") != "fixed":
        out.append("crossing.entry_time_s")
    if fixed("vehicle.quarter_car.suspension_stiffness") is not None:
        out += ["vehicle.quarter_car.body_frequency", "vehicle.quarter_car.wheel_frequency"]
    if fixed("vehicle.quarter_car.suspension_damping") is not None:
        out.append("vehicle.quarter_car.suspension_damping_ratio")
    if fixed("vehicle.sprung_mass.stiffness") is not None:
        out.append("vehicle.sprung_mass.frequency")
    if fixed("vehicle.sprung_mass.damping") is not None:
        out.append("vehicle.sprung_mass.damping_ratio")
    if fixed("simulation.amplitude_mode") != "snr_controlled":
        out.append("snr_control.")
    else:
        # The SNR targets replace the physical amplitudes of the engine and of the white noise.
        out += ["engine.body_accel_rms", "sensor.white_noise.density_ug"]
        for kind in ("roughness", "engine", "wheel"):
            if fixed(f"{kind}.enabled") is False:
                out.append(f"snr_control.targets_db.{kind}")
        if fixed("sensor.white_noise.enabled") is False:
            out.append("snr_control.targets_db.sensor_noise")
    return out


def require_fixed(cfg: Mapping, what: str = "simulate()") -> None:
    """Raise ConfigError if any parameter used by the simulation is still a distribution."""
    paths = distribution_paths(cfg, relevant_only=True)
    if paths:
        listed = "\n  ".join(paths)
        raise ConfigError(
            f"{what} needs a fully fixed scenario, but these parameters are distributions:\n  {listed}\n"
            "Fix them in the configuration, or draw a scenario first with "
            "vehicle_bridge_signal.sample_scenario(config, seed)."
        )


def _deep_merge(base: Mapping, update: Mapping) -> dict:
    out = copy.deepcopy(dict(base))
    for key, value in update.items():
        if key in out and _is_branch(out[key]) and _is_branch(value):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def config_to_yaml(cfg: Mapping) -> str:
    """Serialise a configuration to YAML (keeps key order)."""
    return yaml.safe_dump(_plain(cfg), sort_keys=False, allow_unicode=True, width=110)


def _plain(obj: Any) -> Any:
    """Convert numpy scalars/arrays and tuples to plain Python types for YAML/JSON."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.ndarray):
            return [_plain(v) for v in obj.tolist()]
        if isinstance(obj, np.generic):
            return obj.item()
    return obj


# --------------------------------------------------------------------------------------------
# Reading files and presets
# --------------------------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read configuration file {path}: {exc}") from exc
    try:
        data = yaml_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}:\n{exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: the top level of a configuration must be a mapping, got {type(data).__name__}")
    return dict(data)


@functools.lru_cache(maxsize=1)
def _defaults_cached() -> dict:
    text = resources.files("vehicle_bridge_signal").joinpath("defaults.yaml").read_text(encoding="utf-8")
    return yaml_load(text)


def default_config() -> dict:
    """The packaged defaults (a fresh copy)."""
    return copy.deepcopy(_defaults_cached())


def find_preset(name: str) -> Path:
    """Path of a preset given its name (``smoke``) or file name (``smoke.yaml``)."""
    stem = name[:-5] if name.endswith(".yaml") else name
    candidate = PRESETS_DIR / f"{stem}.yaml"
    if candidate.is_file():
        return candidate
    available = [p.stem for p in sorted(PRESETS_DIR.glob("*.yaml"))] if PRESETS_DIR.is_dir() else []
    hint = difflib.get_close_matches(stem, available, n=3)
    msg = f"Preset '{name}' not found in {PRESETS_DIR}."
    if hint:
        msg += f" Did you mean: {', '.join(hint)}?"
    elif available:
        msg += f" Available presets: {', '.join(available)}."
    else:
        msg += " (Is the package installed in editable mode from its repository folder?)"
    raise ConfigError(msg)


def list_presets() -> list[dict]:
    """Name, description and path of every preset in the presets folder."""
    out = []
    if not PRESETS_DIR.is_dir():
        return out
    for path in sorted(PRESETS_DIR.glob("*.yaml")):
        data = _read_yaml(path)
        out.append({
            "name": path.stem,
            "description": " ".join(str(data.get("description", "")).split()),
            "extends": data.get("extends"),
            "path": str(path),
        })
    return out


def _resolve_reference(ref: str, base_dir: Path | None) -> Path:
    """A reference in ``extends`` or a CLI ``--config``: existing path or preset name."""
    path = Path(ref)
    candidates = [path] if path.is_absolute() else ([base_dir / path] if base_dir else []) + [Path.cwd() / path]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    if path.suffix in ("", ".yaml") and len(path.parts) == 1:
        return find_preset(ref)
    raise ConfigError(f"Configuration '{ref}' not found (looked in {', '.join(str(c) for c in candidates)}).")


def _load_with_extends(data: dict, base_dir: Path | None, chain: tuple[str, ...]) -> dict:
    parents = data.pop("extends", None)
    if parents is None:
        return data
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
        raise ConfigError("'extends' must be a preset name/path or a list of them")
    merged: dict = {}
    for ref in parents:
        path = _resolve_reference(ref, base_dir)
        key = str(path)
        if key in chain:
            raise ConfigError(f"Circular 'extends' chain: {' -> '.join(chain + (key,))}")
        parent = _load_with_extends(_read_yaml(path), path.parent, chain + (key,))
        merged = _deep_merge(merged, parent)
    # Name and description describe the child, never the parent.
    merged.pop("name", None)
    merged.pop("description", None)
    return _deep_merge(merged, data)


def _source_to_mapping(source: Any) -> tuple[dict, Path | None, str | None]:
    if source is None:
        return {}, None, None
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source)), None, None
    if isinstance(source, (str, os.PathLike)):
        path = _resolve_reference(str(source), None)
        return _read_yaml(path), path.parent, str(path)
    raise ConfigError(f"Unsupported configuration source of type {type(source).__name__}")


# --------------------------------------------------------------------------------------------
# Public loading API
# --------------------------------------------------------------------------------------------
def load_config(source: Any = None, overrides: Mapping[str, Any] | list[str] | None = None) -> dict:
    """Load, merge on the defaults and validate a configuration.

    Parameters
    ----------
    source
        ``None`` (defaults only), a mapping, a YAML path, or a preset name (``"smoke"``).
    overrides
        Dotted-key overrides applied last, e.g. ``{"crossing.speed": 12.0}`` or
        ``["crossing.speed=12", "engine.enabled=false"]``.

    Returns
    -------
    dict
        A complete, validated configuration (plain Python types).
    """
    data, base_dir, origin = _source_to_mapping(source)
    data = _load_with_extends(data, base_dir, (origin,) if origin else ())
    defaults = default_config()
    _check_unknown_keys(data, defaults, "")
    cfg = _deep_merge(defaults, data)
    if origin and "name" not in data:
        cfg["name"] = Path(origin).stem
    if overrides:
        cfg = _apply_overrides(cfg, overrides, defaults)
    validate_config(cfg)
    return cfg


def load_preset(name: str, overrides: Mapping[str, Any] | list[str] | None = None) -> dict:
    """Load a preset by name (see :func:`list_presets`)."""
    return load_config(find_preset(name), overrides)


def with_overrides(cfg: Mapping, overrides: Mapping[str, Any] | list[str]) -> dict:
    """Copy of ``cfg`` with dotted-key overrides applied and validated (parametric studies)."""
    out = _apply_overrides(copy.deepcopy(dict(cfg)), overrides, default_config())
    validate_config(out)
    return out


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``'a.b=value'``; the value is read as YAML (numbers, booleans, lists, mappings)."""
    if "=" not in text:
        raise ConfigError(f"Override '{text}' must have the form key.path=value")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"Override '{text}' has an empty key")
    try:
        value = yaml_load(raw) if raw.strip() else None
    except yaml.YAMLError as exc:
        raise ConfigError(f"Cannot parse the value of override '{text}': {exc}") from exc
    return key, value


def _apply_overrides(cfg: dict, overrides: Mapping[str, Any] | list[str], defaults: Mapping) -> dict:
    items = [parse_override(o) for o in overrides] if isinstance(overrides, (list, tuple)) else list(overrides.items())
    for key, value in items:
        try:
            ref = get_path(defaults, key)
        except KeyError:
            raise ConfigError(_unknown_key_message(key, defaults)) from None
        if _is_branch(ref) and not _is_branch(value):
            raise ConfigError(f"Override '{key}' points to a section; give a leaf such as '{key}.<name>'")
        if _is_branch(ref):
            _check_unknown_keys(value, ref, key + ".")
            set_path(cfg, key, _deep_merge(get_path(cfg, key), value))
        else:
            set_path(cfg, key, copy.deepcopy(value))
    return cfg


def _all_paths(tree: Mapping, prefix: str = "") -> list[str]:
    out = []
    for key, value in tree.items():
        out.append(prefix + key)
        if _is_branch(value):
            out.extend(_all_paths(value, prefix + key + "."))
    return out


def _unknown_key_message(path: str, defaults: Mapping) -> str:
    close = difflib.get_close_matches(path, _all_paths(defaults), n=3, cutoff=0.6)
    msg = f"Unknown configuration key '{path}'."
    if close:
        msg += f" Did you mean: {', '.join(close)}?"
    return msg


def _check_unknown_keys(user: Mapping, ref: Mapping, prefix: str) -> None:
    for key, value in user.items():
        path = prefix + str(key)
        if prefix == "" and key in TOP_LEVEL_FREE_KEYS:
            continue
        if key not in ref:
            raise ConfigError(_unknown_key_message(path, default_config()))
        ref_value = ref[key]
        if _is_branch(ref_value):
            if not _is_branch(value):
                raise ConfigError(f"'{path}' is a section (mapping of parameters), got {value!r}")
            _check_unknown_keys(value, ref_value, path + ".")
        elif _is_branch(value):
            raise ConfigError(
                f"'{path}' is a parameter, but a mapping without 'dist' was given: {value!r}. "
                "Use a scalar or a distribution {dist: ..., ...}."
            )


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------
def _c(kind: str, nullable: bool = False, **kw) -> dict:
    return {"kind": kind, "nullable": nullable, **kw}


# Constraint per parameter. kinds: real, pos, nonneg, open01, frac, bool, int, choice, str.
CONSTRAINTS: dict[str, dict] = {
    "simulation.fs_sim": _c("pos"),
    "simulation.fs_out": _c("pos"),
    "simulation.n_samples_out": _c("int", min=16),
    "simulation.warmup_s": _c("nonneg"),
    "simulation.tail_s": _c("nonneg"),
    "simulation.interaction": _c("choice", values=("coupled", "decoupled")),
    "simulation.amplitude_mode": _c("choice", values=("physical", "snr_controlled")),
    "simulation.newmark_gamma": _c("pos"),
    "simulation.newmark_beta": _c("pos"),
    "crossing.speed": _c("pos"),
    "crossing.approach_margin_s": _c("nonneg"),
    "crossing.exit_margin_s": _c("nonneg"),
    "crossing.placement": _c("choice", values=("random", "center", "fixed")),
    "crossing.entry_time_s": _c("nonneg", nullable=True),
    "crossing.max_speed_resamples": _c("int", min=1),
    "bridge.span": _c("pos"),
    "bridge.f1": _c("pos", nullable=True),
    "bridge.f1_factor": _c("pos"),
    "bridge.flexural_rigidity": _c("pos", nullable=True),
    "bridge.mass_per_length": _c("pos"),
    "bridge.damping_ratio": _c("frac", allow_list=True),
    "bridge.max_modes": _c("int", min=1, max=60),
    "bridge.n_label_modes": _c("int", min=1, max=60),
    "vehicle.model": _c("choice", values=("sprung_mass", "quarter_car", "half_car")),
    "vehicle.sprung_mass.mass": _c("pos"),
    "vehicle.sprung_mass.frequency": _c("pos"),
    "vehicle.sprung_mass.damping_ratio": _c("nonneg"),
    "vehicle.sprung_mass.stiffness": _c("pos", nullable=True),
    "vehicle.sprung_mass.damping": _c("nonneg", nullable=True),
    "vehicle.quarter_car.body_mass": _c("pos"),
    "vehicle.quarter_car.unsprung_mass": _c("pos"),
    "vehicle.quarter_car.body_frequency": _c("pos"),
    "vehicle.quarter_car.wheel_frequency": _c("pos"),
    "vehicle.quarter_car.suspension_damping_ratio": _c("nonneg"),
    "vehicle.quarter_car.tyre_damping": _c("nonneg"),
    "vehicle.quarter_car.suspension_stiffness": _c("pos", nullable=True),
    "vehicle.quarter_car.tyre_stiffness": _c("pos", nullable=True),
    "vehicle.quarter_car.suspension_damping": _c("nonneg", nullable=True),
    "vehicle.quarter_car.max_resamples": _c("int", min=1),
    "vehicle.half_car.body_mass": _c("pos"),
    "vehicle.half_car.dynamic_index": _c("pos"),
    "vehicle.half_car.wheelbase": _c("pos"),
    "vehicle.half_car.cg_to_front_ratio": _c("open01"),
    "vehicle.half_car.unsprung_mass": _c("pos"),
    "vehicle.half_car.body_frequency": _c("pos"),
    "vehicle.half_car.wheel_frequency": _c("pos"),
    "vehicle.half_car.suspension_damping_ratio": _c("nonneg"),
    "vehicle.half_car.tyre_damping": _c("nonneg"),
    "vehicle.half_car.sensor_position": _c("real"),
    "vehicle.half_car.engine_position": _c("real", nullable=True),
    "vehicle.half_car.max_resamples": _c("int", min=1),
    "roughness.enabled": _c("bool"),
    "roughness.scope": _c("choice", values=("crossing", "bridge")),
    "roughness.class": _c("choice", nullable=True, values=ROUGHNESS_CLASSES),
    "roughness.gd_n0": _c("pos", nullable=True),
    "roughness.waviness": _c("pos"),
    "roughness.n0": _c("pos"),
    "roughness.n_min": _c("pos"),
    "roughness.n_max": _c("pos"),
    "roughness.profile_period_m": _c("pos"),
    "roughness.smoothing.enabled": _c("bool"),
    "roughness.smoothing.length_m": _c("pos"),
    "roughness.seed": _c("int", nullable=True, min=0),
    "engine.enabled": _c("bool"),
    "engine.n_cylinders": _c("int", min=1, max=16),
    "engine.rpm_mean": _c("pos"),
    "engine.rpm_relative_std": _c("nonneg", max=0.5),
    "engine.rpm_correlation_time_s": _c("pos", min=0.1),
    "engine.firing_amplitude": _c("nonneg"),
    "engine.double_firing_amplitude": _c("nonneg"),
    "engine.half_order_amplitude": _c("nonneg"),
    "engine.body_accel_rms": _c("pos"),
    "wheel.enabled": _c("bool"),
    "wheel.diameter_m": _c("pos"),
    "wheel.eccentricity_m": _c("nonneg"),
    "ambient.enabled": _c("bool"),
    "ambient.modal_accel_rms": _c("nonneg"),
    "mount.enabled": _c("bool"),
    "mount.frequency_hz": _c("pos"),
    "mount.damping_ratio": _c("pos"),
    "sensor.white_noise.enabled": _c("bool"),
    "sensor.white_noise.density_ug": _c("nonneg"),
    "sensor.pink_noise.enabled": _c("bool"),
    "sensor.pink_noise.density_at_1hz_ug": _c("nonneg"),
    "sensor.bias.enabled": _c("bool"),
    "sensor.bias.value_m_s2": _c("real"),
    "sensor.drift.enabled": _c("bool"),
    "sensor.drift.random_walk_m_s2_per_sqrt_s": _c("nonneg"),
    "sensor.anti_alias.enabled": _c("bool"),
    "sensor.anti_alias.order": _c("int", min=1, max=12),
    "sensor.anti_alias.cutoff_ratio": _c("pos", max=0.5, max_open=True),
    "sensor.anti_alias.zero_phase": _c("bool"),
    "sensor.jitter.enabled": _c("bool"),
    "sensor.jitter.std_s": _c("nonneg"),
    "sensor.quantization.enabled": _c("bool"),
    "sensor.quantization.bits": _c("int", min=2, max=32),
    "sensor.quantization.range_g": _c("pos"),
    "sensor.quantization.include_gravity": _c("bool"),
    "snr_control.targets_db.roughness": _c("real"),
    "snr_control.targets_db.engine": _c("real"),
    "snr_control.targets_db.wheel": _c("real"),
    "snr_control.targets_db.sensor_noise": _c("real"),
    "snr_control.max_iterations": _c("int", min=1),
    "dataset.n_samples": _c("int", nullable=True, min=1),
    "dataset.n_bridges": _c("int", nullable=True, min=1),
    "dataset.n_vehicles": _c("int", nullable=True, min=1),
    "dataset.compression": _c("choice", values=("gzip", "lzf", "none")),
    "dataset.compression_level": _c("int", min=0, max=9),
    "dataset.store_contact": _c("bool"),
    "dataset.store_bridge_response": _c("bool"),
}


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _check_scalar(path: str, x: Any, c: dict) -> None:
    kind = c["kind"]
    if x is None:
        if c["nullable"]:
            return
        raise ConfigError(f"'{path}' cannot be null")
    if kind == "bool":
        if not isinstance(x, bool):
            raise ConfigError(f"'{path}' must be true or false, got {x!r}")
        return
    if kind == "choice":
        if x not in c["values"]:
            raise ConfigError(f"'{path}' must be one of {list(c['values'])}, got {x!r}")
        return
    if kind == "str":
        if not isinstance(x, str):
            raise ConfigError(f"'{path}' must be a string, got {x!r}")
        return
    if kind == "int":
        if isinstance(x, bool) or not (isinstance(x, int) or (isinstance(x, float) and x.is_integer())):
            raise ConfigError(f"'{path}' must be an integer, got {x!r}")
    elif not _is_number(x):
        raise ConfigError(f"'{path}' must be a finite number, got {x!r}")
    x = float(x)
    if kind == "pos" and not x > 0:
        raise ConfigError(f"'{path}' must be > 0, got {x}")
    if kind in ("nonneg", "frac") and x < 0:
        raise ConfigError(f"'{path}' must be >= 0, got {x}")
    if kind == "frac" and not x < 1:
        raise ConfigError(f"'{path}' must be < 1 (it is a damping ratio), got {x}")
    if kind == "open01" and not 0 < x < 1:
        raise ConfigError(f"'{path}' must be strictly between 0 and 1, got {x}")
    if "min" in c and x < c["min"]:
        raise ConfigError(f"'{path}' must be >= {c['min']}, got {x}")
    if "max" in c:
        if c.get("max_open") and not x < c["max"]:
            raise ConfigError(f"'{path}' must be < {c['max']}, got {x}")
        if x > c["max"]:
            raise ConfigError(f"'{path}' must be <= {c['max']}, got {x}")


def _check_distribution(path: str, spec: Mapping, c: dict | None, structural: bool) -> None:
    if structural:
        raise ConfigError(f"'{path}' is structural (defines array shapes or the solver) and must be fixed, "
                          f"not a distribution")
    dist = spec.get("dist")
    if dist not in DIST_TYPES:
        raise ConfigError(f"'{path}': unknown distribution '{dist}'. Use one of {list(DIST_TYPES)}")
    required, optional = _DIST_FIELDS[dist]
    keys = set(spec) - {"dist"}
    missing = required - keys
    extra = keys - required - optional
    if missing:
        raise ConfigError(f"'{path}': distribution '{dist}' needs {sorted(missing)}")
    if extra:
        raise ConfigError(f"'{path}': distribution '{dist}' does not accept {sorted(extra)}")
    if dist == "choice":
        values = spec["values"]
        if not isinstance(values, list) or not values:
            raise ConfigError(f"'{path}': 'values' must be a non-empty list")
        weights = spec.get("weights")
        if weights is not None:
            if not isinstance(weights, list) or len(weights) != len(values):
                raise ConfigError(f"'{path}': 'weights' must be a list with one weight per value")
            if not all(_is_number(w) and w >= 0 for w in weights) or sum(weights) <= 0:
                raise ConfigError(f"'{path}': weights must be >= 0 and not all zero")
        if c is not None:
            for v in values:
                _check_scalar(path, v, c)
        return
    for key in keys:
        if not _is_number(spec[key]):
            raise ConfigError(f"'{path}': '{key}' must be a finite number, got {spec[key]!r}")
    if dist in ("uniform", "loguniform"):
        low, high = float(spec["low"]), float(spec["high"])
        if not low < high:
            raise ConfigError(f"'{path}': need low < high, got low={low}, high={high} "
                              "(use a scalar for a fixed value)")
        if dist == "loguniform" and low <= 0:
            raise ConfigError(f"'{path}': loguniform needs low > 0")
        bounds = (low, high)
    else:  # normal
        if not float(spec["std"]) > 0:
            raise ConfigError(f"'{path}': normal needs std > 0")
        low, high = spec.get("low"), spec.get("high")
        if low is not None and high is not None and not low < high:
            raise ConfigError(f"'{path}': need low < high")
        if c is not None and c["kind"] in ("pos", "nonneg", "frac", "open01", "int") and low is None:
            raise ConfigError(f"'{path}': a normal distribution needs a 'low' truncation bound here "
                              f"because the parameter must be {c['kind']}")
        bounds = tuple(b for b in (low, high) if b is not None)
    if c is not None:
        if c["kind"] in ("bool", "choice", "str"):
            raise ConfigError(f"'{path}': use {{dist: choice, values: [...]}} for this parameter")
        for b in bounds:
            cc = dict(c)
            if c["kind"] == "int":
                cc["kind"] = "real"
            # The support of a continuous distribution may touch an open bound only with zero probability.
            if c["kind"] == "pos" and b == bounds[0] and b == 0:
                continue
            _check_scalar(path, b, cc)


def _is_structural(path: str) -> bool:
    return any(path == p.rstrip(".") or path.startswith(p) for p in STRUCTURAL_PREFIXES)


def _value(cfg: Mapping, path: str) -> Any:
    return get_path(cfg, path)


def _fixed_number(cfg: Mapping, path: str) -> float | None:
    v = get_path(cfg, path, None)
    return float(v) if _is_number(v) else None


def _upper_bound(value: Any) -> float | None:
    """Largest value a parameter can take (None if unbounded or not numeric)."""
    if _is_number(value):
        return float(value)
    if is_distribution(value):
        d = value["dist"]
        if d in ("uniform", "loguniform"):
            return float(value["high"])
        if d == "normal":
            return float(value["high"]) if value.get("high") is not None else None
        if d == "choice":
            nums = [float(v) for v in value["values"] if _is_number(v)]
            return max(nums) if nums else None
    return None


def validate_config(cfg: Mapping) -> None:
    """Validate types, ranges and cross-field consistency. Raises :class:`ConfigError`."""
    for top in TOP_LEVEL_FREE_KEYS:
        if top in cfg and top != "extends" and cfg[top] is not None and not isinstance(cfg[top], str):
            raise ConfigError(f"'{top}' must be a string")
    for path, value in iter_leaves({k: v for k, v in cfg.items() if k not in TOP_LEVEL_FREE_KEYS}):
        c = CONSTRAINTS.get(path)
        structural = _is_structural(path)
        if is_distribution(value):
            _check_distribution(path, value, c, structural)
        elif isinstance(value, list) and c is not None and c.get("allow_list"):
            if not value:
                raise ConfigError(f"'{path}': empty list")
            for i, item in enumerate(value):
                if is_distribution(item):
                    raise ConfigError(f"'{path}[{i}]': distributions are not allowed inside a list; use a single "
                                      "distribution (drawn independently for every mode) or a list of numbers")
                _check_scalar(f"{path}[{i}]", item, c)
        elif c is not None:
            _check_scalar(path, value, c)
    _validate_cross_fields(cfg)


def _validate_cross_fields(cfg: Mapping) -> None:
    sim = cfg["simulation"]
    fs_sim, fs_out = float(sim["fs_sim"]), float(sim["fs_out"])
    ratio = fs_sim / fs_out
    if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
        raise ConfigError(f"simulation.fs_sim ({fs_sim}) must be an integer multiple of simulation.fs_out ({fs_out})")
    if round(ratio) < 2:
        warnings.warn("fs_sim equals fs_out: the anti-alias filter cannot act and the engine will alias in the "
                      "simulation itself", ConfigWarning, stacklevel=3)
    gamma, beta = float(sim["newmark_gamma"]), float(sim["newmark_beta"])
    if gamma < 0.5 or beta < 0.25 * (0.5 + gamma) ** 2 - 1e-12:
        raise ConfigError("Newmark parameters must satisfy gamma >= 1/2 and beta >= (gamma + 1/2)^2 / 4 "
                          "(unconditional stability); use gamma=0.5, beta=0.25")

    bridge = cfg["bridge"]
    f_max = bridge["f_max"]
    if f_max != "auto" and not (_is_number(f_max) and f_max > 0):
        raise ConfigError(f"bridge.f_max must be 'auto' or a positive number, got {f_max!r}")
    if _is_number(f_max) and f_max >= fs_out / 2:
        warnings.warn(f"bridge.f_max={f_max} Hz is above the output Nyquist frequency ({fs_out / 2} Hz)",
                      ConfigWarning, stacklevel=3)
    if int(bridge["n_label_modes"]) > int(bridge["max_modes"]):
        raise ConfigError("bridge.n_label_modes (K) cannot exceed bridge.max_modes")
    table = bridge["nominal_f1_table"]
    spans, f1s = table.get("span"), table.get("f1")
    if (not isinstance(spans, list) or not isinstance(f1s, list) or len(spans) != len(f1s) or len(spans) < 2
            or not all(_is_number(s) and s > 0 for s in spans + f1s)
            or any(b <= a for a, b in zip(spans, spans[1:]))):
        raise ConfigError("bridge.nominal_f1_table needs two lists 'span' (strictly increasing) and 'f1' of equal "
                          "length >= 2 with positive values")
    if bridge["flexural_rigidity"] is not None and bridge["f1"] is not None:
        raise ConfigError("Give either bridge.f1 or bridge.flexural_rigidity, not both")

    crossing = cfg["crossing"]
    if crossing["placement"] == "fixed" and crossing["entry_time_s"] is None:
        raise ConfigError("crossing.placement: fixed needs crossing.entry_time_s")

    rough = cfg["roughness"]
    n_min, n_max = _fixed_number(cfg, "roughness.n_min"), _fixed_number(cfg, "roughness.n_max")
    if n_min is not None and n_max is not None and not n_min < n_max:
        raise ConfigError("roughness.n_min must be smaller than roughness.n_max")
    if rough["class"] is None and rough["gd_n0"] is None and rough["enabled"] is not False:
        raise ConfigError("roughness: give a class or gd_n0 (or disable roughness)")
    if isinstance(rough["class"], str) and _is_number(rough["gd_n0"]):
        from .roughness import class_bounds  # local import avoids a cycle
        lo, hi = class_bounds(rough["class"])
        if not lo <= float(rough["gd_n0"]) <= hi:
            raise ConfigError(f"roughness.gd_n0={rough['gd_n0']} is outside the limits of class {rough['class']} "
                              f"[{lo:g}, {hi:g}] m^3; set roughness.class: null to use gd_n0 alone")
    v_max = _upper_bound(crossing["speed"])
    nu_max = _upper_bound(rough["n_max"])
    if v_max is not None and nu_max is not None and v_max * nu_max >= 0.5 * fs_sim:
        raise ConfigError(f"Roughness up to {nu_max} cycles/m at {v_max} m/s excites up to {v_max * nu_max:g} Hz, "
                          f"above the simulation Nyquist frequency {fs_sim / 2:g} Hz: raise simulation.fs_sim or "
                          f"lower roughness.n_max")

    jitter_std = _upper_bound(cfg["sensor"]["jitter"]["std_s"])
    if jitter_std is not None and jitter_std >= 0.45 / fs_out:
        raise ConfigError(f"sensor.jitter.std_s must be < 0.45/fs_out = {0.45 / fs_out:g} s")
    cutoff = _upper_bound(cfg["sensor"]["anti_alias"]["cutoff_ratio"])
    if cutoff is not None and cutoff * fs_out >= 0.5 * fs_sim:
        raise ConfigError("The anti-alias cutoff must be below fs_sim/2")

    eng = cfg["engine"]
    rpm_max = _upper_bound(eng["rpm_mean"])
    cyl_max = _upper_bound(eng["n_cylinders"])
    rel = _upper_bound(eng["rpm_relative_std"]) or 0.0
    if rpm_max is not None and cyl_max is not None and eng["enabled"] is not False:
        top_order = cyl_max if (_upper_bound(eng["double_firing_amplitude"]) or 0) > 0 else cyl_max / 2
        f_top = rpm_max * (1 + 4 * rel) / 60 * top_order
        if f_top >= 0.5 * fs_sim:
            raise ConfigError(f"The highest engine order can reach {f_top:.0f} Hz, above fs_sim/2 = {fs_sim / 2:g} Hz; "
                              "raise simulation.fs_sim")

    parts = cfg["dataset"]["partitions"]
    if parts is not None:
        if (not isinstance(parts, list) or len(parts) < 2
                or not all(_is_number(p) and p > 0 for p in parts)):
            raise ConfigError("dataset.partitions must be null or a list of >= 2 positive fractions")
