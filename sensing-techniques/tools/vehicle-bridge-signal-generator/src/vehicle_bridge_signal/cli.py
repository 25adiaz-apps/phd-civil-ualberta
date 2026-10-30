"""Command-line interface: ``vbsig <command>`` or ``python -m vehicle_bridge_signal <command>``.

Commands
--------
generate   generate an HDF5 dataset (or ``--preview N`` samples as a figure, without a dataset)
simulate   simulate one fixed scenario, print its summary, optionally plot it / save it (.npz)
presets    list the presets and what they are for
info       summary of an existing dataset
check      load and validate a configuration, list its random parameters
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import warnings
from pathlib import Path

import numpy as np

COMMANDS = ("generate", "simulate", "presets", "info", "check")


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", required=True,
                   help="preset name (e.g. smoke) or path to a YAML configuration")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a parameter, e.g. --set crossing.speed=12 (repeatable; value read as YAML)")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vbsig", description="Vehicle-bridge (drive-by) smartphone signal generator")
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a dataset (HDF5) or a preview figure")
    _add_config_args(g)
    g.add_argument("--n-samples", "-n", type=int, help="number of samples (default: dataset.n_samples of the config)")
    g.add_argument("--out", "-o", help="output .h5 file (inside the project that uses the data)")
    g.add_argument("--seed", type=int, default=0, help="base seed (default 0)")
    g.add_argument("--workers", "-j", type=int, default=1, help="processes (0 = all CPUs); result is identical")
    g.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    g.add_argument("--allow-inside-tool", action="store_true", help="allow writing inside the tool folder")
    g.add_argument("--preview", type=int, metavar="N", help="simulate N samples and plot them; no dataset is written")
    g.add_argument("--plot", default="vbsig_preview.png", help="figure path for --preview")
    g.add_argument("--quiet", action="store_true", help="no progress messages")

    s = sub.add_parser("simulate", help="simulate one fixed scenario and plot it")
    _add_config_args(s)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--sample", action="store_true",
                   help="the configuration has distributions: draw one scenario from it with --seed first")
    s.add_argument("--plot", help="save the component plot here (.png/.pdf); spectra go to <name>_psd.<ext>")
    s.add_argument("--save", help="save the arrays and metadata to a .npz file")

    sub.add_parser("presets", help="list presets")
    i = sub.add_parser("info", help="summary of a dataset")
    i.add_argument("path")
    c = sub.add_parser("check", help="validate a configuration")
    _add_config_args(c)
    return ap


def _summary(res) -> str:
    m = res.metadata
    lines = [
        f"bridge   L={m['span']:.2f} m  f1={m['f1']:.3f} Hz  modes={m['n_modes']}  m={m['mass_per_length']:.0f} kg/m",
        f"vehicle  {m['vehicle_model']}  mass={m['vehicle_mass']:.1f} kg  freqs={np.round(m['vehicle_frequencies'], 3)} Hz"
        f"  mu={m['mass_ratio']:.4f}",
        f"crossing v={m['speed']:.2f} m/s  entry={m['entry_time']:.2f} s  exit={m['exit_time']:.2f} s  "
        f"v/(2L)={m['driving_frequency']:.3f} Hz",
        "RMS on the bridge [m/s^2]: bridge={:.3e}  roughness={:.3e}  engine={:.3e}  wheel={:.3e}  noise={:.3e}".format(
            m["rms_bridge_on"], m["rms_roughness_on"], m["rms_engine_on"], m["rms_wheel_on"], m["rms_noise_on"]),
        "bridge/disturbance [dB]: roughness={:.1f}  engine={:.1f}  wheel={:.1f}  noise={:.1f}".format(
            m["ratio_db_bridge_to_roughness"], m["ratio_db_bridge_to_engine"], m["ratio_db_bridge_to_wheel"],
            m["ratio_db_bridge_to_noise"]),
        f"identity |measured - sum(components)| / rms = {res.identity_error():.2e}",
    ]
    if res.warnings:
        lines.append("warnings: " + " | ".join(res.warnings))
    return "\n".join(lines)


def _cmd_generate(args) -> int:
    from .config import load_config
    cfg = load_config(args.config, args.overrides)
    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        from .plotting import plot_preview
        from .sampling import plan_dataset, scenario_for
        from .simulate import simulate
        results = []
        for plan in plan_dataset(cfg, args.preview, args.seed):
            scenario, _ = scenario_for(cfg, plan, args.seed)
            res = simulate(scenario, plan.sample_seed)
            results.append(res)
            print(f"--- sample {plan.index} (bridge_id={plan.bridge_id}, vehicle_id={plan.vehicle_id})\n{_summary(res)}")
        fig = plot_preview(results)
        fig.savefig(args.plot, dpi=130)
        print(f"preview saved to {Path(args.plot).resolve()}")
        return 0
    if args.out is None:
        print("error: generate needs --out (or --preview N)", file=sys.stderr)
        return 2
    from .dataset import generate_dataset
    path = generate_dataset(cfg, args.n_samples, args.out, args.seed, args.workers, overwrite=args.overwrite,
                            progress=not args.quiet, allow_inside_tool=args.allow_inside_tool,
                            command=" ".join(shlex.quote(a) for a in sys.argv))
    print(f"dataset written to {path}")
    return 0


def _cmd_simulate(args) -> int:
    from .config import load_config
    from .simulate import simulate
    cfg = load_config(args.config, args.overrides)
    seed = args.seed
    if args.sample:
        from .sampling import sample_scenario
        cfg, seed = sample_scenario(cfg, args.seed)
    res = simulate(cfg, seed)
    print(_summary(res))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        from .plotting import plot_component_spectra, plot_components
        out = Path(args.plot)
        plot_components(res).savefig(out, dpi=130)
        plot_component_spectra(res).savefig(out.with_name(out.stem + "_psd" + out.suffix), dpi=130)
        print(f"figures saved to {out.resolve()} and {out.with_name(out.stem + '_psd' + out.suffix).resolve()}")
    if args.save:
        arrays = {"t": res.t, "measured": res.measured, "on_bridge": res.on_bridge,
                  **{f"component_{k}": v for k, v in res.components.items()},
                  **{f"contact_{k}": v for k, v in res.contact.items()},
                  "bridge_midspan_displacement": res.bridge_midspan_displacement,
                  "bridge_midspan_acceleration": res.bridge_midspan_acceleration}
        meta = json.dumps(res.metadata, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        np.savez_compressed(args.save, metadata_json=meta, seed=seed, **arrays)
        print(f"arrays saved to {Path(args.save).resolve()}")
    return 0


def _cmd_presets(_args) -> int:
    from .config import PRESETS_DIR, list_presets
    presets = list_presets()
    if not presets:
        print(f"No presets found in {PRESETS_DIR}")
        return 1
    for p in presets:
        print(f"{p['name']:<28} {p['description']}")
    return 0


def _cmd_info(args) -> int:
    from .dataset import load_dataset
    with load_dataset(args.path) as ds:
        a = ds.attrs
        meta = ds.metadata
        print(repr(ds))
        for k in ("schema_version", "created_utc", "git_commit", "git_dirty", "base_seed", "vehicle_model",
                  "interaction", "amplitude_mode", "n_axles", "generation_seconds", "workers", "command"):
            if k in a:
                print(f"  {k:<20} {a[k]}")
        print(f"  {'bridges':<20} {len(np.unique(meta['bridge_id']))}   vehicles {len(np.unique(meta['vehicle_id']))}")
        for k in ("span", "f1", "speed", "mass_ratio", "ratio_db_bridge_to_roughness", "ratio_db_bridge_to_noise"):
            if k in meta:
                v = np.asarray(meta[k], float)
                v = v[np.isfinite(v)]
                if v.size:
                    print(f"  {k:<30} min {v.min():10.4g}  median {np.median(v):10.4g}  max {v.max():10.4g}")
        lift = np.asarray(meta.get("wheel_lift_off", []), bool)
        if lift.size:
            print(f"  samples with wheel lift-off: {int(lift.sum())}")
    return 0


def _cmd_check(args) -> int:
    from .config import config_to_yaml, distribution_paths, load_config
    cfg = load_config(args.config, args.overrides)
    rel = distribution_paths(cfg, relevant_only=True)
    print(f"configuration '{cfg.get('name')}' is valid.")
    print("random parameters used by the simulation:" if rel else "all parameters used by the simulation are fixed.")
    for p in rel:
        print(f"  {p}")
    if not rel:
        print("(simulate can run it directly)")
    print("\nresolved configuration:\n" + config_to_yaml(cfg))
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = _parser().parse_args(argv)
    from .config import ConfigError
    handler = {"generate": _cmd_generate, "simulate": _cmd_simulate, "presets": _cmd_presets, "info": _cmd_info,
               "check": _cmd_check}[args.command]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            return handler(args)
    except (ConfigError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
