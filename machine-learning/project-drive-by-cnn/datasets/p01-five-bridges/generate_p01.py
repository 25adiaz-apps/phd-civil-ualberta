"""Regenerate every data file of p01-five-bridges.

For each set and package the crossings are simulated once with 200 m of road before and after the
bridge (ragged layout). The other three files are derived from it without simulating again, so all
four files hold exactly the same crossings:

    <package>_200m_ragged.h5   simulated
    <package>_200m_padded.h5   same, zero-padded to 19000 samples (95 s)
    <package>_100m_ragged.h5   cropped to 100 m before and after the bridge
    <package>_100m_padded.h5   cropped, zero-padded to 11000 samples (55 s)

Usage (from the repository .venv):  python generate_p01.py [--workers N] [--overwrite]
"""
import argparse
import sys
from pathlib import Path

import vehicle_bridge_signal as vbs

HERE = Path(__file__).resolve().parent
SETS = ("set-fixed-roughness", "set-variable-roughness")
PACKAGES = {"main": ("main.yaml", 1), "generalization": ("generalization.yaml", 2)}   # config, base seed
PAD = {200: 19000, 100: 11000}   # longest crossing: (2 x 200 m + 68 m) / 5 m/s = 93.6 s -> 18721 samples


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=0, help="processes (0 = all CPUs); the data do not depend on it")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    command = " ".join(sys.argv)
    for set_name in SETS:
        data = HERE / set_name / "data"
        for package, (config, seed) in PACKAGES.items():
            print(f"== {set_name} / {package}", flush=True)
            base = data / f"{package}_200m_ragged.h5"
            vbs.generate_dataset(HERE / set_name / config, None, base, seed=seed, workers=args.workers,
                                 overwrite=args.overwrite, command=command)
            opts = dict(overwrite=args.overwrite, progress=False, command=command)
            vbs.derive_dataset(base, data / f"{package}_200m_padded.h5", layout="padded", pad_to_samples=PAD[200],
                               **opts)
            vbs.derive_dataset(base, data / f"{package}_100m_ragged.h5", 100.0, 100.0, layout="ragged", **opts)
            vbs.derive_dataset(base, data / f"{package}_100m_padded.h5", 100.0, 100.0, layout="padded",
                               pad_to_samples=PAD[100], **opts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
