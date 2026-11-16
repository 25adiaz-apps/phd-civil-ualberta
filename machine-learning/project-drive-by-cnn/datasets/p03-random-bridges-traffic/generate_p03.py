"""Regenerate every data file of p03-random-bridges-traffic.

The crossings are simulated once with 100 m of road before and after the bridge (ragged layout) and then
zero-padded without simulating again, so both files hold the same crossings:

    main_ragged.h5   simulated
    main_padded.h5   same, zero-padded to 11000 samples (55 s)

Usage (from the repository .venv):  python generate_p03.py [--workers N] [--overwrite]
"""
import argparse
import sys
from pathlib import Path

import vehicle_bridge_signal as vbs

HERE = Path(__file__).resolve().parent
SEED = 3
PAD = 11000   # longest crossing: (2 x 100 m + 60 m) / 5 m/s = 52 s -> 10400 samples


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=0, help="processes (0 = all CPUs); the data do not depend on it")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    command = " ".join(sys.argv)
    data = HERE / "data"
    base = data / "main_ragged.h5"
    vbs.generate_dataset(HERE / "main.yaml", None, base, seed=SEED, workers=args.workers,
                         overwrite=args.overwrite, command=command)
    vbs.derive_dataset(base, data / "main_padded.h5", layout="padded", pad_to_samples=PAD,
                       overwrite=args.overwrite, progress=False, command=command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
