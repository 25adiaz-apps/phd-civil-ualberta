"""``python -m vehicle_bridge_signal.generate``: dataset generation entry point.

``python -m vehicle_bridge_signal.generate --config default_dataset --n-samples 10000 --out data/v1.h5 --workers 8``
is the same as ``vbsig generate ...``. If the first argument is another command (``simulate``,
``presets``, ``info``, ``check``) it is dispatched to it, e.g.
``python -m vehicle_bridge_signal.generate simulate --config typical_car_smartphone --plot fig.png``.
"""
from __future__ import annotations

import sys

from .cli import COMMANDS, main as cli_main


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in COMMANDS:
        return cli_main(argv)
    if argv and argv[0] in ("-h", "--help"):
        return cli_main(["generate", "--help"])
    return cli_main(["generate", *argv])


if __name__ == "__main__":
    sys.exit(main())
