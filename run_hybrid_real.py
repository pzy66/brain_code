from __future__ import annotations

import sys

from hybrid_controller.app import main as app_main


DEFAULT_ARGS = [
    "--robot-mode",
    "real",
    "--robot-host",
    "192.168.149.1",
    "--robot-port",
    "8888",
    "--vision-mode",
    "fixed_cyl_slots",
    "--move-source",
    "sim",
    "--decision-source",
    "sim",
    "--timing-profile",
    "formal",
    "--scenario-name",
    "basic",
    "--stage-motion-sec",
    "300",
    "--continue-motion-sec",
    "300",
]


def main(argv: list[str] | None = None) -> int:
    extra_args = sys.argv[1:] if argv is None else list(argv)
    return int(app_main([*DEFAULT_ARGS, *extra_args]))


if __name__ == "__main__":
    raise SystemExit(main())
