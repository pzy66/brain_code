from __future__ import annotations

import sys

from hybrid_controller.debug.fake_robot_server import main as fake_robot_main


DEFAULT_ARGS = [
    "--host",
    "127.0.0.1",
    "--port",
    "8899",
    "--timing-profile",
    "fast",
    "--scenario-name",
    "basic",
    "--vision-mode",
    "fixed_cyl_slots",
]


def main(argv: list[str] | None = None) -> int:
    extra_args = sys.argv[1:] if argv is None else list(argv)
    return int(fake_robot_main([*DEFAULT_ARGS, *extra_args]))


if __name__ == "__main__":
    raise SystemExit(main())
