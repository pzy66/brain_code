#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_TOOL = REPO_ROOT / "hybrid_controller" / "tools" / "debug_vision_grasp_flow.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "vision_grasp_flow_debug"


MODE_ARGS: dict[str, tuple[str, ...]] = {
    "dry-run": ("--no-ros",),
    "camera-only": ("--no-ros", "--max-steps", "1"),
    "resolve-only": (),
    "move-only": (),
    "execute-move": ("--execute",),
    "allow-pick": ("--execute", "--allow-pick"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone 04 debug wrapper for camera recognition -> grasp command validation."
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_ARGS),
        default="dry-run",
        help="Safety mode. PICK requires allow-pick.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory. Defaults under artifacts/vision_grasp_flow_debug/.",
    )
    parser.add_argument(
        "tool_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the underlying debug tool. Prefix with -- when needed.",
    )
    return parser


def _strip_separator(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not LEGACY_TOOL.exists():
        raise FileNotFoundError(f"Underlying debug tool not found: {LEGACY_TOOL}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else DEFAULT_OUTPUT_ROOT / str(args.mode)
    forwarded = [
        *MODE_ARGS[str(args.mode)],
        "--output-dir",
        str(output_dir),
        *_strip_separator(list(args.tool_args)),
    ]

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(LEGACY_TOOL), *forwarded]
        runpy.run_path(str(LEGACY_TOOL), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return int(code)
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = previous_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
