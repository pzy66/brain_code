from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


THIS_DIR = Path(__file__).resolve().parent
ASYNC_DIR = THIS_DIR / "2026-04_async_fbcca_idle_decoder"
TARGET_SCRIPT = ASYNC_DIR / "ssvep_training_evaluation_ui.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direct entry: training/evaluation UI from collected dataset."
    )
    parser.add_argument("--serial-port", type=str, default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--board-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--freqs", type=str, default="8,10,12,15", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-manifest", type=str, default="", help="session1 manifest path")
    parser.add_argument("--dataset-manifest-session2", type=str, default="", help="session2 manifest path")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="forward remaining args to ssvep_training_evaluation_ui.py",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    script = TARGET_SCRIPT
    if not script.exists():
        raise FileNotFoundError(f"missing script: {script}")
    cmd = [
        sys.executable,
        str(script),
    ]
    if str(args.dataset_manifest).strip():
        cmd.extend(["--dataset-manifest", str(args.dataset_manifest).strip()])
    if str(args.dataset_manifest_session2).strip():
        cmd.extend(["--dataset-manifest-session2", str(args.dataset_manifest_session2).strip()])
    if args.extra_args:
        cmd.extend(list(args.extra_args))
    print(f"[launcher] {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=str(THIS_DIR))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
