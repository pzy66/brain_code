from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


THIS_DIR = Path(__file__).resolve().parent
ASYNC_DIR = THIS_DIR / "2026-04_async_fbcca_idle_decoder"
TARGET_SCRIPT = ASYNC_DIR / "ssvep_dataset_collection_ui.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct entry: SSVEP dataset collection UI.")
    parser.add_argument("--serial-port", type=str, default="auto", help="serial port, e.g. COM4, or auto")
    parser.add_argument("--board-id", type=int, default=0)
    parser.add_argument("--freqs", type=str, default="8,10,12,15")
    parser.add_argument("--dataset-dir", type=str, default=str(ASYNC_DIR / "profiles" / "datasets"))
    parser.add_argument("--subject-id", type=str, default="subject001")
    parser.add_argument("--session-id", type=str, default="")
    parser.add_argument("--session-index", type=int, default=1)
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="forward remaining args to ssvep_dataset_collection_ui.py",
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
        "--serial-port",
        str(args.serial_port),
        "--board-id",
        str(int(args.board_id)),
        "--freqs",
        str(args.freqs),
        "--dataset-dir",
        str(args.dataset_dir),
        "--subject-id",
        str(args.subject_id),
        "--session-index",
        str(int(args.session_index)),
    ]
    if str(args.session_id).strip():
        cmd.extend(["--session-id", str(args.session_id).strip()])
    if args.extra_args:
        cmd.extend(list(args.extra_args))
    print(f"[launcher] {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=str(THIS_DIR))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

