#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
COLLECTOR_SCRIPT = SCRIPT_DIR / "block_dataset_collector.py"


def _candidate_interpreters() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("BRAIN_PYTHON_EXE", "").strip()
    if override:
        candidates.append(Path(override).expanduser())

    candidates.append(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")

    home = Path.home()
    candidates.extend(
        [
            home / "miniconda3" / "envs" / "brain-vision" / "python.exe",
            home / "anaconda3" / "envs" / "brain-vision" / "python.exe",
            home / "mambaforge" / "envs" / "brain-vision" / "python.exe",
        ]
    )
    return candidates


def _resolve_python() -> Path:
    for candidate in _candidate_interpreters():
        if candidate.exists():
            return candidate.resolve()
    return Path(sys.executable).resolve()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not COLLECTOR_SCRIPT.exists():
        print(f"Collector script not found: {COLLECTOR_SCRIPT}", file=sys.stderr)
        return 1

    python_exe = _resolve_python()
    command = [str(python_exe), str(COLLECTOR_SCRIPT), *args]
    return int(subprocess.call(command, cwd=str(PROJECT_ROOT)))


if __name__ == "__main__":
    raise SystemExit(main())
