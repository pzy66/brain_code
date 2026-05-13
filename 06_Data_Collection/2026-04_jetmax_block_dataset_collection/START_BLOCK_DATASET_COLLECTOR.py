#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
COLLECTOR_SCRIPT = SCRIPT_DIR / "block_dataset_collector.py"
REQUIRED_IMPORTS = ("PyQt5", "cv2", "numpy")


def _candidate_interpreters() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("BRAIN_PYTHON_EXE", "").strip()
    if override:
        candidates.append(Path(override).expanduser())

    candidates.append(PROJECT_ROOT / ".venv" / "python.exe")
    candidates.append(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    candidates.append(PROJECT_ROOT / ".venv" / "bin" / "python")

    home = Path.home()
    candidates.extend(
        [
            home / "miniconda3" / "envs" / "brain-vision" / "python.exe",
            home / "miniconda3" / "envs" / "brain-vision" / "bin" / "python",
            home / "anaconda3" / "envs" / "brain-vision" / "python.exe",
            home / "anaconda3" / "envs" / "brain-vision" / "bin" / "python",
            home / "mambaforge" / "envs" / "brain-vision" / "python.exe",
            home / "mambaforge" / "envs" / "brain-vision" / "bin" / "python",
            Path(sys.executable),
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _check_interpreter(python_exe: Path) -> tuple[bool, str]:
    probe = "; ".join(f"import {name}" for name in REQUIRED_IMPORTS)
    command = [str(python_exe), "-c", probe]
    try:
        result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=8)
    except Exception as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, "ok"
    message = (result.stderr or result.stdout or "").strip()
    return False, message or f"exit code {result.returncode}"


def _resolve_python() -> tuple[Path | None, list[str]]:
    checked: list[str] = []
    for candidate in _candidate_interpreters():
        if not candidate.exists():
            checked.append(f"missing: {candidate}")
            continue
        resolved = candidate.resolve()
        ok, message = _check_interpreter(resolved)
        if ok:
            return resolved, checked
        checked.append(f"invalid: {resolved} -> {message}")
    return None, checked


def _print_startup_error(message: str, details: list[str]) -> None:
    print("[Block Dataset Collector] startup failed", file=sys.stderr)
    print(message, file=sys.stderr)
    if details:
        print("", file=sys.stderr)
        print("Checked interpreters:", file=sys.stderr)
        for detail in details:
            print(f"  - {detail}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Set BRAIN_PYTHON_EXE to a Python environment with PyQt5, opencv-python, and numpy.", file=sys.stderr)


def _maybe_wait_for_double_click(exit_code: int) -> None:
    if exit_code == 0:
        return
    if sys.stdin is not None and sys.stdin.isatty():
        try:
            input("Press Enter to close...")
        except EOFError:
            pass


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not COLLECTOR_SCRIPT.exists():
        _print_startup_error(f"Collector script not found: {COLLECTOR_SCRIPT}", [])
        return 1

    python_exe, checked = _resolve_python()
    if python_exe is None:
        _print_startup_error("No usable Python interpreter was found.", checked)
        _maybe_wait_for_double_click(1)
        return 1

    command = [str(python_exe), str(COLLECTOR_SCRIPT), *args]
    exit_code = int(subprocess.call(command, cwd=str(PROJECT_ROOT)))
    _maybe_wait_for_double_click(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
