from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ARGS = [
    "--input-profile",
    "operator_keyboard",
    "--robot-mode",
    "real",
    "--robot-transport",
    "ros",
    "--robot-host",
    "192.168.149.1",
    "--robot-port",
    "8888",
    "--rosbridge-port",
    "9091",
    "--vision-mode",
    "robot_camera_detection",
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


def _candidate_brain_code_pythons() -> tuple[Path, ...]:
    if os.name == "nt":
        python_parts = ("python.exe",)
        venv_dir = "Scripts"
    else:
        python_parts = ("python",)
        venv_dir = "bin"
    home = Path.home()
    candidates = [PROJECT_ROOT / ".venv" / venv_dir / python_parts[0]]
    for root_name in ("miniconda3", "anaconda3", "mambaforge"):
        candidates.append(home / root_name / "envs" / "brain_code" / venv_dir / python_parts[0])
    return tuple(candidates)


def _enforce_brain_code_interpreter() -> None:
    override = os.environ.get("BRAIN_PYTHON_EXE", "").strip()
    if override:
        override_path = Path(override).expanduser()
        if not override_path.exists():
            raise SystemExit(
                "Interpreter mismatch.\n"
                f"BRAIN_PYTHON_EXE is set but missing: {override_path}\n"
                "Please fix BRAIN_PYTHON_EXE or switch PyCharm interpreter to the repo .venv or your brain_code environment."
            )
        expected = override_path.resolve()
    else:
        candidates = _candidate_brain_code_pythons()
        expected = next((path.resolve() for path in candidates if path.exists()), None)
    if expected is None:
        return
    current = Path(sys.executable).resolve()
    if current == expected:
        return
    raise SystemExit(
        "Interpreter mismatch.\n"
        f"Current: {current}\n"
        f"Expected: {expected}\n"
        "Please switch PyCharm interpreter to the repo .venv or your brain_code environment and run again.\n"
        "Optional override: set BRAIN_PYTHON_EXE to an absolute python.exe path."
    )


def _normalize_legacy_rosbridge_port(args: list[str]) -> list[str]:
    allow_9092 = str(os.environ.get("HYBRID_ALLOW_ROSBRIDGE_9092", "")).strip().lower() in {"1", "true", "yes", "on"}
    if allow_9092:
        return list(args)
    normalized = list(args)
    replaced = False
    index = 0
    while index < len(normalized):
        token = str(normalized[index])
        if token == "--rosbridge-port" and (index + 1) < len(normalized):
            if str(normalized[index + 1]).strip() == "9092":
                normalized[index + 1] = "9091"
                replaced = True
            index += 2
            continue
        index += 1
    if replaced:
        print("[compat] --rosbridge-port 9092 is deprecated; auto-switched to 9091.", flush=True)
    return normalized


def main(argv: list[str] | None = None) -> int:
    _enforce_brain_code_interpreter()
    from hybrid_controller.app import main as app_main

    extra_args = sys.argv[1:] if argv is None else list(argv)
    extra_args = _normalize_legacy_rosbridge_port(extra_args)
    if "--robot-mode" in extra_args:
        try:
            mode = extra_args[extra_args.index("--robot-mode") + 1]
        except IndexError:
            mode = None
        if mode == "fake" and "--robot-host" not in extra_args:
            extra_args.extend(["--robot-host", "127.0.0.1"])
    return int(app_main([*DEFAULT_ARGS, *extra_args]))


if __name__ == "__main__":
    raise SystemExit(main())
