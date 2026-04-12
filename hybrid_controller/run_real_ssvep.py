from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ARGS = [
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
    "ssvep",
    "--timing-profile",
    "formal",
    "--scenario-name",
    "basic",
    "--stage-motion-sec",
    "300",
    "--continue-motion-sec",
    "300",
]


def _enforce_brain_vision_interpreter() -> None:
    override = os.environ.get("BRAIN_PYTHON_EXE", "").strip()
    if override:
        override_path = Path(override).expanduser()
        if not override_path.exists():
            raise SystemExit(
                "Interpreter mismatch.\n"
                f"BRAIN_PYTHON_EXE is set but missing: {override_path}\n"
                "Please fix BRAIN_PYTHON_EXE or switch PyCharm interpreter to brain-vision."
            )
        expected = override_path.resolve()
    else:
        home = Path.home()
        candidates = (
            home / "miniconda3" / "envs" / "brain-vision" / "python.exe",
            home / "anaconda3" / "envs" / "brain-vision" / "python.exe",
            home / "mambaforge" / "envs" / "brain-vision" / "python.exe",
        )
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
        "Please switch PyCharm interpreter to brain-vision and run again.\n"
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
    _enforce_brain_vision_interpreter()
    from hybrid_controller.app import main as app_main

    extra_args = sys.argv[1:] if argv is None else list(argv)
    extra_args = _normalize_legacy_rosbridge_port(extra_args)
    return int(app_main([*DEFAULT_ARGS, *extra_args]))


if __name__ == "__main__":
    raise SystemExit(main())
