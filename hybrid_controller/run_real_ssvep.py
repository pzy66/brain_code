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


def main(argv: list[str] | None = None) -> int:
    _enforce_brain_vision_interpreter()
    from hybrid_controller.app import main as app_main

    extra_args = sys.argv[1:] if argv is None else list(argv)
    return int(app_main([*DEFAULT_ARGS, *extra_args]))


if __name__ == "__main__":
    raise SystemExit(main())
