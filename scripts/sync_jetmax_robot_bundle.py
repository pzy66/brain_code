from __future__ import annotations

import shutil
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    source_runtime = repo_root / "hybrid_controller" / "integrations" / "robot_runtime_py36.py"
    source_requirements = repo_root / "requirements-jetmax-robot-python.txt"

    bundle_root = repo_root / "jetmax_robot"
    bundle_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_runtime, bundle_root / "robot_runtime_py36.py")
    shutil.copy2(source_requirements, bundle_root / "requirements-jetmax-robot-python.txt")

    print("Synced JetMax robot bundle:")
    print(f"  runtime       -> {bundle_root / 'robot_runtime_py36.py'}")
    print(f"  requirements  -> {bundle_root / 'requirements-jetmax-robot-python.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
