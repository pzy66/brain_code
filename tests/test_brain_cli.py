from __future__ import annotations

import subprocess
import sys

from brain_workspace.paths import BRAIN_CODE_ROOT


def _run_brain(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "brain", *args],
        cwd=BRAIN_CODE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_brain_module_help_runs_without_hardware() -> None:
    result = _run_brain("--help")

    assert result.returncode == 0
    assert "Unified brain-code command line" in result.stdout


def test_brain_diagnose_runs_without_hardware_or_assets() -> None:
    result = _run_brain("diagnose")

    assert result.returncode == 0
    assert "datasets_root=" in result.stdout
    assert "missing_optional_asset[vision_model]=" in result.stdout


def test_brain_launch_simulate_runs_without_hardware_or_gui() -> None:
    result = _run_brain("launch", "--simulate")

    assert result.returncode == 0
    assert "launch_target=unified" in result.stdout
    assert "simulate=true" in result.stdout
