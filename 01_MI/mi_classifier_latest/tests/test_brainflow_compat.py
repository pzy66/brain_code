from __future__ import annotations

import os
import platform
import struct
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import brainflow_compat


def _expected_board_controller_name() -> str:
    system = platform.system()
    if system == "Windows":
        return "BoardController.dll" if struct.calcsize("P") * 8 == 64 else "BoardController32.dll"
    if system == "Darwin":
        return "libBoardController.dylib"
    return "libBoardController.so"


def test_resource_filename_supports_module_name_inputs() -> None:
    resource_name = f"lib/{_expected_board_controller_name()}"
    resolved = Path(brainflow_compat.resource_filename("brainflow.board_shim", resource_name))
    assert resolved.exists()
    assert resolved.name == _expected_board_controller_name()


def test_repo_root_startup_hook_allows_brainflow_import_without_pkg_resources() -> None:
    env = os.environ.copy()
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "sys.modules.pop('pkg_resources', None); "
            "from brainflow.board_shim import BoardIds; "
            "from brainflow.data_filter import DataFilter; "
            "import brainflow.ml_model as ml_model; "
            "print(int(BoardIds.SYNTHETIC_BOARD))"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "-1"
