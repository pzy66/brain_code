from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import main


def test_online_cli_returns_nonzero_on_profile_init_error() -> None:
    missing_profile = PROJECT_DIR / "profiles" / "__missing_profile_for_test__.json"
    code = main(
        [
            "online",
            "--serial-port",
            "COM4",
            "--board-id",
            "0",
            "--freqs",
            "8,10,12,15",
            "--profile",
            str(missing_profile),
            "--max-updates",
            "1",
        ]
    )
    assert int(code) == 1
