from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_realtime_online_ui import RealtimeOnlineWindow, build_parser


def _get_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_realtime_parser_supports_shadow_mode_argument() -> None:
    parser = build_parser()
    args = parser.parse_args(["--shadow-mode", "0"])
    assert int(args.shadow_mode) == 0


def test_realtime_window_shadow_mode_default_checked() -> None:
    _ = _get_qapp()
    window = RealtimeOnlineWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))
    try:
        assert bool(window.shadow_mode_check.isChecked()) is True
    finally:
        window.close()

