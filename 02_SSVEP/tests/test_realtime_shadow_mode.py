from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from apps.realtime_online_ui import RealtimeOnlineWindow, build_parser
from ssvep_core.async_fbcca_idle_standalone import default_profile, save_profile
from ssvep_core.runtime_shadow import build_shadow_runtime_chain


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


def test_shadow_runtime_treats_all_zero_logreg_v2_as_global_gate(tmp_path: Path) -> None:
    profile = default_profile((8.0, 10.0, 12.0, 15.0))
    profile_path = tmp_path / "fbcca_profile.json"
    save_profile(profile, profile_path)
    profile_path.with_name("fbcca_profile_v2.json").write_text(
        json.dumps(
            {
                "version": "2.0",
                "gate": {
                    "type": "frequency_specific_logreg",
                    "feature_names": ["top1_score", "ratio"],
                    "per_freq": {
                        "8": {"coef": [0.0, 0.0], "intercept": 0.0},
                        "10": {"coef": [0.0, 0.0], "intercept": 0.0},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    chain, summary = build_shadow_runtime_chain(profile=profile, profile_path=profile_path)

    assert chain.gate.name == "global_gate"
    assert summary["gate_mode"] == "global_gate"
    assert summary["profile_v2_gate_type"] == "frequency_specific_logreg"
    assert summary["profile_v2_effective_gate_type"] == "threshold_only_global_gate"
