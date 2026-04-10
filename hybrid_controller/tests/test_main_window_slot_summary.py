from __future__ import annotations

from hybrid_controller.ui.main_window import MainWindow


def test_format_slot_summary_with_cyl_coordinates() -> None:
    text = MainWindow._format_slot_summary(
        {
            "slot_id": 1,
            "freq_hz": 8.0,
            "cylindrical_center": (-31.7, 128.4),
            "actionable": True,
            "invalid_reason": "",
        }
    )
    assert text == "[1] 8.0Hz theta=-31.7 r=128.4 OK"


def test_format_slot_summary_without_cyl_coordinates() -> None:
    text = MainWindow._format_slot_summary(
        {
            "slot_id": 4,
            "freq_hz": 15.0,
            "actionable": False,
            "invalid_reason": "target_out_of_workspace",
        }
    )
    assert text == "[4] 15.0Hz X:target_out_of_workspace"
