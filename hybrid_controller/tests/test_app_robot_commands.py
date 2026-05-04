from __future__ import annotations

from hybrid_controller.app_robot_commands import (
    build_pick_command_from_slot_payload,
    rewrite_pick_command_with_bias,
)


def test_pick_bias_rewrite_preserves_optional_sucker_angle() -> None:
    rewritten = rewrite_pick_command_with_bias(
        "PICK_WORLD 0 -120 25",
        theta_bias_deg=0.0,
        radius_bias_mm=5.0,
        tangent_bias_mm=0.0,
        pick_z_mm=85.0,
    )

    assert rewritten.endswith(" 25.00")
    assert rewritten.startswith("PICK_WORLD ")


def test_slot_pick_command_appends_angle_only_when_quality_passes() -> None:
    slot = {
        "actionable": True,
        "command_mode": "world",
        "command_point": [10.0, -120.0],
        "grasp_angle_deg": 18.0,
        "grasp_angle_quality": 0.8,
        "grasp_angle_quality_threshold": 0.2,
    }

    assert build_pick_command_from_slot_payload(slot) == "PICK_WORLD 10.00 -120.00 18.00"

    slot["grasp_angle_quality"] = 0.1
    assert build_pick_command_from_slot_payload(slot) == "PICK_WORLD 10.00 -120.00"
