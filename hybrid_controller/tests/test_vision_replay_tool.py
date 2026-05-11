from hybrid_controller.config import AppConfig
from hybrid_controller.tools.replay_vision_debug_bundle import replay
from hybrid_controller.vision.grasp_profile import apply_vision_grasp_profile
from hybrid_controller.vision.grasp_profile import load_vision_grasp_profile


def test_replay_vision_debug_bundle_recreates_pick_decision() -> None:
    config = AppConfig().resolved()
    profile = load_vision_grasp_profile(config)
    config = apply_vision_grasp_profile(config, profile).resolved() if profile.ready else config
    confirm_z = config.vision_pick_confirm_z_mm
    result = replay(
        debug={
            "runtime": {},
            "trace": {"slot_id": 1},
            "packet": {
                "mapping_mode": "absolute_base",
                "calibration_ready": True,
                "slots": [
                    {
                        "slot_id": 1,
                        "valid": True,
                        "actionable": True,
                        "command_mode": "world",
                        "command_point": [10.0, -120.0],
                        "camera_to_world_raw": [10.0, -120.0, 0.0],
                    }
                ],
            },
        },
        snapshot_override={
            "robot_xy": [0.0, -120.0],
            "robot_z": confirm_z,
            "robot_cyl": {"theta_deg": 0.0, "radius_mm": 120.0, "z_mm": confirm_z},
        },
    )

    assert result["decision"]["action"] == "PICK"
    assert result["decision"]["command"] == "PICK_CYL 0.00 170.00"
    assert result["metrics"]["valid_count"] == 1
    assert result["metrics"]["actionable_count"] == 1
    assert result["metrics"]["servo_required_count"] == 0
    assert result["metrics"]["invalid_reasons"] == {}


def test_replay_vision_debug_bundle_reports_invalid_reason_metrics() -> None:
    result = replay(
        debug={
            "runtime": {},
            "trace": {"slot_id": 1},
            "packet": {
                "mapping_mode": "delta_servo",
                "calibration_ready": True,
                "slots": [
                    {
                        "slot_id": 1,
                        "valid": True,
                        "actionable": False,
                        "invalid_reason": "vision_servo_required",
                        "camera_to_world_raw": [20.0, 0.0, 0.0],
                        "center_distance_px": 18.0,
                    },
                    {
                        "slot_id": 2,
                        "valid": True,
                        "actionable": False,
                        "invalid_reason": "grasp_quality_low",
                        "camera_to_world_raw": [0.0, 0.0, 0.0],
                        "grasp_quality": 0.1,
                    },
                ],
            },
        },
        snapshot_override={
            "robot_xy": [0.0, -120.0],
            "robot_z": 190.0,
            "robot_cyl": {"theta_deg": 0.0, "radius_mm": 120.0, "z_mm": 190.0},
        },
    )

    assert result["metrics"]["valid_count"] == 2
    assert result["metrics"]["servo_required_count"] == 1
    assert result["metrics"]["invalid_reasons"]["vision_servo_required"] == 1
    assert result["metrics"]["invalid_reasons"]["grasp_quality_low"] == 1
