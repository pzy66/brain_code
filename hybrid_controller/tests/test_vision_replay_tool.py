from hybrid_controller.tools.replay_vision_debug_bundle import replay


def test_replay_vision_debug_bundle_recreates_pick_decision() -> None:
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
            "robot_z": 130.0,
            "robot_cyl": {"theta_deg": 0.0, "radius_mm": 120.0, "z_mm": 130.0},
        },
    )

    assert result["decision"]["action"] == "PICK"
    assert result["decision"]["command"] == "PICK_WORLD 10.00 -120.00"
