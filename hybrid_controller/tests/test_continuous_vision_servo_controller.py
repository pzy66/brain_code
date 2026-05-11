from hybrid_controller.config import AppConfig
from hybrid_controller.vision.continuous_servo_controller import ContinuousVisionServoController


def _slot(**overrides):
    payload = {
        "slot_id": 1,
        "valid": True,
        "actionable": False,
        "center_distance_px": 80.0,
        "action_tolerance_px": 20.0,
        "servo_command_mode": "cyl",
        "servo_command_point": [10.0, 180.0],
        "invalid_reason": "vision_servo_required",
    }
    payload.update(overrides)
    return payload


def test_continuous_servo_large_error_does_not_descend() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=80.0),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s == 0.0
    assert decision.theta_rate_deg_s > 0.0
    assert decision.radius_rate_mm_s > 0.0


def test_continuous_servo_stops_when_slot_has_non_servo_invalid_reason() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            valid=True,
            actionable=False,
            invalid_reason="vision_mapping_error_high",
            servo_command_point=[0.5, 151.0],
            center_distance_px=8.0,
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "vision_mapping_error_high"
    assert decision.z_rate_mm_s == 0.0


def test_continuous_servo_stops_when_servo_command_is_unavailable() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            valid=True,
            actionable=False,
            invalid_reason="vision_servo_required",
            servo_command_point=None,
            center_distance_px=30.0,
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "servo_command_unavailable"


def test_continuous_servo_holds_near_center_when_grasp_is_temporarily_unstable() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            valid=True,
            actionable=False,
            invalid_reason="grasp_unstable",
            center_distance_px=18.0,
            servo_command_point=None,
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 1},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "hold"
    assert decision.pending is not None
    assert decision.pending.stable_frames == 2


def test_continuous_servo_holds_low_unstable_frame_after_descent() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            valid=True,
            actionable=False,
            invalid_reason="grasp_unstable",
            center_distance_px=89.0,
            servo_command_point=None,
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, 180.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "hold"


def test_continuous_servo_stops_on_large_unstable_jump() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            valid=True,
            actionable=False,
            invalid_reason="grasp_unstable",
            center_distance_px=140.0,
            servo_command_point=None,
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, 180.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "grasp_unstable"


def test_continuous_servo_stops_when_center_distance_is_unavailable() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=None),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "center_distance_unavailable"


def test_continuous_servo_stops_when_robot_pose_is_non_finite() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        current_cyl_pose=(float("nan"), 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "robot_pose_unavailable"


def test_continuous_servo_waits_for_stable_center_before_descent() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0, servo_command_point=[0.5, 151.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s == 0.0


def test_continuous_servo_descends_when_center_is_stable() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0, servo_command_point=[0.5, 151.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 1},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s < 0.0
    assert abs(decision.z_rate_mm_s) <= AppConfig.vision_continuous_servo_z_rate_limit_mm_s


def test_continuous_servo_slows_z_near_confirm_height() -> None:
    config = AppConfig(vision_continuous_servo_z_rate_limit_mm_s=18.0, vision_continuous_servo_z_slow_band_mm=20.0)
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0, servo_command_point=[0.5, 151.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 1},
        current_cyl_pose=(0.0, 150.0, config.vision_pick_confirm_z_mm + 10.0),
    )

    assert decision.action == "SERVO"
    assert 0.0 < abs(decision.z_rate_mm_s) < 18.0


def test_continuous_servo_never_exceeds_configured_low_z_rate_limit() -> None:
    config = AppConfig(vision_continuous_servo_z_rate_limit_mm_s=2.0, vision_continuous_servo_z_slow_band_mm=20.0)
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0, servo_command_point=[0.5, 151.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 1},
        current_cyl_pose=(0.0, 150.0, config.vision_pick_confirm_z_mm + 2.0),
    )

    assert decision.action == "SERVO"
    assert abs(decision.z_rate_mm_s) <= 2.0


def test_continuous_servo_stops_on_stale_frame() -> None:
    controller = ContinuousVisionServoController(AppConfig(vision_continuous_servo_command_timeout_ms=100.0))

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0),
        packet={"frame_id": 1, "queue_age_ms": 120.0},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "frame_stale"


def test_continuous_servo_uses_conservative_max_frame_age() -> None:
    controller = ContinuousVisionServoController(AppConfig(vision_continuous_servo_command_timeout_ms=100.0))

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0),
        packet={"frame_id": 1, "latest_frame_preprocess_age_ms": 10.0, "queue_age_ms": 120.0},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "frame_stale"


def test_continuous_servo_stops_when_frame_age_is_missing() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0),
        packet={"frame_id": 1},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "frame_stale"


def test_continuous_servo_stops_when_frame_age_is_invalid() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0),
        packet={"frame_id": 1, "queue_age_ms": "bad"},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "frame_stale"


def test_continuous_servo_stops_after_lost_frames() -> None:
    controller = ContinuousVisionServoController(AppConfig(vision_continuous_servo_lost_frames=3))

    decision = controller.decide(
        slot_id=1,
        slot_payload=None,
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "lost_frames": 2},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "lost_target"


def test_continuous_servo_reports_pick_ready_at_confirm_height() -> None:
    config = AppConfig(pick_tool_offset_source="command_bias", vision_eye_in_hand_pick_radius_bias_mm=40.0)
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            actionable=True,
            center_distance_px=8.0,
            command="PICK_WORLD 0.00 -150.00",
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 1},
        current_cyl_pose=(5.0, 150.0, config.vision_pick_confirm_z_mm),
    )

    assert decision.action == "PICK_READY"
    assert decision.command == "PICK_CYL 5.00 190.00"


def test_continuous_servo_stops_when_below_confirm_height() -> None:
    config = AppConfig(pick_tool_offset_source="command_bias", vision_eye_in_hand_pick_radius_bias_mm=40.0)
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            actionable=True,
            center_distance_px=8.0,
            command="PICK_WORLD 0.00 -150.00",
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 1},
        current_cyl_pose=(5.0, 150.0, config.vision_pick_confirm_z_mm - config.vision_pick_z_tolerance_mm - 1.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "below_confirm_height"
    assert decision.command is None


def test_continuous_servo_stops_when_pick_command_is_unavailable() -> None:
    config = AppConfig(pick_tool_offset_source="target_pixel")
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            actionable=True,
            center_distance_px=8.0,
            command="",
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 1},
        current_cyl_pose=(5.0, 150.0, config.vision_pick_confirm_z_mm),
    )

    assert decision.action == "STOP"
    assert decision.reason == "pick_command_unavailable"
    assert decision.command is None
