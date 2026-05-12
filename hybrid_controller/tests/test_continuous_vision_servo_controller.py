import pytest

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


def test_continuous_servo_holds_low_confidence_fragment_without_publishing_motion() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            confidence=0.29,
            area_px=479,
            bbox=[398, 108, 428, 128],
            center_distance_px=152.0,
            servo_command_point=[26.7, 192.5],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(25.5, 160.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "hold"
    assert decision.pending is not None
    assert decision.pending.stable_frames == 0
    assert decision.trace["quality_reason"] == "target_confidence_low"


def test_continuous_servo_stops_on_target_center_jump() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            pixel_center=[211, 253],
            center_distance_px=109.8,
            servo_command_point=[0.0, 150.0],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
        pending={
            "slot_id": 1,
            "stable_frames": 1,
            "last_center_px": [326, 237],
            "last_center_distance_px": 6.7,
        },
        current_cyl_pose=(8.76, 162.5, 210.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "target_center_jump"
    assert decision.pending is not None
    assert decision.pending.last_center_px == (326.0, 237.0)


def test_continuous_servo_records_tracking_center_in_pending() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(pixel_center=[324, 238], center_distance_px=4.5),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.pending is not None
    assert decision.pending.last_center_px == (324.0, 238.0)
    assert decision.pending.last_center_distance_px == 4.5


def test_continuous_servo_tracks_grasp_pixel_when_configured() -> None:
    controller = ContinuousVisionServoController(AppConfig(vision_servo_measurement_point="grasp"))

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            pixel_center=[319, 222],
            grasp_pixel=[314, 235],
            center_distance_px=7.8,
            servo_command_point=[0.1, 151.0],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
        pending={
            "slot_id": 1,
            "stable_frames": 1,
            "last_center_px": [314, 235],
            "last_center_distance_px": 7.8,
        },
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.pending is not None
    assert decision.pending.last_center_px == (314.0, 235.0)


def test_continuous_servo_tracks_geometry_center_when_configured() -> None:
    controller = ContinuousVisionServoController(AppConfig(vision_servo_measurement_point="geometry"))

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            pixel_center=[319, 222],
            geometry_center=[322, 239],
            grasp_pixel=[314, 235],
            center_distance_px=2.2,
            servo_command_point=[0.1, 151.0],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
        pending={
            "slot_id": 1,
            "stable_frames": 1,
            "last_center_px": [322, 239],
            "last_center_distance_px": 2.2,
        },
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.pending is not None
    assert decision.pending.last_center_px == (322.0, 239.0)


def test_continuous_servo_tracks_subpixel_geometry_center_by_default() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            pixel_center=[316, 235],
            geometry_center=[316, 235],
            geometry_center_f=[318.8, 238.6],
            grasp_pixel=[302, 235],
            center_distance_px=1.84,
            servo_command_point=[0.1, 151.0],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
        pending={
            "slot_id": 1,
            "stable_frames": 1,
            "last_center_px": [318.8, 238.6],
            "last_center_distance_px": 1.84,
        },
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "SERVO"
    assert decision.pending is not None
    assert decision.pending.last_center_px == pytest.approx((318.8, 238.6))


def test_continuous_servo_holds_near_center_when_grasp_is_temporarily_unstable() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            valid=True,
            actionable=False,
            invalid_reason="grasp_unstable",
            center_distance_px=6.0,
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


def test_continuous_servo_holds_first_near_center_unstable_frame_at_high_view() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            valid=True,
            actionable=False,
            invalid_reason="grasp_unstable",
            center_distance_px=13.6,
            servo_command_point=None,
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, 210.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "hold"
    assert decision.pending is not None
    assert decision.pending.stable_frames == 0


def test_continuous_servo_refines_low_unstable_target_without_descending() -> None:
    controller = ContinuousVisionServoController(AppConfig(vision_continuous_servo_center_allow_descent_px=2.0))

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            valid=True,
            actionable=False,
            invalid_reason="grasp_unstable",
            center_distance_px=14.0,
            servo_command_point=None,
            resolved_cyl=[1.2, 154.0, 85.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, 196.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.z_rate_mm_s == 0.0
    assert decision.theta_rate_deg_s > 0.0
    assert decision.radius_rate_mm_s > 0.0


def test_continuous_servo_refines_low_unstable_target_from_raw_delta() -> None:
    controller = ContinuousVisionServoController(AppConfig(vision_continuous_servo_center_allow_descent_px=2.0))

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            valid=True,
            actionable=False,
            invalid_reason="grasp_unstable",
            center_distance_px=14.0,
            servo_command_point=None,
            camera_to_world_raw=[-0.2, -5.0, 0.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(8.8, 168.4, 196.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.z_rate_mm_s == 0.0


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
    assert decision.reason == "grasp_unstable"


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
    assert decision.pending is not None
    assert decision.pending.descent_anchor_z_mm == 190.0


def test_continuous_servo_pauses_descent_after_one_z_pulse() -> None:
    config = AppConfig(vision_continuous_servo_z_pulse_mm=8.0)
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0, servo_command_point=[0.5, 151.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 5, "descent_anchor_z_mm": 190.0},
        current_cyl_pose=(0.0, 150.0, 181.9),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s == 0.0
    assert decision.pending is not None
    assert decision.pending.descent_anchor_z_mm is None
    assert decision.pending.descent_cooldown_frames > 0


def test_continuous_servo_does_not_descend_outside_configured_descent_band() -> None:
    controller = ContinuousVisionServoController(AppConfig(vision_continuous_servo_center_allow_descent_px=2.0))

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=6.0, servo_command_point=[0.5, 151.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 4},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s == 0.0


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


def test_continuous_servo_stops_on_too_dark_frame_before_motion() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0),
        packet={
            "frame_id": 1,
            "queue_age_ms": 10.0,
            "frame_block_reason": "frame_too_dark",
            "frame_quality": {"too_dark": True, "gray_mean": 16.0, "gray_p95": 16.0},
        },
        pending={"slot_id": 1, "stable_frames": 1},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "frame_too_dark"
    assert decision.theta_rate_deg_s == 0.0
    assert decision.radius_rate_mm_s == 0.0
    assert decision.z_rate_mm_s == 0.0


def test_continuous_servo_ignores_preprocess_age_for_staleness() -> None:
    controller = ContinuousVisionServoController(AppConfig(vision_continuous_servo_command_timeout_ms=100.0))

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0),
        packet={"frame_id": 1, "latest_frame_preprocess_age_ms": 2500.0, "queue_age_ms": 10.0},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.reason != "frame_stale"


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
            center_distance_px=1.5,
            command="PICK_WORLD 0.00 -150.00",
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 1, "pick_ready_frames": 1},
        current_cyl_pose=(5.0, 150.0, config.vision_pick_confirm_z_mm),
    )

    assert decision.action == "PICK_READY"
    assert decision.command == "PICK_CYL 5.00 190.00"


def test_continuous_servo_requires_strict_pick_ready_frames_independent_of_descent_stability() -> None:
    config = AppConfig(
        pick_tool_offset_source="command_bias",
        vision_eye_in_hand_pick_radius_bias_mm=40.0,
        vision_continuous_servo_pick_ready_center_px=2.0,
        vision_continuous_servo_stable_frames=2,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            actionable=True,
            center_distance_px=1.5,
            command="PICK_WORLD 0.00 -150.00",
            servo_command_point=[5.0, 150.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 8, "pick_ready_frames": 0},
        current_cyl_pose=(5.0, 150.0, config.vision_pick_confirm_z_mm),
    )

    assert decision.action == "STOP"
    assert decision.reason == "hold"
    assert decision.command is None
    assert decision.pending is not None
    assert decision.pending.stable_frames == 9
    assert decision.pending.pick_ready_frames == 1


def test_continuous_servo_does_not_pick_ready_until_strictly_centered_at_confirm_height() -> None:
    config = AppConfig(
        pick_tool_offset_source="command_bias",
        vision_eye_in_hand_pick_radius_bias_mm=40.0,
        vision_continuous_servo_pick_ready_center_px=8.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            actionable=True,
            center_distance_px=10.4,
            command="PICK_WORLD 0.00 -150.00",
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 3},
        current_cyl_pose=(5.0, 150.0, config.vision_pick_confirm_z_mm),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.command is None
    assert decision.pending is not None
    assert decision.pending.stable_frames == 0


def test_continuous_servo_refines_actionable_slot_until_strictly_centered() -> None:
    config = AppConfig(
        pick_tool_offset_source="command_bias",
        vision_continuous_servo_pick_ready_center_px=2.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            actionable=True,
            center_distance_px=13.9,
            command="PICK_WORLD 0.00 -150.00",
            servo_command_point=None,
            resolved_cyl=[6.5, 154.0, 85.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 3},
        current_cyl_pose=(5.0, 150.0, config.vision_pick_confirm_z_mm),
    )

    assert decision.action == "SERVO"
    assert decision.command is None
    assert decision.theta_rate_deg_s > 0.0
    assert decision.radius_rate_mm_s > 0.0
    assert decision.z_rate_mm_s == 0.0


def test_continuous_servo_scales_low_height_fine_rates_near_confirm_height() -> None:
    base_config = AppConfig(
        vision_continuous_servo_pick_ready_center_px=2.0,
        vision_continuous_servo_low_height_fine_band_px=20.0,
        vision_continuous_servo_low_height_fine_rate_scale=0.25,
    )
    high_decision = ContinuousVisionServoController(base_config).decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=12.0, servo_command_point=[2.0, 154.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, base_config.vision_pick_confirm_z_mm + 40.0),
    )
    low_decision = ContinuousVisionServoController(base_config).decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=12.0, servo_command_point=[2.0, 154.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, base_config.vision_pick_confirm_z_mm),
    )

    assert high_decision.action == "SERVO"
    assert low_decision.action == "SERVO"
    assert low_decision.theta_rate_deg_s == high_decision.theta_rate_deg_s * 0.25
    assert low_decision.radius_rate_mm_s == high_decision.radius_rate_mm_s * 0.25


def test_continuous_servo_default_low_height_band_slows_visible_low_offset() -> None:
    config = AppConfig(vision_continuous_servo_pick_ready_center_px=2.0)
    high_decision = ContinuousVisionServoController(config).decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.7, servo_command_point=[2.0, 154.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, config.vision_pick_confirm_z_mm + 40.0),
    )
    low_decision = ContinuousVisionServoController(config).decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.7, servo_command_point=[2.0, 154.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, config.vision_pick_confirm_z_mm),
    )

    assert high_decision.action == "SERVO"
    assert low_decision.action == "SERVO"
    assert low_decision.theta_rate_deg_s == pytest.approx(
        high_decision.theta_rate_deg_s * config.vision_continuous_servo_low_height_fine_rate_scale
    )
    assert low_decision.radius_rate_mm_s == pytest.approx(
        high_decision.radius_rate_mm_s * config.vision_continuous_servo_low_height_fine_rate_scale
    )


def test_continuous_servo_stops_on_low_height_error_rebound() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_fine_band_px=10.0,
        vision_continuous_servo_low_height_error_growth_stop_px=2.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=7.9, servo_command_point=[7.65, 175.0]),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 2, "last_center_distance_px": 2.9},
        current_cyl_pose=(7.65, 176.0, 120.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "low_height_error_rebounded"
    assert decision.theta_rate_deg_s == 0.0
    assert decision.radius_rate_mm_s == 0.0
    assert decision.pending is not None
    assert decision.pending.stable_frames == 0
    assert decision.pending.pick_ready_frames == 0


def test_continuous_servo_stops_to_settle_near_center_at_confirm_height() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_settle_stop_band_px=8.0,
        vision_continuous_servo_stable_frames=2,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=7.2, servo_command_point=[1.0, 154.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, 123.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "settle_near_center"
    assert decision.pending is not None
    assert decision.pending.stable_frames == 1


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
            center_distance_px=1.5,
            command="",
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 1, "pick_ready_frames": 1},
        current_cyl_pose=(5.0, 150.0, config.vision_pick_confirm_z_mm),
    )

    assert decision.action == "STOP"
    assert decision.reason == "pick_command_unavailable"
    assert decision.command is None
