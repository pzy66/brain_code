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
    controller = ContinuousVisionServoController(
        AppConfig(vision_continuous_servo_horizontal_mode="servo_command_point")
    )

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


def test_continuous_servo_pixel_jacobian_uses_image_error_direction() -> None:
    controller = ContinuousVisionServoController(
        AppConfig(
            vision_continuous_servo_horizontal_mode="pixel_jacobian",
            vision_continuous_servo_pixel_jacobian_gain=0.35,
            vision_continuous_servo_theta_rate_limit_deg_s=8.0,
            vision_continuous_servo_radius_rate_limit_mm_s=10.0,
        ).resolved()
    )

    right_low = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=200.0,
            geometry_center_f=[514.0, 290.0],
            alignment_target_pixel=[320.0, 240.0],
            servo_command_point=[29.0, 185.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 210.0),
    )
    left_high = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=120.0,
            geometry_center_f=[220.0, 210.0],
            alignment_target_pixel=[320.0, 240.0],
            servo_command_point=[29.0, 185.0],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 210.0),
    )

    assert right_low.action == "SERVO"
    assert right_low.theta_rate_deg_s > 0.0
    assert right_low.radius_rate_mm_s > 0.0
    assert left_high.action == "SERVO"
    assert left_high.theta_rate_deg_s < 0.0
    assert left_high.radius_rate_mm_s < 0.0


def test_continuous_servo_ibvs_dls_reduces_predicted_image_error() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="ibvs_dls",
        vision_continuous_servo_ibvs_gain=0.5,
        vision_continuous_servo_ibvs_damping_px_per_unit=2.0,
        vision_continuous_servo_ibvs_du_dtheta_px_per_deg=-10.0,
        vision_continuous_servo_ibvs_du_dradius_px_per_mm=1.5,
        vision_continuous_servo_ibvs_dv_dtheta_px_per_deg=1.0,
        vision_continuous_servo_ibvs_dv_dradius_px_per_mm=4.0,
        vision_continuous_servo_theta_rate_limit_deg_s=20.0,
        vision_continuous_servo_radius_rate_limit_mm_s=20.0,
    ).resolved()
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=58.3,
            geometry_center_f=[372.0, 214.0],
            alignment_target_pixel=[320.0, 240.0],
            servo_command_point=[29.0, 185.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 210.0),
    )

    assert decision.action == "SERVO"
    before = (52.0 * 52.0 + -26.0 * -26.0) ** 0.5
    predicted_u = 52.0 + (
        config.vision_continuous_servo_ibvs_du_dtheta_px_per_deg * decision.theta_rate_deg_s
        + config.vision_continuous_servo_ibvs_du_dradius_px_per_mm * decision.radius_rate_mm_s
    )
    predicted_v = -26.0 + (
        config.vision_continuous_servo_ibvs_dv_dtheta_px_per_deg * decision.theta_rate_deg_s
        + config.vision_continuous_servo_ibvs_dv_dradius_px_per_mm * decision.radius_rate_mm_s
    )
    after = (predicted_u * predicted_u + predicted_v * predicted_v) ** 0.5
    assert after < before
    assert abs(decision.theta_rate_deg_s) <= config.vision_continuous_servo_theta_rate_limit_deg_s
    assert abs(decision.radius_rate_mm_s) <= config.vision_continuous_servo_radius_rate_limit_mm_s
    assert decision.trace["horizontal_mode"] == "ibvs_dls"
    assert decision.trace["ibvs_jacobian_source"] == "config"
    assert decision.trace["ibvs_predicted_error_after_px"] < decision.trace["ibvs_predicted_error_before_px"]


def test_continuous_servo_ibvs_dls_prefers_profile_jacobian_metadata() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="ibvs_dls",
        vision_continuous_servo_ibvs_profile_jacobian=(-10.0, 1.5, 1.0, 4.0),
        vision_continuous_servo_ibvs_jacobian_source="profile_confirm",
        vision_continuous_servo_theta_rate_limit_deg_s=20.0,
        vision_continuous_servo_radius_rate_limit_mm_s=20.0,
    ).resolved()
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=58.3,
            geometry_center_f=[372.0, 214.0],
            alignment_target_pixel=[320.0, 240.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 210.0),
    )

    assert decision.action == "SERVO"
    assert decision.trace["ibvs_jacobian_source"] == "profile_confirm"
    assert decision.trace["ibvs_jacobian"]["du_dtheta_px_per_deg"] == pytest.approx(-10.0)


def test_continuous_servo_ibvs_dls_handles_singular_jacobian_without_nan() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="ibvs_dls",
        vision_continuous_servo_ibvs_du_dtheta_px_per_deg=0.0,
        vision_continuous_servo_ibvs_du_dradius_px_per_mm=0.0,
        vision_continuous_servo_ibvs_dv_dtheta_px_per_deg=0.0,
        vision_continuous_servo_ibvs_dv_dradius_px_per_mm=0.0,
        vision_continuous_servo_ibvs_damping_px_per_unit=0.0,
    ).resolved()
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=58.3,
            geometry_center_f=[372.0, 214.0],
            alignment_target_pixel=[320.0, 240.0],
            servo_command_point=None,
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "ibvs_jacobian_unavailable"
    assert decision.theta_rate_deg_s == 0.0
    assert decision.radius_rate_mm_s == 0.0


def test_continuous_servo_ibvs_dls_does_not_fallback_to_servo_command_point_when_feedback_missing() -> None:
    controller = ContinuousVisionServoController(AppConfig(vision_continuous_servo_horizontal_mode="ibvs_dls"))

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=None,
            geometry_center_f=None,
            alignment_target_pixel=[320.0, 240.0],
            servo_command_point=[35.0, 210.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "center_distance_unavailable_wait"
    assert decision.theta_rate_deg_s == 0.0
    assert decision.radius_rate_mm_s == 0.0


def test_continuous_servo_ibvs_dls_keeps_mid_error_authority_before_strict_fine_band() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="ibvs_dls",
        vision_continuous_servo_ibvs_gain=0.5,
        vision_continuous_servo_ibvs_du_dtheta_px_per_deg=-10.0,
        vision_continuous_servo_ibvs_du_dradius_px_per_mm=0.0,
        vision_continuous_servo_ibvs_dv_dtheta_px_per_deg=0.0,
        vision_continuous_servo_ibvs_dv_dradius_px_per_mm=4.0,
        vision_continuous_servo_ibvs_damping_px_per_unit=0.0,
        vision_continuous_servo_theta_rate_limit_deg_s=20.0,
        vision_continuous_servo_radius_rate_limit_mm_s=20.0,
        vision_continuous_servo_pixel_axis_fine_band_px=24.0,
        vision_continuous_servo_pixel_axis_fine_rate_scale=0.25,
        vision_continuous_servo_low_height_guard_band_mm=30.0,
        vision_continuous_servo_low_height_fine_band_px=10.0,
        vision_continuous_servo_low_height_fine_rate_scale=0.25,
    ).resolved()
    controller = ContinuousVisionServoController(config)

    mid_error = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=20.0,
            geometry_center_f=[340.0, 240.0],
            alignment_target_pixel=[320.0, 240.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 210.0),
    )
    strict_fine = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=8.0,
            geometry_center_f=[328.0, 240.0],
            alignment_target_pixel=[320.0, 240.0],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 130.0),
    )

    assert mid_error.action == "SERVO"
    assert mid_error.theta_rate_deg_s == pytest.approx(1.0)
    assert strict_fine.action == "SERVO"
    assert strict_fine.theta_rate_deg_s == pytest.approx(0.025)


def test_continuous_servo_switches_measurement_point_in_low_height_guard() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_horizontal_mode="ibvs_dls",
        vision_servo_measurement_point="geometry_subpixel",
        vision_servo_low_height_measurement_point="top_face_subpixel",
        vision_continuous_servo_ibvs_du_dtheta_px_per_deg=-10.0,
        vision_continuous_servo_ibvs_du_dradius_px_per_mm=0.0,
        vision_continuous_servo_ibvs_dv_dtheta_px_per_deg=0.0,
        vision_continuous_servo_ibvs_dv_dradius_px_per_mm=4.0,
        vision_continuous_servo_low_height_guard_band_mm=30.0,
        vision_continuous_servo_center_allow_descent_px=2.0,
    ).resolved()
    controller = ContinuousVisionServoController(config)

    high = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=None,
            geometry_center_f=[340.0, 240.0],
            top_face_center_f=[320.0, 240.0],
            alignment_target_pixel=[320.0, 240.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 170.0),
    )
    low = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=None,
            geometry_center_f=[340.0, 240.0],
            top_face_center_f=[320.0, 240.0],
            alignment_target_pixel=[320.0, 240.0],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 125.0),
    )

    assert high.action == "SERVO"
    assert high.trace["center_distance_px"] == pytest.approx(20.0)
    assert high.trace["measurement_point"] == "geometry_subpixel"
    assert low.trace["measurement_point"] == "top_face_subpixel"
    assert low.trace["center_distance_px"] == pytest.approx(0.0)


def test_continuous_servo_pixel_axis_reverses_after_crossing_center() -> None:
    controller = ContinuousVisionServoController(
        AppConfig(
            vision_continuous_servo_horizontal_mode="pixel_axis",
            vision_continuous_servo_theta_rate_limit_deg_s=8.0,
            vision_continuous_servo_radius_rate_limit_mm_s=8.0,
        ).resolved()
    )

    right_low = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=200.0,
            geometry_center_f=[514.0, 290.0],
            alignment_target_pixel=[320.0, 240.0],
            servo_command_point=[29.0, 185.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 190.0),
    )
    left_low = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=76.0,
            geometry_center_f=[287.0, 309.0],
            alignment_target_pixel=[320.0, 240.0],
            servo_command_point=[45.0, 200.0],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(44.0, 198.0, 190.0),
    )

    assert right_low.action == "SERVO"
    assert right_low.theta_rate_deg_s > 0.0
    assert right_low.radius_rate_mm_s < 0.0
    assert left_low.action == "SERVO"
    assert left_low.theta_rate_deg_s < 0.0
    assert left_low.radius_rate_mm_s < 0.0


def test_continuous_servo_pixel_axis_slows_near_center() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="pixel_axis",
        vision_continuous_servo_pixel_axis_fine_band_px=24.0,
        vision_continuous_servo_pixel_axis_fine_rate_scale=0.25,
        vision_continuous_servo_theta_rate_limit_deg_s=8.0,
        vision_continuous_servo_radius_rate_limit_mm_s=8.0,
    ).resolved()
    controller = ContinuousVisionServoController(config)

    far = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=40.0,
            geometry_center_f=[360.0, 240.0],
            alignment_target_pixel=[320.0, 240.0],
            servo_command_point=[25.0, 180.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 190.0),
    )
    near = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=12.0,
            geometry_center_f=[360.0, 240.0],
            alignment_target_pixel=[320.0, 240.0],
            servo_command_point=[25.0, 180.0],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 190.0),
    )

    assert far.action == "SERVO"
    assert near.action == "SERVO"
    assert near.theta_rate_deg_s == pytest.approx(far.theta_rate_deg_s * 0.25)


def test_continuous_servo_pixel_axis_continues_through_grasp_unstable() -> None:
    controller = ContinuousVisionServoController(
        AppConfig(
            vision_continuous_servo_horizontal_mode="pixel_axis",
            vision_continuous_servo_descent_high_error_px=80.0,
            vision_continuous_servo_center_allow_descent_px=8.0,
            vision_continuous_servo_z_rate_limit_mm_s=8.0,
        ).resolved()
    )

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            actionable=False,
            invalid_reason="grasp_unstable",
            center_distance_px=42.0,
            geometry_center_f=[360.0, 228.0],
            alignment_target_pixel=[320.0, 240.0],
            servo_command_point=None,
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(20.0, 180.0, 190.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.theta_rate_deg_s > 0.0
    assert decision.z_rate_mm_s < 0.0
    assert decision.trace["soft_descent"] is True


def test_continuous_servo_dynamic_descent_gate_tightens_near_confirm() -> None:
    controller = ContinuousVisionServoController(
        AppConfig(
            vision_continuous_servo_horizontal_mode="pixel_axis",
            vision_continuous_servo_descent_high_error_px=80.0,
            vision_continuous_servo_descent_high_error_z_above_confirm_mm=70.0,
            vision_continuous_servo_descent_low_error_z_above_confirm_mm=12.0,
            vision_continuous_servo_center_allow_descent_px=8.0,
            vision_continuous_servo_low_height_descent_allow_px=8.0,
        ).resolved()
    )

    high = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=60.0,
            geometry_center_f=[380.0, 240.0],
            alignment_target_pixel=[320.0, 240.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 190.0),
    )
    low = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=60.0,
            geometry_center_f=[380.0, 240.0],
            alignment_target_pixel=[320.0, 240.0],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(20.0, 180.0, 135.0),
    )

    assert high.action == "SERVO"
    assert high.z_rate_mm_s < 0.0
    assert low.action == "SERVO"
    assert low.z_rate_mm_s == 0.0


def test_continuous_servo_low_height_descent_allow_is_separate_from_final_gate() -> None:
    controller = ContinuousVisionServoController(
        AppConfig(
            vision_pick_confirm_z_mm=120.0,
            vision_pick_z_tolerance_mm=1.0,
            vision_continuous_servo_horizontal_mode="pixel_axis",
            vision_continuous_servo_center_allow_descent_px=8.0,
            vision_continuous_servo_low_height_descent_allow_px=12.0,
            vision_continuous_servo_low_height_pause_descent_band_mm=4.0,
            vision_continuous_servo_z_rate_limit_mm_s=12.0,
            vision_continuous_servo_low_height_z_rate_scale=0.35,
        ).resolved()
    )

    outer_guard = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=10.5,
            geometry_center_f=[330.5, 240.0],
            alignment_target_pixel=[320.0, 240.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(20.0, 180.0, 132.0),
    )
    inner_pause = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=10.5,
            geometry_center_f=[330.5, 240.0],
            alignment_target_pixel=[320.0, 240.0],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(20.0, 180.0, 123.0),
    )

    assert outer_guard.action == "SERVO"
    assert outer_guard.z_rate_mm_s < 0.0
    assert outer_guard.trace["descent_error_allow_px"] == pytest.approx(12.0)
    assert outer_guard.trace["pick_ready_center_px"] == pytest.approx(2.0)
    assert inner_pause.action == "SERVO"
    assert inner_pause.z_rate_mm_s == 0.0


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
    controller = ContinuousVisionServoController(
        AppConfig(vision_continuous_servo_horizontal_mode="servo_command_point")
    )

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


def test_continuous_servo_accepts_large_low_confidence_locked_target() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            confidence=0.34,
            area_px=24040,
            bbox=[253, 128, 425, 348],
            center_distance_px=1.8,
            servo_command_point=[7.6, 176.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 1},
        current_cyl_pose=(7.5, 175.5, 127.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.trace.get("quality_reason") is None


def test_continuous_servo_holds_on_single_target_center_jump() -> None:
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
    assert decision.reason == "hold"
    assert decision.pending is not None
    assert decision.pending.last_center_px == (326.0, 237.0)
    assert decision.pending.stable_frames == 0
    assert decision.pending.descent_anchor_z_mm is None
    assert decision.trace["reason"] == "target_center_jump"


def test_continuous_servo_accepts_large_center_jump_when_error_improves() -> None:
    controller = ContinuousVisionServoController(
        AppConfig(vision_continuous_servo_horizontal_mode="servo_command_point")
    )

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            pixel_center=[440, 280],
            center_distance_px=126.0,
            servo_command_point=[45.2, 199.2],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
        pending={
            "slot_id": 1,
            "stable_frames": 0,
            "last_center_px": [485, 289],
            "last_center_distance_px": 172.0,
        },
        current_cyl_pose=(40.9, 192.3, 190.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.theta_rate_deg_s > 0.0
    assert decision.radius_rate_mm_s > 0.0


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


def test_continuous_servo_tracks_color_block_center_when_configured() -> None:
    controller = ContinuousVisionServoController(AppConfig(vision_servo_measurement_point="color_block_subpixel"))

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            pixel_center=[319, 222],
            geometry_center_f=[322, 239],
            color_block_center_f=[317.5, 238.25],
            center_distance_px=3.05,
            measurement_point="color_block_subpixel",
            servo_command_point=[0.1, 151.0],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        pending={
            "slot_id": 1,
            "stable_frames": 1,
            "last_center_px": [317.5, 238.25],
            "last_center_distance_px": 3.05,
        },
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.pending is not None
    assert decision.pending.last_center_px == (317.5, 238.25)


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
    controller = ContinuousVisionServoController(
        AppConfig(vision_continuous_servo_horizontal_mode="servo_command_point")
    )

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
    controller = ContinuousVisionServoController(
        AppConfig(vision_continuous_servo_horizontal_mode="servo_command_point")
    )

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
    controller = ContinuousVisionServoController(
        AppConfig(
            vision_continuous_servo_horizontal_mode="servo_command_point",
            vision_continuous_servo_center_allow_descent_px=2.0,
        )
    )

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
    controller = ContinuousVisionServoController(
        AppConfig(
            vision_continuous_servo_horizontal_mode="servo_command_point",
            vision_continuous_servo_center_allow_descent_px=2.0,
        )
    )

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
    controller = ContinuousVisionServoController(
        AppConfig(vision_continuous_servo_horizontal_mode="servo_command_point")
    )

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
    controller = ContinuousVisionServoController(
        AppConfig(vision_continuous_servo_horizontal_mode="servo_command_point")
    )

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


def test_continuous_servo_low_height_unstable_frame_refines_without_descending() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=25.0,
        vision_continuous_servo_low_height_unstable_servo_px=60.0,
        vision_continuous_servo_center_stop_descent_px=36.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            valid=True,
            actionable=False,
            invalid_reason="grasp_unstable",
            center_distance_px=39.0,
            servo_command_point=None,
            camera_to_world_raw=[12.0, 3.0, 0.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(10.0, 174.0, 133.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.z_rate_mm_s == 0.0
    assert decision.theta_rate_deg_s != 0.0 or decision.radius_rate_mm_s != 0.0
    assert decision.pending is not None
    assert decision.pending.stable_frames == 0


def test_continuous_servo_stops_when_center_distance_is_unavailable() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=None),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "center_distance_unavailable_wait"
    assert decision.pending is not None
    assert decision.pending.descent_anchor_z_mm is None


def test_continuous_servo_recovers_center_distance_from_packet_target() -> None:
    controller = ContinuousVisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=None,
            geometry_center_f=[316.0, 237.0],
            alignment_target_pixel=None,
            servo_command_point=[0.5, 151.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.trace["center_distance_px"] == pytest.approx(5.0)


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
    controller = ContinuousVisionServoController(
        AppConfig(vision_continuous_servo_horizontal_mode="servo_command_point")
    )

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0, servo_command_point=[0.5, 151.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s == 0.0


def test_continuous_servo_descends_when_center_is_stable() -> None:
    controller = ContinuousVisionServoController(
        AppConfig(vision_continuous_servo_horizontal_mode="servo_command_point")
    )

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


def test_continuous_servo_can_pause_descent_after_configured_z_pulse() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_continuous_servo_z_pulse_mm=3.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0, servo_command_point=[0.5, 151.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 5, "descent_anchor_z_mm": 190.0},
        current_cyl_pose=(0.0, 150.0, 187.0),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s == 0.0
    assert decision.pending is not None
    assert decision.pending.stable_frames == 0
    assert decision.pending.descent_anchor_z_mm is None
    assert decision.pending.descent_cooldown_frames > 0


def test_continuous_servo_does_not_descend_outside_configured_descent_band() -> None:
    controller = ContinuousVisionServoController(
        AppConfig(
            vision_continuous_servo_horizontal_mode="servo_command_point",
            vision_continuous_servo_center_allow_descent_px=2.0,
            vision_continuous_servo_center_stop_descent_px=5.0,
        )
    )

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=6.0, servo_command_point=[0.5, 151.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 4},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s == 0.0


def test_continuous_servo_soft_descends_inside_wide_band() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_center_allow_descent_px=8.0,
        vision_continuous_servo_center_stop_descent_px=36.0,
        vision_continuous_servo_soft_descent_rate_scale=0.25,
        vision_continuous_servo_z_rate_limit_mm_s=12.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=24.0, servo_command_point=[0.5, 151.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, 170.0),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s == pytest.approx(-3.0)
    assert decision.trace["soft_descent"] is True


def test_continuous_servo_soft_descent_stops_inside_pause_band() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_center_allow_descent_px=8.0,
        vision_continuous_servo_center_stop_descent_px=36.0,
        vision_continuous_servo_low_height_pause_descent_band_mm=12.0,
        vision_continuous_servo_soft_descent_min_z_above_confirm_mm=18.0,
        vision_continuous_servo_z_rate_limit_mm_s=12.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=24.0, servo_command_point=[0.5, 151.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, 131.0),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s == 0.0
    assert decision.trace.get("soft_descent") is False


def test_continuous_servo_slows_z_near_confirm_height() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_continuous_servo_z_rate_limit_mm_s=18.0,
        vision_continuous_servo_z_slow_band_mm=20.0,
    )
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


def test_continuous_servo_scales_z_rate_inside_low_height_guard() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_z_rate_limit_mm_s=12.0,
        vision_continuous_servo_z_slow_band_mm=5.0,
        vision_continuous_servo_low_height_guard_band_mm=30.0,
        vision_continuous_servo_low_height_z_rate_scale=0.25,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=7.0, servo_command_point=[0.5, 151.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 1},
        current_cyl_pose=(0.0, 150.0, 140.0),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s == pytest.approx(-3.0)
    assert decision.trace["low_height_guard_active"] is True
    assert decision.trace["z_rate_scale_reason"] == "low_height_guard"
    assert decision.trace["low_height_z_rate_scale"] == pytest.approx(0.25)


def test_continuous_servo_never_exceeds_configured_low_z_rate_limit() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_continuous_servo_z_rate_limit_mm_s=2.0,
        vision_continuous_servo_z_slow_band_mm=20.0,
    )
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


def test_continuous_servo_default_does_not_pulse_pause_descent() -> None:
    controller = ContinuousVisionServoController(
        AppConfig(vision_continuous_servo_horizontal_mode="servo_command_point")
    )

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0, servo_command_point=[0.5, 151.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 5, "descent_anchor_z_mm": 190.0},
        current_cyl_pose=(0.0, 150.0, 184.0),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s < 0.0
    assert decision.pending is not None
    assert decision.pending.stable_frames > 0


def test_continuous_servo_stops_on_stale_frame() -> None:
    controller = ContinuousVisionServoController(
        AppConfig(
            vision_continuous_servo_command_timeout_ms=100.0,
            vision_continuous_servo_stale_frames=3,
        )
    )

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0),
        packet={"frame_id": 1, "queue_age_ms": 120.0},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "frame_stale_wait"
    assert decision.pending is not None
    assert decision.pending.stale_frames == 1
    assert decision.pending.descent_anchor_z_mm is None
    assert decision.pending.descent_cooldown_frames > 0

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0),
        packet={"frame_id": 2, "queue_age_ms": 120.0},
        pending=decision.pending_dict,
        current_cyl_pose=(0.0, 150.0, 190.0),
    )
    assert decision.reason == "frame_stale_wait"
    assert decision.pending is not None
    assert decision.pending.stale_frames == 2

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0),
        packet={"frame_id": 3, "queue_age_ms": 120.0},
        pending=decision.pending_dict,
        current_cyl_pose=(0.0, 150.0, 190.0),
    )
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
    controller = ContinuousVisionServoController(AppConfig(vision_continuous_servo_stale_frames=1))

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=8.0),
        packet={"frame_id": 1},
        current_cyl_pose=(0.0, 150.0, 190.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "frame_stale"


def test_continuous_servo_stops_when_frame_age_is_invalid() -> None:
    controller = ContinuousVisionServoController(AppConfig(vision_continuous_servo_stale_frames=1))

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
        vision_continuous_servo_horizontal_mode="servo_command_point",
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
        vision_continuous_servo_horizontal_mode="servo_command_point",
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
        vision_continuous_servo_horizontal_mode="servo_command_point",
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


def test_continuous_servo_default_low_height_band_only_slows_near_final_gate() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_continuous_servo_pick_ready_center_px=2.0,
    )
    high_decision = ContinuousVisionServoController(config).decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=2.5, servo_command_point=[2.0, 154.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, config.vision_pick_confirm_z_mm + 40.0),
    )
    low_decision = ContinuousVisionServoController(config).decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=2.5, servo_command_point=[2.0, 154.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, config.vision_pick_confirm_z_mm),
    )
    visible_offset = ContinuousVisionServoController(config).decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=10.0, servo_command_point=[4.0, 158.0]),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
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
    assert visible_offset.action == "SERVO"
    assert visible_offset.theta_rate_deg_s == pytest.approx(
        4.0 * config.vision_continuous_servo_theta_gain_deg_s_per_deg
        * config.vision_continuous_servo_low_height_coarse_rate_scale
    )
    assert visible_offset.radius_rate_mm_s == pytest.approx(
        8.0 * config.vision_continuous_servo_radius_gain_mm_s_per_mm
        * config.vision_continuous_servo_low_height_coarse_rate_scale
    )


def test_continuous_servo_keeps_coarse_rates_across_outer_low_height_guard_band() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=25.0,
        vision_continuous_servo_low_height_fine_rate_scale=0.25,
    )
    high_decision = ContinuousVisionServoController(config).decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=20.0, servo_command_point=[2.0, 154.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, 170.0),
    )
    low_decision = ContinuousVisionServoController(config).decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=20.0, servo_command_point=[2.0, 154.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, 142.0),
    )

    assert high_decision.action == "SERVO"
    assert low_decision.action == "SERVO"
    assert low_decision.theta_rate_deg_s == pytest.approx(high_decision.theta_rate_deg_s)
    assert low_decision.radius_rate_mm_s == pytest.approx(high_decision.radius_rate_mm_s)
    assert low_decision.trace["low_height_guard_active"] is True


def test_continuous_servo_scales_rates_near_center_in_low_height_guard_band() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=25.0,
        vision_continuous_servo_low_height_fine_band_px=12.0,
        vision_continuous_servo_low_height_fine_rate_scale=0.25,
    )
    high_decision = ContinuousVisionServoController(config).decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=10.0, servo_command_point=[2.0, 154.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, 170.0),
    )
    low_decision = ContinuousVisionServoController(config).decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=10.0, servo_command_point=[2.0, 154.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, 142.0),
    )

    assert high_decision.action == "SERVO"
    assert low_decision.action == "SERVO"
    assert low_decision.theta_rate_deg_s == pytest.approx(
        high_decision.theta_rate_deg_s * config.vision_continuous_servo_low_height_fine_rate_scale
    )
    assert low_decision.radius_rate_mm_s == pytest.approx(
        high_decision.radius_rate_mm_s * config.vision_continuous_servo_low_height_fine_rate_scale
    )
    assert low_decision.trace["low_height_guard_active"] is True


def test_continuous_servo_continues_slow_descent_in_outer_low_height_guard() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=25.0,
        vision_continuous_servo_low_height_pause_descent_band_mm=12.0,
        vision_continuous_servo_center_allow_descent_px=8.0,
        vision_continuous_servo_center_stop_descent_px=36.0,
        vision_continuous_servo_low_height_z_rate_scale=0.35,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=20.0, servo_command_point=[2.0, 154.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 2},
        current_cyl_pose=(0.0, 150.0, 142.0),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s < 0.0
    assert decision.theta_rate_deg_s != 0.0 or decision.radius_rate_mm_s != 0.0
    assert decision.trace["low_height_guard_active"] is True
    assert decision.trace["low_height_pause_descent_active"] is False
    assert decision.trace["soft_descent"] is True


def test_continuous_servo_pauses_descent_in_inner_low_height_band_until_centered() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=25.0,
        vision_continuous_servo_low_height_pause_descent_band_mm=12.0,
        vision_continuous_servo_center_allow_descent_px=8.0,
        vision_continuous_servo_center_stop_descent_px=36.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=20.0, servo_command_point=[2.0, 154.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 2},
        current_cyl_pose=(0.0, 150.0, 131.0),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s == 0.0
    assert decision.theta_rate_deg_s != 0.0 or decision.radius_rate_mm_s != 0.0
    assert decision.trace["low_height_guard_active"] is True
    assert decision.trace["low_height_pause_descent_active"] is True


def test_continuous_servo_pauses_descent_after_low_height_near_center_rebound() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_horizontal_mode="pixel_axis",
        vision_continuous_servo_low_height_guard_band_mm=45.0,
        vision_continuous_servo_low_height_pause_descent_band_mm=12.0,
        vision_continuous_servo_low_height_fine_band_px=10.0,
        vision_continuous_servo_low_height_descent_rebound_pause_px=10.0,
        vision_continuous_servo_center_allow_descent_px=8.0,
        vision_continuous_servo_descent_high_error_px=80.0,
        vision_continuous_servo_descent_high_error_z_above_confirm_mm=70.0,
        vision_continuous_servo_descent_low_error_z_above_confirm_mm=12.0,
    ).resolved()
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=25.1,
            geometry_center_f=[294.9, 238.9],
            alignment_target_pixel=[320.0, 240.0],
        ),
        packet={"frame_id": 2, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        pending={"slot_id": 1, "stable_frames": 1, "last_center_distance_px": 3.3},
        current_cyl_pose=(48.1, 177.1, 151.9),
    )

    assert decision.action == "SERVO"
    assert decision.z_rate_mm_s == 0.0
    assert decision.trace["low_height_guard_active"] is True
    assert decision.trace["low_height_descent_rebound"] is True
    assert decision.theta_rate_deg_s != 0.0 or decision.radius_rate_mm_s != 0.0


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


def test_continuous_servo_records_low_height_anchor_and_best_error() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=30.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=9.0, servo_command_point=[7.6, 175.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(7.0, 174.0, 140.0),
    )

    assert decision.action == "SERVO"
    assert decision.pending is not None
    assert decision.pending.low_height_anchor_pose == (7.0, 174.0, 140.0)
    assert decision.pending.best_center_distance_px == pytest.approx(9.0)
    assert decision.trace["low_height_anchor_pose"] == [7.0, 174.0, 140.0]
    assert decision.trace["best_center_distance_px"] == pytest.approx(9.0)


def test_continuous_servo_low_height_static_residual_requires_local_model() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_pick_z_tolerance_mm=1.0,
        vision_continuous_servo_horizontal_mode="ibvs_dls",
        vision_continuous_servo_low_height_guard_band_mm=30.0,
        vision_continuous_servo_low_height_static_frames=3,
        vision_continuous_servo_low_height_static_error_min_px=8.0,
        vision_continuous_servo_low_height_static_error_max_px=30.0,
        vision_continuous_servo_low_height_static_improvement_px=1.0,
        vision_continuous_servo_low_height_static_band_mm=6.0,
        vision_continuous_servo_low_height_static_pose_band_mm=1.5,
    ).resolved()
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=24.7,
            geometry_center_f=[344.7, 240.0],
            alignment_target_pixel=[320.0, 240.0],
        ),
        packet={"frame_id": 10, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        pending={
            "slot_id": 1,
            "stable_frames": 0,
            "low_height_anchor_pose": [10.0, 174.0, 124.0],
            "best_center_distance_px": 25.0,
            "low_height_static_frames": 2,
            "low_height_static_reference_px": 25.0,
        },
        current_cyl_pose=(10.2, 174.3, 121.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "low_height_local_model_required"
    assert decision.trace["low_height_static_frames"] == 3
    assert "search_low_height_center.py" in decision.trace["recommendation"]


def test_continuous_servo_low_height_static_residual_not_counted_while_descending() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_pick_z_tolerance_mm=1.0,
        vision_continuous_servo_horizontal_mode="ibvs_dls",
        vision_continuous_servo_low_height_guard_band_mm=30.0,
        vision_continuous_servo_low_height_static_frames=3,
        vision_continuous_servo_low_height_static_error_min_px=8.0,
        vision_continuous_servo_low_height_static_error_max_px=30.0,
        vision_continuous_servo_low_height_static_band_mm=6.0,
        vision_continuous_servo_low_height_static_pose_band_mm=1.5,
    ).resolved()
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=10.0,
            geometry_center_f=[330.0, 240.0],
            alignment_target_pixel=[320.0, 240.0],
        ),
        packet={"frame_id": 10, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        pending={
            "slot_id": 1,
            "stable_frames": 0,
            "low_height_anchor_pose": [10.0, 174.0, 124.0],
            "best_center_distance_px": 3.7,
            "low_height_static_frames": 30,
            "low_height_static_reference_px": 3.7,
        },
        current_cyl_pose=(10.2, 174.3, 124.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.pending is not None
    assert decision.pending.low_height_static_frames == 0
    assert decision.pending.low_height_static_reference_px is None


def test_continuous_servo_low_height_static_residual_waits_until_confirm_band() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_horizontal_mode="ibvs_dls",
        vision_continuous_servo_low_height_guard_band_mm=30.0,
        vision_continuous_servo_low_height_static_frames=3,
        vision_continuous_servo_low_height_static_band_mm=6.0,
    ).resolved()
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            center_distance_px=24.7,
            geometry_center_f=[344.7, 240.0],
            alignment_target_pixel=[320.0, 240.0],
        ),
        packet={"frame_id": 10, "queue_age_ms": 10.0, "alignment_target_pixel": [320.0, 240.0]},
        pending={
            "slot_id": 1,
            "stable_frames": 0,
            "low_height_anchor_pose": [10.0, 174.0, 140.0],
            "best_center_distance_px": 25.0,
            "low_height_static_frames": 20,
            "low_height_static_reference_px": 25.0,
        },
        current_cyl_pose=(10.2, 174.3, 140.0),
    )

    assert decision.reason != "low_height_local_model_required"
    assert decision.action in {"SERVO", "STOP"}


def test_continuous_servo_clears_low_height_anchor_above_guard_band() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=30.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=9.0, servo_command_point=[7.6, 175.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={
            "slot_id": 1,
            "stable_frames": 0,
            "low_height_anchor_pose": [7.0, 174.0, 140.0],
            "best_center_distance_px": 4.0,
        },
        current_cyl_pose=(7.1, 174.5, 151.0),
    )

    assert decision.action == "SERVO"
    assert decision.pending is not None
    assert decision.pending.low_height_anchor_pose is None
    assert decision.pending.best_center_distance_px is None


def test_continuous_servo_allows_corrective_low_height_theta_drift() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=30.0,
        vision_continuous_servo_low_height_max_theta_drift_deg=4.0,
        vision_continuous_servo_low_height_max_radius_drift_mm=8.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=12.0, servo_command_point=[12.0, 176.0]),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
        pending={
            "slot_id": 1,
            "stable_frames": 0,
            "low_height_anchor_pose": [7.0, 174.0, 140.0],
            "best_center_distance_px": 10.0,
        },
        current_cyl_pose=(11.2, 176.0, 134.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.trace["low_height_guard_active"] is True


def test_continuous_servo_stops_on_noncorrective_low_height_theta_drift() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=30.0,
        vision_continuous_servo_low_height_max_theta_drift_deg=4.0,
        vision_continuous_servo_low_height_max_radius_drift_mm=8.0,
        vision_continuous_servo_low_height_best_error_rebound_px=8.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=25.0, servo_command_point=[12.0, 176.0]),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
        pending={
            "slot_id": 1,
            "stable_frames": 0,
            "low_height_anchor_pose": [7.0, 174.0, 140.0],
            "best_center_distance_px": 10.0,
        },
        current_cyl_pose=(11.2, 176.0, 134.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "low_height_mapping_drift"
    assert decision.pending is not None
    assert decision.pending.stable_frames == 0
    assert decision.trace["theta_drift_deg"] == pytest.approx(4.2)


def test_continuous_servo_allows_corrective_low_height_radius_drift() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=30.0,
        vision_continuous_servo_low_height_max_theta_drift_deg=8.0,
        vision_continuous_servo_low_height_max_radius_drift_mm=3.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=12.0, servo_command_point=[7.6, 179.0]),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
        pending={
            "slot_id": 1,
            "stable_frames": 0,
            "low_height_anchor_pose": [7.0, 174.0, 140.0],
            "best_center_distance_px": 10.0,
        },
        current_cyl_pose=(7.5, 177.5, 134.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.trace["low_height_guard_active"] is True


def test_continuous_servo_stops_on_noncorrective_low_height_radius_drift() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=30.0,
        vision_continuous_servo_low_height_max_theta_drift_deg=8.0,
        vision_continuous_servo_low_height_max_radius_drift_mm=3.0,
        vision_continuous_servo_low_height_best_error_rebound_px=8.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=25.0, servo_command_point=[7.6, 179.0]),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
        pending={
            "slot_id": 1,
            "stable_frames": 0,
            "low_height_anchor_pose": [7.0, 174.0, 140.0],
            "best_center_distance_px": 10.0,
        },
        current_cyl_pose=(7.5, 177.5, 134.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "low_height_mapping_drift"
    assert decision.trace["radius_drift_mm"] == pytest.approx(3.5)


def test_continuous_servo_stops_on_low_height_best_error_rebound() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=30.0,
        vision_continuous_servo_low_height_fine_band_px=10.0,
        vision_continuous_servo_center_stop_descent_px=36.0,
        vision_continuous_servo_low_height_unstable_servo_px=60.0,
        vision_continuous_servo_low_height_best_error_rebound_px=6.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=64.0, servo_command_point=[7.6, 175.0]),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
        pending={
            "slot_id": 1,
            "stable_frames": 0,
            "low_height_anchor_pose": [7.0, 174.0, 140.0],
            "best_center_distance_px": 4.0,
        },
        current_cyl_pose=(7.5, 175.0, 134.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "low_height_best_error_rebounded"
    assert decision.trace["best_error_rebound_px"] == pytest.approx(60.0)


def test_continuous_servo_refines_low_height_best_error_rebound_inside_servo_window() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=30.0,
        vision_continuous_servo_low_height_fine_band_px=10.0,
        vision_continuous_servo_center_allow_descent_px=8.0,
        vision_continuous_servo_center_stop_descent_px=36.0,
        vision_continuous_servo_low_height_unstable_servo_px=60.0,
        vision_continuous_servo_low_height_best_error_rebound_px=6.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=21.0, servo_command_point=[7.6, 175.0]),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
        pending={
            "slot_id": 1,
            "stable_frames": 2,
            "low_height_anchor_pose": [7.0, 174.0, 140.0],
            "best_center_distance_px": 4.0,
        },
        current_cyl_pose=(7.5, 175.0, 122.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.z_rate_mm_s == 0.0
    assert decision.theta_rate_deg_s != 0.0 or decision.radius_rate_mm_s != 0.0


def test_continuous_servo_ignores_low_height_guard_above_guard_band() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_low_height_guard_band_mm=30.0,
        vision_continuous_servo_low_height_max_theta_drift_deg=4.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=12.0, servo_command_point=[12.0, 176.0]),
        packet={"frame_id": 2, "queue_age_ms": 10.0},
        pending={
            "slot_id": 1,
            "stable_frames": 0,
            "low_height_anchor_pose": [7.0, 174.0, 140.0],
            "best_center_distance_px": 4.0,
        },
        current_cyl_pose=(20.0, 176.0, 151.0),
    )

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.pending is not None
    assert decision.pending.low_height_anchor_pose is None
    assert decision.pending.best_center_distance_px is None


def test_continuous_servo_keeps_servoing_near_center_at_confirm_height_when_correction_exists() -> None:
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

    assert decision.action == "SERVO"
    assert decision.reason == "continuous_servo"
    assert decision.theta_rate_deg_s != 0.0 or decision.radius_rate_mm_s != 0.0
    assert decision.pending is not None
    assert decision.pending.stable_frames == 1


def test_continuous_servo_only_stops_to_settle_when_no_correction_exists() -> None:
    config = AppConfig(
        vision_continuous_servo_horizontal_mode="servo_command_point",
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_settle_stop_band_px=8.0,
        vision_continuous_servo_stable_frames=2,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(center_distance_px=1.7, servo_command_point=[0.0, 150.0]),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 0},
        current_cyl_pose=(0.0, 150.0, 123.0),
    )

    assert decision.action == "STOP"
    assert decision.reason == "settle_near_center"
    assert decision.pending is not None
    assert decision.pending.stable_frames == 1
    assert decision.trace["zero_rate_hold"] is True


def test_continuous_servo_pick_ready_at_confirm_height_without_settle_stop() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_pick_ready_center_px=2.0,
        vision_continuous_servo_stable_frames=2,
        pick_tool_offset_source="command_bias",
        vision_eye_in_hand_pick_radius_bias_mm=40.0,
    )
    controller = ContinuousVisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload=_slot(
            actionable=True,
            center_distance_px=1.6,
            command="PICK_CYL 7.50 176.00 90.00",
            servo_command_point=[7.5, 176.0],
        ),
        packet={"frame_id": 1, "queue_age_ms": 10.0},
        pending={"slot_id": 1, "stable_frames": 1, "pick_ready_frames": 1},
        current_cyl_pose=(7.5, 176.0, 120.0),
    )

    assert decision.action == "PICK_READY"
    assert decision.reason == "pick_ready"
    assert decision.command == "PICK_CYL 7.50 216.00 90.00"


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
