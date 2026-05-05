from __future__ import annotations

from hybrid_controller.config import AppConfig
from hybrid_controller.vision.target_resolver import resolve_vision_packet


def _packet(
    *,
    mapping_mode: str,
    calibration_ready: bool = True,
    camera_to_world_raw: tuple[float, float, float] = (12.0, -8.0, 0.0),
    world_xyz: tuple[float, float, float] | None = None,
    servo_required: bool = False,
    calibration_profile_required: bool = False,
    calibration_profile_id: str = "",
    estimated_xy_error_mm: float | None = None,
    grasp_quality: float = 1.0,
) -> dict[str, object]:
    slot: dict[str, object] = {
        "slot_id": 1,
        "freq_hz": 8.0,
        "valid": True,
        "camera_to_world_raw": [float(camera_to_world_raw[0]), float(camera_to_world_raw[1]), float(camera_to_world_raw[2])],
        "servo_required": bool(servo_required),
        "calibration_profile_id": calibration_profile_id,
        "grasp_quality": float(grasp_quality),
    }
    if estimated_xy_error_mm is not None:
        slot["estimated_xy_error_mm"] = float(estimated_xy_error_mm)
    if world_xyz is not None:
        slot["world_xyz"] = [float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])]
    return {
        "mapping_mode": mapping_mode,
        "calibration_ready": bool(calibration_ready),
        "calibration_profile_required": bool(calibration_profile_required),
        "calibration_profile_id": calibration_profile_id,
        "slots": [slot],
    }


def _snapshot(robot_xy: tuple[float, float]) -> dict[str, object]:
    return {
        "robot_xy": [float(robot_xy[0]), float(robot_xy[1])],
        "limits_cyl": {
            "theta_deg": [-120.0, 120.0],
            "radius_mm": [50.0, 280.0],
        },
    }


def test_resolve_delta_servo_tracks_robot_pose() -> None:
    packet = _packet(mapping_mode="delta_servo", camera_to_world_raw=(12.0, -8.0, 0.0))

    resolved_a = resolve_vision_packet(
        packet,
        config=AppConfig(),
        snapshot=_snapshot((0.0, -120.0)),
        snapshot_age_ms=20.0,
    ).packet
    resolved_b = resolve_vision_packet(
        packet,
        config=AppConfig(),
        snapshot=_snapshot((40.0, -120.0)),
        snapshot_age_ms=20.0,
    ).packet

    slot_a = resolved_a["slots"][0]
    slot_b = resolved_b["slots"][0]
    assert slot_a["command_point"] == [12.0, -128.0]
    assert slot_b["command_point"] == [52.0, -128.0]
    assert slot_a["actionable"] is True
    assert slot_b["actionable"] is True


def test_resolve_delta_servo_rejects_stale_snapshot() -> None:
    config = AppConfig(vision_snapshot_max_age_ms=200.0)
    resolved = resolve_vision_packet(
        _packet(mapping_mode="delta_servo"),
        config=config,
        snapshot=_snapshot((0.0, -120.0)),
        snapshot_age_ms=500.0,
    ).packet

    slot = resolved["slots"][0]
    assert slot["actionable"] is False
    assert slot["command_point"] is None
    assert slot["invalid_reason"] == "robot_snapshot_stale"


def test_resolve_delta_servo_rejects_pose_stale_for_frame() -> None:
    config = AppConfig(vision_frame_pose_max_age_ms=150.0)
    resolved = resolve_vision_packet(
        _packet(mapping_mode="delta_servo"),
        config=config,
        snapshot=_snapshot((0.0, -120.0)),
        snapshot_age_ms=20.0,
        frame_pose_age_ms=240.0,
    ).packet

    slot = resolved["slots"][0]
    assert slot["actionable"] is False
    assert slot["command_point"] is None
    assert slot["invalid_reason"] == "robot_pose_stale_for_frame"
    assert slot["frame_pose_age_ms"] == 240.0


def test_resolve_absolute_base_uses_world_xyz() -> None:
    resolved = resolve_vision_packet(
        _packet(mapping_mode="absolute_base", world_xyz=(66.0, -170.0, 0.0)),
        config=AppConfig(),
        snapshot=None,
        snapshot_age_ms=float("inf"),
    ).packet

    slot = resolved["slots"][0]
    assert slot["actionable"] is True
    assert slot["command_mode"] == "world"
    assert slot["command_point"] == [66.0, -170.0]
    assert slot["world_xyz"] == [66.0, -170.0, 0.0]


def test_resolve_delta_servo_requires_centering_before_pick() -> None:
    config = AppConfig(vision_servo_move_gain=0.5)
    packet = _packet(
        mapping_mode="delta_servo",
        camera_to_world_raw=(40.0, 0.0, 0.0),
        servo_required=True,
        calibration_profile_id="unit-profile",
        calibration_profile_required=True,
    )

    resolved = resolve_vision_packet(
        packet,
        config=config,
        snapshot=_snapshot((0.0, -120.0)),
        snapshot_age_ms=20.0,
    ).packet

    slot = resolved["slots"][0]
    assert slot["actionable"] is False
    assert slot["command_point"] is None
    assert slot["invalid_reason"] == "vision_servo_required"
    assert slot["servo_command_mode"] == "cyl"
    assert slot["servo_command_point"] is not None


def test_resolve_delta_servo_uses_fine_gain_near_center() -> None:
    config = AppConfig(
        vision_servo_move_gain=0.8,
        vision_servo_fine_move_gain=0.25,
        vision_servo_fine_threshold_px=24.0,
    )
    packet = _packet(
        mapping_mode="delta_servo",
        camera_to_world_raw=(20.0, 0.0, 0.0),
        servo_required=True,
    )
    packet["slots"][0]["center_distance_px"] = 12.0

    resolved = resolve_vision_packet(
        packet,
        config=config,
        snapshot=_snapshot((0.0, -120.0)),
        snapshot_age_ms=20.0,
    ).packet

    slot = resolved["slots"][0]
    assert slot["invalid_reason"] == "vision_servo_required"
    assert slot["servo_command_point"] is not None
    # With fine gain, servo target is current pose plus 5 mm in X, not 16 mm.
    assert -3.0 < slot["servo_command_point"][0] < -2.0


def test_resolve_rejects_profile_required_without_profile_id() -> None:
    resolved = resolve_vision_packet(
        _packet(mapping_mode="delta_servo", calibration_profile_required=True),
        config=AppConfig(),
        snapshot=_snapshot((0.0, -120.0)),
        snapshot_age_ms=20.0,
    ).packet

    slot = resolved["slots"][0]
    assert slot["actionable"] is False
    assert slot["invalid_reason"] == "calibration_profile_unavailable"


def test_resolve_rejects_high_mapping_error() -> None:
    resolved = resolve_vision_packet(
        _packet(mapping_mode="delta_servo", estimated_xy_error_mm=9.0, calibration_profile_id="unit-profile"),
        config=AppConfig(vision_action_max_error_mm=6.0),
        snapshot=_snapshot((0.0, -120.0)),
        snapshot_age_ms=20.0,
    ).packet

    slot = resolved["slots"][0]
    assert slot["actionable"] is False
    assert slot["invalid_reason"] == "vision_mapping_error_high"
