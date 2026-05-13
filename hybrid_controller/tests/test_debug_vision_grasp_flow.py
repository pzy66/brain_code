from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from hybrid_controller.config import AppConfig
from hybrid_controller.tools.calibrate_low_height_alignment import _sample_alignment_target
from hybrid_controller.tools.calibrate_low_height_alignment import _state_is_safe as _low_height_state_is_safe
from hybrid_controller.tools.calibrate_low_height_alignment import main as low_height_alignment_main
from hybrid_controller.tools.debug_vision_grasp_flow import (
    _capture_frames_from_candidates,
    _clamp_refine_target,
    _continuous_decision_for_packet,
    _continuous_confirm_recheck,
    _continuous_low_height_refine_requested,
    _continuous_stop_reason_is_recoverable,
    _continuous_stopped_motion_target,
    _continuous_slot_id_for_selection,
    _continuous_snapshot_blocks_teleop,
    _current_calibration_stage,
    _current_cyl_pose,
    _decision_for_packet,
    main,
    _override_profile_center_tolerance,
    _packet_frame_pose_age_ms,
    _PersistentCaptureReader,
    _pose_from_confirm_recheck,
    _ibvs_jacobian_from_stage_profile,
    _resolve_alignment_target_pixel,
    _resolve_packet,
    _rewrite_final_pick_command_for_debug,
    _fetch_fresh_state_snapshot,
    _frame_pose_age_for_static_snapshot,
    _state_message_to_snapshot,
    _select_latest_frames,
    _select_slot,
    _servo_command_point_from_slot,
    _snapshot_local_age_ms,
    _wait_for_idle,
)
from hybrid_controller.tools.diagnose_vision_centers import diagnose_center_sequence
from hybrid_controller.vision.calibration_profile import VisionCalibrationProfile


class _ClosedCapture:
    def isOpened(self) -> bool:
        return False

    def set(self, *_args, **_kwargs) -> bool:
        return True

    def release(self) -> None:
        return None


class _ReadableCapture:
    def __init__(self) -> None:
        self.read_count = 0

    def isOpened(self) -> bool:
        return True

    def set(self, *_args, **_kwargs) -> bool:
        return True

    def read(self):
        self.read_count += 1
        if self.read_count > 64:
            return False, None
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        return None


class _FakeCv2:
    CAP_ANY = 0
    CAP_FFMPEG = 1900
    CAP_PROP_BUFFERSIZE = 38
    CAP_PROP_OPEN_TIMEOUT_MSEC = 53
    CAP_PROP_READ_TIMEOUT_MSEC = 54

    seen_sources: list[str] = []

    @classmethod
    def VideoCapture(cls, source, *_args):
        cls.seen_sources.append(str(source))
        if "working" in str(source):
            return _ReadableCapture()
        return _ClosedCapture()


def test_debug_capture_tries_stream_candidates_until_frames_are_read() -> None:
    _FakeCv2.seen_sources = []

    selected_url, frames = _capture_frames_from_candidates(
        cv2_module=_FakeCv2,
        stream_urls=("http://camera/empty", "http://camera/working"),
        config=AppConfig().resolved(),
        frame_count=1,
        drain_frames=0,
        timeout_sec=0.5,
    )

    assert selected_url == "http://camera/working"
    assert len(frames) == 1
    assert _FakeCv2.seen_sources == ["http://camera/empty", "http://camera/working"]


def test_debug_capture_backend_http_uses_official_mjpeg_reader(monkeypatch) -> None:
    opened: list[str] = []

    class FakeHttpCapture(_ReadableCapture):
        def __init__(self, url, *, cv2_module, timeout_sec, read_timeout_sec=None):
            super().__init__()
            opened.append(str(url))

    monkeypatch.setattr(
        "hybrid_controller.tools.debug_vision_grasp_flow._HttpMjpegCapture",
        FakeHttpCapture,
    )

    selected_url, frames = _capture_frames_from_candidates(
        cv2_module=_FakeCv2,
        stream_urls=(
            "http://camera:8080/stream?"
            "topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80",
        ),
        config=AppConfig().resolved(),
        frame_count=2,
        drain_frames=0,
        timeout_sec=0.5,
        capture_backend="http",
    )

    assert (
        selected_url
        == "http://camera:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"
    )
    assert len(frames) == 2
    assert opened == [
        "http://camera:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"
    ]


def test_persistent_capture_reader_reuses_single_capture() -> None:
    _FakeCv2.seen_sources = []
    reader = _PersistentCaptureReader(
        cv2_module=_FakeCv2,
        stream_urls=("http://camera/working",),
        config=AppConfig().resolved(),
        capture_backend="auto",
    )
    try:
        selected1, frames1 = reader.read(frame_count=2, drain_frames=0, timeout_sec=0.5)
        selected2, frames2 = reader.read(frame_count=2, drain_frames=0, timeout_sec=0.5)
    finally:
        reader.close()

    assert selected1 == "http://camera/working"
    assert selected2 == "http://camera/working"
    assert len(frames1) == 2
    assert len(frames2) == 2
    assert _FakeCv2.seen_sources == ["http://camera/working"]


def test_persistent_capture_reader_exposes_transport_stats() -> None:
    _FakeCv2.seen_sources = []
    reader = _PersistentCaptureReader(
        cv2_module=_FakeCv2,
        stream_urls=("http://camera/working",),
        config=AppConfig().resolved(),
        capture_backend="auto",
    )
    try:
        stats_before_read = reader.transport_stats()
        selected, frames = reader.read(frame_count=1, drain_frames=0, timeout_sec=0.5)
        stats_after_read = reader.transport_stats()
    finally:
        reader.close()

    assert stats_before_read["open"] is False
    assert selected == "http://camera/working"
    assert len(frames) == 1
    assert stats_after_read["open"] is True
    assert stats_after_read["stream_url"] == "http://camera/working"
    assert stats_after_read["capture_backend"] == "auto"
    assert stats_after_read["reader"] == "_ReadableCapture"


def test_persistent_capture_reader_reopen_resets_consumer_only() -> None:
    _FakeCv2.seen_sources = []
    reader = _PersistentCaptureReader(
        cv2_module=_FakeCv2,
        stream_urls=("http://camera/working",),
        config=AppConfig().resolved(),
        capture_backend="auto",
    )
    try:
        selected1, _frames1 = reader.read(frame_count=1, drain_frames=0, timeout_sec=0.5)
        reader.reopen()
        selected2, frames2 = reader.read(frame_count=1, drain_frames=0, timeout_sec=0.5)
    finally:
        reader.close()

    assert selected1 == "http://camera/working"
    assert selected2 == "http://camera/working"
    assert len(frames2) == 1
    assert _FakeCv2.seen_sources == ["http://camera/working", "http://camera/working"]


def test_debug_center_tolerance_override_updates_stage_profiles() -> None:
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            "servo": {"center_tolerance_px": 8.0},
            "stage_models": {
                "confirm": {
                    "z_mm": 175.0,
                    "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
                    "servo": {"center_tolerance_px": 9.0},
                }
            },
        }
    )

    overridden = _override_profile_center_tolerance(profile, 6.0)

    assert overridden.center_tolerance_px == 6.0
    assert profile.center_tolerance_px == 8.0
    assert overridden.stage_models is not None
    assert overridden.stage_models["confirm"].center_tolerance_px == 6.0


def test_select_latest_frames_keeps_only_newest_frames() -> None:
    frames = [(f"frame-{index}", float(index)) for index in range(5)]

    assert _select_latest_frames(frames, None) == frames
    assert _select_latest_frames(frames, 0) == frames
    assert _select_latest_frames(frames, 2) == frames[-2:]
    assert _select_latest_frames(frames, 99) == frames


def test_debug_packet_frame_pose_age_uses_camera_age_only() -> None:
    assert _packet_frame_pose_age_ms({"latest_frame_preprocess_age_ms": 42.0, "queue_age_ms": 200.0}) == 200.0
    assert _packet_frame_pose_age_ms({"latest_frame_preprocess_age_ms": "bad", "stream_age_ms": 55.0}) == 55.0
    assert _packet_frame_pose_age_ms({"latest_frame_preprocess_age_ms": 20.0, "stream_age_ms": 80.0}) == 80.0
    assert _packet_frame_pose_age_ms({"latest_frame_preprocess_age_ms": 2500.0, "queue_age_ms": 20.0}) == 20.0


def test_state_snapshot_preserves_local_receive_timestamp() -> None:
    snapshot = _state_message_to_snapshot({"state": "IDLE", "_local_receive_ts": 123.45, "busy_action": "teleop"})

    assert snapshot["_local_receive_ts"] == 123.45
    assert snapshot["busy_action"] == "teleop"


def test_debug_current_cyl_pose_rejects_non_finite_values() -> None:
    assert (
        _current_cyl_pose(
            {
                "robot_cyl": {
                    "theta_deg": float("nan"),
                    "radius_mm": 150.0,
                    "z_mm": 190.0,
                }
            }
        )
        is None
    )


def test_debug_snapshot_local_age_uses_receive_timestamp() -> None:
    assert _snapshot_local_age_ms({"_local_receive_ts": 10.0}, now=10.125) == pytest.approx(125.0)
    assert _snapshot_local_age_ms({"_local_receive_ts": 0.0}, now=10.0) == float("inf")


def test_debug_fetch_fresh_state_waits_for_new_state(monkeypatch) -> None:
    now_values = iter([10.30, 10.30, 10.30, 10.10, 10.10])

    class FakeClient:
        def __init__(self) -> None:
            self.messages = [
                {"state": "IDLE", "_local_receive_ts": 10.0},
                {"state": "IDLE", "_local_receive_ts": 10.05},
            ]

        def fetch_state(self, *, timeout_sec: float) -> dict[str, object]:
            del timeout_sec
            return self.messages.pop(0)

    monkeypatch.setattr("hybrid_controller.tools.debug_vision_grasp_flow.time.perf_counter", lambda: next(now_values))
    monkeypatch.setattr("hybrid_controller.tools.debug_vision_grasp_flow.time.sleep", lambda _sec: None)

    snapshot, age_ms = _fetch_fresh_state_snapshot(FakeClient(), timeout_sec=1.0, max_age_ms=100.0)

    assert snapshot["_local_receive_ts"] == 10.05
    assert age_ms == pytest.approx(50.0)


def test_continuous_snapshot_blocks_non_teleop_busy_state() -> None:
    assert _continuous_snapshot_blocks_teleop({"busy": True, "busy_action": "pick"}) is True
    assert _continuous_snapshot_blocks_teleop({"busy": True, "busy_action": "teleop"}) is False
    assert _continuous_snapshot_blocks_teleop({"busy": False, "busy_action": "pick"}) is False
    assert _continuous_snapshot_blocks_teleop({"state": "PICK_DESCEND", "busy": False}) is True
    assert _continuous_snapshot_blocks_teleop({"state": "ERROR", "busy": False}) is True
    assert _continuous_snapshot_blocks_teleop({"state": "MOVING_XY", "busy": False, "busy_action": ""}) is True
    assert _continuous_snapshot_blocks_teleop({"state": "MOVING_XY", "busy": False, "busy_action": "teleop"}) is False
    assert _continuous_snapshot_blocks_teleop({"state": "CARRY_READY", "busy": False, "carrying": True}) is True


def test_debug_wait_for_idle_requires_settled_executor_state(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.snapshots = [
                {"state": "MOVING_XY", "busy": False},
                {"state": "IDLE", "busy": False},
            ]

        def fetch_state(self, *, timeout_sec: float) -> dict[str, object]:
            del timeout_sec
            return self.snapshots.pop(0)

    monkeypatch.setattr("hybrid_controller.tools.debug_vision_grasp_flow.time.sleep", lambda _sec: None)

    snapshot = _wait_for_idle(client=FakeClient(), timeout_sec=1.0, poll_sec=0.01)

    assert snapshot is not None
    assert snapshot["state"] == "IDLE"


def test_low_height_calibration_state_requires_idle_not_moving_xy() -> None:
    assert _low_height_state_is_safe({"state": "MOVING_XY", "busy": False, "carrying": False}) is False
    assert _low_height_state_is_safe({"state": "IDLE", "busy": False, "carrying": False}) is True


def test_debug_resolve_packet_forwards_frame_pose_age() -> None:
    packet = {
        "mapping_mode": "delta_servo",
        "calibration_ready": True,
        "calibration_profile_required": False,
        "slots": [
            {
                "slot_id": 1,
                "valid": True,
                "camera_to_world_raw": [12.0, -8.0, 0.0],
                "grasp_quality": 1.0,
            }
        ],
    }
    resolved = _resolve_packet(
        packet=packet,
        config=AppConfig(vision_frame_pose_max_age_ms=10.0),
        snapshot={"robot_xy": [0.0, -120.0], "limits_cyl": {"theta_deg": [-120.0, 120.0], "radius_mm": [50.0, 280.0]}},
        snapshot_age_ms=1.0,
        frame_pose_age_ms=25.0,
    )

    slot = resolved["slots"][0]
    assert slot["invalid_reason"] == "robot_pose_stale_for_frame"
    assert slot["frame_pose_age_ms"] == 25.0


def test_debug_frame_pose_age_uses_state_timestamp_not_image_processing_age() -> None:
    from hybrid_controller.tools.debug_vision_grasp_flow import _snapshot_frame_pose_age_ms

    packet = {
        "capture_ts": 100.000,
        "stream_age_ms": 325.0,
        "queue_age_ms": 310.0,
    }
    snapshot = {"_local_receive_ts": 100.018}

    assert _snapshot_frame_pose_age_ms(snapshot, packet) == pytest.approx(18.0)


def test_static_debug_snapshot_without_local_timestamp_does_not_fake_frame_pose_age() -> None:
    packet = {
        "capture_ts": 100.000,
        "stream_age_ms": 325.0,
        "queue_age_ms": 310.0,
    }

    assert _frame_pose_age_for_static_snapshot({"state": "IDLE"}, packet) is None


def test_static_debug_snapshot_with_local_timestamp_uses_true_pose_age() -> None:
    packet = {
        "capture_ts": 100.000,
        "stream_age_ms": 325.0,
        "queue_age_ms": 310.0,
    }

    assert _frame_pose_age_for_static_snapshot(
        {"_local_receive_ts": 100.020, "state": "MOVING_XY", "busy": True, "busy_action": "move"},
        packet,
    ) == pytest.approx(20.0)


def test_static_debug_idle_snapshot_does_not_block_on_frame_pose_age() -> None:
    packet = {
        "capture_ts": 100.000,
        "stream_age_ms": 325.0,
        "queue_age_ms": 310.0,
    }

    assert (
        _frame_pose_age_for_static_snapshot(
            {"_local_receive_ts": 100.350, "state": "IDLE", "busy": False, "busy_action": ""},
            packet,
        )
        is None
    )


def test_debug_execute_loop_requires_explicit_opt_in(capsys) -> None:
    exit_code = main(["--execute", "--max-steps", "2", "--no-ros", "--detector", "fallback"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--allow-execute-loop" in captured.err


def test_debug_continuous_mode_requires_persistent_camera(capsys) -> None:
    exit_code = main(["--servo-mode", "continuous", "--execute", "--no-ros", "--detector", "fallback"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--persistent-camera" in captured.err


def test_debug_continuous_mode_requires_execute(capsys) -> None:
    exit_code = main(["--servo-mode", "continuous", "--persistent-camera", "--no-ros", "--detector", "fallback"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--execute" in captured.err


def test_low_height_alignment_guard_rejects_large_offsets(capsys) -> None:
    exit_code = low_height_alignment_main(
        [
            "--slot-id",
            "1",
            "--theta-offsets-deg",
            "0,2.0",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "theta offset" in captured.err


def test_low_height_alignment_sample_alignment_target_requires_consistent_samples() -> None:
    assert _sample_alignment_target(
        [
            {"alignment_target_pixel": [319.5, 241.0]},
            {"alignment_target_pixel": [319.7, 240.9]},
        ]
    ) == pytest.approx((319.5, 241.0))

    with pytest.raises(ValueError, match="alignment_target_pixel changed"):
        _sample_alignment_target(
            [
                {"alignment_target_pixel": [320.0, 240.0]},
                {"alignment_target_pixel": [330.0, 240.0]},
            ]
        )


def test_debug_calibration_stage_switches_to_confirm_after_descent_starts() -> None:
    stage, z_mm = _current_calibration_stage(
        AppConfig(vision_pick_search_z_mm=190.0, vision_pick_confirm_z_mm=120.0),
        {"robot_z": 185.0},
    )

    assert stage == "confirm"
    assert z_mm == 185.0


def test_debug_confirm_stage_uses_low_action_tolerance() -> None:
    packet = {
        "mapping_mode": "delta_servo",
        "calibration_ready": True,
        "calibration_profile_required": False,
        "slots": [
            {
                "slot_id": 1,
                "valid": True,
                "camera_to_world_raw": [0.0, 0.0, 0.0],
                "grasp_quality": 1.0,
            }
        ],
    }

    resolved = _resolve_packet(
        packet=packet,
        config=AppConfig(vision_frame_pose_max_age_ms=1000.0),
        snapshot={
            "robot_xy": [0.0, -120.0],
            "robot_z": 140.0,
            "limits_cyl": {"theta_deg": [-120.0, 120.0], "radius_mm": [50.0, 280.0]},
        },
        snapshot_age_ms=1.0,
        frame_pose_age_ms=1.0,
    )

    assert resolved["slots"][0]["actionable"] is True


def test_debug_confirm_stage_can_require_two_px_alignment() -> None:
    packet = {
        "mapping_mode": "delta_servo",
        "calibration_ready": True,
        "calibration_profile_required": False,
        "slots": [
            {
                "slot_id": 1,
                "valid": True,
                "camera_to_world_raw": [3.0, 0.0, 0.0],
                "center_distance_px": 3.0,
                "servo_required": True,
                "grasp_quality": 1.0,
            }
        ],
    }

    resolved = _resolve_packet(
        packet=packet,
        config=AppConfig(
            vision_frame_pose_max_age_ms=1000.0,
            vision_servo_low_action_tolerance_px=2.0,
            vision_pick_confirm_z_mm=120.0,
            vision_pick_z_tolerance_mm=4.0,
        ),
        snapshot={
            "robot_xy": [0.0, -120.0],
            "robot_z": 120.0,
            "limits_cyl": {"theta_deg": [-120.0, 120.0], "radius_mm": [50.0, 280.0]},
        },
        snapshot_age_ms=1.0,
        frame_pose_age_ms=1.0,
    )

    slot = resolved["slots"][0]
    assert slot["actionable"] is False
    assert slot["invalid_reason"] == "vision_servo_required"
    assert slot["servo_command_point"] is not None


def test_debug_process_frame_batch_marks_black_frame_too_dark() -> None:
    frame = np.full((480, 640, 3), 16, dtype=np.uint8)

    packet, _last_frame, _frame_id, _slots = main.__globals__["_process_frame_batch"](
        frames=[(frame, 1.0)],
        model=None,
        config=AppConfig(vision_action_requires_calibration=False),
        calibration_profile=None,
        snapshot_for_stage={"robot_z": 190.0},
        frame_id_start=0,
        slots=None,
        device=None,
        half=False,
    )

    assert packet["frame_block_reason"] == "frame_too_dark"
    assert packet["frame_quality"]["gray_mean"] == 16.0


def test_debug_command_bias_alignment_target_uses_camera_center() -> None:
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
            "servo": {"target_pixel": [320.0, 240.0]},
            "stage_models": {
                "confirm": {
                    "z_mm": 120.0,
                    "servo": {"target_pixel": [320.0, 223.0]},
                }
            },
        }
    )

    target = _resolve_alignment_target_pixel(
        config=AppConfig(pick_tool_offset_source="command_bias"),
        calibration_profile=profile,
        frame_w=640,
        frame_h=480,
        roi_center=(320, 240),
        calibration_stage="confirm",
        calibration_z_mm=120.0,
    )

    assert target == (320.0, 240.0)


def test_debug_profile_stage_jacobian_converts_xy_response_to_cylindrical() -> None:
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[-0.1, 0.0, 0.0], [0.0, -0.2, 0.0]]},
            "samples_summary": {
                "pixel_to_robot_jacobian": [
                    [10.0, 1.0],
                    [2.0, 4.0],
                ]
            },
        }
    )

    jacobian = _ibvs_jacobian_from_stage_profile(profile, theta_deg=0.0, radius_mm=180.0)

    assert jacobian is not None
    du_dtheta, du_dr, dv_dtheta, dv_dr = jacobian
    assert du_dtheta == pytest.approx(2.0 * 180.0 * 3.141592653589793 / 180.0)
    assert du_dr == pytest.approx(10.0)
    assert dv_dtheta == pytest.approx(4.0 * 180.0 * 3.141592653589793 / 180.0)
    assert dv_dr == pytest.approx(1.0)


def test_debug_command_bias_pick_command_has_single_final_radius_offset() -> None:
    config = AppConfig(
        pick_tool_offset_source="command_bias",
        vision_eye_in_hand_pick_radius_bias_mm=40.0,
        pick_cyl_radius_bias_mm=0.0,
    )

    decision = _decision_for_packet(
        packet={"frame_id": 1},
        config=config,
        snapshot={"robot_cyl": {"theta_deg": 7.0, "radius_mm": 160.0, "z_mm": config.vision_pick_confirm_z_mm}},
        selected_slot={"slot_id": 1, "valid": True, "actionable": True, "command_mode": "world", "command_point": [0.0, -160.0]},
    )

    assert decision["command"] == "PICK_CYL 7.00 200.00"
    assert decision["raw_command"] == "PICK_CYL 7.00 200.00"


def test_debug_final_pick_rewrite_keeps_default_single_bias_when_app_layer_bias_is_zero() -> None:
    command = _rewrite_final_pick_command_for_debug(
        config=AppConfig(),
        command="PICK_CYL 7.00 200.00",
    )

    assert command == "PICK_CYL 7.00 200.00"


def test_continuous_auto_selection_locks_pending_slot_immediately() -> None:
    packet = {
        "slots": [
            {
                "slot_id": 1,
                "valid": True,
                "actionable": False,
                "invalid_reason": "vision_servo_required",
                "center_distance_px": 141.0,
                "confidence": 0.95,
            },
            {
                "slot_id": 3,
                "valid": True,
                "actionable": False,
                "invalid_reason": "vision_servo_required",
                "center_distance_px": 21.0,
                "confidence": 0.94,
            },
        ]
    }

    assert _continuous_slot_id_for_selection(None, {"slot_id": 1, "stable_frames": 0, "lost_frames": 0}) == 1
    assert _select_slot(packet, _continuous_slot_id_for_selection(None, {"slot_id": 1, "stable_frames": 0}))[
        "slot_id"
    ] == 1


def test_continuous_auto_selection_keeps_slot_after_stable_center() -> None:
    assert _continuous_slot_id_for_selection(None, {"slot_id": 3, "stable_frames": 2, "lost_frames": 0}) == 3


def test_debug_continuous_command_bias_pick_command_has_single_final_radius_offset() -> None:
    config = AppConfig(
        pick_tool_offset_source="command_bias",
        vision_eye_in_hand_pick_radius_bias_mm=40.0,
        pick_cyl_radius_bias_mm=0.0,
    )

    decision = _continuous_decision_for_packet(
        packet={"frame_id": 1, "queue_age_ms": 1.0},
        config=config,
        snapshot={"robot_cyl": {"theta_deg": 7.0, "radius_mm": 160.0, "z_mm": config.vision_pick_confirm_z_mm}},
        selected_slot={
            "slot_id": 1,
            "valid": True,
            "actionable": True,
            "pixel_center": [321, 241],
            "center_distance_px": 1.4,
            "action_tolerance_px": 20.0,
            "command": "PICK_WORLD 0.00 -160.00",
        },
        pending={"slot_id": 1, "stable_frames": 1, "pick_ready_frames": 1},
    )

    assert decision["action"] == "PICK_READY"
    assert decision["command"] == "PICK_CYL 7.00 200.00"
    assert decision["raw_command"] == "PICK_CYL 7.00 200.00"


def test_debug_continuous_parser_exposes_stop_at_confirm_switch() -> None:
    parser = main.__globals__["build_parser"]()

    args = parser.parse_args(
        [
            "--servo-mode",
            "continuous",
            "--persistent-camera",
            "--execute",
            "--continuous-stop-at-confirm",
            "--detector",
            "fallback",
        ]
    )

    assert args.continuous_stop_at_confirm is True


def test_debug_continuous_parser_exposes_confirm_z_tolerance_override() -> None:
    parser = main.__globals__["build_parser"]()

    args = parser.parse_args(["--confirm-z-tolerance-mm", "0.6", "--detector", "fallback"])

    assert args.confirm_z_tolerance_mm == pytest.approx(0.6)


def test_debug_continuous_parser_exposes_ibvs_jacobian_overrides() -> None:
    parser = main.__globals__["build_parser"]()

    args = parser.parse_args(
        [
            "--continuous-horizontal-mode",
            "ibvs_dls",
            "--continuous-ibvs-du-dtheta",
            "-5.3",
            "--continuous-ibvs-du-dradius",
            "-3.0",
            "--continuous-ibvs-dv-dtheta",
            "21.7",
            "--continuous-ibvs-dv-dradius",
            "18.6",
            "--detector",
            "fallback",
        ]
    )

    assert args.continuous_horizontal_mode == "ibvs_dls"
    assert args.continuous_ibvs_du_dtheta == pytest.approx(-5.3)
    assert args.continuous_ibvs_du_dradius == pytest.approx(-3.0)
    assert args.continuous_ibvs_dv_dtheta == pytest.approx(21.7)
    assert args.continuous_ibvs_dv_dradius == pytest.approx(18.6)


def test_debug_low_height_centering_check_shortcut_sets_safe_continuous_defaults(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_run_continuous_servo_flow(*, args, config, report, **_kwargs):
        captured["args"] = args
        captured["config"] = config
        captured["report"] = report
        return 0

    monkeypatch.setattr(
        "hybrid_controller.tools.debug_vision_grasp_flow._run_continuous_servo_flow",
        fake_run_continuous_servo_flow,
    )
    monkeypatch.setattr("hybrid_controller.tools.debug_vision_grasp_flow._load_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "hybrid_controller.tools.debug_vision_grasp_flow._resolve_device",
        lambda _device: ("cpu", False),
    )
    monkeypatch.setattr(
        "hybrid_controller.tools.debug_vision_grasp_flow.load_vision_grasp_profile",
        lambda _config: type("Profile", (), {"ready": False, "error": ""})(),
    )
    monkeypatch.setitem(sys.modules, "cv2", type("FakeCv2", (), {})())

    exit_code = main(
        [
            "--low-height-centering-check",
            "--no-ros",
            "--vision-grasp-profile-optional",
            "--detector",
            "fallback",
            "--continuous-horizontal-mode",
            "ibvs_dls",
            "--continuous-ibvs-du-dtheta",
            "-5.3",
            "--continuous-ibvs-du-dradius",
            "-3.0",
            "--continuous-ibvs-dv-dtheta",
            "21.7",
            "--continuous-ibvs-dv-dradius",
            "18.6",
            "--continuous-low-height-z-rate-scale",
            "0.65",
            "--continuous-low-height-pause-descent-band-mm",
            "4.0",
            "--continuous-descent-low-error-z-above-confirm-mm",
            "2.0",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    args = captured["args"]
    config = captured["config"]
    report = captured["report"]
    assert args.servo_mode == "continuous"
    assert args.persistent_camera is True
    assert args.execute is True
    assert args.allow_pick is True
    assert args.allow_real_pick is False
    assert args.continuous_stop_at_confirm is True
    assert args.process_latest_frames == 1
    assert args.timeout_sec == pytest.approx(5.0)
    assert args.capture_backend == "http"
    assert config.vision_pick_confirm_z_mm == pytest.approx(120.0)
    assert config.vision_pick_z_tolerance_mm == pytest.approx(1.0)
    assert config.vision_servo_measurement_point == "geometry_subpixel"
    assert config.vision_continuous_servo_horizontal_mode == "ibvs_dls"
    assert config.vision_continuous_servo_ibvs_du_dtheta_px_per_deg == pytest.approx(-5.3)
    assert config.vision_continuous_servo_ibvs_du_dradius_px_per_mm == pytest.approx(-3.0)
    assert config.vision_continuous_servo_ibvs_dv_dtheta_px_per_deg == pytest.approx(21.7)
    assert config.vision_continuous_servo_ibvs_dv_dradius_px_per_mm == pytest.approx(18.6)
    assert config.vision_continuous_servo_low_height_z_rate_scale == pytest.approx(0.65)
    assert config.vision_continuous_servo_low_height_pause_descent_band_mm == pytest.approx(4.0)
    assert config.vision_continuous_servo_descent_low_error_z_above_confirm_mm == pytest.approx(2.0)
    assert report["low_height_centering_check"] is True
    assert report["confirm_z_mm"] == pytest.approx(120.0)
    assert report["continuous_horizontal_mode"] == "ibvs_dls"
    assert report["continuous_ibvs_jacobian"]["du_dtheta_px_per_deg"] == pytest.approx(-5.3)


def test_debug_low_height_centering_check_preserves_longer_timeout(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_run_continuous_servo_flow(*, args, **_kwargs):
        captured["args"] = args
        return 0

    monkeypatch.setattr(
        "hybrid_controller.tools.debug_vision_grasp_flow._run_continuous_servo_flow",
        fake_run_continuous_servo_flow,
    )
    monkeypatch.setattr("hybrid_controller.tools.debug_vision_grasp_flow._load_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "hybrid_controller.tools.debug_vision_grasp_flow._resolve_device",
        lambda _device: ("cpu", False),
    )
    monkeypatch.setattr(
        "hybrid_controller.tools.debug_vision_grasp_flow.load_vision_grasp_profile",
        lambda _config: type("Profile", (), {"ready": False, "error": ""})(),
    )
    monkeypatch.setitem(sys.modules, "cv2", type("FakeCv2", (), {})())

    exit_code = main(
        [
            "--low-height-centering-check",
            "--no-ros",
            "--vision-grasp-profile-optional",
            "--detector",
            "fallback",
            "--timeout-sec",
            "8",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured["args"].timeout_sec == pytest.approx(8.0)


def test_debug_parser_accepts_low_height_measurement_point(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_run_continuous_servo_flow(*, args, config, report, **_kwargs):
        captured["args"] = args
        captured["config"] = config
        captured["report"] = report
        return 0

    monkeypatch.setattr(
        "hybrid_controller.tools.debug_vision_grasp_flow._run_continuous_servo_flow",
        fake_run_continuous_servo_flow,
    )
    monkeypatch.setattr("hybrid_controller.tools.debug_vision_grasp_flow._load_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "hybrid_controller.tools.debug_vision_grasp_flow._resolve_device",
        lambda _device: ("cpu", False),
    )
    monkeypatch.setattr(
        "hybrid_controller.tools.debug_vision_grasp_flow.load_vision_grasp_profile",
        lambda _config: type("Profile", (), {"ready": False, "error": ""})(),
    )
    monkeypatch.setitem(sys.modules, "cv2", type("FakeCv2", (), {})())

    exit_code = main(
        [
            "--servo-mode",
            "continuous",
            "--persistent-camera",
            "--execute",
            "--no-ros",
            "--vision-grasp-profile-optional",
            "--detector",
            "fallback",
            "--low-height-measurement-point",
            "color_block_subpixel",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured["config"].vision_servo_low_height_measurement_point == "color_block_subpixel"
    assert captured["report"]["low_height_measurement_point"] == "color_block_subpixel"


def test_debug_confirm_recheck_reports_actual_low_height_measurement_point(monkeypatch, tmp_path) -> None:
    args = type(
        "Args",
        (),
        {
            "continuous_confirm_recheck_settle_sec": 0.0,
            "continuous_confirm_recheck_repeats": 2,
            "continuous_confirm_recheck_max_spread_px": 3.0,
            "frames": 1,
            "drain_frames": 0,
            "timeout_sec": 1.0,
            "command_timeout_sec": 1.0,
            "ros_timeout_sec": 1.0,
            "process_latest_frames": 1,
        },
    )()
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_servo_measurement_point="geometry_subpixel",
        vision_servo_low_height_measurement_point="top_face_subpixel",
        vision_continuous_servo_pick_ready_center_px=2.0,
    ).resolved()

    class FakeReader:
        def __init__(self) -> None:
            self.reopen_count = 0

        def reopen(self) -> None:
            self.reopen_count += 1

        def read(self, *, frame_count, drain_frames, timeout_sec):
            del frame_count, drain_frames, timeout_sec
            return "http://camera", [(np.zeros((16, 16, 3), dtype=np.uint8), 10.0)]

        def transport_stats(self) -> dict[str, object]:
            return {"reader": "fake", "frames_rejected": 0}

    class FakeCv2:
        def imwrite(self, *_args, **_kwargs) -> bool:
            return True

    packet = {
        "frame_id": 1,
        "queue_age_ms": 1.0,
        "alignment_target_pixel": [320.0, 240.0],
        "slots": [
            {
                "slot_id": 1,
                "valid": True,
                "measurement_point": "top_face_subpixel",
                "center_distance_px": 0.75,
                "pixel_center_f": [340.0, 240.0],
                "geometry_center_f": [330.0, 240.0],
                "top_face_center_f": [320.75, 240.0],
                "grasp_pixel_f": [331.0, 240.0],
                "alignment_target_pixel": [320.0, 240.0],
            }
        ],
    }

    monkeypatch.setattr("hybrid_controller.tools.debug_vision_grasp_flow.time.sleep", lambda _sec: None)
    monkeypatch.setattr(
        "hybrid_controller.tools.debug_vision_grasp_flow._wait_for_idle",
        lambda **_kwargs: {
            "state": "IDLE",
            "busy": False,
            "robot_z": 120.0,
            "robot_cyl": {"theta_deg": 7.0, "radius_mm": 170.0, "z_mm": 120.0},
        },
    )
    monkeypatch.setattr(
        "hybrid_controller.tools.debug_vision_grasp_flow._process_frame_batch",
        lambda **_kwargs: (dict(packet), np.zeros((16, 16, 3), dtype=np.uint8), 2, None),
    )
    monkeypatch.setattr(
        "hybrid_controller.tools.debug_vision_grasp_flow._resolve_packet",
        lambda **kwargs: dict(kwargs["packet"]),
    )
    monkeypatch.setattr("hybrid_controller.tools.debug_vision_grasp_flow._save_overlay", lambda **_kwargs: None)

    recheck, _frame_id, _debug_slots = _continuous_confirm_recheck(
        args=args,
        config=config,
        cv2_module=FakeCv2(),
        model=None,
        calibration_profile=None,
        reader=FakeReader(),
        client=object(),
        output_dir=tmp_path,
        slot_id=1,
        frame_id=0,
        debug_slots=None,
        device=None,
        half=False,
    )

    assert recheck["measurement_point"] == "top_face_subpixel"
    assert recheck["sample_measurement_points"] == ["top_face_subpixel", "top_face_subpixel"]
    assert recheck["median_center_distance_px"] == pytest.approx(0.75)
    assert recheck["samples"][0]["measurement_point"] == "top_face_subpixel"


def test_center_sequence_diagnostics_ranks_stable_low_height_point(tmp_path) -> None:
    for index, (geometry_x, top_x, color_x) in enumerate(
        ((340.0, 321.0, 320.4), (358.0, 320.5, 320.2), (335.0, 320.0, 320.1)),
        start=1,
    ):
        step = tmp_path / f"step_{index:02d}"
        step.mkdir()
        (step / "raw.jpg").write_bytes(b"not-a-real-jpeg-needed-for-sequence-mode")
        (step / "packet.json").write_text(
            json.dumps(
                {
                    "alignment_target_pixel": [320.0, 240.0],
                    "camera_transport": {
                        "reader": "_HttpMjpegCapture",
                        "content_length_frames": index,
                        "frames_rejected": 0,
                    },
                    "slots": [
                        {
                            "slot_id": 1,
                            "valid": True,
                            "measurement_point": "top_face_subpixel",
                            "center_distance_px": abs(top_x - 320.0),
                            "pixel_center_f": [350.0, 240.0],
                            "geometry_center_f": [geometry_x, 240.0],
                            "color_block_center_f": [color_x, 240.0],
                            "top_face_center_f": [top_x, 240.0],
                            "grasp_pixel_f": [330.0, 240.0],
                            "confidence": 0.9,
                            "area_px": 20000,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    output_path = tmp_path / "sequence_report.json"
    report = diagnose_center_sequence(tmp_path, slot_id=1, output_path=output_path)

    assert output_path.exists()
    assert report["recommended_low_height_packet_key"] == "color_block_center_f"
    assert report["recommended_low_height_measurement_point"] == "color_block_subpixel"
    assert report["point_summary"]["color_block_center_f"]["repeat_spread_px"] <= 0.4
    assert report["point_summary"]["top_face_center_f"]["repeat_spread_px"] <= 1.0
    assert report["point_summary"]["geometry_center_f"]["jump_count"] >= 1
    assert report["camera_transport_summary"]["content_length_frames_last"] == pytest.approx(3.0)


def test_debug_continuous_confirm_stop_keeps_servoing_until_pick_ready_center() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_pick_z_tolerance_mm=4.0,
        vision_continuous_servo_pick_ready_center_px=2.0,
    )

    decision = _continuous_decision_for_packet(
        packet={"frame_id": 1, "queue_age_ms": 1.0},
        config=config,
        snapshot={"robot_cyl": {"theta_deg": 7.65, "radius_mm": 176.0, "z_mm": 120.0}},
        selected_slot={
            "slot_id": 1,
            "valid": True,
            "actionable": False,
            "invalid_reason": "vision_servo_required",
            "center_distance_px": 7.9,
            "servo_command_point": [7.65, 175.0],
            "confidence": 0.9,
            "area_px": 35000,
            "bbox": [208, 107, 433, 358],
        },
        pending={"slot_id": 1, "stable_frames": 0},
    )

    assert decision["action"] == "SERVO"
    assert decision["reason"] == "continuous_servo"
    assert decision["command"] is None
    assert float(decision["trace"]["current_z_mm"]) == pytest.approx(120.0)
    assert float(decision["trace"]["center_distance_px"]) > config.vision_continuous_servo_pick_ready_center_px


def test_debug_low_height_refine_target_is_clamped() -> None:
    target = _clamp_refine_target(
        current_pose=(7.85, 175.0, 120.0),
        target_theta=7.10,
        target_radius=180.0,
        max_theta_step_deg=0.25,
        max_radius_step_mm=1.5,
    )

    assert target == pytest.approx((7.60, 176.5, 120.0))
    assert _servo_command_point_from_slot({"servo_command_point": [7.8, 176.0]}) == pytest.approx((7.8, 176.0))
    assert _servo_command_point_from_slot({"servo_command_point": ["bad", 176.0]}) is None


def test_debug_low_height_refine_target_applies_damping_before_clamp() -> None:
    target = _clamp_refine_target(
        current_pose=(8.35, 179.55, 120.0),
        target_theta=8.83,
        target_radius=180.58,
        max_theta_step_deg=0.25,
        max_radius_step_mm=4.0,
        step_gain=0.45,
    )

    assert target == pytest.approx((8.566, 180.0135, 120.0))


def test_debug_low_height_refine_gain_is_bounded() -> None:
    assert _clamp_refine_target(
        current_pose=(1.0, 100.0, 120.0),
        target_theta=2.0,
        target_radius=110.0,
        max_theta_step_deg=10.0,
        max_radius_step_mm=20.0,
        step_gain=0.0,
    ) == pytest.approx((1.01, 100.1, 120.0))
    assert _clamp_refine_target(
        current_pose=(1.0, 100.0, 120.0),
        target_theta=2.0,
        target_radius=110.0,
        max_theta_step_deg=10.0,
        max_radius_step_mm=20.0,
        step_gain=2.0,
    ) == pytest.approx((2.0, 110.0, 120.0))


def test_debug_low_height_refine_can_start_before_exact_confirm_z() -> None:
    assert _continuous_low_height_refine_requested(
        enabled=True,
        stop_at_confirm=True,
        current_z_mm=125.7,
        confirm_z_mm=120.0,
        center_distance_px=20.0,
        pick_ready_center_px=8.0,
        guard_band_mm=30.0,
    )
    assert not _continuous_low_height_refine_requested(
        enabled=True,
        stop_at_confirm=True,
        current_z_mm=151.0,
        confirm_z_mm=120.0,
        center_distance_px=20.0,
        pick_ready_center_px=8.0,
        guard_band_mm=30.0,
    )
    assert not _continuous_low_height_refine_requested(
        enabled=True,
        stop_at_confirm=True,
        current_z_mm=125.7,
        confirm_z_mm=120.0,
        center_distance_px=5.0,
        pick_ready_center_px=8.0,
        guard_band_mm=30.0,
    )


def test_debug_pose_from_confirm_recheck_uses_stopped_pose() -> None:
    assert _pose_from_confirm_recheck({"pose_cyl": [7.4, 167.8, 120.0]}) == pytest.approx((7.4, 167.8, 120.0))
    assert _pose_from_confirm_recheck({"pose_cyl": ["bad", 167.8, 120.0]}) is None
    assert _pose_from_confirm_recheck({}) is None


def test_debug_continuous_parser_defaults_low_refine_radius_step_to_live_probe_size() -> None:
    parser = main.__globals__["build_parser"]()

    args = parser.parse_args(
        [
            "--servo-mode",
            "continuous",
            "--persistent-camera",
            "--execute",
            "--detector",
            "fallback",
        ]
    )

    assert args.continuous_low_height_refine_max_radius_step_mm == pytest.approx(4.0)
    assert args.continuous_low_height_refine_gain == pytest.approx(0.45)
    assert args.continuous_low_height_discrete_refine is False
    assert args.process_latest_frames == 1


def test_debug_continuous_parser_exposes_low_height_rate_scale_overrides() -> None:
    parser = main.__globals__["build_parser"]()

    args = parser.parse_args(
        [
            "--servo-mode",
            "continuous",
            "--persistent-camera",
            "--execute",
            "--detector",
            "fallback",
            "--continuous-low-height-coarse-rate-scale",
            "0.55",
            "--continuous-low-height-fine-rate-scale",
            "0.25",
        ]
    )

    assert args.continuous_low_height_coarse_rate_scale == pytest.approx(0.55)
    assert args.continuous_low_height_fine_rate_scale == pytest.approx(0.25)


def test_debug_low_height_refine_move_resets_pending_anchor_contract() -> None:
    source = main.__globals__["_run_continuous_servo_flow"].__code__

    assert "reset_after_stopped_refine_move" in source.co_consts


def test_debug_stopped_step_mode_can_convert_soft_descent_to_small_move() -> None:
    target = _continuous_stopped_motion_target(
        current_pose=(7.0, 170.0, 160.0),
        selected_slot={"servo_command_point": [7.5, 172.0]},
        center_distance_px=12.0,
        center_allow_px=8.0,
        z_rate_mm_s=-4.0,
        confirm_z_mm=120.0,
        z_tolerance_mm=4.0,
        z_step_mm=5.0,
        refine_z_band_above_confirm_mm=20.0,
        max_theta_step_deg=0.2,
        max_radius_step_mm=1.0,
    )

    assert target is not None
    reason, pose, meta = target
    assert reason == "stopped_descent_step"
    assert pose == pytest.approx((7.0, 170.0, 155.0))
    assert meta["z_step_mm"] == pytest.approx(5.0)


def test_debug_stopped_step_mode_refines_only_near_confirm_band() -> None:
    target = _continuous_stopped_motion_target(
        current_pose=(7.0, 170.0, 132.0),
        selected_slot={"servo_command_point": [7.5, 172.0]},
        center_distance_px=12.0,
        center_allow_px=8.0,
        z_rate_mm_s=0.0,
        confirm_z_mm=120.0,
        z_tolerance_mm=4.0,
        z_step_mm=5.0,
        refine_z_band_above_confirm_mm=20.0,
        max_theta_step_deg=0.2,
        max_radius_step_mm=1.0,
    )

    assert target is not None
    reason, pose, meta = target
    assert reason == "stopped_horizontal_refine"
    assert pose == pytest.approx((7.2, 171.0, 132.0))
    assert meta["source_point"] == pytest.approx([7.5, 172.0])


def test_debug_low_height_rebound_is_recoverable_near_confirm() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_pick_z_tolerance_mm=4.0,
        vision_continuous_servo_low_height_rebound_recover_band_mm=10.0,
        vision_continuous_servo_low_height_rebound_recover_attempts=2,
    ).resolved()

    assert (
        _continuous_stop_reason_is_recoverable(
            "frame_too_dark",
            trace={},
            pending={"slot_id": 1},
            config=config,
            recoveries_used=99,
        )
        is True
    )

    assert (
        _continuous_stop_reason_is_recoverable(
            "low_height_error_rebounded",
            trace={"current_z_mm": 124.2, "confirm_z_mm": 120.0},
            pending={"slot_id": 1},
            config=config,
            recoveries_used=1,
        )
        is True
    )
    assert (
        _continuous_stop_reason_is_recoverable(
            "low_height_best_error_rebounded",
            trace={"current_z_mm": 124.2, "confirm_z_mm": 120.0},
            pending={"slot_id": 1},
            config=config,
            recoveries_used=1,
        )
        is True
    )


def test_debug_low_height_rebound_recovery_respects_band_and_attempt_limit() -> None:
    config = AppConfig(
        vision_pick_confirm_z_mm=120.0,
        vision_pick_z_tolerance_mm=4.0,
        vision_continuous_servo_low_height_rebound_recover_band_mm=10.0,
        vision_continuous_servo_low_height_rebound_recover_attempts=2,
    ).resolved()

    assert (
        _continuous_stop_reason_is_recoverable(
            "low_height_error_rebounded",
            trace={"current_z_mm": 134.0, "confirm_z_mm": 120.0},
            pending={"slot_id": 1},
            config=config,
            recoveries_used=0,
        )
        is False
    )
    assert (
        _continuous_stop_reason_is_recoverable(
            "low_height_error_rebounded",
            trace={"current_z_mm": 124.0, "confirm_z_mm": 120.0},
            pending={"slot_id": 1},
            config=config,
            recoveries_used=2,
        )
        is False
    )
    assert (
        _continuous_stop_reason_is_recoverable(
            "low_height_best_error_rebounded",
            trace={"current_z_mm": 124.0, "confirm_z_mm": 120.0},
            pending={"slot_id": 1},
            config=config,
            recoveries_used=2,
        )
        is False
    )


def test_low_height_alignment_cli_overrides_measurement_point_after_profile(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def connect(self) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeReader:
        def __init__(self, **_kwargs) -> None:
            return None

        def close(self) -> None:
            return None

        def reopen(self) -> None:
            return None

    def fake_measure_slot(**kwargs):
        captured["config"] = kwargs["config"]
        sample = {
            "pose_cyl": [7.0, 170.0, 120.0],
            "pose_xy": [0.0, -170.0],
            "pixel": [320.0, 240.0],
            "measurement_point": "top_face_subpixel",
        }
        packet = {"frame_id": 1}
        return sample, packet, np.zeros((16, 16, 3), dtype=np.uint8), 1, None

    monkeypatch.setattr(
        "hybrid_controller.tools.calibrate_low_height_alignment.RosBridgeClient",
        lambda **_kwargs: FakeClient(),
    )
    monkeypatch.setattr("hybrid_controller.tools.calibrate_low_height_alignment._PersistentCaptureReader", FakeReader)
    monkeypatch.setattr("hybrid_controller.tools.calibrate_low_height_alignment._freeze_sucker", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "hybrid_controller.tools.calibrate_low_height_alignment._wait_for_idle",
        lambda *_a, **_k: {
            "state": "IDLE",
            "busy": False,
            "robot_cyl": {"theta_deg": 7.0, "radius_mm": 170.0, "z_mm": 120.0},
        },
    )
    monkeypatch.setattr("hybrid_controller.tools.calibrate_low_height_alignment._load_model", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "hybrid_controller.tools.calibrate_low_height_alignment._resolve_device",
        lambda _device: ("cpu", False),
    )
    monkeypatch.setattr(
        "hybrid_controller.tools.calibrate_low_height_alignment.load_vision_grasp_profile",
        lambda _config: type("Profile", (), {"ready": False, "error": ""})(),
    )
    monkeypatch.setattr(
        "hybrid_controller.tools.calibrate_low_height_alignment.VisionCalibrationProfile.load",
        lambda _path: VisionCalibrationProfile.from_dict(
            {
                "profile_id": "unit-profile",
                "image_size": [640, 480],
                "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            }
        ),
    )
    monkeypatch.setattr("hybrid_controller.tools.calibrate_low_height_alignment._measure_slot", fake_measure_slot)

    exit_code = low_height_alignment_main(
        [
            "--slot-id",
            "1",
            "--z-mm",
            "120",
            "--dry-run",
            "--low-height-measurement-point",
            "top_face_subpixel",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured["config"].vision_servo_low_height_measurement_point == "top_face_subpixel"
