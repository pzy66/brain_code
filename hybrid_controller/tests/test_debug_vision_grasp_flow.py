from __future__ import annotations

import numpy as np
import pytest

from hybrid_controller.config import AppConfig
from hybrid_controller.tools.calibrate_low_height_alignment import _state_is_safe as _low_height_state_is_safe
from hybrid_controller.tools.calibrate_low_height_alignment import main as low_height_alignment_main
from hybrid_controller.tools.debug_vision_grasp_flow import (
    _capture_frames_from_candidates,
    _clamp_refine_target,
    _continuous_decision_for_packet,
    _continuous_slot_id_for_selection,
    _continuous_snapshot_blocks_teleop,
    _current_calibration_stage,
    _current_cyl_pose,
    _decision_for_packet,
    main,
    _override_profile_center_tolerance,
    _packet_frame_pose_age_ms,
    _PersistentCaptureReader,
    _resolve_alignment_target_pixel,
    _resolve_packet,
    _rewrite_final_pick_command_for_debug,
    _fetch_fresh_state_snapshot,
    _state_message_to_snapshot,
    _select_latest_frames,
    _select_slot,
    _servo_command_point_from_slot,
    _snapshot_local_age_ms,
    _wait_for_idle,
)
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
        def __init__(self, url, *, cv2_module, timeout_sec):
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


def test_debug_continuous_confirm_stop_requires_pick_ready_center() -> None:
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

    assert decision["action"] == "STOP"
    assert decision["reason"] == "settle_near_center"
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
