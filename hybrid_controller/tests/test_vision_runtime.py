from __future__ import annotations

import json
import time

from PyQt5.QtCore import QCoreApplication

from hybrid_controller.config import AppConfig
from hybrid_controller.config import HIWONDER_CAMERA_TOPIC
from hybrid_controller.config import build_hiwonder_camera_stream_url
from hybrid_controller.vision.calibration_profile import VisionCalibrationProfile
import hybrid_controller.vision.runtime as vision_runtime


def _ensure_app() -> QCoreApplication:
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


class _FakeCapture:
    def isOpened(self) -> bool:
        return True

    def set(self, *_args, **_kwargs) -> bool:
        return True

    def read(self):
        return False, None

    def release(self) -> None:
        return None


class _FakeCv2:
    CAP_PROP_BUFFERSIZE = 38

    @staticmethod
    def VideoCapture(_source):
        return _FakeCapture()


class _FakeYOLO:
    def __init__(self, _weights_path: str) -> None:
        self.weights_path = _weights_path

    def __call__(self, *_args, **_kwargs):
        return []


def test_vision_runtime_start_stop_uses_preloaded_dependencies(monkeypatch) -> None:
    app = _ensure_app()
    monkeypatch.setattr(vision_runtime, "_load_vision_dependencies", lambda: (_FakeCv2, _FakeYOLO))

    statuses: list[str] = []
    runtime = vision_runtime.VisionRuntime(
        AppConfig(robot_mode="fake", vision_mode="robot_camera_detection", vision_infer_interval_ms=200),
        calibration_params=None,
        targets_callback=lambda _targets: None,
        packet_callback=lambda _packet: None,
        frame_callback=lambda _frame: None,
        status_callback=statuses.append,
    )

    runtime.start()
    deadline = time.perf_counter() + 0.2
    while time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert runtime.worker is not None
    assert runtime.thread is not None

    runtime.stop()
    deadline = time.perf_counter() + 0.2
    while time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert runtime.worker is None
    assert runtime.thread is None
    assert runtime.healthcheck()["running"] is False
    assert any("Vision runtime started" in status for status in statuses)


def test_infer_interval_controller_respects_hysteresis_and_bounds() -> None:
    worker = vision_runtime._VisionWorker(  # pylint: disable=protected-access
        AppConfig(
            vision_infer_interval_ms=80,
            vision_infer_interval_min_ms=45,
            vision_infer_interval_max_ms=220,
            vision_infer_target_queue_age_ms=90.0,
            vision_infer_hysteresis_ms=15.0,
            vision_infer_adjust_alpha=0.5,
            vision_infer_max_step_up_ms=40.0,
            vision_infer_max_step_down_ms=20.0,
        ),
        calibration_params=None,
        cv2_module=_FakeCv2,
        yolo_class=_FakeYOLO,
    )
    worker._infer_interval_dynamic_ms = 80.0  # pylint: disable=protected-access

    worker._adjust_infer_interval(infer_ms=30.0, queue_age_ms=95.0)  # within hysteresis
    first = worker._infer_interval_dynamic_ms  # pylint: disable=protected-access
    assert 45.0 <= first <= 220.0

    worker._adjust_infer_interval(infer_ms=45.0, queue_age_ms=260.0)
    second = worker._infer_interval_dynamic_ms  # pylint: disable=protected-access
    assert second >= first

    worker._adjust_infer_interval(infer_ms=15.0, queue_age_ms=10.0)
    third = worker._infer_interval_dynamic_ms  # pylint: disable=protected-access
    assert 45.0 <= third <= 220.0


def test_worker_reports_missing_target_pixel_for_target_pixel_pick_flow(tmp_path) -> None:
    profile_path = tmp_path / "current_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "profile_id": "missing-target-test",
                "image_size": [640, 480],
                "mapping": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
                "servo": {"target_pixel": None},
            }
        ),
        encoding="utf-8",
    )
    worker = vision_runtime._VisionWorker(  # pylint: disable=protected-access
        AppConfig(vision_calibration_profile_path=profile_path, pick_tool_offset_source="target_pixel"),
        calibration_params=None,
        cv2_module=_FakeCv2,
        yolo_class=_FakeYOLO,
    )

    assert worker._calibration_profile is not None  # pylint: disable=protected-access
    assert worker._calibration_profile.target_pixel is None  # pylint: disable=protected-access
    assert worker._pending_status is not None  # pylint: disable=protected-access
    assert "servo.target_pixel is missing" in worker._pending_status  # pylint: disable=protected-access


def test_web_video_url_normalization_preserves_topic_slashes() -> None:
    raw = (
        "http://192.168.149.1:8080/stream?"
        "topic=%2Fusb_cam%2Fimage_rect_color&type=mjpeg&width=640&height=480&quality=80"
    )

    normalized = vision_runtime._normalize_web_video_url(raw)  # pylint: disable=protected-access

    assert "topic=/usb_cam/image_rect_color" in normalized
    assert "%2Fusb_cam%2Fimage_rect_color" not in normalized


def test_default_vision_stream_candidates_use_only_hiwonder_official_topic() -> None:
    candidates = AppConfig(robot_host="192.168.149.1").resolve_vision_stream_candidates()

    assert candidates == (build_hiwonder_camera_stream_url("192.168.149.1"),)
    assert HIWONDER_CAMERA_TOPIC in candidates[0]
    assert candidates[0] == (
        "http://192.168.149.1:8080/stream?"
        "topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"
    )
    forbidden_fragments = (
        "/usb_cam/image_raw",
        "/usb_cam/image_color",
        "stream.mjpg",
        "video_feed",
        "action=stream",
        "/dev/video",
    )
    assert all(fragment not in candidates[0] for fragment in forbidden_fragments)


def test_default_vision_stream_url_matches_single_official_candidate() -> None:
    config = AppConfig(robot_host="192.168.149.1")

    assert config.resolve_vision_stream_url() == build_hiwonder_camera_stream_url("192.168.149.1")
    assert config.resolve_vision_stream_url() == config.resolve_vision_stream_candidates()[0]


def test_worker_alignment_target_uses_stage_profile_target_pixel() -> None:
    worker = vision_runtime._VisionWorker(  # pylint: disable=protected-access
        AppConfig(pick_tool_offset_source="target_pixel"),
        calibration_params=None,
        cv2_module=_FakeCv2,
        yolo_class=_FakeYOLO,
    )
    worker._calibration_profile = VisionCalibrationProfile.from_dict(  # pylint: disable=protected-access
        {
            "profile_id": "stage-target-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
            "servo": {"target_pixel": [320.0, 240.0]},
            "stage_models": {
                "confirm": {
                    "z_mm": 120.0,
                    "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
                    "servo": {"target_pixel": [320.0, 223.0]},
                }
            },
        }
    )

    target = worker._resolve_alignment_target_pixel(  # pylint: disable=protected-access
        640,
        480,
        (320, 240),
        calibration_stage="confirm",
        calibration_z_mm=120.0,
    )

    assert target == (320.0, 223.0)


def test_worker_command_bias_alignment_target_uses_camera_center_instead_of_stage_profile() -> None:
    worker = vision_runtime._VisionWorker(  # pylint: disable=protected-access
        AppConfig(pick_tool_offset_source="command_bias"),
        calibration_params=None,
        cv2_module=_FakeCv2,
        yolo_class=_FakeYOLO,
    )
    worker._calibration_profile = VisionCalibrationProfile.from_dict(  # pylint: disable=protected-access
        {
            "profile_id": "stage-target-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
            "servo": {"target_pixel": [320.0, 240.0]},
            "stage_models": {
                "confirm": {
                    "z_mm": 120.0,
                    "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
                    "servo": {"target_pixel": [320.0, 223.0]},
                }
            },
        }
    )

    target = worker._resolve_alignment_target_pixel(  # pylint: disable=protected-access
        640,
        480,
        (320, 240),
        calibration_stage="confirm",
        calibration_z_mm=120.0,
    )

    assert target == (320.0, 240.0)


def test_worker_calibration_stage_switches_to_confirm_after_descent_starts() -> None:
    worker = vision_runtime._VisionWorker(  # pylint: disable=protected-access
        AppConfig(vision_pick_search_z_mm=190.0, vision_pick_confirm_z_mm=120.0),
        calibration_params=None,
        cv2_module=_FakeCv2,
        yolo_class=_FakeYOLO,
    )
    worker.set_robot_z(185.0)

    stage, z_mm = worker._current_calibration_stage()  # pylint: disable=protected-access

    assert stage == "confirm"
    assert z_mm == 185.0
