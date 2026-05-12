from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import cv2
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
    IMREAD_COLOR = 1

    @staticmethod
    def VideoCapture(_source):
        return _FakeCapture()

    @staticmethod
    def imdecode(buffer, _flags):
        data = bytes(buffer)
        if data == b"\xff\xd8FRAME_A\xff\xd9":
            return np.zeros((48, 64, 3), dtype=np.uint8)
        if data == b"\xff\xd8FRAME_B\xff\xd9":
            return np.full((48, 64, 3), 80, dtype=np.uint8)
        if data == b"\xff\xd8TORN\xff\xd9":
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[210:285] = 255
            return frame
        return None


class _ChunkedResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.closed = False

    def read(self, _size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def close(self) -> None:
        self.closed = True


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


def test_horizontal_tearing_detector_rejects_spliced_mjpeg_frame() -> None:
    clean = cv2.imread(
        str(Path("hybrid_controller/logs/vision_debug/live_read_check/frame_03.jpg"))
    )
    torn = cv2.imread(
        str(Path("hybrid_controller/logs/vision_debug/vision_grasp_20260511_152143/continuous_028/raw.jpg"))
    )

    assert vision_runtime._frame_has_horizontal_tearing(clean) is False  # pylint: disable=protected-access
    assert vision_runtime._frame_has_horizontal_tearing(torn) is True  # pylint: disable=protected-access


def test_horizontal_tearing_detector_rejects_multi_band_low_height_frame() -> None:
    torn = cv2.imread(
        str(Path("hybrid_controller/logs/vision_debug/vision_grasp_20260511_223228/step_01/raw.jpg"))
    )
    clean = cv2.imread(
        str(Path("hybrid_controller/logs/vision_debug/vision_grasp_20260511_222734/step_01/raw.jpg"))
    )

    assert vision_runtime._frame_has_horizontal_tearing(clean) is False  # pylint: disable=protected-access
    assert vision_runtime._frame_has_horizontal_tearing(torn) is True  # pylint: disable=protected-access


def test_temporal_splice_detector_rejects_partial_frame_jump() -> None:
    previous = cv2.imread(
        str(Path("hybrid_controller/logs/vision_debug/vision_grasp_20260511_183500/continuous_096/raw.jpg"))
    )
    torn = cv2.imread(
        str(Path("hybrid_controller/logs/vision_debug/vision_grasp_20260511_183500/continuous_097/raw.jpg"))
    )
    clean = cv2.imread(
        str(Path("hybrid_controller/logs/vision_debug/vision_grasp_20260511_183500/continuous_093/raw.jpg"))
    )

    assert vision_runtime._frame_is_temporal_splice(previous, torn) is True  # pylint: disable=protected-access
    assert vision_runtime._frame_is_temporal_splice(clean, previous) is False  # pylint: disable=protected-access


def test_http_mjpeg_capture_prefers_multipart_content_length(monkeypatch) -> None:
    payload_a = b"\xff\xd8FRAME_A\xff\xd9"
    payload_b = b"\xff\xd8FRAME_B\xff\xd9"
    body = (
        b"--boundarydonotcross\r\n"
        b"Content-type: image/jpeg\r\n"
        b"X-Timestamp: 1.0\r\n"
        b"Content-Length: "
        + str(len(payload_a)).encode("ascii")
        + b"\r\n\r\n"
        + payload_a
        + b"\r\n--boundarydonotcross\r\n"
        b"Content-type: image/jpeg\r\n"
        b"Content-Length: "
        + str(len(payload_b)).encode("ascii")
        + b"\r\n\r\n"
        + payload_b
    )
    response = _ChunkedResponse([body[:17], body[17:53], body[53:]])
    monkeypatch.setattr(vision_runtime, "urlopen", lambda *_args, **_kwargs: response)

    capture = vision_runtime._HttpMjpegCapture(  # pylint: disable=protected-access
        "http://camera:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg",
        cv2_module=_FakeCv2,
        timeout_sec=0.5,
    )

    ok_a, frame_a = capture.read()
    ok_b, frame_b = capture.read()
    stats = capture.stats()

    assert ok_a is True
    assert ok_b is True
    assert frame_a.shape == (48, 64, 3)
    assert frame_b.shape == (48, 64, 3)
    assert int(frame_a[0, 0, 0]) == 0
    assert int(frame_b[0, 0, 0]) == 80
    assert stats["content_length_payloads"] == 2
    assert stats["jpeg_marker_payloads"] == 0
    assert stats["frames_accepted"] == 2
    assert stats["content_length_preferred"] is True


def test_http_mjpeg_capture_waits_for_complete_content_length(monkeypatch) -> None:
    payload = b"\xff\xd8FRAME_A\xff\xd9"
    header = (
        b"--boundarydonotcross\r\n"
        b"Content-type: image/jpeg\r\n"
        b"Content-Length: "
        + str(len(payload)).encode("ascii")
        + b"\r\n\r\n"
    )
    response = _ChunkedResponse([header + payload[:3], payload[3:]])
    monkeypatch.setattr(vision_runtime, "urlopen", lambda *_args, **_kwargs: response)

    capture = vision_runtime._HttpMjpegCapture(  # pylint: disable=protected-access
        "http://camera:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg",
        cv2_module=_FakeCv2,
        timeout_sec=0.5,
    )

    ok, frame = capture.read()

    assert ok is True
    assert frame.shape == (48, 64, 3)


def test_http_mjpeg_capture_skips_rejected_content_length_frame(monkeypatch) -> None:
    payload_bad = b"\xff\xd8TORN\xff\xd9"
    payload_good = b"\xff\xd8FRAME_B\xff\xd9"
    body = (
        b"--boundarydonotcross\r\nContent-Length: "
        + str(len(payload_bad)).encode("ascii")
        + b"\r\n\r\n"
        + payload_bad
        + b"\r\n--boundarydonotcross\r\nContent-Length: "
        + str(len(payload_good)).encode("ascii")
        + b"\r\n\r\n"
        + payload_good
    )
    response = _ChunkedResponse([body])
    monkeypatch.setattr(vision_runtime, "urlopen", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(vision_runtime, "_frame_has_horizontal_tearing", lambda frame: int(frame[0, 0, 0]) == 0)

    capture = vision_runtime._HttpMjpegCapture(  # pylint: disable=protected-access
        "http://camera:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg",
        cv2_module=_FakeCv2,
        timeout_sec=0.5,
    )

    ok, frame = capture.read()
    stats = capture.stats()

    assert ok is True
    assert int(frame[0, 0, 0]) == 80
    assert stats["frames_rejected"] == 1
    assert stats["last_reject_reason"] == "horizontal_tearing:content_length"


def test_http_mjpeg_capture_reopens_after_repeated_rejected_frames(monkeypatch) -> None:
    payload_bad = b"\xff\xd8TORN\xff\xd9"
    payload_good = b"\xff\xd8FRAME_B\xff\xd9"

    def _body(payloads):
        body = bytearray()
        for payload in payloads:
            body.extend(b"--boundarydonotcross\r\nContent-Length: ")
            body.extend(str(len(payload)).encode("ascii"))
            body.extend(b"\r\n\r\n")
            body.extend(payload)
        return bytes(body)

    responses = [
        _ChunkedResponse([_body([payload_bad, payload_bad, payload_bad])]),
        _ChunkedResponse([_body([payload_good])]),
    ]
    opened = []

    def fake_urlopen(*_args, **_kwargs):
        opened.append(True)
        return responses.pop(0)

    monkeypatch.setattr(vision_runtime, "urlopen", fake_urlopen)
    monkeypatch.setattr(vision_runtime, "_frame_has_horizontal_tearing", lambda frame: int(frame[0, 0, 0]) == 0)

    capture = vision_runtime._HttpMjpegCapture(  # pylint: disable=protected-access
        "http://camera:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg",
        cv2_module=_FakeCv2,
        timeout_sec=0.5,
    )

    ok, frame = capture.read()
    stats = capture.stats()

    assert ok is True
    assert int(frame[0, 0, 0]) == 80
    assert len(opened) == 2
    assert stats["buffer_resets"] == 1
    assert stats["reopen_count"] == 1


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
