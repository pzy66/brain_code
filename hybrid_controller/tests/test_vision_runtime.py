from __future__ import annotations

import time

from PyQt5.QtCore import QCoreApplication

from hybrid_controller.config import AppConfig
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
