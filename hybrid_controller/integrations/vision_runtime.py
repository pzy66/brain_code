from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal

from hybrid_controller.adapters.vision_adapter import VisionTarget
from hybrid_controller.config import AppConfig


class _VisionWorker(QObject):
    targets_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self._running = False
        self._capture = None
        self._model = None
        self._timer: Optional[QTimer] = None

    def start(self) -> None:
        try:
            import cv2
            from ultralytics import YOLO
        except Exception as error:
            self.error_occurred.emit(f"Vision runtime dependencies missing: {error}")
            self.finished.emit()
            return

        stream_url = self.config.resolve_vision_stream_url()
        source = int(stream_url) if stream_url.isdigit() else stream_url
        self._model = YOLO(str(self.config.vision_weights_path))
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            self.error_occurred.emit(f"Failed to open vision source: {stream_url}")
            self.finished.emit()
            return

        self._running = True
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._process_frame)
        self._timer.start(self.config.vision_infer_interval_ms)
        self.status_changed.emit(f"Vision runtime started with source {stream_url}")

    def stop(self) -> None:
        self._running = False
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        self.finished.emit()

    def _process_frame(self) -> None:
        if not self._running or self._capture is None or self._model is None:
            return
        ok, frame = self._capture.read()
        if not ok:
            self.error_occurred.emit("Vision frame read failed.")
            return
        try:
            results = self._model(frame, verbose=False)
            targets: list[VisionTarget] = []
            result0 = results[0]
            if result0.boxes is not None:
                boxes = result0.boxes.xyxy.cpu().numpy().tolist()
                confidences = result0.boxes.conf.cpu().numpy().tolist()
                for index, (box, confidence) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    targets.append(
                        VisionTarget(
                            id=index,
                            bbox=(x1, y1, x2, y2),
                            center_px=center,
                            raw_center=center,
                            confidence=float(confidence),
                        )
                    )
            self.targets_ready.emit(targets)
        except Exception as error:
            self.error_occurred.emit(f"Vision inference error: {error}")


class VisionRuntime:
    def __init__(
        self,
        config: AppConfig,
        targets_callback: Callable[[list[VisionTarget]], None],
        status_callback: Callable[[str], None],
    ) -> None:
        self.config = config
        self.targets_callback = targets_callback
        self.status_callback = status_callback
        self.thread: Optional[QThread] = None
        self.worker: Optional[_VisionWorker] = None

    def start(self) -> None:
        if self.worker is not None:
            return
        self.thread = QThread()
        self.worker = _VisionWorker(self.config)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.worker.targets_ready.connect(self.targets_callback)
        self.worker.status_changed.connect(self.status_callback)
        self.worker.error_occurred.connect(self.status_callback)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(2000)
            self.thread = None
        self.worker = None

    def healthcheck(self) -> dict[str, object]:
        return {
            "running": self.worker is not None,
            "weights": str(self.config.vision_weights_path),
            "source": self.config.resolve_vision_stream_url(),
        }
