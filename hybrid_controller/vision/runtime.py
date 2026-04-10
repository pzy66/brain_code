from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import numpy as np
from PyQt5.QtCore import QMetaObject, QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot

from hybrid_controller.adapters.vision_adapter import VisionTarget
from hybrid_controller.config import AppConfig
from hybrid_controller.vision.processing import (
    SlotState,
    VisionCalibration,
    annotate_slots_with_cylindrical,
    build_vision_packet,
    extract_candidates,
    packet_to_targets,
    update_slots,
)


def _resolve_weights_path(config: AppConfig) -> str:
    candidate = config.vision_weights_path
    if candidate.exists():
        return str(candidate)
    search_roots = (
        Path.cwd(),
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[3],
    )
    for root in search_roots:
        alternate = (root / candidate).resolve()
        if alternate.exists():
            return str(alternate)
    return str((Path(__file__).resolve().parents[2] / candidate).resolve())


def _load_vision_dependencies() -> tuple[object, object]:
    import cv2
    from ultralytics import YOLO

    return cv2, YOLO


def _resolve_vision_device(request: str) -> tuple[str | None, bool]:
    normalized = str(request or "auto").strip().lower()
    if normalized in {"", "auto"}:
        try:
            import torch

            if torch.cuda.is_available():
                return "0", True
        except Exception:
            pass
        return "cpu", False
    if normalized == "cpu":
        return "cpu", False
    return str(request).strip(), False


class _VisionWorker(QObject):
    targets_ready = pyqtSignal(object)
    packet_ready = pyqtSignal(object)
    frame_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        config: AppConfig,
        *,
        calibration_params: dict[str, object] | None,
        cv2_module: object,
        yolo_class: object,
    ) -> None:
        super().__init__()
        self.config = config
        self._cv2 = cv2_module
        self._yolo_class = yolo_class
        self._calibration: VisionCalibration | None = None
        self._pending_status: str | None = None
        if calibration_params:
            try:
                self._calibration = VisionCalibration.from_param_dict(calibration_params)
            except Exception as error:
                self._calibration = None
                self._pending_status = f"Vision calibration unavailable: {error}"
        self._running = False
        self._capture = None
        self._model = None
        self._timer: Optional[QTimer] = None
        self._infer_interval_ms = max(10, int(self.config.vision_infer_interval_ms))
        self._infer_interval_dynamic_ms = float(self._infer_interval_ms)
        self._infer_interval_min_ms = max(10.0, float(self.config.vision_infer_interval_min_ms))
        self._infer_interval_max_ms = max(self._infer_interval_min_ms, float(self.config.vision_infer_interval_max_ms))
        self._infer_hysteresis_ms = max(0.0, float(self.config.vision_infer_hysteresis_ms))
        self._adaptive_infer_enabled = bool(self.config.vision_adaptive_infer_enabled)
        self._infer_max_step_up_ms = max(1.0, float(self.config.vision_infer_max_step_up_ms))
        self._infer_max_step_down_ms = max(1.0, float(self.config.vision_infer_max_step_down_ms))
        self._display_interval_ms = max(16, int(self.config.ui_refresh_interval_ms))
        self._frame_emit_interval_sec = 1.0 / max(1.0, 1000.0 / float(self._display_interval_ms))
        self._last_frame_emit_ts = 0.0
        self._active_stream_url: str | None = None
        self._stream_candidates = tuple(str(item) for item in self.config.resolve_vision_stream_candidates())
        self._candidate_cursor = 0
        self._last_connect_attempt_ts = 0.0
        self._connect_interval_sec = max(0.2, float(self.config.vision_reconnect_interval_ms) / 1000.0)
        self._frame_id = 0
        self._stop_requested = False
        self._capture_counter = 0
        self._capture_window_start = time.perf_counter()
        self._capture_fps = 0.0
        self._capture_total_frames = 0
        self._last_capture_ts = 0.0
        self._capture_thread: threading.Thread | None = None
        self._capture_stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_seq = 0
        self._last_infer_frame_seq = 0
        self._infer_total_frames = 0
        self._dropped_total_frames = 0
        self._capture_lost = False
        self._predict_device, auto_half = _resolve_vision_device(str(self.config.vision_device))
        self._predict_half = bool(self.config.vision_half or auto_half)
        self._slots = [SlotState(slot=index + 1, freq_hz=config.ssvep_freqs[index]) for index in range(config.vision_max_targets)]

    @pyqtSlot()
    def start(self) -> None:
        if self._running or self._stop_requested:
            return

        weights_path = _resolve_weights_path(self.config)
        self._model = self._yolo_class(weights_path)
        self._warmup_model()
        self._running = True
        self._infer_interval_dynamic_ms = float(self._infer_interval_ms)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._process_frame)
        self._timer.start(int(round(self._infer_interval_dynamic_ms)))
        self.status_changed.emit("Vision runtime started.")
        if self._pending_status is not None:
            self.status_changed.emit(self._pending_status)
        self._ensure_capture(force=True)
        if self._capture is not None and self._active_stream_url is not None:
            self.status_changed.emit(f"Vision runtime started with source {self._active_stream_url}")
        else:
            checked = ", ".join(self._stream_candidates)
            self.status_changed.emit(f"Vision runtime waiting for stream... checked=[{checked}]")

    @pyqtSlot()
    def stop(self) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        self._running = False
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        self._stop_capture_pump()
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        with self._frame_lock:
            self._latest_frame = None
            self._latest_frame_seq = 0
            self._last_infer_frame_seq = 0
            self._capture_lost = False
        self._model = None
        self._active_stream_url = None
        self.finished.emit()

    def _release_capture(self) -> None:
        self._stop_capture_pump()
        if self._capture is None:
            return
        try:
            self._capture.release()
        except Exception:
            pass
        self._capture = None
        self._active_stream_url = None
        with self._frame_lock:
            self._capture_lost = False

    def _ensure_capture(self, *, force: bool) -> bool:
        if self._capture is not None:
            return True
        now = time.perf_counter()
        if not force and (now - self._last_connect_attempt_ts) < self._connect_interval_sec:
            return False
        self._last_connect_attempt_ts = now
        candidate_count = len(self._stream_candidates)
        if candidate_count == 0:
            return False

        for offset in range(candidate_count):
            index = (self._candidate_cursor + offset) % candidate_count
            stream_url = self._stream_candidates[index]
            capture = self._try_open_capture(stream_url)
            if capture is None:
                continue
            self._capture = capture
            self._active_stream_url = stream_url
            self._candidate_cursor = (index + 1) % candidate_count
            with self._frame_lock:
                self._capture_lost = False
            self._start_capture_pump(capture)
            self.status_changed.emit(f"Vision stream connected: {stream_url}")
            return True

        checked = ", ".join(self._stream_candidates)
        self.status_changed.emit(f"Vision stream unavailable, retrying... checked=[{checked}]")
        return False

    def _endpoint_reachable(self, stream_url: str) -> bool:
        value = str(stream_url).strip()
        if value.isdigit():
            return True
        try:
            parsed = urlparse(value)
        except Exception:
            return True
        if parsed.scheme not in {"http", "https", "rtsp", "tcp"}:
            return True
        if not parsed.hostname:
            return True
        if parsed.port is not None:
            port = int(parsed.port)
        elif parsed.scheme in {"https"}:
            port = 443
        elif parsed.scheme in {"rtsp"}:
            port = 554
        else:
            port = 80
        timeout_sec = max(0.05, float(self.config.vision_endpoint_probe_timeout_ms) / 1000.0)
        try:
            with socket.create_connection((str(parsed.hostname), int(port)), timeout=timeout_sec):
                return True
        except OSError:
            return False

    def _try_open_capture(self, stream_url: str):
        if not self._endpoint_reachable(stream_url):
            return None
        source = int(stream_url) if stream_url.isdigit() else stream_url
        backend = getattr(self._cv2, "CAP_ANY", 0)
        if isinstance(source, str) and hasattr(self._cv2, "CAP_FFMPEG"):
            backend = getattr(self._cv2, "CAP_FFMPEG")
        try:
            capture = self._cv2.VideoCapture(source, backend)
        except TypeError:
            capture = self._cv2.VideoCapture(source)
        if hasattr(self._cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(self._cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(self._cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(self.config.vision_open_timeout_ms))
        if hasattr(self._cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(self._cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(self.config.vision_read_timeout_ms))
        if not capture.isOpened():
            try:
                capture.release()
            except Exception:
                pass
            return None

        probe_reads = max(1, int(self.config.vision_probe_reads))
        probe_sleep = max(0.0, float(self.config.vision_probe_sleep_ms) / 1000.0)
        for _ in range(probe_reads):
            ok, frame = capture.read()
            if ok and frame is not None:
                return capture
            if probe_sleep > 0.0:
                time.sleep(probe_sleep)
        try:
            capture.release()
        except Exception:
            pass
        return None

    def _start_capture_pump(self, capture) -> None:
        self._stop_capture_pump()
        self._capture_stop_event.clear()
        thread = threading.Thread(
            target=self._capture_loop,
            args=(capture,),
            name="hybrid-vision-capture",
            daemon=True,
        )
        self._capture_thread = thread
        thread.start()

    def _stop_capture_pump(self) -> None:
        self._capture_stop_event.set()
        thread = self._capture_thread
        self._capture_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._capture_stop_event.clear()

    def _capture_loop(self, capture) -> None:
        while self._running and not self._capture_stop_event.is_set():
            try:
                ok, frame = capture.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                with self._frame_lock:
                    self._capture_lost = True
                return

            now = time.perf_counter()
            self._capture_counter += 1
            self._capture_total_frames += 1
            elapsed = now - self._capture_window_start
            if elapsed >= 1.0:
                current_fps = self._capture_counter / elapsed
                self._capture_fps = (
                    current_fps if self._capture_fps <= 0 else (self._capture_fps * 0.8 + current_fps * 0.2)
                )
                self._capture_counter = 0
                self._capture_window_start = now

            with self._frame_lock:
                self._latest_frame = frame
                self._latest_frame_seq += 1
                self._last_capture_ts = now
            self._emit_frame_from_capture(frame, now)

    def _latest_frame_snapshot(self) -> tuple[object | None, int, float]:
        with self._frame_lock:
            return self._latest_frame, int(self._latest_frame_seq), float(self._last_capture_ts)

    def _schedule_next_tick(self) -> None:
        timer = self._timer
        if self._running and timer is not None:
            timer.start(int(round(self._infer_interval_dynamic_ms)))

    def _adjust_infer_interval(self, *, infer_ms: float, queue_age_ms: float) -> None:
        if not self._adaptive_infer_enabled:
            self._infer_interval_dynamic_ms = float(self._infer_interval_ms)
            return
        target_queue_age_ms = max(1.0, float(self.config.vision_infer_target_queue_age_ms))
        alpha = min(1.0, max(0.05, float(self.config.vision_infer_adjust_alpha)))
        current = float(self._infer_interval_dynamic_ms)
        queue_error = float(queue_age_ms) - target_queue_age_ms

        floor_from_infer = max(self._infer_interval_min_ms, float(infer_ms) * 0.75)
        hysteresis = float(self._infer_hysteresis_ms)
        desired = current
        if abs(queue_error) <= hysteresis:
            desired = max(floor_from_infer, current)
        elif queue_error > 0.0:
            step_up = min(self._infer_max_step_up_ms, queue_error * 0.22 + infer_ms * 0.06)
            desired = min(self._infer_interval_max_ms, current + step_up)
        else:
            step_down = min(self._infer_max_step_down_ms, abs(queue_error) * 0.10 + infer_ms * 0.03)
            desired = max(floor_from_infer, current - step_down)

        smoothed = (1.0 - alpha) * current + alpha * desired
        self._infer_interval_dynamic_ms = max(
            self._infer_interval_min_ms,
            min(self._infer_interval_max_ms, smoothed),
        )

    def _emit_frame_from_capture(self, frame, capture_ts: float) -> None:
        if not self._running:
            return
        if (capture_ts - self._last_frame_emit_ts) < self._frame_emit_interval_sec:
            return
        self._last_frame_emit_ts = capture_ts
        try:
            self.frame_ready.emit(frame.copy())
        except Exception:
            return

    def _predict_frame(self, frame):
        model = self._model
        if model is None:
            return []
        if hasattr(model, "predict"):
            predict_kwargs: dict[str, object] = {
                "source": frame,
                "imgsz": int(self.config.vision_model_imgsz),
                "conf": float(self.config.vision_confidence_threshold),
                "iou": float(self.config.vision_iou_threshold),
                "max_det": int(self.config.vision_max_det),
                "verbose": False,
            }
            if self._predict_device:
                predict_kwargs["device"] = self._predict_device
            if self._predict_half:
                predict_kwargs["half"] = True
            return model.predict(**predict_kwargs)
        return model(frame, verbose=False)

    def _warmup_model(self) -> None:
        warmup_runs = max(0, int(self.config.vision_warmup_runs))
        if warmup_runs <= 0:
            return
        dummy_size = max(128, int(self.config.vision_model_imgsz))
        dummy = np.zeros((dummy_size, dummy_size, 3), dtype=np.uint8)
        for _ in range(warmup_runs):
            try:
                self._predict_frame(dummy)
            except Exception:
                return

    def _process_frame(self) -> None:
        if not self._running or self._model is None:
            return
        try:
            if self._capture is None:
                self._ensure_capture(force=False)
                return

            with self._frame_lock:
                capture_lost = bool(self._capture_lost)
            if capture_lost:
                self._release_capture()
                self.status_changed.emit("Vision stream lost, reconnecting...")
                self._ensure_capture(force=True)
                return

            frame, frame_seq, capture_ts = self._latest_frame_snapshot()
            if frame is None:
                return
            previous_infer_seq = int(self._last_infer_frame_seq)
            if frame_seq == previous_infer_seq:
                return
            if previous_infer_seq > 0 and frame_seq > previous_infer_seq:
                dropped = max(0, int(frame_seq - previous_infer_seq - 1))
                self._dropped_total_frames += dropped
            self._infer_total_frames += 1
            self._last_infer_frame_seq = frame_seq

            self._frame_id += 1
            frame_h, frame_w = frame.shape[:2]
            roi_center = self._resolve_roi_center(frame_w, frame_h)
            roi_radius = self._resolve_roi_radius(frame_w, frame_h)

            infer_start = time.perf_counter()
            try:
                results = self._predict_frame(frame)
            except Exception as error:
                self.error_occurred.emit(f"Vision inference error: {error}")
                return
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            if not results:
                return
            result0 = results[0]
            candidates, detected_count = extract_candidates(
                result0,
                frame_shape=(frame_h, frame_w),
                roi_center=roi_center,
                roi_radius=roi_radius,
                max_det=self.config.vision_max_targets,
                confidence_threshold=self.config.vision_confidence_threshold,
            )
            update_slots(
                self._slots,
                candidates,
                match_distance=120.0,
                lost_ttl=6,
            )
            annotate_slots_with_cylindrical(
                self._slots,
                calibration=self._calibration,
                world_scale_xy=float(self.config.vision_world_scale_xy),
                world_offset_xy_mm=(
                    float(self.config.vision_world_offset_xy_mm[0]),
                    float(self.config.vision_world_offset_xy_mm[1]),
                ),
                mapping_mode=str(self.config.vision_mapping_mode),
            )
            packet = build_vision_packet(
                frame_id=self._frame_id,
                frame_size=(frame_w, frame_h),
                roi_center=roi_center,
                roi_radius=roi_radius,
                slots=self._slots,
                capture_fps=self._capture_fps,
                infer_ms=infer_ms,
                queue_age_ms=max(0.0, (time.perf_counter() - capture_ts) * 1000.0),
                detected_count=detected_count,
                calibration_ready=self._calibration is not None,
                mapping_mode=str(self.config.vision_mapping_mode),
            )
            packet["infer_interval_ms"] = float(self._infer_interval_dynamic_ms)
            total_infer_frames = max(1, int(self._infer_total_frames))
            packet["frame_drop_ratio"] = float(self._dropped_total_frames) / float(
                self._dropped_total_frames + total_infer_frames
            )
            targets = packet_to_targets(packet)
            self.packet_ready.emit(packet)
            self.targets_ready.emit(targets)
            self._adjust_infer_interval(
                infer_ms=float(infer_ms),
                queue_age_ms=float(packet.get("queue_age_ms", 0.0)),
            )
        finally:
            self._schedule_next_tick()

    def _resolve_roi_center(self, frame_w: int, frame_h: int) -> tuple[int, int]:
        x = int(round(float(self.config.roi_center[0])))
        y = int(round(float(self.config.roi_center[1])))
        if 0 <= x < frame_w and 0 <= y < frame_h:
            return (x, y)
        return (frame_w // 2, frame_h // 2)

    def _resolve_roi_radius(self, frame_w: int, frame_h: int) -> int:
        radius = int(round(float(self.config.roi_radius)))
        if radius > 0:
            return radius
        return max(40, int(round(min(frame_w, frame_h) * 0.28)))


class VisionRuntime:
    def __init__(
        self,
        config: AppConfig,
        *,
        calibration_params: dict[str, object] | None,
        targets_callback: Callable[[list[VisionTarget]], None],
        packet_callback: Callable[[dict[str, object]], None],
        frame_callback: Callable[[np.ndarray], None],
        status_callback: Callable[[str], None],
    ) -> None:
        self.config = config
        self.calibration_params = calibration_params
        self.targets_callback = targets_callback
        self.packet_callback = packet_callback
        self.frame_callback = frame_callback
        self.status_callback = status_callback
        self.thread: Optional[QThread] = None
        self.worker: Optional[_VisionWorker] = None
        self._last_packet: dict[str, object] | None = None

    def start(self) -> None:
        if self.worker is not None:
            return
        try:
            cv2_module, yolo_class = _load_vision_dependencies()
        except Exception as error:
            self.status_callback(f"Vision runtime dependencies missing: {error}")
            return
        self.thread = QThread()
        self.worker = _VisionWorker(
            self.config,
            calibration_params=self.calibration_params,
            cv2_module=cv2_module,
            yolo_class=yolo_class,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.worker.targets_ready.connect(self.targets_callback)
        self.worker.packet_ready.connect(self._handle_packet_ready)
        self.worker.packet_ready.connect(self.packet_callback)
        self.worker.frame_ready.connect(self.frame_callback)
        self.worker.status_changed.connect(self.status_callback)
        self.worker.error_occurred.connect(self.status_callback)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

    def stop(self) -> None:
        worker = self.worker
        thread = self.thread
        self.worker = None
        self.thread = None
        if worker is not None:
            try:
                if thread is not None and thread.isRunning():
                    QMetaObject.invokeMethod(worker, "stop", Qt.BlockingQueuedConnection)
                else:
                    worker.stop()
            except RuntimeError:
                pass
        if thread is not None:
            thread.quit()
            thread.wait(2000)
            thread.deleteLater()

    def healthcheck(self) -> dict[str, object]:
        return {
            "running": self.worker is not None,
            "weights": _resolve_weights_path(self.config),
            "source": self.config.resolve_vision_stream_url(),
            "source_candidates": self.config.resolve_vision_stream_candidates(),
            "last_packet": self._last_packet,
            "calibration_ready": self.calibration_params is not None,
        }

    def _handle_packet_ready(self, packet: dict[str, object]) -> None:
        self._last_packet = dict(packet)
