from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qsl, quote, urlparse, urlunparse
from urllib.request import Request, urlopen

import numpy as np
from PyQt5.QtCore import Q_ARG, QMetaObject, QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot

from hybrid_controller.adapters.vision_adapter import VisionTarget
from hybrid_controller.config import AppConfig
from hybrid_controller.vision.calibration_profile import VisionCalibrationProfile
from hybrid_controller.vision.processing import (
    SlotState,
    VisionCalibration,
    annotate_slots_with_cylindrical,
    build_vision_packet,
    extract_candidates,
    frame_brightness_quality,
    packet_to_targets,
    update_slots,
)


def _normalize_web_video_url(source: str) -> str:
    value = str(source).strip()
    try:
        parsed = urlparse(value)
    except Exception:
        return value
    if parsed.scheme not in {"http", "https"} or not parsed.path.endswith("/stream") or not parsed.query:
        return value
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    if not query_items:
        return value
    normalized_query = "&".join(
        f"{quote(str(key), safe='')}={quote(str(val), safe='/')}" for key, val in query_items
    )
    return urlunparse(parsed._replace(query=normalized_query))


def _frame_has_horizontal_tearing(frame: object) -> bool:
    """Detect obvious cross-frame horizontal slice artifacts in MJPEG images."""
    if frame is None:
        return False
    try:
        arr = np.asarray(frame)
    except Exception:
        return False
    if arr.ndim < 2:
        return False
    height, width = arr.shape[:2]
    if height < 40 or width < 40:
        return False
    if arr.ndim == 2:
        sample = arr[:, :, None]
    else:
        sample = arr[:, :, : min(3, arr.shape[2])]
    diffs = np.mean(np.abs(np.diff(sample.astype(np.int16), axis=0)), axis=(1, 2))
    if diffs.size < 20:
        return False
    row_edge_fraction = np.mean(np.mean(np.abs(np.diff(sample.astype(np.int16), axis=0)), axis=2) > 18.0, axis=1)
    edge_rows = np.flatnonzero(row_edge_fraction > 0.20)
    if edge_rows.size >= 6 and float(np.percentile(row_edge_fraction, 95)) >= 0.18:
        clusters_for_edges: list[list[int]] = [[int(edge_rows[0])]]
        for row_index in edge_rows[1:]:
            row_index = int(row_index)
            if row_index - clusters_for_edges[-1][-1] <= 4:
                clusters_for_edges[-1].append(row_index)
            else:
                clusters_for_edges.append([row_index])
        strong_edge_clusters = [
            cluster
            for cluster in clusters_for_edges
            if len(cluster) >= 2 and max(float(row_edge_fraction[int(row)]) for row in cluster) >= 0.24
        ]
        central_top = int(height * 0.25)
        central_bottom = int(height * 0.75)
        central_clusters = [
            cluster
            for cluster in strong_edge_clusters
            if central_top <= (cluster[0] + cluster[-1]) // 2 <= central_bottom
        ]
        if len(strong_edge_clusters) >= 4 and len(central_clusters) >= 3:
            return True
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median))) + 1e-6
    p95 = float(np.percentile(diffs, 95))
    threshold = max(12.0, median + 20.0 * mad, p95 * 4.0)
    strong_rows = np.flatnonzero(diffs > threshold)
    if strong_rows.size < 2:
        return False
    clusters: list[list[int]] = [[int(strong_rows[0])]]
    for row_index in strong_rows[1:]:
        row_index = int(row_index)
        if row_index - clusters[-1][-1] <= 4:
            clusters[-1].append(row_index)
        else:
            clusters.append([row_index])
    if len(clusters) < 2:
        return False
    row_diffs = np.abs(np.diff(sample.astype(np.int16), axis=0))
    strong_clusters = 0
    for cluster in clusters:
        cluster_rows = [int(row) for row in cluster]
        cluster_score = 0.0
        for row_index in cluster_rows:
            per_column = np.mean(row_diffs[int(row_index)], axis=1)
            cluster_score = max(cluster_score, float(np.mean(per_column > 25.0)))
        if cluster_score >= 0.18:
            strong_clusters += 1
    if strong_clusters >= 2:
        return True
    return False


def _frame_is_temporal_splice(previous_frame: object, frame: object) -> bool:
    """Reject MJPEG frames whose regions are visibly assembled from different moments.

    The official JetMax stream sometimes delivers a frame that decodes as a valid
    JPEG but contains a large partial-frame jump during motion.  A single-frame
    check misses those cases because the JPEG itself is syntactically valid, so
    this compares coarse image structure against the last accepted frame.
    """
    if previous_frame is None or frame is None:
        return False
    try:
        previous = np.asarray(previous_frame)
        current = np.asarray(frame)
    except Exception:
        return False
    if previous.shape != current.shape or current.ndim < 2:
        return False
    height, width = current.shape[:2]
    if height < 80 or width < 80:
        return False
    previous_small = previous[::8, ::8, :3] if previous.ndim >= 3 else previous[::8, ::8]
    current_small = current[::8, ::8, :3] if current.ndim >= 3 else current[::8, ::8]
    diff = np.mean(np.abs(current_small.astype(np.int16) - previous_small.astype(np.int16)), axis=-1)
    changed = diff > 28.0
    changed_ratio = float(np.mean(changed))
    if changed_ratio < 0.18:
        return False
    row_ratio = np.mean(changed, axis=1)
    col_ratio = np.mean(changed, axis=0)
    active_rows = np.flatnonzero(row_ratio > 0.22)
    active_cols = np.flatnonzero(col_ratio > 0.22)
    row_span = 0.0 if active_rows.size == 0 else float(active_rows[-1] - active_rows[0] + 1) / float(changed.shape[0])
    col_span = 0.0 if active_cols.size == 0 else float(active_cols[-1] - active_cols[0] + 1) / float(changed.shape[1])
    # Normal robot motion changes the scene broadly and smoothly.  Spliced MJPEG
    # frames tend to replace a large slice while leaving another large slice from
    # the previous moment, which yields a broad but non-global change mask.
    return 0.18 <= changed_ratio <= 0.58 and (row_span >= 0.35 or col_span >= 0.35)


def _multipart_content_length(header_block: bytes) -> int | None:
    for raw_line in header_block.splitlines():
        if b":" not in raw_line:
            continue
        key, raw_value = raw_line.split(b":", 1)
        if key.strip().lower() != b"content-length":
            continue
        try:
            value = int(raw_value.strip())
        except ValueError:
            return None
        return value if 0 < value <= 4_000_000 else None
    return None


class _HttpMjpegCapture:
    """Small MJPEG reader for the locked JetMax web_video_server URL.

    This reader consumes bytes from the official desktop URL only. It must not be
    expanded into endpoint discovery, ROS topic scanning, or robot-side camera
    service management.
    """

    def __init__(
        self,
        url: str,
        *,
        cv2_module: object,
        timeout_sec: float,
        read_timeout_sec: float | None = None,
    ) -> None:
        self._url = _normalize_web_video_url(str(url))
        self._cv2 = cv2_module
        self._open_timeout_sec = max(0.2, float(timeout_sec))
        self._read_timeout_sec = max(
            0.1,
            float(self._open_timeout_sec if read_timeout_sec is None else read_timeout_sec),
        )
        self._response = None
        self._buffer = bytearray()
        self._last_frame = None
        self._consecutive_rejected_frames = 0
        self._stats: dict[str, object] = {
            "open_count": 0,
            "bytes_read": 0,
            "content_length_payloads": 0,
            "jpeg_marker_payloads": 0,
            "frames_accepted": 0,
            "frames_rejected": 0,
            "read_timeouts": 0,
            "read_errors": 0,
            "buffer_resets": 0,
            "reopen_count": 0,
            "last_reject_reason": "",
            "last_read_error": "",
        }
        self._open()

    def _open(self) -> None:
        request = Request(self._url, headers={"User-Agent": "hybrid-controller/vision"})
        self._response = urlopen(request, timeout=self._open_timeout_sec)
        self._set_response_read_timeout()
        self._stats["open_count"] = int(self._stats.get("open_count", 0)) + 1

    def _set_response_read_timeout(self) -> None:
        response = self._response
        if response is None:
            return
        for attr_path in (("fp", "raw", "_sock"), ("fp", "raw", "_fp", "fp", "raw", "_sock")):
            target = response
            try:
                for attr in attr_path:
                    target = getattr(target, attr)
                target.settimeout(self._read_timeout_sec)
                return
            except Exception:
                continue

    def isOpened(self) -> bool:
        return self._response is not None

    def stats(self) -> dict[str, object]:
        """Return transport counters for debug reports without touching the sender."""
        result = dict(self._stats)
        result["buffer_bytes"] = int(len(self._buffer))
        result["consecutive_rejected_frames"] = int(self._consecutive_rejected_frames)
        result["url"] = self._url
        result["reader"] = "http_multipart_mjpeg"
        result["content_length_preferred"] = True
        return result

    def read(self):
        if self._response is None:
            return False, None
        deadline = time.perf_counter() + self._read_timeout_sec
        while time.perf_counter() < deadline:
            while True:
                read_status, frame = self._read_buffered_frame()
                if read_status == "ready" and frame is not None:
                    return True, frame
                if read_status == "need_data":
                    break
            try:
                chunk = self._read_response_chunk()
            except (TimeoutError, socket.timeout):
                self._stats["read_timeouts"] = int(self._stats.get("read_timeouts", 0)) + 1
                self._stats["last_read_error"] = "timeout"
                while True:
                    read_status, frame = self._read_buffered_frame()
                    if read_status == "ready" and frame is not None:
                        return True, frame
                    if read_status == "need_data":
                        self._reopen_consumer()
                        return False, None
            except OSError as error:
                self._stats["read_errors"] = int(self._stats.get("read_errors", 0)) + 1
                self._stats["last_read_error"] = str(error)
                self._reopen_consumer()
                return False, None
            if not chunk:
                return False, None
            self._stats["bytes_read"] = int(self._stats.get("bytes_read", 0)) + len(chunk)
            self._buffer.extend(chunk)
            if len(self._buffer) > 4_000_000:
                del self._buffer[:-1_000_000]
        return False, None

    def _read_response_chunk(self) -> bytes:
        response = self._response
        if response is None:
            return b""
        read1 = getattr(response, "read1", None)
        if callable(read1):
            return read1(4096)
        return response.read(1)

    def _read_buffered_frame(self):
        header_end = self._buffer.find(b"\r\n\r\n")
        if header_end >= 0:
            header = bytes(self._buffer[:header_end])
            content_length = _multipart_content_length(header)
            if content_length is not None:
                payload_start = header_end + 4
                payload_end = payload_start + content_length
                if len(self._buffer) < payload_end:
                    return "need_data", None
                jpg = bytes(self._buffer[payload_start:payload_end])
                del self._buffer[:payload_end]
                self._stats["content_length_payloads"] = int(self._stats.get("content_length_payloads", 0)) + 1
                frame = self._decode_frame(jpg, source="content_length")
                return ("ready", frame) if frame is not None else ("skipped", None)
            if len(self._buffer) > 8192:
                del self._buffer[: header_end + 4]

        start = self._buffer.find(b"\xff\xd8")
        end = self._buffer.find(b"\xff\xd9", start + 2 if start >= 0 else 0)
        if start < 0 or end < 0:
            return "need_data", None
        jpg = bytes(self._buffer[start : end + 2])
        del self._buffer[: end + 2]
        self._stats["jpeg_marker_payloads"] = int(self._stats.get("jpeg_marker_payloads", 0)) + 1
        frame = self._decode_frame(jpg, source="jpeg_marker_scan")
        return ("ready", frame) if frame is not None else ("skipped", None)

    def _decode_frame(self, jpg: bytes, *, source: str):
        if not jpg.startswith(b"\xff\xd8") or not jpg.endswith(b"\xff\xd9"):
            self._note_rejected_frame(f"invalid_jpeg:{source}")
            return None
        frame = self._cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), self._cv2.IMREAD_COLOR)
        if frame is not None and not _frame_has_horizontal_tearing(frame):
            if _frame_is_temporal_splice(self._last_frame, frame):
                self._note_rejected_frame(f"temporal_splice:{source}")
                return None
            self._last_frame = frame
            self._consecutive_rejected_frames = 0
            self._stats["frames_accepted"] = int(self._stats.get("frames_accepted", 0)) + 1
            return frame
        if frame is not None:
            self._note_rejected_frame(f"horizontal_tearing:{source}")
        else:
            self._note_rejected_frame(f"decode_failed:{source}")
        return None

    def _note_rejected_frame(self, reason: str) -> None:
        self._stats["frames_rejected"] = int(self._stats.get("frames_rejected", 0)) + 1
        self._stats["last_reject_reason"] = str(reason)
        self._consecutive_rejected_frames += 1
        if self._consecutive_rejected_frames < 3:
            return
        self._reopen_consumer()

    def _reopen_consumer(self) -> None:
        self._buffer.clear()
        self._last_frame = None
        self._consecutive_rejected_frames = 0
        self._stats["buffer_resets"] = int(self._stats.get("buffer_resets", 0)) + 1
        response = self._response
        self._response = None
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        try:
            self._stats["reopen_count"] = int(self._stats.get("reopen_count", 0)) + 1
            self._open()
        except Exception:
            self._response = None

    def release(self) -> None:
        response = self._response
        self._response = None
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def _is_web_video_mjpeg_stream(source: object) -> bool:
    if not isinstance(source, str):
        return False
    try:
        parsed = urlparse(source)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.path.endswith("/stream"):
        return False
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    requested_type = str(query.get("type", "")).strip().lower()
    return requested_type in {"", "mjpeg"}


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
        self._calibration_profile: VisionCalibrationProfile | None = None
        self._calibration_profile_path = Path(self.config.vision_calibration_profile_path)
        self._calibration_profile_mtime: float | None = None
        self._pending_status: str | None = None
        profile_path = self._calibration_profile_path
        if profile_path.exists():
            try:
                self._calibration_profile = VisionCalibrationProfile.load(profile_path)
                self._calibration_profile_mtime = float(profile_path.stat().st_mtime)
                if (
                    self._calibration_profile is not None
                    and str(self.config.pick_tool_offset_source).strip().lower() == "target_pixel"
                    and self._calibration_profile.target_pixel is None
                ):
                    self._pending_status = (
                        "Vision calibration profile loaded, but servo.target_pixel is missing. "
                        "Run suction-target calibration before automatic pick alignment."
                    )
            except Exception as error:
                self._calibration_profile = None
                self._pending_status = f"Vision calibration profile unavailable: {error}"
        elif bool(self.config.vision_calibration_profile_required):
            self._pending_status = f"Vision calibration profile missing: {profile_path}"
        if calibration_params:
            try:
                merged_params = dict(calibration_params)
                if self._calibration_profile is not None:
                    if self._calibration_profile.dist_coeffs is not None and "D" not in merged_params:
                        merged_params["D"] = self._calibration_profile.dist_coeffs.reshape(-1).tolist()
                    if self._calibration_profile.image_size is not None and "image_size" not in merged_params:
                        merged_params["image_size"] = list(self._calibration_profile.image_size)
                    if not str(merged_params.get("profile_id", "")).strip():
                        merged_params["profile_id"] = self._calibration_profile.profile_id
                self._calibration = VisionCalibration.from_param_dict(merged_params)
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
        self._capture_drain_remaining = max(0, int(getattr(self.config, "vision_stream_drain_grabs", 0)))
        self._capture_ready_frames = 0
        self._capture_drained_frames = 0
        self._capture_rejected_frames = 0
        self._capture_startup_reference_frame = None
        self._capture_transport_stats: dict[str, object] = {}
        self._last_capture_ts = 0.0
        self._robot_pose_lock = threading.Lock()
        self._robot_z_mm: float | None = None
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
                self._latest_frame = None
                self._latest_frame_seq = 0
                self._last_infer_frame_seq = 0
                self._capture_drain_remaining = max(0, int(getattr(self.config, "vision_stream_drain_grabs", 0)))
                self._capture_ready_frames = 0
                self._capture_startup_reference_frame = None
            self._start_capture_pump(capture)
            self.status_changed.emit(f"Vision stream connected: {stream_url}")
            return True

        checked = ", ".join(self._stream_candidates)
        self.status_changed.emit(f"Vision stream unavailable, retrying... checked=[{checked}]")
        return False

    def _try_open_capture(self, stream_url: str):
        stream_url = _normalize_web_video_url(str(stream_url))
        source = int(stream_url) if stream_url.isdigit() else stream_url
        if _is_web_video_mjpeg_stream(source):
            # Prefer direct multipart MJPEG parsing for the Hiwonder official stream.
            # This stays on the PC side and never starts/restarts JetMax camera nodes.
            timeout_sec = max(0.2, float(self.config.vision_open_timeout_ms) / 1000.0)
            read_timeout_sec = max(0.1, float(self.config.vision_read_timeout_ms) / 1000.0)
            capture = None
            try:
                capture = _HttpMjpegCapture(
                    str(source),
                    cv2_module=self._cv2,
                    timeout_sec=timeout_sec,
                    read_timeout_sec=read_timeout_sec,
                )
                ok, frame = capture.read()
                if ok and frame is not None:
                    return capture
            except Exception:
                pass
            if capture is not None:
                capture.release()
        backend_candidates = [getattr(self._cv2, "CAP_ANY", 0)]
        if isinstance(source, str) and hasattr(self._cv2, "CAP_FFMPEG"):
            parsed = urlparse(source)
            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
            requested_type = str(query.get("type", "")).strip().lower()
            ffmpeg_backend = getattr(self._cv2, "CAP_FFMPEG")
            if parsed.scheme in {"rtsp", "tcp"}:
                backend_candidates = [ffmpeg_backend, getattr(self._cv2, "CAP_ANY", 0)]
            elif parsed.scheme in {"http", "https"} and requested_type not in {"h264", "vp8", "vp9"}:
                # JetMax official web_video_server MJPEG streams are more reliable with OpenCV's default backend.
                backend_candidates = [getattr(self._cv2, "CAP_ANY", 0), ffmpeg_backend]
            else:
                backend_candidates = [ffmpeg_backend, getattr(self._cv2, "CAP_ANY", 0)]
        deduped_backends: list[int] = []
        for backend in backend_candidates:
            if backend not in deduped_backends:
                deduped_backends.append(backend)

        probe_reads = max(1, int(self.config.vision_probe_reads))
        probe_sleep = max(0.0, float(self.config.vision_probe_sleep_ms) / 1000.0)
        for backend in deduped_backends:
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
                continue

            try:
                for _ in range(probe_reads):
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        return capture
                    if probe_sleep > 0.0:
                        time.sleep(probe_sleep)
            except Exception:
                pass
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
            if _frame_has_horizontal_tearing(frame):
                with self._frame_lock:
                    self._capture_rejected_frames += 1
                    self._capture_ready_frames = 0
                continue
            if not self._frame_is_stable_for_startup(frame):
                continue

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
                stats_getter = getattr(capture, "stats", None)
                if callable(stats_getter):
                    try:
                        self._capture_transport_stats = dict(stats_getter())
                    except Exception:
                        pass
            self._emit_frame_from_capture(frame, now)

    def _frame_is_stable_for_startup(self, frame) -> bool:
        with self._frame_lock:
            previous_frame = self._capture_startup_reference_frame
        if _frame_is_temporal_splice(previous_frame, frame):
            with self._frame_lock:
                self._capture_rejected_frames += 1
                self._capture_ready_frames = 0
                self._capture_startup_reference_frame = None
            return False
        try:
            frame_reference = frame.copy()
        except Exception:
            frame_reference = frame
        with self._frame_lock:
            self._capture_startup_reference_frame = frame_reference
            if self._capture_drain_remaining > 0:
                self._capture_drain_remaining -= 1
                self._capture_drained_frames += 1
                self._capture_ready_frames = 0
                return False
            self._capture_ready_frames += 1
            ready_frames = int(self._capture_ready_frames)
        # Require a short clean run after every open/reopen before the frame can
        # drive UI, recognition, or servo. This is PC-side only; it does not touch
        # the Hiwonder camera sender.
        return ready_frames >= 2

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
            frame_quality = frame_brightness_quality(
                frame,
                min_mean=float(getattr(self.config, "vision_frame_min_brightness_mean", 30.0)),
                min_p95=float(getattr(self.config, "vision_frame_min_brightness_p95", 45.0)),
            )
            self._reload_calibration_profile_if_needed()
            roi_center = self._resolve_roi_center(frame_w, frame_h)
            roi_radius = self._resolve_roi_radius(frame_w, frame_h)
            calibration_stage, calibration_z_mm = self._current_calibration_stage()
            alignment_target_pixel = self._resolve_alignment_target_pixel(
                frame_w,
                frame_h,
                roi_center,
                calibration_stage=calibration_stage,
                calibration_z_mm=calibration_z_mm,
            )
            action_center_tolerance_px = float(self.config.vision_servo_action_tolerance_px)
            if str(calibration_stage or "").strip().lower() == "search":
                action_center_tolerance_px = max(
                    action_center_tolerance_px,
                    float(getattr(self.config, "vision_servo_search_action_tolerance_px", action_center_tolerance_px)),
                )
            elif str(calibration_stage or "").strip().lower() in {"confirm", "pick"}:
                action_center_tolerance_px = float(
                    getattr(self.config, "vision_servo_low_action_tolerance_px", action_center_tolerance_px)
                )
            low_height_shape_fallback = (
                bool(getattr(self.config, "vision_low_height_shape_fallback_enabled", True))
                and str(calibration_stage or "").strip().lower() in {"confirm", "pick"}
            )

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
                frame_bgr=frame,
                fallback_to_frame=bool(self.config.vision_frame_fallback_enabled),
                prefer_frame_fallback=low_height_shape_fallback,
                fallback_min_area_ratio=float(
                    getattr(self.config, "vision_low_height_shape_fallback_min_area_ratio", 1.20)
                ),
                fallback_reject_edge_touch=bool(
                    low_height_shape_fallback
                    and getattr(self.config, "vision_low_height_reject_edge_fallback_candidates", True)
                ),
            )
            update_slots(
                self._slots,
                candidates,
                match_distance=120.0,
                lost_ttl=6,
                grasp_history_len=int(self.config.vision_grasp_history_frames),
                center_stability_tolerance_px=float(self.config.vision_center_stability_tolerance_px),
                grasp_stability_tolerance_px=float(self.config.vision_grasp_stability_tolerance_px),
                grasp_history_reset_px=float(self.config.vision_grasp_history_reset_px),
                grasp_angle_stability_tolerance_deg=float(
                    self.config.vision_grasp_angle_stability_tolerance_deg
                ),
            )
            annotate_slots_with_cylindrical(
                self._slots,
                calibration=self._calibration,
                calibration_profile=self._calibration_profile,
                frame_size=(frame_w, frame_h),
                roi_center=roi_center,
                world_scale_xy=float(self.config.vision_world_scale_xy),
                world_offset_xy_mm=(
                    float(self.config.vision_world_offset_xy_mm[0]),
                    float(self.config.vision_world_offset_xy_mm[1]),
                ),
                mapping_mode=str(self.config.vision_mapping_mode),
                calibration_profile_required=bool(self.config.vision_calibration_profile_required),
                action_error_threshold_mm=float(self.config.vision_action_max_error_mm),
                center_tolerance_px=float(self.config.vision_servo_center_tolerance_px),
                action_center_tolerance_px=action_center_tolerance_px,
                alignment_target_pixel=alignment_target_pixel,
                alignment_target_required=str(self.config.pick_tool_offset_source).strip().lower() == "target_pixel",
                calibration_stage=calibration_stage,
                calibration_z_mm=calibration_z_mm,
                grasp_quality_threshold=float(self.config.vision_grasp_quality_threshold),
                required_stable_frames=int(self.config.vision_grasp_stable_frames),
                grasp_angle_stability_tolerance_deg=float(
                    self.config.vision_grasp_angle_stability_tolerance_deg
                ),
                servo_measurement_point=str(getattr(self.config, "vision_servo_measurement_point", "center")),
                low_height_servo_measurement_point=str(
                    getattr(self.config, "vision_servo_low_height_measurement_point", "")
                ),
                low_height_confirm_z_mm=float(
                    getattr(self.config, "vision_pick_confirm_z_mm", self.config.robot_approach_z)
                ),
                low_height_guard_band_mm=float(
                    getattr(self.config, "vision_continuous_servo_low_height_guard_band_mm", 30.0)
                ),
            )
            calibration_ready = self._calibration is not None or (
                self._calibration_profile is not None
                and (self._calibration_profile.has_pixel_to_delta_model or self._calibration_profile.has_stage_models)
            )
            calibration_ready = calibration_ready and (
                not bool(self.config.vision_calibration_profile_required) or self._calibration_profile is not None
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
                capture_ts=float(capture_ts),
                stream_age_ms=max(0.0, (time.perf_counter() - capture_ts) * 1000.0),
                detected_count=detected_count,
                calibration_ready=calibration_ready,
                mapping_mode=str(self.config.vision_mapping_mode),
                calibration_profile_id="" if self._calibration_profile is None else self._calibration_profile.profile_id,
                calibration_profile_required=bool(self.config.vision_calibration_profile_required),
                alignment_target_pixel=alignment_target_pixel,
                calibration_stage=calibration_stage,
                calibration_z_mm=calibration_z_mm,
                frame_quality=frame_quality,
            )
            packet["infer_interval_ms"] = float(self._infer_interval_dynamic_ms)
            packet["camera_transport"] = dict(self._capture_transport_stats)
            packet["camera_startup_drain"] = {
                "drained_frames": int(self._capture_drained_frames),
                "ready_frames": int(self._capture_ready_frames),
                "rejected_frames": int(self._capture_rejected_frames),
            }
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

    @staticmethod
    def _coerce_frame_pixel(value: object, frame_w: int, frame_h: int) -> tuple[float, float] | None:
        if not isinstance(value, (tuple, list)) or len(value) < 2:
            return None
        try:
            x = float(value[0])
            y = float(value[1])
        except (TypeError, ValueError):
            return None
        if 0.0 <= x < float(frame_w) and 0.0 <= y < float(frame_h):
            return (x, y)
        return None

    def _resolve_alignment_target_pixel(
        self,
        frame_w: int,
        frame_h: int,
        roi_center: tuple[int, int],
        *,
        calibration_stage: str | None = None,
        calibration_z_mm: float | None = None,
    ) -> tuple[float, float] | None:
        configured = self._coerce_frame_pixel(self.config.vision_pick_target_pixel, frame_w, frame_h)
        if configured is not None:
            return configured
        if str(self.config.pick_tool_offset_source).strip().lower() == "command_bias":
            return (float(roi_center[0]), float(roi_center[1]))
        if self._calibration_profile is not None:
            try:
                active_profile = self._calibration_profile.model_for_stage(
                    calibration_stage,
                    z_mm=calibration_z_mm,
                    allow_fallback=True,
                )
            except Exception:
                active_profile = self._calibration_profile
            profile_target = self._coerce_frame_pixel(active_profile.target_pixel, frame_w, frame_h)
            if profile_target is not None:
                return profile_target
        if str(self.config.pick_tool_offset_source).strip().lower() == "target_pixel":
            return None
        return (float(roi_center[0]), float(roi_center[1]))

    def _reload_calibration_profile_if_needed(self) -> None:
        profile_path = self._calibration_profile_path
        if not profile_path.exists():
            return
        try:
            mtime = float(profile_path.stat().st_mtime)
        except OSError:
            return
        if self._calibration_profile_mtime is not None and mtime <= float(self._calibration_profile_mtime):
            return
        try:
            self._calibration_profile = VisionCalibrationProfile.load(profile_path)
            self._calibration_profile_mtime = mtime
            target = self._calibration_profile.target_pixel
            if str(self.config.pick_tool_offset_source).strip().lower() == "command_bias":
                suffix = " target=roi_center command_bias"
            else:
                suffix = "" if target is None else f" target=({target[0]:.1f},{target[1]:.1f})"
            self.status_changed.emit(f"Vision calibration profile reloaded: {self._calibration_profile.profile_id}{suffix}")
        except Exception as error:
            self.status_changed.emit(f"Vision calibration profile reload failed: {error}")

    @pyqtSlot()
    def reset_tracking(self) -> None:
        for slot in self._slots:
            slot.clear()
        self.status_changed.emit("Vision tracking reset after robot motion.")

    @pyqtSlot(float)
    def set_robot_z(self, z_mm: float) -> None:
        with self._robot_pose_lock:
            self._robot_z_mm = float(z_mm)

    def _current_calibration_stage(self) -> tuple[str, float]:
        with self._robot_pose_lock:
            robot_z = self._robot_z_mm
        search_z = float(getattr(self.config, "vision_pick_search_z_mm", self.config.robot_carry_z))
        confirm_z = float(getattr(self.config, "vision_pick_confirm_z_mm", self.config.robot_approach_z))
        pick_z = float(getattr(self.config, "robot_pick_z", confirm_z))
        tolerance = max(0.5, float(getattr(self.config, "vision_pick_z_tolerance_mm", 4.0)))
        if robot_z is None:
            return ("search", search_z)
        z_value = float(robot_z)
        if abs(z_value - pick_z) <= tolerance:
            return ("pick", pick_z)
        if abs(z_value - confirm_z) <= tolerance:
            return ("confirm", confirm_z)
        if z_value < search_z - tolerance:
            return ("confirm", z_value)
        return ("search", search_z)


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

    def reset_tracking(self) -> None:
        worker = self.worker
        thread = self.thread
        if worker is None:
            return
        try:
            if thread is not None and thread.isRunning():
                QMetaObject.invokeMethod(worker, "reset_tracking", Qt.QueuedConnection)
            else:
                worker.reset_tracking()
        except RuntimeError:
            pass

    def set_robot_z(self, z_mm: float) -> None:
        worker = self.worker
        thread = self.thread
        if worker is None:
            return
        try:
            if thread is not None and thread.isRunning():
                QMetaObject.invokeMethod(worker, "set_robot_z", Qt.QueuedConnection, Q_ARG(float, float(z_mm)))
            else:
                worker.set_robot_z(float(z_mm))
        except RuntimeError:
            pass

    def healthcheck(self) -> dict[str, object]:
        return {
            "running": self.worker is not None,
            "weights": _resolve_weights_path(self.config),
            "source": self.config.resolve_vision_stream_url(),
            "source_candidates": self.config.resolve_vision_stream_candidates(),
            "last_packet": self._last_packet,
            "calibration_ready": self.calibration_params is not None,
            "calibration_profile": str(self.config.vision_calibration_profile_path),
        }

    def _handle_packet_ready(self, packet: dict[str, object]) -> None:
        self._last_packet = dict(packet)
