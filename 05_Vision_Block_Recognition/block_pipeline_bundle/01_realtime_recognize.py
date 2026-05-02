#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
except Exception:  # pragma: no cover - optional dependency path
    torch = None

from PyQt5.QtCore import QPointF, QRect, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QApplication, QLabel, QWidget


MAX_SLOTS = 4
DEFAULT_FREQS = (8.0, 10.0, 12.0, 15.0)
DEFAULT_REFRESH_RATE_HZ = 240.0
DEFAULT_STIM_MEAN = 0.5
DEFAULT_STIM_AMP = 0.5
DEFAULT_STIM_PHI = 0.0
DEFAULT_IMGSZ = 512
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.5
DEFAULT_MAX_DET = 6
DEFAULT_MATCH_DISTANCE = 120.0
DEFAULT_LOST_TTL = 6
DEFAULT_WINDOW_SIZE = (1280, 720)
MORPH_KERNEL_SIZE = 3


def log_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_int_tuple(values: Optional[Tuple[int, int]]) -> Optional[List[int]]:
    if values is None:
        return None
    return [int(values[0]), int(values[1])]


def safe_bbox_list(values: Optional[Tuple[int, int, int, int]]) -> Optional[List[int]]:
    if values is None:
        return None
    return [int(values[0]), int(values[1]), int(values[2]), int(values[3])]


def parse_source(raw: str) -> Union[int, str]:
    text = str(raw).strip()
    if not text:
        return 0
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def parse_freqs(raw: str) -> List[float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    freqs = [float(part) for part in parts]
    if len(freqs) != MAX_SLOTS:
        raise ValueError(f"--freqs must contain exactly {MAX_SLOTS} values")
    return freqs


def resolve_device(device_request: str) -> Tuple[str, bool]:
    request = str(device_request).strip().lower()
    if request in ("", "auto"):
        if torch is not None and torch.cuda.is_available():
            return "0", True
        return "cpu", False
    if request == "cpu":
        return "cpu", False
    return str(device_request).strip(), True


def fit_rect(src_width: int, src_height: int, dst_width: int, dst_height: int) -> QRect:
    if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
        return QRect()
    scale = min(dst_width / float(src_width), dst_height / float(src_height))
    draw_width = max(1, int(round(src_width * scale)))
    draw_height = max(1, int(round(src_height * scale)))
    left = (dst_width - draw_width) // 2
    top = (dst_height - draw_height) // 2
    return QRect(left, top, draw_width, draw_height)


def bbox_iou(
    box_a: Optional[Tuple[int, int, int, int]],
    box_b: Optional[Tuple[int, int, int, int]],
) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def euclidean_distance(point_a: Tuple[int, int], point_b: Tuple[int, int]) -> float:
    return math.hypot(float(point_a[0] - point_b[0]), float(point_a[1] - point_b[1]))


def largest_component(binary_mask: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    if binary_mask.ndim != 2:
        return None, 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return None, 0
    component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component_index, cv2.CC_STAT_AREA])
    component = np.where(labels == component_index, 255, 0).astype(np.uint8)
    return component, area


def mask_to_geometry(
    mask: np.ndarray,
    frame_shape: Tuple[int, int],
) -> Optional[Tuple[List[Tuple[int, int]], Tuple[int, int], Tuple[int, int, int, int], int]]:
    frame_h, frame_w = frame_shape
    if mask.shape != (frame_h, frame_w):
        mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

    binary = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    component, _ = largest_component(binary)
    if component is None:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    component = cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel, iterations=1)
    component = cv2.morphologyEx(component, cv2.MORPH_OPEN, kernel, iterations=1)
    component, area_px = largest_component(component)
    if component is None or area_px <= 0:
        return None

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None

    moments = cv2.moments(component, binaryImage=True)
    if moments["m00"] <= 0:
        return None

    center_x = int(round(moments["m10"] / moments["m00"]))
    center_y = int(round(moments["m01"] / moments["m00"]))
    x, y, w, h = cv2.boundingRect(contour)
    epsilon = max(1.0, 0.004 * cv2.arcLength(contour, True))
    contour = cv2.approxPolyDP(contour, epsilon, True)
    polygon = [(int(point[0][0]), int(point[0][1])) for point in contour]
    if len(polygon) < 3:
        return None

    bbox = (int(x), int(y), int(x + w), int(y + h))
    return polygon, (center_x, center_y), bbox, int(area_px)


@dataclass
class AppConfig:
    source: Union[int, str]
    weights: Path
    device_request: str
    device: str
    half: bool
    imgsz: int
    conf: float
    iou: float
    max_det: int
    roi_radius: int
    requested_refresh_rate_hz: float
    refresh_rate_hz: float
    screen_refresh_rate_hz: float
    stim_freqs: List[float]
    stim_mean: float
    stim_amp: float
    stim_phi: float
    match_distance: float
    lost_ttl: int
    warmup_runs: int
    exit_after_sec: float
    fullscreen: bool


@dataclass
class DetectionCandidate:
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    area_px: int
    confidence: float
    polygon: List[Tuple[int, int]]
    distance_to_roi: float


@dataclass
class SlotState:
    slot: int
    freq_hz: float
    valid: bool = False
    observed: bool = False
    pixel_center: Optional[Tuple[int, int]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    area_px: int = 0
    confidence: float = 0.0
    polygon: List[Tuple[int, int]] = field(default_factory=list)
    age: int = 0
    lost_frames: int = 0

    def assign(self, candidate: DetectionCandidate, increment_age: bool) -> None:
        self.valid = True
        self.observed = True
        self.pixel_center = candidate.center
        self.bbox = candidate.bbox
        self.area_px = int(candidate.area_px)
        self.confidence = float(candidate.confidence)
        self.polygon = list(candidate.polygon)
        self.lost_frames = 0
        self.age = self.age + 1 if increment_age else 1

    def mark_missing(self) -> None:
        if not self.valid:
            return
        self.observed = False
        self.lost_frames += 1
        self.age += 1

    def clear(self) -> None:
        self.valid = False
        self.observed = False
        self.pixel_center = None
        self.bbox = None
        self.area_px = 0
        self.confidence = 0.0
        self.polygon = []
        self.age = 0
        self.lost_frames = 0

    def to_packet(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "freq_hz": float(self.freq_hz),
            "valid": bool(self.valid),
            "observed": bool(self.observed),
            "pixel_center": self.pixel_center,
            "bbox": self.bbox,
            "area_px": int(self.area_px),
            "confidence": float(self.confidence),
            "polygon": list(self.polygon),
            "age": int(self.age),
            "lost_frames": int(self.lost_frames),
        }


def load_app_config(argv: Optional[Sequence[str]] = None) -> AppConfig:
    parser = argparse.ArgumentParser(description="Single-file block detection and SSVEP mask flicker viewer")
    parser.add_argument("--source", type=str, default="0", help="Camera index, video file, or RTSP/HTTP stream URL")
    parser.add_argument("--weights", type=str, required=True, help="Path to a YOLO segmentation weight file")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', '0', or other Ultralytics device string")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument("--roi-radius", type=int, default=0, help="ROI radius in source pixels; 0 means auto")
    parser.add_argument("--refresh-rate", type=float, default=DEFAULT_REFRESH_RATE_HZ)
    parser.add_argument("--freqs", type=str, default="8,10,12,15")
    parser.add_argument("--stim-mean", type=float, default=DEFAULT_STIM_MEAN)
    parser.add_argument("--stim-amp", type=float, default=DEFAULT_STIM_AMP)
    parser.add_argument("--stim-phi", type=float, default=DEFAULT_STIM_PHI)
    parser.add_argument("--match-distance", type=float, default=DEFAULT_MATCH_DISTANCE)
    parser.add_argument("--lost-ttl", type=int, default=DEFAULT_LOST_TTL)
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of startup warmup inferences before live processing")
    parser.add_argument("--exit-after-sec", type=float, default=0.0, help="Automatically close the viewer after N seconds; 0 disables it")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args(argv)

    weights = Path(args.weights).expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weight file not found: {weights}")

    stim_freqs = parse_freqs(args.freqs)
    device, half = resolve_device(args.device)
    return AppConfig(
        source=parse_source(args.source),
        weights=weights,
        device_request=str(args.device),
        device=device,
        half=half,
        imgsz=max(64, int(args.imgsz)),
        conf=clamp01(args.conf),
        iou=clamp01(args.iou),
        max_det=max(MAX_SLOTS, int(args.max_det)),
        roi_radius=max(0, int(args.roi_radius)),
        requested_refresh_rate_hz=max(1.0, float(args.refresh_rate)),
        refresh_rate_hz=max(1.0, float(args.refresh_rate)),
        screen_refresh_rate_hz=0.0,
        stim_freqs=stim_freqs,
        stim_mean=float(args.stim_mean),
        stim_amp=max(0.0, float(args.stim_amp)),
        stim_phi=float(args.stim_phi),
        match_distance=max(1.0, float(args.match_distance)),
        lost_ttl=max(1, int(args.lost_ttl)),
        warmup_runs=max(0, int(args.warmup_runs)),
        exit_after_sec=max(0.0, float(args.exit_after_sec)),
        fullscreen=bool(args.fullscreen),
    )


class WindowsTimerResolution:
    def __init__(self) -> None:
        self._winmm = None

    def __enter__(self) -> "WindowsTimerResolution":
        if sys.platform.startswith("win"):
            try:
                self._winmm = ctypes.WinDLL("winmm")
                self._winmm.timeBeginPeriod(1)
            except Exception:
                self._winmm = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._winmm is not None:
            try:
                self._winmm.timeEndPeriod(1)
            except Exception:
                pass
        return False


class StimClockThread(QThread):
    tick = pyqtSignal(float, int, float)

    def __init__(self, refresh_rate_hz: float) -> None:
        super().__init__()
        self.refresh_rate_hz = float(refresh_rate_hz)
        if self.refresh_rate_hz <= 0:
            raise ValueError("refresh_rate_hz must be positive")
        self.period_ns = int(round(1_000_000_000 / self.refresh_rate_hz))
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self._stop_event.clear()
        frame_idx = 0
        start_ns = time.perf_counter_ns()
        next_tick_ns = start_ns

        with WindowsTimerResolution():
            while not self._stop_event.is_set():
                now_ns = time.perf_counter_ns()
                remain_ns = next_tick_ns - now_ns
                if remain_ns > 1_000_000:
                    sleep_s = max((remain_ns - 200_000) / 1_000_000_000.0, 0.0)
                    if self._stop_event.wait(sleep_s):
                        break
                    continue

                while not self._stop_event.is_set():
                    now_ns = time.perf_counter_ns()
                    remain_ns = next_tick_ns - now_ns
                    if remain_ns <= 0:
                        break
                    if remain_ns > 200_000:
                        time.sleep(0)
                        continue
                    if remain_ns > 50_000:
                        time.sleep(0.0001)
                        continue

                if self._stop_event.is_set():
                    break

                emit_ns = time.perf_counter_ns()
                jitter_ms = (emit_ns - next_tick_ns) / 1_000_000.0
                rel_t = (next_tick_ns - start_ns) / 1_000_000_000.0
                self.tick.emit(rel_t, frame_idx, float(jitter_ms))

                frame_idx += 1
                next_tick_ns = start_ns + frame_idx * self.period_ns
                if emit_ns - next_tick_ns > self.period_ns:
                    frame_idx = int((emit_ns - start_ns) // self.period_ns) + 1
                    next_tick_ns = start_ns + frame_idx * self.period_ns


class CameraLoader:
    def __init__(self, source: Union[int, str]) -> None:
        self.source = source
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._last_capture_ts = 0.0
        self._fps = 0.0
        self._status = "camera initializing"
        self._frame_counter = 0
        self._fps_window_start = time.perf_counter()

    def start(self) -> "CameraLoader":
        if self._thread is not None:
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="camera-loader", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def get_latest(self, last_seq: int) -> Optional[Tuple[int, np.ndarray, float, float, str]]:
        with self._lock:
            if self._frame is None or self._seq == last_seq:
                return None
            return self._seq, self._frame, self._last_capture_ts, self._fps, self._status

    def peek_latest(self) -> Optional[Tuple[int, np.ndarray, float, float, str]]:
        with self._lock:
            if self._frame is None:
                return None
            return self._seq, self._frame, self._last_capture_ts, self._fps, self._status

    def status_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            age_ms = (time.perf_counter() - self._last_capture_ts) * 1000.0 if self._last_capture_ts > 0 else -1.0
            return {
                "status": self._status,
                "capture_fps": float(self._fps),
                "last_frame_age_ms": float(age_ms),
            }

    def _open_capture(self) -> cv2.VideoCapture:
        backend = cv2.CAP_ANY
        if isinstance(self.source, str) and hasattr(cv2, "CAP_FFMPEG"):
            backend = cv2.CAP_FFMPEG
        try:
            capture = cv2.VideoCapture(self.source, backend)
        except TypeError:
            capture = cv2.VideoCapture(self.source)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
        return capture

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def _run(self) -> None:
        while not self._stop_event.is_set():
            capture = self._open_capture()
            if capture is None or not capture.isOpened():
                self._set_status("camera reconnecting")
                if capture is not None:
                    capture.release()
                if self._stop_event.wait(1.0):
                    return
                continue

            with self._lock:
                self._capture = capture
                self._status = "camera connected"

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    self._set_status("camera stream lost")
                    break

                now = time.perf_counter()
                self._frame_counter += 1
                elapsed = now - self._fps_window_start
                if elapsed >= 1.0:
                    current_fps = self._frame_counter / elapsed
                    self._fps = current_fps if self._fps <= 0 else (self._fps * 0.8 + current_fps * 0.2)
                    self._frame_counter = 0
                    self._fps_window_start = now

                with self._lock:
                    self._seq += 1
                    self._frame = frame
                    self._last_capture_ts = now
                    self._status = "camera connected"

            capture.release()
            with self._lock:
                if self._capture is capture:
                    self._capture = None
            if self._stop_event.wait(0.5):
                return


class InferenceWorker(QThread):
    packet_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, camera: CameraLoader, config: AppConfig) -> None:
        super().__init__()
        self.camera = camera
        self.config = config
        self._stop_event = threading.Event()
        self._model: Optional[YOLO] = None
        self._roi_radius_cache = 0
        self._slots = [SlotState(slot=index + 1, freq_hz=config.stim_freqs[index]) for index in range(MAX_SLOTS)]

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            self._model = YOLO(str(self.config.weights))
            self._warmup_model()
        except Exception:
            self.error.emit(f"Failed to load model:\n{traceback.format_exc()}")
            return

        last_seq = 0
        while not self._stop_event.is_set():
            latest = self.camera.get_latest(last_seq)
            if latest is None:
                self._stop_event.wait(0.002)
                continue

            frame_id, frame, capture_ts, capture_fps, camera_status = latest
            last_seq = frame_id
            frame_h, frame_w = frame.shape[:2]
            roi_center = (frame_w // 2, frame_h // 2)
            roi_radius = self._resolve_roi_radius(frame_w, frame_h)
            queue_age_ms = max(0.0, (time.perf_counter() - capture_ts) * 1000.0)

            infer_start = time.perf_counter()
            try:
                result = self._predict(frame)
            except Exception:
                self.error.emit(f"Inference failed:\n{traceback.format_exc()}")
                return
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            post_start = time.perf_counter()
            candidates, detected_count = self._extract_candidates(result, (frame_h, frame_w), roi_center, roi_radius)
            self._update_slots(candidates)
            packet = {
                "frame_id": int(frame_id),
                "image_size": (int(frame_w), int(frame_h)),
                "roi_center": roi_center,
                "roi_radius": int(roi_radius),
                "slots": [slot.to_packet() for slot in self._slots],
                "capture_fps": float(capture_fps),
                "queue_age_ms": float(queue_age_ms),
                "infer_ms": float(infer_ms),
                "post_ms": float((time.perf_counter() - post_start) * 1000.0),
                "detected_count": int(detected_count),
                "camera_status": str(camera_status),
            }
            self.packet_ready.emit(packet)

    def _warmup_model(self) -> None:
        assert self._model is not None
        if self.config.warmup_runs <= 0:
            return
        dummy_size = max(128, int(self.config.imgsz))
        dummy = np.zeros((dummy_size, dummy_size, 3), dtype=np.uint8)
        warmup_start = time.perf_counter()
        for _ in range(self.config.warmup_runs):
            self._predict(dummy)
        elapsed_ms = (time.perf_counter() - warmup_start) * 1000.0
        log_stderr(
            f"Model warmup complete: runs={self.config.warmup_runs}, imgsz={self.config.imgsz}, "
            f"elapsed_ms={elapsed_ms:.1f}"
        )

    def _resolve_roi_radius(self, frame_w: int, frame_h: int) -> int:
        if self.config.roi_radius > 0:
            return self.config.roi_radius
        if self._roi_radius_cache <= 0:
            self._roi_radius_cache = max(10, int(round(min(frame_w, frame_h) * 0.35)))
        return self._roi_radius_cache

    def _predict(self, frame: np.ndarray):
        assert self._model is not None
        predict_kwargs: Dict[str, Any] = {
            "source": frame,
            "imgsz": self.config.imgsz,
            "conf": self.config.conf,
            "iou": self.config.iou,
            "max_det": self.config.max_det,
            "verbose": False,
            "device": self.config.device,
        }
        if self.config.half:
            predict_kwargs["half"] = True
        results = self._model.predict(**predict_kwargs)
        return results[0]

    def _extract_candidates(
        self,
        result: Any,
        frame_shape: Tuple[int, int],
        roi_center: Tuple[int, int],
        roi_radius: int,
    ) -> Tuple[List[DetectionCandidate], int]:
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None or boxes.conf is None or masks.data is None:
            return [], 0

        confidences = boxes.conf.detach().cpu().numpy()
        mask_data = masks.data.detach().cpu().numpy()
        count = min(len(confidences), len(mask_data))
        candidates: List[DetectionCandidate] = []
        for index in range(count):
            geometry = mask_to_geometry(mask_data[index], frame_shape)
            if geometry is None:
                continue
            polygon, center, bbox, area_px = geometry
            distance_to_roi = euclidean_distance(center, roi_center)
            if distance_to_roi > roi_radius:
                continue
            candidates.append(
                DetectionCandidate(
                    center=center,
                    bbox=bbox,
                    area_px=int(area_px),
                    confidence=float(confidences[index]),
                    polygon=polygon,
                    distance_to_roi=float(distance_to_roi),
                )
            )

        candidates.sort(key=lambda item: (-item.area_px, item.distance_to_roi))
        return candidates[: self.config.max_det], len(candidates)

    def _update_slots(self, candidates: List[DetectionCandidate]) -> None:
        matched_slots: set[int] = set()
        matched_candidates: set[int] = set()
        pairs: List[Tuple[float, float, float, int, int]] = []

        for slot_index, slot in enumerate(self._slots):
            if not slot.valid or slot.pixel_center is None:
                continue
            for candidate_index, candidate in enumerate(candidates):
                distance = euclidean_distance(slot.pixel_center, candidate.center)
                overlap = bbox_iou(slot.bbox, candidate.bbox)
                if distance > self.config.match_distance and overlap <= 0.05:
                    continue
                score = (distance / self.config.match_distance) + (1.0 - overlap) * 0.35
                pairs.append((score, distance, -overlap, slot_index, candidate_index))

        pairs.sort()
        for _, _, _, slot_index, candidate_index in pairs:
            if slot_index in matched_slots or candidate_index in matched_candidates:
                continue
            self._slots[slot_index].assign(candidates[candidate_index], increment_age=True)
            matched_slots.add(slot_index)
            matched_candidates.add(candidate_index)

        for slot_index, slot in enumerate(self._slots):
            if slot_index in matched_slots or not slot.valid:
                continue
            slot.mark_missing()
            if slot.lost_frames > self.config.lost_ttl:
                slot.clear()

        remaining_candidates = [candidates[index] for index in range(len(candidates)) if index not in matched_candidates]
        free_slots = [slot for slot in self._slots if not slot.valid]
        for slot, candidate in zip(free_slots, remaining_candidates):
            slot.assign(candidate, increment_age=False)


class VisionWindow(QWidget):
    def __init__(self, config: AppConfig, camera: CameraLoader) -> None:
        super().__init__()
        self.config = config
        self.camera = camera
        self.selected_slot = 0
        self.current_t = 0.0
        self.current_tick_idx = -1
        self.tick_jitter_ms = 0.0
        self.display_fps = 0.0
        self.packet_fps = 0.0
        self._paint_counter = 0
        self._paint_window_start = time.perf_counter()
        self._packet_counter = 0
        self._packet_window_start = time.perf_counter()
        self._packet: Optional[Dict[str, Any]] = None
        self._background_image: Optional[QImage] = None
        self._background_pixmap: Optional[QPixmap] = None
        self._image_rect = QRect()
        self._image_size = (0, 0)
        self._displayed_frame_seq = 0
        self._camera_status = "camera initializing"
        self._capture_fps = 0.0
        self._infer_ms = 0.0
        self._post_ms = 0.0
        self._queue_age_ms = 0.0
        self._detected_count = 0
        self._last_frame_age_ms = -1.0
        self._error_text = ""
        self._hud_font = QFont("Consolas", 11)
        self._label_font = QFont("Consolas", 12, QFont.Bold)
        self.setWindowTitle("Block Center SSVEP")
        self.setStyleSheet("background-color: black;")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._video_label = QLabel(self)
        self._video_label.setFocusPolicy(Qt.NoFocus)
        self._video_label.setAlignment(Qt.AlignCenter)
        self._overlay = OverlayWidget(self)
        self._overlay.raise_()
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._poll_camera_status)
        self._status_timer.start(250)
        self._frame_timer = QTimer(self)
        self._frame_timer.setTimerType(Qt.PreciseTimer)
        self._frame_timer.timeout.connect(self._poll_latest_frame)
        self._frame_timer.start(15)
        if self.config.exit_after_sec > 0:
            QTimer.singleShot(int(round(self.config.exit_after_sec * 1000.0)), self.close)

    def on_stim_tick(self, rel_t: float, tick_index: int, jitter_ms: float) -> None:
        self.current_t = float(rel_t)
        self.current_tick_idx = int(tick_index)
        self.tick_jitter_ms = float(jitter_ms)
        self._overlay.update()

    def on_vision_packet(self, packet: Dict[str, Any]) -> None:
        self._packet = packet
        self._camera_status = packet.get("camera_status", self._camera_status)
        self._capture_fps = float(packet.get("capture_fps", self._capture_fps))
        self._infer_ms = float(packet.get("infer_ms", self._infer_ms))
        self._post_ms = float(packet.get("post_ms", self._post_ms))
        self._queue_age_ms = float(packet.get("queue_age_ms", self._queue_age_ms))
        self._detected_count = int(packet.get("detected_count", self._detected_count))
        self._update_packet_fps()
        self._emit_output()
        self._overlay.update()

    def on_worker_error(self, message: str) -> None:
        self._error_text = message.strip()
        log_stderr(self._error_text)
        self._overlay.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._overlay.setGeometry(self.rect())
        self._rebuild_background_pixmap()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        if Qt.Key_1 <= event.key() <= Qt.Key_4:
            self.selected_slot = event.key() - Qt.Key_0
            log_stderr(f"selected_slot={self.selected_slot}")
            self._emit_output()
            self._overlay.update()
            return
        if event.key() == Qt.Key_0:
            self.selected_slot = 0
            log_stderr("selected_slot=0")
            self._emit_output()
            self._overlay.update()
            return
        super().keyPressEvent(event)

    def _poll_camera_status(self) -> None:
        snapshot = self.camera.status_snapshot()
        self._camera_status = str(snapshot.get("status", self._camera_status))
        self._capture_fps = float(snapshot.get("capture_fps", self._capture_fps))
        self._last_frame_age_ms = float(snapshot.get("last_frame_age_ms", self._last_frame_age_ms))
        self._overlay.update()

    def _poll_latest_frame(self) -> None:
        latest = self.camera.peek_latest()
        if latest is None:
            return
        frame_seq, frame, capture_ts, capture_fps, camera_status = latest
        self._camera_status = camera_status
        self._capture_fps = float(capture_fps)
        self._last_frame_age_ms = max(0.0, (time.perf_counter() - capture_ts) * 1000.0)
        if frame_seq == self._displayed_frame_seq:
            return
        self._displayed_frame_seq = frame_seq
        self._update_background(frame)

    def _update_background(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
        self._background_image = qimage
        self._image_size = (width, height)
        self._rebuild_background_pixmap()

    def _rebuild_background_pixmap(self) -> None:
        if self._background_image is None:
            self._background_pixmap = None
            self._image_rect = QRect()
            self._video_label.clear()
            self._video_label.setGeometry(QRect())
            return
        dst_rect = fit_rect(
            self._background_image.width(),
            self._background_image.height(),
            max(1, self.width()),
            max(1, self.height()),
        )
        pixmap = QPixmap.fromImage(self._background_image)
        self._background_pixmap = pixmap.scaled(
            dst_rect.size(),
            Qt.IgnoreAspectRatio,
            Qt.FastTransformation,
        )
        self._image_rect = dst_rect
        self._video_label.setGeometry(dst_rect)
        self._video_label.setPixmap(self._background_pixmap)
        self._video_label.show()

    def _map_point(self, point: Tuple[int, int]) -> QPointF:
        if self._image_rect.isNull() or self._image_size[0] <= 0 or self._image_size[1] <= 0:
            return QPointF(0.0, 0.0)
        scale_x = self._image_rect.width() / float(self._image_size[0])
        scale_y = self._image_rect.height() / float(self._image_size[1])
        return QPointF(
            self._image_rect.left() + point[0] * scale_x,
            self._image_rect.top() + point[1] * scale_y,
        )

    def _draw_slots(self, painter: QPainter) -> None:
        if self._packet is None:
            return

        for slot in self._packet.get("slots", []):
            if not slot.get("valid"):
                continue
            polygon = slot.get("polygon") or []
            if len(polygon) < 3:
                continue
            mapped_polygon = QPolygonF([self._map_point((int(x), int(y))) for x, y in polygon])
            luminance = clamp01(
                self.config.stim_mean
                + self.config.stim_amp
                * math.sin(2.0 * math.pi * float(slot["freq_hz"]) * self.current_t + self.config.stim_phi)
            )
            gray_value = int(round(255.0 * luminance))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(gray_value, gray_value, gray_value))
            painter.drawPolygon(mapped_polygon)

        for slot in self._packet.get("slots", []):
            if not slot.get("valid"):
                continue
            polygon = slot.get("polygon") or []
            center = slot.get("pixel_center")
            if len(polygon) < 3 or center is None:
                continue
            mapped_polygon = QPolygonF([self._map_point((int(x), int(y))) for x, y in polygon])
            border_color = QColor(0, 255, 180) if int(slot["slot"]) == self.selected_slot else QColor(255, 255, 255)
            if not slot.get("observed", False):
                border_color = QColor(255, 200, 0)
            painter.setPen(QPen(border_color, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(mapped_polygon)

            mapped_center = self._map_point((int(center[0]), int(center[1])))
            painter.setPen(QPen(border_color, 2))
            center_x = int(round(mapped_center.x()))
            center_y = int(round(mapped_center.y()))
            painter.drawLine(center_x - 8, center_y, center_x + 8, center_y)
            painter.drawLine(center_x, center_y - 8, center_x, center_y + 8)
            painter.setFont(self._label_font)
            label = f"[{slot['slot']}] {slot['freq_hz']:g}Hz"
            painter.drawText(center_x + 10, center_y - 10, label)

    def _draw_roi(self, painter: QPainter) -> None:
        if self._packet is None or self._image_rect.isNull():
            return
        roi_center = self._packet.get("roi_center")
        roi_radius = int(self._packet.get("roi_radius", 0))
        if roi_center is None or roi_radius <= 0:
            return
        mapped_center = self._map_point((int(roi_center[0]), int(roi_center[1])))
        scale = self._image_rect.width() / float(self._image_size[0]) if self._image_size[0] > 0 else 1.0
        mapped_radius = max(1, int(round(roi_radius * scale)))
        painter.setPen(QPen(QColor(255, 128, 0), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(mapped_center, mapped_radius, mapped_radius)

    def _draw_hud(self, painter: QPainter) -> None:
        metrics = [
            f"camera_status={self._camera_status}",
            f"capture_fps={self._capture_fps:.1f}",
            f"packet_fps={self.packet_fps:.1f}",
            f"infer_ms={self._infer_ms:.1f}",
            f"post_ms={self._post_ms:.1f}",
            f"queue_age_ms={self._queue_age_ms:.1f}",
            f"display_fps={self.display_fps:.1f}",
            f"detected_count={self._detected_count}",
            f"stim_refresh_hz={self.config.refresh_rate_hz:.1f}",
            f"tick_jitter_ms={self.tick_jitter_ms:.3f}",
            f"screen_refresh_hz={self.config.screen_refresh_rate_hz:.1f}",
            f"last_frame_age_ms={self._last_frame_age_ms:.1f}",
            f"selected_slot={self.selected_slot or 'None'}",
        ]
        if self._error_text:
            metrics.append("error=see red banner")

        painter.setFont(self._hud_font)
        line_height = 20
        box_width = 420
        box_height = 12 + line_height * len(metrics)
        painter.fillRect(12, 12, box_width, box_height, QColor(0, 0, 0, 180))
        painter.setPen(QColor(255, 255, 255))
        for index, line in enumerate(metrics):
            painter.drawText(24, 32 + index * line_height, line)

        if self._error_text:
            painter.fillRect(12, self.height() - 90, self.width() - 24, 78, QColor(120, 0, 0, 220))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Consolas", 11, QFont.Bold))
            painter.drawText(
                QRect(24, self.height() - 84, self.width() - 48, 66),
                Qt.TextWordWrap,
                self._error_text,
            )

    def paint_overlay(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.Antialiasing, False)
        if self._background_pixmap is None or self._image_rect.isNull():
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Consolas", 18, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for camera frames...")
        else:
            if self._packet is not None:
                self._draw_slots(painter)
                self._draw_roi(painter)
        self._draw_hud(painter)
        self._update_display_fps()

    def _update_display_fps(self) -> None:
        self._paint_counter += 1
        now = time.perf_counter()
        elapsed = now - self._paint_window_start
        if elapsed >= 1.0:
            current_fps = self._paint_counter / elapsed
            self.display_fps = current_fps if self.display_fps <= 0 else (self.display_fps * 0.8 + current_fps * 0.2)
            self._paint_counter = 0
            self._paint_window_start = now

    def _update_packet_fps(self) -> None:
        self._packet_counter += 1
        now = time.perf_counter()
        elapsed = now - self._packet_window_start
        if elapsed >= 1.0:
            current_fps = self._packet_counter / elapsed
            self.packet_fps = current_fps if self.packet_fps <= 0 else (self.packet_fps * 0.8 + current_fps * 0.2)
            self._packet_counter = 0
            self._packet_window_start = now

    def _emit_output(self) -> None:
        if self._packet is None:
            return
        selected_center: Optional[Tuple[int, int]] = None
        slots_payload: List[Dict[str, Any]] = []
        for slot in self._packet.get("slots", []):
            slot_index = int(slot["slot"])
            pixel_center = slot.get("pixel_center")
            if self.selected_slot == slot_index and slot.get("valid") and pixel_center is not None:
                selected_center = (int(pixel_center[0]), int(pixel_center[1]))
            slots_payload.append(
                {
                    "slot": slot_index,
                    "freq_hz": float(slot["freq_hz"]),
                    "valid": bool(slot["valid"]),
                    "observed": bool(slot.get("observed", False)),
                    "pixel_center": safe_int_tuple(slot.get("pixel_center")),
                    "bbox": safe_bbox_list(slot.get("bbox")),
                    "area_px": int(slot.get("area_px", 0)),
                    "confidence": round(float(slot.get("confidence", 0.0)), 6),
                    "age": int(slot.get("age", 0)),
                    "lost_frames": int(slot.get("lost_frames", 0)),
                }
            )

        payload = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame_id": int(self._packet["frame_id"]),
            "image_size": [int(self._packet["image_size"][0]), int(self._packet["image_size"][1])],
            "selected_slot": self.selected_slot or None,
            "selected_center": safe_int_tuple(selected_center),
            "capture_fps": round(self._capture_fps, 3),
            "packet_fps": round(self.packet_fps, 3),
            "queue_age_ms": round(self._queue_age_ms, 3),
            "infer_ms": round(self._infer_ms, 3),
            "post_ms": round(self._post_ms, 3),
            "detected_count": int(self._detected_count),
            "slots": slots_payload,
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()


class OverlayWidget(QWidget):
    def __init__(self, controller: VisionWindow) -> None:
        super().__init__(controller)
        self.controller = controller
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.NoFocus)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        self.controller.paint_overlay(painter)


def apply_screen_refresh_guard(config: AppConfig, app: QApplication) -> None:
    screen = app.primaryScreen()
    if screen is None:
        return
    try:
        screen_refresh = float(screen.refreshRate())
    except Exception:
        return
    config.screen_refresh_rate_hz = max(0.0, screen_refresh)
    if screen_refresh > 1.0 and screen_refresh + 0.5 < config.requested_refresh_rate_hz:
        log_stderr(
            f"Requested refresh rate {config.requested_refresh_rate_hz:.1f}Hz exceeds screen refresh "
            f"{screen_refresh:.1f}Hz; using {screen_refresh:.1f}Hz instead."
        )
        config.refresh_rate_hz = screen_refresh


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        config = load_app_config(argv)
    except Exception as exc:
        log_stderr(str(exc))
        return 1

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    apply_screen_refresh_guard(config, app)
    log_stderr(
        f"Starting viewer with source={config.source!r}, weights='{config.weights}', "
        f"device='{config.device}', half={config.half}, refresh_rate_hz={config.refresh_rate_hz:.1f}"
    )

    camera = CameraLoader(config.source).start()
    worker = InferenceWorker(camera, config)
    stim_clock = StimClockThread(config.refresh_rate_hz)
    window = VisionWindow(config, camera)
    window.resize(*DEFAULT_WINDOW_SIZE)

    worker.packet_ready.connect(window.on_vision_packet, type=Qt.QueuedConnection)
    worker.error.connect(window.on_worker_error, type=Qt.QueuedConnection)
    stim_clock.tick.connect(window.on_stim_tick, type=Qt.QueuedConnection)

    worker.start()
    stim_clock.start(QThread.TimeCriticalPriority)

    if config.fullscreen:
        window.showFullScreen()
    else:
        window.show()

    exit_code = 0
    try:
        exit_code = app.exec_()
    finally:
        stim_clock.request_stop()
        stim_clock.wait(2000)
        worker.request_stop()
        worker.wait(2000)
        camera.stop()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
