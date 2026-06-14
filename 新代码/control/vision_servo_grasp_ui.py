"""Simple JetMax vision-servo grasp debug UI.

Run this file in VS Code:

    conda activate brain_robot
    cd E:\\brain_control\\code\\control
    python .\vision_servo_grasp_ui.py

The UI is intentionally small: video, wood-block detection, flicker labels,
camera-centering servo, and pick/place/reset controls.
"""

from __future__ import annotations

import math
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import roslibpy
except ImportError:
    roslibpy = None

try:
    import paramiko
except ImportError:
    paramiko = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "192.168.149.1"
DEFAULT_MODEL = ROOT / "best_seg.pt"
VIDEO_TOPIC = "/usb_cam/image_rect_color"
MAX_SLOTS = 4
FREQUENCIES_HZ = [8.0, 10.0, 12.0, 15.0]
DEFAULT_PICK_FORWARD_OFFSET_MM = 42.0
DEFAULT_PICK_LOW_Z_MM = 91.0
KEYBOARD_THETA_STEP_DEG = 3.0
KEYBOARD_RADIUS_STEP_MM = 10.0
KEYBOARD_Z_STEP_MM = 10.0
KEYBOARD_MOVE_INTERVAL_SEC = 0.25
SERVO_COMMAND_INTERVAL_SEC = 0.18
DEFAULT_MATCH_DISTANCE = 120.0
DEFAULT_LOST_TTL = 6
MORPH_KERNEL_SIZE = 3
LOCKED_TARGET_MAX_JUMP_PX = 85.0
LOCKED_TARGET_MIN_IOU = 0.08


@dataclass(frozen=True)
class Target:
    target_id: int
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    angle_deg: float
    score: float
    source: str
    frequency_hz: float
    polygon: list[tuple[int, int]]
    observed: bool = True


@dataclass(frozen=True)
class DetectionCandidate:
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    area_px: int
    confidence: float
    polygon: list[tuple[int, int]]
    source: str
    angle_deg: float = 0.0


@dataclass
class SlotState:
    slot: int
    freq_hz: float
    valid: bool = False
    observed: bool = False
    center: tuple[int, int] | None = None
    bbox: tuple[int, int, int, int] | None = None
    area_px: int = 0
    confidence: float = 0.0
    polygon: list[tuple[int, int]] = field(default_factory=list)
    source: str = ""
    angle_deg: float = 0.0
    age: int = 0
    lost_frames: int = 0

    def assign(self, candidate: DetectionCandidate, *, increment_age: bool) -> None:
        self.valid = True
        self.observed = True
        self.center = candidate.center
        self.bbox = candidate.bbox
        self.area_px = int(candidate.area_px)
        self.confidence = float(candidate.confidence)
        self.polygon = list(candidate.polygon)
        self.source = str(candidate.source)
        self.angle_deg = float(candidate.angle_deg)
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
        self.center = None
        self.bbox = None
        self.area_px = 0
        self.confidence = 0.0
        self.polygon = []
        self.source = ""
        self.angle_deg = 0.0
        self.age = 0
        self.lost_frames = 0


def tcp_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def bbox_iou(
    box_a: tuple[int, int, int, int] | None,
    box_b: tuple[int, int, int, int] | None,
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


def point_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1]))


def largest_component(binary_mask: np.ndarray) -> tuple[np.ndarray | None, int]:
    if binary_mask.ndim != 2:
        return None, 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return None, 0
    component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component_index, cv2.CC_STAT_AREA])
    component = np.where(labels == component_index, 255, 0).astype(np.uint8)
    return component, area


def mask_to_candidate(
    mask: np.ndarray,
    frame_shape: tuple[int, int],
    *,
    confidence: float,
    source: str,
) -> DetectionCandidate | None:
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
    moments = cv2.moments(component, binaryImage=True)
    if moments["m00"] <= 0:
        return None
    cx = int(round(moments["m10"] / moments["m00"]))
    cy = int(round(moments["m01"] / moments["m00"]))
    x, y, w, h = cv2.boundingRect(contour)
    epsilon = max(1.0, 0.004 * cv2.arcLength(contour, True))
    polygon_contour = cv2.approxPolyDP(contour, epsilon, True)
    polygon = [(int(point[0][0]), int(point[0][1])) for point in polygon_contour]
    if len(polygon) < 3:
        return None
    rect = cv2.minAreaRect(contour)
    return DetectionCandidate(
        center=(cx, cy),
        bbox=(int(x), int(y), int(x + w), int(y + h)),
        area_px=int(area_px),
        confidence=float(confidence),
        polygon=polygon,
        source=str(source),
        angle_deg=float(rect[2]),
    )


def build_stream_url(host: str) -> str:
    return (
        "http://{host}:8080/stream?topic={topic}&type=mjpeg"
        "&width=640&height=480&quality=80"
    ).format(host=host, topic=VIDEO_TOPIC)


def bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class VideoThread(QThread):
    frame_ready = pyqtSignal(object)
    status = pyqtSignal(str)

    def __init__(self, host: str) -> None:
        super().__init__()
        self.host = host
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        url = build_stream_url(self.host)
        self.status.emit("Opening video: " + url)
        while not self._stop.is_set():
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                self.status.emit("Video open failed, retrying...")
                cap.release()
                time.sleep(1.0)
                continue
            self.status.emit("Video connected.")
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.status.emit("Video frame lost, reconnecting...")
                    break
                self.frame_ready.emit(frame)
                time.sleep(0.001)
            cap.release()
            time.sleep(0.5)


class Detector:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_loaded = False
        self.slots = [SlotState(slot=index + 1, freq_hz=FREQUENCIES_HZ[index]) for index in range(MAX_SLOTS)]

    def load_model(self) -> None:
        if self.model_loaded:
            return
        self.model_loaded = True
        if YOLO is not None and DEFAULT_MODEL.exists():
            self.model = YOLO(str(DEFAULT_MODEL))

    def detect(self, frame: np.ndarray, *, use_yolo: bool, min_area: int) -> list[Target]:
        candidates: list[DetectionCandidate] = []
        if use_yolo:
            self.load_model()
            if self.model is not None:
                candidates.extend(self._detect_yolo(frame))
        candidates.extend(self._detect_contours(frame, min_area=max(80, int(min_area))))
        candidates = self._dedupe_candidates(candidates)
        self._update_slots(candidates)
        return self._targets_from_slots()

    def _detect_yolo(self, frame: np.ndarray) -> list[DetectionCandidate]:
        result = self.model.predict(frame, imgsz=512, conf=0.25, iou=0.5, max_det=8, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        candidates: list[DetectionCandidate] = []
        if boxes is None or boxes.conf is None:
            return candidates
        confidences = boxes.conf.detach().cpu().numpy()
        frame_h, frame_w = frame.shape[:2]
        if masks is not None and masks.data is not None:
            mask_data = masks.data.detach().cpu().numpy()
            count = min(len(confidences), len(mask_data))
            for index in range(count):
                candidate = mask_to_candidate(
                    mask_data[index],
                    (frame_h, frame_w),
                    confidence=float(confidences[index]),
                    source="yolo",
                )
                if candidate is not None:
                    candidates.append(candidate)
            return candidates

        if boxes.xyxy is None:
            return candidates
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
        for index, box in enumerate(xyxy):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(frame_w - 1, x1))
            x2 = max(0, min(frame_w - 1, x2))
            y1 = max(0, min(frame_h - 1, y1))
            y2 = max(0, min(frame_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            candidates.append(
                DetectionCandidate(
                    center=(int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))),
                    bbox=(x1, y1, x2, y2),
                    area_px=int((x2 - x1) * (y2 - y1)),
                    confidence=float(confidences[index]),
                    polygon=polygon,
                    source="yolo-box",
                    angle_deg=0.0,
                )
            )
        return candidates

    def _detect_contours(self, frame: np.ndarray, *, min_area: int) -> list[DetectionCandidate]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_h, frame_w = frame.shape[:2]
        hue_ranges = [
            (np.array([0, 45, 35]), np.array([12, 255, 255])),      # red/orange
            (np.array([170, 45, 35]), np.array([179, 255, 255])),   # red wrap
            (np.array([12, 35, 35]), np.array([38, 255, 255])),     # yellow/brown top
            (np.array([38, 40, 30]), np.array([95, 255, 255])),     # green/cyan
            (np.array([90, 35, 30]), np.array([145, 255, 255])),    # blue/purple
        ]
        kernel = np.ones((5, 5), np.uint8)
        contours: list[np.ndarray] = []
        for low, high in hue_ranges:
            mask = cv2.inRange(hsv, low, high)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(found)
        raw: list[DetectionCandidate] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < float(min_area):
                continue
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            if rw <= 4 or rh <= 4:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if area > 0.35 * float(frame_w * frame_h):
                continue
            touches_border = x <= 2 or y <= 2 or (x + w) >= frame_w - 2 or (y + h) >= frame_h - 2
            if touches_border and area > 0.08 * float(frame_w * frame_h):
                continue
            if w < 18 or h < 18:
                continue
            aspect = max(float(w) / max(1.0, float(h)), float(h) / max(1.0, float(w)))
            if aspect > 2.6:
                continue
            fill_ratio = area / max(1.0, float(w * h))
            if fill_ratio < 0.18:
                continue
            epsilon = max(1.0, 0.006 * cv2.arcLength(contour, True))
            polygon_contour = cv2.approxPolyDP(contour, epsilon, True)
            polygon = [(int(point[0][0]), int(point[0][1])) for point in polygon_contour]
            if len(polygon) < 3:
                continue
            raw.append(
                DetectionCandidate(
                    center=(int(round(cx)), int(round(cy))),
                    bbox=(x, y, x + w, y + h),
                    area_px=int(area),
                    confidence=min(1.0, area / 20000.0),
                    polygon=polygon,
                    source="contour",
                    angle_deg=float(angle),
                )
            )
        return sorted(raw, key=lambda t: (-t.area_px, t.center[1], t.center[0]))[:8]

    def _dedupe_candidates(self, candidates: list[DetectionCandidate]) -> list[DetectionCandidate]:
        ordered = sorted(
            candidates,
            key=lambda item: (0 if item.source.startswith("yolo") else 1, -item.confidence, -item.area_px),
        )
        kept: list[DetectionCandidate] = []
        for candidate in ordered:
            duplicate = False
            for existing in kept:
                if bbox_iou(candidate.bbox, existing.bbox) >= 0.25:
                    duplicate = True
                    break
                if point_distance(candidate.center, existing.center) <= 35.0:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        kept.sort(key=lambda item: (-item.area_px, item.center[1], item.center[0]))
        return kept[:8]

    def _update_slots(self, candidates: list[DetectionCandidate]) -> None:
        matched_slots: set[int] = set()
        matched_candidates: set[int] = set()
        pairs: list[tuple[float, int, int]] = []

        for slot_index, slot in enumerate(self.slots):
            if not slot.valid or slot.center is None:
                continue
            for candidate_index, candidate in enumerate(candidates):
                distance = point_distance(slot.center, candidate.center)
                overlap = bbox_iou(slot.bbox, candidate.bbox)
                if distance > DEFAULT_MATCH_DISTANCE and overlap <= 0.05:
                    continue
                score = (distance / DEFAULT_MATCH_DISTANCE) + (1.0 - overlap) * 0.35
                pairs.append((score, slot_index, candidate_index))

        pairs.sort()
        for _score, slot_index, candidate_index in pairs:
            if slot_index in matched_slots or candidate_index in matched_candidates:
                continue
            self.slots[slot_index].assign(candidates[candidate_index], increment_age=True)
            matched_slots.add(slot_index)
            matched_candidates.add(candidate_index)

        for slot_index, slot in enumerate(self.slots):
            if slot_index in matched_slots or not slot.valid:
                continue
            slot.mark_missing()
            if slot.lost_frames > DEFAULT_LOST_TTL:
                slot.clear()

        remaining = [candidate for i, candidate in enumerate(candidates) if i not in matched_candidates]
        free_slots = [slot for slot in self.slots if not slot.valid]
        for slot, candidate in zip(free_slots, remaining):
            slot.assign(candidate, increment_age=False)

    def _targets_from_slots(self) -> list[Target]:
        targets: list[Target] = []
        for slot in self.slots:
            if not slot.valid or slot.center is None or slot.bbox is None:
                continue
            targets.append(
                Target(
                    target_id=int(slot.slot),
                    center=(float(slot.center[0]), float(slot.center[1])),
                    bbox=slot.bbox,
                    angle_deg=float(slot.angle_deg),
                    score=float(slot.confidence),
                    source=str(slot.source),
                    frequency_hz=float(slot.freq_hz),
                    polygon=list(slot.polygon),
                    observed=bool(slot.observed),
                )
            )
        return targets


class RobotBridge:
    def __init__(self, host: str) -> None:
        if roslibpy is None:
            raise RuntimeError("roslibpy is not installed. Run: pip install roslibpy")
        self.host = host
        self.ros: Any | None = None
        self.state: dict[str, Any] = {}
        self.state_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def connect(self) -> None:
        self.ros = roslibpy.Ros(host=self.host, port=9091)
        self.ros.run(timeout=8)
        if not self.ros.is_connected:
            raise RuntimeError("Failed to connect rosbridge ws://{}:9091".format(self.host))
        topic = roslibpy.Topic(self.ros, "/hybrid_controller/state", "hybrid_controller_ros/RobotState")
        topic.subscribe(self._on_state)

    def close(self) -> None:
        if self.ros is not None:
            self.ros.close()
            self.ros = None

    def _on_state(self, message: dict[str, Any]) -> None:
        self.state = dict(message)
        try:
            self.state_queue.put_nowait(dict(message))
        except queue.Full:
            pass

    def service(self, name: str, service_type: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.ros is None:
            raise RuntimeError("ROS is not connected.")
        service = roslibpy.Service(self.ros, name, service_type)
        return service.call(roslibpy.ServiceRequest(args or {}), timeout=20)

    def reset(self) -> dict[str, Any]:
        return self.service("/hybrid_controller/reset", "std_srvs/Trigger")

    def abort(self) -> dict[str, Any]:
        return self.service("/hybrid_controller/abort", "std_srvs/Trigger")

    def sucker_off(self) -> dict[str, Any]:
        return self.service("/hybrid_controller/sucker_off", "std_srvs/Trigger")

    def pick_here(self) -> dict[str, Any]:
        return self.service("/hybrid_controller/pick_here", "std_srvs/Trigger")

    def place(self) -> dict[str, Any]:
        return self.service("/hybrid_controller/place", "std_srvs/Trigger")

    def move_cyl_auto(self, theta_deg: float, radius_mm: float) -> dict[str, Any]:
        return self.service(
            "/hybrid_controller/move_cyl_auto",
            "hybrid_controller_ros/MoveCylAuto",
            {"theta_deg": float(theta_deg), "radius_mm": float(radius_mm)},
        )

    def move_cyl(self, theta_deg: float, radius_mm: float, z_mm: float) -> dict[str, Any]:
        return self.service(
            "/hybrid_controller/move_cyl",
            "hybrid_controller_ros/MoveCyl",
            {
                "theta_deg": float(theta_deg),
                "radius_mm": float(radius_mm),
                "z_mm": float(z_mm),
            },
        )

    def get_pick_tuning(self) -> dict[str, Any]:
        return self.service("/hybrid_controller/get_pick_tuning", "hybrid_controller_ros/GetPickTuning")

    def set_pick_tuning(self, tuning: dict[str, Any]) -> dict[str, Any]:
        return self.service("/hybrid_controller/set_pick_tuning", "hybrid_controller_ros/SetPickTuning", tuning)

    def pick_cyl(self, theta_deg: float, radius_mm: float, sucker_angle_deg: float = 0.0) -> dict[str, Any]:
        return self.service(
            "/hybrid_controller/pick_cyl",
            "hybrid_controller_ros/PickCyl",
            {
                "theta_deg": float(theta_deg),
                "radius_mm": float(radius_mm),
                "use_sucker_rotation": False,
                "sucker_rotation_deg": float(sucker_angle_deg),
            },
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("JetMax Vision Servo Grasp Debug")
        self.resize(1120, 760)
        self.setFocusPolicy(Qt.StrongFocus)

        self.host = DEFAULT_HOST
        self.video_thread: VideoThread | None = None
        self.robot: RobotBridge | None = None
        self.detector = Detector()
        self.frame: np.ndarray | None = None
        self.targets: list[Target] = []
        self.selected_id = 1
        self.servo_target_id: int | None = None
        self.locked_servo_target: Target | None = None
        self.servo_running = False
        self.center_then_pick = False
        self.hide_target_overlay = False
        self.busy_command = False
        self.last_command_ts = 0.0
        self.last_keyboard_move_ts = 0.0
        self.stable_center_frames = 0
        self.stim_start_ts = time.perf_counter()
        self.last_detect_ts = 0.0

        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.video_label.setMinimumSize(720, 540)
        self.video_label.setStyleSheet("background:#111; color:#ddd;")
        self.video_label.setText("No video")

        self.status_label = QLabel("Ready")
        self.state_label = QLabel("Robot: disconnected")
        self.target_label = QLabel("Target: none")

        self.host_box = QComboBox()
        self.host_box.setEditable(True)
        self.host_box.addItem(DEFAULT_HOST)

        self.target_box = QComboBox()
        self.target_box.addItem("1")
        self.target_box.currentTextChanged.connect(self._target_changed)

        self.use_yolo = QCheckBox("YOLO")
        self.use_yolo.setChecked(DEFAULT_MODEL.exists() and YOLO is not None)
        self.flicker_enabled = QCheckBox("Flicker")
        self.flicker_enabled.setChecked(True)
        self.invert_theta = QCheckBox("Invert theta")
        self.invert_radius = QCheckBox("Invert radius")

        self.min_area = self._spin_int(120, 50000, 900)
        self.tolerance_px = self._spin_int(4, 120, 28)
        self.theta_gain = self._spin_float(0.001, 0.20, 0.022, 3)
        self.radius_gain = self._spin_float(0.001, 0.80, 0.085, 3)
        self.max_theta_step = self._spin_float(0.2, 15.0, 3.0, 1)
        self.max_radius_step = self._spin_float(1.0, 40.0, 8.0, 1)
        self.low_forward_enabled = QCheckBox("Low forward")
        self.low_forward_enabled.setChecked(True)
        self.pick_forward_offset = self._spin_float(-80.0, 120.0, DEFAULT_PICK_FORWARD_OFFSET_MM, 1)
        self.pick_forward_offset.setSingleStep(5.0)
        self.pick_forward_offset.setSuffix(" mm")
        self.descend_z = self._spin_float(80.0, 160.0, DEFAULT_PICK_LOW_Z_MM, 1)
        self.descend_z.setSingleStep(5.0)
        self.descend_z.setSuffix(" mm")

        self.connect_btn = QPushButton("Connect")
        self.video_btn = QPushButton("Start Video")
        self.camera_reset_btn = QPushButton("Reset Camera")
        self.center_btn = QPushButton("Start Center")
        self.step_btn = QPushButton("Servo Step")
        self.pick_btn = QPushButton("Pick")
        self.center_pick_btn = QPushButton("Center + Pick")
        self.reset_btn = QPushButton("Reset")
        self.abort_btn = QPushButton("Abort")
        self.sucker_btn = QPushButton("Sucker Off")
        self.place_btn = QPushButton("Place")

        self.connect_btn.clicked.connect(self.connect_robot)
        self.video_btn.clicked.connect(self.toggle_video)
        self.camera_reset_btn.clicked.connect(self.reset_camera)
        self.center_btn.clicked.connect(self.toggle_servo)
        self.step_btn.clicked.connect(self.servo_step)
        self.pick_btn.clicked.connect(self.pick_selected)
        self.center_pick_btn.clicked.connect(self.center_and_pick)
        self.reset_btn.clicked.connect(lambda: self.call_robot("reset"))
        self.abort_btn.clicked.connect(lambda: self.call_robot("abort"))
        self.sucker_btn.clicked.connect(lambda: self.call_robot("sucker_off"))
        self.place_btn.clicked.connect(self.place_selected)

        root = QWidget()
        outer = QHBoxLayout(root)
        outer.addWidget(self.video_label, stretch=1)
        outer.addWidget(self._side_panel(), stretch=0)
        self.setCentralWidget(root)

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.render)
        self.render_timer.start(4)

        self.servo_timer = QTimer(self)
        self.servo_timer.timeout.connect(self._servo_tick)
        self.servo_timer.start(120)

        self.state_timer = QTimer(self)
        self.state_timer.timeout.connect(self._poll_robot_state)
        self.state_timer.start(300)

        self._install_keyboard_shortcuts()

    def _side_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        connection = QGroupBox("Connection")
        grid = QGridLayout(connection)
        grid.addWidget(QLabel("JetMax IP"), 0, 0)
        grid.addWidget(self.host_box, 0, 1)
        grid.addWidget(self.connect_btn, 1, 0)
        grid.addWidget(self.video_btn, 1, 1)
        grid.addWidget(self.camera_reset_btn, 2, 0, 1, 2)
        layout.addWidget(connection)

        detect = QGroupBox("Detection")
        grid = QGridLayout(detect)
        grid.addWidget(QLabel("Target"), 0, 0)
        grid.addWidget(self.target_box, 0, 1)
        grid.addWidget(self.use_yolo, 1, 0)
        grid.addWidget(self.flicker_enabled, 1, 1)
        grid.addWidget(QLabel("Min area"), 2, 0)
        grid.addWidget(self.min_area, 2, 1)
        layout.addWidget(detect)

        servo = QGroupBox("Servo")
        grid = QGridLayout(servo)
        grid.addWidget(QLabel("Tolerance px"), 0, 0)
        grid.addWidget(self.tolerance_px, 0, 1)
        grid.addWidget(QLabel("Theta gain"), 1, 0)
        grid.addWidget(self.theta_gain, 1, 1)
        grid.addWidget(QLabel("Radius gain"), 2, 0)
        grid.addWidget(self.radius_gain, 2, 1)
        grid.addWidget(QLabel("Max theta"), 3, 0)
        grid.addWidget(self.max_theta_step, 3, 1)
        grid.addWidget(QLabel("Max radius"), 4, 0)
        grid.addWidget(self.max_radius_step, 4, 1)
        grid.addWidget(self.invert_theta, 5, 0)
        grid.addWidget(self.invert_radius, 5, 1)
        grid.addWidget(self.step_btn, 6, 0)
        grid.addWidget(self.center_btn, 6, 1)
        layout.addWidget(servo)

        actions = QGroupBox("Actions")
        grid = QGridLayout(actions)
        grid.addWidget(self.low_forward_enabled, 0, 0)
        grid.addWidget(QLabel("Forward"), 0, 1)
        grid.addWidget(self.pick_forward_offset, 1, 0, 1, 2)
        grid.addWidget(QLabel("Descend Z"), 2, 0)
        grid.addWidget(self.descend_z, 2, 1)
        grid.addWidget(self.pick_btn, 3, 0)
        grid.addWidget(self.center_pick_btn, 3, 1)
        grid.addWidget(self.reset_btn, 4, 0)
        grid.addWidget(self.abort_btn, 4, 1)
        grid.addWidget(self.sucker_btn, 5, 0)
        grid.addWidget(self.place_btn, 5, 1)
        layout.addWidget(actions)

        layout.addWidget(self.state_label)
        layout.addWidget(self.target_label)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        return panel

    def _spin_int(self, low: int, high: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        return spin

    def _spin_float(self, low: float, high: float, value: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        spin.setDecimals(decimals)
        spin.setSingleStep(10 ** (-decimals))
        return spin

    def _install_keyboard_shortcuts(self) -> None:
        bindings = {
            "A": (-KEYBOARD_THETA_STEP_DEG, 0.0, 0.0, "left"),
            "D": (KEYBOARD_THETA_STEP_DEG, 0.0, 0.0, "right"),
            "W": (0.0, KEYBOARD_RADIUS_STEP_MM, 0.0, "forward"),
            "S": (0.0, -KEYBOARD_RADIUS_STEP_MM, 0.0, "back"),
            "Q": (0.0, 0.0, KEYBOARD_Z_STEP_MM, "up"),
            "E": (0.0, 0.0, -KEYBOARD_Z_STEP_MM, "down"),
        }
        for key, (dtheta, dradius, dz, label) in bindings.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.setAutoRepeat(True)
            shortcut.activated.connect(
                lambda dtheta=dtheta, dradius=dradius, dz=dz, label=label: self.keyboard_move(
                    dtheta,
                    dradius,
                    dz,
                    label,
                )
            )

    def _target_changed(self, text: str) -> None:
        try:
            self.selected_id = int(text)
        except ValueError:
            self.selected_id = 1
        if not self.servo_running and not self.hide_target_overlay:
            self._clear_servo_lock()

    def connect_robot(self) -> None:
        self.host = self.host_box.currentText().strip() or DEFAULT_HOST
        if not tcp_open(self.host, 9091):
            self.set_status("rosbridge 9091 is not open. Start JetMax runtime first.")
            QMessageBox.warning(self, "ROS not reachable", "JetMax rosbridge 9091 is not open.")
            return
        try:
            if self.robot is not None:
                self.robot.close()
            self.robot = RobotBridge(self.host)
            self.robot.connect()
            self.set_status("Robot connected.")
        except Exception as exc:
            self.set_status("Robot connect failed: " + str(exc))
            QMessageBox.warning(self, "Connect failed", str(exc))

    def toggle_video(self) -> None:
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait(1500)
            self.video_thread = None
            self.video_btn.setText("Start Video")
            self.set_status("Video stopped.")
            return
        self.host = self.host_box.currentText().strip() or DEFAULT_HOST
        self.video_thread = VideoThread(self.host)
        self.video_thread.frame_ready.connect(self._on_frame)
        self.video_thread.status.connect(self.set_status)
        self.video_thread.start()
        self.video_btn.setText("Stop Video")

    def reset_camera(self) -> None:
        if paramiko is None:
            QMessageBox.warning(self, "Missing dependency", "paramiko is not installed.")
            return
        self.host = self.host_box.currentText().strip() or DEFAULT_HOST
        self.set_status("Resetting JetMax camera...")
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self.host, username="hiwonder", password="hiwonder", timeout=8)
            script = r"""#!/bin/bash
set -e
cp /home/hiwonder/ros/autostart/usb_cam.launch /home/hiwonder/ros/autostart/usb_cam.launch.bak-ui-$(date +%Y%m%d-%H%M%S)
python3 - <<'PY'
from pathlib import Path
import re
p = Path('/home/hiwonder/ros/autostart/usb_cam.launch')
s = p.read_text()
s = re.sub(r'<param name="video_device" value="[^"]+" />', '<param name="video_device" value="/dev/usb_cam0" />', s)
s = re.sub(r'<param name="framerate" value="[^"]+" />', '<param name="framerate" value="20" />', s)
s = re.sub(r'<param name="io_method" value="[^"]+" />', '<param name="io_method" value="mmap" />', s)
p.write_text(s)
PY
echo hiwonder | sudo -S systemctl stop usb_cam.service
sleep 2
echo hiwonder | sudo -S modprobe -r uvcvideo || true
sleep 2
echo hiwonder | sudo -S modprobe uvcvideo quirks=128 nodrop=1 timeout=5000 || true
sleep 4
echo hiwonder | sudo -S sh -c 'echo -1 > /sys/module/usbcore/parameters/autosuspend' || true
echo hiwonder | sudo -S sh -c 'for p in /sys/bus/usb/devices/*/power/control; do echo on > $p 2>/dev/null || true; done'
echo hiwonder | sudo -S sh -c 'for p in /sys/bus/usb/devices/*/power/autosuspend_delay_ms; do echo -1 > $p 2>/dev/null || true; done'
echo hiwonder | sudo -S sh -c 'echo 0 > /sys/bus/usb/devices/1-2.2/authorized' || true
sleep 3
echo hiwonder | sudo -S sh -c 'echo 1 > /sys/bus/usb/devices/1-2.2/authorized' || true
sleep 5
echo hiwonder | sudo -S sh -c 'echo on > /sys/bus/usb/devices/1-2.2/power/control' || true
echo hiwonder | sudo -S sh -c 'echo -1 > /sys/bus/usb/devices/1-2.2/power/autosuspend_delay_ms' || true
timeout 8 v4l2-ctl --device=/dev/usb_cam0 --set-fmt-video=width=640,height=480,pixelformat=YUYV --set-parm=20 || true
timeout 10 v4l2-ctl --device=/dev/usb_cam0 --stream-mmap --stream-count=3 || true
echo hiwonder | sudo -S systemctl start usb_cam.service
sleep 8
source /opt/ros/melodic/setup.bash
timeout 6 rostopic hz /usb_cam/image_rect_color || true
"""
            sftp = client.open_sftp()
            with sftp.file("/tmp/reset_camera_from_ui.sh", "w") as handle:
                handle.write(script)
            sftp.chmod("/tmp/reset_camera_from_ui.sh", 0o755)
            sftp.close()
            _, stdout, stderr = client.exec_command("bash /tmp/reset_camera_from_ui.sh", timeout=120)
            out = stdout.read().decode("utf-8", "replace") + stderr.read().decode("utf-8", "replace")
            client.close()
            if "average rate" in out:
                self.set_status("Camera reset OK.")
                if self.video_thread is None:
                    self.toggle_video()
            else:
                self.set_status("Camera reset ran, but no frame rate confirmed.")
        except Exception as exc:
            self.set_status("Camera reset failed: " + str(exc))

    def _on_frame(self, frame: object) -> None:
        self.frame = np.asarray(frame)

    def _poll_robot_state(self) -> None:
        if self.robot is None:
            return
        while True:
            try:
                state = self.robot.state_queue.get_nowait()
            except queue.Empty:
                break
            theta = float(state.get("theta_deg", 0.0))
            radius = float(state.get("radius_mm", 0.0))
            z = float(state.get("z_mm", 0.0))
            self.state_label.setText(
                "Robot: {state}  theta={theta:.1f}  r={radius:.1f}  z={z:.1f}".format(
                    state=state.get("state", "?"),
                    theta=theta,
                    radius=radius,
                    z=z,
                )
            )

    def render(self) -> None:
        if self.frame is None:
            return
        frame = self.frame.copy()
        now = time.perf_counter()
        if (now - self.last_detect_ts) >= 0.10:
            self.targets = self.detector.detect(
                frame,
                use_yolo=self.use_yolo.isChecked(),
                min_area=int(self.min_area.value()),
            )
            self.last_detect_ts = now
            self._refresh_locked_servo_target()
            self._update_target_choices()
        self._draw_overlay(frame)
        self.video_label.setPixmap(
            bgr_to_pixmap(frame).scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _update_target_choices(self) -> None:
        ids = [str(t.target_id) for t in self.targets]
        current = str(self.selected_id)
        if self.servo_target_id is not None:
            locked = str(self.servo_target_id)
            if [self.target_box.itemText(i) for i in range(self.target_box.count())] != ids:
                self.target_box.blockSignals(True)
                self.target_box.clear()
                self.target_box.addItems(ids or [locked])
                if locked not in ids:
                    self.target_box.addItem(locked)
                index = self.target_box.findText(locked)
                if index >= 0:
                    self.target_box.setCurrentIndex(index)
                self.target_box.blockSignals(False)
            self.selected_id = int(self.servo_target_id)
            return
        if [self.target_box.itemText(i) for i in range(self.target_box.count())] == ids:
            return
        self.target_box.blockSignals(True)
        self.target_box.clear()
        self.target_box.addItems(ids or ["1"])
        index = ids.index(current) if current in ids else 0
        self.target_box.setCurrentIndex(index)
        self.target_box.blockSignals(False)
        self._target_changed(self.target_box.currentText())

    def _draw_overlay(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        cv2.drawMarker(frame, center, (255, 255, 255), cv2.MARKER_CROSS, 28, 2)
        selected = self.servo_target() if self.servo_target_id is not None else self.selected_target()
        current_t = time.perf_counter() - self.stim_start_ts
        hide_target_overlay = bool(self.hide_target_overlay)
        if self.flicker_enabled.isChecked() and not hide_target_overlay:
            overlay = frame.copy()
            for target in self.targets:
                if len(target.polygon) < 3:
                    continue
                luminance = clamp01(0.5 + 0.5 * math.sin(2.0 * math.pi * target.frequency_hz * current_t))
                gray = int(round(255.0 * luminance))
                points = np.array(target.polygon, dtype=np.int32)
                cv2.fillPoly(overlay, [points], (gray, gray, gray))
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, dst=frame)

        if not hide_target_overlay:
            for target in self.targets:
                x1, y1, x2, y2 = target.bbox
                color = (0, 220, 255)
                if selected is not None and target.target_id == selected.target_id:
                    color = (0, 80, 255)
                if not target.observed:
                    color = (0, 180, 180)
                if len(target.polygon) >= 3:
                    cv2.polylines(frame, [np.array(target.polygon, dtype=np.int32)], True, color, 3)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cx, cy = target.center
                cv2.circle(frame, (int(cx), int(cy)), 5, color, -1)
                cv2.putText(
                    frame,
                    "#{0} {1:g}Hz {2}".format(target.target_id, target.frequency_hz, target.source),
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        if selected is None:
            self.target_label.setText("Target: none")
        else:
            err_x = selected.center[0] - center[0]
            err_y = selected.center[1] - center[1]
            dist = math.hypot(err_x, err_y)
            cv2.line(frame, center, (int(selected.center[0]), int(selected.center[1])), (255, 255, 0), 2)
            if hide_target_overlay:
                cv2.circle(frame, (int(selected.center[0]), int(selected.center[1])), 5, (255, 255, 0), -1)
            self.target_label.setText(
                "Target #{0}: err=({1:.0f},{2:.0f}) dist={3:.1f}px freq={4:g}Hz".format(
                    selected.target_id,
                    err_x,
                    err_y,
                    dist,
                    selected.frequency_hz,
                )
            )

    def selected_target(self, *, allow_fallback: bool = True) -> Target | None:
        for target in self.targets:
            if target.target_id == self.selected_id:
                return target
        return self.targets[0] if allow_fallback and self.targets else None

    def servo_target(self) -> Target | None:
        if self.locked_servo_target is not None:
            return self.locked_servo_target
        if self.servo_target_id is None:
            return self.selected_target()
        for target in self.targets:
            if target.target_id == self.servo_target_id:
                return target
        return None

    def _target_with_locked_id(self, target: Target, locked_id: int) -> Target:
        return Target(
            target_id=int(locked_id),
            center=target.center,
            bbox=target.bbox,
            angle_deg=target.angle_deg,
            score=target.score,
            source=target.source,
            frequency_hz=target.frequency_hz,
            polygon=list(target.polygon),
            observed=target.observed,
        )

    def _matches_locked_target(self, target: Target, locked: Target) -> bool:
        center_jump = point_distance(target.center, locked.center)
        overlap = bbox_iou(target.bbox, locked.bbox)
        return center_jump <= LOCKED_TARGET_MAX_JUMP_PX or overlap >= LOCKED_TARGET_MIN_IOU

    def _refresh_locked_servo_target(self) -> None:
        locked = self.locked_servo_target
        locked_id = self.servo_target_id
        if locked is None or locked_id is None or not self.targets:
            return

        same_id = [target for target in self.targets if target.target_id == locked_id]
        for target in same_id:
            if self._matches_locked_target(target, locked):
                self.locked_servo_target = self._target_with_locked_id(target, locked_id)
                return

        nearest = min(self.targets, key=lambda target: point_distance(target.center, locked.center))
        if self._matches_locked_target(nearest, locked):
            self.locked_servo_target = self._target_with_locked_id(nearest, locked_id)

    def _clear_servo_lock(self) -> None:
        self.servo_target_id = None
        self.locked_servo_target = None

    def _lock_servo_target(self) -> bool:
        target = self.selected_target()
        if target is None:
            self.set_status("No target to center.")
            return False
        self.servo_target_id = int(target.target_id)
        self.locked_servo_target = target
        self.selected_id = int(target.target_id)
        index = self.target_box.findText(str(target.target_id))
        if index >= 0:
            self.target_box.blockSignals(True)
            self.target_box.setCurrentIndex(index)
            self.target_box.blockSignals(False)
        return True

    def _finish_centering(self) -> None:
        locked_id = self.servo_target_id
        self.servo_running = False
        self.center_btn.setText("Start Center")
        self.flicker_enabled.setChecked(False)
        if locked_id is not None:
            self.selected_id = int(locked_id)
            index = self.target_box.findText(str(locked_id))
            if index >= 0:
                self.target_box.blockSignals(True)
                self.target_box.setCurrentIndex(index)
                self.target_box.blockSignals(False)

    def toggle_servo(self) -> None:
        if not self.servo_running and not self._lock_servo_target():
            return
        self.servo_running = not self.servo_running
        self.center_then_pick = False
        self.stable_center_frames = 0
        if not self.servo_running:
            self._clear_servo_lock()
            self.hide_target_overlay = False
        self.center_btn.setText("Stop Center" if self.servo_running else "Start Center")
        self.set_status("Centering started." if self.servo_running else "Centering stopped.")

    def center_and_pick(self) -> None:
        if not self._lock_servo_target():
            return
        self.servo_running = True
        self.center_then_pick = True
        self.hide_target_overlay = True
        self.flicker_enabled.setChecked(False)
        self.stable_center_frames = 0
        self.center_btn.setText("Stop Center")
        self.set_status("Center then pick started.")

    def _servo_tick(self) -> None:
        if not self.servo_running:
            return
        if self.busy_command or (time.monotonic() - self.last_command_ts) < SERVO_COMMAND_INTERVAL_SEC:
            return
        done = self.servo_step()
        if done:
            self.stable_center_frames += 1
            if self.stable_center_frames >= 3:
                should_pick = self.center_then_pick
                self.center_then_pick = False
                self._finish_centering()
                if should_pick:
                    self.pick_selected()
                self.hide_target_overlay = False
        elif not done:
            self.stable_center_frames = 0

    def servo_step(self) -> bool:
        if self.robot is None:
            self.set_status("Connect robot first.")
            return False
        target = self.servo_target()
        if target is None or self.frame is None:
            self.set_status("No locked target.")
            return False
        state = dict(self.robot.state)
        if state.get("busy"):
            self.set_status("Robot busy: " + str(state.get("busy_action", "")))
            return False
        h, w = self.frame.shape[:2]
        err_x = float(target.center[0] - w / 2.0)
        err_y = float(target.center[1] - h / 2.0)
        dist = math.hypot(err_x, err_y)
        if dist <= float(self.tolerance_px.value()):
            self.set_status("Centered: {:.1f}px".format(dist))
            return True

        theta_sign = -1.0 if self.invert_theta.isChecked() else 1.0
        radius_sign = -1.0 if self.invert_radius.isChecked() else 1.0
        dtheta = theta_sign * clamp(
            -err_x * float(self.theta_gain.value()),
            -float(self.max_theta_step.value()),
            float(self.max_theta_step.value()),
        )
        dradius = radius_sign * clamp(
            -err_y * float(self.radius_gain.value()),
            -float(self.max_radius_step.value()),
            float(self.max_radius_step.value()),
        )
        theta = clamp(
            float(state.get("theta_deg", 0.0)) + dtheta,
            float(state.get("theta_min_deg", -120.0)),
            float(state.get("theta_max_deg", 120.0)),
        )
        radius = clamp(
            float(state.get("radius_mm", 160.0)) + dradius,
            float(state.get("radius_min_mm", 50.0)),
            float(state.get("radius_max_mm", 280.0)),
        )
        self.busy_command = True
        self.last_command_ts = time.monotonic()
        try:
            response = self.robot.move_cyl_auto(theta, radius)
            self.set_status("Servo move -> theta={:.1f}, r={:.1f}, {}".format(theta, radius, response))
        except Exception as exc:
            self.set_status("Servo failed: " + str(exc))
        finally:
            self.busy_command = False
        return False

    def keyboard_move(self, dtheta: float, dradius: float, dz: float, label: str) -> None:
        if self.robot is None:
            self.set_status("Connect robot first.")
            return
        if self.servo_running:
            return
        now = time.monotonic()
        if self.busy_command or (now - self.last_keyboard_move_ts) < KEYBOARD_MOVE_INTERVAL_SEC:
            return
        state = dict(self.robot.state)
        if not state:
            self.set_status("No robot state.")
            return
        if state.get("busy"):
            self.set_status("Robot busy: " + str(state.get("busy_action", "")))
            return
        carrying = bool(state.get("carrying", False)) or str(state.get("state", "")) == "CARRY_READY"
        if not carrying:
            self.set_status("Keyboard move is enabled after pick.")
            return

        theta = clamp(
            float(state.get("theta_deg", 0.0)) + float(dtheta),
            float(state.get("theta_min_deg", -120.0)),
            float(state.get("theta_max_deg", 120.0)),
        )
        radius = clamp(
            float(state.get("radius_mm", 160.0)) + float(dradius),
            float(state.get("radius_min_mm", 50.0)),
            float(state.get("radius_max_mm", 280.0)),
        )
        z = clamp(
            float(state.get("z_mm", 160.0)) + float(dz),
            float(state.get("z_min_mm", 80.0)),
            float(state.get("z_max_mm", 212.8)),
        )

        self.busy_command = True
        self.last_keyboard_move_ts = now
        try:
            response = self.robot.move_cyl(theta, radius, z)
            self.set_status(
                "Keyboard {0} -> theta={1:.1f}, r={2:.1f}, z={3:.1f}, {4}".format(
                    label,
                    theta,
                    radius,
                    z,
                    response,
                )
            )
        except Exception as exc:
            self.set_status("Keyboard move failed: " + str(exc))
        finally:
            self.busy_command = False

    def apply_descend_z_tuning(self) -> str:
        if self.robot is None:
            raise RuntimeError("ROS is not connected.")
        tuning = dict(self.robot.get_pick_tuning())
        descend_z = float(self.descend_z.value())
        tuning_payload = {
            "pick_approach_z_mm": float(tuning.get("pick_approach_z_mm", 130.0)),
            "pick_descend_z_mm": descend_z,
            "pick_pre_suction_sec": float(tuning.get("pick_pre_suction_sec", 0.25)),
            "pick_bottom_hold_sec": float(tuning.get("pick_bottom_hold_sec", 0.15)),
            "pick_lift_sec": float(tuning.get("pick_lift_sec", 0.8)),
            "place_descend_z_mm": descend_z,
            "place_release_mode": str(tuning.get("place_release_mode", "release")),
            "place_release_sec": float(tuning.get("place_release_sec", 0.25)),
            "place_post_release_hold_sec": float(tuning.get("place_post_release_hold_sec", 0.10)),
            "z_carry_floor_mm": float(tuning.get("z_carry_floor_mm", 160.0)),
        }
        response = self.robot.set_pick_tuning(tuning_payload)
        if not bool(response.get("ok", False)):
            raise RuntimeError("set_pick_tuning failed: " + str(response))
        return " descend_z={:.1f};".format(descend_z)

    def pick_selected(self) -> None:
        if self.robot is None:
            self.set_status("Connect robot first.")
            return
        if self.servo_target_id is None and not self._lock_servo_target():
            return
        locked_target_id = self.servo_target_id
        state = dict(self.robot.state)
        if not state:
            self.set_status("No robot state.")
            return
        theta = float(state.get("theta_deg", 0.0))
        base_radius = float(state.get("radius_mm", 160.0))
        radius_min = float(state.get("radius_min_mm", 50.0))
        radius_max = float(state.get("radius_max_mm", 280.0))
        forward_offset = float(self.pick_forward_offset.value())
        pick_radius = clamp(base_radius + forward_offset, radius_min, radius_max)
        try:
            tuning_text = self.apply_descend_z_tuning()
            low_forward_text = ""
            low_forward_used = False
            if self.low_forward_enabled.isChecked() and abs(pick_radius - base_radius) > 0.1:
                low_z = float(self.descend_z.value())
                z_min = float(state.get("z_min_mm", low_z))
                z_max = float(state.get("z_max_mm", low_z))
                low_z = clamp(low_z, z_min, z_max)
                approach_z = clamp(float(state.get("z_mm", low_z)), z_min, z_max)
                if approach_z < low_z:
                    approach_z = low_z
                self.robot.move_cyl(theta, pick_radius, approach_z)
                self.robot.move_cyl(theta, pick_radius, low_z)
                low_forward_text += " forward-first z={:.1f}->{:.1f};".format(approach_z, low_z)
                low_forward_used = True
            response = self.robot.pick_here() if low_forward_used else self.robot.pick_cyl(theta, pick_radius, 0.0)
            self.set_status(
                "Pick target #{}, camera r={:.1f}, suction r={:.1f}, offset={:.1f};{} {}".format(
                    locked_target_id,
                    base_radius,
                    pick_radius,
                    pick_radius - base_radius,
                    tuning_text + low_forward_text,
                    response,
                )
            )
        except Exception as exc:
            self.set_status("Pick failed: " + str(exc))

    def place_selected(self) -> None:
        if self.robot is None:
            self.set_status("Connect robot first.")
            return
        try:
            tuning_text = self.apply_descend_z_tuning()
            response = self.robot.place()
            self._clear_servo_lock()
            self.hide_target_overlay = False
            self.set_status("place ->{} {}".format(tuning_text, response))
        except Exception as exc:
            self.set_status("place failed: " + str(exc))

    def call_robot(self, action: str) -> None:
        if self.robot is None:
            self.set_status("Connect robot first.")
            return
        try:
            response = getattr(self.robot, action)()
            if action in {"reset", "abort", "sucker_off"}:
                self._clear_servo_lock()
                self.hide_target_overlay = False
            self.set_status(action + " -> " + str(response))
        except Exception as exc:
            self.set_status(action + " failed: " + str(exc))

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
        print(text, flush=True)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait(1500)
        if self.robot is not None:
            self.robot.close()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
