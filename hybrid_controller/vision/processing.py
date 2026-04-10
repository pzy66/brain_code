from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from hybrid_controller.adapters.vision_adapter import VisionTarget
from hybrid_controller.cylindrical import cartesian_to_cylindrical


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


def euclidean_distance(point_a: tuple[int, int], point_b: tuple[int, int]) -> float:
    return math.hypot(float(point_a[0] - point_b[0]), float(point_a[1] - point_b[1]))


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


def mask_to_geometry(
    mask: np.ndarray,
    frame_shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int, int, int], int] | None:
    frame_h, frame_w = frame_shape
    if mask.shape != (frame_h, frame_w):
        mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

    binary = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    component, _ = largest_component(binary)
    if component is None:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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


def bbox_to_geometry(
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int, int, int], int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox_xyxy]
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    polygon = [(left, top), (right, top), (right, bottom), (left, bottom)]
    center = (int(round((left + right) / 2.0)), int(round((top + bottom) / 2.0)))
    bbox = (left, top, right, bottom)
    area_px = max(0, right - left) * max(0, bottom - top)
    return polygon, center, bbox, int(area_px)


@dataclass(frozen=True, slots=True)
class DetectionCandidate:
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    area_px: int
    confidence: float
    polygon: list[tuple[int, int]]
    distance_to_roi: float


@dataclass
class SlotState:
    slot: int
    freq_hz: float
    valid: bool = False
    observed: bool = False
    pixel_center: tuple[int, int] | None = None
    bbox: tuple[int, int, int, int] | None = None
    area_px: int = 0
    confidence: float = 0.0
    polygon: list[tuple[int, int]] = field(default_factory=list)
    age: int = 0
    lost_frames: int = 0
    command_mode: str = "cyl"
    command_point: tuple[float, float] | None = None
    cylindrical_center: tuple[float, float, float] | None = None
    world_xyz: tuple[float, float, float] | None = None
    mapping_mode: str = "absolute_base"
    camera_to_world_raw: tuple[float, float, float] | None = None
    actionable: bool = False
    invalid_reason: str = ""
    resolved_base_xy: tuple[float, float] | None = None
    resolved_cyl: tuple[float, float, float] | None = None

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
        self.command_point = None
        self.cylindrical_center = None
        self.world_xyz = None
        self.mapping_mode = "absolute_base"
        self.camera_to_world_raw = None
        self.actionable = False
        self.invalid_reason = ""
        self.resolved_base_xy = None
        self.resolved_cyl = None

    def to_packet(self) -> dict[str, Any]:
        return {
            "slot_id": int(self.slot),
            "slot": int(self.slot),
            "freq_hz": float(self.freq_hz),
            "valid": bool(self.valid),
            "observed": bool(self.observed),
            "pixel_center": None if self.pixel_center is None else [int(self.pixel_center[0]), int(self.pixel_center[1])],
            "bbox": None if self.bbox is None else [int(v) for v in self.bbox],
            "area_px": int(self.area_px),
            "confidence": float(self.confidence),
            "polygon": [[int(x), int(y)] for x, y in self.polygon],
            "age": int(self.age),
            "lost_frames": int(self.lost_frames),
            "command_mode": self.command_mode,
            "command_point": None if self.command_point is None else [float(v) for v in self.command_point],
            "cylindrical_center": None if self.cylindrical_center is None else [float(v) for v in self.cylindrical_center],
            "world_xyz": None if self.world_xyz is None else [float(v) for v in self.world_xyz],
            "mapping_mode": self.mapping_mode,
            "camera_to_world_raw": (
                None if self.camera_to_world_raw is None else [float(v) for v in self.camera_to_world_raw]
            ),
            "actionable": bool(self.actionable),
            "invalid_reason": str(self.invalid_reason),
            "resolved_base_xy": None if self.resolved_base_xy is None else [float(v) for v in self.resolved_base_xy],
            "resolved_cyl": None if self.resolved_cyl is None else [float(v) for v in self.resolved_cyl],
        }


@dataclass(frozen=True, slots=True)
class VisionCalibration:
    K: np.ndarray
    R: np.ndarray
    T: np.ndarray

    @classmethod
    def from_param_dict(cls, params: dict[str, Any]) -> "VisionCalibration":
        if not isinstance(params, dict):
            raise ValueError("Calibration params must be a dict.")
        k_mat = np.array(params["K"], dtype=np.float64).reshape(3, 3)
        r_vec = np.array(params["R"], dtype=np.float64).reshape(3, 1)
        t_vec = np.array(params["T"], dtype=np.float64).reshape(3, 1)
        np.linalg.inv(k_mat)
        r_mat = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(r_vec, r_mat)
        np.linalg.inv(r_mat)
        return cls(K=k_mat, R=r_vec, T=t_vec)

    def camera_to_world(self, pixel_x: float, pixel_y: float) -> tuple[float, float, float]:
        inv_k = np.asmatrix(self.K).I
        r_mat = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(self.R, r_mat)
        inv_r = np.asmatrix(r_mat).I
        trans_plane_to_cam = np.dot(inv_r, np.asmatrix(self.T))
        coords = np.zeros((3, 1), dtype=np.float64)
        coords[0][0] = float(pixel_x)
        coords[1][0] = float(pixel_y)
        coords[2][0] = 1.0
        world_pt_cam = np.dot(inv_k, coords)
        world_pt_plane = np.dot(inv_r, world_pt_cam)
        scale = trans_plane_to_cam[2][0] / world_pt_plane[2][0]
        scaled = np.multiply(scale, world_pt_plane)
        reprojection = np.asmatrix(scaled) - np.asmatrix(trans_plane_to_cam)
        x_mm, y_mm, z_mm = reprojection.T.tolist()[0]
        values = (float(x_mm), float(y_mm), float(z_mm))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("camera_to_world produced non-finite values.")
        return values


def extract_candidates(
    result: object,
    *,
    frame_shape: tuple[int, int],
    roi_center: tuple[int, int],
    roi_radius: int,
    max_det: int,
    confidence_threshold: float,
) -> tuple[list[DetectionCandidate], int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "conf", None) is None:
        return [], 0
    masks = getattr(result, "masks", None)
    mask_data = None if masks is None or getattr(masks, "data", None) is None else masks.data.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    boxes_xyxy = boxes.xyxy.detach().cpu().numpy()
    count = min(len(confidences), len(boxes_xyxy))
    candidates: list[DetectionCandidate] = []
    for index in range(count):
        confidence = float(confidences[index])
        if confidence < float(confidence_threshold):
            continue
        geometry = None
        if mask_data is not None and index < len(mask_data):
            geometry = mask_to_geometry(mask_data[index], frame_shape)
        if geometry is None:
            geometry = bbox_to_geometry(tuple(float(v) for v in boxes_xyxy[index]))
        polygon, center, bbox, area_px = geometry
        distance_to_roi = euclidean_distance(center, roi_center)
        if distance_to_roi > roi_radius:
            continue
        candidates.append(
            DetectionCandidate(
                center=center,
                bbox=bbox,
                area_px=int(area_px),
                confidence=confidence,
                polygon=polygon,
                distance_to_roi=float(distance_to_roi),
            )
        )
    candidates.sort(key=lambda item: (item.distance_to_roi, -item.area_px, -item.confidence))
    return candidates[: int(max_det)], len(candidates)


def update_slots(
    slots: list[SlotState],
    candidates: list[DetectionCandidate],
    *,
    match_distance: float,
    lost_ttl: int,
) -> None:
    matched_slots: set[int] = set()
    matched_candidates: set[int] = set()
    pairs: list[tuple[float, float, float, int, int]] = []

    for slot_index, slot in enumerate(slots):
        if not slot.valid or slot.pixel_center is None:
            continue
        for candidate_index, candidate in enumerate(candidates):
            distance = euclidean_distance(slot.pixel_center, candidate.center)
            overlap = bbox_iou(slot.bbox, candidate.bbox)
            if distance > float(match_distance) and overlap <= 0.05:
                continue
            score = (distance / float(match_distance)) + (1.0 - overlap) * 0.35
            pairs.append((score, distance, -overlap, slot_index, candidate_index))

    pairs.sort()
    for _, _, _, slot_index, candidate_index in pairs:
        if slot_index in matched_slots or candidate_index in matched_candidates:
            continue
        slots[slot_index].assign(candidates[candidate_index], increment_age=True)
        matched_slots.add(slot_index)
        matched_candidates.add(candidate_index)

    for slot_index, slot in enumerate(slots):
        if slot_index in matched_slots or not slot.valid:
            continue
        slot.mark_missing()
        if slot.lost_frames > int(lost_ttl):
            slot.clear()

    remaining_candidates = [candidates[index] for index in range(len(candidates)) if index not in matched_candidates]
    free_slots = [slot for slot in slots if not slot.valid]
    for slot, candidate in zip(free_slots, remaining_candidates):
        slot.assign(candidate, increment_age=False)


def annotate_slots_with_cylindrical(
    slots: Iterable[SlotState],
    *,
    calibration: VisionCalibration | None,
    world_scale_xy: float = 1.0,
    world_offset_xy_mm: tuple[float, float] = (0.0, 0.0),
    mapping_mode: str = "absolute_base",
) -> None:
    scale_xy = float(world_scale_xy)
    offset_x = float(world_offset_xy_mm[0])
    offset_y = float(world_offset_xy_mm[1])
    mapping_mode_text = str(mapping_mode or "absolute_base").strip().lower()
    if mapping_mode_text not in {"absolute_base", "delta_servo"}:
        mapping_mode_text = "absolute_base"
    for slot in slots:
        slot.command_mode = "world"
        slot.command_point = None
        slot.cylindrical_center = None
        slot.world_xyz = None
        slot.mapping_mode = mapping_mode_text
        slot.camera_to_world_raw = None
        slot.actionable = False
        slot.invalid_reason = ""
        slot.resolved_base_xy = None
        slot.resolved_cyl = None
        if not slot.valid or slot.pixel_center is None or calibration is None:
            if slot.valid and calibration is None:
                slot.invalid_reason = "calibration_unavailable"
            continue
        try:
            raw_world_xyz = calibration.camera_to_world(float(slot.pixel_center[0]), float(slot.pixel_center[1]))
        except Exception as error:
            slot.invalid_reason = f"camera_to_world_failed:{error}"
            continue
        slot.camera_to_world_raw = raw_world_xyz
        mapped_world_xyz = (
            float(raw_world_xyz[0]) * scale_xy + offset_x,
            float(raw_world_xyz[1]) * scale_xy + offset_y,
            float(raw_world_xyz[2]),
        )
        if mapping_mode_text == "absolute_base":
            cylindrical_center = cartesian_to_cylindrical(*mapped_world_xyz)
            slot.world_xyz = mapped_world_xyz
            slot.cylindrical_center = cylindrical_center
            slot.command_point = (float(mapped_world_xyz[0]), float(mapped_world_xyz[1]))
            slot.actionable = True
            slot.resolved_base_xy = (float(mapped_world_xyz[0]), float(mapped_world_xyz[1]))
            slot.resolved_cyl = (
                float(cylindrical_center[0]),
                float(cylindrical_center[1]),
                float(cylindrical_center[2]),
            )
            continue
        slot.invalid_reason = "awaiting_robot_snapshot_delta_resolve"


def packet_to_targets(packet: dict[str, Any]) -> list[VisionTarget]:
    targets: list[VisionTarget] = []
    for slot in packet.get("slots", []):
        if not slot.get("valid"):
            continue
        bbox = tuple(float(v) for v in slot.get("bbox") or (0.0, 0.0, 0.0, 0.0))
        pixel_center = tuple(float(v) for v in slot.get("pixel_center") or (0.0, 0.0))
        command_point_raw = slot.get("command_point")
        command_point = None if command_point_raw is None else tuple(float(v) for v in command_point_raw)
        cylindrical_center_raw = slot.get("cylindrical_center")
        cylindrical_center = (
            None
            if cylindrical_center_raw is None
            else tuple(float(v) for v in cylindrical_center_raw)
        )
        world_xyz_raw = slot.get("world_xyz")
        world_xyz = None if world_xyz_raw is None else tuple(float(v) for v in world_xyz_raw)
        targets.append(
            VisionTarget(
                id=int(slot.get("slot_id", slot.get("slot", 0))),
                bbox=bbox,
                center_px=pixel_center,
                raw_center=pixel_center,
                confidence=float(slot.get("confidence", 0.0)),
                command_mode=str(slot.get("command_mode", "pixel")),
                command_point=command_point,
                display_center=pixel_center,
                slot_id=int(slot.get("slot_id", slot.get("slot", 0))),
                freq_hz=float(slot.get("freq_hz", 0.0)),
                cylindrical_center=cylindrical_center,
                world_xyz=world_xyz,
                mapping_mode=str(slot.get("mapping_mode", packet.get("mapping_mode", "absolute_base"))),
                actionable=bool(slot.get("actionable", command_point is not None)),
                invalid_reason=str(slot.get("invalid_reason", "")),
            )
        )
    return targets


def build_vision_packet(
    *,
    frame_id: int,
    frame_size: tuple[int, int],
    roi_center: tuple[int, int],
    roi_radius: int,
    slots: list[SlotState],
    capture_fps: float,
    infer_ms: float,
    queue_age_ms: float,
    detected_count: int,
    calibration_ready: bool,
    mapping_mode: str = "absolute_base",
) -> dict[str, Any]:
    return {
        "frame_id": int(frame_id),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "image_size": [int(frame_size[0]), int(frame_size[1])],
        "roi_center": [int(roi_center[0]), int(roi_center[1])],
        "roi_radius": int(roi_radius),
        "capture_fps": float(capture_fps),
        "infer_ms": float(infer_ms),
        "queue_age_ms": float(queue_age_ms),
        "detected_count": int(detected_count),
        "selected_slot": None,
        "slots": [slot.to_packet() for slot in slots],
        "calibration_ready": bool(calibration_ready),
        "mapping_mode": str(mapping_mode),
    }
