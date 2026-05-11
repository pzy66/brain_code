from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from hybrid_controller.adapters.vision_adapter import VisionTarget
from hybrid_controller.cylindrical import cartesian_to_cylindrical
from hybrid_controller.vision.calibration_profile import VisionCalibrationProfile


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


def median_point(points: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not points:
        return None
    xs = np.array([point[0] for point in points], dtype=np.float64)
    ys = np.array([point[1] for point in points], dtype=np.float64)
    return (int(round(float(np.median(xs)))), int(round(float(np.median(ys)))))


def _angle_distance_deg(angle_a: float, angle_b: float) -> float:
    delta = float(angle_a) - float(angle_b)
    while delta <= -45.0:
        delta += 90.0
    while delta > 45.0:
        delta -= 90.0
    return float(delta)


def median_grasp_angle_deg(angles: list[float]) -> float | None:
    values = [float(angle) for angle in angles if math.isfinite(float(angle))]
    if not values:
        return None
    reference = values[-1]
    aligned = [reference + _angle_distance_deg(value, reference) for value in values]
    return normalize_rect_grasp_angle_deg(float(np.median(np.array(aligned, dtype=np.float64))), 1.0, 1.0)


def normalize_rect_grasp_angle_deg(angle_deg: float, width_px: float, height_px: float) -> float:
    """Return the grasp angle of a minAreaRect long edge in [-45, 45] deg."""
    value = float(angle_deg)
    if float(width_px) < float(height_px):
        value += 90.0
    while value <= -45.0:
        value += 90.0
    while value > 45.0:
        value -= 90.0
    return float(value)


@dataclass(frozen=True, slots=True)
class GeometryResult:
    polygon: list[tuple[int, int]]
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    area_px: int
    grasp_pixel: tuple[int, int]
    grasp_quality: float
    oriented_bbox: list[tuple[int, int]]
    grasp_angle_deg: float | None = None
    grasp_angle_quality: float = 0.0

    def as_legacy_tuple(self) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int, int, int], int]:
        return self.polygon, self.center, self.bbox, int(self.area_px)


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


def _estimate_top_face_grasp_pixel_and_quality(
    component: np.ndarray,
    frame_bgr: np.ndarray | None,
    fallback_pixel: tuple[int, int],
) -> tuple[tuple[int, int], float]:
    dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    _, max_value, _, max_loc = cv2.minMaxLoc(dist)
    core = np.where(dist >= max(4.0, 0.35 * float(max_value)), 255, 0).astype(np.uint8)
    core = cv2.bitwise_and(core, component)
    moments = cv2.moments(core, binaryImage=True)
    if moments["m00"] > 0:
        core_pixel = (
            int(round(moments["m10"] / moments["m00"])),
            int(round(moments["m01"] / moments["m00"])),
        )
    else:
        core_pixel = (int(max_loc[0]), int(max_loc[1])) if max_value > 0 else fallback_pixel

    if frame_bgr is None or frame_bgr.shape[:2] != component.shape[:2]:
        return core_pixel, 0.0

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    ys, xs = np.where(component > 0)
    if len(xs) < 20:
        return core_pixel, 0.0
    values = hsv[ys, xs]
    saturation = values[:, 1]
    brightness = values[:, 2]
    chromatic = saturation > 45
    if int(np.count_nonzero(chromatic)) < 20:
        return core_pixel, 0.0

    brightness_cutoff = float(np.percentile(brightness[chromatic], 35.0))
    selected_indexes = np.where(chromatic & (brightness >= brightness_cutoff))[0]
    if len(selected_indexes) < 20:
        return core_pixel, 0.0

    top_mask = np.zeros(component.shape, dtype=np.uint8)
    top_mask[ys[selected_indexes], xs[selected_indexes]] = 255
    top_mask = cv2.morphologyEx(
        top_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    top_mask = cv2.bitwise_and(top_mask, component)
    top_moments = cv2.moments(top_mask, binaryImage=True)
    if top_moments["m00"] <= 0:
        return core_pixel, 0.0
    color_pixel = (
        int(round(top_moments["m10"] / top_moments["m00"])),
        int(round(top_moments["m01"] / top_moments["m00"])),
    )
    if euclidean_distance(color_pixel, core_pixel) > 35.0:
        return core_pixel, 0.0
    component_area = max(1, int(np.count_nonzero(component)))
    top_area = int(np.count_nonzero(top_mask))
    top_ratio = float(top_area) / float(component_area)
    top_fill_quality = clamp01((top_ratio - 0.12) / 0.45)
    distance_quality = clamp01(1.0 - euclidean_distance(color_pixel, core_pixel) / 35.0)
    quality = clamp01(0.65 * top_fill_quality + 0.35 * distance_quality)
    return (
        int(round((float(core_pixel[0]) + float(color_pixel[0])) / 2.0)),
        int(round((float(core_pixel[1]) + float(color_pixel[1])) / 2.0)),
    ), float(quality)


def estimate_top_face_grasp_pixel(
    component: np.ndarray,
    frame_bgr: np.ndarray | None,
    fallback_pixel: tuple[int, int],
) -> tuple[int, int]:
    pixel, _ = _estimate_top_face_grasp_pixel_and_quality(component, frame_bgr, fallback_pixel)
    return pixel


def contour_to_grasp_geometry(
    component: np.ndarray,
    contour: np.ndarray,
    area_px: int,
    *,
    frame_bgr: np.ndarray | None = None,
) -> GeometryResult | None:
    moments = cv2.moments(component, binaryImage=True)
    if moments["m00"] <= 0:
        return None

    center_x = int(round(moments["m10"] / moments["m00"]))
    center_y = int(round(moments["m01"] / moments["m00"]))
    x, y, w, h = cv2.boundingRect(contour)
    epsilon = max(1.0, 0.004 * cv2.arcLength(contour, True))
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    polygon = [(int(point[0][0]), int(point[0][1])) for point in simplified]
    if len(polygon) < 3:
        return None

    rect = cv2.minAreaRect(contour)
    rect_center = rect[0]
    rect_size = rect[1]
    rect_angle = float(rect[2])
    rect_w = max(1.0, float(rect_size[0]))
    rect_h = max(1.0, float(rect_size[1]))
    rect_area = rect_w * rect_h
    fill_ratio = clamp01(float(area_px) / rect_area)
    aspect_ratio = max(rect_w, rect_h) / max(1.0, min(rect_w, rect_h))
    aspect_score = clamp01(1.0 - max(0.0, aspect_ratio - 3.0) / 3.0)
    angle_quality = clamp01((aspect_ratio - 1.05) / 0.75) * fill_ratio
    area_score = clamp01(float(area_px) / 600.0)
    grasp_quality = clamp01(0.45 * fill_ratio + 0.35 * aspect_score + 0.20 * area_score)
    box_points = cv2.boxPoints(rect)
    oriented_bbox = [(int(round(point[0])), int(round(point[1]))) for point in box_points]
    grasp_x = int(round(float(rect_center[0])))
    grasp_y = int(round(float(rect_center[1])))
    rect_grasp_pixel = (grasp_x, grasp_y)
    safe_pixel, top_quality = _estimate_top_face_grasp_pixel_and_quality(component, frame_bgr, rect_grasp_pixel)
    max_top_shift_px = max(8.0, min(rect_w, rect_h) * 0.35)
    bbox = (int(x), int(y), int(x + w), int(y + h))
    edge_touch = x <= 1 or y <= 1 or (x + w) >= component.shape[1] - 1 or (y + h) >= component.shape[0] - 1
    if top_quality >= 0.35 and euclidean_distance(safe_pixel, rect_grasp_pixel) <= max_top_shift_px:
        grasp_pixel = safe_pixel
    else:
        # The default pick point is the deepest point inside the mask, not the
        # projected silhouette center. This is more robust when the visible
        # side face shifts the contour away from the true suction point.
        grasp_pixel = safe_pixel
    if edge_touch:
        grasp_quality = min(float(grasp_quality), 0.15)
    return GeometryResult(
        polygon=polygon,
        center=(center_x, center_y),
        bbox=bbox,
        area_px=int(area_px),
        grasp_pixel=grasp_pixel,
        grasp_quality=float(grasp_quality),
        oriented_bbox=oriented_bbox,
        grasp_angle_deg=normalize_rect_grasp_angle_deg(rect_angle, rect_w, rect_h),
        grasp_angle_quality=float(angle_quality),
    )


def mask_to_geometry(
    mask: np.ndarray,
    frame_shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int, int, int], int] | None:
    result = mask_to_grasp_geometry(mask, frame_shape)
    return None if result is None else result.as_legacy_tuple()


def mask_to_grasp_geometry(
    mask: np.ndarray,
    frame_shape: tuple[int, int],
    *,
    frame_bgr: np.ndarray | None = None,
) -> GeometryResult | None:
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

    return contour_to_grasp_geometry(component, contour, int(area_px), frame_bgr=frame_bgr)


def bbox_to_geometry(
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int, int, int], int]:
    return bbox_to_grasp_geometry(bbox_xyxy).as_legacy_tuple()


def bbox_to_grasp_geometry(
    bbox_xyxy: tuple[float, float, float, float],
) -> GeometryResult:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox_xyxy]
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    polygon = [(left, top), (right, top), (right, bottom), (left, bottom)]
    center = (int(round((left + right) / 2.0)), int(round((top + bottom) / 2.0)))
    bbox = (left, top, right, bottom)
    area_px = max(0, right - left) * max(0, bottom - top)
    fill_score = 1.0 if area_px > 0 else 0.0
    area_score = clamp01(float(area_px) / 600.0)
    return GeometryResult(
        polygon=polygon,
        center=center,
        bbox=bbox,
        area_px=int(area_px),
        grasp_pixel=center,
        grasp_quality=clamp01(0.65 * fill_score + 0.35 * area_score),
        oriented_bbox=polygon,
        grasp_angle_deg=0.0,
        grasp_angle_quality=0.0,
    )


def block_candidate_score(
    item: "DetectionCandidate",
    *,
    largest_area_px: float,
    roi_radius: float,
    frame_shape: tuple[int, int] | None = None,
) -> float:
    area_ratio = min(1.0, float(item.area_px) / max(1.0, float(largest_area_px)))
    distance_penalty = min(1.0, float(item.distance_to_roi) / max(1.0, float(roi_radius)))
    edge_penalty = 0.0
    if frame_shape is not None:
        frame_h, frame_w = frame_shape
        bbox = item.bbox
        edge_touch = bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= frame_w - 1 or bbox[3] >= frame_h - 1
        edge_penalty = 0.45 if edge_touch else 0.0
    return (
        2.4 * float(item.grasp_quality)
        + 1.4 * area_ratio
        + 0.4 * float(item.confidence)
        - 0.5 * distance_penalty
        - edge_penalty
    )


def frame_to_block_candidates(
    frame_bgr: np.ndarray,
    *,
    roi_center: tuple[int, int],
    roi_radius: int,
    max_det: int,
    min_area_px: int = 700,
) -> list[DetectionCandidate]:
    """Color-agnostic fallback for visible painted blocks on a pale workspace.

    It deliberately filters by filled rectangular shape so workspace lines and
    dashed target marks do not become pick candidates.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
        return []
    frame_h, frame_w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    chroma_lab = np.sqrt(
        np.square(lab[:, :, 1].astype(np.float32) - 128.0)
        + np.square(lab[:, :, 2].astype(np.float32) - 128.0)
    )
    mask = np.where(((saturation > 45) | (chroma_lab > 18.0)) & (value > 35), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[DetectionCandidate] = []
    frame_area = float(frame_w * frame_h)
    for contour in contours:
        area_px = int(round(float(cv2.contourArea(contour))))
        if area_px < int(min_area_px) or float(area_px) > frame_area * 0.45:
            continue
        if len(contour) < 3:
            continue
        rect = cv2.minAreaRect(contour)
        rect_w = max(1.0, float(rect[1][0]))
        rect_h = max(1.0, float(rect[1][1]))
        long_side = max(rect_w, rect_h)
        short_side = min(rect_w, rect_h)
        if short_side < 18.0:
            continue
        aspect_ratio = long_side / short_side
        if aspect_ratio > 2.4:
            continue
        fill_ratio = float(area_px) / max(1.0, rect_w * rect_h)
        if fill_ratio < 0.45:
            continue
        component = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.drawContours(component, [contour], contourIdx=-1, color=255, thickness=-1)
        geometry = contour_to_grasp_geometry(component, contour, area_px, frame_bgr=frame_bgr)
        if geometry is None:
            continue
        distance_to_roi = euclidean_distance(geometry.center, roi_center)
        if distance_to_roi > float(roi_radius):
            continue
        edge_touch = (
            geometry.bbox[0] <= 1
            or geometry.bbox[1] <= 1
            or geometry.bbox[2] >= frame_w - 1
            or geometry.bbox[3] >= frame_h - 1
        )
        quality = clamp01(float(geometry.grasp_quality) * (0.35 if edge_touch else 1.0))
        candidates.append(
            DetectionCandidate(
                center=geometry.center,
                grasp_pixel=geometry.grasp_pixel,
                bbox=geometry.bbox,
                area_px=int(geometry.area_px),
                confidence=0.35 if edge_touch else 0.55,
                polygon=geometry.polygon,
                grasp_quality=float(quality),
                oriented_bbox=list(geometry.oriented_bbox),
                grasp_angle_deg=geometry.grasp_angle_deg,
                grasp_angle_quality=float(geometry.grasp_angle_quality),
                distance_to_roi=float(distance_to_roi),
            )
        )
    if candidates:
        largest_area = max(1.0, float(max(item.area_px for item in candidates)))
        roi_scale = max(1.0, float(roi_radius))

        def candidate_score(item: DetectionCandidate) -> float:
            return block_candidate_score(
                item,
                largest_area_px=largest_area,
                roi_radius=roi_scale,
                frame_shape=(frame_h, frame_w),
            )

        candidates.sort(key=lambda item: (-candidate_score(item), item.distance_to_roi, -item.area_px))
    return candidates[: int(max_det)]


@dataclass(frozen=True, slots=True)
class DetectionCandidate:
    center: tuple[int, int]
    grasp_pixel: tuple[int, int]
    bbox: tuple[int, int, int, int]
    area_px: int
    confidence: float
    polygon: list[tuple[int, int]]
    grasp_quality: float
    oriented_bbox: list[tuple[int, int]]
    distance_to_roi: float
    grasp_angle_deg: float | None = None
    grasp_angle_quality: float = 0.0


@dataclass
class SlotState:
    slot: int
    freq_hz: float
    valid: bool = False
    observed: bool = False
    pixel_center: tuple[int, int] | None = None
    center_history: list[tuple[int, int]] = field(default_factory=list)
    grasp_pixel: tuple[int, int] | None = None
    grasp_history: list[tuple[int, int]] = field(default_factory=list)
    grasp_angle_history: list[float] = field(default_factory=list)
    area_history: list[int] = field(default_factory=list)
    bbox: tuple[int, int, int, int] | None = None
    area_px: int = 0
    confidence: float = 0.0
    polygon: list[tuple[int, int]] = field(default_factory=list)
    oriented_bbox: list[tuple[int, int]] = field(default_factory=list)
    grasp_quality: float = 0.0
    grasp_angle_deg: float | None = None
    grasp_angle_quality: float = 0.0
    grasp_angle_stability_deg: float | None = None
    age: int = 0
    lost_frames: int = 0
    command_mode: str = "cyl"
    command_point: tuple[float, float] | None = None
    servo_command_mode: str = "cyl"
    servo_command_point: tuple[float, float] | None = None
    cylindrical_center: tuple[float, float, float] | None = None
    world_xyz: tuple[float, float, float] | None = None
    mapping_mode: str = "absolute_base"
    camera_to_world_raw: tuple[float, float, float] | None = None
    undistorted_pixel: tuple[float, float] | None = None
    alignment_target_pixel: tuple[float, float] | None = None
    center_distance_px: float | None = None
    center_tolerance_px: float | None = None
    action_tolerance_px: float | None = None
    estimated_xy_error_mm: float | None = None
    center_stable_frames: int = 0
    center_stability_px: float | None = None
    grasp_stable_frames: int = 0
    grasp_stability_px: float | None = None
    area_stability_ratio: float | None = None
    servo_required: bool = False
    calibration_profile_id: str = ""
    actionable: bool = False
    invalid_reason: str = ""
    resolved_base_xy: tuple[float, float] | None = None
    resolved_cyl: tuple[float, float, float] | None = None

    def assign(
        self,
        candidate: DetectionCandidate,
        increment_age: bool,
        *,
        grasp_history_len: int = 5,
        center_stability_tolerance_px: float = 6.0,
        grasp_stability_tolerance_px: float = 6.0,
        grasp_history_reset_px: float = 22.0,
        grasp_angle_stability_tolerance_deg: float = 15.0,
    ) -> None:
        self.valid = True
        self.observed = True
        self.pixel_center = candidate.center
        history_len = max(1, int(grasp_history_len))
        self.center_history.append(candidate.center)
        if len(self.center_history) > history_len:
            del self.center_history[:-history_len]
        if self.center_history:
            center_median = median_point(self.center_history) or candidate.center
            self.center_stability_px = max(euclidean_distance(point, center_median) for point in self.center_history)
            self.center_stable_frames = (
                len(self.center_history)
                if float(self.center_stability_px) <= float(center_stability_tolerance_px)
                else 1
            )
        else:
            self.center_stability_px = None
            self.center_stable_frames = 0
        previous_median = median_point(self.grasp_history)
        if previous_median is not None and euclidean_distance(candidate.grasp_pixel, previous_median) > float(
            grasp_history_reset_px
        ):
            self.grasp_history = []
            self.grasp_angle_history = []
            self.area_history = []
        self.grasp_history.append(candidate.grasp_pixel)
        if len(self.grasp_history) > history_len:
            del self.grasp_history[:-history_len]
        median = median_point(self.grasp_history) or candidate.grasp_pixel
        self.grasp_pixel = median
        if self.grasp_history:
            self.grasp_stability_px = max(euclidean_distance(point, median) for point in self.grasp_history)
            self.grasp_stable_frames = (
                len(self.grasp_history)
                if float(self.grasp_stability_px) <= float(grasp_stability_tolerance_px)
                else 1
            )
        else:
            self.grasp_stability_px = None
            self.grasp_stable_frames = 0
        self.bbox = candidate.bbox
        self.area_px = int(candidate.area_px)
        self.area_history.append(int(candidate.area_px))
        if len(self.area_history) > history_len:
            del self.area_history[:-history_len]
        if self.area_history:
            area_values = np.array(self.area_history, dtype=np.float64)
            median_area = max(1.0, float(np.median(area_values)))
            self.area_stability_ratio = float(np.max(np.abs(area_values - median_area)) / median_area)
        else:
            self.area_stability_ratio = None
        self.confidence = float(candidate.confidence)
        self.polygon = list(candidate.polygon)
        self.oriented_bbox = list(candidate.oriented_bbox)
        self.grasp_quality = float(candidate.grasp_quality)
        self.grasp_angle_quality = float(candidate.grasp_angle_quality)
        if candidate.grasp_angle_deg is None or not math.isfinite(float(candidate.grasp_angle_deg)):
            self.grasp_angle_history = []
            self.grasp_angle_deg = None
            self.grasp_angle_stability_deg = None
        else:
            current_angle = float(candidate.grasp_angle_deg)
            previous_angle = median_grasp_angle_deg(self.grasp_angle_history)
            if previous_angle is not None and abs(_angle_distance_deg(current_angle, previous_angle)) > 35.0:
                self.grasp_angle_history = []
            self.grasp_angle_history.append(current_angle)
            if len(self.grasp_angle_history) > history_len:
                del self.grasp_angle_history[:-history_len]
            median_angle = median_grasp_angle_deg(self.grasp_angle_history)
            self.grasp_angle_deg = None if median_angle is None else float(median_angle)
            if self.grasp_angle_history and median_angle is not None:
                self.grasp_angle_stability_deg = max(
                    abs(_angle_distance_deg(angle, median_angle)) for angle in self.grasp_angle_history
                )
                if float(self.grasp_angle_stability_deg) > float(grasp_angle_stability_tolerance_deg):
                    self.grasp_angle_quality = min(float(self.grasp_angle_quality), 0.1)
            else:
                self.grasp_angle_stability_deg = None
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
        self.center_history = []
        self.grasp_pixel = None
        self.grasp_history = []
        self.grasp_angle_history = []
        self.area_history = []
        self.bbox = None
        self.area_px = 0
        self.confidence = 0.0
        self.polygon = []
        self.oriented_bbox = []
        self.grasp_quality = 0.0
        self.grasp_angle_deg = None
        self.grasp_angle_quality = 0.0
        self.grasp_angle_stability_deg = None
        self.age = 0
        self.lost_frames = 0
        self.command_point = None
        self.servo_command_point = None
        self.servo_command_mode = "cyl"
        self.cylindrical_center = None
        self.world_xyz = None
        self.mapping_mode = "absolute_base"
        self.camera_to_world_raw = None
        self.undistorted_pixel = None
        self.alignment_target_pixel = None
        self.center_distance_px = None
        self.center_tolerance_px = None
        self.action_tolerance_px = None
        self.estimated_xy_error_mm = None
        self.center_stable_frames = 0
        self.center_stability_px = None
        self.grasp_stable_frames = 0
        self.grasp_stability_px = None
        self.area_stability_ratio = None
        self.servo_required = False
        self.calibration_profile_id = ""
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
            "grasp_pixel": None if self.grasp_pixel is None else [int(self.grasp_pixel[0]), int(self.grasp_pixel[1])],
            "bbox": None if self.bbox is None else [int(v) for v in self.bbox],
            "area_px": int(self.area_px),
            "area_stability_ratio": None if self.area_stability_ratio is None else float(self.area_stability_ratio),
            "confidence": float(self.confidence),
            "polygon": [[int(x), int(y)] for x, y in self.polygon],
            "oriented_bbox": [[int(x), int(y)] for x, y in self.oriented_bbox],
            "grasp_quality": float(self.grasp_quality),
            "grasp_angle_deg": None if self.grasp_angle_deg is None else float(self.grasp_angle_deg),
            "grasp_angle_quality": float(self.grasp_angle_quality),
            "grasp_angle_stability_deg": (
                None if self.grasp_angle_stability_deg is None else float(self.grasp_angle_stability_deg)
            ),
            "age": int(self.age),
            "lost_frames": int(self.lost_frames),
            "command_mode": self.command_mode,
            "command_point": None if self.command_point is None else [float(v) for v in self.command_point],
            "servo_command_mode": self.servo_command_mode,
            "servo_command_point": None if self.servo_command_point is None else [float(v) for v in self.servo_command_point],
            "cylindrical_center": None if self.cylindrical_center is None else [float(v) for v in self.cylindrical_center],
            "world_xyz": None if self.world_xyz is None else [float(v) for v in self.world_xyz],
            "mapping_mode": self.mapping_mode,
            "camera_to_world_raw": (
                None if self.camera_to_world_raw is None else [float(v) for v in self.camera_to_world_raw]
            ),
            "undistorted_pixel": (
                None if self.undistorted_pixel is None else [float(v) for v in self.undistorted_pixel]
            ),
            "alignment_target_pixel": (
                None if self.alignment_target_pixel is None else [float(v) for v in self.alignment_target_pixel]
            ),
            "center_distance_px": None if self.center_distance_px is None else float(self.center_distance_px),
            "center_tolerance_px": None if self.center_tolerance_px is None else float(self.center_tolerance_px),
            "action_tolerance_px": None if self.action_tolerance_px is None else float(self.action_tolerance_px),
            "estimated_xy_error_mm": (
                None if self.estimated_xy_error_mm is None else float(self.estimated_xy_error_mm)
            ),
            "center_stable_frames": int(self.center_stable_frames),
            "center_stability_px": None if self.center_stability_px is None else float(self.center_stability_px),
            "grasp_stable_frames": int(self.grasp_stable_frames),
            "grasp_stability_px": None if self.grasp_stability_px is None else float(self.grasp_stability_px),
            "servo_required": bool(self.servo_required),
            "calibration_profile_id": str(self.calibration_profile_id),
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
    D: np.ndarray | None = None
    image_size: tuple[int, int] | None = None
    profile_id: str = ""

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
        dist_value = None
        for key in ("D", "dist_coeffs", "distCoeffs", "distortion_coefficients"):
            if key in params:
                dist_value = params[key]
                break
        dist = None if dist_value is None else np.array(dist_value, dtype=np.float64).reshape(-1, 1)
        if dist is not None and not np.all(np.isfinite(dist)):
            raise ValueError("Calibration distortion coefficients contain non-finite values.")
        image_size = None
        size_value = params.get("image_size", params.get("frame_size"))
        if isinstance(size_value, (tuple, list)) and len(size_value) >= 2:
            try:
                width = int(round(float(size_value[0])))
                height = int(round(float(size_value[1])))
                if width > 0 and height > 0:
                    image_size = (width, height)
            except (TypeError, ValueError):
                image_size = None
        return cls(
            K=k_mat,
            R=r_vec,
            T=t_vec,
            D=dist,
            image_size=image_size,
            profile_id=str(params.get("profile_id", "")),
        )

    def undistort_pixel(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        if self.D is None or self.D.size == 0:
            return (float(pixel_x), float(pixel_y))
        points = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(points, self.K, self.D, P=self.K)
        return (float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1]))

    def camera_to_world(self, pixel_x: float, pixel_y: float) -> tuple[float, float, float]:
        inv_k = np.asmatrix(self.K).I
        r_mat = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(self.R, r_mat)
        inv_r = np.asmatrix(r_mat).I
        trans_plane_to_cam = np.dot(inv_r, np.asmatrix(self.T))
        coords = np.zeros((3, 1), dtype=np.float64)
        undistorted_x, undistorted_y = self.undistort_pixel(float(pixel_x), float(pixel_y))
        coords[0][0] = float(undistorted_x)
        coords[1][0] = float(undistorted_y)
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
    frame_bgr: np.ndarray | None = None,
    fallback_to_frame: bool = False,
) -> tuple[list[DetectionCandidate], int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "conf", None) is None:
        if bool(fallback_to_frame) and frame_bgr is not None:
            candidates = frame_to_block_candidates(
                frame_bgr,
                roi_center=roi_center,
                roi_radius=roi_radius,
                max_det=max_det,
            )
            return candidates, len(candidates)
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
            geometry = mask_to_grasp_geometry(mask_data[index], frame_shape, frame_bgr=frame_bgr)
        if geometry is None:
            geometry = bbox_to_grasp_geometry(tuple(float(v) for v in boxes_xyxy[index]))
        polygon = geometry.polygon
        center = geometry.center
        bbox = geometry.bbox
        area_px = geometry.area_px
        distance_to_roi = euclidean_distance(center, roi_center)
        if distance_to_roi > roi_radius:
            continue
        candidates.append(
            DetectionCandidate(
                center=center,
                grasp_pixel=geometry.grasp_pixel,
                bbox=bbox,
                area_px=int(area_px),
                confidence=confidence,
                polygon=polygon,
                grasp_quality=float(geometry.grasp_quality),
                oriented_bbox=list(geometry.oriented_bbox),
                grasp_angle_deg=geometry.grasp_angle_deg,
                grasp_angle_quality=float(geometry.grasp_angle_quality),
                distance_to_roi=float(distance_to_roi),
            )
        )
    if candidates:
        largest_area = max(1.0, float(max(item.area_px for item in candidates)))
        roi_scale = max(1.0, float(roi_radius))
        candidates.sort(
            key=lambda item: (
                -block_candidate_score(
                    item,
                    largest_area_px=largest_area,
                    roi_radius=roi_scale,
                    frame_shape=frame_shape,
                ),
                item.distance_to_roi,
                -item.area_px,
                -item.confidence,
            )
        )
    if not candidates and bool(fallback_to_frame) and frame_bgr is not None:
        candidates = frame_to_block_candidates(
            frame_bgr,
            roi_center=roi_center,
            roi_radius=roi_radius,
            max_det=max_det,
        )
        return candidates, len(candidates)
    return candidates[: int(max_det)], len(candidates)


def update_slots(
    slots: list[SlotState],
    candidates: list[DetectionCandidate],
    *,
    match_distance: float,
    lost_ttl: int,
    grasp_history_len: int = 5,
    center_stability_tolerance_px: float = 6.0,
    grasp_stability_tolerance_px: float = 6.0,
    grasp_history_reset_px: float = 22.0,
    grasp_angle_stability_tolerance_deg: float = 15.0,
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
        slots[slot_index].assign(
            candidates[candidate_index],
            increment_age=True,
            grasp_history_len=grasp_history_len,
            center_stability_tolerance_px=center_stability_tolerance_px,
            grasp_stability_tolerance_px=grasp_stability_tolerance_px,
            grasp_history_reset_px=grasp_history_reset_px,
            grasp_angle_stability_tolerance_deg=grasp_angle_stability_tolerance_deg,
        )
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
        slot.assign(
            candidate,
            increment_age=False,
            grasp_history_len=grasp_history_len,
            center_stability_tolerance_px=center_stability_tolerance_px,
            grasp_stability_tolerance_px=grasp_stability_tolerance_px,
            grasp_history_reset_px=grasp_history_reset_px,
            grasp_angle_stability_tolerance_deg=grasp_angle_stability_tolerance_deg,
        )


def annotate_slots_with_cylindrical(
    slots: Iterable[SlotState],
    *,
    calibration: VisionCalibration | None,
    calibration_profile: VisionCalibrationProfile | None = None,
    frame_size: tuple[int, int] | None = None,
    roi_center: tuple[int, int] | None = None,
    world_scale_xy: float = 1.0,
    world_offset_xy_mm: tuple[float, float] = (0.0, 0.0),
    mapping_mode: str = "absolute_base",
    calibration_profile_required: bool = False,
    action_error_threshold_mm: float = 6.0,
    center_tolerance_px: float = 8.0,
    action_center_tolerance_px: float = 14.0,
    alignment_target_pixel: tuple[float, float] | None = None,
    alignment_target_required: bool = False,
    calibration_stage: str | None = None,
    calibration_z_mm: float | None = None,
    grasp_quality_threshold: float = 0.25,
    required_stable_frames: int = 3,
    grasp_angle_stability_tolerance_deg: float = 15.0,
) -> None:
    scale_xy = float(world_scale_xy)
    offset_x = float(world_offset_xy_mm[0])
    offset_y = float(world_offset_xy_mm[1])
    mapping_mode_text = str(mapping_mode or "absolute_base").strip().lower()
    if mapping_mode_text not in {"absolute_base", "delta_servo"}:
        mapping_mode_text = "delta_servo"
    for slot in slots:
        slot.command_mode = "world"
        slot.command_point = None
        slot.servo_command_mode = "cyl"
        slot.servo_command_point = None
        slot.cylindrical_center = None
        slot.world_xyz = None
        slot.mapping_mode = mapping_mode_text
        slot.camera_to_world_raw = None
        slot.undistorted_pixel = None
        slot.alignment_target_pixel = None
        slot.center_distance_px = None
        slot.center_tolerance_px = None
        slot.action_tolerance_px = None
        slot.estimated_xy_error_mm = None
        slot.servo_required = False
        slot.calibration_profile_id = ""
        slot.actionable = False
        slot.invalid_reason = ""
        slot.resolved_base_xy = None
        slot.resolved_cyl = None
        if not slot.valid or slot.pixel_center is None:
            continue
        if slot.grasp_pixel is None:
            slot.grasp_pixel = slot.pixel_center
        if slot.grasp_quality < float(grasp_quality_threshold):
            slot.invalid_reason = "grasp_quality_low"
            continue
        if slot.area_stability_ratio is not None and float(slot.area_stability_ratio) > 0.40:
            slot.invalid_reason = "grasp_unstable"
            continue
        if calibration_profile_required and calibration_profile is None:
            slot.invalid_reason = "calibration_profile_unavailable"
            continue
        active_profile = calibration_profile
        if calibration_profile is not None:
            try:
                active_profile = calibration_profile.model_for_stage(
                    calibration_stage,
                    z_mm=calibration_z_mm,
                    allow_fallback=True,
                )
            except Exception as error:
                slot.invalid_reason = f"calibration_stage_model_failed:{error}"
                continue
        if active_profile is not None and not active_profile.is_valid_for_image_size(frame_size):
            slot.invalid_reason = "calibration_profile_image_size_mismatch"
            continue
        if calibration is None and (active_profile is None or not active_profile.has_pixel_to_delta_model):
            if slot.valid:
                slot.invalid_reason = "calibration_unavailable"
            continue

        point_for_mapping = slot.pixel_center
        effective_alignment_target = alignment_target_pixel
        if effective_alignment_target is None and active_profile is not None:
            effective_alignment_target = active_profile.target_pixel
        if effective_alignment_target is None and bool(alignment_target_required):
            slot.invalid_reason = "alignment_target_unavailable"
            continue
        if effective_alignment_target is None and roi_center is not None:
            effective_alignment_target = (float(roi_center[0]), float(roi_center[1]))
        if effective_alignment_target is not None:
            effective_alignment_target = (
                float(effective_alignment_target[0]),
                float(effective_alignment_target[1]),
            )
            slot.alignment_target_pixel = effective_alignment_target
        try:
            if active_profile is not None and active_profile.has_pixel_to_delta_model:
                mapped = active_profile.map_pixel_to_delta(
                    (float(point_for_mapping[0]), float(point_for_mapping[1])),
                    frame_size=frame_size,
                    target_pixel=effective_alignment_target,
                    stage=calibration_stage,
                    z_mm=calibration_z_mm,
                )
                raw_world_xyz = (float(mapped.delta_xy_mm[0]), float(mapped.delta_xy_mm[1]), 0.0)
                slot.undistorted_pixel = mapped.undistorted_pixel
                slot.estimated_xy_error_mm = mapped.estimated_error_mm
                slot.calibration_profile_id = mapped.profile_id
            else:
                assert calibration is not None
                undistorted = calibration.undistort_pixel(float(point_for_mapping[0]), float(point_for_mapping[1]))
                raw_world_xyz = calibration.camera_to_world(float(point_for_mapping[0]), float(point_for_mapping[1]))
                if mapping_mode_text == "delta_servo" and effective_alignment_target is not None:
                    target_world_xyz = calibration.camera_to_world(
                        float(effective_alignment_target[0]),
                        float(effective_alignment_target[1]),
                    )
                    raw_world_xyz = (
                        float(raw_world_xyz[0] - target_world_xyz[0]),
                        float(raw_world_xyz[1] - target_world_xyz[1]),
                        float(raw_world_xyz[2] - target_world_xyz[2]),
                    )
                slot.undistorted_pixel = undistorted
                slot.calibration_profile_id = calibration.profile_id
                if active_profile is not None:
                    slot.estimated_xy_error_mm = active_profile.estimate_error_mm(undistorted)
                    slot.calibration_profile_id = active_profile.profile_id
        except Exception as error:
            slot.invalid_reason = f"camera_to_world_failed:{error}"
            continue
        slot.camera_to_world_raw = raw_world_xyz
        if slot.estimated_xy_error_mm is not None and float(slot.estimated_xy_error_mm) > float(action_error_threshold_mm):
            slot.invalid_reason = "vision_mapping_error_high"
            continue

        if effective_alignment_target is not None:
            distance_to_center = math.hypot(
                float(point_for_mapping[0]) - float(effective_alignment_target[0]),
                float(point_for_mapping[1]) - float(effective_alignment_target[1]),
            )
            configured_tolerance = float(center_tolerance_px)
            profile_tolerance = (
                min(float(active_profile.center_tolerance_px), configured_tolerance)
                if active_profile is not None
                else configured_tolerance
            )
            action_tolerance = max(float(profile_tolerance), float(action_center_tolerance_px))
            slot.center_distance_px = float(distance_to_center)
            slot.center_tolerance_px = float(profile_tolerance)
            slot.action_tolerance_px = float(action_tolerance)
            is_low_confirm_stage = str(calibration_stage or "").strip().lower() in {"confirm", "pick"}
            center_aligned = distance_to_center <= action_tolerance
            low_confirm_area_unstable = (
                is_low_confirm_stage
                and not center_aligned
                and slot.area_stability_ratio is not None
                and float(slot.area_stability_ratio) > 0.15
            )
            unstable_for_low_servo = (
                int(required_stable_frames) > 1
                and (
                    int(slot.center_stable_frames) < int(required_stable_frames)
                    or low_confirm_area_unstable
                    or (
                        slot.grasp_angle_stability_deg is not None
                        and float(slot.grasp_angle_stability_deg) > float(grasp_angle_stability_tolerance_deg)
                    )
                )
            )
            if mapping_mode_text == "delta_servo" and low_confirm_area_unstable:
                slot.invalid_reason = "grasp_unstable"
                continue
            if (
                mapping_mode_text == "delta_servo"
                and distance_to_center > action_tolerance
                and is_low_confirm_stage
                and unstable_for_low_servo
            ):
                slot.invalid_reason = "grasp_unstable"
                continue
            if mapping_mode_text == "delta_servo" and distance_to_center > action_tolerance:
                slot.servo_required = True
            elif (
                mapping_mode_text == "delta_servo"
                and int(required_stable_frames) > 1
                and int(slot.center_stable_frames) < int(required_stable_frames)
                and not is_low_confirm_stage
            ):
                slot.invalid_reason = "grasp_unstable"
                continue
            elif (
                mapping_mode_text == "delta_servo"
                and int(required_stable_frames) > 1
                and int(slot.grasp_stable_frames) < int(required_stable_frames)
                and not is_low_confirm_stage
            ):
                slot.invalid_reason = "grasp_unstable"
                continue
            elif (
                mapping_mode_text == "delta_servo"
                and slot.grasp_angle_stability_deg is not None
                and float(slot.grasp_angle_stability_deg) > float(grasp_angle_stability_tolerance_deg)
            ):
                slot.invalid_reason = "grasp_unstable"
                continue

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
        grasp_pixel_raw = slot.get("grasp_pixel")
        grasp_pixel = None if grasp_pixel_raw is None else tuple(float(v) for v in grasp_pixel_raw)
        undistorted_raw = slot.get("undistorted_pixel")
        undistorted_pixel = None if undistorted_raw is None else tuple(float(v) for v in undistorted_raw)
        alignment_target_raw = slot.get("alignment_target_pixel") or packet.get("alignment_target_pixel")
        alignment_target_pixel = (
            None if alignment_target_raw is None else tuple(float(v) for v in alignment_target_raw)
        )
        servo_command_raw = slot.get("servo_command_point")
        servo_command_point = None if servo_command_raw is None else tuple(float(v) for v in servo_command_raw)
        grasp_angle_raw = slot.get("grasp_angle_deg")
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
                grasp_pixel=grasp_pixel,
                undistorted_pixel=undistorted_pixel,
                alignment_target_pixel=alignment_target_pixel,
                estimated_xy_error_mm=(
                    None if slot.get("estimated_xy_error_mm") is None else float(slot.get("estimated_xy_error_mm"))
                ),
                servo_required=bool(slot.get("servo_required", False)),
                servo_command_mode=str(slot.get("servo_command_mode", "cyl")),
                servo_command_point=servo_command_point,
                calibration_profile_id=str(slot.get("calibration_profile_id", "")),
                grasp_quality=float(slot.get("grasp_quality", 0.0)),
                grasp_angle_deg=None if grasp_angle_raw is None else float(grasp_angle_raw),
                grasp_angle_quality=float(slot.get("grasp_angle_quality", 0.0)),
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
    capture_ts: float | None = None,
    stream_age_ms: float | None = None,
    mapping_mode: str = "absolute_base",
    calibration_profile_id: str = "",
    calibration_profile_required: bool = False,
    alignment_target_pixel: tuple[float, float] | None = None,
    calibration_stage: str | None = None,
    calibration_z_mm: float | None = None,
) -> dict[str, Any]:
    return {
        "frame_id": int(frame_id),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "image_size": [int(frame_size[0]), int(frame_size[1])],
        "roi_center": [int(roi_center[0]), int(roi_center[1])],
        "roi_radius": int(roi_radius),
        "alignment_target_pixel": (
            None
            if alignment_target_pixel is None
            else [float(alignment_target_pixel[0]), float(alignment_target_pixel[1])]
        ),
        "capture_fps": float(capture_fps),
        "capture_ts": None if capture_ts is None else float(capture_ts),
        "infer_ms": float(infer_ms),
        "queue_age_ms": float(queue_age_ms),
        "stream_age_ms": None if stream_age_ms is None else float(stream_age_ms),
        "detected_count": int(detected_count),
        "selected_slot": None,
        "slots": [slot.to_packet() for slot in slots],
        "calibration_ready": bool(calibration_ready),
        "mapping_mode": str(mapping_mode),
        "calibration_profile_id": str(calibration_profile_id),
        "calibration_profile_required": bool(calibration_profile_required),
        "calibration_stage": None if calibration_stage is None else str(calibration_stage),
        "calibration_z_mm": None if calibration_z_mm is None else float(calibration_z_mm),
    }
