from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _point(values: Any) -> tuple[int, int] | None:
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return None
    return (int(round(float(values[0]))), int(round(float(values[1]))))


def _draw_point(
    image: Any,
    point: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
    *,
    dy: int = 0,
) -> None:
    cv2.drawMarker(image, point, color, cv2.MARKER_TILTED_CROSS, 28, 2)
    cv2.circle(image, point, 7, color, 2)
    origin = (point[0] + 8, point[1] - 8 + int(dy))
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)


def _draw_text(image: Any, text: str, origin: tuple[int, int]) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def _foreground_component_centers(
    frame: Any,
    bbox: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    if bbox is None:
        return {}
    frame_h, frame_w = frame.shape[:2]
    pad = 35
    x1 = max(0, int(bbox[0]) - pad)
    y1 = max(0, int(bbox[1]) - pad)
    x2 = min(frame_w, int(bbox[2]) + pad)
    y2 = min(frame_h, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return {}
    roi = frame[y1:y2, x1:x2]
    rect_x = max(1, int(bbox[0]) - x1)
    rect_y = max(1, int(bbox[1]) - y1)
    rect_w = max(2, min(roi.shape[1] - rect_x - 1, int(bbox[2]) - int(bbox[0])))
    rect_h = max(2, min(roi.shape[0] - rect_y - 1, int(bbox[3]) - int(bbox[1])))
    if rect_w <= 2 or rect_h <= 2:
        return {"foreground_roi": [x1, y1, x2, y2], "foreground_mask_area_px": 0}
    grabcut_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            roi,
            grabcut_mask,
            (rect_x, rect_y, rect_w, rect_h),
            bgd_model,
            fgd_model,
            3,
            cv2.GC_INIT_WITH_RECT,
        )
        mask = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
    except cv2.error:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"foreground_roi": [x1, y1, x2, y2], "foreground_mask_area_px": 0}
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return {"foreground_roi": [x1, y1, x2, y2], "foreground_mask_area_px": area}
    centroid = (
        int(round(moments["m10"] / moments["m00"] + x1)),
        int(round(moments["m01"] / moments["m00"] + y1)),
    )
    rect = cv2.minAreaRect(contour)
    rect_center = (
        int(round(float(rect[0][0]) + x1)),
        int(round(float(rect[0][1]) + y1)),
    )
    full_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    dist = cv2.distanceTransform(full_mask, cv2.DIST_L2, 5)
    _, max_value, _, max_loc = cv2.minMaxLoc(dist)
    core = np.where(dist >= max(5.0, 0.55 * float(max_value)), 255, 0).astype(np.uint8)
    core_moments = cv2.moments(core, binaryImage=True)
    if core_moments["m00"] > 0:
        core_center = (
            int(round(core_moments["m10"] / core_moments["m00"])),
            int(round(core_moments["m01"] / core_moments["m00"])),
        )
    else:
        core_center = (int(max_loc[0]), int(max_loc[1]))
    return {
        "foreground_roi": [x1, y1, x2, y2],
        "foreground_mask_area_px": area,
        "foreground_centroid": centroid,
        "foreground_minrect_center": rect_center,
        "foreground_core_center": core_center,
        "foreground_minrect_size": [float(rect[1][0]), float(rect[1][1])],
        "foreground_minrect_angle_deg": float(rect[2]),
    }


def _green_hsv_component_centers(
    frame: Any,
    bbox: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    frame_h, frame_w = frame.shape[:2]
    if bbox is None:
        x1, y1, x2, y2 = 0, 0, frame_w, frame_h
    else:
        pad = 35
        x1 = max(0, int(bbox[0]) - pad)
        y1 = max(0, int(bbox[1]) - pad)
        x2 = min(frame_w, int(bbox[2]) + pad)
        y2 = min(frame_h, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return {}
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 35, 20]), np.array([95, 255, 220]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 300.0]
    if not contours:
        return {
            "green_hsv_roi": [x1, y1, x2, y2],
            "green_hsv_mask_area_px": 0.0,
        }
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        centroid = (
            int(round(moments["m10"] / moments["m00"] + x1)),
            int(round(moments["m01"] / moments["m00"] + y1)),
        )
    else:
        centroid = None
    rect = cv2.minAreaRect(contour)
    rect_center = (
        int(round(float(rect[0][0]) + x1)),
        int(round(float(rect[0][1]) + y1)),
    )
    return {
        "green_hsv_roi": [x1, y1, x2, y2],
        "green_hsv_mask_area_px": area,
        "green_hsv_centroid": centroid,
        "green_hsv_minrect_center": rect_center,
        "green_hsv_minrect_size": [float(rect[1][0]), float(rect[1][1])],
        "green_hsv_minrect_angle_deg": float(rect[2]),
    }


def _load_paths(bundle: Path) -> tuple[Path, Path]:
    raw_path = bundle / "raw.jpg" if bundle.is_dir() else bundle
    packet_path = bundle / "packet.json" if bundle.is_dir() else bundle.with_name("packet.json")
    return raw_path, packet_path


def diagnose_centers(bundle: Path, output_path: Path | None = None) -> dict[str, Any]:
    raw_path, packet_path = _load_paths(bundle)
    frame = cv2.imread(str(raw_path))
    if frame is None:
        raise RuntimeError(f"Cannot read raw frame: {raw_path}")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    slots = packet.get("slots")
    if not isinstance(slots, list) or not slots:
        raise RuntimeError(f"No slots in packet: {packet_path}")
    slot = dict(slots[0])

    bbox_center = None
    bbox_tuple = None
    bbox = slot.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        bbox_tuple = (x1, y1, x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 255), 2)
        bbox_center = (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))

    rect_center = None
    oriented_bbox = [_point(point) for point in slot.get("oriented_bbox") or []]
    oriented_bbox = [point for point in oriented_bbox if point is not None]
    if len(oriented_bbox) >= 4:
        for start, end in zip(oriented_bbox, oriented_bbox[1:] + oriented_bbox[:1]):
            cv2.line(frame, start, end, (255, 160, 40), 2)
        rect_center = (
            int(round(sum(point[0] for point in oriented_bbox) / len(oriented_bbox))),
            int(round(sum(point[1] for point in oriented_bbox) / len(oriented_bbox))),
        )

    target = _point(slot.get("alignment_target_pixel") or packet.get("alignment_target_pixel"))
    pixel_center = _point(slot.get("pixel_center"))
    grasp_pixel = _point(slot.get("grasp_pixel"))
    metrics: dict[str, Any] = {
        "raw": str(raw_path),
        "packet": str(packet_path),
        "pixel_center": pixel_center,
        "grasp_pixel": grasp_pixel,
        "target_pixel": target,
        "bbox_center": bbox_center,
        "rect_center": rect_center,
    }
    metrics.update(_foreground_component_centers(frame, bbox_tuple))
    metrics.update(_green_hsv_component_centers(frame, bbox_tuple))

    labels: list[tuple[str, tuple[int, int] | None, tuple[int, int, int], int]] = [
        ("target", target, (255, 80, 220), 0),
        ("pixel_center", pixel_center, (255, 255, 0), 14),
        ("grasp", grasp_pixel, (0, 255, 255), 28),
        ("bbox_center", bbox_center, (40, 220, 255), 42),
        ("rect_center", rect_center, (255, 160, 40), 56),
        ("fg_centroid", _point(metrics.get("foreground_centroid")), (0, 255, 0), 70),
        ("fg_core", _point(metrics.get("foreground_core_center")), (0, 0, 255), 84),
        ("green_hsv", _point(metrics.get("green_hsv_centroid")), (80, 255, 80), 98),
    ]
    for label, point, color, dy in labels:
        if point is not None:
            _draw_point(frame, point, color, label, dy=dy)

    summary_lines: list[str] = []
    if target is not None:
        for key in (
            "pixel_center",
            "grasp_pixel",
            "bbox_center",
            "rect_center",
            "foreground_centroid",
            "foreground_core_center",
            "green_hsv_centroid",
            "green_hsv_minrect_center",
        ):
            point = metrics[key]
            if point is not None:
                distance = float(math.dist(point, target))
                metrics[f"{key}_to_target_px"] = distance
                summary_lines.append(f"{key}={point}, d_target={distance:.2f}px")
    if pixel_center is not None:
        for key in (
            "bbox_center",
            "rect_center",
            "grasp_pixel",
            "foreground_centroid",
            "foreground_core_center",
            "green_hsv_centroid",
            "green_hsv_minrect_center",
        ):
            point = metrics[key]
            if point is not None:
                distance = float(math.dist(point, pixel_center))
                metrics[f"{key}_to_pixel_center_px"] = distance

    for index, text in enumerate(summary_lines[:5]):
        _draw_text(frame, text, (12, 24 + index * 22))

    if output_path is None:
        output_path = raw_path.with_name("center_diagnostics.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    metrics["output"] = str(output_path)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw and report vision center diagnostics for a debug bundle step.")
    parser.add_argument("bundle", type=Path, help="Step directory containing raw.jpg and packet.json, or raw.jpg path.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    metrics = diagnose_centers(Path(args.bundle), output_path=args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
