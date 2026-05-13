from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_controller.vision.processing import estimate_color_block_center


def _point(values: Any) -> tuple[int, int] | None:
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return None
    return (int(round(float(values[0]))), int(round(float(values[1]))))


def _point_f(values: Any) -> tuple[float, float] | None:
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return None
    try:
        x_value = float(values[0])
        y_value = float(values[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x_value) and math.isfinite(y_value)):
        return None
    return (x_value, y_value)


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


def _iter_bundle_steps(path: Path) -> list[Path]:
    candidate = Path(path)
    if candidate.is_file():
        return [candidate]
    if (candidate / "raw.jpg").exists() and (candidate / "packet.json").exists():
        return [candidate]
    steps: list[Path] = []
    for packet_path in candidate.rglob("packet.json"):
        step = packet_path.parent
        if (step / "raw.jpg").exists():
            steps.append(step)
    return sorted(steps, key=lambda item: str(item))


def _slot_from_packet(packet: Mapping[str, object], slot_id: int | None) -> dict[str, object] | None:
    slots = packet.get("slots")
    if not isinstance(slots, list):
        return None
    valid_slots = [dict(slot) for slot in slots if isinstance(slot, Mapping) and bool(slot.get("valid", False))]
    if slot_id is not None:
        for slot in valid_slots:
            try:
                if int(slot.get("slot_id", slot.get("slot", -1))) == int(slot_id):
                    return slot
            except (TypeError, ValueError):
                continue
        return None
    valid_slots.sort(
        key=lambda slot: (
            0 if bool(slot.get("actionable", False)) else 1,
            float(slot.get("center_distance_px", float("inf")) or float("inf")),
            -float(slot.get("confidence", 0.0) or 0.0),
        )
    )
    return valid_slots[0] if valid_slots else None


def _component_from_slot(frame_shape: tuple[int, int], slot: Mapping[str, object]) -> np.ndarray | None:
    frame_h, frame_w = frame_shape
    component = np.zeros((frame_h, frame_w), dtype=np.uint8)
    polygon = []
    for point in slot.get("polygon") or []:
        parsed = _point(point)
        if parsed is not None:
            polygon.append(parsed)
    if len(polygon) >= 3:
        cv2.fillPoly(component, [np.array(polygon, dtype=np.int32)], 255)
        return component
    bbox = slot.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        except (TypeError, ValueError):
            return None
        x1 = max(0, min(frame_w - 1, x1))
        x2 = max(0, min(frame_w, x2))
        y1 = max(0, min(frame_h - 1, y1))
        y2 = max(0, min(frame_h, y2))
        if x2 > x1 and y2 > y1:
            component[y1:y2, x1:x2] = 255
            return component
    return None


def _annotate_diagnostic_color_block(frame: Any, slot: dict[str, object]) -> bool:
    if _point_f(slot.get("color_block_center_f") or slot.get("color_block_center")) is not None:
        return False
    if frame is None or not hasattr(frame, "shape"):
        return False
    component = _component_from_slot(tuple(int(v) for v in frame.shape[:2]), slot)
    if component is None:
        return False
    point = estimate_color_block_center(component, frame)
    if point is None:
        return False
    slot["color_block_center_f"] = [float(point[0]), float(point[1])]
    slot["color_block_center"] = [int(round(float(point[0]))), int(round(float(point[1])))]
    slot["color_block_center_source"] = "diagnostic_recomputed_from_raw"
    return True


def _target_pixel(slot: Mapping[str, object], packet: Mapping[str, object]) -> tuple[float, float] | None:
    return _point_f(slot.get("alignment_target_pixel")) or _point_f(packet.get("alignment_target_pixel"))


def _candidate_points(slot: Mapping[str, object]) -> dict[str, tuple[float, float]]:
    candidates: dict[str, tuple[float, float]] = {}
    for key in (
        "pixel_center_f",
        "pixel_center",
        "geometry_center_f",
        "geometry_center",
        "color_block_center_f",
        "color_block_center",
        "top_face_center_f",
        "top_face_center",
        "grasp_pixel_f",
        "grasp_pixel",
    ):
        point = _point_f(slot.get(key))
        if point is not None:
            candidates[key] = point
    return candidates


def _measurement_mode_for_packet_key(key: str) -> str:
    mapping = {
        "pixel_center_f": "center_subpixel",
        "pixel_center": "center",
        "geometry_center_f": "geometry_subpixel",
        "geometry_center": "geometry",
        "color_block_center_f": "color_block_subpixel",
        "color_block_center": "color_block",
        "top_face_center_f": "top_face_subpixel",
        "top_face_center": "top_face",
        "grasp_pixel_f": "grasp_subpixel",
        "grasp_pixel": "grasp",
    }
    return mapping.get(str(key), "")


def _median(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("inf")
    middle = len(finite) // 2
    if len(finite) % 2:
        return finite[middle]
    return 0.5 * (finite[middle - 1] + finite[middle])


def _max_pairwise_spread(points: list[tuple[float, float]]) -> float:
    spread = 0.0
    for left_index, left in enumerate(points):
        for right in points[left_index + 1 :]:
            spread = max(spread, math.hypot(left[0] - right[0], left[1] - right[1]))
    return float(spread)


def _jump_count(points: list[tuple[float, float]], *, threshold_px: float) -> int:
    if len(points) < 2:
        return 0
    jumps = 0
    for previous, current in zip(points, points[1:]):
        if math.hypot(current[0] - previous[0], current[1] - previous[1]) > float(threshold_px):
            jumps += 1
    return int(jumps)


def _point_summary_for_payloads(
    by_key: Mapping[str, Mapping[str, object]],
    *,
    jump_threshold_px: float,
) -> dict[str, object]:
    point_summary: dict[str, object] = {}
    for key, payload in by_key.items():
        distances = list(payload.get("distances_px", []))  # type: ignore[arg-type]
        points = list(payload.get("points", []))  # type: ignore[arg-type]
        point_summary[key] = {
            "samples": int(len(distances)),
            "median_error_px": _median([float(value) for value in distances]),
            "min_error_px": min(distances) if distances else float("inf"),
            "max_error_px": max(distances) if distances else float("inf"),
            "repeat_spread_px": _max_pairwise_spread(points),
            "jump_count": _jump_count(points, threshold_px=float(jump_threshold_px)),
        }
    return point_summary


def _rank_point_summary(point_summary: Mapping[str, object]) -> str:
    ranked = sorted(
        point_summary.items(),
        key=lambda item: (
            int(item[1]["jump_count"]),  # type: ignore[index]
            float(item[1]["median_error_px"]),  # type: ignore[index]
            float(item[1]["repeat_spread_px"]),  # type: ignore[index]
        ),
    )
    return ranked[0][0] if ranked else ""


def _rank_grouped_point_summary(grouped_summary: Mapping[str, object]) -> str:
    ranked = sorted(
        grouped_summary.items(),
        key=lambda item: (
            -int(item[1]["stable_group_count"]),  # type: ignore[index]
            int(item[1]["jump_count_sum"]),  # type: ignore[index]
            float(item[1]["median_group_error_px"]),  # type: ignore[index]
            float(item[1]["median_group_repeat_spread_px"]),  # type: ignore[index]
        ),
    )
    return ranked[0][0] if ranked else ""


def _group_label_for_step(bundle_root: Path, step: Path) -> str:
    try:
        parent = step.parent.relative_to(bundle_root)
    except ValueError:
        parent = step.parent
    text = str(parent).replace("\\", "/").strip("/")
    return text if text and text != "." else "all"


def _grouped_point_summary(
    grouped_by_key: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    jump_threshold_px: float,
) -> tuple[dict[str, object], dict[str, object]]:
    group_summary: dict[str, object] = {}
    aggregate: dict[str, dict[str, list[float] | int]] = {}
    for group_label, by_key in grouped_by_key.items():
        point_summary = _point_summary_for_payloads(by_key, jump_threshold_px=float(jump_threshold_px))
        recommended_key = _rank_point_summary(point_summary)
        group_summary[group_label] = {
            "point_summary": point_summary,
            "recommended_low_height_packet_key": recommended_key,
            "recommended_low_height_measurement_point": _measurement_mode_for_packet_key(recommended_key),
        }
        for key, summary in point_summary.items():
            entry = aggregate.setdefault(
                key,
                {
                    "group_median_errors_px": [],
                    "group_repeat_spreads_px": [],
                    "stable_group_count": 0,
                    "jump_count_sum": 0,
                    "group_count": 0,
                },
            )
            entry["group_count"] = int(entry["group_count"]) + 1  # type: ignore[index,arg-type]
            entry["jump_count_sum"] = int(entry["jump_count_sum"]) + int(summary["jump_count"])  # type: ignore[index,arg-type]
            entry["group_median_errors_px"].append(float(summary["median_error_px"]))  # type: ignore[index,union-attr]
            entry["group_repeat_spreads_px"].append(float(summary["repeat_spread_px"]))  # type: ignore[index,union-attr]
            if int(summary["jump_count"]) == 0 and float(summary["repeat_spread_px"]) <= float(jump_threshold_px):
                entry["stable_group_count"] = int(entry["stable_group_count"]) + 1  # type: ignore[index,arg-type]
    grouped_ranking: dict[str, object] = {}
    for key, payload in aggregate.items():
        errors = [float(v) for v in payload["group_median_errors_px"]]  # type: ignore[index]
        spreads = [float(v) for v in payload["group_repeat_spreads_px"]]  # type: ignore[index]
        grouped_ranking[key] = {
            "group_count": int(payload["group_count"]),  # type: ignore[arg-type]
            "stable_group_count": int(payload["stable_group_count"]),  # type: ignore[arg-type]
            "jump_count_sum": int(payload["jump_count_sum"]),  # type: ignore[arg-type]
            "median_group_error_px": _median(errors),
            "median_group_repeat_spread_px": _median(spreads),
        }
    return group_summary, grouped_ranking


def _transport_summary(stats: Iterable[Mapping[str, object]]) -> dict[str, object]:
    values = [dict(item) for item in stats]
    if not values:
        return {}
    keys = (
        "content_length_frames",
        "frames_decoded",
        "frames_rejected",
        "buffer_resets",
        "reopen_count",
        "read_timeouts",
    )
    summary: dict[str, object] = {"samples": len(values)}
    for key in keys:
        numbers: list[float] = []
        for item in values:
            try:
                value = float(item.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                numbers.append(value)
        if numbers:
            summary[f"{key}_max"] = max(numbers)
            summary[f"{key}_last"] = numbers[-1]
    summary["reader"] = values[-1].get("reader") or values[-1].get("capture_backend")
    summary["stream_url"] = values[-1].get("stream_url")
    return summary


def diagnose_center_sequence(
    bundle_root: Path,
    *,
    slot_id: int | None = None,
    jump_threshold_px: float = 12.0,
    output_path: Path | None = None,
) -> dict[str, Any]:
    steps = _iter_bundle_steps(bundle_root)
    samples: list[dict[str, object]] = []
    by_key: dict[str, dict[str, object]] = {}
    grouped_by_key: dict[str, dict[str, dict[str, object]]] = {}
    transport_stats: list[Mapping[str, object]] = []
    for index, step in enumerate(steps, start=1):
        raw_path, packet_path = _load_paths(step)
        if not packet_path.exists():
            continue
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        if not isinstance(packet, Mapping):
            continue
        slot = _slot_from_packet(packet, slot_id)
        if slot is None:
            samples.append({"index": index, "path": str(step), "valid": False, "reason": "slot_missing"})
            continue
        raw_frame = cv2.imread(str(raw_path)) if raw_path.exists() else None
        color_block_recomputed = _annotate_diagnostic_color_block(raw_frame, slot)
        target = _target_pixel(slot, packet)
        points = _candidate_points(slot)
        distances: dict[str, float] = {}
        group_label = _group_label_for_step(bundle_root, step)
        group_payload = grouped_by_key.setdefault(group_label, {})
        if target is not None:
            for key, point in points.items():
                distances[key] = float(math.hypot(point[0] - target[0], point[1] - target[1]))
                entry = by_key.setdefault(key, {"distances_px": [], "points": []})
                entry["distances_px"].append(distances[key])  # type: ignore[index,union-attr]
                entry["points"].append(point)  # type: ignore[index,union-attr]
                group_entry = group_payload.setdefault(key, {"distances_px": [], "points": []})
                group_entry["distances_px"].append(distances[key])  # type: ignore[index,union-attr]
                group_entry["points"].append(point)  # type: ignore[index,union-attr]
        transport = packet.get("camera_transport")
        if isinstance(transport, Mapping):
            transport_stats.append(transport)
        samples.append(
            {
                "index": index,
                "path": str(step),
                "group": group_label,
                "raw": str(raw_path),
                "packet": str(packet_path),
                "slot_id": slot.get("slot_id", slot.get("slot")),
                "target_pixel": None if target is None else [target[0], target[1]],
                "measurement_point": slot.get("measurement_point"),
                "center_distance_px": slot.get("center_distance_px"),
                "bbox": slot.get("bbox"),
                "area_px": slot.get("area_px"),
                "confidence": slot.get("confidence"),
                "points": {key: [value[0], value[1]] for key, value in points.items()},
                "distances_px": distances,
                "color_block_center_source": slot.get("color_block_center_source"),
                "color_block_recomputed_from_raw": bool(color_block_recomputed),
                "image_age_ms": packet.get("image_age_ms", packet.get("queue_age_ms")),
                "frame_pose_age_ms": packet.get("frame_pose_age_ms"),
            }
        )
    point_summary = _point_summary_for_payloads(by_key, jump_threshold_px=float(jump_threshold_px))
    group_summary, grouped_ranking = _grouped_point_summary(grouped_by_key, jump_threshold_px=float(jump_threshold_px))
    recommended_key = _rank_point_summary(point_summary)
    grouped_recommended_key = _rank_grouped_point_summary(grouped_ranking)
    recommended_mode = _measurement_mode_for_packet_key(recommended_key)
    report: dict[str, Any] = {
        "source": str(bundle_root),
        "slot_id": slot_id,
        "sample_count": len(samples),
        "jump_threshold_px": float(jump_threshold_px),
        "camera_contract": (
            "Read-only diagnostics only. This report does not start, restart, scan, or repair the JetMax camera sender."
        ),
        "point_summary": point_summary,
        "recommended_low_height_packet_key": recommended_key,
        "recommended_low_height_measurement_point": recommended_mode,
        "group_summary": group_summary,
        "grouped_point_ranking": grouped_ranking,
        "grouped_recommended_low_height_packet_key": grouped_recommended_key,
        "grouped_recommended_low_height_measurement_point": _measurement_mode_for_packet_key(grouped_recommended_key),
        "camera_transport_summary": _transport_summary(transport_stats),
        "samples": samples,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


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
    _annotate_diagnostic_color_block(frame, slot)

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
    color_block_center = _point(slot.get("color_block_center_f") or slot.get("color_block_center"))
    top_face_center = _point(slot.get("top_face_center_f") or slot.get("top_face_center"))
    grasp_pixel = _point(slot.get("grasp_pixel"))
    metrics: dict[str, Any] = {
        "raw": str(raw_path),
        "packet": str(packet_path),
        "pixel_center": pixel_center,
        "color_block_center": color_block_center,
        "top_face_center": top_face_center,
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
        ("color_block", color_block_center, (60, 180, 255), 28),
        ("top_face", top_face_center, (0, 255, 180), 42),
        ("grasp", grasp_pixel, (0, 255, 255), 56),
        ("bbox_center", bbox_center, (40, 220, 255), 70),
        ("rect_center", rect_center, (255, 160, 40), 84),
        ("fg_centroid", _point(metrics.get("foreground_centroid")), (0, 255, 0), 98),
        ("fg_core", _point(metrics.get("foreground_core_center")), (0, 0, 255), 112),
        ("green_hsv", _point(metrics.get("green_hsv_centroid")), (80, 255, 80), 126),
    ]
    for label, point, color, dy in labels:
        if point is not None:
            _draw_point(frame, point, color, label, dy=dy)

    summary_lines: list[str] = []
    if target is not None:
        for key in (
            "pixel_center",
            "color_block_center",
            "top_face_center",
            "grasp_pixel",
            "bbox_center",
            "rect_center",
            "foreground_centroid",
            "foreground_core_center",
            "green_hsv_centroid",
            "green_hsv_minrect_center",
        ):
            point = metrics.get(key)
            if point is not None:
                distance = float(math.dist(point, target))
                metrics[f"{key}_to_target_px"] = distance
                summary_lines.append(f"{key}={point}, d_target={distance:.2f}px")
    if pixel_center is not None:
        for key in (
            "bbox_center",
            "rect_center",
            "color_block_center",
            "top_face_center",
            "grasp_pixel",
            "foreground_centroid",
            "foreground_core_center",
            "green_hsv_centroid",
            "green_hsv_minrect_center",
        ):
            point = metrics.get(key)
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
    parser.add_argument("bundle", type=Path, help="Step directory containing raw.jpg and packet.json, raw.jpg path, or a debug bundle root.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="Summarize center-point stability across every raw.jpg/packet.json step under the bundle root.",
    )
    parser.add_argument("--slot-id", type=int, default=None)
    parser.add_argument("--jump-threshold-px", type=float, default=12.0)
    args = parser.parse_args(argv)
    if bool(args.sequence):
        metrics = diagnose_center_sequence(
            Path(args.bundle),
            slot_id=args.slot_id,
            jump_threshold_px=float(args.jump_threshold_px),
            output_path=args.output,
        )
    else:
        metrics = diagnose_centers(Path(args.bundle), output_path=args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
