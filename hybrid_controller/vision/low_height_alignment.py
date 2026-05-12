from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class LowHeightAlignmentSample:
    theta_deg: float
    radius_mm: float
    z_mm: float
    x_mm: float
    y_mm: float
    pixel_x: float
    pixel_y: float
    slot_id: int | None = None
    frame_id: int | None = None
    center_distance_px: float | None = None
    confidence: float | None = None
    area_px: float | None = None


@dataclass(frozen=True, slots=True)
class LowHeightResponseModel:
    matrix: tuple[tuple[float, float, float], tuple[float, float, float]]
    pixel_to_robot_jacobian: tuple[tuple[float, float], tuple[float, float]]
    target_pixel: tuple[float, float]
    target_robot_xy_mm: tuple[float, float]
    z_mm: float
    rms_pixel_error_px: float
    max_pixel_error_px: float
    condition_number: float
    used_sample_count: int
    rejected_sample_count: int

    def to_stage_model_payload(
        self,
        *,
        profile_id: str,
        image_size: tuple[int, int] = (640, 480),
        center_tolerance_px: float = 2.0,
        servo_gain: float = 0.45,
        max_attempts: int = 4,
        source: str = "low_height_response_calibration",
    ) -> dict[str, object]:
        return {
            "profile_id": str(profile_id),
            "stage": "confirm",
            "z_mm": float(self.z_mm),
            "image_size": [int(image_size[0]), int(image_size[1])],
            "pixel_to_delta": {
                "model": "affine",
                "matrix": [[float(value) for value in row] for row in self.matrix],
            },
            "servo": {
                "target_pixel": [float(self.target_pixel[0]), float(self.target_pixel[1])],
                "center_tolerance_px": float(center_tolerance_px),
                "gain": float(servo_gain),
                "max_attempts": int(max_attempts),
            },
            "residual": {
                "median_error_mm": None,
                "p95_error_mm": None,
                "max_error_mm": None,
                "points": [],
            },
            "samples_summary": {
                "source": str(source),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "used_sample_count": int(self.used_sample_count),
                "rejected_sample_count": int(self.rejected_sample_count),
                "rms_pixel_error_px": float(self.rms_pixel_error_px),
                "max_pixel_error_px": float(self.max_pixel_error_px),
                "condition_number": float(self.condition_number),
                "target_robot_xy_mm": [float(self.target_robot_xy_mm[0]), float(self.target_robot_xy_mm[1])],
                "pixel_to_robot_jacobian": [
                    [float(value) for value in row] for row in self.pixel_to_robot_jacobian
                ],
            },
            "limits": {"max_allowed_error_mm": 8.0},
        }


def sample_from_mapping(payload: Mapping[str, object]) -> LowHeightAlignmentSample:
    cyl = payload.get("pose_cyl") or payload.get("robot_cyl")
    xy = payload.get("pose_xy") or payload.get("robot_xy")
    pixel = (
        payload.get("pixel")
        or payload.get("geometry_center_f")
        or payload.get("geometry_center")
        or payload.get("pixel_center_f")
        or payload.get("pixel_center")
    )
    if not isinstance(cyl, (tuple, list)) or len(cyl) < 3:
        raise ValueError("sample missing pose_cyl")
    if not isinstance(xy, (tuple, list)) or len(xy) < 2:
        raise ValueError("sample missing pose_xy")
    if not isinstance(pixel, (tuple, list)) or len(pixel) < 2:
        raise ValueError("sample missing pixel")
    return LowHeightAlignmentSample(
        theta_deg=_finite_float(cyl[0], "theta_deg"),
        radius_mm=_finite_float(cyl[1], "radius_mm"),
        z_mm=_finite_float(cyl[2], "z_mm"),
        x_mm=_finite_float(xy[0], "x_mm"),
        y_mm=_finite_float(xy[1], "y_mm"),
        pixel_x=_finite_float(pixel[0], "pixel_x"),
        pixel_y=_finite_float(pixel[1], "pixel_y"),
        slot_id=_optional_int(payload.get("slot_id")),
        frame_id=_optional_int(payload.get("frame_id")),
        center_distance_px=_optional_float(payload.get("center_distance_px")),
        confidence=_optional_float(payload.get("confidence")),
        area_px=_optional_float(payload.get("area_px")),
    )


def fit_low_height_response_model(
    samples: Iterable[LowHeightAlignmentSample | Mapping[str, object]],
    *,
    target_pixel: tuple[float, float] = (320.0, 240.0),
    z_mm: float | None = None,
    min_samples: int = 4,
    max_condition_number: float = 250.0,
) -> LowHeightResponseModel:
    parsed = [
        item if isinstance(item, LowHeightAlignmentSample) else sample_from_mapping(item)
        for item in samples
    ]
    parsed = [item for item in parsed if _sample_is_finite(item)]
    if len(parsed) < max(3, int(min_samples)):
        raise ValueError("low_height_alignment_requires_more_samples")

    xy = np.array([[item.x_mm, item.y_mm] for item in parsed], dtype=np.float64)
    px = np.array([[item.pixel_x, item.pixel_y] for item in parsed], dtype=np.float64)

    used_mask, affine, residuals = _fit_pixel_response_with_outlier_rejection(xy, px)
    used_count = int(np.count_nonzero(used_mask))
    if used_count < max(3, int(min_samples)):
        raise ValueError("low_height_alignment_not_enough_inlier_samples")

    condition_number = float(np.linalg.cond(affine))
    if not math.isfinite(condition_number) or condition_number > float(max_condition_number):
        raise ValueError(f"low_height_alignment_jacobian_ill_conditioned:{condition_number:.2f}")

    inverse = np.linalg.pinv(affine)
    pixel_to_delta = -inverse.T
    target = np.array([float(target_pixel[0]), float(target_pixel[1])], dtype=np.float64)
    xy_used = xy[used_mask]
    px_used = px[used_mask]
    xy_mean = np.mean(xy_used, axis=0)
    px_mean = np.mean(px_used, axis=0)
    target_robot_xy = xy_mean + (target - px_mean) @ inverse
    if z_mm is None:
        z_value = float(np.median(np.array([item.z_mm for item in parsed], dtype=np.float64)))
    else:
        z_value = float(z_mm)

    matrix = (
        (float(pixel_to_delta[0, 0]), float(pixel_to_delta[0, 1]), 0.0),
        (float(pixel_to_delta[1, 0]), float(pixel_to_delta[1, 1]), 0.0),
    )
    residuals_used = residuals[used_mask]
    return LowHeightResponseModel(
        matrix=matrix,
        pixel_to_robot_jacobian=(
            (float(affine[0, 0]), float(affine[0, 1])),
            (float(affine[1, 0]), float(affine[1, 1])),
        ),
        target_pixel=(float(target[0]), float(target[1])),
        target_robot_xy_mm=(float(target_robot_xy[0]), float(target_robot_xy[1])),
        z_mm=z_value,
        rms_pixel_error_px=float(math.sqrt(float(np.mean(np.square(residuals_used))))),
        max_pixel_error_px=float(np.max(residuals_used)),
        condition_number=condition_number,
        used_sample_count=used_count,
        rejected_sample_count=int(len(parsed) - used_count),
    )


def merge_confirm_stage_model(
    profile_payload: Mapping[str, object],
    stage_payload: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(profile_payload)
    stage_models_raw = merged.get("stage_models")
    stage_models = dict(stage_models_raw) if isinstance(stage_models_raw, Mapping) else {}
    stage_models["confirm"] = dict(stage_payload)
    merged["stage_models"] = stage_models
    return merged


def _fit_pixel_response_with_outlier_rejection(
    xy: np.ndarray,
    px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones((xy.shape[0],), dtype=bool)
    affine, residuals = _fit_pixel_response(xy, px, mask)
    if xy.shape[0] < 6:
        return mask, affine, residuals
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    threshold = max(3.0, median + 3.5 * max(mad, 1e-6))
    next_mask = residuals <= threshold
    if int(np.count_nonzero(next_mask)) < 4 or np.array_equal(next_mask, mask):
        return mask, affine, residuals
    affine, residuals = _fit_pixel_response(xy, px, next_mask)
    return next_mask, affine, residuals


def _fit_pixel_response(
    xy: np.ndarray,
    px: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xy_used = xy[mask]
    px_used = px[mask]
    xy_center = np.mean(xy_used, axis=0)
    px_center = np.mean(px_used, axis=0)
    xy_delta = xy_used - xy_center
    px_delta = px_used - px_center
    affine, *_ = np.linalg.lstsq(xy_delta, px_delta, rcond=None)
    predicted = (xy - xy_center) @ affine + px_center
    residuals = np.linalg.norm(predicted - px, axis=1)
    if not np.all(np.isfinite(affine)):
        raise ValueError("low_height_alignment_jacobian_non_finite")
    return affine, residuals


def _sample_is_finite(sample: LowHeightAlignmentSample) -> bool:
    values = (
        sample.theta_deg,
        sample.radius_mm,
        sample.z_mm,
        sample.x_mm,
        sample.y_mm,
        sample.pixel_x,
        sample.pixel_y,
    )
    return all(math.isfinite(float(value)) for value in values)


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
