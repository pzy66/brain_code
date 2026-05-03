from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VisionMappingResult:
    delta_xy_mm: tuple[float, float]
    undistorted_pixel: tuple[float, float]
    estimated_error_mm: float | None
    profile_id: str


def _as_optional_matrix(value: object, *, rows: int | None = None, cols: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.array(value, dtype=np.float64)
    if rows is not None and cols is not None:
        try:
            array = array.reshape(rows, cols)
        except ValueError as error:
            raise ValueError(f"matrix cannot be reshaped to {rows}x{cols}") from error
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix contains non-finite values")
    return array


def _as_float_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return (x, y)


def _as_image_size(value: object) -> tuple[int, int] | None:
    pair = _as_float_pair(value)
    if pair is None:
        return None
    width = int(round(pair[0]))
    height = int(round(pair[1]))
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


@dataclass(frozen=True, slots=True)
class VisionCalibrationProfile:
    profile_id: str
    image_size: tuple[int, int] | None = None
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    target_pixel: tuple[float, float] | None = None
    model: str = "affine"
    pixel_to_delta_matrix: np.ndarray | None = None
    median_error_mm: float | None = None
    p95_error_mm: float | None = None
    max_error_mm: float | None = None
    max_allowed_error_mm: float = 6.0
    center_tolerance_px: float = 8.0
    servo_gain: float = 0.8
    max_servo_attempts: int = 3
    residual_points: tuple[tuple[float, float, float], ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VisionCalibrationProfile":
        if not isinstance(payload, dict):
            raise ValueError("Vision calibration profile must be a dict.")

        profile_id = str(payload.get("profile_id") or payload.get("id") or "vision-profile").strip()
        image_size = _as_image_size(payload.get("image_size") or payload.get("frame_size"))
        camera_matrix = _as_optional_matrix(_first_present(payload, ("K", "camera_matrix")), rows=3, cols=3)
        dist_value = _first_present(payload, ("D", "dist_coeffs", "distCoeffs", "distortion_coefficients"))
        dist_coeffs = None if dist_value is None else np.array(dist_value, dtype=np.float64).reshape(-1, 1)
        if dist_coeffs is not None and not np.all(np.isfinite(dist_coeffs)):
            raise ValueError("distortion coefficients contain non-finite values")

        mapping = payload.get("pixel_to_delta") or payload.get("pixel_to_delta_model") or {}
        if not isinstance(mapping, dict):
            mapping = {}
        model = str(mapping.get("model") or payload.get("model") or "affine").strip().lower()
        matrix_value = mapping.get("matrix") if "matrix" in mapping else payload.get("pixel_to_delta_matrix")
        pixel_to_delta_matrix = None
        if matrix_value is not None:
            matrix_raw = np.array(matrix_value, dtype=np.float64)
            if model == "homography":
                pixel_to_delta_matrix = matrix_raw.reshape(3, 3)
            else:
                model = "affine"
                pixel_to_delta_matrix = matrix_raw.reshape(2, 3)
            if not np.all(np.isfinite(pixel_to_delta_matrix)):
                raise ValueError("pixel-to-delta matrix contains non-finite values")

        residual = payload.get("residual") or payload.get("validation") or {}
        if not isinstance(residual, dict):
            residual = {}
        limits = payload.get("limits") or {}
        if not isinstance(limits, dict):
            limits = {}
        servo = payload.get("servo") or {}
        if not isinstance(servo, dict):
            servo = {}
        target_pixel = _as_float_pair(_first_present(servo, ("target_pixel", "suction_pick_pixel", "alignment_target_pixel")))
        if target_pixel is None:
            target_pixel = _as_float_pair(
                _first_present(
                    payload,
                    (
                        "target_pixel",
                        "suction_pick_pixel",
                        "alignment_target_pixel",
                        "vision_pick_target_pixel",
                    ),
                )
            )

        residual_points_raw = payload.get("residual_points") or residual.get("points") or []
        residual_points: list[tuple[float, float, float]] = []
        if isinstance(residual_points_raw, list):
            for item in residual_points_raw:
                if not isinstance(item, dict):
                    continue
                pixel = _as_float_pair(item.get("pixel") or item.get("undistorted_pixel"))
                try:
                    error_mm = float(item.get("error_mm"))
                except (TypeError, ValueError):
                    continue
                if pixel is None or not math.isfinite(error_mm):
                    continue
                residual_points.append((float(pixel[0]), float(pixel[1]), float(error_mm)))

        return cls(
            profile_id=profile_id or "vision-profile",
            image_size=image_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            target_pixel=target_pixel,
            model=model if model in {"affine", "homography"} else "affine",
            pixel_to_delta_matrix=pixel_to_delta_matrix,
            median_error_mm=_optional_float(residual.get("median_error_mm")),
            p95_error_mm=_optional_float(residual.get("p95_error_mm")),
            max_error_mm=_optional_float(residual.get("max_error_mm")),
            max_allowed_error_mm=float(limits.get("max_allowed_error_mm", payload.get("max_allowed_error_mm", 6.0))),
            center_tolerance_px=float(servo.get("center_tolerance_px", payload.get("center_tolerance_px", 8.0))),
            servo_gain=max(0.05, min(1.0, float(servo.get("gain", payload.get("servo_gain", 0.8))))),
            max_servo_attempts=max(1, int(servo.get("max_attempts", payload.get("max_servo_attempts", 3)))),
            residual_points=tuple(residual_points),
        )

    @classmethod
    def load(cls, path: str | Path) -> "VisionCalibrationProfile":
        profile_path = Path(path)
        with profile_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, object]:
        mapping: dict[str, object] = {"model": self.model}
        if self.pixel_to_delta_matrix is not None:
            mapping["matrix"] = self.pixel_to_delta_matrix.tolist()
        return {
            "profile_id": self.profile_id,
            "image_size": None if self.image_size is None else [int(self.image_size[0]), int(self.image_size[1])],
            "K": None if self.camera_matrix is None else self.camera_matrix.tolist(),
            "D": None if self.dist_coeffs is None else self.dist_coeffs.reshape(-1).tolist(),
            "pixel_to_delta": mapping,
            "residual": {
                "median_error_mm": self.median_error_mm,
                "p95_error_mm": self.p95_error_mm,
                "max_error_mm": self.max_error_mm,
                "points": [
                    {"pixel": [float(x), float(y)], "error_mm": float(error)}
                    for x, y, error in self.residual_points
                ],
            },
            "limits": {"max_allowed_error_mm": float(self.max_allowed_error_mm)},
            "servo": {
                "target_pixel": None if self.target_pixel is None else [float(self.target_pixel[0]), float(self.target_pixel[1])],
                "center_tolerance_px": float(self.center_tolerance_px),
                "gain": float(self.servo_gain),
                "max_attempts": int(self.max_servo_attempts),
            },
        }

    @property
    def has_pixel_to_delta_model(self) -> bool:
        return self.pixel_to_delta_matrix is not None

    def is_valid_for_image_size(self, frame_size: tuple[int, int] | None) -> bool:
        if self.image_size is None or frame_size is None:
            return True
        return (int(self.image_size[0]), int(self.image_size[1])) == (int(frame_size[0]), int(frame_size[1]))

    def undistort_pixel(self, pixel: tuple[float, float]) -> tuple[float, float]:
        x = float(pixel[0])
        y = float(pixel[1])
        if self.camera_matrix is None or self.dist_coeffs is None or self.dist_coeffs.size == 0:
            return (x, y)
        points = np.array([[[x, y]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(points, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        return (float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1]))

    def estimate_error_mm(self, pixel: tuple[float, float]) -> float | None:
        if self.residual_points:
            px, py = float(pixel[0]), float(pixel[1])
            nearest = min(
                self.residual_points,
                key=lambda item: math.hypot(float(item[0]) - px, float(item[1]) - py),
            )
            return float(nearest[2])
        if self.p95_error_mm is not None:
            return float(self.p95_error_mm)
        if self.max_error_mm is not None:
            return float(self.max_error_mm)
        if self.median_error_mm is not None:
            return float(self.median_error_mm)
        return None

    def _map_undistorted_pixel_to_delta(self, undistorted: tuple[float, float]) -> tuple[float, float]:
        if self.pixel_to_delta_matrix is None:
            raise ValueError("calibration_profile_missing_pixel_to_delta_model")
        x, y = float(undistorted[0]), float(undistorted[1])
        vector = np.array([x, y, 1.0], dtype=np.float64)
        if self.model == "homography":
            mapped = self.pixel_to_delta_matrix @ vector
            denom = float(mapped[2])
            if abs(denom) < 1e-9:
                raise ValueError("calibration_profile_homography_degenerate")
            return (float(mapped[0] / denom), float(mapped[1] / denom))
        mapped = self.pixel_to_delta_matrix @ vector
        return (float(mapped[0]), float(mapped[1]))

    def map_pixel_to_delta(
        self,
        pixel: tuple[float, float],
        *,
        frame_size: tuple[int, int] | None = None,
        target_pixel: tuple[float, float] | None = None,
    ) -> VisionMappingResult:
        if not self.is_valid_for_image_size(frame_size):
            raise ValueError("calibration_profile_image_size_mismatch")
        if self.pixel_to_delta_matrix is None:
            raise ValueError("calibration_profile_missing_pixel_to_delta_model")

        undistorted = self.undistort_pixel(pixel)
        delta_xy = self._map_undistorted_pixel_to_delta(undistorted)
        effective_target = target_pixel if target_pixel is not None else self.target_pixel
        target_error_mm: float | None = None
        if effective_target is not None:
            target_undistorted = self.undistort_pixel(effective_target)
            target_delta_xy = self._map_undistorted_pixel_to_delta(target_undistorted)
            delta_xy = (
                float(delta_xy[0] - target_delta_xy[0]),
                float(delta_xy[1] - target_delta_xy[1]),
            )
            target_error_mm = self.estimate_error_mm(target_undistorted)
        if not all(math.isfinite(value) for value in delta_xy):
            raise ValueError("calibration_profile_mapping_non_finite")
        estimated_error_mm = self.estimate_error_mm(undistorted)
        if target_error_mm is not None:
            estimated_error_mm = (
                target_error_mm
                if estimated_error_mm is None
                else max(float(estimated_error_mm), float(target_error_mm))
            )
        return VisionMappingResult(
            delta_xy_mm=delta_xy,
            undistorted_pixel=undistorted,
            estimated_error_mm=estimated_error_mm,
            profile_id=self.profile_id,
        )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
