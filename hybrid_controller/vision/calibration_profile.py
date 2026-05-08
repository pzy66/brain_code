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


@dataclass(frozen=True, slots=True)
class ResidualGrid:
    model: str = "grid"
    x_values: tuple[float, ...] = ()
    y_values: tuple[float, ...] = ()
    correction_dx_mm: np.ndarray | None = None
    correction_dy_mm: np.ndarray | None = None
    error_mm: np.ndarray | None = None
    points: tuple[tuple[float, float, float, float, float | None], ...] = ()
    power: float = 2.0
    max_neighbors: int = 6
    max_distance_px: float | None = None

    @classmethod
    def from_payload(cls, payload: object) -> "ResidualGrid | None":
        if not isinstance(payload, dict):
            return None
        model = str(payload.get("model") or "grid").strip().lower()
        points = _parse_residual_correction_points(payload.get("points") or [])
        x_values = _as_float_tuple(payload.get("x_values") or payload.get("xs"))
        y_values = _as_float_tuple(payload.get("y_values") or payload.get("ys"))
        correction_dx = _as_optional_grid(payload.get("correction_dx_mm") or payload.get("dx_mm"))
        correction_dy = _as_optional_grid(payload.get("correction_dy_mm") or payload.get("dy_mm"))
        error_grid = _as_optional_grid(payload.get("error_mm"))
        if x_values and y_values and correction_dx is not None and correction_dy is not None:
            expected_shape = (len(y_values), len(x_values))
            if correction_dx.shape != expected_shape or correction_dy.shape != expected_shape:
                raise ValueError("residual_grid correction shape must match y_values x x_values")
            if error_grid is not None and error_grid.shape != expected_shape:
                raise ValueError("residual_grid error shape must match y_values x x_values")
            return cls(
                model="grid",
                x_values=tuple(x_values),
                y_values=tuple(y_values),
                correction_dx_mm=correction_dx,
                correction_dy_mm=correction_dy,
                error_mm=error_grid,
                points=tuple(points),
                power=max(0.1, float(payload.get("power", 2.0))),
                max_neighbors=max(1, int(payload.get("max_neighbors", 6))),
                max_distance_px=_optional_float(payload.get("max_distance_px")),
            )
        if points:
            return cls(
                model="nearest" if model == "nearest" else "idw",
                points=tuple(points),
                power=max(0.1, float(payload.get("power", 2.0))),
                max_neighbors=max(1, int(payload.get("max_neighbors", 6))),
                max_distance_px=_optional_float(payload.get("max_distance_px")),
            )
        return None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"model": self.model}
        if self.x_values and self.y_values and self.correction_dx_mm is not None and self.correction_dy_mm is not None:
            payload.update(
                {
                    "x_values": [float(v) for v in self.x_values],
                    "y_values": [float(v) for v in self.y_values],
                    "correction_dx_mm": self.correction_dx_mm.tolist(),
                    "correction_dy_mm": self.correction_dy_mm.tolist(),
                }
            )
            if self.error_mm is not None:
                payload["error_mm"] = self.error_mm.tolist()
        if self.points:
            payload["points"] = [
                {
                    "pixel": [float(x), float(y)],
                    "correction_mm": [float(dx), float(dy)],
                    "error_mm": None if error is None else float(error),
                }
                for x, y, dx, dy, error in self.points
            ]
            payload["power"] = float(self.power)
            payload["max_neighbors"] = int(self.max_neighbors)
        if self.max_distance_px is not None:
            payload["max_distance_px"] = float(self.max_distance_px)
        return payload

    def correction_and_error(self, pixel: tuple[float, float]) -> tuple[float, float, float | None]:
        x = float(pixel[0])
        y = float(pixel[1])
        if self.x_values and self.y_values and self.correction_dx_mm is not None and self.correction_dy_mm is not None:
            correction_x = self._bilinear(self.correction_dx_mm, x, y)
            correction_y = self._bilinear(self.correction_dy_mm, x, y)
            error = None if self.error_mm is None else self._bilinear(self.error_mm, x, y)
            return (float(correction_x), float(correction_y), None if error is None else float(error))
        if self.points:
            return self._idw(x, y)
        return (0.0, 0.0, None)

    def _bilinear(self, grid: np.ndarray, x: float, y: float) -> float:
        xs = self.x_values
        ys = self.y_values
        if len(xs) == 1 and len(ys) == 1:
            return float(grid[0, 0])
        if x < xs[0] or x > xs[-1] or y < ys[0] or y > ys[-1]:
            raise ValueError("calibration_profile_residual_grid_out_of_bounds")
        ix = _lower_interval_index(xs, x)
        iy = _lower_interval_index(ys, y)
        x0, x1 = xs[ix], xs[ix + 1]
        y0, y1 = ys[iy], ys[iy + 1]
        tx = 0.0 if abs(x1 - x0) < 1e-9 else (x - x0) / (x1 - x0)
        ty = 0.0 if abs(y1 - y0) < 1e-9 else (y - y0) / (y1 - y0)
        v00 = float(grid[iy, ix])
        v10 = float(grid[iy, ix + 1])
        v01 = float(grid[iy + 1, ix])
        v11 = float(grid[iy + 1, ix + 1])
        return float((1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11)

    def _idw(self, x: float, y: float) -> tuple[float, float, float | None]:
        distances = [
            (math.hypot(float(px) - x, float(py) - y), dx, dy, error)
            for px, py, dx, dy, error in self.points
        ]
        distances.sort(key=lambda item: item[0])
        nearest = float(distances[0][0])
        if self.max_distance_px is not None and nearest > float(self.max_distance_px):
            raise ValueError("calibration_profile_residual_grid_out_of_bounds")
        if nearest <= 1e-9 or self.model == "nearest":
            _, dx, dy, error = distances[0]
            return (float(dx), float(dy), None if error is None else float(error))
        selected = distances[: max(1, int(self.max_neighbors))]
        weights = [1.0 / max(1e-9, float(distance) ** float(self.power)) for distance, *_ in selected]
        total = sum(weights)
        correction_x = sum(weight * float(item[1]) for weight, item in zip(weights, selected)) / total
        correction_y = sum(weight * float(item[2]) for weight, item in zip(weights, selected)) / total
        error_values = [(weight, item[3]) for weight, item in zip(weights, selected) if item[3] is not None]
        error = None
        if error_values:
            error = sum(weight * float(value) for weight, value in error_values) / sum(weight for weight, _ in error_values)
        return (float(correction_x), float(correction_y), None if error is None else float(error))


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


def _as_float_tuple(value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    values: list[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return ()
        if not math.isfinite(number):
            return ()
        values.append(number)
    return tuple(values)


def _as_optional_grid(value: object) -> np.ndarray | None:
    if value is None:
        return None
    array = np.array(value, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("grid contains invalid values")
    return array


def _parse_residual_correction_points(value: object) -> list[tuple[float, float, float, float, float | None]]:
    points: list[tuple[float, float, float, float, float | None]] = []
    if not isinstance(value, list):
        return points
    for item in value:
        if not isinstance(item, dict):
            continue
        pixel = _as_float_pair(item.get("pixel") or item.get("undistorted_pixel"))
        correction = _as_float_pair(item.get("correction_mm") or item.get("correction"))
        if correction is None:
            correction = _as_float_pair((item.get("correction_dx_mm"), item.get("correction_dy_mm")))
        if pixel is None or correction is None:
            continue
        error_mm = _optional_float(item.get("error_mm"))
        points.append((float(pixel[0]), float(pixel[1]), float(correction[0]), float(correction[1]), error_mm))
    return points


def _parse_workspace_polygon(payload: object) -> tuple[tuple[float, float], ...]:
    raw_polygon: object = None
    if isinstance(payload, dict):
        raw_polygon = (
            payload.get("undistorted_pixel_polygon")
            or payload.get("pixel_polygon")
            or payload.get("polygon")
        )
    elif isinstance(payload, list):
        raw_polygon = payload
    if not isinstance(raw_polygon, list):
        return ()
    points: list[tuple[float, float]] = []
    for item in raw_polygon:
        pair = _as_float_pair(item)
        if pair is not None:
            points.append((float(pair[0]), float(pair[1])))
    return tuple(points) if len(points) >= 3 else ()


def _point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    if len(polygon) < 3:
        return True
    contour = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= -1e-6


def _lower_interval_index(values: tuple[float, ...], value: float) -> int:
    if len(values) < 2:
        return 0
    if value >= values[-1]:
        return len(values) - 2
    for index in range(len(values) - 1):
        if values[index] <= value <= values[index + 1]:
            return index
    return 0


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


def _merge_stage_payload(parent: dict[str, Any], child: dict[str, Any], stage_name: str) -> dict[str, Any]:
    merged = dict(child)
    merged.setdefault("stage", str(stage_name).strip().lower())
    for key in (
        "image_size",
        "frame_size",
        "K",
        "camera_matrix",
        "D",
        "dist_coeffs",
        "distCoeffs",
        "distortion_coefficients",
        "hand_eye",
        "T_camera_to_tool",
        "pixel_to_delta",
        "pixel_to_delta_model",
        "pixel_to_delta_matrix",
        "model",
        "residual",
        "validation",
        "residual_points",
        "residual_grid",
        "valid_workspace",
        "created_at",
    ):
        if key not in merged and key in parent:
            merged[key] = parent[key]
    parent_servo = parent.get("servo")
    child_servo = merged.get("servo")
    if isinstance(parent_servo, dict):
        servo = dict(parent_servo)
        if isinstance(child_servo, dict):
            servo.update(child_servo)
        merged["servo"] = servo
    parent_limits = parent.get("limits")
    child_limits = merged.get("limits")
    if isinstance(parent_limits, dict):
        limits = dict(parent_limits)
        if isinstance(child_limits, dict):
            limits.update(child_limits)
        merged["limits"] = limits
    merged.pop("stage_models", None)
    return merged


@dataclass(frozen=True, slots=True)
class VisionCalibrationProfile:
    profile_id: str
    stage: str = ""
    z_mm: float | None = None
    image_size: tuple[int, int] | None = None
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    hand_eye_camera_to_tool: np.ndarray | None = None
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
    residual_grid: ResidualGrid | None = None
    valid_workspace: tuple[tuple[float, float], ...] = ()
    samples_summary: dict[str, Any] | None = None
    created_at: str = ""
    stage_models: dict[str, "VisionCalibrationProfile"] | None = None

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
        hand_eye_payload = payload.get("hand_eye") or {}
        if not isinstance(hand_eye_payload, dict):
            hand_eye_payload = {}
        hand_eye_matrix_value = hand_eye_payload.get("T_camera_to_tool")
        if hand_eye_matrix_value is None:
            hand_eye_matrix_value = payload.get("T_camera_to_tool")
        hand_eye_camera_to_tool = _as_optional_matrix(
            hand_eye_matrix_value,
            rows=4,
            cols=4,
        )

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

        residual_grid = ResidualGrid.from_payload(payload.get("residual_grid") or residual.get("grid"))
        valid_workspace = _parse_workspace_polygon(payload.get("valid_workspace") or limits.get("valid_workspace"))
        samples_summary = payload.get("samples_summary")
        if not isinstance(samples_summary, dict):
            samples_summary = None
        stage_models_payload = payload.get("stage_models") or {}
        stage_models: dict[str, VisionCalibrationProfile] = {}
        if isinstance(stage_models_payload, dict):
            for stage_name, stage_payload in stage_models_payload.items():
                if not isinstance(stage_payload, dict):
                    continue
                merged_payload = _merge_stage_payload(payload, stage_payload, str(stage_name))
                stage_models[str(stage_name).strip().lower()] = cls.from_dict(merged_payload)

        return cls(
            profile_id=profile_id or "vision-profile",
            stage=str(payload.get("stage") or "").strip().lower(),
            z_mm=_optional_float(payload.get("z_mm")),
            image_size=image_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            hand_eye_camera_to_tool=hand_eye_camera_to_tool,
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
            residual_grid=residual_grid,
            valid_workspace=valid_workspace,
            samples_summary=samples_summary,
            created_at=str(payload.get("created_at") or ""),
            stage_models=stage_models or None,
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
            "stage": self.stage,
            "z_mm": self.z_mm,
            "image_size": None if self.image_size is None else [int(self.image_size[0]), int(self.image_size[1])],
            "K": None if self.camera_matrix is None else self.camera_matrix.tolist(),
            "D": None if self.dist_coeffs is None else self.dist_coeffs.reshape(-1).tolist(),
            "hand_eye": {
                "T_camera_to_tool": None
                if self.hand_eye_camera_to_tool is None
                else self.hand_eye_camera_to_tool.tolist()
            },
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
            "residual_grid": None if self.residual_grid is None else self.residual_grid.to_dict(),
            "valid_workspace": (
                None
                if not self.valid_workspace
                else {"undistorted_pixel_polygon": [[float(x), float(y)] for x, y in self.valid_workspace]}
            ),
            "samples_summary": self.samples_summary,
            "created_at": self.created_at,
            "servo": {
                "target_pixel": None if self.target_pixel is None else [float(self.target_pixel[0]), float(self.target_pixel[1])],
                "center_tolerance_px": float(self.center_tolerance_px),
                "gain": float(self.servo_gain),
                "max_attempts": int(self.max_servo_attempts),
            },
            "stage_models": (
                {}
                if not self.stage_models
                else {str(name): model.to_dict() for name, model in self.stage_models.items()}
            ),
        }

    @property
    def has_pixel_to_delta_model(self) -> bool:
        return self.pixel_to_delta_matrix is not None

    @property
    def has_stage_models(self) -> bool:
        return bool(self.stage_models)

    def model_for_stage(
        self,
        stage: str | None = None,
        *,
        z_mm: float | None = None,
        allow_fallback: bool = True,
    ) -> "VisionCalibrationProfile":
        models = self.stage_models or {}
        if not models:
            if allow_fallback:
                return self
            raise ValueError("calibration_profile_stage_model_unavailable")
        key = str(stage or "").strip().lower()
        if key and key in models:
            return models[key]
        if key:
            if allow_fallback:
                return self
            raise ValueError("calibration_profile_stage_model_unavailable")
        if z_mm is not None:
            candidates = [model for model in models.values() if model.z_mm is not None]
            if candidates:
                return min(candidates, key=lambda model: abs(float(model.z_mm) - float(z_mm)))
        if allow_fallback:
            return self
        raise ValueError("calibration_profile_stage_model_unavailable")

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
        if self.residual_grid is not None:
            try:
                _, _, error = self.residual_grid.correction_and_error(pixel)
            except ValueError:
                return float("inf")
            if error is not None:
                return float(error)
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

    def _ensure_valid_undistorted_pixel(self, undistorted: tuple[float, float]) -> None:
        if not _point_in_polygon(undistorted, self.valid_workspace):
            raise ValueError("calibration_profile_point_outside_valid_workspace")

    def _map_undistorted_pixel_to_delta(self, undistorted: tuple[float, float]) -> tuple[float, float]:
        if self.pixel_to_delta_matrix is None:
            raise ValueError("calibration_profile_missing_pixel_to_delta_model")
        self._ensure_valid_undistorted_pixel(undistorted)
        x, y = float(undistorted[0]), float(undistorted[1])
        vector = np.array([x, y, 1.0], dtype=np.float64)
        if self.model == "homography":
            mapped = self.pixel_to_delta_matrix @ vector
            denom = float(mapped[2])
            if abs(denom) < 1e-9:
                raise ValueError("calibration_profile_homography_degenerate")
            delta_x, delta_y = float(mapped[0] / denom), float(mapped[1] / denom)
        else:
            mapped = self.pixel_to_delta_matrix @ vector
            delta_x, delta_y = float(mapped[0]), float(mapped[1])
        if self.residual_grid is not None:
            correction_x, correction_y, _ = self.residual_grid.correction_and_error(undistorted)
            delta_x += float(correction_x)
            delta_y += float(correction_y)
        return (float(delta_x), float(delta_y))

    def map_pixel_to_delta(
        self,
        pixel: tuple[float, float],
        *,
        frame_size: tuple[int, int] | None = None,
        target_pixel: tuple[float, float] | None = None,
        stage: str | None = None,
        z_mm: float | None = None,
        allow_stage_fallback: bool = True,
    ) -> VisionMappingResult:
        if stage is not None or z_mm is not None:
            stage_model = self.model_for_stage(stage, z_mm=z_mm, allow_fallback=allow_stage_fallback)
            if stage_model is not self:
                return stage_model.map_pixel_to_delta(
                    pixel,
                    frame_size=frame_size,
                    target_pixel=target_pixel,
                    allow_stage_fallback=allow_stage_fallback,
                )
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
