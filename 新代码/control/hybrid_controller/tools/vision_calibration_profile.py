from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


def _read_samples(path: Path) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                sample = {
                    "pixel_x": float(row["pixel_x"]),
                    "pixel_y": float(row["pixel_y"]),
                    "delta_x_mm": float(row["delta_x_mm"]),
                    "delta_y_mm": float(row["delta_y_mm"]),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "CSV must include pixel_x,pixel_y,delta_x_mm,delta_y_mm numeric columns."
                ) from error
            if not all(math.isfinite(value) for value in sample.values()):
                continue
            samples.append(sample)
    if len(samples) < 4:
        raise ValueError("Need at least 4 calibration samples.")
    return samples


def _load_json_array(path_text: str) -> object | None:
    if not str(path_text or "").strip():
        return None
    return json.loads(Path(path_text).read_text(encoding="utf-8"))


def _prepare_samples(
    samples: list[dict[str, float]],
    *,
    camera_matrix: object | None,
    dist_coeffs: object | None,
) -> list[dict[str, float]]:
    prepared = [dict(sample) for sample in samples]
    if camera_matrix is None or dist_coeffs is None:
        for sample in prepared:
            sample["fit_pixel_x"] = float(sample["pixel_x"])
            sample["fit_pixel_y"] = float(sample["pixel_y"])
        return prepared

    import cv2

    k_mat = np.array(camera_matrix, dtype=np.float64).reshape(3, 3)
    d_vec = np.array(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    points = np.array([[[sample["pixel_x"], sample["pixel_y"]]] for sample in prepared], dtype=np.float64)
    undistorted = cv2.undistortPoints(points, k_mat, d_vec, P=k_mat)
    for sample, point in zip(prepared, undistorted[:, 0, :]):
        sample["fit_pixel_x"] = float(point[0])
        sample["fit_pixel_y"] = float(point[1])
    return prepared


def _split_train_validation(
    samples: list[dict[str, float]],
    *,
    validation_ratio: float,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    ratio = max(0.0, min(0.5, float(validation_ratio)))
    if ratio <= 0.0 or len(samples) < 8:
        return list(samples), []
    stride = max(3, int(round(1.0 / ratio)))
    validation = [sample for index, sample in enumerate(samples) if (index + 1) % stride == 0]
    train = [sample for index, sample in enumerate(samples) if (index + 1) % stride != 0]
    if len(train) < 4 or len(validation) < 1:
        return list(samples), []
    return train, validation


def _fit_affine(samples: list[dict[str, float]]) -> np.ndarray:
    a_rows: list[list[float]] = []
    bx: list[float] = []
    by: list[float] = []
    for sample in samples:
        a_rows.append([sample["fit_pixel_x"], sample["fit_pixel_y"], 1.0])
        bx.append(sample["delta_x_mm"])
        by.append(sample["delta_y_mm"])
    a = np.array(a_rows, dtype=np.float64)
    mx, *_ = np.linalg.lstsq(a, np.array(bx, dtype=np.float64), rcond=None)
    my, *_ = np.linalg.lstsq(a, np.array(by, dtype=np.float64), rcond=None)
    return np.vstack([mx, my])


def _fit_homography(samples: list[dict[str, float]]) -> np.ndarray:
    import cv2

    src = np.array([[sample["fit_pixel_x"], sample["fit_pixel_y"]] for sample in samples], dtype=np.float64)
    dst = np.array([[sample["delta_x_mm"], sample["delta_y_mm"]] for sample in samples], dtype=np.float64)
    matrix, _ = cv2.findHomography(src, dst, method=0)
    if matrix is None:
        raise ValueError("cv2.findHomography failed.")
    return np.array(matrix, dtype=np.float64).reshape(3, 3)


def _predict(matrix: np.ndarray, model: str, x: float, y: float) -> tuple[float, float]:
    vector = np.array([float(x), float(y), 1.0], dtype=np.float64)
    if model == "homography":
        mapped = matrix @ vector
        denom = float(mapped[2])
        if abs(denom) < 1e-9:
            return (float("nan"), float("nan"))
        return (float(mapped[0] / denom), float(mapped[1] / denom))
    mapped = matrix @ vector
    return (float(mapped[0]), float(mapped[1]))


def _build_residual_points(samples: list[dict[str, float]], matrix: np.ndarray, model: str) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for sample in samples:
        pred_x, pred_y = _predict(matrix, model, sample["fit_pixel_x"], sample["fit_pixel_y"])
        correction_x = float(sample["delta_x_mm"] - pred_x)
        correction_y = float(sample["delta_y_mm"] - pred_y)
        base_error = math.hypot(correction_x, correction_y)
        points.append(
            {
                "pixel": [float(sample["fit_pixel_x"]), float(sample["fit_pixel_y"])],
                "correction_mm": [correction_x, correction_y],
                "error_mm": float(base_error),
            }
        )
    return points


def _apply_idw_residual(
    residual_points: list[dict[str, object]],
    x: float,
    y: float,
    *,
    max_neighbors: int,
    power: float = 2.0,
) -> tuple[float, float]:
    distances: list[tuple[float, float, float]] = []
    for point in residual_points:
        px, py = [float(value) for value in point["pixel"]]
        dx, dy = [float(value) for value in point["correction_mm"]]
        distance = math.hypot(px - float(x), py - float(y))
        if distance <= 1e-9:
            return (dx, dy)
        distances.append((distance, dx, dy))
    distances.sort(key=lambda item: item[0])
    selected = distances[: max(1, int(max_neighbors))]
    weights = [1.0 / max(1e-9, float(distance) ** float(power)) for distance, *_ in selected]
    total = sum(weights)
    return (
        float(sum(weight * item[1] for weight, item in zip(weights, selected)) / total),
        float(sum(weight * item[2] for weight, item in zip(weights, selected)) / total),
    )


def _apply_idw_error(
    residual_points: list[dict[str, object]],
    x: float,
    y: float,
    *,
    max_neighbors: int,
    power: float = 2.0,
) -> float:
    distances: list[tuple[float, float]] = []
    for point in residual_points:
        px, py = [float(value) for value in point["pixel"]]
        error = float(point.get("error_mm", 0.0) or 0.0)
        distance = math.hypot(px - float(x), py - float(y))
        if distance <= 1e-9:
            return error
        distances.append((distance, error))
    distances.sort(key=lambda item: item[0])
    selected = distances[: max(1, int(max_neighbors))]
    weights = [1.0 / max(1e-9, float(distance) ** float(power)) for distance, _ in selected]
    total = sum(weights)
    return float(sum(weight * item[1] for weight, item in zip(weights, selected)) / total)


def _build_regular_residual_grid(
    samples: list[dict[str, float]],
    residual_points: list[dict[str, object]],
    *,
    grid_size: int,
    max_neighbors: int,
    power: float,
) -> dict[str, object]:
    if not residual_points:
        return {"model": "grid"}
    size = max(2, int(grid_size))
    xs = np.linspace(
        float(min(sample["fit_pixel_x"] for sample in samples)),
        float(max(sample["fit_pixel_x"] for sample in samples)),
        size,
    )
    ys = np.linspace(
        float(min(sample["fit_pixel_y"] for sample in samples)),
        float(max(sample["fit_pixel_y"] for sample in samples)),
        size,
    )
    dx_grid: list[list[float]] = []
    dy_grid: list[list[float]] = []
    error_grid: list[list[float]] = []
    for y in ys:
        dx_row: list[float] = []
        dy_row: list[float] = []
        error_row: list[float] = []
        for x in xs:
            correction_x, correction_y = _apply_idw_residual(
                residual_points,
                float(x),
                float(y),
                max_neighbors=max_neighbors,
                power=power,
            )
            dx_row.append(float(correction_x))
            dy_row.append(float(correction_y))
            error_row.append(
                _apply_idw_error(
                    residual_points,
                    float(x),
                    float(y),
                    max_neighbors=max_neighbors,
                    power=power,
                )
            )
        dx_grid.append(dx_row)
        dy_grid.append(dy_row)
        error_grid.append(error_row)
    return {
        "model": "grid",
        "x_values": [float(value) for value in xs],
        "y_values": [float(value) for value in ys],
        "correction_dx_mm": dx_grid,
        "correction_dy_mm": dy_grid,
        "error_mm": error_grid,
        "points": residual_points,
        "power": float(power),
        "max_neighbors": int(max_neighbors),
    }


def _errors(
    samples: list[dict[str, float]],
    matrix: np.ndarray,
    model: str,
    *,
    residual_points: list[dict[str, object]] | None,
    max_neighbors: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample in samples:
        pred_x, pred_y = _predict(matrix, model, sample["fit_pixel_x"], sample["fit_pixel_y"])
        correction_x = correction_y = 0.0
        if residual_points:
            correction_x, correction_y = _apply_idw_residual(
                residual_points,
                sample["fit_pixel_x"],
                sample["fit_pixel_y"],
                max_neighbors=max_neighbors,
            )
            pred_x += correction_x
            pred_y += correction_y
        error = math.hypot(pred_x - sample["delta_x_mm"], pred_y - sample["delta_y_mm"])
        rows.append(
            {
                "pixel": [sample["pixel_x"], sample["pixel_y"]],
                "undistorted_pixel": [sample["fit_pixel_x"], sample["fit_pixel_y"]],
                "expected_delta_mm": [sample["delta_x_mm"], sample["delta_y_mm"]],
                "predicted_delta_mm": [pred_x, pred_y],
                "residual_correction_mm": [correction_x, correction_y],
                "error_mm": float(error),
            }
        )
    return rows


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def _workspace_polygon(samples: list[dict[str, float]]) -> list[list[float]]:
    points = np.array([[sample["fit_pixel_x"], sample["fit_pixel_y"]] for sample in samples], dtype=np.float32)
    if len(points) < 3:
        return []
    try:
        import cv2

        hull = cv2.convexHull(points.reshape((-1, 1, 2)))[:, 0, :]
        return [[float(x), float(y)] for x, y in hull]
    except Exception:
        min_x = float(np.min(points[:, 0]))
        max_x = float(np.max(points[:, 0]))
        min_y = float(np.min(points[:, 1]))
        max_y = float(np.max(points[:, 1]))
        return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]


def _write_error_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "pixel_x",
                "pixel_y",
                "undistorted_x",
                "undistorted_y",
                "expected_dx_mm",
                "expected_dy_mm",
                "predicted_dx_mm",
                "predicted_dy_mm",
                "correction_dx_mm",
                "correction_dy_mm",
                "error_mm",
            ]
        )
        for row in rows:
            pixel = row["pixel"]
            undistorted = row["undistorted_pixel"]
            expected = row["expected_delta_mm"]
            predicted = row["predicted_delta_mm"]
            correction = row["residual_correction_mm"]
            writer.writerow(
                [
                    float(pixel[0]),
                    float(pixel[1]),
                    float(undistorted[0]),
                    float(undistorted[1]),
                    float(expected[0]),
                    float(expected[1]),
                    float(predicted[0]),
                    float(predicted[1]),
                    float(correction[0]),
                    float(correction[1]),
                    float(row["error_mm"]),
                ]
            )


def _write_heatmap(path: Path, rows: list[dict[str, object]], *, width: int, height: int) -> None:
    try:
        import cv2
    except Exception:
        return
    image = np.full((int(height), int(width), 3), 245, dtype=np.uint8)
    if not rows:
        return
    max_error = max(1.0, max(float(row["error_mm"]) for row in rows))
    for row in rows:
        x, y = [float(value) for value in row["pixel"]]
        ratio = max(0.0, min(1.0, float(row["error_mm"]) / max_error))
        color = (0, int(round(220 * (1.0 - ratio))), int(round(255 * ratio)))
        point = (int(round(x)), int(round(y)))
        cv2.circle(image, point, 8, color, -1)
        cv2.putText(
            image,
            f"{float(row['error_mm']):.1f}",
            (point[0] + 8, max(15, point[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _parse_target_pixel(value: str) -> list[float] | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--target-pixel must be formatted as x,y")
    x, y = float(parts[0]), float(parts[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("--target-pixel contains non-finite values")
    return [x, y]


def build_profile(samples: list[dict[str, float]], *, model: str, args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    camera_matrix = _load_json_array(str(args.camera_matrix_json))
    dist_coeffs = _load_json_array(str(args.dist_coeffs_json))
    prepared = _prepare_samples(samples, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    train_samples, validation_samples = _split_train_validation(prepared, validation_ratio=float(args.validation_ratio))
    if model == "homography":
        matrix = _fit_homography(train_samples)
    else:
        model = "affine"
        matrix = _fit_affine(train_samples)

    residual_points: list[dict[str, object]] = []
    if str(args.residual_model) != "none":
        residual_points = _build_residual_points(train_samples, matrix, model)
    max_neighbors = max(1, int(args.residual_max_neighbors))
    validation_base = validation_samples if validation_samples else prepared
    error_rows = _errors(
        validation_base,
        matrix,
        model,
        residual_points=residual_points,
        max_neighbors=max_neighbors,
    )
    error_values = [float(row["error_mm"]) for row in error_rows]
    residual = {
        "median_error_mm": _percentile(error_values, 50),
        "p95_error_mm": _percentile(error_values, 95),
        "max_error_mm": max(error_values),
        "points": [{"pixel": row["undistorted_pixel"], "error_mm": row["error_mm"]} for row in error_rows],
    }
    target_pixel = _parse_target_pixel(str(args.target_pixel))
    if str(args.residual_model) == "none":
        residual_grid = {"model": "none"}
    elif str(args.residual_model) == "grid":
        residual_grid: dict[str, object] = _build_regular_residual_grid(
            train_samples,
            residual_points,
            grid_size=int(args.grid_size),
            max_neighbors=max_neighbors,
            power=float(args.residual_power),
        )
    else:
        residual_grid = {
            "model": "idw",
            "points": residual_points,
            "power": float(args.residual_power),
            "max_neighbors": max_neighbors,
        }
    if args.residual_max_distance_px > 0:
        residual_grid["max_distance_px"] = float(args.residual_max_distance_px)
    profile: dict[str, object] = {
        "profile_id": str(args.profile_id),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "image_size": [int(args.image_width), int(args.image_height)],
        "K": camera_matrix,
        "D": dist_coeffs,
        "pixel_to_delta": {"model": model, "matrix": matrix.tolist()},
        "residual": residual,
        "residual_grid": residual_grid,
        "valid_workspace": {"undistorted_pixel_polygon": _workspace_polygon(train_samples)},
        "samples_summary": {
            "total_samples": len(prepared),
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "validation_ratio": float(args.validation_ratio),
            "grid_size": int(args.grid_size),
            "residual_model": str(args.residual_model),
            "source_csv": str(args.samples_csv),
        },
        "limits": {"max_allowed_error_mm": float(args.max_allowed_error_mm)},
        "servo": {
            "target_pixel": target_pixel,
            "center_tolerance_px": float(args.center_tolerance_px),
            "gain": float(args.servo_gain),
            "max_attempts": int(args.max_servo_attempts),
        },
    }
    return profile, error_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit an eye-in-hand vision calibration profile from grid samples.")
    parser.add_argument("--samples-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-id", default="eye-in-hand-current")
    parser.add_argument("--model", choices=("affine", "homography"), default="affine")
    parser.add_argument("--residual-model", choices=("grid", "idw", "none"), default="grid")
    parser.add_argument("--residual-power", type=float, default=2.0)
    parser.add_argument("--residual-max-neighbors", type=int, default=6)
    parser.add_argument("--residual-max-distance-px", type=float, default=0.0)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--max-allowed-error-mm", type=float, default=6.0)
    parser.add_argument("--center-tolerance-px", type=float, default=8.0)
    parser.add_argument("--servo-gain", type=float, default=0.8)
    parser.add_argument("--max-servo-attempts", type=int, default=5)
    parser.add_argument("--target-pixel", default="")
    parser.add_argument("--camera-matrix-json", default="")
    parser.add_argument("--dist-coeffs-json", default="")
    parser.add_argument("--validation-csv", type=Path, default=None)
    parser.add_argument("--heatmap-output", type=Path, default=None)
    args = parser.parse_args(argv)

    samples = _read_samples(args.samples_csv)
    profile, error_rows = build_profile(samples, model=str(args.model), args=args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    validation_csv = args.validation_csv or args.output.with_name(args.output.stem + "_validation.csv")
    heatmap_output = args.heatmap_output or args.output.with_name(args.output.stem + "_heatmap.png")
    _write_error_csv(Path(validation_csv), error_rows)
    _write_heatmap(Path(heatmap_output), error_rows, width=int(args.image_width), height=int(args.image_height))

    residual = profile["residual"]
    print(
        "wrote {path} samples={count} train={train} validation={validation} median={median:.2f}mm p95={p95:.2f}mm max={max_error:.2f}mm validation_csv={csv_path} heatmap={heatmap}".format(
            path=args.output,
            count=len(samples),
            train=int(profile["samples_summary"]["train_samples"]),
            validation=int(profile["samples_summary"]["validation_samples"]),
            median=float(residual["median_error_mm"]),
            p95=float(residual["p95_error_mm"]),
            max_error=float(residual["max_error_mm"]),
            csv_path=validation_csv,
            heatmap=heatmap_output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
