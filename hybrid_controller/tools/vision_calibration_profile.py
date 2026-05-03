from __future__ import annotations

import argparse
import csv
import json
import math
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


def _fit_affine(samples: list[dict[str, float]]) -> np.ndarray:
    a_rows: list[list[float]] = []
    bx: list[float] = []
    by: list[float] = []
    for sample in samples:
        a_rows.append([sample["pixel_x"], sample["pixel_y"], 1.0])
        bx.append(sample["delta_x_mm"])
        by.append(sample["delta_y_mm"])
    a = np.array(a_rows, dtype=np.float64)
    mx, *_ = np.linalg.lstsq(a, np.array(bx, dtype=np.float64), rcond=None)
    my, *_ = np.linalg.lstsq(a, np.array(by, dtype=np.float64), rcond=None)
    return np.vstack([mx, my])


def _fit_homography(samples: list[dict[str, float]]) -> np.ndarray:
    import cv2

    src = np.array([[sample["pixel_x"], sample["pixel_y"]] for sample in samples], dtype=np.float64)
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


def _errors(samples: list[dict[str, float]], matrix: np.ndarray, model: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample in samples:
        pred_x, pred_y = _predict(matrix, model, sample["pixel_x"], sample["pixel_y"])
        error = math.hypot(pred_x - sample["delta_x_mm"], pred_y - sample["delta_y_mm"])
        rows.append(
            {
                "pixel": [sample["pixel_x"], sample["pixel_y"]],
                "expected_delta_mm": [sample["delta_x_mm"], sample["delta_y_mm"]],
                "predicted_delta_mm": [pred_x, pred_y],
                "error_mm": float(error),
            }
        )
    return rows


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def build_profile(samples: list[dict[str, float]], *, model: str, args: argparse.Namespace) -> dict[str, object]:
    if model == "homography":
        matrix = _fit_homography(samples)
    else:
        model = "affine"
        matrix = _fit_affine(samples)
    error_rows = _errors(samples, matrix, model)
    error_values = [float(row["error_mm"]) for row in error_rows]
    residual = {
        "median_error_mm": _percentile(error_values, 50),
        "p95_error_mm": _percentile(error_values, 95),
        "max_error_mm": max(error_values),
        "points": [{"pixel": row["pixel"], "error_mm": row["error_mm"]} for row in error_rows],
    }
    profile: dict[str, object] = {
        "profile_id": str(args.profile_id),
        "image_size": [int(args.image_width), int(args.image_height)],
        "pixel_to_delta": {"model": model, "matrix": matrix.tolist()},
        "residual": residual,
        "limits": {"max_allowed_error_mm": float(args.max_allowed_error_mm)},
        "servo": {
            "center_tolerance_px": float(args.center_tolerance_px),
            "gain": float(args.servo_gain),
            "max_attempts": int(args.max_servo_attempts),
        },
    }
    if args.camera_matrix_json:
        profile["K"] = json.loads(Path(args.camera_matrix_json).read_text(encoding="utf-8"))
    if args.dist_coeffs_json:
        profile["D"] = json.loads(Path(args.dist_coeffs_json).read_text(encoding="utf-8"))
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit an eye-in-hand vision calibration profile from grid samples.")
    parser.add_argument("--samples-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-id", default="eye-in-hand-current")
    parser.add_argument("--model", choices=("affine", "homography"), default="affine")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--max-allowed-error-mm", type=float, default=6.0)
    parser.add_argument("--center-tolerance-px", type=float, default=8.0)
    parser.add_argument("--servo-gain", type=float, default=0.8)
    parser.add_argument("--max-servo-attempts", type=int, default=3)
    parser.add_argument("--camera-matrix-json", default="")
    parser.add_argument("--dist-coeffs-json", default="")
    args = parser.parse_args(argv)

    samples = _read_samples(args.samples_csv)
    profile = build_profile(samples, model=str(args.model), args=args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    residual = profile["residual"]
    print(
        "wrote {path} samples={count} median={median:.2f}mm p95={p95:.2f}mm max={max_error:.2f}mm".format(
            path=args.output,
            count=len(samples),
            median=float(residual["median_error_mm"]),
            p95=float(residual["p95_error_mm"]),
            max_error=float(residual["max_error_mm"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
