from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _as_matrix(value: object, *, rows: int, cols: int, name: str) -> np.ndarray:
    matrix = np.array(value, dtype=np.float64).reshape(rows, cols)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _rt_to_transform(rotation: object, translation: object, *, name: str) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _as_matrix(rotation, rows=3, cols=3, name=f"{name}.R")
    transform[:3, 3] = _as_matrix(translation, rows=3, cols=1, name=f"{name}.t").reshape(3)
    return transform


def _sample_transform(sample: dict[str, object], *, matrix_key: str, r_key: str, t_key: str) -> np.ndarray:
    if matrix_key in sample:
        return _as_matrix(sample[matrix_key], rows=4, cols=4, name=matrix_key)
    return _rt_to_transform(sample[r_key], sample[t_key], name=matrix_key)


def _load_samples(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("samples")
    if not isinstance(payload, list) or len(payload) < 3:
        raise ValueError("Hand-eye input must contain at least 3 samples.")
    samples = [sample for sample in payload if isinstance(sample, dict)]
    if len(samples) < 3:
        raise ValueError("Hand-eye input samples must be JSON objects.")
    return samples


def _method_code(name: str) -> int:
    import cv2

    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    key = str(name or "tsai").strip().lower()
    if key not in methods:
        raise ValueError(f"Unsupported hand-eye method: {name}")
    return int(methods[key])


def _translation_consistency_mm(
    transforms_gripper2base: list[np.ndarray],
    transforms_target2cam: list[np.ndarray],
    transform_camera2tool: np.ndarray,
) -> dict[str, float]:
    base_target_translations: list[np.ndarray] = []
    for t_gripper2base, t_target2cam in zip(transforms_gripper2base, transforms_target2cam):
        t_target2base = t_gripper2base @ transform_camera2tool @ t_target2cam
        base_target_translations.append(np.array(t_target2base[:3, 3], dtype=np.float64))
    if not base_target_translations:
        return {"median_mm": float("nan"), "p95_mm": float("nan"), "max_mm": float("nan")}
    center = np.median(np.vstack(base_target_translations), axis=0)
    errors = np.array([float(np.linalg.norm(point - center)) for point in base_target_translations], dtype=np.float64)
    return {
        "median_mm": float(np.median(errors)),
        "p95_mm": float(np.percentile(errors, 95)),
        "max_mm": float(np.max(errors)),
    }


def calibrate_hand_eye(samples: list[dict[str, object]], *, method: str) -> dict[str, object]:
    import cv2

    transforms_gripper2base = [
        _sample_transform(
            sample,
            matrix_key="T_gripper2base",
            r_key="R_gripper2base",
            t_key="t_gripper2base",
        )
        for sample in samples
    ]
    transforms_target2cam = [
        _sample_transform(
            sample,
            matrix_key="T_target2cam",
            r_key="R_target2cam",
            t_key="t_target2cam",
        )
        for sample in samples
    ]
    r_gripper2base = [transform[:3, :3] for transform in transforms_gripper2base]
    t_gripper2base = [transform[:3, 3].reshape(3, 1) for transform in transforms_gripper2base]
    r_target2cam = [transform[:3, :3] for transform in transforms_target2cam]
    t_target2cam = [transform[:3, 3].reshape(3, 1) for transform in transforms_target2cam]

    r_camera2tool, t_camera2tool = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=_method_code(method),
    )
    transform_camera2tool = np.eye(4, dtype=np.float64)
    transform_camera2tool[:3, :3] = np.array(r_camera2tool, dtype=np.float64).reshape(3, 3)
    transform_camera2tool[:3, 3] = np.array(t_camera2tool, dtype=np.float64).reshape(3)
    residual = _translation_consistency_mm(transforms_gripper2base, transforms_target2cam, transform_camera2tool)
    return {
        "hand_eye": {
            "T_camera_to_tool": transform_camera2tool.tolist(),
            "method": str(method),
            "sample_count": int(len(samples)),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "target_translation_consistency_mm": residual,
        }
    }


def _merge_profile(profile_path: Path | None, result: dict[str, object]) -> dict[str, object]:
    if profile_path is None:
        payload: dict[str, object] = {}
    else:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Profile JSON must be an object.")
    payload["hand_eye"] = result["hand_eye"]
    samples_summary = payload.get("samples_summary")
    if not isinstance(samples_summary, dict):
        samples_summary = {}
    hand_eye_payload = result.get("hand_eye")
    if isinstance(hand_eye_payload, dict):
        samples_summary["hand_eye"] = {
            "sample_count": hand_eye_payload.get("sample_count"),
            "method": hand_eye_payload.get("method"),
            "target_translation_consistency_mm": hand_eye_payload.get("target_translation_consistency_mm"),
        }
    payload["samples_summary"] = samples_summary
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate eye-in-hand T_camera_to_tool from ChArUco pose samples.")
    parser.add_argument("--samples", type=Path, required=True, help="JSON list or object with a samples list.")
    parser.add_argument("--method", choices=("tsai", "park", "horaud", "andreff", "daniilidis"), default="tsai")
    parser.add_argument("--profile-in", type=Path, default=None, help="Optional existing calibration profile to update.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path. Omit to print to stdout.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = calibrate_hand_eye(_load_samples(args.samples), method=str(args.method))
    payload = _merge_profile(args.profile_in, result)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    residual = result["hand_eye"]["target_translation_consistency_mm"]  # type: ignore[index]
    if isinstance(residual, dict) and math.isfinite(float(residual.get("p95_mm", float("nan")))):
        print(
            "Hand-eye residual: median={0:.2f}mm p95={1:.2f}mm max={2:.2f}mm".format(
                float(residual["median_mm"]),
                float(residual["p95_mm"]),
                float(residual["max_mm"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
