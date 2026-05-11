from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class VisionGraspProfile:
    profile_id: str
    real_pick_enabled: bool
    vision_pick_confirm_z_mm: float
    vision_eye_in_hand_pick_radius_bias_mm: float
    pick_cyl_radius_bias_mm: float
    sucker_rotation_angle_quality_threshold: float
    vision_servo_low_action_tolerance_px: float
    vision_pick_z_tolerance_mm: float


@dataclass(frozen=True, slots=True)
class VisionGraspProfileLoadResult:
    path: Path
    ready: bool
    missing: bool
    error: str
    profile: VisionGraspProfile | None

    @property
    def profile_id(self) -> str:
        return "" if self.profile is None else self.profile.profile_id


def load_vision_grasp_profile(config: object) -> VisionGraspProfileLoadResult:
    path = Path(getattr(config, "vision_grasp_profile_path"))
    if not path.exists():
        return VisionGraspProfileLoadResult(
            path=path,
            ready=False,
            missing=True,
            error=f"vision_grasp_profile_missing:{path}",
            profile=None,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, Mapping):
            raise ValueError("profile root must be a JSON object")
        profile = _profile_from_payload(payload, config=config)
    except Exception as error:
        return VisionGraspProfileLoadResult(
            path=path,
            ready=False,
            missing=False,
            error=f"vision_grasp_profile_invalid:{error}",
            profile=None,
        )
    return VisionGraspProfileLoadResult(path=path, ready=True, missing=False, error="", profile=profile)


def apply_vision_grasp_profile(config: object, result: VisionGraspProfileLoadResult) -> object:
    profile = result.profile
    if profile is None:
        return config
    return replace(
        config,
        vision_pick_confirm_z_mm=float(profile.vision_pick_confirm_z_mm),
        vision_eye_in_hand_pick_radius_bias_mm=float(profile.vision_eye_in_hand_pick_radius_bias_mm),
        pick_cyl_radius_bias_mm=float(profile.pick_cyl_radius_bias_mm),
        sucker_rotation_angle_quality_threshold=float(profile.sucker_rotation_angle_quality_threshold),
        vision_servo_low_action_tolerance_px=float(profile.vision_servo_low_action_tolerance_px),
        vision_pick_z_tolerance_mm=float(profile.vision_pick_z_tolerance_mm),
    )


def _profile_from_payload(payload: Mapping[str, object], *, config: object) -> VisionGraspProfile:
    z_limits = getattr(config, "robot_height_limits_mm", (80.0, 212.8))
    z_min = float(z_limits[0])
    z_max = float(z_limits[1])
    return VisionGraspProfile(
        profile_id=_text(payload.get("profile_id"), "default-vision-grasp-profile"),
        real_pick_enabled=bool(payload.get("real_pick_enabled", True)),
        vision_pick_confirm_z_mm=_float(
            payload.get("vision_pick_confirm_z_mm"),
            float(getattr(config, "vision_pick_confirm_z_mm")),
            z_min,
            z_max,
        ),
        vision_eye_in_hand_pick_radius_bias_mm=_float(
            payload.get("vision_eye_in_hand_pick_radius_bias_mm"),
            float(getattr(config, "vision_eye_in_hand_pick_radius_bias_mm")),
            -80.0,
            120.0,
        ),
        pick_cyl_radius_bias_mm=_float(
            payload.get("pick_cyl_radius_bias_mm"),
            float(getattr(config, "pick_cyl_radius_bias_mm")),
            -80.0,
            120.0,
        ),
        sucker_rotation_angle_quality_threshold=_float(
            payload.get("sucker_rotation_angle_quality_threshold"),
            float(getattr(config, "sucker_rotation_angle_quality_threshold")),
            0.0,
            1.0,
        ),
        vision_servo_low_action_tolerance_px=_float(
            payload.get("vision_servo_low_action_tolerance_px"),
            float(getattr(config, "vision_servo_low_action_tolerance_px")),
            1.0,
            80.0,
        ),
        vision_pick_z_tolerance_mm=_float(
            payload.get("vision_pick_z_tolerance_mm"),
            float(getattr(config, "vision_pick_z_tolerance_mm")),
            0.5,
            30.0,
        ),
    )


def _float(value: object, default: float, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return max(float(lower), min(float(upper), number))


def _text(value: object, default: str) -> str:
    text = str(value or "").strip()
    return text or str(default)
