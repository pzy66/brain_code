from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple


ValidationResult = dict[str, object]


@dataclass(frozen=True, slots=True)
class CylindricalPose:
    theta_deg: float
    radius_mm: float
    z_mm: float

    def normalized(self) -> "CylindricalPose":
        return CylindricalPose(
            theta_deg=normalize_theta_deg(self.theta_deg),
            radius_mm=float(self.radius_mm),
            z_mm=float(self.z_mm),
        )

    def as_cartesian(self) -> tuple[float, float, float]:
        return cylindrical_to_cartesian(self.theta_deg, self.radius_mm, self.z_mm)

    @classmethod
    def from_cartesian(cls, x_mm: float, y_mm: float, z_mm: float) -> "CylindricalPose":
        theta_deg, radius_mm, z_value = cartesian_to_cylindrical(x_mm, y_mm, z_mm)
        return cls(theta_deg=theta_deg, radius_mm=radius_mm, z_mm=z_value)


def normalize_theta_deg(theta_deg: float) -> float:
    theta = float(theta_deg)
    while theta <= -180.0:
        theta += 360.0
    while theta > 180.0:
        theta -= 360.0
    return theta


def cylindrical_to_cartesian(theta_deg: float, radius_mm: float, z_mm: float) -> Tuple[float, float, float]:
    theta_rad = math.radians(float(theta_deg))
    radius = float(radius_mm)
    return (
        radius * math.sin(theta_rad),
        -radius * math.cos(theta_rad),
        float(z_mm),
    )


def cartesian_to_cylindrical(x_mm: float, y_mm: float, z_mm: float) -> Tuple[float, float, float]:
    radius = math.hypot(float(x_mm), float(y_mm))
    theta_deg = math.degrees(math.atan2(float(x_mm), -float(y_mm)))
    return (normalize_theta_deg(theta_deg), radius, float(z_mm))


def clamp(value: float, limits: Tuple[float, float]) -> float:
    return max(float(limits[0]), min(float(limits[1]), float(value)))


def within_limits(value: float, limits: Tuple[float, float]) -> bool:
    return float(limits[0]) <= float(value) <= float(limits[1])


def sample_range(start: float, stop: float, step: float) -> List[float]:
    start_value = float(start)
    stop_value = float(stop)
    step_value = abs(float(step))
    if step_value <= 0.0:
        raise ValueError("step must be positive")
    values: List[float] = []
    current = start_value
    while current <= stop_value + 1e-6:
        values.append(round(current, 6))
        current += step_value
    if not values or abs(values[-1] - stop_value) > 1e-6:
        values.append(round(stop_value, 6))
    return values


def interpolate_auto_z(profile_points: Iterable[Tuple[float, float]], radius_mm: float) -> float:
    points = sorted((float(radius), float(z_mm)) for radius, z_mm in profile_points)
    if not points:
        raise ValueError("auto-z profile is empty")
    radius = float(radius_mm)
    if radius <= points[0][0]:
        return points[0][1]
    if radius >= points[-1][0]:
        return points[-1][1]
    for index in range(1, len(points)):
        left_r, left_z = points[index - 1]
        right_r, right_z = points[index]
        if radius > right_r:
            continue
        if abs(right_r - left_r) <= 1e-6:
            return right_z
        ratio = (radius - left_r) / (right_r - left_r)
        return left_z + (right_z - left_z) * ratio
    return points[-1][1]


def build_auto_z_profile(
    *,
    radius_limits: Tuple[float, float],
    z_limits: Tuple[float, float],
    radius_step_mm: float,
    z_step_mm: float,
    validator: Callable[[float, float, float], ValidationResult],
    preferred_z_mm: float,
    profile_theta_deg: float = 0.0,
    target_z_provider: Callable[[float], float] | None = None,
    posture_tolerance: float = 0.0,
) -> Tuple[Tuple[float, float], ...]:
    candidate_map: dict[float, list[tuple[float, float, float, float, float]]] = {}
    radii = sample_range(radius_limits[0], radius_limits[1], radius_step_mm)
    z_candidates = sample_range(z_limits[0], z_limits[1], z_step_mm)
    preferred_z = float(preferred_z_mm)
    posture_tolerance_value = max(0.0, float(posture_tolerance))

    for radius in radii:
        target_z_mm = float(target_z_provider(radius)) if target_z_provider is not None else preferred_z
        valid_candidates: list[tuple[float, float, float, float, float]] = []
        for z_value in z_candidates:
            report = validator(float(profile_theta_deg), float(radius), float(z_value))
            if not report.get("ok", False):
                continue
            margin = float(report.get("margin", 0.0))
            neutral_distance = float(report.get("neutral_distance", abs(z_value - preferred_z)))
            posture_distance = float(report.get("posture_distance", neutral_distance))
            target_distance = abs(float(z_value) - target_z_mm)
            valid_candidates.append((float(z_value), margin, neutral_distance, posture_distance, target_distance))
        if valid_candidates:
            candidate_map[float(radius)] = sorted(valid_candidates, key=lambda entry: entry[0])

    if not candidate_map:
        raise ValueError("No valid auto-z profile points could be generated")

    def _filter_posture_candidates(
        candidates: list[tuple[float, float, float, float, float]],
    ) -> list[tuple[float, float, float, float, float]]:
        best_posture_distance = min(abs(entry[3]) for entry in candidates)
        return [
            entry
            for entry in candidates
            if abs(entry[3]) <= best_posture_distance + posture_tolerance_value + 1e-6
        ]

    def _pick_outermost(candidates: list[tuple[float, float, float, float, float]]) -> float:
        filtered_candidates = _filter_posture_candidates(candidates)
        z_value, _, _, _, _ = min(
            filtered_candidates,
            key=lambda entry: (
                entry[4],
                -entry[1],
                abs(entry[0] - preferred_z),
                abs(entry[2]),
                entry[0],
            ),
        )
        return float(z_value)

    def _pick_inward(candidates: list[tuple[float, float, float, float, float]], next_z: float) -> float:
        filtered_candidates = _filter_posture_candidates(candidates)
        z_value, _, _, _, _ = min(
            filtered_candidates,
            key=lambda entry: (
                entry[4],
                abs(entry[0] - float(next_z)),
                -entry[1],
                abs(entry[0] - preferred_z),
                abs(entry[2]),
                entry[0],
            ),
        )
        return float(z_value)

    profile_desc: list[tuple[float, float]] = []
    next_selected_z: float | None = None
    for radius in sorted(candidate_map.keys(), reverse=True):
        candidates = candidate_map[radius]
        if next_selected_z is None:
            selected_z = _pick_outermost(candidates)
        else:
            selected_z = _pick_inward(candidates, next_selected_z)
        profile_desc.append((radius, selected_z))
        next_selected_z = selected_z

    return tuple(sorted(profile_desc, key=lambda entry: entry[0]))


def nearest_profile_radius_limits(profile_points: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    points = sorted((float(radius), float(z_mm)) for radius, z_mm in profile_points)
    if not points:
        raise ValueError("auto-z profile is empty")
    return (points[0][0], points[-1][0])
