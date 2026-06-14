from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RobotPoseSample:
    snapshot: dict[str, object]
    received_wall_ts: float
    received_perf_ts: float
    robot_ts: float | None = None
    robot_xy: tuple[float, float] | None = None
    cyl_pose: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class RobotPoseMatch:
    sample: RobotPoseSample
    age_ms: float


class RobotPoseBuffer:
    def __init__(self, *, capacity: int = 180) -> None:
        self._samples: deque[RobotPoseSample] = deque(maxlen=max(1, int(capacity)))

    def clear(self) -> None:
        self._samples.clear()

    def add_snapshot(
        self,
        snapshot: Mapping[str, object],
        *,
        received_wall_ts: float,
        received_perf_ts: float,
    ) -> RobotPoseSample | None:
        if not isinstance(snapshot, Mapping):
            return None
        sample = RobotPoseSample(
            snapshot=dict(snapshot),
            received_wall_ts=float(received_wall_ts),
            received_perf_ts=float(received_perf_ts),
            robot_ts=_optional_float(snapshot.get("robot_ts")),
            robot_xy=_robot_xy(snapshot),
            cyl_pose=_robot_cyl(snapshot),
        )
        self._samples.append(sample)
        return sample

    def nearest(self, capture_perf_ts: object) -> RobotPoseMatch | None:
        try:
            target = float(capture_perf_ts)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(target) or not self._samples:
            return None
        sample = min(self._samples, key=lambda item: abs(float(item.received_perf_ts) - target))
        return RobotPoseMatch(sample=sample, age_ms=abs(float(sample.received_perf_ts) - target) * 1000.0)


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _robot_xy(snapshot: Mapping[str, object]) -> tuple[float, float] | None:
    raw = snapshot.get("robot_xy")
    if not isinstance(raw, (tuple, list)) or len(raw) < 2:
        return None
    try:
        x = float(raw[0])
        y = float(raw[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return (x, y)


def _robot_cyl(snapshot: Mapping[str, object]) -> tuple[float, float, float] | None:
    raw = snapshot.get("robot_cyl")
    if not isinstance(raw, Mapping):
        return None
    try:
        theta = float(raw.get("theta_deg"))
        radius = float(raw.get("radius_mm"))
        z = float(raw.get("z_mm", snapshot.get("robot_z", 0.0)))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (theta, radius, z)):
        return None
    return (theta, radius, z)
