import math
import time
from typing import Callable, Dict, Optional, Tuple

from hybrid_controller.cylindrical import CylindricalPose, interpolate_auto_z


class CylindricalTeleopCommand:
    __slots__ = ("theta_rate_deg_s", "radius_rate_mm_s", "enabled", "timestamp")

    def __init__(self, theta_rate_deg_s=0.0, radius_rate_mm_s=0.0, enabled=False, timestamp=0.0):
        self.theta_rate_deg_s = float(theta_rate_deg_s)
        self.radius_rate_mm_s = float(radius_rate_mm_s)
        self.enabled = bool(enabled)
        self.timestamp = float(timestamp)


class CylindricalTeleopStep:
    __slots__ = ("pose", "theta_rate_deg_s", "radius_rate_mm_s", "stale")

    def __init__(self, pose, theta_rate_deg_s, radius_rate_mm_s, stale):
        self.pose = pose
        self.theta_rate_deg_s = float(theta_rate_deg_s)
        self.radius_rate_mm_s = float(radius_rate_mm_s)
        self.stale = bool(stale)


class CylindricalTeleopKernel:
    def __init__(
        self,
        *,
        theta_limits_deg: Tuple[float, float],
        radius_limits_mm: Tuple[float, float],
        auto_z_profile: Tuple[Tuple[float, float], ...],
        validator: Callable[[float, float, float], Dict[str, object]],
        tick_hz: float = 20.0,
        deadman_timeout_sec: float = 0.2,
        theta_accel_deg_s2: float = 240.0,
        radius_accel_mm_s2: float = 240.0,
    ) -> None:
        self.theta_limits_deg = (float(theta_limits_deg[0]), float(theta_limits_deg[1]))
        self.radius_limits_mm = (float(radius_limits_mm[0]), float(radius_limits_mm[1]))
        self.auto_z_profile = tuple((float(radius), float(z_mm)) for radius, z_mm in auto_z_profile)
        self.validator = validator
        self.tick_hz = max(float(tick_hz), 1.0)
        self.tick_sec = 1.0 / self.tick_hz
        self.deadman_timeout_sec = max(float(deadman_timeout_sec), self.tick_sec)
        self.theta_accel_deg_s2 = max(float(theta_accel_deg_s2), 1.0)
        self.radius_accel_mm_s2 = max(float(radius_accel_mm_s2), 1.0)
        self._command = CylindricalTeleopCommand(timestamp=time.monotonic())
        self._theta_rate_deg_s = 0.0
        self._radius_rate_mm_s = 0.0

    def update_command(
        self,
        *,
        theta_rate_deg_s: float,
        radius_rate_mm_s: float,
        enabled: bool,
        timestamp: Optional[float] = None,
    ) -> None:
        self._command = CylindricalTeleopCommand(
            theta_rate_deg_s=float(theta_rate_deg_s),
            radius_rate_mm_s=float(radius_rate_mm_s),
            enabled=bool(enabled),
            timestamp=float(time.monotonic() if timestamp is None else timestamp),
        )

    def clear_command(self, *, timestamp: Optional[float] = None) -> None:
        self.update_command(theta_rate_deg_s=0.0, radius_rate_mm_s=0.0, enabled=False, timestamp=timestamp)

    def step(self, current_pose: CylindricalPose, *, now: Optional[float] = None):
        current_time = float(time.monotonic() if now is None else now)
        command = self._command
        stale = (current_time - float(command.timestamp)) > self.deadman_timeout_sec
        target_theta_rate = 0.0 if stale or not command.enabled else float(command.theta_rate_deg_s)
        target_radius_rate = 0.0 if stale or not command.enabled else float(command.radius_rate_mm_s)

        self._theta_rate_deg_s = self._ramp(
            current=self._theta_rate_deg_s,
            target=target_theta_rate,
            max_delta=self.theta_accel_deg_s2 * self.tick_sec,
        )
        self._radius_rate_mm_s = self._ramp(
            current=self._radius_rate_mm_s,
            target=target_radius_rate,
            max_delta=self.radius_accel_mm_s2 * self.tick_sec,
        )

        if abs(self._theta_rate_deg_s) < 1e-6 and abs(self._radius_rate_mm_s) < 1e-6:
            return None

        next_theta = self._clamp(
            float(current_pose.theta_deg) + self._theta_rate_deg_s * self.tick_sec,
            self.theta_limits_deg,
        )
        next_radius = self._clamp(
            float(current_pose.radius_mm) + self._radius_rate_mm_s * self.tick_sec,
            self.radius_limits_mm,
        )
        next_z = float(interpolate_auto_z(self.auto_z_profile, next_radius))

        validation = self.validator(next_theta, next_radius, next_z)
        if not bool(validation.get("ok", False)):
            self._theta_rate_deg_s = 0.0
            self._radius_rate_mm_s = 0.0
            return None

        next_pose = CylindricalPose(theta_deg=next_theta, radius_mm=next_radius, z_mm=next_z).normalized()
        return CylindricalTeleopStep(
            pose=next_pose,
            theta_rate_deg_s=self._theta_rate_deg_s,
            radius_rate_mm_s=self._radius_rate_mm_s,
            stale=stale,
        )

    @staticmethod
    def _ramp(*, current: float, target: float, max_delta: float) -> float:
        delta = float(target) - float(current)
        if abs(delta) <= float(max_delta):
            return float(target)
        return float(current) + math.copysign(float(max_delta), delta)

    @staticmethod
    def _clamp(value: float, limits: Tuple[float, float]) -> float:
        return max(float(limits[0]), min(float(limits[1]), float(value)))
