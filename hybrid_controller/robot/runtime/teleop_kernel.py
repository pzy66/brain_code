import math
import time
from typing import Callable, Dict, Iterable, Optional, Tuple

try:
    from hybrid_controller.cylindrical import CylindricalPose, interpolate_auto_z
except ImportError:  # pragma: no cover - exercised on JetMax robot-only deploys
    class CylindricalPose:
        __slots__ = ("theta_deg", "radius_mm", "z_mm")

        def __init__(self, theta_deg, radius_mm, z_mm):
            self.theta_deg = float(theta_deg)
            self.radius_mm = float(radius_mm)
            self.z_mm = float(z_mm)

        def normalized(self):
            theta = float(self.theta_deg)
            while theta <= -180.0:
                theta += 360.0
            while theta > 180.0:
                theta -= 360.0
            return CylindricalPose(theta_deg=theta, radius_mm=self.radius_mm, z_mm=self.z_mm)

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


class CylindricalTeleopCommand:
    __slots__ = ("theta_rate_deg_s", "radius_rate_mm_s", "z_rate_mm_s", "use_auto_z", "enabled", "timestamp")

    def __init__(
        self,
        theta_rate_deg_s=0.0,
        radius_rate_mm_s=0.0,
        z_rate_mm_s=0.0,
        use_auto_z=True,
        enabled=False,
        timestamp=0.0,
    ):
        self.theta_rate_deg_s = float(theta_rate_deg_s)
        self.radius_rate_mm_s = float(radius_rate_mm_s)
        self.z_rate_mm_s = float(z_rate_mm_s)
        self.use_auto_z = bool(use_auto_z)
        self.enabled = bool(enabled)
        self.timestamp = float(timestamp)


class CylindricalTeleopStep:
    __slots__ = ("pose", "theta_rate_deg_s", "radius_rate_mm_s", "z_rate_mm_s", "stale")

    def __init__(self, pose, theta_rate_deg_s, radius_rate_mm_s, z_rate_mm_s, stale):
        self.pose = pose
        self.theta_rate_deg_s = float(theta_rate_deg_s)
        self.radius_rate_mm_s = float(radius_rate_mm_s)
        self.z_rate_mm_s = float(z_rate_mm_s)
        self.stale = bool(stale)


class CylindricalTeleopKernel:
    def __init__(
        self,
        *,
        theta_limits_deg: Tuple[float, float],
        radius_limits_mm: Tuple[float, float],
        z_limits_mm: Tuple[float, float] = (0.0, 300.0),
        auto_z_profile: Tuple[Tuple[float, float], ...],
        validator: Callable[[float, float, float], Dict[str, object]],
        tick_hz: float = 20.0,
        deadman_timeout_sec: float = 0.2,
        theta_accel_deg_s2: float = 240.0,
        radius_accel_mm_s2: float = 240.0,
        z_accel_mm_s2: float = 120.0,
        use_auto_z: bool = True,
    ) -> None:
        self.theta_limits_deg = (float(theta_limits_deg[0]), float(theta_limits_deg[1]))
        self.radius_limits_mm = (float(radius_limits_mm[0]), float(radius_limits_mm[1]))
        self.z_limits_mm = (float(z_limits_mm[0]), float(z_limits_mm[1]))
        self.auto_z_profile = tuple((float(radius), float(z_mm)) for radius, z_mm in auto_z_profile)
        self.validator = validator
        self.tick_hz = max(float(tick_hz), 1.0)
        self.tick_sec = 1.0 / self.tick_hz
        self.deadman_timeout_sec = max(float(deadman_timeout_sec), self.tick_sec)
        self.theta_accel_deg_s2 = max(float(theta_accel_deg_s2), 1.0)
        self.radius_accel_mm_s2 = max(float(radius_accel_mm_s2), 1.0)
        self.z_accel_mm_s2 = max(float(z_accel_mm_s2), 1.0)
        self.use_auto_z = bool(use_auto_z)
        self._command = CylindricalTeleopCommand(timestamp=time.monotonic())
        self._theta_rate_deg_s = 0.0
        self._radius_rate_mm_s = 0.0
        self._z_rate_mm_s = 0.0

    def update_command(
        self,
        *,
        theta_rate_deg_s: float,
        radius_rate_mm_s: float,
        z_rate_mm_s: float = 0.0,
        use_auto_z: bool = True,
        enabled: bool,
        timestamp: Optional[float] = None,
    ) -> None:
        self._command = CylindricalTeleopCommand(
            theta_rate_deg_s=float(theta_rate_deg_s),
            radius_rate_mm_s=float(radius_rate_mm_s),
            z_rate_mm_s=float(z_rate_mm_s),
            use_auto_z=bool(use_auto_z),
            enabled=bool(enabled),
            timestamp=float(time.monotonic() if timestamp is None else timestamp),
        )

    def clear_command(self, *, timestamp: Optional[float] = None) -> None:
        self.update_command(
            theta_rate_deg_s=0.0,
            radius_rate_mm_s=0.0,
            z_rate_mm_s=0.0,
            use_auto_z=self.use_auto_z,
            enabled=False,
            timestamp=timestamp,
        )

    def step(self, current_pose: CylindricalPose, *, now: Optional[float] = None):
        current_time = float(time.monotonic() if now is None else now)
        command = self._command
        stale = (current_time - float(command.timestamp)) > self.deadman_timeout_sec
        target_theta_rate = 0.0 if stale or not command.enabled else float(command.theta_rate_deg_s)
        target_radius_rate = 0.0 if stale or not command.enabled else float(command.radius_rate_mm_s)
        target_z_rate = 0.0 if stale or not command.enabled else float(command.z_rate_mm_s)

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
        self._z_rate_mm_s = self._ramp(
            current=self._z_rate_mm_s,
            target=target_z_rate,
            max_delta=self.z_accel_mm_s2 * self.tick_sec,
        )

        if abs(self._theta_rate_deg_s) < 1e-6 and abs(self._radius_rate_mm_s) < 1e-6 and abs(self._z_rate_mm_s) < 1e-6:
            return None

        next_theta = self._clamp(
            float(current_pose.theta_deg) + self._theta_rate_deg_s * self.tick_sec,
            self.theta_limits_deg,
        )
        next_radius = self._clamp(
            float(current_pose.radius_mm) + self._radius_rate_mm_s * self.tick_sec,
            self.radius_limits_mm,
        )
        if self.use_auto_z and bool(command.use_auto_z) and abs(self._z_rate_mm_s) < 1e-6:
            next_z = float(interpolate_auto_z(self.auto_z_profile, next_radius))
        else:
            next_z = self._clamp(
                float(current_pose.z_mm) + self._z_rate_mm_s * self.tick_sec,
                self.z_limits_mm,
            )

        validation = self.validator(next_theta, next_radius, next_z)
        if not bool(validation.get("ok", False)):
            self._theta_rate_deg_s = 0.0
            self._radius_rate_mm_s = 0.0
            self._z_rate_mm_s = 0.0
            return None

        next_pose = CylindricalPose(theta_deg=next_theta, radius_mm=next_radius, z_mm=next_z).normalized()
        return CylindricalTeleopStep(
            pose=next_pose,
            theta_rate_deg_s=self._theta_rate_deg_s,
            radius_rate_mm_s=self._radius_rate_mm_s,
            z_rate_mm_s=self._z_rate_mm_s,
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
