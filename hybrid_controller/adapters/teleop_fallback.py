from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TeleopFallbackPlan:
    target_theta_deg: float
    target_radius_mm: float


class RosTeleopFallbackController:
    def __init__(
        self,
        *,
        stall_sec: float = 0.45,
        step_sec: float = 0.35,
        interval_sec: float = 0.35,
        pose_theta_epsilon_deg: float = 0.2,
        pose_radius_epsilon_mm: float = 0.8,
        target_theta_epsilon_deg: float = 0.05,
        target_radius_epsilon_mm: float = 0.2,
    ) -> None:
        self.stall_sec = float(stall_sec)
        self.step_sec = float(step_sec)
        self.interval_sec = float(interval_sec)
        self.pose_theta_epsilon_deg = float(pose_theta_epsilon_deg)
        self.pose_radius_epsilon_mm = float(pose_radius_epsilon_mm)
        self.target_theta_epsilon_deg = float(target_theta_epsilon_deg)
        self.target_radius_epsilon_mm = float(target_radius_epsilon_mm)
        self.reset()

    def reset(self) -> None:
        self.pending = False
        self.last_pose: tuple[float, float] | None = None
        self.last_pose_ts = 0.0
        self.last_fallback_ts = 0.0

    def next_plan(
        self,
        *,
        current_pose: tuple[float, float],
        theta_rate_deg_s: float,
        radius_rate_mm_s: float,
        now_monotonic: float,
        theta_limits_deg: tuple[float, float],
        radius_limits_mm: tuple[float, float],
    ) -> TeleopFallbackPlan | None:
        if self.pending:
            return None
        if self.last_pose is None or self._pose_moved(self.last_pose, current_pose):
            self.last_pose = (float(current_pose[0]), float(current_pose[1]))
            self.last_pose_ts = float(now_monotonic)
            return None
        if (float(now_monotonic) - float(self.last_pose_ts)) < self.stall_sec:
            return None
        if (float(now_monotonic) - float(self.last_fallback_ts)) < self.interval_sec:
            return None

        target_theta = self._clamp(
            float(current_pose[0]) + float(theta_rate_deg_s) * self.step_sec,
            theta_limits_deg,
        )
        target_radius = self._clamp(
            float(current_pose[1]) + float(radius_rate_mm_s) * self.step_sec,
            radius_limits_mm,
        )
        if (
            abs(float(target_theta) - float(current_pose[0])) < self.target_theta_epsilon_deg
            and abs(float(target_radius) - float(current_pose[1])) < self.target_radius_epsilon_mm
        ):
            return None

        self.pending = True
        self.last_fallback_ts = float(now_monotonic)
        return TeleopFallbackPlan(
            target_theta_deg=float(target_theta),
            target_radius_mm=float(target_radius),
        )

    def handle_service_result(self, *, ok: bool, message: str, now_monotonic: float) -> bool:
        self.pending = False
        if bool(ok):
            self.last_pose_ts = float(now_monotonic)
            return False
        if str(message or "").strip().upper() == "BUSY":
            return False
        return True

    def handle_service_call_exception(self) -> None:
        self.pending = False

    def _pose_moved(self, previous: tuple[float, float], current: tuple[float, float]) -> bool:
        theta_delta = abs(float(current[0]) - float(previous[0]))
        radius_delta = abs(float(current[1]) - float(previous[1]))
        return theta_delta >= self.pose_theta_epsilon_deg or radius_delta >= self.pose_radius_epsilon_mm

    @staticmethod
    def _clamp(value: float, limits: tuple[float, float]) -> float:
        low = float(min(limits[0], limits[1]))
        high = float(max(limits[0], limits[1]))
        return max(low, min(high, float(value)))
