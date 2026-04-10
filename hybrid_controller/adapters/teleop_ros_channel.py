from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RosTeleopCommand:
    theta_rate_deg_s: float
    radius_rate_mm_s: float
    enabled: bool


class RosTeleopPublishPlanner:
    def __init__(self, *, keepalive_interval_sec: float = 0.12, epsilon: float = 1e-3) -> None:
        self.keepalive_interval_sec = max(0.01, float(keepalive_interval_sec))
        self.epsilon = max(1e-6, float(epsilon))
        self.reset()

    @property
    def active(self) -> bool:
        return bool(self._active)

    def reset(self) -> None:
        self._active = False
        self._last_rates = (0.0, 0.0)
        self._last_publish_ts = 0.0

    def on_publish_failed(self) -> None:
        self._active = False

    def next_command(
        self,
        *,
        theta_rate_deg_s: float,
        radius_rate_mm_s: float,
        now_monotonic: float,
    ) -> RosTeleopCommand | None:
        theta_rate = float(theta_rate_deg_s)
        radius_rate = float(radius_rate_mm_s)
        now = float(now_monotonic)

        if abs(theta_rate) < self.epsilon:
            theta_rate = 0.0
        if abs(radius_rate) < self.epsilon:
            radius_rate = 0.0

        if theta_rate == 0.0 and radius_rate == 0.0:
            if not self._active:
                return None
            self._active = False
            self._last_rates = (0.0, 0.0)
            self._last_publish_ts = now
            return RosTeleopCommand(theta_rate_deg_s=0.0, radius_rate_mm_s=0.0, enabled=False)

        last_theta, last_radius = self._last_rates
        changed = (
            abs(theta_rate - float(last_theta)) >= self.epsilon
            or abs(radius_rate - float(last_radius)) >= self.epsilon
        )
        keepalive_due = (now - float(self._last_publish_ts)) >= self.keepalive_interval_sec
        should_publish = changed or keepalive_due or not self._active
        if not should_publish:
            return None

        self._active = True
        self._last_rates = (theta_rate, radius_rate)
        self._last_publish_ts = now
        return RosTeleopCommand(
            theta_rate_deg_s=theta_rate,
            radius_rate_mm_s=radius_rate,
            enabled=True,
        )
