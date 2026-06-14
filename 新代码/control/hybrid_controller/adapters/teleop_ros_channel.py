from __future__ import annotations

import time
from dataclasses import dataclass


_UINT32_MAX = 2**32 - 1
_CMD_SEQ_HEADROOM = 100_000
_CMD_SEQ_MODULUS = _UINT32_MAX - _CMD_SEQ_HEADROOM


def new_teleop_cmd_seq_base(now_wall: float | None = None) -> int:
    """Return a uint32-safe sequence base that normally increases across process restarts."""

    now = time.time() if now_wall is None else float(now_wall)
    return max(1, int(now * 1000.0) % _CMD_SEQ_MODULUS)


def next_teleop_cmd_seq(current: int) -> int:
    value = int(current) + 1
    if value <= 0 or value >= _UINT32_MAX:
        return 1
    return value


@dataclass(frozen=True, slots=True)
class RosTeleopCommand:
    theta_rate_deg_s: float
    radius_rate_mm_s: float
    z_rate_mm_s: float
    use_auto_z: bool
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
        self._last_command = (0.0, 0.0, 0.0, True)
        self._last_publish_ts = 0.0

    def on_publish_failed(self) -> None:
        self._active = False

    def next_command(
        self,
        *,
        theta_rate_deg_s: float,
        radius_rate_mm_s: float,
        z_rate_mm_s: float = 0.0,
        use_auto_z: bool = True,
        now_monotonic: float,
    ) -> RosTeleopCommand | None:
        theta_rate = float(theta_rate_deg_s)
        radius_rate = float(radius_rate_mm_s)
        z_rate = float(z_rate_mm_s)
        auto_z = bool(use_auto_z)
        now = float(now_monotonic)

        if abs(theta_rate) < self.epsilon:
            theta_rate = 0.0
        if abs(radius_rate) < self.epsilon:
            radius_rate = 0.0
        if abs(z_rate) < self.epsilon:
            z_rate = 0.0

        if theta_rate == 0.0 and radius_rate == 0.0 and z_rate == 0.0:
            if not self._active:
                return None
            self._active = False
            self._last_command = (0.0, 0.0, 0.0, auto_z)
            self._last_publish_ts = now
            return RosTeleopCommand(
                theta_rate_deg_s=0.0,
                radius_rate_mm_s=0.0,
                z_rate_mm_s=0.0,
                use_auto_z=auto_z,
                enabled=False,
            )

        last_theta, last_radius, last_z, last_auto_z = self._last_command
        changed = (
            abs(theta_rate - float(last_theta)) >= self.epsilon
            or abs(radius_rate - float(last_radius)) >= self.epsilon
            or abs(z_rate - float(last_z)) >= self.epsilon
            or bool(auto_z) != bool(last_auto_z)
        )
        keepalive_due = (now - float(self._last_publish_ts)) >= self.keepalive_interval_sec
        should_publish = changed or keepalive_due or not self._active
        if not should_publish:
            return None

        self._active = True
        self._last_command = (theta_rate, radius_rate, z_rate, auto_z)
        self._last_publish_ts = now
        return RosTeleopCommand(
            theta_rate_deg_s=theta_rate,
            radius_rate_mm_s=radius_rate,
            z_rate_mm_s=z_rate,
            use_auto_z=auto_z,
            enabled=True,
        )
