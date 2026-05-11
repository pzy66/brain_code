from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from hybrid_controller.vision.servo_controller import _biased_pick_cyl_command
from hybrid_controller.vision.servo_controller import _pick_command_sucker_rotation


@dataclass(frozen=True, slots=True)
class ContinuousServoPending:
    slot_id: int
    stable_frames: int = 0
    lost_frames: int = 0

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object] | None,
        *,
        slot_id: int,
    ) -> "ContinuousServoPending":
        if not isinstance(payload, Mapping):
            return cls(slot_id=int(slot_id))
        try:
            payload_slot_id = int(payload.get("slot_id", slot_id))
        except (TypeError, ValueError):
            payload_slot_id = int(slot_id)
        if payload_slot_id != int(slot_id):
            return cls(slot_id=int(slot_id))
        return cls(
            slot_id=int(slot_id),
            stable_frames=_safe_int(payload.get("stable_frames"), 0),
            lost_frames=_safe_int(payload.get("lost_frames"), 0),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": int(self.slot_id),
            "stable_frames": int(self.stable_frames),
            "lost_frames": int(self.lost_frames),
        }


@dataclass(frozen=True, slots=True)
class ContinuousServoDecision:
    action: str
    status: str
    reason: str
    theta_rate_deg_s: float = 0.0
    radius_rate_mm_s: float = 0.0
    z_rate_mm_s: float = 0.0
    command: str | None = None
    pending: ContinuousServoPending | None = None
    trace: dict[str, object] = field(default_factory=dict)

    @property
    def pending_dict(self) -> dict[str, object] | None:
        return None if self.pending is None else self.pending.to_dict()


class ContinuousVisionServoController:
    """Pure decision layer for continuous eye-in-hand visual servoing."""

    def __init__(self, config: object) -> None:
        self.config = config

    def decide(
        self,
        *,
        slot_id: int,
        slot_payload: Mapping[str, object] | None,
        packet: Mapping[str, object] | None,
        pending: Mapping[str, object] | None = None,
        current_cyl_pose: tuple[float, float, float] | None = None,
        frame_pose_age_ms: float | None = None,
    ) -> ContinuousServoDecision:
        current = ContinuousServoPending.from_mapping(pending, slot_id=int(slot_id))
        if current_cyl_pose is None:
            return self._stop("robot_pose_unavailable", current=current)
        try:
            theta_deg, radius_mm, current_z = (
                float(current_cyl_pose[0]),
                float(current_cyl_pose[1]),
                float(current_cyl_pose[2]),
            )
        except (TypeError, ValueError):
            return self._stop("robot_pose_unavailable", current=current)
        if not (math.isfinite(theta_deg) and math.isfinite(radius_mm) and math.isfinite(current_z)):
            return self._stop("robot_pose_unavailable", current=current)
        if self._frame_is_stale(packet=packet, frame_pose_age_ms=frame_pose_age_ms):
            return self._stop("frame_stale", current=current)
        if slot_payload is None or not bool(slot_payload.get("valid", False)):
            lost = current.lost_frames + 1
            next_pending = ContinuousServoPending(slot_id=int(slot_id), stable_frames=0, lost_frames=lost)
            if lost >= max(1, int(getattr(self.config, "vision_continuous_servo_lost_frames", 3))):
                return self._stop("lost_target", current=next_pending)
            return ContinuousServoDecision(
                action="STOP",
                status=f"lost target {lost}",
                reason="lost_target_wait",
                pending=next_pending,
                trace={"lost_frames": int(lost)},
            )
        try:
            payload_slot_id = int(slot_payload.get("slot_id", slot_payload.get("slot", slot_id)))
        except (TypeError, ValueError):
            payload_slot_id = int(slot_id)
        if payload_slot_id != int(slot_id):
            return self._stop("slot_switched", current=current)
        confirm_z = float(getattr(self.config, "vision_pick_confirm_z_mm", getattr(self.config, "robot_approach_z", 130.0)))
        z_tolerance = max(0.0, float(getattr(self.config, "vision_pick_z_tolerance_mm", 4.0)))
        center_distance_px = _safe_float(slot_payload.get("center_distance_px"), float("inf"))
        if not math.isfinite(center_distance_px):
            return self._stop("center_distance_unavailable", current=current)
        center_allow_px = max(
            float(getattr(self.config, "vision_continuous_servo_center_allow_descent_px", 24.0)),
            _safe_float(slot_payload.get("action_tolerance_px"), 0.0),
        )
        center_stop_px = max(
            center_allow_px,
            float(getattr(self.config, "vision_continuous_servo_center_stop_descent_px", 36.0)),
        )
        stable_frames = current.stable_frames + 1 if center_distance_px <= center_allow_px else 0
        next_pending = ContinuousServoPending(slot_id=int(slot_id), stable_frames=stable_frames, lost_frames=0)
        required_stable = max(1, int(getattr(self.config, "vision_continuous_servo_stable_frames", 2)))
        invalid_reason = str(slot_payload.get("invalid_reason") or "").strip()
        if (
            invalid_reason == "grasp_unstable"
            and center_distance_px <= max(center_stop_px, center_allow_px * 4.0)
            and (current.stable_frames > 0 or current_z <= confirm_z + 60.0)
        ):
            return ContinuousServoDecision(
                action="STOP",
                status=f"holding unstable grasp slot={int(slot_id)}",
                reason="hold",
                pending=next_pending,
                trace={
                    "center_distance_px": float(center_distance_px),
                    "center_allow_descent_px": float(center_allow_px),
                    "center_stop_descent_px": float(center_stop_px),
                    "stable_frames": int(stable_frames),
                    "current_z_mm": float(current_z),
                    "confirm_z_mm": float(confirm_z),
                    "invalid_reason": invalid_reason,
                },
            )
        if not bool(slot_payload.get("actionable", False)) and invalid_reason != "vision_servo_required":
            return self._stop(invalid_reason or "slot_not_actionable", current=current)
        if invalid_reason == "vision_servo_required":
            servo_point = slot_payload.get("servo_command_point")
            if not isinstance(servo_point, (tuple, list)) or len(servo_point) < 2:
                return self._stop("servo_command_unavailable", current=current)

        if current_z < confirm_z - z_tolerance:
            return self._stop("below_confirm_height", current=next_pending)

        at_confirm_height = abs(current_z - confirm_z) <= z_tolerance
        if at_confirm_height and bool(slot_payload.get("actionable", False)):
            if stable_frames >= required_stable:
                command = self._pick_command(slot_payload=slot_payload, current_cyl_pose=(theta_deg, radius_mm, current_z))
                if not command:
                    return self._stop("pick_command_unavailable", current=next_pending)
                return ContinuousServoDecision(
                    action="PICK_READY",
                    status=f"pick_ready slot={int(slot_id)}",
                    reason="pick_ready",
                    command=command,
                    pending=next_pending,
                    trace={
                        "stable_frames": int(stable_frames),
                        "center_distance_px": float(center_distance_px),
                        "confirm_z_mm": float(confirm_z),
                    },
                )

        theta_rate, radius_rate = self._horizontal_rates(slot_payload, current_cyl_pose=(theta_deg, radius_mm, current_z))
        z_rate = 0.0
        if stable_frames >= required_stable and current_z > confirm_z + z_tolerance:
            z_rate = -self._z_rate(current_z=current_z, confirm_z=confirm_z)
        elif center_distance_px > center_stop_px:
            z_rate = 0.0

        if abs(theta_rate) < 1e-6 and abs(radius_rate) < 1e-6 and abs(z_rate) < 1e-6:
            return ContinuousServoDecision(
                action="STOP",
                status=f"holding slot={int(slot_id)}",
                reason="hold",
                pending=next_pending,
                trace={
                    "center_distance_px": float(center_distance_px),
                    "stable_frames": int(stable_frames),
                    "current_z_mm": float(current_z),
                    "confirm_z_mm": float(confirm_z),
                },
            )

        return ContinuousServoDecision(
            action="SERVO",
            status=f"servo slot={int(slot_id)}",
            reason="continuous_servo",
            theta_rate_deg_s=theta_rate,
            radius_rate_mm_s=radius_rate,
            z_rate_mm_s=z_rate,
            pending=next_pending,
            trace={
                "center_distance_px": float(center_distance_px),
                "center_allow_descent_px": float(center_allow_px),
                "center_stop_descent_px": float(center_stop_px),
                "stable_frames": int(stable_frames),
                "current_z_mm": float(current_z),
                "confirm_z_mm": float(confirm_z),
            },
        )

    def _horizontal_rates(
        self,
        slot_payload: Mapping[str, object],
        *,
        current_cyl_pose: tuple[float, float, float],
    ) -> tuple[float, float]:
        point = slot_payload.get("servo_command_point")
        if not isinstance(point, (tuple, list)) or len(point) < 2:
            return (0.0, 0.0)
        target_theta = _safe_float(point[0], current_cyl_pose[0])
        target_radius = _safe_float(point[1], current_cyl_pose[1])
        theta_error = target_theta - float(current_cyl_pose[0])
        radius_error = target_radius - float(current_cyl_pose[1])
        theta_gain = max(0.0, float(getattr(self.config, "vision_continuous_servo_theta_gain_deg_s_per_deg", 2.0)))
        radius_gain = max(0.0, float(getattr(self.config, "vision_continuous_servo_radius_gain_mm_s_per_mm", 1.2)))
        theta_limit = max(0.1, float(getattr(self.config, "vision_continuous_servo_theta_rate_limit_deg_s", 18.0)))
        radius_limit = max(0.1, float(getattr(self.config, "vision_continuous_servo_radius_rate_limit_mm_s", 35.0)))
        return (
            _clamp(theta_error * theta_gain, -theta_limit, theta_limit),
            _clamp(radius_error * radius_gain, -radius_limit, radius_limit),
        )

    def _z_rate(self, *, current_z: float, confirm_z: float) -> float:
        z_limit = max(0.1, float(getattr(self.config, "vision_continuous_servo_z_rate_limit_mm_s", 18.0)))
        slow_band = max(0.1, float(getattr(self.config, "vision_continuous_servo_z_slow_band_mm", 20.0)))
        remaining = max(0.0, float(current_z) - float(confirm_z))
        if remaining <= slow_band:
            min_crawl_rate = min(3.0, z_limit)
            return min(z_limit, max(min_crawl_rate, z_limit * remaining / slow_band))
        return z_limit

    def _pick_command(
        self,
        *,
        slot_payload: Mapping[str, object],
        current_cyl_pose: tuple[float, float, float],
    ) -> str | None:
        raw_command = str(slot_payload.get("command", "") or "")
        offset_source = str(getattr(self.config, "pick_tool_offset_source", "command_bias")).strip().lower()
        if offset_source != "command_bias":
            return raw_command or None
        return _biased_pick_cyl_command(
            current_cyl_pose=current_cyl_pose,
            radius_bias_mm=float(getattr(self.config, "vision_eye_in_hand_pick_radius_bias_mm", 0.0)),
            sucker_rotation_deg=_pick_command_sucker_rotation(raw_command),
        )

    def _frame_is_stale(self, *, packet: Mapping[str, object] | None, frame_pose_age_ms: float | None) -> bool:
        max_age_ms = max(1.0, float(getattr(self.config, "vision_continuous_servo_command_timeout_ms", 250.0)))
        ages: list[float] = []
        if frame_pose_age_ms is not None:
            try:
                ages.append(max(0.0, float(frame_pose_age_ms)))
            except (TypeError, ValueError):
                return True
        if isinstance(packet, Mapping):
            for key in ("latest_frame_preprocess_age_ms", "stream_age_ms", "queue_age_ms"):
                if packet.get(key) is not None:
                    age = _safe_float(packet.get(key), None)
                    if not math.isfinite(age):
                        return True
                    ages.append(max(0.0, float(age)))
        if not ages:
            return True
        return max(ages) > max_age_ms

    @staticmethod
    def _stop(reason: str, *, current: ContinuousServoPending) -> ContinuousServoDecision:
        return ContinuousServoDecision(
            action="STOP",
            status=f"stop reason={reason}",
            reason=reason,
            pending=current,
            trace={"reason": reason},
        )


def _safe_float(value: object, default: float | None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        if default is None:
            return float("nan")
        return float(default)
    return result if math.isfinite(result) else (float("nan") if default is None else float(default))


def _safe_int(value: object, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))
