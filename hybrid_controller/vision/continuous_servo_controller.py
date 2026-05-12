from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from hybrid_controller.cylindrical import cartesian_to_cylindrical
from hybrid_controller.cylindrical import cylindrical_to_cartesian
from hybrid_controller.vision.servo_controller import _biased_pick_cyl_command
from hybrid_controller.vision.servo_controller import _pick_command_sucker_rotation


@dataclass(frozen=True, slots=True)
class ContinuousServoPending:
    slot_id: int
    stable_frames: int = 0
    pick_ready_frames: int = 0
    lost_frames: int = 0
    source: str = ""
    last_center_px: tuple[float, float] | None = None
    last_center_distance_px: float | None = None
    descent_anchor_z_mm: float | None = None
    descent_cooldown_frames: int = 0

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
        last_center = _safe_pair(payload.get("last_center_px"))
        last_center_distance = _safe_float(payload.get("last_center_distance_px"), float("nan"))
        descent_anchor_z = _safe_float(payload.get("descent_anchor_z_mm"), float("nan"))
        return cls(
            slot_id=int(slot_id),
            stable_frames=_safe_int(payload.get("stable_frames"), 0),
            pick_ready_frames=_safe_int(payload.get("pick_ready_frames"), 0),
            lost_frames=_safe_int(payload.get("lost_frames"), 0),
            source=str(payload.get("source", "") or ""),
            last_center_px=last_center,
            last_center_distance_px=last_center_distance if math.isfinite(last_center_distance) else None,
            descent_anchor_z_mm=descent_anchor_z if math.isfinite(descent_anchor_z) else None,
            descent_cooldown_frames=_safe_int(payload.get("descent_cooldown_frames"), 0),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": int(self.slot_id),
            "stable_frames": int(self.stable_frames),
            "pick_ready_frames": int(self.pick_ready_frames),
            "lost_frames": int(self.lost_frames),
            "source": str(self.source),
            "last_center_px": (
                None if self.last_center_px is None else [float(self.last_center_px[0]), float(self.last_center_px[1])]
            ),
            "last_center_distance_px": (
                None if self.last_center_distance_px is None else float(self.last_center_distance_px)
            ),
            "descent_anchor_z_mm": None if self.descent_anchor_z_mm is None else float(self.descent_anchor_z_mm),
            "descent_cooldown_frames": int(self.descent_cooldown_frames),
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
        frame_block_reason = ""
        if isinstance(packet, Mapping):
            frame_block_reason = str(packet.get("frame_block_reason") or "").strip()
        if frame_block_reason:
            return self._stop(frame_block_reason, current=current)
        if self._frame_is_stale(packet=packet, frame_pose_age_ms=frame_pose_age_ms):
            return self._stop("frame_stale", current=current)
        if slot_payload is None or not bool(slot_payload.get("valid", False)):
            lost = current.lost_frames + 1
            next_pending = ContinuousServoPending(
                slot_id=int(slot_id),
                stable_frames=0,
                pick_ready_frames=0,
                lost_frames=lost,
                source=str(current.source),
                last_center_px=current.last_center_px,
                last_center_distance_px=current.last_center_distance_px,
                descent_anchor_z_mm=current.descent_anchor_z_mm,
                descent_cooldown_frames=current.descent_cooldown_frames,
            )
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
        quality_reason = self._slot_quality_reject_reason(slot_payload)
        if quality_reason:
            return ContinuousServoDecision(
                action="STOP",
                status=f"holding low-quality target slot={int(slot_id)} reason={quality_reason}",
                reason="hold",
                pending=ContinuousServoPending(
                    slot_id=int(slot_id),
                    stable_frames=0,
                    pick_ready_frames=0,
                    lost_frames=0,
                    source=str(current.source),
                    last_center_px=current.last_center_px,
                    last_center_distance_px=current.last_center_distance_px,
                    descent_anchor_z_mm=current.descent_anchor_z_mm,
                    descent_cooldown_frames=current.descent_cooldown_frames,
                ),
                trace={"quality_reason": quality_reason},
            )
        confirm_z = float(getattr(self.config, "vision_pick_confirm_z_mm", getattr(self.config, "robot_approach_z", 130.0)))
        z_tolerance = max(0.0, float(getattr(self.config, "vision_pick_z_tolerance_mm", 4.0)))
        center_distance_px = _safe_float(slot_payload.get("center_distance_px"), float("inf"))
        if not math.isfinite(center_distance_px):
            return self._stop("center_distance_unavailable", current=current)
        low_height_error_rebounded = self._low_height_error_rebounded(
            current_z=current_z,
            confirm_z=confirm_z,
            center_distance_px=center_distance_px,
            previous_center_distance_px=current.last_center_distance_px,
        )
        tracking_px = self._tracking_point(slot_payload)
        tracking_reason = self._tracking_reject_reason(
            current=current,
            center_px=tracking_px,
            center_distance_px=center_distance_px,
        )
        if tracking_reason:
            return self._stop(tracking_reason, current=current)
        center_allow_px = max(0.1, float(getattr(self.config, "vision_continuous_servo_center_allow_descent_px", 8.0)))
        center_stop_px = max(
            center_allow_px,
            float(getattr(self.config, "vision_continuous_servo_center_stop_descent_px", 36.0)),
        )
        settle_band_px = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_settle_stop_band_px", center_allow_px)),
        )
        required_stable = max(1, int(getattr(self.config, "vision_continuous_servo_stable_frames", 2)))
        stable_frames = current.stable_frames + 1 if center_distance_px <= center_allow_px else 0
        z_pulse_mm = max(0.5, float(getattr(self.config, "vision_continuous_servo_z_pulse_mm", 8.0)))
        descent_anchor_z = current.descent_anchor_z_mm
        descent_cooldown_frames = max(0, int(current.descent_cooldown_frames))
        if stable_frames == 0 or (descent_anchor_z is not None and current_z <= float(descent_anchor_z) - z_pulse_mm):
            descent_anchor_z = None
            descent_cooldown_frames = max(descent_cooldown_frames, required_stable)
        elif descent_cooldown_frames > 0 and stable_frames > 0:
            descent_cooldown_frames = max(0, descent_cooldown_frames - 1)
        next_pending = ContinuousServoPending(
            slot_id=int(slot_id),
            stable_frames=stable_frames,
            lost_frames=0,
            source=str(current.source),
            last_center_px=tracking_px,
            last_center_distance_px=float(center_distance_px),
            descent_anchor_z_mm=descent_anchor_z,
            descent_cooldown_frames=descent_cooldown_frames,
        )
        if center_distance_px > max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_pick_ready_center_px", 8.0)),
        ):
            next_pending = self._replace_pending(next_pending, pick_ready_frames=0)
        if low_height_error_rebounded:
            rebounded_pending = ContinuousServoPending(
                slot_id=int(next_pending.slot_id),
                stable_frames=0,
                pick_ready_frames=0,
                lost_frames=0,
                source=str(next_pending.source),
                last_center_px=next_pending.last_center_px,
                last_center_distance_px=next_pending.last_center_distance_px,
                descent_anchor_z_mm=next_pending.descent_anchor_z_mm,
                descent_cooldown_frames=int(next_pending.descent_cooldown_frames),
            )
            return ContinuousServoDecision(
                action="STOP",
                status=f"settling after low-height rebound slot={int(slot_id)}",
                reason="low_height_error_rebounded",
                pending=rebounded_pending,
                trace={
                    "center_distance_px": float(center_distance_px),
                    "last_center_distance_px": None
                    if current.last_center_distance_px is None
                    else float(current.last_center_distance_px),
                    "current_z_mm": float(current_z),
                    "confirm_z_mm": float(confirm_z),
                    "growth_stop_px": float(
                        getattr(self.config, "vision_continuous_servo_low_height_error_growth_stop_px", 2.0)
                    ),
                },
            )
        invalid_reason = str(slot_payload.get("invalid_reason") or "").strip()
        unstable_hold_window_px = max(center_stop_px, center_allow_px * 4.0)
        unstable_near_center = center_distance_px <= center_allow_px
        unstable_after_lock_or_descent = (
            center_distance_px <= unstable_hold_window_px
            and (current.stable_frames > 0 or current_z <= confirm_z + 60.0)
        )
        if invalid_reason == "grasp_unstable" and center_distance_px <= unstable_hold_window_px:
            if center_distance_px <= center_stop_px:
                candidate_point = self._fallback_servo_command_point(
                    slot_payload,
                    current_cyl_pose=(theta_deg, radius_mm, current_z),
                )
                if candidate_point is not None:
                    invalid_reason = "vision_servo_required"
                    slot_payload = dict(slot_payload)
                    slot_payload["servo_command_point"] = candidate_point
                    slot_payload["servo_command_mode"] = str(slot_payload.get("servo_command_mode", "cyl"))
                else:
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
            else:
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
        if current_z < confirm_z - z_tolerance:
            return self._stop("below_confirm_height", current=next_pending)

        if (
            current_z <= confirm_z + z_tolerance * 2.0
            and center_distance_px <= settle_band_px
            and stable_frames < required_stable
        ):
            return ContinuousServoDecision(
                action="STOP",
                status=f"settling near center slot={int(slot_id)}",
                reason="settle_near_center",
                pending=next_pending,
                trace={
                    "center_distance_px": float(center_distance_px),
                    "settle_stop_band_px": float(settle_band_px),
                    "stable_frames": int(stable_frames),
                    "required_stable_frames": int(required_stable),
                    "current_z_mm": float(current_z),
                    "confirm_z_mm": float(confirm_z),
                },
            )

        at_confirm_height = abs(current_z - confirm_z) <= z_tolerance
        if at_confirm_height and bool(slot_payload.get("actionable", False)):
            pick_ready_center_px = max(
                0.1,
                float(getattr(self.config, "vision_continuous_servo_pick_ready_center_px", 8.0)),
            )
            pick_ready_stable_frames = (
                current.pick_ready_frames + 1 if center_distance_px <= pick_ready_center_px else 0
            )
            next_pending = ContinuousServoPending(
                slot_id=int(slot_id),
                stable_frames=stable_frames,
                pick_ready_frames=pick_ready_stable_frames,
                lost_frames=0,
                source=str(current.source),
                last_center_px=tracking_px,
                last_center_distance_px=float(center_distance_px),
                descent_anchor_z_mm=descent_anchor_z,
                descent_cooldown_frames=descent_cooldown_frames,
            )
            if pick_ready_stable_frames >= required_stable:
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
                        "pick_ready_frames": int(pick_ready_stable_frames),
                        "center_distance_px": float(center_distance_px),
                        "confirm_z_mm": float(confirm_z),
                        "pick_ready_center_px": float(pick_ready_center_px),
                    },
                )

        slot_for_horizontal = slot_payload
        if not bool(slot_payload.get("actionable", False)) and invalid_reason != "vision_servo_required":
            return self._stop(invalid_reason or "slot_not_actionable", current=current)
        if invalid_reason == "vision_servo_required":
            fallback_point = self._fallback_servo_command_point(
                slot_payload,
                current_cyl_pose=(theta_deg, radius_mm, current_z),
            )
            if fallback_point is None:
                return self._stop("servo_command_unavailable", current=current)
            slot_for_horizontal = dict(slot_payload)
            slot_for_horizontal["servo_command_point"] = fallback_point
            slot_for_horizontal["servo_command_mode"] = str(slot_payload.get("servo_command_mode", "cyl"))
        if bool(slot_payload.get("actionable", False)):
            pick_ready_center_px = max(
                0.1,
                float(getattr(self.config, "vision_continuous_servo_pick_ready_center_px", 8.0)),
            )
            if center_distance_px > pick_ready_center_px:
                fallback_point = self._fallback_servo_command_point(
                    slot_payload,
                    current_cyl_pose=(theta_deg, radius_mm, current_z),
                )
                if fallback_point is not None:
                    slot_for_horizontal = dict(slot_payload)
                    slot_for_horizontal["servo_command_point"] = fallback_point
                    slot_for_horizontal["servo_command_mode"] = str(slot_payload.get("servo_command_mode", "cyl"))

        theta_rate, radius_rate = self._horizontal_rates(
            slot_for_horizontal,
            current_cyl_pose=(theta_deg, radius_mm, current_z),
            center_distance_px=center_distance_px,
            confirm_z=confirm_z,
        )
        z_rate = 0.0
        if stable_frames >= required_stable and descent_cooldown_frames <= 0 and current_z > confirm_z + z_tolerance:
            if descent_anchor_z is None:
                descent_anchor_z = float(current_z)
                next_pending = ContinuousServoPending(
                    slot_id=int(slot_id),
                    stable_frames=stable_frames,
                    pick_ready_frames=0,
                    lost_frames=0,
                    source=str(current.source),
                    last_center_px=tracking_px,
                    last_center_distance_px=float(center_distance_px),
                    descent_anchor_z_mm=descent_anchor_z,
                    descent_cooldown_frames=0,
                )
            if current_z > float(descent_anchor_z) - z_pulse_mm:
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
                "pick_ready_frames": int(next_pending.pick_ready_frames),
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
                "settle_stop_band_px": float(settle_band_px),
                "pick_ready_center_px": float(
                    max(0.1, float(getattr(self.config, "vision_continuous_servo_pick_ready_center_px", 8.0)))
                ),
                "fine_pulse": bool(center_distance_px <= float(
                    max(0.1, float(getattr(self.config, "vision_continuous_servo_fine_pulse_center_px", 16.0)))
                )),
                "stable_frames": int(stable_frames),
                "pick_ready_frames": int(next_pending.pick_ready_frames),
                "current_z_mm": float(current_z),
                "confirm_z_mm": float(confirm_z),
                "descent_anchor_z_mm": None if descent_anchor_z is None else float(descent_anchor_z),
                "descent_cooldown_frames": int(descent_cooldown_frames),
                "z_pulse_mm": float(z_pulse_mm),
            },
        )

    def _horizontal_rates(
        self,
        slot_payload: Mapping[str, object],
        *,
        current_cyl_pose: tuple[float, float, float],
        center_distance_px: float | None = None,
        confirm_z: float | None = None,
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
        theta_rate = _clamp(theta_error * theta_gain, -theta_limit, theta_limit)
        radius_rate = _clamp(radius_error * radius_gain, -radius_limit, radius_limit)
        if self._use_low_height_fine_scale(
            current_z=float(current_cyl_pose[2]),
            confirm_z=confirm_z,
            center_distance_px=center_distance_px,
        ):
            scale = max(
                0.05,
                min(1.0, float(getattr(self.config, "vision_continuous_servo_low_height_fine_rate_scale", 0.35))),
            )
            theta_rate *= scale
            radius_rate *= scale
        return (
            theta_rate,
            radius_rate,
        )

    def _use_low_height_fine_scale(
        self,
        *,
        current_z: float,
        confirm_z: float | None,
        center_distance_px: float | None,
    ) -> bool:
        if confirm_z is None or center_distance_px is None:
            return False
        try:
            center_distance = float(center_distance_px)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(center_distance):
            return False
        z_tolerance = max(0.5, float(getattr(self.config, "vision_pick_z_tolerance_mm", 4.0)))
        if float(current_z) > float(confirm_z) + z_tolerance * 2.0:
            return False
        fine_band = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_low_height_fine_band_px", 20.0)),
        )
        return center_distance <= fine_band

    def _low_height_error_rebounded(
        self,
        *,
        current_z: float,
        confirm_z: float,
        center_distance_px: float,
        previous_center_distance_px: float | None,
    ) -> bool:
        if previous_center_distance_px is None:
            return False
        previous = _safe_float(previous_center_distance_px, float("nan"))
        if not (math.isfinite(previous) and math.isfinite(float(center_distance_px))):
            return False
        z_tolerance = max(0.5, float(getattr(self.config, "vision_pick_z_tolerance_mm", 4.0)))
        if float(current_z) > float(confirm_z) + z_tolerance * 2.0:
            return False
        fine_band = max(0.1, float(getattr(self.config, "vision_continuous_servo_low_height_fine_band_px", 10.0)))
        if min(previous, float(center_distance_px)) > fine_band:
            return False
        growth = float(center_distance_px) - previous
        growth_stop = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_low_height_error_growth_stop_px", 2.0)),
        )
        return growth >= growth_stop

    def _fallback_servo_command_point(
        self,
        slot_payload: Mapping[str, object],
        *,
        current_cyl_pose: tuple[float, float, float],
    ) -> list[float] | None:
        for key in ("servo_command_point", "resolved_cyl", "cylindrical_center"):
            point = slot_payload.get(key)
            if isinstance(point, (tuple, list)) and len(point) >= 2:
                target_theta = _safe_float(point[0], float("nan"))
                target_radius = _safe_float(point[1], float("nan"))
                if math.isfinite(target_theta) and math.isfinite(target_radius):
                    return [float(target_theta), float(target_radius)]

        raw_delta = slot_payload.get("camera_to_world_raw")
        if not isinstance(raw_delta, (tuple, list)) or len(raw_delta) < 2:
            return None
        delta_x = _safe_float(raw_delta[0], float("nan"))
        delta_y = _safe_float(raw_delta[1], float("nan"))
        if not (math.isfinite(delta_x) and math.isfinite(delta_y)):
            return None

        gain = max(0.05, min(1.0, float(getattr(self.config, "vision_servo_move_gain", 0.45))))
        center_distance_px = _safe_float(slot_payload.get("center_distance_px"), float("inf"))
        fine_threshold_px = max(0.0, float(getattr(self.config, "vision_servo_fine_threshold_px", 0.0)))
        if math.isfinite(center_distance_px) and center_distance_px <= fine_threshold_px:
            gain = min(gain, max(0.05, min(1.0, float(getattr(self.config, "vision_servo_fine_move_gain", 0.20)))))

        current_x, current_y, _ = cylindrical_to_cartesian(
            float(current_cyl_pose[0]),
            float(current_cyl_pose[1]),
            float(current_cyl_pose[2]),
        )
        target_theta, target_radius, _ = cartesian_to_cylindrical(
            float(current_x) + float(delta_x) * gain,
            float(current_y) + float(delta_y) * gain,
            float(current_cyl_pose[2]),
        )
        if not (math.isfinite(target_theta) and math.isfinite(target_radius)):
            return None
        return [float(target_theta), float(target_radius)]

    def _slot_quality_reject_reason(self, slot_payload: Mapping[str, object]) -> str:
        min_confidence = max(
            0.0,
            min(1.0, float(getattr(self.config, "vision_continuous_servo_min_confidence", 0.55))),
        )
        confidence = _safe_float(slot_payload.get("confidence"), 1.0)
        if math.isfinite(confidence) and confidence < min_confidence:
            return "target_confidence_low"
        min_area = max(1.0, float(getattr(self.config, "vision_continuous_servo_min_area_px", 1500)))
        area = _safe_float(slot_payload.get("area_px"), min_area)
        if math.isfinite(area) and area < min_area:
            return "target_area_too_small"
        bbox = slot_payload.get("bbox")
        if isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
            width = abs(_safe_float(bbox[2], 0.0) - _safe_float(bbox[0], 0.0))
            height = abs(_safe_float(bbox[3], 0.0) - _safe_float(bbox[1], 0.0))
            min_side = max(8.0, math.sqrt(min_area) * 0.45)
            if width < min_side or height < min_side:
                return "target_bbox_too_small"
        return ""

    def _tracking_point(self, slot_payload: Mapping[str, object]) -> tuple[float, float] | None:
        measurement_point = str(getattr(self.config, "vision_servo_measurement_point", "geometry_subpixel")).strip().lower()
        if measurement_point == "grasp_subpixel":
            grasp_px = _safe_pair(slot_payload.get("grasp_pixel_f"))
            if grasp_px is not None:
                return grasp_px
        if measurement_point == "grasp":
            grasp_px = _safe_pair(slot_payload.get("grasp_pixel"))
            if grasp_px is not None:
                return grasp_px
        if measurement_point == "geometry_subpixel":
            geometry_px = _safe_pair(slot_payload.get("geometry_center_f"))
            if geometry_px is not None:
                return geometry_px
        if measurement_point == "geometry":
            geometry_px = _safe_pair(slot_payload.get("geometry_center"))
            if geometry_px is not None:
                return geometry_px
        if measurement_point == "center_subpixel":
            center_px = _safe_pair(slot_payload.get("pixel_center_f"))
            if center_px is not None:
                return center_px
        return _safe_pair(slot_payload.get("pixel_center"))

    def _tracking_reject_reason(
        self,
        *,
        current: ContinuousServoPending,
        center_px: tuple[float, float] | None,
        center_distance_px: float,
    ) -> str:
        if center_px is None:
            return ""
        if current.last_center_px is not None:
            jump = math.hypot(
                float(center_px[0]) - float(current.last_center_px[0]),
                float(center_px[1]) - float(current.last_center_px[1]),
            )
            max_jump = max(1.0, float(getattr(self.config, "vision_continuous_servo_max_center_jump_px", 45.0)))
            if jump > max_jump:
                return "target_center_jump"
        if current.last_center_distance_px is not None:
            growth = float(center_distance_px) - float(current.last_center_distance_px)
            max_growth = max(1.0, float(getattr(self.config, "vision_continuous_servo_max_error_growth_px", 35.0)))
            if growth > max_growth:
                return "target_error_diverged"
        return ""

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
            for key in ("stream_age_ms", "queue_age_ms"):
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

    @staticmethod
    def _replace_pending(
        pending: ContinuousServoPending,
        *,
        pick_ready_frames: int | None = None,
    ) -> ContinuousServoPending:
        return ContinuousServoPending(
            slot_id=int(pending.slot_id),
            stable_frames=int(pending.stable_frames),
            pick_ready_frames=(
                int(pending.pick_ready_frames) if pick_ready_frames is None else max(0, int(pick_ready_frames))
            ),
            lost_frames=int(pending.lost_frames),
            source=str(pending.source),
            last_center_px=pending.last_center_px,
            last_center_distance_px=pending.last_center_distance_px,
            descent_anchor_z_mm=pending.descent_anchor_z_mm,
            descent_cooldown_frames=int(pending.descent_cooldown_frames),
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


def _safe_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    first = _safe_float(value[0], float("nan"))
    second = _safe_float(value[1], float("nan"))
    if not (math.isfinite(first) and math.isfinite(second)):
        return None
    return (float(first), float(second))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))
