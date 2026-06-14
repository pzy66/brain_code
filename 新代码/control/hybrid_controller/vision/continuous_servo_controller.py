from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from hybrid_controller.cylindrical import cartesian_to_cylindrical
from hybrid_controller.cylindrical import cylindrical_to_cartesian
from hybrid_controller.config import HIWONDER_CAMERA_HEIGHT
from hybrid_controller.config import HIWONDER_CAMERA_WIDTH
from hybrid_controller.config import normalize_servo_measurement_point
from hybrid_controller.vision.servo_controller import _biased_pick_cyl_command
from hybrid_controller.vision.servo_controller import _pick_command_sucker_rotation


@dataclass(frozen=True, slots=True)
class ContinuousServoPending:
    slot_id: int
    stable_frames: int = 0
    pick_ready_frames: int = 0
    lost_frames: int = 0
    stale_frames: int = 0
    source: str = ""
    last_center_px: tuple[float, float] | None = None
    last_center_distance_px: float | None = None
    descent_anchor_z_mm: float | None = None
    descent_cooldown_frames: int = 0
    low_height_anchor_pose: tuple[float, float, float] | None = None
    best_center_distance_px: float | None = None
    low_height_static_frames: int = 0
    low_height_static_reference_px: float | None = None
    motion_guard_anchor_pose: tuple[float, float, float] | None = None
    motion_guard_anchor_px: tuple[float, float] | None = None
    motion_guard_static_frames: int = 0

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
        low_height_anchor_pose = _safe_triplet(payload.get("low_height_anchor_pose"))
        best_center_distance = _safe_float(payload.get("best_center_distance_px"), float("nan"))
        low_height_static_reference = _safe_float(payload.get("low_height_static_reference_px"), float("nan"))
        motion_guard_anchor_pose = _safe_triplet(payload.get("motion_guard_anchor_pose"))
        motion_guard_anchor_px = _safe_pair(payload.get("motion_guard_anchor_px"))
        return cls(
            slot_id=int(slot_id),
            stable_frames=_safe_int(payload.get("stable_frames"), 0),
            pick_ready_frames=_safe_int(payload.get("pick_ready_frames"), 0),
            lost_frames=_safe_int(payload.get("lost_frames"), 0),
            stale_frames=_safe_int(payload.get("stale_frames"), 0),
            source=str(payload.get("source", "") or ""),
            last_center_px=last_center,
            last_center_distance_px=last_center_distance if math.isfinite(last_center_distance) else None,
            descent_anchor_z_mm=descent_anchor_z if math.isfinite(descent_anchor_z) else None,
            descent_cooldown_frames=_safe_int(payload.get("descent_cooldown_frames"), 0),
            low_height_anchor_pose=low_height_anchor_pose,
            best_center_distance_px=best_center_distance if math.isfinite(best_center_distance) else None,
            low_height_static_frames=_safe_int(payload.get("low_height_static_frames"), 0),
            low_height_static_reference_px=(
                low_height_static_reference if math.isfinite(low_height_static_reference) else None
            ),
            motion_guard_anchor_pose=motion_guard_anchor_pose,
            motion_guard_anchor_px=motion_guard_anchor_px,
            motion_guard_static_frames=_safe_int(payload.get("motion_guard_static_frames"), 0),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": int(self.slot_id),
            "stable_frames": int(self.stable_frames),
            "pick_ready_frames": int(self.pick_ready_frames),
            "lost_frames": int(self.lost_frames),
            "stale_frames": int(self.stale_frames),
            "source": str(self.source),
            "last_center_px": (
                None if self.last_center_px is None else [float(self.last_center_px[0]), float(self.last_center_px[1])]
            ),
            "last_center_distance_px": (
                None if self.last_center_distance_px is None else float(self.last_center_distance_px)
            ),
            "descent_anchor_z_mm": None if self.descent_anchor_z_mm is None else float(self.descent_anchor_z_mm),
            "descent_cooldown_frames": int(self.descent_cooldown_frames),
            "low_height_anchor_pose": (
                None
                if self.low_height_anchor_pose is None
                else [
                    float(self.low_height_anchor_pose[0]),
                    float(self.low_height_anchor_pose[1]),
                    float(self.low_height_anchor_pose[2]),
                ]
            ),
            "best_center_distance_px": (
                None if self.best_center_distance_px is None else float(self.best_center_distance_px)
            ),
            "low_height_static_frames": int(self.low_height_static_frames),
            "low_height_static_reference_px": (
                None if self.low_height_static_reference_px is None else float(self.low_height_static_reference_px)
            ),
            "motion_guard_anchor_pose": (
                None
                if self.motion_guard_anchor_pose is None
                else [
                    float(self.motion_guard_anchor_pose[0]),
                    float(self.motion_guard_anchor_pose[1]),
                    float(self.motion_guard_anchor_pose[2]),
                ]
            ),
            "motion_guard_anchor_px": (
                None
                if self.motion_guard_anchor_px is None
                else [float(self.motion_guard_anchor_px[0]), float(self.motion_guard_anchor_px[1])]
            ),
            "motion_guard_static_frames": int(self.motion_guard_static_frames),
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


@dataclass(frozen=True, slots=True)
class _IbvsJacobian:
    matrix: tuple[float, float, float, float]
    source: str
    gain: float
    damping: float
    det: float
    condition_number: float | None


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
            stale = current.stale_frames + 1
            next_pending = ContinuousServoPending(
                slot_id=int(slot_id),
                stable_frames=0,
                pick_ready_frames=0,
                lost_frames=0,
                stale_frames=stale,
                source=str(current.source),
                last_center_px=current.last_center_px,
                last_center_distance_px=current.last_center_distance_px,
                descent_anchor_z_mm=None,
                descent_cooldown_frames=max(
                    1,
                    int(getattr(self.config, "vision_continuous_servo_stable_frames", 2)),
                ),
                low_height_anchor_pose=current.low_height_anchor_pose,
                best_center_distance_px=current.best_center_distance_px,
                low_height_static_frames=current.low_height_static_frames,
                low_height_static_reference_px=current.low_height_static_reference_px,
            )
            if stale >= max(1, int(getattr(self.config, "vision_continuous_servo_stale_frames", 3))):
                return self._stop("frame_stale", current=next_pending)
            return ContinuousServoDecision(
                action="STOP",
                status=f"waiting for fresh frame {stale}",
                reason="frame_stale_wait",
                pending=next_pending,
                trace={"stale_frames": int(stale)},
            )
        if slot_payload is None or not bool(slot_payload.get("valid", False)):
            lost = current.lost_frames + 1
            next_pending = ContinuousServoPending(
                slot_id=int(slot_id),
                stable_frames=0,
                pick_ready_frames=0,
                lost_frames=lost,
                stale_frames=0,
                source=str(current.source),
                last_center_px=current.last_center_px,
                last_center_distance_px=current.last_center_distance_px,
                descent_anchor_z_mm=current.descent_anchor_z_mm,
                descent_cooldown_frames=current.descent_cooldown_frames,
                low_height_anchor_pose=current.low_height_anchor_pose,
                best_center_distance_px=current.best_center_distance_px,
                low_height_static_frames=current.low_height_static_frames,
                low_height_static_reference_px=current.low_height_static_reference_px,
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
                    stale_frames=0,
                    source=str(current.source),
                    last_center_px=current.last_center_px,
                    last_center_distance_px=current.last_center_distance_px,
                    descent_anchor_z_mm=current.descent_anchor_z_mm,
                    descent_cooldown_frames=current.descent_cooldown_frames,
                    low_height_anchor_pose=current.low_height_anchor_pose,
                    best_center_distance_px=current.best_center_distance_px,
                    low_height_static_frames=current.low_height_static_frames,
                    low_height_static_reference_px=current.low_height_static_reference_px,
                ),
                trace={"quality_reason": quality_reason},
            )
        confirm_z = float(getattr(self.config, "vision_pick_confirm_z_mm", getattr(self.config, "robot_approach_z", 130.0)))
        z_tolerance = max(0.0, float(getattr(self.config, "vision_pick_z_tolerance_mm", 4.0)))
        measurement_point = self._measurement_point_for_pose(current_z=current_z, confirm_z=confirm_z)
        center_distance_px = self._center_distance_px(slot_payload, packet, measurement_point=measurement_point)
        if not math.isfinite(center_distance_px):
            return ContinuousServoDecision(
                action="STOP",
                status=f"waiting for measurable target center slot={int(slot_id)}",
                reason="center_distance_unavailable_wait",
                pending=ContinuousServoPending(
                    slot_id=int(slot_id),
                    stable_frames=0,
                    pick_ready_frames=0,
                    lost_frames=0,
                    stale_frames=0,
                    source=str(current.source),
                    last_center_px=current.last_center_px,
                    last_center_distance_px=current.last_center_distance_px,
                    descent_anchor_z_mm=None,
                    descent_cooldown_frames=max(
                        1,
                        int(getattr(self.config, "vision_continuous_servo_stable_frames", 2)),
                    ),
                    low_height_anchor_pose=current.low_height_anchor_pose,
                    best_center_distance_px=current.best_center_distance_px,
                    low_height_static_frames=current.low_height_static_frames,
                    low_height_static_reference_px=current.low_height_static_reference_px,
                ),
                trace={"reason": "center_distance_unavailable"},
            )
        low_height_error_rebounded = self._low_height_error_rebounded(
            current_z=current_z,
            confirm_z=confirm_z,
            center_distance_px=center_distance_px,
            previous_center_distance_px=current.last_center_distance_px,
        )
        tracking_px = self._tracking_point(slot_payload, measurement_point=measurement_point)
        tracking_reason = self._tracking_reject_reason(
            current=current,
            center_px=tracking_px,
            center_distance_px=center_distance_px,
        )
        if tracking_reason:
            return ContinuousServoDecision(
                action="STOP",
                status=f"holding after {tracking_reason} slot={int(slot_id)}",
                reason="hold",
                pending=ContinuousServoPending(
                    slot_id=int(slot_id),
                    stable_frames=0,
                    pick_ready_frames=0,
                    lost_frames=0,
                    stale_frames=0,
                    source=str(current.source),
                    last_center_px=current.last_center_px,
                    last_center_distance_px=current.last_center_distance_px,
                    descent_anchor_z_mm=None,
                    descent_cooldown_frames=max(
                        1,
                        int(getattr(self.config, "vision_continuous_servo_stable_frames", 2)),
                    ),
                    low_height_anchor_pose=current.low_height_anchor_pose,
                    best_center_distance_px=current.best_center_distance_px,
                    low_height_static_frames=current.low_height_static_frames,
                    low_height_static_reference_px=current.low_height_static_reference_px,
                ),
                trace={
                    "reason": tracking_reason,
                    "center_distance_px": float(center_distance_px),
                    "last_center_distance_px": None
                    if current.last_center_distance_px is None
                    else float(current.last_center_distance_px),
                },
            )
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
        z_pulse_mm = max(0.0, float(getattr(self.config, "vision_continuous_servo_z_pulse_mm", 0.0)))
        pulse_mode_enabled = z_pulse_mm > 0.0
        descent_anchor_z = current.descent_anchor_z_mm
        descent_cooldown_frames = max(0, int(current.descent_cooldown_frames))
        completed_descent_pulse = (
            pulse_mode_enabled
            and descent_anchor_z is not None
            and current_z <= float(descent_anchor_z) - z_pulse_mm
        )
        if stable_frames == 0 or completed_descent_pulse:
            descent_anchor_z = None
            descent_cooldown_frames = max(descent_cooldown_frames, required_stable)
            if completed_descent_pulse:
                stable_frames = 0
        elif descent_cooldown_frames > 0 and stable_frames > 0:
            descent_cooldown_frames = max(0, descent_cooldown_frames - 1)
        next_pending = ContinuousServoPending(
            slot_id=int(slot_id),
            stable_frames=stable_frames,
            lost_frames=0,
            stale_frames=0,
            source=str(current.source),
            last_center_px=tracking_px,
            last_center_distance_px=float(center_distance_px),
            descent_anchor_z_mm=descent_anchor_z,
            descent_cooldown_frames=descent_cooldown_frames,
            low_height_anchor_pose=current.low_height_anchor_pose,
            best_center_distance_px=current.best_center_distance_px,
            low_height_static_frames=current.low_height_static_frames,
            low_height_static_reference_px=current.low_height_static_reference_px,
            motion_guard_anchor_pose=current.motion_guard_anchor_pose,
            motion_guard_anchor_px=current.motion_guard_anchor_px,
            motion_guard_static_frames=current.motion_guard_static_frames,
        )
        if center_distance_px > max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_pick_ready_center_px", 8.0)),
        ):
            next_pending = self._replace_pending(next_pending, pick_ready_frames=0)
        next_pending = self._update_low_height_guard_pending(
            next_pending,
            current_cyl_pose=(theta_deg, radius_mm, current_z),
            confirm_z=confirm_z,
            center_distance_px=center_distance_px,
        )
        in_low_height_guard_zone = self._is_low_height_guard_zone(current_z=current_z, confirm_z=confirm_z)
        in_low_height_pause_zone = self._is_low_height_pause_descent_zone(current_z=current_z, confirm_z=confirm_z)
        descent_allow_px = self._descent_error_allow_px(current_z=current_z, confirm_z=confirm_z)
        low_height_best_confirm_descent_allowed = self._low_height_best_confirm_descent_allowed(
            current_z=current_z,
            confirm_z=confirm_z,
            center_distance_px=center_distance_px,
            best_center_distance_px=next_pending.best_center_distance_px,
            descent_allow_px=descent_allow_px,
            in_low_height_pause_zone=in_low_height_pause_zone,
        )
        low_height_guard_reason = self._low_height_guard_stop_reason(
            pending=next_pending,
            current_cyl_pose=(theta_deg, radius_mm, current_z),
            confirm_z=confirm_z,
            center_distance_px=center_distance_px,
            allow_confirm_descent=low_height_best_confirm_descent_allowed,
        )
        if low_height_guard_reason:
            stopped_pending = self._replace_pending(next_pending, stable_frames=0, pick_ready_frames=0)
            return ContinuousServoDecision(
                action="STOP",
                status=f"stopping low-height guard slot={int(slot_id)} reason={low_height_guard_reason}",
                reason=low_height_guard_reason,
                pending=stopped_pending,
                trace=self._low_height_guard_trace(
                    pending=stopped_pending,
                    current_cyl_pose=(theta_deg, radius_mm, current_z),
                    confirm_z=confirm_z,
                    center_distance_px=center_distance_px,
                ),
            )
        if low_height_error_rebounded:
            rebounded_pending = ContinuousServoPending(
                slot_id=int(next_pending.slot_id),
                stable_frames=0,
                pick_ready_frames=0,
                lost_frames=0,
                stale_frames=0,
                source=str(next_pending.source),
                last_center_px=next_pending.last_center_px,
                last_center_distance_px=next_pending.last_center_distance_px,
                descent_anchor_z_mm=next_pending.descent_anchor_z_mm,
                descent_cooldown_frames=int(next_pending.descent_cooldown_frames),
                low_height_anchor_pose=next_pending.low_height_anchor_pose,
                best_center_distance_px=next_pending.best_center_distance_px,
                low_height_static_frames=next_pending.low_height_static_frames,
                low_height_static_reference_px=next_pending.low_height_static_reference_px,
                motion_guard_anchor_pose=next_pending.motion_guard_anchor_pose,
                motion_guard_anchor_px=next_pending.motion_guard_anchor_px,
                motion_guard_static_frames=next_pending.motion_guard_static_frames,
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
        original_invalid_reason = str(slot_payload.get("invalid_reason") or "").strip()
        invalid_reason = original_invalid_reason
        horizontal_mode = str(
            getattr(self.config, "vision_continuous_servo_horizontal_mode", "servo_command_point")
        ).strip().lower()
        if _safe_pair(slot_payload.get("alignment_target_pixel")) is None:
            packet_target = _safe_pair(packet.get("alignment_target_pixel")) if isinstance(packet, Mapping) else None
            legacy_synthetic_target = packet_target is None
            if packet_target is None:
                packet_target = (
                    float(HIWONDER_CAMERA_WIDTH) * 0.5,
                    float(HIWONDER_CAMERA_HEIGHT) * 0.5,
                )
            slot_payload = dict(slot_payload)
            slot_payload["alignment_target_pixel"] = [float(packet_target[0]), float(packet_target[1])]
            if legacy_synthetic_target:
                slot_payload["_synthetic_alignment_target_pixel"] = True
        has_ibvs_feedback = (
            self._ibvs_tracking_point(
                slot_payload,
                measurement_point=measurement_point,
                current_cyl_pose=(theta_deg, radius_mm, current_z),
                confirm_z=confirm_z,
            )
            is not None
            and _safe_pair(slot_payload.get("alignment_target_pixel")) is not None
        )
        pixel_control_ready = (
            horizontal_mode in {"pixel_axis", "pixel_jacobian", "ibvs_dls"}
            and (
                tracking_px is not None
                or (
                    horizontal_mode == "ibvs_dls"
                    and self._ibvs_tracking_point(
                        slot_payload,
                        measurement_point=measurement_point,
                        current_cyl_pose=(theta_deg, radius_mm, current_z),
                        confirm_z=confirm_z,
                    )
                    is not None
                )
            )
            and _safe_pair(slot_payload.get("alignment_target_pixel")) is not None
        )
        if horizontal_mode == "ibvs_dls":
            pixel_control_ready = pixel_control_ready and self._ibvs_jacobian_available(
                current_cyl_pose=(theta_deg, radius_mm, current_z)
            )
        direct_pixel_horizontal = bool(pixel_control_ready)
        if (
            horizontal_mode == "ibvs_dls"
            and invalid_reason in {"vision_servo_required", ""}
            and not bool(slot_payload.get("actionable", False))
                    and has_ibvs_feedback
                    and not bool(slot_payload.get("_synthetic_alignment_target_pixel", False))
                    and not direct_pixel_horizontal
        ):
            reason = "ibvs_target_unavailable"
            if self._ibvs_tracking_point(
                slot_payload,
                measurement_point=measurement_point,
                current_cyl_pose=(theta_deg, radius_mm, current_z),
                confirm_z=confirm_z,
            ) is not None and _safe_pair(slot_payload.get("alignment_target_pixel")) is not None:
                reason = "ibvs_jacobian_unavailable"
            return ContinuousServoDecision(
                action="STOP",
                status=f"holding ibvs without usable image feedback slot={int(slot_id)}",
                reason=reason,
                pending=next_pending,
                trace={
                    "reason": reason,
                    "horizontal_mode": "ibvs_dls",
                    "tracking_point_available": tracking_px is not None,
                    "alignment_target_available": _safe_pair(slot_payload.get("alignment_target_pixel")) is not None,
                },
            )
        if (
            invalid_reason == "vision_servo_required"
            and horizontal_mode == "ibvs_dls"
            and tracking_px is not None
            and _safe_pair(slot_payload.get("alignment_target_pixel")) is not None
            and not self._ibvs_jacobian_available(current_cyl_pose=(theta_deg, radius_mm, current_z))
        ):
            return ContinuousServoDecision(
                action="STOP",
                status=f"holding ibvs without usable jacobian slot={int(slot_id)}",
                reason="ibvs_jacobian_unavailable",
                pending=next_pending,
                trace={"reason": "ibvs_jacobian_unavailable"},
            )
        if invalid_reason == "grasp_unstable" and direct_pixel_horizontal:
            invalid_reason = "vision_servo_required"
        unstable_hold_window_px = max(center_stop_px, center_allow_px * 4.0)
        unstable_near_center = center_distance_px <= center_allow_px
        unstable_after_lock_or_descent = (
            center_distance_px <= unstable_hold_window_px
            and (current.stable_frames > 0 or current_z <= confirm_z + 60.0)
        )
        low_height_unstable_servo_limit_px = max(
            center_stop_px,
            float(getattr(self.config, "vision_continuous_servo_low_height_unstable_servo_px", 60.0)),
        )
        low_height_unstable_servo_allowed = (
            self._is_low_height_guard_zone(current_z=current_z, confirm_z=confirm_z)
            and center_distance_px <= low_height_unstable_servo_limit_px
        )
        if invalid_reason == "grasp_unstable" and (
            center_distance_px <= unstable_hold_window_px or low_height_unstable_servo_allowed
        ):
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
                candidate_point = None
                if low_height_unstable_servo_allowed:
                    candidate_point = self._fallback_servo_command_point(
                        slot_payload,
                        current_cyl_pose=(theta_deg, radius_mm, current_z),
                    )
                if candidate_point is not None:
                    invalid_reason = "vision_servo_required"
                    slot_payload = dict(slot_payload)
                    slot_payload["servo_command_point"] = candidate_point
                    slot_payload["servo_command_mode"] = str(slot_payload.get("servo_command_mode", "cyl"))
                    next_pending = self._replace_pending(next_pending, stable_frames=0, pick_ready_frames=0)
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
                stale_frames=0,
                source=str(current.source),
                last_center_px=tracking_px,
                last_center_distance_px=float(center_distance_px),
                descent_anchor_z_mm=descent_anchor_z,
                descent_cooldown_frames=descent_cooldown_frames,
                low_height_anchor_pose=next_pending.low_height_anchor_pose,
                best_center_distance_px=next_pending.best_center_distance_px,
                low_height_static_frames=next_pending.low_height_static_frames,
                low_height_static_reference_px=next_pending.low_height_static_reference_px,
                motion_guard_anchor_pose=next_pending.motion_guard_anchor_pose,
                motion_guard_anchor_px=next_pending.motion_guard_anchor_px,
                motion_guard_static_frames=next_pending.motion_guard_static_frames,
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
            slot_for_horizontal = dict(slot_payload)
            if not direct_pixel_horizontal:
                if (
                    horizontal_mode == "ibvs_dls"
                    and has_ibvs_feedback
                    and not bool(slot_payload.get("_synthetic_alignment_target_pixel", False))
                ):
                    return ContinuousServoDecision(
                        action="STOP",
                        status=f"holding ibvs without usable image feedback slot={int(slot_id)}",
                        reason="ibvs_jacobian_unavailable",
                        pending=next_pending,
                        trace={
                            "reason": "ibvs_jacobian_unavailable",
                            "horizontal_mode": "ibvs_dls",
                        },
                    )
                fallback_point = self._fallback_servo_command_point(
                    slot_payload,
                    current_cyl_pose=(theta_deg, radius_mm, current_z),
                )
                if fallback_point is None:
                    return self._stop("servo_command_unavailable", current=current)
                slot_for_horizontal["servo_command_point"] = fallback_point
                slot_for_horizontal["servo_command_mode"] = str(slot_payload.get("servo_command_mode", "cyl"))
            elif horizontal_mode == "ibvs_dls" and not self._ibvs_jacobian_available(
                current_cyl_pose=(theta_deg, radius_mm, current_z)
            ):
                return ContinuousServoDecision(
                    action="STOP",
                    status=f"holding ibvs without usable jacobian slot={int(slot_id)}",
                    reason="ibvs_jacobian_unavailable",
                    pending=next_pending,
                    trace={"reason": "ibvs_jacobian_unavailable"},
                )
        if bool(slot_payload.get("actionable", False)):
            pick_ready_center_px = max(
                0.1,
                float(getattr(self.config, "vision_continuous_servo_pick_ready_center_px", 8.0)),
            )
            if horizontal_mode != "ibvs_dls" and center_distance_px > pick_ready_center_px:
                fallback_point = self._fallback_servo_command_point(
                    slot_payload,
                    current_cyl_pose=(theta_deg, radius_mm, current_z),
                )
                if fallback_point is not None:
                    slot_for_horizontal = dict(slot_payload)
                    slot_for_horizontal["servo_command_point"] = fallback_point
                    slot_for_horizontal["servo_command_mode"] = str(slot_payload.get("servo_command_mode", "cyl"))

        theta_rate, radius_rate, horizontal_trace = self._horizontal_rates(
            slot_for_horizontal,
            current_cyl_pose=(theta_deg, radius_mm, current_z),
            center_distance_px=center_distance_px,
            confirm_z=confirm_z,
        )
        if (
            str(horizontal_trace.get("horizontal_mode", "")).lower() == "ibvs_dls"
            and str(horizontal_trace.get("horizontal_reason", "")).lower() == "ibvs_unavailable"
        ):
            return ContinuousServoDecision(
                action="STOP",
                status=f"holding ibvs without usable image feedback slot={int(slot_id)}",
                reason="ibvs_rate_unavailable",
                pending=next_pending,
                trace={
                    "reason": "ibvs_rate_unavailable",
                    "center_distance_px": float(center_distance_px),
                    "current_z_mm": float(current_z),
                    "confirm_z_mm": float(confirm_z),
                    **horizontal_trace,
                },
            )
        low_height_descent_rebound = self._low_height_descent_rebound(
            current_z=current_z,
            confirm_z=confirm_z,
            center_distance_px=center_distance_px,
            previous_center_distance_px=current.last_center_distance_px,
        )
        low_height_best_descent_pause = self._low_height_best_descent_pause(
            current_z=current_z,
            confirm_z=confirm_z,
            center_distance_px=center_distance_px,
            best_center_distance_px=next_pending.best_center_distance_px,
        )
        z_rate_scale_reason = self._z_rate_scale_reason(
            current_z=current_z,
            confirm_z=confirm_z,
            center_distance_px=center_distance_px,
        )
        z_rate = 0.0
        soft_descent = False
        if stable_frames >= required_stable and descent_cooldown_frames <= 0 and current_z > confirm_z + z_tolerance:
            if descent_anchor_z is None:
                descent_anchor_z = float(current_z)
                next_pending = ContinuousServoPending(
                    slot_id=int(slot_id),
                    stable_frames=stable_frames,
                    pick_ready_frames=0,
                    lost_frames=0,
                    stale_frames=0,
                    source=str(current.source),
                    last_center_px=tracking_px,
                    last_center_distance_px=float(center_distance_px),
                    descent_anchor_z_mm=descent_anchor_z,
                    descent_cooldown_frames=0,
                    low_height_anchor_pose=next_pending.low_height_anchor_pose,
                    best_center_distance_px=next_pending.best_center_distance_px,
                    low_height_static_frames=next_pending.low_height_static_frames,
                    low_height_static_reference_px=next_pending.low_height_static_reference_px,
                    motion_guard_anchor_pose=next_pending.motion_guard_anchor_pose,
                    motion_guard_anchor_px=next_pending.motion_guard_anchor_px,
                    motion_guard_static_frames=next_pending.motion_guard_static_frames,
                )
            if (not pulse_mode_enabled) or current_z > float(descent_anchor_z) - z_pulse_mm:
                z_rate = -self._z_rate(current_z=current_z, confirm_z=confirm_z)
                if z_rate_scale_reason:
                    z_rate *= self._low_height_z_rate_scale()
            if low_height_descent_rebound and not low_height_best_confirm_descent_allowed:
                z_rate = 0.0
            if low_height_best_descent_pause and not low_height_best_confirm_descent_allowed:
                z_rate = 0.0
            if (
                in_low_height_pause_zone
                and center_distance_px > center_allow_px
                and not low_height_best_confirm_descent_allowed
            ):
                z_rate = 0.0
        elif (
            bool(getattr(self.config, "vision_continuous_servo_soft_descent_enabled", True))
            and (
                (
                    direct_pixel_horizontal
                    and (center_distance_px <= descent_allow_px or low_height_best_confirm_descent_allowed)
                )
                or (
                    not direct_pixel_horizontal
                    and center_distance_px > center_allow_px
                    and center_distance_px <= center_stop_px
                )
            )
            and (
                original_invalid_reason in {"", "vision_servo_required"}
                or (direct_pixel_horizontal and original_invalid_reason == "grasp_unstable")
            )
            and (not in_low_height_pause_zone or low_height_best_confirm_descent_allowed)
            and current_z
            > confirm_z
            + (
                z_tolerance
                if in_low_height_guard_zone
                else max(
                    z_tolerance,
                    float(getattr(self.config, "vision_continuous_servo_soft_descent_min_z_above_confirm_mm", 18.0)),
                )
            )
        ):
            soft_scale = max(
                0.0,
                min(1.0, float(getattr(self.config, "vision_continuous_servo_soft_descent_rate_scale", 0.35))),
            )
            if soft_scale > 0.0:
                z_rate = -self._z_rate(current_z=current_z, confirm_z=confirm_z) * soft_scale
                if z_rate_scale_reason:
                    z_rate *= self._low_height_z_rate_scale()
                if low_height_descent_rebound and not low_height_best_confirm_descent_allowed:
                    z_rate = 0.0
                if low_height_best_descent_pause and not low_height_best_confirm_descent_allowed:
                    z_rate = 0.0
                soft_descent = True
        elif center_distance_px > center_stop_px:
            z_rate = 0.0

        next_pending, motion_guard_trace = self._motion_response_guard_update(
            next_pending,
            current_cyl_pose=(theta_deg, radius_mm, current_z),
            current_px=tracking_px,
        )
        if motion_guard_trace is not None:
            return ContinuousServoDecision(
                action="STOP",
                status="stopping because camera image did not respond to robot motion",
                reason="camera_motion_response_missing",
                pending=self._replace_pending(next_pending, stable_frames=0, pick_ready_frames=0),
                trace={
                    "center_distance_px": float(center_distance_px),
                    "current_z_mm": float(current_z),
                    "confirm_z_mm": float(confirm_z),
                    **motion_guard_trace,
                },
            )

        if abs(theta_rate) < 1e-6 and abs(radius_rate) < 1e-6 and abs(z_rate) < 1e-6:
            if (
                current_z <= confirm_z + z_tolerance * 2.0
                and center_distance_px <= settle_band_px
                and stable_frames < required_stable
            ):
                hold_reason = "settle_near_center"
                hold_status = f"settling near center slot={int(slot_id)}"
            else:
                hold_reason = "hold"
                hold_status = f"holding slot={int(slot_id)}"
            return ContinuousServoDecision(
                action="STOP",
                status=hold_status,
                reason=hold_reason,
                pending=next_pending,
                trace={
                    "center_distance_px": float(center_distance_px),
                    "settle_stop_band_px": float(settle_band_px),
                    "stable_frames": int(stable_frames),
                    "required_stable_frames": int(required_stable),
                    "pick_ready_frames": int(next_pending.pick_ready_frames),
                    "current_z_mm": float(current_z),
                    "confirm_z_mm": float(confirm_z),
                    "zero_rate_hold": True,
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
                "descent_error_allow_px": float(descent_allow_px),
                "settle_stop_band_px": float(settle_band_px),
                "pick_ready_center_px": float(
                    max(0.1, float(getattr(self.config, "vision_continuous_servo_pick_ready_center_px", 8.0)))
                ),
                "fine_pulse": bool(
                    float(getattr(self.config, "vision_continuous_servo_fine_pulse_center_px", 0.0)) > 0.0
                    and center_distance_px
                    <= float(max(0.1, float(getattr(self.config, "vision_continuous_servo_fine_pulse_center_px", 0.0))))
                ),
                "stable_frames": int(stable_frames),
                "pick_ready_frames": int(next_pending.pick_ready_frames),
                "current_z_mm": float(current_z),
                "confirm_z_mm": float(confirm_z),
                "descent_anchor_z_mm": None if descent_anchor_z is None else float(descent_anchor_z),
                "descent_cooldown_frames": int(descent_cooldown_frames),
                "z_pulse_mm": float(z_pulse_mm),
                "soft_descent": bool(soft_descent),
                "soft_descent_rate_scale": float(
                    getattr(self.config, "vision_continuous_servo_soft_descent_rate_scale", 0.35)
                ),
                "low_height_guard_active": bool(in_low_height_guard_zone),
                "low_height_pause_descent_active": bool(in_low_height_pause_zone),
                "low_height_descent_rebound": bool(low_height_descent_rebound),
                "low_height_best_descent_pause": bool(low_height_best_descent_pause),
                "low_height_best_confirm_descent_allowed": bool(low_height_best_confirm_descent_allowed),
                "low_height_best_confirm_descent_allow_px": float(
                    max(
                        descent_allow_px,
                        float(
                            getattr(
                                self.config,
                                "vision_continuous_servo_low_height_best_confirm_descent_allow_px",
                                descent_allow_px,
                            )
                        ),
                    )
                ),
                "z_rate_scale_reason": z_rate_scale_reason,
                "low_height_z_rate_scale": self._low_height_z_rate_scale(),
                "low_height_anchor_pose": (
                    None if next_pending.low_height_anchor_pose is None else list(next_pending.low_height_anchor_pose)
                ),
                "measurement_point": str(measurement_point),
                "base_measurement_point": normalize_servo_measurement_point(
                    getattr(self.config, "vision_servo_measurement_point", "geometry_subpixel")
                ),
                "low_height_measurement_point": str(
                    getattr(self.config, "vision_servo_low_height_measurement_point", "") or ""
                ),
                "best_center_distance_px": (
                    None
                    if next_pending.best_center_distance_px is None
                    else float(next_pending.best_center_distance_px)
                ),
                "low_height_static_frames": int(next_pending.low_height_static_frames),
                "low_height_static_reference_px": (
                    None
                    if next_pending.low_height_static_reference_px is None
                    else float(next_pending.low_height_static_reference_px)
                ),
                **horizontal_trace,
            },
        )

    def _motion_response_guard_update(
        self,
        pending: ContinuousServoPending,
        *,
        current_cyl_pose: tuple[float, float, float],
        current_px: tuple[float, float] | None,
    ) -> tuple[ContinuousServoPending, dict[str, object] | None]:
        if not bool(getattr(self.config, "vision_continuous_servo_camera_motion_guard_enabled", False)):
            return pending, None
        if current_px is None:
            return self._with_motion_guard(pending, anchor_pose=None, anchor_px=None, static_frames=0), None
        current_point = (float(current_px[0]), float(current_px[1]))
        anchor_pose = pending.motion_guard_anchor_pose
        anchor_px = pending.motion_guard_anchor_px
        if anchor_pose is None or anchor_px is None:
            return (
                self._with_motion_guard(
                    pending,
                    anchor_pose=current_cyl_pose,
                    anchor_px=current_point,
                    static_frames=0,
                ),
                None,
            )

        robot_delta = self._cyl_horizontal_delta_mm(anchor_pose, current_cyl_pose)
        min_robot = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_camera_motion_guard_min_robot_mm", 8.0)),
        )
        if robot_delta < min_robot:
            return self._with_motion_guard(pending, anchor_pose=anchor_pose, anchor_px=anchor_px, static_frames=0), None

        pixel_delta = math.hypot(float(current_point[0]) - float(anchor_px[0]), float(current_point[1]) - float(anchor_px[1]))
        max_pixel = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_camera_motion_guard_max_pixel_px", 2.5)),
        )
        if pixel_delta > max_pixel:
            return (
                self._with_motion_guard(
                    pending,
                    anchor_pose=current_cyl_pose,
                    anchor_px=current_point,
                    static_frames=0,
                ),
                None,
            )

        static_frames = int(pending.motion_guard_static_frames) + 1
        guarded = self._with_motion_guard(
            pending,
            anchor_pose=anchor_pose,
            anchor_px=anchor_px,
            static_frames=static_frames,
        )
        required = max(
            1,
            int(getattr(self.config, "vision_continuous_servo_camera_motion_guard_static_frames", 5)),
        )
        if static_frames < required:
            return guarded, None
        return guarded, {
            "reason": "camera_motion_response_missing",
            "robot_horizontal_delta_mm": float(robot_delta),
            "pixel_delta_px": float(pixel_delta),
            "motion_guard_static_frames": int(static_frames),
            "motion_guard_required_static_frames": int(required),
            "motion_guard_min_robot_mm": float(min_robot),
            "motion_guard_max_pixel_px": float(max_pixel),
            "motion_guard_anchor_pose": [float(anchor_pose[0]), float(anchor_pose[1]), float(anchor_pose[2])],
            "motion_guard_current_pose": [
                float(current_cyl_pose[0]),
                float(current_cyl_pose[1]),
                float(current_cyl_pose[2]),
            ],
            "motion_guard_anchor_px": [float(anchor_px[0]), float(anchor_px[1])],
            "motion_guard_current_px": [float(current_point[0]), float(current_point[1])],
        }

    @staticmethod
    def _cyl_horizontal_delta_mm(
        first: tuple[float, float, float],
        second: tuple[float, float, float],
    ) -> float:
        first_xy = cylindrical_to_cartesian(float(first[0]), float(first[1]), float(first[2]))
        second_xy = cylindrical_to_cartesian(float(second[0]), float(second[1]), float(second[2]))
        return float(math.hypot(float(second_xy[0]) - float(first_xy[0]), float(second_xy[1]) - float(first_xy[1])))

    def _horizontal_rates(
        self,
        slot_payload: Mapping[str, object],
        *,
        current_cyl_pose: tuple[float, float, float],
        center_distance_px: float | None = None,
        confirm_z: float | None = None,
    ) -> tuple[float, float, dict[str, object]]:
        mode = str(getattr(self.config, "vision_continuous_servo_horizontal_mode", "servo_command_point")).strip().lower()
        if mode == "ibvs_dls":
            rates = self._ibvs_dls_horizontal_rates(
                slot_payload,
                current_cyl_pose=current_cyl_pose,
                center_distance_px=center_distance_px,
                confirm_z=confirm_z,
            )
            if rates is not None:
                return rates
            return (0.0, 0.0, {"horizontal_mode": "ibvs_dls", "horizontal_reason": "ibvs_unavailable"})
        if mode == "pixel_jacobian":
            rates = self._pixel_jacobian_horizontal_rates(
                slot_payload,
                current_cyl_pose=current_cyl_pose,
                center_distance_px=center_distance_px,
                confirm_z=confirm_z,
            )
            if rates is not None:
                return (rates[0], rates[1], {"horizontal_mode": "pixel_jacobian"})
        if mode == "pixel_axis":
            rates = self._pixel_axis_horizontal_rates(
                slot_payload,
                current_cyl_pose=current_cyl_pose,
                center_distance_px=center_distance_px,
                confirm_z=confirm_z,
            )
            if rates is not None:
                return (rates[0], rates[1], {"horizontal_mode": "pixel_axis"})
        point = slot_payload.get("servo_command_point")
        if not isinstance(point, (tuple, list)) or len(point) < 2:
            return (0.0, 0.0, {"horizontal_mode": "servo_command_point", "horizontal_reason": "missing_target"})
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
        low_height_scale_reason = self._low_height_fine_scale_reason(
            current_z=float(current_cyl_pose[2]),
            confirm_z=confirm_z,
            center_distance_px=center_distance_px,
        )
        if low_height_scale_reason:
            scale = self._low_height_horizontal_scale(center_distance_px=center_distance_px)
            theta_rate *= scale
            radius_rate *= scale
        return (theta_rate, radius_rate, {"horizontal_mode": "servo_command_point"})

    def _ibvs_dls_horizontal_rates(
        self,
        slot_payload: Mapping[str, object],
        *,
        current_cyl_pose: tuple[float, float, float],
        center_distance_px: float | None = None,
        confirm_z: float | None = None,
    ) -> tuple[float, float, dict[str, object]] | None:
        tracking_point = self._ibvs_tracking_point(slot_payload, current_cyl_pose=current_cyl_pose, confirm_z=confirm_z)
        target = _safe_pair(slot_payload.get("alignment_target_pixel"))
        if tracking_point is None or target is None:
            return None
        error_x = float(tracking_point[0]) - float(target[0])
        error_y = float(tracking_point[1]) - float(target[1])
        if not (math.isfinite(error_x) and math.isfinite(error_y)):
            return None

        # Image-based visual servoing control: q_dot = -lambda * J^+_damped * e.
        # J maps [theta_deg, radius_mm] motion to [u_px, v_px] image motion.
        jacobian = self._ibvs_jacobian(current_cyl_pose=current_cyl_pose)
        if jacobian is None:
            return None
        j00, j01, j10, j11 = jacobian.matrix
        gain = float(jacobian.gain)
        damping = float(jacobian.damping)
        # Solve (J^T J + mu^2 I) q = -gain * J^T e for the two horizontal axes.
        a00 = j00 * j00 + j10 * j10 + damping * damping
        a01 = j00 * j01 + j10 * j11
        a11 = j01 * j01 + j11 * j11 + damping * damping
        b0 = -(j00 * error_x + j10 * error_y) * gain
        b1 = -(j01 * error_x + j11 * error_y) * gain
        det = a00 * a11 - a01 * a01
        if not math.isfinite(det) or abs(det) < 1e-9:
            return None
        theta_rate = (b0 * a11 - a01 * b1) / det
        radius_rate = (a00 * b1 - a01 * b0) / det
        if not (math.isfinite(theta_rate) and math.isfinite(radius_rate)):
            return None

        raw_theta_rate = float(theta_rate)
        raw_radius_rate = float(radius_rate)
        theta_rate, radius_rate = self._apply_pixel_horizontal_rate_shaping(
            theta_rate,
            radius_rate,
            current_cyl_pose=current_cyl_pose,
            center_distance_px=center_distance_px,
            confirm_z=confirm_z,
            apply_pixel_axis_fine_scale=False,
        )
        predicted_before = math.hypot(error_x, error_y)
        predicted_after_x = error_x + j00 * theta_rate + j01 * radius_rate
        predicted_after_y = error_y + j10 * theta_rate + j11 * radius_rate
        predicted_after = math.hypot(predicted_after_x, predicted_after_y)
        return (
            theta_rate,
            radius_rate,
            {
                "horizontal_mode": "ibvs_dls",
                "ibvs_jacobian_source": str(jacobian.source),
                "ibvs_jacobian": {
                    "du_dtheta_px_per_deg": float(j00),
                    "du_dradius_px_per_mm": float(j01),
                    "dv_dtheta_px_per_deg": float(j10),
                    "dv_dradius_px_per_mm": float(j11),
                },
                "ibvs_gain": float(gain),
                "ibvs_damping_px_per_unit": float(damping),
                "ibvs_condition_number": jacobian.condition_number,
                "ibvs_dls_det": float(jacobian.det),
                "ibvs_raw_theta_rate_deg_s": float(raw_theta_rate),
                "ibvs_raw_radius_rate_mm_s": float(raw_radius_rate),
                "ibvs_predicted_error_before_px": float(predicted_before),
                "ibvs_predicted_error_after_px": float(predicted_after),
                "ibvs_predicted_error_delta_px": float(predicted_after - predicted_before),
            },
        )

    def _ibvs_jacobian_available(
        self,
        *,
        current_cyl_pose: tuple[float, float, float] | None = None,
    ) -> bool:
        return self._ibvs_jacobian(current_cyl_pose=current_cyl_pose) is not None

    def _ibvs_jacobian(
        self,
        *,
        current_cyl_pose: tuple[float, float, float] | None = None,
    ) -> _IbvsJacobian | None:
        source = str(getattr(self.config, "vision_continuous_servo_ibvs_jacobian_source", "config") or "config")
        fitted = _safe_quad(getattr(self.config, "vision_continuous_servo_ibvs_fitted_jacobian", None))
        profile = _safe_quad(getattr(self.config, "vision_continuous_servo_ibvs_profile_jacobian", None))
        configured = (
            float(getattr(self.config, "vision_continuous_servo_ibvs_du_dtheta_px_per_deg", -14.0)),
            float(getattr(self.config, "vision_continuous_servo_ibvs_du_dradius_px_per_mm", 0.0)),
            float(getattr(self.config, "vision_continuous_servo_ibvs_dv_dtheta_px_per_deg", 0.0)),
            float(getattr(self.config, "vision_continuous_servo_ibvs_dv_dradius_px_per_mm", 3.5)),
        )
        if fitted is not None:
            matrix = fitted
            source = "fitted_low_height"
        elif profile is not None:
            matrix = profile
            source = "profile_confirm"
        else:
            matrix = configured
            if not source:
                source = "config"
        j00, j01, j10, j11 = matrix
        if not all(math.isfinite(value) for value in (j00, j01, j10, j11)):
            return None
        gain = max(0.01, min(1.0, float(getattr(self.config, "vision_continuous_servo_ibvs_gain", 0.45))))
        damping = max(0.0, float(getattr(self.config, "vision_continuous_servo_ibvs_damping_px_per_unit", 2.0)))
        a00 = j00 * j00 + j10 * j10 + damping * damping
        a01 = j00 * j01 + j10 * j11
        a11 = j01 * j01 + j11 * j11 + damping * damping
        det = a00 * a11 - a01 * a01
        if not math.isfinite(det) or abs(det) < 1e-9:
            return None
        condition_number = _matrix_condition_number_2x2((j00, j01, j10, j11))
        return _IbvsJacobian(
            matrix=(j00, j01, j10, j11),
            source=source,
            gain=gain,
            damping=damping,
            det=float(det),
            condition_number=condition_number,
        )

    def _pixel_jacobian_horizontal_rates(
        self,
        slot_payload: Mapping[str, object],
        *,
        current_cyl_pose: tuple[float, float, float],
        center_distance_px: float | None = None,
        confirm_z: float | None = None,
    ) -> tuple[float, float] | None:
        tracking_point = self._tracking_point(slot_payload, current_cyl_pose=current_cyl_pose, confirm_z=confirm_z)
        target = _safe_pair(slot_payload.get("alignment_target_pixel"))
        if tracking_point is None or target is None:
            return None
        error_x = float(tracking_point[0]) - float(target[0])
        error_y = float(tracking_point[1]) - float(target[1])
        if not (math.isfinite(error_x) and math.isfinite(error_y)):
            return None
        gain = max(0.01, min(1.0, float(getattr(self.config, "vision_continuous_servo_pixel_jacobian_gain", 0.35))))
        dtheta = (
            float(getattr(self.config, "vision_continuous_servo_pixel_jacobian_dtheta_dx", 0.0)) * error_x
            + float(getattr(self.config, "vision_continuous_servo_pixel_jacobian_dtheta_dy", 0.0)) * error_y
        )
        dradius = (
            float(getattr(self.config, "vision_continuous_servo_pixel_jacobian_dr_dx", 0.0)) * error_x
            + float(getattr(self.config, "vision_continuous_servo_pixel_jacobian_dr_dy", 0.0)) * error_y
        )
        if not (math.isfinite(dtheta) and math.isfinite(dradius)):
            return None
        theta_gain = max(0.0, float(getattr(self.config, "vision_continuous_servo_theta_gain_deg_s_per_deg", 2.0)))
        radius_gain = max(0.0, float(getattr(self.config, "vision_continuous_servo_radius_gain_mm_s_per_mm", 1.2)))
        theta_rate = dtheta * gain * theta_gain
        radius_rate = dradius * gain * radius_gain
        theta_rate, radius_rate = self._apply_pixel_horizontal_rate_shaping(
            theta_rate,
            radius_rate,
            current_cyl_pose=current_cyl_pose,
            center_distance_px=center_distance_px,
            confirm_z=confirm_z,
        )
        return (theta_rate, radius_rate)

    def _pixel_axis_horizontal_rates(
        self,
        slot_payload: Mapping[str, object],
        *,
        current_cyl_pose: tuple[float, float, float],
        center_distance_px: float | None = None,
        confirm_z: float | None = None,
    ) -> tuple[float, float] | None:
        tracking_point = self._tracking_point(slot_payload, current_cyl_pose=current_cyl_pose, confirm_z=confirm_z)
        target = _safe_pair(slot_payload.get("alignment_target_pixel"))
        if tracking_point is None or target is None:
            return None
        error_x = float(tracking_point[0]) - float(target[0])
        error_y = float(tracking_point[1]) - float(target[1])
        if not (math.isfinite(error_x) and math.isfinite(error_y)):
            return None
        theta_limit = max(0.1, float(getattr(self.config, "vision_continuous_servo_theta_rate_limit_deg_s", 18.0)))
        radius_limit = max(0.1, float(getattr(self.config, "vision_continuous_servo_radius_rate_limit_mm_s", 35.0)))
        theta_gain = float(getattr(self.config, "vision_continuous_servo_pixel_axis_theta_deg_s_per_px", 0.08))
        radius_gain = float(getattr(self.config, "vision_continuous_servo_pixel_axis_radius_mm_s_per_px", -0.06))
        theta_rate = error_x * theta_gain
        radius_rate = error_y * radius_gain
        theta_rate, radius_rate = self._apply_pixel_horizontal_rate_shaping(
            theta_rate,
            radius_rate,
            current_cyl_pose=current_cyl_pose,
            center_distance_px=center_distance_px,
            confirm_z=confirm_z,
        )
        return (theta_rate, radius_rate)

    def _apply_pixel_horizontal_rate_shaping(
        self,
        theta_rate: float,
        radius_rate: float,
        *,
        current_cyl_pose: tuple[float, float, float],
        center_distance_px: float | None,
        confirm_z: float | None,
        apply_pixel_axis_fine_scale: bool = True,
    ) -> tuple[float, float]:
        theta_limit = max(0.1, float(getattr(self.config, "vision_continuous_servo_theta_rate_limit_deg_s", 18.0)))
        radius_limit = max(0.1, float(getattr(self.config, "vision_continuous_servo_radius_rate_limit_mm_s", 35.0)))
        theta_rate = _clamp(float(theta_rate), -theta_limit, theta_limit)
        radius_rate = _clamp(float(radius_rate), -radius_limit, radius_limit)
        if confirm_z is not None:
            height_scale = self._height_horizontal_rate_scale(
                current_z=float(current_cyl_pose[2]),
                confirm_z=float(confirm_z),
                center_distance_px=center_distance_px,
            )
            theta_rate *= height_scale
            radius_rate *= height_scale
        if apply_pixel_axis_fine_scale and center_distance_px is not None:
            try:
                center_distance = float(center_distance_px)
            except (TypeError, ValueError):
                center_distance = float("inf")
            fine_band = max(
                0.1,
                float(getattr(self.config, "vision_continuous_servo_pixel_axis_fine_band_px", 24.0)),
            )
            if math.isfinite(center_distance) and center_distance <= fine_band:
                fine_scale = max(
                    0.05,
                    min(1.0, float(getattr(self.config, "vision_continuous_servo_pixel_axis_fine_rate_scale", 0.35))),
                )
                theta_rate *= fine_scale
                radius_rate *= fine_scale
        low_height_scale_reason = self._low_height_fine_scale_reason(
            current_z=float(current_cyl_pose[2]),
            confirm_z=confirm_z,
            center_distance_px=center_distance_px,
        )
        if low_height_scale_reason:
            apply_low_height_fine_scale = True
            if low_height_scale_reason == "low_height_pause":
                try:
                    center_distance = float(center_distance_px)
                except (TypeError, ValueError):
                    center_distance = float("inf")
                fine_band = max(
                    0.1,
                    float(getattr(self.config, "vision_continuous_servo_low_height_fine_band_px", 3.0)),
                )
                apply_low_height_fine_scale = math.isfinite(center_distance) and center_distance <= fine_band
            if apply_low_height_fine_scale:
                scale = self._low_height_horizontal_scale(center_distance_px=center_distance_px)
                theta_rate *= scale
                radius_rate *= scale
                theta_rate, radius_rate = self._apply_low_height_min_horizontal_rate(
                    theta_rate,
                    radius_rate,
                    center_distance_px=center_distance_px,
                )
        return (theta_rate, radius_rate)

    def _height_horizontal_rate_scale(
        self,
        *,
        current_z: float,
        confirm_z: float,
        center_distance_px: float | None = None,
    ) -> float:
        high_z = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_descent_high_error_z_above_confirm_mm", 70.0)),
        )
        low_z = max(
            0.0,
            min(high_z, float(getattr(self.config, "vision_continuous_servo_descent_low_error_z_above_confirm_mm", 12.0))),
        )
        low_scale = self._low_height_horizontal_scale(center_distance_px=center_distance_px)
        above_confirm = max(0.0, float(current_z) - float(confirm_z))
        if above_confirm >= high_z:
            return 1.0
        if above_confirm <= low_z:
            return float(low_scale)
        ratio = (above_confirm - low_z) / max(1e-6, high_z - low_z)
        return float(low_scale + (1.0 - low_scale) * ratio)

    def _low_height_horizontal_scale(self, *, center_distance_px: float | None) -> float:
        fine_scale = max(
            0.05,
            min(1.0, float(getattr(self.config, "vision_continuous_servo_low_height_fine_rate_scale", 0.35))),
        )
        coarse_scale = max(
            fine_scale,
            min(1.0, float(getattr(self.config, "vision_continuous_servo_low_height_coarse_rate_scale", 0.70))),
        )
        if center_distance_px is None:
            return fine_scale
        try:
            center_distance = float(center_distance_px)
        except (TypeError, ValueError):
            return fine_scale
        if not math.isfinite(center_distance):
            return fine_scale
        fine_band = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_low_height_fine_band_px", 3.0)),
        )
        return fine_scale if center_distance <= fine_band else coarse_scale

    def _apply_low_height_min_horizontal_rate(
        self,
        theta_rate: float,
        radius_rate: float,
        *,
        center_distance_px: float | None,
    ) -> tuple[float, float]:
        try:
            center_distance = float(center_distance_px)
        except (TypeError, ValueError):
            return (theta_rate, radius_rate)
        if not math.isfinite(center_distance):
            return (theta_rate, radius_rate)
        pick_ready_center = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_pick_ready_center_px", 2.0)),
        )
        fine_band = max(
            pick_ready_center,
            float(getattr(self.config, "vision_continuous_servo_low_height_fine_band_px", 3.0)),
        )
        if center_distance <= pick_ready_center or center_distance > fine_band:
            return (theta_rate, radius_rate)
        theta_min = max(
            0.0,
            float(getattr(self.config, "vision_continuous_servo_low_height_min_theta_rate_deg_s", 0.0)),
        )
        radius_min = max(
            0.0,
            float(getattr(self.config, "vision_continuous_servo_low_height_min_radius_rate_mm_s", 0.0)),
        )
        if theta_min > 0.0 and 1e-6 < abs(theta_rate) < theta_min:
            theta_rate = math.copysign(theta_min, theta_rate)
        if radius_min > 0.0 and 1e-6 < abs(radius_rate) < radius_min:
            radius_rate = math.copysign(radius_min, radius_rate)
        return (theta_rate, radius_rate)

    def _low_height_fine_scale_reason(
        self,
        *,
        current_z: float,
        confirm_z: float | None,
        center_distance_px: float | None,
    ) -> str:
        if confirm_z is None or center_distance_px is None:
            return ""
        try:
            center_distance = float(center_distance_px)
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(center_distance):
            return ""
        guard_band = max(
            0.0,
            float(getattr(self.config, "vision_continuous_servo_low_height_guard_band_mm", 30.0)),
        )
        if float(current_z) <= float(confirm_z) + guard_band:
            if self._is_low_height_pause_descent_zone(current_z=float(current_z), confirm_z=float(confirm_z)):
                return "low_height_pause"
            fine_band = max(
                0.1,
                float(getattr(self.config, "vision_continuous_servo_low_height_fine_band_px", 20.0)),
            )
            return "low_height_guard_fine" if center_distance <= fine_band else ""
        z_tolerance = max(0.5, float(getattr(self.config, "vision_pick_z_tolerance_mm", 4.0)))
        if float(current_z) > float(confirm_z) + z_tolerance * 2.0:
            return ""
        fine_band = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_low_height_fine_band_px", 20.0)),
        )
        return "near_confirm_fine" if center_distance <= fine_band else ""

    def _z_rate_scale_reason(
        self,
        *,
        current_z: float,
        confirm_z: float,
        center_distance_px: float,
    ) -> str:
        if self._is_low_height_guard_zone(current_z=float(current_z), confirm_z=float(confirm_z)):
            return "low_height_guard"
        return self._low_height_fine_scale_reason(
            current_z=float(current_z),
            confirm_z=float(confirm_z),
            center_distance_px=float(center_distance_px),
        )

    def _low_height_z_rate_scale(self) -> float:
        return max(
            0.05,
            min(1.0, float(getattr(self.config, "vision_continuous_servo_low_height_z_rate_scale", 0.35))),
        )

    def _use_low_height_fine_scale(
        self,
        *,
        current_z: float,
        confirm_z: float | None,
        center_distance_px: float | None,
    ) -> bool:
        return bool(
            self._low_height_fine_scale_reason(
                current_z=current_z,
                confirm_z=confirm_z,
                center_distance_px=center_distance_px,
            )
        )

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

    def _low_height_descent_rebound(
        self,
        *,
        current_z: float,
        confirm_z: float,
        center_distance_px: float,
        previous_center_distance_px: float | None,
    ) -> bool:
        if not self._is_low_height_guard_zone(current_z=float(current_z), confirm_z=float(confirm_z)):
            return False
        if previous_center_distance_px is None:
            return False
        previous = _safe_float(previous_center_distance_px, float("nan"))
        current = _safe_float(center_distance_px, float("nan"))
        if not (math.isfinite(previous) and math.isfinite(current)):
            return False
        fine_band = max(0.1, float(getattr(self.config, "vision_continuous_servo_low_height_fine_band_px", 10.0)))
        if previous > fine_band:
            return False
        growth = current - previous
        rebound_pause = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_low_height_descent_rebound_pause_px", 10.0)),
        )
        return growth >= rebound_pause

    def _low_height_best_descent_pause(
        self,
        *,
        current_z: float,
        confirm_z: float,
        center_distance_px: float,
        best_center_distance_px: float | None,
    ) -> bool:
        if not self._is_low_height_guard_zone(current_z=float(current_z), confirm_z=float(confirm_z)):
            return False
        best = _safe_float(best_center_distance_px, float("nan")) if best_center_distance_px is not None else float("nan")
        current = _safe_float(center_distance_px, float("nan"))
        if not (math.isfinite(best) and math.isfinite(current)):
            return False
        pick_ready_center = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_pick_ready_center_px", 2.0)),
        )
        fine_band = max(
            pick_ready_center,
            float(getattr(self.config, "vision_continuous_servo_low_height_fine_band_px", 3.0)),
        )
        if best > fine_band:
            return False
        rebound_limit = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_low_height_best_error_descent_pause_px", 4.0)),
        )
        return current > best + rebound_limit

    def _low_height_best_confirm_descent_allowed(
        self,
        *,
        current_z: float,
        confirm_z: float,
        center_distance_px: float,
        best_center_distance_px: float | None,
        descent_allow_px: float,
        in_low_height_pause_zone: bool,
    ) -> bool:
        if not bool(in_low_height_pause_zone):
            return False
        z_tolerance = max(0.0, float(getattr(self.config, "vision_pick_z_tolerance_mm", 4.0)))
        if float(current_z) <= float(confirm_z) + z_tolerance:
            return False
        best = _safe_float(best_center_distance_px, float("nan")) if best_center_distance_px is not None else float("nan")
        current = _safe_float(center_distance_px, float("nan"))
        allow = _safe_float(descent_allow_px, float("nan"))
        if not (math.isfinite(best) and math.isfinite(current) and math.isfinite(allow)):
            return False
        pick_ready_center = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_pick_ready_center_px", 2.0)),
        )
        best_confirm_allow = max(
            pick_ready_center,
            allow,
            float(getattr(self.config, "vision_continuous_servo_low_height_best_confirm_descent_allow_px", allow)),
        )
        return best <= pick_ready_center and current <= best_confirm_allow

    def _is_low_height_guard_zone(self, *, current_z: float, confirm_z: float) -> bool:
        guard_band = max(
            0.0,
            float(getattr(self.config, "vision_continuous_servo_low_height_guard_band_mm", 30.0)),
        )
        return float(current_z) <= float(confirm_z) + guard_band

    def _is_low_height_pause_descent_zone(self, *, current_z: float, confirm_z: float) -> bool:
        pause_band = max(
            0.0,
            float(getattr(self.config, "vision_continuous_servo_low_height_pause_descent_band_mm", 12.0)),
        )
        return float(current_z) <= float(confirm_z) + pause_band

    def _update_low_height_guard_pending(
        self,
        pending: ContinuousServoPending,
        *,
        current_cyl_pose: tuple[float, float, float],
        confirm_z: float,
        center_distance_px: float,
    ) -> ContinuousServoPending:
        if not self._is_low_height_guard_zone(current_z=float(current_cyl_pose[2]), confirm_z=confirm_z):
            return ContinuousServoPending(
                slot_id=int(pending.slot_id),
                stable_frames=int(pending.stable_frames),
                pick_ready_frames=int(pending.pick_ready_frames),
                lost_frames=int(pending.lost_frames),
                stale_frames=int(pending.stale_frames),
                source=str(pending.source),
                last_center_px=pending.last_center_px,
                last_center_distance_px=pending.last_center_distance_px,
                descent_anchor_z_mm=pending.descent_anchor_z_mm,
                descent_cooldown_frames=int(pending.descent_cooldown_frames),
                low_height_anchor_pose=None,
                best_center_distance_px=None,
                low_height_static_frames=0,
                low_height_static_reference_px=None,
                motion_guard_anchor_pose=pending.motion_guard_anchor_pose,
                motion_guard_anchor_px=pending.motion_guard_anchor_px,
                motion_guard_static_frames=pending.motion_guard_static_frames,
            )
        static_band = max(
            0.0,
            float(getattr(self.config, "vision_continuous_servo_low_height_static_band_mm", 6.0)),
        )
        static_pose_band = max(
            0.0,
            float(getattr(self.config, "vision_continuous_servo_low_height_static_pose_band_mm", 1.5)),
        )
        static_measurement_zone = float(current_cyl_pose[2]) <= float(confirm_z) + min(static_band, static_pose_band)
        anchor = pending.low_height_anchor_pose
        if anchor is None:
            anchor = (float(current_cyl_pose[0]), float(current_cyl_pose[1]), float(current_cyl_pose[2]))
        best = pending.best_center_distance_px
        if best is None or not math.isfinite(float(best)):
            best = float(center_distance_px)
        else:
            best = min(float(best), float(center_distance_px))
        static_reference = pending.low_height_static_reference_px
        current_distance = float(center_distance_px)
        if not static_measurement_zone:
            static_reference = None
            static_frames = 0
        elif static_reference is None or not math.isfinite(float(static_reference)):
            static_reference = current_distance
            static_frames = 1
        else:
            min_improvement = max(
                0.05,
                float(getattr(self.config, "vision_continuous_servo_low_height_static_improvement_px", 1.0)),
            )
            if current_distance < float(static_reference) - min_improvement:
                static_reference = current_distance
                static_frames = 1
            else:
                static_frames = int(pending.low_height_static_frames) + 1
        return ContinuousServoPending(
            slot_id=int(pending.slot_id),
            stable_frames=int(pending.stable_frames),
            pick_ready_frames=int(pending.pick_ready_frames),
            lost_frames=int(pending.lost_frames),
            stale_frames=int(pending.stale_frames),
            source=str(pending.source),
            last_center_px=pending.last_center_px,
            last_center_distance_px=pending.last_center_distance_px,
            descent_anchor_z_mm=pending.descent_anchor_z_mm,
            descent_cooldown_frames=int(pending.descent_cooldown_frames),
            low_height_anchor_pose=anchor,
            best_center_distance_px=float(best),
            low_height_static_frames=int(static_frames),
            low_height_static_reference_px=None if static_reference is None else float(static_reference),
            motion_guard_anchor_pose=pending.motion_guard_anchor_pose,
            motion_guard_anchor_px=pending.motion_guard_anchor_px,
            motion_guard_static_frames=pending.motion_guard_static_frames,
        )

    def _low_height_guard_stop_reason(
        self,
        *,
        pending: ContinuousServoPending,
        current_cyl_pose: tuple[float, float, float],
        confirm_z: float,
        center_distance_px: float,
        allow_confirm_descent: bool = False,
    ) -> str:
        if not self._is_low_height_guard_zone(current_z=float(current_cyl_pose[2]), confirm_z=confirm_z):
            return ""
        anchor = pending.low_height_anchor_pose
        if anchor is None:
            return ""
        theta_drift = abs(float(current_cyl_pose[0]) - float(anchor[0]))
        radius_drift = abs(float(current_cyl_pose[1]) - float(anchor[1]))
        max_theta_drift = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_low_height_max_theta_drift_deg", 8.0)),
        )
        max_radius_drift = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_low_height_max_radius_drift_mm", 8.0)),
        )
        best = pending.best_center_distance_px
        pick_ready_center = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_pick_ready_center_px", 2.0)),
        )
        center_stop_px = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_center_stop_descent_px", 36.0)),
        )
        low_height_servo_limit_px = max(
            center_stop_px,
            float(getattr(self.config, "vision_continuous_servo_low_height_unstable_servo_px", 60.0)),
        )
        best_distance = _safe_float(best, float("nan")) if best is not None else float("nan")
        if theta_drift > max_theta_drift or radius_drift > max_radius_drift:
            if not math.isfinite(best_distance):
                return "low_height_mapping_drift"
            rebound_limit = max(
                0.1,
                float(getattr(self.config, "vision_continuous_servo_low_height_best_error_rebound_px", 8.0)),
            )
            corrective_drift = (
                float(center_distance_px) <= low_height_servo_limit_px
                and float(center_distance_px) <= best_distance + rebound_limit
            )
            if not corrective_drift:
                return "low_height_mapping_drift"
        if not math.isfinite(best_distance):
            return ""
        static_reason = self._low_height_static_residual_stop_reason(
            pending=pending,
            current_z=float(current_cyl_pose[2]),
            confirm_z=float(confirm_z),
            center_distance_px=float(center_distance_px),
        )
        if static_reason and not bool(allow_confirm_descent):
            return static_reason
        fine_band = max(
            pick_ready_center,
            float(getattr(self.config, "vision_continuous_servo_low_height_fine_band_px", 10.0)),
        )
        if best_distance > fine_band:
            return ""
        rebound = float(center_distance_px) - best_distance
        rebound_limit = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_low_height_best_error_rebound_px", 8.0)),
        )
        if rebound > rebound_limit:
            if float(center_distance_px) > low_height_servo_limit_px:
                return "low_height_best_error_rebounded"
        return ""

    def _low_height_guard_trace(
        self,
        *,
        pending: ContinuousServoPending,
        current_cyl_pose: tuple[float, float, float],
        confirm_z: float,
        center_distance_px: float,
    ) -> dict[str, object]:
        anchor = pending.low_height_anchor_pose
        theta_drift = None if anchor is None else abs(float(current_cyl_pose[0]) - float(anchor[0]))
        radius_drift = None if anchor is None else abs(float(current_cyl_pose[1]) - float(anchor[1]))
        best = pending.best_center_distance_px
        rebound = None if best is None else float(center_distance_px) - float(best)
        return {
            "center_distance_px": float(center_distance_px),
            "best_center_distance_px": None if best is None else float(best),
            "best_error_rebound_px": rebound,
            "low_height_anchor_pose": None if anchor is None else list(anchor),
            "theta_drift_deg": theta_drift,
            "radius_drift_mm": radius_drift,
            "current_z_mm": float(current_cyl_pose[2]),
            "confirm_z_mm": float(confirm_z),
            "low_height_guard_band_mm": float(
                getattr(self.config, "vision_continuous_servo_low_height_guard_band_mm", 30.0)
            ),
            "max_theta_drift_deg": float(
                getattr(self.config, "vision_continuous_servo_low_height_max_theta_drift_deg", 8.0)
            ),
            "max_radius_drift_mm": float(
                getattr(self.config, "vision_continuous_servo_low_height_max_radius_drift_mm", 8.0)
            ),
            "best_error_rebound_limit_px": float(
                getattr(self.config, "vision_continuous_servo_low_height_best_error_rebound_px", 8.0)
            ),
            "low_height_static_frames": int(pending.low_height_static_frames),
            "low_height_static_reference_px": (
                None
                if pending.low_height_static_reference_px is None
                else float(pending.low_height_static_reference_px)
            ),
            "low_height_static_error_min_px": float(
                getattr(self.config, "vision_continuous_servo_low_height_static_error_min_px", 8.0)
            ),
            "low_height_static_error_max_px": float(
                getattr(self.config, "vision_continuous_servo_low_height_static_error_max_px", 30.0)
            ),
            "low_height_static_required_frames": int(
                getattr(self.config, "vision_continuous_servo_low_height_static_frames", 12)
            ),
            "low_height_static_band_mm": float(
                getattr(self.config, "vision_continuous_servo_low_height_static_band_mm", 6.0)
            ),
            "low_height_static_pose_band_mm": float(
                getattr(self.config, "vision_continuous_servo_low_height_static_pose_band_mm", 1.5)
            ),
            "recommendation": (
                "Run search_low_height_center.py or calibrate_low_height_alignment.py at the stopped low height."
            ),
        }

    def _low_height_static_residual_stop_reason(
        self,
        *,
        pending: ContinuousServoPending,
        current_z: float,
        confirm_z: float,
        center_distance_px: float,
    ) -> str:
        if bool(getattr(self.config, "vision_continuous_servo_low_height_static_stop_enabled", True)) is False:
            return ""
        static_band = max(
            0.0,
            float(getattr(self.config, "vision_continuous_servo_low_height_static_band_mm", 6.0)),
        )
        if float(current_z) > float(confirm_z) + static_band:
            return ""
        current = float(center_distance_px)
        if not math.isfinite(current):
            return ""
        min_px = max(
            0.0,
            float(getattr(self.config, "vision_continuous_servo_low_height_static_error_min_px", 8.0)),
        )
        max_px = max(
            min_px,
            float(getattr(self.config, "vision_continuous_servo_low_height_static_error_max_px", 30.0)),
        )
        if current < min_px or current > max_px:
            return ""
        required_frames = max(
            1,
            int(getattr(self.config, "vision_continuous_servo_low_height_static_frames", 12)),
        )
        if int(pending.low_height_static_frames) < required_frames:
            return ""
        if pending.low_height_anchor_pose is None:
            return ""
        reference = pending.low_height_static_reference_px
        if reference is None or not math.isfinite(float(reference)):
            return ""
        min_improvement = max(
            0.05,
            float(getattr(self.config, "vision_continuous_servo_low_height_static_improvement_px", 1.0)),
        )
        if float(reference) - current >= min_improvement:
            return ""
        return "low_height_local_model_required"

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
        if (
            bool(slot_payload.get("low_height_local_center_override", False))
            and str(slot_payload.get("invalid_reason", "") or "").strip() == "vision_servo_required"
            and not bool(slot_payload.get("actionable", False))
        ):
            return ""
        min_confidence = max(
            0.0,
            min(1.0, float(getattr(self.config, "vision_continuous_servo_min_confidence", 0.55))),
        )
        hard_min_confidence = max(
            0.0,
            min(
                min_confidence,
                float(getattr(self.config, "vision_continuous_servo_hard_min_confidence", 0.20)),
            ),
        )
        min_area = max(1.0, float(getattr(self.config, "vision_continuous_servo_min_area_px", 1500)))
        large_area_ratio = max(
            1.0,
            float(getattr(self.config, "vision_continuous_servo_low_confidence_large_area_ratio", 12.0)),
        )
        area = _safe_float(slot_payload.get("area_px"), min_area)
        confidence = _safe_float(slot_payload.get("confidence"), 1.0)
        if (
            math.isfinite(confidence)
            and confidence < min_confidence
            and (
                confidence < hard_min_confidence
                or not math.isfinite(area)
                or area < min_area * large_area_ratio
            )
        ):
            return "target_confidence_low"
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

    def _measurement_point_for_pose(
        self,
        *,
        current_z: float | None = None,
        confirm_z: float | None = None,
    ) -> str:
        base = normalize_servo_measurement_point(
            getattr(self.config, "vision_servo_measurement_point", "geometry_subpixel")
        )
        low_height = str(getattr(self.config, "vision_servo_low_height_measurement_point", "") or "").strip().lower()
        if low_height:
            low_height = normalize_servo_measurement_point(low_height, default=base)
        if low_height and current_z is not None and confirm_z is not None:
            try:
                z_value = float(current_z)
                confirm_value = float(confirm_z)
            except (TypeError, ValueError):
                return base
            if math.isfinite(z_value) and math.isfinite(confirm_value) and self._is_low_height_guard_zone(
                current_z=z_value,
                confirm_z=confirm_value,
            ):
                return low_height
        return base

    def _tracking_point(
        self,
        slot_payload: Mapping[str, object],
        *,
        measurement_point: str | None = None,
        current_cyl_pose: tuple[float, float, float] | None = None,
        confirm_z: float | None = None,
    ) -> tuple[float, float] | None:
        if measurement_point is None:
            current_z = None if current_cyl_pose is None else float(current_cyl_pose[2])
            measurement_point = self._measurement_point_for_pose(current_z=current_z, confirm_z=confirm_z)
        else:
            measurement_point = normalize_servo_measurement_point(measurement_point)
        if measurement_point == "color_block_subpixel":
            color_px = _safe_pair(slot_payload.get("color_block_center_f"))
            if color_px is not None:
                return color_px
            return None
        if measurement_point == "color_block":
            color_px = _safe_pair(slot_payload.get("color_block_center"))
            if color_px is not None:
                return color_px
            return None
        if measurement_point == "top_face_subpixel":
            top_face_px = _safe_pair(slot_payload.get("top_face_center_f"))
            if top_face_px is not None:
                return top_face_px
        if measurement_point == "top_face":
            top_face_px = _safe_pair(slot_payload.get("top_face_center"))
            if top_face_px is not None:
                return top_face_px
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

    def _ibvs_tracking_point(
        self,
        slot_payload: Mapping[str, object],
        *,
        measurement_point: str | None = None,
        current_cyl_pose: tuple[float, float, float] | None = None,
        confirm_z: float | None = None,
    ) -> tuple[float, float] | None:
        point = self._tracking_point(
            slot_payload,
            measurement_point=measurement_point,
            current_cyl_pose=current_cyl_pose,
            confirm_z=confirm_z,
        )
        if point is not None:
            return point
        center_distance = _safe_float(slot_payload.get("center_distance_px"), float("nan"))
        if not math.isfinite(center_distance):
            return None
        target = _safe_pair(slot_payload.get("alignment_target_pixel")) or (
            float(HIWONDER_CAMERA_WIDTH) * 0.5,
            float(HIWONDER_CAMERA_HEIGHT) * 0.5,
        )
        # Some older tests and replay bundles only contain scalar center error.
        # Use it as a synthetic pixel offset from the camera center; do not use
        # legacy servo_command_point as a fallback motion target in IBVS mode.
        legacy_point = _safe_pair(slot_payload.get("servo_command_point"))
        if legacy_point is not None:
            theta_error = float(legacy_point[0]) - (
                float(current_cyl_pose[0]) if current_cyl_pose is not None else 0.0
            )
            radius_error = float(legacy_point[1]) - (
                float(current_cyl_pose[1]) if current_cyl_pose is not None else 0.0
            )
            if abs(theta_error) > 1e-6 or abs(radius_error) > 1e-6:
                return (
                    float(target[0]) - math.copysign(abs(float(center_distance)), theta_error or 1.0),
                    float(target[1]) + math.copysign(abs(float(center_distance)), radius_error or 1.0),
                )
        return (float(target[0]) + float(center_distance), float(target[1]))

    def _center_distance_px(
        self,
        slot_payload: Mapping[str, object],
        packet: Mapping[str, object] | None,
        *,
        measurement_point: str | None = None,
    ) -> float:
        configured = _safe_float(slot_payload.get("center_distance_px"), float("nan"))
        slot_measurement = str(slot_payload.get("measurement_point", "") or "").strip().lower()
        normalized_measurement = (
            normalize_servo_measurement_point(measurement_point)
            if measurement_point is not None
            else normalize_servo_measurement_point(slot_measurement)
        )
        if math.isfinite(configured) and (
            not slot_measurement
            or normalize_servo_measurement_point(slot_measurement, default=normalized_measurement)
            == normalized_measurement
        ):
            return float(configured)
        point = self._tracking_point(slot_payload, measurement_point=normalized_measurement)
        target = _safe_pair(slot_payload.get("alignment_target_pixel"))
        if target is None and isinstance(packet, Mapping):
            target = _safe_pair(packet.get("alignment_target_pixel"))
        if point is None or target is None:
            return float("inf")
        return float(math.hypot(float(point[0]) - float(target[0]), float(point[1]) - float(target[1])))

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
                previous_distance = (
                    None
                    if current.last_center_distance_px is None
                    else _safe_float(current.last_center_distance_px, float("nan"))
                )
                if previous_distance is None or not math.isfinite(previous_distance):
                    return "target_center_jump"
                improvement = previous_distance - float(center_distance_px)
                relock_margin = max(3.0, max_jump * 0.25)
                if improvement < relock_margin:
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

    def _descent_error_allow_px(self, *, current_z: float, confirm_z: float) -> float:
        center_allow = max(0.1, float(getattr(self.config, "vision_continuous_servo_center_allow_descent_px", 8.0)))
        low_allow = max(
            center_allow,
            float(getattr(self.config, "vision_continuous_servo_low_height_descent_allow_px", center_allow)),
        )
        high_allow = max(
            low_allow,
            float(getattr(self.config, "vision_continuous_servo_descent_high_error_px", 80.0)),
        )
        high_z = max(
            0.1,
            float(getattr(self.config, "vision_continuous_servo_descent_high_error_z_above_confirm_mm", 70.0)),
        )
        low_z = max(
            0.0,
            min(high_z, float(getattr(self.config, "vision_continuous_servo_descent_low_error_z_above_confirm_mm", 12.0))),
        )
        above_confirm = max(0.0, float(current_z) - float(confirm_z))
        if above_confirm >= high_z:
            return float(high_allow)
        if above_confirm <= low_z:
            return float(low_allow)
        ratio = (above_confirm - low_z) / max(1e-6, high_z - low_z)
        return float(low_allow + (high_allow - low_allow) * ratio)

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
        stable_frames: int | None = None,
        pick_ready_frames: int | None = None,
    ) -> ContinuousServoPending:
        return ContinuousServoPending(
            slot_id=int(pending.slot_id),
            stable_frames=(int(pending.stable_frames) if stable_frames is None else max(0, int(stable_frames))),
            pick_ready_frames=(
                int(pending.pick_ready_frames) if pick_ready_frames is None else max(0, int(pick_ready_frames))
            ),
            lost_frames=int(pending.lost_frames),
            stale_frames=int(pending.stale_frames),
            source=str(pending.source),
            last_center_px=pending.last_center_px,
            last_center_distance_px=pending.last_center_distance_px,
            descent_anchor_z_mm=pending.descent_anchor_z_mm,
            descent_cooldown_frames=int(pending.descent_cooldown_frames),
            low_height_anchor_pose=pending.low_height_anchor_pose,
            best_center_distance_px=pending.best_center_distance_px,
            low_height_static_frames=int(pending.low_height_static_frames),
            low_height_static_reference_px=pending.low_height_static_reference_px,
            motion_guard_anchor_pose=pending.motion_guard_anchor_pose,
            motion_guard_anchor_px=pending.motion_guard_anchor_px,
            motion_guard_static_frames=pending.motion_guard_static_frames,
        )

    @staticmethod
    def _with_motion_guard(
        pending: ContinuousServoPending,
        *,
        anchor_pose: tuple[float, float, float] | None,
        anchor_px: tuple[float, float] | None,
        static_frames: int,
    ) -> ContinuousServoPending:
        return ContinuousServoPending(
            slot_id=int(pending.slot_id),
            stable_frames=int(pending.stable_frames),
            pick_ready_frames=int(pending.pick_ready_frames),
            lost_frames=int(pending.lost_frames),
            stale_frames=int(pending.stale_frames),
            source=str(pending.source),
            last_center_px=pending.last_center_px,
            last_center_distance_px=pending.last_center_distance_px,
            descent_anchor_z_mm=pending.descent_anchor_z_mm,
            descent_cooldown_frames=int(pending.descent_cooldown_frames),
            low_height_anchor_pose=pending.low_height_anchor_pose,
            best_center_distance_px=pending.best_center_distance_px,
            low_height_static_frames=int(pending.low_height_static_frames),
            low_height_static_reference_px=pending.low_height_static_reference_px,
            motion_guard_anchor_pose=anchor_pose,
            motion_guard_anchor_px=anchor_px,
            motion_guard_static_frames=max(0, int(static_frames)),
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


def _safe_quad(value: object) -> tuple[float, float, float, float] | None:
    if isinstance(value, Mapping):
        raw = (
            value.get("du_dtheta_px_per_deg"),
            value.get("du_dradius_px_per_mm"),
            value.get("dv_dtheta_px_per_deg"),
            value.get("dv_dradius_px_per_mm"),
        )
    else:
        raw = value
    if not isinstance(raw, (tuple, list)) or len(raw) < 4:
        return None
    values = tuple(_safe_float(raw[index], float("nan")) for index in range(4))
    if not all(math.isfinite(item) for item in values):
        return None
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _safe_triplet(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        return None
    first = _safe_float(value[0], float("nan"))
    second = _safe_float(value[1], float("nan"))
    third = _safe_float(value[2], float("nan"))
    if not (math.isfinite(first) and math.isfinite(second) and math.isfinite(third)):
        return None
    return (float(first), float(second), float(third))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _matrix_condition_number_2x2(matrix: tuple[float, float, float, float]) -> float | None:
    j00, j01, j10, j11 = matrix
    ata00 = j00 * j00 + j10 * j10
    ata01 = j00 * j01 + j10 * j11
    ata11 = j01 * j01 + j11 * j11
    trace = ata00 + ata11
    det = ata00 * ata11 - ata01 * ata01
    disc = max(0.0, trace * trace - 4.0 * det)
    lambda_max = (trace + math.sqrt(disc)) * 0.5
    lambda_min = (trace - math.sqrt(disc)) * 0.5
    if lambda_min <= 1e-12 or lambda_max <= 0.0:
        return None
    return float(math.sqrt(lambda_max / lambda_min))
