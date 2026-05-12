from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

from hybrid_controller.app_robot_commands import build_pick_command_from_slot_payload


SERVO_SEARCH_HIGH = "SEARCH_HIGH"
SERVO_COARSE_CENTER = "COARSE_CENTER"
SERVO_LOW_CONFIRM = "LOW_CONFIRM"
SERVO_FINE_CENTER = "FINE_CENTER"
SERVO_WAIT_STABLE = "WAIT_STABLE"
SERVO_PICK_READY = "PICK_READY"
SERVO_PICK_SENT = "PICK_SENT"
SERVO_DONE = "DONE"
SERVO_FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class VisionServoPending:
    slot_id: int
    state: str = SERVO_SEARCH_HIGH
    attempts: int = 0
    waiting_for_ack: bool = False
    min_frame_id: int = 0
    stability_wait_frames: int = 0
    command: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None, *, slot_id: int | None = None) -> "VisionServoPending | None":
        if not isinstance(payload, Mapping):
            if slot_id is None:
                return None
            return cls(slot_id=int(slot_id))
        try:
            resolved_slot_id = int(payload.get("slot_id", slot_id))
        except (TypeError, ValueError):
            return None
        raw_state = payload.get("state", payload.get("stage", SERVO_SEARCH_HIGH))
        state = normalize_state(raw_state)
        try:
            attempts = int(payload.get("attempts", 0))
        except (TypeError, ValueError):
            attempts = 0
        try:
            min_frame_id = int(payload.get("min_frame_id", 0))
        except (TypeError, ValueError):
            min_frame_id = 0
        try:
            stability_wait_frames = int(payload.get("stability_wait_frames", 0))
        except (TypeError, ValueError):
            stability_wait_frames = 0
        return cls(
            slot_id=resolved_slot_id,
            state=state,
            attempts=max(0, attempts),
            waiting_for_ack=bool(payload.get("waiting_for_ack", False)),
            min_frame_id=max(0, min_frame_id),
            stability_wait_frames=max(0, stability_wait_frames),
            command=str(payload.get("command", "") or ""),
        )

    def to_dict(self) -> dict[str, object]:
        stage = "low_confirm" if self.state == SERVO_LOW_CONFIRM else "search"
        if self.state == SERVO_FINE_CENTER:
            stage = "fine_center"
        if self.state == SERVO_WAIT_STABLE:
            stage = "wait_stable"
        return {
            "slot_id": int(self.slot_id),
            "state": str(self.state),
            "stage": stage,
            "attempts": int(self.attempts),
            "waiting_for_ack": bool(self.waiting_for_ack),
            "min_frame_id": int(self.min_frame_id),
            "stability_wait_frames": int(self.stability_wait_frames),
            "command": str(self.command),
        }


@dataclass(frozen=True, slots=True)
class VisionServoDecision:
    action: str
    state: str
    status: str
    message: str = ""
    command: str | None = None
    pending: VisionServoPending | None = None
    reason: str = ""
    trace: dict[str, object] = field(default_factory=dict)

    @property
    def pending_dict(self) -> dict[str, object] | None:
        return None if self.pending is None else self.pending.to_dict()


def normalize_state(value: object) -> str:
    text = str(value or "").strip().upper()
    legacy = {
        "SEARCH": SERVO_SEARCH_HIGH,
        "HIGH": SERVO_SEARCH_HIGH,
        "LOW": SERVO_LOW_CONFIRM,
        "LOW_CONFIRM": SERVO_LOW_CONFIRM,
        "WAIT": SERVO_WAIT_STABLE,
        "WAIT_STABLE": SERVO_WAIT_STABLE,
        "STABLE": SERVO_WAIT_STABLE,
        "PICK": SERVO_PICK_READY,
        "PICK_READY": SERVO_PICK_READY,
        "PICK_SENT": SERVO_PICK_SENT,
        "DONE": SERVO_DONE,
        "FAILED": SERVO_FAILED,
    }
    if text in legacy:
        return legacy[text]
    if text in {
        SERVO_SEARCH_HIGH,
        SERVO_COARSE_CENTER,
        SERVO_LOW_CONFIRM,
        SERVO_FINE_CENTER,
        SERVO_WAIT_STABLE,
        SERVO_PICK_READY,
        SERVO_PICK_SENT,
        SERVO_DONE,
        SERVO_FAILED,
    }:
        return text
    return SERVO_SEARCH_HIGH


def _biased_pick_cyl_command(
    *,
    current_cyl_pose: tuple[float, float, float] | None,
    radius_bias_mm: float,
    sucker_rotation_deg: float | None = None,
) -> str | None:
    if current_cyl_pose is None or abs(float(radius_bias_mm)) < 1e-6:
        return None
    try:
        theta_deg = float(current_cyl_pose[0])
        radius_mm = float(current_cyl_pose[1]) + float(radius_bias_mm)
    except (TypeError, ValueError):
        return None
    suffix = "" if sucker_rotation_deg is None else f" {float(sucker_rotation_deg):.2f}"
    return f"PICK_CYL {theta_deg:.2f} {radius_mm:.2f}{suffix}"


def _pick_command_sucker_rotation(command: str | None) -> float | None:
    parts = str(command or "").strip().split()
    if len(parts) < 4 or parts[0].strip().upper() not in {"PICK_CYL", "PICK_WORLD"}:
        return None
    try:
        return float(parts[3])
    except (TypeError, ValueError):
        return None


class VisionServoController:
    """Pure decision layer for eye-in-hand visual pick servoing."""

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
        at_confirm_z: bool = False,
        eye_in_hand_enabled: bool = True,
    ) -> VisionServoDecision:
        current = VisionServoPending.from_mapping(pending, slot_id=int(slot_id)) or VisionServoPending(slot_id=int(slot_id))
        frame_id = self._frame_id(packet)
        if current.waiting_for_ack:
            return VisionServoDecision(
                action="WAIT",
                state=current.state,
                status=f"waiting_ack slot={int(slot_id)}",
                pending=current,
            )
        if frame_id < current.min_frame_id:
            return VisionServoDecision(
                action="WAIT",
                state=current.state,
                status=f"waiting_fresh_frame slot={int(slot_id)}",
                pending=current,
            )
        frame_block_reason = ""
        if isinstance(packet, Mapping):
            frame_block_reason = str(packet.get("frame_block_reason") or "").strip()
        if frame_block_reason:
            return VisionServoDecision(
                action="CANCEL",
                state=SERVO_FAILED,
                status=f"cancelled slot={int(slot_id)} reason={frame_block_reason}",
                message=f"Vision servo cancelled slot {int(slot_id)}: {frame_block_reason}.",
                reason=frame_block_reason,
            )
        if slot_payload is None or not bool(slot_payload.get("valid", False)):
            return VisionServoDecision(
                action="CANCEL",
                state=SERVO_FAILED,
                status=f"lost_target slot={int(slot_id)}",
                message=f"Vision servo lost slot {int(slot_id)}; pick cancelled.",
                reason="lost_target",
            )

        slot_for_command = dict(slot_payload)
        slot_for_command["grasp_angle_quality_threshold"] = float(
            getattr(self.config, "sucker_rotation_angle_quality_threshold", 0.20)
        )
        command = build_pick_command_from_slot_payload(slot_for_command)
        if command is not None:
            if bool(eye_in_hand_enabled):
                confirm_z = float(getattr(self.config, "vision_pick_confirm_z_mm", getattr(self.config, "robot_approach_z", 130.0)))
                pick_z_tolerance = max(0.0, float(getattr(self.config, "vision_pick_z_tolerance_mm", 4.0)))
                current_z = None if current_cyl_pose is None else float(current_cyl_pose[2])
                at_confirm_height = bool(at_confirm_z) or (
                    current_z is not None and abs(float(current_z) - confirm_z) <= pick_z_tolerance
                )
                if not at_confirm_height:
                    return self._descent_confirm_decision(
                        slot_id=int(slot_id),
                        frame_id=frame_id,
                        attempts=current.attempts,
                        current=current,
                        current_cyl_pose=current_cyl_pose,
                    )
            offset_source = str(getattr(self.config, "pick_tool_offset_source", "target_pixel")).strip().lower()
            pick_radius_bias_mm = 0.0
            if offset_source == "command_bias":
                pick_radius_bias_mm = float(getattr(self.config, "vision_eye_in_hand_pick_radius_bias_mm", 0.0))
            biased_command = _biased_pick_cyl_command(
                current_cyl_pose=current_cyl_pose,
                radius_bias_mm=pick_radius_bias_mm,
                sucker_rotation_deg=_pick_command_sucker_rotation(command),
            )
            if biased_command is not None:
                command = biased_command
            return VisionServoDecision(
                action="PICK",
                state=SERVO_PICK_READY,
                status=f"centered slot={int(slot_id)}; picking",
                message=f"Vision servo centered slot {int(slot_id)}; sending PICK.",
                command=command,
                reason="pick_ready",
                trace={
                    "slot_id": int(slot_id),
                    "frame_id": int(frame_id),
                    "pick_tool_offset_source": offset_source,
                    "pick_radius_bias_mm": pick_radius_bias_mm,
                },
            )

        reason = str(slot_payload.get("invalid_reason") or "not_actionable")
        if reason == "grasp_unstable":
            return self._wait_stable_decision(slot_id=int(slot_id), frame_id=frame_id, current=current)
        if reason == "vision_servo_required":
            return self._move_decision(
                slot_id=int(slot_id),
                slot_payload=slot_payload,
                frame_id=frame_id,
                current=current,
                eye_in_hand_enabled=bool(eye_in_hand_enabled),
                current_cyl_pose=current_cyl_pose,
            )
        return VisionServoDecision(
            action="CANCEL",
            state=SERVO_FAILED,
            status=f"cancelled slot={int(slot_id)} reason={reason}",
            message=f"Vision servo cancelled slot {int(slot_id)}: {reason}.",
            reason=reason,
        )

    def acknowledge_move(self, pending: Mapping[str, object], *, current_frame_id: int) -> dict[str, object] | None:
        current = VisionServoPending.from_mapping(pending)
        if current is None or not current.waiting_for_ack:
            return None
        return VisionServoPending(
            slot_id=current.slot_id,
            state=current.state,
            attempts=current.attempts,
            waiting_for_ack=False,
            min_frame_id=int(current_frame_id) + 1,
            stability_wait_frames=0,
            command=current.command,
        ).to_dict()

    def _descent_confirm_decision(
        self,
        *,
        slot_id: int,
        frame_id: int,
        attempts: int,
        current: VisionServoPending,
        current_cyl_pose: tuple[float, float, float] | None,
    ) -> VisionServoDecision:
        if current_cyl_pose is None:
            return VisionServoDecision(
                action="CANCEL",
                state=SERVO_FAILED,
                status=f"cancelled slot={int(slot_id)} reason=robot_pose_unavailable",
                message="Vision pick low confirm cancelled: robot pose unavailable.",
                reason="robot_pose_unavailable",
            )
        theta_deg, radius_mm, current_z = current_cyl_pose
        search_z = float(getattr(self.config, "vision_pick_search_z_mm", getattr(self.config, "robot_carry_z", 190.0)))
        confirm_z = float(getattr(self.config, "vision_pick_confirm_z_mm", getattr(self.config, "robot_approach_z", 130.0)))
        legacy_step_mm = max(0.1, float(getattr(self.config, "vision_pick_descent_step_mm", abs(search_z - confirm_z))))
        coarse_step_mm = max(
            0.1,
            float(getattr(self.config, "vision_pick_descent_coarse_step_mm", legacy_step_mm)),
        )
        fine_step_mm = max(
            0.1,
            float(getattr(self.config, "vision_pick_descent_fine_step_mm", legacy_step_mm)),
        )
        fine_band_mm = max(0.0, float(getattr(self.config, "vision_pick_descent_fine_band_mm", 0.0)))
        current_z = float(current_z)
        remaining_z_mm = abs(current_z - confirm_z)
        step_mm = fine_step_mm if remaining_z_mm <= fine_band_mm else coarse_step_mm
        if current_z > confirm_z:
            next_z = max(confirm_z, current_z - step_mm)
        elif current_z < confirm_z:
            next_z = min(confirm_z, current_z + step_mm)
        else:
            next_z = confirm_z
        command = f"MOVE_CYL {float(theta_deg):.2f} {float(radius_mm):.2f} {float(next_z):.2f}"
        pending = VisionServoPending(
            slot_id=int(slot_id),
            state=SERVO_LOW_CONFIRM,
            attempts=max(0, int(attempts)),
            waiting_for_ack=True,
            min_frame_id=int(frame_id) + 1,
            stability_wait_frames=0,
            command=command,
        )
        return VisionServoDecision(
            action="MOVE",
            state=SERVO_LOW_CONFIRM,
            status=f"descent_confirm slot={int(slot_id)} z={next_z:.1f}/{confirm_z:.1f}",
            message=f"Vision pick descending with visual confirmation: {command}",
            command=command,
            pending=pending,
            reason="descent_confirm",
            trace={
                "slot_id": int(slot_id),
                "frame_id": int(frame_id),
                "current_z_mm": float(current_z),
                "target_z_mm": float(next_z),
                "confirm_z_mm": float(confirm_z),
                "descent_step_mm": float(step_mm),
                "descent_coarse_step_mm": float(coarse_step_mm),
                "descent_fine_step_mm": float(fine_step_mm),
                "descent_fine_band_mm": float(fine_band_mm),
                "remaining_z_mm": float(remaining_z_mm),
            },
        )

    def _wait_stable_decision(
        self,
        *,
        slot_id: int,
        frame_id: int,
        current: VisionServoPending,
    ) -> VisionServoDecision:
        wait_frames = int(current.stability_wait_frames) + 1
        max_wait = max(1, int(getattr(self.config, "vision_grasp_stability_wait_frames", 10)))
        if wait_frames > max_wait:
            return VisionServoDecision(
                action="CANCEL",
                state=SERVO_FAILED,
                status=f"cancelled slot={int(slot_id)} reason=grasp_unstable",
                message=f"Vision servo cancelled slot {int(slot_id)}: grasp point did not stabilize in {max_wait} fresh frames.",
                reason="grasp_unstable",
            )
        pending = VisionServoPending(
            slot_id=int(slot_id),
            state=SERVO_WAIT_STABLE,
            attempts=max(0, int(current.attempts)),
            waiting_for_ack=False,
            min_frame_id=int(frame_id) + 1,
            stability_wait_frames=wait_frames,
            command="WAIT_STABLE",
        )
        return VisionServoDecision(
            action="WAIT_STABLE",
            state=SERVO_WAIT_STABLE,
            status=f"stabilizing {wait_frames}/{max_wait} slot={int(slot_id)}",
            message=f"Vision grasp for slot {int(slot_id)} is near center; waiting for stable frames.",
            pending=pending,
            reason="grasp_unstable",
        )

    def _move_decision(
        self,
        *,
        slot_id: int,
        slot_payload: Mapping[str, object],
        frame_id: int,
        current: VisionServoPending,
        eye_in_hand_enabled: bool,
        current_cyl_pose: tuple[float, float, float] | None = None,
    ) -> VisionServoDecision:
        point = slot_payload.get("servo_command_point")
        mode = str(slot_payload.get("servo_command_mode", "cyl")).strip().lower()
        if mode != "cyl" or not isinstance(point, (tuple, list)) or len(point) < 2:
            return VisionServoDecision(
                action="CANCEL",
                state=SERVO_FAILED,
                status=f"cancelled slot={int(slot_id)} reason=servo_command_unavailable",
                message=f"Vision servo cancelled slot {int(slot_id)}: servo_command_unavailable.",
                reason="servo_command_unavailable",
            )
        max_attempts = max(1, int(getattr(self.config, "vision_servo_max_attempts", 5)))
        if int(current.attempts) >= max_attempts:
            return VisionServoDecision(
                action="CANCEL",
                state=SERVO_FAILED,
                status=f"max_attempts slot={int(slot_id)}",
                message=f"Vision servo reached {max_attempts} attempts for slot {int(slot_id)}.",
                reason="max_attempts",
            )
        try:
            theta_deg = float(point[0])
            radius_mm = float(point[1])
        except (TypeError, ValueError):
            return VisionServoDecision(
                action="CANCEL",
                state=SERVO_FAILED,
                status=f"cancelled slot={int(slot_id)} reason=servo_command_invalid",
                message=f"Vision servo cancelled slot {int(slot_id)}: servo_command_invalid.",
                reason="servo_command_invalid",
            )
        if bool(eye_in_hand_enabled):
            search_z = float(getattr(self.config, "vision_pick_search_z_mm", getattr(self.config, "robot_carry_z", 190.0)))
            confirm_z = float(getattr(self.config, "vision_pick_confirm_z_mm", getattr(self.config, "robot_approach_z", 130.0)))
            pick_z_tolerance = max(0.0, float(getattr(self.config, "vision_pick_z_tolerance_mm", 4.0)))
            current_z = None if current_cyl_pose is None else float(current_cyl_pose[2])
            at_confirm_height = current_z is not None and abs(float(current_z) - confirm_z) <= pick_z_tolerance
            if current.state == SERVO_LOW_CONFIRM or at_confirm_height:
                center_distance = _safe_float(slot_payload.get("center_distance_px"), float("nan"))
                untrusted_error_px = max(
                    0.0,
                    float(getattr(self.config, "vision_low_confirm_untrusted_error_px", 12.0)),
                )
                if math.isfinite(center_distance) and center_distance > untrusted_error_px:
                    return VisionServoDecision(
                        action="CANCEL",
                        state=SERVO_FAILED,
                        status=f"cancelled slot={int(slot_id)} reason=low_confirm_alignment_untrusted",
                        message=(
                            "Low-height visual centering is outside the trusted range; "
                            "run local low-height calibration or reset the block pose before moving."
                        ),
                        reason="low_confirm_alignment_untrusted",
                        trace={
                            "center_distance_px": float(center_distance),
                            "untrusted_error_px": float(untrusted_error_px),
                            "current_z_mm": None if current_z is None else float(current_z),
                            "confirm_z_mm": float(confirm_z),
                        },
                    )
                target_z = confirm_z if at_confirm_height or current_z is None else current_z
                next_state = SERVO_FINE_CENTER
            else:
                started_descent = current_z is not None and float(current_z) < search_z - pick_z_tolerance
                if started_descent:
                    target_z = float(current_z)
                    next_state = SERVO_FINE_CENTER
                else:
                    target_z = search_z
                    next_state = SERVO_COARSE_CENTER
            command = f"MOVE_CYL {theta_deg:.2f} {radius_mm:.2f} {target_z:.2f}"
        else:
            next_state = SERVO_COARSE_CENTER
            command = f"MOVE_CYL_AUTO {theta_deg:.2f} {radius_mm:.2f}"
        next_attempts = int(current.attempts) + 1
        pending = VisionServoPending(
            slot_id=int(slot_id),
            state=next_state,
            attempts=next_attempts,
            waiting_for_ack=True,
            min_frame_id=int(frame_id) + 1,
            stability_wait_frames=0,
            command=command,
        )
        return VisionServoDecision(
            action="MOVE",
            state=next_state,
            status=f"move {next_attempts}/{max_attempts} slot={int(slot_id)}",
            message=f"Vision servo move {next_attempts}/{max_attempts} for slot {int(slot_id)}: {command}",
            command=command,
            pending=pending,
            reason="vision_servo_required",
        )

    @staticmethod
    def _frame_id(packet: Mapping[str, object] | None) -> int:
        if not isinstance(packet, Mapping):
            return 0
        try:
            return int(packet.get("frame_id", 0))
        except (TypeError, ValueError):
            return 0


def _safe_float(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)
