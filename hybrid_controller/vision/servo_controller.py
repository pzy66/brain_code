from __future__ import annotations

from dataclasses import dataclass, field
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
            if bool(eye_in_hand_enabled) and not bool(at_confirm_z):
                return self._low_confirm_decision(
                    slot_id=int(slot_id),
                    frame_id=frame_id,
                    attempts=current.attempts,
                    current_cyl_pose=current_cyl_pose,
                )
            return VisionServoDecision(
                action="PICK",
                state=SERVO_PICK_READY,
                status=f"centered slot={int(slot_id)}; picking",
                message=f"Vision servo centered slot {int(slot_id)}; sending PICK.",
                command=command,
                reason="pick_ready",
                trace={"slot_id": int(slot_id), "frame_id": int(frame_id)},
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

    def _low_confirm_decision(
        self,
        *,
        slot_id: int,
        frame_id: int,
        attempts: int,
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
        theta_deg, radius_mm, _ = current_cyl_pose
        confirm_z = float(getattr(self.config, "vision_pick_confirm_z_mm", getattr(self.config, "robot_approach_z", 130.0)))
        command = f"MOVE_CYL {float(theta_deg):.2f} {float(radius_mm):.2f} {confirm_z:.2f}"
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
            status=f"low_confirm slot={int(slot_id)} z={confirm_z:.1f}",
            message=f"Vision pick lowering for final confirmation: {command}",
            command=command,
            pending=pending,
            reason="low_confirm",
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
            if current.state == SERVO_LOW_CONFIRM:
                target_z = float(getattr(self.config, "vision_pick_confirm_z_mm", getattr(self.config, "robot_approach_z", 130.0)))
                next_state = SERVO_FINE_CENTER
            else:
                target_z = float(getattr(self.config, "vision_pick_search_z_mm", getattr(self.config, "robot_carry_z", 190.0)))
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
