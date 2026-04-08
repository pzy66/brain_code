from __future__ import annotations

from dataclasses import asdict
from typing import Any

from hybrid_controller.adapters.vision_adapter import VisionTarget, snapshot_targets
from hybrid_controller.cylindrical import (
    build_auto_z_profile,
    cartesian_to_cylindrical,
    cylindrical_to_cartesian,
    interpolate_auto_z,
    nearest_profile_radius_limits,
    within_limits,
)
from hybrid_controller.config import AppConfig

from .context import TaskContext
from .events import Effect, Event
from .state_machine import TaskState


class TaskController:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.state = TaskState.IDLE
        start_cyl = cartesian_to_cylindrical(
            config.robot_start_xy[0],
            config.robot_start_xy[1],
            config.robot_carry_z,
        )
        self._auto_z_profile = build_auto_z_profile(
            radius_limits=config.robot_radius_limits_mm,
            z_limits=config.robot_height_limits_mm,
            radius_step_mm=config.robot_auto_z_profile_radius_step_mm,
            z_step_mm=config.robot_auto_z_profile_height_step_mm,
            validator=self._validate_cylindrical_target,
            preferred_z_mm=config.robot_auto_z_preferred_mm,
            profile_theta_deg=0.0,
        )
        self.context = TaskContext(
            robot_xy=config.robot_start_xy,
            robot_cyl=start_cyl,
            robot_auto_z=interpolate_auto_z(self._auto_z_profile, start_cyl[1]),
        )
        self._timer_seq = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "context": {
                **asdict(self.context),
                "frozen_targets": [target.to_dict() for target in self.context.frozen_targets],
                "latest_vision_targets": [target.to_dict() for target in self.context.latest_vision_targets],
            },
        }

    def handle_event(self, event: Event) -> list[Effect]:
        event = event.with_default_timestamp()
        effects: list[Effect] = []

        if event.type == "vision_update":
            self._store_vision_targets(event, effects)
        elif event.type == "reset_task":
            self._reset_task(effects)
        elif event.type == "start_task":
            self._start_task(event.timestamp, effects)
        elif event.type == "move":
            self._handle_move(event, effects)
        elif event.type == "timer_expired":
            self._handle_timer_expired(event, effects)
        elif event.type == "decision_confirm":
            self._handle_decision_confirm(event, effects)
        elif event.type == "decision_cancel":
            self._handle_decision_cancel(event, effects)
        elif event.type == "target_selected":
            self._handle_target_selected(event, effects)
        elif event.type == "robot_ack":
            self._handle_robot_ack(event, effects)
        elif event.type == "robot_busy":
            self.context.last_robot_status = "BUSY"
            self._handle_robot_failure("Robot reported BUSY", effects)
        elif event.type == "robot_error":
            self.context.last_error = str(event.value or "Unknown robot error")
            self._handle_robot_failure(self.context.last_error, effects)

        return effects

    def _start_task(self, timestamp: float, effects: list[Effect]) -> None:
        latest_targets = list(self.context.latest_vision_targets)
        robot_xy = self.context.robot_xy
        robot_cyl = self.context.robot_cyl
        self._cancel_active_timer(effects)
        self.context = TaskContext(
            robot_xy=robot_xy,
            robot_cyl=robot_cyl,
            robot_auto_z=self.context.robot_auto_z,
            latest_vision_targets=latest_targets,
        )
        self._enter_motion_state(TaskState.S1_MI_MOVE, self.config.stage_motion_sec, timestamp, effects)

    def _reset_task(self, effects: list[Effect]) -> None:
        latest_targets = list(self.context.latest_vision_targets)
        robot_xy = self.context.robot_xy
        robot_cyl = self.context.robot_cyl
        self._cancel_active_timer(effects)
        self.context = TaskContext(
            robot_xy=robot_xy,
            robot_cyl=robot_cyl,
            robot_auto_z=self.context.robot_auto_z,
            latest_vision_targets=latest_targets,
        )
        self._set_state(TaskState.IDLE, effects)

    def _store_vision_targets(self, event: Event, effects: list[Effect]) -> None:
        raw_targets = event.value or []
        targets = [target for target in raw_targets if isinstance(target, VisionTarget)]
        self.context.latest_vision_targets = list(targets)
        if self.state == TaskState.S2_TARGET_SELECT and not self.context.frozen_targets:
            self.context.frozen_targets = snapshot_targets(
                targets=targets,
                roi_center=self.config.roi_center,
                roi_radius=self.config.roi_radius,
                limit=self.config.vision_max_targets,
            )
            if self.context.frozen_targets:
                effects.append(
                    Effect(
                        "log",
                        {"message": f"Frozen {len(self.context.frozen_targets)} targets for stage 2 selection."},
                    )
                )
            else:
                effects.append(Effect("log", {"message": "No targets available in ROI for stage 2 selection."}))

    def _handle_move(self, event: Event, effects: list[Effect]) -> None:
        if self.state not in (TaskState.S1_MI_MOVE, TaskState.S3_MI_CARRY):
            return
        if self.context.pending_robot_xy is not None or self.context.pending_robot_cyl is not None:
            effects.append(Effect("log", {"message": "Ignoring move input while waiting for MOVE acknowledgement."}))
            return
        if self.config.motion_coordinate_mode == "cylindrical":
            dtheta, dr = self._resolve_move_cyl_delta(event)
            cur_theta, cur_radius, _ = self.context.robot_cyl
            next_theta = max(
                self.config.robot_theta_limits_deg[0],
                min(self.config.robot_theta_limits_deg[1], cur_theta + dtheta),
            )
            effective_radius_limits = nearest_profile_radius_limits(self._auto_z_profile)
            next_radius = max(effective_radius_limits[0], min(effective_radius_limits[1], cur_radius + dr))
            next_z = interpolate_auto_z(self._auto_z_profile, next_radius)
            next_x, next_y, _ = cylindrical_to_cartesian(next_theta, next_radius, next_z)
            self.context.pending_robot_cyl = (next_theta, next_radius, next_z)
            self.context.pending_robot_xy = (next_x, next_y)
            command = f"MOVE_CYL_AUTO {next_theta:.2f} {next_radius:.2f}"
        else:
            dx, dy = self._resolve_move_delta(event)
            cur_x, cur_y = self.context.robot_xy
            min_x, max_x = self.config.motion_bounds_x
            min_y, max_y = self.config.motion_bounds_y
            next_x = max(min_x, min(max_x, cur_x + dx))
            next_y = max(min_y, min(max_y, cur_y + dy))
            self.context.pending_robot_xy = (next_x, next_y)
            command = f"MOVE {next_x:.2f} {next_y:.2f}"
        self.context.robot_busy = True
        self.context.robot_execution_state = "MOVING_XY"
        effects.append(Effect("robot_command", {"command": command}))

    def _handle_timer_expired(self, event: Event, effects: list[Effect]) -> None:
        timer_id = self._extract_timer_id(event.value)
        if timer_id != self.context.active_timer_id:
            return
        self.context.active_timer_id = None
        self.context.motion_deadline_ts = None
        if self.state == TaskState.S1_MI_MOVE:
            self._set_state(TaskState.S1_DECISION, effects)
        elif self.state == TaskState.S3_MI_CARRY:
            self._set_state(TaskState.S3_DECISION, effects)

    def _handle_decision_confirm(self, event: Event, effects: list[Effect]) -> None:
        if self.state == TaskState.S1_DECISION:
            self._set_state(TaskState.S2_TARGET_SELECT, effects)
            self.context.frozen_targets = snapshot_targets(
                targets=self.context.latest_vision_targets,
                roi_center=self.config.roi_center,
                roi_radius=self.config.roi_radius,
                limit=self.config.vision_max_targets,
            )
            if not self.context.frozen_targets:
                effects.append(Effect("log", {"message": "Entered target selection with no candidates yet."}))
            return

        if self.state == TaskState.S2_GRAB_CONFIRM and self.context.selected_target_command_point is not None:
            if self.context.robot_busy:
                effects.append(Effect("log", {"message": "Robot is still moving; wait before confirming PICK."}))
                return
            command_mode = str(self.context.selected_target_command_mode or "pixel").strip().lower()
            raw_x, raw_y = self.context.selected_target_command_point
            self._set_state(TaskState.S2_PICKING, effects)
            if command_mode == "world":
                command = f"PICK_WORLD {raw_x:.2f} {raw_y:.2f}"
            elif command_mode == "cyl":
                command = f"PICK_CYL {raw_x:.2f} {raw_y:.2f}"
            else:
                command = f"PICK {raw_x:.2f} {raw_y:.2f}"
            effects.append(Effect("robot_command", {"command": command}))
            return

        if self.state == TaskState.S3_DECISION:
            if self.context.robot_busy:
                effects.append(Effect("log", {"message": "Robot is still moving; wait before confirming PLACE."}))
                return
            self._set_state(TaskState.S3_PLACING, effects)
            effects.append(Effect("robot_command", {"command": "PLACE"}))

    def _handle_decision_cancel(self, event: Event, effects: list[Effect]) -> None:
        if self.state == TaskState.S1_DECISION:
            self._enter_motion_state(TaskState.S1_MI_MOVE, self.config.continue_motion_sec, event.timestamp, effects)
            return

        if self.state == TaskState.S2_GRAB_CONFIRM:
            self.context.clear_selection()
            self.context.frozen_targets = []
            self._enter_motion_state(TaskState.S1_MI_MOVE, self.config.continue_motion_sec, event.timestamp, effects)
            return

        if self.state == TaskState.S3_DECISION:
            self._enter_motion_state(TaskState.S3_MI_CARRY, self.config.continue_motion_sec, event.timestamp, effects)

    def _handle_target_selected(self, event: Event, effects: list[Effect]) -> None:
        if self.state != TaskState.S2_TARGET_SELECT:
            return
        try:
            target_index = int(event.value)
        except (TypeError, ValueError):
            return
        if target_index < 0 or target_index >= len(self.context.frozen_targets):
            return
        target = self.context.frozen_targets[target_index]
        self.context.selected_target_id = target.id
        self.context.selected_target_raw_center = target.raw_center
        self.context.selected_target_command_mode = str(getattr(target, "command_mode", "pixel"))
        command_point = getattr(target, "command_point", None)
        self.context.selected_target_command_point = None if command_point is None else tuple(command_point)
        self._set_state(TaskState.S2_GRAB_CONFIRM, effects)

    def _handle_robot_ack(self, event: Event, effects: list[Effect]) -> None:
        ack = str(event.value or "").strip().upper()
        if not ack:
            return
        self.context.last_robot_status = ack
        if ack == "MOVE":
            self.context.robot_busy = False
            self.context.robot_execution_state = "IDLE" if not self.context.carrying else "CARRY_READY"
            if self.context.pending_robot_xy is not None:
                self.context.robot_xy = self.context.pending_robot_xy
                self.context.pending_robot_xy = None
            if self.context.pending_robot_cyl is not None:
                self.context.robot_cyl = self.context.pending_robot_cyl
                self.context.robot_auto_z = self.context.pending_robot_cyl[2]
                self.context.pending_robot_cyl = None
            return
        if ack == "PICK_DONE" and self.state == TaskState.S2_PICKING:
            self.context.robot_busy = False
            self.context.robot_execution_state = "CARRY_READY"
            self.context.carrying = True
            self.context.clear_selection()
            self.context.frozen_targets = []
            self._enter_motion_state(TaskState.S3_MI_CARRY, self.config.stage_motion_sec, event.timestamp, effects)
            return
        if ack == "PLACE_DONE" and self.state == TaskState.S3_PLACING:
            self.context.robot_busy = False
            self.context.robot_execution_state = "IDLE"
            self.context.carrying = False
            self._set_state(TaskState.FINISHED, effects)

    def _handle_robot_failure(self, message: str, effects: list[Effect]) -> None:
        self.context.pending_robot_xy = None
        self.context.pending_robot_cyl = None
        self.context.robot_busy = True
        effects.append(Effect("log", {"message": message}))
        if self.state in (TaskState.S2_PICKING, TaskState.S3_PLACING):
            self._cancel_active_timer(effects)
            self._set_state(TaskState.ERROR, effects)

    def _enter_motion_state(
        self,
        state: TaskState,
        duration_sec: float,
        timestamp: float,
        effects: list[Effect],
    ) -> None:
        self._cancel_active_timer(effects)
        self._set_state(state, effects)
        self._timer_seq += 1
        timer_id = f"motion-{self._timer_seq}"
        self.context.active_timer_id = timer_id
        self.context.motion_deadline_ts = timestamp + duration_sec
        effects.append(
            Effect(
                "start_timer",
                {"timer_id": timer_id, "duration_sec": duration_sec, "deadline_ts": self.context.motion_deadline_ts},
            )
        )

    def _cancel_active_timer(self, effects: list[Effect]) -> None:
        if self.context.active_timer_id:
            effects.append(Effect("cancel_timer", {"timer_id": self.context.active_timer_id}))
        self.context.active_timer_id = None
        self.context.motion_deadline_ts = None

    def _set_state(self, new_state: TaskState, effects: list[Effect]) -> None:
        if self.state == new_state:
            return
        old_state = self.state
        self.state = new_state
        effects.append(Effect("state_changed", {"from": old_state.value, "to": new_state.value}))

    def _resolve_move_delta(self, event: Event) -> tuple[float, float]:
        value = event.value
        step = float(self.config.sim_move_step_mm if event.source == "sim" else self.config.mi_step_mm)
        if isinstance(value, dict):
            if "dx" in value or "dy" in value:
                return float(value.get("dx", 0.0)), float(value.get("dy", 0.0))
            value = value.get("direction")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return float(value[0]), float(value[1])
        mapping = {
            "left": (-step, 0.0),
            "right": (step, 0.0),
            "forward": (0.0, step),
            "backward": (0.0, -step),
        }
        return mapping.get(str(value).strip().lower(), (0.0, 0.0))

    def _resolve_move_cyl_delta(self, event: Event) -> tuple[float, float]:
        value = event.value
        theta_step = float(self.config.teleop_theta_step_deg if event.source == "sim" else self.config.teleop_theta_step_deg)
        radius_step = float(self.config.teleop_radius_step_mm if event.source == "sim" else self.config.mi_step_mm)
        if isinstance(value, dict):
            if "dtheta" in value or "dr" in value:
                return float(value.get("dtheta", 0.0)), float(value.get("dr", 0.0))
            value = value.get("direction")
        mapping = {
            "left": (-theta_step, 0.0),
            "right": (theta_step, 0.0),
            "forward": (0.0, radius_step),
            "backward": (0.0, -radius_step),
        }
        return mapping.get(str(value).strip().lower(), (0.0, 0.0))

    def _validate_cylindrical_target(self, theta_deg: float, radius_mm: float, z_mm: float) -> dict[str, object]:
        if not within_limits(theta_deg, self.config.robot_theta_limits_deg):
            return {"ok": False, "message": f"Theta {theta_deg:.2f} deg is outside limits."}
        if not within_limits(radius_mm, self.config.robot_radius_limits_mm):
            return {"ok": False, "message": f"Radius {radius_mm:.2f} mm is outside limits."}
        if not within_limits(z_mm, self.config.robot_height_limits_mm):
            return {"ok": False, "message": f"Height {z_mm:.2f} mm is outside limits."}
        x_mm, y_mm, _ = cylindrical_to_cartesian(theta_deg, radius_mm, z_mm)
        if not within_limits(x_mm, self.config.robot_limits_x):
            return {"ok": False, "message": f"X {x_mm:.2f} mm is outside safe bounds."}
        if not within_limits(y_mm, self.config.robot_limits_y):
            return {"ok": False, "message": f"Y {y_mm:.2f} mm is outside safe bounds."}
        margin = min(
            x_mm - self.config.robot_limits_x[0],
            self.config.robot_limits_x[1] - x_mm,
            y_mm - self.config.robot_limits_y[0],
            self.config.robot_limits_y[1] - y_mm,
            radius_mm - self.config.robot_radius_limits_mm[0],
            self.config.robot_height_limits_mm[1] - z_mm,
        )
        return {"ok": True, "message": None, "margin": float(margin), "neutral_distance": abs(z_mm - self.config.robot_auto_z_preferred_mm)}

    @staticmethod
    def _extract_timer_id(value: Any) -> str | None:
        if isinstance(value, dict):
            timer_id = value.get("timer_id")
            return None if timer_id is None else str(timer_id)
        if value is None:
            return None
        return str(value)
