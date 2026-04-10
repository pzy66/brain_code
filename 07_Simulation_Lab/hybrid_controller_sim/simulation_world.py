from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

from hybrid_controller.adapters.control_sim_slots import ControlSimSlotCatalog
from hybrid_controller.adapters.vision_adapter import VisionTarget
from hybrid_controller.cylindrical import (
    build_auto_z_profile,
    cartesian_to_cylindrical,
    cylindrical_to_cartesian,
    interpolate_auto_z,
    nearest_profile_radius_limits,
    within_limits,
)
from hybrid_controller.config import AppConfig
from hybrid_controller.robot.runtime.robot_protocol import RobotErrorCode, RobotExecutorState, is_busy_state


@dataclass(slots=True)
class PickSlotState:
    slot_id: int
    name: str
    world_xy: tuple[float, float]
    pixel_xy: tuple[float, float]
    occupied: bool = True
    object_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "name": self.name,
            "world_xy": self.world_xy,
            "pixel_xy": self.pixel_xy,
            "occupied": self.occupied,
            "object_id": self.object_id,
        }


@dataclass(slots=True)
class PlaceSlotState:
    slot_id: int
    name: str
    world_xy: tuple[float, float]
    pixel_xy: tuple[float, float]
    object_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "name": self.name,
            "world_xy": self.world_xy,
            "pixel_xy": self.pixel_xy,
            "object_id": self.object_id,
            "occupied": self.object_id is not None,
        }


class SimulationWorld:
    def __init__(self, config: AppConfig, scenario_name: str = "basic") -> None:
        self.config = config
        self.slot_catalog = ControlSimSlotCatalog(config)
        self._lock = threading.RLock()
        self._revision = 0
        self.scenario_name = "basic"
        self.robot_xy: tuple[float, float] = config.robot_start_xy
        self.robot_z: float = float(config.robot_carry_z)
        self.robot_cyl: tuple[float, float, float] = cartesian_to_cylindrical(
            config.robot_start_xy[0],
            config.robot_start_xy[1],
            config.robot_carry_z,
        )
        self.home_pose: tuple[float, float, float] = (
            float(config.robot_start_xy[0]),
            float(config.robot_start_xy[1]),
            float(config.robot_carry_z),
        )
        self.auto_z_profile = build_auto_z_profile(
            radius_limits=config.robot_radius_limits_mm,
            z_limits=config.robot_height_limits_mm,
            radius_step_mm=config.robot_auto_z_profile_radius_step_mm,
            z_step_mm=config.robot_auto_z_profile_height_step_mm,
            validator=self._validate_cylindrical_target,
            preferred_z_mm=config.robot_auto_z_preferred_mm,
            profile_theta_deg=0.0,
        )
        self.motion_target_xy: tuple[float, float] | None = None
        self.motion_target_cyl: tuple[float, float, float] | None = None
        self.motion_started_xy: tuple[float, float] | None = None
        self.motion_started_at: float | None = None
        self.motion_duration_sec: float = 0.0
        self.robot_state: RobotExecutorState = RobotExecutorState.IDLE
        self.pending_pick_slot_id: int | None = None
        self.sucker_on: bool = False
        self.carrying_target_id: int | None = None
        self.last_ack: str | None = None
        self.last_error: str | None = None
        self.last_error_code: str | None = None
        self.control_kernel: str = "cylindrical_kernel"
        self.pick_slots: dict[int, PickSlotState] = {}
        self.place_slots: dict[int, PlaceSlotState] = {}
        self._pick_error_used = False
        self._place_error_used = False
        self.reset(scenario_name)

    def reset(self, scenario: str | None = None) -> dict[str, object]:
        name = str(scenario or self.scenario_name or "basic").strip().lower()
        with self._lock:
            self.scenario_name = name
            self.robot_xy = self.config.robot_start_xy
            self.robot_z = float(self.config.robot_carry_z)
            self.robot_cyl = cartesian_to_cylindrical(
                self.config.robot_start_xy[0],
                self.config.robot_start_xy[1],
                self.config.robot_carry_z,
            )
            self.motion_target_xy = None
            self.motion_target_cyl = None
            self.motion_started_xy = None
            self.motion_started_at = None
            self.motion_duration_sec = 0.0
            self.robot_state = RobotExecutorState.IDLE
            self.pending_pick_slot_id = None
            self.sucker_on = False
            self.carrying_target_id = None
            self.last_ack = None
            self.last_error = None
            self.last_error_code = None
            self.control_kernel = "cylindrical_kernel"
            self._pick_error_used = False
            self._place_error_used = False
            self.pick_slots = {}
            self.place_slots = {}

            for slot in self.slot_catalog.list_pick_slots(source=self._pick_slot_source()):
                occupied = True
                object_id = slot.slot_id
                if name == "empty_roi":
                    occupied = False
                    object_id = None
                elif name == "sparse_targets" and slot.slot_id != 1:
                    occupied = False
                    object_id = None
                self.pick_slots[slot.slot_id] = PickSlotState(
                    slot_id=slot.slot_id,
                    name=slot.name,
                    world_xy=slot.world_xy,
                    pixel_xy=slot.pixel_xy,
                    occupied=occupied,
                    object_id=object_id,
                )

            for slot in self.slot_catalog.list_place_slots():
                self.place_slots[slot.slot_id] = PlaceSlotState(
                    slot_id=slot.slot_id,
                    name=slot.name,
                    world_xy=slot.world_xy,
                    pixel_xy=slot.pixel_xy,
                )

            self._bump_revision()
            return self._snapshot_locked()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            self._advance_motion_locked()
            return self._snapshot_locked()

    def visible_targets(self) -> list[VisionTarget]:
        with self._lock:
            targets = []
            command_mode = "pixel"
            if self.config.vision_mode == "fixed_world_slots":
                command_mode = "world"
            elif self.config.vision_mode == "fixed_cyl_slots":
                command_mode = "cyl"
            for target in self.slot_catalog.build_selection_targets(
                source=self._pick_slot_source(),
                command_mode=command_mode,
            ):
                slot_state = self.pick_slots.get(target.id)
                if slot_state is not None and slot_state.occupied:
                    targets.append(target)
            return targets

    def handle_move(self, x: float, y: float) -> dict[str, object]:
        with self._lock:
            self._advance_motion_locked()
            if self.robot_state == RobotExecutorState.ERROR:
                return self._error_result(
                    RobotErrorCode.INVALID_STATE,
                    "Robot is in ERROR state; reset is required.",
                )
            if self.robot_state not in {RobotExecutorState.IDLE, RobotExecutorState.CARRY_READY}:
                return self._busy_result()

            tx = self._clamp(x, self.config.robot_limits_x)
            ty = self._clamp(y, self.config.robot_limits_y)
            duration = self._plan_motion_locked((tx, ty), state=RobotExecutorState.MOVING_XY)
            self.control_kernel = "legacy_cartesian"
            self.robot_z = float(self.config.robot_carry_z)
            self.last_ack = None
            self.last_error = None
            self.last_error_code = None
            self._bump_revision()
            return {
                "status": "accepted",
                "duration_sec": duration,
                "target_xy": (tx, ty),
                "state": self.robot_state.value,
            }

    def handle_move_cyl(self, theta_deg: float, radius_mm: float, z_mm: float) -> dict[str, object]:
        with self._lock:
            self._advance_motion_locked()
            if self.robot_state == RobotExecutorState.ERROR:
                return self._error_result(RobotErrorCode.INVALID_STATE, "Robot is in ERROR state; reset is required.")
            if self.robot_state not in {RobotExecutorState.IDLE, RobotExecutorState.CARRY_READY}:
                return self._busy_result()
            validation = self._validate_cylindrical_target(theta_deg, radius_mm, z_mm)
            if not validation["ok"]:
                return self._error_result(RobotErrorCode.TARGET_OUT_OF_WORKSPACE, validation["message"])
            tx, ty, _ = cylindrical_to_cartesian(theta_deg, radius_mm, z_mm)
            duration = self._plan_motion_locked((tx, ty), state=RobotExecutorState.MOVING_XY)
            self.control_kernel = "cylindrical_kernel"
            self.motion_target_cyl = (float(theta_deg), float(radius_mm), float(z_mm))
            self.robot_z = float(z_mm)
            self.last_ack = None
            self.last_error = None
            self.last_error_code = None
            self._bump_revision()
            return {
                "status": "accepted",
                "duration_sec": duration,
                "target_xy": (tx, ty),
                "target_cyl": self.motion_target_cyl,
                "state": self.robot_state.value,
            }

    def handle_move_cyl_auto(self, theta_deg: float, radius_mm: float) -> dict[str, object]:
        effective_limits = nearest_profile_radius_limits(self.auto_z_profile)
        if not within_limits(radius_mm, effective_limits):
            return self._error_result(
                RobotErrorCode.TARGET_OUT_OF_WORKSPACE,
                f"Radius {float(radius_mm):.2f} mm is outside auto-z profile range {effective_limits}.",
            )
        desired_z_mm = interpolate_auto_z(self.auto_z_profile, radius_mm)
        current_radius_mm = float(self.robot_cyl[1])
        current_z_mm = float(self.robot_z)
        delta_radius_mm = abs(float(radius_mm) - current_radius_mm)
        max_down_step_mm = max(
            float(self.config.robot_auto_z_min_delta_mm),
            delta_radius_mm * float(self.config.robot_auto_z_down_per_radius_mm),
        )
        max_up_step_mm = max(
            float(self.config.robot_auto_z_min_delta_mm),
            delta_radius_mm * float(self.config.robot_auto_z_up_per_radius_mm),
        )
        if desired_z_mm < current_z_mm:
            target_z_mm = max(desired_z_mm, current_z_mm - max_down_step_mm)
        else:
            target_z_mm = min(desired_z_mm, current_z_mm + max_up_step_mm)
        return self.handle_move_cyl(theta_deg, radius_mm, target_z_mm)

    def begin_pick(self, pixel_x: float, pixel_y: float) -> dict[str, object]:
        with self._lock:
            self._advance_motion_locked()
            if self.robot_state == RobotExecutorState.ERROR:
                return self._error_result(
                    RobotErrorCode.INVALID_STATE,
                    "Cannot pick while robot is in ERROR state.",
                )
            if self.robot_state not in {RobotExecutorState.IDLE, RobotExecutorState.CARRY_READY}:
                return self._busy_result()
            if self.carrying_target_id is not None:
                return self._error_result(
                    RobotErrorCode.INVALID_STATE,
                    "Cannot pick while already carrying a target.",
                )
            if self.scenario_name == "invalid_pick_slot":
                return self._error_result(
                    RobotErrorCode.TARGET_OUT_OF_WORKSPACE,
                    "Invalid synthetic pick slot.",
                )

            slot = self.slot_catalog.resolve_pick_slot(float(pixel_x), float(pixel_y))
            if slot is None:
                return self._error_result(
                    RobotErrorCode.TARGET_OUT_OF_WORKSPACE,
                    "No synthetic pick slot matched the requested pixel.",
                )

            slot_state = self.pick_slots.get(slot.slot_id)
            if slot_state is None or not slot_state.occupied or slot_state.object_id is None:
                return self._error_result(
                    RobotErrorCode.INVALID_STATE,
                    f"Pick slot {slot.slot_id} is empty.",
                )

            target_x = self._clamp(slot.world_xy[0], self.config.robot_limits_x)
            target_y = self._clamp(slot.world_xy[1], self.config.robot_limits_y)
            overflow = max(abs(slot.world_xy[0] - target_x), abs(slot.world_xy[1] - target_y))
            if overflow > float(self.config.robot_target_margin_mm):
                return self._error_result(
                    RobotErrorCode.TARGET_OUT_OF_WORKSPACE,
                    f"Target exceeds workspace by {overflow:.2f} mm.",
                )

            duration = self._plan_motion_locked((target_x, target_y), state=RobotExecutorState.PICK_APPROACH)
            self.pending_pick_slot_id = slot.slot_id
            self.control_kernel = "legacy_cartesian"
            self.robot_z = float(self.config.robot_carry_z)
            self.sucker_on = False
            self.last_ack = "PICK_STARTED"
            self.last_error = None
            self.last_error_code = None
            self._bump_revision()
            return {
                "status": "started",
                "delay_sec": duration,
                "phase": self.robot_state.value,
                "target_slot_id": slot.slot_id,
                "state": self.robot_state.value,
            }

    def begin_pick_world(self, x: float, y: float) -> dict[str, object]:
        with self._lock:
            self._advance_motion_locked()
            if self.robot_state == RobotExecutorState.ERROR:
                return self._error_result(
                    RobotErrorCode.INVALID_STATE,
                    "Cannot pick while robot is in ERROR state.",
                )
            if self.robot_state not in {RobotExecutorState.IDLE, RobotExecutorState.CARRY_READY}:
                return self._busy_result()
            if self.carrying_target_id is not None:
                return self._error_result(
                    RobotErrorCode.INVALID_STATE,
                    "Cannot pick while already carrying a target.",
                )
            if self.scenario_name == "invalid_world_slot":
                return self._error_result(
                    RobotErrorCode.TARGET_OUT_OF_WORKSPACE,
                    "Invalid fixed world pick slot.",
                )
            target_x = float(x)
            target_y = float(y)
            if target_x < float(self.config.robot_limits_x[0]) or target_x > float(self.config.robot_limits_x[1]):
                overflow = min(
                    abs(target_x - float(self.config.robot_limits_x[0])),
                    abs(target_x - float(self.config.robot_limits_x[1])),
                )
                return self._error_result(
                    RobotErrorCode.TARGET_OUT_OF_WORKSPACE,
                    f"Target exceeds workspace by {overflow:.2f} mm.",
                )
            if target_y < float(self.config.robot_limits_y[0]) or target_y > float(self.config.robot_limits_y[1]):
                overflow = min(
                    abs(target_y - float(self.config.robot_limits_y[0])),
                    abs(target_y - float(self.config.robot_limits_y[1])),
                )
                return self._error_result(
                    RobotErrorCode.TARGET_OUT_OF_WORKSPACE,
                    f"Target exceeds workspace by {overflow:.2f} mm.",
                )

            slot = self.slot_catalog.resolve_world_pick_slot(target_x, target_y, source=self._pick_slot_source())
            if slot is None:
                return self._error_result(
                    RobotErrorCode.TARGET_OUT_OF_WORKSPACE,
                    "No fixed world pick slot matched the requested coordinate.",
                )
            slot_state = self.pick_slots.get(slot.slot_id)
            if slot_state is None or not slot_state.occupied or slot_state.object_id is None:
                return self._error_result(
                    RobotErrorCode.INVALID_STATE,
                    f"Pick slot {slot.slot_id} is empty.",
                )

            duration = self._plan_motion_locked((target_x, target_y), state=RobotExecutorState.PICK_APPROACH)
            self.pending_pick_slot_id = slot.slot_id
            self.control_kernel = "legacy_cartesian"
            self.robot_z = float(self.config.robot_carry_z)
            self.sucker_on = False
            self.last_ack = "PICK_STARTED"
            self.last_error = None
            self.last_error_code = None
            self._bump_revision()
            return {
                "status": "started",
                "delay_sec": duration,
                "phase": self.robot_state.value,
                "target_slot_id": slot.slot_id,
                "state": self.robot_state.value,
            }

    def begin_pick_cyl(self, theta_deg: float, radius_mm: float) -> dict[str, object]:
        validation = self._validate_cylindrical_target(theta_deg, radius_mm, self.config.robot_pick_z)
        if not validation["ok"]:
            return self._error_result(RobotErrorCode.TARGET_OUT_OF_WORKSPACE, validation["message"])
        x_mm, y_mm, _ = cylindrical_to_cartesian(theta_deg, radius_mm, self.config.robot_pick_z)
        result = self.begin_pick_world(x_mm, y_mm)
        if result.get("status") == "started":
            self.control_kernel = "cylindrical_kernel"
            self._bump_revision()
        return result

    def step_pick(self) -> dict[str, object]:
        with self._lock:
            self._advance_motion_locked()
            if self.pending_pick_slot_id is None:
                return self._error_result(RobotErrorCode.INVALID_STATE, "Pick was not in progress.")

            slot_state = self.pick_slots[self.pending_pick_slot_id]
            if self.robot_state == RobotExecutorState.PICK_APPROACH:
                if self.motion_target_xy is not None:
                    return {
                        "status": "progress",
                        "delay_sec": self._remaining_motion_delay_locked(),
                        "phase": self.robot_state.value,
                    }
                if self.scenario_name == "pick_error" and not self._pick_error_used:
                    self._pick_error_used = True
                    return self._enter_error_locked(
                        RobotErrorCode.HARDWARE_FAILURE,
                        "Injected pick failure",
                        preserve_carrying=False,
                        disable_sucker=True,
                    )
                self.robot_xy = slot_state.world_xy
                self.robot_z = float(self.config.robot_approach_z)
                self.sucker_on = True
                self._clear_motion_locked()
                self.robot_state = RobotExecutorState.PICK_SUCTION_ON
                self._bump_revision()
                return {"status": "progress", "delay_sec": self._phase_delay(0.1), "phase": self.robot_state.value}

            if self.robot_state == RobotExecutorState.PICK_SUCTION_ON:
                self.robot_state = RobotExecutorState.PICK_DESCEND
                self.robot_z = float(self.config.robot_pick_z)
                self._bump_revision()
                return {"status": "progress", "delay_sec": self._phase_delay(0.9), "phase": self.robot_state.value}

            if self.robot_state == RobotExecutorState.PICK_DESCEND:
                slot_state.occupied = False
                self.carrying_target_id = slot_state.object_id
                slot_state.object_id = None
                self.robot_state = RobotExecutorState.PICK_LIFT
                self.robot_z = float(self.config.robot_carry_z)
                self._bump_revision()
                return {"status": "progress", "delay_sec": self._phase_delay(0.8), "phase": self.robot_state.value}

            if self.robot_state == RobotExecutorState.PICK_LIFT:
                self.robot_state = RobotExecutorState.CARRY_READY
                self.pending_pick_slot_id = None
                self.last_ack = "PICK_DONE"
                self.last_error = None
                self.last_error_code = None
                self._bump_revision()
                return {"status": "done", "message": "PICK_DONE", "target_id": self.carrying_target_id}

            return self._error_result(
                RobotErrorCode.INVALID_STATE,
                f"Unexpected pick phase: {self.robot_state.value}",
            )

    def begin_place(self) -> dict[str, object]:
        with self._lock:
            self._advance_motion_locked()
            if self.robot_state == RobotExecutorState.ERROR:
                return self._error_result(
                    RobotErrorCode.INVALID_STATE,
                    "Cannot place while robot is in ERROR state.",
                )
            if self.robot_state not in {RobotExecutorState.IDLE, RobotExecutorState.CARRY_READY}:
                return self._busy_result()
            if self.carrying_target_id is None:
                return self._error_result(
                    RobotErrorCode.INVALID_STATE,
                    "Cannot place without a carried target.",
                )

            self.robot_state = RobotExecutorState.PLACE_DESCEND
            self.last_ack = "PLACE_STARTED"
            self.last_error = None
            self.last_error_code = None
            self._bump_revision()
            return {"status": "started", "delay_sec": self._phase_delay(0.8), "phase": self.robot_state.value}

    def step_place(self) -> dict[str, object]:
        with self._lock:
            self._advance_motion_locked()
            if self.carrying_target_id is None:
                return self._error_result(RobotErrorCode.INVALID_STATE, "Place was not in progress.")

            if self.robot_state == RobotExecutorState.PLACE_DESCEND:
                self.robot_z = float(self.config.robot_pick_z)
                if self.scenario_name == "place_error" and not self._place_error_used:
                    self._place_error_used = True
                    return self._enter_error_locked(
                        RobotErrorCode.HARDWARE_FAILURE,
                        "Injected place failure",
                        preserve_carrying=True,
                        disable_sucker=False,
                    )
                self.robot_state = RobotExecutorState.PLACE_RELEASE
                self._bump_revision()
                return {"status": "progress", "delay_sec": self._phase_delay(0.2), "phase": self.robot_state.value}

            if self.robot_state == RobotExecutorState.PLACE_RELEASE:
                self.sucker_on = False
                place_slot = self.slot_catalog.nearest_place_slot(self.robot_xy)
                if place_slot is not None:
                    self.place_slots[place_slot.slot_id].object_id = self.carrying_target_id
                else:
                    dynamic_slot_id = 1000 + int(self.carrying_target_id)
                    self.place_slots[dynamic_slot_id] = PlaceSlotState(
                        slot_id=dynamic_slot_id,
                        name=f"Drop-{self.carrying_target_id}",
                        world_xy=(float(self.robot_xy[0]), float(self.robot_xy[1])),
                        pixel_xy=self._project_world_to_pixel(self.robot_xy),
                        object_id=self.carrying_target_id,
                    )
                self.robot_state = RobotExecutorState.PLACE_LIFT
                self._bump_revision()
                return {"status": "progress", "delay_sec": self._phase_delay(0.6), "phase": self.robot_state.value}

            if self.robot_state == RobotExecutorState.PLACE_LIFT:
                placed_target_id = self.carrying_target_id
                self.robot_z = float(self.config.robot_carry_z)
                self.carrying_target_id = None
                self.robot_state = RobotExecutorState.IDLE
                self.last_ack = "PLACE_DONE"
                self.last_error = None
                self.last_error_code = None
                self._bump_revision()
                return {"status": "done", "message": "PLACE_DONE", "target_id": placed_target_id}

            return self._error_result(
                RobotErrorCode.INVALID_STATE,
                f"Unexpected place phase: {self.robot_state.value}",
            )

    def abort_current_action(
        self,
        message: str,
        *,
        code: RobotErrorCode = RobotErrorCode.CONNECTION_LOST,
        ) -> dict[str, object]:
        with self._lock:
            self._clear_motion_locked()
            self.pending_pick_slot_id = None
            self.sucker_on = False
            self.robot_xy = self.home_pose[:2]
            self.robot_z = float(self.config.robot_carry_z)
            self.robot_cyl = cartesian_to_cylindrical(self.robot_xy[0], self.robot_xy[1], self.robot_z)
            self.carrying_target_id = None
            self.robot_state = RobotExecutorState.ERROR
            self.last_error_code = code.value
            self.last_error = str(message)
            self._bump_revision()
            return self._snapshot_locked()

    def reset_error(self) -> dict[str, object]:
        with self._lock:
            if self.robot_state != RobotExecutorState.ERROR:
                return self._snapshot_locked()
            self._clear_motion_locked()
            self.pending_pick_slot_id = None
            self.sucker_on = False
            self.robot_xy = self.home_pose[:2]
            self.robot_z = float(self.config.robot_carry_z)
            self.robot_cyl = cartesian_to_cylindrical(self.robot_xy[0], self.robot_xy[1], self.robot_z)
            self.last_error = None
            self.last_error_code = None
            self.control_kernel = "cylindrical_kernel"
            self.robot_state = RobotExecutorState.CARRY_READY if self.carrying_target_id is not None else RobotExecutorState.IDLE
            self._bump_revision()
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, object]:
        return {
            "revision": self._revision,
            "scenario_name": self.scenario_name,
            "robot_xy": self.robot_xy,
            "robot_z": self.robot_z,
            "robot_cyl": {
                "theta_deg": self.robot_cyl[0],
                "radius_mm": self.robot_cyl[1],
                "z_mm": self.robot_cyl[2],
            },
            "home_pose": self.home_pose,
            "limits_x": self.config.robot_limits_x,
            "limits_y": self.config.robot_limits_y,
            "limits_cyl": {
                "theta_deg": self.config.robot_theta_limits_deg,
                "radius_mm": nearest_profile_radius_limits(self.auto_z_profile),
                "z_mm": self.config.robot_height_limits_mm,
            },
            "travel_z": self.config.robot_travel_z,
            "approach_z": self.config.robot_approach_z,
            "pick_z": self.config.robot_pick_z,
            "carry_z": self.config.robot_carry_z,
            "auto_z_enabled": True,
            "auto_z_current": interpolate_auto_z(self.auto_z_profile, self.robot_cyl[1]),
            "ik_valid": True,
            "validation_error": None,
            "motion_target_xy": self.motion_target_xy,
            "motion_target_cyl": self.motion_target_cyl,
            "motion_duration_sec": self.motion_duration_sec,
            "busy": is_busy_state(self.robot_state),
            "busy_action": self._busy_action_name(),
            "state": self.robot_state.value,
            "action_phase": self.robot_state.value,
            "carrying": self.carrying_target_id is not None,
            "control_kernel": self.control_kernel,
            "calibration_ready": True,
            "sucker_on": self.sucker_on,
            "carrying_target_id": self.carrying_target_id,
            "pending_pick_slot_id": self.pending_pick_slot_id,
            "last_ack": self.last_ack,
            "last_error_code": self.last_error_code,
            "last_error": self.last_error,
            "pick_slots": [slot.to_dict() for slot in self.pick_slots.values()],
            "place_slots": [slot.to_dict() for slot in self.place_slots.values()],
        }

    def _advance_motion_locked(self) -> None:
        if self.motion_target_xy is None or self.motion_started_xy is None or self.motion_started_at is None:
            return
        if self.motion_duration_sec <= 0:
            self.robot_xy = self.motion_target_xy
            if self.motion_target_cyl is not None:
                self.robot_cyl = self.motion_target_cyl
            self._clear_motion_locked()
            if self.robot_state == RobotExecutorState.MOVING_XY:
                self.robot_state = RobotExecutorState.CARRY_READY if self.carrying_target_id is not None else RobotExecutorState.IDLE
                self.last_ack = "MOVE"
            return
        elapsed = time.monotonic() - self.motion_started_at
        if elapsed >= self.motion_duration_sec:
            self.robot_xy = self.motion_target_xy
            if self.motion_target_cyl is not None:
                self.robot_cyl = self.motion_target_cyl
            self._clear_motion_locked()
            if self.robot_state == RobotExecutorState.MOVING_XY:
                self.robot_state = RobotExecutorState.CARRY_READY if self.carrying_target_id is not None else RobotExecutorState.IDLE
                self.last_ack = "MOVE"
            return
        ratio = max(0.0, min(1.0, elapsed / self.motion_duration_sec))
        sx, sy = self.motion_started_xy
        tx, ty = self.motion_target_xy
        self.robot_xy = (sx + (tx - sx) * ratio, sy + (ty - sy) * ratio)
        self.robot_cyl = cartesian_to_cylindrical(self.robot_xy[0], self.robot_xy[1], self.robot_z)

    def _plan_motion_locked(self, target_xy: tuple[float, float], *, state: RobotExecutorState) -> float:
        cur_x, cur_y = self.robot_xy
        tx, ty = target_xy
        dist = math.hypot(tx - cur_x, ty - cur_y)
        duration = max(0.04, dist / float(self.config.robot_move_speed_mm_s))
        if self.config.timing_profile == "fast":
            duration = max(0.01, duration * 0.25)
        self.motion_started_xy = (cur_x, cur_y)
        self.motion_target_xy = (tx, ty)
        self.motion_target_cyl = cartesian_to_cylindrical(tx, ty, self.robot_z)
        self.motion_started_at = time.monotonic()
        self.motion_duration_sec = duration
        self.robot_state = state
        return duration

    def _phase_delay(self, formal_sec: float) -> float:
        if self.config.timing_profile == "fast":
            return max(0.05, float(formal_sec) * 0.25)
        return float(formal_sec)

    def _enter_error_locked(
        self,
        code: RobotErrorCode,
        message: str,
        *,
        preserve_carrying: bool,
        disable_sucker: bool,
    ) -> dict[str, object]:
        self._clear_motion_locked()
        self.pending_pick_slot_id = None
        self.robot_xy = self.home_pose[:2]
        self.robot_z = float(self.config.robot_carry_z)
        if disable_sucker:
            self.sucker_on = False
        if not preserve_carrying:
            self.carrying_target_id = None
        self.robot_state = RobotExecutorState.ERROR
        self.last_error_code = code.value
        self.last_error = str(message)
        self._bump_revision()
        return {"status": "error", "code": code.value, "message": self.last_error}

    def _busy_result(self) -> dict[str, object]:
        return {
            "status": "busy",
            "code": RobotErrorCode.BUSY.value,
            "message": f"Robot is busy in state {self.robot_state.value}.",
        }

    def _error_result(self, code: RobotErrorCode, message: str) -> dict[str, object]:
        self.last_error_code = code.value
        self.last_error = str(message)
        self._bump_revision()
        return {"status": "error", "code": code.value, "message": self.last_error}

    def _clear_motion_locked(self) -> None:
        self.motion_target_xy = None
        self.motion_target_cyl = None
        self.motion_started_xy = None
        self.motion_started_at = None
        self.motion_duration_sec = 0.0

    def _remaining_motion_delay_locked(self) -> float:
        if self.motion_started_at is None or self.motion_duration_sec <= 0.0:
            return 0.0
        remaining = self.motion_duration_sec - (time.monotonic() - self.motion_started_at)
        return max(0.0, remaining)

    def _project_world_to_pixel(self, world_xy: tuple[float, float]) -> tuple[float, float]:
        x_min, x_max = (float(self.config.robot_limits_x[0]), float(self.config.robot_limits_x[1]))
        y_min, y_max = (float(self.config.robot_limits_y[0]), float(self.config.robot_limits_y[1]))
        roi_x, roi_y = (float(self.config.roi_center[0]), float(self.config.roi_center[1]))
        roi_radius = float(self.config.roi_radius)
        x_span = max(1.0, x_max - x_min)
        y_span = max(1.0, y_max - y_min)
        x_ratio = (float(world_xy[0]) - x_min) / x_span
        y_ratio = (float(world_xy[1]) - y_min) / y_span
        pixel_x = roi_x + (x_ratio - 0.5) * roi_radius * 1.8
        pixel_y = roi_y + (0.5 - y_ratio) * roi_radius * 1.3
        return (round(pixel_x, 2), round(pixel_y, 2))

    def _busy_action_name(self) -> str | None:
        state = self.robot_state
        if state in {
            RobotExecutorState.PICK_APPROACH,
            RobotExecutorState.PICK_SUCTION_ON,
            RobotExecutorState.PICK_DESCEND,
            RobotExecutorState.PICK_LIFT,
        }:
            return "pick"
        if state in {
            RobotExecutorState.PLACE_DESCEND,
            RobotExecutorState.PLACE_RELEASE,
            RobotExecutorState.PLACE_LIFT,
        }:
            return "place"
        if state == RobotExecutorState.MOVING_XY:
            return "move"
        if state == RobotExecutorState.RECOVERING:
            return "recover"
        return None

    @staticmethod
    def _clamp(value: float, bounds: tuple[float, float]) -> float:
        return max(float(bounds[0]), min(float(bounds[1]), float(value)))

    def _bump_revision(self) -> None:
        self._revision += 1

    def _pick_slot_source(self) -> str:
        if self.config.vision_mode in {"fixed_world_slots", "fixed_cyl_slots"}:
            return "hardware"
        return "sim"

    def _validate_cylindrical_target(self, theta_deg: float, radius_mm: float, z_mm: float) -> dict[str, object]:
        if not within_limits(theta_deg, self.config.robot_theta_limits_deg):
            return {"ok": False, "message": f"Theta {float(theta_deg):.2f} deg is outside limits."}
        if not within_limits(radius_mm, self.config.robot_radius_limits_mm):
            return {"ok": False, "message": f"Radius {float(radius_mm):.2f} mm is outside limits."}
        if not within_limits(z_mm, self.config.robot_height_limits_mm):
            return {"ok": False, "message": f"Height {float(z_mm):.2f} mm is outside limits."}
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
        return {"ok": True, "message": None, "margin": float(margin), "neutral_distance": abs(float(z_mm) - self.config.robot_auto_z_preferred_mm)}
