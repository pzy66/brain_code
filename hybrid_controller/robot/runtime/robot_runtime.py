from __future__ import annotations

import argparse
import json
import math
import socketserver
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Protocol, TextIO

from hybrid_controller.cylindrical import (
    CylindricalPose,
    build_auto_z_profile,
    cartesian_to_cylindrical,
    cylindrical_to_cartesian,
    interpolate_auto_z,
    nearest_profile_radius_limits,
    within_limits,
)

from .robot_protocol import (
    RobotCommandError,
    RobotErrorCode,
    RobotExecutorState,
    format_error_line,
    is_busy_state,
)
from .runtime_core import CommandParseError, parse_command_text

JETMAX_L4_MM = 16.8
SERVO1_MIN_PULSE = 0
SERVO1_MAX_PULSE = 1000
SERVO2_MAX_PULSE = 700
SERVO2_MIN_PULSE = 0
SERVO3_MIN_PULSE = 470
SERVO3_MAX_PULSE = 1000
MIN_RADIUS_MM = 50.0
MAX_Z_MM = 225.0


class HardwareProtocol(Protocol):
    def go_home(self) -> None: ...

    def get_position(self) -> tuple[float, float, float]: ...

    def set_position(self, position: tuple[float, float, float], duration: float) -> None: ...

    def set_position_relatively(self, delta: tuple[float, float, float], duration: float) -> None: ...

    def set_sucker(self, state: bool) -> None: ...


class CalibrationProtocol(Protocol):
    def is_ready(self) -> bool: ...

    def readiness_code(self) -> str | None: ...

    def readiness_message(self) -> str | None: ...

    def camera_to_world(self, pixel_x: float, pixel_y: float) -> tuple[float, float, float]: ...


@dataclass(frozen=True)
class RobotLimits:
    x_min: float = -140.0
    x_max: float = 140.0
    y_min: float = -200.0
    y_max: float = -40.0
    theta_min_deg: float = -120.0
    theta_max_deg: float = 120.0
    radius_min_mm: float = 50.0
    radius_max_mm: float = 280.0
    auto_radius_min_mm: float = 80.0
    auto_radius_max_mm: float = 260.0
    z_min_mm: float = 80.0
    z_max_mm: float = 212.8
    cylindrical_xy_workspace_enabled: bool = False
    z_approach: float = 130.0
    z_pick: float = 85.0
    z_carry: float = 160.0
    move_speed_xy_mm_s: float = 150.0
    target_margin_mm: float = 15.0
    auto_z_profile_radius_step_mm: float = 5.0
    auto_z_profile_height_step_mm: float = 5.0
    auto_z_preferred_mm: float = 160.0
    auto_z_plateau_min_radius_mm: float = 145.0
    auto_z_plateau_max_radius_mm: float = 185.0
    auto_z_plateau_z_mm: float = 205.0
    auto_z_retract_drop_per_radius_mm: float = 0.8
    auto_z_extend_drop_per_radius_mm: float = 0.4
    auto_z_posture_tolerance_deg: float = 8.0
    auto_z_down_per_radius_mm: float = 0.5
    auto_z_up_per_radius_mm: float = 1.0
    auto_z_min_delta_mm: float = 3.0
    motion_min_duration_sec: float = 0.25
    motion_settle_sec: float = 0.08
    teleop_min_duration_sec: float = 0.12
    teleop_settle_sec: float = 0.02


@dataclass(frozen=True)
class PickTuning:
    pick_approach_z_mm: float
    pick_descend_z_mm: float
    pick_pre_suction_sec: float
    pick_bottom_hold_sec: float
    pick_lift_sec: float
    place_descend_z_mm: float
    place_release_mode: str
    place_release_sec: float
    place_post_release_hold_sec: float
    z_carry_floor_mm: float

    def to_dict(self) -> dict[str, object]:
        return {
            "pick_approach_z_mm": float(self.pick_approach_z_mm),
            "pick_descend_z_mm": float(self.pick_descend_z_mm),
            "pick_pre_suction_sec": float(self.pick_pre_suction_sec),
            "pick_bottom_hold_sec": float(self.pick_bottom_hold_sec),
            "pick_lift_sec": float(self.pick_lift_sec),
            "place_descend_z_mm": float(self.place_descend_z_mm),
            "place_release_mode": str(self.place_release_mode),
            "place_release_sec": float(self.place_release_sec),
            "place_post_release_hold_sec": float(self.place_post_release_hold_sec),
            "z_carry_floor_mm": float(self.z_carry_floor_mm),
        }


@dataclass(frozen=True)
class PickPlan:
    pixel_x: float
    pixel_y: float
    offset_x: float
    offset_y: float
    raw_target_x: float
    raw_target_y: float
    target_x: float
    target_y: float


@dataclass(frozen=True)
class WorldPickPlan:
    target_x: float
    target_y: float


@dataclass(frozen=True)
class CylindricalMovePlan:
    theta_deg: float
    radius_mm: float
    z_mm: float
    x_mm: float
    y_mm: float


@dataclass(frozen=True)
class CylindricalPickPlan:
    theta_deg: float
    radius_mm: float
    target_x: float
    target_y: float


@dataclass(frozen=True)
class PlacePlan:
    target_x: float
    target_y: float


class ActuatorAdapter:
    def __init__(
        self,
        hardware: HardwareProtocol,
        *,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._hardware = hardware
        self._log = log or (lambda message: None)

    def go_home(self) -> None:
        self._log("ActuatorAdapter.go_home()")
        self._hardware.go_home()

    def get_position(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._hardware.get_position())

    def move_xyz(self, x: float, y: float, z: float, duration: float) -> None:
        self._hardware.set_position((float(x), float(y), float(z)), float(duration))

    def move_relative_xyz(self, dx: float, dy: float, dz: float, duration: float) -> None:
        relative_move = getattr(self._hardware, "set_position_relatively", None)
        if callable(relative_move):
            relative_move((float(dx), float(dy), float(dz)), float(duration))
            return
        cur_x, cur_y, cur_z = self.get_position()
        self.move_xyz(cur_x + float(dx), cur_y + float(dy), cur_z + float(dz), duration)

    def set_sucker(self, state: bool) -> None:
        self._hardware.set_sucker(bool(state))

    def release_sucker(self, duration_sec: float) -> bool:
        release = getattr(self._hardware, "release_sucker", None)
        if not callable(release):
            return False
        release(float(duration_sec))
        return True


class LegacyCartesianKernel:
    name = "legacy_cartesian"

    def __init__(self, executor: "RobotExecutor") -> None:
        self._executor = executor

    def move_xy(self, x: float, y: float) -> tuple[float, float, float]:
        self._executor._mark_control_kernel(self.name)
        return self._executor._move_xy_legacy_impl(x, y)

    def begin_pick(self, pixel_x: float, pixel_y: float) -> PickPlan:
        self._executor._mark_control_kernel(self.name)
        return self._executor._begin_pick_pixel_legacy_impl(pixel_x, pixel_y)

    def begin_pick_world(self, x: float, y: float) -> WorldPickPlan:
        self._executor._mark_control_kernel(self.name)
        return self._executor._begin_pick_world_legacy_impl(x, y)


class CylindricalKernel:
    name = "cylindrical_kernel"

    def __init__(self, executor: "RobotExecutor") -> None:
        self._executor = executor

    def current_pose(self) -> CylindricalPose:
        position = self._executor.actuator.get_position()
        return CylindricalPose.from_cartesian(*position)

    def move_cyl(self, theta_deg: float, radius_mm: float, z_mm: float) -> CylindricalMovePlan:
        self._executor._mark_control_kernel(self.name)
        pose = CylindricalPose(theta_deg=theta_deg, radius_mm=radius_mm, z_mm=z_mm).normalized()
        return self._executor._move_cyl_impl(pose)

    def move_cyl_auto(self, theta_deg: float, radius_mm: float) -> CylindricalMovePlan:
        self._executor._mark_control_kernel(self.name)
        return self._executor._move_cyl_auto_impl(float(theta_deg), float(radius_mm))

    def begin_pick_cyl(self, theta_deg: float, radius_mm: float) -> CylindricalPickPlan:
        self._executor._mark_control_kernel(self.name)
        return self._executor._begin_pick_cyl_impl(float(theta_deg), float(radius_mm))


class CalibrationProvider:
    def __init__(self) -> None:
        self._cv2 = None
        self._np = None
        self._ready = False
        self._error_code: str | None = RobotErrorCode.CALIBRATION_UNAVAILABLE.value
        self._error_message: str | None = "Calibration parameters are unavailable."
        self._K = None
        self._R = None
        self._T = None

        try:
            import cv2
            import numpy as np
            import rospy
        except Exception as error:  # pragma: no cover - robot-only branch
            self._error_message = f"Calibration dependencies unavailable: {error}"
            return

        self._cv2 = cv2
        self._np = np
        try:
            params = rospy.get_param("/camera_cal/block_params", None)
        except Exception as error:
            self._error_message = f"Failed to read /camera_cal/block_params: {error}"
            return
        if not params:
            self._error_message = "Missing /camera_cal/block_params."
            return

        try:
            self._K = np.array(params["K"], dtype=np.float64).reshape(3, 3)
            self._R = np.array(params["R"], dtype=np.float64).reshape(3, 1)
            self._T = np.array(params["T"], dtype=np.float64).reshape(3, 1)
        except Exception as error:
            self._error_code = RobotErrorCode.CALIBRATION_INVALID.value
            self._error_message = f"Malformed calibration params: {error}"
            return

        if not self._validate_loaded_params():
            return
        self._ready = True
        self._error_code = None
        self._error_message = None

    def is_ready(self) -> bool:
        return self._ready

    def readiness_code(self) -> str | None:
        return self._error_code

    def readiness_message(self) -> str | None:
        return self._error_message

    def camera_to_world(self, pixel_x: float, pixel_y: float) -> tuple[float, float, float]:
        if not self._ready or self._np is None or self._cv2 is None:
            raise RobotCommandError(
                self._error_code or RobotErrorCode.CALIBRATION_UNAVAILABLE,
                self._error_message or "Calibration is not ready.",
            )

        np = self._np
        inv_k = np.asmatrix(self._K).I
        r_mat = np.zeros((3, 3), dtype=np.float64)
        self._cv2.Rodrigues(self._R, r_mat)
        inv_r = np.asmatrix(r_mat).I
        trans_plane_to_cam = np.dot(inv_r, np.asmatrix(self._T))

        coords = np.zeros((3, 1), dtype=np.float64)
        coords[0][0] = pixel_x
        coords[1][0] = pixel_y
        coords[2][0] = 1.0

        world_pt_cam = np.dot(inv_k, coords)
        world_pt_plane = np.dot(inv_r, world_pt_cam)
        scale = trans_plane_to_cam[2][0] / world_pt_plane[2][0]
        scaled = np.multiply(scale, world_pt_plane)
        reprojection = np.asmatrix(scaled) - np.asmatrix(trans_plane_to_cam)
        x, y, z = reprojection.T.tolist()[0]
        values = (float(x), float(y), float(z))
        if not all(math.isfinite(value) for value in values):
            raise RobotCommandError(RobotErrorCode.CALIBRATION_INVALID, "camera_to_world produced non-finite values.")
        return values

    def _validate_loaded_params(self) -> bool:
        np = self._np
        cv2 = self._cv2
        try:
            matrices = (self._K, self._R, self._T)
            if not all(matrix is not None for matrix in matrices):
                raise ValueError("K/R/T must all be present.")
            if not np.isfinite(self._K).all() or not np.isfinite(self._R).all() or not np.isfinite(self._T).all():
                raise ValueError("K/R/T contain non-finite values.")
            np.linalg.inv(self._K)
            r_mat = np.zeros((3, 3), dtype=np.float64)
            cv2.Rodrigues(self._R, r_mat)
            np.linalg.inv(r_mat)
        except Exception as error:
            self._error_code = RobotErrorCode.CALIBRATION_INVALID.value
            self._error_message = f"Invalid calibration params: {error}"
            return False
        return True


class _UnavailableCalibrationProvider:
    def __init__(self, message: str = "Calibration provider unavailable.") -> None:
        self._message = str(message)

    def is_ready(self) -> bool:
        return False

    def readiness_code(self) -> str | None:
        return RobotErrorCode.CALIBRATION_UNAVAILABLE.value

    def readiness_message(self) -> str | None:
        return self._message

    def camera_to_world(self, pixel_x: float, pixel_y: float) -> tuple[float, float, float]:
        raise RobotCommandError(RobotErrorCode.CALIBRATION_UNAVAILABLE, self._message)


class _HardwareBridge:
    def __init__(self) -> None:
        try:
            import hiwonder
            import rospy
        except Exception as error:  # pragma: no cover - hardware-only branch
            raise RuntimeError(
                "Robot runtime requires hiwonder and rospy in the robot environment."
            ) from error

        self._rospy = rospy
        self.jetmax = hiwonder.JetMax()
        self.sucker = hiwonder.Sucker()
        rospy.init_node("hybrid_controller_robot_runtime", anonymous=True, disable_signals=True)

    def go_home(self) -> None:  # pragma: no cover - hardware-only branch
        self.jetmax.go_home(1.5)
        time.sleep(1.5)

    def get_position(self) -> tuple[float, float, float]:  # pragma: no cover - hardware-only branch
        return tuple(float(value) for value in self.jetmax.position)

    def set_position(self, position: tuple[float, float, float], duration: float) -> None:  # pragma: no cover
        self.jetmax.set_position(position, duration)

    def set_position_relatively(self, delta: tuple[float, float, float], duration: float) -> None:  # pragma: no cover
        relative_move = getattr(self.jetmax, "set_position_relatively", None)
        if callable(relative_move):
            relative_move(delta, duration)
            return
        current = tuple(float(value) for value in self.jetmax.position)
        self.jetmax.set_position(
            (current[0] + float(delta[0]), current[1] + float(delta[1]), current[2] + float(delta[2])),
            duration,
        )

    def set_sucker(self, state: bool) -> None:  # pragma: no cover - hardware-only branch
        self.sucker.set_state(bool(state))

    def release_sucker(self, duration_sec: float) -> None:  # pragma: no cover - hardware-only branch
        release = getattr(self.sucker, "release", None)
        if callable(release):
            release(float(duration_sec))
            return
        self.sucker.set_state(False)


class RobotExecutor:
    def __init__(
        self,
        hardware: HardwareProtocol,
        calibration: CalibrationProtocol,
        *,
        limits: RobotLimits | None = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.hardware = hardware
        self.actuator = ActuatorAdapter(hardware, log=log)
        self.calibration = calibration
        self.limits = limits or RobotLimits()
        self._log = log or (lambda message: None)
        self._lock = threading.RLock()
        self._state = RobotExecutorState.IDLE
        self._carrying = False
        self._revision = 0
        self._last_error_code: str | None = None
        self._last_error_message: str | None = None
        self._control_kernel = CylindricalKernel.name
        self._home_pose = tuple(float(value) for value in self.actuator.get_position())
        self._home_cyl = cartesian_to_cylindrical(*self._home_pose)
        self._abort_requested = threading.Event()
        self._reference_pulses: tuple[float, float, float] | None = None
        self._reference_forearm_pitch_deg: float | None = None
        self._auto_z_profile = self._build_auto_z_profile()
        self._pick_tuning = self._default_pick_tuning()
        self._last_post_pick_settle_z = float(self._pick_tuning.z_carry_floor_mm)
        self._last_release_mode_effective = "off"
        self.legacy_kernel = LegacyCartesianKernel(self)
        self.cylindrical_kernel = CylindricalKernel(self)

    @property
    def state(self) -> RobotExecutorState:
        with self._lock:
            return self._state

    @property
    def carrying(self) -> bool:
        with self._lock:
            return self._carrying

    def is_busy(self) -> bool:
        with self._lock:
            return is_busy_state(self._state)

    def healthcheck(self) -> dict[str, object]:
        with self._lock:
            position = tuple(float(value) for value in self.actuator.get_position())
            theta_deg, radius_mm, z_mm = cartesian_to_cylindrical(*position)
            validation = self._validate_pose_xyz(
                *position,
                enforce_workspace=self.limits.cylindrical_xy_workspace_enabled,
            )
            effective_radius_limits = nearest_profile_radius_limits(self._auto_z_profile)
            pick_tuning = self._pick_tuning
            return {
                "revision": self._revision,
                "state": self._state.value,
                "action_phase": self._state.value,
                "busy_action": self._busy_action_name(),
                "busy": is_busy_state(self._state),
                "carrying": self._carrying,
                "control_kernel": self._control_kernel,
                "position_xyz": position,
                "robot_xy": position[:2],
                "robot_z": position[2],
                "robot_cyl": {
                    "theta_deg": round(theta_deg, 3),
                    "radius_mm": round(radius_mm, 3),
                    "z_mm": round(z_mm, 3),
                },
                "home_pose": self._home_pose,
                "home_cyl": {
                    "theta_deg": round(self._home_cyl[0], 3),
                    "radius_mm": round(self._home_cyl[1], 3),
                    "z_mm": round(self._home_cyl[2], 3),
                },
                "limits_x": (self.limits.x_min, self.limits.x_max),
                "limits_y": (self.limits.y_min, self.limits.y_max),
                "limits_cyl": {
                    "theta_deg": (self.limits.theta_min_deg, self.limits.theta_max_deg),
                    "radius_mm": effective_radius_limits,
                    "z_mm": (self.limits.z_min_mm, self.limits.z_max_mm),
                },
                "limits_cyl_auto": {
                    "theta_deg": (self.limits.theta_min_deg, self.limits.theta_max_deg),
                    "radius_mm": (self.limits.auto_radius_min_mm, self.limits.auto_radius_max_mm),
                    "z_mm": (self.limits.z_min_mm, self.limits.z_max_mm),
                },
                "cylindrical_xy_workspace_enabled": self.limits.cylindrical_xy_workspace_enabled,
                "travel_z": pick_tuning.pick_approach_z_mm,
                "approach_z": pick_tuning.pick_approach_z_mm,
                "pick_z": pick_tuning.pick_descend_z_mm,
                "carry_z": self.limits.z_carry,
                "auto_z_enabled": True,
                "auto_z_current": round(interpolate_auto_z(self._auto_z_profile, radius_mm), 3),
                "ik_valid": bool(validation.get("ok", False)),
                "validation_error": validation.get("message"),
                "last_error_code": self._last_error_code,
                "last_error": self._last_error_message,
                "last_error_message": self._last_error_message,
                "calibration_ready": self.calibration.is_ready(),
                "pick_slots": [],
                "place_slots": [],
                "pick_tuning": pick_tuning.to_dict(),
                "post_pick_settle_z": float(self._last_post_pick_settle_z),
                "release_mode_effective": str(self._last_release_mode_effective),
            }

    def get_pick_tuning(self) -> dict[str, object]:
        with self._lock:
            return self._pick_tuning.to_dict()

    def set_pick_tuning(self, payload: dict[str, object]) -> dict[str, object]:
        with self._lock:
            self._pick_tuning = self._normalize_pick_tuning(payload)
            self._revision += 1
            return self._pick_tuning.to_dict()

    def move_xy(self, x: float, y: float) -> tuple[float, float, float]:
        return self.legacy_kernel.move_xy(x, y)

    def move_cyl(self, theta_deg: float, radius_mm: float, z_mm: float) -> CylindricalMovePlan:
        return self.cylindrical_kernel.move_cyl(theta_deg, radius_mm, z_mm)

    def move_cyl_auto(self, theta_deg: float, radius_mm: float) -> CylindricalMovePlan:
        return self.cylindrical_kernel.move_cyl_auto(theta_deg, radius_mm)

    def begin_pick(self, pixel_x: float, pixel_y: float) -> PickPlan:
        return self.legacy_kernel.begin_pick(pixel_x, pixel_y)

    def begin_pick_world(self, x: float, y: float) -> WorldPickPlan:
        return self.legacy_kernel.begin_pick_world(x, y)

    def begin_pick_cyl(self, theta_deg: float, radius_mm: float) -> CylindricalPickPlan:
        return self.cylindrical_kernel.begin_pick_cyl(theta_deg, radius_mm)

    def _move_xy_legacy_impl(self, x: float, y: float) -> tuple[float, float, float]:
        with self._lock:
            self._ensure_operable_for_motion()
            self._clear_last_error()
            self._abort_requested.clear()
            tx = self._clamp(x, self.limits.x_min, self.limits.x_max)
            ty = self._clamp(y, self.limits.y_min, self.limits.y_max)
            self._set_state(RobotExecutorState.MOVING_XY)
            current = self.actuator.get_position()

        try:
            self._move_to_carry_height(current)
            cx, cy, _ = self.actuator.get_position()
            duration = max(self.limits.motion_min_duration_sec, math.hypot(tx - cx, ty - cy) / self.limits.move_speed_xy_mm_s)
            self._log(f"MOVE -> ({tx:.2f}, {ty:.2f}, {self.limits.z_carry:.2f}) in {duration:.2f}s")
            self.actuator.move_xyz(tx, ty, self.limits.z_carry, duration)
            self._sleep_with_abort(duration + self.limits.motion_settle_sec)
        except Exception as error:
            with self._lock:
                self._last_error_code = RobotErrorCode.HARDWARE_FAILURE.value
                self._last_error_message = str(error)
                self._set_state(RobotExecutorState.ERROR)
            raise

        with self._lock:
            self._set_state(RobotExecutorState.CARRY_READY if self._carrying else RobotExecutorState.IDLE)
            return (tx, ty, self.limits.z_carry)

    def _move_cyl_impl(self, pose: CylindricalPose) -> CylindricalMovePlan:
        with self._lock:
            self._ensure_operable_for_motion()
            self._clear_last_error()
            self._abort_requested.clear()
            plan = self._build_cylindrical_move_plan(pose)
            self._set_state(RobotExecutorState.MOVING_XY)
            current = self.actuator.get_position()

        try:
            self._move_to_safe_pose(
                current,
                plan,
                min_duration=self.limits.motion_min_duration_sec,
                settle_sec=self.limits.motion_settle_sec,
            )
        except Exception as error:
            with self._lock:
                self._last_error_code = RobotErrorCode.HARDWARE_FAILURE.value
                self._last_error_message = str(error)
                self._set_state(RobotExecutorState.ERROR)
            raise

        with self._lock:
            self._set_state(RobotExecutorState.CARRY_READY if self._carrying else RobotExecutorState.IDLE)
            return plan

    def _move_cyl_auto_impl(self, theta_deg: float, radius_mm: float) -> CylindricalMovePlan:
        profile_radius_limits = nearest_profile_radius_limits(self._auto_z_profile)
        effective_radius_limits = (
            max(profile_radius_limits[0], self.limits.auto_radius_min_mm),
            min(profile_radius_limits[1], self.limits.auto_radius_max_mm),
        )
        if not within_limits(radius_mm, effective_radius_limits):
            raise RobotCommandError(
                RobotErrorCode.TARGET_OUT_OF_WORKSPACE,
                f"Radius {float(radius_mm):.2f} mm is outside auto-z profile range {effective_radius_limits}.",
            )
        current_theta_deg, current_radius_mm, current_z_mm = cartesian_to_cylindrical(*self.actuator.get_position())
        _ = current_theta_deg
        desired_z_mm = interpolate_auto_z(self._auto_z_profile, radius_mm)
        delta_radius_mm = abs(float(radius_mm) - float(current_radius_mm))
        max_down_step_mm = max(
            float(self.limits.auto_z_min_delta_mm),
            delta_radius_mm * float(self.limits.auto_z_down_per_radius_mm),
        )
        max_up_step_mm = max(
            float(self.limits.auto_z_min_delta_mm),
            delta_radius_mm * float(self.limits.auto_z_up_per_radius_mm),
        )
        if desired_z_mm < float(current_z_mm):
            z_mm = max(desired_z_mm, float(current_z_mm) - max_down_step_mm)
        else:
            z_mm = min(desired_z_mm, float(current_z_mm) + max_up_step_mm)
        with self._lock:
            self._ensure_operable_for_motion()
            self._clear_last_error()
            self._abort_requested.clear()
            plan = self._build_cylindrical_move_plan(
                CylindricalPose(theta_deg=theta_deg, radius_mm=radius_mm, z_mm=z_mm).normalized()
            )
            self._set_state(RobotExecutorState.MOVING_XY)
            current = self.actuator.get_position()

        try:
            self._move_to_safe_pose(
                current,
                plan,
                min_duration=self.limits.teleop_min_duration_sec,
                settle_sec=self.limits.teleop_settle_sec,
            )
        except Exception as error:
            with self._lock:
                self._last_error_code = RobotErrorCode.HARDWARE_FAILURE.value
                self._last_error_message = str(error)
                self._set_state(RobotExecutorState.ERROR)
            raise

        with self._lock:
            self._set_state(RobotExecutorState.CARRY_READY if self._carrying else RobotExecutorState.IDLE)
            return plan

    def _begin_pick_pixel_legacy_impl(self, pixel_x: float, pixel_y: float) -> PickPlan:
        with self._lock:
            self._ensure_operable_for_action(action="pick")
            if self._carrying:
                raise RobotCommandError(RobotErrorCode.INVALID_STATE, "Cannot pick while already carrying a target.")
            self._clear_last_error()
            self._abort_requested.clear()

        if not self.calibration.is_ready():
            raise RobotCommandError(
                self.calibration.readiness_code() or RobotErrorCode.CALIBRATION_UNAVAILABLE,
                self.calibration.readiness_message() or "Calibration is not ready.",
            )

        offset_x, offset_y, _ = self.calibration.camera_to_world(pixel_x, pixel_y)
        cur_x, cur_y, _ = self.actuator.get_position()
        raw_target_x = cur_x + offset_x
        raw_target_y = cur_y + offset_y
        target_x = self._clamp(raw_target_x, self.limits.x_min, self.limits.x_max)
        target_y = self._clamp(raw_target_y, self.limits.y_min, self.limits.y_max)
        overflow = max(abs(raw_target_x - target_x), abs(raw_target_y - target_y))
        if overflow > self.limits.target_margin_mm:
            raise RobotCommandError(
                RobotErrorCode.TARGET_OUT_OF_WORKSPACE,
                f"Target exceeds workspace by {overflow:.2f} mm.",
            )

        plan = PickPlan(
            pixel_x=float(pixel_x),
            pixel_y=float(pixel_y),
            offset_x=float(offset_x),
            offset_y=float(offset_y),
            raw_target_x=float(raw_target_x),
            raw_target_y=float(raw_target_y),
            target_x=float(target_x),
            target_y=float(target_y),
        )
        with self._lock:
            self._set_state(RobotExecutorState.PICK_APPROACH)
        return plan

    def _begin_pick_world_legacy_impl(self, x: float, y: float) -> WorldPickPlan:
        with self._lock:
            self._ensure_operable_for_action(action="pick")
            if self._carrying:
                raise RobotCommandError(RobotErrorCode.INVALID_STATE, "Cannot pick while already carrying a target.")
            self._clear_last_error()
            self._abort_requested.clear()
            target_x = float(x)
            target_y = float(y)
            theta_deg, radius_mm, _ = cartesian_to_cylindrical(target_x, target_y, self.limits.z_pick)
            cyl_validation = self._validate_cylindrical_target(theta_deg, radius_mm, self.limits.z_pick)
            if not cyl_validation.get("ok", False):
                raise RobotCommandError(
                    RobotErrorCode.TARGET_OUT_OF_WORKSPACE,
                    str(cyl_validation.get("message") or "World target violates cylindrical limits."),
                )
            self._set_state(RobotExecutorState.PICK_APPROACH)
            return WorldPickPlan(target_x=target_x, target_y=target_y)

    def _begin_pick_cyl_impl(self, theta_deg: float, radius_mm: float) -> CylindricalPickPlan:
        with self._lock:
            self._ensure_operable_for_action(action="pick")
            if self._carrying:
                raise RobotCommandError(RobotErrorCode.INVALID_STATE, "Cannot pick while already carrying a target.")
            self._clear_last_error()
            self._abort_requested.clear()
            pose = CylindricalPose(theta_deg=theta_deg, radius_mm=radius_mm, z_mm=self.limits.z_pick).normalized()
            validation = self._validate_cylindrical_target(pose.theta_deg, pose.radius_mm, pose.z_mm)
            if not validation.get("ok", False):
                raise RobotCommandError(
                    RobotErrorCode.TARGET_OUT_OF_WORKSPACE,
                    str(validation.get("message") or "Invalid cylindrical pick target."),
                )
            x_mm, y_mm, _ = pose.as_cartesian()
            self._set_state(RobotExecutorState.PICK_APPROACH)
            return CylindricalPickPlan(
                theta_deg=float(pose.theta_deg),
                radius_mm=float(pose.radius_mm),
                target_x=float(x_mm),
                target_y=float(y_mm),
            )

    def complete_pick(self, plan: PickPlan | WorldPickPlan | CylindricalPickPlan) -> None:
        if isinstance(plan, PickPlan):
            self._log(
                f"PICK target pixel=({plan.pixel_x:.2f}, {plan.pixel_y:.2f}) "
                f"offset=({plan.offset_x:.2f}, {plan.offset_y:.2f}) -> ({plan.target_x:.2f}, {plan.target_y:.2f})"
            )
        elif isinstance(plan, CylindricalPickPlan):
            self._log(
                f"PICK_CYL target=(theta={plan.theta_deg:.2f}, r={plan.radius_mm:.2f}) "
                f"-> ({plan.target_x:.2f}, {plan.target_y:.2f})"
            )
        else:
            self._log(f"PICK_WORLD target=({plan.target_x:.2f}, {plan.target_y:.2f})")
        target_x = float(plan.target_x)
        target_y = float(plan.target_y)
        tuning = self._pick_tuning_snapshot()
        settle_z = self._compute_settle_z_for_xy(
            target_x,
            target_y,
            carry_floor_mm=tuning.z_carry_floor_mm,
        )
        self._raise_if_abort_requested()
        self.actuator.move_xyz(target_x, target_y, tuning.pick_approach_z_mm, 1.0)
        self._sleep_with_abort(1.0)

        with self._lock:
            self._set_state(RobotExecutorState.PICK_SUCTION_ON)
        self._raise_if_abort_requested()
        self.actuator.set_sucker(True)
        self._sleep_with_abort(tuning.pick_pre_suction_sec)

        with self._lock:
            self._set_state(RobotExecutorState.PICK_DESCEND)
        self._raise_if_abort_requested()
        descend_duration_sec = 0.8
        self.actuator.move_xyz(target_x, target_y, tuning.pick_descend_z_mm, descend_duration_sec)
        self._sleep_with_abort(descend_duration_sec + tuning.pick_bottom_hold_sec)

        with self._lock:
            self._set_state(RobotExecutorState.PICK_LIFT)
        self._raise_if_abort_requested()
        self.actuator.move_xyz(target_x, target_y, settle_z, tuning.pick_lift_sec)
        self._sleep_with_abort(tuning.pick_lift_sec)

        with self._lock:
            self._carrying = True
            self._last_post_pick_settle_z = float(settle_z)
            self._set_state(RobotExecutorState.CARRY_READY)
            self._clear_last_error()

    def begin_place(self) -> PlacePlan:
        with self._lock:
            self._ensure_operable_for_action(action="place")
            if not self._carrying:
                raise RobotCommandError(RobotErrorCode.INVALID_STATE, "Cannot place without a carried target.")
            self._clear_last_error()
            self._abort_requested.clear()
            cur_x, cur_y, _ = self.actuator.get_position()
            self._set_state(RobotExecutorState.PLACE_DESCEND)
            return PlacePlan(target_x=float(cur_x), target_y=float(cur_y))

    def complete_place(self, plan: PlacePlan) -> None:
        tuning = self._pick_tuning_snapshot()
        settle_z = self._compute_settle_z_for_xy(
            plan.target_x,
            plan.target_y,
            carry_floor_mm=tuning.z_carry_floor_mm,
        )
        current_x, current_y, current_z = self.actuator.get_position()
        if float(current_z) < float(tuning.z_carry_floor_mm):
            self._raise_if_abort_requested()
            self.actuator.move_xyz(current_x, current_y, settle_z, 0.5)
            self._sleep_with_abort(0.5)

        self._raise_if_abort_requested()
        self.actuator.move_xyz(plan.target_x, plan.target_y, tuning.place_descend_z_mm, 0.8)
        self._sleep_with_abort(0.8)

        with self._lock:
            self._set_state(RobotExecutorState.PLACE_RELEASE)
        self._raise_if_abort_requested()
        release_mode = self._apply_place_release(
            mode=tuning.place_release_mode,
            release_sec=tuning.place_release_sec,
        )
        self._sleep_with_abort(tuning.place_post_release_hold_sec)

        with self._lock:
            self._set_state(RobotExecutorState.PLACE_LIFT)
        self._raise_if_abort_requested()
        self.actuator.move_xyz(plan.target_x, plan.target_y, settle_z, 0.6)
        self._sleep_with_abort(0.6)

        with self._lock:
            self._carrying = False
            self._last_release_mode_effective = str(release_mode)
            self._set_state(RobotExecutorState.IDLE)
            self._clear_last_error()

    def abort(
        self,
        *,
        code: RobotErrorCode | str = RobotErrorCode.ABORTED,
        message: str = "Abort requested by operator.",
    ) -> None:
        self._abort_requested.set()
        with self._lock:
            busy = is_busy_state(self._state)
        if busy:
            return
        self._set_abort_error(code, message)

    def reset_error(self) -> None:
        with self._lock:
            if self._state != RobotExecutorState.ERROR:
                return
            self._abort_requested.clear()
        self._recover_to_safe_pose(disable_sucker=True)
        with self._lock:
            self._clear_last_error()
            self._set_state(RobotExecutorState.CARRY_READY if self._carrying else RobotExecutorState.IDLE)

    def handle_pick_failure(self, error: Exception) -> RobotCommandError:
        return self._handle_action_failure(
            error=error,
            default_code=RobotErrorCode.HARDWARE_FAILURE,
            disable_sucker=True,
        )

    def handle_place_failure(self, error: Exception) -> RobotCommandError:
        return self._handle_action_failure(
            error=error,
            default_code=RobotErrorCode.HARDWARE_FAILURE,
            disable_sucker=False,
        )

    def _handle_action_failure(
        self,
        *,
        error: Exception,
        default_code: RobotErrorCode,
        disable_sucker: bool,
    ) -> RobotCommandError:
        if isinstance(error, RobotCommandError):
            command_error = error
        else:
            command_error = RobotCommandError(default_code, str(error))
        with self._lock:
            self._set_state(RobotExecutorState.RECOVERING)
            self._last_error_code = command_error.code
            self._last_error_message = command_error.message
        try:
            self._recover_to_safe_pose(
                disable_sucker=disable_sucker or command_error.code == RobotErrorCode.ABORTED.value
            )
        except Exception as recover_error:
            with self._lock:
                self._set_state(RobotExecutorState.ERROR)
                self._last_error_code = RobotErrorCode.RECOVER_FAILED.value
                self._last_error_message = str(recover_error)
            return RobotCommandError(RobotErrorCode.RECOVER_FAILED, str(recover_error))

        with self._lock:
            self._set_state(RobotExecutorState.ERROR)
        return command_error

    def _recover_to_safe_pose(self, *, disable_sucker: bool) -> None:
        if disable_sucker:
            try:
                self.actuator.set_sucker(False)
            except Exception as error:
                self._log(f"Failed to disable sucker during recovery: {error}")
        try:
            cur_x, cur_y, _ = self.actuator.get_position()
            self.actuator.move_xyz(cur_x, cur_y, self.limits.z_carry, 0.6)
            self._sleep_with_abort(0.6, allow_abort=False)
        except Exception as error:
            self._log(f"Failed to lift robot during recovery: {error}")
            raise
        self.actuator.go_home()
        self._log("Robot returned home after recovery.")

    def _move_to_safe_pose(
        self,
        current_position: tuple[float, float, float],
        plan: CylindricalMovePlan,
        *,
        min_duration: float,
        settle_sec: float,
    ) -> None:
        cur_x, cur_y, cur_z = current_position
        duration_xyz = max(
            float(min_duration),
            math.sqrt(
                (plan.x_mm - cur_x) ** 2
                + (plan.y_mm - cur_y) ** 2
                + (plan.z_mm - cur_z) ** 2
            )
            / self.limits.move_speed_xy_mm_s,
        )
        self._log(
            f"MOVE_CYL(theta={plan.theta_deg:.2f}, r={plan.radius_mm:.2f}, z={plan.z_mm:.2f}) "
            f"-> ({plan.x_mm:.2f}, {plan.y_mm:.2f}, {plan.z_mm:.2f}) in {duration_xyz:.2f}s"
        )
        self.actuator.move_xyz(plan.x_mm, plan.y_mm, plan.z_mm, duration_xyz)
        self._sleep_with_abort(duration_xyz + float(settle_sec))

    def _move_to_carry_height(self, current_position: tuple[float, float, float]) -> None:
        cur_x, cur_y, cur_z = current_position
        if abs(cur_z - self.limits.z_carry) <= 1e-6:
            return
        self.actuator.move_xyz(cur_x, cur_y, self.limits.z_carry, 0.6)
        self._sleep_with_abort(0.6, allow_abort=False)

    def _pick_tuning_snapshot(self) -> PickTuning:
        with self._lock:
            return self._pick_tuning

    def _compute_settle_z_for_xy(self, x_mm: float, y_mm: float, *, carry_floor_mm: float) -> float:
        _, radius_mm, _ = cartesian_to_cylindrical(float(x_mm), float(y_mm), float(self.limits.z_pick))
        auto_z = float(interpolate_auto_z(self._auto_z_profile, float(radius_mm)))
        settle_z = max(float(carry_floor_mm), auto_z)
        settle_z = self._clamp(settle_z, float(self.limits.z_min_mm), float(self.limits.z_max_mm))
        return float(settle_z)

    def _apply_place_release(self, *, mode: str, release_sec: float) -> str:
        normalized_mode = str(mode or "release").strip().lower()
        release_duration = max(0.0, float(release_sec))
        if normalized_mode == "release":
            used_release = self.actuator.release_sucker(release_duration)
            if used_release:
                return "release"
        self.actuator.set_sucker(False)
        self._sleep_with_abort(release_duration)
        if normalized_mode == "release":
            return "off_fallback"
        return "off"

    def _build_cylindrical_move_plan(self, pose: CylindricalPose) -> CylindricalMovePlan:
        pose = pose.normalized()
        validation = self._validate_cylindrical_target(pose.theta_deg, pose.radius_mm, pose.z_mm)
        if not validation.get("ok", False):
            raise RobotCommandError(
                RobotErrorCode.TARGET_OUT_OF_WORKSPACE,
                str(validation.get("message") or "Invalid cylindrical move target."),
            )
        x_mm, y_mm, _ = pose.as_cartesian()
        return CylindricalMovePlan(
            theta_deg=float(pose.theta_deg),
            radius_mm=float(pose.radius_mm),
            z_mm=float(pose.z_mm),
            x_mm=float(x_mm),
            y_mm=float(y_mm),
        )

    def _build_reference_pulses(self) -> tuple[float, float, float] | None:
        try:
            import justkinematics

            angles = justkinematics.inverse(JETMAX_L4_MM, self._home_pose)
            self._reference_forearm_pitch_deg = self._estimate_forearm_pitch_deg(angles)
            pulses = tuple(float(value) for value in justkinematics.deg_to_pulse(angles))
            if len(pulses) != 3:
                return None
            return pulses
        except Exception:
            self._reference_forearm_pitch_deg = None
            return None

    def _default_pick_tuning(self) -> PickTuning:
        return PickTuning(
            pick_approach_z_mm=float(self.limits.z_approach),
            pick_descend_z_mm=float(self.limits.z_pick),
            pick_pre_suction_sec=0.25,
            pick_bottom_hold_sec=0.15,
            pick_lift_sec=0.8,
            place_descend_z_mm=float(self.limits.z_pick),
            place_release_mode="release",
            place_release_sec=0.25,
            place_post_release_hold_sec=0.10,
            z_carry_floor_mm=float(self.limits.z_carry),
        )

    def _normalize_pick_tuning(self, payload: dict[str, object] | None) -> PickTuning:
        current = self._pick_tuning if hasattr(self, "_pick_tuning") else self._default_pick_tuning()
        data = current.to_dict()
        if isinstance(payload, dict):
            data.update(payload)

        z_min = float(self.limits.z_min_mm)
        z_max = float(self.limits.z_max_mm)

        def _to_float(name: str, lower: float, upper: float) -> float:
            value = data.get(name, getattr(current, name))
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = float(getattr(current, name))
            return self._clamp(numeric, float(lower), float(upper))

        def _to_sec(name: str, upper: float = 3.0) -> float:
            value = data.get(name, getattr(current, name))
            return self._clamp(float(value), 0.0, float(upper))

        release_mode = str(data.get("place_release_mode", current.place_release_mode) or "release").strip().lower()
        if release_mode not in {"release", "off"}:
            release_mode = "release"

        return PickTuning(
            pick_approach_z_mm=_to_float("pick_approach_z_mm", z_min, z_max),
            pick_descend_z_mm=_to_float("pick_descend_z_mm", z_min, z_max),
            pick_pre_suction_sec=_to_sec("pick_pre_suction_sec"),
            pick_bottom_hold_sec=_to_sec("pick_bottom_hold_sec"),
            pick_lift_sec=_to_sec("pick_lift_sec"),
            place_descend_z_mm=_to_float("place_descend_z_mm", z_min, z_max),
            place_release_mode=release_mode,
            place_release_sec=_to_sec("place_release_sec"),
            place_post_release_hold_sec=_to_sec("place_post_release_hold_sec"),
            z_carry_floor_mm=_to_float("z_carry_floor_mm", z_min, z_max),
        )

    def _build_auto_z_profile(self) -> tuple[tuple[float, float], ...]:
        self._reference_pulses = self._build_reference_pulses()
        return build_auto_z_profile(
            radius_limits=(self.limits.radius_min_mm, self.limits.radius_max_mm),
            z_limits=(self.limits.z_min_mm, self.limits.z_max_mm),
            radius_step_mm=self.limits.auto_z_profile_radius_step_mm,
            z_step_mm=self.limits.auto_z_profile_height_step_mm,
            validator=self._validate_cylindrical_target,
            preferred_z_mm=self.limits.auto_z_preferred_mm,
            profile_theta_deg=0.0,
            target_z_provider=self._auto_z_target_mm,
            posture_tolerance=self.limits.auto_z_posture_tolerance_deg,
        )

    def _auto_z_target_mm(self, radius_mm: float) -> float:
        radius_value = float(radius_mm)
        plateau_min = float(self.limits.auto_z_plateau_min_radius_mm)
        plateau_max = float(self.limits.auto_z_plateau_max_radius_mm)
        plateau_z = float(self.limits.auto_z_plateau_z_mm)
        if plateau_min > plateau_max:
            plateau_min, plateau_max = plateau_max, plateau_min
        if plateau_min <= radius_value <= plateau_max:
            return plateau_z
        if radius_value < plateau_min:
            delta_radius = plateau_min - radius_value
            return plateau_z - delta_radius * float(self.limits.auto_z_retract_drop_per_radius_mm)
        delta_radius = radius_value - plateau_max
        return plateau_z - delta_radius * float(self.limits.auto_z_extend_drop_per_radius_mm)

    def _validate_cylindrical_target(self, theta_deg: float, radius_mm: float, z_mm: float) -> dict[str, object]:
        theta_value = float(theta_deg)
        radius_value = float(radius_mm)
        z_value = float(z_mm)
        if not within_limits(theta_value, (self.limits.theta_min_deg, self.limits.theta_max_deg)):
            return {"ok": False, "message": f"Theta {theta_value:.2f} deg is outside limits."}
        if not within_limits(radius_value, (self.limits.radius_min_mm, self.limits.radius_max_mm)):
            return {"ok": False, "message": f"Radius {radius_value:.2f} mm is outside limits."}
        if not within_limits(z_value, (self.limits.z_min_mm, self.limits.z_max_mm)):
            return {"ok": False, "message": f"Height {z_value:.2f} mm is outside limits."}
        x_mm, y_mm, z_cart = cylindrical_to_cartesian(theta_value, radius_value, z_value)
        return self._validate_pose_xyz(
            x_mm,
            y_mm,
            z_cart,
            enforce_workspace=self.limits.cylindrical_xy_workspace_enabled,
        )

    def _validate_pose_xyz(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        *,
        enforce_workspace: bool = True,
    ) -> dict[str, object]:
        radius_mm = math.hypot(float(x_mm), float(y_mm))
        if radius_mm < MIN_RADIUS_MM:
            return {"ok": False, "message": f"Radius {radius_mm:.2f} mm is below {MIN_RADIUS_MM:.2f} mm."}
        if z_mm > MAX_Z_MM:
            return {"ok": False, "message": f"Height {z_mm:.2f} mm exceeds {MAX_Z_MM:.2f} mm."}
        margin_candidates = [radius_mm - MIN_RADIUS_MM, MAX_Z_MM - z_mm]
        if enforce_workspace:
            if not (self.limits.x_min <= x_mm <= self.limits.x_max):
                return {"ok": False, "message": f"X {x_mm:.2f} mm is outside safe bounds."}
            if not (self.limits.y_min <= y_mm <= self.limits.y_max):
                return {"ok": False, "message": f"Y {y_mm:.2f} mm is outside safe bounds."}
            margin_candidates.extend(
                [
                    x_mm - self.limits.x_min,
                    self.limits.x_max - x_mm,
                    y_mm - self.limits.y_min,
                    self.limits.y_max - y_mm,
                ]
            )

        margin = min(margin_candidates)
        neutral_distance = abs(z_mm - self.limits.auto_z_preferred_mm)

        try:
            import justkinematics

            angles = justkinematics.inverse(JETMAX_L4_MM, (float(x_mm), float(y_mm), float(z_mm)))
            pulses = tuple(float(value) for value in justkinematics.deg_to_pulse(angles))
            if len(pulses) != 3 or not all(math.isfinite(value) for value in pulses):
                return {"ok": False, "message": "IK produced invalid servo pulses."}
            if pulses[0] < SERVO1_MIN_PULSE:
                return {"ok": False, "message": f"Servo1 pulse {pulses[0]:.2f} would be clamped low."}
            if pulses[0] > SERVO1_MAX_PULSE:
                return {"ok": False, "message": f"Servo1 pulse {pulses[0]:.2f} would be clamped high."}
            if pulses[1] < SERVO2_MIN_PULSE:
                return {"ok": False, "message": f"Servo2 pulse {pulses[1]:.2f} would be clamped low."}
            if pulses[2] < SERVO3_MIN_PULSE:
                return {"ok": False, "message": f"Servo3 pulse {pulses[2]:.2f} would be clamped."}
            if pulses[1] > SERVO2_MAX_PULSE:
                return {"ok": False, "message": f"Servo2 pulse {pulses[1]:.2f} would be clamped."}
            if pulses[2] > SERVO3_MAX_PULSE:
                return {"ok": False, "message": f"Servo3 pulse {pulses[2]:.2f} would be clamped high."}
            pulse_margin = min(
                pulses[0] - SERVO1_MIN_PULSE,
                SERVO1_MAX_PULSE - pulses[0],
                pulses[1] - SERVO2_MIN_PULSE,
                SERVO2_MAX_PULSE - pulses[1],
                pulses[2] - SERVO3_MIN_PULSE,
                SERVO3_MAX_PULSE - pulses[2],
            )
            margin = min(margin, pulse_margin)
            if self._reference_pulses is not None:
                neutral_distance = sum(abs(pulses[index] - self._reference_pulses[index]) for index in range(3))
            posture_distance = self._estimate_forearm_posture_distance(angles)
        except Exception:
            posture_distance = abs(neutral_distance)

        return {
            "ok": True,
            "message": None,
            "margin": float(margin),
            "neutral_distance": float(neutral_distance),
            "posture_distance": float(posture_distance),
        }

    @staticmethod
    def _estimate_forearm_pitch_deg(angles: tuple[float, ...] | list[float]) -> float:
        angle_values = tuple(float(value) for value in angles)
        if len(angle_values) < 3:
            return 0.0
        return float(angle_values[1] + angle_values[2])

    def _estimate_forearm_posture_distance(self, angles: tuple[float, ...] | list[float]) -> float:
        forearm_pitch_deg = self._estimate_forearm_pitch_deg(angles)
        reference_pitch_deg = self._reference_forearm_pitch_deg
        if reference_pitch_deg is None:
            reference_pitch_deg = 0.0
        return abs(float(forearm_pitch_deg) - float(reference_pitch_deg))

    def _ensure_operable_for_motion(self) -> None:
        if self._state == RobotExecutorState.ERROR:
            raise RobotCommandError(RobotErrorCode.INVALID_STATE, "Robot is in ERROR state; reset is required.")
        if self._state not in {RobotExecutorState.IDLE, RobotExecutorState.CARRY_READY}:
            raise RobotCommandError(RobotErrorCode.BUSY, f"Robot is busy in state {self._state.value}.")

    def _ensure_operable_for_action(self, *, action: str) -> None:
        if self._state == RobotExecutorState.ERROR:
            raise RobotCommandError(RobotErrorCode.INVALID_STATE, f"Cannot {action} while robot is in ERROR state.")
        if self._state not in {RobotExecutorState.IDLE, RobotExecutorState.CARRY_READY}:
            raise RobotCommandError(RobotErrorCode.BUSY, f"Cannot {action} while robot is {self._state.value}.")

    def _set_state(self, state: RobotExecutorState) -> None:
        self._state = state
        self._revision += 1
        self._log(f"Executor state -> {state.value}")

    def _clear_last_error(self) -> None:
        self._last_error_code = None
        self._last_error_message = None
        self._revision += 1

    def _busy_action_name(self) -> str | None:
        if self._state in {
            RobotExecutorState.PICK_APPROACH,
            RobotExecutorState.PICK_SUCTION_ON,
            RobotExecutorState.PICK_DESCEND,
            RobotExecutorState.PICK_LIFT,
        }:
            return "pick"
        if self._state in {
            RobotExecutorState.PLACE_DESCEND,
            RobotExecutorState.PLACE_RELEASE,
            RobotExecutorState.PLACE_LIFT,
        }:
            return "place"
        if self._state == RobotExecutorState.MOVING_XY:
            return "move"
        if self._state == RobotExecutorState.RECOVERING:
            return "recover"
        return None

    def _mark_control_kernel(self, kernel_name: str) -> None:
        with self._lock:
            if self._control_kernel != str(kernel_name):
                self._control_kernel = str(kernel_name)
                self._revision += 1

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _sleep_with_abort(self, duration: float, *, allow_abort: bool = True) -> None:
        deadline = time.monotonic() + max(0.0, float(duration))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if allow_abort and self._abort_requested.is_set():
                raise RobotCommandError(RobotErrorCode.ABORTED, "Abort requested by operator.")
            time.sleep(min(0.05, remaining))

    def _raise_if_abort_requested(self) -> None:
        if self._abort_requested.is_set():
            raise RobotCommandError(RobotErrorCode.ABORTED, "Abort requested by operator.")

    def _set_abort_error(self, code: RobotErrorCode | str, message: str) -> None:
        self._recover_to_safe_pose(disable_sucker=True)
        with self._lock:
            code_text = code.value if isinstance(code, RobotErrorCode) else str(code)
            self._set_state(RobotExecutorState.ERROR)
            self._last_error_code = code_text
            self._last_error_message = str(message)


class RobotTcpGateway:
    def __init__(
        self,
        executor: RobotExecutor,
        *,
        send_line: Callable[[object, str], None],
        log: Callable[[str], None],
    ) -> None:
        self.executor = executor
        self._send_line = send_line
        self._log = log

    def dispatch_command(self, line: str, stream) -> None:
        self._log(f"RX <= {line}")
        try:
            command, args = parse_command_text(line)
            if command == "PING":
                self._send_line(stream, "ACK PONG")
                return

            if command == "STATUS":
                self._send_line(stream, f"ACK STATUS {json.dumps(self.executor.healthcheck(), ensure_ascii=True)}")
                return

            if command == "ABORT":
                self.executor.abort()
                self._send_line(stream, "ACK ABORT")
                return

            if command == "RESET":
                self.executor.reset_error()
                self._send_line(stream, "ACK RESET")
                return

            if command == "MOVE":
                self.executor.legacy_kernel.move_xy(float(args[0]), float(args[1]))
                self._send_line(stream, "ACK MOVE")
                return

            if command == "MOVE_CYL":
                self.executor.cylindrical_kernel.move_cyl(float(args[0]), float(args[1]), float(args[2]))
                self._send_line(stream, "ACK MOVE")
                return

            if command == "MOVE_CYL_AUTO":
                self.executor.cylindrical_kernel.move_cyl_auto(float(args[0]), float(args[1]))
                self._send_line(stream, "ACK MOVE")
                return

            if command == "PICK":
                plan = self.executor.legacy_kernel.begin_pick(float(args[0]), float(args[1]))
                self._send_line(stream, "ACK PICK_STARTED")
                threading.Thread(
                    target=self._pick_worker,
                    args=(plan, stream),
                    name="robot-pick",
                    daemon=True,
                ).start()
                return

            if command == "PICK_WORLD":
                plan = self.executor.legacy_kernel.begin_pick_world(float(args[0]), float(args[1]))
                self._send_line(stream, "ACK PICK_STARTED")
                threading.Thread(
                    target=self._pick_worker,
                    args=(plan, stream),
                    name="robot-pick-world",
                    daemon=True,
                ).start()
                return

            if command == "PICK_CYL":
                plan = self.executor.cylindrical_kernel.begin_pick_cyl(float(args[0]), float(args[1]))
                self._send_line(stream, "ACK PICK_STARTED")
                threading.Thread(
                    target=self._pick_worker,
                    args=(plan, stream),
                    name="robot-pick-cyl",
                    daemon=True,
                ).start()
                return

            if command == "PLACE":
                plan = self.executor.begin_place()
                self._send_line(stream, "ACK PLACE_STARTED")
                threading.Thread(
                    target=self._place_worker,
                    args=(plan, stream),
                    name="robot-place",
                    daemon=True,
                ).start()
                return

            self._send_line(stream, f"ERR Unsupported command: {line}")
        except CommandParseError as error:
            if str(error) == "Empty command":
                self._send_line(stream, "ERR Empty command")
                return
            self._log(f"Command parse error: {error}")
            self._send_line(stream, f"ERR {error}")
        except ValueError as error:
            self._log(f"Command value error: {error}")
            self._send_line(stream, f"ERR {error}")
        except RobotCommandError as error:
            if error.code == RobotErrorCode.BUSY.value:
                self._send_line(stream, "BUSY")
            else:
                self._send_line(stream, format_error_line(error.code, error.message))
        except Exception as error:
            self._log(f"Command dispatch error: {error}")
            self._send_line(stream, format_error_line(RobotErrorCode.HARDWARE_FAILURE, str(error)))

    def _pick_worker(self, plan: PickPlan | WorldPickPlan | CylindricalPickPlan, stream) -> None:
        try:
            self.executor.complete_pick(plan)
            self._send_line(stream, "ACK PICK_DONE")
            self._log("PICK completed.")
        except Exception as error:
            failure = self.executor.handle_pick_failure(error)
            if failure.code == RobotErrorCode.ABORTED.value:
                self._log("PICK aborted.")
                return
            self._log(f"PICK failed: {failure.code}: {failure.message}")
            self._send_line(stream, format_error_line(failure.code, failure.message))

    def _place_worker(self, plan: PlacePlan, stream) -> None:
        try:
            self.executor.complete_place(plan)
            self._send_line(stream, "ACK PLACE_DONE")
            self._log("PLACE completed.")
        except Exception as error:
            failure = self.executor.handle_place_failure(error)
            if failure.code == RobotErrorCode.ABORTED.value:
                self._log("PLACE aborted.")
                return
            self._log(f"PLACE failed: {failure.code}: {failure.message}")
            self._send_line(stream, format_error_line(failure.code, failure.message))


class _RobotRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:  # pragma: no cover - network path
        runtime: RobotRuntime = self.server.runtime
        runtime._log(f"Client connected: {self.client_address}")
        try:
            while True:
                raw = self.rfile.readline()
                if not raw:
                    return
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                runtime.dispatch_command(line, self.wfile)
        finally:
            runtime._log(f"Client disconnected: {self.client_address}")


class _RobotTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], runtime: "RobotRuntime") -> None:
        super().__init__(server_address, _RobotRequestHandler)
        self.runtime = runtime


class RobotRuntime:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8888,
        *,
        limits: RobotLimits | None = None,
        hardware_factory: Optional[Callable[[], HardwareProtocol]] = None,
        calibration_factory: Optional[Callable[[], CalibrationProtocol]] = None,
        log_stream: TextIO | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.limits = limits or RobotLimits()
        self._hardware_factory = hardware_factory or _HardwareBridge
        self._calibration_factory = calibration_factory or CalibrationProvider
        self._log_stream = log_stream
        self._hardware: Optional[HardwareProtocol] = None
        self._calibration: Optional[CalibrationProtocol] = None
        self._executor: Optional[RobotExecutor] = None
        self._gateway: Optional[RobotTcpGateway] = None
        self._server: Optional[_RobotTCPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._write_lock = threading.Lock()
        self._stopped = threading.Event()

    def start(self) -> None:
        if self._server is not None:
            return
        self._log("Initializing hardware bridge...")
        self._hardware = self._hardware_factory()
        self._log("Initializing calibration provider...")
        self._calibration = self._calibration_factory()
        self._log("Sending robot to home pose...")
        self._hardware.go_home()
        self._initialize_executor()
        self._server = _RobotTCPServer((self.host, self.port), self)
        self.host, self.port = self._server.server_address
        self._thread = threading.Thread(target=self._server.serve_forever, name="robot-runtime", daemon=True)
        self._thread.start()
        self._stopped.clear()
        self._log(f"Robot runtime listening on {self.host}:{self.port}")

    def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            self._log("Stopping robot runtime...")
            server.shutdown()
            server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._stopped.set()

    def wait_forever(self) -> None:
        try:
            while self._server is not None and not self._stopped.wait(0.5):
                pass
        except KeyboardInterrupt:  # pragma: no cover - manual operation
            self._log("KeyboardInterrupt received, shutting down.")
            self.stop()

    def healthcheck(self) -> dict[str, object]:
        self._initialize_executor()
        if self._executor is None:
            return {
                "running": self._server is not None,
                "host": self.host,
                "port": self.port,
                "hardware_ready": self._hardware is not None,
                "busy": False,
                "state": RobotExecutorState.ERROR.value,
                "carrying": False,
                "control_kernel": CylindricalKernel.name,
                "position_xyz": None,
                "last_error_code": None,
                "last_error_message": "Executor unavailable",
                "calibration_ready": False,
            }
        return {
            "running": self._server is not None,
            "host": self.host,
            "port": self.port,
            "hardware_ready": self._hardware is not None,
            **self._executor.healthcheck(),
        }

    def dispatch_command(self, line: str, stream) -> None:
        self._initialize_executor()
        if self._gateway is None:
            self._send_line(stream, format_error_line(RobotErrorCode.HARDWARE_FAILURE, "Robot executor unavailable."))
            return
        self._gateway.dispatch_command(line, stream)

    def reset_error(self) -> None:
        self._initialize_executor()
        if self._executor is not None:
            self._executor.reset_error()

    def abort(self) -> None:
        self._initialize_executor()
        if self._executor is not None:
            self._executor.abort()

    def _initialize_executor(self) -> None:
        if self._executor is not None and self._gateway is not None:
            return
        if self._hardware is None:
            return
        if self._calibration is None:
            self._calibration = _UnavailableCalibrationProvider("Calibration provider was not initialized.")
        self._executor = RobotExecutor(self._hardware, self._calibration, limits=self.limits, log=self._log)
        self._gateway = RobotTcpGateway(self._executor, send_line=self._send_line, log=self._log)

    def _send_line(self, stream, line: str) -> None:
        try:
            with self._write_lock:
                stream.write((line + "\n").encode("utf-8"))
                stream.flush()
            self._log(f"TX => {line}")
        except Exception as error:
            self._log(f"Failed to send '{line}': {error}")

    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        if self._log_stream is not None:
            print(line, file=self._log_stream, flush=True)
        else:
            print(line, flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid Controller robot runtime")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host, default 0.0.0.0")
    parser.add_argument("--port", type=int, default=8888, help="Bind port, default 8888")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    runtime = RobotRuntime(host=args.host, port=args.port)
    runtime.start()
    runtime.wait_forever()
    runtime.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    raise SystemExit(main())
