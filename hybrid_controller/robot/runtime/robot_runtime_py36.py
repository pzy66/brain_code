#!/usr/bin/env python3
"""JetMax robot runtime compatible with Python 3.6 on the official image."""

import argparse
import json
import math
import socketserver
import threading
import time
import os
import sys

try:
    from hybrid_controller.robot.runtime.runtime_core import CommandParseError, parse_command_text
except Exception:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    try:
        from runtime_core import CommandParseError, parse_command_text
    except Exception:
        _REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        from hybrid_controller.robot.runtime.runtime_core import CommandParseError, parse_command_text


STATE_IDLE = "IDLE"
STATE_MOVING_XY = "MOVING_XY"
STATE_PICK_APPROACH = "PICK_APPROACH"
STATE_PICK_SUCTION_ON = "PICK_SUCTION_ON"
STATE_PICK_DESCEND = "PICK_DESCEND"
STATE_PICK_LIFT = "PICK_LIFT"
STATE_CARRY_READY = "CARRY_READY"
STATE_PLACE_DESCEND = "PLACE_DESCEND"
STATE_PLACE_RELEASE = "PLACE_RELEASE"
STATE_PLACE_LIFT = "PLACE_LIFT"
STATE_RECOVERING = "RECOVERING"
STATE_ERROR = "ERROR"

ERR_ABORTED = "aborted"
ERR_CALIBRATION_UNAVAILABLE = "calibration_unavailable"
ERR_TARGET_OUT_OF_WORKSPACE = "target_out_of_workspace"
ERR_INVALID_STATE = "invalid_state"
ERR_BUSY = "busy"
ERR_HARDWARE_FAILURE = "hardware_failure"
ERR_RECOVER_FAILED = "recover_failed"
ERR_CONNECTION_LOST = "connection_lost"

JETMAX_L4_MM = 16.8
SERVO1_MIN_PULSE = 0
SERVO1_MAX_PULSE = 1000
SERVO2_MAX_PULSE = 700
SERVO2_MIN_PULSE = 0
SERVO3_MIN_PULSE = 470
SERVO3_MAX_PULSE = 1000
MIN_RADIUS_MM = 50.0
MAX_Z_MM = 225.0


def format_error_line(code, message):
    return "ERR {0}: {1}".format(str(code), str(message))


def cylindrical_to_cartesian(theta_deg, radius_mm, z_mm):
    theta_rad = math.radians(float(theta_deg))
    radius = float(radius_mm)
    return (
        radius * math.sin(theta_rad),
        -radius * math.cos(theta_rad),
        float(z_mm),
    )


def cartesian_to_cylindrical(x_mm, y_mm, z_mm):
    radius = math.hypot(float(x_mm), float(y_mm))
    theta_deg = math.degrees(math.atan2(float(x_mm), -float(y_mm)))
    return (float(theta_deg), float(radius), float(z_mm))


def within_limits(value, limits):
    return float(limits[0]) <= float(value) <= float(limits[1])


def sample_range(start, stop, step):
    values = []
    current = float(start)
    stop_value = float(stop)
    step_value = abs(float(step))
    while current <= stop_value + 1e-6:
        values.append(round(current, 6))
        current += step_value
    if not values or abs(values[-1] - stop_value) > 1e-6:
        values.append(round(stop_value, 6))
    return values


def interpolate_auto_z(profile_points, radius_mm):
    points = sorted((float(radius), float(z_mm)) for radius, z_mm in profile_points)
    if radius_mm <= points[0][0]:
        return points[0][1]
    if radius_mm >= points[-1][0]:
        return points[-1][1]
    for index in range(1, len(points)):
        left_r, left_z = points[index - 1]
        right_r, right_z = points[index]
        if radius_mm > right_r:
            continue
        if abs(right_r - left_r) <= 1e-6:
            return right_z
        ratio = (float(radius_mm) - left_r) / (right_r - left_r)
        return left_z + (right_z - left_z) * ratio
    return points[-1][1]


def nearest_profile_radius_limits(profile_points):
    points = sorted((float(radius), float(z_mm)) for radius, z_mm in profile_points)
    return (points[0][0], points[-1][0])


class RobotCommandError(Exception):
    def __init__(self, code, message):
        Exception.__init__(self, str(message))
        self.code = str(code)
        self.message = str(message)


class AbortRequested(Exception):
    pass


class Sender(object):
    def __init__(self, wfile):
        self._wfile = wfile
        self._lock = threading.Lock()
        self._closed = False

    def send_line(self, line):
        payload = (str(line).rstrip("\n") + "\n").encode("utf-8")
        with self._lock:
            if self._closed:
                raise IOError("sender closed")
            self._wfile.write(payload)
            self._wfile.flush()

    def close(self):
        with self._lock:
            self._closed = True


class ActuatorAdapter(object):
    def __init__(self, arm, sucker):
        self._arm = arm
        self._sucker = sucker

    def go_home(self, duration):
        self._arm.go_home(float(duration), 1)

    def get_position(self):
        position = getattr(self._arm, "position", (0.0, 0.0, 0.0))
        return (float(position[0]), float(position[1]), float(position[2]))

    def move_xyz(self, x, y, z, duration):
        self._arm.set_position((float(x), float(y), float(z)), float(duration))

    def move_relative_xyz(self, dx, dy, dz, duration):
        relative_move = getattr(self._arm, "set_position_relatively", None)
        if callable(relative_move):
            relative_move((float(dx), float(dy), float(dz)), float(duration))
            return
        current = getattr(self._arm, "position", (0.0, 0.0, 0.0))
        self._arm.set_position(
            (
                float(current[0]) + float(dx),
                float(current[1]) + float(dy),
                float(current[2]) + float(dz),
            ),
            float(duration),
        )

    def set_sucker(self, state):
        self._sucker.set_state(bool(state))

    def release_sucker(self, duration_sec):
        release = getattr(self._sucker, "release", None)
        if callable(release):
            release(float(duration_sec))
            return True
        return False


class LegacyCartesianKernel(object):
    name = "legacy_cartesian"

    def __init__(self, executor):
        self._executor = executor

    def start_move(self, sender, target_x, target_y):
        self._executor._mark_control_kernel(self.name)
        return self._executor._legacy_start_move_impl(sender, target_x, target_y)

    def start_pick_pixel(self, sender, pixel_x, pixel_y):
        self._executor._mark_control_kernel(self.name)
        return self._executor.handle_pick_pixel()

    def start_pick_world(self, sender, target_x, target_y):
        self._executor._mark_control_kernel(self.name)
        return self._executor._legacy_start_pick_world_impl(sender, target_x, target_y)


class CylindricalKernel(object):
    name = "cylindrical_kernel"

    def __init__(self, executor):
        self._executor = executor

    def start_move(self, sender, theta_deg, radius_mm, z_mm):
        self._executor._mark_control_kernel(self.name)
        return self._executor._cyl_start_move_impl(sender, theta_deg, radius_mm, z_mm)

    def start_move_auto(self, sender, theta_deg, radius_mm):
        self._executor._mark_control_kernel(self.name)
        return self._executor._cyl_start_move_auto_impl(sender, theta_deg, radius_mm)

    def start_pick(self, sender, theta_deg, radius_mm):
        self._executor._mark_control_kernel(self.name)
        return self._executor._cyl_start_pick_impl(sender, theta_deg, radius_mm)


class JetMaxExecutor(object):
    def __init__(self, limits, home_pose):
        import hiwonder
        import justkinematics
        import rospy

        self.hiwonder = hiwonder
        self.justkinematics = justkinematics
        self.rospy = rospy
        if not rospy.core.is_initialized():
            rospy.init_node("hybrid_robot_runtime", anonymous=True, disable_signals=True)
        self.arm = hiwonder.JetMax()
        self.sucker = hiwonder.Sucker()
        self.actuator = ActuatorAdapter(self.arm, self.sucker)
        rospy.sleep(0.15)

        self.x_min = float(limits["x_min"])
        self.x_max = float(limits["x_max"])
        self.y_min = float(limits["y_min"])
        self.y_max = float(limits["y_max"])
        self.cylindrical_xy_workspace_enabled = bool(limits.get("cylindrical_xy_workspace_enabled", False))
        self.z_approach = float(limits["z_approach"])
        self.z_pick = float(limits["z_pick"])
        self.z_carry = float(limits["z_carry"])
        self.pick_approach_z_mm = float(limits.get("pick_approach_z_mm", self.z_approach))
        self.pick_descend_z_mm = float(limits.get("pick_descend_z_mm", self.z_pick))
        self.pick_pre_suction_sec = float(limits.get("pick_pre_suction_sec", 0.25))
        self.pick_bottom_hold_sec = float(limits.get("pick_bottom_hold_sec", 0.15))
        self.pick_lift_sec = float(limits.get("pick_lift_sec", 0.8))
        self.place_descend_z_mm = float(limits.get("place_descend_z_mm", self.z_pick))
        self.place_release_mode = str(limits.get("place_release_mode", "release"))
        self.place_release_sec = float(limits.get("place_release_sec", 0.25))
        self.place_post_release_hold_sec = float(limits.get("place_post_release_hold_sec", 0.10))
        self.z_carry_floor_mm = float(limits.get("z_carry_floor_mm", self.z_carry))
        self.move_speed_xy = float(limits["move_speed_xy"])
        self.theta_limits = tuple(float(value) for value in limits.get("theta_limits", (-120.0, 120.0)))
        self.radius_limits = tuple(float(value) for value in limits.get("radius_limits", (50.0, 280.0)))
        self.auto_radius_limits = tuple(float(value) for value in limits.get("auto_radius_limits", (80.0, 260.0)))
        self.z_limits = tuple(float(value) for value in limits.get("z_limits", (80.0, 212.8)))
        self.auto_z_radius_step = float(limits.get("auto_z_radius_step", 5.0))
        self.auto_z_height_step = float(limits.get("auto_z_height_step", 5.0))
        self.auto_z_preferred = float(limits.get("auto_z_preferred", self.z_carry))
        self.auto_z_plateau_min = float(limits.get("auto_z_plateau_min", 145.0))
        self.auto_z_plateau_max = float(limits.get("auto_z_plateau_max", 185.0))
        self.auto_z_plateau_z = float(limits.get("auto_z_plateau_z", 205.0))
        self.auto_z_retract_drop_per_radius = float(limits.get("auto_z_retract_drop_per_radius", 0.8))
        self.auto_z_extend_drop_per_radius = float(limits.get("auto_z_extend_drop_per_radius", 0.4))
        self.auto_z_posture_tolerance = float(limits.get("auto_z_posture_tolerance", 8.0))
        self.auto_z_down_per_radius = float(limits.get("auto_z_down_per_radius", 0.5))
        self.auto_z_up_per_radius = float(limits.get("auto_z_up_per_radius", 1.0))
        self.auto_z_min_delta = float(limits.get("auto_z_min_delta", 3.0))
        self.motion_min_duration = float(limits.get("motion_min_duration", 0.25))
        self.motion_settle_sec = float(limits.get("motion_settle_sec", 0.08))
        self.teleop_min_duration = float(limits.get("teleop_min_duration", 0.12))
        self.teleop_settle_sec = float(limits.get("teleop_settle_sec", 0.02))
        origin = getattr(self.arm, "origin", None)
        if origin is not None and len(origin) == 3:
            self.home_pose = (
                float(origin[0]),
                float(origin[1]),
                float(origin[2]),
            )
        else:
            self.home_pose = (
                float(home_pose[0]),
                float(home_pose[1]),
                float(home_pose[2]),
            )

        self._lock = threading.RLock()
        self._abort_event = threading.Event()
        self._action_thread = None
        self._state = STATE_IDLE
        self._busy = False
        self._busy_action = ""
        self._carrying = False
        self._control_kernel = CylindricalKernel.name
        self._last_error_code = ""
        self._last_error = ""
        self._last_ack = ""
        self._commanded_pose = (
            float(self.home_pose[0]),
            float(self.home_pose[1]),
            float(self.home_pose[2]),
        )
        self._commanded_cyl = cartesian_to_cylindrical(
            self._commanded_pose[0], self._commanded_pose[1], self._commanded_pose[2]
        )
        self._reference_forearm_pitch_deg = None
        self._reference_pulses = self._build_reference_pulses()
        self._auto_z_profile = self._build_auto_z_profile()
        self._last_post_pick_settle_z = float(self.z_carry_floor_mm)
        self._last_release_mode_effective = "off"
        self.legacy_kernel = LegacyCartesianKernel(self)
        self.cylindrical_kernel = CylindricalKernel(self)

    def snapshot(self):
        x, y, z = self._get_position()
        theta_deg, radius_mm, z_mm = cartesian_to_cylindrical(x, y, z)
        validation = self._validate_pose_xyz(x, y, z, enforce_workspace=self.cylindrical_xy_workspace_enabled)
        with self._lock:
            return {
                "state": self._state,
                "busy": self._busy,
                "busy_action": self._busy_action,
                "carrying": self._carrying,
                "control_kernel": self._control_kernel,
                "robot_xy": [round(x, 3), round(y, 3)],
                "robot_z": round(z, 3),
                "robot_cyl": {
                    "theta_deg": round(theta_deg, 3),
                    "radius_mm": round(radius_mm, 3),
                    "z_mm": round(z_mm, 3),
                },
                "home_pose": [self.home_pose[0], self.home_pose[1], self.home_pose[2]],
                "limits_x": [self.x_min, self.x_max],
                "limits_y": [self.y_min, self.y_max],
                "limits_cyl": {
                    "theta_deg": [self.theta_limits[0], self.theta_limits[1]],
                    "radius_mm": list(nearest_profile_radius_limits(self._auto_z_profile)),
                    "z_mm": [self.z_limits[0], self.z_limits[1]],
                },
                "limits_cyl_auto": {
                    "theta_deg": [self.theta_limits[0], self.theta_limits[1]],
                    "radius_mm": [self.auto_radius_limits[0], self.auto_radius_limits[1]],
                    "z_mm": [self.z_limits[0], self.z_limits[1]],
                },
                "cylindrical_xy_workspace_enabled": self.cylindrical_xy_workspace_enabled,
                "approach_z": self.pick_approach_z_mm,
                "pick_z": self.pick_descend_z_mm,
                "carry_z": self.z_carry,
                "auto_z_enabled": True,
                "auto_z_current": round(interpolate_auto_z(self._auto_z_profile, radius_mm), 3),
                "ik_valid": bool(validation.get("ok", False)),
                "validation_error": validation.get("message"),
                "last_error_code": self._last_error_code,
                "last_error": self._last_error,
                "last_ack": self._last_ack,
                "calibration_ready": self._has_calibration(),
                "pick_tuning": self.get_pick_tuning(),
                "post_pick_settle_z": float(self._last_post_pick_settle_z),
                "release_mode_effective": str(self._last_release_mode_effective),
            }

    def get_pick_tuning(self):
        with self._lock:
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

    def set_pick_tuning(self, payload):
        with self._lock:
            data = self.get_pick_tuning()
            if isinstance(payload, dict):
                data.update(payload)
            z_min = float(self.z_limits[0])
            z_max = float(self.z_limits[1])
            self.pick_approach_z_mm = self._clamp(float(data["pick_approach_z_mm"]), z_min, z_max)
            self.pick_descend_z_mm = self._clamp(float(data["pick_descend_z_mm"]), z_min, z_max)
            self.pick_pre_suction_sec = self._clamp(float(data["pick_pre_suction_sec"]), 0.0, 3.0)
            self.pick_bottom_hold_sec = self._clamp(float(data["pick_bottom_hold_sec"]), 0.0, 3.0)
            self.pick_lift_sec = self._clamp(float(data["pick_lift_sec"]), 0.0, 3.0)
            self.place_descend_z_mm = self._clamp(float(data["place_descend_z_mm"]), z_min, z_max)
            release_mode = str(data.get("place_release_mode", "release") or "release").strip().lower()
            self.place_release_mode = "release" if release_mode not in {"release", "off"} else release_mode
            self.place_release_sec = self._clamp(float(data["place_release_sec"]), 0.0, 3.0)
            self.place_post_release_hold_sec = self._clamp(float(data["place_post_release_hold_sec"]), 0.0, 3.0)
            self.z_carry_floor_mm = self._clamp(float(data["z_carry_floor_mm"]), z_min, z_max)
            return self.get_pick_tuning()

    def start_move(self, sender, target_x, target_y):
        return self.legacy_kernel.start_move(sender, target_x, target_y)

    def start_move_cyl(self, sender, theta_deg, radius_mm, z_mm):
        return self.cylindrical_kernel.start_move(sender, theta_deg, radius_mm, z_mm)

    def start_move_cyl_auto(self, sender, theta_deg, radius_mm):
        return self.cylindrical_kernel.start_move_auto(sender, theta_deg, radius_mm)

    def start_pick_world(self, sender, target_x, target_y):
        return self.legacy_kernel.start_pick_world(sender, target_x, target_y)

    def start_pick_cyl(self, sender, theta_deg, radius_mm):
        return self.cylindrical_kernel.start_pick(sender, theta_deg, radius_mm)

    def _legacy_start_move_impl(self, sender, target_x, target_y):
        target_x = self._clamp(float(target_x), self.x_min, self.x_max)
        target_y = self._clamp(float(target_y), self.y_min, self.y_max)
        with self._lock:
            if self._busy:
                return "BUSY"
            self._busy = True
            self._busy_action = "move"
            self._state = STATE_MOVING_XY
            self._abort_event.clear()
            self._action_thread = threading.Thread(
                target=self._run_move,
                args=(sender, target_x, target_y),
                name="jetmax-move",
                daemon=True,
            )
            self._action_thread.start()
        return None

    def _cyl_start_move_impl(self, sender, theta_deg, radius_mm, z_mm):
        validation = self._validate_cyl_target(theta_deg, radius_mm, z_mm)
        if not validation["ok"]:
            return format_error_line(ERR_TARGET_OUT_OF_WORKSPACE, validation["message"])
        target_x, target_y, target_z = cylindrical_to_cartesian(theta_deg, radius_mm, z_mm)
        with self._lock:
            if self._busy:
                return "BUSY"
            self._busy = True
            self._busy_action = "move"
            self._state = STATE_MOVING_XY
            self._abort_event.clear()
            self._action_thread = threading.Thread(
                target=self._run_move_cyl,
                args=(sender, float(theta_deg), float(radius_mm), float(target_x), float(target_y), float(target_z)),
                name="jetmax-move-cyl",
                daemon=True,
            )
            self._action_thread.start()
        return None

    def _cyl_start_move_auto_impl(self, sender, theta_deg, radius_mm):
        profile_limits = nearest_profile_radius_limits(self._auto_z_profile)
        effective_limits = (
            max(profile_limits[0], self.auto_radius_limits[0]),
            min(profile_limits[1], self.auto_radius_limits[1]),
        )
        if not within_limits(radius_mm, effective_limits):
            return format_error_line(
                ERR_TARGET_OUT_OF_WORKSPACE,
                "Radius {0:.2f} mm is outside auto-z profile range {1}.".format(float(radius_mm), effective_limits),
            )
        current_theta_deg, current_radius_mm, current_z_mm = cartesian_to_cylindrical(*self._get_position())
        desired_z_mm = interpolate_auto_z(self._auto_z_profile, float(radius_mm))
        delta_radius_mm = abs(float(radius_mm) - float(current_radius_mm))
        max_down_step_mm = max(self.auto_z_min_delta, delta_radius_mm * self.auto_z_down_per_radius)
        max_up_step_mm = max(self.auto_z_min_delta, delta_radius_mm * self.auto_z_up_per_radius)
        if desired_z_mm < float(current_z_mm):
            z_mm = max(desired_z_mm, float(current_z_mm) - max_down_step_mm)
        else:
            z_mm = min(desired_z_mm, float(current_z_mm) + max_up_step_mm)
        validation = self._validate_cyl_target(theta_deg, radius_mm, z_mm)
        if not validation["ok"]:
            return format_error_line(ERR_TARGET_OUT_OF_WORKSPACE, validation["message"])
        target_x, target_y, target_z = cylindrical_to_cartesian(theta_deg, radius_mm, z_mm)
        with self._lock:
            if self._busy:
                return "BUSY"
            self._busy = True
            self._busy_action = "move"
            self._state = STATE_MOVING_XY
            self._abort_event.clear()
            self._action_thread = threading.Thread(
                target=self._run_move_cyl_auto,
                args=(sender, float(theta_deg), float(radius_mm), float(target_x), float(target_y), float(target_z)),
                name="jetmax-move-cyl-auto",
                daemon=True,
            )
            self._action_thread.start()
        return None

    def _legacy_start_pick_world_impl(self, sender, target_x, target_y):
        target_x = float(target_x)
        target_y = float(target_y)
        theta_deg, radius_mm, _ = cartesian_to_cylindrical(target_x, target_y, self.z_pick)
        validation = self._validate_cyl_target(theta_deg, radius_mm, self.z_pick)
        if not validation["ok"]:
            return format_error_line(
                ERR_TARGET_OUT_OF_WORKSPACE,
                validation["message"],
            )
        with self._lock:
            if self._busy:
                return "BUSY"
            if self._carrying:
                return format_error_line(ERR_INVALID_STATE, "Already carrying an object.")
            self._busy = True
            self._busy_action = "pick"
            self._state = STATE_PICK_APPROACH
            self._abort_event.clear()
            self._action_thread = threading.Thread(
                target=self._run_pick_world,
                args=(sender, target_x, target_y),
                name="jetmax-pick",
                daemon=True,
            )
            self._action_thread.start()
        return "ACK PICK_STARTED"

    def _cyl_start_pick_impl(self, sender, theta_deg, radius_mm):
        validation = self._validate_cyl_target(theta_deg, radius_mm, self.z_pick)
        if not validation["ok"]:
            return format_error_line(ERR_TARGET_OUT_OF_WORKSPACE, validation["message"])
        target_x, target_y, _ = cylindrical_to_cartesian(theta_deg, radius_mm, self.z_pick)
        return self._legacy_start_pick_world_impl(sender, target_x, target_y)

    def start_place(self, sender):
        with self._lock:
            if self._busy:
                return "BUSY"
            if not self._carrying:
                return format_error_line(ERR_INVALID_STATE, "No object is currently carried.")
            self._busy = True
            self._busy_action = "place"
            self._state = STATE_PLACE_DESCEND
            self._abort_event.clear()
            self._action_thread = threading.Thread(
                target=self._run_place,
                args=(sender,),
                name="jetmax-place",
                daemon=True,
            )
            self._action_thread.start()
        return "ACK PLACE_STARTED"

    def abort(self):
        self._abort_event.set()
        action_thread = None
        with self._lock:
            action_thread = self._action_thread
        if action_thread is not None and action_thread.is_alive():
            action_thread.join(timeout=6.0)
        else:
            self._recover_after_abort()
        with self._lock:
            self._last_ack = "ABORT"
        return "ACK ABORT"

    def reset(self):
        with self._lock:
            if self._busy:
                return "BUSY"
        try:
            self.actuator.set_sucker(False)
            self._abort_event.clear()
            self._go_home(2.0, False)
        except Exception as error:
            self._set_error(ERR_RECOVER_FAILED, "RESET failed: {0}".format(error))
            return format_error_line(ERR_RECOVER_FAILED, "RESET failed: {0}".format(error))
        with self._lock:
            self._state = STATE_IDLE
            self._busy = False
            self._busy_action = ""
            self._carrying = False
            self._last_error_code = ""
            self._last_error = ""
            self._last_ack = "RESET"
        return "ACK RESET"

    def handle_pick_pixel(self):
        return format_error_line(
            ERR_CALIBRATION_UNAVAILABLE,
            "PICK pixel mode is not enabled in the JetMax python3.6 runtime.",
        )

    def _compute_settle_z(self, x_mm, y_mm, carry_floor_mm):
        _, radius_mm, _ = cartesian_to_cylindrical(float(x_mm), float(y_mm), float(self.pick_descend_z_mm))
        auto_z = float(interpolate_auto_z(self._auto_z_profile, float(radius_mm)))
        settle_z = max(float(carry_floor_mm), auto_z)
        return self._clamp(float(settle_z), float(self.z_limits[0]), float(self.z_limits[1]))

    def _apply_place_release(self):
        mode = str(self.place_release_mode or "release").strip().lower()
        if mode == "release":
            if self.actuator.release_sucker(float(self.place_release_sec)):
                return "release"
        self.actuator.set_sucker(False)
        self._sleep_with_abort(float(self.place_release_sec))
        if mode == "release":
            return "off_fallback"
        return "off"

    def _run_move(self, sender, target_x, target_y):
        try:
            current_x, current_y, current_z = self._get_position()
            move_z = current_z if current_z >= self.z_carry else self.z_carry
            if move_z - current_z > 2.0:
                self._move_to(current_x, current_y, self.z_carry, None, True)
            self._set_state(STATE_MOVING_XY, True, "move")
            duration = max(0.25, math.hypot(target_x - current_x, target_y - current_y) / self.move_speed_xy)
            self._move_to(target_x, target_y, move_z, duration, True)
            with self._lock:
                self._state = STATE_CARRY_READY if self._carrying else STATE_IDLE
                self._busy = False
                self._busy_action = ""
                self._last_ack = "MOVE"
            sender.send_line("ACK MOVE")
        except AbortRequested:
            self._recover_after_abort()
        except Exception as error:
            self._recover_from_error(ERR_HARDWARE_FAILURE, "MOVE failed: {0}".format(error), sender)

    def _run_move_cyl(self, sender, theta_deg, radius_mm, target_x, target_y, target_z):
        try:
            current_x, current_y, current_z = self._get_position()
            self._set_state(STATE_MOVING_XY, True, "move")
            duration_xyz = max(
                self.motion_min_duration,
                math.sqrt(
                    (target_x - current_x) ** 2 +
                    (target_y - current_y) ** 2 +
                    (target_z - current_z) ** 2
                ) / self.move_speed_xy
            )
            self._move_to(target_x, target_y, target_z, duration_xyz, True, settle_sec=self.motion_settle_sec)
            with self._lock:
                self._commanded_cyl = (float(theta_deg), float(radius_mm), float(target_z))
                self._state = STATE_CARRY_READY if self._carrying else STATE_IDLE
                self._busy = False
                self._busy_action = ""
                self._last_ack = "MOVE"
            sender.send_line("ACK MOVE")
        except AbortRequested:
            self._recover_after_abort()
        except Exception as error:
            self._recover_from_error(ERR_HARDWARE_FAILURE, "MOVE_CYL failed: {0}".format(error), sender)

    def _run_move_cyl_auto(self, sender, theta_deg, radius_mm, target_x, target_y, target_z):
        try:
            current_x, current_y, current_z = self._get_position()
            self._set_state(STATE_MOVING_XY, True, "move")
            duration_xyz = max(
                self.teleop_min_duration,
                math.sqrt(
                    (target_x - current_x) ** 2 +
                    (target_y - current_y) ** 2 +
                    (target_z - current_z) ** 2
                ) / self.move_speed_xy
            )
            self._move_to(target_x, target_y, target_z, duration_xyz, True, settle_sec=self.teleop_settle_sec)
            with self._lock:
                self._commanded_cyl = (float(theta_deg), float(radius_mm), float(target_z))
                self._state = STATE_CARRY_READY if self._carrying else STATE_IDLE
                self._busy = False
                self._busy_action = ""
                self._last_ack = "MOVE"
            sender.send_line("ACK MOVE")
        except AbortRequested:
            self._recover_after_abort()
        except Exception as error:
            self._recover_from_error(ERR_HARDWARE_FAILURE, "MOVE_CYL_AUTO failed: {0}".format(error), sender)

    def _run_pick_world(self, sender, target_x, target_y):
        try:
            current_x, current_y, current_z = self._get_position()
            settle_z = self._compute_settle_z(target_x, target_y, self.z_carry_floor_mm)
            if abs(current_z - settle_z) > 2.0:
                self._move_to(current_x, current_y, settle_z, None, True)
            self._set_state(STATE_PICK_APPROACH, True, "pick")
            self._move_to(target_x, target_y, self.pick_approach_z_mm, None, True)
            self._set_state(STATE_PICK_SUCTION_ON, True, "pick")
            self.actuator.set_sucker(True)
            self._sleep_with_abort(float(self.pick_pre_suction_sec))
            self._set_state(STATE_PICK_DESCEND, True, "pick")
            self._move_to(target_x, target_y, self.pick_descend_z_mm, 0.6, True)
            self._sleep_with_abort(float(self.pick_bottom_hold_sec))
            self._set_state(STATE_PICK_LIFT, True, "pick")
            self._move_to(target_x, target_y, settle_z, self.pick_lift_sec, True)
            with self._lock:
                self._carrying = True
                self._last_post_pick_settle_z = float(settle_z)
                self._state = STATE_CARRY_READY
                self._busy = False
                self._busy_action = ""
                self._last_ack = "PICK_DONE"
            sender.send_line("ACK PICK_DONE")
        except AbortRequested:
            self._recover_after_abort()
        except Exception as error:
            self._recover_from_error(ERR_HARDWARE_FAILURE, "PICK_WORLD failed: {0}".format(error), sender)

    def _run_place(self, sender):
        try:
            current_x, current_y, current_z = self._get_position()
            settle_z = self._compute_settle_z(current_x, current_y, self.z_carry_floor_mm)
            if current_z < float(self.z_carry_floor_mm):
                self._move_to(current_x, current_y, settle_z, None, True)
            self._set_state(STATE_PLACE_DESCEND, True, "place")
            self._move_to(current_x, current_y, self.place_descend_z_mm, 0.6, True)
            self._set_state(STATE_PLACE_RELEASE, True, "place")
            release_mode = self._apply_place_release()
            self._sleep_with_abort(float(self.place_post_release_hold_sec))
            self._set_state(STATE_PLACE_LIFT, True, "place")
            self._move_to(current_x, current_y, settle_z, 0.8, True)
            with self._lock:
                self._carrying = False
                self._state = STATE_IDLE
                self._busy = False
                self._busy_action = ""
                self._last_ack = "PLACE_DONE"
                self._last_release_mode_effective = str(release_mode)
            sender.send_line("ACK PLACE_DONE")
        except AbortRequested:
            self._recover_after_abort()
        except Exception as error:
            self._recover_from_error(ERR_HARDWARE_FAILURE, "PLACE failed: {0}".format(error), sender)

    def _recover_after_abort(self):
        try:
            self._set_state(STATE_RECOVERING, True, "abort")
            self.actuator.set_sucker(False)
            self._go_home(2.0, False)
            self._set_error(ERR_ABORTED, "Abort requested by operator.")
        except Exception as error:
            self._set_error(ERR_RECOVER_FAILED, "Abort recovery failed: {0}".format(error))

    def _recover_from_error(self, code, message, sender):
        try:
            self._set_state(STATE_RECOVERING, True, "recover")
            self.actuator.set_sucker(False)
            self._go_home(2.0, False)
        except Exception as recover_error:
            self._set_error(ERR_RECOVER_FAILED, "Recovery failed: {0}".format(recover_error))
            try:
                sender.send_line(format_error_line(ERR_RECOVER_FAILED, "Recovery failed: {0}".format(recover_error)))
            except Exception:
                pass
            return
        self._set_error(code, message)
        try:
            sender.send_line(format_error_line(code, message))
        except Exception:
            pass

    def _move_to(self, x, y, z, duration, allow_abort, settle_sec=None):
        if duration is None:
            current_x, current_y, current_z = self._get_position()
            distance = math.sqrt((x - current_x) ** 2 + (y - current_y) ** 2 + (z - current_z) ** 2)
            duration = max(self.motion_min_duration, distance / self.move_speed_xy)
        self.actuator.move_xyz(float(x), float(y), float(z), float(duration))
        self._commanded_pose = (float(x), float(y), float(z))
        self._commanded_cyl = cartesian_to_cylindrical(float(x), float(y), float(z))
        settle = self.motion_settle_sec if settle_sec is None else float(settle_sec)
        self._sleep_with_abort(float(duration) + settle, allow_abort)

    def _go_home(self, duration, allow_abort):
        self.actuator.go_home(float(duration))
        self._commanded_pose = (
            float(self.home_pose[0]),
            float(self.home_pose[1]),
            float(self.home_pose[2]),
        )
        self._sleep_with_abort(float(duration) + 0.1, allow_abort)

    def _sleep_with_abort(self, seconds, allow_abort=True):
        deadline = time.time() + max(0.0, float(seconds))
        while time.time() < deadline:
            if allow_abort and self._abort_event.is_set():
                raise AbortRequested()
            time.sleep(0.03)

    def _build_reference_pulses(self):
        try:
            angles = self.justkinematics.inverse(JETMAX_L4_MM, self.home_pose)
            self._reference_forearm_pitch_deg = self._estimate_forearm_pitch_deg(angles)
            pulses = tuple(float(value) for value in self.justkinematics.deg_to_pulse(angles))
            if len(pulses) != 3:
                return None
            return pulses
        except Exception:
            self._reference_forearm_pitch_deg = None
            return None

    def _build_auto_z_profile(self):
        candidate_map = {}
        radii = sample_range(self.radius_limits[0], self.radius_limits[1], self.auto_z_radius_step)
        z_candidates = sample_range(self.z_limits[0], self.z_limits[1], self.auto_z_height_step)
        posture_tolerance = max(0.0, float(self.auto_z_posture_tolerance))
        for radius in radii:
            target_z = self._auto_z_target(radius)
            valid_candidates = []
            for z_value in z_candidates:
                report = self._validate_cyl_target(0.0, radius, z_value)
                if not report["ok"]:
                    continue
                margin = float(report.get("margin", 0.0))
                neutral_distance = float(report.get("neutral_distance", abs(float(z_value) - self.auto_z_preferred)))
                posture_distance = float(report.get("posture_distance", neutral_distance))
                target_distance = abs(float(z_value) - float(target_z))
                valid_candidates.append((float(z_value), margin, neutral_distance, posture_distance, target_distance))
            if valid_candidates:
                candidate_map[float(radius)] = sorted(valid_candidates, key=lambda entry: entry[0])

        if not candidate_map:
            return ((self.radius_limits[0], self.z_carry),)

        def _filter_posture_candidates(candidates):
            best_posture_distance = min(abs(entry[3]) for entry in candidates)
            return [
                entry
                for entry in candidates
                if abs(entry[3]) <= best_posture_distance + posture_tolerance + 1e-6
            ]

        profile_desc = []
        next_selected_z = None
        for radius in sorted(candidate_map.keys(), reverse=True):
            candidates = candidate_map[radius]
            if next_selected_z is None:
                filtered = _filter_posture_candidates(candidates)
                selected = min(
                    filtered,
                    key=lambda entry: (
                        entry[4],
                        -entry[1],
                        abs(entry[0] - self.auto_z_preferred),
                        abs(entry[2]),
                        entry[0],
                    ),
                )[0]
            else:
                filtered = _filter_posture_candidates(candidates)
                selected = min(
                    filtered,
                    key=lambda entry: (
                        entry[4],
                        abs(entry[0] - float(next_selected_z)),
                        -entry[1],
                        abs(entry[0] - self.auto_z_preferred),
                        abs(entry[2]),
                        entry[0],
                    ),
                )[0]
            profile_desc.append((float(radius), float(selected)))
            next_selected_z = float(selected)
        return tuple(sorted(profile_desc, key=lambda entry: entry[0]))

    def _auto_z_target(self, radius):
        radius_value = float(radius)
        plateau_min = min(float(self.auto_z_plateau_min), float(self.auto_z_plateau_max))
        plateau_max = max(float(self.auto_z_plateau_min), float(self.auto_z_plateau_max))
        plateau_z = float(self.auto_z_plateau_z)
        if plateau_min <= radius_value <= plateau_max:
            return plateau_z
        if radius_value < plateau_min:
            delta_radius = plateau_min - radius_value
            return plateau_z - delta_radius * float(self.auto_z_retract_drop_per_radius)
        delta_radius = radius_value - plateau_max
        return plateau_z - delta_radius * float(self.auto_z_extend_drop_per_radius)

    def _validate_cyl_target(self, theta_deg, radius_mm, z_mm):
        theta_value = float(theta_deg)
        radius_value = float(radius_mm)
        z_value = float(z_mm)
        if not within_limits(theta_value, self.theta_limits):
            return {"ok": False, "message": "Theta {0:.2f} deg is outside limits.".format(theta_value)}
        if not within_limits(radius_value, self.radius_limits):
            return {"ok": False, "message": "Radius {0:.2f} mm is outside limits.".format(radius_value)}
        if not within_limits(z_value, self.z_limits):
            return {"ok": False, "message": "Height {0:.2f} mm is outside limits.".format(z_value)}
        x_mm, y_mm, z_cart = cylindrical_to_cartesian(theta_value, radius_value, z_value)
        return self._validate_pose_xyz(
            x_mm,
            y_mm,
            z_cart,
            enforce_workspace=self.cylindrical_xy_workspace_enabled,
        )

    def _validate_pose_xyz(self, x_mm, y_mm, z_mm, enforce_workspace=True):
        radius_mm = math.hypot(float(x_mm), float(y_mm))
        if radius_mm < MIN_RADIUS_MM:
            return {"ok": False, "message": "Radius {0:.2f} mm is below {1:.2f} mm.".format(radius_mm, MIN_RADIUS_MM)}
        if float(z_mm) > MAX_Z_MM:
            return {"ok": False, "message": "Height {0:.2f} mm exceeds {1:.2f} mm.".format(float(z_mm), MAX_Z_MM)}
        margin_candidates = [radius_mm - MIN_RADIUS_MM, MAX_Z_MM - float(z_mm)]
        if enforce_workspace:
            if not self._within_workspace(float(x_mm), float(y_mm)):
                return {"ok": False, "message": "Target ({0:.2f}, {1:.2f}) is outside workspace.".format(float(x_mm), float(y_mm))}
            margin_candidates.extend(
                [
                    float(x_mm) - self.x_min,
                    self.x_max - float(x_mm),
                    float(y_mm) - self.y_min,
                    self.y_max - float(y_mm),
                ]
            )
        margin = min(margin_candidates)
        neutral_distance = abs(float(z_mm) - self.auto_z_preferred)
        try:
            angles = self.justkinematics.inverse(JETMAX_L4_MM, (float(x_mm), float(y_mm), float(z_mm)))
            pulses = tuple(float(value) for value in self.justkinematics.deg_to_pulse(angles))
            if len(pulses) != 3 or not all(math.isfinite(value) for value in pulses):
                return {"ok": False, "message": "IK produced invalid servo pulses."}
            if pulses[0] < SERVO1_MIN_PULSE:
                return {"ok": False, "message": "Servo1 pulse {0:.2f} would be clamped low.".format(pulses[0])}
            if pulses[0] > SERVO1_MAX_PULSE:
                return {"ok": False, "message": "Servo1 pulse {0:.2f} would be clamped high.".format(pulses[0])}
            if pulses[1] < SERVO2_MIN_PULSE:
                return {"ok": False, "message": "Servo2 pulse {0:.2f} would be clamped low.".format(pulses[1])}
            if pulses[2] < SERVO3_MIN_PULSE:
                return {"ok": False, "message": "Servo3 pulse {0:.2f} would be clamped.".format(pulses[2])}
            if pulses[1] > SERVO2_MAX_PULSE:
                return {"ok": False, "message": "Servo2 pulse {0:.2f} would be clamped.".format(pulses[1])}
            if pulses[2] > SERVO3_MAX_PULSE:
                return {"ok": False, "message": "Servo3 pulse {0:.2f} would be clamped high.".format(pulses[2])}
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
        except Exception as error:
            return {"ok": False, "message": "IK failed: {0}".format(error)}
        return {
            "ok": True,
            "message": None,
            "margin": float(margin),
            "neutral_distance": float(neutral_distance),
            "posture_distance": float(posture_distance),
        }

    @staticmethod
    def _estimate_forearm_pitch_deg(angles):
        angle_values = tuple(float(value) for value in angles)
        if len(angle_values) < 3:
            return 0.0
        return float(angle_values[1] + angle_values[2])

    def _estimate_forearm_posture_distance(self, angles):
        forearm_pitch_deg = self._estimate_forearm_pitch_deg(angles)
        reference_pitch_deg = self._reference_forearm_pitch_deg
        if reference_pitch_deg is None:
            reference_pitch_deg = 0.0
        return abs(float(forearm_pitch_deg) - float(reference_pitch_deg))

    def _set_state(self, state, busy, busy_action):
        with self._lock:
            self._state = state
            self._busy = bool(busy)
            self._busy_action = str(busy_action or "")

    def _mark_control_kernel(self, kernel_name):
        with self._lock:
            self._control_kernel = str(kernel_name or self._control_kernel)

    def _set_error(self, code, message):
        with self._lock:
            self._state = STATE_ERROR
            self._busy = False
            self._busy_action = ""
            self._last_error_code = str(code)
            self._last_error = str(message)

    def _get_position(self):
        position = self._commanded_pose
        return (float(position[0]), float(position[1]), float(position[2]))

    def _has_calibration(self):
        try:
            params = self.rospy.get_param("/camera_cal/block_params", None)
        except Exception:
            return False
        return bool(params)

    def _within_workspace(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, float(value)))


class RobotGateway(object):
    def __init__(self, executor):
        self.executor = executor

    def handle(self, line, sender):
        text = str(line).strip()
        if not text:
            return None
        try:
            command, args = parse_command_text(text)
            snapshot = self.executor.snapshot()
            state = str(snapshot.get("state", ""))
            if state == STATE_ERROR and command not in {"PING", "STATUS", "ABORT", "RESET"}:
                return format_error_line(
                    ERR_INVALID_STATE,
                    "Robot is in ERROR state; only ABORT/RESET/STATUS/PING are allowed.",
                )
            if command == "PING":
                return "ACK PONG"
            if command == "STATUS":
                return "ACK STATUS {0}".format(json.dumps(snapshot, separators=(",", ":")))
            if command == "RESET":
                return self.executor.reset()
            if command == "ABORT":
                return self.executor.abort()
            if command == "MOVE":
                return self.executor.legacy_kernel.start_move(sender, float(args[0]), float(args[1]))
            if command == "MOVE_CYL":
                return self.executor.cylindrical_kernel.start_move(sender, float(args[0]), float(args[1]), float(args[2]))
            if command == "MOVE_CYL_AUTO":
                return self.executor.cylindrical_kernel.start_move_auto(sender, float(args[0]), float(args[1]))
            if command == "PICK":
                return self.executor.legacy_kernel.start_pick_pixel(sender, float(args[0]), float(args[1]))
            if command == "PICK_WORLD":
                return self.executor.legacy_kernel.start_pick_world(sender, float(args[0]), float(args[1]))
            if command == "PICK_CYL":
                return self.executor.cylindrical_kernel.start_pick(sender, float(args[0]), float(args[1]))
            if command == "PLACE":
                return self.executor.start_place(sender)
        except CommandParseError as error:
            if str(error) == "Empty command":
                return None
            return format_error_line(ERR_INVALID_STATE, str(error))
        except RobotCommandError as error:
            return format_error_line(error.code, error.message)
        except Exception as error:
            return format_error_line(ERR_HARDWARE_FAILURE, str(error))
        return format_error_line(ERR_INVALID_STATE, "Unsupported command: {0}".format(command))


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class RobotRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        sender = Sender(self.wfile)
        try:
            while True:
                raw = self.rfile.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                print("[client {0}] <= {1}".format(self.client_address[0], line), flush=True)
                response = self.server.gateway.handle(line, sender)
                if response:
                    sender.send_line(response)
                    print("[client {0}] => {1}".format(self.client_address[0], response), flush=True)
        finally:
            sender.close()


def main():
    parser = argparse.ArgumentParser(description="JetMax python3.6 TCP runtime")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()

    executor = JetMaxExecutor(
        limits={
            "x_min": -140.0,
            "x_max": 140.0,
            "y_min": -200.0,
            "y_max": -40.0,
            "cylindrical_xy_workspace_enabled": False,
            "theta_limits": (-120.0, 120.0),
            "radius_limits": (50.0, 280.0),
            "auto_radius_limits": (80.0, 260.0),
            "z_limits": (80.0, 212.8),
            "z_approach": 130.0,
            "z_pick": 85.0,
            "z_carry": 160.0,
            "move_speed_xy": 150.0,
            "motion_min_duration": 0.25,
            "motion_settle_sec": 0.08,
            "teleop_min_duration": 0.12,
            "teleop_settle_sec": 0.02,
            "auto_z_radius_step": 5.0,
            "auto_z_height_step": 5.0,
            "auto_z_preferred": 160.0,
            "auto_z_plateau_min": 145.0,
            "auto_z_plateau_max": 185.0,
            "auto_z_plateau_z": 205.0,
            "auto_z_retract_drop_per_radius": 0.8,
            "auto_z_extend_drop_per_radius": 0.4,
            "auto_z_posture_tolerance": 8.0,
            "auto_z_down_per_radius": 0.5,
            "auto_z_up_per_radius": 1.0,
            "auto_z_min_delta": 3.0,
        },
        home_pose=(0.0, -120.0, 160.0),
    )
    server = ThreadedTCPServer((args.host, int(args.port)), RobotRequestHandler)
    server.gateway = RobotGateway(executor)
    print("JetMax runtime listening on {0}:{1}".format(args.host, int(args.port)), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
