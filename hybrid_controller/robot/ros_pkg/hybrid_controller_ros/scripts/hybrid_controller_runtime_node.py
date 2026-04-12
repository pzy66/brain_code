#!/usr/bin/env python3

import os
import sys
import threading
import time
import math
from collections import deque

import rospy
from std_srvs.srv import Trigger, TriggerResponse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.environ.get("HYBRID_CONTROLLER_REPO_ROOT") or os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RUNTIME_DIR = os.path.join(REPO_ROOT, "hybrid_controller", "robot", "runtime")
if RUNTIME_DIR not in sys.path:
    sys.path.insert(0, RUNTIME_DIR)

from robot_runtime_py36 import JetMaxExecutor, cylindrical_to_cartesian, interpolate_auto_z
from hybrid_controller_ros.msg import CylindricalTeleop, RobotState
from hybrid_controller_ros.srv import (
    GetPickTuning,
    GetPickTuningResponse,
    MoveCyl,
    MoveCylAuto,
    MoveCylAutoResponse,
    MoveCylResponse,
    PickCyl,
    PickCylResponse,
    PickWorld,
    PickWorldResponse,
    SetPickTuning,
    SetPickTuningResponse,
)


class CylindricalPose(object):
    def __init__(self, theta_deg=0.0, radius_mm=0.0, z_mm=0.0):
        self.theta_deg = float(theta_deg)
        self.radius_mm = float(radius_mm)
        self.z_mm = float(z_mm)

    def normalized(self):
        return CylindricalPose(self.theta_deg, self.radius_mm, self.z_mm)


class _TeleopStep(object):
    def __init__(self, pose, stale):
        self.pose = pose
        self.stale = bool(stale)


class _TeleopKernel(object):
    def __init__(
        self,
        theta_limits_deg,
        radius_limits_mm,
        auto_z_profile,
        validator,
        tick_hz=20.0,
        deadman_timeout_sec=0.2,
        theta_accel_deg_s2=220.0,
        radius_accel_mm_s2=220.0,
    ):
        self.theta_limits_deg = (float(theta_limits_deg[0]), float(theta_limits_deg[1]))
        self.radius_limits_mm = (float(radius_limits_mm[0]), float(radius_limits_mm[1]))
        self.auto_z_profile = tuple((float(radius), float(z_mm)) for radius, z_mm in auto_z_profile)
        self.validator = validator
        self.tick_hz = max(float(tick_hz), 1.0)
        self.tick_sec = 1.0 / self.tick_hz
        self.deadman_timeout_sec = max(float(deadman_timeout_sec), self.tick_sec)
        self.theta_accel_deg_s2 = max(float(theta_accel_deg_s2), 1.0)
        self.radius_accel_mm_s2 = max(float(radius_accel_mm_s2), 1.0)
        self.theta_rate_deg_s = 0.0
        self.radius_rate_mm_s = 0.0
        self.command_theta_rate_deg_s = 0.0
        self.command_radius_rate_mm_s = 0.0
        self.command_enabled = False
        self.command_ts = time.monotonic()

    def update_command(self, theta_rate_deg_s=0.0, radius_rate_mm_s=0.0, enabled=False, timestamp=None):
        self.command_theta_rate_deg_s = float(theta_rate_deg_s)
        self.command_radius_rate_mm_s = float(radius_rate_mm_s)
        self.command_enabled = bool(enabled)
        self.command_ts = float(time.monotonic() if timestamp is None else timestamp)

    def clear_command(self, timestamp=None):
        self.update_command(0.0, 0.0, False, timestamp=timestamp)

    def step(self, current_pose, now=None):
        current_time = float(time.monotonic() if now is None else now)
        stale = (current_time - float(self.command_ts)) > self.deadman_timeout_sec
        target_theta_rate = 0.0 if stale or not self.command_enabled else float(self.command_theta_rate_deg_s)
        target_radius_rate = 0.0 if stale or not self.command_enabled else float(self.command_radius_rate_mm_s)
        self.theta_rate_deg_s = self._ramp(
            self.theta_rate_deg_s,
            target_theta_rate,
            self.theta_accel_deg_s2 * self.tick_sec,
        )
        self.radius_rate_mm_s = self._ramp(
            self.radius_rate_mm_s,
            target_radius_rate,
            self.radius_accel_mm_s2 * self.tick_sec,
        )
        if abs(self.theta_rate_deg_s) < 1e-6 and abs(self.radius_rate_mm_s) < 1e-6:
            return None

        next_theta = self._clamp(
            float(current_pose.theta_deg) + self.theta_rate_deg_s * self.tick_sec,
            self.theta_limits_deg,
        )
        next_radius = self._clamp(
            float(current_pose.radius_mm) + self.radius_rate_mm_s * self.tick_sec,
            self.radius_limits_mm,
        )
        next_z = float(interpolate_auto_z(self.auto_z_profile, next_radius))
        validation = self.validator(next_theta, next_radius, next_z)
        if not bool(validation.get("ok", False)):
            self.theta_rate_deg_s = 0.0
            self.radius_rate_mm_s = 0.0
            return None
        pose = CylindricalPose(next_theta, next_radius, next_z).normalized()
        return _TeleopStep(pose, stale)

    @staticmethod
    def _ramp(current, target, max_delta):
        delta = float(target) - float(current)
        if abs(delta) <= float(max_delta):
            return float(target)
        return float(current) + math.copysign(float(max_delta), delta)

    @staticmethod
    def _clamp(value, limits):
        return max(float(limits[0]), min(float(limits[1]), float(value)))


class _BufferedSender(object):
    def __init__(self):
        self.lines = []
        self.condition = threading.Condition()

    def send_line(self, line):
        with self.condition:
            self.lines.append(str(line).strip())
            self.condition.notify_all()

    def wait_for_line(self, timeout):
        deadline = time.time() + float(timeout)
        with self.condition:
            while True:
                if self.lines:
                    return self.lines.pop(0)
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self.condition.wait(timeout=remaining)


class _NoopSender(object):
    def send_line(self, _line):
        return None


class _QueuedAction(object):
    def __init__(self, name, start_fn, args):
        self.name = str(name)
        self.start_fn = start_fn
        self.args = tuple(args)
        self.enqueued_ts = float(time.time())


class HybridControllerRuntimeNode(object):
    def __init__(self):
        self.executor = JetMaxExecutor(
            limits={
                "x_min": -140.0,
                "x_max": 140.0,
                "y_min": -200.0,
                "y_max": -40.0,
                "cylindrical_xy_workspace_enabled": False,
                "z_approach": 130.0,
                "z_pick": 85.0,
                "z_carry": 160.0,
                "pick_approach_z_mm": 130.0,
                "pick_descend_z_mm": 85.0,
                "pick_pre_suction_sec": 0.25,
                "pick_bottom_hold_sec": 0.15,
                "pick_lift_sec": 0.8,
                "place_descend_z_mm": 85.0,
                "place_release_mode": "release",
                "place_release_sec": 0.25,
                "place_post_release_hold_sec": 0.10,
                "z_carry_floor_mm": 160.0,
                "move_speed_xy": 150.0,
                "theta_limits": (-120.0, 120.0),
                "radius_limits": (50.0, 280.0),
                "auto_radius_limits": (80.0, 260.0),
                "z_limits": (80.0, 212.8),
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
                "motion_min_duration": 0.25,
                "motion_settle_sec": 0.08,
                "teleop_min_duration": 0.12,
                "teleop_settle_sec": 0.02,
            },
            home_pose=(0.0, -120.0, 160.0),
        )
        self.teleop_kernel = _TeleopKernel(
            theta_limits_deg=self.executor.theta_limits,
            radius_limits_mm=self.executor.auto_radius_limits,
            auto_z_profile=self.executor._auto_z_profile,
            validator=self._validate_cyl_target_for_teleop,
            tick_hz=20.0,
            deadman_timeout_sec=0.6,
            theta_accel_deg_s2=180.0,
            radius_accel_mm_s2=200.0,
        )
        self._teleop_command_duration_sec = 0.08
        self._teleop_active = False
        self._teleop_validate_counter = 0
        self._teleop_full_validate_every_ticks = 6
        self._teleop_last_cmd_seq = 0
        self._teleop_last_client_ts = 0.0
        self._state_seq = 0
        self._last_state_signature = None
        self._null_sender = _NoopSender()
        self._action_submit_lock = threading.Lock()
        self._action_queue = deque()
        self._action_queue_lock = threading.Lock()
        self._action_queue_event = threading.Event()
        self._action_worker_stop = threading.Event()
        self._action_worker_thread = threading.Thread(
            target=self._action_worker_loop,
            name="hybrid-controller-action-queue",
            daemon=True,
        )
        self._action_worker_thread.start()
        # Keep ROS queues minimal so we always process freshest state/teleop data
        # under Wi-Fi jitter instead of draining stale backlog.
        self._state_pub = rospy.Publisher("/hybrid_controller/state", RobotState, queue_size=1)
        rospy.Subscriber("/hybrid_controller/teleop_cyl_cmd", CylindricalTeleop, self._on_teleop_cmd, queue_size=1)
        rospy.Service("/hybrid_controller/reset", Trigger, self._handle_reset)
        rospy.Service("/hybrid_controller/abort", Trigger, self._handle_abort)
        rospy.Service("/hybrid_controller/place", Trigger, self._handle_place)
        rospy.Service("/hybrid_controller/move_cyl", MoveCyl, self._handle_move_cyl)
        rospy.Service("/hybrid_controller/move_cyl_auto", MoveCylAuto, self._handle_move_cyl_auto)
        rospy.Service("/hybrid_controller/pick_cyl", PickCyl, self._handle_pick_cyl)
        rospy.Service("/hybrid_controller/pick_world", PickWorld, self._handle_pick_world)
        rospy.Service("/hybrid_controller/get_pick_tuning", GetPickTuning, self._handle_get_pick_tuning)
        rospy.Service("/hybrid_controller/set_pick_tuning", SetPickTuning, self._handle_set_pick_tuning)
        rospy.Timer(rospy.Duration(0.05), self._on_teleop_tick)
        rospy.Timer(rospy.Duration(1.0 / 15.0), self._publish_state)

    def _on_teleop_cmd(self, message):
        cmd_seq = int(getattr(message, "cmd_seq", 0) or 0)
        if cmd_seq > 0 and self._teleop_last_cmd_seq > 0 and cmd_seq <= self._teleop_last_cmd_seq:
            return
        if cmd_seq > 0:
            self._teleop_last_cmd_seq = int(cmd_seq)
        client_ts = float(getattr(message, "client_ts", 0.0) or 0.0)
        if client_ts > 0.0:
            if self._teleop_last_client_ts > 0.0 and client_ts < self._teleop_last_client_ts:
                return
            if (time.time() - client_ts) > 2.0:
                return
            self._teleop_last_client_ts = float(client_ts)
        self.teleop_kernel.update_command(
            theta_rate_deg_s=float(message.theta_rate_deg_s),
            radius_rate_mm_s=float(message.radius_rate_mm_s),
            enabled=bool(message.enabled),
            timestamp=time.monotonic(),
        )

    def _validate_cyl_target_for_teleop(self, theta_deg, radius_mm, z_mm):
        theta_value = float(theta_deg)
        radius_value = float(radius_mm)
        z_value = float(z_mm)
        if not (float(self.executor.theta_limits[0]) <= theta_value <= float(self.executor.theta_limits[1])):
            return {"ok": False, "message": "theta_out_of_limits"}
        if not (float(self.executor.auto_radius_limits[0]) <= radius_value <= float(self.executor.auto_radius_limits[1])):
            return {"ok": False, "message": "radius_out_of_limits"}
        if not (float(self.executor.z_limits[0]) <= z_value <= float(self.executor.z_limits[1])):
            return {"ok": False, "message": "z_out_of_limits"}
        if bool(self.executor.cylindrical_xy_workspace_enabled):
            x_mm, y_mm, _ = cylindrical_to_cartesian(theta_value, radius_value, z_value)
            if not bool(self.executor._within_workspace(float(x_mm), float(y_mm))):
                return {"ok": False, "message": "workspace_out_of_limits"}
        self._teleop_validate_counter += 1
        if self._teleop_validate_counter >= int(self._teleop_full_validate_every_ticks):
            self._teleop_validate_counter = 0
            return self.executor._validate_cyl_target(theta_value, radius_value, z_value)
        return {"ok": True, "message": None, "margin": 0.0, "neutral_distance": 0.0, "posture_distance": 0.0}

    def _on_teleop_tick(self, _event):
        with self.executor._lock:
            state = str(self.executor._state)
            current_commanded_cyl = tuple(self.executor._commanded_cyl)
        if state.startswith("PICK") or state.startswith("PLACE") or state in {"ERROR", "RECOVERING"}:
            self._stop_teleop()
            return
        current_pose = CylindricalPose(
            theta_deg=float(current_commanded_cyl[0]),
            radius_mm=float(current_commanded_cyl[1]),
            z_mm=float(current_commanded_cyl[2]),
        )
        step = self.teleop_kernel.step(current_pose, now=time.monotonic())
        if step is None:
            self._stop_teleop()
            return
        target_pose = step.pose.normalized()
        current_xyz = cylindrical_to_cartesian(current_pose.theta_deg, current_pose.radius_mm, current_pose.z_mm)
        target_xyz = cylindrical_to_cartesian(target_pose.theta_deg, target_pose.radius_mm, target_pose.z_mm)
        dx = float(target_xyz[0] - current_xyz[0])
        dy = float(target_xyz[1] - current_xyz[1])
        dz = float(target_xyz[2] - current_xyz[2])
        self.executor._mark_control_kernel("cylindrical_ros_teleop")
        try:
            self.executor.actuator.move_relative_xyz(dx, dy, dz, self._teleop_command_duration_sec)
        except Exception:
            self.executor.actuator.move_xyz(
                target_xyz[0],
                target_xyz[1],
                target_xyz[2],
                self._teleop_command_duration_sec,
            )
        with self.executor._lock:
            self.executor._commanded_pose = (float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]))
            self.executor._commanded_cyl = (float(target_pose.theta_deg), float(target_pose.radius_mm), float(target_pose.z_mm))
            self.executor._state = "MOVING_XY"
            self.executor._busy = False
            self.executor._busy_action = "teleop"
        self._teleop_active = True

    def _stop_teleop(self):
        self.teleop_kernel.clear_command(timestamp=time.monotonic())
        if not self._teleop_active:
            return
        with self.executor._lock:
            if self.executor._busy_action == "teleop":
                self.executor._state = "CARRY_READY" if self.executor._carrying else "IDLE"
                self.executor._busy = False
                self.executor._busy_action = ""
        self._teleop_active = False

    def _handle_move_cyl(self, request):
        ok, message = self._submit_or_queue_action(
            "move_cyl",
            self.executor.start_move_cyl,
            float(request.theta_deg),
            float(request.radius_mm),
            float(request.z_mm),
        )
        return MoveCylResponse(ok=ok, message=message)

    def _handle_move_cyl_auto(self, request):
        ok, message = self._submit_or_queue_action(
            "move_cyl_auto",
            self.executor.start_move_cyl_auto,
            float(request.theta_deg),
            float(request.radius_mm),
        )
        return MoveCylAutoResponse(ok=ok, message=message)

    def _handle_pick_cyl(self, request):
        ok, message = self._submit_or_queue_action(
            "pick_cyl",
            self.executor.start_pick_cyl,
            float(request.theta_deg),
            float(request.radius_mm),
        )
        return PickCylResponse(ok=ok, message=message)

    def _handle_pick_world(self, request):
        ok, message = self._submit_or_queue_action(
            "pick_world",
            self.executor.start_pick_world,
            float(request.x_mm),
            float(request.y_mm),
        )
        return PickWorldResponse(ok=ok, message=message)

    def _handle_get_pick_tuning(self, _request):
        tuning = self.executor.get_pick_tuning()
        response = GetPickTuningResponse()
        response.ok = True
        response.message = "ok"
        response.pick_approach_z_mm = float(tuning.get("pick_approach_z_mm", 0.0))
        response.pick_descend_z_mm = float(tuning.get("pick_descend_z_mm", 0.0))
        response.pick_pre_suction_sec = float(tuning.get("pick_pre_suction_sec", 0.0))
        response.pick_bottom_hold_sec = float(tuning.get("pick_bottom_hold_sec", 0.0))
        response.pick_lift_sec = float(tuning.get("pick_lift_sec", 0.0))
        response.place_descend_z_mm = float(tuning.get("place_descend_z_mm", 0.0))
        response.place_release_mode = str(tuning.get("place_release_mode", "release"))
        response.place_release_sec = float(tuning.get("place_release_sec", 0.0))
        response.place_post_release_hold_sec = float(tuning.get("place_post_release_hold_sec", 0.0))
        response.z_carry_floor_mm = float(tuning.get("z_carry_floor_mm", 0.0))
        return response

    def _handle_set_pick_tuning(self, request):
        payload = {
            "pick_approach_z_mm": float(request.pick_approach_z_mm),
            "pick_descend_z_mm": float(request.pick_descend_z_mm),
            "pick_pre_suction_sec": float(request.pick_pre_suction_sec),
            "pick_bottom_hold_sec": float(request.pick_bottom_hold_sec),
            "pick_lift_sec": float(request.pick_lift_sec),
            "place_descend_z_mm": float(request.place_descend_z_mm),
            "place_release_mode": str(request.place_release_mode),
            "place_release_sec": float(request.place_release_sec),
            "place_post_release_hold_sec": float(request.place_post_release_hold_sec),
            "z_carry_floor_mm": float(request.z_carry_floor_mm),
        }
        try:
            self.executor.set_pick_tuning(payload)
            return SetPickTuningResponse(ok=True, message="ok")
        except Exception as error:
            return SetPickTuningResponse(ok=False, message=str(error))

    def _handle_place(self, _request):
        ok, message = self._submit_or_queue_action("place", self.executor.start_place)
        return TriggerResponse(success=bool(ok), message=str(message))

    def _handle_abort(self, _request):
        self._stop_teleop()
        self._clear_action_queue()
        response = self.executor.abort()
        ok = str(response).strip().upper().startswith("ACK")
        return TriggerResponse(success=ok, message=str(response))

    def _handle_reset(self, _request):
        self._stop_teleop()
        self._clear_action_queue()
        response = self.executor.reset()
        ok = str(response).strip().upper().startswith("ACK")
        return TriggerResponse(success=ok, message=str(response))

    def _submit_or_queue_action(self, action_name, start_fn, *args):
        self._stop_teleop()
        with self._action_submit_lock:
            immediate = start_fn(self._null_sender, *args)
            if immediate == "BUSY":
                self._enqueue_action(_QueuedAction(action_name, start_fn, args))
                return True, "ACCEPTED queued:{0}".format(str(action_name))
            if isinstance(immediate, str) and str(immediate).startswith("ERR"):
                return False, str(immediate)
            if isinstance(immediate, str) and str(immediate).startswith("ACK"):
                return True, str(immediate)
            return True, "ACCEPTED {0}".format(str(action_name))

    def _enqueue_action(self, action):
        with self._action_queue_lock:
            self._action_queue.append(action)
            self._action_queue_event.set()

    def _clear_action_queue(self):
        with self._action_queue_lock:
            self._action_queue.clear()
            self._action_queue_event.clear()

    def _action_worker_loop(self):
        while not rospy.is_shutdown() and not self._action_worker_stop.is_set():
            self._action_queue_event.wait(timeout=0.2)
            if rospy.is_shutdown() or self._action_worker_stop.is_set():
                return
            action = None
            with self._action_queue_lock:
                if self._action_queue:
                    action = self._action_queue.popleft()
                if not self._action_queue:
                    self._action_queue_event.clear()
            if action is None:
                continue
            self._run_queued_action(action)

    def _run_queued_action(self, action):
        while not rospy.is_shutdown() and not self._action_worker_stop.is_set():
            immediate = action.start_fn(self._null_sender, *action.args)
            if immediate == "BUSY":
                time.sleep(0.03)
                continue
            if isinstance(immediate, str) and str(immediate).startswith("ERR"):
                rospy.logwarn("Queued action %s rejected: %s", action.name, immediate)
                return
            rospy.loginfo("Queued action %s started.", action.name)
            return

    def _publish_state(self, _event):
        snapshot = self.executor.snapshot()
        signature = (
            str(snapshot.get("state", "")),
            bool(snapshot.get("busy", False)),
            str(snapshot.get("busy_action", "")),
            bool(snapshot.get("carrying", False)),
            str(snapshot.get("last_ack", "")),
            str(snapshot.get("last_error_code", "")),
            str(snapshot.get("last_error", "")),
            tuple(snapshot.get("robot_xy", [0.0, 0.0])),
            float(snapshot.get("robot_z", 0.0) or 0.0),
        )
        if signature != self._last_state_signature:
            self._state_seq += 1
            self._last_state_signature = signature
        msg = RobotState()
        msg.state = str(snapshot.get("state", ""))
        msg.state_seq = int(self._state_seq)
        msg.robot_ts = float(time.time())
        msg.busy = bool(snapshot.get("busy", False))
        msg.carrying = bool(snapshot.get("carrying", False))
        robot_cyl = snapshot.get("robot_cyl", {})
        robot_xy = snapshot.get("robot_xy", [0.0, 0.0])
        limits_cyl = snapshot.get("limits_cyl", {})
        limits_cyl_auto = snapshot.get("limits_cyl_auto", {})
        msg.theta_deg = float(robot_cyl.get("theta_deg", 0.0))
        msg.radius_mm = float(robot_cyl.get("radius_mm", 0.0))
        msg.z_mm = float(robot_cyl.get("z_mm", snapshot.get("robot_z", 0.0)))
        msg.x_mm = float(robot_xy[0])
        msg.y_mm = float(robot_xy[1])
        msg.auto_z_current = float(snapshot.get("auto_z_current", msg.z_mm))
        msg.theta_min_deg = float(limits_cyl.get("theta_deg", [0.0, 0.0])[0])
        msg.theta_max_deg = float(limits_cyl.get("theta_deg", [0.0, 0.0])[1])
        msg.radius_min_mm = float(limits_cyl.get("radius_mm", [0.0, 0.0])[0])
        msg.radius_max_mm = float(limits_cyl.get("radius_mm", [0.0, 0.0])[1])
        msg.auto_radius_min_mm = float(limits_cyl_auto.get("radius_mm", [0.0, 0.0])[0])
        msg.auto_radius_max_mm = float(limits_cyl_auto.get("radius_mm", [0.0, 0.0])[1])
        msg.control_kernel = str(snapshot.get("control_kernel", ""))
        msg.busy_action = str(snapshot.get("busy_action", ""))
        msg.last_error_code = str(snapshot.get("last_error_code", ""))
        msg.last_error_message = str(snapshot.get("last_error", ""))
        msg.calibration_ready = bool(snapshot.get("calibration_ready", False))
        msg.ik_valid = bool(snapshot.get("ik_valid", True))
        msg.validation_error = str(snapshot.get("validation_error", ""))
        pick_tuning = snapshot.get("pick_tuning", {}) if isinstance(snapshot, dict) else {}
        msg.pick_approach_z_mm = float(pick_tuning.get("pick_approach_z_mm", snapshot.get("approach_z", 0.0)))
        msg.pick_descend_z_mm = float(pick_tuning.get("pick_descend_z_mm", snapshot.get("pick_z", 0.0)))
        msg.pick_pre_suction_sec = float(pick_tuning.get("pick_pre_suction_sec", 0.0))
        msg.pick_bottom_hold_sec = float(pick_tuning.get("pick_bottom_hold_sec", 0.0))
        msg.pick_lift_sec = float(pick_tuning.get("pick_lift_sec", 0.0))
        msg.place_descend_z_mm = float(pick_tuning.get("place_descend_z_mm", snapshot.get("pick_z", 0.0)))
        msg.place_release_mode = str(pick_tuning.get("place_release_mode", "release"))
        msg.place_release_sec = float(pick_tuning.get("place_release_sec", 0.0))
        msg.place_post_release_hold_sec = float(pick_tuning.get("place_post_release_hold_sec", 0.0))
        msg.z_carry_floor_mm = float(pick_tuning.get("z_carry_floor_mm", snapshot.get("carry_z", 0.0)))
        msg.post_pick_settle_z = float(snapshot.get("post_pick_settle_z", 0.0) or 0.0)
        msg.release_mode_effective = str(snapshot.get("release_mode_effective", ""))
        msg.last_ack = str(snapshot.get("last_ack", ""))
        self._state_pub.publish(msg)


def main():
    rospy.init_node("hybrid_controller_runtime_node", anonymous=True, disable_signals=True)
    HybridControllerRuntimeNode()
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
