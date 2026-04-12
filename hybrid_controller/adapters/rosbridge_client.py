from __future__ import annotations

import ast
import json
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from hybrid_controller.controller.events import Event

try:
    import roslibpy
except ImportError:  # pragma: no cover - optional dependency
    roslibpy = None

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


@dataclass(frozen=True, slots=True)
class RosServiceResult:
    ok: bool
    message: str
    raw: dict[str, object]


class RosbridgeClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        state_callback: Optional[Callable[[dict[str, object]], None]] = None,
        event_callback: Optional[Callable[[Event], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.state_callback = state_callback
        self.event_callback = event_callback
        self.status_callback = status_callback
        self._ros = None
        self._teleop_topic = None
        self._state_topic = None
        self._services: dict[str, object] = {}
        self._connected = False
        self._last_state: dict[str, object] | None = None
        self._state_lock = threading.Lock()
        self._last_state_seq: int = 0
        self._last_robot_ts: float = 0.0

    def connect(self) -> None:
        if roslibpy is None:
            raise RuntimeError("roslibpy is not installed in the current environment.")
        if self._ros is not None and self.is_connected():
            return
        if self._ros is not None:
            try:
                self._ros.close()
            except Exception:
                pass
            self._ros = None
        self._ros = roslibpy.Ros(host=self.host, port=self.port)
        self._ros.on_ready(self._on_ready, run_in_thread=False)
        self._ros.on("error", self._on_error)
        self._ros.on("close", self._on_close)
        self._ros.run()

    def close(self) -> None:
        if self._state_topic is not None:
            try:
                self._state_topic.unsubscribe()
            except Exception:
                pass
            self._state_topic = None
        if self._teleop_topic is not None:
            try:
                self._teleop_topic.unadvertise()
            except Exception:
                pass
            self._teleop_topic = None
        ros = self._ros
        self._ros = None
        self._services = {}
        self._connected = False
        with self._state_lock:
            self._last_state = None
            self._last_state_seq = 0
            self._last_robot_ts = 0.0
        if ros is not None:
            try:
                # Do not terminate the underlying Twisted reactor here.
                # `terminate()` makes subsequent `run()` calls in the same process fail with
                # `twisted.internet.error.ReactorNotRestartable` during reconnect flows.
                ros.close()
            except Exception:
                pass

    def is_connected(self) -> bool:
        ros = self._ros
        return bool(self._connected and ros is not None and ros.is_connected)

    def publish_teleop(
        self,
        *,
        theta_rate_deg_s: float,
        radius_rate_mm_s: float,
        enabled: bool,
        cmd_seq: int = 0,
        client_ts: float = 0.0,
    ) -> None:
        if self._teleop_topic is None:
            raise RuntimeError("ROS teleop topic is not ready.")
        message = roslibpy.Message(
            {
                "theta_rate_deg_s": float(theta_rate_deg_s),
                "radius_rate_mm_s": float(radius_rate_mm_s),
                "enabled": bool(enabled),
                "cmd_seq": int(max(0, int(cmd_seq))),
                "client_ts": float(client_ts),
            }
        )
        self._teleop_topic.publish(message)

    def stop_teleop(self) -> None:
        if self._teleop_topic is None:
            return
        self.publish_teleop(theta_rate_deg_s=0.0, radius_rate_mm_s=0.0, enabled=False)

    def send_move_cyl(self, theta_deg: float, radius_mm: float, z_mm: float, *, callback: Optional[Callable[[RosServiceResult], None]] = None) -> None:
        self._call_service("move_cyl", {"theta_deg": float(theta_deg), "radius_mm": float(radius_mm), "z_mm": float(z_mm)}, callback=callback)

    def send_move_cyl_auto(self, theta_deg: float, radius_mm: float, *, callback: Optional[Callable[[RosServiceResult], None]] = None) -> None:
        self._call_service("move_cyl_auto", {"theta_deg": float(theta_deg), "radius_mm": float(radius_mm)}, callback=callback)

    def send_pick_cyl(self, theta_deg: float, radius_mm: float, *, callback: Optional[Callable[[RosServiceResult], None]] = None) -> None:
        self._call_service("pick_cyl", {"theta_deg": float(theta_deg), "radius_mm": float(radius_mm)}, callback=callback)

    def send_pick_world(self, x_mm: float, y_mm: float, *, callback: Optional[Callable[[RosServiceResult], None]] = None) -> None:
        self._call_service("pick_world", {"x_mm": float(x_mm), "y_mm": float(y_mm)}, callback=callback)

    def get_pick_tuning(self, *, callback: Optional[Callable[[RosServiceResult], None]] = None) -> None:
        self._call_service("get_pick_tuning", {}, callback=callback)

    def set_pick_tuning(
        self,
        tuning: dict[str, object],
        *,
        callback: Optional[Callable[[RosServiceResult], None]] = None,
    ) -> None:
        payload = {
            "pick_approach_z_mm": float(tuning.get("pick_approach_z_mm", 0.0)),
            "pick_descend_z_mm": float(tuning.get("pick_descend_z_mm", 0.0)),
            "pick_pre_suction_sec": float(tuning.get("pick_pre_suction_sec", 0.0)),
            "pick_bottom_hold_sec": float(tuning.get("pick_bottom_hold_sec", 0.0)),
            "pick_lift_sec": float(tuning.get("pick_lift_sec", 0.0)),
            "place_descend_z_mm": float(tuning.get("place_descend_z_mm", 0.0)),
            "place_release_mode": str(tuning.get("place_release_mode", "release")),
            "place_release_sec": float(tuning.get("place_release_sec", 0.0)),
            "place_post_release_hold_sec": float(tuning.get("place_post_release_hold_sec", 0.0)),
            "z_carry_floor_mm": float(tuning.get("z_carry_floor_mm", 0.0)),
        }
        self._call_service("set_pick_tuning", payload, callback=callback)

    def send_place(self, *, callback: Optional[Callable[[RosServiceResult], None]] = None) -> None:
        self._call_trigger("place", callback=callback)

    def send_abort(self, *, callback: Optional[Callable[[RosServiceResult], None]] = None) -> None:
        self._call_trigger("abort", callback=callback)

    def send_reset(self, *, callback: Optional[Callable[[RosServiceResult], None]] = None) -> None:
        self._call_trigger("reset", callback=callback)

    def get_param(self, name: str, *, default: object = None, timeout_sec: float = 2.0) -> object:
        service = self._services.get("rosapi_get_param")
        if service is None:
            raise RuntimeError("ROS service 'rosapi_get_param' is not ready.")
        if roslibpy is None:
            raise RuntimeError("roslibpy is not installed in the current environment.")

        request = roslibpy.ServiceRequest(
            {
                "name": str(name),
                "default": "" if default is None else json.dumps(default, ensure_ascii=False),
            }
        )
        result_holder: dict[str, object] = {"done": False, "value": None, "error": None}
        done = threading.Event()

        def on_success(response: dict[str, object]) -> None:
            result_holder["done"] = True
            result_holder["value"] = response.get("value")
            done.set()

        def on_error(error: object) -> None:
            result_holder["done"] = True
            result_holder["error"] = str(error)
            done.set()

        service.call(request, callback=on_success, errback=on_error)
        if not done.wait(timeout=float(timeout_sec)):
            raise RuntimeError(f"Timed out reading ROS param '{name}'.")
        if result_holder["error"] is not None:
            raise RuntimeError(str(result_holder["error"]))
        return self._parse_ros_param_value(result_holder["value"], default=default)

    def _call_trigger(self, name: str, *, callback: Optional[Callable[[RosServiceResult], None]]) -> None:
        self._call_service(name, {}, callback=callback)

    def _call_service(
        self,
        name: str,
        args: dict[str, object],
        *,
        callback: Optional[Callable[[RosServiceResult], None]],
    ) -> None:
        service = self._services.get(name)
        if service is None:
            raise RuntimeError(f"ROS service '{name}' is not ready.")
        request = roslibpy.ServiceRequest(args)

        def on_success(response: dict[str, object]) -> None:
            result = RosServiceResult(
                ok=bool(response.get("ok", response.get("success", False))),
                message=str(response.get("message", "")),
                raw=dict(response),
            )
            if callback is not None:
                callback(result)

        def on_error(error: object) -> None:
            result = RosServiceResult(ok=False, message=str(error), raw={"error": str(error)})
            if callback is not None:
                callback(result)

        service.call(request, callback=on_success, errback=on_error)

    def _on_ready(self, *args: object) -> None:
        assert roslibpy is not None
        assert self._ros is not None
        self._connected = True
        self._teleop_topic = roslibpy.Topic(
            self._ros,
            "/hybrid_controller/teleop_cyl_cmd",
            "hybrid_controller_ros/CylindricalTeleop",
            queue_size=1,
        )
        self._teleop_topic.advertise()
        self._state_topic = roslibpy.Topic(
            self._ros,
            "/hybrid_controller/state",
            "hybrid_controller_ros/RobotState",
            queue_length=1,
        )
        self._state_topic.subscribe(self._handle_state)
        self._services = {
            "reset": roslibpy.Service(self._ros, "/hybrid_controller/reset", "std_srvs/Trigger"),
            "abort": roslibpy.Service(self._ros, "/hybrid_controller/abort", "std_srvs/Trigger"),
            "place": roslibpy.Service(self._ros, "/hybrid_controller/place", "std_srvs/Trigger"),
            "move_cyl": roslibpy.Service(self._ros, "/hybrid_controller/move_cyl", "hybrid_controller_ros/MoveCyl"),
            "move_cyl_auto": roslibpy.Service(self._ros, "/hybrid_controller/move_cyl_auto", "hybrid_controller_ros/MoveCylAuto"),
            "pick_cyl": roslibpy.Service(self._ros, "/hybrid_controller/pick_cyl", "hybrid_controller_ros/PickCyl"),
            "pick_world": roslibpy.Service(self._ros, "/hybrid_controller/pick_world", "hybrid_controller_ros/PickWorld"),
            "get_pick_tuning": roslibpy.Service(self._ros, "/hybrid_controller/get_pick_tuning", "hybrid_controller_ros/GetPickTuning"),
            "set_pick_tuning": roslibpy.Service(self._ros, "/hybrid_controller/set_pick_tuning", "hybrid_controller_ros/SetPickTuning"),
            "rosapi_get_param": roslibpy.Service(self._ros, "/rosapi/get_param", "rosapi/GetParam"),
        }
        self._emit_status("ROS bridge connected.")

    def _on_error(self, error: object) -> None:
        self._emit_status(f"ROS bridge error: {error}")

    def _on_close(self, *args: object) -> None:
        self._connected = False
        self._emit_status("ROS bridge closed.")
        if self.event_callback is not None:
            self.event_callback(Event(source="robot", type="robot_disconnected", value="ROS bridge connection lost"))

    def _handle_state(self, message: dict[str, object]) -> None:
        snapshot = self._message_to_snapshot(message)
        current_seq = int(snapshot.get("state_seq", 0) or 0)
        current_robot_ts = float(snapshot.get("robot_ts", 0.0) or 0.0)
        with self._state_lock:
            previous = None if self._last_state is None else dict(self._last_state)
            if current_seq > 0 and self._last_state_seq > 0:
                if current_seq < self._last_state_seq:
                    return
                # Runtime node publishes a heartbeat at fixed rate and only increments
                # state_seq when the semantic state changes.
                # Accept equal-seq heartbeats when robot_ts moves forward.
                if current_seq == self._last_state_seq:
                    if current_robot_ts <= 0.0:
                        return
                    if self._last_robot_ts > 0.0 and current_robot_ts <= self._last_robot_ts:
                        return
            if current_seq <= 0 and current_robot_ts > 0.0 and self._last_robot_ts > 0.0 and current_robot_ts < self._last_robot_ts:
                return
            self._last_state = dict(snapshot)
            if current_seq > 0:
                self._last_state_seq = int(current_seq)
            if current_robot_ts > 0.0:
                self._last_robot_ts = float(current_robot_ts)
        if self.state_callback is not None:
            self.state_callback(snapshot)
        if self.event_callback is None or previous is None:
            return
        previous_ack = str(previous.get("last_ack") or "").strip().upper()
        current_ack = str(snapshot.get("last_ack") or "").strip().upper()
        current_state = str(snapshot.get("state") or "")
        previous_error = str(previous.get("last_error_code") or "")
        current_error = str(snapshot.get("last_error_code") or "")
        if current_ack and current_ack != previous_ack:
            self.event_callback(Event(source="robot", type="robot_ack", value=current_ack))
        if current_state == "ERROR" and current_error and current_error != previous_error:
            self.event_callback(Event(source="robot", type="robot_error", value=str(snapshot.get("last_error") or current_error)))

    def latest_state_snapshot(self) -> dict[str, object] | None:
        with self._state_lock:
            if self._last_state is None:
                return None
            return dict(self._last_state)

    @staticmethod
    def _message_to_snapshot(message: dict[str, object]) -> dict[str, object]:
        theta_deg = float(message.get("theta_deg", 0.0))
        radius_mm = float(message.get("radius_mm", 0.0))
        z_mm = float(message.get("z_mm", 0.0))
        x_mm = float(message.get("x_mm", 0.0))
        y_mm = float(message.get("y_mm", 0.0))
        return {
            "state": str(message.get("state", "")),
            "state_seq": int(message.get("state_seq", 0) or 0),
            "robot_ts": float(message.get("robot_ts", time.time()) or time.time()),
            "busy": bool(message.get("busy", False)),
            "busy_action": str(message.get("busy_action", "")),
            "carrying": bool(message.get("carrying", False)),
            "robot_xy": [x_mm, y_mm],
            "robot_z": z_mm,
            "robot_cyl": {"theta_deg": theta_deg, "radius_mm": radius_mm, "z_mm": z_mm},
            "limits_cyl": {
                "theta_deg": [float(message.get("theta_min_deg", 0.0)), float(message.get("theta_max_deg", 0.0))],
                "radius_mm": [float(message.get("radius_min_mm", 0.0)), float(message.get("radius_max_mm", 0.0))],
            },
            "limits_cyl_auto": {
                "theta_deg": [float(message.get("theta_min_deg", 0.0)), float(message.get("theta_max_deg", 0.0))],
                "radius_mm": [float(message.get("auto_radius_min_mm", 0.0)), float(message.get("auto_radius_max_mm", 0.0))],
            },
            "auto_z_current": float(message.get("auto_z_current", z_mm)),
            "control_kernel": str(message.get("control_kernel", "")),
            "last_error_code": str(message.get("last_error_code", "")),
            "last_error": str(message.get("last_error_message", "")),
            "calibration_ready": bool(message.get("calibration_ready", False)),
            "ik_valid": bool(message.get("ik_valid", True)),
            "validation_error": str(message.get("validation_error", "")),
            "last_ack": str(message.get("last_ack", "")),
            "pick_tuning": {
                "pick_approach_z_mm": float(message.get("pick_approach_z_mm", 0.0)),
                "pick_descend_z_mm": float(message.get("pick_descend_z_mm", 0.0)),
                "pick_pre_suction_sec": float(message.get("pick_pre_suction_sec", 0.0)),
                "pick_bottom_hold_sec": float(message.get("pick_bottom_hold_sec", 0.0)),
                "pick_lift_sec": float(message.get("pick_lift_sec", 0.0)),
                "place_descend_z_mm": float(message.get("place_descend_z_mm", 0.0)),
                "place_release_mode": str(message.get("place_release_mode", "release")),
                "place_release_sec": float(message.get("place_release_sec", 0.0)),
                "place_post_release_hold_sec": float(message.get("place_post_release_hold_sec", 0.0)),
                "z_carry_floor_mm": float(message.get("z_carry_floor_mm", 0.0)),
            },
            "post_pick_settle_z": float(message.get("post_pick_settle_z", 0.0)),
            "release_mode_effective": str(message.get("release_mode_effective", "")),
        }

    def _emit_status(self, message: str) -> None:
        if self.status_callback is not None:
            self.status_callback(str(message))

    @staticmethod
    def _parse_ros_param_value(raw_value: object, *, default: object = None) -> object:
        if raw_value is None:
            return default
        text = str(raw_value).strip()
        if not text:
            return default
        parsers = []
        if yaml is not None:
            parsers.append(yaml.safe_load)
        parsers.extend((json.loads, ast.literal_eval))
        for parser in parsers:
            try:
                return parser(text)
            except Exception:
                continue
        return text
