from __future__ import annotations

import types

from hybrid_controller.adapters.rosbridge_client import RosServiceResult
from hybrid_controller.app import HybridControllerApplication
from hybrid_controller.config import AppConfig


class _DummyMainWindow:
    def __init__(self) -> None:
        self.logs: list[str] = []

    def append_log(self, line: str) -> None:
        self.logs.append(str(line))


class _DummyRobotClient:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def send_command(self, command: str) -> None:
        self.commands.append(str(command))


class _DummyContext:
    def __init__(self) -> None:
        self.selected_target_id = 1
        self.latest_vision_targets: list[object] = []


class _DummyController:
    def __init__(self) -> None:
        self.context = _DummyContext()


class _DummyRosClient:
    def __init__(self) -> None:
        self.pick_world_calls: list[tuple[float, float]] = []
        self.pick_tuning_calls: list[dict[str, object]] = []

    def send_pick_world(self, x_mm: float, y_mm: float, *, callback) -> None:  # noqa: ANN001
        self.pick_world_calls.append((float(x_mm), float(y_mm)))
        callback(RosServiceResult(ok=True, message="ACK PICK_DONE", raw={}))

    def set_pick_tuning(self, tuning: dict[str, object], *, callback) -> None:  # noqa: ANN001
        self.pick_tuning_calls.append(dict(tuning))
        callback(RosServiceResult(ok=True, message="ok", raw={}))


def _make_ros_app_stub() -> HybridControllerApplication:
    app = HybridControllerApplication.__new__(HybridControllerApplication)
    app.config = AppConfig(robot_mode="real", robot_transport="ros", vision_mode="robot_camera_detection")
    app.main_window = _DummyMainWindow()
    app.robot_client = _DummyRobotClient()
    app.ros_client = _DummyRosClient()
    app.controller = _DummyController()
    app._pick_cyl_radius_bias_mm = 0.0
    app._pick_cyl_theta_bias_deg = 0.0
    app._pick_tuning_state = {
        "pick_approach_z_mm": 130.0,
        "pick_descend_z_mm": 85.0,
        "pick_pre_suction_sec": 0.25,
        "pick_bottom_hold_sec": 0.15,
        "pick_lift_sec": 0.8,
        "place_descend_z_mm": 85.0,
        "place_release_mode": "release",
        "place_release_sec": 0.25,
        "place_post_release_hold_sec": 0.1,
        "z_carry_floor_mm": 160.0,
    }
    app._active_pick_trace = None
    app._latest_vision_packet = None
    app.slot_catalog = None
    app._next_pending_command_seq = 1
    app._pending_command = None
    app.runtime_info = {
        "vision_mapping_mode": "delta_servo",
        "vision_last_resolved_base_xy": [10.0, -120.0],
        "vision_last_resolved_cyl": [0.0, 120.0, 85.0],
        "vision_snapshot_age_ms": 12.0,
    }
    app._queued_events: list[object] = []
    app._status_lines: list[tuple[str, str]] = []
    app._pick_traces: list[tuple[tuple[object, ...], dict[str, object]]] = []
    app._fetch_remote_robot_snapshot = lambda: {"robot_xy": [0.0, -120.0], "robot_cyl": {"theta_deg": 0.0, "radius_mm": 120.0, "z_mm": 160.0}}
    app._queue_event = lambda event: app._queued_events.append(event)
    app._queue_runtime_status = lambda component, message: app._status_lines.append((str(component), str(message)))
    app.dispatch_event = lambda event: app._queued_events.append(event)
    app._handle_runtime_status = lambda component, message: app._status_lines.append((str(component), str(message)))
    app.logger = types.SimpleNamespace(write=lambda *args, **kwargs: app._pick_traces.append((args, kwargs)))
    app._request_remote_snapshot = lambda: None
    return app


def test_pick_world_routes_through_ros_without_tcp_fallback() -> None:
    app = _make_ros_app_stub()

    app._send_robot_text_command("PICK_WORLD 10 -120")

    assert app.ros_client.pick_world_calls == [(10.0, -120.0)]
    assert app.robot_client.commands == []
    assert app._rt_get("pending_command_seq", 0) == 1
    assert str(app._rt_get("pending_command", "")) == "PICK_WORLD 10 -120"


def test_pick_world_ros_mode_without_ros_client_rejects_instead_of_tcp_fallback() -> None:
    app = _make_ros_app_stub()
    app.ros_client = None

    app._send_robot_text_command("PICK_WORLD 12 -118")

    assert app.robot_client.commands == []
    assert app._status_lines
    assert "ROS transport command rejected" in app._status_lines[-1][1]
    assert app._queued_events
    assert any(getattr(event, "type", "") == "robot_error" for event in app._queued_events)


def test_manual_pick_in_robot_camera_detection_mode_does_not_fallback_to_slot_catalog() -> None:
    app = _make_ros_app_stub()
    app._latest_vision_packet = {
        "slots": [
            {
                "slot_id": 1,
                "valid": True,
                "actionable": False,
                "invalid_reason": "robot_snapshot_stale",
                "command_mode": "world",
                "command_point": None,
            }
        ]
    }
    app.slot_catalog = types.SimpleNamespace(
        list_pick_slots=lambda source=None: [
            types.SimpleNamespace(slot_id=1, world_xy=(-50.0, -160.0), cylindrical_trz=(0.0, 160.0, 85.0))
        ]
    )

    command = app._build_manual_pick_command(1)

    assert command is None


def test_pick_world_bias_rewrite_applies_theta_and_radius_bias() -> None:
    app = _make_ros_app_stub()
    app._pick_cyl_radius_bias_mm = -5.0
    app._pick_cyl_theta_bias_deg = 3.0

    rewritten = app._rewrite_outgoing_robot_command("PICK_WORLD 0 -120")

    assert rewritten.startswith("PICK_WORLD ")
    assert rewritten != "PICK_WORLD 0 -120"
