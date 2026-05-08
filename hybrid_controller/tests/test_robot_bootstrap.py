from __future__ import annotations

from hybrid_controller.app import HybridControllerApplication
from hybrid_controller.config import AppConfig


class _ConnectedRosClient:
    def is_connected(self) -> bool:
        return True


def _make_app(config: AppConfig) -> HybridControllerApplication:
    app = HybridControllerApplication.__new__(HybridControllerApplication)
    app.config = config
    app.runtime_info = {
        "robot_start_active": False,
        "robot_health": "unknown",
        "preflight_ok": False,
        "preflight_message": "unknown",
    }
    app.ros_client = None
    app.vision_runtime = None
    app._shutdown_started = False
    app._next_robot_bootstrap_probe_ts = 0.0
    app._last_auto_robot_start_ts = 0.0
    app._last_ros_runtime_unavailable_log_ts = 0.0
    app._auto_start_blocked = False
    app._auto_start_block_reason = ""
    app._rt_get = lambda key, default=None: app.runtime_info.get(key, default)  # type: ignore[method-assign]
    app._rt_set = lambda key, value: app.runtime_info.__setitem__(key, value)  # type: ignore[method-assign]
    app._rt_update = lambda payload: app.runtime_info.update(payload)  # type: ignore[method-assign]
    return app


def test_robot_bootstrap_auto_starts_when_rosbridge_port_is_closed() -> None:
    app = _make_app(
        AppConfig(
            robot_mode="real",
            robot_transport="ros",
            robot_bootstrap_retry_enabled=True,
            robot_bootstrap_probe_interval_sec=1.0,
        )
    )
    starts: list[str] = []
    statuses: list[tuple[str, str]] = []
    app._probe_tcp_port = lambda **_: False  # type: ignore[method-assign]
    app._on_robot_start_requested = lambda: starts.append("start")  # type: ignore[method-assign]
    app._queue_runtime_status = lambda component, message: statuses.append((component, message))  # type: ignore[method-assign]

    app._pump_robot_bootstrap()

    assert starts == ["start"]
    assert app.runtime_info["robot_health"] == "waiting_for_robot_runtime"
    assert app.runtime_info["preflight_message"] == "waiting_for_robot_runtime"
    assert any("Auto-start robot runtime" in message for _, message in statuses)


def test_robot_bootstrap_reconnects_when_rosbridge_port_returns() -> None:
    app = _make_app(
        AppConfig(
            robot_mode="real",
            robot_transport="ros",
            robot_bootstrap_retry_enabled=True,
            robot_bootstrap_probe_interval_sec=1.0,
        )
    )
    connects: list[str] = []
    statuses: list[tuple[str, str]] = []
    app._probe_tcp_port = lambda **_: True  # type: ignore[method-assign]
    app._on_robot_connect_requested = lambda: connects.append("connect")  # type: ignore[method-assign]
    app._queue_runtime_status = lambda component, message: statuses.append((component, message))  # type: ignore[method-assign]

    app._pump_robot_bootstrap()

    assert connects == ["connect"]
    assert any("reconnecting robot client" in message for _, message in statuses)


def test_robot_bootstrap_does_nothing_when_ros_client_is_connected() -> None:
    app = _make_app(AppConfig(robot_mode="real", robot_transport="ros"))
    app.ros_client = _ConnectedRosClient()  # type: ignore[assignment]
    probes: list[str] = []
    app._probe_tcp_port = lambda **_: probes.append("probe") or False  # type: ignore[method-assign]

    app._pump_robot_bootstrap()

    assert probes == []


def test_robot_bootstrap_can_be_disabled() -> None:
    app = _make_app(
        AppConfig(
            robot_mode="real",
            robot_transport="ros",
            robot_bootstrap_retry_enabled=False,
        )
    )
    starts: list[str] = []
    app._probe_tcp_port = lambda **_: False  # type: ignore[method-assign]
    app._on_robot_start_requested = lambda: starts.append("start")  # type: ignore[method-assign]

    app._pump_robot_bootstrap()

    assert starts == []


def test_stale_ros_state_recovery_is_enabled_by_default() -> None:
    assert AppConfig().robot_auto_restart_on_state_stale is True


def test_robot_runtime_health_marks_disconnected_when_rosbridge_connected_but_state_stale() -> None:
    app = _make_app(AppConfig(robot_mode="real", robot_transport="ros", robot_state_stale_threshold_ms=700.0))
    app.ros_client = _ConnectedRosClient()  # type: ignore[assignment]
    app._compute_remote_snapshot_age_ms = lambda: 2000.0  # type: ignore[method-assign]
    app._maybe_recover_ros_runtime_from_stale_state = lambda **_: None  # type: ignore[method-assign]
    app._update_runtime_health()

    assert app.runtime_info["robot_connected"] is False
    assert app.runtime_info["robot_health"] == "state_stale"


def test_pending_command_can_resolve_from_state_snapshot_ack() -> None:
    app = _make_app(AppConfig(robot_mode="real", robot_transport="ros"))
    events: list[object] = []
    app._pending_command = {"expected_ack": "MOVE", "command": "MOVE_CYL 0 160 130"}
    app.dispatch_event = lambda event: events.append(event)  # type: ignore[method-assign]

    app._dispatch_robot_ack_from_state_if_pending({"last_ack": "MOVE", "busy": False})

    assert len(events) == 1
    assert getattr(events[0], "type", "") == "robot_ack"
    assert getattr(events[0], "value", "") == "MOVE"


def test_pending_command_does_not_resolve_from_state_snapshot_while_busy() -> None:
    app = _make_app(AppConfig(robot_mode="real", robot_transport="ros"))
    events: list[object] = []
    app._pending_command = {"expected_ack": "MOVE", "command": "MOVE_CYL 0 160 130"}
    app.dispatch_event = lambda event: events.append(event)  # type: ignore[method-assign]

    app._dispatch_robot_ack_from_state_if_pending({"last_ack": "MOVE", "busy": True})

    assert events == []


def test_pending_command_does_not_resolve_from_stale_repeated_ack() -> None:
    app = _make_app(AppConfig(robot_mode="real", robot_transport="ros"))
    events: list[object] = []
    app._pending_command = {
        "expected_ack": "MOVE",
        "command": "MOVE_CYL 0 160 130",
        "ack_baseline_last_ack": "MOVE",
        "ack_baseline_state_seq": 41,
    }
    app.dispatch_event = lambda event: events.append(event)  # type: ignore[method-assign]

    app._dispatch_robot_ack_from_state_if_pending({"last_ack": "MOVE", "busy": False, "state_seq": 41})

    assert events == []


def test_pending_command_resolves_from_repeated_ack_after_state_seq_advances() -> None:
    app = _make_app(AppConfig(robot_mode="real", robot_transport="ros"))
    events: list[object] = []
    app._pending_command = {
        "expected_ack": "MOVE",
        "command": "MOVE_CYL 0 160 130",
        "ack_baseline_last_ack": "MOVE",
        "ack_baseline_state_seq": 41,
    }
    app.dispatch_event = lambda event: events.append(event)  # type: ignore[method-assign]

    app._dispatch_robot_ack_from_state_if_pending({"last_ack": "MOVE", "busy": False, "state_seq": 42})

    assert len(events) == 1
    assert getattr(events[0], "type", "") == "robot_ack"
    assert getattr(events[0], "value", "") == "MOVE"
