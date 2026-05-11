from __future__ import annotations

import importlib
import threading

from hybrid_controller.app import HybridControllerApplication
from hybrid_controller.config import AppConfig
from hybrid_controller.runtime_state import RobotSnapshotEnvelope


class _ConnectedRosClient:
    def is_connected(self) -> bool:
        return True


class _ReadyRosClient:
    def __init__(self, snapshot: dict[str, object] | None = None, *, connected: bool = True) -> None:
        self.snapshot = snapshot if snapshot is not None else {"state": "IDLE", "busy": False, "carrying": False}
        self.connected = connected

    def is_connected(self) -> bool:
        return bool(self.connected)

    def latest_state_snapshot(self) -> dict[str, object] | None:
        if self.snapshot is None:
            return None
        return dict(self.snapshot)


class _FakeRosbridgeClient:
    instances: list["_FakeRosbridgeClient"] = []
    connect_result: bool = True
    state_snapshot: dict[str, object] | None = {"state": "IDLE", "busy": False, "carrying": False}

    def __init__(
        self,
        host: str,
        port: int,
        *,
        state_callback=None,
        event_callback=None,
        status_callback=None,
    ) -> None:
        self.host = host
        self.port = port
        self.connected = False
        _FakeRosbridgeClient.instances.append(self)

    def connect(self) -> None:
        self.connected = bool(self.connect_result)

    def is_connected(self) -> bool:
        return self.connected

    def latest_state_snapshot(self) -> dict[str, object] | None:
        if self.state_snapshot is None:
            return None
        return dict(self.state_snapshot)

    def get_pick_tuning(self, *, callback=None) -> None:
        return None


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
    app._remote_snapshot_lock = threading.Lock()
    app._remote_snapshot_cache = None
    app._remote_snapshot_envelope = None
    app._shutdown_started = False
    app._next_robot_bootstrap_probe_ts = 0.0
    app._last_auto_robot_start_ts = 0.0
    app._auto_robot_start_attempts = 0
    app._last_ros_runtime_unavailable_log_ts = 0.0
    app._vision_auto_start_deferred_reason = ""
    app._last_vision_auto_start_deferred_log_ts = 0.0
    app._auto_start_blocked = False
    app._auto_start_block_reason = ""
    app._rt_get = lambda key, default=None: app.runtime_info.get(key, default)  # type: ignore[method-assign]
    app._rt_set = lambda key, value: app.runtime_info.__setitem__(key, value)  # type: ignore[method-assign]
    app._rt_update = lambda payload: app.runtime_info.update(payload)  # type: ignore[method-assign]
    app._log_runtime = lambda component, message: None  # type: ignore[method-assign]
    app._queue_event = lambda event: None  # type: ignore[method-assign]
    app._queue_remote_snapshot = lambda snapshot: None  # type: ignore[method-assign]
    app._sync_pick_tuning_from_robot = lambda: None  # type: ignore[method-assign]
    app._capture_world_snapshot = lambda **_: None  # type: ignore[method-assign]
    app._latest_world_snapshot = None
    app._evaluate_preflight_from_snapshot = lambda snapshot: None  # type: ignore[method-assign]
    app._on_remote_snapshot_received = lambda snapshot: None  # type: ignore[method-assign]
    app._fetch_vision_calibration_params = lambda: None  # type: ignore[method-assign]
    app._queue_vision_packet = lambda packet: None  # type: ignore[method-assign]
    app._queue_vision_frame = lambda frame: None  # type: ignore[method-assign]
    app._queue_runtime_status = lambda component, message: None  # type: ignore[method-assign]
    return app


def test_robot_bootstrap_default_does_not_probe_or_auto_start() -> None:
    app = _make_app(AppConfig(robot_mode="real", robot_transport="ros"))
    probes: list[str] = []
    starts: list[str] = []
    app._probe_tcp_port = lambda **_: probes.append("probe") or False  # type: ignore[method-assign]
    app._on_robot_start_requested = lambda: starts.append("start")  # type: ignore[method-assign]

    app._pump_robot_bootstrap()

    assert probes == []
    assert starts == []


def test_ros_connect_skips_preconnect_probe_by_default() -> None:
    _FakeRosbridgeClient.instances = []
    _FakeRosbridgeClient.connect_result = True
    _FakeRosbridgeClient.state_snapshot = {"state": "IDLE", "busy": False, "carrying": False}
    app = _make_app(AppConfig(robot_mode="real", robot_transport="ros"))
    probes: list[str] = []
    starts: list[str] = []
    app._probe_tcp_port = lambda **_: probes.append("probe") or False  # type: ignore[method-assign]
    app._on_robot_start_requested = lambda: starts.append("start")  # type: ignore[method-assign]
    app._create_rosbridge_client = lambda: _FakeRosbridgeClient("127.0.0.1", 9091)  # type: ignore[method-assign]

    app._setup_robot_mode()

    assert probes == []
    assert starts == []
    assert app.runtime_info["robot_connected"] is True
    assert app.runtime_info["robot_health"] == "ok"
    assert len(_FakeRosbridgeClient.instances) == 1


def test_ros_connect_requires_initial_state_before_marking_ready() -> None:
    _FakeRosbridgeClient.instances = []
    _FakeRosbridgeClient.connect_result = True
    _FakeRosbridgeClient.state_snapshot = None
    app = _make_app(
        AppConfig(
            robot_mode="real",
            robot_transport="ros",
            robot_auto_start_on_ros_unavailable=True,
            robot_auto_start_max_attempts=1,
            ros_runtime_state_grace_sec=0.01,
        )
    )
    starts: list[str] = []
    statuses: list[tuple[str, str]] = []
    logs: list[tuple[str, str]] = []
    app._on_robot_start_requested = lambda: starts.append("start")  # type: ignore[method-assign]
    app._queue_runtime_status = lambda component, message: statuses.append((component, message))  # type: ignore[method-assign]
    app._log_runtime = lambda component, message: logs.append((component, message))  # type: ignore[method-assign]
    app._create_rosbridge_client = lambda: _FakeRosbridgeClient("127.0.0.1", 9091)  # type: ignore[method-assign]

    app._setup_robot_mode()

    assert starts == ["start"]
    assert app.runtime_info["robot_connected"] is False
    assert app.runtime_info["robot_health"] == "state_unavailable"
    assert app.runtime_info["preflight_message"] == "state_unavailable"
    assert any("ros_state_unavailable" in message for _, message in statuses)
    assert any("/hybrid_controller/state" in message for _, message in logs)


def test_ros_connect_probe_is_explicit_opt_in() -> None:
    _FakeRosbridgeClient.instances = []
    _FakeRosbridgeClient.connect_result = True
    app = _make_app(
        AppConfig(
            robot_mode="real",
            robot_transport="ros",
            ros_probe_before_connect=True,
        )
    )
    probes: list[str] = []
    starts: list[str] = []
    app._probe_tcp_port = lambda **_: probes.append("probe") or False  # type: ignore[method-assign]
    app._on_robot_start_requested = lambda: starts.append("start")  # type: ignore[method-assign]
    app._create_rosbridge_client = lambda: _FakeRosbridgeClient("127.0.0.1", 9091)  # type: ignore[method-assign]

    app._setup_robot_mode()

    assert probes == ["probe"]
    assert starts == []
    assert app.runtime_info["robot_connected"] is False
    assert app.runtime_info["robot_health"] == "rosbridge_port_closed"
    assert _FakeRosbridgeClient.instances == []


def test_startup_ros_connect_timeout_triggers_one_auto_start_without_probe() -> None:
    _FakeRosbridgeClient.instances = []
    _FakeRosbridgeClient.connect_result = False
    app = _make_app(
        AppConfig(
            robot_mode="real",
            robot_transport="ros",
            robot_auto_start_on_ros_unavailable=True,
            robot_auto_start_max_attempts=1,
            rosbridge_timeout_sec=0.01,
        )
    )
    probes: list[str] = []
    starts: list[str] = []
    statuses: list[tuple[str, str]] = []
    app._probe_tcp_port = lambda **_: probes.append("probe") or False  # type: ignore[method-assign]
    app._on_robot_start_requested = lambda: starts.append("start")  # type: ignore[method-assign]
    app._queue_runtime_status = lambda component, message: statuses.append((component, message))  # type: ignore[method-assign]
    app._create_rosbridge_client = lambda: _FakeRosbridgeClient("127.0.0.1", 9091)  # type: ignore[method-assign]

    app._setup_robot_mode()
    app._last_auto_robot_start_ts = 0.0
    app._setup_robot_mode()

    assert probes == []
    assert starts == ["start"]
    assert app.runtime_info["robot_connected"] is False
    assert app.runtime_info["robot_health"] == "rosbridge_connecting"
    assert app.runtime_info["preflight_message"] == "rosbridge_connecting"
    assert any("rosbridge_connect_timeout" in message for _, message in statuses)
    assert any("attempt limit reached" in message for _, message in statuses)


def test_robot_bootstrap_auto_starts_when_explicitly_enabled_and_rosbridge_port_is_closed() -> None:
    app = _make_app(
        AppConfig(
            robot_mode="real",
            robot_transport="ros",
            robot_bootstrap_retry_enabled=True,
            robot_auto_start_on_ros_unavailable=True,
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


def test_robot_auto_start_attempt_limit_blocks_repeat_starts() -> None:
    app = _make_app(
        AppConfig(
            robot_mode="real",
            robot_transport="ros",
            robot_auto_start_on_ros_unavailable=True,
            robot_auto_start_max_attempts=1,
            robot_auto_start_cooldown_sec=1.0,
        )
    )
    starts: list[str] = []
    statuses: list[tuple[str, str]] = []
    app._on_robot_start_requested = lambda: starts.append("start")  # type: ignore[method-assign]
    app._queue_runtime_status = lambda component, message: statuses.append((component, message))  # type: ignore[method-assign]

    assert app._maybe_auto_start_robot_runtime("unit_first") is True
    app._last_auto_robot_start_ts = 0.0
    app._last_ros_runtime_unavailable_log_ts = 0.0
    assert app._maybe_auto_start_robot_runtime("unit_second") is False

    assert starts == ["start"]
    assert any("attempt limit reached" in message for _, message in statuses)


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


def test_remote_snapshot_poller_waits_for_explicit_robot_connection() -> None:
    app = _make_app(AppConfig(robot_mode="real", robot_transport="tcp"))
    app._remote_snapshot_poller = None
    app._robot_connection_requested = False

    app._start_remote_snapshot_poller()

    assert app._remote_snapshot_poller is None


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


def test_runtime_auto_start_and_stale_restart_are_disabled_by_default() -> None:
    config = AppConfig()
    assert config.robot_auto_start_on_ros_unavailable is False
    assert config.robot_bootstrap_retry_enabled is False
    assert config.robot_auto_restart_on_state_stale is False
    assert config.robot_auto_start_max_attempts == 1


def test_vision_auto_start_reads_official_stream_without_ros_state(monkeypatch) -> None:
    started: list[str] = []

    class FakeVisionRuntime:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            started.append("start")

    app = _make_app(
        AppConfig(
            robot_mode="real",
            robot_transport="ros",
            vision_mode="robot_camera_detection",
            vision_auto_start=True,
        )
    )
    vision_runtime_module = importlib.import_module("hybrid_controller.vision.runtime")
    monkeypatch.setattr(vision_runtime_module, "VisionRuntime", FakeVisionRuntime)

    app._setup_vision_mode()

    assert started == ["start"]
    assert isinstance(app.vision_runtime, FakeVisionRuntime)
    assert app.runtime_info["vision_health"] == "starting_without_calibration"
    assert app._vision_auto_start_deferred_reason == ""


def test_vision_auto_start_defers_while_robot_runtime_start_is_active() -> None:
    app = _make_app(
        AppConfig(
            robot_mode="real",
            robot_transport="ros",
            vision_mode="robot_camera_detection",
            vision_auto_start=True,
        )
    )
    app.runtime_info["robot_start_active"] = True
    app.ros_client = _ReadyRosClient()  # type: ignore[assignment]

    app._setup_vision_mode()

    assert app.vision_runtime is None
    assert app.runtime_info["vision_health"] == "waiting_for_robot_runtime:robot_runtime_start_active"


def test_vision_auto_start_runs_when_ros_state_is_fresh(monkeypatch) -> None:
    started: list[str] = []

    class FakeVisionRuntime:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def start(self) -> None:
            started.append("start")

    app = _make_app(
        AppConfig(
            robot_mode="real",
            robot_transport="ros",
            vision_mode="robot_camera_detection",
            vision_auto_start=True,
            robot_state_stale_threshold_ms=700.0,
        )
    )
    snapshot = {"state": "IDLE", "busy": False, "carrying": False}
    app.ros_client = _ReadyRosClient(snapshot)  # type: ignore[assignment]
    app._remote_snapshot_cache = dict(snapshot)
    app._remote_snapshot_envelope = RobotSnapshotEnvelope(
        payload=dict(snapshot),
        ts=__import__("time").time(),
        transport="unit",
        ok=True,
        error="",
    )
    vision_runtime_module = importlib.import_module("hybrid_controller.vision.runtime")
    monkeypatch.setattr(vision_runtime_module, "VisionRuntime", FakeVisionRuntime)

    app._setup_vision_mode()

    assert started == ["start"]
    assert isinstance(app.vision_runtime, FakeVisionRuntime)
    assert app.runtime_info["vision_health"] == "starting_without_calibration"
    assert app._vision_auto_start_deferred_reason == ""


def test_deferred_vision_starts_after_robot_runtime_start_finishes(monkeypatch) -> None:
    started: list[str] = []

    class FakeVisionRuntime:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            started.append("start")

    app = _make_app(
        AppConfig(
            robot_mode="real",
            robot_transport="ros",
            vision_mode="robot_camera_detection",
            vision_auto_start=True,
            robot_state_stale_threshold_ms=700.0,
        )
    )
    app.runtime_info["robot_start_active"] = True
    app._setup_vision_mode()
    assert app.vision_runtime is None
    vision_runtime_module = importlib.import_module("hybrid_controller.vision.runtime")
    monkeypatch.setattr(vision_runtime_module, "VisionRuntime", FakeVisionRuntime)
    app.runtime_info["robot_start_active"] = False
    app.runtime_info["robot_health"] = "reconnecting"
    app._maybe_start_deferred_vision(source="unit_runtime_finished")

    assert started == ["start"]
    assert app.runtime_info["vision_health"] == "starting_without_calibration"


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
