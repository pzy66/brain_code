from __future__ import annotations

from hybrid_controller.app import HybridControllerApplication
from hybrid_controller.config import AppConfig


class _FakeTimer:
    def __init__(self, interval_ms: int) -> None:
        self._interval_ms = int(interval_ms)
        self.started_intervals: list[int] = []

    def interval(self) -> int:
        return self._interval_ms

    def start(self, interval_ms: int) -> None:
        self._interval_ms = int(interval_ms)
        self.started_intervals.append(int(interval_ms))


class _FakeTeleopPlanner:
    active = False


def _make_scheduler_app(config: AppConfig | None = None) -> HybridControllerApplication:
    app = HybridControllerApplication.__new__(HybridControllerApplication)
    app.config = config or AppConfig()
    app._pressed_move_tokens = set()
    app._teleop_timer_interval_ms = None
    app._teleop_ros_planner = _FakeTeleopPlanner()
    app.timers = {}
    return app


def test_realtime_scheduler_uses_slow_tick_while_keyboard_is_idle() -> None:
    app = _make_scheduler_app(
        AppConfig(
            move_source="sim",
            teleop_repeat_interval_ms=50,
            idle_runtime_tick_interval_ms=250,
        )
    )

    assert app._desired_realtime_tick_interval_ms() == 250

    app._pressed_move_tokens.add("w")

    assert app._desired_realtime_tick_interval_ms() == 50


def test_realtime_scheduler_restarts_timer_when_motion_state_changes() -> None:
    app = _make_scheduler_app(
        AppConfig(
            move_source="sim",
            teleop_repeat_interval_ms=50,
            idle_runtime_tick_interval_ms=250,
        )
    )
    timer = _FakeTimer(250)
    app.timers["teleop-step"] = timer
    app._teleop_timer_interval_ms = 250

    app._pressed_move_tokens.add("d")
    app._sync_realtime_timer_interval()

    assert timer.started_intervals == [50]

    app._pressed_move_tokens.clear()
    app._sync_realtime_timer_interval()

    assert timer.started_intervals == [50, 250]


def test_realtime_tick_skips_teleop_pump_when_idle() -> None:
    app = _make_scheduler_app()
    calls: list[str] = []
    app._pump_robot_bootstrap = lambda: calls.append("bootstrap")  # type: ignore[method-assign]
    app._pump_ros_reconnect = lambda: calls.append("reconnect")  # type: ignore[method-assign]
    app._pump_input_sources = lambda: calls.append("input")  # type: ignore[method-assign]
    app._pump_teleop_command = lambda: calls.append("teleop")  # type: ignore[method-assign]
    app._check_pending_command_timeout = lambda: calls.append("timeout")  # type: ignore[method-assign]

    app._on_realtime_tick()

    assert calls == ["bootstrap", "reconnect", "input", "timeout"]

    app._pressed_move_tokens.add("w")
    app._on_realtime_tick()

    assert calls[-5:] == ["bootstrap", "reconnect", "input", "teleop", "timeout"]
