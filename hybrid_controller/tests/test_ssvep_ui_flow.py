from __future__ import annotations

from hybrid_controller.app import HybridControllerApplication


class _DummyMainWindow:
    def __init__(self) -> None:
        self.runtime_config = {
            "serial_port": "COM9",
            "board_id": 1,
            "prepare_sec": 0.5,
            "active_sec": 3.5,
            "rest_sec": 0.5,
            "target_repeats": 2,
            "idle_repeats": 4,
            "win_sec": 2.5,
            "step_sec": 0.5,
        }

    def ssvep_runtime_config(self) -> dict[str, object]:
        return dict(self.runtime_config)

    def ssvep_pretrain_config(self) -> dict[str, object]:
        return {
            "preset": "fast",
            "target_repeats": 2,
            "idle_repeats": 4,
            "estimated_sec": 54.0,
        }


class _DummySSVEPRuntime:
    def __init__(self) -> None:
        self.connected = True
        self.config_calls: list[dict[str, object]] = []

    def set_runtime_config(self, **kwargs: object) -> None:
        self.config_calls.append(dict(kwargs))


class _DummySSVEPCoordinator:
    def __init__(self) -> None:
        self.connect_calls = 0
        self.pretrain_calls = 0
        self.online_calls = 0

    def connect_device(self) -> None:
        self.connect_calls += 1

    def start_pretrain(self) -> None:
        self.pretrain_calls += 1

    def start_online(self) -> None:
        self.online_calls += 1


def _make_ssvep_app_stub() -> HybridControllerApplication:
    app = HybridControllerApplication.__new__(HybridControllerApplication)
    app.main_window = _DummyMainWindow()
    app.ssvep_runtime = _DummySSVEPRuntime()
    app.ssvep_coordinator = _DummySSVEPCoordinator()
    app._rt_updates: list[dict[str, object]] = []
    app._statuses: list[tuple[str, str]] = []
    app._rt_update = lambda payload: app._rt_updates.append(dict(payload))
    app._handle_runtime_status = lambda component, message: app._statuses.append((str(component), str(message)))
    return app


def test_ssvep_connect_pretrain_and_online_apply_current_ui_config() -> None:
    app = _make_ssvep_app_stub()

    app._on_ssvep_connect_requested()
    app._on_ssvep_pretrain_requested()
    app._on_ssvep_start_requested()

    assert app.ssvep_coordinator.connect_calls == 1
    assert app.ssvep_coordinator.pretrain_calls == 1
    assert app.ssvep_coordinator.online_calls == 1
    assert len(app.ssvep_runtime.config_calls) == 3
    assert all(call["serial_port"] == "COM9" for call in app.ssvep_runtime.config_calls)
    assert all(call["board_id"] == 1 for call in app.ssvep_runtime.config_calls)
    assert all(call["target_repeats"] == 2 for call in app.ssvep_runtime.config_calls)
    assert all(call["idle_repeats"] == 4 for call in app.ssvep_runtime.config_calls)
    assert any("pretrain=fast" in message for _, message in app._statuses)
