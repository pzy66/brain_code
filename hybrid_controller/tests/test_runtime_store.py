from __future__ import annotations

from hybrid_controller.config import AppConfig
from hybrid_controller.runtime_state import RuntimeAction, RuntimeInfoCompat, RuntimeStore


def test_runtime_store_reducer_updates_and_perf_fields() -> None:
    store = RuntimeStore.from_config(AppConfig())
    store.dispatch(
        RuntimeAction.update(
            {
                "robot_connected": True,
                "robot_health": "ok",
                "ssvep_running": True,
                "queue_age_ms": 42.5,
                "infer_interval_ms": 88.0,
                "frame_drop_ratio": 0.25,
                "remote_snapshot_age_ms": 35.0,
            }
        )
    )

    state = store.state
    assert state.robot_connected is True
    assert state.robot_health == "ok"
    assert state.ssvep_running is True
    assert abs(state.perf.queue_age_ms - 42.5) < 1e-6
    assert abs(state.perf.infer_interval_ms - 88.0) < 1e-6
    assert abs(state.perf.frame_drop_ratio - 0.25) < 1e-6
    assert abs(state.perf.remote_snapshot_age_ms - 35.0) < 1e-6


def test_runtime_info_compat_mirrors_runtime_store() -> None:
    store = RuntimeStore.from_config(AppConfig())
    runtime_info = RuntimeInfoCompat(store)

    runtime_info["robot_connected"] = True
    runtime_info["ssvep_profile_count"] = 3
    runtime_info["ui_refresh_ms_ema"] = 7.5
    runtime_info["custom_note"] = "ok"

    assert runtime_info.get("robot_connected") is True
    assert runtime_info.get("ssvep_profile_count") == 3
    assert abs(float(runtime_info.get("ui_refresh_ms_ema", 0.0)) - 7.5) < 1e-6
    assert runtime_info.get("custom_note") == "ok"
    assert store.state.robot_connected is True
    assert store.state.ssvep_profile_count == 3
    assert abs(store.state.perf.ui_refresh_ms_ema - 7.5) < 1e-6

