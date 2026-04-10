from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeAction:
    kind: str
    key: str | None = None
    value: Any = None
    payload: Mapping[str, Any] | None = None

    @staticmethod
    def set(key: str, value: Any) -> "RuntimeAction":
        return RuntimeAction(kind="set", key=str(key), value=value)

    @staticmethod
    def update(payload: Mapping[str, Any]) -> "RuntimeAction":
        return RuntimeAction(kind="update", payload=dict(payload))


@dataclass(slots=True)
class PerfState:
    ui_refresh_ms_ema: float = 0.0
    queue_age_ms: float = 0.0
    infer_interval_ms: float = 0.0
    frame_drop_ratio: float = 0.0
    remote_snapshot_age_ms: float = 0.0


@dataclass(slots=True)
class RuntimeState:
    simulation_enabled: bool = False
    timing_profile: str = "formal"
    scenario_name: str = "basic"
    move_source: str = "sim"
    decision_source: str = "sim"
    robot_mode: str = "real"
    robot_transport: str = "tcp"
    vision_mode: str = "robot_camera_detection"
    robot_connected: bool = False
    robot_start_active: bool = False
    robot_health: str = "unknown"
    vision_health: str = "unknown"
    last_robot_ack: str = "--"
    last_robot_error: str = "--"
    last_ssvep_raw: str = "--"
    target_frequency_map: list[tuple[str, object]] = field(default_factory=list)
    preflight_ok: bool = True
    preflight_message: str = "not_required"
    calibration_ready: bool | None = None
    robot_cyl: dict[str, object] | None = None
    limits_cyl: dict[str, object] | None = None
    auto_z_current: float | None = None
    control_kernel: str = "cylindrical_kernel"
    ssvep_runtime_status: str = "stopped"
    ssvep_running: bool = False
    ssvep_stim_enabled: bool = False
    ssvep_busy: bool = False
    ssvep_connected: bool = False
    ssvep_connect_active: bool = False
    ssvep_pretrain_active: bool = False
    ssvep_online_active: bool = False
    ssvep_profile_path: str = "--"
    ssvep_profile_source: str = "fallback"
    ssvep_last_pretrain_time: str = "--"
    ssvep_latest_profile_path: str = "--"
    ssvep_profile_count: int = 0
    ssvep_available_profiles: tuple[tuple[str, str], ...] = ()
    ssvep_allow_fallback_profile: bool = True
    ssvep_status_hint: str = "No trained profile. You can pretrain first, or start with fallback."
    ssvep_mode: str = "idle"
    ssvep_last_state: str = "--"
    ssvep_last_selected_freq: str = "--"
    ssvep_last_margin: str = "--"
    ssvep_last_ratio: str = "--"
    ssvep_last_stable_windows: str = "--"
    ssvep_last_error: str = "--"
    ssvep_model_name: str = "fbcca"
    ssvep_debug_keyboard: bool = True
    perf: PerfState = field(default_factory=PerfState)


@dataclass(frozen=True, slots=True)
class RobotSnapshotEnvelope:
    payload: dict[str, object] | None
    ts: float
    transport: str
    ok: bool
    error: str = ""

    @property
    def has_payload(self) -> bool:
        return isinstance(self.payload, dict)


_PERF_KEY_MAP: dict[str, str] = {
    "ui_refresh_ms_ema": "ui_refresh_ms_ema",
    "queue_age_ms": "queue_age_ms",
    "infer_interval_ms": "infer_interval_ms",
    "frame_drop_ratio": "frame_drop_ratio",
    "remote_snapshot_age_ms": "remote_snapshot_age_ms",
}


class RuntimeStore:
    def __init__(self, state: RuntimeState) -> None:
        self._state = state
        self._field_names = {item.name for item in fields(RuntimeState)} - {"perf"}
        self._extra: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: object) -> "RuntimeStore":
        return cls(
            RuntimeState(
                simulation_enabled=bool(getattr(config, "simulation_enabled", False)),
                timing_profile=str(getattr(config, "timing_profile", "formal")),
                scenario_name=str(getattr(config, "scenario_name", "basic")),
                move_source=str(getattr(config, "move_source", "sim")),
                decision_source=str(getattr(config, "decision_source", "sim")),
                robot_mode=str(getattr(config, "robot_mode", "real")),
                robot_transport=str(getattr(config, "robot_transport", "tcp")),
                vision_mode=str(getattr(config, "vision_mode", "robot_camera_detection")),
                ssvep_profile_path=str(getattr(config, "ssvep_current_profile_path", "--")),
                ssvep_allow_fallback_profile=bool(getattr(config, "ssvep_allow_fallback_profile", True)),
                ssvep_model_name=str(getattr(config, "ssvep_model_name", "fbcca")),
                ssvep_debug_keyboard=bool(getattr(config, "ssvep_keyboard_debug_enabled", True)),
            )
        )

    @property
    def state(self) -> RuntimeState:
        return self._state

    def dispatch(self, action: RuntimeAction) -> None:
        if action.kind == "set":
            if not action.key:
                raise ValueError("RuntimeAction(kind='set') requires key.")
            self.set_value(action.key, action.value)
            return
        if action.kind == "update":
            if action.payload is None:
                return
            for key, value in action.payload.items():
                self.set_value(str(key), value)
            return
        raise ValueError(f"Unsupported RuntimeAction kind: {action.kind}")

    def set_value(self, key: str, value: Any) -> None:
        text = str(key)
        perf_key = _PERF_KEY_MAP.get(text)
        if perf_key is not None:
            setattr(self._state.perf, perf_key, float(value))
            return
        if text in self._field_names:
            setattr(self._state, text, self._normalize_value(text, value))
            return
        self._extra[text] = value

    def get_value(self, key: str, default: Any = None) -> Any:
        text = str(key)
        perf_key = _PERF_KEY_MAP.get(text)
        if perf_key is not None:
            return getattr(self._state.perf, perf_key)
        if text in self._field_names:
            return getattr(self._state, text)
        return self._extra.get(text, default)

    def delete_value(self, key: str) -> None:
        text = str(key)
        perf_key = _PERF_KEY_MAP.get(text)
        if perf_key is not None:
            setattr(self._state.perf, perf_key, 0.0)
            return
        if text in self._field_names:
            return
        self._extra.pop(text, None)

    def to_legacy_dict(self) -> dict[str, Any]:
        payload = {name: getattr(self._state, name) for name in sorted(self._field_names)}
        payload.update(
            {
                "ui_refresh_ms_ema": float(self._state.perf.ui_refresh_ms_ema),
                "queue_age_ms": float(self._state.perf.queue_age_ms),
                "infer_interval_ms": float(self._state.perf.infer_interval_ms),
                "frame_drop_ratio": float(self._state.perf.frame_drop_ratio),
                "remote_snapshot_age_ms": float(self._state.perf.remote_snapshot_age_ms),
            }
        )
        payload.update(self._extra)
        return payload

    def _normalize_value(self, key: str, value: Any) -> Any:
        if key == "target_frequency_map":
            if isinstance(value, Iterable):
                return [tuple(item) for item in value]  # type: ignore[misc]
            return []
        if key == "ssvep_available_profiles":
            if isinstance(value, Iterable):
                normalized: list[tuple[str, str]] = []
                for item in value:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        normalized.append((str(item[0]), str(item[1])))
                return tuple(normalized)
            return ()
        return value


class RuntimeInfoCompat(MutableMapping[str, Any]):
    def __init__(self, store: RuntimeStore) -> None:
        self._store = store

    def __getitem__(self, key: str) -> Any:
        value = self._store.get_value(key, None)
        if value is None and key not in self:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self._store.set_value(str(key), value)

    def __delitem__(self, key: str) -> None:
        self._store.delete_value(str(key))

    def __iter__(self):
        return iter(self._store.to_legacy_dict())

    def __len__(self) -> int:
        return len(self._store.to_legacy_dict())

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        payload = self._store.to_legacy_dict()
        return key in payload

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get_value(str(key), default)

