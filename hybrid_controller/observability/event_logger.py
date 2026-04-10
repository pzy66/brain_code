from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import threading
import time
from pathlib import Path
from typing import Any

from hybrid_controller.controller.events import Effect, Event


class EventLogger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, kind: str, **payload: Any) -> None:
        record = {"ts": time.time(), "kind": kind, **payload}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=self._json_default) + "\n")

    def log_event(self, event: Event, state: str) -> None:
        self.write(
            "event",
            state=state,
            source=event.source,
            type=event.type,
            value=event.value,
            confidence=event.confidence,
            event_ts=event.timestamp,
        )

    def log_effect(self, effect: Effect, state: str) -> None:
        self.write("effect", state=state, effect_type=effect.type, payload=effect.payload)

    def log_raw_input(self, source: str, payload: Any) -> None:
        self.write("raw_input", source=source, payload=payload)

    def log_runtime_status(self, component: str, message: str) -> None:
        self.write("runtime_status", component=component, message=message)

    def log_vision_snapshot(self, targets: Any, state: str, *, scenario: str) -> None:
        self.write("vision_snapshot", state=state, scenario=scenario, targets=targets)

    def log_world_snapshot(self, snapshot: dict[str, Any], *, reason: str) -> None:
        self.write("world_snapshot", reason=reason, snapshot=snapshot)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return sorted(value)
        return repr(value)
