from __future__ import annotations

import atexit
from dataclasses import asdict, is_dataclass
import json
from queue import Empty, Full, Queue
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
        self._queue: Queue[dict[str, Any]] = Queue(maxsize=8192)
        self._stop_event = threading.Event()
        self._closed = False
        self._close_lock = threading.Lock()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="hybrid-event-logger",
            daemon=True,
        )
        self._writer_thread.start()
        atexit.register(self.shutdown)

    def write(self, kind: str, **payload: Any) -> None:
        record = {"ts": time.time(), "kind": kind, **payload}
        if self._closed:
            return
        try:
            self._queue.put_nowait(record)
            return
        except Full:
            # Fallback path: if the queue is temporarily full, do a direct write
            # to avoid silently dropping important records.
            self._write_record(record)

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

    def shutdown(self, timeout_sec: float = 1.5) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            try:
                self._queue.put_nowait({})
            except Full:
                pass
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=max(0.1, float(timeout_sec)))
        self._drain_queue_sync()

    def _writer_loop(self) -> None:
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                break
            try:
                record = self._queue.get(timeout=0.15)
            except Empty:
                continue
            if not record:
                continue
            self._write_record(record)

    def _drain_queue_sync(self) -> None:
        while True:
            try:
                record = self._queue.get_nowait()
            except Empty:
                return
            if not record:
                continue
            self._write_record(record)

    def _write_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=self._json_default) + "\n")

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
