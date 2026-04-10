from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from hybrid_controller.controller.events import Event


class ReplaySource:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def iter_records(self, kinds: set[str] | None = None) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if kinds is not None and payload.get("kind") not in kinds:
                    continue
                yield payload

    def iter_events(self) -> Iterator[Event]:
        for payload in self.iter_records({"event"}):
            yield Event(
                source=payload["source"],
                type=payload["type"],
                value=payload.get("value"),
                confidence=payload.get("confidence"),
                timestamp=float(payload.get("event_ts", payload.get("ts", 0.0))),
            )

    def iter_world_snapshots(self) -> Iterator[dict[str, Any]]:
        for payload in self.iter_records({"world_snapshot"}):
            yield payload["snapshot"]

    def replay(self, callback: Callable[[Event], None], sleep: bool = False) -> None:
        previous_ts = None
        for event in self.iter_events():
            if sleep and previous_ts is not None and event.timestamp > previous_ts:
                time.sleep(event.timestamp - previous_ts)
            callback(event)
            previous_ts = event.timestamp

    def replay_records(
        self,
        callback: Callable[[dict[str, Any]], None],
        *,
        sleep: bool = False,
        kinds: set[str] | None = None,
    ) -> None:
        previous_ts = None
        for payload in self.iter_records(kinds):
            record_ts = float(payload.get("ts", 0.0))
            if sleep and previous_ts is not None and record_ts > previous_ts:
                time.sleep(record_ts - previous_ts)
            callback(payload)
            previous_ts = record_ts
