from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

EventSource = Literal["mi", "ssvep", "vision", "robot", "sim", "system"]
EventType = Literal[
    "start_task",
    "move",
    "timer_expired",
    "decision_confirm",
    "decision_cancel",
    "target_selected",
    "vision_update",
    "robot_ack",
    "robot_busy",
    "robot_error",
    "reset_task",
]
EffectType = Literal["state_changed", "robot_command", "start_timer", "cancel_timer", "log"]


@dataclass(slots=True)
class Event:
    source: EventSource
    type: EventType
    value: Any = None
    confidence: Optional[float] = None
    timestamp: float = 0.0

    def with_default_timestamp(self) -> "Event":
        if self.timestamp > 0:
            return self
        return Event(
            source=self.source,
            type=self.type,
            value=self.value,
            confidence=self.confidence,
            timestamp=time.time(),
        )


@dataclass(slots=True)
class Effect:
    type: EffectType
    payload: Any = None
