from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, runtime_checkable


MiMoveDirection = Literal["left", "right", "forward", "backward"]


@dataclass(frozen=True, slots=True)
class InputFrame:
    source: str
    direction: MiMoveDirection | None = None
    confidence: float | None = None
    timestamp: float | None = None
    raw: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class InputHealth:
    source: str
    connected: bool
    running: bool
    ready: bool
    status: str
    last_error: str = "--"


@runtime_checkable
class InputProvider(Protocol):
    def connect(self) -> bool: ...

    def start(self) -> bool: ...

    def stop(self) -> None: ...

    def read_frame(self) -> InputFrame | None: ...

    def set_mode(self, mode: str) -> None: ...

    def healthcheck(self) -> InputHealth: ...

