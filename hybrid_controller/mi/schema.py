from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


MiDirection = Literal["left", "right", "forward", "backward"]


@dataclass(frozen=True, slots=True)
class MiFrameSchema:
    direction: MiDirection | None
    confidence: float
    timestamp: float
    raw_label: str = ""
    raw: Mapping[str, object] | None = None

