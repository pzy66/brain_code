from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class VisionDetection:
    target_id: int | None
    pixel_xy: tuple[float, float] | None
    bbox: tuple[float, float, float, float] | None = None
    mask: Any | None = None
    confidence: float = 0.0
    angle_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VisionDetectionFrame:
    frame_id: int
    timestamp: str
    detections: tuple[VisionDetection, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": int(self.frame_id),
            "timestamp": str(self.timestamp),
            "detections": [detection.to_dict() for detection in self.detections],
        }
