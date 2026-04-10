from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable


@dataclass(frozen=True, slots=True)
class VisionTarget:
    id: int
    bbox: tuple[float, float, float, float]
    center_px: tuple[float, float]
    raw_center: tuple[float, float]
    confidence: float
    command_mode: str = "pixel"
    command_point: tuple[float, float] | None = None
    display_center: tuple[float, float] | None = None
    slot_id: int | None = None
    freq_hz: float | None = None
    cylindrical_center: tuple[float, float, float] | None = None
    world_xyz: tuple[float, float, float] | None = None
    mapping_mode: str = "absolute_base"
    actionable: bool = True
    invalid_reason: str = ""

    def __post_init__(self) -> None:
        if self.command_point is None and str(self.command_mode or "").strip().lower() == "pixel":
            object.__setattr__(self, "command_point", self.raw_center)
        if self.display_center is None:
            object.__setattr__(self, "display_center", self.center_px)

    def distance_to(self, point: tuple[float, float]) -> float:
        center = self.display_center or self.center_px
        return hypot(center[0] - point[0], center[1] - point[1])

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "bbox": self.bbox,
            "center_px": self.center_px,
            "raw_center": self.raw_center,
            "confidence": self.confidence,
            "command_mode": self.command_mode,
            "command_point": self.command_point,
            "display_center": self.display_center or self.center_px,
            "slot_id": self.slot_id,
            "freq_hz": self.freq_hz,
            "cylindrical_center": self.cylindrical_center,
            "world_xyz": self.world_xyz,
            "mapping_mode": self.mapping_mode,
            "actionable": self.actionable,
            "invalid_reason": self.invalid_reason,
        }


def normalize_detections(detections: Iterable[object]) -> list[VisionTarget]:
    normalized: list[VisionTarget] = []
    for index, item in enumerate(detections):
        if isinstance(item, VisionTarget):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            bbox = tuple(item["bbox"])
            center_px = tuple(item.get("center_px") or _center_of_bbox(bbox))
            raw_center = tuple(item.get("raw_center") or center_px)
            command_mode = str(item.get("command_mode", "pixel"))
            command_point_raw = item.get("command_point")
            if command_point_raw is None:
                command_point = _as_xy(raw_center) if command_mode.strip().lower() == "pixel" else None
            else:
                command_point = _as_xy(tuple(command_point_raw))
            display_center = tuple(item.get("display_center") or center_px)
            normalized.append(
                VisionTarget(
                    id=int(item.get("id", index)),
                    bbox=_as_box(bbox),
                    center_px=_as_xy(center_px),
                    raw_center=_as_xy(raw_center),
                    confidence=float(item.get("confidence", 1.0)),
                    command_mode=command_mode,
                    command_point=command_point,
                    display_center=_as_xy(display_center),
                    slot_id=None if item.get("slot_id") is None else int(item.get("slot_id")),
                    freq_hz=None if item.get("freq_hz") is None else float(item.get("freq_hz")),
                    cylindrical_center=(
                        None
                        if item.get("cylindrical_center") is None
                        else _as_xyz(item.get("cylindrical_center"))
                    ),
                    world_xyz=(
                        None
                        if item.get("world_xyz") is None
                        else _as_xyz(item.get("world_xyz"))
                    ),
                    mapping_mode=str(item.get("mapping_mode", "absolute_base")),
                    actionable=bool(item.get("actionable", True)),
                    invalid_reason=str(item.get("invalid_reason", "")),
                )
            )
            continue
        if isinstance(item, (list, tuple)) and len(item) == 4:
            bbox = _as_box(item)
            center = _center_of_bbox(bbox)
            normalized.append(
                VisionTarget(
                    id=index,
                    bbox=bbox,
                    center_px=center,
                    raw_center=center,
                    confidence=1.0,
                    command_mode="pixel",
                    command_point=center,
                    display_center=center,
                )
            )
    return normalized


def snapshot_targets(
    targets: Iterable[VisionTarget],
    roi_center: tuple[float, float],
    roi_radius: float,
    limit: int,
) -> list[VisionTarget]:
    ranked = [
        (target.distance_to(roi_center), target)
        for target in targets
        if target.distance_to(roi_center) <= roi_radius
    ]
    ranked.sort(key=lambda item: (item[0], -item[1].confidence, item[1].id))
    return [item[1] for item in ranked[:limit]]


def _center_of_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _as_box(values: object) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = values
    return float(x1), float(y1), float(x2), float(y2)


def _as_xy(values: object) -> tuple[float, float]:
    x, y = values
    return float(x), float(y)


def _as_xyz(values: object) -> tuple[float, float, float]:
    x, y, z = values
    return float(x), float(y), float(z)
