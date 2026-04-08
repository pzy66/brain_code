from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from hybrid_controller.cylindrical import cartesian_to_cylindrical, cylindrical_to_cartesian
from hybrid_controller.adapters.vision_adapter import VisionTarget
from hybrid_controller.config import AppConfig, ControlSimSlotSpec


@dataclass(frozen=True, slots=True)
class ControlSimSlot:
    slot_id: int
    name: str
    world_xy: tuple[float, float]
    pixel_xy: tuple[float, float]
    cylindrical_trz: tuple[float, float, float]
    role: str

    def to_target(self, *, confidence: float = 1.0, command_mode: str = "pixel") -> VisionTarget:
        center_x, center_y = self.pixel_xy
        half_size = 28.0
        bbox = (center_x - half_size, center_y - half_size, center_x + half_size, center_y + half_size)
        command_mode = str(command_mode or "pixel").strip().lower()
        if command_mode == "pixel":
            command_point = self.pixel_xy
        elif command_mode == "cyl":
            command_point = self.cylindrical_trz[:2]
        else:
            command_point = self.world_xy
        return VisionTarget(
            id=self.slot_id,
            bbox=bbox,
            center_px=self.pixel_xy,
            raw_center=command_point,
            confidence=float(confidence),
            command_mode=command_mode,
            command_point=command_point,
            display_center=self.pixel_xy,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "name": self.name,
            "world_xy": self.world_xy,
            "pixel_xy": self.pixel_xy,
            "cylindrical_trz": self.cylindrical_trz,
            "role": self.role,
        }


class ControlSimSlotCatalog:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._pick_slots = self._normalize_slots(config.sim_pick_slots, role="pick")
        self._place_slots = self._normalize_slots(config.sim_place_slots, role="place")
        self._hardware_pick_slots = self._normalize_slots(config.hardware_pick_slots, role="pick")

    def list_pick_slots(self, *, source: str = "sim") -> list[ControlSimSlot]:
        return list(self._pick_slots_for_source(source))

    def list_place_slots(self) -> list[ControlSimSlot]:
        return list(self._place_slots)

    def build_selection_targets(self, *, source: str = "sim", command_mode: str = "pixel") -> list[VisionTarget]:
        targets: list[VisionTarget] = []
        for index, slot in enumerate(self._pick_slots_for_source(source)):
            targets.append(slot.to_target(confidence=max(0.7, 1.0 - index * 0.05), command_mode=command_mode))
        return targets

    def resolve_pick_slot(self, pixel_x: float, pixel_y: float) -> ControlSimSlot | None:
        tolerance = float(self.config.control_sim_slot_tolerance_px)
        candidates = sorted(
            (
                (hypot(slot.pixel_xy[0] - pixel_x, slot.pixel_xy[1] - pixel_y), slot)
                for slot in self._pick_slots
            ),
            key=lambda item: (item[0], item[1].slot_id),
        )
        if not candidates or candidates[0][0] > tolerance:
            return None
        return candidates[0][1]

    def resolve_world_pick_slot(self, world_x: float, world_y: float, *, source: str = "hardware") -> ControlSimSlot | None:
        tolerance = float(self.config.robot_target_margin_mm)
        candidates = sorted(
            (
                (hypot(slot.world_xy[0] - world_x, slot.world_xy[1] - world_y), slot)
                for slot in self._pick_slots_for_source(source)
            ),
            key=lambda item: (item[0], item[1].slot_id),
        )
        if not candidates or candidates[0][0] > tolerance:
            return None
        return candidates[0][1]

    def nearest_place_slot(self, world_xy: tuple[float, float]) -> ControlSimSlot | None:
        candidates = sorted(
            (
                (hypot(slot.world_xy[0] - world_xy[0], slot.world_xy[1] - world_xy[1]), slot)
                for slot in self._place_slots
            ),
            key=lambda item: (item[0], item[1].slot_id),
        )
        if not candidates:
            return None
        if candidates[0][0] > float(self.config.control_sim_place_snap_distance_mm):
            return None
        return candidates[0][1]

    def _pick_slots_for_source(self, source: str) -> tuple[ControlSimSlot, ...]:
        normalized = str(source or "sim").strip().lower()
        if normalized in {"hardware", "fixed_world_slots"}:
            return self._hardware_pick_slots
        return self._pick_slots

    @staticmethod
    def _normalize_slots(
        specs: tuple[ControlSimSlotSpec, ...],
        *,
        role: str,
    ) -> tuple[ControlSimSlot, ...]:
        return tuple(
            ControlSimSlot(
                slot_id=int(spec.slot_id),
                name=str(spec.name),
                world_xy=ControlSimSlotCatalog._resolve_world_xy(spec),
                pixel_xy=(float(spec.pixel_xy[0]), float(spec.pixel_xy[1])),
                cylindrical_trz=ControlSimSlotCatalog._resolve_cylindrical(spec),
                role=role,
            )
            for spec in specs
        )

    @staticmethod
    def _resolve_world_xy(spec: ControlSimSlotSpec) -> tuple[float, float]:
        if spec.world_xy is not None:
            return (float(spec.world_xy[0]), float(spec.world_xy[1]))
        if spec.cylindrical_trz is None:
            raise ValueError(f"Slot {spec.slot_id} must define world_xy or cylindrical_trz")
        x_mm, y_mm, _ = cylindrical_to_cartesian(
            float(spec.cylindrical_trz[0]),
            float(spec.cylindrical_trz[1]),
            float(spec.cylindrical_trz[2]),
        )
        return (x_mm, y_mm)

    @staticmethod
    def _resolve_cylindrical(spec: ControlSimSlotSpec) -> tuple[float, float, float]:
        if spec.cylindrical_trz is not None:
            return (
                float(spec.cylindrical_trz[0]),
                float(spec.cylindrical_trz[1]),
                float(spec.cylindrical_trz[2]),
            )
        if spec.world_xy is None:
            raise ValueError(f"Slot {spec.slot_id} must define world_xy or cylindrical_trz")
        theta_deg, radius_mm, _ = cartesian_to_cylindrical(
            float(spec.world_xy[0]),
            float(spec.world_xy[1]),
            85.0,
        )
        return (theta_deg, radius_mm, 85.0)
