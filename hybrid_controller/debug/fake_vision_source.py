from __future__ import annotations

from hybrid_controller.adapters.vision_adapter import VisionTarget

from .simulation_world import SimulationWorld


class FakeVisionSource:
    def __init__(self, world: SimulationWorld) -> None:
        self.world = world

    def snapshot(self) -> list[VisionTarget]:
        return self.world.visible_targets()
