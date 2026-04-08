from __future__ import annotations

from hybrid_controller.controller.events import Event


class SimInputAdapter:
    def __init__(self, move_source: str = "sim", decision_source: str = "sim") -> None:
        self.move_source = move_source
        self.decision_source = decision_source

    def set_sources(self, *, move_source: str | None = None, decision_source: str | None = None) -> None:
        if move_source is not None:
            self.move_source = move_source
        if decision_source is not None:
            self.decision_source = decision_source

    def handle_key_token(self, token: str) -> list[Event]:
        normalized = str(token or "").strip().lower()
        if not normalized:
            return []
        if normalized == "n":
            return [Event(source="sim", type="start_task")]
        if normalized == "r":
            return [Event(source="sim", type="reset_task")]
        if self.move_source == "sim":
            move_mapping = {
                "a": "left",
                "left": "left",
                "d": "right",
                "right": "right",
                "w": "forward",
                "up": "forward",
                "s": "backward",
                "down": "backward",
            }
            direction = move_mapping.get(normalized)
            if direction is not None:
                return [Event(source="sim", type="move", value=direction)]
        if self.decision_source == "sim":
            if normalized in {"enter", "c"}:
                return [Event(source="sim", type="decision_confirm")]
            if normalized in {"escape", "x"}:
                return [Event(source="sim", type="decision_cancel")]
            if normalized in {"1", "2", "3", "4"}:
                return [Event(source="sim", type="target_selected", value=int(normalized) - 1)]
        return []
