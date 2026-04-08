from __future__ import annotations

import time
from typing import Any, Optional

from hybrid_controller.config import AppConfig
from hybrid_controller.controller.events import Event


class MIAdapter:
    CLASS_TO_DIRECTION = {
        "left_hand": "left",
        "left hand": "left",
        "right_hand": "right",
        "right hand": "right",
        "feet": "backward",
        "tongue": "forward",
    }

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._last_direction: Optional[str] = None
        self._stable_count = 0
        self._last_emit_ms = -10**9
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        self._last_direction = None
        self._stable_count = 0

    def latest_intent(self) -> Optional[str]:
        return self._last_direction

    def process_result(self, result: dict[str, Any], timestamp_ms: Optional[int] = None) -> Optional[Event]:
        if not self._running:
            return None
        direction = self._extract_direction(result)
        confidence = self._extract_confidence(result)
        if direction is None or confidence < self.config.mi_min_confidence:
            self._last_direction = None
            self._stable_count = 0
            return None
        if direction == self._last_direction:
            self._stable_count += 1
        else:
            self._last_direction = direction
            self._stable_count = 1
        if self._stable_count < self.config.mi_stable_windows:
            return None
        now_ms = int(time.monotonic() * 1000) if timestamp_ms is None else int(timestamp_ms)
        if now_ms - self._last_emit_ms < self.config.mi_emit_interval_ms:
            return None
        self._last_emit_ms = now_ms
        return Event(source="mi", type="move", value=direction, confidence=confidence)

    def _extract_direction(self, result: dict[str, Any]) -> Optional[str]:
        candidates = [
            result.get("stable_prediction"),
            result.get("stable_prediction_display_name"),
            result.get("prediction"),
            result.get("prediction_display_name"),
        ]
        for item in candidates:
            if item is None:
                continue
            normalized = str(item).strip().lower().replace("_", " ")
            normalized = normalized.replace("(", " ").replace(")", " ")
            normalized = " ".join(normalized.split())
            direction = self.CLASS_TO_DIRECTION.get(normalized)
            if direction is not None:
                return direction
        return None

    @staticmethod
    def _extract_confidence(result: dict[str, Any]) -> float:
        for key in ("stable_confidence", "confidence"):
            value = result.get(key)
            if value is not None:
                return float(value)
        return 0.0
