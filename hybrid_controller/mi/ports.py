from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class BrainFlowPortResolver:
    """
    Minimal serial-port resolver for future BrainFlow integration.
    This module intentionally has no BrainFlow dependency in the placeholder phase.
    """

    env_keys: tuple[str, ...] = ("MI_SERIAL_PORT", "BRAINFLOW_SERIAL_PORT")

    def list_candidate_ports(self) -> list[str]:
        candidates: list[str] = []
        for env_key in self.env_keys:
            value = os.environ.get(env_key, "").strip()
            if value:
                candidates.append(value)
        try:
            from serial.tools import list_ports  # type: ignore[import]
        except Exception:
            return self._dedupe(candidates)
        for port in list_ports.comports():
            name = str(getattr(port, "device", "")).strip()
            if name:
                candidates.append(name)
        return self._dedupe(candidates)

    def first_candidate(self) -> str | None:
        candidates = self.list_candidate_ports()
        return candidates[0] if candidates else None

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

