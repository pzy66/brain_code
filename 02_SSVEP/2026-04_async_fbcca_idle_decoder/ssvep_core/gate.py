from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from async_fbcca_idle_standalone import AsyncDecisionGate, ThresholdProfile


@dataclass
class AsyncGateAdapter:
    gate: AsyncDecisionGate

    @classmethod
    def from_profile(cls, profile: ThresholdProfile) -> "AsyncGateAdapter":
        return cls(gate=AsyncDecisionGate.from_profile(profile))

    def update(self, features: dict[str, Any]) -> dict[str, Any]:
        return dict(self.gate.update(features))

    def reset(self) -> None:
        self.gate.reset()

