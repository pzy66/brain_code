from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .accumulator import EvidenceAccumulator, EvidenceAccumulatorConfig
from .state_machine import FiveStateMachine, StateMachineConfig


@dataclass(frozen=True)
class DecisionEngineConfig:
    evidence: EvidenceAccumulatorConfig = EvidenceAccumulatorConfig()
    state: StateMachineConfig = StateMachineConfig()


class DecisionEngine:
    def __init__(self, config: Optional[DecisionEngineConfig] = None) -> None:
        self.config = config or DecisionEngineConfig()
        self.accumulator = EvidenceAccumulator(self.config.evidence)
        self.state_machine = FiveStateMachine(self.config.state)

    def reset(self) -> None:
        self.accumulator.reset(0.0)
        self.state_machine.reset()

    def step(
        self,
        pred_freq: Optional[float],
        gate_score: float,
        consistency: float,
        *,
        prior: float = 0.0,
        timestamp_s: Optional[float] = None,
    ) -> dict[str, object]:
        evidence_score = self.accumulator.update(
            gate_score=float(gate_score),
            consistency=float(consistency),
            prior=float(prior),
        )
        result = self.state_machine.step(
            pred_freq=pred_freq,
            gate_score=float(gate_score),
            evidence_score=float(evidence_score),
            consistency=float(consistency),
            upper_commit_th=float(self.config.evidence.upper_commit_th),
            lower_idle_th=float(self.config.evidence.lower_idle_th),
            timestamp_s=timestamp_s,
        )
        if bool(result.get("commit", False)):
            self.accumulator.reset(0.0)
        if str(result.get("state", "")) == "Idle" and float(evidence_score) <= float(self.config.evidence.lower_idle_th):
            self.accumulator.reset(0.0)
            evidence_score = 0.0
        payload = {
            "state": str(result.get("state", "Idle")),
            "commit": bool(result.get("commit", False)),
            "selected_freq": result.get("selected_freq"),
            "stable_windows": int(result.get("stable_windows", 0) or 0),
            "refractory_remaining_sec": float(result.get("refractory_remaining_sec", 0.0) or 0.0),
            "evidence_score": float(evidence_score),
        }
        return payload

