from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .accumulator import EvidenceAccumulator, EvidenceAccumulatorConfig
from .state_machine import FiveStateMachine, StateMachineConfig

_UNSET = object()


@dataclass(frozen=True)
class DecisionEngineConfig:
    evidence: EvidenceAccumulatorConfig = EvidenceAccumulatorConfig()
    state: StateMachineConfig = StateMachineConfig()


class DecisionEngine:
    def __init__(self, config: Optional[DecisionEngineConfig] = None) -> None:
        self.config = config or DecisionEngineConfig()
        self.accumulator = EvidenceAccumulator(self.config.evidence)
        self.state_machine = FiveStateMachine(self.config.state)
        self.active_freq: Optional[float] = None

    def reset(self) -> None:
        self.accumulator.reset(0.0)
        self.state_machine.reset()
        self.active_freq = None

    def step(
        self,
        pred_freq: Optional[float],
        gate_score: float,
        consistency: float,
        *,
        gate_open_freq: object = _UNSET,
        prior: float = 0.0,
        timestamp_s: Optional[float] = None,
    ) -> dict[str, object]:
        if gate_open_freq is _UNSET:
            effective_pred = pred_freq
            gate_open_value = pred_freq
        else:
            gate_open_value = None if gate_open_freq is None else float(gate_open_freq)
            effective_pred = gate_open_value
        evidence_score = self.accumulator.update(
            gate_score=float(gate_score),
            consistency=float(consistency),
            prior=float(prior),
        )
        result = self.state_machine.step(
            pred_freq=effective_pred,
            gate_score=float(gate_score),
            evidence_score=float(evidence_score),
            consistency=float(consistency),
            upper_commit_th=float(self.config.evidence.upper_commit_th),
            lower_idle_th=float(self.config.evidence.lower_idle_th),
            timestamp_s=timestamp_s,
        )
        state_name = str(result.get("state", "Idle"))
        tracked_freq = result.get("selected_freq")
        commit = bool(result.get("commit", False))
        commit_freq = None if tracked_freq is None else float(tracked_freq)
        release = bool(
            self.active_freq is not None
            and state_name == "Idle"
            and gate_open_value is None
        )
        if commit and commit_freq is not None:
            self.active_freq = float(commit_freq)
        elif release:
            self.active_freq = None
        visible_selected = self.active_freq
        if commit:
            self.accumulator.reset(0.0)
        if str(result.get("state", "")) == "Idle" and float(evidence_score) <= float(self.config.evidence.lower_idle_th):
            self.accumulator.reset(0.0)
            evidence_score = 0.0
        payload = {
            "state": state_name,
            "commit": bool(commit),
            "selected_freq": visible_selected,
            "tracked_freq": None if tracked_freq is None else float(tracked_freq),
            "commit_freq": None if not commit or commit_freq is None else float(commit_freq),
            "release": bool(release),
            "stable_windows": int(result.get("stable_windows", 0) or 0),
            "refractory_remaining_sec": float(result.get("refractory_remaining_sec", 0.0) or 0.0),
            "decision_input_score": float(gate_score),
            "evidence_score": float(evidence_score),
            "pred_freq_raw": None if pred_freq is None else float(pred_freq),
            "gate_open_freq": gate_open_value,
        }
        return payload
