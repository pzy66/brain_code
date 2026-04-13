from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional


class DecisionState:
    IDLE = "Idle"
    CANDIDATE = "Candidate"
    ARMED = "Armed"
    COMMIT = "Commit"
    REFRACTORY = "Refractory"


@dataclass(frozen=True)
class StateMachineConfig:
    candidate_min_windows: int = 2
    armed_min_windows: int = 3
    commit_consistency_th: float = 0.6
    enter_gate_th: float = 0.0
    exit_gate_th: float = -0.2
    refractory_sec: float = 0.8


class FiveStateMachine:
    def __init__(self, config: Optional[StateMachineConfig] = None) -> None:
        self.config = config or StateMachineConfig()
        self.state = DecisionState.IDLE
        self.selected_freq: Optional[float] = None
        self.stable_windows = 0
        self._refractory_until = 0.0

    def reset(self) -> None:
        self.state = DecisionState.IDLE
        self.selected_freq = None
        self.stable_windows = 0
        self._refractory_until = 0.0

    def _to_idle(self) -> None:
        self.state = DecisionState.IDLE
        self.selected_freq = None
        self.stable_windows = 0

    def step(
        self,
        *,
        pred_freq: Optional[float],
        gate_score: float,
        evidence_score: float,
        consistency: float,
        upper_commit_th: float,
        lower_idle_th: float,
        timestamp_s: Optional[float] = None,
    ) -> dict[str, object]:
        cfg = self.config
        now = float(time.monotonic() if timestamp_s is None else timestamp_s)
        freq_value = None if pred_freq is None else float(pred_freq)
        commit = False

        if self.state == DecisionState.REFRACTORY and now >= float(self._refractory_until):
            self._to_idle()

        if self.state == DecisionState.REFRACTORY:
            return {
                "state": DecisionState.REFRACTORY,
                "commit": False,
                "selected_freq": None,
                "stable_windows": 0,
                "refractory_remaining_sec": max(float(self._refractory_until - now), 0.0),
            }

        if freq_value is None:
            self._to_idle()
            return {
                "state": self.state,
                "commit": False,
                "selected_freq": None,
                "stable_windows": 0,
                "refractory_remaining_sec": 0.0,
            }

        if self.selected_freq is not None and abs(float(self.selected_freq) - freq_value) <= 1e-8:
            self.stable_windows += 1
        else:
            self.selected_freq = freq_value
            self.stable_windows = 1

        if float(gate_score) <= float(cfg.exit_gate_th) or float(evidence_score) <= float(lower_idle_th):
            self._to_idle()
            return {
                "state": self.state,
                "commit": False,
                "selected_freq": None,
                "stable_windows": 0,
                "refractory_remaining_sec": 0.0,
            }

        if self.state == DecisionState.IDLE:
            if self.stable_windows >= int(cfg.candidate_min_windows) and float(gate_score) >= float(cfg.enter_gate_th):
                self.state = DecisionState.CANDIDATE
            return {
                "state": self.state,
                "commit": False,
                "selected_freq": self.selected_freq,
                "stable_windows": int(self.stable_windows),
                "refractory_remaining_sec": 0.0,
            }

        if self.state == DecisionState.CANDIDATE:
            if self.stable_windows >= int(cfg.armed_min_windows) and float(evidence_score) >= 0.75 * float(upper_commit_th):
                self.state = DecisionState.ARMED
            return {
                "state": self.state,
                "commit": False,
                "selected_freq": self.selected_freq,
                "stable_windows": int(self.stable_windows),
                "refractory_remaining_sec": 0.0,
            }

        if self.state == DecisionState.ARMED:
            if float(consistency) >= float(cfg.commit_consistency_th) and float(evidence_score) >= float(upper_commit_th):
                commit = True
                committed_freq = self.selected_freq
                self.state = DecisionState.COMMIT
                self._refractory_until = now + float(cfg.refractory_sec)
                self.state = DecisionState.REFRACTORY
                self.selected_freq = None
                self.stable_windows = 0
                return {
                    "state": DecisionState.COMMIT,
                    "commit": True,
                    "selected_freq": committed_freq,
                    "stable_windows": int(cfg.armed_min_windows),
                    "refractory_remaining_sec": float(cfg.refractory_sec),
                }

        return {
            "state": self.state,
            "commit": bool(commit),
            "selected_freq": self.selected_freq,
            "stable_windows": int(self.stable_windows),
            "refractory_remaining_sec": 0.0,
        }

