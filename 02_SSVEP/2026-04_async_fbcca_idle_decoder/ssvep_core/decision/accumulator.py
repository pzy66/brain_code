from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceAccumulatorConfig:
    lambda_decay: float = 0.85
    beta_consistency: float = 0.5
    upper_commit_th: float = 2.2
    lower_idle_th: float = 0.4


class EvidenceAccumulator:
    def __init__(self, config: EvidenceAccumulatorConfig | None = None) -> None:
        self.config = config or EvidenceAccumulatorConfig()
        self.score = 0.0

    def reset(self, value: float = 0.0) -> None:
        self.score = float(value)

    def update(
        self,
        *,
        gate_score: float,
        consistency: float,
        prior: float = 0.0,
    ) -> float:
        cfg = self.config
        self.score = (
            float(cfg.lambda_decay) * float(self.score)
            + float(gate_score)
            + float(cfg.beta_consistency) * float(consistency)
            + float(prior)
        )
        return float(self.score)

