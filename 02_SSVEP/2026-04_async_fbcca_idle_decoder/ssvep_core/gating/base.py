from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


def sigmoid(value: float) -> float:
    x = float(value)
    if x >= 0.0:
        z = float(np.exp(-x))
        return float(1.0 / (1.0 + z))
    z = float(np.exp(x))
    return float(z / (1.0 + z))


def logit(probability: float, *, eps: float = 1e-6) -> float:
    p = float(np.clip(float(probability), eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


@dataclass(frozen=True)
class GateOutput:
    p_control: float
    gate_score: float
    pred_freq: Optional[float]
    gate_name: str
    feature_vector: tuple[float, ...] = ()
    model_found: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "p_control": float(self.p_control),
            "gate_score": float(self.gate_score),
            "pred_freq": None if self.pred_freq is None else float(self.pred_freq),
            "gate_name": str(self.gate_name),
            "feature_vector": [float(item) for item in self.feature_vector],
            "model_found": bool(self.model_found),
        }


class BaseGate(ABC):
    name: str

    def __init__(self, *, name: str) -> None:
        self.name = str(name)

    @abstractmethod
    def predict(self, feature_row: Mapping[str, Any], pred_freq: Optional[float]) -> GateOutput:
        raise NotImplementedError

    def reset(self) -> None:
        return

