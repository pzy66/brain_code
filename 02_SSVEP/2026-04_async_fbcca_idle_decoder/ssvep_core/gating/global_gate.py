from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .base import BaseGate, GateOutput, logit, sigmoid


@dataclass(frozen=True)
class GlobalGateConfig:
    enter_score_th: float = 0.30
    enter_ratio_th: float = 1.05
    enter_margin_th: float = 0.04
    weight_top1: float = 2.4
    weight_ratio: float = 1.6
    weight_margin: float = 1.4
    weight_norm: float = 0.9
    weight_entropy: float = 0.7
    bias: float = -1.0


class GlobalThresholdGate(BaseGate):
    def __init__(self, config: Optional[GlobalGateConfig] = None) -> None:
        super().__init__(name="global_gate")
        self.config = config or GlobalGateConfig()

    @classmethod
    def from_profile(cls, profile: Any) -> "GlobalThresholdGate":
        cfg = GlobalGateConfig(
            enter_score_th=float(getattr(profile, "enter_score_th", 0.30)),
            enter_ratio_th=float(getattr(profile, "enter_ratio_th", 1.05)),
            enter_margin_th=float(getattr(profile, "enter_margin_th", 0.04)),
        )
        return cls(config=cfg)

    def _safe_feature(self, feature_row: Mapping[str, Any], key: str, default: float) -> float:
        value = feature_row.get(key, default)
        try:
            return float(value)
        except Exception:
            return float(default)

    def predict(self, feature_row: Mapping[str, Any], pred_freq: Optional[float]) -> GateOutput:
        cfg = self.config
        top1 = self._safe_feature(feature_row, "top1_score", 0.0)
        ratio = self._safe_feature(feature_row, "ratio", 1.0)
        margin = self._safe_feature(feature_row, "margin", 0.0)
        norm_top1 = self._safe_feature(feature_row, "normalized_top1", 0.0)
        entropy = self._safe_feature(feature_row, "score_entropy", self._safe_feature(feature_row, "entropy", 0.0))

        z = float(cfg.bias)
        z += float(cfg.weight_top1) * float(top1 - cfg.enter_score_th)
        z += float(cfg.weight_ratio) * float(ratio - cfg.enter_ratio_th)
        z += float(cfg.weight_margin) * float(margin - cfg.enter_margin_th)
        z += float(cfg.weight_norm) * float(norm_top1)
        z -= float(cfg.weight_entropy) * float(entropy)
        p = sigmoid(z)
        return GateOutput(
            p_control=float(p),
            gate_score=float(logit(p)),
            pred_freq=None if pred_freq is None else float(pred_freq),
            gate_name=self.name,
            feature_vector=(float(top1), float(ratio), float(margin), float(norm_top1), float(entropy)),
            model_found=True,
        )

