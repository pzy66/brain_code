from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .base import BaseGate, GateOutput, logit, sigmoid

DEFAULT_GATE_FEATURE_NAMES = (
    "top1_score",
    "top2_score",
    "margin",
    "ratio",
    "score_entropy",
    "normalized_top1",
    "consistency",
    "margin_mean_k",
    "ratio_mean_k",
)


@dataclass
class LogRegFitConfig:
    learning_rate: float = 0.08
    epochs: int = 320
    l2: float = 1e-3
    min_samples: int = 24


@dataclass(frozen=True)
class FrequencyLogRegModel:
    freq: float
    coef: tuple[float, ...]
    intercept: float
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "coef": [float(item) for item in self.coef],
            "intercept": float(self.intercept),
            "mean": [float(item) for item in self.mean],
            "std": [float(item) for item in self.std],
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(output):
        return float(default)
    return float(output)


def _row_label_to_role(row: Mapping[str, Any]) -> str:
    role = str(row.get("trial_role", "")).strip().lower()
    if role:
        return role
    label = str(row.get("label", "")).strip().lower()
    if row.get("expected_freq") is not None and not label.startswith("switch_to_"):
        return "control"
    if "switch" in label or "transition" in label or "scan" in label or "hard" in label:
        return "hard_idle"
    if "long_idle" in label or "long idle" in label:
        return "hard_idle"
    return "clean_idle"


class PerFrequencyLogRegGate(BaseGate):
    def __init__(
        self,
        *,
        feature_names: Sequence[str] = DEFAULT_GATE_FEATURE_NAMES,
        models: Optional[dict[str, FrequencyLogRegModel]] = None,
    ) -> None:
        super().__init__(name="per_frequency_logreg")
        self.feature_names = tuple(str(item) for item in feature_names)
        self.models = dict(models or {})

    @classmethod
    def from_payload(
        cls,
        *,
        payload: Mapping[str, Any],
        feature_names: Optional[Sequence[str]] = None,
    ) -> "PerFrequencyLogRegGate":
        names = tuple(feature_names or payload.get("feature_names") or DEFAULT_GATE_FEATURE_NAMES)
        per_freq = dict(payload.get("per_freq", {}))
        models: dict[str, FrequencyLogRegModel] = {}
        for freq_key, item_raw in per_freq.items():
            item = dict(item_raw or {})
            coef = tuple(float(value) for value in item.get("coef", [0.0] * len(names)))
            mean = tuple(float(value) for value in item.get("mean", [0.0] * len(names)))
            std = tuple(
                float(value) if float(value) > 1e-8 else 1.0
                for value in item.get("std", [1.0] * len(names))
            )
            if len(coef) != len(names):
                coef = tuple([0.0] * len(names))
            if len(mean) != len(names):
                mean = tuple([0.0] * len(names))
            if len(std) != len(names):
                std = tuple([1.0] * len(names))
            models[str(freq_key)] = FrequencyLogRegModel(
                freq=float(freq_key),
                coef=coef,
                intercept=float(item.get("intercept", 0.0)),
                mean=mean,
                std=std,
            )
        return cls(feature_names=names, models=models)

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "frequency_specific_logreg",
            "feature_names": [str(item) for item in self.feature_names],
            "per_freq": {key: value.to_payload() for key, value in self.models.items()},
        }

    def _build_vector(self, row: Mapping[str, Any]) -> np.ndarray:
        values = [_safe_float(row.get(name, 0.0), 0.0) for name in self.feature_names]
        return np.asarray(values, dtype=np.float64)

    def predict(self, feature_row: Mapping[str, Any], pred_freq: Optional[float]) -> GateOutput:
        if pred_freq is None:
            return GateOutput(
                p_control=0.0,
                gate_score=logit(1e-6),
                pred_freq=None,
                gate_name=self.name,
                feature_vector=tuple(float(_safe_float(feature_row.get(name, 0.0), 0.0)) for name in self.feature_names),
                model_found=False,
            )
        key = f"{float(pred_freq):g}"
        model = self.models.get(key)
        vector = self._build_vector(feature_row)
        if model is None:
            return GateOutput(
                p_control=0.5,
                gate_score=0.0,
                pred_freq=float(pred_freq),
                gate_name=self.name,
                feature_vector=tuple(float(item) for item in vector.tolist()),
                model_found=False,
            )
        mean = np.asarray(model.mean, dtype=np.float64)
        std = np.asarray(model.std, dtype=np.float64)
        x = (vector - mean) / np.maximum(std, 1e-6)
        coef = np.asarray(model.coef, dtype=np.float64)
        z = float(np.dot(coef, x) + float(model.intercept))
        p = sigmoid(z)
        return GateOutput(
            p_control=float(p),
            gate_score=float(logit(p)),
            pred_freq=float(pred_freq),
            gate_name=self.name,
            feature_vector=tuple(float(item) for item in vector.tolist()),
            model_found=True,
        )

    def fit(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        freqs: Sequence[float],
        fit_config: Optional[LogRegFitConfig] = None,
    ) -> dict[str, Any]:
        cfg = fit_config or LogRegFitConfig()
        summary: dict[str, Any] = {"feature_names": list(self.feature_names), "per_freq": {}}
        fitted_models: dict[str, FrequencyLogRegModel] = {}
        for freq in freqs:
            freq_value = float(freq)
            freq_key = f"{freq_value:g}"
            filtered = [
                row
                for row in rows
                if abs(_safe_float(row.get("pred_freq", freq_value), freq_value) - freq_value) <= 1e-8
            ]
            if len(filtered) < int(cfg.min_samples):
                model = FrequencyLogRegModel(
                    freq=freq_value,
                    coef=tuple([0.0] * len(self.feature_names)),
                    intercept=0.0,
                    mean=tuple([0.0] * len(self.feature_names)),
                    std=tuple([1.0] * len(self.feature_names)),
                )
                fitted_models[freq_key] = model
                summary["per_freq"][freq_key] = {
                    "samples": int(len(filtered)),
                    "positive_ratio": 0.0,
                    "status": "insufficient_samples",
                }
                continue

            x = np.asarray([self._build_vector(row) for row in filtered], dtype=np.float64)
            y = np.asarray(
                [
                    1.0
                    if (
                        _row_label_to_role(row) == "control"
                        and abs(_safe_float(row.get("expected_freq", float("nan")), float("nan")) - freq_value) <= 1e-8
                    )
                    else 0.0
                    for row in filtered
                ],
                dtype=np.float64,
            )
            if float(np.sum(y)) <= 0.0 or float(np.sum(1.0 - y)) <= 0.0:
                model = FrequencyLogRegModel(
                    freq=freq_value,
                    coef=tuple([0.0] * len(self.feature_names)),
                    intercept=float(logit(float(np.clip(np.mean(y), 1e-3, 1.0 - 1e-3)))),
                    mean=tuple([0.0] * len(self.feature_names)),
                    std=tuple([1.0] * len(self.feature_names)),
                )
                fitted_models[freq_key] = model
                summary["per_freq"][freq_key] = {
                    "samples": int(len(filtered)),
                    "positive_ratio": float(np.mean(y)),
                    "status": "single_class_fallback",
                }
                continue

            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            std = np.where(std > 1e-6, std, 1.0)
            x_norm = (x - mean) / std
            weight_pos = float(np.sum(1.0 - y) / max(np.sum(y), 1e-6))

            w = np.zeros((x_norm.shape[1],), dtype=np.float64)
            b = 0.0
            for _ in range(max(1, int(cfg.epochs))):
                logits = np.dot(x_norm, w) + b
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
                sample_weights = np.where(y > 0.5, weight_pos, 1.0)
                error = (probs - y) * sample_weights
                grad_w = np.dot(x_norm.T, error) / max(len(x_norm), 1) + float(cfg.l2) * w
                grad_b = float(np.sum(error) / max(len(x_norm), 1))
                w -= float(cfg.learning_rate) * grad_w
                b -= float(cfg.learning_rate) * grad_b

            model = FrequencyLogRegModel(
                freq=freq_value,
                coef=tuple(float(item) for item in w.tolist()),
                intercept=float(b),
                mean=tuple(float(item) for item in mean.tolist()),
                std=tuple(float(item) for item in std.tolist()),
            )
            fitted_models[freq_key] = model
            summary["per_freq"][freq_key] = {
                "samples": int(len(filtered)),
                "positive_ratio": float(np.mean(y)),
                "status": "ok",
            }

        self.models = fitted_models
        return summary

