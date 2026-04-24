from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .base import logit, sigmoid
from ..trial_roles import resolve_trial_role

GLOBAL_CORRECTNESS_LOGISTIC = "global_correctness_logistic"
BAYESIAN_GAP_GMM = "bayesian_gap_gmm"
FREQ_SHRUNK_LOGISTIC = "freq_shrunk_logistic"
FREQ_SHRUNK_GAP_GMM = "freq_shrunk_gap_gmm"
CORRECTNESS_VARIANTS = (
    GLOBAL_CORRECTNESS_LOGISTIC,
    BAYESIAN_GAP_GMM,
    FREQ_SHRUNK_LOGISTIC,
    FREQ_SHRUNK_GAP_GMM,
)
DEFAULT_CORRECTNESS_FEATURE_NAMES = (
    "gate_score",
    "margin",
    "ratio",
    "consistency",
    "normalized_top1",
    "score_entropy",
    "gap_12",
    "gap_13",
    "gap_14",
)
DEFAULT_BAYESIAN_GAP_FEATURE_NAMES = (
    "gap_12",
    "gap_13",
    "gap_14",
    "normalized_top1",
    "score_entropy",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except Exception:
        return float(default)
    return float(output) if np.isfinite(output) else float(default)


def _clip_probability(value: float) -> float:
    return float(np.clip(float(value), 1e-6, 1.0 - 1e-6))


def _freq_key(value: Any) -> str:
    freq = _safe_float(value, float("nan"))
    if not np.isfinite(freq):
        return ""
    return f"{float(freq):g}"


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = np.asarray(scores[np.asarray(y_true, dtype=float) > 0.5], dtype=float)
    negatives = np.asarray(scores[np.asarray(y_true, dtype=float) <= 0.5], dtype=float)
    if positives.size <= 0 or negatives.size <= 0:
        return None
    total = float(positives.size * negatives.size)
    wins = 0.0
    for value in positives:
        wins += float(np.sum(value > negatives))
        wins += 0.5 * float(np.sum(value == negatives))
    return float(wins / max(total, 1.0))


def _trial_key(row: Mapping[str, Any], *, fallback_index: int) -> str:
    value = row.get("trial_id")
    if value is None:
        return f"row_{int(fallback_index)}"
    try:
        return str(int(value))
    except Exception:
        return str(value)


def _build_sample_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> np.ndarray:
    sample_count = int(len(rows))
    if sample_count <= 0:
        return np.asarray([], dtype=np.float64)
    normalized_mode = str(mode or "per_trial_equal").strip().lower()
    if normalized_mode != "per_trial_equal":
        return np.ones((sample_count,), dtype=np.float64)
    trial_keys = [_trial_key(row, fallback_index=index) for index, row in enumerate(rows)]
    trial_counts: dict[str, int] = {}
    for key in trial_keys:
        trial_counts[key] = int(trial_counts.get(key, 1) + 0) if key in trial_counts else 1
        if key in trial_counts and trial_counts[key] > 1:
            continue
    trial_counts = {}
    for key in trial_keys:
        trial_counts[key] = int(trial_counts.get(key, 0)) + 1
    return np.asarray(
        [1.0 / max(int(trial_counts.get(key, 1)), 1) for key in trial_keys],
        dtype=np.float64,
    )


def _weighted_mean_and_std(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= 0:
        mean = np.zeros((x.shape[1] if x.ndim == 2 else 0,), dtype=np.float64)
        std = np.ones_like(mean)
        return mean, std
    normalized_weights = np.asarray(weights, dtype=np.float64)
    total_weight = float(np.sum(normalized_weights))
    if total_weight <= 1e-12:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
    else:
        mean = np.average(x, axis=0, weights=normalized_weights)
        centered = np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
        variance = np.average(centered * centered, axis=0, weights=normalized_weights)
        std = np.sqrt(np.maximum(variance, 0.0))
    std = np.where(np.asarray(std, dtype=np.float64) > 1e-6, std, 1.0)
    return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)


def _normalize_rows(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = _weighted_mean_and_std(x, weights)
    return (np.asarray(x, dtype=np.float64) - mean) / np.maximum(std, 1e-6), mean, std


def _effective_class_weights(y: np.ndarray, base_sample_weights: np.ndarray) -> np.ndarray:
    weighted_positive_mass = float(np.sum(base_sample_weights[y > 0.5]))
    weighted_negative_mass = float(np.sum(base_sample_weights[y <= 0.5]))
    if weighted_positive_mass <= 1e-12 or weighted_negative_mass <= 1e-12:
        return np.asarray(base_sample_weights, dtype=np.float64)
    weight_pos = float(weighted_negative_mass / max(weighted_positive_mass, 1e-6))
    return np.asarray(
        np.where(y > 0.5, weight_pos * np.asarray(base_sample_weights, dtype=np.float64), base_sample_weights),
        dtype=np.float64,
    )


def _pred_freq_indices(rows: Sequence[Mapping[str, Any]], freq_order: Sequence[float]) -> np.ndarray:
    lookup = {_freq_key(freq): index for index, freq in enumerate(freq_order)}
    output = []
    for row in rows:
        output.append(int(lookup.get(_freq_key(row.get("pred_freq")), -1)))
    return np.asarray(output, dtype=np.int64)


def _shrinkage_from_mass(mass: float, *, full_mass: float = 12.0) -> float:
    return float(np.clip(float(mass) / max(float(full_mass), 1e-6), 0.0, 1.0))


def _ece_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    weights: np.ndarray,
    *,
    bin_count: int = 10,
) -> Optional[float]:
    if y_true.size <= 0 or y_prob.size <= 0 or weights.size <= 0:
        return None
    bins = np.linspace(0.0, 1.0, int(max(bin_count, 1)) + 1)
    total_weight = float(np.sum(weights))
    if total_weight <= 1e-12:
        return None
    value = 0.0
    for index in range(len(bins) - 1):
        left = float(bins[index])
        right = float(bins[index + 1])
        if index == len(bins) - 2:
            mask = np.logical_and(y_prob >= left, y_prob <= right)
        else:
            mask = np.logical_and(y_prob >= left, y_prob < right)
        if not np.any(mask):
            continue
        bucket_weights = np.asarray(weights[mask], dtype=np.float64)
        bucket_mass = float(np.sum(bucket_weights))
        if bucket_mass <= 1e-12:
            continue
        bucket_acc = float(np.average(y_true[mask], weights=bucket_weights))
        bucket_conf = float(np.average(y_prob[mask], weights=bucket_weights))
        value += abs(bucket_acc - bucket_conf) * (bucket_mass / total_weight)
    return float(value)


def _logsumexp(values: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size <= 0:
        return np.asarray(-np.inf, dtype=np.float64)
    max_value = np.max(array, axis=axis, keepdims=True)
    stabilized = np.exp(array - max_value)
    summed = np.sum(stabilized, axis=axis, keepdims=True)
    output = max_value + np.log(np.maximum(summed, 1e-300))
    if axis is None:
        return np.asarray(float(output.reshape(-1)[0]), dtype=np.float64)
    return np.asarray(np.squeeze(output, axis=axis), dtype=np.float64)


def _diag_logpdf_matrix(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    safe_var = np.maximum(np.asarray(var, dtype=np.float64), 1e-6)
    centered = np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    quadratic = np.sum((centered * centered) / safe_var.reshape(1, -1), axis=1)
    log_det = float(np.sum(np.log(2.0 * np.pi * safe_var)))
    return np.asarray(-0.5 * (quadratic + log_det), dtype=np.float64)


def _weighted_diag_stats(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = float(np.sum(weights))
    if x.size <= 0 or total <= 1e-12:
        dim = int(x.shape[1]) if x.ndim == 2 else 0
        return (
            np.zeros((dim,), dtype=np.float64),
            np.ones((dim,), dtype=np.float64),
        )
    mean = np.average(x, axis=0, weights=weights)
    centered = np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    var = np.average(centered * centered, axis=0, weights=weights)
    return np.asarray(mean, dtype=np.float64), np.maximum(np.asarray(var, dtype=np.float64), 1e-6)


def _fit_single_diag_component(x: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    mean, var = _weighted_diag_stats(x, weights)
    total_weight = float(np.sum(np.asarray(weights, dtype=np.float64)))
    return {
        "weight": 1.0,
        "mean": [float(item) for item in mean.tolist()],
        "var": [float(item) for item in var.tolist()],
        "component_weight_mass": float(total_weight),
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if x.size <= 0:
        return 0.0
    if np.sum(w) <= 1e-12:
        return float(np.quantile(x, quantile))
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cumulative = np.cumsum(w_sorted)
    target = float(np.clip(float(quantile), 0.0, 1.0)) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = int(np.clip(index, 0, x_sorted.size - 1))
    return float(x_sorted[index])


def _fit_diag_gmm(
    x: np.ndarray,
    weights: np.ndarray,
    *,
    component_count: int,
    iterations: int = 32,
) -> dict[str, Any]:
    sample_count = int(x.shape[0])
    if sample_count <= 0 or int(component_count) <= 1:
        component = _fit_single_diag_component(x, weights)
        log_density = _diag_logpdf_matrix(
            x,
            np.asarray(component["mean"], dtype=np.float64),
            np.asarray(component["var"], dtype=np.float64),
        )
        log_likelihood = float(np.sum(np.asarray(weights, dtype=np.float64) * log_density))
        parameter_count = 2 * int(x.shape[1]) if x.ndim == 2 else 0
        bic = float(-2.0 * log_likelihood + parameter_count * np.log(max(sample_count, 1)))
        return {
            "components": [component],
            "component_count": 1,
            "log_likelihood": float(log_likelihood),
            "bic": float(bic),
        }

    total_weight = float(np.sum(weights))
    if total_weight <= 1e-12:
        weights = np.ones((sample_count,), dtype=np.float64)
        total_weight = float(sample_count)
    projection = np.asarray(x[:, 0], dtype=np.float64)
    lower = _weighted_quantile(projection, weights, 0.25)
    upper = _weighted_quantile(projection, weights, 0.75)
    global_mean, global_var = _weighted_diag_stats(x, weights)
    means = np.vstack(
        [
            global_mean + (lower - float(np.mean(projection))) * np.eye(1, global_mean.size, 0).reshape(-1),
            global_mean + (upper - float(np.mean(projection))) * np.eye(1, global_mean.size, 0).reshape(-1),
        ]
    )[: int(component_count)]
    if means.shape[0] < int(component_count):
        means = np.vstack([means, np.tile(global_mean.reshape(1, -1), (int(component_count) - means.shape[0], 1))])
    vars_ = np.tile(global_var.reshape(1, -1), (int(component_count), 1))
    mixture = np.full((int(component_count),), 1.0 / float(component_count), dtype=np.float64)

    for _ in range(max(int(iterations), 1)):
        log_probs = []
        for component_index in range(int(component_count)):
            log_probs.append(
                np.log(np.maximum(mixture[component_index], 1e-8))
                + _diag_logpdf_matrix(x, means[component_index], vars_[component_index])
            )
        stacked = np.stack(log_probs, axis=1)
        log_norm = _logsumexp(stacked, axis=1).reshape(-1, 1)
        responsibilities = np.exp(stacked - log_norm)
        effective_resp = responsibilities * weights.reshape(-1, 1)
        component_mass = np.sum(effective_resp, axis=0)
        safe_mass = np.maximum(component_mass, 1e-8)
        mixture = np.asarray(safe_mass / np.sum(safe_mass), dtype=np.float64)
        for component_index in range(int(component_count)):
            component_weights = effective_resp[:, component_index]
            mean, var = _weighted_diag_stats(x, component_weights)
            means[component_index] = mean
            vars_[component_index] = np.maximum(var, 1e-6)

    final_log_probs = []
    for component_index in range(int(component_count)):
        final_log_probs.append(
            np.log(np.maximum(mixture[component_index], 1e-8))
            + _diag_logpdf_matrix(x, means[component_index], vars_[component_index])
        )
    final_stacked = np.stack(final_log_probs, axis=1)
    final_log_likelihood = float(np.sum(weights * _logsumexp(final_stacked, axis=1)))
    parameter_count = int(component_count) * (2 * int(x.shape[1])) + (int(component_count) - 1)
    bic = float(-2.0 * final_log_likelihood + parameter_count * np.log(max(sample_count, 1)))
    return {
        "components": [
            {
                "weight": float(mixture[component_index]),
                "mean": [float(item) for item in means[component_index].tolist()],
                "var": [float(item) for item in vars_[component_index].tolist()],
                "component_weight_mass": float(np.sum(weights * np.exp(final_stacked[:, component_index] - _logsumexp(final_stacked, axis=1)))),
            }
            for component_index in range(int(component_count))
        ],
        "component_count": int(component_count),
        "log_likelihood": float(final_log_likelihood),
        "bic": float(bic),
    }


def _fit_best_diag_mixture(
    x: np.ndarray,
    weights: np.ndarray,
    *,
    max_components: int,
) -> dict[str, Any]:
    sample_count = int(x.shape[0])
    if sample_count <= 0:
        return _fit_diag_gmm(x, weights, component_count=1)
    candidates = [_fit_diag_gmm(x, weights, component_count=1)]
    if int(max_components) >= 2 and sample_count >= 8:
        candidates.append(_fit_diag_gmm(x, weights, component_count=2))
    return min(candidates, key=lambda item: float(item.get("bic", float("inf"))))


def _adjust_mixture_components(
    components: Sequence[Mapping[str, Any]],
    *,
    global_mean: np.ndarray,
    global_var: np.ndarray,
    freq_mean: Optional[np.ndarray],
    freq_var: Optional[np.ndarray],
    shrinkage: float,
) -> list[dict[str, Any]]:
    if not components:
        return []
    if freq_mean is None or freq_var is None or float(shrinkage) <= 1e-12:
        return [dict(component) for component in components]
    safe_shrinkage = float(np.clip(float(shrinkage), 0.0, 1.0))
    mean_shift = np.asarray(freq_mean, dtype=np.float64) - np.asarray(global_mean, dtype=np.float64)
    adjusted: list[dict[str, Any]] = []
    for component in components:
        component_mean = np.asarray(component.get("mean", []), dtype=np.float64)
        component_var = np.asarray(component.get("var", []), dtype=np.float64)
        if component_mean.size != mean_shift.size or component_var.size != np.asarray(global_var, dtype=np.float64).size:
            adjusted.append(dict(component))
            continue
        adjusted.append(
            {
                **dict(component),
                "mean": [
                    float(item)
                    for item in (
                        component_mean + safe_shrinkage * mean_shift
                    ).tolist()
                ],
                "var": [
                    float(item)
                    for item in np.maximum(
                        (1.0 - safe_shrinkage) * component_var + safe_shrinkage * np.asarray(freq_var, dtype=np.float64),
                        1e-6,
                    ).tolist()
                ],
            }
        )
    return adjusted


def _mixture_logpdf(x: np.ndarray, components: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if x.size <= 0:
        return np.asarray([], dtype=np.float64)
    if not components:
        return np.full((x.shape[0],), -np.inf, dtype=np.float64)
    stacked = []
    for component in components:
        weight = max(_safe_float(component.get("weight", 0.0), 0.0), 1e-8)
        mean = np.asarray(component.get("mean", []), dtype=np.float64)
        var = np.asarray(component.get("var", []), dtype=np.float64)
        stacked.append(np.log(weight) + _diag_logpdf_matrix(x, mean, var))
    return np.asarray(_logsumexp(np.stack(stacked, axis=1), axis=1), dtype=np.float64)


@dataclass(frozen=True)
class CorrectnessCalibratorConfig:
    variant: str = GLOBAL_CORRECTNESS_LOGISTIC
    learning_rate: float = 0.08
    epochs: int = 320
    l2: float = 1e-3
    min_positive_windows: int = 24
    min_negative_windows: int = 24
    sample_weight_mode: str = "per_trial_equal"
    max_gmm_components: int = 2


@dataclass(frozen=True)
class CorrectnessCalibratorModel:
    variant: str
    feature_names: tuple[str, ...]
    freq_order: tuple[float, ...]
    payload: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "variant": str(self.variant),
            "feature_names": [str(item) for item in self.feature_names],
            "freq_order": [float(item) for item in self.freq_order],
            "payload": dict(self.payload),
        }


class CorrectnessCalibrator:
    def __init__(
        self,
        *,
        feature_names: Optional[Sequence[str]] = None,
        model: Optional[CorrectnessCalibratorModel] = None,
    ) -> None:
        self.feature_names = None if feature_names is None else tuple(str(item) for item in feature_names)
        self.model = model

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CorrectnessCalibrator":
        model_payload = dict(payload.get("model", {}) or {})
        model = None
        if model_payload:
            if "payload" in model_payload:
                model = CorrectnessCalibratorModel(
                    variant=str(model_payload.get("variant", GLOBAL_CORRECTNESS_LOGISTIC)),
                    feature_names=tuple(model_payload.get("feature_names", DEFAULT_CORRECTNESS_FEATURE_NAMES)),
                    freq_order=tuple(float(item) for item in model_payload.get("freq_order", [])),
                    payload=dict(model_payload.get("payload", {}) or {}),
                )
            else:
                # Backward-compatible restore for older logistic payloads.
                model = CorrectnessCalibratorModel(
                    variant=GLOBAL_CORRECTNESS_LOGISTIC,
                    feature_names=tuple(payload.get("feature_names", DEFAULT_CORRECTNESS_FEATURE_NAMES)),
                    freq_order=tuple(float(item) for item in model_payload.get("freq_order", [])),
                    payload={
                        "coef": [float(item) for item in model_payload.get("coef", [])],
                        "intercept": float(model_payload.get("intercept", 0.0)),
                        "mean": [float(item) for item in model_payload.get("mean", [])],
                        "std": [float(item) for item in model_payload.get("std", [])],
                    },
                )
        return cls(
            feature_names=tuple(payload.get("feature_names", DEFAULT_CORRECTNESS_FEATURE_NAMES)),
            model=model,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "feature_names": [
                str(item)
                for item in (
                    self.feature_names
                    if self.feature_names is not None
                    else (
                        self.model.feature_names
                        if self.model is not None
                        else DEFAULT_CORRECTNESS_FEATURE_NAMES
                    )
                )
            ],
            "model": None if self.model is None else self.model.to_payload(),
        }

    @staticmethod
    def _normalize_variant(value: Optional[str]) -> str:
        variant = str(value or GLOBAL_CORRECTNESS_LOGISTIC).strip().lower()
        if variant not in CORRECTNESS_VARIANTS:
            return GLOBAL_CORRECTNESS_LOGISTIC
        return variant

    def _resolved_feature_names(self, *, variant: str) -> tuple[str, ...]:
        if self.feature_names:
            return tuple(str(item) for item in self.feature_names)
        if str(variant) in {BAYESIAN_GAP_GMM, FREQ_SHRUNK_GAP_GMM}:
            return tuple(DEFAULT_BAYESIAN_GAP_FEATURE_NAMES)
        return tuple(DEFAULT_CORRECTNESS_FEATURE_NAMES)

    def _build_vector(
        self,
        row: Mapping[str, Any],
        *,
        feature_names: Sequence[str],
        freq_order: Sequence[float],
        include_one_hot: bool,
    ) -> np.ndarray:
        base = [_safe_float(row.get(name, 0.0), 0.0) for name in feature_names]
        if not include_one_hot:
            return np.asarray(base, dtype=np.float64)
        pred_freq = row.get("pred_freq")
        pred_value = None if pred_freq is None else _safe_float(pred_freq, float("nan"))
        if pred_value is None or not np.isfinite(pred_value):
            one_hot = [0.0] * len(freq_order)
        else:
            one_hot = [1.0 if abs(float(pred_value) - float(freq)) <= 1e-8 else 0.0 for freq in freq_order]
        return np.asarray([*base, *one_hot], dtype=np.float64)

    @staticmethod
    def _label_for_row(row: Mapping[str, Any]) -> float:
        pred_freq = row.get("pred_freq")
        expected_freq = row.get("expected_freq")
        if pred_freq is None or expected_freq is None:
            return 0.0
        pred_value = _safe_float(pred_freq, float("nan"))
        expected_value = _safe_float(expected_freq, float("nan"))
        if not np.isfinite(pred_value) or not np.isfinite(expected_value):
            return 0.0
        return 1.0 if resolve_trial_role(row) == "control" and abs(pred_value - expected_value) <= 1e-8 else 0.0

    def fit(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        freqs: Sequence[float],
        config: Optional[CorrectnessCalibratorConfig] = None,
    ) -> dict[str, Any]:
        cfg = config or CorrectnessCalibratorConfig()
        variant = self._normalize_variant(cfg.variant)
        feature_names = self._resolved_feature_names(variant=variant)
        freq_order = tuple(float(item) for item in freqs)
        filtered: list[dict[str, Any]] = []
        for row in rows:
            pred_freq = row.get("pred_freq")
            pred_value = None if pred_freq is None else _safe_float(pred_freq, float("nan"))
            if pred_value is None or not np.isfinite(pred_value):
                continue
            filtered.append(dict(row))
        y = np.asarray([self._label_for_row(row) for row in filtered], dtype=np.float64)
        base_sample_weights = _build_sample_weights(filtered, mode=str(cfg.sample_weight_mode))
        positive_windows = int(np.sum(y > 0.5))
        negative_windows = int(np.sum(y <= 0.5))
        positive_trials = len(
            {
                _trial_key(row, fallback_index=index)
                for index, row in enumerate(filtered)
                if float(y[index]) > 0.5
            }
        )
        negative_trials = len(
            {
                _trial_key(row, fallback_index=index)
                for index, row in enumerate(filtered)
                if float(y[index]) <= 0.5
            }
        )
        valid = bool(
            positive_windows >= int(cfg.min_positive_windows)
            and negative_windows >= int(cfg.min_negative_windows)
        )

        if variant in {BAYESIAN_GAP_GMM, FREQ_SHRUNK_GAP_GMM}:
            x = np.asarray(
                [
                    self._build_vector(
                        row,
                        feature_names=feature_names,
                        freq_order=freq_order,
                        include_one_hot=False,
                    )
                    for row in filtered
                ],
                dtype=np.float64,
            )
            x_norm, mean, std = _normalize_rows(x, base_sample_weights) if x.size else (
                np.zeros((0, len(feature_names)), dtype=np.float64),
                np.zeros((len(feature_names),), dtype=np.float64),
                np.ones((len(feature_names),), dtype=np.float64),
            )
            positive_mask = y > 0.5
            negative_mask = np.logical_not(positive_mask)
            pos_x = np.asarray(x_norm[positive_mask], dtype=np.float64)
            neg_x = np.asarray(x_norm[negative_mask], dtype=np.float64)
            pos_w = np.asarray(base_sample_weights[positive_mask], dtype=np.float64)
            neg_w = np.asarray(base_sample_weights[negative_mask], dtype=np.float64)
            weighted_positive_mass = float(np.sum(pos_w))
            weighted_negative_mass = float(np.sum(neg_w))
            total_mass = float(weighted_positive_mass + weighted_negative_mass)
            prior_positive = float(weighted_positive_mass / total_mass) if total_mass > 1e-12 else 0.5
            prior_negative = float(1.0 - prior_positive)
            positive_fit = _fit_best_diag_mixture(pos_x, pos_w, max_components=int(cfg.max_gmm_components))
            negative_fit = _fit_best_diag_mixture(neg_x, neg_w, max_components=int(cfg.max_gmm_components))
            payload = {
                "mean": [float(item) for item in mean.tolist()],
                "std": [float(item) for item in std.tolist()],
                "prior_positive": float(_clip_probability(prior_positive)),
                "prior_negative": float(_clip_probability(prior_negative)),
                "positive_components": [dict(item) for item in positive_fit.get("components", [])],
                "negative_components": [dict(item) for item in negative_fit.get("components", [])],
                "positive_component_count": int(positive_fit.get("component_count", 1)),
                "negative_component_count": int(negative_fit.get("component_count", 1)),
            }
            if variant == FREQ_SHRUNK_GAP_GMM:
                pred_indices = _pred_freq_indices(filtered, freq_order)
                global_pos_mean, global_pos_var = _weighted_diag_stats(pos_x, pos_w)
                global_neg_mean, global_neg_var = _weighted_diag_stats(neg_x, neg_w)
                per_freq_stats: dict[str, dict[str, Any]] = {}
                for freq_index, freq in enumerate(freq_order):
                    freq_key = _freq_key(freq)
                    pos_mask = np.logical_and(positive_mask, pred_indices == int(freq_index))
                    neg_mask = np.logical_and(negative_mask, pred_indices == int(freq_index))
                    freq_pos_x = np.asarray(x_norm[pos_mask], dtype=np.float64)
                    freq_neg_x = np.asarray(x_norm[neg_mask], dtype=np.float64)
                    freq_pos_w = np.asarray(base_sample_weights[pos_mask], dtype=np.float64)
                    freq_neg_w = np.asarray(base_sample_weights[neg_mask], dtype=np.float64)
                    pos_mass = float(np.sum(freq_pos_w))
                    neg_mass = float(np.sum(freq_neg_w))
                    pos_mean, pos_var = _weighted_diag_stats(freq_pos_x, freq_pos_w)
                    neg_mean, neg_var = _weighted_diag_stats(freq_neg_x, freq_neg_w)
                    per_freq_stats[str(freq_key)] = {
                        "positive_mean": [float(item) for item in pos_mean.tolist()],
                        "positive_var": [float(item) for item in pos_var.tolist()],
                        "negative_mean": [float(item) for item in neg_mean.tolist()],
                        "negative_var": [float(item) for item in neg_var.tolist()],
                        "positive_mass": float(pos_mass),
                        "negative_mass": float(neg_mass),
                        "positive_shrinkage": float(_shrinkage_from_mass(pos_mass)),
                        "negative_shrinkage": float(_shrinkage_from_mass(neg_mass)),
                    }
                payload.update(
                    {
                        "global_positive_mean": [float(item) for item in global_pos_mean.tolist()],
                        "global_positive_var": [float(item) for item in global_pos_var.tolist()],
                        "global_negative_mean": [float(item) for item in global_neg_mean.tolist()],
                        "global_negative_var": [float(item) for item in global_neg_var.tolist()],
                        "per_freq_stats": per_freq_stats,
                    }
                )
            self.model = CorrectnessCalibratorModel(
                variant=str(variant),
                feature_names=tuple(feature_names),
                freq_order=freq_order,
                payload=payload,
            )
        else:
            x = np.asarray(
                [
                    self._build_vector(
                        row,
                        feature_names=feature_names,
                        freq_order=freq_order,
                        include_one_hot=(variant == GLOBAL_CORRECTNESS_LOGISTIC),
                    )
                    for row in filtered
                ],
                dtype=np.float64,
            )
            if x.size <= 0:
                dim = len(feature_names) + (len(freq_order) if variant == GLOBAL_CORRECTNESS_LOGISTIC else 0)
                mean = np.zeros((dim,), dtype=np.float64)
                std = np.ones_like(mean)
                w = np.zeros_like(mean)
                intercept = 0.0
                payload = {
                    "coef": [float(item) for item in w.tolist()],
                    "intercept": float(intercept),
                    "mean": [float(item) for item in mean.tolist()],
                    "std": [float(item) for item in std.tolist()],
                }
            else:
                x_norm, mean, std = _normalize_rows(x, base_sample_weights)
                w = np.zeros((x_norm.shape[1],), dtype=np.float64)
                weighted_positive_mass = float(np.sum(base_sample_weights[y > 0.5]))
                weighted_negative_mass = float(np.sum(base_sample_weights[y <= 0.5]))
                total_base_weight = float(np.sum(base_sample_weights))
                intercept = float(
                    logit(
                        _clip_probability(
                            (weighted_positive_mass / total_base_weight)
                            if total_base_weight > 1e-12
                            else (np.mean(y) if y.size else 0.5)
                        )
                    )
                )
                effective_weights = _effective_class_weights(y, base_sample_weights)
                if valid and weighted_positive_mass > 1e-12 and weighted_negative_mass > 1e-12:
                    total_effective_weight = float(np.sum(effective_weights))
                    for _ in range(max(int(cfg.epochs), 1)):
                        logits = np.dot(x_norm, w) + intercept
                        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
                        error = (probs - y) * effective_weights
                        denom = max(total_effective_weight, 1e-6)
                        grad_w = np.dot(x_norm.T, error) / denom + float(cfg.l2) * w
                        grad_b = float(np.sum(error) / denom)
                        w -= float(cfg.learning_rate) * grad_w
                        intercept -= float(cfg.learning_rate) * grad_b
                payload = {
                    "coef": [float(item) for item in w.tolist()],
                    "intercept": float(intercept),
                    "mean": [float(item) for item in mean.tolist()],
                    "std": [float(item) for item in std.tolist()],
                }
                if variant == FREQ_SHRUNK_LOGISTIC:
                    base_logits = np.dot(x_norm, w) + intercept
                    pred_indices = _pred_freq_indices(filtered, freq_order)
                    slope = 1.0
                    recal_intercept = 0.0
                    freq_bias = np.zeros((len(freq_order),), dtype=np.float64)
                    if valid and np.any(pred_indices >= 0):
                        total_effective_weight = float(np.sum(effective_weights))
                        for _ in range(max(int(cfg.epochs), 1)):
                            extra_bias = np.where(pred_indices >= 0, freq_bias[np.maximum(pred_indices, 0)], 0.0)
                            recal_logits = slope * base_logits + recal_intercept + extra_bias
                            probs = 1.0 / (1.0 + np.exp(-np.clip(recal_logits, -50.0, 50.0)))
                            error = (probs - y) * effective_weights
                            denom = max(total_effective_weight, 1e-6)
                            grad_slope = float(np.sum(error * base_logits) / denom) + 1e-2 * float(slope - 1.0)
                            grad_intercept = float(np.sum(error) / denom)
                            grad_bias = np.zeros_like(freq_bias)
                            for freq_index in range(len(freq_order)):
                                mask = pred_indices == int(freq_index)
                                if np.any(mask):
                                    grad_bias[freq_index] = float(np.sum(error[mask]) / denom) + 1e-2 * float(freq_bias[freq_index])
                            slope -= float(cfg.learning_rate) * grad_slope
                            recal_intercept -= float(cfg.learning_rate) * grad_intercept
                            freq_bias -= float(cfg.learning_rate) * grad_bias
                    bias_payload: dict[str, float] = {}
                    shrink_payload: dict[str, float] = {}
                    for freq_index, freq in enumerate(freq_order):
                        freq_key = _freq_key(freq)
                        freq_mass = float(np.sum(base_sample_weights[pred_indices == int(freq_index)]))
                        shrinkage = _shrinkage_from_mass(freq_mass)
                        bias_payload[str(freq_key)] = float(freq_bias[freq_index] * shrinkage)
                        shrink_payload[str(freq_key)] = float(shrinkage)
                    payload.update(
                        {
                            "base_coef": [float(item) for item in w.tolist()],
                            "base_intercept": float(intercept),
                            "base_mean": [float(item) for item in mean.tolist()],
                            "base_std": [float(item) for item in std.tolist()],
                            "recalibration_slope": float(slope),
                            "recalibration_intercept": float(recal_intercept),
                            "recalibration_freq_bias": bias_payload,
                            "recalibration_freq_shrinkage": shrink_payload,
                        }
                    )
            self.model = CorrectnessCalibratorModel(
                variant=str(variant),
                feature_names=tuple(feature_names),
                freq_order=freq_order,
                payload=payload,
            )

        self.feature_names = tuple(feature_names)
        if x.size <= 0:
            predicted = np.asarray([], dtype=np.float64)
            brier_score = None
            auc_roc = None
        else:
            predicted = np.asarray(
                [self.predict(row)["p_correct"] for row in filtered],
                dtype=np.float64,
            )
            brier_score = (
                float(np.average((predicted - y) ** 2, weights=base_sample_weights))
                if predicted.size
                else None
            )
            auc_roc = _binary_auc(y, predicted)
        return {
            "valid": bool(valid),
            "variant": str(variant),
            "sample_count": int(len(filtered)),
            "positive_windows": int(positive_windows),
            "negative_windows": int(negative_windows),
            "positive_trials": int(positive_trials),
            "negative_trials": int(negative_trials),
            "sample_weight_mode": str(cfg.sample_weight_mode),
            "brier_score": None if brier_score is None else float(brier_score),
            "auc_roc": None if auc_roc is None else float(auc_roc),
            "ece": None
            if predicted.size <= 0
            else _ece_score(y, predicted, base_sample_weights),
            "feature_names": [str(item) for item in feature_names],
            "freq_order": [float(item) for item in freq_order],
        }

    def predict(self, row: Mapping[str, Any]) -> dict[str, float]:
        if self.model is None:
            return {"p_correct": 0.5, "correctness_logit": 0.0}
        pred_freq = row.get("pred_freq")
        pred_value = None if pred_freq is None else _safe_float(pred_freq, float("nan"))
        if pred_value is None or not np.isfinite(pred_value):
            return {"p_correct": 1e-6, "correctness_logit": float(logit(1e-6))}

        variant = self._normalize_variant(self.model.variant)
        payload = dict(self.model.payload)
        if variant in {BAYESIAN_GAP_GMM, FREQ_SHRUNK_GAP_GMM}:
            vector = self._build_vector(
                row,
                feature_names=self.model.feature_names,
                freq_order=self.model.freq_order,
                include_one_hot=False,
            )
            mean = np.asarray(payload.get("mean", [0.0] * vector.size), dtype=np.float64)
            std = np.asarray(payload.get("std", [1.0] * vector.size), dtype=np.float64)
            normalized = ((vector - mean) / np.maximum(std, 1e-6)).reshape(1, -1)
            positive_components = list(payload.get("positive_components", []) or [])
            negative_components = list(payload.get("negative_components", []) or [])
            if variant == FREQ_SHRUNK_GAP_GMM:
                freq_key = _freq_key(pred_value)
                per_freq_stats = dict(payload.get("per_freq_stats", {}) or {})
                freq_stats = dict(per_freq_stats.get(freq_key, {}) or {})
                positive_components = _adjust_mixture_components(
                    positive_components,
                    global_mean=np.asarray(
                        payload.get("global_positive_mean", [0.0] * vector.size),
                        dtype=np.float64,
                    ),
                    global_var=np.asarray(
                        payload.get("global_positive_var", [1.0] * vector.size),
                        dtype=np.float64,
                    ),
                    freq_mean=np.asarray(
                        freq_stats.get("positive_mean", [0.0] * vector.size),
                        dtype=np.float64,
                    )
                    if freq_stats
                    else None,
                    freq_var=np.asarray(
                        freq_stats.get("positive_var", [1.0] * vector.size),
                        dtype=np.float64,
                    )
                    if freq_stats
                    else None,
                    shrinkage=_safe_float(freq_stats.get("positive_shrinkage", 0.0), 0.0),
                )
                negative_components = _adjust_mixture_components(
                    negative_components,
                    global_mean=np.asarray(
                        payload.get("global_negative_mean", [0.0] * vector.size),
                        dtype=np.float64,
                    ),
                    global_var=np.asarray(
                        payload.get("global_negative_var", [1.0] * vector.size),
                        dtype=np.float64,
                    ),
                    freq_mean=np.asarray(
                        freq_stats.get("negative_mean", [0.0] * vector.size),
                        dtype=np.float64,
                    )
                    if freq_stats
                    else None,
                    freq_var=np.asarray(
                        freq_stats.get("negative_var", [1.0] * vector.size),
                        dtype=np.float64,
                    )
                    if freq_stats
                    else None,
                    shrinkage=_safe_float(freq_stats.get("negative_shrinkage", 0.0), 0.0),
                )
            prior_positive = _clip_probability(payload.get("prior_positive", 0.5))
            prior_negative = _clip_probability(payload.get("prior_negative", 0.5))
            log_positive = float(np.log(prior_positive) + _mixture_logpdf(normalized, positive_components)[0])
            log_negative = float(np.log(prior_negative) + _mixture_logpdf(normalized, negative_components)[0])
            stacked = np.asarray([log_negative, log_positive], dtype=np.float64)
            normalizer = float(_logsumexp(stacked))
            p = float(np.exp(log_positive - normalizer))
            p = _clip_probability(p)
            return {
                "p_correct": float(p),
                "correctness_logit": float(logit(p)),
            }

        include_one_hot = bool(variant == GLOBAL_CORRECTNESS_LOGISTIC)
        vector = self._build_vector(
            row,
            feature_names=self.model.feature_names,
            freq_order=self.model.freq_order,
            include_one_hot=include_one_hot,
        )
        coef = np.asarray(
            payload.get(
                "base_coef" if variant == FREQ_SHRUNK_LOGISTIC else "coef",
                payload.get("coef", []),
            ),
            dtype=np.float64,
        )
        mean = np.asarray(
            payload.get(
                "base_mean" if variant == FREQ_SHRUNK_LOGISTIC else "mean",
                payload.get("mean", [0.0] * vector.size),
            ),
            dtype=np.float64,
        )
        std = np.asarray(
            payload.get(
                "base_std" if variant == FREQ_SHRUNK_LOGISTIC else "std",
                payload.get("std", [1.0] * vector.size),
            ),
            dtype=np.float64,
        )
        if vector.size != coef.size:
            return {"p_correct": 0.5, "correctness_logit": 0.0}
        normalized = (vector - mean) / np.maximum(std, 1e-6)
        base_intercept = float(
            payload.get(
                "base_intercept" if variant == FREQ_SHRUNK_LOGISTIC else "intercept",
                payload.get("intercept", 0.0),
            )
        )
        z = float(np.dot(coef, normalized) + base_intercept)
        if variant == FREQ_SHRUNK_LOGISTIC:
            freq_key = _freq_key(pred_value)
            freq_bias_map = dict(payload.get("recalibration_freq_bias", {}) or {})
            z = float(
                float(payload.get("recalibration_slope", 1.0)) * z
                + float(payload.get("recalibration_intercept", 0.0))
                + _safe_float(freq_bias_map.get(freq_key, 0.0), 0.0)
            )
        p = float(sigmoid(z))
        return {
            "p_correct": float(_clip_probability(p)),
            "correctness_logit": float(z),
        }
