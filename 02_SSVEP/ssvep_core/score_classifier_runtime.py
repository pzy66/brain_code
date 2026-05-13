from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np


CLASSIFIER_DERIVED_FEATURE_NAMES = (
    "top1_score",
    "top2_score",
    "margin",
    "ratio",
    "normalized_top1",
    "score_entropy",
)
FULL_REFERENCE_BANK_FEATURE_NAMES = (
    "top_command_score",
    "top_all_score",
    "command_rank_in_all",
    "top_command_to_top_all_ratio",
    "nearest_noncommand_margin",
    "all_bank_entropy",
)
CLASSIFIER_CONFIDENCE_GATE_POLICY = "confidence_threshold"
CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY = "lrt_multiwindow_reject_gate"
DEFAULT_LRT_MULTIWINDOW_DECAY = 0.65
CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW = "baseline_lrtmw"
CLASSIFIER_GATE_VARIANT_LRTMW_MARGIN = "lrtmw_margin_gate"
CLASSIFIER_GATE_VARIANT_LRTMW_ENTROPY = "lrtmw_entropy_gate"
CLASSIFIER_GATE_VARIANT_SUBJECT_THRESHOLD_FLOOR = "subject_threshold_floor"
CLASSIFIER_GATE_VARIANT_NS2_AWARE = "ns2_aware_gate"
CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE = "subject_floor_ns2_aware_gate"
CLASSIFIER_GATE_VARIANT_WEAK_SUBJECT_GUARD = "weak_subject_guard"
CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD = "frequency_specific_threshold_gate"
CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC = "frequency_specific_logistic_gate"
CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC = "conditional_frequency_specific_logistic_gate"
CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO = "tenp5_ns2_hard_negative_veto"
CLASSIFIER_GATE_VARIANTS = (
    CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
    CLASSIFIER_GATE_VARIANT_LRTMW_MARGIN,
    CLASSIFIER_GATE_VARIANT_LRTMW_ENTROPY,
    CLASSIFIER_GATE_VARIANT_SUBJECT_THRESHOLD_FLOOR,
    CLASSIFIER_GATE_VARIANT_NS2_AWARE,
    CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE,
    CLASSIFIER_GATE_VARIANT_WEAK_SUBJECT_GUARD,
    CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
    CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
    CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC,
    CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO,
)
CLASSIFIER_RUNTIME_SAFE_GATE_VARIANTS = (
    CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
)
CLASSIFIER_RESEARCH_ONLY_GATE_VARIANTS = tuple(
    variant for variant in CLASSIFIER_GATE_VARIANTS if variant not in CLASSIFIER_RUNTIME_SAFE_GATE_VARIANTS
)


def parse_classifier_gate_variant(raw: Any) -> str:
    value = str(raw or CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW).strip().lower()
    if value in {"", "baseline", "lrtmw", "lrt_multiwindow_reject_gate"}:
        return CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW
    aliases = {
        "frequency-specific-threshold": CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
        "frequency_specific_threshold": CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
        "freq_specific_threshold": CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
        "freqspec_threshold": CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
        "frequency-specific-logistic": CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
        "frequency_specific_logistic": CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
        "freq_specific_logistic": CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
        "freqspec_logistic": CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
        "conditional-frequency-specific-logistic": CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC,
        "conditional_frequency_specific_logistic": CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC,
        "conditional_freq_specific_logistic": CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC,
        "conditional_freqspec_logistic": CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC,
        "freqspec_conditional_logistic": CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC,
        "tenp5-ns2-hard-negative-veto": CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO,
        "tenp5_ns2_veto": CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO,
        "10p5_ns2_hard_negative_veto": CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO,
        "10.5_ns2_hard_negative_veto": CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO,
        "10p5-ns2-hard-negative-veto": CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO,
        "10.5-ns2-hard-negative-veto": CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO,
        "subject-floor-ns2-aware": CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE,
        "subject_floor_ns2_aware": CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE,
        "subject_threshold_floor_ns2_aware": CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE,
        "floor_ns2_aware": CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE,
        "floor_ns2": CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE,
    }
    value = aliases.get(value, value)
    if value not in CLASSIFIER_GATE_VARIANTS:
        raise ValueError(f"unsupported classifier gate variant: {raw}")
    return value


def is_classifier_gate_variant_runtime_safe(raw: Any) -> bool:
    return parse_classifier_gate_variant(raw) in CLASSIFIER_RUNTIME_SAFE_GATE_VARIANTS


def require_runtime_safe_classifier_gate_variant(raw: Any) -> str:
    variant = parse_classifier_gate_variant(raw)
    if variant not in CLASSIFIER_RUNTIME_SAFE_GATE_VARIANTS:
        raise ValueError(
            "research-only classifier gate variant is not allowed for realtime startup: "
            f"{variant}. Re-run or export a session no-control/baseline_lrtmw profile for online use."
        )
    return variant


def freq_label(freq: float) -> str:
    return f"{float(freq):g}"


def classifier_labels(freqs: Sequence[float]) -> tuple[str, ...]:
    return ("idle", *(freq_label(float(freq)) for freq in freqs))


def classifier_feature_names(
    freqs: Sequence[float],
    *,
    score_source_name: str = "fbcca",
    score_bank_mode: str = "command_only",
) -> list[str]:
    names = [f"{str(score_source_name).strip().lower()}_score_{freq_label(freq)}" for freq in freqs]
    names.extend(CLASSIFIER_DERIVED_FEATURE_NAMES)
    if parse_score_bank_mode(score_bank_mode) == "full_reference_bank":
        names.extend(FULL_REFERENCE_BANK_FEATURE_NAMES)
    return names


def parse_score_bank_mode(raw: str) -> str:
    value = str(raw or "command_only").strip().lower()
    if value not in {"command_only", "full_reference_bank"}:
        raise ValueError(f"unsupported score_bank_mode: {raw}")
    return value


def normalize_frequency_specific_control_state_gates(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        try:
            freq_key = freq_label(float(key))
        except Exception:
            freq_key = str(key).strip()
        if not freq_key or not isinstance(value, Mapping):
            continue
        normalized[freq_key] = dict(value)
    return normalized


def score_matrix_to_features(score_matrix: np.ndarray) -> np.ndarray:
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("score_matrix must have shape (windows, at least 2 freqs)")
    order = np.argsort(scores, axis=1)[:, ::-1]
    top1 = np.take_along_axis(scores, order[:, 0:1], axis=1)[:, 0]
    top2 = np.take_along_axis(scores, order[:, 1:2], axis=1)[:, 0]
    score_sum = np.sum(scores, axis=1)
    safe_sum = np.maximum(score_sum, 1e-12)
    probs = np.clip(scores / safe_sum[:, None], 1e-12, None)
    probs = probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-12)
    entropy = -np.sum(probs * np.log(probs), axis=1) / np.log(float(scores.shape[1]))
    return np.column_stack(
        [
            scores,
            top1,
            top2,
            top1 - top2,
            top1 / np.maximum(top2, 1e-12),
            top1 / safe_sum,
            entropy,
        ]
    ).astype(np.float64, copy=False)


def full_reference_bank_features(
    *,
    command_score_matrix: np.ndarray,
    all_score_matrix: np.ndarray,
    command_freqs: Sequence[float],
    all_freqs: Sequence[float],
) -> np.ndarray:
    command_scores = np.asarray(command_score_matrix, dtype=np.float64)
    all_scores = np.asarray(all_score_matrix, dtype=np.float64)
    if command_scores.ndim != 2 or all_scores.ndim != 2:
        raise ValueError("command and full-bank score matrices must be 2D")
    if command_scores.shape[0] != all_scores.shape[0]:
        raise ValueError("command and full-bank score matrices must have the same window count")
    command_freq_tuple = tuple(round(float(freq), 10) for freq in command_freqs)
    all_freq_tuple = tuple(round(float(freq), 10) for freq in all_freqs)
    if len(command_freq_tuple) != command_scores.shape[1]:
        raise ValueError("command frequency count does not match command score matrix")
    if len(all_freq_tuple) != all_scores.shape[1]:
        raise ValueError("full-bank frequency count does not match full-bank score matrix")
    all_index_by_freq = {freq: index for index, freq in enumerate(all_freq_tuple)}
    command_indices: list[int] = []
    for freq in command_freq_tuple:
        if freq not in all_index_by_freq:
            raise ValueError(f"command frequency {freq:g} is missing from full reference bank")
        command_indices.append(int(all_index_by_freq[freq]))

    command_from_all = all_scores[:, np.asarray(command_indices, dtype=int)]
    top_command = np.max(command_from_all, axis=1)
    top_all = np.max(all_scores, axis=1)
    sorted_all = np.argsort(all_scores, axis=1)[:, ::-1]
    command_index_set = set(command_indices)
    command_rank_rows: list[int] = []
    top_noncommand_rows: list[float] = []
    for row_index, order in enumerate(sorted_all):
        rank = 1
        best_command_rank: Optional[int] = None
        top_noncommand: Optional[float] = None
        for column in order:
            col = int(column)
            if col in command_index_set:
                if best_command_rank is None:
                    best_command_rank = int(rank)
            elif top_noncommand is None:
                top_noncommand = float(all_scores[row_index, col])
            if best_command_rank is not None and top_noncommand is not None:
                break
            rank += 1
        command_rank_rows.append(int(best_command_rank or len(all_freq_tuple)))
        top_noncommand_rows.append(float(top_noncommand if top_noncommand is not None else 0.0))
    score_sum = np.sum(np.clip(all_scores, 0.0, None), axis=1)
    safe_sum = np.maximum(score_sum, 1e-12)
    probs = np.clip(np.clip(all_scores, 0.0, None) / safe_sum[:, None], 1e-12, None)
    probs = probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-12)
    entropy = -np.sum(probs * np.log(probs), axis=1) / np.log(float(max(all_scores.shape[1], 2)))
    return np.column_stack(
        [
            top_command,
            top_all,
            np.asarray(command_rank_rows, dtype=np.float64),
            top_command / np.maximum(top_all, 1e-12),
            top_command - np.asarray(top_noncommand_rows, dtype=np.float64),
            entropy,
        ]
    ).astype(np.float64, copy=False)


def score_matrices_to_features(
    *,
    command_score_matrix: np.ndarray,
    command_freqs: Sequence[float],
    score_bank_mode: str = "command_only",
    all_score_matrix: Optional[np.ndarray] = None,
    all_freqs: Sequence[float] = (),
) -> np.ndarray:
    command_features = score_matrix_to_features(command_score_matrix)
    mode = parse_score_bank_mode(score_bank_mode)
    if mode == "command_only":
        return command_features
    if all_score_matrix is None:
        raise ValueError("full_reference_bank mode requires all_score_matrix")
    full_features = full_reference_bank_features(
        command_score_matrix=command_score_matrix,
        all_score_matrix=all_score_matrix,
        command_freqs=command_freqs,
        all_freqs=all_freqs,
    )
    return np.column_stack([command_features, full_features]).astype(np.float64, copy=False)


def smooth_classifier_probabilities(probs: np.ndarray, smoothing_windows: int = 1) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    width = max(1, int(smoothing_windows))
    if values.ndim != 2 or width <= 1 or values.shape[0] <= 1:
        return values
    smoothed = np.empty_like(values, dtype=np.float64)
    cumulative = np.cumsum(values, axis=0)
    for index in range(int(values.shape[0])):
        start = max(0, int(index) - width + 1)
        total = cumulative[index] - (cumulative[start - 1] if start > 0 else 0.0)
        smoothed[index] = total / float(index - start + 1)
    row_sum = np.sum(smoothed, axis=1, keepdims=True)
    return smoothed / np.maximum(row_sum, 1e-12)


def normalize_ridge5_state(state: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(state or {})
    labels = tuple(str(label) for label in payload.get("labels", ()))
    if len(labels) < 2 or "idle" not in labels:
        raise ValueError("ridge5 classifier state must include idle plus command labels")
    feature_mean = np.asarray(payload.get("feature_mean"), dtype=np.float64).reshape(-1)
    feature_std = np.asarray(payload.get("feature_std"), dtype=np.float64).reshape(-1)
    weights = np.asarray(payload.get("weights"), dtype=np.float64)
    if feature_mean.size <= 0 or feature_std.shape != feature_mean.shape:
        raise ValueError("ridge5 classifier state has invalid feature_mean/feature_std")
    if weights.ndim != 2 or weights.shape[0] != feature_mean.size + 1 or weights.shape[1] != len(labels):
        raise ValueError(
            "ridge5 classifier weights shape mismatch: "
            f"weights={tuple(weights.shape)} features={feature_mean.size} labels={len(labels)}"
        )
    feature_std = np.where(feature_std > 1e-9, feature_std, 1.0)
    normalized: dict[str, Any] = {
        **payload,
        "labels": labels,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "weights": weights,
        "command_confidence_th": float(payload.get("command_confidence_th", 0.0)),
        "smoothing_windows": max(1, int(payload.get("smoothing_windows", 1))),
        "gate_policy": str(payload.get("gate_policy", CLASSIFIER_CONFIDENCE_GATE_POLICY)).strip().lower(),
        "gate_variant": parse_classifier_gate_variant(payload.get("gate_variant")),
        "lrt_feature_indices": tuple(int(index) for index in payload.get("lrt_feature_indices", ()) or ()),
        "lrt_window_th": float(payload.get("lrt_window_th", 0.0)),
        "lrt_enter_th": float(payload.get("lrt_enter_th", 0.0)),
        "lrt_decay": float(payload.get("lrt_decay", DEFAULT_LRT_MULTIWINDOW_DECAY)),
        "l2": float(payload.get("l2", 0.0)),
        "fit_summary": dict(payload.get("fit_summary", {}) or {}),
        "frequency_specific_control_state_gates": normalize_frequency_specific_control_state_gates(
            payload.get("frequency_specific_control_state_gates")
        ),
    }
    for key in (
        "score_shape_margin_index",
        "score_shape_ratio_index",
        "score_shape_entropy_index",
    ):
        normalized[key] = None if payload.get(key) is None else int(payload.get(key))
    for key in (
        "score_shape_margin_th",
        "score_shape_ratio_th",
        "score_shape_entropy_th",
        "lrt_window_floor_th",
    ):
        value = payload.get(key)
        normalized[key] = None if value is None else float(value)
    normalized["weak_subject_guard_active"] = bool(payload.get("weak_subject_guard_active", False))
    normalized["weak_subject_guard_reasons"] = list(payload.get("weak_subject_guard_reasons", []) or [])
    for key in (
        "lrt_feature_mean_control",
        "lrt_feature_std_control",
        "lrt_feature_mean_idle",
        "lrt_feature_std_idle",
    ):
        value = payload.get(key)
        normalized[key] = None if value is None else np.asarray(value, dtype=np.float64).reshape(-1)
    return normalized


def ridge5_predict_windows_from_state(
    state: Mapping[str, Any],
    feature_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = normalize_ridge5_state(state)
    features = np.asarray(feature_matrix, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != int(model["feature_mean"].shape[0]):
        raise ValueError(
            "ridge5 feature matrix shape mismatch: "
            f"features={tuple(features.shape)} expected_features={int(model['feature_mean'].shape[0])}"
        )
    z = (features - model["feature_mean"]) / model["feature_std"]
    design = np.column_stack([np.ones(int(z.shape[0]), dtype=np.float64), z])
    logits = design @ np.asarray(model["weights"], dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-12)
    return probs.astype(np.float64, copy=False), np.asarray(model["labels"], dtype=object)


def lrt_window_evidence_from_state(
    state: Mapping[str, Any],
    feature_matrix: np.ndarray,
) -> np.ndarray:
    model = dict(state or {})
    features = np.asarray(feature_matrix, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] <= 0:
        return np.zeros(0, dtype=np.float64)
    indices = tuple(int(index) for index in model.get("lrt_feature_indices", ()) or ())
    if not indices:
        raise ValueError("lrt multi-window gate requires configured feature indices")
    selected = features[:, np.asarray(indices, dtype=int)]
    control_mean_raw = model.get("lrt_feature_mean_control")
    control_std_raw = model.get("lrt_feature_std_control")
    idle_mean_raw = model.get("lrt_feature_mean_idle")
    idle_std_raw = model.get("lrt_feature_std_idle")
    if control_mean_raw is None or control_std_raw is None or idle_mean_raw is None or idle_std_raw is None:
        raise ValueError("lrt multi-window gate is missing control/idle feature statistics")
    control_mean = np.asarray(control_mean_raw, dtype=np.float64).reshape(-1)
    control_std = np.asarray(control_std_raw, dtype=np.float64).reshape(-1)
    idle_mean = np.asarray(idle_mean_raw, dtype=np.float64).reshape(-1)
    idle_std = np.asarray(idle_std_raw, dtype=np.float64).reshape(-1)
    if (
        control_mean.shape != selected.shape[1:]
        or control_std.shape != selected.shape[1:]
        or idle_mean.shape != selected.shape[1:]
        or idle_std.shape != selected.shape[1:]
    ):
        raise ValueError("lrt multi-window gate feature statistics do not match feature indices")
    control_std = np.maximum(control_std, 1e-6)
    idle_std = np.maximum(idle_std, 1e-6)
    control_z = (selected - control_mean) / control_std
    idle_z = (selected - idle_mean) / idle_std
    log_control = -0.5 * np.sum(control_z * control_z, axis=1) - np.sum(np.log(control_std))
    log_idle = -0.5 * np.sum(idle_z * idle_z, axis=1) - np.sum(np.log(idle_std))
    return (log_control - log_idle).astype(np.float64, copy=False)


def score_shape_gate_mask_from_state(state: Mapping[str, Any], feature_matrix: np.ndarray) -> np.ndarray:
    model = normalize_ridge5_state(state)
    features = np.asarray(feature_matrix, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("feature_matrix must be 2D")
    mask = np.ones(int(features.shape[0]), dtype=bool)
    margin_index = model.get("score_shape_margin_index")
    margin_th = model.get("score_shape_margin_th")
    if margin_index is not None and margin_th is not None:
        mask &= features[:, int(margin_index)] + 1e-12 >= float(margin_th)
    ratio_index = model.get("score_shape_ratio_index")
    ratio_th = model.get("score_shape_ratio_th")
    if ratio_index is not None and ratio_th is not None:
        mask &= features[:, int(ratio_index)] + 1e-12 >= float(ratio_th)
    entropy_index = model.get("score_shape_entropy_index")
    entropy_th = model.get("score_shape_entropy_th")
    if entropy_index is not None and entropy_th is not None:
        mask &= features[:, int(entropy_index)] <= float(entropy_th) + 1e-12
    return mask


def command_confidence_from_probs(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    label_values = np.asarray(labels, dtype=object)
    idle_matches = np.where(label_values == "idle")[0]
    if idle_matches.size <= 0:
        raise ValueError("classifier labels must include idle")
    idle_index = int(idle_matches[0])
    return 1.0 - values[:, idle_index]
