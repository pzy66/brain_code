from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .async_fbcca_idle_standalone import (
    CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
    DEFAULT_GPU_CACHE_MODE,
    DEFAULT_GPU_DEVICE_ID,
    DEFAULT_GPU_PRECISION_NAME,
    DEFAULT_LRT_MULTIWINDOW_DECAY,
    DEFAULT_PROFILE_PATH,
    ThresholdProfile,
    TrialSpec,
    create_decoder,
    extract_window_batch,
    json_safe,
    save_profile,
)
from .profile_v2 import build_profile_v2
from .score_classifier_runtime import (
    CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
    classifier_feature_names,
    freq_label,
    lrt_window_evidence_from_state,
    normalize_frequency_specific_control_state_gates,
    ridge5_predict_windows_from_state,
    score_matrices_to_features,
    smooth_classifier_probabilities,
)


SESSION_NC_CLASSIFIER_SCHEMA_VERSION = "session_no_control_fbcca_ridge5_v1"
SESSION_NC_DEFAULT_FREQS = (8.0, 10.0, 12.0, 15.0)
SESSION_NC_FULL_REFERENCE_BANK_FREQS = (8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0)
SESSION_NC_WIN_SEC = 2.0
SESSION_NC_STEP_SEC = 0.25
SESSION_NC_MIN_ENTER_WINDOWS = 2
SESSION_NC_MIN_EXIT_WINDOWS = 1
SESSION_NC_SMOOTHING_WINDOWS = 3
SESSION_NC_MAX_GAP_WINDOWS = 0
SESSION_NC_RIDGE_L2 = 0.3
SESSION_NC_DECODER_NAME = "fbcca_fixed_all8"
SESSION_NC_DECODER_MODEL_PARAMS = {"Nh": 5, "subband_weight_mode": "chen_fixed"}
SESSION_NC_SCORE_BANK_MODE = "full_reference_bank"
SESSION_NC_SCORE_SOURCE_NAME = "fbcca"


@dataclass(frozen=True)
class SessionScoredTrial:
    trial: TrialSpec
    score_matrix: np.ndarray
    feature_matrix: np.ndarray
    duration_sec: float
    all_score_matrix: Optional[np.ndarray] = None
    all_freqs: tuple[float, ...] = ()


def _freq_tuple(values: Sequence[float]) -> tuple[float, float, float, float]:
    freqs = tuple(float(item) for item in values)
    if len(freqs) != 4:
        raise ValueError("session no-control classifier requires exactly 4 command frequencies")
    return freqs  # type: ignore[return-value]


def _classifier_labels(freqs: Sequence[float]) -> tuple[str, ...]:
    return ("idle", *(freq_label(float(freq)) for freq in freqs))


def _trial_true_label(trial: TrialSpec) -> str:
    return "idle" if trial.expected_freq is None else freq_label(float(trial.expected_freq))


def _trial_metadata(trial: TrialSpec) -> dict[str, Any]:
    metadata = getattr(trial, "metadata", None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _truthy_flag(value: Any, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _is_training_calibration_trial(trial: TrialSpec) -> bool:
    metadata = _trial_metadata(trial)
    label = str(getattr(trial, "label", "")).strip().lower()
    if "command_test" in label or label.endswith("_test") or "_test_" in label:
        return False
    if not metadata:
        return True
    if not _truthy_flag(metadata.get("valid", True), True):
        return False
    split_role = str(metadata.get("split_role", "calibration")).strip().lower()
    if split_role and split_role != "calibration":
        return False
    state_type = str(metadata.get("state_type", "")).strip().lower()
    if state_type == "baseline":
        return False
    if state_type in ("command", "no_control"):
        return True
    # Legacy manifests have no state_type; keep the previous behavior.
    return True


def _array_payload(value: np.ndarray) -> list[Any]:
    return np.asarray(value, dtype=np.float64).tolist()


def _safe_quantile(values: Sequence[Any] | np.ndarray, quantile: float, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float(default)
    return float(np.quantile(arr, min(max(float(quantile), 0.0), 1.0)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else float(default)
    except Exception:
        return float(default)


def _one_hot_labels(labels: Sequence[str], class_labels: Sequence[str]) -> np.ndarray:
    label_to_index = {str(label): int(index) for index, label in enumerate(class_labels)}
    target = np.zeros((len(labels), len(class_labels)), dtype=np.float64)
    for row_index, label in enumerate(labels):
        target[row_index, label_to_index[str(label)]] = 1.0
    return target


def _fit_balanced_ridge_classifier(
    scored_trials: Sequence[SessionScoredTrial],
    *,
    freqs: Sequence[float],
    l2: float = SESSION_NC_RIDGE_L2,
) -> dict[str, Any]:
    labels = _classifier_labels(freqs)
    rows: list[np.ndarray] = []
    targets: list[str] = []
    per_label_windows = {label: 0 for label in labels}
    per_label_trials = {label: 0 for label in labels}
    for item in scored_trials:
        if not _is_training_calibration_trial(item.trial):
            continue
        label = _trial_true_label(item.trial)
        if label not in per_label_windows:
            continue
        features = np.asarray(item.feature_matrix, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] <= 0:
            continue
        rows.append(features)
        targets.extend([label] * int(features.shape[0]))
        per_label_windows[label] += int(features.shape[0])
        per_label_trials[label] += 1
    missing = [label for label in labels if per_label_windows.get(label, 0) <= 0]
    if missing:
        raise ValueError(f"session classifier calibration missing classes: {missing}")
    x = np.vstack(rows).astype(np.float64, copy=False)
    y = np.asarray(targets, dtype=object)
    feature_mean = np.mean(x, axis=0)
    feature_std = np.std(x, axis=0)
    feature_std = np.where(feature_std > 1e-9, feature_std, 1.0)
    z = (x - feature_mean) / feature_std
    design = np.column_stack([np.ones(int(z.shape[0]), dtype=np.float64), z])
    target = _one_hot_labels(y, labels)
    sample_weights = np.ones(int(y.shape[0]), dtype=np.float64)
    for label in labels:
        mask = y == label
        count = int(np.sum(mask))
        if count > 0:
            sample_weights[mask] = 1.0 / float(count)
    sample_weights *= float(len(y)) / max(float(np.sum(sample_weights)), 1e-12)
    sqrt_weights = np.sqrt(sample_weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_target = target * sqrt_weights[:, None]
    reg = np.eye(int(design.shape[1]), dtype=np.float64) * max(float(l2), 0.0)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(weighted_design.T @ weighted_design + reg, weighted_design.T @ weighted_target)
    return {
        "freqs": [float(freq) for freq in freqs],
        "labels": list(labels),
        "feature_mean": _array_payload(feature_mean),
        "feature_std": _array_payload(feature_std),
        "weights": _array_payload(np.asarray(weights, dtype=np.float64)),
        "l2": float(l2),
        "per_label_windows": {label: int(per_label_windows[label]) for label in labels},
        "per_label_trials": {label: int(per_label_trials[label]) for label in labels},
        "calibration_windows": int(z.shape[0]),
    }


def _score_trials(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    sampling_rate: int,
    freqs: Sequence[float],
    full_reference_bank_freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
    gpu_warmup: bool,
    gpu_cache_policy: str,
) -> list[SessionScoredTrial]:
    command_decoder = create_decoder(
        SESSION_NC_DECODER_NAME,
        sampling_rate=int(sampling_rate),
        freqs=freqs,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        model_params=dict(SESSION_NC_DECODER_MODEL_PARAMS),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=bool(gpu_warmup),
        gpu_cache_policy=str(gpu_cache_policy),
    )
    full_bank_decoder = create_decoder(
        SESSION_NC_DECODER_NAME,
        sampling_rate=int(sampling_rate),
        freqs=full_reference_bank_freqs,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        model_params=dict(SESSION_NC_DECODER_MODEL_PARAMS),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=bool(gpu_warmup),
        gpu_cache_policy=str(gpu_cache_policy),
    )
    scored: list[SessionScoredTrial] = []
    for trial, segment in trial_segments:
        matrix = np.ascontiguousarray(np.asarray(segment, dtype=np.float64))
        if matrix.ndim != 2 or matrix.shape[0] < int(command_decoder.win_samples):
            continue
        windows = extract_window_batch(
            matrix,
            win_samples=int(command_decoder.win_samples),
            step_samples=int(command_decoder.step_samples),
        )
        command_scores = np.asarray(command_decoder.score_windows_batch(windows), dtype=np.float64)
        all_scores = np.asarray(full_bank_decoder.score_windows_batch(windows), dtype=np.float64)
        features = score_matrices_to_features(
            command_score_matrix=command_scores,
            command_freqs=freqs,
            score_bank_mode=SESSION_NC_SCORE_BANK_MODE,
            all_score_matrix=all_scores,
            all_freqs=full_reference_bank_freqs,
        )
        scored.append(
            SessionScoredTrial(
                trial=trial,
                score_matrix=command_scores,
                feature_matrix=features,
                duration_sec=float(matrix.shape[0]) / float(max(int(sampling_rate), 1)),
                all_score_matrix=all_scores,
                all_freqs=tuple(float(freq) for freq in full_reference_bank_freqs),
            )
        )
    return scored


def _lrt_feature_indices(freqs: Sequence[float]) -> tuple[int, ...]:
    names = classifier_feature_names(
        freqs,
        score_source_name=SESSION_NC_SCORE_SOURCE_NAME,
        score_bank_mode=SESSION_NC_SCORE_BANK_MODE,
    )
    wanted = (
        "top1_score",
        "margin",
        "ratio",
        "normalized_top1",
        "score_entropy",
        "top_command_to_top_all_ratio",
        "nearest_noncommand_margin",
        "all_bank_entropy",
    )
    indices = [int(names.index(name)) for name in wanted if name in names]
    if not indices:
        raise ValueError("session no-control LRT gate requires full-reference-bank features")
    return tuple(indices)


def _fit_lrt_gate_state(
    base_state: Mapping[str, Any],
    scored_trials: Sequence[SessionScoredTrial],
    *,
    freqs: Sequence[float],
    smoothing_windows: int,
) -> dict[str, Any]:
    labels = np.asarray(base_state["labels"], dtype=object)
    feature_indices = _lrt_feature_indices(freqs)
    control_rows: list[np.ndarray] = []
    idle_rows: list[np.ndarray] = []
    trial_counts = {"control": 0, "idle": 0}
    window_counts = {"control": 0, "idle": 0}
    for item in scored_trials:
        if not _is_training_calibration_trial(item.trial):
            continue
        features = np.asarray(item.feature_matrix, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] <= 0:
            continue
        probs, _labels = ridge5_predict_windows_from_state(base_state, features)
        smoothed = smooth_classifier_probabilities(probs, smoothing_windows=max(1, int(smoothing_windows)))
        pred_indices = np.argmax(smoothed, axis=1)
        lrt_rows = features[:, np.asarray(feature_indices, dtype=int)]
        true_label = _trial_true_label(item.trial)
        if true_label == "idle":
            idle_rows.append(lrt_rows)
            trial_counts["idle"] += 1
            window_counts["idle"] += int(lrt_rows.shape[0])
            continue
        command_rows = [
            lrt_rows[row_index]
            for row_index, pred_index in enumerate(pred_indices)
            if str(labels[int(pred_index)]) == true_label
        ]
        if command_rows:
            stacked = np.vstack(command_rows).astype(np.float64, copy=False)
            control_rows.append(stacked)
            trial_counts["control"] += 1
            window_counts["control"] += int(stacked.shape[0])
    if not control_rows or not idle_rows:
        raise ValueError("session LRT gate requires command and no-control calibration windows")
    control = np.vstack(control_rows).astype(np.float64, copy=False)
    idle = np.vstack(idle_rows).astype(np.float64, copy=False)
    control_mean = np.mean(control, axis=0)
    idle_mean = np.mean(idle, axis=0)
    control_std = np.maximum(np.std(control, axis=0), 1e-6)
    idle_std = np.maximum(np.std(idle, axis=0), 1e-6)
    local_state = {
        **dict(base_state),
        "lrt_feature_indices": list(range(len(feature_indices))),
        "lrt_feature_mean_control": _array_payload(control_mean),
        "lrt_feature_std_control": _array_payload(control_std),
        "lrt_feature_mean_idle": _array_payload(idle_mean),
        "lrt_feature_std_idle": _array_payload(idle_std),
    }
    control_scores = lrt_window_evidence_from_state(local_state, control)
    idle_scores = lrt_window_evidence_from_state(local_state, idle)
    window_th = _safe_quantile(idle_scores, 0.95, 0.0)
    if control_scores.size:
        window_th = min(float(window_th), _safe_quantile(control_scores, 0.50, window_th))
    window_th = max(float(window_th), 0.0)
    return {
        "lrt_feature_indices": [int(index) for index in feature_indices],
        "lrt_feature_names": [
            classifier_feature_names(
                freqs,
                score_source_name=SESSION_NC_SCORE_SOURCE_NAME,
                score_bank_mode=SESSION_NC_SCORE_BANK_MODE,
            )[index]
            for index in feature_indices
        ],
        "lrt_feature_mean_control": _array_payload(control_mean),
        "lrt_feature_std_control": _array_payload(control_std),
        "lrt_feature_mean_idle": _array_payload(idle_mean),
        "lrt_feature_std_idle": _array_payload(idle_std),
        "lrt_window_th": float(window_th),
        "lrt_enter_th": 0.0,
        "lrt_decay": float(DEFAULT_LRT_MULTIWINDOW_DECAY),
        "trial_counts": trial_counts,
        "window_counts": window_counts,
        "control_lrt_p50": _safe_quantile(control_scores, 0.50, 0.0),
        "idle_lrt_p95": _safe_quantile(idle_scores, 0.95, 0.0),
    }


def _count_trials(scored_trials: Sequence[SessionScoredTrial], freqs: Sequence[float]) -> dict[str, Any]:
    per_label = {label: 0 for label in _classifier_labels(freqs)}
    per_label_windows = {label: 0 for label in _classifier_labels(freqs)}
    for item in scored_trials:
        if not _is_training_calibration_trial(item.trial):
            continue
        label = _trial_true_label(item.trial)
        if label not in per_label:
            continue
        per_label[label] += 1
        per_label_windows[label] += int(np.asarray(item.feature_matrix).shape[0])
    return {
        "per_label_trials": per_label,
        "per_label_windows": per_label_windows,
        "control_trials": int(sum(count for label, count in per_label.items() if label != "idle")),
        "idle_trials": int(per_label.get("idle", 0)),
        "total_trials": int(sum(per_label.values())),
    }


def _profile_v2_payload(profile: ThresholdProfile, metrics: Mapping[str, Any]) -> dict[str, Any]:
    state = dict(dict(profile.model_params or {}).get("state") or {})
    per_freq_gate = normalize_frequency_specific_control_state_gates(
        state.get("frequency_specific_control_state_gates")
    )
    return build_profile_v2(
        base_profile=profile,
        per_freq_gate=per_freq_gate,
        metrics=dict(metrics),
        feature_names=tuple(
            classifier_feature_names(
                profile.freqs,
                score_source_name=SESSION_NC_SCORE_SOURCE_NAME,
                score_bank_mode=SESSION_NC_SCORE_BANK_MODE,
            )
        ),
        gate_type="session_no_control_lrt_multiwindow",
        evidence={
            "lambda": float(state.get("lrt_decay", DEFAULT_LRT_MULTIWINDOW_DECAY)),
            "upper_commit_th": float(state.get("lrt_enter_th", 0.0)),
            "lower_idle_th": float(state.get("lrt_window_th", 0.0)),
        },
    ).to_payload()


def save_session_no_control_profile_bundle(
    profile: ThresholdProfile,
    output_profile_path: Path,
    metrics: Mapping[str, Any],
) -> tuple[Path, Path]:
    profile_path = Path(output_profile_path).expanduser().resolve()
    if profile_path == Path(DEFAULT_PROFILE_PATH).expanduser().resolve():
        raise ValueError("session no-control profile must not overwrite default_profile.json")
    if profile_path.name in {"default_profile.json", "default_profile_v2.json"}:
        raise ValueError("session no-control profile must not overwrite default_profile.json")
    save_profile(profile, profile_path)
    if "_profile" in profile_path.stem:
        profile_v2_stem = profile_path.stem.replace("_profile", "_profile_v2", 1)
    else:
        profile_v2_stem = f"{profile_path.stem}_v2"
    profile_v2_path = profile_path.with_name(f"{profile_v2_stem}.json")
    from .async_fbcca_idle_standalone import atomic_write_text, json_dumps

    atomic_write_text(profile_v2_path, json_dumps(json_safe(_profile_v2_payload(profile, metrics))) + "\n")
    return profile_path, profile_v2_path.resolve()


def fit_session_no_control_fbcca_ridge_profile(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    sampling_rate: int,
    available_board_channels: Sequence[int],
    freqs: Sequence[float] = SESSION_NC_DEFAULT_FREQS,
    full_reference_bank_freqs: Sequence[float] = SESSION_NC_FULL_REFERENCE_BANK_FREQS,
    win_sec: float = SESSION_NC_WIN_SEC,
    step_sec: float = SESSION_NC_STEP_SEC,
    min_enter_windows: int = SESSION_NC_MIN_ENTER_WINDOWS,
    min_exit_windows: int = SESSION_NC_MIN_EXIT_WINDOWS,
    smoothing_windows: int = SESSION_NC_SMOOTHING_WINDOWS,
    max_gap_windows: int = SESSION_NC_MAX_GAP_WINDOWS,
    l2: float = SESSION_NC_RIDGE_L2,
    compute_backend: str = "cpu",
    gpu_device: int = DEFAULT_GPU_DEVICE_ID,
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
    gpu_warmup: bool = False,
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE,
) -> tuple[ThresholdProfile, dict[str, Any]]:
    freq_tuple = _freq_tuple(freqs)
    full_freq_tuple = tuple(float(freq) for freq in full_reference_bank_freqs)
    scored_trials = _score_trials(
        trial_segments,
        sampling_rate=int(sampling_rate),
        freqs=freq_tuple,
        full_reference_bank_freqs=full_freq_tuple,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=bool(gpu_warmup),
        gpu_cache_policy=str(gpu_cache_policy),
    )
    coverage = _count_trials(scored_trials, freq_tuple)
    missing = [
        label
        for label, count in dict(coverage["per_label_trials"]).items()
        if int(count) <= 0
    ]
    if missing:
        raise ValueError(f"session no-control classifier missing required calibration labels: {missing}")
    base_state = _fit_balanced_ridge_classifier(scored_trials, freqs=freq_tuple, l2=float(l2))
    lrt_payload = _fit_lrt_gate_state(
        base_state,
        scored_trials,
        freqs=freq_tuple,
        smoothing_windows=max(1, int(smoothing_windows)),
    )
    feature_names = classifier_feature_names(
        freq_tuple,
        score_source_name=SESSION_NC_SCORE_SOURCE_NAME,
        score_bank_mode=SESSION_NC_SCORE_BANK_MODE,
    )
    state = {
        **base_state,
        "command_confidence_th": 0.0,
        "smoothing_windows": max(1, int(smoothing_windows)),
        "gate_policy": CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        "gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
        "lrt_feature_indices": list(lrt_payload["lrt_feature_indices"]),
        "lrt_feature_mean_control": list(lrt_payload["lrt_feature_mean_control"]),
        "lrt_feature_std_control": list(lrt_payload["lrt_feature_std_control"]),
        "lrt_feature_mean_idle": list(lrt_payload["lrt_feature_mean_idle"]),
        "lrt_feature_std_idle": list(lrt_payload["lrt_feature_std_idle"]),
        "lrt_window_th": float(lrt_payload["lrt_window_th"]),
        "lrt_enter_th": float(lrt_payload["lrt_enter_th"]),
        "lrt_decay": float(lrt_payload["lrt_decay"]),
        "fit_summary": {
            "schema_version": SESSION_NC_CLASSIFIER_SCHEMA_VERSION,
            "classifier": "fbcca_score_ridge_5class",
            "score_source_name": SESSION_NC_SCORE_SOURCE_NAME,
            "score_bank_mode": SESSION_NC_SCORE_BANK_MODE,
            "decoder": SESSION_NC_DECODER_NAME,
            "decoder_model_params": dict(SESSION_NC_DECODER_MODEL_PARAMS),
            "gate_policy": CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
            "gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
            "min_enter_windows": max(1, int(min_enter_windows)),
            "max_gap_windows": max(0, int(max_gap_windows)),
            "smoothing_windows": max(1, int(smoothing_windows)),
            "calibration_counts": coverage,
            "lrt_gate": dict(lrt_payload),
        },
    }
    metadata = {
        "source": "realtime_online_ui_session_no_control_classifier",
        "schema_version": SESSION_NC_CLASSIFIER_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "fbcca_ridge5",
        "scorer": "fbcca_fixed_all8",
        "score_bank": SESSION_NC_SCORE_BANK_MODE,
        "gate": CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        "gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
        "no_control_calibration": {
            "enabled": True,
            "source": "session_idle_trials",
            "trial_count": int(coverage["idle_trials"]),
            "fit_split": "current_session_pretrain_trials",
        },
        "calibration_counts": coverage,
        "full_reference_bank_freqs": [float(freq) for freq in full_freq_tuple],
        "recommended_for_realtime_note": "Research mainline runtime profile; keep default_profile.json unchanged until validated on local sessions.",
    }
    profile = ThresholdProfile(
        freqs=freq_tuple,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        enter_score_th=999.0,
        enter_ratio_th=999.0,
        enter_margin_th=999.0,
        exit_score_th=999.0,
        exit_ratio_th=999.0,
        min_enter_windows=max(1, int(min_enter_windows)),
        min_exit_windows=max(1, int(min_exit_windows)),
        model_name="fbcca_score_ridge_5class",
        model_params={
            "state": json_safe(state),
            "score_source_name": SESSION_NC_SCORE_SOURCE_NAME,
            "score_bank_mode": SESSION_NC_SCORE_BANK_MODE,
            "decoder_name": SESSION_NC_DECODER_NAME,
            "decoder_model_params": dict(SESSION_NC_DECODER_MODEL_PARAMS),
            "feature_names": list(feature_names),
            "full_reference_bank_freqs": [float(freq) for freq in full_freq_tuple],
            "max_gap_windows": max(0, int(max_gap_windows)),
        },
        gate_policy=CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        eeg_channels=tuple(int(channel) for channel in available_board_channels),
        channel_weight_mode=None,
        channel_weights=None,
        subband_weight_mode="chen_fixed",
        recommended_for_realtime=True,
        profile_validation_status={
            "status": "trained_from_current_session",
            "schema_version": SESSION_NC_CLASSIFIER_SCHEMA_VERSION,
            "calibration_counts": coverage,
            "lrt_gate": dict(lrt_payload),
        },
        metadata=metadata,
    )
    quality = {
        "status": "ok",
        "schema_version": SESSION_NC_CLASSIFIER_SCHEMA_VERSION,
        "calibration_counts": coverage,
        "feature_count": int(len(feature_names)),
        "score_bank_mode": SESSION_NC_SCORE_BANK_MODE,
        "lrt_window_th": float(lrt_payload["lrt_window_th"]),
        "lrt_enter_th": float(lrt_payload["lrt_enter_th"]),
        "min_enter_windows": int(profile.min_enter_windows),
        "smoothing_windows": int(smoothing_windows),
        "model_name": str(profile.model_name),
    }
    return profile, quality
