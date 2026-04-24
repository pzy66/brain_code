from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from .async_fbcca_idle_standalone import (
    DEFAULT_ASYNC_DECISION_TIME_MODE,
    DEFAULT_CONTROL_STATE_MODE,
    DEFAULT_GATE_POLICY,
    DEFAULT_GPU_CACHE_MODE,
    DEFAULT_GPU_DEVICE_ID,
    DEFAULT_GPU_PRECISION_NAME,
    DEFAULT_NH,
    DEFAULT_PAPER_DECISION_TIME_MODE,
    MODEL_FEATURE_NAMES,
    ThresholdProfile,
    TrialSpec,
    atomic_write_text,
    build_feature_rows_with_decoder,
    compute_classification_metrics,
    create_decoder,
    extract_window_batch,
    fit_threshold_profile,
    json_dumps,
    json_safe,
    parse_tdca_decoder_variant,
    parse_compute_backend_name,
    parse_control_state_mode,
    parse_decision_time_mode,
    profile_meets_acceptance,
    save_profile,
)

from .compute_backend import resolve_compute_backend
from .dataset import LoadedDataset, infer_trial_role, load_collection_dataset, summarize_trial_roles
from .decision import DecisionEngine, DecisionEngineConfig, EvidenceAccumulatorConfig, StateMachineConfig
from .gating import (
    CorrectnessCalibrator,
    CorrectnessCalibratorConfig,
    PerFrequencyLogRegGate,
    RollingFeatureHistory,
)
from .gating.correctness_calibrator import (
    BAYESIAN_GAP_GMM,
    GLOBAL_CORRECTNESS_LOGISTIC,
)
from .gating.per_freq_logreg_gate import LogRegFitConfig
from .profile_v2 import DEFAULT_GATE_FEATURES, build_profile_v2
from .run_artifacts import (
    make_run_tag,
    publish_deployed_profile,
    resolve_ssvep_run_artifacts,
)
from .trial_roles import resolve_trial_role


DEFAULT_TDCA_LOCAL_MODEL = "tdca"
DEFAULT_TDCA_LOCAL_CHANNEL_MODE = "all8"
DEFAULT_TDCA_LOCAL_WIN_CANDIDATES = (2.0, 2.5, 3.0, 3.5)
DEFAULT_TDCA_LOCAL_DELAY_STEPS = (2, 3, 4, 5)
DEFAULT_TDCA_LOCAL_N_COMPONENTS = (2, 3, 4)
DEFAULT_TDCA_LOCAL_STEP_SEC = 0.25
DEFAULT_TDCA_LOCAL_REPEAT_COUNT = 5
DEFAULT_TDCA_LOCAL_TOP_K = 8
DEFAULT_DECISION_GRID_CANDIDATE_MIN_WINDOWS = (1, 2)
DEFAULT_DECISION_GRID_ARMED_MIN_WINDOWS = (1, 2, 3)
DEFAULT_DECISION_GRID_LAMBDA = (0.65, 0.75, 0.85, 0.90)
DEFAULT_DECISION_GRID_UPPER = (0.0, 0.4, 0.8, 1.2)
DEFAULT_DECISION_GRID_LOWER = (-1.2, -0.8, -0.4, 0.0)
DEFAULT_DECISION_GRID_REFRACTORY = (0.0, 0.2, 0.4, 0.8)
DEFAULT_DECISION_COMMIT_CONSISTENCY_TH = 0.6
DEFAULT_DECISION_BETA_CONSISTENCY = 0.5
DEFAULT_LOGREG_FIT_CONFIG = LogRegFitConfig()
DEFAULT_CORRECTNESS_CALIBRATOR_CONFIG = CorrectnessCalibratorConfig()
DEFAULT_BASELINE_MODELS = ("fbcca", "trca_r")
DEFAULT_BASELINE_WIN_SEC = 3.0
DEFAULT_TDCA_LOCAL_DECODER_VARIANT = "tdca_like_legacy"
DEFAULT_TDCA_LOCAL_DECODER_VARIANTS = ("tdca_like_legacy", "tdca_paper_aligned")
DEFAULT_TDCA_LOCAL_SEARCH_PRESET = "reduced13"
TDCA_LOCAL_SEARCH_PRESETS = ("smoke4", "reduced13", "full96")
DEFAULT_CONFIDENCE_VARIANT = GLOBAL_CORRECTNESS_LOGISTIC
DEFAULT_CONFIDENCE_VARIANTS = (
    GLOBAL_CORRECTNESS_LOGISTIC,
    BAYESIAN_GAP_GMM,
)
DEFAULT_CONFIDENCE_TRAINING_SCHEME = "oof_gate_logreg_on_train_split"
DEFAULT_GATE_ENTER_P_GRID = (0.50, 0.60, 0.70, 0.80)
DEFAULT_GATE_EXIT_P_GRID = (0.15, 0.25, 0.35, 0.45)
DEFAULT_DECISION_EVIDENCE_VARIANT = "centered_logit_over_enter_threshold"
DEFAULT_TUNE_MIN_CONTROL_TRIALS_PER_FREQ = 4
DEFAULT_TUNE_MIN_IDLE_TRIALS = 10
DEFAULT_TDCA_LOCAL_DATA_DEPLOYMENT_MIN_SESSIONS = 2
TDCA_DECODER_VARIANT_METADATA = {
    "tdca_like_legacy": {
        "algorithm_alignment": "not-paper-exact",
        "paper_tdca_projection_enabled": False,
    },
    "tdca_paper_aligned": {
        "algorithm_alignment": "paper-aligned",
        "paper_tdca_projection_enabled": True,
    },
}
DEFAULT_REPLAY_BENCHMARK_SHAPE = (256, 8, 64)
DEFAULT_REPLAY_BENCHMARK_REPEATS = 3
DEFAULT_GPU_REPLAY_MIN_WINDOWS = 256
DEFAULT_GPU_REPLAY_MIN_SPEEDUP = 1.5
DEFAULT_BASELINE_DECISION_PARAMS = {
    "candidate_min_windows": 1,
    "armed_min_windows": 2,
    "lambda_decay": 0.85,
    "upper_commit_th": 2.2,
    "lower_idle_th": 0.4,
    "refractory_sec": 0.8,
}


class GateReplayState:
    def __init__(self, profile: ThresholdProfile) -> None:
        self.profile = profile
        self.reset()

    def reset(self) -> None:
        self._candidate_freq: Optional[float] = None
        self._candidate_windows = 0
        self._gate_open_freq: Optional[float] = None
        self._exit_windows = 0
        self._switch_candidate_freq: Optional[float] = None
        self._switch_candidate_windows = 0

    def _threshold_payload(self, *, freq_value: Optional[float], selected_fallback: bool = False) -> dict[str, Any]:
        resolved_freq = self._gate_open_freq if selected_fallback and self._gate_open_freq is not None else freq_value
        return _profile_threshold_payload(self.profile, freq_value=resolved_freq)

    def _enter_pass(self, row: Mapping[str, Any], *, pred_freq: Optional[float]) -> bool:
        if pred_freq is None:
            return False
        payload = self._threshold_payload(freq_value=pred_freq)
        enter_p_th = _clip_probability(payload.get("enter_p_th", self.profile.enter_p_th if self.profile.enter_p_th is not None else 0.65), 0.65)
        return bool(_row_correctness_probability(row) >= float(enter_p_th))

    def _switch_pass(self, row: Mapping[str, Any], *, pred_freq: Optional[float]) -> bool:
        if self._gate_open_freq is None or pred_freq is None:
            return False
        if abs(float(pred_freq) - float(self._gate_open_freq)) <= 1e-8:
            return False
        payload = self._threshold_payload(freq_value=pred_freq)
        enter_p_th = _clip_probability(payload.get("enter_p_th", self.profile.enter_p_th if self.profile.enter_p_th is not None else 0.65), 0.65)
        return bool(_row_correctness_probability(row) >= float(enter_p_th))

    def _exit_fail(self, row: Mapping[str, Any], *, pred_freq: Optional[float]) -> bool:
        if self._gate_open_freq is None:
            return True
        payload = self._threshold_payload(freq_value=pred_freq, selected_fallback=True)
        if pred_freq is None or abs(float(pred_freq) - float(self._gate_open_freq)) > 1e-8:
            return True
        exit_p_th = _clip_probability(payload.get("exit_p_th", self.profile.exit_p_th if self.profile.exit_p_th is not None else 0.30), 0.30)
        return bool(_row_correctness_probability(row) < float(exit_p_th))

    def update(self, row_raw: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(row_raw)
        pred_freq = row.get("pred_freq")
        pred_freq_value = None if pred_freq is None else _safe_float(pred_freq, float("nan"))
        if pred_freq_value is not None and not np.isfinite(pred_freq_value):
            pred_freq_value = None
        enter_pass = self._enter_pass(row, pred_freq=pred_freq_value)
        switch_pass = self._switch_pass(row, pred_freq=pred_freq_value)
        exit_fail = False
        gate_event = "hold"
        gate_switched = False
        if self._gate_open_freq is None:
            min_enter_windows = int(
                self._threshold_payload(freq_value=pred_freq_value).get("min_enter_windows", self.profile.min_enter_windows)
            )
            if enter_pass and pred_freq_value is not None:
                if self._candidate_freq is None or abs(float(pred_freq_value) - float(self._candidate_freq)) > 1e-8:
                    self._candidate_freq = float(pred_freq_value)
                    self._candidate_windows = 1
                else:
                    self._candidate_windows += 1
                if self._candidate_windows >= max(int(min_enter_windows), 1):
                    self._gate_open_freq = float(pred_freq_value)
                    self._exit_windows = 0
                    self._switch_candidate_freq = None
                    self._switch_candidate_windows = 0
                    gate_event = "enter"
            else:
                self._candidate_freq = None
                self._candidate_windows = 0
                self._switch_candidate_freq = None
                self._switch_candidate_windows = 0
        else:
            pred_is_current = (
                pred_freq_value is not None and abs(float(pred_freq_value) - float(self._gate_open_freq)) <= 1e-8
            )
            if switch_pass and pred_freq_value is not None and not pred_is_current:
                min_switch_windows = int(
                    self._threshold_payload(freq_value=pred_freq_value).get(
                        "min_switch_windows",
                        self.profile.min_switch_windows,
                    )
                )
                if self._switch_candidate_freq is None or abs(float(pred_freq_value) - float(self._switch_candidate_freq)) > 1e-8:
                    self._switch_candidate_freq = float(pred_freq_value)
                    self._switch_candidate_windows = 1
                else:
                    self._switch_candidate_windows += 1
                self._exit_windows = 0
                if self._switch_candidate_windows >= max(int(min_switch_windows), 1):
                    self._gate_open_freq = float(pred_freq_value)
                    self._candidate_freq = None
                    self._candidate_windows = 0
                    self._switch_candidate_freq = None
                    self._switch_candidate_windows = 0
                    gate_event = "switch"
                    gate_switched = True
            else:
                self._switch_candidate_freq = None
                self._switch_candidate_windows = 0
                exit_fail = self._exit_fail(row, pred_freq=pred_freq_value)
                min_exit_windows = int(
                    self._threshold_payload(freq_value=pred_freq_value, selected_fallback=True).get(
                        "min_exit_windows",
                        self.profile.min_exit_windows,
                    )
                )
                if exit_fail:
                    self._exit_windows += 1
                    if self._exit_windows >= max(int(min_exit_windows), 1):
                        self._gate_open_freq = None
                        self._candidate_freq = None
                        self._candidate_windows = 0
                        self._exit_windows = 0
                        gate_event = "exit"
                else:
                    self._exit_windows = 0
        row["gate_open_freq"] = None if self._gate_open_freq is None else float(self._gate_open_freq)
        row["gate_is_open"] = bool(self._gate_open_freq is not None)
        row["enter_pass"] = bool(enter_pass)
        row["switch_pass"] = bool(switch_pass)
        row["exit_fail"] = bool(exit_fail)
        row["gate_candidate_windows"] = int(self._candidate_windows)
        row["gate_exit_windows"] = int(self._exit_windows)
        row["gate_switch_candidate_freq"] = None if self._switch_candidate_freq is None else float(self._switch_candidate_freq)
        row["gate_switch_candidate_windows"] = int(self._switch_candidate_windows)
        row["gate_event"] = str(gate_event)
        row["gate_switched"] = bool(gate_switched)
        return row


@dataclass(frozen=True)
class RepeatedGroupSplit:
    repeat_index: int
    train_indices: tuple[int, ...]
    gate_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]
    fingerprint: str


@dataclass(frozen=True)
class MergedLocalDataset:
    manifest_paths: tuple[Path, ...]
    datasets: tuple[LoadedDataset, ...]
    trial_segments: tuple[tuple[TrialSpec, np.ndarray], ...]
    sampling_rate: int
    freqs: tuple[float, float, float, float]
    board_eeg_channels: tuple[int, ...]
    subject_id: str
    session_ids: tuple[str, ...]
    trial_role_counts: dict[str, int]
    quality_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TDCALocalOptConfig:
    dataset_manifest_session1: Path
    output_profile_path: Path
    report_path: Path
    dataset_manifests: tuple[Path, ...] = ()
    report_root_dir: Optional[Path] = None
    organize_report_dir: bool = False
    model_names: tuple[str, ...] = (DEFAULT_TDCA_LOCAL_MODEL,)
    channel_modes: tuple[str, ...] = (DEFAULT_TDCA_LOCAL_CHANNEL_MODE,)
    multi_seed_count: int = DEFAULT_TDCA_LOCAL_REPEAT_COUNT
    win_candidates: tuple[float, ...] = DEFAULT_TDCA_LOCAL_WIN_CANDIDATES
    tdca_delay_steps: tuple[int, ...] = DEFAULT_TDCA_LOCAL_DELAY_STEPS
    tdca_n_components: tuple[int, ...] = DEFAULT_TDCA_LOCAL_N_COMPONENTS
    search_preset: str = DEFAULT_TDCA_LOCAL_SEARCH_PRESET
    step_sec: float = DEFAULT_TDCA_LOCAL_STEP_SEC
    Nh: int = DEFAULT_NH
    seed: int = 20260410
    compute_backend: str = "auto"
    gpu_device: int = DEFAULT_GPU_DEVICE_ID
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME
    gpu_warmup: bool = True
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE
    control_state_mode: str = "frequency-specific-logistic"
    decision_time_mode: str = DEFAULT_PAPER_DECISION_TIME_MODE
    async_decision_time_mode: str = DEFAULT_ASYNC_DECISION_TIME_MODE
    progress_heartbeat_sec: float = 5.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except Exception:
        return float(default)
    return float(output) if np.isfinite(output) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _freq_label(freq: float) -> str:
    return f"{float(freq):g}"


def _profile_threshold_payload(
    profile: ThresholdProfile,
    *,
    freq_value: Optional[float],
) -> dict[str, Any]:
    payload_map = dict(profile.frequency_specific_thresholds or {})
    if not payload_map or freq_value is None:
        return {}
    payload = payload_map.get(_freq_label(float(freq_value)))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _tdca_variant_metadata(decoder_variant: Optional[str]) -> dict[str, Any]:
    variant = parse_tdca_decoder_variant(decoder_variant)
    payload = dict(TDCA_DECODER_VARIANT_METADATA.get(variant, {}))
    payload["decoder_variant"] = str(variant)
    return payload


def _tdca_variant_priority(decoder_variant: Optional[str]) -> int:
    variant = parse_tdca_decoder_variant(decoder_variant)
    return 0 if variant == "tdca_paper_aligned" else 1


def _confidence_variant_priority(confidence_variant: Optional[str]) -> int:
    variant = str(confidence_variant or DEFAULT_CONFIDENCE_VARIANT).strip().lower()
    return 0 if variant == GLOBAL_CORRECTNESS_LOGISTIC else 1


def _resolved_confidence_variant(
    row: Mapping[str, Any],
    *,
    default: str = DEFAULT_CONFIDENCE_VARIANT,
) -> str:
    candidate = dict(row.get("candidate", {}))
    value = row.get("confidence_variant", candidate.get("confidence_variant", default))
    variant = str(value or default).strip().lower()
    if variant not in DEFAULT_CONFIDENCE_VARIANTS:
        return str(default)
    return variant


def _nearest_freq(freq: float, freqs: Sequence[float]) -> float:
    items = [float(item) for item in freqs]
    if not items:
        raise ValueError("frequency list is empty")
    return min(items, key=lambda item: abs(float(item) - float(freq)))


def _latency_or_penalty(value: Any, penalty: float = 1_000_000.0) -> float:
    output = _safe_float(value, penalty)
    return float(output) if np.isfinite(output) else float(penalty)


def _median(values: Sequence[Any], default: float = 0.0) -> float:
    numeric = np.asarray(
        [float(item) for item in values if np.isfinite(_safe_float(item, float("nan")))],
        dtype=float,
    )
    if numeric.size == 0:
        return float(default)
    return float(np.median(numeric))


def _clip_probability(value: Any, default: float = 0.5) -> float:
    return float(np.clip(_safe_float(value, default), 1e-6, 1.0 - 1e-6))


def _p_to_logit(value: Any, default: float = 0.5) -> float:
    p = _clip_probability(value, default)
    return float(np.log(p / max(1.0 - p, 1e-12)))


def _logit_to_probability(value: Any, default: float = 0.5) -> float:
    candidate = _safe_float(value, float("nan"))
    if not np.isfinite(candidate):
        return _clip_probability(default, default)
    clipped = float(np.clip(candidate, -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _row_correctness_probability(row: Mapping[str, Any]) -> float:
    if "p_correct" in row:
        return _clip_probability(row.get("p_correct", 0.0), 0.0)
    if "p_control" in row:
        return _clip_probability(row.get("p_control", 0.0), 0.0)
    if "correctness_logit" in row:
        return _logit_to_probability(row.get("correctness_logit", 0.0), 0.5)
    if "control_log_lr" in row:
        return _logit_to_probability(row.get("control_log_lr", 0.0), 0.5)
    if "gate_score" in row:
        return _logit_to_probability(row.get("gate_score", 0.0), 0.5)
    return 0.0


def _row_correctness_label(row: Mapping[str, Any]) -> float:
    expected = row.get("expected_freq")
    pred = row.get("pred_freq")
    if expected is None or pred is None:
        return 0.0
    expected_value = _safe_float(expected, float("nan"))
    pred_value = _safe_float(pred, float("nan"))
    if not np.isfinite(expected_value) or not np.isfinite(pred_value):
        return 0.0
    if resolve_trial_role(row) != "control":
        return 0.0
    return 1.0 if abs(float(expected_value) - float(pred_value)) <= 1e-8 else 0.0


def _decision_evidence_reference_logit(
    *,
    profile: ThresholdProfile,
    pred_freq: Optional[float],
) -> float:
    payload = _profile_threshold_payload(profile, freq_value=pred_freq)
    explicit_reference = payload.get("decision_evidence_reference", payload.get("enter_reference_logit"))
    if explicit_reference is not None:
        candidate = _safe_float(explicit_reference, float("nan"))
        if np.isfinite(candidate):
            return float(candidate)
    enter_p_th = _clip_probability(
        payload.get(
            "enter_p_th",
            profile.enter_p_th if profile.enter_p_th is not None else 0.65,
        ),
        0.65,
    )
    return float(_p_to_logit(enter_p_th, 0.65))


def _decision_evidence_row(
    *,
    row: Mapping[str, Any],
    profile: ThresholdProfile,
) -> dict[str, Any]:
    scored = dict(row)
    pred_freq = scored.get("pred_freq")
    pred_freq_value = None if pred_freq is None else _safe_float(pred_freq, float("nan"))
    if pred_freq_value is not None and not np.isfinite(pred_freq_value):
        pred_freq_value = None
    raw_logit = _safe_float(
        scored.get(
            "correctness_logit",
            scored.get("control_log_lr", scored.get("gate_score", 0.0)),
        ),
        0.0,
    )
    reference_logit = _decision_evidence_reference_logit(
        profile=profile,
        pred_freq=pred_freq_value,
    )
    scored["decision_evidence_raw"] = float(raw_logit)
    scored["decision_evidence_reference"] = float(reference_logit)
    scored["decision_evidence_centered"] = float(raw_logit - reference_logit)
    scored["decision_evidence_variant"] = DEFAULT_DECISION_EVIDENCE_VARIANT
    return scored


def _binary_auc(y_true: Sequence[Any], y_score: Sequence[Any]) -> Optional[float]:
    positives = np.asarray(
        [
            _safe_float(score, 0.0)
            for label, score in zip(y_true, y_score)
            if float(_safe_float(label, 0.0)) > 0.5
        ],
        dtype=float,
    )
    negatives = np.asarray(
        [
            _safe_float(score, 0.0)
            for label, score in zip(y_true, y_score)
            if float(_safe_float(label, 0.0)) <= 0.5
        ],
        dtype=float,
    )
    if positives.size <= 0 or negatives.size <= 0:
        return None
    total = float(positives.size * negatives.size)
    wins = 0.0
    for value in positives:
        wins += float(np.sum(value > negatives))
        wins += 0.5 * float(np.sum(value == negatives))
    return float(wins / max(total, 1.0))


def _quantile_candidates(
    values: np.ndarray,
    quantiles: Sequence[float],
    *,
    floor: Optional[float] = None,
) -> tuple[float, ...]:
    numeric = np.asarray(values, dtype=float)
    if numeric.size <= 0:
        return tuple()
    candidates: list[float] = []
    for quantile in quantiles:
        q = float(np.clip(_safe_float(quantile, 0.5), 0.0, 1.0))
        candidate = float(np.quantile(numeric, q))
        if floor is not None and candidate < float(floor):
            candidate = float(floor)
        if np.isfinite(candidate):
            candidates.append(float(candidate))
    return tuple(dict.fromkeys(float(item) for item in candidates))


def _rank_metrics_key(metrics: Mapping[str, Any]) -> tuple[float, float, float, float, float, float]:
    control_recall = float(metrics.get("control_recall", 0.0))
    control_recall_at_3s = float(metrics.get("control_recall_at_3s", control_recall))
    return (
        float(metrics.get("idle_fp_per_min", float("inf"))),
        _latency_or_penalty(metrics.get("release_latency_s", float("inf"))),
        _latency_or_penalty(metrics.get("switch_latency_s", float("inf"))),
        -control_recall_at_3s,
        -control_recall,
        float(metrics.get("inference_ms", float("inf"))),
    )


def _fingerprint_for_split(
    *,
    train_indices: Sequence[int],
    gate_indices: Sequence[int],
    holdout_indices: Sequence[int],
) -> str:
    payload = "|".join(
        [
            ",".join(str(int(item)) for item in sorted(train_indices)),
            ",".join(str(int(item)) for item in sorted(gate_indices)),
            ",".join(str(int(item)) for item in sorted(holdout_indices)),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _candidate_key(
    *,
    decoder_variant: str,
    win_sec: float,
    delay_steps: int,
    n_components: int,
    confidence_variant: str,
) -> str:
    return (
        f"variant={parse_tdca_decoder_variant(decoder_variant)}|win={float(win_sec):g}|"
        f"delay={int(delay_steps)}|n_components={int(n_components)}|"
        f"confidence={str(confidence_variant).strip().lower()}"
    )


def _default_decision_params() -> dict[str, Any]:
    return dict(DEFAULT_BASELINE_DECISION_PARAMS)


def _normalize_name_list(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(item).strip().lower() for item in values if str(item).strip())


def _validate_tdca_local_config(config: TDCALocalOptConfig) -> None:
    model_names = _normalize_name_list(config.model_names)
    channel_modes = _normalize_name_list(config.channel_modes)
    if model_names != (DEFAULT_TDCA_LOCAL_MODEL,):
        raise ValueError(f"tdca-local-opt only supports model_names=('tdca',); got {config.model_names}")
    if channel_modes != (DEFAULT_TDCA_LOCAL_CHANNEL_MODE,):
        raise ValueError(f"tdca-local-opt only supports channel_modes=('all8',); got {config.channel_modes}")
    preset = str(config.search_preset or DEFAULT_TDCA_LOCAL_SEARCH_PRESET).strip().lower()
    if preset not in TDCA_LOCAL_SEARCH_PRESETS:
        raise ValueError(f"tdca-local-opt only supports search_preset in {TDCA_LOCAL_SEARCH_PRESETS}; got {config.search_preset}")


def _estimate_window_count(
    *,
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    sampling_rate: int,
    win_sec: float,
    step_sec: float,
) -> int:
    win_samples = max(int(round(float(win_sec) * int(sampling_rate))), 1)
    step_samples = max(int(round(float(step_sec) * int(sampling_rate))), 1)
    total = 0
    for _trial, segment in trial_segments:
        matrix = np.asarray(segment, dtype=float)
        if matrix.shape[0] < win_samples:
            continue
        total += 1 + max(int((matrix.shape[0] - win_samples) // step_samples), 0)
    return int(total)


def _microbenchmark_replay_backend(
    backend: Any,
    *,
    sample_shape: tuple[int, int, int] = DEFAULT_REPLAY_BENCHMARK_SHAPE,
    repeats: int = DEFAULT_REPLAY_BENCHMARK_REPEATS,
) -> dict[str, Any]:
    shape = tuple(max(int(item), 1) for item in sample_shape)
    repeat_count = max(int(repeats), 1)
    host_array = np.linspace(0.0, 1.0, int(np.prod(shape, dtype=np.int64)), dtype=np.float32).reshape(shape)
    warmup_ms = float(getattr(backend, "benchmark_warmup", lambda: 0.0)())
    total_ms_values: list[float] = []
    compute_ms_values: list[float] = []
    for _ in range(repeat_count):
        device_value, upload_ms = backend.to_device(host_array)
        t0 = time.perf_counter()
        reduced = backend.xp.sum(device_value * device_value, axis=-1)
        sync_ms = float(backend.synchronize())
        _host_value, download_ms = backend.to_host(reduced)
        t1 = time.perf_counter()
        total_ms_values.append(float((t1 - t0) * 1000.0 + upload_ms))
        compute_ms_values.append(float(max((t1 - t0) * 1000.0 - download_ms - sync_ms, 0.0)))
    return {
        "backend_name": str(getattr(backend, "backend_name", "cpu")),
        "sample_shape": [int(item) for item in shape],
        "repeats": int(repeat_count),
        "warmup_overhead_ms": float(warmup_ms),
        "total_ms": _median(total_ms_values, default=float("inf")),
        "compute_ms": _median(compute_ms_values, default=float("inf")),
    }


def _resolve_replay_backend_policy(
    *,
    env_preflight: Mapping[str, Any],
    estimated_window_count: int,
) -> dict[str, Any]:
    effective_backend = str(env_preflight.get("effective_backend", "cpu"))
    gpu_speedup = _safe_float(env_preflight.get("gpu_replay_speedup", 0.0), 0.0)
    gpu_ready = bool(env_preflight.get("gpu_backend_available", False))
    estimated_count = max(int(estimated_window_count), 0)
    if effective_backend != "cuda":
        reason = "preflight_effective_backend_not_cuda"
        chosen_backend = "cpu"
    elif not gpu_ready:
        reason = "gpu_backend_not_available"
        chosen_backend = "cpu"
    elif estimated_count < int(DEFAULT_GPU_REPLAY_MIN_WINDOWS):
        reason = "estimated_windows_below_threshold"
        chosen_backend = "cpu"
    elif gpu_speedup < float(DEFAULT_GPU_REPLAY_MIN_SPEEDUP):
        reason = "gpu_speedup_below_threshold"
        chosen_backend = "cpu"
    else:
        reason = "batched_replay_not_implemented"
        chosen_backend = "cpu"
    return {
        "effective_replay_backend": str(chosen_backend),
        "gpu_replay_speedup": float(gpu_speedup),
        "gpu_replay_eligible": bool(chosen_backend == "cuda"),
        "gpu_replay_reason": str(reason),
        "estimated_window_count": int(estimated_count),
    }


def _default_model_params(
    *,
    model_name: str,
    Nh: int,
    delay_steps: Optional[int] = None,
    n_components: Optional[int] = None,
    decoder_variant: Optional[str] = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"Nh": int(Nh)}
    if str(model_name).strip().lower() == DEFAULT_TDCA_LOCAL_MODEL:
        params["delay_steps"] = int(delay_steps if delay_steps is not None else DEFAULT_TDCA_LOCAL_DELAY_STEPS[0])
        params["n_components"] = int(n_components if n_components is not None else DEFAULT_TDCA_LOCAL_N_COMPONENTS[0])
        resolved_variant = parse_tdca_decoder_variant(decoder_variant or DEFAULT_TDCA_LOCAL_DECODER_VARIANT)
        params["decoder_variant"] = str(resolved_variant)
        params["training_latency_sec"] = float(0.14 if resolved_variant == "tdca_paper_aligned" else 0.0)
    return params


def _baseline_candidate(model_name: str, *, Nh: int) -> dict[str, Any]:
    return {
        "model_name": str(model_name),
        "channel_mode": DEFAULT_TDCA_LOCAL_CHANNEL_MODE,
        "win_sec": float(DEFAULT_BASELINE_WIN_SEC),
        "model_params": _default_model_params(model_name=str(model_name), Nh=int(Nh)),
    }


def _resolve_search_plan(config: TDCALocalOptConfig) -> dict[str, Any]:
    preset = str(config.search_preset or DEFAULT_TDCA_LOCAL_SEARCH_PRESET).strip().lower()
    custom_override = bool(
        tuple(float(item) for item in config.win_candidates) != tuple(float(item) for item in DEFAULT_TDCA_LOCAL_WIN_CANDIDATES)
        or tuple(int(item) for item in config.tdca_delay_steps) != tuple(int(item) for item in DEFAULT_TDCA_LOCAL_DELAY_STEPS)
        or tuple(int(item) for item in config.tdca_n_components) != tuple(int(item) for item in DEFAULT_TDCA_LOCAL_N_COMPONENTS)
        or int(config.multi_seed_count) != int(DEFAULT_TDCA_LOCAL_REPEAT_COUNT)
    )
    if custom_override:
        return {
            "search_preset": "custom",
            "repeats": int(config.multi_seed_count),
            "candidate_grid": [
                {
                    "decoder_variant": str(decoder_variant),
                    "confidence_variant": str(confidence_variant),
                    "win_sec": float(win_sec),
                    "delay_steps": int(delay_steps),
                    "n_components": int(n_components),
                }
                for decoder_variant, confidence_variant, win_sec, delay_steps, n_components in product(
                    tuple(str(item) for item in DEFAULT_TDCA_LOCAL_DECODER_VARIANTS),
                    tuple(str(item) for item in DEFAULT_CONFIDENCE_VARIANTS),
                    tuple(float(item) for item in config.win_candidates),
                    tuple(int(item) for item in config.tdca_delay_steps),
                    tuple(int(item) for item in config.tdca_n_components),
                )
            ],
        }
    if preset == "smoke4":
        candidate_grid = [
            {
                "decoder_variant": str(decoder_variant),
                "confidence_variant": str(confidence_variant),
                "win_sec": float(win_sec),
                "delay_steps": int(delay_steps),
                "n_components": int(n_components),
            }
            for decoder_variant, win_sec, delay_steps, n_components in (
                ("tdca_like_legacy", 2.0, 2, 2),
                ("tdca_like_legacy", 2.5, 2, 2),
                ("tdca_paper_aligned", 3.0, 3, 2),
                ("tdca_paper_aligned", 3.5, 3, 2),
            )
            for confidence_variant in DEFAULT_CONFIDENCE_VARIANTS
        ]
        repeats = 1
    elif preset == "reduced13":
        candidate_grid = [
            *[
                {
                    "decoder_variant": "tdca_like_legacy",
                    "confidence_variant": str(confidence_variant),
                    "win_sec": float(win_sec),
                    "delay_steps": int(delay_steps),
                    "n_components": 2,
                }
                for confidence_variant, win_sec, delay_steps in product(DEFAULT_CONFIDENCE_VARIANTS, (2.0, 2.5, 3.0), (2, 3, 4))
            ],
            *[
                {
                    "decoder_variant": "tdca_paper_aligned",
                    "confidence_variant": str(confidence_variant),
                    "win_sec": float(win_sec),
                    "delay_steps": int(delay_steps),
                    "n_components": 2,
                }
                for confidence_variant, win_sec, delay_steps in product(DEFAULT_CONFIDENCE_VARIANTS, (3.0, 3.5), (3, 4))
            ],
        ]
        repeats = 5
    else:
        candidate_grid = [
            {
                "decoder_variant": str(decoder_variant),
                "confidence_variant": str(confidence_variant),
                "win_sec": float(win_sec),
                "delay_steps": int(delay_steps),
                "n_components": int(n_components),
            }
            for decoder_variant, confidence_variant, win_sec, delay_steps, n_components in product(
                tuple(str(item) for item in DEFAULT_TDCA_LOCAL_DECODER_VARIANTS),
                tuple(str(item) for item in DEFAULT_CONFIDENCE_VARIANTS),
                tuple(float(item) for item in DEFAULT_TDCA_LOCAL_WIN_CANDIDATES),
                tuple(int(item) for item in DEFAULT_TDCA_LOCAL_DELAY_STEPS),
                tuple(int(item) for item in DEFAULT_TDCA_LOCAL_N_COMPONENTS),
            )
        ]
        repeats = 5
    return {
        "search_preset": str(preset),
        "repeats": int(repeats),
        "candidate_grid": list(candidate_grid),
    }


def _candidate_rank_tuple(row: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in row.get("rank_key", []))


def _tdca_board_sort_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    decoder_variant = (
        row.get("decoder_variant")
        or dict(row.get("candidate", {})).get("decoder_variant")
        or DEFAULT_TDCA_LOCAL_DECODER_VARIANT
    )
    confidence_variant = (
        row.get("confidence_variant")
        or dict(row.get("candidate", {})).get("confidence_variant")
        or DEFAULT_CONFIDENCE_VARIANT
    )
    gate_valid = bool(row.get("gate_calibration_valid", True))
    return (
        0.0 if gate_valid else 1.0,
        *_candidate_rank_tuple(row),
        float(_tdca_variant_priority(str(decoder_variant))),
        float(_confidence_variant_priority(str(confidence_variant))),
    )


def _resolve_report_paths(config: TDCALocalOptConfig) -> dict[str, Path]:
    artifacts = resolve_ssvep_run_artifacts(
        task="tdca-local-opt",
        report_path=Path(config.report_path).expanduser().resolve(),
        output_profile_path=Path(config.output_profile_path).expanduser().resolve(),
        organize_report_dir=bool(config.organize_report_dir),
        report_root_dir=(
            Path(config.report_root_dir).expanduser().resolve()
            if config.report_root_dir is not None
            else None
        ),
        run_tag=make_run_tag(task="tdca-local-opt"),
    )
    return {
        "run_tag": Path(artifacts.run_dir).name,
        "report_dir": artifacts.run_dir,
        "report_json": artifacts.report_json,
        "report_md": artifacts.report_md,
        "run_config": artifacts.run_config,
        "selection_snapshot": artifacts.selection_snapshot,
        "run_log": artifacts.run_log,
        "progress_snapshot": artifacts.progress_snapshot,
        "output_profile": artifacts.output_profile,
        "profile_v2": artifacts.profile_v2,
        "figures_dir": artifacts.figures_dir,
    }


def _append_run_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {text}\n")


def _tdca_progress_label(stage: str) -> str:
    return {
        "prepare": "准备",
        "baseline_opening": "基线开场对比",
        "candidate_search": "TDCA 候选搜索",
        "decision_search": "异步决策搜索",
        "baseline_seal": "基线封板对比",
        "finalize": "保存产物",
        "complete": "完成",
    }.get(str(stage).strip().lower(), str(stage or "处理中"))


def _tdca_progress_percent(stage: str, *, run_index: int = 0, run_total: int = 0) -> int:
    ranges = {
        "prepare": (0.0, 12.0),
        "baseline_opening": (12.0, 18.0),
        "candidate_search": (18.0, 72.0),
        "decision_search": (72.0, 92.0),
        "baseline_seal": (92.0, 96.0),
        "finalize": (96.0, 99.0),
        "complete": (100.0, 100.0),
    }
    stage_name = str(stage).strip().lower()
    start, end = ranges.get(stage_name, (0.0, 0.0))
    if stage_name == "complete":
        return 100
    if run_total <= 0:
        fraction = 0.0
    else:
        fraction = min(max(float(run_index) / max(float(run_total), 1.0), 0.0), 1.0)
    return int(round(start + (end - start) * fraction))


def preflight_tdca_local_env(
    *,
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
) -> dict[str, Any]:
    requested_backend = parse_compute_backend_name(compute_backend)
    try:
        backend = resolve_compute_backend(
            requested_backend,
            gpu_device=int(gpu_device),
            precision=str(gpu_precision),
        )
        backend_error = None
    except Exception as exc:
        backend = None
        backend_error = str(exc)
    if requested_backend == "cuda" and backend is None:
        raise RuntimeError(
            f"explicit compute_backend=cuda failed preflight before search started: {backend_error}"
        )
    if backend is None:
        backend = resolve_compute_backend("cpu", gpu_device=int(gpu_device), precision=str(gpu_precision))
    transfer_benchmark: dict[str, Any]
    try:
        transfer_benchmark = backend.microbenchmark_transfer(sample_shape=(256, 8), repeats=2, use_pinned=True)
    except Exception as exc:
        transfer_benchmark = {"error": str(exc)}
    cpu_backend = resolve_compute_backend("cpu", gpu_device=int(gpu_device), precision=str(gpu_precision))
    cpu_replay_benchmark = _microbenchmark_replay_backend(cpu_backend)
    try:
        gpu_backend = resolve_compute_backend("cuda", gpu_device=int(gpu_device), precision=str(gpu_precision))
        gpu_backend_available = True
        gpu_replay_benchmark = _microbenchmark_replay_backend(gpu_backend)
        gpu_backend_error = None
    except Exception as exc:
        gpu_backend = None
        gpu_backend_available = False
        gpu_replay_benchmark = {"backend_name": "cuda", "error": str(exc)}
        gpu_backend_error = str(exc)
    cpu_total_ms = _safe_float(cpu_replay_benchmark.get("total_ms"), float("inf"))
    gpu_total_ms = _safe_float(gpu_replay_benchmark.get("total_ms"), float("inf"))
    if gpu_backend_available and np.isfinite(cpu_total_ms) and np.isfinite(gpu_total_ms) and gpu_total_ms > 1e-12:
        gpu_replay_speedup = float(cpu_total_ms / gpu_total_ms)
    else:
        gpu_replay_speedup = 0.0
    payload = {
        "requested_backend": str(requested_backend),
        "effective_backend": str(getattr(backend, "backend_name", "cpu")),
        "gpu_device": int(gpu_device),
        "gpu_precision": str(gpu_precision),
        "backend_description": dict(backend.describe()),
        "transfer_benchmark": transfer_benchmark,
        "cpu_replay_benchmark": cpu_replay_benchmark,
        "gpu_replay_benchmark": gpu_replay_benchmark,
        "gpu_backend_available": bool(gpu_backend_available),
        "gpu_backend_error": gpu_backend_error,
        "gpu_replay_speedup": float(gpu_replay_speedup),
        "preflight_error": backend_error,
        "notes": [
            "explicit cuda requests fail during preflight instead of during replay search",
            "auto is allowed to fall back to cpu",
            "gpu replay is only enabled for 256+ estimated windows and >=1.5x speedup",
        ],
    }
    return payload


def backfill_manifest_trial_roles(manifest_path: Path) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    raw = Path(path).read_text(encoding="utf-8")
    manifest = dict(json.loads(raw))
    trials = [dict(row) for row in list(manifest.get("trials", [])) if isinstance(row, Mapping)]
    changed = False
    for row in trials:
        inferred = infer_trial_role(
            label=str(row.get("label", "")),
            expected_freq=None if row.get("expected_freq") is None else float(row.get("expected_freq")),
        )
        if str(row.get("trial_role", "")).strip().lower() != str(inferred):
            row["trial_role"] = str(inferred)
            changed = True
    quality_summary = dict(manifest.get("quality_summary", {}))
    role_counts = summarize_trial_roles(trials)
    if dict(quality_summary.get("trial_role_counts", {})) != role_counts:
        quality_summary["trial_role_counts"] = role_counts
        changed = True
    manifest["trials"] = trials
    manifest["quality_summary"] = quality_summary
    if changed:
        atomic_write_text(path, json_dumps(json_safe(manifest)) + "\n")
    return {
        "manifest_path": str(path),
        "changed": bool(changed),
        "trial_count": int(len(trials)),
        "trial_role_counts": role_counts,
    }


def _build_quality_row(dataset: LoadedDataset) -> dict[str, Any]:
    quality_summary = dict(dataset.manifest.get("quality_summary", {}))
    trials = [dict(row) for row in list(dataset.manifest.get("trials", [])) if isinstance(row, Mapping)]
    role_counts = summarize_trial_roles(trials)
    return {
        "manifest_path": str(dataset.manifest_path),
        "session_id": str(dataset.session_id),
        "subject_id": str(dataset.subject_id),
        "trial_count": int(len(trials)),
        "trial_role_counts": role_counts,
        "quality_summary": quality_summary,
    }


def _load_merged_dataset(config: TDCALocalOptConfig) -> MergedLocalDataset:
    manifest_paths = tuple(
        dict.fromkeys(
            Path(path).expanduser().resolve()
            for path in (config.dataset_manifests or (config.dataset_manifest_session1,))
        )
    )
    if not manifest_paths:
        raise ValueError("at least one dataset manifest is required")

    datasets: list[LoadedDataset] = []
    merged_segments: list[tuple[TrialSpec, np.ndarray]] = []
    quality_rows: list[dict[str, Any]] = []
    subject_id = ""
    sampling_rate: Optional[int] = None
    freqs: Optional[tuple[float, float, float, float]] = None
    board_channels: Optional[tuple[int, ...]] = None
    global_trial_id = 0
    block_offset = 0

    for path in manifest_paths:
        backfill_manifest_trial_roles(path)
        dataset = load_collection_dataset(path)
        datasets.append(dataset)
        quality_rows.append(_build_quality_row(dataset))
        if subject_id and str(dataset.subject_id) != subject_id:
            raise ValueError(
                f"subject mismatch across manifests: {subject_id} vs {dataset.subject_id} ({path})"
            )
        subject_id = str(dataset.subject_id)
        if sampling_rate is None:
            sampling_rate = int(dataset.sampling_rate)
            freqs = tuple(float(item) for item in dataset.freqs)  # type: ignore[assignment]
            board_channels = tuple(int(item) for item in dataset.board_eeg_channels)
        else:
            if int(dataset.sampling_rate) != int(sampling_rate):
                raise ValueError(f"sampling_rate mismatch in {path}")
            if tuple(float(item) for item in dataset.freqs) != tuple(freqs or ()):
                raise ValueError(f"frequency set mismatch in {path}")
            if tuple(int(item) for item in dataset.board_eeg_channels) != tuple(board_channels or ()):
                raise ValueError(f"board_eeg_channels mismatch in {path}")
        for trial, segment in dataset.trial_segments:
            merged_segments.append(
                (
                    TrialSpec(
                        label=str(trial.label),
                        expected_freq=None if trial.expected_freq is None else float(trial.expected_freq),
                        trial_id=int(global_trial_id),
                        block_index=int(block_offset + max(int(trial.block_index), 0)),
                    ),
                    np.ascontiguousarray(np.asarray(segment, dtype=np.float64)),
                )
            )
            global_trial_id += 1
        block_offset += len(dataset.trial_segments) + 1

    trial_role_counts = summarize_trial_roles(
        [
            {
                "label": str(trial.label),
                "expected_freq": None if trial.expected_freq is None else float(trial.expected_freq),
                "trial_role": infer_trial_role(
                    label=str(trial.label),
                    expected_freq=None if trial.expected_freq is None else float(trial.expected_freq),
                ),
            }
            for trial, _segment in merged_segments
        ]
    )
    if sampling_rate is None or freqs is None or board_channels is None:
        raise RuntimeError("merged dataset resolution failed")
    return MergedLocalDataset(
        manifest_paths=manifest_paths,
        datasets=tuple(datasets),
        trial_segments=tuple(merged_segments),
        sampling_rate=int(sampling_rate),
        freqs=tuple(freqs),  # type: ignore[arg-type]
        board_eeg_channels=tuple(board_channels),
        subject_id=str(subject_id),
        session_ids=tuple(str(dataset.session_id) for dataset in datasets),
        trial_role_counts=trial_role_counts,
        quality_rows=tuple(dict(row) for row in quality_rows),
    )


def _allocate_split_counts(count: int) -> tuple[int, int, int]:
    total = max(int(count), 0)
    if total <= 1:
        return total, 0, 0
    if total == 2:
        return 1, 0, 1
    if total == 3:
        return 1, 1, 1
    train_count = max(1, int(round(total * 0.6)))
    gate_count = max(1, int(round(total * 0.2)))
    holdout_count = total - train_count - gate_count
    if holdout_count <= 0:
        holdout_count = 1
        if gate_count > 1:
            gate_count -= 1
        else:
            train_count = max(1, train_count - 1)
    return train_count, gate_count, holdout_count


def build_repeated_group_splits(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    repeats: int,
    seed: int,
) -> list[RepeatedGroupSplit]:
    indexed = list(enumerate(trial_segments))
    grouped: dict[str, list[int]] = {}
    for index, (trial, _segment) in indexed:
        role = infer_trial_role(
            label=str(trial.label),
            expected_freq=None if trial.expected_freq is None else float(trial.expected_freq),
        )
        group_key = f"{role}|{str(trial.label).strip().lower()}"
        grouped.setdefault(group_key, []).append(int(index))

    output: list[RepeatedGroupSplit] = []
    for repeat_index in range(max(int(repeats), 1)):
        rng = np.random.default_rng(int(seed) + repeat_index * 1009)
        train_indices: list[int] = []
        gate_indices: list[int] = []
        holdout_indices: list[int] = []
        for group_key in sorted(grouped):
            values = list(grouped[group_key])
            if len(values) > 1:
                values = [int(item) for item in rng.permutation(values).tolist()]
            train_count, gate_count, holdout_count = _allocate_split_counts(len(values))
            train_indices.extend(values[:train_count])
            gate_indices.extend(values[train_count : train_count + gate_count])
            holdout_indices.extend(values[train_count + gate_count : train_count + gate_count + holdout_count])
        if not holdout_indices and train_indices:
            holdout_indices.append(train_indices.pop())
        fingerprint = _fingerprint_for_split(
            train_indices=train_indices,
            gate_indices=gate_indices,
            holdout_indices=holdout_indices,
        )
        output.append(
            RepeatedGroupSplit(
                repeat_index=int(repeat_index),
                train_indices=tuple(sorted(int(item) for item in train_indices)),
                gate_indices=tuple(sorted(int(item) for item in gate_indices)),
                holdout_indices=tuple(sorted(int(item) for item in holdout_indices)),
                fingerprint=str(fingerprint),
            )
        )
    return output


def _subset_segments(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    indices: Sequence[int],
) -> list[tuple[TrialSpec, np.ndarray]]:
    keep = {int(item) for item in indices}
    return [trial_segments[index] for index in range(len(trial_segments)) if index in keep]


def _feature_stats(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    if not rows:
        means = {str(name): 0.0 for name in MODEL_FEATURE_NAMES}
        stds = {str(name): 1.0 for name in MODEL_FEATURE_NAMES}
        return means, stds
    matrix = np.asarray(
        [[_safe_float(row.get(name, 0.0), 0.0) for name in MODEL_FEATURE_NAMES] for row in rows],
        dtype=float,
    )
    mean_values = np.mean(matrix, axis=0)
    std_values = np.maximum(np.std(matrix, axis=0), 1e-6)
    return (
        {str(name): float(value) for name, value in zip(MODEL_FEATURE_NAMES, mean_values)},
        {str(name): float(value) for name, value in zip(MODEL_FEATURE_NAMES, std_values)},
    )


def _attach_history_features(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    first_seen: dict[tuple[str, int], int] = {}
    for index, row in enumerate(rows):
        key = (str(row.get("label", "")), _safe_int(row.get("trial_id", -1), -1))
        grouped.setdefault(key, []).append(dict(row))
        first_seen.setdefault(key, int(index))
    ordered_keys = sorted(grouped.keys(), key=lambda item: int(first_seen.get(item, 0)))
    output: list[dict[str, Any]] = []
    for key in ordered_keys:
        history = RollingFeatureHistory(window_size=4)
        history.reset()
        trial_rows = sorted(grouped[key], key=lambda item: _safe_int(item.get("window_index", 0), 0))
        for row in trial_rows:
            raw_pred_freq = row.get("pred_freq")
            pred_freq = None if raw_pred_freq is None else _safe_float(raw_pred_freq, float("nan"))
            if pred_freq is not None and not np.isfinite(pred_freq):
                pred_freq = None
            hist = history.update(
                pred_freq=pred_freq,
                margin=_safe_float(row.get("margin", 0.0), 0.0),
                ratio=_safe_float(row.get("ratio", 1.0), 1.0),
            )
            enriched = dict(row)
            enriched["pred_freq"] = pred_freq
            enriched["consistency"] = float(hist.get("consistency", 0.0))
            enriched["margin_mean_k"] = float(hist.get("margin_mean_k", enriched.get("margin", 0.0)))
            enriched["ratio_mean_k"] = float(hist.get("ratio_mean_k", enriched.get("ratio", 1.0)))
            output.append(enriched)
    return output


def _score_rows_with_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    gate: PerFrequencyLogRegGate,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row_raw in rows:
        row = dict(row_raw)
        pred_freq = row.get("pred_freq")
        pred_freq_value = None if pred_freq is None else _safe_float(pred_freq, float("nan"))
        if pred_freq_value is None or not np.isfinite(pred_freq_value):
            pred_freq_value = None
        gate_out = gate.predict(row, pred_freq_value)
        row["pred_freq"] = pred_freq_value
        row["p_control"] = float(gate_out.p_control)
        row["gate_score"] = float(gate_out.gate_score)
        row["control_log_lr"] = float(gate_out.gate_score)
        scored.append(row)
    return scored


def _score_rows_with_correctness(
    rows: Sequence[Mapping[str, Any]],
    *,
    calibrator: CorrectnessCalibrator,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    calibrator_payload = dict(calibrator.to_payload())
    calibrator_model = dict(calibrator_payload.get("model", {}) or {})
    variant = str(calibrator_model.get("variant", DEFAULT_CONFIDENCE_VARIANT) or DEFAULT_CONFIDENCE_VARIANT)
    for row_raw in rows:
        row = dict(row_raw)
        correctness = calibrator.predict(row)
        row["p_correct"] = float(correctness["p_correct"])
        row["correctness_logit"] = float(correctness["correctness_logit"])
        row["confidence_variant"] = str(variant)
        if str(variant) == BAYESIAN_GAP_GMM:
            row["p_correct_bayes"] = float(correctness["p_correct"])
            row["correctness_logit_bayes"] = float(correctness["correctness_logit"])
        else:
            row["p_correct_logistic"] = float(correctness["p_correct"])
            row["correctness_logit_logistic"] = float(correctness["correctness_logit"])
        scored.append(row)
    return scored


def _resolve_oof_group_key(rows: Sequence[Mapping[str, Any]]) -> str:
    block_values = {
        _safe_int(row.get("block_index", -1), -1)
        for row in rows
        if row.get("block_index") is not None
    }
    valid_blocks = {value for value in block_values if value >= 0}
    return "block_index" if len(valid_blocks) >= 3 else "trial_id"


def _group_rows_for_oof(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
) -> dict[int, list[tuple[int, dict[str, Any]]]]:
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        group_value = _safe_int(row.get(group_key, index), index)
        grouped.setdefault(int(group_value), []).append((int(index), dict(row)))
    return grouped


def _build_oof_train_scored_rows(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    freqs: Sequence[float],
    fit_config: LogRegFitConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_key = _resolve_oof_group_key(train_rows)
    grouped = _group_rows_for_oof(train_rows, group_key=group_key)
    if len(grouped) < 2 and group_key != "trial_id":
        group_key = "trial_id"
        grouped = _group_rows_for_oof(train_rows, group_key=group_key)
    if len(grouped) < 2:
        fallback_gate = PerFrequencyLogRegGate()
        fallback_gate.fit(rows=train_rows, freqs=freqs, fit_config=fit_config)
        return (
            _score_rows_with_gate(train_rows, gate=fallback_gate),
            {
                "confidence_training_scheme": str(DEFAULT_CONFIDENCE_TRAINING_SCHEME),
                "oof_group_key": str(group_key),
                "oof_group_count": int(len(grouped)),
                "oof_fold_count": 1,
                "used_oof": False,
                "fallback_reason": "insufficient_groups_for_oof",
            },
        )

    scored_by_index: dict[int, dict[str, Any]] = {}
    ordered_group_ids = sorted(grouped.keys())
    for heldout_group in ordered_group_ids:
        heldout_entries = list(grouped.get(int(heldout_group), []))
        fold_train_rows = [
            dict(row)
            for group_id, entries in grouped.items()
            if int(group_id) != int(heldout_group)
            for _, row in entries
        ]
        if not fold_train_rows:
            continue
        gate_model = PerFrequencyLogRegGate()
        gate_model.fit(rows=fold_train_rows, freqs=freqs, fit_config=fit_config)
        heldout_rows = [dict(row) for _, row in heldout_entries]
        scored_rows = _score_rows_with_gate(heldout_rows, gate=gate_model)
        for (row_index, _), scored_row in zip(heldout_entries, scored_rows):
            scored_by_index[int(row_index)] = dict(scored_row)
    ordered_rows = [
        dict(scored_by_index[index])
        for index in sorted(scored_by_index.keys())
    ]
    return (
        ordered_rows,
        {
            "confidence_training_scheme": str(DEFAULT_CONFIDENCE_TRAINING_SCHEME),
            "oof_group_key": str(group_key),
            "oof_group_count": int(len(grouped)),
            "oof_fold_count": int(len(grouped)),
            "used_oof": True,
            "fallback_reason": "",
        },
    )


def _tag_tune_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    origin: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["tune_origin"] = str(origin)
        output.append(enriched)
    return output


def _tune_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    freqs: Sequence[float],
) -> dict[str, Any]:
    control_trials_by_freq: dict[str, set[int]] = {f"{float(freq):g}": set() for freq in freqs}
    idle_trials: set[int] = set()
    positive_windows = 0
    negative_windows = 0
    positive_trials: set[int] = set()
    negative_trials: set[int] = set()
    tune_origin_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        origin = str(row.get("tune_origin", "") or "")
        if origin:
            tune_origin_counts[origin] = int(tune_origin_counts.get(origin, 0) or 0) + 1
        trial_id = _safe_int(row.get("trial_id", index), index)
        role = resolve_trial_role(row)
        expected = row.get("expected_freq")
        expected_freq = None if expected is None else _safe_float(expected, float("nan"))
        if role == "control" and expected_freq is not None and np.isfinite(expected_freq):
            key = f"{float(expected_freq):g}"
            control_trials_by_freq.setdefault(key, set()).add(int(trial_id))
        else:
            idle_trials.add(int(trial_id))
        pred_freq = row.get("pred_freq")
        pred_freq_value = None if pred_freq is None else _safe_float(pred_freq, float("nan"))
        positive = bool(
            role == "control"
            and pred_freq_value is not None
            and expected_freq is not None
            and np.isfinite(pred_freq_value)
            and np.isfinite(expected_freq)
            and abs(float(pred_freq_value) - float(expected_freq)) <= 1e-8
        )
        if positive:
            positive_windows += 1
            positive_trials.add(int(trial_id))
        else:
            negative_windows += 1
            negative_trials.add(int(trial_id))
    control_trial_counts = {key: int(len(values)) for key, values in control_trials_by_freq.items()}
    min_control_trials_by_freq = min(control_trial_counts.values(), default=0)
    idle_trial_count = int(len(idle_trials))
    valid = bool(
        min_control_trials_by_freq >= int(DEFAULT_TUNE_MIN_CONTROL_TRIALS_PER_FREQ)
        and idle_trial_count >= int(DEFAULT_TUNE_MIN_IDLE_TRIALS)
    )
    invalid_reasons: list[str] = []
    if min_control_trials_by_freq < int(DEFAULT_TUNE_MIN_CONTROL_TRIALS_PER_FREQ):
        invalid_reasons.append("insufficient_control_trials_per_freq")
    if idle_trial_count < int(DEFAULT_TUNE_MIN_IDLE_TRIALS):
        invalid_reasons.append("insufficient_idle_trials")
    return {
        "rows_total": int(len(rows)),
        "control_trials_by_freq": control_trial_counts,
        "idle_trial_count": int(idle_trial_count),
        "positive_windows": int(positive_windows),
        "negative_windows": int(negative_windows),
        "positive_trials": int(len(positive_trials)),
        "negative_trials": int(len(negative_trials)),
        "tune_origin_counts": {str(key): int(value) for key, value in tune_origin_counts.items()},
        "min_control_trials_by_freq": int(min_control_trials_by_freq),
        "valid": bool(valid),
        "invalid_reasons": list(invalid_reasons),
    }


def _measure_decoder_inference_ms(
    decoder: Any,
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    max_windows: int = 64,
) -> float:
    latencies_ms: list[float] = []
    for _trial, segment in trial_segments:
        if len(latencies_ms) >= int(max_windows):
            break
        matrix = np.asarray(segment, dtype=float)
        if matrix.shape[0] < int(decoder.win_samples):
            continue
        windows = extract_window_batch(
            matrix,
            win_samples=int(decoder.win_samples),
            step_samples=int(decoder.step_samples),
        )
        for window in windows:
            t0 = time.perf_counter()
            decoder.analyze_window(np.asarray(window, dtype=np.float64))
            t1 = time.perf_counter()
            latencies_ms.append(float((t1 - t0) * 1000.0))
            if len(latencies_ms) >= int(max_windows):
                break
    return _median(latencies_ms, default=float("inf"))


def _gate_score_partitions(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    freq: float,
) -> tuple[np.ndarray, np.ndarray]:
    control: list[float] = []
    idle: list[float] = []
    freq_value = float(freq)
    for row in scored_rows:
        pred_freq = row.get("pred_freq")
        if pred_freq is None or abs(_safe_float(pred_freq, float("nan")) - freq_value) > 1e-8:
            continue
        expected = row.get("expected_freq")
        role = resolve_trial_role(row)
        score = _safe_float(row.get("gate_score", 0.0), 0.0)
        if expected is not None and role == "control" and abs(_safe_float(expected, float("nan")) - freq_value) <= 1e-8:
            control.append(float(score))
        else:
            idle.append(float(score))
    return np.asarray(control, dtype=float), np.asarray(idle, dtype=float)


def _control_trial_counts_by_freq(
    rows: Sequence[Mapping[str, Any]],
    *,
    freqs: Sequence[float],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for freq in freqs:
        freq_value = float(freq)
        trial_ids = {
            _safe_int(row.get("trial_id", -1), -1)
            for row in rows
            if resolve_trial_role(row) == "control"
            and row.get("expected_freq") is not None
            and abs(_safe_float(row.get("expected_freq"), float("nan")) - freq_value) <= 1e-8
        }
        counts[_freq_label(freq_value)] = int(len({item for item in trial_ids if item >= 0}))
    return counts


def _gate_calibration_summary(
    *,
    scored_rows: Sequence[Mapping[str, Any]],
    freqs: Sequence[float],
    calibrator_summary: Mapping[str, Any],
) -> dict[str, Any]:
    control_trials_by_freq = _control_trial_counts_by_freq(scored_rows, freqs=freqs)
    idle_trial_ids = {
        _safe_int(row.get("trial_id", -1), -1)
        for row in scored_rows
        if resolve_trial_role(row) != "control"
    }
    positive_windows = int(calibrator_summary.get("positive_windows", 0) or 0)
    negative_windows = int(calibrator_summary.get("negative_windows", 0) or 0)
    min_control_trials = min((int(value) for value in control_trials_by_freq.values()), default=0)
    idle_trial_count = int(len({item for item in idle_trial_ids if item >= 0}))
    invalid_reasons: list[str] = []
    if min_control_trials <= 0:
        invalid_reasons.append("missing_control_trial_for_some_freq")
    if idle_trial_count <= 0:
        invalid_reasons.append("missing_idle_trials")
    if positive_windows < int(DEFAULT_CORRECTNESS_CALIBRATOR_CONFIG.min_positive_windows):
        invalid_reasons.append("positive_windows_below_min")
    if negative_windows < int(DEFAULT_CORRECTNESS_CALIBRATOR_CONFIG.min_negative_windows):
        invalid_reasons.append("negative_windows_below_min")
    if not bool(calibrator_summary.get("valid", False)):
        invalid_reasons.append("correctness_calibrator_invalid")
    return {
        "positive_windows": int(positive_windows),
        "negative_windows": int(negative_windows),
        "positive_trials": int(calibrator_summary.get("positive_trials", 0) or 0),
        "negative_trials": int(calibrator_summary.get("negative_trials", 0) or 0),
        "sample_weight_mode": str(calibrator_summary.get("sample_weight_mode", "")),
        "confidence_training_scheme": str(
            calibrator_summary.get("confidence_training_scheme", DEFAULT_CONFIDENCE_TRAINING_SCHEME)
        ),
        "oof_group_key": str(calibrator_summary.get("oof_group_key", "")),
        "oof_group_count": int(calibrator_summary.get("oof_group_count", 0) or 0),
        "control_trials_by_freq": dict(control_trials_by_freq),
        "idle_trial_count": int(idle_trial_count),
        "min_control_trials_by_freq": int(min_control_trials),
        "brier_score": calibrator_summary.get("brier_score"),
        "auc_roc": calibrator_summary.get("auc_roc"),
        "calibrator_valid": bool(calibrator_summary.get("valid", False)),
        "gate_calibration_valid": not invalid_reasons,
        "invalid_reasons": list(dict.fromkeys(str(item) for item in invalid_reasons)),
    }


def _select_enter_exit_logit(
    *,
    control_scores: np.ndarray,
    idle_scores: np.ndarray,
    enter_fallback: Optional[float],
    exit_fallback: Optional[float],
) -> tuple[float, float, dict[str, Any]]:
    all_scores = np.concatenate([control_scores, idle_scores]) if (control_scores.size or idle_scores.size) else np.asarray([], dtype=float)

    def _bounded_fallback(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        candidate = _safe_float(value, float("nan"))
        if not np.isfinite(candidate):
            return None
        if not all_scores.size:
            return float(candidate)
        min_score = float(np.min(all_scores))
        max_score = float(np.max(all_scores))
        if candidate < min_score or candidate > max_score:
            return None
        return float(candidate)

    bounded_enter_fallback = _bounded_fallback(enter_fallback)
    bounded_exit_fallback = _bounded_fallback(exit_fallback)
    enter_candidates = sorted(
        {
            *(_quantile_candidates(control_scores, (0.05, 0.10, 0.20, 0.30, 0.40), floor=-1_000_000.0)),
            *(_quantile_candidates(idle_scores, (0.80, 0.90, 0.95, 0.98), floor=-1_000_000.0)),
            *(() if bounded_enter_fallback is None else (float(bounded_enter_fallback),)),
        }
    )
    exit_candidates = sorted(
        {
            *(_quantile_candidates(control_scores, (0.02, 0.05, 0.10, 0.20, 0.30), floor=-1_000_000.0)),
            *(_quantile_candidates(idle_scores, (0.50, 0.60, 0.70, 0.80, 0.90), floor=-1_000_000.0)),
            *(() if bounded_exit_fallback is None else (float(bounded_exit_fallback),)),
        }
    )
    if not enter_candidates:
        enter_candidates = [float(enter_fallback if enter_fallback is not None else 0.0)]
    if not exit_candidates:
        exit_candidates = [float(exit_fallback if exit_fallback is not None else enter_candidates[0])]
    best_enter = float(enter_candidates[0])
    best_exit = float(exit_candidates[0])
    enter_objective: Optional[tuple[float, float, float]] = None
    enter_rows: list[dict[str, float]] = []
    for candidate in enter_candidates:
        idle_fp = float(np.mean(idle_scores >= candidate)) if idle_scores.size else 0.0
        control_recall = float(np.mean(control_scores >= candidate)) if control_scores.size else 0.0
        enter_rows.append(
            {
                "candidate": float(candidate),
                "idle_fp": float(idle_fp),
                "control_recall": float(control_recall),
            }
        )
    positive_enter_rows = [dict(item) for item in enter_rows if float(item.get("control_recall", 0.0)) > 0.0]
    enter_selection_rows = positive_enter_rows or enter_rows
    for item in enter_selection_rows:
        objective = (
            float(item.get("idle_fp", 0.0)),
            -float(item.get("control_recall", 0.0)),
            float(item.get("candidate", 0.0)),
        )
        if enter_objective is None or objective < enter_objective:
            enter_objective = objective
            best_enter = float(item.get("candidate", 0.0))
    exit_objective: Optional[tuple[float, float, float]] = None
    exit_rows: list[dict[str, float]] = []
    for candidate in exit_candidates:
        control_drop_rate = float(np.mean(control_scores < candidate)) if control_scores.size else 0.0
        idle_clear_rate = float(np.mean(idle_scores < candidate)) if idle_scores.size else 0.0
        exit_rows.append(
            {
                "candidate": float(candidate),
                "control_drop_rate": float(control_drop_rate),
                "idle_clear_rate": float(idle_clear_rate),
            }
        )
    positive_exit_rows = [dict(item) for item in exit_rows if float(item.get("idle_clear_rate", 0.0)) > 0.0]
    exit_selection_rows = positive_exit_rows or exit_rows
    for item in exit_selection_rows:
        objective = (
            float(item.get("control_drop_rate", 0.0)),
            -float(item.get("idle_clear_rate", 0.0)),
            float(item.get("candidate", 0.0)),
        )
        if exit_objective is None or objective < exit_objective:
            exit_objective = objective
            best_exit = float(item.get("candidate", 0.0))
    return (
        float(best_enter),
        float(best_exit),
        {
            "enter_candidates": [float(item) for item in enter_candidates],
            "exit_candidates": [float(item) for item in exit_candidates],
            "enter_rows": [dict(item) for item in enter_rows],
            "exit_rows": [dict(item) for item in exit_rows],
            "enter_selection_mode": "positive_control_recall" if positive_enter_rows else "all_candidates",
            "exit_selection_mode": "positive_idle_clear" if positive_exit_rows else "all_candidates",
            "enter_objective": None if enter_objective is None else [float(item) for item in enter_objective],
            "exit_objective": None if exit_objective is None else [float(item) for item in exit_objective],
        },
    )


def _build_frequency_specific_thresholds(
    *,
    base_profile: ThresholdProfile,
    freqs: Sequence[float],
    gate_calibration_summary: Mapping[str, Any],
    enter_p_th: float,
    exit_p_th: float,
    min_enter_windows: int,
    min_exit_windows: int,
    min_switch_windows: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], bool]:
    output: dict[str, dict[str, Any]] = {}
    board: list[dict[str, Any]] = []
    base_payload = dict(base_profile.frequency_specific_thresholds or {})
    control_trials_by_freq = dict(gate_calibration_summary.get("control_trials_by_freq", {}) or {})
    idle_trial_count = int(gate_calibration_summary.get("idle_trial_count", 0) or 0)
    global_valid = bool(gate_calibration_summary.get("gate_calibration_valid", False))
    enter_logit = _p_to_logit(enter_p_th, 0.65)
    exit_logit = _p_to_logit(exit_p_th, 0.30)
    for freq in freqs:
        freq_key = _freq_label(float(freq))
        payload = dict(base_payload.get(freq_key, {}))
        control_trial_count = int(control_trials_by_freq.get(freq_key, 0) or 0)
        freq_valid = bool(global_valid and control_trial_count > 0 and idle_trial_count > 0)
        invalid_reason = "" if freq_valid else "gate_calibration_invalid"
        payload.setdefault("enter_score_th", float(base_profile.enter_score_th))
        payload.setdefault("enter_ratio_th", float(base_profile.enter_ratio_th))
        payload.setdefault("enter_margin_th", float(base_profile.enter_margin_th))
        payload.setdefault("exit_score_th", float(base_profile.exit_score_th))
        payload.setdefault("exit_ratio_th", float(base_profile.exit_ratio_th))
        payload.setdefault("switch_enter_score_th", float(base_profile.switch_enter_score_th or base_profile.enter_score_th))
        payload.setdefault("switch_enter_ratio_th", float(base_profile.switch_enter_ratio_th or base_profile.enter_ratio_th))
        payload.setdefault("switch_enter_margin_th", float(base_profile.switch_enter_margin_th or base_profile.enter_margin_th))
        payload["enter_p_th"] = float(enter_p_th)
        payload["exit_p_th"] = float(exit_p_th)
        payload["enter_log_lr_th"] = float(enter_logit)
        payload["exit_log_lr_th"] = float(exit_logit)
        payload["min_enter_windows"] = int(min_enter_windows)
        payload["min_exit_windows"] = int(min_exit_windows)
        payload["min_switch_windows"] = int(min_switch_windows)
        output[freq_key] = payload
        board.append(
            {
                "freq": float(freq),
                "enter_p_th": float(enter_p_th),
                "exit_p_th": float(exit_p_th),
                "enter_logit_th": float(enter_logit),
                "exit_logit_th": float(exit_logit),
                "min_switch_windows": int(min_switch_windows),
                "n_control_rows": int(control_trial_count),
                "n_idle_rows": int(idle_trial_count),
                "n_control_trials": int(control_trial_count),
                "n_idle_trials": int(idle_trial_count),
                "calibration_valid": bool(freq_valid),
                "invalid_reason": str(invalid_reason),
                "detail": {
                    "mode": "fixed_probability_grid",
                    "gate_calibration_summary": dict(gate_calibration_summary),
                },
            }
        )
    return output, board, bool(global_valid)


def _search_gate_profile(
    *,
    base_profile: ThresholdProfile,
    scored_search_rows: Sequence[Mapping[str, Any]],
    freqs: Sequence[float],
    inference_ms: float,
    decision_time_mode: str,
    async_decision_time_mode: str,
    gate_calibration_summary: Mapping[str, Any],
    confidence_variant: str,
) -> tuple[ThresholdProfile, list[dict[str, Any]], list[dict[str, Any]], bool]:
    best_profile = base_profile
    best_objective: Optional[tuple[float, float, float, float, float, float]] = None
    gate_board: list[dict[str, Any]] = []
    threshold_board: list[dict[str, Any]] = []
    any_valid = bool(gate_calibration_summary.get("gate_calibration_valid", False))
    for enter_p_th, exit_p_th, min_enter_windows, min_exit_windows, min_switch_windows in product(
        DEFAULT_GATE_ENTER_P_GRID,
        DEFAULT_GATE_EXIT_P_GRID,
        (1, 2),
        (1, 2),
        (1, 2),
    ):
        per_freq_thresholds, freq_board, gate_calibration_valid = _build_frequency_specific_thresholds(
            base_profile=base_profile,
            freqs=freqs,
            gate_calibration_summary=gate_calibration_summary,
            enter_p_th=float(enter_p_th),
            exit_p_th=float(exit_p_th),
            min_enter_windows=int(min_enter_windows),
            min_exit_windows=int(min_exit_windows),
            min_switch_windows=int(min_switch_windows),
        )
        any_valid = bool(any_valid or gate_calibration_valid)
        candidate_profile = replace(
            base_profile,
            min_enter_windows=int(min_enter_windows),
            min_exit_windows=int(min_exit_windows),
            min_switch_windows=int(min_switch_windows),
            enter_p_th=float(enter_p_th),
            exit_p_th=float(exit_p_th),
            enter_log_lr_th=float(_p_to_logit(enter_p_th, 0.65)),
            exit_log_lr_th=float(_p_to_logit(exit_p_th, 0.30)),
            frequency_specific_thresholds=per_freq_thresholds,
            confidence_variant=str(confidence_variant),
            control_state_mode=parse_control_state_mode("frequency-specific-logistic"),
        )
        if gate_calibration_valid:
            metrics = dict(
                _evaluate_structured_rows(
                    scored_rows=scored_search_rows,
                    profile=candidate_profile,
                    freqs=freqs,
                    decision_params=_default_decision_params(),
                    inference_ms=float(inference_ms),
                    decision_time_mode=str(decision_time_mode),
                    async_decision_time_mode=str(async_decision_time_mode),
                ).get("async_metrics", {})
            )
        else:
            metrics = {
                "idle_fp_per_min": float("inf"),
                "control_recall": 0.0,
                "control_recall_at_2s": 0.0,
                "control_recall_at_3s": 0.0,
                "switch_detect_rate": 0.0,
                "switch_detect_rate_at_2.8s": 0.0,
                "release_detect_rate": 0.0,
                "switch_latency_s": float("inf"),
                "release_latency_s": float("inf"),
                "detection_latency_s": float("inf"),
                "inference_ms": float(inference_ms),
            }
        objective = _rank_metrics_key(metrics)
        gate_board.append(
            {
                "enter_p_th": float(enter_p_th),
                "exit_p_th": float(exit_p_th),
                "min_enter_windows": int(min_enter_windows),
                "min_exit_windows": int(min_exit_windows),
                "min_switch_windows": int(min_switch_windows),
                "gate_calibration_valid": bool(gate_calibration_valid),
                "min_gate_control_rows": int(gate_calibration_summary.get("min_control_trials_by_freq", 0) or 0),
                "min_gate_idle_rows": int(gate_calibration_summary.get("idle_trial_count", 0) or 0),
                "positive_windows": int(gate_calibration_summary.get("positive_windows", 0) or 0),
                "negative_windows": int(gate_calibration_summary.get("negative_windows", 0) or 0),
                "metrics": dict(metrics),
                "rank_key": [float(item) for item in objective],
            }
        )
        for item in freq_board:
            threshold_board.append(
                {
                    "min_enter_windows": int(min_enter_windows),
                    "min_exit_windows": int(min_exit_windows),
                    "min_switch_windows": int(min_switch_windows),
                    **dict(item),
                }
            )
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_profile = candidate_profile
    gate_board.sort(key=lambda item: tuple(float(value) for value in item.get("rank_key", [])))
    return best_profile, gate_board, threshold_board, any_valid


def _make_decision_params_key(params: Mapping[str, Any]) -> str:
    return "|".join(
        [
            f"candidate_min_windows={int(params.get('candidate_min_windows', 0))}",
            f"armed_min_windows={int(params.get('armed_min_windows', 0))}",
            f"lambda_decay={float(params.get('lambda_decay', 0.0)):.2f}",
            f"upper_commit_th={float(params.get('upper_commit_th', 0.0)):.1f}",
            f"lower_idle_th={float(params.get('lower_idle_th', 0.0)):.1f}",
            f"refractory_sec={float(params.get('refractory_sec', 0.0)):.1f}",
        ]
    )


def _make_decision_engine_config(
    *,
    profile: ThresholdProfile,
    decision_params: Mapping[str, Any],
) -> DecisionEngineConfig:
    enter_gate_logit = float(
        _p_to_logit(profile.enter_p_th if profile.enter_p_th is not None else 0.65, 0.65)
    )
    exit_gate_logit = float(
        _p_to_logit(profile.exit_p_th if profile.exit_p_th is not None else 0.30, 0.30)
    )
    return DecisionEngineConfig(
        evidence=EvidenceAccumulatorConfig(
            lambda_decay=float(decision_params.get("lambda_decay", 0.85)),
            beta_consistency=float(DEFAULT_DECISION_BETA_CONSISTENCY),
            upper_commit_th=float(decision_params.get("upper_commit_th", 2.2)),
            lower_idle_th=float(decision_params.get("lower_idle_th", 0.4)),
        ),
        state=StateMachineConfig(
            candidate_min_windows=int(decision_params.get("candidate_min_windows", 1)),
            armed_min_windows=int(decision_params.get("armed_min_windows", 2)),
            commit_consistency_th=float(DEFAULT_DECISION_COMMIT_CONSISTENCY_TH),
            enter_gate_th=0.0,
            exit_gate_th=float(exit_gate_logit - enter_gate_logit),
            refractory_sec=float(decision_params.get("refractory_sec", 0.8)),
        ),
    )


def _evaluate_structured_rows(
    *,
    scored_rows: Sequence[Mapping[str, Any]],
    profile: ThresholdProfile,
    freqs: Sequence[float],
    decision_params: Mapping[str, Any],
    inference_ms: float,
    decision_time_mode: str,
    async_decision_time_mode: str,
) -> dict[str, Any]:
    rows = [dict(row) for row in scored_rows]
    if not rows:
        empty_metrics = {
            "idle_fp_per_min": 0.0,
            "control_recall": 0.0,
            "control_recall_at_2s": 0.0,
            "control_recall_at_3s": 0.0,
            "switch_detect_rate": 0.0,
            "switch_detect_rate_at_2.8s": 0.0,
            "release_detect_rate": 0.0,
            "switch_latency_s": float("inf"),
            "release_latency_s": float("inf"),
            "detection_latency_s": float("inf"),
            "inference_ms": float(inference_ms),
        }
        return {
            "async_metrics": empty_metrics,
            "metrics_4class": {},
            "metrics_2class": {},
            "metrics_5class": None,
            "paper_lens_metrics_4class": {},
            "paper_lens_metrics_2class": {},
            "async_lens_metrics_4class": {},
            "async_lens_metrics_2class": {},
            "decision_params": dict(decision_params),
            "trial_events": [],
        }

    gate = GateReplayState(profile)
    history = RollingFeatureHistory(window_size=4)
    engine = DecisionEngine(_make_decision_engine_config(profile=profile, decision_params=decision_params))
    gate.reset()
    history.reset()
    engine.reset()

    grouped_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    first_seen: dict[tuple[str, int], int] = {}
    for index, row in enumerate(rows):
        key = (str(row.get("label", "")), _safe_int(row.get("trial_id", -1), -1))
        grouped_rows.setdefault(key, []).append(dict(row))
        first_seen.setdefault(key, int(index))
    ordered_trials = sorted(grouped_rows.keys(), key=lambda item: int(first_seen.get(item, 0)))

    control_trials = 0
    control_detected_trials = 0
    control_detected_trials_at_2s = 0
    control_detected_trials_at_3s = 0
    idle_selected_events = 0
    idle_duration_sec = 0.0
    switch_trials = 0
    switch_detected_trials = 0
    switch_detected_trials_at_2p8s = 0
    release_trials = 0
    release_detected_trials = 0
    detection_latencies: list[float] = []
    switch_latencies: list[float] = []
    release_latencies: list[float] = []
    y4_true: list[str] = []
    y4_pred: list[str] = []
    y2_true: list[str] = []
    y2_pred: list[str] = []
    times4: list[float] = []
    times2: list[float] = []
    last_control_freq: Optional[float] = None
    previous_trial_expected: Optional[float] = None
    trial_events: list[dict[str, Any]] = []
    selected_active_prev = False
    stream_index = 0

    for key in ordered_trials:
        trial_rows = sorted(grouped_rows[key], key=lambda item: _safe_int(item.get("window_index", 0), 0))
        if not trial_rows:
            continue
        label_text = str(trial_rows[0].get("label", ""))
        expected = trial_rows[0].get("expected_freq")
        expected_freq = None if expected is None else _safe_float(expected, float("nan"))
        if expected_freq is not None and not np.isfinite(expected_freq):
            expected_freq = None
        trial_duration = float(
            profile.win_sec
            + max(_safe_int(trial_rows[-1].get("window_index", len(trial_rows) - 1), len(trial_rows) - 1), 0)
            * float(profile.step_sec)
        )
        penalty_latency = float(trial_duration + profile.win_sec)
        first_correct_latency: Optional[float] = None
        first_any_latency: Optional[float] = None
        first_any_freq: Optional[float] = None
        first_release_latency: Optional[float] = None
        tracked_freq_first_seen: Optional[float] = None
        commit_freq_first_seen: Optional[float] = None
        first_gate_pass_latency: Optional[float] = None
        last_pred_freq: Optional[float] = None
        idle_commit_seen = False
        raw_correct_seen = False
        gate_pass_correct_seen = False
        gate_switch_count = 0
        gate_event = "hold"
        previous_gate_open_freq: Optional[float] = None
        max_p_correct = 0.0
        max_decision_evidence: Optional[float] = None

        for row in trial_rows:
            raw_pred_freq = row.get("pred_freq")
            pred_freq_raw = None if raw_pred_freq is None else _safe_float(raw_pred_freq, float("nan"))
            if pred_freq_raw is not None and np.isfinite(pred_freq_raw):
                last_pred_freq = float(pred_freq_raw)
                if expected_freq is not None and abs(float(pred_freq_raw) - float(expected_freq)) <= 1e-8:
                    raw_correct_seen = True
            gate_row = gate.update(dict(row))
            gate_row = _decision_evidence_row(row=gate_row, profile=profile)
            hist = history.update(
                pred_freq=pred_freq_raw,
                margin=_safe_float(gate_row.get("margin", 0.0), 0.0),
                ratio=_safe_float(gate_row.get("ratio", 1.0), 1.0),
            )
            timestamp_s = float(stream_index) * float(profile.step_sec)
            decision = engine.step(
                pred_freq_raw,
                _safe_float(gate_row.get("decision_evidence_centered", 0.0), 0.0),
                float(hist["consistency"]),
                gate_open_freq=gate_row.get("gate_open_freq"),
                timestamp_s=timestamp_s,
            )
            stream_index += 1
            commit = bool(decision.get("commit", False))
            committed_freq = decision.get("commit_freq")
            tracked_freq = decision.get("tracked_freq")
            decision_selected = decision.get("selected_freq")
            window_index = _safe_int(row.get("window_index", 0), 0)
            latency_value = float(profile.win_sec + window_index * profile.step_sec)
            gate_open_freq = gate_row.get("gate_open_freq")
            gate_open_freq_value = None if gate_open_freq is None else _safe_float(gate_open_freq, float("nan"))
            p_correct = _row_correctness_probability(gate_row)
            decision_evidence_value = _safe_float(gate_row.get("decision_evidence_centered", 0.0), 0.0)
            max_p_correct = max(float(max_p_correct), float(p_correct))
            if max_decision_evidence is None or float(decision_evidence_value) > float(max_decision_evidence):
                max_decision_evidence = float(decision_evidence_value)
            if (
                expected_freq is not None
                and gate_open_freq_value is not None
                and np.isfinite(gate_open_freq_value)
                and abs(float(gate_open_freq_value) - float(expected_freq)) <= 1e-8
            ):
                gate_pass_correct_seen = True
                if first_gate_pass_latency is None:
                    first_gate_pass_latency = float(latency_value)
            if tracked_freq is not None and tracked_freq_first_seen is None:
                tracked_freq_first_seen = float(latency_value)
            if commit and first_any_latency is None:
                first_any_latency = float(latency_value)
                first_any_freq = None if committed_freq is None else _safe_float(committed_freq, float("nan"))
            if commit and commit_freq_first_seen is None:
                commit_freq_first_seen = float(latency_value)
            if expected_freq is None:
                if commit and committed_freq is not None and not idle_commit_seen:
                    idle_selected_events += 1
                    idle_commit_seen = True
                if (
                    previous_trial_expected is not None
                    and first_release_latency is None
                    and selected_active_prev
                    and decision_selected is None
                ):
                    first_release_latency = float(latency_value)
            else:
                if (
                    commit
                    and committed_freq is not None
                    and abs(_safe_float(committed_freq, float("nan")) - float(expected_freq)) <= 1e-8
                    and first_correct_latency is None
                ):
                    first_correct_latency = float(latency_value)
            if gate_row.get("gate_event") == "switch":
                gate_switch_count += 1
            if gate_row.get("gate_event") not in {"", "hold", None}:
                gate_event = str(gate_row.get("gate_event"))
            elif previous_gate_open_freq is None and gate_open_freq is not None:
                gate_event = "enter"
            elif previous_gate_open_freq is not None and gate_open_freq is None:
                gate_event = "exit"
            previous_gate_open_freq = None if gate_open_freq is None else float(gate_open_freq)
            selected_active_prev = decision_selected is not None

        if expected_freq is None:
            is_release_trial = previous_trial_expected is not None
            idle_duration_sec += float(max(trial_duration, 0.0))
            y2_true.append("idle")
            y2_pred.append("control" if first_any_latency is not None else "idle")
            times2.append(float(first_any_latency if first_any_latency is not None else penalty_latency))
            if is_release_trial:
                release_trials += 1
                if first_release_latency is None:
                    release_latencies.append(float(penalty_latency))
                else:
                    release_detected_trials += 1
                    release_latencies.append(float(first_release_latency))
            previous_trial_expected = None
            trial_events.append(
                {
                    "label": label_text,
                    "trial_id": int(_safe_int(trial_rows[0].get("trial_id", -1), -1)),
                    "expected_freq": None,
                    "first_any_latency_s": first_any_latency,
                    "first_release_latency_s": first_release_latency,
                    "tracked_freq_first_seen_s": tracked_freq_first_seen,
                    "commit_freq_first_seen_s": commit_freq_first_seen,
                    "first_gate_pass_latency_s": first_gate_pass_latency,
                    "gate_event": str(gate_event),
                    "gate_switch_count": int(gate_switch_count),
                    "trial_duration_s": float(trial_duration),
                    "release_trial": bool(is_release_trial),
                    "commit_seen": bool(first_any_latency is not None),
                    "max_p_correct": float(max_p_correct),
                    "max_decision_evidence": float(max_decision_evidence if max_decision_evidence is not None else 0.0),
                    "raw_correct_seen": False,
                    "gate_pass_correct_seen": False,
                }
            )
            continue

        control_trials += 1
        y2_true.append("control")
        y2_pred.append("control" if first_any_latency is not None else "idle")
        times2.append(float(first_any_latency if first_any_latency is not None else penalty_latency))
        if first_correct_latency is not None:
            control_detected_trials += 1
            detection_latencies.append(float(first_correct_latency))
            if float(first_correct_latency) <= 2.0:
                control_detected_trials_at_2s += 1
            if float(first_correct_latency) <= 3.0:
                control_detected_trials_at_3s += 1
        is_switch_trial = last_control_freq is not None and abs(float(last_control_freq) - float(expected_freq)) > 1e-8
        if is_switch_trial:
            switch_trials += 1
            if first_correct_latency is None:
                switch_latencies.append(float(penalty_latency))
            else:
                switch_detected_trials += 1
                switch_latencies.append(float(first_correct_latency))
                if float(first_correct_latency) <= 2.8:
                    switch_detected_trials_at_2p8s += 1

        pred4_freq = first_any_freq
        if pred4_freq is None and last_pred_freq is not None:
            pred4_freq = last_pred_freq
        if pred4_freq is None:
            pred4_freq = float(expected_freq)
        y4_true.append(_freq_label(float(expected_freq)))
        y4_pred.append(_freq_label(_nearest_freq(float(pred4_freq), freqs)))
        times4.append(float(first_any_latency if first_any_latency is not None else penalty_latency))
        last_control_freq = float(expected_freq)
        previous_trial_expected = float(expected_freq)
        trial_events.append(
            {
                "label": label_text,
                "trial_id": int(_safe_int(trial_rows[0].get("trial_id", -1), -1)),
                "expected_freq": float(expected_freq),
                "first_correct_latency_s": first_correct_latency,
                "first_any_latency_s": first_any_latency,
                "tracked_freq_first_seen_s": tracked_freq_first_seen,
                "commit_freq_first_seen_s": commit_freq_first_seen,
                "first_gate_pass_latency_s": first_gate_pass_latency,
                "gate_event": str(gate_event),
                "gate_switch_count": int(gate_switch_count),
                "trial_duration_s": float(trial_duration),
                "switch_trial": bool(is_switch_trial),
                "commit_seen": bool(first_any_latency is not None),
                "max_p_correct": float(max_p_correct),
                "max_decision_evidence": float(max_decision_evidence if max_decision_evidence is not None else 0.0),
                "raw_correct_seen": bool(raw_correct_seen),
                "gate_pass_correct_seen": bool(gate_pass_correct_seen),
            }
        )

    idle_minutes = float(max(idle_duration_sec, 0.0)) / 60.0
    async_metrics = {
        "idle_fp_per_min": float(idle_selected_events / idle_minutes) if idle_minutes > 1e-12 else 0.0,
        "control_recall": float(control_detected_trials / control_trials) if control_trials else 0.0,
        "control_recall_at_2s": float(control_detected_trials_at_2s / control_trials) if control_trials else 0.0,
        "control_recall_at_3s": float(control_detected_trials_at_3s / control_trials) if control_trials else 0.0,
        "switch_detect_rate": float(switch_detected_trials / switch_trials) if switch_trials else 0.0,
        "switch_detect_rate_at_2.8s": float(switch_detected_trials_at_2p8s / switch_trials) if switch_trials else 0.0,
        "release_detect_rate": float(release_detected_trials / release_trials) if release_trials else 0.0,
        "switch_latency_s": _median(switch_latencies, default=float("inf")),
        "release_latency_s": _median(release_latencies, default=float("inf")),
        "detection_latency_s": _median(detection_latencies, default=float("inf")),
        "control_trials": float(control_trials),
        "switch_trials": float(switch_trials),
        "release_trials": float(release_trials),
        "idle_fp_event_count": float(idle_selected_events),
        "idle_time_sec": float(idle_duration_sec),
        "inference_ms": float(inference_ms),
    }
    metrics_4class = compute_classification_metrics(
        y_true=y4_true,
        y_pred=y4_pred,
        labels=[_freq_label(float(freq)) for freq in freqs],
        decision_time_samples_s=times4,
        itr_class_count=4,
        decision_time_fallback_s=float(profile.win_sec),
    )
    metrics_2class = compute_classification_metrics(
        y_true=y2_true,
        y_pred=y2_pred,
        labels=["idle", "control"],
        decision_time_samples_s=times2,
        itr_class_count=2,
        decision_time_fallback_s=float(profile.win_sec),
    )
    decision_mode = parse_decision_time_mode(decision_time_mode)
    async_mode = parse_decision_time_mode(async_decision_time_mode)
    return {
        "async_metrics": async_metrics,
        "metrics_4class": metrics_4class,
        "metrics_2class": metrics_2class,
        "metrics_5class": None,
        "paper_lens_metrics_4class": dict(metrics_4class),
        "paper_lens_metrics_2class": dict(metrics_2class),
        "async_lens_metrics_4class": dict(metrics_4class),
        "async_lens_metrics_2class": dict(metrics_2class),
        "paper_lens_decision_time_mode": str(decision_mode),
        "async_lens_decision_time_mode": str(async_mode),
        "decision_params": dict(decision_params),
        "trial_events": trial_events,
    }


def _build_candidate_context(
    *,
    merged_dataset: MergedLocalDataset,
    split: RepeatedGroupSplit,
    model_name: str,
    win_sec: float,
    model_params: Mapping[str, Any],
    confidence_variant: str,
    decoder_compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
    gpu_warmup: bool,
    gpu_cache_policy: str,
    control_state_mode: str,
    decision_time_mode: str,
    async_decision_time_mode: str,
    replay_policy: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    train_segments = _subset_segments(merged_dataset.trial_segments, split.train_indices)
    gate_segments = _subset_segments(merged_dataset.trial_segments, split.gate_indices)
    holdout_segments = _subset_segments(merged_dataset.trial_segments, split.holdout_indices)
    if not train_segments:
        raise ValueError("train split is empty")
    decoder = create_decoder(
        str(model_name),
        sampling_rate=int(merged_dataset.sampling_rate),
        freqs=merged_dataset.freqs,
        win_sec=float(win_sec),
        step_sec=float(DEFAULT_TDCA_LOCAL_STEP_SEC),
        model_params=dict(model_params),
        decoder_compute_backend=str(decoder_compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=bool(gpu_warmup),
        gpu_cache_policy=str(gpu_cache_policy),
    )
    if bool(getattr(decoder, "requires_fit", False)):
        decoder.fit(train_segments)
    training_window_policy = str(
        getattr(
            decoder,
            "training_window_policy",
            dict(model_params).get("training_window_policy", "last_window_only"),
        )
    )
    training_latency_sec = _safe_float(
        getattr(decoder, "training_latency_sec", dict(model_params).get("training_latency_sec", 0.0)),
        0.0,
    )
    analysis_latency_sec = _safe_float(
        getattr(decoder, "analysis_latency_sec", dict(model_params).get("analysis_latency_sec", training_latency_sec)),
        training_latency_sec,
    )
    effective_raw_window_sec = _safe_float(
        getattr(decoder, "effective_raw_window_sec", float(win_sec) + float(analysis_latency_sec)),
        float(win_sec) + float(analysis_latency_sec),
    )
    paper_alignment_level = str(getattr(decoder, "paper_alignment_level", "partial"))
    train_rows = _attach_history_features(build_feature_rows_with_decoder(decoder, train_segments))
    gate_rows = _attach_history_features(build_feature_rows_with_decoder(decoder, gate_segments or train_segments))
    holdout_rows = _attach_history_features(build_feature_rows_with_decoder(decoder, holdout_segments or gate_segments or train_segments))
    gate_model = PerFrequencyLogRegGate()
    gate_fit_summary = gate_model.fit(
        rows=train_rows,
        freqs=merged_dataset.freqs,
        fit_config=DEFAULT_LOGREG_FIT_CONFIG,
    )
    oof_train_scored_rows, oof_summary = _build_oof_train_scored_rows(
        train_rows=train_rows,
        freqs=merged_dataset.freqs,
        fit_config=DEFAULT_LOGREG_FIT_CONFIG,
    )
    scored_gate_rows = _score_rows_with_gate(gate_rows, gate=gate_model)
    scored_holdout_rows = _score_rows_with_gate(holdout_rows, gate=gate_model)
    correctness_calibrator = CorrectnessCalibrator()
    correctness_fit_summary = correctness_calibrator.fit(
        rows=oof_train_scored_rows,
        freqs=merged_dataset.freqs,
        config=replace(DEFAULT_CORRECTNESS_CALIBRATOR_CONFIG, variant=str(confidence_variant)),
    )
    correctness_fit_summary = {
        **dict(correctness_fit_summary),
        **dict(oof_summary),
    }
    oof_train_scored_rows = _score_rows_with_correctness(oof_train_scored_rows, calibrator=correctness_calibrator)
    scored_gate_rows = _score_rows_with_correctness(scored_gate_rows, calibrator=correctness_calibrator)
    scored_holdout_rows = _score_rows_with_correctness(scored_holdout_rows, calibrator=correctness_calibrator)
    scored_tune_rows = [
        *_tag_tune_rows(oof_train_scored_rows, origin="train_oof"),
        *_tag_tune_rows(scored_gate_rows, origin="gate"),
    ]
    tune_summary = _tune_summary(scored_tune_rows, freqs=merged_dataset.freqs)
    gate_calibration_summary = _gate_calibration_summary(
        scored_rows=scored_tune_rows,
        freqs=merged_dataset.freqs,
        calibrator_summary=correctness_fit_summary,
    )
    gate_calibration_summary = {
        **dict(gate_calibration_summary),
        **dict(tune_summary),
    }
    gate_invalid_reasons = [str(item) for item in gate_calibration_summary.get("invalid_reasons", []) if str(item)]
    if not bool(tune_summary.get("valid", False)):
        gate_calibration_summary["gate_calibration_valid"] = False
        gate_invalid_reasons.extend(str(item) for item in tune_summary.get("invalid_reasons", []) if str(item))
        gate_invalid_reasons.append("tune_rows_insufficient")
    gate_calibration_summary["invalid_reasons"] = sorted(set(gate_invalid_reasons))
    control_rows = [
        row
        for row in train_rows
        if resolve_trial_role(row) == "control"
    ]
    idle_rows = [
        row
        for row in train_rows
        if resolve_trial_role(row) != "control"
    ]
    control_feature_means, control_feature_stds = _feature_stats(control_rows)
    idle_feature_means, idle_feature_stds = _feature_stats(idle_rows)
    base_profile = fit_threshold_profile(
        train_rows,
        freqs=merged_dataset.freqs,
        win_sec=float(effective_raw_window_sec),
        step_sec=float(DEFAULT_TDCA_LOCAL_STEP_SEC),
        min_enter_windows=1,
        min_exit_windows=1,
        gate_policy=DEFAULT_GATE_POLICY,
        evaluation_rows=gate_rows,
        dynamic_stop_enabled=False,
        control_state_mode=parse_control_state_mode(control_state_mode),
    )
    base_profile = replace(
        base_profile,
        model_name=str(model_name),
        model_params=dict(model_params),
        eeg_channels=tuple(int(item) for item in merged_dataset.board_eeg_channels),
        control_feature_means=control_feature_means,
        control_feature_stds=control_feature_stds,
        idle_feature_means=idle_feature_means,
        idle_feature_stds=idle_feature_stds,
        control_state_mode=parse_control_state_mode(control_state_mode),
        runtime_backend_preference=str(decoder_compute_backend),
        runtime_precision_preference=str(gpu_precision),
        enter_p_th=0.65,
        exit_p_th=0.30,
        confidence_variant=str(confidence_variant),
        training_window_policy=str(training_window_policy),
    )
    inference_ms = _measure_decoder_inference_ms(decoder, holdout_segments or gate_segments or train_segments)
    gate_profile, gate_board, threshold_board, gate_calibration_valid = _search_gate_profile(
        base_profile=base_profile,
        scored_search_rows=scored_tune_rows,
        freqs=merged_dataset.freqs,
        inference_ms=float(inference_ms),
        decision_time_mode=str(decision_time_mode),
        async_decision_time_mode=str(async_decision_time_mode),
        gate_calibration_summary=gate_calibration_summary,
        confidence_variant=str(confidence_variant),
    )
    default_bundle = _evaluate_structured_rows(
        scored_rows=scored_holdout_rows,
        profile=gate_profile,
        freqs=merged_dataset.freqs,
        decision_params=_default_decision_params(),
        inference_ms=float(inference_ms),
        decision_time_mode=str(decision_time_mode),
        async_decision_time_mode=str(async_decision_time_mode),
    )
    state_payload = None
    try:
        state_payload = decoder.get_state()
    except Exception:
        state_payload = None
    return {
        "decoder": decoder,
        "model_name": str(model_name),
        "model_params": {
            **dict(model_params),
            "training_window_policy": str(training_window_policy),
            "training_latency_sec": float(training_latency_sec),
            "analysis_latency_sec": float(analysis_latency_sec),
            "effective_raw_window_sec": float(effective_raw_window_sec),
            "paper_alignment_level": str(paper_alignment_level),
        },
        "decoder_variant": str(dict(model_params).get("decoder_variant", DEFAULT_TDCA_LOCAL_DECODER_VARIANT)),
        "confidence_variant": str(confidence_variant),
        "state_payload": state_payload,
        "train_segments": train_segments,
        "gate_segments": gate_segments,
        "holdout_segments": holdout_segments,
        "train_rows": train_rows,
        "oof_train_scored_rows": oof_train_scored_rows,
        "gate_rows": gate_rows,
        "holdout_rows": holdout_rows,
        "scored_gate_rows": scored_gate_rows,
        "scored_tune_rows": scored_tune_rows,
        "scored_holdout_rows": scored_holdout_rows,
        "gate_model": gate_model,
        "gate_fit_summary": gate_fit_summary,
        "correctness_calibrator": correctness_calibrator,
        "correctness_fit_summary": correctness_fit_summary,
        "gate_calibration_summary": gate_calibration_summary,
        "tune_summary": tune_summary,
        "tune_rows_valid": bool(tune_summary.get("valid", False)),
        "gate_profile": gate_profile,
        "gate_search_board": gate_board,
        "gate_exit_threshold_board": threshold_board,
        "gate_calibration_valid": bool(gate_calibration_valid),
        "default_holdout_bundle": default_bundle,
        "inference_ms": float(inference_ms),
        "replay_backend_policy": dict(replay_policy or {}),
        "training_window_policy": str(training_window_policy),
        "training_latency_sec": float(training_latency_sec),
        "analysis_latency_sec": float(analysis_latency_sec),
        "effective_raw_window_sec": float(effective_raw_window_sec),
        "paper_alignment_level": str(paper_alignment_level),
        "confidence_training_scheme": str(DEFAULT_CONFIDENCE_TRAINING_SCHEME),
        "oof_group_key": str(correctness_fit_summary.get("oof_group_key", "")),
        "oof_group_count": int(correctness_fit_summary.get("oof_group_count", 0) or 0),
        "sample_weight_mode": str(correctness_fit_summary.get("sample_weight_mode", "")),
        "positive_trials": int(correctness_fit_summary.get("positive_trials", 0) or 0),
        "negative_trials": int(correctness_fit_summary.get("negative_trials", 0) or 0),
    }


def _aggregate_metric_bundle(bundles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    async_rows = [dict(item.get("async_metrics", {})) for item in bundles]
    metrics4_rows = [dict(item.get("metrics_4class", {})) for item in bundles]
    metrics2_rows = [dict(item.get("metrics_2class", {})) for item in bundles]
    async_keys = sorted({key for row in async_rows for key in row.keys()})
    metrics4_keys = sorted({key for row in metrics4_rows for key in row.keys() if key not in {"confusion_matrix", "labels", "y_true", "y_pred", "decision_time_samples_s"}})
    metrics2_keys = sorted({key for row in metrics2_rows for key in row.keys() if key not in {"confusion_matrix", "labels", "y_true", "y_pred", "decision_time_samples_s"}})
    return {
        "async_metrics": {key: _median([row.get(key) for row in async_rows], default=0.0) for key in async_keys},
        "metrics_4class": {key: _median([row.get(key) for row in metrics4_rows], default=0.0) for key in metrics4_keys},
        "metrics_2class": {key: _median([row.get(key) for row in metrics2_rows], default=0.0) for key in metrics2_keys},
    }


def _sanitize_report_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for row in rows:
        clean = {str(key): value for key, value in dict(row).items() if not str(key).startswith("_")}
        sanitized.append(clean)
    return sanitized


def _error_type_for_control_event(event: Mapping[str, Any]) -> str:
    if not bool(event.get("raw_correct_seen", False)):
        return "decoder_miss"
    if not bool(event.get("gate_pass_correct_seen", False)):
        return "confidence_reject_miss"
    return "decision_miss"


def _build_error_attribution_board(
    *,
    candidate_row: Mapping[str, Any],
    holdout_bundles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, int]] = {
        "control": {"total": 0, "success": 0, "decoder_miss": 0, "confidence_reject_miss": 0, "decision_miss": 0},
        "switch": {"total": 0, "success": 0, "decoder_miss": 0, "confidence_reject_miss": 0, "decision_miss": 0},
        "release": {"total": 0, "success": 0, "decoder_miss": 0, "confidence_reject_miss": 0, "decision_miss": 0},
    }
    for bundle in holdout_bundles:
        for event in bundle.get("trial_events", []) or []:
            event_dict = dict(event)
            expected = event_dict.get("expected_freq")
            if expected is None:
                if not bool(event_dict.get("release_trial", False)):
                    continue
                totals["release"]["total"] += 1
                if event_dict.get("first_release_latency_s") is not None:
                    totals["release"]["success"] += 1
                else:
                    totals["release"]["decision_miss"] += 1
                continue
            totals["control"]["total"] += 1
            first_correct = event_dict.get("first_correct_latency_s")
            if first_correct is not None:
                totals["control"]["success"] += 1
            else:
                totals["control"][_error_type_for_control_event(event_dict)] += 1
            if not bool(event_dict.get("switch_trial", False)):
                continue
            totals["switch"]["total"] += 1
            if first_correct is not None and float(first_correct) <= 2.8:
                totals["switch"]["success"] += 1
            else:
                totals["switch"][_error_type_for_control_event(event_dict)] += 1
    board: list[dict[str, Any]] = []
    for event_type in ("control", "switch", "release"):
        payload = dict(totals[event_type])
        board.append(
            {
                "candidate_key": str(candidate_row.get("candidate_key", "")),
                "decoder_variant": str(candidate_row.get("decoder_variant", "")),
                "confidence_variant": str(candidate_row.get("confidence_variant", DEFAULT_CONFIDENCE_VARIANT)),
                "event_type": str(event_type),
                **{str(key): int(value) for key, value in payload.items()},
            }
        )
    return board


def _trial_event_index_by_repeat(
    bundles: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, int], dict[str, Any]]:
    index: dict[tuple[int, str, int], dict[str, Any]] = {}
    for repeat_index, bundle in enumerate(bundles):
        for event in bundle.get("trial_events", []) or []:
            event_dict = dict(event)
            label = str(event_dict.get("label", ""))
            trial_id = int(_safe_int(event_dict.get("trial_id", -1), -1))
            index[(int(repeat_index), label, int(trial_id))] = event_dict
    return index


def _build_contrast_error_board(
    *,
    candidate_row: Mapping[str, Any],
    tdca_holdout_bundles: Sequence[Mapping[str, Any]],
    fbcca_holdout_bundles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    tdca_index = _trial_event_index_by_repeat(tdca_holdout_bundles)
    fbcca_index = _trial_event_index_by_repeat(fbcca_holdout_bundles)
    output: list[dict[str, Any]] = []
    for key, fbcca_event in fbcca_index.items():
        expected = fbcca_event.get("expected_freq")
        if expected is None:
            continue
        fbcca_success = fbcca_event.get("first_correct_latency_s") is not None
        if not fbcca_success:
            continue
        tdca_event = dict(tdca_index.get(key, {}))
        tdca_success = tdca_event.get("first_correct_latency_s") is not None
        if tdca_success:
            continue
        output.append(
            {
                "repeat_index": int(key[0]),
                "label": str(fbcca_event.get("label", "")),
                "trial_id": int(_safe_int(fbcca_event.get("trial_id", -1), -1)),
                "expected_freq": float(_safe_float(expected, 0.0)),
                "switch_trial": bool(fbcca_event.get("switch_trial", False)),
                "fbcca_first_correct_latency_s": float(_safe_float(fbcca_event.get("first_correct_latency_s"), 0.0)),
                "tdca_first_correct_latency_s": None
                if tdca_event.get("first_correct_latency_s") is None
                else float(_safe_float(tdca_event.get("first_correct_latency_s"), 0.0)),
                "tdca_error_type": _error_type_for_control_event(tdca_event) if tdca_event else "missing_trial_event",
                "tdca_raw_correct_seen": bool(tdca_event.get("raw_correct_seen", False)),
                "tdca_gate_pass_correct_seen": bool(tdca_event.get("gate_pass_correct_seen", False)),
                "candidate_key": str(candidate_row.get("candidate_key", "")),
                "decoder_variant": str(candidate_row.get("decoder_variant", "")),
                "confidence_variant": str(candidate_row.get("confidence_variant", DEFAULT_CONFIDENCE_VARIANT)),
            }
        )
    return output


def _build_confidence_diagnostics_board(
    *,
    candidate_row: Mapping[str, Any],
    tune_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    bin_edges = np.linspace(0.0, 1.0, 11)
    for split_name, split_rows in (("tune", tune_rows), ("holdout", holdout_rows)):
        rows = [dict(row) for row in split_rows if row.get("pred_freq") is not None]
        if not rows:
            continue
        y_true = np.asarray([_row_correctness_label(row) for row in rows], dtype=float)
        y_score = np.asarray([_row_correctness_probability(row) for row in rows], dtype=float)
        brier_score = float(np.mean((y_score - y_true) ** 2)) if y_score.size else None
        auc_roc = _binary_auc(y_true, y_score)
        for index in range(len(bin_edges) - 1):
            lower = float(bin_edges[index])
            upper = float(bin_edges[index + 1])
            if index >= len(bin_edges) - 2:
                mask = np.logical_and(y_score >= lower, y_score <= upper)
            else:
                mask = np.logical_and(y_score >= lower, y_score < upper)
            if not np.any(mask):
                continue
            output.append(
                {
                    "candidate_key": str(candidate_row.get("candidate_key", "")),
                    "decoder_variant": str(candidate_row.get("decoder_variant", "")),
                    "confidence_variant": str(candidate_row.get("confidence_variant", DEFAULT_CONFIDENCE_VARIANT)),
                    "split": str(split_name),
                    "bin_lower": float(lower),
                    "bin_upper": float(upper),
                    "row_count": int(np.sum(mask)),
                    "empirical_correct_rate": float(np.mean(y_true[mask])),
                    "mean_p_correct": float(np.mean(y_score[mask])),
                    "brier_score": None if brier_score is None else float(brier_score),
                    "auc_roc": None if auc_roc is None else float(auc_roc),
                }
            )
    return output


def _build_decision_bottleneck_summary(
    *,
    candidate_row: Mapping[str, Any],
    holdout_bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    control_events: list[dict[str, Any]] = []
    switch_events: list[dict[str, Any]] = []
    release_events: list[dict[str, Any]] = []
    for bundle in holdout_bundles:
        for event in bundle.get("trial_events", []) or []:
            event_dict = dict(event)
            if event_dict.get("expected_freq") is None:
                if bool(event_dict.get("release_trial", False)):
                    release_events.append(event_dict)
                continue
            control_events.append(event_dict)
            if bool(event_dict.get("switch_trial", False)):
                switch_events.append(event_dict)
    failure_breakdown = {
        "decoder_miss": 0,
        "confidence_reject_miss": 0,
        "decision_miss": 0,
    }
    for event in control_events:
        if event.get("first_correct_latency_s") is not None:
            continue
        failure_breakdown[_error_type_for_control_event(event)] += 1
    gate_pass_latencies = [
        _safe_float(event.get("first_gate_pass_latency_s"), float("nan"))
        for event in control_events
        if event.get("first_gate_pass_latency_s") is not None
    ]
    max_p_correct_values = [
        _safe_float(event.get("max_p_correct"), float("nan"))
        for event in control_events
        if np.isfinite(_safe_float(event.get("max_p_correct"), float("nan")))
    ]
    max_decision_values = [
        _safe_float(event.get("max_decision_evidence"), float("nan"))
        for event in control_events
        if np.isfinite(_safe_float(event.get("max_decision_evidence"), float("nan")))
    ]
    return {
        "candidate_key": str(candidate_row.get("candidate_key", "")),
        "decoder_variant": str(candidate_row.get("decoder_variant", "")),
        "confidence_variant": str(candidate_row.get("confidence_variant", DEFAULT_CONFIDENCE_VARIANT)),
        "control_trials": int(len(control_events)),
        "switch_trials": int(len(switch_events)),
        "release_trials": int(len(release_events)),
        "raw_correct_seen_count": int(sum(1 for event in control_events if bool(event.get("raw_correct_seen", False)))),
        "gate_pass_correct_seen_count": int(sum(1 for event in control_events if bool(event.get("gate_pass_correct_seen", False)))),
        "commit_seen_count": int(sum(1 for event in control_events if bool(event.get("commit_seen", False)))),
        "median_first_gate_pass_latency_s": None if not gate_pass_latencies else float(np.median(np.asarray(gate_pass_latencies, dtype=float))),
        "median_max_p_correct": None if not max_p_correct_values else float(np.median(np.asarray(max_p_correct_values, dtype=float))),
        "median_max_decision_evidence": None if not max_decision_values else float(np.median(np.asarray(max_decision_values, dtype=float))),
        "failure_breakdown": {str(key): int(value) for key, value in failure_breakdown.items()},
    }


def _decision_param_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for values in product(
        DEFAULT_DECISION_GRID_CANDIDATE_MIN_WINDOWS,
        DEFAULT_DECISION_GRID_ARMED_MIN_WINDOWS,
        DEFAULT_DECISION_GRID_LAMBDA,
        DEFAULT_DECISION_GRID_UPPER,
        DEFAULT_DECISION_GRID_LOWER,
        DEFAULT_DECISION_GRID_REFRACTORY,
    ):
        candidate_min_windows, armed_min_windows, lambda_decay, upper_commit_th, lower_idle_th, refractory_sec = values
        grid.append(
            {
                "candidate_min_windows": int(candidate_min_windows),
                "armed_min_windows": int(armed_min_windows),
                "lambda_decay": float(lambda_decay),
                "upper_commit_th": float(upper_commit_th),
                "lower_idle_th": float(lower_idle_th),
                "refractory_sec": float(refractory_sec),
            }
        )
    return grid


def _render_markdown(report_payload: Mapping[str, Any]) -> str:
    async_metrics = dict(report_payload.get("chosen_async_metrics", {}) or {})
    metrics4 = dict(report_payload.get("chosen_metrics_4class", {}) or {})
    status_reasons = [str(item) for item in report_payload.get("status_reasons", [])]
    chosen_rationale = str(report_payload.get("chosen_model_rationale", ""))
    gate_valid = bool(dict(report_payload.get("chosen_metrics", {}) or {}).get("gate_calibration_valid", report_payload.get("gate_calibration_valid", False)))
    decision_rows = list(report_payload.get("decision_search_board", []) or [])
    decision_effective = bool(decision_rows) and "decision_search_not_effective" not in status_reasons
    gate_summary = dict(report_payload.get("gate_calibration_summary", {}) or {})
    data_summary = dict(report_payload.get("data_sufficiency_summary", {}) or {})
    decision_bottleneck = dict(report_payload.get("decision_bottleneck_summary", {}) or {})
    failure_breakdown = dict(decision_bottleneck.get("failure_breakdown", {}) or {})
    lines = [
        "# TDCA Local Opt",
        "",
        f"- Generated at: `{report_payload.get('generated_at', '')}`",
        f"- Task: `{report_payload.get('task', '')}`",
        f"- Search preset: `{report_payload.get('search_preset', '')}`",
        f"- Chosen model: `{report_payload.get('chosen_model', '')}`",
        f"- Decoder variant: `{report_payload.get('decoder_variant', '')}`",
        f"- Confidence variant: `{report_payload.get('confidence_variant', '')}`",
        f"- Confidence training scheme: `{report_payload.get('confidence_training_scheme', '')}`",
        f"- Decision evidence variant: `{report_payload.get('decision_evidence_variant', '')}`",
        f"- OOF group key: `{report_payload.get('oof_group_key', '')}`",
        f"- OOF group count: `{report_payload.get('oof_group_count', '')}`",
        f"- Sample weight mode: `{report_payload.get('sample_weight_mode', '')}`",
        f"- Training window policy: `{report_payload.get('training_window_policy', '')}`",
        f"- Training latency sec: `{report_payload.get('training_latency_sec', '')}`",
        f"- Analysis latency sec: `{report_payload.get('analysis_latency_sec', '')}`",
        f"- Effective raw window sec: `{report_payload.get('effective_raw_window_sec', '')}`",
        f"- Paper alignment level: `{report_payload.get('paper_alignment_level', '')}`",
        f"- Profile saved: `{report_payload.get('profile_saved', False)}`",
        f"- Profile path: `{report_payload.get('chosen_profile_path', '')}`",
        f"- Report status: `{report_payload.get('status', 'ok')}`",
        f"- Status reasons: `{', '.join(status_reasons) if status_reasons else 'none'}`",
        f"- Chosen model rationale: `{chosen_rationale or 'n/a'}`",
        f"- Gate calibration: `{'valid' if gate_valid else 'invalid'}`",
        f"- Decision search: `{'effective' if decision_effective else 'ineffective'}`",
        f"- Decision search target: `{report_payload.get('decision_search_target', '')}`",
        f"- Final selection target: `{report_payload.get('final_selection_target', '')}`",
        f"- Run valid for deployment: `{bool(report_payload.get('run_valid_for_deployment', False))}`",
        "",
        "## Async Metrics",
        "",
        f"- idle_fp_per_min: `{async_metrics.get('idle_fp_per_min', '')}`",
        f"- control_recall: `{async_metrics.get('control_recall', '')}`",
        f"- control_recall_at_3s: `{async_metrics.get('control_recall_at_3s', '')}`",
        f"- switch_latency_s: `{async_metrics.get('switch_latency_s', '')}`",
        f"- release_latency_s: `{async_metrics.get('release_latency_s', '')}`",
        f"- inference_ms: `{async_metrics.get('inference_ms', '')}`",
        "",
        "## 4-Class",
        "",
        f"- acc: `{metrics4.get('acc', '')}`",
        f"- macro_f1: `{metrics4.get('macro_f1', '')}`",
        f"- itr_bpm: `{metrics4.get('itr_bpm', '')}`",
        "",
        "## Gate Calibration",
        "",
        f"- positive_windows: `{gate_summary.get('positive_windows', '')}`",
        f"- negative_windows: `{gate_summary.get('negative_windows', '')}`",
        f"- positive_trials: `{gate_summary.get('positive_trials', '')}`",
        f"- negative_trials: `{gate_summary.get('negative_trials', '')}`",
        f"- idle_trial_count: `{gate_summary.get('idle_trial_count', '')}`",
        f"- brier_score: `{gate_summary.get('brier_score', '')}`",
        f"- auc_roc: `{gate_summary.get('auc_roc', '')}`",
        f"- diagnostics_rows: `{len(report_payload.get('confidence_diagnostics_board', []) or [])}`",
        "",
        "## Tune Summary",
        "",
        f"- rows_total: `{report_payload.get('tune_summary', {}).get('rows_total', '')}`",
        f"- min_control_trials_by_freq: `{report_payload.get('tune_summary', {}).get('min_control_trials_by_freq', '')}`",
        f"- idle_trial_count: `{report_payload.get('tune_summary', {}).get('idle_trial_count', '')}`",
        f"- tune_rows_valid: `{report_payload.get('tune_rows_valid', False)}`",
        "",
        "## Data Sufficiency",
        "",
        f"- session_count: `{data_summary.get('session_count', '')}`",
        f"- trial_count: `{data_summary.get('trial_count', '')}`",
        f"- unique_split_fingerprints: `{data_summary.get('unique_split_fingerprints', '')}`",
        f"- current_sessions_sufficient_for_deployment: `{data_summary.get('current_sessions_sufficient_for_deployment', '')}`",
        "",
        "## Decision Bottleneck",
        "",
        f"- control_trials: `{decision_bottleneck.get('control_trials', '')}`",
        f"- switch_trials: `{decision_bottleneck.get('switch_trials', '')}`",
        f"- release_trials: `{decision_bottleneck.get('release_trials', '')}`",
        f"- raw_correct_seen_count: `{decision_bottleneck.get('raw_correct_seen_count', '')}`",
        f"- gate_pass_correct_seen_count: `{decision_bottleneck.get('gate_pass_correct_seen_count', '')}`",
        f"- commit_seen_count: `{decision_bottleneck.get('commit_seen_count', '')}`",
        f"- median_first_gate_pass_latency_s: `{decision_bottleneck.get('median_first_gate_pass_latency_s', '')}`",
        f"- median_max_p_correct: `{decision_bottleneck.get('median_max_p_correct', '')}`",
        f"- median_max_decision_evidence: `{decision_bottleneck.get('median_max_decision_evidence', '')}`",
        f"- failure_breakdown: `{json.dumps(json_safe(failure_breakdown), ensure_ascii=False)}`",
    ]
    return "\n".join(lines).strip() + "\n"


def _split_replay_policy(
    *,
    merged_dataset: MergedLocalDataset,
    split: RepeatedGroupSplit,
    win_sec: float,
    env_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    all_indices = tuple(
        sorted(
            {
                *(int(item) for item in split.train_indices),
                *(int(item) for item in split.gate_indices),
                *(int(item) for item in split.holdout_indices),
            }
        )
    )
    estimated_window_count = _estimate_window_count(
        trial_segments=_subset_segments(merged_dataset.trial_segments, all_indices),
        sampling_rate=int(merged_dataset.sampling_rate),
        win_sec=float(win_sec),
        step_sec=float(DEFAULT_TDCA_LOCAL_STEP_SEC),
    )
    return _resolve_replay_backend_policy(
        env_preflight=env_preflight,
        estimated_window_count=int(estimated_window_count),
    )


def _run_baseline_suite(
    *,
    merged_dataset: MergedLocalDataset,
    splits: Sequence[RepeatedGroupSplit],
    env_preflight: Mapping[str, Any],
    config: TDCALocalOptConfig,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    decision_params = _default_decision_params()
    for model_name in DEFAULT_BASELINE_MODELS:
        candidate = _baseline_candidate(model_name, Nh=int(config.Nh))
        holdout_bundles: list[dict[str, Any]] = []
        inference_values: list[float] = []
        replay_policies: list[dict[str, Any]] = []
        for split in splits:
            replay_policy = _split_replay_policy(
                merged_dataset=merged_dataset,
                split=split,
                win_sec=float(candidate["win_sec"]),
                env_preflight=env_preflight,
            )
            context = _build_candidate_context(
                merged_dataset=merged_dataset,
                split=split,
                model_name=str(model_name),
                win_sec=float(candidate["win_sec"]),
                model_params=dict(candidate["model_params"]),
                confidence_variant=str(DEFAULT_CONFIDENCE_VARIANT),
                decoder_compute_backend=str(config.compute_backend),
                gpu_device=int(config.gpu_device),
                gpu_precision=str(config.gpu_precision),
                gpu_warmup=bool(config.gpu_warmup),
                gpu_cache_policy=str(config.gpu_cache_policy),
                control_state_mode=str(config.control_state_mode),
                decision_time_mode=str(config.decision_time_mode),
                async_decision_time_mode=str(config.async_decision_time_mode),
                replay_policy=replay_policy,
            )
            bundle = _evaluate_structured_rows(
                scored_rows=context["scored_holdout_rows"],
                profile=context["gate_profile"],
                freqs=merged_dataset.freqs,
                decision_params=decision_params,
                inference_ms=float(context["inference_ms"]),
                decision_time_mode=str(config.decision_time_mode),
                async_decision_time_mode=str(config.async_decision_time_mode),
            )
            holdout_bundles.append(bundle)
            inference_values.append(float(context["inference_ms"]))
            replay_policies.append(dict(replay_policy))
        aggregated = _aggregate_metric_bundle(holdout_bundles)
        async_metrics = dict(aggregated.get("async_metrics", {}))
        async_metrics["inference_ms"] = _median(inference_values, default=float("inf"))
        policy = replay_policies[0] if replay_policies else {}
        row = {
            "model_name": str(model_name),
            "decoder_variant": "n/a",
            "algorithm_alignment": "not_applicable",
            "paper_tdca_projection_enabled": False,
            "candidate": dict(candidate),
            "split_fingerprints": [str(split.fingerprint) for split in splits],
            "decision_params": dict(decision_params),
            "metrics_median": async_metrics,
            "metrics_4class_median": dict(aggregated.get("metrics_4class", {})),
            "metrics_2class_median": dict(aggregated.get("metrics_2class", {})),
            "inference_ms": float(async_metrics.get("inference_ms", float("inf"))),
            "effective_replay_backend": str(policy.get("effective_replay_backend", "cpu")),
            "gpu_replay_speedup": float(policy.get("gpu_replay_speedup", 0.0)),
            "gpu_replay_eligible": bool(policy.get("gpu_replay_eligible", False)),
            "gpu_replay_reason": str(policy.get("gpu_replay_reason", "")),
            "_holdout_bundles": [dict(item) for item in holdout_bundles],
        }
        row["rank_key"] = [float(item) for item in _rank_metrics_key(async_metrics)]
        results.append(row)
    results.sort(key=lambda item: _candidate_rank_tuple(item))
    return results


def run_tdca_local_opt(
    config: TDCALocalOptConfig,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_fn: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    report_paths = _resolve_report_paths(config)
    report_dir = Path(report_paths["report_dir"]).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    resolved_profile_path = Path(report_paths["output_profile"]).expanduser().resolve()
    resolved_profile_v2_path = Path(report_paths["profile_v2"]).expanduser().resolve()
    run_tag = str(report_paths.get("run_tag", report_dir.name))
    run_log_path = Path(report_paths["run_log"]).expanduser().resolve()
    progress_snapshot_path = Path(report_paths["progress_snapshot"]).expanduser().resolve()

    def log(text: str) -> None:
        _append_run_log(run_log_path, text)
        if log_fn is not None:
            log_fn(text)

    started_at = time.perf_counter()
    progress_state: dict[str, Any] = {
        "task": "tdca-local-opt",
        "stage": "prepare",
        "stage_label": _tdca_progress_label("prepare"),
        "detail": "准备运行目录",
        "run_index": 0,
        "run_total": 0,
        "config_index": 0,
        "config_total": 0,
        "report_dir": str(report_dir),
        "report_path": str(Path(report_paths["report_json"]).expanduser().resolve()),
        "profile_path": str(resolved_profile_path),
        "progress_percent": 0,
        "elapsed_s": 0.0,
        "eta_s": None,
    }

    def emit_progress(*, force: bool = False, **updates: Any) -> None:
        explicit_progress = updates.get("progress_percent") if "progress_percent" in updates else None
        progress_state.update(updates)
        stage_name = str(progress_state.get("stage", "") or "prepare")
        run_index = int(progress_state.get("run_index", 0) or 0)
        run_total = int(progress_state.get("run_total", 0) or 0)
        elapsed_s = float(max(time.perf_counter() - started_at, 0.0))
        if isinstance(explicit_progress, (int, float)):
            progress_percent = explicit_progress
        else:
            progress_percent = _tdca_progress_percent(stage_name, run_index=run_index, run_total=run_total)
        progress_state["stage_label"] = str(progress_state.get("stage_label") or _tdca_progress_label(stage_name))
        progress_state["progress_percent"] = int(max(0, min(100, int(progress_percent))))
        progress_state["elapsed_s"] = float(elapsed_s)
        if progress_state["progress_percent"] > 0 and progress_state["progress_percent"] < 100:
            eta_s = elapsed_s * ((100.0 / float(progress_state["progress_percent"])) - 1.0)
            progress_state["eta_s"] = float(max(eta_s, 0.0))
        elif progress_state["progress_percent"] >= 100:
            progress_state["eta_s"] = 0.0
        else:
            progress_state["eta_s"] = None
        atomic_write_text(progress_snapshot_path, json_dumps(json_safe(progress_state)) + "\n")
        if progress_fn is not None:
            progress_fn(dict(progress_state))

    _validate_tdca_local_config(config)
    run_config_payload = dict(asdict(config))
    run_config_payload["output_profile_path"] = str(resolved_profile_path)
    run_config_payload["resolved_profile_v2_path"] = str(resolved_profile_v2_path)
    run_config_payload["report_path"] = str(Path(report_paths["report_json"]).expanduser().resolve())
    run_config_payload["report_dir"] = str(report_dir)
    run_config_payload["run_tag"] = str(run_tag)
    atomic_write_text(
        Path(report_paths["run_config"]).expanduser().resolve(),
        json_dumps(json_safe(run_config_payload)) + "\n",
    )
    log(f"[tdca-local-opt] report directory prepared: {report_dir}")
    emit_progress(
        force=True,
        stage="prepare",
        stage_label=_tdca_progress_label("prepare"),
        detail=f"运行目录已准备：{report_dir}",
        run_index=1,
        run_total=4,
    )
    env_preflight = preflight_tdca_local_env(
        compute_backend=str(config.compute_backend),
        gpu_device=int(config.gpu_device),
        gpu_precision=str(config.gpu_precision),
    )
    log(
        "[tdca-local-opt] compute backend prepared: "
        f"requested={config.compute_backend} effective={env_preflight.get('effective_backend','cpu')}"
    )
    emit_progress(
        force=True,
        stage="prepare",
        stage_label=_tdca_progress_label("prepare"),
        detail="后端预检完成，开始载入数据",
        run_index=2,
        run_total=4,
    )
    merged_dataset = _load_merged_dataset(config)
    emit_progress(
        force=True,
        stage="prepare",
        stage_label=_tdca_progress_label("prepare"),
        detail=f"数据已载入：trial={len(merged_dataset.trial_segments)}",
        run_index=3,
        run_total=4,
    )
    search_plan = _resolve_search_plan(config)
    splits = build_repeated_group_splits(
        merged_dataset.trial_segments,
        repeats=int(search_plan["repeats"]),
        seed=int(config.seed),
    )
    split_fingerprints = [str(item.fingerprint) for item in splits]
    unique_fingerprint_count = len(set(split_fingerprints))
    seed_effective = {
        "requested_repeats": int(search_plan["repeats"]),
        "generated_repeats": int(len(splits)),
        "unique_split_fingerprints": int(unique_fingerprint_count),
        "effective_repeats": int(unique_fingerprint_count),
        "invalid": bool(int(search_plan["repeats"]) > 1 and unique_fingerprint_count < 2),
    }
    status = "invalid" if bool(seed_effective["invalid"]) else "ok"
    status_reasons: list[str] = []
    if bool(seed_effective["invalid"]):
        status_reasons.append("split_fingerprints_not_effective")

    emit_progress(
        force=True,
        stage="prepare",
        stage_label=_tdca_progress_label("prepare"),
        detail=f"分组完成：preset={search_plan['search_preset']} repeat={len(splits)}",
        run_index=4,
        run_total=4,
    )
    emit_progress(
        force=True,
        stage="baseline_opening",
        stage_label=_tdca_progress_label("baseline_opening"),
        detail="开始基线开场对比",
        run_index=0,
        run_total=len(DEFAULT_BASELINE_MODELS),
    )
    baseline_opening = _run_baseline_suite(
        merged_dataset=merged_dataset,
        splits=splits,
        env_preflight=env_preflight,
        config=config,
    )
    emit_progress(
        force=True,
        stage="baseline_opening",
        stage_label=_tdca_progress_label("baseline_opening"),
        detail="基线开场对比完成",
        run_index=len(DEFAULT_BASELINE_MODELS),
        run_total=len(DEFAULT_BASELINE_MODELS),
    )

    candidate_results: dict[str, list[dict[str, Any]]] = {}
    gate_search_by_candidate: dict[str, list[dict[str, Any]]] = {}
    gate_exit_by_candidate: dict[str, list[dict[str, Any]]] = {}
    candidate_context_cache: dict[tuple[str, int], dict[str, Any]] = {}
    candidate_grid = [dict(item) for item in search_plan["candidate_grid"]]
    total_candidates = len(candidate_grid) * max(len(splits), 1)
    progress_index = 0
    for split in splits:
        for candidate in candidate_grid:
            progress_index += 1
            key = _candidate_key(
                decoder_variant=str(candidate["decoder_variant"]),
                win_sec=float(candidate["win_sec"]),
                delay_steps=int(candidate["delay_steps"]),
                n_components=int(candidate["n_components"]),
                confidence_variant=str(candidate.get("confidence_variant", DEFAULT_CONFIDENCE_VARIANT)),
            )
            log(
                f"[tdca-local-opt] repeat={split.repeat_index + 1}/{len(splits)} "
                f"candidate={progress_index}/{total_candidates} {key}"
            )
            emit_progress(
                stage="candidate_search",
                stage_label=_tdca_progress_label("candidate_search"),
                detail=(
                    f"repeat {split.repeat_index + 1}/{len(splits)} | "
                    f"candidate {progress_index}/{total_candidates} | {key}"
                ),
                model_name="tdca",
                run_index=progress_index,
                run_total=total_candidates,
            )
            replay_policy = _split_replay_policy(
                merged_dataset=merged_dataset,
                split=split,
                win_sec=float(candidate["win_sec"]),
                env_preflight=env_preflight,
            )
            context = _build_candidate_context(
                merged_dataset=merged_dataset,
                split=split,
                model_name=DEFAULT_TDCA_LOCAL_MODEL,
                win_sec=float(candidate["win_sec"]),
                model_params=_default_model_params(
                    model_name=DEFAULT_TDCA_LOCAL_MODEL,
                    Nh=int(config.Nh),
                    delay_steps=int(candidate["delay_steps"]),
                    n_components=int(candidate["n_components"]),
                    decoder_variant=str(candidate["decoder_variant"]),
                ),
                confidence_variant=str(candidate.get("confidence_variant", DEFAULT_CONFIDENCE_VARIANT)),
                decoder_compute_backend=str(config.compute_backend),
                gpu_device=int(config.gpu_device),
                gpu_precision=str(config.gpu_precision),
                gpu_warmup=bool(config.gpu_warmup),
                gpu_cache_policy=str(config.gpu_cache_policy),
                control_state_mode=str(config.control_state_mode),
                decision_time_mode=str(config.decision_time_mode),
                async_decision_time_mode=str(config.async_decision_time_mode),
                replay_policy=replay_policy,
            )
            candidate_context_cache[(key, int(split.repeat_index))] = context
            candidate_results.setdefault(key, []).append(
                {
                    "repeat_index": int(split.repeat_index),
                    "candidate": dict(candidate),
                    "holdout_bundle": dict(context["default_holdout_bundle"]),
                    "inference_ms": float(context["inference_ms"]),
                    "replay_backend_policy": dict(replay_policy),
                    "gate_calibration_summary": dict(context.get("gate_calibration_summary", {})),
                    "tune_summary": dict(context.get("tune_summary", {})),
                    "training_window_policy": str(context.get("training_window_policy", "last_window_only")),
                    "training_latency_sec": float(context.get("training_latency_sec", 0.0)),
                    "analysis_latency_sec": float(context.get("analysis_latency_sec", 0.0)),
                    "effective_raw_window_sec": float(context.get("effective_raw_window_sec", candidate["win_sec"])),
                    "paper_alignment_level": str(context.get("paper_alignment_level", "partial")),
                    "confidence_training_scheme": str(
                        context.get("confidence_training_scheme", DEFAULT_CONFIDENCE_TRAINING_SCHEME)
                    ),
                    "oof_group_key": str(context.get("oof_group_key", "")),
                    "oof_group_count": int(context.get("oof_group_count", 0) or 0),
                    "sample_weight_mode": str(context.get("sample_weight_mode", "")),
                    "positive_trials": int(context.get("positive_trials", 0) or 0),
                    "negative_trials": int(context.get("negative_trials", 0) or 0),
                }
            )
            gate_search_by_candidate.setdefault(key, []).extend(
                [
                    {
                        "repeat_index": int(split.repeat_index),
                        "decoder_variant": str(candidate["decoder_variant"]),
                        **dict(row),
                    }
                    for row in list(context["gate_search_board"])
                ]
            )
            gate_exit_by_candidate.setdefault(key, []).extend(
                [
                    {
                        "repeat_index": int(split.repeat_index),
                        "decoder_variant": str(candidate["decoder_variant"]),
                        **dict(row),
                    }
                    for row in list(context["gate_exit_threshold_board"])
                ]
            )

    tdca_search_board: list[dict[str, Any]] = []
    for key, rows in candidate_results.items():
        bundles = [dict(item.get("holdout_bundle", {})) for item in rows]
        aggregated = _aggregate_metric_bundle(bundles)
        async_metrics = dict(aggregated.get("async_metrics", {}))
        async_metrics["inference_ms"] = _median([item.get("inference_ms") for item in rows], default=float("inf"))
        rank_key = _rank_metrics_key(async_metrics)
        sample_candidate = dict(rows[0].get("candidate", {}))
        variant_metadata = _tdca_variant_metadata(sample_candidate.get("decoder_variant"))
        gate_rows = list(gate_exit_by_candidate.get(str(key), []))
        gate_summaries = [dict(item.get("gate_calibration_summary", {})) for item in rows]
        gate_summary = dict(gate_summaries[0]) if gate_summaries else {}
        gate_valid = bool(gate_summary.get("gate_calibration_valid", False))
        min_gate_control_rows = min((_safe_int(item.get("n_control_rows", 0), 0) for item in gate_rows), default=0)
        min_gate_idle_rows = min((_safe_int(item.get("n_idle_rows", 0), 0) for item in gate_rows), default=0)
        enter_logit_th_median = _median([item.get("enter_logit_th") for item in gate_rows], default=0.0)
        exit_logit_th_median = _median([item.get("exit_logit_th") for item in gate_rows], default=0.0)
        enter_p_th_median = _median([item.get("enter_p_th") for item in gate_rows], default=0.65)
        exit_p_th_median = _median([item.get("exit_p_th") for item in gate_rows], default=0.30)
        confidence_variant = _resolved_confidence_variant(sample_candidate)
        tdca_search_board.append(
            {
                "candidate_key": str(key),
                "candidate": sample_candidate,
                "decoder_variant": str(variant_metadata["decoder_variant"]),
                "algorithm_alignment": str(variant_metadata["algorithm_alignment"]),
                "paper_tdca_projection_enabled": bool(variant_metadata["paper_tdca_projection_enabled"]),
                "confidence_variant": str(confidence_variant),
                "confidence_training_scheme": str(
                    rows[0].get("confidence_training_scheme", DEFAULT_CONFIDENCE_TRAINING_SCHEME)
                ),
                "oof_group_key": str(rows[0].get("oof_group_key", "")),
                "oof_group_count": int(rows[0].get("oof_group_count", 0) or 0),
                "sample_weight_mode": str(rows[0].get("sample_weight_mode", "")),
                "positive_trials": int(rows[0].get("positive_trials", 0) or 0),
                "negative_trials": int(rows[0].get("negative_trials", 0) or 0),
                "training_window_policy": str(rows[0].get("training_window_policy", "last_window_only")),
                "training_latency_sec": float(rows[0].get("training_latency_sec", 0.0) or 0.0),
                "analysis_latency_sec": float(rows[0].get("analysis_latency_sec", 0.0) or 0.0),
                "effective_raw_window_sec": float(
                    rows[0].get("effective_raw_window_sec", sample_candidate.get("win_sec", 0.0)) or 0.0
                ),
                "paper_alignment_level": str(rows[0].get("paper_alignment_level", "partial")),
                "tune_summary": dict(rows[0].get("tune_summary", {})),
                "gate_calibration_valid": bool(gate_valid),
                "min_gate_control_rows": int(min_gate_control_rows),
                "min_gate_idle_rows": int(min_gate_idle_rows),
                "enter_p_th_median": float(enter_p_th_median),
                "exit_p_th_median": float(exit_p_th_median),
                "enter_logit_th_median": float(enter_logit_th_median),
                "exit_logit_th_median": float(exit_logit_th_median),
                "gate_calibration_summary": dict(gate_summary),
                "repeat_count": int(len(rows)),
                "metrics_median": async_metrics,
                "metrics_4class_median": dict(aggregated.get("metrics_4class", {})),
                "metrics_2class_median": dict(aggregated.get("metrics_2class", {})),
                "rank_key": [float(item) for item in rank_key],
            }
        )
    tdca_search_board.sort(key=_tdca_board_sort_key)
    top_candidates: list[dict[str, Any]] = []
    selected_candidate_keys: set[str] = set()
    eligible_candidate_rows = [dict(item) for item in tdca_search_board if bool(item.get("gate_calibration_valid", False))]
    for decoder_variant in DEFAULT_TDCA_LOCAL_DECODER_VARIANTS:
        variant_rows = [
            dict(item)
            for item in eligible_candidate_rows
            if str(item.get("decoder_variant", "")) == str(decoder_variant)
        ]
        for row in variant_rows[:4]:
            candidate_key = str(row.get("candidate_key", ""))
            if candidate_key and candidate_key not in selected_candidate_keys:
                top_candidates.append(dict(row))
                selected_candidate_keys.add(candidate_key)
    if len(top_candidates) < DEFAULT_TDCA_LOCAL_TOP_K:
        for row in eligible_candidate_rows:
            candidate_key = str(row.get("candidate_key", ""))
            if candidate_key and candidate_key not in selected_candidate_keys:
                top_candidates.append(dict(row))
                selected_candidate_keys.add(candidate_key)
            if len(top_candidates) >= DEFAULT_TDCA_LOCAL_TOP_K:
                break
    top_candidates.sort(key=_tdca_board_sort_key)
    if tdca_search_board and not top_candidates:
        status = "invalid"
        status_reasons.append("gate_calibration_invalid_all_candidates")
        if any(
            "tune_rows_insufficient" in dict(row.get("gate_calibration_summary", {})).get("invalid_reasons", [])
            for row in tdca_search_board
        ):
            status_reasons.append("tune_rows_insufficient")
    variant_summary: list[dict[str, Any]] = []
    selected_for_decision = {str(row.get("candidate_key", "")) for row in top_candidates}
    for decoder_variant in DEFAULT_TDCA_LOCAL_DECODER_VARIANTS:
        variant_rows = [
            dict(item)
            for item in tdca_search_board
            if str(item.get("decoder_variant", "")) == str(decoder_variant)
        ]
        if not variant_rows:
            continue
        best_variant_row = dict(variant_rows[0])
        variant_summary.append(
            {
                "decoder_variant": str(decoder_variant),
                "algorithm_alignment": str(best_variant_row.get("algorithm_alignment", "")),
                "paper_tdca_projection_enabled": bool(best_variant_row.get("paper_tdca_projection_enabled", False)),
                "confidence_variant": str(best_variant_row.get("confidence_variant", DEFAULT_CONFIDENCE_VARIANT)),
                "confidence_training_scheme": str(
                    best_variant_row.get("confidence_training_scheme", DEFAULT_CONFIDENCE_TRAINING_SCHEME)
                ),
                "training_window_policy": str(best_variant_row.get("training_window_policy", "last_window_only")),
                "training_latency_sec": float(best_variant_row.get("training_latency_sec", 0.0) or 0.0),
                "best_candidate": dict(best_variant_row.get("candidate", {})),
                "metrics_median": dict(best_variant_row.get("metrics_median", {})),
                "rank_key": [float(item) for item in best_variant_row.get("rank_key", [])],
                "selected_for_decision_search": bool(str(best_variant_row.get("candidate_key", "")) in selected_for_decision),
            }
        )

    decision_grid = _decision_param_grid()
    total_decision_runs = max(len(top_candidates) * len(decision_grid), 1)
    decision_progress_index = 0
    decision_aggregate_rows: list[dict[str, Any]] = []
    final_candidate_rows: list[dict[str, Any]] = []
    candidate_best_global_params: dict[str, dict[str, Any]] = {}
    for candidate_row in top_candidates:
        candidate_key = str(candidate_row.get("candidate_key", ""))
        candidate_confidence_variant = _resolved_confidence_variant(candidate_row)
        gate_eval_by_param: dict[str, list[dict[str, Any]]] = {}
        param_payload_by_key: dict[str, dict[str, Any]] = {}
        candidate_tune_rows: list[dict[str, Any]] = []
        for split in splits:
            context = candidate_context_cache[(candidate_key, int(split.repeat_index))]
            candidate_tune_rows.extend([dict(row) for row in context.get("scored_tune_rows", []) or []])
            for params in decision_grid:
                param_key = _make_decision_params_key(params)
                param_payload_by_key[param_key] = dict(params)
                bundle = _evaluate_structured_rows(
                    scored_rows=context["scored_tune_rows"],
                    profile=context["gate_profile"],
                    freqs=merged_dataset.freqs,
                    decision_params=params,
                    inference_ms=float(context["inference_ms"]),
                    decision_time_mode=str(config.decision_time_mode),
                    async_decision_time_mode=str(config.async_decision_time_mode),
                )
                gate_eval_by_param.setdefault(param_key, []).append(bundle)
        candidate_decision_board: list[dict[str, Any]] = []
        for param_key, bundles in gate_eval_by_param.items():
            decision_progress_index += 1
            aggregated = _aggregate_metric_bundle(bundles)
            async_metrics = dict(aggregated.get("async_metrics", {}))
            rank_key = _rank_metrics_key(async_metrics)
            variant_metadata = _tdca_variant_metadata(dict(candidate_row.get("candidate", {})).get("decoder_variant"))
            emit_progress(
                stage="decision_search",
                stage_label=_tdca_progress_label("decision_search"),
                detail=(
                    f"candidate {decision_progress_index}/{total_decision_runs} | "
                    f"{candidate_key} | params={param_key}"
                ),
                model_name="tdca",
                run_index=decision_progress_index,
                run_total=total_decision_runs,
            )
            row = {
                "candidate_key": str(candidate_key),
                "candidate": dict(candidate_row.get("candidate", {})),
                "decoder_variant": str(variant_metadata["decoder_variant"]),
                "algorithm_alignment": str(variant_metadata["algorithm_alignment"]),
                "paper_tdca_projection_enabled": bool(variant_metadata["paper_tdca_projection_enabled"]),
                "confidence_variant": str(candidate_confidence_variant),
                "confidence_training_scheme": str(
                    candidate_row.get("confidence_training_scheme", DEFAULT_CONFIDENCE_TRAINING_SCHEME)
                ),
                "oof_group_key": str(candidate_row.get("oof_group_key", "")),
                "oof_group_count": int(candidate_row.get("oof_group_count", 0) or 0),
                "sample_weight_mode": str(candidate_row.get("sample_weight_mode", "")),
                "positive_trials": int(candidate_row.get("positive_trials", 0) or 0),
                "negative_trials": int(candidate_row.get("negative_trials", 0) or 0),
                "training_window_policy": str(candidate_row.get("training_window_policy", "last_window_only")),
                "training_latency_sec": float(candidate_row.get("training_latency_sec", 0.0) or 0.0),
                "analysis_latency_sec": float(candidate_row.get("analysis_latency_sec", 0.0) or 0.0),
                "effective_raw_window_sec": float(
                    candidate_row.get("effective_raw_window_sec", dict(candidate_row.get("candidate", {})).get("win_sec", 0.0)) or 0.0
                ),
                "paper_alignment_level": str(candidate_row.get("paper_alignment_level", "partial")),
                "decision_evidence_variant": DEFAULT_DECISION_EVIDENCE_VARIANT,
                "tune_summary": dict(candidate_row.get("tune_summary", {})),
                "selection_target": "tune_split",
                "decision_params": dict(param_payload_by_key[param_key]),
                "metrics_median": async_metrics,
                "metrics_4class_median": dict(aggregated.get("metrics_4class", {})),
                "metrics_2class_median": dict(aggregated.get("metrics_2class", {})),
                "rank_key": [float(item) for item in rank_key],
            }
            candidate_decision_board.append(row)
            decision_aggregate_rows.append(dict(row))
        candidate_decision_board.sort(key=_tdca_board_sort_key)
        best_global_params = dict(candidate_decision_board[0]["decision_params"])
        candidate_best_global_params[candidate_key] = best_global_params
        holdout_bundles: list[dict[str, Any]] = []
        candidate_holdout_rows: list[dict[str, Any]] = []
        for split in splits:
            context = candidate_context_cache[(candidate_key, int(split.repeat_index))]
            candidate_holdout_rows.extend([dict(row) for row in context.get("scored_holdout_rows", []) or []])
            holdout_bundles.append(
                _evaluate_structured_rows(
                    scored_rows=context["scored_holdout_rows"],
                    profile=context["gate_profile"],
                    freqs=merged_dataset.freqs,
                    decision_params=best_global_params,
                    inference_ms=float(context["inference_ms"]),
                    decision_time_mode=str(config.decision_time_mode),
                    async_decision_time_mode=str(config.async_decision_time_mode),
                )
            )
        aggregated_holdout = _aggregate_metric_bundle(holdout_bundles)
        async_metrics = dict(aggregated_holdout.get("async_metrics", {}))
        async_metrics["inference_ms"] = _median(
            [bundle.get("async_metrics", {}).get("inference_ms") for bundle in holdout_bundles],
            default=float("inf"),
        )
        rank_key = _rank_metrics_key(async_metrics)
        variant_metadata = _tdca_variant_metadata(dict(candidate_row.get("candidate", {})).get("decoder_variant"))
        confidence_diagnostics_board = _build_confidence_diagnostics_board(
            candidate_row=candidate_row,
            tune_rows=candidate_tune_rows,
            holdout_rows=candidate_holdout_rows,
        )
        decision_bottleneck_summary = _build_decision_bottleneck_summary(
            candidate_row=candidate_row,
            holdout_bundles=holdout_bundles,
        )
        final_candidate_rows.append(
            {
                "candidate_key": str(candidate_key),
                "candidate": dict(candidate_row.get("candidate", {})),
                "decoder_variant": str(variant_metadata["decoder_variant"]),
                "algorithm_alignment": str(variant_metadata["algorithm_alignment"]),
                "paper_tdca_projection_enabled": bool(variant_metadata["paper_tdca_projection_enabled"]),
                "confidence_variant": str(candidate_confidence_variant),
                "confidence_training_scheme": str(
                    candidate_row.get("confidence_training_scheme", DEFAULT_CONFIDENCE_TRAINING_SCHEME)
                ),
                "oof_group_key": str(candidate_row.get("oof_group_key", "")),
                "oof_group_count": int(candidate_row.get("oof_group_count", 0) or 0),
                "sample_weight_mode": str(candidate_row.get("sample_weight_mode", "")),
                "positive_trials": int(candidate_row.get("positive_trials", 0) or 0),
                "negative_trials": int(candidate_row.get("negative_trials", 0) or 0),
                "training_window_policy": str(candidate_row.get("training_window_policy", "last_window_only")),
                "training_latency_sec": float(candidate_row.get("training_latency_sec", 0.0) or 0.0),
                "analysis_latency_sec": float(candidate_row.get("analysis_latency_sec", 0.0) or 0.0),
                "effective_raw_window_sec": float(
                    candidate_row.get("effective_raw_window_sec", dict(candidate_row.get("candidate", {})).get("win_sec", 0.0)) or 0.0
                ),
                "paper_alignment_level": str(candidate_row.get("paper_alignment_level", "partial")),
                "decision_evidence_variant": DEFAULT_DECISION_EVIDENCE_VARIANT,
                "confidence_diagnostics_board": confidence_diagnostics_board,
                "decision_bottleneck_summary": decision_bottleneck_summary,
                "tune_summary": dict(candidate_row.get("tune_summary", {})),
                "gate_calibration_valid": bool(candidate_row.get("gate_calibration_valid", False)),
                "gate_calibration_summary": dict(candidate_row.get("gate_calibration_summary", {})),
                "selection_target": "holdout_split",
                "decision_params": best_global_params,
                "repeat_count": int(len(holdout_bundles)),
                "metrics_median": async_metrics,
                "metrics_4class_median": dict(aggregated_holdout.get("metrics_4class", {})),
                "metrics_2class_median": dict(aggregated_holdout.get("metrics_2class", {})),
                "rank_key": [float(item) for item in rank_key],
            }
        )

    decision_aggregate_rows.sort(key=_tdca_board_sort_key)
    final_candidate_rows.sort(key=_tdca_board_sort_key)
    decision_release_values = {
        round(_safe_float(dict(row.get("metrics_median", {})).get("release_latency_s", float("inf")), float("inf")), 6)
        for row in decision_aggregate_rows
        if np.isfinite(_safe_float(dict(row.get("metrics_median", {})).get("release_latency_s", float("inf")), float("inf")))
    }
    decision_switch_values = {
        round(_safe_float(dict(row.get("metrics_median", {})).get("switch_latency_s", float("inf")), float("inf")), 6)
        for row in decision_aggregate_rows
        if np.isfinite(_safe_float(dict(row.get("metrics_median", {})).get("switch_latency_s", float("inf")), float("inf")))
    }
    if len(decision_aggregate_rows) > 1 and len(decision_release_values) < 2 and len(decision_switch_values) < 2:
        status = "invalid"
        status_reasons.append("decision_search_not_effective")
    chosen_row = dict(final_candidate_rows[0]) if final_candidate_rows else (dict(tdca_search_board[0]) if tdca_search_board else {})
    emit_progress(
        force=True,
        stage="baseline_seal",
        stage_label=_tdca_progress_label("baseline_seal"),
        detail="开始基线封板对比",
        run_index=0,
        run_total=len(DEFAULT_BASELINE_MODELS),
    )
    baseline_seal = _run_baseline_suite(
        merged_dataset=merged_dataset,
        splits=splits,
        env_preflight=env_preflight,
        config=config,
    )
    emit_progress(
        force=True,
        stage="baseline_seal",
        stage_label=_tdca_progress_label("baseline_seal"),
        detail="基线封板对比完成",
        run_index=len(DEFAULT_BASELINE_MODELS),
        run_total=len(DEFAULT_BASELINE_MODELS),
    )
    chosen_model_rationale = ""
    if status != "ok":
        chosen_model_rationale = "invalid_run_not_comparable"
    elif chosen_row and baseline_seal:
        best_baseline_rank = min((_candidate_rank_tuple(row) for row in baseline_seal), default=tuple())
        chosen_rank_tuple = _candidate_rank_tuple(chosen_row)
        chosen_model_rationale = (
            "tdca_not_clearly_superior"
            if best_baseline_rank and chosen_rank_tuple >= best_baseline_rank
            else "tdca_superior_on_primary_ranking"
        )
    error_attribution_board: list[dict[str, Any]] = []
    for candidate_row in final_candidate_rows:
        candidate_key = str(candidate_row.get("candidate_key", ""))
        holdout_bundles: list[dict[str, Any]] = []
        for split in splits:
            context = candidate_context_cache.get((candidate_key, int(split.repeat_index)))
            if context is None:
                continue
            holdout_bundles.append(
                _evaluate_structured_rows(
                    scored_rows=context["scored_holdout_rows"],
                    profile=context["gate_profile"],
                    freqs=merged_dataset.freqs,
                    decision_params=dict(candidate_row.get("decision_params", {})),
                    inference_ms=float(context["inference_ms"]),
                    decision_time_mode=str(config.decision_time_mode),
                    async_decision_time_mode=str(config.async_decision_time_mode),
                )
            )
        error_attribution_board.extend(
            _build_error_attribution_board(
                candidate_row=candidate_row,
                holdout_bundles=holdout_bundles,
            )
        )
    fbcca_baseline_row = next(
        (dict(row) for row in baseline_seal if str(row.get("model_name", "")) == "fbcca"),
        None,
    )
    contrast_error_board: list[dict[str, Any]] = []
    if chosen_row and fbcca_baseline_row is not None:
        contrast_error_board = _build_contrast_error_board(
            candidate_row=chosen_row,
            tdca_holdout_bundles=[
                _evaluate_structured_rows(
                    scored_rows=candidate_context_cache[(str(chosen_row.get("candidate_key", "")), int(split.repeat_index))]["scored_holdout_rows"],
                    profile=candidate_context_cache[(str(chosen_row.get("candidate_key", "")), int(split.repeat_index))]["gate_profile"],
                    freqs=merged_dataset.freqs,
                    decision_params=dict(chosen_row.get("decision_params", {})),
                    inference_ms=float(candidate_context_cache[(str(chosen_row.get("candidate_key", "")), int(split.repeat_index))]["inference_ms"]),
                    decision_time_mode=str(config.decision_time_mode),
                    async_decision_time_mode=str(config.async_decision_time_mode),
                )
                for split in splits
                if (str(chosen_row.get("candidate_key", "")), int(split.repeat_index)) in candidate_context_cache
            ],
            fbcca_holdout_bundles=[dict(item) for item in fbcca_baseline_row.get("_holdout_bundles", []) or []],
        )
    baseline_opening_report = _sanitize_report_rows(baseline_opening)
    baseline_seal_report = _sanitize_report_rows(baseline_seal)
    chosen_confidence_variant = _resolved_confidence_variant(chosen_row)
    chosen_training_latency_sec = float(chosen_row.get("training_latency_sec", 0.0) or 0.0)
    chosen_analysis_latency_sec = float(chosen_row.get("analysis_latency_sec", 0.0) or 0.0)
    chosen_effective_raw_window_sec = float(
        chosen_row.get("effective_raw_window_sec", dict(chosen_row.get("candidate", {})).get("win_sec", 0.0)) or 0.0
    )
    chosen_paper_alignment_level = str(chosen_row.get("paper_alignment_level", "partial"))
    chosen_tune_summary = dict(chosen_row.get("tune_summary", {}))
    chosen_confidence_diagnostics_board = [dict(item) for item in chosen_row.get("confidence_diagnostics_board", []) or []]
    chosen_decision_bottleneck_summary = dict(chosen_row.get("decision_bottleneck_summary", {}))

    profile_saved = False
    profile_v2_saved = False
    chosen_profile_path: Optional[str] = None
    profile_v2_path: Optional[str] = None
    chosen_profile: Optional[ThresholdProfile] = None
    chosen_replay_policy: dict[str, Any] = {}
    chosen_gate_calibration_summary: dict[str, Any] = dict(chosen_row.get("gate_calibration_summary", {}))
    chosen_variant_metadata = _tdca_variant_metadata(
        dict(chosen_row.get("candidate", {})).get("decoder_variant", DEFAULT_TDCA_LOCAL_DECODER_VARIANT)
    )
    if chosen_row and status == "ok":
        candidate_meets_acceptance = bool(profile_meets_acceptance(dict(chosen_row.get("metrics_median", {}))))
        candidate_data_sufficiency_summary = {
            "session_count": int(len(merged_dataset.manifest_paths)),
            "trial_count": int(len(merged_dataset.trial_segments)),
            "unique_split_fingerprints": int(seed_effective.get("unique_split_fingerprints", 0)),
            "minimum_sessions_for_deployment": int(DEFAULT_TDCA_LOCAL_DATA_DEPLOYMENT_MIN_SESSIONS),
            "current_sessions_sufficient_for_deployment": bool(
                len(merged_dataset.manifest_paths) >= DEFAULT_TDCA_LOCAL_DATA_DEPLOYMENT_MIN_SESSIONS
            ),
        }
        candidate_run_valid_for_deployment = bool(
            candidate_meets_acceptance and candidate_data_sufficiency_summary["current_sessions_sufficient_for_deployment"]
        )
        emit_progress(
            force=True,
            stage="finalize",
            stage_label=_tdca_progress_label("finalize"),
            detail="基于全量数据重建最终 profile",
            run_index=1,
            run_total=3,
        )
        chosen_candidate = dict(chosen_row.get("candidate", {}))
        full_split = RepeatedGroupSplit(
            repeat_index=0,
            train_indices=tuple(range(len(merged_dataset.trial_segments))),
            gate_indices=tuple(range(len(merged_dataset.trial_segments))),
            holdout_indices=tuple(range(len(merged_dataset.trial_segments))),
            fingerprint="full-data",
        )
        full_replay_policy = _split_replay_policy(
            merged_dataset=merged_dataset,
            split=full_split,
            win_sec=float(chosen_candidate.get("win_sec", 3.0)),
            env_preflight=env_preflight,
        )
        chosen_replay_policy = dict(full_replay_policy)
        full_context = _build_candidate_context(
            merged_dataset=merged_dataset,
            split=full_split,
            model_name=DEFAULT_TDCA_LOCAL_MODEL,
            win_sec=float(chosen_candidate.get("win_sec", 3.0)),
            model_params=_default_model_params(
                model_name=DEFAULT_TDCA_LOCAL_MODEL,
                Nh=int(config.Nh),
                delay_steps=int(chosen_candidate.get("delay_steps", 2)),
                n_components=int(chosen_candidate.get("n_components", 2)),
                decoder_variant=str(chosen_candidate.get("decoder_variant", DEFAULT_TDCA_LOCAL_DECODER_VARIANT)),
            ),
            confidence_variant=str(chosen_confidence_variant),
            decoder_compute_backend=str(config.compute_backend),
            gpu_device=int(config.gpu_device),
            gpu_precision=str(config.gpu_precision),
            gpu_warmup=bool(config.gpu_warmup),
            gpu_cache_policy=str(config.gpu_cache_policy),
            control_state_mode=str(config.control_state_mode),
            decision_time_mode=str(config.decision_time_mode),
            async_decision_time_mode=str(config.async_decision_time_mode),
            replay_policy=full_replay_policy,
        )
        final_profile = replace(
            full_context["gate_profile"],
            model_name=DEFAULT_TDCA_LOCAL_MODEL,
            model_params={
                **dict(full_context.get("model_params", {})),
                "state": full_context.get("state_payload"),
            },
            benchmark_metrics=dict(chosen_row.get("metrics_median", {})),
            confidence_variant=str(chosen_confidence_variant),
            training_window_policy=str(full_context.get("training_window_policy", "last_window_only")),
            metadata={
                "task": "tdca-local-opt",
                "env_preflight": env_preflight,
                "split_fingerprints": split_fingerprints,
                "seed_effective": seed_effective,
                "search_preset": str(search_plan.get("search_preset", DEFAULT_TDCA_LOCAL_SEARCH_PRESET)),
                "decision_params": dict(chosen_row.get("decision_params", {})),
                "candidate": dict(chosen_candidate),
                "decoder_variant": str(chosen_variant_metadata["decoder_variant"]),
                "algorithm_alignment": str(chosen_variant_metadata["algorithm_alignment"]),
                "paper_tdca_projection_enabled": bool(chosen_variant_metadata["paper_tdca_projection_enabled"]),
                "confidence_variant": str(chosen_confidence_variant),
                "confidence_training_scheme": str(
                    full_context.get("confidence_training_scheme", DEFAULT_CONFIDENCE_TRAINING_SCHEME)
                ),
                "decision_evidence_variant": DEFAULT_DECISION_EVIDENCE_VARIANT,
                "decision_evidence_raw": "correctness_logit",
                "decision_evidence_reference": "logit(enter_p_th_for_pred_freq)",
                "oof_group_key": str(full_context.get("oof_group_key", "")),
                "oof_group_count": int(full_context.get("oof_group_count", 0) or 0),
                "sample_weight_mode": str(full_context.get("sample_weight_mode", "")),
                "positive_trials": int(full_context.get("positive_trials", 0) or 0),
                "negative_trials": int(full_context.get("negative_trials", 0) or 0),
                "training_window_policy": str(full_context.get("training_window_policy", "last_window_only")),
                "training_latency_sec": float(full_context.get("training_latency_sec", 0.0) or 0.0),
                "analysis_latency_sec": float(full_context.get("analysis_latency_sec", 0.0) or 0.0),
                "effective_raw_window_sec": float(full_context.get("effective_raw_window_sec", 0.0) or 0.0),
                "paper_alignment_level": str(full_context.get("paper_alignment_level", "partial")),
                "tune_summary": dict(full_context.get("tune_summary", {})),
                "gate_calibration_summary": dict(full_context.get("gate_calibration_summary", {})),
                "confidence_diagnostics_board": [dict(item) for item in chosen_confidence_diagnostics_board],
                "decision_bottleneck_summary": dict(chosen_decision_bottleneck_summary),
                "data_sufficiency_summary": dict(candidate_data_sufficiency_summary),
                "run_valid_for_deployment": bool(candidate_run_valid_for_deployment),
                "decision_search_target": "tune_split",
                "final_selection_target": "holdout_split",
                "effective_replay_backend": str(full_replay_policy.get("effective_replay_backend", "cpu")),
                "gpu_replay_speedup": float(full_replay_policy.get("gpu_replay_speedup", 0.0)),
                "gpu_replay_eligible": bool(full_replay_policy.get("gpu_replay_eligible", False)),
                "gpu_replay_reason": str(full_replay_policy.get("gpu_replay_reason", "")),
                "chosen_model_rationale": str(chosen_model_rationale),
            },
            recommended_for_realtime=True,
        )
        save_profile(final_profile, resolved_profile_path)
        chosen_profile = final_profile
        chosen_profile_path = str(resolved_profile_path)
        profile_saved = True
        chosen_gate_calibration_summary = dict(full_context.get("gate_calibration_summary", {}))
        emit_progress(
            force=True,
            stage="finalize",
            stage_label=_tdca_progress_label("finalize"),
            detail="主 profile 已保存，开始导出 profile_v2",
            run_index=2,
            run_total=3,
            profile_path=chosen_profile_path,
        )

        gate_payload = dict(full_context["gate_model"].to_payload())
        per_freq_gate = {}
        per_freq_thresholds = dict(final_profile.frequency_specific_thresholds or {})
        for freq_key, payload in dict(gate_payload.get("per_freq", {})).items():
            merged_payload = dict(payload)
            merged_payload.update(dict(per_freq_thresholds.get(str(freq_key), {})))
            per_freq_gate[str(freq_key)] = merged_payload
        profile_v2 = build_profile_v2(
            base_profile=final_profile,
            per_freq_gate=per_freq_gate,
            metrics=dict(chosen_row.get("metrics_median", {})),
            feature_names=tuple(getattr(full_context["gate_model"], "feature_names", DEFAULT_GATE_FEATURES)),
            evidence=dict(chosen_row.get("decision_params", {})),
            refractory_sec=float(dict(chosen_row.get("decision_params", {})).get("refractory_sec", 0.8)),
        )
        profile_v2_output = resolved_profile_v2_path
        atomic_write_text(profile_v2_output, json_dumps(json_safe(profile_v2.to_payload())) + "\n")
        profile_v2_saved = True
        profile_v2_path = str(profile_v2_output)
        publish_deployed_profile(
            source_profile=resolved_profile_path,
            source_profile_v2=profile_v2_output,
            run_dir=report_dir,
            task="tdca-local-opt",
            run_tag=run_tag,
            report_json=Path(report_paths["report_json"]).expanduser().resolve(),
            extra_metadata={
                "chosen_model": str(final_profile.model_name),
                "decoder_variant": str(chosen_variant_metadata["decoder_variant"]),
            },
        )
        emit_progress(
            force=True,
            stage="finalize",
            stage_label=_tdca_progress_label("finalize"),
            detail="profile_v2 已保存，开始生成报告",
            run_index=3,
            run_total=3,
            profile_path=chosen_profile_path,
        )
    else:
        emit_progress(
            force=True,
            stage="finalize",
            stage_label=_tdca_progress_label("finalize"),
            detail="本次运行未保存 profile，仍将输出报告",
            run_index=3,
            run_total=3,
        )

    chosen_async_metrics = dict(chosen_row.get("metrics_median", {}))
    chosen_metrics_4class = dict(chosen_row.get("metrics_4class_median", {}))
    chosen_metrics_2class = dict(chosen_row.get("metrics_2class_median", {}))
    chosen_candidate_key = str(chosen_row.get("candidate_key", ""))
    chosen_confidence_variant = _resolved_confidence_variant(chosen_row)
    chosen_confidence_training_scheme = str(
        chosen_row.get("confidence_training_scheme", DEFAULT_CONFIDENCE_TRAINING_SCHEME)
    )
    chosen_oof_group_key = str(chosen_row.get("oof_group_key", ""))
    chosen_oof_group_count = int(chosen_row.get("oof_group_count", 0) or 0)
    chosen_sample_weight_mode = str(chosen_row.get("sample_weight_mode", ""))
    chosen_positive_trials = int(chosen_row.get("positive_trials", 0) or 0)
    chosen_negative_trials = int(chosen_row.get("negative_trials", 0) or 0)
    chosen_training_latency_sec = float(chosen_row.get("training_latency_sec", 0.0) or 0.0)
    chosen_analysis_latency_sec = float(chosen_row.get("analysis_latency_sec", 0.0) or 0.0)
    chosen_effective_raw_window_sec = float(
        chosen_row.get("effective_raw_window_sec", dict(chosen_row.get("candidate", {})).get("win_sec", 0.0)) or 0.0
    )
    chosen_paper_alignment_level = str(chosen_row.get("paper_alignment_level", "partial"))
    chosen_tune_summary = dict(chosen_row.get("tune_summary", {}))
    chosen_meets_acceptance = bool(profile_meets_acceptance(chosen_async_metrics)) if chosen_row else False
    data_sufficiency_summary = {
        "session_count": int(len(merged_dataset.manifest_paths)),
        "trial_count": int(len(merged_dataset.trial_segments)),
        "unique_split_fingerprints": int(seed_effective.get("unique_split_fingerprints", 0)),
        "minimum_sessions_for_deployment": int(DEFAULT_TDCA_LOCAL_DATA_DEPLOYMENT_MIN_SESSIONS),
        "current_sessions_sufficient_for_deployment": bool(len(merged_dataset.manifest_paths) >= DEFAULT_TDCA_LOCAL_DATA_DEPLOYMENT_MIN_SESSIONS),
    }
    run_valid_for_deployment = bool(
        status == "ok"
        and profile_saved
        and chosen_meets_acceptance
        and bool(data_sufficiency_summary["current_sessions_sufficient_for_deployment"])
    )
    report_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "tdca_local_opt",
        "task": "tdca-local-opt",
        "search_preset": str(search_plan.get("search_preset", DEFAULT_TDCA_LOCAL_SEARCH_PRESET)),
        "status": status,
        "status_reasons": list(status_reasons),
        "dataset_manifest_session1": str(Path(config.dataset_manifest_session1).expanduser().resolve()),
        "selected_dataset_manifests_session1": [str(path) for path in merged_dataset.manifest_paths],
        "selected_dataset_count_session1": int(len(merged_dataset.manifest_paths)),
        "report_path": str(Path(report_paths["report_json"]).expanduser().resolve()),
        "report_dir": str(report_dir),
        "run_log_path": str(run_log_path),
        "progress_snapshot_path": str(progress_snapshot_path),
        "sampling_rate": int(merged_dataset.sampling_rate),
        "freqs": [float(item) for item in merged_dataset.freqs],
        "board_eeg_channels": [int(item) for item in merged_dataset.board_eeg_channels],
        "quality_summary_session1": [dict(item) for item in merged_dataset.quality_rows],
        "quality_total_trials_session1": int(len(merged_dataset.trial_segments)),
        "quality_kept_trials_session1": int(len(merged_dataset.trial_segments)),
        "quality_filter": {},
        "split_fingerprints": split_fingerprints,
        "seed_effective": seed_effective,
        "env_preflight": env_preflight,
        "decoder_variant": str(chosen_variant_metadata["decoder_variant"]),
        "algorithm_alignment": str(chosen_variant_metadata["algorithm_alignment"]),
        "paper_tdca_projection_enabled": bool(chosen_variant_metadata["paper_tdca_projection_enabled"]),
        "confidence_variant": str(chosen_confidence_variant),
        "confidence_training_scheme": str(chosen_confidence_training_scheme),
        "decision_evidence_variant": DEFAULT_DECISION_EVIDENCE_VARIANT,
        "decision_evidence_raw": "correctness_logit",
        "decision_evidence_reference": "logit(enter_p_th_for_pred_freq)",
        "oof_group_key": str(chosen_oof_group_key),
        "oof_group_count": int(chosen_oof_group_count),
        "sample_weight_mode": str(chosen_sample_weight_mode),
        "positive_trials": int(chosen_positive_trials),
        "negative_trials": int(chosen_negative_trials),
        "training_window_policy": str(chosen_row.get("training_window_policy", "last_window_only")),
        "training_latency_sec": float(chosen_training_latency_sec),
        "analysis_latency_sec": float(chosen_analysis_latency_sec),
        "effective_raw_window_sec": float(chosen_effective_raw_window_sec),
        "paper_alignment_level": str(chosen_paper_alignment_level),
        "effective_replay_backend": str(chosen_replay_policy.get("effective_replay_backend", "cpu")),
        "gpu_replay_speedup": float(chosen_replay_policy.get("gpu_replay_speedup", env_preflight.get("gpu_replay_speedup", 0.0))),
        "gpu_replay_eligible": bool(chosen_replay_policy.get("gpu_replay_eligible", False)),
        "gpu_replay_reason": str(chosen_replay_policy.get("gpu_replay_reason", "")),
        "decision_search_target": "tune_split",
        "final_selection_target": "holdout_split",
        "tdca_search_board": tdca_search_board,
        "gate_exit_search_board": gate_exit_by_candidate.get(chosen_candidate_key, []),
        "decision_search_board": decision_aggregate_rows,
        "holdout_selection_board": final_candidate_rows,
        "variant_summary": variant_summary,
        "tune_summary": dict(chosen_tune_summary),
        "tune_rows_valid": bool(chosen_tune_summary.get("valid", False)),
        "confidence_diagnostics_board": [dict(item) for item in chosen_confidence_diagnostics_board],
        "decision_bottleneck_summary": dict(chosen_decision_bottleneck_summary),
        "ranking_boards": {
            "end_to_end": final_candidate_rows,
            "classifier_only": [],
        },
        "model_results": final_candidate_rows,
        "chosen_model": DEFAULT_TDCA_LOCAL_MODEL if chosen_row else "",
        "chosen_rank": 1 if chosen_row else 0,
        "async_metrics": chosen_async_metrics,
        "chosen_metrics": chosen_async_metrics,
        "chosen_async_metrics": chosen_async_metrics,
        "chosen_metrics_4class": chosen_metrics_4class,
        "chosen_metrics_2class": chosen_metrics_2class,
        "chosen_metrics_5class": None,
        "gate_calibration_valid": bool(chosen_row.get("gate_calibration_valid", False)),
        "gate_calibration_summary": dict(chosen_gate_calibration_summary),
        "min_gate_control_rows": int(chosen_row.get("min_gate_control_rows", 0) or 0),
        "min_gate_idle_rows": int(chosen_row.get("min_gate_idle_rows", 0) or 0),
        "enter_p_th_median": float(chosen_row.get("enter_p_th_median", 0.65) or 0.65),
        "exit_p_th_median": float(chosen_row.get("exit_p_th_median", 0.30) or 0.30),
        "enter_logit_th_median": float(chosen_row.get("enter_logit_th_median", 0.0) or 0.0),
        "exit_logit_th_median": float(chosen_row.get("exit_logit_th_median", 0.0) or 0.0),
        "chosen_profile_path": chosen_profile_path,
        "chosen_meets_acceptance": bool(chosen_meets_acceptance),
        "run_valid_for_deployment": bool(run_valid_for_deployment),
        "data_sufficiency_summary": dict(data_sufficiency_summary),
        "profile_saved": bool(profile_saved),
        "profile_v2_saved": bool(profile_v2_saved),
        "profile_v2_path": profile_v2_path,
        "recommended_model": DEFAULT_TDCA_LOCAL_MODEL if chosen_row else "",
        "gate_policy": DEFAULT_GATE_POLICY,
        "control_state_mode": str(config.control_state_mode),
        "decision_time_mode": str(config.decision_time_mode),
        "async_decision_time_mode": str(config.async_decision_time_mode),
        "chosen_model_rationale": str(chosen_model_rationale),
        "baseline_opening": baseline_opening_report,
        "baseline_seal": baseline_seal_report,
        "error_attribution_board": error_attribution_board,
        "contrast_error_board": contrast_error_board,
    }

    report_paths["report_dir"].mkdir(parents=True, exist_ok=True)
    atomic_write_text(report_paths["run_config"], json_dumps(json_safe(asdict(config))) + "\n")
    atomic_write_text(
        report_paths["selection_snapshot"],
        json_dumps(
            json_safe(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "selected_dataset_manifests_session1": [str(path) for path in merged_dataset.manifest_paths],
                    "search_preset": str(search_plan.get("search_preset", DEFAULT_TDCA_LOCAL_SEARCH_PRESET)),
                    "split_fingerprints": split_fingerprints,
                    "seed_effective": seed_effective,
                    "decoder_variant": str(chosen_variant_metadata["decoder_variant"]),
                    "algorithm_alignment": str(chosen_variant_metadata["algorithm_alignment"]),
                    "paper_tdca_projection_enabled": bool(chosen_variant_metadata["paper_tdca_projection_enabled"]),
                    "confidence_variant": str(chosen_confidence_variant),
                    "confidence_training_scheme": str(chosen_confidence_training_scheme),
                    "decision_evidence_variant": DEFAULT_DECISION_EVIDENCE_VARIANT,
                    "decision_evidence_raw": "correctness_logit",
                    "decision_evidence_reference": "logit(enter_p_th_for_pred_freq)",
                    "oof_group_key": str(chosen_oof_group_key),
                    "oof_group_count": int(chosen_oof_group_count),
                    "sample_weight_mode": str(chosen_sample_weight_mode),
                    "positive_trials": int(chosen_positive_trials),
                    "negative_trials": int(chosen_negative_trials),
                    "training_window_policy": str(chosen_row.get("training_window_policy", "last_window_only")),
                    "training_latency_sec": float(chosen_training_latency_sec),
                    "analysis_latency_sec": float(chosen_analysis_latency_sec),
                    "effective_raw_window_sec": float(chosen_effective_raw_window_sec),
                    "paper_alignment_level": str(chosen_paper_alignment_level),
                    "decision_search_target": "tune_split",
                    "final_selection_target": "holdout_split",
                    "variant_summary": variant_summary,
                    "tune_summary": dict(chosen_tune_summary),
                    "gate_calibration_summary": dict(chosen_gate_calibration_summary),
                    "confidence_diagnostics_board": [dict(item) for item in chosen_confidence_diagnostics_board],
                    "decision_bottleneck_summary": dict(chosen_decision_bottleneck_summary),
                    "holdout_selection_board": final_candidate_rows,
                    "error_attribution_board": error_attribution_board,
                    "contrast_error_board": contrast_error_board,
                    "run_valid_for_deployment": bool(run_valid_for_deployment),
                    "data_sufficiency_summary": dict(data_sufficiency_summary),
                    "chosen_model_rationale": str(chosen_model_rationale),
                }
            )
        )
        + "\n",
    )
    atomic_write_text(report_paths["report_json"], json_dumps(json_safe(report_payload)) + "\n")
    atomic_write_text(report_paths["report_md"], _render_markdown(report_payload))
    log(f"[tdca-local-opt] report saved: {report_paths['report_json']}")
    if chosen_profile is not None:
        log(f"[tdca-local-opt] profile saved: {chosen_profile_path}")
    emit_progress(
        force=True,
        stage="complete",
        stage_label=_tdca_progress_label("complete"),
        detail="TDCA 本地异步优化完成",
        run_index=1,
        run_total=1,
        report_path=str(report_paths["report_json"]),
        profile_path=str(chosen_profile_path or resolved_profile_path),
        progress_percent=100,
    )
    return report_payload
