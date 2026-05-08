from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from dataclasses import replace
from datetime import datetime
from itertools import combinations, product
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.async_fbcca_idle_standalone import (
    TrialSpec,
    compute_classification_metrics,
    default_profile,
    evaluate_decoder_on_trials_v2,
    extract_window_batch,
    create_decoder,
    json_dumps,
    json_safe,
    load_decoder_from_profile,
    load_profile,
)
from ssvep_core.dataset import save_collection_dataset_bundle
from ssvep_core.external_beta_dataset import (
    BETA_FREQS,
    BETA_REQUIRED_CHANNELS,
    build_beta_segments,
    load_beta_subject,
    resolve_beta_command_frequencies,
)
from ssvep_core.external_wang2016_dataset import (
    WANG2016_FREQS,
    WANG2016_REQUIRED_CHANNELS,
    build_wang2016_segments,
    load_wang2016_subject,
    resolve_wang2016_command_frequencies,
)
from ssvep_core.fast_fbcca_pretrain import FastFBCCAPretrainConfig, run_fast_fbcca_personalization
from ssvep_core.fbcca_threshold_pretrain import FBCCAThresholdPretrainConfig, run_fbcca_threshold_pretrain
from ssvep_core.stimulus_profiles import frame_lock_frequency_report


DEFAULT_FREQS = (9.8, 12.0, 14.8, 15.8)
DEFAULT_DATASETS = ("wang2016", "beta")
DEFAULT_METHODS = ("zero_shot_default", "fast_fbcca", "threshold_pretrain", "fbcca_lda5", "fbcca_ridge5")
SUPPORTED_SHORT_PRETRAIN_METHODS = ("itcca5", "ecca5", "trca5", "trca_r5", "tdca5")
SUPPORTED_METHODS = DEFAULT_METHODS + SUPPORTED_SHORT_PRETRAIN_METHODS
DEFAULT_CALIBRATION_BLOCKS = (1, 2, 3)
DEFAULT_IDLE_MULTIPLIERS = (1.0, 2.0)
DEFAULT_FAST_WIN_SEC_CANDIDATES = (1.5, 2.0, 2.5)
DEFAULT_FAST_TEMPLATE_WEIGHT_CANDIDATES = (0.15, 0.25, 0.35)
DEFAULT_THRESHOLD_WIN_SEC_CANDIDATES = (1.5, 2.0, 2.5, 3.0)
DEFAULT_THRESHOLD_GATE_POLICIES = ("balanced", "speed")
DEFAULT_THRESHOLD_MIN_ENTER_CANDIDATES = (1, 2)
DEFAULT_THRESHOLD_MIN_EXIT_CANDIDATES = (1, 2)
DEFAULT_THRESHOLD_CONTROL_STATE_MODES = ("unified", "frequency-specific-threshold")
DEFAULT_CLASSIFIER_WIN_SEC_CANDIDATES = (1.5, 2.0, 2.5)
DEFAULT_CLASSIFIER_MIN_ENTER_CANDIDATES = (1, 2)
DEFAULT_CLASSIFIER_MAX_GAP_CANDIDATES = (0,)
DEFAULT_CLASSIFIER_THRESHOLD_POLICY = "balanced"
CLASSIFIER_THRESHOLD_POLICIES = ("balanced", "balanced_recall_guard")
DEFAULT_CLASSIFIER_THRESHOLD_MIN_CONTROL_RECALL = 0.80
DEFAULT_CLASSIFIER_IDLE_FP_BUDGET_PER_MIN = 1.0
DEFAULT_CLASSIFIER_IDLE_SELECTED_WINDOWS_BUDGET_PER_MIN = 6.0
DEFAULT_RIDGE_L2_CANDIDATES = (0.03, 0.1, 0.3, 1.0, 3.0)
DEFAULT_MAX_SPLITS_PER_SUBJECT = 6
DEFAULT_STEP_SEC = 0.25
CLASSIFIER_DERIVED_FEATURE_NAMES = (
    "top1_score",
    "top2_score",
    "margin",
    "ratio",
    "normalized_top1",
    "score_entropy",
)


@dataclass(frozen=True)
class ScoreMethodSpec:
    method_name: str
    decoder_name: str
    score_source_name: str
    classifier_kind: str
    decoder_model_params: dict[str, Any]
    fit_decoder: bool
    extra_required_win_sec: float = 0.0


SCORE_METHOD_SPECS: dict[str, ScoreMethodSpec] = {
    "fbcca_lda5": ScoreMethodSpec(
        method_name="fbcca_lda5",
        decoder_name="fbcca",
        score_source_name="fbcca",
        classifier_kind="lda",
        decoder_model_params={"Nh": 5, "subband_weight_mode": "chen_fixed"},
        fit_decoder=False,
    ),
    "fbcca_ridge5": ScoreMethodSpec(
        method_name="fbcca_ridge5",
        decoder_name="fbcca",
        score_source_name="fbcca",
        classifier_kind="ridge",
        decoder_model_params={"Nh": 5, "subband_weight_mode": "chen_fixed"},
        fit_decoder=False,
    ),
    "itcca5": ScoreMethodSpec(
        method_name="itcca5",
        decoder_name="itcca",
        score_source_name="itcca",
        classifier_kind="ridge",
        decoder_model_params={},
        fit_decoder=True,
    ),
    "ecca5": ScoreMethodSpec(
        method_name="ecca5",
        decoder_name="ecca_paper",
        score_source_name="ecca",
        classifier_kind="ridge",
        decoder_model_params={},
        fit_decoder=True,
    ),
    "trca5": ScoreMethodSpec(
        method_name="trca5",
        decoder_name="trca",
        score_source_name="trca",
        classifier_kind="ridge",
        decoder_model_params={},
        fit_decoder=True,
    ),
    "trca_r5": ScoreMethodSpec(
        method_name="trca_r5",
        decoder_name="trca_r",
        score_source_name="trca_r",
        classifier_kind="ridge",
        decoder_model_params={},
        fit_decoder=True,
    ),
    "tdca5": ScoreMethodSpec(
        method_name="tdca5",
        decoder_name="tdca",
        score_source_name="tdca",
        classifier_kind="ridge",
        decoder_model_params={"decoder_variant": "tdca_paper_aligned"},
        fit_decoder=True,
        extra_required_win_sec=0.14,
    ),
}


@dataclass(frozen=True)
class ExternalSubjectSpec:
    dataset: str
    subject: str
    mat_path: Path
    freqs: tuple[float, float, float, float]
    channel_loc_path: Optional[Path] = None


@dataclass(frozen=True)
class SplitPlan:
    subject: str
    dataset: str
    split_index: int
    seed: int
    calibration_blocks: tuple[int, ...]
    holdout_blocks: tuple[int, ...]


@dataclass(frozen=True)
class ScoredTrial:
    trial: TrialSpec
    score_matrix: np.ndarray
    feature_matrix: np.ndarray
    duration_sec: float


@dataclass(frozen=True)
class FBCCALDA5Model:
    freqs: tuple[float, float, float, float]
    labels: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    class_means: np.ndarray
    pooled_var: np.ndarray
    command_confidence_th: float
    fit_summary: dict[str, Any]


@dataclass(frozen=True)
class FBCCARidge5Model:
    freqs: tuple[float, float, float, float]
    labels: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    weights: np.ndarray
    l2: float
    command_confidence_th: float
    fit_summary: dict[str, Any]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(json_safe(payload)) + "\n", encoding="utf-8")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _classifier_quality_score(metrics: Mapping[str, Any]) -> float:
    async_macro = _safe_float(metrics.get("async_macro_f1_5class"), 0.0)
    fixed_macro = _safe_float(metrics.get("fixed_macro_f1_5class"), 0.0)
    async_acc = _safe_float(metrics.get("async_acc_5class"), 0.0)
    return float(0.65 * async_macro + 0.25 * fixed_macro + 0.10 * async_acc)


def _classifier_rank_key(metrics: Mapping[str, Any], *, tie_breaker: float = 0.0) -> tuple[float, ...]:
    control_recall = _safe_float(metrics.get("control_recall"), 0.0)
    control_recall_at_2s = _safe_float(metrics.get("control_recall_at_2s"), 0.0)
    control_recall_at_2p5s = _safe_float(metrics.get("control_recall_at_2.5s"), 0.0)
    control_recall_at_3s = _safe_float(metrics.get("control_recall_at_3s"), 0.0)
    recall_shortfall = max(0.0, float(DEFAULT_CLASSIFIER_THRESHOLD_MIN_CONTROL_RECALL) - float(control_recall))
    idle_fp = _safe_float(metrics.get("idle_fp_per_min"), float("inf"))
    idle_selected = _safe_float(metrics.get("idle_selected_windows_per_min"), float("inf"))
    idle_fp_excess = max(0.0, idle_fp - float(DEFAULT_CLASSIFIER_IDLE_FP_BUDGET_PER_MIN))
    idle_selected_excess = max(
        0.0,
        idle_selected - float(DEFAULT_CLASSIFIER_IDLE_SELECTED_WINDOWS_BUDGET_PER_MIN),
    )
    severe_idle_fp = 1 if idle_fp_excess > 1e-12 else 0
    severe_idle_selected = 1 if idle_selected_excess > 1e-12 else 0
    return (
        float(severe_idle_fp),
        float(severe_idle_selected),
        float(recall_shortfall),
        float(idle_fp_excess),
        float(idle_selected_excess),
        -_classifier_quality_score(metrics),
        -float(control_recall),
        -float(control_recall_at_2p5s),
        -float(control_recall_at_3s),
        -float(control_recall_at_2s),
        float(idle_fp),
        float(idle_selected),
        _safe_float(metrics.get("detection_latency_s"), float("inf")),
        -_safe_float(metrics.get("fixed_macro_f1_5class"), 0.0),
        -_safe_float(metrics.get("fixed_acc_5class"), 0.0),
        float(tie_breaker),
    )


def _classifier_recall_guard_rank_key(metrics: Mapping[str, Any], *, tie_breaker: float = 0.0) -> tuple[float, ...]:
    idle_fp = _safe_float(metrics.get("idle_fp_per_min"), float("inf"))
    idle_selected = _safe_float(metrics.get("idle_selected_windows_per_min"), float("inf"))
    control_recall = _safe_float(metrics.get("control_recall"), 0.0)
    detection_latency = _safe_float(metrics.get("detection_latency_s"), float("inf"))
    idle_fp_excess = max(0.0, idle_fp - float(DEFAULT_CLASSIFIER_IDLE_FP_BUDGET_PER_MIN))
    idle_selected_excess = max(
        0.0,
        idle_selected - float(DEFAULT_CLASSIFIER_IDLE_SELECTED_WINDOWS_BUDGET_PER_MIN),
    )
    all_idle_like = (
        control_recall <= 1e-12
        or not np.isfinite(float(detection_latency))
    )
    return (
        float(1 if idle_fp_excess > 1e-12 else 0),
        float(1 if all_idle_like else 0),
        -float(control_recall),
        -_safe_float(metrics.get("control_recall_at_2.5s"), 0.0),
        -_safe_float(metrics.get("control_recall_at_3s"), 0.0),
        -_safe_float(metrics.get("async_macro_f1_5class"), 0.0),
        float(idle_selected_excess),
        float(idle_fp_excess),
        float(idle_fp),
        float(idle_selected),
        float(detection_latency),
        -_safe_float(metrics.get("fixed_macro_f1_5class"), 0.0),
        -_safe_float(metrics.get("fixed_acc_5class"), 0.0),
        float(tie_breaker),
    )


def _classifier_threshold_rank_key(
    metrics: Mapping[str, Any],
    *,
    policy: str,
    tie_breaker: float = 0.0,
) -> tuple[float, ...]:
    normalized = str(policy or DEFAULT_CLASSIFIER_THRESHOLD_POLICY).strip().lower()
    if normalized == "balanced_recall_guard":
        return _classifier_recall_guard_rank_key(metrics, tie_breaker=tie_breaker)
    if normalized != "balanced":
        raise ValueError(
            "classifier threshold policy must be one of "
            f"{','.join(CLASSIFIER_THRESHOLD_POLICIES)}; got {policy}"
        )
    return _classifier_rank_key(metrics, tie_breaker=tie_breaker)


def _csv_float_tuple(raw: str | None, *, default: Sequence[float]) -> tuple[float, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple(float(value) for value in default)
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one comma-separated float value")
    return values


def _csv_int_tuple(raw: str | None, *, default: Sequence[int]) -> tuple[int, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple(int(value) for value in default)
    values = tuple(int(float(item.strip())) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one comma-separated integer value")
    return values


def _csv_str_tuple(raw: str | None, *, default: Sequence[str]) -> tuple[str, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple(str(value) for value in default)
    values = tuple(str(item).strip() for item in text.split(",") if str(item).strip())
    if not values:
        raise ValueError("expected at least one comma-separated string value")
    return values


def _csv_dataset_tuple(raw: str | None) -> tuple[str, ...]:
    values = tuple(str(item).strip().lower() for item in str(raw or "").split(",") if str(item).strip())
    if not values:
        return DEFAULT_DATASETS
    invalid = [value for value in values if value not in DEFAULT_DATASETS]
    if invalid:
        raise ValueError(f"datasets must be drawn from {','.join(DEFAULT_DATASETS)}; got {','.join(invalid)}")
    ordered: list[str] = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return tuple(ordered)


def _csv_method_tuple(raw: str | None) -> tuple[str, ...]:
    values = tuple(str(item).strip().lower() for item in str(raw or "").split(",") if str(item).strip())
    if not values:
        return DEFAULT_METHODS
    invalid = [value for value in values if value not in SUPPORTED_METHODS]
    if invalid:
        raise ValueError(f"methods must be drawn from {','.join(SUPPORTED_METHODS)}; got {','.join(invalid)}")
    ordered: list[str] = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return tuple(ordered)


def _parse_classifier_threshold_policy(raw: str | None) -> str:
    policy = str(raw or DEFAULT_CLASSIFIER_THRESHOLD_POLICY).strip().lower()
    if not policy:
        policy = DEFAULT_CLASSIFIER_THRESHOLD_POLICY
    if policy not in CLASSIFIER_THRESHOLD_POLICIES:
        raise ValueError(
            "classifier threshold policy must be one of "
            f"{','.join(CLASSIFIER_THRESHOLD_POLICIES)}; got {raw}"
        )
    return policy


def _score_method_spec(method_name: str) -> ScoreMethodSpec:
    normalized = str(method_name).strip().lower()
    spec = SCORE_METHOD_SPECS.get(normalized)
    if spec is None:
        raise ValueError(f"unsupported score method: {method_name}")
    return spec


def _score_method_cache_namespace(method_name: str) -> str:
    spec = _score_method_spec(method_name)
    if spec.fit_decoder:
        return str(method_name).strip().lower()
    return str(spec.score_source_name).strip().lower()


def _method_effective_window_sec(*, method_name: str, win_sec: float, sampling_rate: int) -> float:
    spec = _score_method_spec(method_name)
    fs = max(int(sampling_rate), 1)
    analysis_samples = max(1, int(round(float(win_sec) * float(fs))))
    extra_samples = max(0, int(round(float(spec.extra_required_win_sec) * float(fs))))
    return float(analysis_samples + extra_samples) / float(fs)


def _method_latency_window_sec(*, method_name: str, win_sec: float, sampling_rate: int) -> float:
    return _method_effective_window_sec(
        method_name=str(method_name),
        win_sec=float(win_sec),
        sampling_rate=int(sampling_rate),
    )


def _classifier_recipe_id(*, win_sec: float, min_enter_windows: int, max_gap_windows: int = 0) -> str:
    base = f"win{float(win_sec):g}_me{int(min_enter_windows)}"
    if int(max_gap_windows) > 0:
        base = f"{base}_gap{int(max_gap_windows)}"
    return base.replace(".", "p")


def _score_method_candidate_pairs(
    *,
    method_name: str,
    win_sec_candidates: Sequence[float],
    min_enter_candidates: Sequence[int],
    max_supported_win_sec: float,
    sampling_rate: int,
) -> list[tuple[float, int]]:
    supported_win_secs = [
        float(win_sec)
        for win_sec in win_sec_candidates
        if _method_effective_window_sec(
            method_name=method_name,
            win_sec=float(win_sec),
            sampling_rate=int(sampling_rate),
        )
        <= float(max_supported_win_sec) + 1e-9
    ]
    return [
        (float(win_sec), int(min_enter))
        for win_sec, min_enter in product(supported_win_secs, min_enter_candidates)
    ]


def _build_score_method_decoder(
    *,
    method_name: str,
    freqs: Sequence[float],
    sampling_rate: int,
    win_sec: float,
    step_sec: float,
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
) -> Any:
    spec = _score_method_spec(method_name)
    return create_decoder(
        spec.decoder_name,
        sampling_rate=int(sampling_rate),
        freqs=freqs,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        model_params=dict(spec.decoder_model_params),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=True,
    )


def _parse_subject_whitelist(raw: str | None) -> tuple[tuple[str, str], ...]:
    text = str(raw or "").strip()
    if not text:
        return ()
    entries: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in text.split(","):
        item = str(token).strip()
        if not item:
            continue
        dataset = "*"
        subject = item
        if ":" in item:
            dataset_part, subject_part = item.split(":", 1)
            dataset = str(dataset_part).strip().lower() or "*"
            subject = str(subject_part).strip()
        if dataset != "*" and dataset not in DEFAULT_DATASETS:
            raise ValueError(f"subject whitelist dataset must be one of {','.join(DEFAULT_DATASETS)}; got {dataset}")
        subject_key = str(subject).strip().upper()
        if not subject_key:
            raise ValueError("subject whitelist subject token must be non-empty")
        key = (dataset, subject_key)
        if key in seen:
            continue
        seen.add(key)
        entries.append(key)
    return tuple(entries)


def _subject_allowed(
    dataset: str,
    subject: str,
    subject_whitelist: Sequence[tuple[str, str]] = (),
) -> bool:
    if not subject_whitelist:
        return True
    dataset_key = str(dataset).strip().lower()
    subject_key = str(subject).strip().upper()
    for allowed_dataset, allowed_subject in subject_whitelist:
        if allowed_subject != subject_key:
            continue
        if allowed_dataset in {"*", dataset_key}:
            return True
    return False


def _freq_token(freqs: Sequence[float]) -> str:
    return "_".join(f"{float(freq):g}".replace(".", "p") for freq in freqs)


def _trial_sort_key(item: tuple[TrialSpec, np.ndarray]) -> tuple[int, int, str]:
    trial, _segment = item
    return (int(trial.block_index), int(trial.trial_id), str(trial.label))


def _count_segments(segments: Sequence[tuple[TrialSpec, np.ndarray]], freqs: Sequence[float]) -> dict[str, Any]:
    per_freq = {f"{float(freq):g}": 0 for freq in freqs}
    idle = 0
    blocks: set[int] = set()
    for trial, _segment in segments:
        blocks.add(int(trial.block_index))
        if trial.expected_freq is None:
            idle += 1
            continue
        key = f"{float(trial.expected_freq):g}"
        if key in per_freq:
            per_freq[key] += 1
    return {
        "total": int(len(segments)),
        "control": int(sum(per_freq.values())),
        "idle": int(idle),
        "per_freq": per_freq,
        "block_count": int(len(blocks)),
        "blocks": sorted(int(block) for block in blocks),
    }


def _collection_duration_sec(segments: Sequence[tuple[TrialSpec, np.ndarray]], sampling_rate: int) -> float:
    fs = max(int(sampling_rate), 1)
    samples = sum(int(np.asarray(segment).shape[0]) for _trial, segment in segments)
    return float(samples) / float(fs)


def _max_supported_win_sec(segments: Sequence[tuple[TrialSpec, np.ndarray]], sampling_rate: int) -> float:
    fs = max(int(sampling_rate), 1)
    lengths = [float(np.asarray(segment).shape[0]) / float(fs) for _trial, segment in segments if np.asarray(segment).ndim == 2]
    if not lengths:
        return 0.0
    return float(min(lengths))


def _required_channel_names(dataset: str) -> tuple[str, ...]:
    if str(dataset) == "wang2016":
        return tuple(str(name) for name in WANG2016_REQUIRED_CHANNELS)
    if str(dataset) == "beta":
        return tuple(str(name) for name in BETA_REQUIRED_CHANNELS)
    raise ValueError(f"unsupported dataset: {dataset}")


def enumerate_external_subjects(
    *,
    datasets: Sequence[str],
    freqs: Sequence[float],
    wang_raw_dir: Path,
    wang_channels_loc: Path,
    beta_raw_dir: Path,
    subject_limit_per_dataset: int = 0,
    subject_whitelist: Sequence[tuple[str, str]] = (),
) -> list[ExternalSubjectSpec]:
    requested_freqs = tuple(float(value) for value in freqs)
    if len(requested_freqs) != 4:
        raise ValueError("freqs must contain exactly 4 values")
    rows: list[ExternalSubjectSpec] = []
    dataset_limit = max(int(subject_limit_per_dataset), 0)
    if "wang2016" in datasets:
        files = sorted(Path(wang_raw_dir).expanduser().resolve().glob("S*.mat"))
        filtered_files = [path for path in files if _subject_allowed("wang2016", path.stem.upper(), subject_whitelist)]
        if dataset_limit > 0:
            filtered_files = filtered_files[:dataset_limit]
        for path in filtered_files:
            rows.append(
                ExternalSubjectSpec(
                    dataset="wang2016",
                    subject=path.stem.upper(),
                    mat_path=path,
                    channel_loc_path=Path(wang_channels_loc).expanduser().resolve(),
                    freqs=requested_freqs,  # type: ignore[arg-type]
                )
            )
    if "beta" in datasets:
        files = sorted(Path(beta_raw_dir).expanduser().resolve().glob("S*.mat"))
        filtered_files = [path for path in files if _subject_allowed("beta", path.stem.upper(), subject_whitelist)]
        if dataset_limit > 0:
            filtered_files = filtered_files[:dataset_limit]
        for path in filtered_files:
            rows.append(
                ExternalSubjectSpec(
                    dataset="beta",
                    subject=path.stem.upper(),
                    mat_path=path,
                    freqs=requested_freqs,  # type: ignore[arg-type]
                )
            )
    return rows


def load_external_subject_segments(spec: ExternalSubjectSpec) -> tuple[int, list[tuple[TrialSpec, np.ndarray]], dict[str, Any]]:
    freqs = tuple(float(value) for value in spec.freqs)
    if spec.dataset == "wang2016":
        if spec.channel_loc_path is None:
            raise ValueError("wang2016 requires channel_loc_path")
        subject = load_wang2016_subject(spec.mat_path, spec.channel_loc_path)
        resolved_freqs, target_index = resolve_wang2016_command_frequencies(freqs)
        segments = build_wang2016_segments(
            subject,
            freqs=resolved_freqs,
            include_hard_idle=True,
            include_pre_stim_idle=False,
        )
        metadata = {
            "dataset": "wang2016",
            "subject": subject.subject,
            "mat_path": str(subject.mat_path),
            "channel_loc_path": str(spec.channel_loc_path),
            "sampling_rate": 250,
            "required_channel_names": list(WANG2016_REQUIRED_CHANNELS),
            "selected_channel_names": list(subject.selected_channel_names),
            "selected_channel_indices_zero_based": list(subject.selected_channel_indices),
            "target_index_by_freq": {f"{float(freq):g}": int(index) for freq, index in target_index.items()},
            "idle_proxy_note": (
                "Idle/no-control is proxied with non-command target stimulus trials from the external benchmark."
            ),
        }
        return 250, segments, metadata
    if spec.dataset == "beta":
        subject = load_beta_subject(spec.mat_path)
        resolved_freqs, target_index = resolve_beta_command_frequencies(subject, freqs)
        segments = build_beta_segments(
            subject,
            freqs=resolved_freqs,
            include_hard_idle=True,
            include_pre_stim_idle=False,
        )
        metadata = {
            "dataset": "beta",
            "subject": subject.subject,
            "mat_path": str(subject.mat_path),
            "sampling_rate": int(subject.sampling_rate),
            "required_channel_names": list(BETA_REQUIRED_CHANNELS),
            "selected_channel_names": list(subject.selected_channel_names),
            "selected_channel_indices_zero_based": list(subject.selected_channel_indices),
            "target_index_by_freq": {f"{float(freq):g}": int(index) for freq, index in target_index.items()},
            "idle_proxy_note": (
                "Idle/no-control is proxied with non-command target stimulus trials from the external benchmark."
            ),
        }
        return int(subject.sampling_rate), segments, metadata
    raise ValueError(f"unsupported dataset: {spec.dataset}")


def build_block_split_plans(
    *,
    dataset: str,
    subject: str,
    block_indices: Sequence[int],
    calibration_block_count: int,
    max_splits: int,
    seed: int,
) -> list[SplitPlan]:
    blocks = tuple(sorted(int(block) for block in block_indices))
    if len(blocks) <= 1:
        return []
    if calibration_block_count <= 0 or calibration_block_count >= len(blocks):
        return []
    combos = [tuple(int(item) for item in combo) for combo in combinations(blocks, calibration_block_count)]
    if max_splits > 0 and len(combos) > max_splits:
        rng = random.Random(int(seed) + int(calibration_block_count) * 1009 + len(blocks) * 17)
        rng.shuffle(combos)
        combos = combos[: int(max_splits)]
        combos = sorted(tuple(sorted(combo)) for combo in combos)
    plans: list[SplitPlan] = []
    for index, calibration_blocks in enumerate(combos):
        holdout_blocks = tuple(block for block in blocks if block not in calibration_blocks)
        if not holdout_blocks:
            continue
        plans.append(
            SplitPlan(
                dataset=str(dataset),
                subject=str(subject),
                split_index=int(index),
                seed=int(seed) + int(index),
                calibration_blocks=tuple(calibration_blocks),
                holdout_blocks=tuple(holdout_blocks),
            )
        )
    return plans


def select_split_segments(
    segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    freqs: Sequence[float],
    calibration_blocks: Sequence[int],
    holdout_blocks: Sequence[int],
    idle_multiplier: float,
    seed: int,
) -> tuple[list[tuple[TrialSpec, np.ndarray]], list[tuple[TrialSpec, np.ndarray]], dict[str, Any]]:
    calibration_block_set = {int(block) for block in calibration_blocks}
    holdout_block_set = {int(block) for block in holdout_blocks}
    calibration_control = [
        item
        for item in segments
        if int(item[0].block_index) in calibration_block_set and item[0].expected_freq is not None
    ]
    calibration_idle_pool = [
        item for item in segments if int(item[0].block_index) in calibration_block_set and item[0].expected_freq is None
    ]
    holdout_segments = [item for item in segments if int(item[0].block_index) in holdout_block_set]
    control_count = len(calibration_control)
    if control_count <= 0:
        raise ValueError("calibration split produced no control trials")
    if not holdout_segments:
        raise ValueError("holdout split produced no holdout trials")
    idle_budget = int(round(float(control_count) * max(float(idle_multiplier), 0.0)))
    if idle_budget > len(calibration_idle_pool):
        idle_budget = len(calibration_idle_pool)
    rng = random.Random(int(seed))
    sampled_idle = list(calibration_idle_pool)
    if idle_budget < len(sampled_idle):
        sampled_idle = rng.sample(sampled_idle, idle_budget)
    calibration_segments = sorted([*calibration_control, *sampled_idle], key=_trial_sort_key)
    holdout_segments = sorted(list(holdout_segments), key=_trial_sort_key)
    summary = {
        "seed": int(seed),
        "idle_multiplier": float(idle_multiplier),
        "idle_pool_count": int(len(calibration_idle_pool)),
        "idle_selected_count": int(len(sampled_idle)),
        "calibration_blocks": [int(block) for block in calibration_blocks],
        "holdout_blocks": [int(block) for block in holdout_blocks],
        "calibration_counts": _count_segments(calibration_segments, freqs),
        "holdout_counts": _count_segments(holdout_segments, freqs),
    }
    return calibration_segments, holdout_segments, summary


def _minimal_protocol_config(
    *,
    spec: ExternalSubjectSpec,
    freqs: Sequence[float],
    sampling_rate: int,
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
) -> dict[str, Any]:
    counts = _count_segments(trial_segments, freqs)
    active_samples = max(int(np.asarray(segment).shape[0]) for _trial, segment in trial_segments)
    active_sec = float(active_samples) / float(max(int(sampling_rate), 1))
    return {
        "protocol_name": "external-short-pretrain-split-v1",
        "source_dataset": str(spec.dataset),
        "subject_file": str(spec.mat_path),
        "sampling_rate": int(sampling_rate),
        "active_sec": float(active_sec),
        "planned_total_trials": int(counts["total"]),
        "saved_trial_count": int(counts["total"]),
        "control_trial_count": int(counts["control"]),
        "clean_idle_trial_count": 0,
        "hard_idle_trial_count": int(counts["idle"]),
        "freqs": [float(freq) for freq in freqs],
        "selected_channel_policy": "strict_required_8_channels_only",
        "selected_channel_names": list(_required_channel_names(spec.dataset)),
        "include_hard_idle_non_command_targets": True,
        "include_pre_stimulus_clean_idle": False,
        "idle_proxy_note": "External hard-idle proxy uses non-command target stimulus trials.",
    }


def save_split_dataset(
    *,
    dataset_root: Path,
    spec: ExternalSubjectSpec,
    split_plan: SplitPlan,
    split_label: str,
    sampling_rate: int,
    freqs: Sequence[float],
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
) -> dict[str, str]:
    token = _freq_token(freqs)
    session_id = (
        f"{spec.dataset}_{spec.subject.lower()}_{split_label}_"
        f"cal{'-'.join(str(block) for block in split_plan.calibration_blocks)}_{token}"
    )
    subject_id = f"{spec.dataset}_{spec.subject.lower()}"
    protocol = _minimal_protocol_config(
        spec=spec,
        freqs=freqs,
        sampling_rate=sampling_rate,
        trial_segments=trial_segments,
    )
    return save_collection_dataset_bundle(
        dataset_root=Path(dataset_root),
        session_id=session_id,
        subject_id=subject_id,
        serial_port=f"external_{spec.dataset}",
        board_id=-1,
        sampling_rate=int(sampling_rate),
        freqs=tuple(float(freq) for freq in freqs),
        board_eeg_channels=tuple(range(len(_required_channel_names(spec.dataset)))),
        protocol_config=protocol,
        trial_segments=trial_segments,
    )


def _freq_label(freq: float) -> str:
    return f"{float(freq):g}"


def _classifier_labels(freqs: Sequence[float]) -> tuple[str, ...]:
    return ("idle", *(_freq_label(float(freq)) for freq in freqs))


def _trial_true_label(trial: TrialSpec) -> str:
    return "idle" if trial.expected_freq is None else _freq_label(float(trial.expected_freq))


def _score_matrix_to_features(score_matrix: np.ndarray) -> np.ndarray:
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


def _classifier_feature_names(freqs: Sequence[float], *, score_source_name: str = "fbcca") -> list[str]:
    names = [f"{str(score_source_name).strip().lower()}_score_{_freq_label(freq)}" for freq in freqs]
    names.extend(CLASSIFIER_DERIVED_FEATURE_NAMES)
    return names


def _build_fbcca_decoder_for_scoring(
    *,
    freqs: Sequence[float],
    sampling_rate: int,
    win_sec: float,
    step_sec: float,
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
) -> Any:
    return _build_score_method_decoder(
        method_name="fbcca_lda5",
        freqs=freqs,
        sampling_rate=sampling_rate,
        win_sec=win_sec,
        step_sec=step_sec,
        compute_backend=compute_backend,
        gpu_device=gpu_device,
        gpu_precision=gpu_precision,
    )


def _score_trials_for_classifier(
    *,
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    decoder: Any,
) -> list[ScoredTrial]:
    scored: list[ScoredTrial] = []
    for trial, segment in trial_segments:
        matrix = np.ascontiguousarray(np.asarray(segment, dtype=np.float64))
        if matrix.ndim != 2 or matrix.shape[0] < int(decoder.win_samples):
            continue
        windows = extract_window_batch(
            matrix,
            win_samples=int(decoder.win_samples),
            step_samples=int(decoder.step_samples),
        )
        if hasattr(decoder, "score_windows_batch"):
            score_matrix = np.asarray(decoder.score_windows_batch(windows), dtype=np.float64)
        elif hasattr(decoder, "analyze_windows_batch"):
            payloads = decoder.analyze_windows_batch(windows)
            score_matrix = np.vstack(
                [np.asarray(dict(item)["scores"], dtype=np.float64).reshape(1, -1) for item in payloads]
            )
        else:
            rows: list[np.ndarray] = []
            for window in windows:
                if hasattr(decoder, "score_window"):
                    row = np.asarray(decoder.score_window(window), dtype=np.float64)
                elif hasattr(decoder, "analyze_window"):
                    row = np.asarray(dict(decoder.analyze_window(window))["scores"], dtype=np.float64)
                else:
                    raise TypeError("decoder does not expose a supported scoring API")
                rows.append(row.reshape(1, -1))
            score_matrix = np.vstack(rows)
        scored.append(
            ScoredTrial(
                trial=trial,
                score_matrix=score_matrix,
                feature_matrix=_score_matrix_to_features(score_matrix),
                duration_sec=float(matrix.shape[0]) / float(max(int(decoder.fs), 1)),
            )
        )
    return scored


def _scoring_cache_key(
    *,
    freqs: Sequence[float],
    sampling_rate: int,
    step_sec: float,
    win_sec: float,
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
) -> tuple[Any, ...]:
    return (
        tuple(round(float(freq), 10) for freq in freqs),
        int(sampling_rate),
        round(float(step_sec), 9),
        round(float(win_sec), 9),
        str(compute_backend),
        int(gpu_device),
        str(gpu_precision),
    )


def _trial_segment_cache_key(item: tuple[TrialSpec, np.ndarray]) -> tuple[Any, ...]:
    trial, segment = item
    matrix = np.asarray(segment)
    expected_freq = None if trial.expected_freq is None else round(float(trial.expected_freq), 10)
    return (
        int(trial.block_index),
        int(trial.trial_id),
        str(trial.label),
        expected_freq,
        tuple(int(value) for value in matrix.shape),
        int(id(segment)),
    )


def _score_segment_subset_cached(
    *,
    freqs: Sequence[float],
    sampling_rate: int,
    step_sec: float,
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
    win_sec: float,
    segments: Sequence[tuple[TrialSpec, np.ndarray]],
    context: str,
    decoder_cache: dict[tuple[Any, ...], Any],
    scored_cache: dict[tuple[Any, ...], dict[tuple[Any, ...], ScoredTrial]],
) -> list[ScoredTrial]:
    runtime_key = _scoring_cache_key(
        freqs=freqs,
        sampling_rate=int(sampling_rate),
        step_sec=float(step_sec),
        win_sec=float(win_sec),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
    )
    segment_keys = [_trial_segment_cache_key(item) for item in segments]
    if len(set(segment_keys)) != len(segment_keys):
        raise ValueError(f"{context} contains duplicate trial segment cache keys")

    runtime_scored_cache = scored_cache.setdefault(runtime_key, {})
    missing_segments = [
        item for item, segment_key in zip(segments, segment_keys) if segment_key not in runtime_scored_cache
    ]
    if missing_segments:
        decoder = decoder_cache.get(runtime_key)
        if decoder is None:
            decoder = _build_fbcca_decoder_for_scoring(
                freqs=freqs,
                sampling_rate=int(sampling_rate),
                win_sec=float(win_sec),
                step_sec=float(step_sec),
                compute_backend=str(compute_backend),
                gpu_device=int(gpu_device),
                gpu_precision=str(gpu_precision),
            )
            decoder_cache[runtime_key] = decoder
        scored_missing = _score_trials_for_classifier(
            trial_segments=missing_segments,
            decoder=decoder,
        )
        if len(scored_missing) != len(missing_segments):
            raise RuntimeError(
                f"{context} scored {len(scored_missing)} of {len(missing_segments)} requested trial segments"
            )
        for item, scored in zip(missing_segments, scored_missing):
            runtime_scored_cache[_trial_segment_cache_key(item)] = scored

    scored_subset = [runtime_scored_cache[segment_key] for segment_key in segment_keys]
    _validate_scored_trial_coverage(
        scored_subset,
        freqs=freqs,
        context=context,
    )
    return scored_subset


def _fbcca_lda5_predict_windows(model: FBCCALDA5Model, feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(feature_matrix, dtype=np.float64)
    z = (features - model.feature_mean) / model.feature_std
    diff = z[:, None, :] - model.class_means[None, :, :]
    distances = np.sum((diff * diff) / model.pooled_var[None, None, :], axis=2)
    logits = -0.5 * distances
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-12)
    return probs, np.asarray(model.labels, dtype=object)


def _fbcca_ridge5_predict_windows(model: FBCCARidge5Model, feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(feature_matrix, dtype=np.float64)
    z = (features - model.feature_mean) / model.feature_std
    design = np.column_stack([np.ones(int(z.shape[0]), dtype=np.float64), z])
    logits = design @ np.asarray(model.weights, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-12)
    return probs, np.asarray(model.labels, dtype=object)


def _predict_fbcca_lda5_trial(
    model: FBCCALDA5Model | FBCCARidge5Model,
    item: ScoredTrial,
    *,
    min_enter_windows: int,
    max_gap_windows: int = 0,
) -> tuple[str, float, float]:
    probs, labels = _predict_classifier_windows(model, item.feature_matrix)
    return _predict_fbcca_lda5_trial_from_probs(
        model,
        probs,
        labels,
        min_enter_windows=min_enter_windows,
        max_gap_windows=max(0, int(max_gap_windows)),
    )


def _predict_fbcca_lda5_trial_from_probs(
    model: FBCCALDA5Model | FBCCARidge5Model,
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    min_enter_windows: int,
    max_gap_windows: int = 0,
) -> tuple[str, float, float]:
    if probs.size <= 0:
        return "idle", 0.0, float("inf")
    idle_index = int(np.where(labels == "idle")[0][0])
    needed = max(1, int(min_enter_windows))
    max_gap = max(0, int(max_gap_windows))
    streak_label = ""
    streak_count = 0
    gap_count = 0
    last_command_index = 0
    last_command_confidence = 0.0
    best_command_label = "idle"
    best_command_confidence = 0.0
    best_command_index = 0
    for index, row in enumerate(probs):
        pred_index = int(np.argmax(row))
        pred_label = str(labels[pred_index])
        command_confidence = float(1.0 - row[idle_index])
        if pred_label != "idle" and command_confidence > best_command_confidence:
            best_command_label = pred_label
            best_command_confidence = command_confidence
            best_command_index = int(index)
        if pred_label != "idle" and command_confidence >= float(model.command_confidence_th):
            if pred_label == streak_label:
                streak_count += 1
            else:
                streak_label = pred_label
                streak_count = 1
            gap_count = 0
            last_command_index = int(index)
            last_command_confidence = float(command_confidence)
            if streak_count >= needed:
                return pred_label, command_confidence, float(index)
        elif streak_label and gap_count < max_gap:
            gap_count += 1
        else:
            streak_label = ""
            streak_count = 0
            gap_count = 0
            last_command_index = 0
            last_command_confidence = 0.0
    if streak_label and streak_count >= needed:
        return streak_label, last_command_confidence, float(last_command_index)
    if needed <= 1 and best_command_confidence >= float(model.command_confidence_th):
        return best_command_label, best_command_confidence, float(best_command_index)
    return "idle", 0.0, 0.0


def _predict_classifier_windows(
    model: FBCCALDA5Model | FBCCARidge5Model,
    feature_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(model, FBCCARidge5Model):
        return _fbcca_ridge5_predict_windows(model, feature_matrix)
    return _fbcca_lda5_predict_windows(model, feature_matrix)


def _build_classifier_probability_cache(
    model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
) -> tuple[tuple[ScoredTrial, np.ndarray, np.ndarray], ...]:
    return tuple(
        (item, *_predict_classifier_windows(model, item.feature_matrix))
        for item in scored_trials
    )


def _predict_fbcca_lda5_fixed_trial(
    model: FBCCALDA5Model | FBCCARidge5Model,
    item: ScoredTrial,
) -> tuple[str, str, float]:
    probs, labels = _predict_classifier_windows(model, item.feature_matrix)
    return _predict_fbcca_lda5_fixed_from_probs(model, probs, labels)


def _predict_fbcca_lda5_fixed_from_probs(
    model: FBCCALDA5Model | FBCCARidge5Model,
    probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[str, str, float]:
    if probs.size <= 0:
        return "idle", str(model.labels[1]), 0.0
    row = probs[-1]
    pred_5 = str(labels[int(np.argmax(row))])
    command_labels = list(model.labels[1:])
    command_indices = [int(np.where(labels == label)[0][0]) for label in command_labels]
    pred_4 = command_labels[int(np.argmax(row[command_indices]))]
    confidence = float(np.max(row))
    return pred_5, pred_4, confidence


def _validate_scored_trial_coverage(
    scored_trials: Sequence[ScoredTrial],
    *,
    freqs: Sequence[float],
    context: str,
) -> None:
    expected_labels = set(_classifier_labels(freqs))
    present_labels = {_trial_true_label(item.trial) for item in scored_trials}
    missing = sorted(label for label in expected_labels if label not in present_labels)
    if missing:
        raise ValueError(f"{context} missing scored classes: {missing}")


def _score_split_once(
    *,
    freqs: Sequence[float],
    sampling_rate: int,
    step_sec: float,
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
    calibration_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    holdout_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    win_sec: float,
    context: str,
) -> tuple[list[ScoredTrial], list[ScoredTrial]]:
    decoder = _build_fbcca_decoder_for_scoring(
        freqs=freqs,
        sampling_rate=int(sampling_rate),
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
    )
    calibration_scored = _score_trials_for_classifier(
        trial_segments=calibration_segments,
        decoder=decoder,
    )
    holdout_scored = _score_trials_for_classifier(
        trial_segments=holdout_segments,
        decoder=decoder,
    )
    _validate_scored_trial_coverage(
        calibration_scored,
        freqs=freqs,
        context=f"{context} calibration",
    )
    _validate_scored_trial_coverage(
        holdout_scored,
        freqs=freqs,
        context=f"{context} holdout",
    )
    return calibration_scored, holdout_scored


def _score_split_once_for_method(
    *,
    method_name: str,
    freqs: Sequence[float],
    sampling_rate: int,
    step_sec: float,
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
    calibration_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    holdout_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    win_sec: float,
    context: str,
) -> tuple[list[ScoredTrial], list[ScoredTrial]]:
    spec = _score_method_spec(method_name)
    if not spec.fit_decoder:
        return _score_split_once(
            freqs=freqs,
            sampling_rate=int(sampling_rate),
            step_sec=float(step_sec),
            compute_backend=str(compute_backend),
            gpu_device=int(gpu_device),
            gpu_precision=str(gpu_precision),
            calibration_segments=calibration_segments,
            holdout_segments=holdout_segments,
            win_sec=float(win_sec),
            context=context,
        )
    decoder = _build_score_method_decoder(
        method_name=method_name,
        freqs=freqs,
        sampling_rate=int(sampling_rate),
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
    )
    if getattr(decoder, "requires_fit", False):
        decoder.fit(calibration_segments)
    calibration_scored = _score_trials_for_classifier(
        trial_segments=calibration_segments,
        decoder=decoder,
    )
    holdout_scored = _score_trials_for_classifier(
        trial_segments=holdout_segments,
        decoder=decoder,
    )
    _validate_scored_trial_coverage(
        calibration_scored,
        freqs=freqs,
        context=f"{context} calibration",
    )
    _validate_scored_trial_coverage(
        holdout_scored,
        freqs=freqs,
        context=f"{context} holdout",
    )
    return calibration_scored, holdout_scored


def _fbcca_lda5_threshold_candidates(
    model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    probability_cache: Optional[Sequence[tuple[ScoredTrial, np.ndarray, np.ndarray]]] = None,
) -> tuple[float, ...]:
    values: list[float] = [0.0, 1.0]
    cache = (
        tuple(probability_cache)
        if probability_cache is not None
        else _build_classifier_probability_cache(model, scored_trials)
    )
    for _item, probs, labels in cache:
        if probs.size <= 0:
            continue
        idle_index = int(np.where(labels == "idle")[0][0])
        pred_indices = np.argmax(probs, axis=1)
        command_mask = pred_indices != idle_index
        if not np.any(command_mask):
            continue
        command_conf = 1.0 - probs[:, idle_index]
        values.extend(float(value) for value in command_conf[command_mask])
    rounded = np.unique(np.round(np.asarray(values, dtype=np.float64), 6))
    if rounded.size <= 0:
        return (0.0,)
    return tuple(float(value) for value in rounded.tolist())


def _select_fbcca_lda5_confidence_threshold(
    base_model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int = 0,
    threshold_policy: str = DEFAULT_CLASSIFIER_THRESHOLD_POLICY,
) -> dict[str, Any]:
    best_threshold = 0.0
    best_rank: Optional[tuple[float, ...]] = None
    best_bundle: dict[str, Any] = {}
    probability_cache = _build_classifier_probability_cache(base_model, scored_trials)
    candidates = _fbcca_lda5_threshold_candidates(
        base_model,
        scored_trials,
        probability_cache=probability_cache,
    )
    for threshold in candidates:
        model = replace(base_model, command_confidence_th=float(threshold))
        bundle = _evaluate_fbcca_lda5_model(
            model,
            scored_trials,
            win_sec=float(win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            probability_cache=probability_cache,
        )
        fixed5 = dict(bundle.get("fixed_window_metrics_5class") or {})
        async5 = dict(bundle.get("async_lens_metrics_5class") or {})
        async_metrics = dict(bundle.get("async_metrics") or {})
        selected_metrics = {
            **async_metrics,
            "async_acc_5class": _safe_float(async5.get("acc"), 0.0),
            "async_macro_f1_5class": _safe_float(async5.get("macro_f1"), 0.0),
            "fixed_acc_5class": _safe_float(fixed5.get("acc"), 0.0),
            "fixed_macro_f1_5class": _safe_float(fixed5.get("macro_f1"), 0.0),
        }
        rank = _classifier_threshold_rank_key(
            selected_metrics,
            policy=str(threshold_policy),
            tie_breaker=float(threshold),
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_threshold = float(threshold)
            best_bundle = bundle
    best_async = dict(best_bundle.get("async_metrics") or {})
    best_async5 = dict(best_bundle.get("async_lens_metrics_5class") or {})
    best_fixed5 = dict(best_bundle.get("fixed_window_metrics_5class") or {})
    return {
        "command_confidence_th": float(best_threshold),
        "threshold_policy": _parse_classifier_threshold_policy(threshold_policy),
        "max_gap_windows": max(0, int(max_gap_windows)),
        "candidate_count": int(len(candidates)),
        "candidate_thresholds_preview": [float(value) for value in candidates[:12]],
        "selected_metrics": {
            "idle_fp_per_min": _safe_float(best_async.get("idle_fp_per_min"), float("inf")),
            "idle_selected_windows_per_min": _safe_float(
                best_async.get("idle_selected_windows_per_min"),
                float("inf"),
            ),
            "control_recall": _safe_float(best_async.get("control_recall"), 0.0),
            "control_recall_at_2s": _safe_float(best_async.get("control_recall_at_2s"), 0.0),
            "control_recall_at_2.5s": _safe_float(best_async.get("control_recall_at_2.5s"), 0.0),
            "control_recall_at_3s": _safe_float(best_async.get("control_recall_at_3s"), 0.0),
            "detection_latency_s": _safe_float(best_async.get("detection_latency_s"), float("inf")),
            "async_acc_5class": _safe_float(best_async5.get("acc"), 0.0),
            "async_macro_f1_5class": _safe_float(best_async5.get("macro_f1"), 0.0),
            "fixed_acc_5class": _safe_float(best_fixed5.get("acc"), 0.0),
            "fixed_macro_f1_5class": _safe_float(best_fixed5.get("macro_f1"), 0.0),
        },
    }


def _fit_fbcca_lda5_base_model(
    scored_trials: Sequence[ScoredTrial],
    *,
    freqs: Sequence[float],
    score_source_name: str = "fbcca",
) -> FBCCALDA5Model:
    labels = _classifier_labels(freqs)
    rows: list[np.ndarray] = []
    y: list[str] = []
    per_label_windows = {label: 0 for label in labels}
    per_label_trials = {label: 0 for label in labels}
    for item in scored_trials:
        label = _trial_true_label(item.trial)
        if label not in per_label_windows:
            continue
        features = np.asarray(item.feature_matrix, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] <= 0:
            continue
        rows.append(features)
        y.extend([label] * int(features.shape[0]))
        per_label_windows[label] += int(features.shape[0])
        per_label_trials[label] += 1
    missing = [label for label in labels if per_label_windows.get(label, 0) <= 0]
    if missing:
        raise ValueError(f"classifier calibration missing classes: {missing}")
    x = np.vstack(rows).astype(np.float64, copy=False)
    y_array = np.asarray(y, dtype=object)
    feature_mean = np.mean(x, axis=0)
    feature_std = np.std(x, axis=0)
    feature_std = np.where(feature_std > 1e-9, feature_std, 1.0)
    z = (x - feature_mean) / feature_std
    class_means: list[np.ndarray] = []
    pooled_accum = np.zeros(z.shape[1], dtype=np.float64)
    denom = 0
    for label in labels:
        class_rows = z[y_array == label]
        mean = np.mean(class_rows, axis=0)
        class_means.append(mean)
        centered = class_rows - mean
        pooled_accum += np.sum(centered * centered, axis=0)
        denom += max(int(class_rows.shape[0]) - 1, 0)
    pooled_var = pooled_accum / float(max(denom, 1))
    pooled_var = np.where(pooled_var > 1e-6, pooled_var, 1.0)
    base_model = FBCCALDA5Model(
        freqs=tuple(float(freq) for freq in freqs),  # type: ignore[arg-type]
        labels=labels,
        feature_mean=feature_mean,
        feature_std=feature_std,
        class_means=np.vstack(class_means),
        pooled_var=pooled_var,
        command_confidence_th=0.0,
        fit_summary={
            "classifier": _classifier_name_for_model(
                FBCCALDA5Model(
                    freqs=tuple(float(freq) for freq in freqs),  # type: ignore[arg-type]
                    labels=labels,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    class_means=np.vstack(class_means),
                    pooled_var=pooled_var,
                    command_confidence_th=0.0,
                    fit_summary={},
                ),
                score_source_name=score_source_name,
            ),
            "score_source_name": str(score_source_name).strip().lower(),
            "calibration_windows": int(z.shape[0]),
            "per_label_windows": {label: int(per_label_windows[label]) for label in labels},
            "per_label_trials": {label: int(per_label_trials[label]) for label in labels},
        },
    )
    return base_model


def _fit_fbcca_lda5_model(
    scored_trials: Sequence[ScoredTrial],
    *,
    freqs: Sequence[float],
    win_sec: float = 1.5,
    step_sec: float = DEFAULT_STEP_SEC,
    min_enter_windows: int = 1,
    max_gap_windows: int = 0,
    threshold_policy: str = DEFAULT_CLASSIFIER_THRESHOLD_POLICY,
    base_model: Optional[FBCCALDA5Model] = None,
    score_source_name: str = "fbcca",
) -> FBCCALDA5Model:
    if base_model is not None:
        model = base_model
    else:
        try:
            model = _fit_fbcca_lda5_base_model(
                scored_trials,
                freqs=freqs,
                score_source_name=score_source_name,
            )
        except TypeError as exc:
            if "score_source_name" not in str(exc):
                raise
            model = _fit_fbcca_lda5_base_model(
                scored_trials,
                freqs=freqs,
            )
    threshold_selection = _select_fbcca_lda5_confidence_threshold(
        model,
        scored_trials,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
        threshold_policy=str(threshold_policy),
    )
    confidence_th = float(threshold_selection.get("command_confidence_th", 0.0))
    return replace(
        model,
        command_confidence_th=max(float(confidence_th), 0.0),
        fit_summary={
            **model.fit_summary,
            "score_source_name": str(score_source_name).strip().lower(),
            "classifier": _classifier_name_for_model(model, score_source_name=score_source_name),
            "command_confidence_th": max(float(confidence_th), 0.0),
            "min_enter_windows": max(1, int(min_enter_windows)),
            "max_gap_windows": max(0, int(max_gap_windows)),
            "threshold_policy": _parse_classifier_threshold_policy(threshold_policy),
            "threshold_selection": dict(threshold_selection),
        },
    )


def _one_hot_labels(labels: Sequence[str], class_labels: Sequence[str]) -> np.ndarray:
    label_to_index = {str(label): int(index) for index, label in enumerate(class_labels)}
    target = np.zeros((len(labels), len(class_labels)), dtype=np.float64)
    for row_index, label in enumerate(labels):
        target[row_index, label_to_index[str(label)]] = 1.0
    return target


def _fit_fbcca_ridge5_base_model(
    scored_trials: Sequence[ScoredTrial],
    *,
    freqs: Sequence[float],
    l2: float,
    score_source_name: str = "fbcca",
) -> FBCCARidge5Model:
    labels = _classifier_labels(freqs)
    rows: list[np.ndarray] = []
    y: list[str] = []
    per_label_windows = {label: 0 for label in labels}
    per_label_trials = {label: 0 for label in labels}
    for item in scored_trials:
        label = _trial_true_label(item.trial)
        if label not in per_label_windows:
            continue
        features = np.asarray(item.feature_matrix, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] <= 0:
            continue
        rows.append(features)
        y.extend([label] * int(features.shape[0]))
        per_label_windows[label] += int(features.shape[0])
        per_label_trials[label] += 1
    missing = [label for label in labels if per_label_windows.get(label, 0) <= 0]
    if missing:
        raise ValueError(f"classifier calibration missing classes: {missing}")
    x = np.vstack(rows).astype(np.float64, copy=False)
    y_array = np.asarray(y, dtype=object)
    feature_mean = np.mean(x, axis=0)
    feature_std = np.std(x, axis=0)
    feature_std = np.where(feature_std > 1e-9, feature_std, 1.0)
    z = (x - feature_mean) / feature_std
    design = np.column_stack([np.ones(int(z.shape[0]), dtype=np.float64), z])
    target = _one_hot_labels(y_array, labels)
    sample_weights = np.ones(int(y_array.shape[0]), dtype=np.float64)
    for label in labels:
        mask = y_array == label
        count = int(np.sum(mask))
        if count > 0:
            sample_weights[mask] = 1.0 / float(count)
    sample_weights *= float(len(y_array)) / max(float(np.sum(sample_weights)), 1e-12)
    sqrt_weights = np.sqrt(sample_weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_target = target * sqrt_weights[:, None]
    reg = np.eye(int(design.shape[1]), dtype=np.float64) * max(float(l2), 0.0)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(weighted_design.T @ weighted_design + reg, weighted_design.T @ weighted_target)
    return FBCCARidge5Model(
        freqs=tuple(float(freq) for freq in freqs),  # type: ignore[arg-type]
        labels=labels,
        feature_mean=feature_mean,
        feature_std=feature_std,
        weights=np.asarray(weights, dtype=np.float64),
        l2=float(l2),
        command_confidence_th=0.0,
        fit_summary={
            "classifier": _classifier_name_for_model(
                FBCCARidge5Model(
                    freqs=tuple(float(freq) for freq in freqs),  # type: ignore[arg-type]
                    labels=labels,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    weights=np.asarray(weights, dtype=np.float64),
                    l2=float(l2),
                    command_confidence_th=0.0,
                    fit_summary={},
                ),
                score_source_name=score_source_name,
            ),
            "score_source_name": str(score_source_name).strip().lower(),
            "l2": float(l2),
            "class_weighting": "balanced_window_inverse_frequency",
            "calibration_windows": int(z.shape[0]),
            "per_label_windows": {label: int(per_label_windows[label]) for label in labels},
            "per_label_trials": {label: int(per_label_trials[label]) for label in labels},
        },
    )


def _fit_fbcca_ridge5_model(
    scored_trials: Sequence[ScoredTrial],
    *,
    freqs: Sequence[float],
    win_sec: float = 1.5,
    step_sec: float = DEFAULT_STEP_SEC,
    min_enter_windows: int = 1,
    max_gap_windows: int = 0,
    threshold_policy: str = DEFAULT_CLASSIFIER_THRESHOLD_POLICY,
    l2_candidates: Sequence[float] = DEFAULT_RIDGE_L2_CANDIDATES,
    base_models: Optional[Sequence[FBCCARidge5Model]] = None,
    score_source_name: str = "fbcca",
) -> FBCCARidge5Model:
    best_model: Optional[FBCCARidge5Model] = None
    best_rank: Optional[tuple[float, ...]] = None
    candidates = (
        list(base_models)
        if base_models is not None
        else []
    )
    if base_models is None:
        for l2 in l2_candidates:
            try:
                base_model = _fit_fbcca_ridge5_base_model(
                    scored_trials,
                    freqs=freqs,
                    l2=float(l2),
                    score_source_name=score_source_name,
                )
            except TypeError as exc:
                if "score_source_name" not in str(exc):
                    raise
                base_model = _fit_fbcca_ridge5_base_model(
                    scored_trials,
                    freqs=freqs,
                    l2=float(l2),
                )
            candidates.append(base_model)
    for base_model in candidates:
        threshold_selection = _select_fbcca_lda5_confidence_threshold(
            base_model,
            scored_trials,
            win_sec=float(win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            threshold_policy=str(threshold_policy),
        )
        model = replace(
            base_model,
            command_confidence_th=max(float(threshold_selection.get("command_confidence_th", 0.0)), 0.0),
            fit_summary={
                **base_model.fit_summary,
                "score_source_name": str(score_source_name).strip().lower(),
                "classifier": _classifier_name_for_model(base_model, score_source_name=score_source_name),
                "command_confidence_th": max(float(threshold_selection.get("command_confidence_th", 0.0)), 0.0),
                "min_enter_windows": max(1, int(min_enter_windows)),
                "max_gap_windows": max(0, int(max_gap_windows)),
                "threshold_policy": _parse_classifier_threshold_policy(threshold_policy),
                "threshold_selection": dict(threshold_selection),
            },
        )
        selected = dict(threshold_selection.get("selected_metrics") or {})
        rank = _classifier_threshold_rank_key(
            selected,
            policy=str(threshold_policy),
            tie_breaker=float(base_model.l2),
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_model = model
    if best_model is None:
        raise ValueError("fbcca_ridge5 could not fit any candidate")
    return best_model


def _evaluate_fbcca_lda5_model(
    model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int = 0,
    probability_cache: Optional[Sequence[tuple[ScoredTrial, np.ndarray, np.ndarray]]] = None,
) -> dict[str, Any]:
    labels5 = list(model.labels)
    fixed_y5_true: list[str] = []
    fixed_y5_pred: list[str] = []
    fixed_times5: list[float] = []
    fixed_y4_true: list[str] = []
    fixed_y4_pred: list[str] = []
    fixed_times4: list[float] = []
    async_y5_true: list[str] = []
    async_y5_pred: list[str] = []
    async_times5: list[float] = []
    async_y4_true: list[str] = []
    async_y4_pred: list[str] = []
    async_times4: list[float] = []
    control_total = 0
    control_correct = 0
    control_correct_at_2s = 0
    control_correct_at_2p5s = 0
    control_correct_at_3s = 0
    detection_latencies: list[float] = []
    idle_total = 0
    idle_selected_events = 0
    idle_selected_windows = 0
    per_freq_total = {_freq_label(freq): 0 for freq in model.freqs}
    per_freq_correct = {_freq_label(freq): 0 for freq in model.freqs}
    command_labels = list(labels5[1:])
    cache = (
        tuple(probability_cache)
        if probability_cache is not None
        else _build_classifier_probability_cache(model, scored_trials)
    )
    for item, probs, labels in cache:
        true_label = _trial_true_label(item.trial)
        fixed_pred_5, fixed_pred_4, _fixed_confidence = _predict_fbcca_lda5_fixed_from_probs(model, probs, labels)
        fixed_y5_true.append(true_label)
        fixed_y5_pred.append(fixed_pred_5)
        fixed_times5.append(float(win_sec))
        if true_label != "idle":
            fixed_y4_true.append(true_label)
            fixed_y4_pred.append(fixed_pred_4)
            fixed_times4.append(float(win_sec))

        async_pred_label, confidence, first_index = _predict_fbcca_lda5_trial_from_probs(
            model,
            probs,
            labels,
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
        )
        async_latency = (
            float(win_sec) + float(first_index) * float(step_sec)
            if async_pred_label != "idle"
            else float(item.duration_sec + win_sec)
            if true_label != "idle"
            else float(win_sec)
        )
        async_y5_true.append(true_label)
        async_y5_pred.append(async_pred_label)
        async_times5.append(float(async_latency))
        if true_label == "idle":
            idle_total += 1
            idle_index = int(np.where(labels == "idle")[0][0])
            command_conf = 1.0 - probs[:, idle_index]
            selected_mask = (np.argmax(probs, axis=1) != idle_index) & (
                command_conf >= float(model.command_confidence_th)
            )
            idle_selected_windows += int(np.sum(selected_mask))
            if async_pred_label != "idle":
                idle_selected_events += 1
        else:
            async_y4_true.append(true_label)
            missed_pred = next((label for label in command_labels if label != true_label), command_labels[0])
            async_y4_pred.append(async_pred_label if async_pred_label != "idle" else missed_pred)
            async_times4.append(float(async_latency))
            control_total += 1
            per_freq_total[true_label] += 1
            if async_pred_label == true_label:
                control_correct += 1
                per_freq_correct[true_label] += 1
                detection_latencies.append(float(async_latency))
                if float(async_latency) <= 2.0:
                    control_correct_at_2s += 1
                if float(async_latency) <= 2.5:
                    control_correct_at_2p5s += 1
                if float(async_latency) <= 3.0:
                    control_correct_at_3s += 1
    fixed_metrics5 = compute_classification_metrics(
        y_true=fixed_y5_true,
        y_pred=fixed_y5_pred,
        labels=labels5,
        decision_time_samples_s=fixed_times5,
        itr_class_count=5,
        decision_time_fallback_s=float(win_sec),
    )
    fixed_metrics4 = compute_classification_metrics(
        y_true=fixed_y4_true,
        y_pred=fixed_y4_pred,
        labels=command_labels,
        decision_time_samples_s=fixed_times4,
        itr_class_count=4,
        decision_time_fallback_s=float(win_sec),
    )
    async_metrics5 = compute_classification_metrics(
        y_true=async_y5_true,
        y_pred=async_y5_pred,
        labels=labels5,
        decision_time_samples_s=async_times5,
        itr_class_count=5,
        decision_time_fallback_s=float(win_sec),
    )
    async_metrics4 = compute_classification_metrics(
        y_true=async_y4_true,
        y_pred=async_y4_pred,
        labels=command_labels,
        decision_time_samples_s=async_times4,
        itr_class_count=4,
        decision_time_fallback_s=float(win_sec),
    )
    idle_duration_sec = float(sum(item.duration_sec for item in scored_trials if item.trial.expected_freq is None))
    idle_minutes = idle_duration_sec / 60.0
    idle_fp_per_min = float(idle_selected_events / idle_minutes) if idle_minutes > 1e-12 else 0.0
    idle_selected_windows_per_min = float(idle_selected_windows / idle_minutes) if idle_minutes > 1e-12 else 0.0
    median_detection_latency = (
        float(np.median(np.asarray(detection_latencies, dtype=np.float64)))
        if detection_latencies
        else float("inf")
    )
    async_metrics = {
        "idle_fp_per_min": idle_fp_per_min,
        "idle_selected_windows_per_min": idle_selected_windows_per_min,
        "control_recall": float(control_correct / control_total) if control_total else 0.0,
        "control_recall_at_2s": float(control_correct_at_2s / control_total) if control_total else 0.0,
        "control_recall_at_2.5s": float(control_correct_at_2p5s / control_total) if control_total else 0.0,
        "control_recall_at_3s": float(control_correct_at_3s / control_total) if control_total else 0.0,
        "switch_latency_s": float("inf"),
        "release_latency_s": float("inf"),
        "switch_latency_supported": False,
        "release_latency_supported": False,
        "detection_latency_s": median_detection_latency,
        "idle_trial_fp_rate": float(idle_selected_events / idle_total) if idle_total else 0.0,
        "idle_trials": float(idle_total),
        "idle_fp_trials": float(idle_selected_events),
        "idle_selected_windows": float(idle_selected_windows),
        "control_trials": float(control_total),
    }
    return {
        "metric_scope": "5class",
        "fixed_window_metrics_4class": fixed_metrics4,
        "async_lens_metrics_4class": async_metrics4,
        "fixed_window_metrics_5class": fixed_metrics5,
        "async_lens_metrics_5class": async_metrics5,
        "async_metrics": async_metrics,
        "classifier_metrics_5class": async_metrics5,
        "classifier_trial_events": [
            {"true": true, "pred": pred, "decision_time_s": float(time)}
            for true, pred, time in zip(async_y5_true, async_y5_pred, async_times5)
        ],
        "per_frequency_recall": {
            label: float(per_freq_correct[label] / per_freq_total[label]) if per_freq_total[label] else 0.0
            for label in per_freq_total
        },
        "model_summary": dict(model.fit_summary),
        "command_confidence_th": float(model.command_confidence_th),
        "min_enter_windows": int(min_enter_windows),
        "max_gap_windows": max(0, int(max_gap_windows)),
    }


def _evaluation_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    if "fixed_window_metrics_5class" in bundle or "async_lens_metrics_5class" in bundle:
        return {
            "metric_scope": str(bundle.get("metric_scope", "")),
            "fixed_window_metrics_5class": dict(bundle.get("fixed_window_metrics_5class") or {}),
            "async_lens_metrics_5class": dict(bundle.get("async_lens_metrics_5class") or {}),
            "fixed_window_metrics_4class": dict(bundle.get("fixed_window_metrics_4class") or {}),
            "async_lens_metrics_4class": dict(bundle.get("async_lens_metrics_4class") or {}),
            "async_metrics": dict(bundle.get("async_metrics") or {}),
        }
    return {
        "metric_scope": str(bundle.get("metric_scope", "")),
        "fixed_window_metrics_5class": dict(bundle.get("metrics_5class") or {}),
        "async_lens_metrics_5class": dict(bundle.get("async_lens_metrics_5class") or {}),
        "fixed_window_metrics_4class": dict(bundle.get("metrics_4class") or {}),
        "async_lens_metrics_4class": dict(bundle.get("async_lens_metrics_4class") or {}),
        "async_metrics": dict(bundle.get("async_metrics") or {}),
    }


def _array_payload(value: np.ndarray) -> list[Any]:
    return np.asarray(value, dtype=np.float64).tolist()


def _classifier_name_for_model(
    model: FBCCALDA5Model | FBCCARidge5Model,
    *,
    score_source_name: str = "fbcca",
) -> str:
    prefix = str(score_source_name).strip().lower()
    if isinstance(model, FBCCARidge5Model):
        return f"{prefix}_score_ridge_5class"
    return f"{prefix}_score_lda_5class"


def _classifier_state_payload(model: FBCCALDA5Model | FBCCARidge5Model) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "freqs": [float(freq) for freq in model.freqs],
        "labels": [str(label) for label in model.labels],
        "feature_mean": _array_payload(model.feature_mean),
        "feature_std": _array_payload(model.feature_std),
        "command_confidence_th": float(model.command_confidence_th),
        "fit_summary": dict(model.fit_summary),
    }
    if isinstance(model, FBCCARidge5Model):
        payload.update(
            {
                "weights": _array_payload(model.weights),
                "l2": float(model.l2),
            }
        )
    else:
        payload.update(
            {
                "class_means": _array_payload(model.class_means),
                "pooled_var": _array_payload(model.pooled_var),
            }
        )
    return payload


def _classifier_candidate_artifact(
    *,
    model: FBCCALDA5Model | FBCCARidge5Model,
    spec: ExternalSubjectSpec,
    split_plan: SplitPlan,
    split_summary: Mapping[str, Any],
    sampling_rate: int,
    freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    eval_payload: Mapping[str, Any],
    score_source_name: str = "fbcca",
    decoder_name: Optional[str] = None,
    decoder_model_params: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    normalized_source_name = str(score_source_name).strip().lower()
    feature_names = _classifier_feature_names(freqs, score_source_name=normalized_source_name)
    return {
        "artifact_schema_version": "external_fbcca_classifier_candidate_v1",
        "status": "candidate_only",
        "runtime_loadable": False,
        "runtime_load_note": (
            "Benchmark artifact only; realtime loading still needs a registered decoder "
            "that restores this state through load_decoder_from_profile()."
        ),
        "model_name": _classifier_name_for_model(model, score_source_name=normalized_source_name),
        "model_family": f"{normalized_source_name}_score_classifier_5class",
        "feature_contract": {
            "feature_source": f"{normalized_source_name}_score_matrix",
            "feature_builder": "_score_matrix_to_features",
            "feature_names": feature_names,
            "feature_count": int(len(feature_names)),
            "command_score_count": int(len(freqs)),
            "derived_feature_names": list(CLASSIFIER_DERIVED_FEATURE_NAMES),
        },
        "windowing": {
            "win_sec": float(win_sec),
            "step_sec": float(step_sec),
            "min_enter_windows": int(min_enter_windows),
        },
        "training_provenance": {
            "dataset": str(spec.dataset),
            "subject": str(spec.subject),
            "source_mat_path": str(spec.mat_path),
            "channel_loc_path": "" if spec.channel_loc_path is None else str(spec.channel_loc_path),
            "sampling_rate": int(sampling_rate),
            "score_source_name": normalized_source_name,
            "decoder_name": str(decoder_name or normalized_source_name),
            "decoder_model_params": json_safe(dict(decoder_model_params or {})),
            "required_channel_names": list(_required_channel_names(spec.dataset)),
            "only_required_channels_used": True,
            "command_freqs": [float(freq) for freq in freqs],
            "split_index": int(split_plan.split_index),
            "seed": int(split_plan.seed),
            "calibration_blocks": [int(block) for block in split_plan.calibration_blocks],
            "holdout_blocks": [int(block) for block in split_plan.holdout_blocks],
            "idle_multiplier": _safe_float(dict(split_summary).get("idle_multiplier"), 1.0),
            "idle_proxy_note": (
                "Idle/no-control is proxied with non-command target stimulus trials from external benchmarks."
            ),
        },
        "state": _classifier_state_payload(model),
        "holdout_eval": dict(eval_payload),
        "summary_metrics": _extract_row_metrics(dict(eval_payload)),
    }


def _write_classifier_candidate_artifact(
    artifact_dir: Optional[Path],
    *,
    recipe_id: str,
    artifact: Mapping[str, Any],
) -> str:
    if artifact_dir is None:
        return ""
    artifact_path = Path(artifact_dir) / f"{recipe_id}_candidate.json"
    _write_json(artifact_path, dict(artifact))
    return str(artifact_path)


def _extract_row_metrics(eval_payload: dict[str, Any]) -> dict[str, float]:
    fixed_4 = dict(eval_payload.get("fixed_window_metrics_4class") or {})
    fixed_5 = dict(eval_payload.get("fixed_window_metrics_5class") or {})
    async_4 = dict(eval_payload.get("async_lens_metrics_4class") or {})
    async_5 = dict(eval_payload.get("async_lens_metrics_5class") or {})
    async_metrics = dict(eval_payload.get("async_metrics") or {})
    return {
        "fixed_acc_4class": _safe_float(fixed_4.get("acc"), 0.0),
        "fixed_macro_f1_4class": _safe_float(fixed_4.get("macro_f1"), 0.0),
        "fixed_acc_5class": _safe_float(fixed_5.get("acc"), 0.0),
        "fixed_macro_f1_5class": _safe_float(fixed_5.get("macro_f1"), 0.0),
        "fixed_itr_bpm_5class": _safe_float(fixed_5.get("itr_bpm"), 0.0),
        "async_acc_4class": _safe_float(async_4.get("acc"), 0.0),
        "async_macro_f1_4class": _safe_float(async_4.get("macro_f1"), 0.0),
        "async_acc_5class": _safe_float(async_5.get("acc"), 0.0),
        "async_macro_f1_5class": _safe_float(async_5.get("macro_f1"), 0.0),
        "async_itr_bpm_5class": _safe_float(async_5.get("itr_bpm"), 0.0),
        "idle_fp_per_min": _safe_float(async_metrics.get("idle_fp_per_min"), float("inf")),
        "idle_selected_windows_per_min": _safe_float(
            async_metrics.get("idle_selected_windows_per_min"),
            float("inf"),
        ),
        "control_recall": _safe_float(async_metrics.get("control_recall"), 0.0),
        "control_recall_at_2s": _safe_float(async_metrics.get("control_recall_at_2s"), 0.0),
        "control_recall_at_2.5s": _safe_float(async_metrics.get("control_recall_at_2.5s"), 0.0),
        "control_recall_at_3s": _safe_float(async_metrics.get("control_recall_at_3s"), 0.0),
        "detection_latency_s": _safe_float(async_metrics.get("detection_latency_s"), float("inf")),
        "switch_latency_supported": float(1.0 if bool(async_metrics.get("switch_latency_supported", True)) else 0.0),
        "release_latency_supported": float(1.0 if bool(async_metrics.get("release_latency_supported", True)) else 0.0),
        "switch_latency_s": _safe_float(async_metrics.get("switch_latency_s"), float("inf")),
        "release_latency_s": _safe_float(async_metrics.get("release_latency_s"), float("inf")),
    }


def _evaluate_profile(
    *,
    profile: Any,
    sampling_rate: int,
    holdout_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
) -> dict[str, Any]:
    decoder = load_decoder_from_profile(
        profile,
        sampling_rate=int(sampling_rate),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=True,
    )
    return evaluate_decoder_on_trials_v2(
        decoder,
        profile,
        holdout_segments,
        metric_scope="5class",
        decision_time_mode="fixed-window",
        async_decision_time_mode="first-correct",
        paper_decision_time_mode="fixed-window",
    )


def run_zero_shot_default(
    *,
    spec: ExternalSubjectSpec,
    split_plan: SplitPlan,
    split_summary: dict[str, Any],
    sampling_rate: int,
    freqs: Sequence[float],
    holdout_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
) -> dict[str, Any]:
    max_win_sec = _max_supported_win_sec(holdout_segments, sampling_rate)
    profile = default_profile(freqs)
    effective_win_sec = min(float(profile.win_sec), float(max_win_sec)) if max_win_sec > 0.0 else float(profile.win_sec)
    if effective_win_sec > 0.0 and abs(effective_win_sec - float(profile.win_sec)) > 1e-9:
        profile = replace(
            profile,
            win_sec=float(effective_win_sec),
            step_sec=min(float(profile.step_sec), float(effective_win_sec)),
        )
    bundle = _evaluate_profile(
        profile=profile,
        sampling_rate=sampling_rate,
        holdout_segments=holdout_segments,
        compute_backend=compute_backend,
        gpu_device=gpu_device,
        gpu_precision=gpu_precision,
    )
    eval_payload = _evaluation_payload(bundle)
    return {
        "method": "zero_shot_default",
        "recipe_id": f"default_profile_win{float(profile.win_sec):g}".replace(".", "p"),
        "dataset": str(spec.dataset),
        "subject": str(spec.subject),
        "split_index": int(split_plan.split_index),
        "calibration_blocks": [int(block) for block in split_plan.calibration_blocks],
        "holdout_blocks": [int(block) for block in split_plan.holdout_blocks],
        "split_summary": dict(split_summary),
        "calibration_profile": {
            "status": "not_applicable",
            "profile_path": "",
        },
        "holdout_eval": eval_payload,
        "summary_metrics": _extract_row_metrics(eval_payload),
    }


def run_fast_fbcca_method(
    *,
    method_dir: Path,
    spec: ExternalSubjectSpec,
    split_plan: SplitPlan,
    split_summary: dict[str, Any],
    sampling_rate: int,
    freqs: Sequence[float],
    calibration_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    holdout_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
    step_sec: float,
    win_sec: float,
    template_weight: float,
) -> dict[str, Any]:
    recipe_id = f"win{float(win_sec):g}_tw{float(template_weight):g}".replace(".", "p")
    profile_path = Path(method_dir) / recipe_id / "fbcca_profile.json"
    config = FastFBCCAPretrainConfig(
        base_profile_path=PROJECT_DIR / "profiles" / "fbcca_base_profile.json",
        fallback_profile_path=profile_path.parent / "missing_fbcca_profile.json",
        output_profile_path=profile_path,
        history_profile_path=None,
        freqs=tuple(float(freq) for freq in freqs),  # type: ignore[arg-type]
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        template_weight=float(template_weight),
        template_win_sec=float(win_sec),
        seed=int(split_plan.seed),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=True,
        fallback_to_base_on_low_quality=False,
    )
    profile, payload = run_fast_fbcca_personalization(
        config,
        trial_segments=calibration_segments,
        sampling_rate=int(sampling_rate),
        available_board_channels=tuple(range(len(_required_channel_names(spec.dataset)))),
        collection_duration_sec=_collection_duration_sec(calibration_segments, sampling_rate),
        log_fn=None,
    )
    bundle = _evaluate_profile(
        profile=profile,
        sampling_rate=sampling_rate,
        holdout_segments=holdout_segments,
        compute_backend=compute_backend,
        gpu_device=gpu_device,
        gpu_precision=gpu_precision,
    )
    eval_payload = _evaluation_payload(bundle)
    return {
        "method": "fast_fbcca",
        "recipe_id": recipe_id,
        "dataset": str(spec.dataset),
        "subject": str(spec.subject),
        "split_index": int(split_plan.split_index),
        "calibration_blocks": [int(block) for block in split_plan.calibration_blocks],
        "holdout_blocks": [int(block) for block in split_plan.holdout_blocks],
        "split_summary": dict(split_summary),
        "calibration_profile": {
            "status": str(payload.get("status", "")),
            "profile_path": str(payload.get("profile_path", "")),
            "profile_v2_path": str(payload.get("profile_v2_path", "")),
            "template_enabled": bool(payload.get("template_enabled", False)),
            "gate_calibration_enabled": bool(payload.get("gate_calibration_enabled", False)),
            "gate_feature_source": str(payload.get("gate_feature_source", "")),
            "quality_metrics": dict(payload.get("quality_metrics", {}) or {}),
            "quality_summary": dict(payload.get("quality_summary", {}) or {}),
            "fallback_reasons": list(payload.get("fallback_reasons", []) or []),
            "release_fallback_candidate": bool(payload.get("release_fallback_candidate", False)),
            "release_fallback_reasons": list(payload.get("release_fallback_reasons", []) or []),
            "recommended_for_realtime": bool(payload.get("recommended_for_realtime", False)),
        },
        "holdout_eval": eval_payload,
        "summary_metrics": _extract_row_metrics(eval_payload),
    }


def run_fbcca_lda5_method(
    *,
    artifact_dir: Optional[Path] = None,
    spec: ExternalSubjectSpec,
    split_plan: SplitPlan,
    split_summary: dict[str, Any],
    sampling_rate: int,
    freqs: Sequence[float],
    calibration_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    holdout_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
    step_sec: float,
    win_sec: float,
    min_enter_windows: int,
    max_gap_windows: int = 0,
    threshold_policy: str = DEFAULT_CLASSIFIER_THRESHOLD_POLICY,
    calibration_scored: Optional[Sequence[ScoredTrial]] = None,
    holdout_scored: Optional[Sequence[ScoredTrial]] = None,
    base_model: Optional[FBCCALDA5Model] = None,
) -> dict[str, Any]:
    recipe_id = _classifier_recipe_id(
        win_sec=float(win_sec),
        min_enter_windows=int(min_enter_windows),
        max_gap_windows=max(0, int(max_gap_windows)),
    )
    if calibration_scored is None or holdout_scored is None:
        calibration_scored, holdout_scored = _score_split_once(
            freqs=freqs,
            sampling_rate=int(sampling_rate),
            step_sec=float(step_sec),
            compute_backend=str(compute_backend),
            gpu_device=int(gpu_device),
            gpu_precision=str(gpu_precision),
            calibration_segments=calibration_segments,
            holdout_segments=holdout_segments,
            win_sec=float(win_sec),
            context=f"fbcca_lda5 dataset={spec.dataset} subject={spec.subject}",
        )
    else:
        calibration_scored = list(calibration_scored)
        holdout_scored = list(holdout_scored)
    model = _fit_fbcca_lda5_model(
        calibration_scored,
        freqs=freqs,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
        threshold_policy=str(threshold_policy),
        base_model=base_model,
    )
    bundle = _evaluate_fbcca_lda5_model(
        model,
        holdout_scored,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
    )
    eval_payload = _evaluation_payload(bundle)
    candidate_artifact = _classifier_candidate_artifact(
        model=model,
        spec=spec,
        split_plan=split_plan,
        split_summary=split_summary,
        sampling_rate=int(sampling_rate),
        freqs=freqs,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        min_enter_windows=int(min_enter_windows),
        eval_payload=eval_payload,
        score_source_name="fbcca",
        decoder_name="fbcca",
        decoder_model_params={"Nh": 5, "subband_weight_mode": "chen_fixed"},
    )
    candidate_artifact_path = _write_classifier_candidate_artifact(
        artifact_dir,
        recipe_id=recipe_id,
        artifact=candidate_artifact,
    )
    return {
        "method": "fbcca_lda5",
        "recipe_id": recipe_id,
        "dataset": str(spec.dataset),
        "subject": str(spec.subject),
        "split_index": int(split_plan.split_index),
        "calibration_blocks": [int(block) for block in split_plan.calibration_blocks],
        "holdout_blocks": [int(block) for block in split_plan.holdout_blocks],
        "split_summary": dict(split_summary),
        "calibration_profile": {
            "status": "ok",
            "classifier": "fbcca_score_lda_5class",
            "fit_summary": dict(model.fit_summary),
            "command_confidence_th": float(model.command_confidence_th),
            "min_enter_windows": int(min_enter_windows),
            "max_gap_windows": max(0, int(max_gap_windows)),
            "threshold_policy": _parse_classifier_threshold_policy(threshold_policy),
            "feature_count": int(model.feature_mean.shape[0]),
            "feature_names": _classifier_feature_names(freqs),
            "candidate_artifact": candidate_artifact,
            "candidate_artifact_path": candidate_artifact_path,
        },
        "holdout_eval": eval_payload,
        "summary_metrics": _extract_row_metrics(eval_payload),
    }


def run_fbcca_ridge5_method(
    *,
    artifact_dir: Optional[Path] = None,
    spec: ExternalSubjectSpec,
    split_plan: SplitPlan,
    split_summary: dict[str, Any],
    sampling_rate: int,
    freqs: Sequence[float],
    calibration_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    holdout_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
    step_sec: float,
    win_sec: float,
    min_enter_windows: int,
    max_gap_windows: int = 0,
    threshold_policy: str = DEFAULT_CLASSIFIER_THRESHOLD_POLICY,
    calibration_scored: Optional[Sequence[ScoredTrial]] = None,
    holdout_scored: Optional[Sequence[ScoredTrial]] = None,
    base_models: Optional[Sequence[FBCCARidge5Model]] = None,
    method_name: str = "fbcca_ridge5",
    score_source_name: str = "fbcca",
    decoder_name: Optional[str] = None,
    decoder_model_params: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    recipe_id = _classifier_recipe_id(
        win_sec=float(win_sec),
        min_enter_windows=int(min_enter_windows),
        max_gap_windows=max(0, int(max_gap_windows)),
    )
    latency_win_sec = _method_latency_window_sec(
        method_name=str(method_name),
        win_sec=float(win_sec),
        sampling_rate=int(sampling_rate),
    )
    if calibration_scored is None or holdout_scored is None:
        calibration_scored, holdout_scored = _score_split_once_for_method(
            method_name=str(method_name),
            freqs=freqs,
            sampling_rate=int(sampling_rate),
            step_sec=float(step_sec),
            compute_backend=str(compute_backend),
            gpu_device=int(gpu_device),
            gpu_precision=str(gpu_precision),
            calibration_segments=calibration_segments,
            holdout_segments=holdout_segments,
            win_sec=float(win_sec),
            context=f"{str(method_name)} dataset={spec.dataset} subject={spec.subject}",
        )
    else:
        calibration_scored = list(calibration_scored)
        holdout_scored = list(holdout_scored)
    model = _fit_fbcca_ridge5_model(
        calibration_scored,
        freqs=freqs,
        win_sec=float(latency_win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
        threshold_policy=str(threshold_policy),
        base_models=base_models,
        score_source_name=str(score_source_name),
    )
    bundle = _evaluate_fbcca_lda5_model(
        model,
        holdout_scored,
        win_sec=float(latency_win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
    )
    eval_payload = _evaluation_payload(bundle)
    candidate_artifact = _classifier_candidate_artifact(
        model=model,
        spec=spec,
        split_plan=split_plan,
        split_summary=split_summary,
        sampling_rate=int(sampling_rate),
        freqs=freqs,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        min_enter_windows=int(min_enter_windows),
        eval_payload=eval_payload,
        score_source_name=str(score_source_name),
        decoder_name=str(decoder_name or _score_method_spec(method_name).decoder_name),
        decoder_model_params=dict(decoder_model_params or _score_method_spec(method_name).decoder_model_params),
    )
    candidate_artifact_path = _write_classifier_candidate_artifact(
        artifact_dir,
        recipe_id=recipe_id,
        artifact=candidate_artifact,
    )
    return {
        "method": str(method_name),
        "recipe_id": recipe_id,
        "dataset": str(spec.dataset),
        "subject": str(spec.subject),
        "split_index": int(split_plan.split_index),
        "calibration_blocks": [int(block) for block in split_plan.calibration_blocks],
        "holdout_blocks": [int(block) for block in split_plan.holdout_blocks],
        "split_summary": dict(split_summary),
        "calibration_profile": {
            "status": "ok",
            "classifier": _classifier_name_for_model(model, score_source_name=score_source_name),
            "fit_summary": dict(model.fit_summary),
            "command_confidence_th": float(model.command_confidence_th),
            "min_enter_windows": int(min_enter_windows),
            "max_gap_windows": max(0, int(max_gap_windows)),
            "threshold_policy": _parse_classifier_threshold_policy(threshold_policy),
            "feature_count": int(model.feature_mean.shape[0]),
            "feature_names": _classifier_feature_names(freqs, score_source_name=score_source_name),
            "score_source_name": str(score_source_name).strip().lower(),
            "decoder_name": str(decoder_name or _score_method_spec(method_name).decoder_name),
            "decoder_model_params": dict(decoder_model_params or _score_method_spec(method_name).decoder_model_params),
            "l2": float(model.l2),
            "candidate_artifact": candidate_artifact,
            "candidate_artifact_path": candidate_artifact_path,
        },
        "holdout_eval": eval_payload,
        "summary_metrics": _extract_row_metrics(eval_payload),
    }


def run_threshold_pretrain_method(
    *,
    method_dir: Path,
    dataset_root: Path,
    spec: ExternalSubjectSpec,
    split_plan: SplitPlan,
    split_summary: dict[str, Any],
    sampling_rate: int,
    freqs: Sequence[float],
    calibration_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    holdout_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
    step_sec: float,
    win_sec_candidates: Sequence[float],
    gate_policies: Sequence[str],
    min_enter_candidates: Sequence[int],
    min_exit_candidates: Sequence[int],
    control_state_modes: Sequence[str],
) -> dict[str, Any]:
    calibration_bundle = save_split_dataset(
        dataset_root=dataset_root,
        spec=spec,
        split_plan=split_plan,
        split_label=f"split{int(split_plan.split_index):02d}",
        sampling_rate=sampling_rate,
        freqs=freqs,
        trial_segments=calibration_segments,
    )
    manifest_path = Path(calibration_bundle["dataset_manifest"]).expanduser().resolve()
    run_dir = Path(method_dir) / f"split{int(split_plan.split_index):02d}"
    config = FBCCAThresholdPretrainConfig(
        dataset_manifest_session1=manifest_path,
        output_profile_path=run_dir / "fbcca_profile.json",
        report_path=run_dir / "report.json",
        report_root_dir=run_dir,
        organize_report_dir=False,
        win_sec=float(win_sec_candidates[0]),
        step_sec=float(step_sec),
        win_sec_candidates=tuple(float(value) for value in win_sec_candidates),
        gate_policy_candidates=tuple(str(value) for value in gate_policies),
        min_enter_windows_candidates=tuple(int(value) for value in min_enter_candidates),
        min_exit_windows_candidates=tuple(int(value) for value in min_exit_candidates),
        control_state_mode_candidates=tuple(str(value) for value in control_state_modes),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=True,
        publish_realtime=False,
        progress_heartbeat_sec=30.0,
    )
    payload = run_fbcca_threshold_pretrain(config, log_fn=None)
    profile = load_profile(Path(payload["profile_path"]).expanduser().resolve(), fallback_freqs=freqs, require_exists=True)
    bundle = _evaluate_profile(
        profile=profile,
        sampling_rate=sampling_rate,
        holdout_segments=holdout_segments,
        compute_backend=compute_backend,
        gpu_device=gpu_device,
        gpu_precision=gpu_precision,
    )
    eval_payload = _evaluation_payload(bundle)
    chosen = dict(payload.get("chosen_candidate", {}) or {})
    recipe_id = (
        "search_"
        f"w{float(chosen.get('win_sec', 0.0)):g}_"
        f"gp{str(chosen.get('gate_policy', ''))}_"
        f"me{int(chosen.get('min_enter_windows', 0) or 0)}_"
        f"mx{int(chosen.get('min_exit_windows', 0) or 0)}_"
        f"cs{str(chosen.get('control_state_mode', ''))}"
    ).replace(".", "p")
    return {
        "method": "threshold_pretrain",
        "recipe_id": recipe_id,
        "aggregate_recipe_id": "selected_policy_grid_search",
        "selected_recipe_id": recipe_id,
        "dataset": str(spec.dataset),
        "subject": str(spec.subject),
        "split_index": int(split_plan.split_index),
        "calibration_blocks": [int(block) for block in split_plan.calibration_blocks],
        "holdout_blocks": [int(block) for block in split_plan.holdout_blocks],
        "split_summary": dict(split_summary),
        "calibration_manifest": str(manifest_path),
        "calibration_profile": {
            "status": "ok",
            "profile_path": str(payload.get("profile_path", "")),
            "profile_v2_path": str(payload.get("profile_v2_path", "")),
            "report_path": str(payload.get("report_path", "")),
            "run_valid_for_deployment": bool(payload.get("run_valid_for_deployment", False)),
            "status_reasons": list(payload.get("status_reasons", []) or []),
            "chosen_candidate": chosen,
            "chosen_async_metrics": dict(payload.get("chosen_async_metrics", {}) or {}),
        },
        "holdout_eval": eval_payload,
        "summary_metrics": _extract_row_metrics(eval_payload),
    }


def aggregate_recipe_rows(
    rows: Sequence[dict[str, Any]],
    *,
    expected_subject_count: Optional[int] = None,
) -> list[dict[str, Any]]:
    all_subjects = {
        (str(row.get("dataset", "")), str(row.get("subject", "")))
        for row in rows
        if str(row.get("dataset", "")).strip() and str(row.get("subject", "")).strip()
    }
    resolved_expected_subject_count = (
        max(0, int(expected_subject_count))
        if expected_subject_count is not None
        else int(len(all_subjects))
    )
    grouped: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        calibration_blocks = tuple(int(item) for item in row.get("calibration_blocks", []))
        aggregate_recipe_id = str(row.get("aggregate_recipe_id") or row.get("recipe_id", ""))
        key = (
            str(row.get("method", "")),
            aggregate_recipe_id,
            int(len(calibration_blocks)),
            float(dict(row.get("split_summary", {}) or {}).get("idle_multiplier", 0.0)),
        )
        grouped[key].append(dict(row))

    summaries: list[dict[str, Any]] = []
    for key, key_rows in grouped.items():
        per_subject: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in key_rows:
            subject_key = (str(row.get("dataset", "")), str(row.get("subject", "")))
            per_subject[subject_key].append(row)

        subject_summaries: list[dict[str, Any]] = []
        for (dataset, subject), subject_rows in sorted(per_subject.items()):
            metrics = [dict(item.get("summary_metrics", {}) or {}) for item in subject_rows]
            subject_summaries.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "split_count": int(len(subject_rows)),
                    "mean_fixed_acc_4class": float(
                        np.mean([_safe_float(m.get("fixed_acc_4class"), 0.0) for m in metrics])
                    ),
                    "mean_fixed_macro_f1_4class": float(
                        np.mean([_safe_float(m.get("fixed_macro_f1_4class"), 0.0) for m in metrics])
                    ),
                    "mean_fixed_acc_5class": float(
                        np.mean([_safe_float(m.get("fixed_acc_5class"), 0.0) for m in metrics])
                    ),
                    "mean_fixed_macro_f1_5class": float(
                        np.mean([_safe_float(m.get("fixed_macro_f1_5class"), 0.0) for m in metrics])
                    ),
                    "mean_async_acc_4class": float(
                        np.mean([_safe_float(m.get("async_acc_4class"), 0.0) for m in metrics])
                    ),
                    "mean_async_macro_f1_4class": float(
                        np.mean([_safe_float(m.get("async_macro_f1_4class"), 0.0) for m in metrics])
                    ),
                    "mean_async_acc_5class": float(
                        np.mean([_safe_float(m.get("async_acc_5class"), 0.0) for m in metrics])
                    ),
                    "mean_async_macro_f1_5class": float(
                        np.mean([_safe_float(m.get("async_macro_f1_5class"), 0.0) for m in metrics])
                    ),
                    "mean_idle_fp_per_min": float(
                        np.mean([_safe_float(m.get("idle_fp_per_min"), float("inf")) for m in metrics])
                    ),
                    "mean_idle_selected_windows_per_min": float(
                        np.mean([_safe_float(m.get("idle_selected_windows_per_min"), float("inf")) for m in metrics])
                    ),
                    "mean_control_recall": float(np.mean([_safe_float(m.get("control_recall"), 0.0) for m in metrics])),
                    "mean_control_recall_at_2s": float(
                        np.mean([_safe_float(m.get("control_recall_at_2s"), 0.0) for m in metrics])
                    ),
                    "mean_control_recall_at_2.5s": float(
                        np.mean([_safe_float(m.get("control_recall_at_2.5s"), 0.0) for m in metrics])
                    ),
                    "mean_control_recall_at_3s": float(
                        np.mean([_safe_float(m.get("control_recall_at_3s"), 0.0) for m in metrics])
                    ),
                    "mean_detection_latency_s": float(
                        np.mean([_safe_float(m.get("detection_latency_s"), float("inf")) for m in metrics])
                    ),
                    "switch_latency_supported": bool(
                        all(_safe_float(m.get("switch_latency_supported"), 1.0) >= 0.5 for m in metrics)
                    ),
                    "release_latency_supported": bool(
                        all(_safe_float(m.get("release_latency_supported"), 1.0) >= 0.5 for m in metrics)
                    ),
                    "mean_switch_latency_s": float(
                        np.mean([_safe_float(m.get("switch_latency_s"), float("inf")) for m in metrics])
                    ),
                    "mean_release_latency_s": float(
                        np.mean([_safe_float(m.get("release_latency_s"), float("inf")) for m in metrics])
                    ),
                }
            )

        subject_metric = lambda field: float(
            np.mean([_safe_float(item.get(field), 0.0) for item in subject_summaries])
        ) if subject_summaries else 0.0
        calibration_patterns = sorted(
            {
                tuple(int(item) for item in row.get("calibration_blocks", []))
                for row in key_rows
            }
        )
        selected_recipe_counts: dict[str, int] = {}
        for row in key_rows:
            selected_recipe = str(row.get("selected_recipe_id") or row.get("recipe_id", ""))
            if not selected_recipe:
                continue
            selected_recipe_counts[selected_recipe] = int(selected_recipe_counts.get(selected_recipe, 0)) + 1
        summary = {
            "method": key[0],
            "recipe_id": key[1],
            "calibration_blocks": (
                [int(item) for item in calibration_patterns[0]]
                if len(calibration_patterns) == 1
                else []
            ),
            "calibration_block_patterns": [[int(item) for item in pattern] for pattern in calibration_patterns],
            "calibration_block_count": int(key[2]),
            "idle_multiplier": float(key[3]),
            "subject_count": int(len(subject_summaries)),
            "expected_subject_count": int(resolved_expected_subject_count),
            "coverage_subject_count": int(len(subject_summaries)),
            "shared_eligible": bool(
                resolved_expected_subject_count > 0
                and len(subject_summaries) == resolved_expected_subject_count
            ),
            "split_count": int(len(key_rows)),
            "mean_fixed_acc_4class": subject_metric("mean_fixed_acc_4class"),
            "mean_fixed_macro_f1_4class": subject_metric("mean_fixed_macro_f1_4class"),
            "mean_fixed_acc_5class": subject_metric("mean_fixed_acc_5class"),
            "mean_fixed_macro_f1_5class": subject_metric("mean_fixed_macro_f1_5class"),
            "mean_async_acc_4class": subject_metric("mean_async_acc_4class"),
            "mean_async_macro_f1_4class": subject_metric("mean_async_macro_f1_4class"),
            "mean_async_acc_5class": subject_metric("mean_async_acc_5class"),
            "mean_async_macro_f1_5class": subject_metric("mean_async_macro_f1_5class"),
            "mean_idle_fp_per_min": subject_metric("mean_idle_fp_per_min"),
            "mean_idle_selected_windows_per_min": subject_metric("mean_idle_selected_windows_per_min"),
            "mean_control_recall": subject_metric("mean_control_recall"),
            "mean_control_recall_at_2s": subject_metric("mean_control_recall_at_2s"),
            "mean_control_recall_at_2.5s": subject_metric("mean_control_recall_at_2.5s"),
            "mean_control_recall_at_3s": subject_metric("mean_control_recall_at_3s"),
            "mean_detection_latency_s": subject_metric("mean_detection_latency_s"),
            "switch_latency_supported": bool(
                all(bool(item.get("switch_latency_supported", True)) for item in subject_summaries)
            ),
            "release_latency_supported": bool(
                all(bool(item.get("release_latency_supported", True)) for item in subject_summaries)
            ),
            "mean_switch_latency_s": subject_metric("mean_switch_latency_s"),
            "mean_release_latency_s": subject_metric("mean_release_latency_s"),
            "selected_recipe_counts": selected_recipe_counts,
            "subjects": subject_summaries,
        }
        summaries.append(summary)
    summaries.sort(key=_summary_rank_key)
    return summaries


def _summary_rank_key(summary: dict[str, Any]) -> tuple[float, ...]:
    return _classifier_rank_key(
        {
            "idle_fp_per_min": summary.get("mean_idle_fp_per_min"),
            "idle_selected_windows_per_min": summary.get("mean_idle_selected_windows_per_min"),
            "control_recall": summary.get("mean_control_recall"),
            "control_recall_at_2s": summary.get("mean_control_recall_at_2s"),
            "control_recall_at_2.5s": summary.get("mean_control_recall_at_2.5s"),
            "control_recall_at_3s": summary.get("mean_control_recall_at_3s"),
            "async_macro_f1_5class": summary.get("mean_async_macro_f1_5class"),
            "async_acc_5class": summary.get("mean_async_acc_5class"),
            "fixed_macro_f1_5class": summary.get("mean_fixed_macro_f1_5class"),
            "fixed_acc_5class": summary.get("mean_fixed_acc_5class"),
            "detection_latency_s": summary.get("mean_detection_latency_s"),
        },
        tie_breaker=-_safe_float(summary.get("mean_fixed_macro_f1_4class"), 0.0),
    )


def _shared_recipe_summaries(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(summary) for summary in summaries if bool(summary.get("shared_eligible", False))]


def render_markdown_summary(
    *,
    run_id: str,
    freqs: Sequence[float],
    subjects: Sequence[ExternalSubjectSpec],
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    shared_summaries: Optional[Sequence[dict[str, Any]]] = None,
) -> str:
    resolved_shared_summaries = (
        [dict(summary) for summary in shared_summaries]
        if shared_summaries is not None
        else _shared_recipe_summaries(summaries)
    )
    lines = [
        "# External Short-Pretrain 5-Class Benchmark",
        "",
        f"- run_id: `{run_id}`",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- freqs: `{','.join(f'{float(freq):g}' for freq in freqs)}`",
        f"- 240hz_all_integer_frames_per_cycle: `{bool(frame_lock_frequency_report(freqs, refresh_rate_hz=240.0).get('all_integer_frames_per_cycle', False))}`",
        f"- subject_count: `{len(subjects)}`",
        f"- row_count: `{len(rows)}`",
        "",
        "> Idle/no-control is proxied with non-command target stimulus trials from the public external benchmarks.",
        "",
    ]

    def append_recipe_table(title: str, table_summaries: Sequence[dict[str, Any]]) -> None:
        lines.extend(
            [
                f"## {title}",
                "",
                "| Rank | Method | Recipe | Coverage | Cal Blocks | Idle Mult | Mean Fixed 5c Acc | Mean Fixed 5c Macro-F1 | Mean Async 5c Acc | Mean Async 5c Macro-F1 | Mean Idle FP/min | Mean Control Recall | Recall <=2.5s | Recall <=3s | Mean Detection Latency s |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if not table_summaries:
            lines.append("| - | - | - | 0/0 | - | - | - | - | - | - | - | - | - | - | - |")
            return
        for index, summary in enumerate(list(table_summaries)[:10], start=1):
            expected_subjects = int(summary.get("expected_subject_count", 0) or 0)
            covered_subjects = int(
                summary.get("coverage_subject_count", summary.get("subject_count", 0)) or 0
            )
            coverage = (
                f"{covered_subjects}/{expected_subjects}"
                if expected_subjects > 0
                else str(covered_subjects)
            )
            lines.append(
                "| {rank} | {method} | `{recipe}` | {coverage} | {cal_blocks} | {idle_mult:.2f} | {acc:.4f} | {f1:.4f} | {async_acc:.4f} | {async_f1:.4f} | {idle:.4f} | {recall:.4f} | {recall_2p5:.4f} | {recall_3:.4f} | {latency:.4f} |".format(
                    rank=index,
                    method=str(summary.get("method", "")),
                    recipe=str(summary.get("recipe_id", "")),
                    coverage=coverage,
                    cal_blocks=int(summary.get("calibration_block_count", 0)),
                    idle_mult=float(summary.get("idle_multiplier", 0.0)),
                    acc=float(summary.get("mean_fixed_acc_5class", 0.0)),
                    f1=float(summary.get("mean_fixed_macro_f1_5class", 0.0)),
                    async_acc=float(summary.get("mean_async_acc_5class", 0.0)),
                    async_f1=float(summary.get("mean_async_macro_f1_5class", 0.0)),
                    idle=float(summary.get("mean_idle_fp_per_min", float("inf"))),
                    recall=float(summary.get("mean_control_recall", 0.0)),
                    recall_2p5=float(summary.get("mean_control_recall_at_2.5s", 0.0)),
                    recall_3=float(summary.get("mean_control_recall_at_3s", 0.0)),
                    latency=float(summary.get("mean_detection_latency_s", float("inf"))),
                )
            )

    append_recipe_table("Top Shared Recipes", resolved_shared_summaries)
    lines.append("")
    append_recipe_table("Top Recipes", summaries)
    lines.extend(
        [
            "",
            "## Subjects",
            "",
            "| Dataset | Subject | |",
            "|---|---|---|",
        ]
    )
    for spec in subjects:
        lines.append(f"| {spec.dataset} | {spec.subject} | `{spec.mat_path}` |")
    return "\n".join(lines).strip() + "\n"


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    run_id = str(args.run_id or f"external_short_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_root = Path(args.output_root).expanduser().resolve() / run_id
    report_root = output_root / "reports"
    dataset_root = Path(args.dataset_root).expanduser().resolve() / run_id
    report_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    log_path = report_root / "benchmark.log"
    progress_path = report_root / "progress_snapshot.json"
    partial_summary_path = report_root / "partial_summary.json"

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    datasets = _csv_dataset_tuple(args.datasets)
    methods = _csv_method_tuple(args.methods)
    freqs = _csv_float_tuple(args.freqs, default=DEFAULT_FREQS)
    calibration_blocks = _csv_int_tuple(args.calibration_blocks, default=DEFAULT_CALIBRATION_BLOCKS)
    idle_multipliers = _csv_float_tuple(args.idle_multipliers, default=DEFAULT_IDLE_MULTIPLIERS)
    fast_win_sec_candidates = _csv_float_tuple(args.fast_win_sec_candidates, default=DEFAULT_FAST_WIN_SEC_CANDIDATES)
    fast_template_weight_candidates = _csv_float_tuple(
        args.fast_template_weight_candidates,
        default=DEFAULT_FAST_TEMPLATE_WEIGHT_CANDIDATES,
    )
    threshold_win_sec_candidates = _csv_float_tuple(
        args.threshold_win_sec_candidates,
        default=DEFAULT_THRESHOLD_WIN_SEC_CANDIDATES,
    )
    threshold_gate_policies = _csv_str_tuple(
        args.threshold_gate_policies,
        default=DEFAULT_THRESHOLD_GATE_POLICIES,
    )
    threshold_min_enter_candidates = _csv_int_tuple(
        args.threshold_min_enter_candidates,
        default=DEFAULT_THRESHOLD_MIN_ENTER_CANDIDATES,
    )
    threshold_min_exit_candidates = _csv_int_tuple(
        args.threshold_min_exit_candidates,
        default=DEFAULT_THRESHOLD_MIN_EXIT_CANDIDATES,
    )
    threshold_control_state_modes = _csv_str_tuple(
        args.threshold_control_state_modes,
        default=DEFAULT_THRESHOLD_CONTROL_STATE_MODES,
    )
    classifier_win_sec_candidates = _csv_float_tuple(
        args.classifier_win_sec_candidates,
        default=DEFAULT_CLASSIFIER_WIN_SEC_CANDIDATES,
    )
    classifier_min_enter_candidates = _csv_int_tuple(
        args.classifier_min_enter_candidates,
        default=DEFAULT_CLASSIFIER_MIN_ENTER_CANDIDATES,
    )
    classifier_max_gap_candidates = _csv_int_tuple(
        args.classifier_max_gap_candidates,
        default=DEFAULT_CLASSIFIER_MAX_GAP_CANDIDATES,
    )
    classifier_threshold_policy = _parse_classifier_threshold_policy(args.classifier_threshold_policy)
    subject_whitelist = _parse_subject_whitelist(getattr(args, "subject_whitelist", ""))

    subjects = enumerate_external_subjects(
        datasets=datasets,
        freqs=freqs,
        wang_raw_dir=Path(args.wang_raw_dir),
        wang_channels_loc=Path(args.wang_channels_loc),
        beta_raw_dir=Path(args.beta_raw_dir),
        subject_limit_per_dataset=int(args.subject_limit_per_dataset),
        subject_whitelist=subject_whitelist,
    )
    if not subjects:
        raise RuntimeError("no external subjects found for the requested datasets")

    rows: list[dict[str, Any]] = []
    subject_manifest: list[dict[str, Any]] = []
    completed_rows_total = 0
    subject_count_total = int(len(subjects))

    def emit_progress(
        *,
        stage: str,
        detail: str,
        percent: float,
        current_dataset: str = "",
        current_subject: str = "",
        current_method: str = "",
    ) -> None:
        _write_json(
            progress_path,
            {
                "task": "external-short-pretrain-benchmark",
                "stage": str(stage),
                "stage_label": str(stage),
                "detail": str(detail),
                "progress_percent": float(max(0.0, min(100.0, percent))),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "run_id": run_id,
                "row_count": int(len(rows)),
                "completed_recipe_count": int(completed_rows_total),
                "subject_count": int(subject_count_total),
                "current_dataset": str(current_dataset),
                "current_subject": str(current_subject),
                "current_method": str(current_method),
                "report_dir": str(report_root),
                "summary_path": str(report_root / "summary.json"),
            },
        )

    def emit_partial(stage: str, detail: str) -> None:
        partial_summaries = aggregate_recipe_rows(rows, expected_subject_count=len(subjects))
        partial_shared_summaries = _shared_recipe_summaries(partial_summaries)
        _write_json(
            partial_summary_path,
            {
                "task": "external-short-pretrain-benchmark",
                "status": "running",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "run_id": run_id,
                "freqs": [float(freq) for freq in freqs],
                "datasets": list(datasets),
                "methods": list(methods),
                "expected_subject_count": int(len(subjects)),
                "row_count": int(len(rows)),
                "completed_recipe_count": int(completed_rows_total),
                "current_stage": str(stage),
                "current_detail": str(detail),
                "best_recipe": dict(partial_summaries[0]) if partial_summaries else {},
                "best_shared_recipe": dict(partial_shared_summaries[0]) if partial_shared_summaries else {},
                "recipe_summaries": partial_summaries,
                "shared_recipe_summaries": partial_shared_summaries,
                "subjects": subject_manifest,
                "idle_proxy_note": "Idle/no-control is proxied with non-command target stimulus trials from external benchmarks.",
            },
        )

    emit_progress(stage="start", detail="benchmark initialized", percent=0.0)
    emit_partial("start", "benchmark initialized")

    for subject_index, spec in enumerate(subjects, start=1):
        log(f"load subject dataset={spec.dataset} subject={spec.subject} path={spec.mat_path}")
        sampling_rate, segments, source_metadata = load_external_subject_segments(spec)
        counts = _count_segments(segments, freqs)
        max_supported_win_sec = _max_supported_win_sec(segments, sampling_rate)
        subject_manifest.append(
            {
                "dataset": str(spec.dataset),
                "subject": str(spec.subject),
                "mat_path": str(spec.mat_path),
                "channel_loc_path": "" if spec.channel_loc_path is None else str(spec.channel_loc_path),
                "sampling_rate": int(sampling_rate),
                "max_supported_win_sec": float(max_supported_win_sec),
                "counts": counts,
                "source_metadata": source_metadata,
            }
        )
        blocks = list(counts["blocks"])
        fast_candidates = [
            (float(win_sec), float(template_weight))
            for win_sec, template_weight in product(
                fast_win_sec_candidates,
                fast_template_weight_candidates,
            )
            if float(win_sec) <= float(max_supported_win_sec) + 1e-9
        ]
        threshold_supported_wins = tuple(
            float(win_sec)
            for win_sec in threshold_win_sec_candidates
            if float(win_sec) <= float(max_supported_win_sec) + 1e-9
        )
        score_method_candidate_pairs_by_method: dict[str, list[tuple[float, int]]] = {}
        for method_name in methods:
            if method_name not in SCORE_METHOD_SPECS:
                continue
            score_method_candidate_pairs_by_method[method_name] = _score_method_candidate_pairs(
                method_name=method_name,
                win_sec_candidates=classifier_win_sec_candidates,
                min_enter_candidates=classifier_min_enter_candidates,
                max_supported_win_sec=float(max_supported_win_sec),
                sampling_rate=int(sampling_rate),
            )
        plans_by_calibration: list[tuple[int, list[SplitPlan]]] = []
        subject_planned_rows = 0
        rows_per_split = 0
        if "zero_shot_default" in methods:
            rows_per_split += 1
        if "fast_fbcca" in methods:
            rows_per_split += len(fast_candidates)
        for method_name, candidate_pairs in score_method_candidate_pairs_by_method.items():
            rows_per_split += int(len(candidate_pairs) * len(classifier_max_gap_candidates))
        if "threshold_pretrain" in methods and threshold_supported_wins:
            rows_per_split += 1
        for calibration_block_count in calibration_blocks:
            plans = build_block_split_plans(
                dataset=spec.dataset,
                subject=spec.subject,
                block_indices=blocks,
                calibration_block_count=int(calibration_block_count),
                max_splits=int(args.max_splits_per_subject),
                seed=int(args.seed),
            )
            plans_by_calibration.append((int(calibration_block_count), plans))
            subject_planned_rows += int(len(plans) * len(idle_multipliers) * rows_per_split)
        subject_completed_rows = 0

        def append_row(row_payload: dict[str, Any], *, method_name: str, detail: str) -> None:
            nonlocal completed_rows_total, subject_completed_rows
            rows.append(row_payload)
            completed_rows_total += 1
            subject_completed_rows += 1
            subject_fraction = float(subject_completed_rows) / float(max(subject_planned_rows, 1))
            percent = 100.0 * (
                (float(subject_index - 1) + min(max(subject_fraction, 0.0), 1.0)) / float(max(subject_count_total, 1))
            )
            emit_progress(
                stage="evaluate_recipe",
                detail=detail,
                percent=percent,
                current_dataset=str(spec.dataset),
                current_subject=str(spec.subject),
                current_method=str(method_name),
            )
            emit_partial("evaluate_recipe", detail)

        emit_progress(
            stage="load_subject",
            detail=f"loaded dataset={spec.dataset} subject={spec.subject}",
            percent=100.0 * float(subject_index - 1) / float(max(subject_count_total, 1)),
            current_dataset=str(spec.dataset),
            current_subject=str(spec.subject),
        )
        emit_partial("load_subject", f"loaded dataset={spec.dataset} subject={spec.subject}")
        for calibration_block_count, plans in plans_by_calibration:
            if not plans:
                log(
                    f"skip subject={spec.subject} dataset={spec.dataset} calibration_blocks={int(calibration_block_count)}"
                )
                continue
            for plan in plans:
                for idle_multiplier in idle_multipliers:
                    calibration_segments, holdout_segments, split_summary = select_split_segments(
                        segments,
                        freqs=freqs,
                        calibration_blocks=plan.calibration_blocks,
                        holdout_blocks=plan.holdout_blocks,
                        idle_multiplier=float(idle_multiplier),
                        seed=int(plan.seed),
                    )
                    split_summary["calibration_duration_sec"] = _collection_duration_sec(
                        calibration_segments,
                        sampling_rate,
                    )
                    split_summary["holdout_duration_sec"] = _collection_duration_sec(holdout_segments, sampling_rate)
                    method_root = report_root / spec.dataset / spec.subject / (
                        f"cb{int(calibration_block_count)}_idle{float(idle_multiplier):g}_split{int(plan.split_index):02d}"
                    )
                    score_bundle_cache: dict[tuple[str, float], tuple[list[ScoredTrial], list[ScoredTrial]]] = {}
                    lda_base_cache: dict[tuple[str, float], FBCCALDA5Model] = {}
                    ridge_base_cache: dict[tuple[str, float], list[FBCCARidge5Model]] = {}

                    def _method_cache_key(method_name: str, win_sec: float) -> tuple[str, float]:
                        namespace = _score_method_cache_namespace(method_name)
                        return (namespace, round(float(win_sec), 9))

                    def scored_for_method_win(method_name: str, win_sec: float) -> tuple[list[ScoredTrial], list[ScoredTrial]]:
                        cache_key = _method_cache_key(method_name, float(win_sec))
                        if cache_key not in score_bundle_cache:
                            score_bundle_cache[cache_key] = _score_split_once_for_method(
                                method_name=str(method_name),
                                freqs=freqs,
                                sampling_rate=sampling_rate,
                                step_sec=float(args.step_sec),
                                compute_backend=str(args.compute_backend),
                                gpu_device=int(args.gpu_device),
                                gpu_precision=str(args.gpu_precision),
                                calibration_segments=calibration_segments,
                                holdout_segments=holdout_segments,
                                win_sec=float(win_sec),
                                context=(
                                    f"{str(method_name)} dataset={spec.dataset} subject={spec.subject} "
                                    f"split={int(plan.split_index)} win={float(win_sec):g}"
                                ),
                            )
                        return score_bundle_cache[cache_key]

                    def lda_base_for_win(method_name: str, win_sec: float) -> FBCCALDA5Model:
                        cache_key = _method_cache_key(method_name, float(win_sec))
                        if cache_key not in lda_base_cache:
                            calibration_scored, _holdout_scored = scored_for_method_win(method_name, float(win_sec))
                            try:
                                lda_base_cache[cache_key] = _fit_fbcca_lda5_base_model(
                                    calibration_scored,
                                    freqs=freqs,
                                    score_source_name=_score_method_spec(method_name).score_source_name,
                                )
                            except TypeError as exc:
                                if "score_source_name" not in str(exc):
                                    raise
                                lda_base_cache[cache_key] = _fit_fbcca_lda5_base_model(
                                    calibration_scored,
                                    freqs=freqs,
                                )
                        return lda_base_cache[cache_key]

                    def ridge_bases_for_win(method_name: str, win_sec: float) -> list[FBCCARidge5Model]:
                        cache_key = _method_cache_key(method_name, float(win_sec))
                        if cache_key not in ridge_base_cache:
                            calibration_scored, _holdout_scored = scored_for_method_win(method_name, float(win_sec))
                            ridge_base_models: list[FBCCARidge5Model] = []
                            for l2 in DEFAULT_RIDGE_L2_CANDIDATES:
                                try:
                                    base_model = _fit_fbcca_ridge5_base_model(
                                        calibration_scored,
                                        freqs=freqs,
                                        l2=float(l2),
                                        score_source_name=_score_method_spec(method_name).score_source_name,
                                    )
                                except TypeError as exc:
                                    if "score_source_name" not in str(exc):
                                        raise
                                    base_model = _fit_fbcca_ridge5_base_model(
                                        calibration_scored,
                                        freqs=freqs,
                                        l2=float(l2),
                                    )
                                ridge_base_models.append(base_model)
                            ridge_base_cache[cache_key] = ridge_base_models
                        return ridge_base_cache[cache_key]

                    if "zero_shot_default" in methods:
                        detail = (
                            "zero_shot_default "
                            f"dataset={spec.dataset} subject={spec.subject} split={int(plan.split_index)} "
                            f"cal_blocks={len(plan.calibration_blocks)} idle_mult={float(idle_multiplier):g}"
                        )
                        log(detail)
                        append_row(
                            run_zero_shot_default(
                                spec=spec,
                                split_plan=plan,
                                split_summary=split_summary,
                                sampling_rate=sampling_rate,
                                freqs=freqs,
                                holdout_segments=holdout_segments,
                                compute_backend=str(args.compute_backend),
                                gpu_device=int(args.gpu_device),
                                gpu_precision=str(args.gpu_precision),
                            ),
                            method_name="zero_shot_default",
                            detail=detail,
                        )
                    if "fast_fbcca" in methods:
                        if not fast_candidates:
                            log(
                                f"skip fast_fbcca dataset={spec.dataset} subject={spec.subject} "
                                f"because no win candidate fits max_supported_win_sec={float(max_supported_win_sec):g}s"
                            )
                        for win_sec, template_weight in fast_candidates:
                            detail = (
                                "fast_fbcca "
                                f"dataset={spec.dataset} subject={spec.subject} split={int(plan.split_index)} "
                                f"cal_blocks={len(plan.calibration_blocks)} idle_mult={float(idle_multiplier):g} "
                                f"win={float(win_sec):g} tw={float(template_weight):g}"
                            )
                            log(detail)
                            append_row(
                                run_fast_fbcca_method(
                                    method_dir=method_root / "fast_fbcca",
                                    spec=spec,
                                    split_plan=plan,
                                    split_summary=split_summary,
                                    sampling_rate=sampling_rate,
                                    freqs=freqs,
                                    calibration_segments=calibration_segments,
                                    holdout_segments=holdout_segments,
                                    compute_backend=str(args.compute_backend),
                                    gpu_device=int(args.gpu_device),
                                    gpu_precision=str(args.gpu_precision),
                                    step_sec=float(args.step_sec),
                                    win_sec=float(win_sec),
                                    template_weight=float(template_weight),
                                ),
                                method_name="fast_fbcca",
                                detail=detail,
                            )
                    if "fbcca_lda5" in methods:
                        lda_candidates = score_method_candidate_pairs_by_method.get("fbcca_lda5", [])
                        if not lda_candidates:
                            log(
                                f"skip fbcca_lda5 dataset={spec.dataset} subject={spec.subject} "
                                f"because no win candidate fits max_supported_win_sec={float(max_supported_win_sec):g}s"
                            )
                        for win_sec, min_enter in lda_candidates:
                            calibration_scored, holdout_scored = scored_for_method_win("fbcca_lda5", float(win_sec))
                            lda_base_model = lda_base_for_win("fbcca_lda5", float(win_sec))
                            for max_gap in classifier_max_gap_candidates:
                                detail = (
                                    "fbcca_lda5 "
                                    f"dataset={spec.dataset} subject={spec.subject} split={int(plan.split_index)} "
                                    f"cal_blocks={len(plan.calibration_blocks)} idle_mult={float(idle_multiplier):g} "
                                    f"win={float(win_sec):g} min_enter={int(min_enter)} max_gap={int(max_gap)} "
                                    f"threshold_policy={classifier_threshold_policy}"
                                )
                                log(detail)
                                append_row(
                                    run_fbcca_lda5_method(
                                        artifact_dir=method_root / "fbcca_lda5",
                                        spec=spec,
                                        split_plan=plan,
                                        split_summary=split_summary,
                                        sampling_rate=sampling_rate,
                                        freqs=freqs,
                                        calibration_segments=calibration_segments,
                                        holdout_segments=holdout_segments,
                                        compute_backend=str(args.compute_backend),
                                        gpu_device=int(args.gpu_device),
                                        gpu_precision=str(args.gpu_precision),
                                        step_sec=float(args.step_sec),
                                        win_sec=float(win_sec),
                                        min_enter_windows=int(min_enter),
                                        max_gap_windows=int(max_gap),
                                        threshold_policy=classifier_threshold_policy,
                                        calibration_scored=calibration_scored,
                                        holdout_scored=holdout_scored,
                                        base_model=lda_base_model,
                                    ),
                                    method_name="fbcca_lda5",
                                    detail=detail,
                                )
                    if "fbcca_ridge5" in methods:
                        ridge_candidates = score_method_candidate_pairs_by_method.get("fbcca_ridge5", [])
                        if not ridge_candidates:
                            log(
                                f"skip fbcca_ridge5 dataset={spec.dataset} subject={spec.subject} "
                                f"because no win candidate fits max_supported_win_sec={float(max_supported_win_sec):g}s"
                            )
                        for win_sec, min_enter in ridge_candidates:
                            calibration_scored, holdout_scored = scored_for_method_win("fbcca_ridge5", float(win_sec))
                            ridge_base_models = ridge_bases_for_win("fbcca_ridge5", float(win_sec))
                            for max_gap in classifier_max_gap_candidates:
                                detail = (
                                    "fbcca_ridge5 "
                                    f"dataset={spec.dataset} subject={spec.subject} split={int(plan.split_index)} "
                                    f"cal_blocks={len(plan.calibration_blocks)} idle_mult={float(idle_multiplier):g} "
                                    f"win={float(win_sec):g} min_enter={int(min_enter)} max_gap={int(max_gap)} "
                                    f"threshold_policy={classifier_threshold_policy}"
                                )
                                log(detail)
                                append_row(
                                    run_fbcca_ridge5_method(
                                        artifact_dir=method_root / "fbcca_ridge5",
                                        spec=spec,
                                        split_plan=plan,
                                        split_summary=split_summary,
                                        sampling_rate=sampling_rate,
                                        freqs=freqs,
                                        calibration_segments=calibration_segments,
                                        holdout_segments=holdout_segments,
                                        compute_backend=str(args.compute_backend),
                                        gpu_device=int(args.gpu_device),
                                        gpu_precision=str(args.gpu_precision),
                                        step_sec=float(args.step_sec),
                                        win_sec=float(win_sec),
                                        min_enter_windows=int(min_enter),
                                        max_gap_windows=int(max_gap),
                                        threshold_policy=classifier_threshold_policy,
                                        calibration_scored=calibration_scored,
                                        holdout_scored=holdout_scored,
                                        base_models=ridge_base_models,
                                    ),
                                    method_name="fbcca_ridge5",
                                    detail=detail,
                                )
                    for method_name in SUPPORTED_SHORT_PRETRAIN_METHODS:
                        if method_name not in methods or method_name in {"fbcca_lda5", "fbcca_ridge5"}:
                            continue
                        method_candidates = score_method_candidate_pairs_by_method.get(method_name, [])
                        if not method_candidates:
                            log(
                                f"skip {method_name} dataset={spec.dataset} subject={spec.subject} "
                                f"because no win candidate fits max_supported_win_sec={float(max_supported_win_sec):g}s"
                            )
                            continue
                        method_spec = _score_method_spec(method_name)
                        for win_sec, min_enter in method_candidates:
                            calibration_scored, holdout_scored = scored_for_method_win(method_name, float(win_sec))
                            ridge_base_models = ridge_bases_for_win(method_name, float(win_sec))
                            for max_gap in classifier_max_gap_candidates:
                                detail = (
                                    f"{method_name} "
                                    f"dataset={spec.dataset} subject={spec.subject} split={int(plan.split_index)} "
                                    f"cal_blocks={len(plan.calibration_blocks)} idle_mult={float(idle_multiplier):g} "
                                    f"win={float(win_sec):g} min_enter={int(min_enter)} max_gap={int(max_gap)} "
                                    f"threshold_policy={classifier_threshold_policy}"
                                )
                                log(detail)
                                append_row(
                                    run_fbcca_ridge5_method(
                                        artifact_dir=method_root / method_name,
                                        spec=spec,
                                        split_plan=plan,
                                        split_summary=split_summary,
                                        sampling_rate=sampling_rate,
                                        freqs=freqs,
                                        calibration_segments=calibration_segments,
                                        holdout_segments=holdout_segments,
                                        compute_backend=str(args.compute_backend),
                                        gpu_device=int(args.gpu_device),
                                        gpu_precision=str(args.gpu_precision),
                                        step_sec=float(args.step_sec),
                                        win_sec=float(win_sec),
                                        min_enter_windows=int(min_enter),
                                        max_gap_windows=int(max_gap),
                                        threshold_policy=classifier_threshold_policy,
                                        calibration_scored=calibration_scored,
                                        holdout_scored=holdout_scored,
                                        base_models=ridge_base_models,
                                        method_name=str(method_name),
                                        score_source_name=str(method_spec.score_source_name),
                                        decoder_name=str(method_spec.decoder_name),
                                        decoder_model_params=dict(method_spec.decoder_model_params),
                                    ),
                                    method_name=str(method_name),
                                    detail=detail,
                                )
                    if "threshold_pretrain" in methods:
                        if not threshold_supported_wins:
                            log(
                                f"skip threshold_pretrain dataset={spec.dataset} subject={spec.subject} "
                                f"because no win candidate fits max_supported_win_sec={float(max_supported_win_sec):g}s"
                            )
                            continue
                        detail = (
                            "threshold_pretrain "
                            f"dataset={spec.dataset} subject={spec.subject} split={int(plan.split_index)} "
                            f"cal_blocks={len(plan.calibration_blocks)} idle_mult={float(idle_multiplier):g}"
                        )
                        log(detail)
                        append_row(
                            run_threshold_pretrain_method(
                                method_dir=method_root / "threshold_pretrain",
                                dataset_root=dataset_root / "threshold_pretrain",
                                spec=spec,
                                split_plan=plan,
                                split_summary=split_summary,
                                sampling_rate=sampling_rate,
                                freqs=freqs,
                                calibration_segments=calibration_segments,
                                holdout_segments=holdout_segments,
                                compute_backend=str(args.compute_backend),
                                gpu_device=int(args.gpu_device),
                                gpu_precision=str(args.gpu_precision),
                                step_sec=float(args.step_sec),
                                win_sec_candidates=threshold_supported_wins,
                                gate_policies=threshold_gate_policies,
                                min_enter_candidates=threshold_min_enter_candidates,
                                min_exit_candidates=threshold_min_exit_candidates,
                                control_state_modes=threshold_control_state_modes,
                            ),
                            method_name="threshold_pretrain",
                            detail=detail,
                        )

    summaries = aggregate_recipe_rows(rows, expected_subject_count=len(subjects))
    shared_summaries = _shared_recipe_summaries(summaries)
    summary = {
        "task": "external-short-pretrain-benchmark",
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "config": json_safe(vars(args)),
        "freqs": [float(freq) for freq in freqs],
        "datasets": list(datasets),
        "methods": list(methods),
        "subject_whitelist": [[dataset, subject] for dataset, subject in subject_whitelist],
        "subjects": subject_manifest,
        "expected_subject_count": int(len(subjects)),
        "row_count": int(len(rows)),
        "rows": rows,
        "recipe_summaries": summaries,
        "shared_recipe_summaries": shared_summaries,
        "best_recipe": dict(summaries[0]) if summaries else {},
        "best_shared_recipe": dict(shared_summaries[0]) if shared_summaries else {},
        "idle_proxy_note": "Idle/no-control is proxied with non-command target stimulus trials from external benchmarks.",
    }
    _write_json(report_root / "summary.json", summary)
    emit_progress(stage="complete", detail="benchmark complete", percent=100.0)
    (report_root / "summary.md").write_text(
        render_markdown_summary(
            run_id=run_id,
            freqs=freqs,
            subjects=subjects,
            rows=rows,
            summaries=summaries,
            shared_summaries=shared_summaries,
        ),
        encoding="utf-8",
    )
    _write_json(partial_summary_path, {**summary, "status": "ok"})
    log(f"complete summary={report_root / 'summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark short-pretrain 5-class SSVEP classifiers on external datasets.")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--datasets", type=str, default="wang2016,beta")
    parser.add_argument(
        "--methods",
        type=str,
        default="zero_shot_default,fast_fbcca,threshold_pretrain,fbcca_lda5,fbcca_ridge5",
    )
    parser.add_argument("--freqs", type=str, default=",".join(f"{float(freq):g}" for freq in DEFAULT_FREQS))
    parser.add_argument("--wang-raw-dir", type=Path, required=True)
    parser.add_argument("--wang-channels-loc", type=Path, required=True)
    parser.add_argument("--beta-raw-dir", type=Path, required=True)
    parser.add_argument("--subject-limit-per-dataset", type=int, default=0)
    parser.add_argument("--subject-whitelist", type=str, default="")
    parser.add_argument("--calibration-blocks", type=str, default="1,2,3")
    parser.add_argument("--idle-multipliers", type=str, default="1.0,2.0")
    parser.add_argument("--max-splits-per-subject", type=int, default=DEFAULT_MAX_SPLITS_PER_SUBJECT)
    parser.add_argument("--step-sec", type=float, default=DEFAULT_STEP_SEC)
    parser.add_argument("--compute-backend", type=str, default="cuda")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-precision", type=str, default="float32")
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--fast-win-sec-candidates", type=str, default="1.5,2.0,2.5")
    parser.add_argument("--fast-template-weight-candidates", type=str, default="0.15,0.25,0.35")
    parser.add_argument("--classifier-win-sec-candidates", type=str, default="1.5,2.0,2.5")
    parser.add_argument("--classifier-min-enter-candidates", type=str, default="1,2")
    parser.add_argument("--classifier-max-gap-candidates", type=str, default="0")
    parser.add_argument(
        "--classifier-threshold-policy",
        type=str,
        default=DEFAULT_CLASSIFIER_THRESHOLD_POLICY,
        choices=CLASSIFIER_THRESHOLD_POLICIES,
    )
    parser.add_argument("--threshold-win-sec-candidates", type=str, default="1.5,2.0,2.5,3.0")
    parser.add_argument("--threshold-gate-policies", type=str, default="balanced,speed")
    parser.add_argument("--threshold-min-enter-candidates", type=str, default="1,2")
    parser.add_argument("--threshold-min-exit-candidates", type=str, default="1,2")
    parser.add_argument(
        "--threshold-control-state-modes",
        type=str,
        default="unified,frequency-specific-threshold",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_benchmark(args)
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
