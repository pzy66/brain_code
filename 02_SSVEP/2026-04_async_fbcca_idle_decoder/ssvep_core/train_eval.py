from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
import re
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np
from scipy.stats import chi2, wilcoxon

from async_fbcca_idle_standalone import (
    AsyncDecisionGate,
    DEFAULT_COMPUTE_BACKEND_NAME,
    DEFAULT_BENCHMARK_MODELS,
    DEFAULT_BENCHMARK_MULTI_SEED_COUNT,
    DEFAULT_BENCHMARK_SEED_STEP,
    DEFAULT_BENCHMARK_CHANNEL_MODES,
    DEFAULT_CALIBRATION_SEED,
    DEFAULT_CHANNEL_WEIGHT_MODE,
    DEFAULT_SUBBAND_WEIGHT_MODE,
    DEFAULT_FBCCA_WEIGHT_CV_FOLDS,
    DEFAULT_ASYNC_DECISION_TIME_MODE,
    DEFAULT_DATA_POLICY,
    DEFAULT_DYNAMIC_STOP_ALPHA,
    DEFAULT_DYNAMIC_STOP_ENABLED,
    DEFAULT_EXPORT_FIGURES,
    DEFAULT_GATE_POLICY,
    DEFAULT_GPU_CACHE_MODE,
    DEFAULT_GPU_DEVICE_ID,
    DEFAULT_GPU_PRECISION_NAME,
    DEFAULT_JOINT_WEIGHT_ITERS,
    DEFAULT_METRIC_SCOPE,
    DEFAULT_PAPER_DECISION_TIME_MODE,
    DEFAULT_RANKING_POLICY,
    DEFAULT_SPATIAL_FILTER_MODE,
    DEFAULT_NH,
    DEFAULT_SPATIAL_RANK_CANDIDATES,
    DEFAULT_SPATIAL_SOURCE_MODEL,
    DEFAULT_WIN_SEC_CANDIDATES,
    BenchmarkRunner,
    atomic_copy_text_file,
    atomic_write_text,
    benchmark_metric_definition_payload,
    benchmark_rank_key,
    compute_classification_metrics,
    create_decoder,
    evaluate_decoder_on_trials,
    evaluate_decoder_on_trials_v2,
    json_dumps,
    json_safe,
    load_decoder_from_profile,
    load_profile,
    model_implementation_level,
    model_method_note,
    normalize_model_name,
    parse_channel_mode_list,
    parse_data_policy,
    parse_decision_time_mode,
    parse_gate_policy,
    parse_metric_scope,
    parse_model_list,
    parse_ranking_policy,
    pack_evaluation_metrics_for_ranking,
    profile_meets_acceptance,
    save_profile,
    select_auto_eeg_channels_for_model,
    split_trial_segments_for_benchmark,
    summarize_benchmark_robustness,
    _subset_trial_segments_by_positions,
)

from .dataset import LoadedDataset, build_protocol_signature, load_collection_dataset
_REPORTING_IMPORT_ERROR: Optional[Exception] = None
try:
    from .reporting import export_evaluation_figures
except Exception as exc:  # pragma: no cover - optional dependency fallback
    _REPORTING_IMPORT_ERROR = exc

    def export_evaluation_figures(*_args, **_kwargs):
        raise RuntimeError("matplotlib is required for figure export") from _REPORTING_IMPORT_ERROR


DEFAULT_EVALUATION_MODE = "staged"
DEFAULT_TRAIN_EVAL_TASK = "fbcca-weighted-compare"
DEFAULT_QUICK_SCREEN_TOP_K = 4
DEFAULT_FORCE_INCLUDE_MODELS = ("legacy_fbcca_202603", "fbcca")
DEFAULT_PROGRESS_HEARTBEAT_SEC = 5.0


@dataclass(frozen=True)
class OfflineTrainEvalConfig:
    dataset_manifest_session1: Path
    dataset_manifest_session2: Optional[Path]
    output_profile_path: Path
    report_path: Path
    dataset_manifests: tuple[Path, ...] = ()
    report_root_dir: Optional[Path] = None
    organize_report_dir: bool = False
    dataset_selection_snapshot: Optional[dict[str, Any]] = None
    quality_min_sample_ratio: float = 0.9
    quality_max_retry_count: int = 3
    strict_protocol_consistency: bool = True
    strict_subject_consistency: bool = True
    model_names: tuple[str, ...] = tuple(DEFAULT_BENCHMARK_MODELS)
    channel_modes: tuple[str, ...] = tuple(DEFAULT_BENCHMARK_CHANNEL_MODES)
    multi_seed_count: int = DEFAULT_BENCHMARK_MULTI_SEED_COUNT
    seed_step: int = DEFAULT_BENCHMARK_SEED_STEP
    win_candidates: tuple[float, ...] = tuple(DEFAULT_WIN_SEC_CANDIDATES)
    gate_policy: str = DEFAULT_GATE_POLICY
    channel_weight_mode: Optional[str] = DEFAULT_CHANNEL_WEIGHT_MODE
    subband_weight_mode: Optional[str] = DEFAULT_SUBBAND_WEIGHT_MODE
    spatial_filter_mode: Optional[str] = DEFAULT_SPATIAL_FILTER_MODE
    spatial_rank_candidates: tuple[int, ...] = tuple(DEFAULT_SPATIAL_RANK_CANDIDATES)
    joint_weight_iters: int = DEFAULT_JOINT_WEIGHT_ITERS
    weight_cv_folds: int = DEFAULT_FBCCA_WEIGHT_CV_FOLDS
    spatial_source_model: str = DEFAULT_SPATIAL_SOURCE_MODEL
    metric_scope: str = DEFAULT_METRIC_SCOPE
    decision_time_mode: str = DEFAULT_PAPER_DECISION_TIME_MODE
    async_decision_time_mode: str = DEFAULT_ASYNC_DECISION_TIME_MODE
    data_policy: str = DEFAULT_DATA_POLICY
    export_figures: bool = DEFAULT_EXPORT_FIGURES
    ranking_policy: str = DEFAULT_RANKING_POLICY
    dynamic_stop_enabled: bool = DEFAULT_DYNAMIC_STOP_ENABLED
    dynamic_stop_alpha: float = DEFAULT_DYNAMIC_STOP_ALPHA
    seed: int = DEFAULT_CALIBRATION_SEED
    task: str = DEFAULT_TRAIN_EVAL_TASK
    evaluation_mode: str = DEFAULT_EVALUATION_MODE
    quick_screen_top_k: int = DEFAULT_QUICK_SCREEN_TOP_K
    force_include_models: tuple[str, ...] = DEFAULT_FORCE_INCLUDE_MODELS
    progress_heartbeat_sec: float = DEFAULT_PROGRESS_HEARTBEAT_SEC
    compute_backend: str = DEFAULT_COMPUTE_BACKEND_NAME
    gpu_device: int = DEFAULT_GPU_DEVICE_ID
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME
    gpu_warmup: bool = True
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE


def _sanitize_path_token(value: str, *, fallback: str) -> str:
    token = "".join(ch if (str(ch).isalnum() or ch in {"-", "_"}) else "_" for ch in str(value).strip())
    token = token.strip("_")
    return token or fallback


def _resolve_report_paths(
    config: OfflineTrainEvalConfig,
    *,
    subject_id: str,
) -> dict[str, Path]:
    now = datetime.now()
    if bool(config.organize_report_dir):
        root_dir = (
            Path(config.report_root_dir).expanduser().resolve()
            if config.report_root_dir is not None
            else Path(config.report_path).expanduser().resolve().parent
        )
        run_date = now.strftime("%Y%m%d")
        run_stamp = now.strftime("%Y%m%d_%H%M%S")
        subject_tag = _sanitize_path_token(subject_id, fallback="mixed")
        report_dir = root_dir / run_date / f"run_{run_stamp}_{subject_tag}"
        report_json = report_dir / "offline_train_eval.json"
    else:
        report_json = Path(config.report_path).expanduser().resolve()
        report_dir = report_json.parent
    return {
        "report_dir": report_dir,
        "report_json": report_json,
        "report_md": report_json.with_suffix(".md"),
        "selection_snapshot": report_dir / "selection_snapshot.json",
        "run_config": report_dir / "run_config.json",
        "run_log": report_dir / "run.log",
        "progress_snapshot": report_dir / "progress_snapshot.json",
    }


def _collect_manifest_stage_values(dataset: LoadedDataset) -> set[str]:
    stages: set[str] = set()
    for row in list(dataset.manifest.get("trials", [])):
        if not isinstance(row, dict):
            continue
        value = str(row.get("stage", "")).strip().lower()
        if value:
            stages.add(value)
    return stages


def _safe_float(value: Any, *, default: float) -> float:
    try:
        converted = float(value)
    except Exception:
        return float(default)
    return float(converted) if np.isfinite(converted) else float(default)


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _canonical_protocol_signature(dataset: LoadedDataset) -> dict[str, Any]:
    cfg = dict(dataset.protocol_config or {})
    return {
        "prepare_sec": round(_safe_float(cfg.get("prepare_sec", 1.0), default=1.0), 6),
        "active_sec": round(_safe_float(cfg.get("active_sec", 4.0), default=4.0), 6),
        "rest_sec": round(_safe_float(cfg.get("rest_sec", 1.0), default=1.0), 6),
        "step_sec": round(_safe_float(cfg.get("step_sec", 0.25), default=0.25), 6),
        "target_repeats": _safe_int(cfg.get("target_repeats", 0), default=0),
        "idle_repeats": _safe_int(cfg.get("idle_repeats", 0), default=0),
        "switch_trials": _safe_int(cfg.get("switch_trials", 0), default=0),
        "preset_name": str(cfg.get("preset_name", "")).strip().lower(),
    }


def _manifest_protocol_signature(dataset: LoadedDataset) -> str:
    raw = dataset.manifest.get("protocol_signature")
    if isinstance(raw, str) and raw.strip():
        return str(raw).strip()
    cfg_raw = dataset.protocol_config.get("protocol_signature")
    if isinstance(cfg_raw, str) and cfg_raw.strip():
        return str(cfg_raw).strip()
    return ""


def _derived_protocol_signature(dataset: LoadedDataset) -> str:
    return build_protocol_signature(
        sampling_rate=int(dataset.sampling_rate),
        protocol_config=dict(dataset.protocol_config or {}),
        freqs=tuple(float(value) for value in dataset.freqs),
        board_eeg_channels=tuple(int(value) for value in dataset.board_eeg_channels),
    )


def _protocol_mismatch_details(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keys = set(reference.keys()) | set(candidate.keys())
    diff: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        ref_value = reference.get(key)
        cand_value = candidate.get(key)
        if ref_value != cand_value:
            diff[str(key)] = {"reference": ref_value, "candidate": cand_value}
    return diff


def _sorted_manifest_trial_rows(dataset: LoadedDataset) -> list[dict[str, Any]]:
    rows = [row for row in list(dataset.manifest.get("trials", [])) if isinstance(row, dict)]
    rows.sort(key=lambda row: int(row.get("order_index", 0)))
    return rows


def _apply_trial_quality_filter(
    dataset: LoadedDataset,
    *,
    min_sample_ratio: float,
    max_retry_count: int,
) -> tuple[LoadedDataset, dict[str, Any]]:
    segments = list(dataset.trial_segments)
    rows = _sorted_manifest_trial_rows(dataset)
    fallback_target = max(
        1,
        int(
            round(
                _safe_float(dataset.protocol_config.get("active_sec", 0.0), default=0.0)
                * float(max(int(dataset.sampling_rate), 1))
            )
        ),
    )
    if fallback_target <= 0:
        fallback_target = max(1, int(segments[0][1].shape[0])) if segments else 1

    kept_segments: list[tuple[Any, Any]] = []
    kept_rows: list[dict[str, Any]] = []
    sample_ratios_all: list[float] = []
    sample_ratios_kept: list[float] = []
    dropped_shortfall = 0
    dropped_retry = 0

    ratio_th = max(0.0, float(min_sample_ratio))
    retry_th = max(0, int(max_retry_count))
    for idx, (trial, segment) in enumerate(segments):
        row = rows[idx] if idx < len(rows) else {}
        used_samples = _safe_int(row.get("used_samples", int(segment.shape[0])), default=int(segment.shape[0]))
        target_samples = _safe_int(row.get("target_samples", fallback_target), default=fallback_target)
        target_samples = max(1, int(target_samples))
        retry_count = max(0, _safe_int(row.get("retry_count", 0), default=0))
        sample_ratio = float(used_samples) / float(target_samples)
        sample_ratios_all.append(sample_ratio)

        pass_ratio = sample_ratio >= ratio_th
        pass_retry = retry_count <= retry_th
        if not pass_ratio:
            dropped_shortfall += 1
        if not pass_retry:
            dropped_retry += 1
        if pass_ratio and pass_retry:
            kept_segments.append((trial, segment))
            sample_ratios_kept.append(sample_ratio)
            if idx < len(rows):
                kept_rows.append(dict(rows[idx]))

    filtered_manifest = dict(dataset.manifest)
    if rows:
        filtered_manifest["trials"] = kept_rows

    filtered_dataset = LoadedDataset(
        manifest_path=dataset.manifest_path,
        npz_path=dataset.npz_path,
        session_id=str(dataset.session_id),
        subject_id=str(dataset.subject_id),
        sampling_rate=int(dataset.sampling_rate),
        freqs=tuple(dataset.freqs),
        board_eeg_channels=tuple(dataset.board_eeg_channels),
        protocol_config=dict(dataset.protocol_config),
        trial_segments=list(kept_segments),
        manifest=filtered_manifest,
    )
    total_trials = int(len(segments))
    kept_trials = int(len(kept_segments))
    summary = {
        "manifest_path": str(dataset.manifest_path),
        "session_id": str(dataset.session_id),
        "subject_id": str(dataset.subject_id),
        "total_trials": total_trials,
        "kept_trials": kept_trials,
        "dropped_trials": int(max(total_trials - kept_trials, 0)),
        "drop_ratio": float(max(total_trials - kept_trials, 0) / max(total_trials, 1)),
        "dropped_shortfall": int(dropped_shortfall),
        "dropped_retry": int(dropped_retry),
        "rows_with_quality": int(len(rows)),
        "min_sample_ratio": float(ratio_th),
        "max_retry_count": int(retry_th),
        "sample_ratio_mean_all": float(np.mean(np.asarray(sample_ratios_all, dtype=float))) if sample_ratios_all else 0.0,
        "sample_ratio_mean_kept": (
            float(np.mean(np.asarray(sample_ratios_kept, dtype=float))) if sample_ratios_kept else 0.0
        ),
    }
    return filtered_dataset, summary


def _load_session1_dataset(
    config: OfflineTrainEvalConfig,
) -> tuple[LoadedDataset, tuple[Path, ...], set[str], list[dict[str, Any]], dict[str, Any]]:
    manifest_paths: tuple[Path, ...]
    if config.dataset_manifests:
        manifest_paths = tuple(Path(path).expanduser().resolve() for path in config.dataset_manifests)
    else:
        manifest_paths = (Path(config.dataset_manifest_session1).expanduser().resolve(),)
    data_policy = parse_data_policy(config.data_policy)
    datasets = [load_collection_dataset(path) for path in manifest_paths]
    if not datasets:
        raise RuntimeError("no dataset manifests selected for training/evaluation")
    reference = datasets[0]
    merged_trials: list[tuple[Any, Any]] = []
    merged_manifest_trials: list[dict[str, Any]] = []
    subjects: set[str] = set()
    session_ids: list[str] = []
    stage_values: set[str] = set()
    quality_rows: list[dict[str, Any]] = []
    reference_protocol = _canonical_protocol_signature(reference)
    reference_manifest_signature = _manifest_protocol_signature(reference)
    reference_derived_signature = _derived_protocol_signature(reference)
    if data_policy == "new-only" and not reference_manifest_signature:
        raise RuntimeError(
            "data-policy=new-only requires protocol_signature in manifest; "
            f"missing at {manifest_paths[0]}"
        )
    protocol_signature_expected = (
        str(reference_manifest_signature)
        if reference_manifest_signature
        else str(reference_derived_signature)
    )
    protocol_signature_source = "manifest" if reference_manifest_signature else "derived"
    protocol_check = {
        "data_policy": str(data_policy),
        "strict_protocol_consistency": bool(config.strict_protocol_consistency),
        "strict_subject_consistency": bool(config.strict_subject_consistency),
        "reference_protocol": dict(reference_protocol),
        "protocol_signature_expected": str(protocol_signature_expected),
        "protocol_signature_source": str(protocol_signature_source),
        "reference_subject_id": str(reference.subject_id),
        "checked_dataset_count": int(len(datasets)),
        "excluded_sessions": [],
    }
    for source_path, ds in zip(manifest_paths, datasets):
        if int(ds.sampling_rate) != int(reference.sampling_rate):
            raise RuntimeError("selected datasets have inconsistent sampling rates")
        if tuple(ds.freqs) != tuple(reference.freqs):
            raise RuntimeError("selected datasets have inconsistent freqs")
        if tuple(ds.board_eeg_channels) != tuple(reference.board_eeg_channels):
            raise RuntimeError("selected datasets have inconsistent EEG channel mapping")
        candidate_manifest_signature = _manifest_protocol_signature(ds)
        candidate_signature = (
            str(candidate_manifest_signature) if candidate_manifest_signature else str(_derived_protocol_signature(ds))
        )
        if data_policy == "new-only":
            if not candidate_manifest_signature:
                raise RuntimeError(
                    "data-policy=new-only requires protocol_signature in each manifest; "
                    f"missing at {source_path}"
                )
            if candidate_signature != protocol_signature_expected:
                raise RuntimeError(
                    "selected datasets have inconsistent protocol_signature under new-only policy; "
                    f"manifest={source_path}; expected={protocol_signature_expected}; got={candidate_signature}"
                )
        if bool(config.strict_protocol_consistency):
            candidate_protocol = _canonical_protocol_signature(ds)
            mismatches = _protocol_mismatch_details(reference_protocol, candidate_protocol)
            if mismatches:
                raise RuntimeError(
                    "selected datasets have inconsistent protocol_config under strict mode; "
                    f"manifest={source_path}; mismatches={mismatches}"
                )
        if bool(config.strict_subject_consistency):
            if str(ds.subject_id) != str(reference.subject_id):
                raise RuntimeError(
                    "selected datasets include different subject_id under strict mode; "
                    f"reference={reference.subject_id}, candidate={ds.subject_id}, manifest={source_path}"
                )
        filtered_ds, quality_summary = _apply_trial_quality_filter(
            ds,
            min_sample_ratio=float(config.quality_min_sample_ratio),
            max_retry_count=int(config.quality_max_retry_count),
        )
        quality_summary["manifest_path"] = str(source_path)
        quality_summary["protocol_signature"] = str(candidate_signature)
        quality_rows.append(dict(quality_summary))
        if not filtered_ds.trial_segments:
            raise RuntimeError(
                "selected dataset has zero trials after quality filtering; "
                f"manifest={source_path} min_sample_ratio={config.quality_min_sample_ratio} "
                f"max_retry_count={config.quality_max_retry_count}"
            )
        merged_trials.extend(list(filtered_ds.trial_segments))
        subjects.add(str(filtered_ds.subject_id))
        session_ids.append(str(filtered_ds.session_id))
        stage_values.update(_collect_manifest_stage_values(filtered_ds))
        for row in list(filtered_ds.manifest.get("trials", [])):
            if isinstance(row, dict):
                merged_manifest_trials.append(dict(row))
    if not merged_trials:
        raise RuntimeError("selected datasets contain no trial segments")
    if len(datasets) == 1:
        return filtered_ds, manifest_paths, stage_values, quality_rows, protocol_check
    merged_subject = next(iter(subjects)) if len(subjects) == 1 else "mixed"
    merged_dataset = LoadedDataset(
        manifest_path=reference.manifest_path,
        npz_path=reference.npz_path,
        session_id=f"merged_{_sanitize_path_token('_'.join(session_ids[:3]), fallback='sessions')}_{len(datasets)}",
        subject_id=merged_subject,
        sampling_rate=int(reference.sampling_rate),
        freqs=tuple(reference.freqs),
        board_eeg_channels=tuple(reference.board_eeg_channels),
        protocol_config={
            **dict(reference.protocol_config),
            "merged_dataset_count": int(len(datasets)),
            "merged_manifest_paths": [str(path) for path in manifest_paths],
            "quality_min_sample_ratio": float(config.quality_min_sample_ratio),
            "quality_max_retry_count": int(config.quality_max_retry_count),
        },
        trial_segments=list(merged_trials),
        manifest={
            **dict(reference.manifest),
            "session_id": f"merged_{len(datasets)}",
            "subject_id": merged_subject,
            "trials": merged_manifest_trials,
            "merged_sources": [str(path) for path in manifest_paths],
        },
    )
    return merged_dataset, manifest_paths, stage_values, quality_rows, protocol_check


def _split_session_for_train_eval(dataset: LoadedDataset, *, seed: int) -> tuple[list[Any], list[Any], list[Any]]:
    train_segments, gate_segments, holdout_segments = split_trial_segments_for_benchmark(
        dataset.trial_segments,
        seed=int(seed),
    )
    if not train_segments or not gate_segments or not holdout_segments:
        raise RuntimeError(
            "session1 split invalid; need enough trials for train/gate/holdout "
            f"(got train={len(train_segments)}, gate={len(gate_segments)}, holdout={len(holdout_segments)})"
        )
    return train_segments, gate_segments, holdout_segments


def _classifier_only_rank_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        -float(metrics.get("acc_4class", 0.0)),
        -float(metrics.get("macro_f1_4class", 0.0)),
        -float(metrics.get("itr_bpm_4class", 0.0)),
        float(metrics.get("mean_decision_time_s_4class", float("inf"))),
        float(metrics.get("inference_ms", float("inf"))),
    )


def _quick_screen_rank_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics.get("idle_fp_rate_proxy", float("inf"))),
        -float(metrics.get("control_recall_proxy", 0.0)),
        -float(metrics.get("acc_4class", 0.0)),
        -float(metrics.get("macro_f1_4class", 0.0)),
        -float(metrics.get("itr_bpm_4class", 0.0)),
        float(metrics.get("mean_decision_time_s_4class", float("inf"))),
        float(metrics.get("inference_ms", float("inf"))),
    )


def _parse_evaluation_mode(value: Any) -> str:
    mode = str(value or DEFAULT_EVALUATION_MODE).strip().lower()
    if mode not in {"staged", "full"}:
        raise ValueError(f"unsupported evaluation_mode: {value}")
    return mode


def _safe_metric_mean(values: Sequence[float], *, higher_is_better: bool) -> float:
    finite = [float(item) for item in values if np.isfinite(float(item))]
    if finite:
        return float(np.mean(np.asarray(finite, dtype=float)))
    return 0.0 if higher_is_better else float("inf")


def _metrics_source_name(
    result: dict[str, Any],
    *,
    prefer_cross_session: bool,
) -> str:
    if prefer_cross_session and isinstance(result.get("cross_session_metrics"), dict):
        return "cross_session"
    return "session1_holdout"


def _ranking_metrics_from_result(
    result: dict[str, Any],
    *,
    prefer_cross_session: bool,
) -> dict[str, float]:
    if prefer_cross_session:
        metrics = result.get("cross_session_metrics")
        if isinstance(metrics, dict) and metrics:
            return {str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        return {str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
    return {}


def _evaluate_classifier_only_fixed_window(
    *,
    decoder: Any,
    trial_segments: Sequence[tuple[Any, np.ndarray]],
) -> dict[str, Any]:
    labels = [f"{float(freq):g}" for freq in decoder.freqs]
    rows = _collect_fixed_window_rows(decoder=decoder, trial_segments=trial_segments)
    control_rows = [row for row in rows if bool(row.get("is_control", False))]
    y_true = [str(row["expected_label"]) for row in control_rows]
    y_pred = [str(row["pred_label"]) for row in control_rows]
    decision_times = [float(row["decision_time_s"]) for row in control_rows]
    inference_ms = [float(row["inference_ms"]) for row in control_rows]

    metrics_4class = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        decision_time_samples_s=decision_times,
        itr_class_count=4,
        decision_time_fallback_s=float(decoder.win_sec),
    )
    ranking_metrics = {
        "acc_4class": float(metrics_4class.get("acc", 0.0)),
        "macro_f1_4class": float(metrics_4class.get("macro_f1", 0.0)),
        "itr_bpm_4class": float(metrics_4class.get("itr_bpm", 0.0)),
        "mean_decision_time_s_4class": float(metrics_4class.get("mean_decision_time_s", float("inf"))),
        "inference_ms": (
            float(np.mean(np.asarray(inference_ms, dtype=float))) if inference_ms else float("inf")
        ),
    }
    return {
        "metrics_4class": metrics_4class,
        "classifier_only_metrics": ranking_metrics,
    }


def _collect_fixed_window_rows(
    *,
    decoder: Any,
    trial_segments: Sequence[tuple[Any, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    freqs = [float(freq) for freq in decoder.freqs]
    for trial, segment in trial_segments:
        matrix = np.asarray(segment, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < int(decoder.win_samples):
            continue
        window = np.ascontiguousarray(matrix[-int(decoder.win_samples) :, :], dtype=np.float64)
        infer_t0 = time.perf_counter()
        features = decoder.analyze_window(window)
        infer_t1 = time.perf_counter()
        pred_freq = float(features.get("pred_freq", freqs[0]))
        nearest = min(freqs, key=lambda item: abs(item - pred_freq))
        expected = getattr(trial, "expected_freq", None)
        expected_label = None if expected is None else f"{float(expected):g}"
        pred_label = f"{float(nearest):g}"
        top1_score = float(features.get("top1_score", 0.0) or 0.0)
        top2_score = float(features.get("top2_score", 0.0) or 0.0)
        margin = float(features.get("margin", top1_score - top2_score) or 0.0)
        ratio = float(features.get("ratio", 0.0) or 0.0)
        rows.append(
            {
                "trial_id": None if getattr(trial, "trial_id", None) is None else int(getattr(trial, "trial_id")),
                "label": str(getattr(trial, "label", "")),
                "stage": str(getattr(trial, "stage", "")),
                "is_control": expected is not None,
                "expected_freq": None if expected is None else float(expected),
                "expected_label": expected_label,
                "pred_freq": float(nearest),
                "pred_label": pred_label,
                "correct": bool(expected is not None and expected_label == pred_label),
                "top1_score": top1_score,
                "top2_score": top2_score,
                "margin": margin,
                "ratio": ratio,
                "decision_time_s": float(decoder.win_sec),
                "inference_ms": float((infer_t1 - infer_t0) * 1000.0),
            }
        )
    return rows


def _fit_quick_screen_idle_threshold(
    *,
    train_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    control_rows = [row for row in train_rows if bool(row.get("is_control", False))]
    idle_rows = [row for row in train_rows if not bool(row.get("is_control", False))]
    control_scores = np.asarray([float(row.get("top1_score", 0.0)) for row in control_rows], dtype=float)
    idle_scores = np.asarray([float(row.get("top1_score", 0.0)) for row in idle_rows], dtype=float)
    correct_mask = np.asarray([bool(row.get("correct", False)) for row in control_rows], dtype=bool)

    if control_scores.size == 0 and idle_scores.size == 0:
        return {
            "threshold": 0.0,
            "idle_fp_rate_proxy": 0.0,
            "control_recall_proxy": 0.0,
            "control_trial_count": 0,
            "idle_trial_count": 0,
        }

    combined = np.concatenate([arr for arr in (control_scores, idle_scores) if arr.size > 0], axis=0)
    quantiles = np.linspace(0.0, 1.0, 11)
    candidates = sorted(
        {
            float(max(value, 0.0))
            for value in np.quantile(combined, quantiles).tolist()
            if np.isfinite(float(value))
        }
    )
    candidates = [0.0, *candidates, float(max(np.max(combined), 0.0)) + 1e-6]

    best: Optional[dict[str, Any]] = None
    for threshold in candidates:
        idle_positive = int(np.sum(idle_scores >= threshold)) if idle_scores.size else 0
        idle_fp_rate = (
            float(idle_positive) / float(idle_scores.size)
            if idle_scores.size
            else 0.0
        )
        control_detected = int(np.sum(np.logical_and(correct_mask, control_scores >= threshold))) if control_scores.size else 0
        control_recall = (
            float(control_detected) / float(control_scores.size)
            if control_scores.size
            else 0.0
        )
        candidate = {
            "threshold": float(threshold),
            "idle_fp_rate_proxy": float(idle_fp_rate),
            "control_recall_proxy": float(control_recall),
            "control_trial_count": int(control_scores.size),
            "idle_trial_count": int(idle_scores.size),
        }
        if best is None or (
            float(candidate["idle_fp_rate_proxy"]),
            -float(candidate["control_recall_proxy"]),
            float(candidate["threshold"]),
        ) < (
            float(best["idle_fp_rate_proxy"]),
            -float(best["control_recall_proxy"]),
            float(best["threshold"]),
        ):
            best = candidate

    return dict(best or {})


def _evaluate_quick_screen_idle_proxy(
    *,
    gate_rows: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    control_rows = [row for row in gate_rows if bool(row.get("is_control", False))]
    idle_rows = [row for row in gate_rows if not bool(row.get("is_control", False))]
    control_positive = [
        row for row in control_rows if float(row.get("top1_score", 0.0)) >= float(threshold)
    ]
    idle_positive = [
        row for row in idle_rows if float(row.get("top1_score", 0.0)) >= float(threshold)
    ]
    control_correct_positive = [
        row for row in control_positive if bool(row.get("correct", False))
    ]
    return {
        "threshold": float(threshold),
        "idle_fp_rate_proxy": (
            float(len(idle_positive)) / float(len(idle_rows))
            if idle_rows
            else 0.0
        ),
        "control_recall_proxy": (
            float(len(control_correct_positive)) / float(len(control_rows))
            if control_rows
            else 0.0
        ),
        "control_activation_rate_proxy": (
            float(len(control_positive)) / float(len(control_rows))
            if control_rows
            else 0.0
        ),
        "idle_positive_count": int(len(idle_positive)),
        "idle_trial_count": int(len(idle_rows)),
        "control_correct_positive_count": int(len(control_correct_positive)),
        "control_trial_count": int(len(control_rows)),
    }


def _quick_screen_models(
    *,
    model_names: Sequence[str],
    train_segments: Sequence[tuple[Any, np.ndarray]],
    gate_segments: Sequence[tuple[Any, np.ndarray]],
    sampling_rate: int,
    freqs: Sequence[float],
    step_sec: float,
    win_candidates: Sequence[float],
    seed: int,
    log_fn: Callable[[str], None],
    compute_backend: str = DEFAULT_COMPUTE_BACKEND_NAME,
    gpu_device: int = DEFAULT_GPU_DEVICE_ID,
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
    gpu_warmup: bool = True,
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE,
) -> list[dict[str, Any]]:
    control_gate_segments = [
        (trial, segment) for trial, segment in gate_segments if getattr(trial, "expected_freq", None) is not None
    ]
    if not control_gate_segments:
        raise RuntimeError("quick screen requires non-empty control gate segments")

    max_available_sec = min(
        float(np.asarray(segment, dtype=float).shape[0]) / max(float(sampling_rate), 1.0)
        for _trial, segment in gate_segments
    )
    valid_win_candidates = tuple(
        sorted({float(value) for value in win_candidates if 0.25 < float(value) <= max_available_sec + 1e-8})
    )
    if not valid_win_candidates:
        raise RuntimeError(
            f"quick screen has no valid win candidates; max_available_sec={max_available_sec:.3f}"
        )

    results: list[dict[str, Any]] = []
    for model_index, raw_model_name in enumerate(model_names, start=1):
        model_name = normalize_model_name(raw_model_name)
        log_fn(f"Stage A model start: {model_index}/{len(model_names)} model={model_name}")
        best_entry: Optional[dict[str, Any]] = None
        for config_index, win_sec in enumerate(valid_win_candidates, start=1):
            log_fn(
                f"Stage A config start: model={model_name} {config_index}/{len(valid_win_candidates)} "
                f"win={float(win_sec):g}s"
            )
            try:
                decoder = create_decoder(
                    model_name,
                    sampling_rate=sampling_rate,
                    freqs=freqs,
                    win_sec=float(win_sec),
                    step_sec=float(step_sec),
                    model_params={"Nh": DEFAULT_NH},
                    compute_backend=compute_backend,
                    gpu_device=int(gpu_device),
                    gpu_precision=gpu_precision,
                    gpu_warmup=bool(gpu_warmup),
                    gpu_cache_policy=gpu_cache_policy,
                )
                if getattr(decoder, "requires_fit", False):
                    decoder.fit(train_segments)
                quick_metrics = _evaluate_classifier_only_fixed_window(decoder=decoder, trial_segments=control_gate_segments)
                train_rows = _collect_fixed_window_rows(decoder=decoder, trial_segments=train_segments)
                gate_rows = _collect_fixed_window_rows(decoder=decoder, trial_segments=gate_segments)
                threshold_fit = _fit_quick_screen_idle_threshold(train_rows=train_rows)
                quick_screen_metrics = _evaluate_quick_screen_idle_proxy(
                    gate_rows=gate_rows,
                    threshold=float(threshold_fit.get("threshold", 0.0)),
                )
                classifier_only_metrics = dict(quick_metrics.get("classifier_only_metrics", {}))
                entry = {
                    "stage": "quick_screen",
                    "model_name": str(model_name),
                    "implementation_level": model_implementation_level(model_name),
                    "method_note": model_method_note(model_name),
                    "channel_mode": "all8",
                    "eval_seed": int(seed),
                    "selected_eeg_channels": [],
                    "best_win_sec": float(win_sec),
                    "metrics_4class": dict(quick_metrics.get("metrics_4class", {})),
                    "classifier_only_metrics": classifier_only_metrics,
                    "quick_screen_metrics": dict(quick_screen_metrics),
                    "quick_screen_threshold_fit": dict(threshold_fit),
                    "quick_screen_pass": bool(
                        (
                            float(classifier_only_metrics.get("acc_4class", 0.0)) > 0.0
                            or float(classifier_only_metrics.get("macro_f1_4class", 0.0)) > 0.0
                        )
                        and float(quick_screen_metrics.get("control_recall_proxy", 0.0)) > 0.0
                    ),
                }
                if best_entry is None or _quick_screen_rank_key(
                    quick_screen_metrics | classifier_only_metrics
                ) < _quick_screen_rank_key(
                    dict(best_entry.get("quick_screen_metrics", {})) | dict(best_entry.get("classifier_only_metrics", {}))
                ):
                    best_entry = entry
                log_fn(
                    f"Stage A config done: model={model_name} {config_index}/{len(valid_win_candidates)} "
                    f"idle_fp_proxy={float(quick_screen_metrics.get('idle_fp_rate_proxy', float('inf'))):.4f} "
                    f"recall_proxy={float(quick_screen_metrics.get('control_recall_proxy', 0.0)):.4f} "
                    f"acc_4class={float(classifier_only_metrics.get('acc_4class', 0.0)):.4f} "
                    f"macro_f1={float(classifier_only_metrics.get('macro_f1_4class', 0.0)):.4f}"
                )
            except Exception as exc:
                log_fn(
                    f"Stage A config failed: model={model_name} {config_index}/{len(valid_win_candidates)} "
                    f"win={float(win_sec):g}s error={exc}"
                )
        if best_entry is None:
            results.append(
                {
                    "stage": "quick_screen",
                    "model_name": str(model_name),
                    "implementation_level": model_implementation_level(model_name),
                    "method_note": model_method_note(model_name),
                    "channel_mode": "all8",
                    "eval_seed": int(seed),
                    "error": "quick_screen_failed",
                    "quick_screen_pass": False,
                }
            )
            log_fn(f"Stage A model done: model={model_name} status=failed")
            continue
        results.append(best_entry)
        log_fn(
            f"Stage A model done: model={model_name} "
            f"best_win={float(best_entry.get('best_win_sec', 0.0)):g}s "
            f"idle_fp_proxy={float(dict(best_entry.get('quick_screen_metrics', {})).get('idle_fp_rate_proxy', float('inf'))):.4f} "
            f"recall_proxy={float(dict(best_entry.get('quick_screen_metrics', {})).get('control_recall_proxy', 0.0)):.4f} "
            f"acc_4class={float(best_entry['classifier_only_metrics'].get('acc_4class', 0.0)):.4f}"
        )

    successful = [item for item in results if isinstance(item.get("classifier_only_metrics"), dict)]
    successful.sort(
        key=lambda item: _quick_screen_rank_key(
            dict(item.get("quick_screen_metrics", {})) | dict(item["classifier_only_metrics"])
        )
    )
    for rank, item in enumerate(successful, start=1):
        item["quick_screen_rank"] = int(rank)
    return results


def _aggregate_family_results(
    run_items: Sequence[dict[str, Any]],
    *,
    ranking_policy: str,
    prefer_cross_session: bool,
) -> dict[str, Any]:
    source_name = "cross_session" if prefer_cross_session else "session1_holdout"
    families: dict[str, dict[str, Any]] = {}
    for item in run_items:
        model_name = str(item.get("model_name", "")).strip()
        if not model_name:
            continue
        row = families.setdefault(
            model_name,
            {
                "model_name": model_name,
                "implementation_level": model_implementation_level(model_name),
                "method_note": model_method_note(model_name),
                "runs_total": 0,
                "runs_success": 0,
                "runs_failed": 0,
                "channel_modes": set(),
                "eval_seeds": set(),
                "metrics_values": {},
                "accepted_values": [],
                "rank_values": [],
            },
        )
        row["runs_total"] += 1
        row["channel_modes"].add(str(item.get("channel_mode", "")))
        row["eval_seeds"].add(int(item.get("eval_seed", -1)))
        if "metrics" not in item:
            row["runs_failed"] += 1
            continue
        metrics = _ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session)
        if not metrics:
            row["runs_failed"] += 1
            continue
        row["runs_success"] += 1
        row["accepted_values"].append(1.0 if profile_meets_acceptance(metrics) else 0.0)
        rank_value = item.get("deployment_rank")
        if rank_value is not None:
            row["rank_values"].append(float(rank_value))
        for key, value in metrics.items():
            row["metrics_values"].setdefault(str(key), []).append(float(value))

    ranked_families: list[dict[str, Any]] = []
    for family in families.values():
        metrics_mean: dict[str, float] = {}
        metrics_std: dict[str, float] = {}
        for metric_name, values in family["metrics_values"].items():
            higher_is_better = metric_name in {
                "control_recall",
                "control_recall_at_2s",
                "control_recall_at_3s",
                "switch_detect_rate",
                "switch_detect_rate_at_2.8s",
                "itr_bpm",
                "acc_4class",
                "macro_f1_4class",
                "itr_bpm_4class",
            }
            metrics_mean[metric_name] = _safe_metric_mean(values, higher_is_better=higher_is_better)
            metrics_std[metric_name] = (
                float(np.std(np.asarray(values, dtype=float)))
                if values
                else (0.0 if higher_is_better else float("inf"))
            )
        ranked_families.append(
            {
                "model_name": str(family["model_name"]),
                "implementation_level": str(family["implementation_level"]),
                "method_note": str(family["method_note"]),
                "runs_total": int(family["runs_total"]),
                "runs_success": int(family["runs_success"]),
                "runs_failed": int(family["runs_failed"]),
                "channel_modes": sorted(
                    value for value in family["channel_modes"] if str(value).strip()
                ),
                "eval_seeds": sorted(
                    int(value) for value in family["eval_seeds"] if int(value) >= 0
                ),
                "metrics_mean": metrics_mean,
                "metrics_std": metrics_std,
                "accept_rate": (
                    float(np.mean(np.asarray(family["accepted_values"], dtype=float)))
                    if family["accepted_values"]
                    else 0.0
                ),
                "mean_deployment_rank": (
                    float(np.mean(np.asarray(family["rank_values"], dtype=float)))
                    if family["rank_values"]
                    else float("inf")
                ),
                "ranking_source": source_name,
            }
        )

    ranked_families.sort(
        key=lambda item: (
            benchmark_rank_key(dict(item.get("metrics_mean", {})), ranking_policy=ranking_policy),
            float(item.get("mean_deployment_rank", float("inf"))),
        )
    )
    for rank, item in enumerate(ranked_families, start=1):
        item["rank"] = int(rank)
    return {
        "ranking_source": source_name,
        "families": ranked_families,
    }


def _extract_4class_vectors(result: dict[str, Any]) -> tuple[list[str], list[str], list[float], list[str]]:
    metrics_4 = dict(result.get("classifier_only_metrics_4class") or result.get("metrics_4class") or {})
    y_true = [str(item) for item in metrics_4.get("y_true", [])]
    y_pred = [str(item) for item in metrics_4.get("y_pred", [])]
    labels = [str(item) for item in metrics_4.get("labels", [])]
    raw_times = list(metrics_4.get("decision_time_samples_s", []))
    times = [float(item) for item in raw_times[: len(y_true)] if np.isfinite(float(item))]
    if len(times) < len(y_true):
        fallback = float(metrics_4.get("mean_decision_time_s", float("inf")))
        if not np.isfinite(fallback):
            fallback = 0.0
        times.extend([fallback] * (len(y_true) - len(times)))
    return y_true, y_pred, times[: len(y_true)], labels


def _bootstrap_4class_summary(
    *,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    decision_time_samples_s: Sequence[float],
    labels: Sequence[str],
    seed: int,
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    n_trials = min(len(y_true), len(y_pred), len(decision_time_samples_s))
    if n_trials <= 0:
        return {
            "n_trials": 0,
            "n_bootstrap": 0,
            "acc_mean": 0.0,
            "acc_ci95": [0.0, 0.0],
            "macro_f1_mean": 0.0,
            "macro_f1_ci95": [0.0, 0.0],
            "decision_time_mean_s": float("inf"),
            "decision_time_ci95_s": [float("inf"), float("inf")],
        }
    y_true_arr = np.asarray([str(item) for item in y_true[:n_trials]], dtype=object)
    y_pred_arr = np.asarray([str(item) for item in y_pred[:n_trials]], dtype=object)
    time_arr = np.asarray([float(item) for item in decision_time_samples_s[:n_trials]], dtype=float)
    if n_trials == 1:
        metrics = compute_classification_metrics(
            y_true=y_true_arr.tolist(),
            y_pred=y_pred_arr.tolist(),
            labels=labels,
            decision_time_samples_s=time_arr.tolist(),
            itr_class_count=4,
            decision_time_fallback_s=max(float(time_arr[0]), 1e-3),
        )
        acc = float(metrics.get("acc", 0.0))
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        decision_mean = float(metrics.get("mean_decision_time_s", float("inf")))
        return {
            "n_trials": 1,
            "n_bootstrap": 1,
            "acc_mean": acc,
            "acc_ci95": [acc, acc],
            "macro_f1_mean": macro_f1,
            "macro_f1_ci95": [macro_f1, macro_f1],
            "decision_time_mean_s": decision_mean,
            "decision_time_ci95_s": [decision_mean, decision_mean],
        }

    rng = np.random.default_rng(int(seed))
    acc_values: list[float] = []
    f1_values: list[float] = []
    decision_values: list[float] = []
    draws = max(50, int(n_bootstrap))
    for _ in range(draws):
        sampled_idx = rng.integers(0, n_trials, size=n_trials)
        metrics = compute_classification_metrics(
            y_true=y_true_arr[sampled_idx].tolist(),
            y_pred=y_pred_arr[sampled_idx].tolist(),
            labels=labels,
            decision_time_samples_s=time_arr[sampled_idx].tolist(),
            itr_class_count=4,
            decision_time_fallback_s=max(float(np.median(time_arr[sampled_idx])), 1e-3),
        )
        acc_values.append(float(metrics.get("acc", 0.0)))
        f1_values.append(float(metrics.get("macro_f1", 0.0)))
        decision_values.append(float(metrics.get("mean_decision_time_s", float("inf"))))

    acc_arr = np.asarray(acc_values, dtype=float)
    f1_arr = np.asarray(f1_values, dtype=float)
    decision_arr = np.asarray(decision_values, dtype=float)
    return {
        "n_trials": int(n_trials),
        "n_bootstrap": int(draws),
        "acc_mean": float(np.mean(acc_arr)),
        "acc_ci95": [float(np.percentile(acc_arr, 2.5)), float(np.percentile(acc_arr, 97.5))],
        "macro_f1_mean": float(np.mean(f1_arr)),
        "macro_f1_ci95": [float(np.percentile(f1_arr, 2.5)), float(np.percentile(f1_arr, 97.5))],
        "decision_time_mean_s": float(np.mean(decision_arr)),
        "decision_time_ci95_s": [
            float(np.percentile(decision_arr, 2.5)),
            float(np.percentile(decision_arr, 97.5)),
        ],
    }


def _paired_mcnemar(
    baseline_true: Sequence[str],
    baseline_pred: Sequence[str],
    candidate_true: Sequence[str],
    candidate_pred: Sequence[str],
) -> dict[str, Any]:
    n = min(len(baseline_true), len(baseline_pred), len(candidate_true), len(candidate_pred))
    if n <= 0:
        return {"n_pairs": 0, "b": 0, "c": 0, "chi2": 0.0, "p_value": 1.0}
    base_correct = np.asarray(
        [str(baseline_true[idx]) == str(baseline_pred[idx]) for idx in range(n)],
        dtype=bool,
    )
    cand_correct = np.asarray(
        [str(candidate_true[idx]) == str(candidate_pred[idx]) for idx in range(n)],
        dtype=bool,
    )
    b = int(np.sum(np.logical_and(base_correct, np.logical_not(cand_correct))))
    c = int(np.sum(np.logical_and(np.logical_not(base_correct), cand_correct)))
    discordant = b + c
    if discordant <= 0:
        return {"n_pairs": int(n), "b": b, "c": c, "chi2": 0.0, "p_value": 1.0}
    chi2_stat = float(((abs(b - c) - 1.0) ** 2) / float(discordant))
    p_value = float(chi2.sf(chi2_stat, df=1))
    return {"n_pairs": int(n), "b": b, "c": c, "chi2": chi2_stat, "p_value": p_value}


def _paired_wilcoxon(
    baseline_times: Sequence[float],
    candidate_times: Sequence[float],
) -> dict[str, Any]:
    n = min(len(baseline_times), len(candidate_times))
    if n <= 0:
        return {"n_pairs": 0, "statistic": 0.0, "p_value": 1.0}
    baseline = np.asarray([float(item) for item in baseline_times[:n]], dtype=float)
    candidate = np.asarray([float(item) for item in candidate_times[:n]], dtype=float)
    valid = np.isfinite(baseline) & np.isfinite(candidate)
    baseline = baseline[valid]
    candidate = candidate[valid]
    if baseline.size == 0:
        return {"n_pairs": 0, "statistic": 0.0, "p_value": 1.0}
    if np.allclose(baseline, candidate):
        return {"n_pairs": int(baseline.size), "statistic": 0.0, "p_value": 1.0}
    try:
        stat, p_value = wilcoxon(candidate, baseline, zero_method="wilcox", correction=True)
    except Exception:
        return {"n_pairs": int(baseline.size), "statistic": 0.0, "p_value": 1.0}
    return {"n_pairs": int(baseline.size), "statistic": float(stat), "p_value": float(p_value)}


def _model_variant_name(result: dict[str, Any]) -> str:
    name = str(result.get("model_name", ""))
    if name == "fbcca_fixed_all8":
        return "fbcca_plain_all8"
    if name == "fbcca_cw_all8":
        return "fbcca_channel_weighted"
    if name == "fbcca_sw_all8":
        return "fbcca_subband_weighted"
    if name == "fbcca_cw_sw_all8":
        return "fbcca_channel_subband_weighted"
    if name == "fbcca_cw_sw_trca_shared":
        return "fbcca_channel_subband_trca_shared"
    if name != "fbcca":
        return name
    runtime_channel = str(result.get("runtime_channel_weight_mode") or "").strip().lower()
    runtime_subband = str(result.get("runtime_subband_weight_mode") or "").strip().lower()
    runtime_spatial = str(result.get("runtime_spatial_filter_mode") or "").strip().lower()
    has_channel = runtime_channel == "fbcca_diag"
    has_subband = runtime_subband not in {"", "none", "chen_fixed"}
    if has_channel and has_subband and runtime_spatial == "trca_shared":
        return "fbcca_channel_subband_trca_shared"
    if has_channel and has_subband:
        return "fbcca_channel_subband_weighted"
    if has_channel:
        return "fbcca_channel_weighted"
    if has_subband:
        return "fbcca_subband_weighted"
    return "fbcca_plain_all8"


def _build_ab_comparisons(model_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, dict[str, Any]] = {}
    for row in model_results:
        if "metrics" not in row:
            continue
        by_variant[_model_variant_name(row)] = dict(row)

    comparisons = [
        ("legacy_fbcca_202603", "fbcca_plain_all8"),
        ("fbcca_plain_all8", "fbcca_channel_weighted"),
        ("fbcca_plain_all8", "fbcca_subband_weighted"),
        ("fbcca_plain_all8", "fbcca_channel_subband_weighted"),
        ("fbcca_channel_weighted", "fbcca_channel_subband_weighted"),
        ("fbcca_channel_subband_weighted", "fbcca_channel_subband_trca_shared"),
        ("balanced", "speed"),
    ]
    rows: list[dict[str, Any]] = []
    for baseline_name, candidate_name in comparisons:
        if baseline_name == "balanced" and candidate_name == "speed":
            baseline = (
                by_variant.get("fbcca_channel_subband_trca_shared")
                or by_variant.get("fbcca_channel_subband_weighted")
                or by_variant.get("fbcca_channel_weighted")
                or by_variant.get("fbcca_plain_all8")
            )
            candidate = None
            for item in model_results:
                if str(item.get("model_name", "")) == "fbcca" and str(item.get("gate_policy", "")).strip().lower() == "speed":
                    candidate = dict(item)
                    break
        else:
            baseline = by_variant.get(baseline_name)
            candidate = by_variant.get(candidate_name)
        entry = {
            "comparison": f"{baseline_name} -> {candidate_name}",
            "baseline": baseline_name,
            "candidate": candidate_name,
            "available": bool(baseline and candidate),
        }
        if not baseline or not candidate:
            rows.append(entry)
            continue
        base_metrics = dict(baseline.get("metrics", {}))
        cand_metrics = dict(candidate.get("metrics", {}))
        idle_delta = float(cand_metrics.get("idle_fp_per_min", float("inf"))) - float(
            base_metrics.get("idle_fp_per_min", float("inf"))
        )
        recall_delta = float(cand_metrics.get("control_recall", 0.0)) - float(base_metrics.get("control_recall", 0.0))
        switch_delta = float(cand_metrics.get("switch_latency_s", float("inf"))) - float(
            base_metrics.get("switch_latency_s", float("inf"))
        )
        entry.update(
            {
                "baseline_model_name": str(baseline.get("model_name", "")),
                "candidate_model_name": str(candidate.get("model_name", "")),
                "idle_fp_delta_per_min": idle_delta,
                "control_recall_delta": recall_delta,
                "switch_latency_delta_s": switch_delta,
                "acceptable_gain": bool(
                    idle_delta <= 0.2 and (recall_delta >= 0.08 or switch_delta <= -0.8)
                ),
            }
        )
        rows.append(entry)
    return rows


def _render_training_eval_markdown_legacy(report_payload: dict[str, Any]) -> str:
    chosen_async = dict(report_payload.get("chosen_async_metrics", {}))
    chosen_4 = dict(report_payload.get("chosen_metrics_4class", {}))
    chosen_2 = dict(report_payload.get("chosen_metrics_2class", {}))
    chosen_5_raw = report_payload.get("chosen_metrics_5class")
    chosen_5 = dict(chosen_5_raw) if isinstance(chosen_5_raw, dict) else None
    lines = [
        "# SSVEP Offline Training-Evaluation Report",
        "",
        f"- Generated at: `{report_payload.get('generated_at', '')}`",
        f"- Session1 manifest: `{report_payload.get('dataset_manifest_session1', '')}`",
        f"- Session2 manifest: `{report_payload.get('dataset_manifest_session2', '')}`",
        f"- Chosen model: `{report_payload.get('chosen_model', '')}`",
        f"- Chosen rank: `{report_payload.get('chosen_rank', '')}`",
        f"- Chosen meets acceptance: `{report_payload.get('chosen_meets_acceptance', False)}`",
        f"- Profile saved: `{report_payload.get('profile_saved', False)}`",
        f"- Profile path: `{report_payload.get('chosen_profile_path', '')}`",
        f"- Metric scope: `{report_payload.get('metric_scope', '')}`",
        f"- Decision-time mode: `{report_payload.get('decision_time_mode', '')}`",
        f"- Async decision-time mode: `{report_payload.get('async_decision_time_mode', '')}`",
        f"- Data policy: `{report_payload.get('data_policy', '')}`",
        f"- Ranking policy: `{report_payload.get('ranking_policy', '')}`",
        "",
        "## 6.2.1 SSVEP分类准确率",
        "",
        "| 口径 | Acc_SSVEP | Macro-F1 | Mean Decision Time(s) | ITR(bits/min) |",
        "|---|---:|---:|---:|---:|",
        "| 四分类 (8/10/12/15Hz) | {acc4:.4f} | {f14:.4f} | {t4:.4f} | {itr4:.4f} |".format(
            acc4=float(chosen_4.get("acc", 0.0)),
            f14=float(chosen_4.get("macro_f1", 0.0)),
            t4=float(chosen_4.get("mean_decision_time_s", float("inf"))),
            itr4=float(chosen_4.get("itr_bpm", 0.0)),
        ),
        "| 二分类 (control vs idle) | {acc2:.4f} | {f12:.4f} | {t2:.4f} | {itr2:.4f} |".format(
            acc2=float(chosen_2.get("acc", 0.0)),
            f12=float(chosen_2.get("macro_f1", 0.0)),
            t2=float(chosen_2.get("mean_decision_time_s", float("inf"))),
            itr2=float(chosen_2.get("itr_bpm", 0.0)),
        ),
    ]
    if isinstance(chosen_5, dict):
        lines.append(
            "| 五分类 (idle+4freq) | {acc5:.4f} | {f15:.4f} | {t5:.4f} | {itr5:.4f} |".format(
                acc5=float(chosen_5.get("acc", 0.0)),
                f15=float(chosen_5.get("macro_f1", 0.0)),
                t5=float(chosen_5.get("mean_decision_time_s", float("inf"))),
                itr5=float(chosen_5.get("itr_bpm", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## 6.2.2 异步可用性评测",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            "| idle_fp_per_min | {idle:.4f} |".format(idle=float(chosen_async.get("idle_fp_per_min", float("inf")))),
            "| control_recall | {recall:.4f} |".format(recall=float(chosen_async.get("control_recall", 0.0))),
            "| control_recall_at_3s | {recall3:.4f} |".format(
                recall3=float(chosen_async.get("control_recall_at_3s", 0.0))
            ),
            "| switch_detect_rate_at_2.8s | {switch_rate_deadline:.4f} |".format(
                switch_rate_deadline=float(chosen_async.get("switch_detect_rate_at_2.8s", 0.0))
            ),
            "| switch_latency_s | {switch_lat:.4f} |".format(
                switch_lat=float(chosen_async.get("switch_latency_s", float("inf")))
            ),
            "| release_latency_s | {release_lat:.4f} |".format(
                release_lat=float(chosen_async.get("release_latency_s", float("inf")))
            ),
            "| inference_ms | {infer:.4f} |".format(
                infer=float(chosen_async.get("inference_ms", float("inf")))
            ),
            "",
            "## Ranked Models - End-to-End (Session1 Holdout)",
            "",
            "| Rank | Model | Impl | idle_fp_per_min | control_recall | control_recall_at_3s | switch_detect_rate_at_2.8s | switch_latency_s | release_latency_s | Acc_4class | MacroF1_4class | ITR_4class | inference_ms | Accept |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for item in report_payload.get("model_results", []):
        if "metrics" not in item:
            continue
        metrics = dict(item.get("metrics", {}))
        lines.append(
            "| {rank} | {model} | {impl} | {idle:.4f} | {recall:.4f} | {recall3:.4f} | {switch_rate_deadline:.4f} | {switch_lat:.4f} | {release_lat:.4f} | {acc4:.4f} | {f14:.4f} | {itr4:.4f} | {infer:.4f} | {ok} |".format(
                rank=int(item.get("rank", 0) or 0),
                model=str(item.get("model_name", "")),
                impl=str(item.get("implementation_level", "")),
                idle=float(metrics.get("idle_fp_per_min", float("inf"))),
                recall=float(metrics.get("control_recall", 0.0)),
                recall3=float(metrics.get("control_recall_at_3s", 0.0)),
                switch_rate_deadline=float(metrics.get("switch_detect_rate_at_2.8s", 0.0)),
                switch_lat=float(metrics.get("switch_latency_s", float("inf"))),
                release_lat=float(metrics.get("release_latency_s", float("inf"))),
                acc4=float(metrics.get("acc_4class", 0.0)),
                f14=float(metrics.get("macro_f1_4class", 0.0)),
                itr4=float(metrics.get("itr_bpm_4class", 0.0)),
                infer=float(metrics.get("inference_ms", float("inf"))),
                ok="Y" if bool(item.get("meets_acceptance")) else "N",
            )
        )
    lines.extend(
        [
            "",
            "## Ranked Models - Classifier-only (Unified Fixed-window Lens)",
            "",
            "| Rank | Model | Acc_4class | MacroF1_4class | ITR_4class | DecisionTime_4class(s) | inference_ms |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    classifier_board = (
        dict(report_payload.get("ranking_boards", {})).get("classifier_only", [])
        if isinstance(report_payload.get("ranking_boards", {}), dict)
        else []
    )
    for row in classifier_board:
        metrics = dict(row.get("metrics", {}))
        lines.append(
            "| {rank} | {model} | {acc4:.4f} | {f14:.4f} | {itr4:.4f} | {dt4:.4f} | {infer:.4f} |".format(
                rank=int(row.get("rank", 0) or 0),
                model=str(row.get("model_name", "")),
                acc4=float(metrics.get("acc_4class", 0.0)),
                f14=float(metrics.get("macro_f1_4class", 0.0)),
                itr4=float(metrics.get("itr_bpm_4class", 0.0)),
                dt4=float(metrics.get("mean_decision_time_s_4class", float("inf"))),
                infer=float(metrics.get("inference_ms", float("inf"))),
            )
        )

    ab_rows = list(report_payload.get("ab_comparisons", []))
    if ab_rows:
        lines.extend(
            [
                "",
                "## A/B Improvement Attribution",
                "",
                "| Comparison | Available | idle_fp delta(/min) | recall delta | switch latency delta(s) | Acceptable gain |",
                "|---|:---:|---:|---:|---:|:---:|",
            ]
        )
        for row in ab_rows:
            lines.append(
                "| {name} | {ok} | {idle:.4f} | {recall:.4f} | {switch:.4f} | {gain} |".format(
                    name=str(row.get("comparison", "")),
                    ok="Y" if bool(row.get("available", False)) else "N",
                    idle=float(row.get("idle_fp_delta_per_min", 0.0)),
                    recall=float(row.get("control_recall_delta", 0.0)),
                    switch=float(row.get("switch_latency_delta_s", 0.0)),
                    gain="Y" if bool(row.get("acceptable_gain", False)) else "N",
                )
            )

    stat_rows = [item for item in report_payload.get("model_results", []) if "bootstrap_4class" in item]
    if stat_rows:
        lines.extend(
            [
                "",
                "## Statistical Summary (Trial-level)",
                "",
                f"- Baseline model for paired tests: `{report_payload.get('stats_baseline_model', '')}`",
                "",
                "| Model | Acc mean [95%CI] | MacroF1 mean [95%CI] | Decision mean(s) [95%CI] | McNemar p | Wilcoxon p |",
                "|---|---|---|---|---:|---:|",
            ]
        )
        for item in stat_rows:
            boot = dict(item.get("bootstrap_4class", {}))
            acc_ci = list(boot.get("acc_ci95", [0.0, 0.0]))
            f1_ci = list(boot.get("macro_f1_ci95", [0.0, 0.0]))
            time_ci = list(boot.get("decision_time_ci95_s", [float("inf"), float("inf")]))
            paired = dict(item.get("paired_vs_baseline", {}))
            mcnemar = dict(paired.get("mcnemar_4class_acc", {}))
            wil = dict(paired.get("wilcoxon_decision_time", {}))
            lines.append(
                "| {model} | {acc_m:.4f} [{acc_l:.4f}, {acc_u:.4f}] | {f1_m:.4f} [{f1_l:.4f}, {f1_u:.4f}] | {dt_m:.4f} [{dt_l:.4f}, {dt_u:.4f}] | {mcp:.4f} | {wlp:.4f} |".format(
                    model=str(item.get("model_name", "")),
                    acc_m=float(boot.get("acc_mean", 0.0)),
                    acc_l=float(acc_ci[0]),
                    acc_u=float(acc_ci[1]),
                    f1_m=float(boot.get("macro_f1_mean", 0.0)),
                    f1_l=float(f1_ci[0]),
                    f1_u=float(f1_ci[1]),
                    dt_m=float(boot.get("decision_time_mean_s", float("inf"))),
                    dt_l=float(time_ci[0]),
                    dt_u=float(time_ci[1]),
                    mcp=float(mcnemar.get("p_value", 1.0)),
                    wlp=float(wil.get("p_value", 1.0)),
                )
            )

    if any("cross_session_metrics" in item for item in report_payload.get("model_results", [])):
        lines.extend(
            [
                "",
                "## 会话内 vs 跨会话（Session2）",
                "",
                "| Model | Session2 idle_fp_per_min | Session2 control_recall | Session2 switch_latency_s | Session2 release_latency_s | Session2 Acc_4class | Session2 MacroF1_4class |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in report_payload.get("model_results", []):
            metrics = item.get("cross_session_metrics")
            if not isinstance(metrics, dict):
                continue
            lines.append(
                "| {model} | {idle:.4f} | {recall:.4f} | {switch_lat:.4f} | {release_lat:.4f} | {acc4:.4f} | {f14:.4f} |".format(
                    model=str(item.get("model_name", "")),
                    idle=float(metrics.get("idle_fp_per_min", float("inf"))),
                    recall=float(metrics.get("control_recall", 0.0)),
                    switch_lat=float(metrics.get("switch_latency_s", float("inf"))),
                    release_lat=float(metrics.get("release_latency_s", float("inf"))),
                    acc4=float(metrics.get("acc_4class", 0.0)),
                    f14=float(metrics.get("macro_f1_4class", 0.0)),
                )
            )
    lines.extend(
        [
            "",
            "## 图表文件",
            "",
            f"- confusion_4class: `{dict(report_payload.get('figures', {})).get('confusion_4class', '')}`",
            f"- confusion_2class: `{dict(report_payload.get('figures', {})).get('confusion_2class', '')}`",
            f"- decision_time_hist: `{dict(report_payload.get('figures', {})).get('decision_time_hist', '')}`",
            f"- model_radar_async_vs_cls: `{dict(report_payload.get('figures', {})).get('model_radar_async_vs_cls', '')}`",
        ]
    )
    frontend_weights = list(chosen_frontend.get("channel_weights", []) or [])
    frontend_weight_stats = dict(chosen_frontend.get("channel_weight_stats", {}) or {})
    joint_summary = dict(chosen_frontend.get("joint_weight_training", {}) or {})
    top_channels = list(chosen_frontend.get("top_weight_channels", []) or [])
    if chosen_frontend:
        frontend_lines = [
            "",
            "## FBCCA 前端与权重摘要",
            "",
            f"- channel_weight_mode: `{chosen_frontend.get('channel_weight_mode', None)}`",
            f"- spatial_filter_mode: `{chosen_frontend.get('spatial_filter_mode', None)}`",
            f"- spatial_filter_rank: `{chosen_frontend.get('spatial_filter_rank', None)}`",
        ]
        if frontend_weights:
            frontend_lines.append(f"- channel_weights: `{frontend_weights}`")
        if frontend_weight_stats:
            frontend_lines.append(
                "- channel_weight_stats: "
                f"`count={int(frontend_weight_stats.get('count', 0) or 0)}, "
                f"min={float(frontend_weight_stats.get('min', 0.0)):.4f}, "
                f"max={float(frontend_weight_stats.get('max', 0.0)):.4f}, "
                f"mean={float(frontend_weight_stats.get('mean', 0.0)):.4f}, "
                f"std={float(frontend_weight_stats.get('std', 0.0)):.4f}`"
            )
        if top_channels:
            pairs = ", ".join(
                f"ch{int(item.get('board_channel', 0))}={float(item.get('weight', 0.0)):.4f}"
                for item in top_channels
                if isinstance(item, dict)
            )
            frontend_lines.append(f"- top_weight_channels: `{pairs}`")
        if joint_summary:
            frontend_lines.append(
                "- joint_weight_training: "
                f"`mode={joint_summary.get('mode', '')}, "
                f"iters={int(joint_summary.get('joint_weight_iters', 0) or 0)}, "
                f"iteration_count={int(joint_summary.get('iteration_count', 0) or 0)}, "
                f"improved_iteration_count={int(joint_summary.get('improved_iteration_count', 0) or 0)}, "
                f"selected_spatial_rank={joint_summary.get('selected_spatial_rank', None)}`"
            )
            frontend_lines.append(f"- joint_objective: `{list(joint_summary.get('objective', []))}`")
        figure_marker = next((idx for idx, line in enumerate(lines) if line == "## 鍥捐〃鏂囦欢"), len(lines))
        lines[figure_marker:figure_marker] = frontend_lines
    return "\n".join(lines).strip() + "\n"


def _render_training_eval_markdown(report_payload: dict[str, Any]) -> str:
    chosen_async = dict(report_payload.get("chosen_async_metrics", {}))
    chosen_4 = dict(report_payload.get("chosen_metrics_4class", {}))
    chosen_2 = dict(report_payload.get("chosen_metrics_2class", {}))
    chosen_frontend = dict(report_payload.get("chosen_frontend_summary", {}))
    quality_filter = dict(report_payload.get("quality_filter", {}))
    quality_rows_s1 = [
        row for row in list(report_payload.get("quality_summary_session1", [])) if isinstance(row, dict)
    ]
    lines = [
        "# SSVEP Offline Training-Evaluation Report",
        "",
        f"- Generated at: `{report_payload.get('generated_at', '')}`",
        f"- Session1 manifest: `{report_payload.get('dataset_manifest_session1', '')}`",
        f"- Session2 manifest: `{report_payload.get('dataset_manifest_session2', '')}`",
        f"- Chosen model: `{report_payload.get('chosen_model', '')}`",
        f"- Chosen rank: `{report_payload.get('chosen_rank', '')}`",
        f"- Chosen meets acceptance: `{report_payload.get('chosen_meets_acceptance', False)}`",
        f"- Profile saved: `{report_payload.get('profile_saved', False)}`",
        f"- Profile path: `{report_payload.get('chosen_profile_path', '')}`",
        "",
        "## 数据质量过滤",
        "",
        f"- min_sample_ratio: `{float(quality_filter.get('min_sample_ratio', 0.0)):.3f}`",
        f"- max_retry_count: `{int(quality_filter.get('max_retry_count', 0) or 0)}`",
        f"- Session1 kept/total: `{int(report_payload.get('quality_kept_trials_session1', 0) or 0)}`/`{int(report_payload.get('quality_total_trials_session1', 0) or 0)}`",
        "",
        "| Session | Kept/Total | Drop Ratio | Drop by Shortfall | Drop by Retry |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in quality_rows_s1:
        lines.append(
            "| {sid} | {kept}/{total} | {drop:.3f} | {shortfall} | {retry} |".format(
                sid=str(row.get("session_id", "")),
                kept=int(row.get("kept_trials", 0) or 0),
                total=int(row.get("total_trials", 0) or 0),
                drop=float(row.get("drop_ratio", 0.0) or 0.0),
                shortfall=int(row.get("dropped_shortfall", 0) or 0),
                retry=int(row.get("dropped_retry", 0) or 0),
            )
        )
    lines.extend(
        [
            "",
            "## 6.2.1 SSVEP分类准确率",
            "",
            "| 口径 | Acc_SSVEP | Macro-F1 | Mean Decision Time(s) | ITR(bits/min) |",
            "|---|---:|---:|---:|---:|",
            "| 四分类(8/10/12/15Hz) | {acc4:.4f} | {f14:.4f} | {t4:.4f} | {itr4:.4f} |".format(
                acc4=float(chosen_4.get("acc", 0.0)),
                f14=float(chosen_4.get("macro_f1", 0.0)),
                t4=float(chosen_4.get("mean_decision_time_s", float("inf"))),
                itr4=float(chosen_4.get("itr_bpm", 0.0)),
            ),
            "| 二分类(control vs idle) | {acc2:.4f} | {f12:.4f} | {t2:.4f} | {itr2:.4f} |".format(
                acc2=float(chosen_2.get("acc", 0.0)),
                f12=float(chosen_2.get("macro_f1", 0.0)),
                t2=float(chosen_2.get("mean_decision_time_s", float("inf"))),
                itr2=float(chosen_2.get("itr_bpm", 0.0)),
            ),
            "",
            "## 6.2.2 异步可用性评测",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            "| idle_fp_per_min | {idle:.4f} |".format(idle=float(chosen_async.get("idle_fp_per_min", float("inf")))),
            "| control_recall | {recall:.4f} |".format(recall=float(chosen_async.get("control_recall", 0.0))),
            "| switch_latency_s | {switch_lat:.4f} |".format(
                switch_lat=float(chosen_async.get("switch_latency_s", float("inf")))
            ),
            "| release_latency_s | {release_lat:.4f} |".format(
                release_lat=float(chosen_async.get("release_latency_s", float("inf")))
            ),
            "| inference_ms | {infer:.4f} |".format(
                infer=float(chosen_async.get("inference_ms", float("inf")))
            ),
            "",
            "## 图表文件",
            "",
            f"- confusion_4class: `{dict(report_payload.get('figures', {})).get('confusion_4class', '')}`",
            f"- confusion_2class: `{dict(report_payload.get('figures', {})).get('confusion_2class', '')}`",
            f"- decision_time_hist: `{dict(report_payload.get('figures', {})).get('decision_time_hist', '')}`",
            f"- model_radar_async_vs_cls: `{dict(report_payload.get('figures', {})).get('model_radar_async_vs_cls', '')}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _render_training_eval_markdown(report_payload: dict[str, Any]) -> str:
    return _render_training_eval_markdown_clean(report_payload)


def _render_training_eval_markdown_clean(report_payload: dict[str, Any]) -> str:
    """Final clean Chinese report renderer used by staged train/eval."""

    def _f(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(out):
            return float(default)
        return float(out)

    def _bool_zh(value: Any) -> str:
        return "是" if bool(value) else "否"

    chosen_async = dict(report_payload.get("chosen_async_metrics", {}) or {})
    chosen_4 = dict(report_payload.get("chosen_metrics_4class", {}) or {})
    chosen_2 = dict(report_payload.get("chosen_metrics_2class", {}) or {})
    quality_filter = dict(report_payload.get("quality_filter", {}) or {})
    weight_notes = dict(report_payload.get("weight_definition_notes", {}) or {})
    figures = dict(report_payload.get("figures", {}) or {})
    fbcca_weight_table = [
        dict(row) for row in list(report_payload.get("fbcca_weight_table", []) or []) if isinstance(row, dict)
    ]
    model_compare_table = [
        dict(row) for row in list(report_payload.get("model_compare_table", []) or []) if isinstance(row, dict)
    ]
    ab_rows = [
        dict(row) for row in list(report_payload.get("ab_comparisons", []) or []) if isinstance(row, dict)
    ]

    lines = [
        "# SSVEP训练评测报告",
        "",
        "## 总览",
        "",
        f"- 生成时间：`{report_payload.get('generated_at', '')}`",
        f"- 任务模式：`{report_payload.get('task', '')}`",
        f"- 推荐模型族：`{report_payload.get('recommended_model', '')}`",
        f"- 部署候选模型：`{report_payload.get('chosen_model', '')}`",
        f"- 是否达到验收阈值：`{_bool_zh(report_payload.get('chosen_meets_acceptance', False))}`",
        f"- 默认实时profile是否覆盖：`{_bool_zh(report_payload.get('default_profile_saved', False))}`",
        f"- 最佳候选profile：`{report_payload.get('best_candidate_profile_path', '')}`",
        f"- 最佳加权FBCCA profile：`{report_payload.get('best_fbcca_weighted_profile_path', '')}`",
        "",
        "## 权重定义",
        "",
        f"- 通道权重：{weight_notes.get('channel_weights', '8个通道缩放权重。')}",
        f"- 子带权重：{weight_notes.get('subband_weights', '5个全局子带融合权重。')}",
        f"- 当前主线：{weight_notes.get('separable_weighting', '可分离权重：8个通道权重 + 5个全局子带权重，不训练完整8x5矩阵。')}",
        f"- 空间滤波：{weight_notes.get('spatial_filter_state', 'TRCA/shared空间前端仅作为独立对照项。')}",
        "",
        "## 数据质量与协议一致性",
        "",
        f"- Session1：`{report_payload.get('dataset_manifest_session1', '')}`",
        f"- Session2：`{report_payload.get('dataset_manifest_session2', '')}`",
        f"- 保留trial：`{int(report_payload.get('quality_kept_trials_session1', 0) or 0)}` / `{int(report_payload.get('quality_total_trials_session1', 0) or 0)}`",
        f"- 最小样本比例：`{_f(quality_filter.get('min_sample_ratio', 0.0)):.3f}`",
        f"- 数据策略：`{report_payload.get('data_policy', '')}`",
        f"- 协议签名：`{report_payload.get('protocol_signature_expected', '')}`",
        "",
        "## 6.2.1 SSVEP分类准确率",
        "",
        "| 口径 | Acc_SSVEP | Macro-F1 | 平均决策时间(s) | ITR(bits/min) |",
        "|---|---:|---:|---:|---:|",
        "| 四分类 8/10/12/15Hz | {acc:.4f} | {f1:.4f} | {dt:.4f} | {itr:.4f} |".format(
            acc=_f(chosen_4.get("acc", 0.0)),
            f1=_f(chosen_4.get("macro_f1", 0.0)),
            dt=_f(chosen_4.get("mean_decision_time_s", 0.0)),
            itr=_f(chosen_4.get("itr_bpm", 0.0)),
        ),
        "| 二分类 control vs idle | {acc:.4f} | {f1:.4f} | {dt:.4f} | {itr:.4f} |".format(
            acc=_f(chosen_2.get("acc", 0.0)),
            f1=_f(chosen_2.get("macro_f1", 0.0)),
            dt=_f(chosen_2.get("mean_decision_time_s", 0.0)),
            itr=_f(chosen_2.get("itr_bpm", 0.0)),
        ),
        "",
        "## 6.2.2 异步可用性",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| idle_fp_per_min | {_f(chosen_async.get('idle_fp_per_min', 0.0)):.4f} |",
        f"| control_recall | {_f(chosen_async.get('control_recall', 0.0)):.4f} |",
        f"| switch_latency_s | {_f(chosen_async.get('switch_latency_s', 0.0)):.4f} |",
        f"| release_latency_s | {_f(chosen_async.get('release_latency_s', 0.0)):.4f} |",
        f"| inference_ms | {_f(chosen_async.get('inference_ms', 0.0)):.4f} |",
        "",
        "## FBCCA权重增益表",
        "",
        "| Rank | Model | 权重策略 | Acc4 | MacroF1_4 | ITR4 | idleFP/min | Recall | 通道权重数 | 子带权重 | Accepted |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    if fbcca_weight_table:
        for row in fbcca_weight_table:
            metrics = dict(row.get("metrics", {}) or {})
            channel_weights = list(row.get("channel_weights", []) or [])
            subband_weights = [round(_f(value), 6) for value in list(row.get("subband_weights", []) or [])]
            lines.append(
                "| {rank} | {model} | {strategy} | {acc:.4f} | {f1:.4f} | {itr:.4f} | {idle:.4f} | {recall:.4f} | {cw_count} | `{sw}` | {ok} |".format(
                    rank=int(row.get("rank", 0) or 0),
                    model=str(row.get("model_name", "")),
                    strategy=str(row.get("weight_strategy", "")),
                    acc=_f(row.get("acc_4class", 0.0)),
                    f1=_f(row.get("macro_f1_4class", 0.0)),
                    itr=_f(row.get("itr_bpm_4class", 0.0)),
                    idle=_f(metrics.get("idle_fp_per_min", 0.0)),
                    recall=_f(metrics.get("control_recall", 0.0)),
                    cw_count=len(channel_weights),
                    sw=subband_weights,
                    ok="Y" if bool(row.get("meets_acceptance", False)) else "N",
                )
            )
    else:
        lines.append("|  | 无FBCCA权重候选 |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | `[]` | N |")

    lines.extend(
        [
            "",
            "## 加权提升归因",
            "",
            "| 对比 | 可用 | idleFP变化 | Recall变化 | Switch时延变化(s) | 是否可接受提升 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    if ab_rows:
        for row in ab_rows:
            lines.append(
                "| {name} | {available} | {idle:.4f} | {recall:.4f} | {switch:.4f} | {gain} |".format(
                    name=str(row.get("comparison", "")),
                    available="Y" if bool(row.get("available", False)) else "N",
                    idle=_f(row.get("idle_fp_delta_per_min", 0.0)),
                    recall=_f(row.get("control_recall_delta", 0.0)),
                    switch=_f(row.get("switch_latency_delta_s", 0.0)),
                    gain="Y" if bool(row.get("acceptable_gain", False)) else "N",
                )
            )
    else:
        lines.append("| 无归因对比 | N | 0.0000 | 0.0000 | 0.0000 | N |")

    lines.extend(
        [
            "",
            "## 全模型识别效果对比",
            "",
            "| Rank | Model | Impl | Acc4 | MacroF1_4 | ITR4 | idleFP/min | Recall | Switch(s) | Release(s) | Inference(ms) | Accepted |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if model_compare_table:
        for row in model_compare_table:
            lines.append(
                "| {rank} | {model} | {impl} | {acc:.4f} | {f1:.4f} | {itr:.4f} | {idle:.4f} | {recall:.4f} | {switch:.4f} | {release:.4f} | {infer:.4f} | {ok} |".format(
                    rank=int(row.get("rank", 0) or 0),
                    model=str(row.get("model_name", "")),
                    impl=str(row.get("implementation_level", "")),
                    acc=_f(row.get("acc_4class", 0.0)),
                    f1=_f(row.get("macro_f1_4class", 0.0)),
                    itr=_f(row.get("itr_bpm_4class", 0.0)),
                    idle=_f(row.get("idle_fp_per_min", 0.0)),
                    recall=_f(row.get("control_recall", 0.0)),
                    switch=_f(row.get("switch_latency_s", 0.0)),
                    release=_f(row.get("release_latency_s", 0.0)),
                    infer=_f(row.get("inference_ms", 0.0)),
                    ok="Y" if bool(row.get("meets_acceptance", False)) else "N",
                )
            )
    else:
        lines.append("|  | 无模型对比结果 |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N |")

    lines.extend(
        [
            "",
            "## 图表文件",
            "",
            f"- 四分类混淆矩阵：`{figures.get('confusion_4class', '')}`",
            f"- 二分类混淆矩阵：`{figures.get('confusion_2class', '')}`",
            f"- 决策时间直方图：`{figures.get('decision_time_hist', '')}`",
            f"- 模型雷达图：`{figures.get('model_radar_async_vs_cls', '')}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _render_training_eval_markdown(report_payload: dict[str, Any]) -> str:
    """Render the current SSVEP train/eval report in clear Chinese.

    This final definition is intentionally placed last so staged evaluation imports
    the clean report renderer instead of older compatibility renderers above.
    """

    def _f(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(out):
            return float(default)
        return float(out)

    def _b(value: Any) -> str:
        return "是" if bool(value) else "否"

    chosen_async = dict(report_payload.get("chosen_async_metrics", {}) or {})
    chosen_4 = dict(report_payload.get("chosen_metrics_4class", {}) or {})
    chosen_2 = dict(report_payload.get("chosen_metrics_2class", {}) or {})
    quality_filter = dict(report_payload.get("quality_filter", {}) or {})
    weight_notes = dict(report_payload.get("weight_definition_notes", {}) or {})
    figures = dict(report_payload.get("figures", {}) or {})
    fbcca_weight_table = [
        dict(row) for row in list(report_payload.get("fbcca_weight_table", []) or []) if isinstance(row, dict)
    ]
    model_compare_table = [
        dict(row) for row in list(report_payload.get("model_compare_table", []) or []) if isinstance(row, dict)
    ]
    ab_rows = [
        dict(row) for row in list(report_payload.get("ab_comparisons", []) or []) if isinstance(row, dict)
    ]

    lines: list[str] = [
        "# SSVEP 训练评测报告",
        "",
        "## 总览",
        "",
        f"- 生成时间：`{report_payload.get('generated_at', '')}`",
        f"- 任务模式：`{report_payload.get('task', '')}`",
        f"- 推荐模型族：`{report_payload.get('recommended_model', '')}`",
        f"- 部署候选模型：`{report_payload.get('chosen_model', '')}`",
        f"- 是否达到验收阈值：`{_b(report_payload.get('chosen_meets_acceptance', False))}`",
        f"- 默认实时 profile 是否覆盖：`{_b(report_payload.get('default_profile_saved', False))}`",
        f"- 最佳候选 profile：`{report_payload.get('best_candidate_profile_path', '')}`",
        f"- 最佳加权 FBCCA profile：`{report_payload.get('best_fbcca_weighted_profile_path', '')}`",
        "",
        "## 权重定义",
        "",
        f"- 通道权重：{weight_notes.get('channel_weights', '8 个全通道缩放权重。')}",
        f"- 子带权重：{weight_notes.get('subband_weights', '5 个全局 filter-bank 子带融合权重。')}",
        f"- 当前主线：{weight_notes.get('separable_weighting', '可分离权重：channel_weights[8] + subband_weights[5]，不训练完整 8x5 矩阵。')}",
        f"- 空间滤波：{weight_notes.get('spatial_filter_state', 'TRCA/shared 空间前端是独立对照项。')}",
        "",
        "## 数据质量与协议一致性",
        "",
        f"- Session1：`{report_payload.get('dataset_manifest_session1', '')}`",
        f"- Session2：`{report_payload.get('dataset_manifest_session2', '')}`",
        f"- 保留 trial：`{int(report_payload.get('quality_kept_trials_session1', 0) or 0)}` / `{int(report_payload.get('quality_total_trials_session1', 0) or 0)}`",
        f"- 最小样本比例：`{_f(quality_filter.get('min_sample_ratio', 0.0)):.3f}`",
        f"- 数据策略：`{report_payload.get('data_policy', '')}`",
        f"- 协议签名：`{report_payload.get('protocol_signature_expected', '')}`",
        "",
        "## 6.2.1 SSVEP 分类准确率",
        "",
        "| 口径 | Acc_SSVEP | Macro-F1 | 平均决策时间(s) | ITR(bits/min) |",
        "|---|---:|---:|---:|---:|",
        "| 四分类 8/10/12/15Hz | {acc:.4f} | {f1:.4f} | {dt:.4f} | {itr:.4f} |".format(
            acc=_f(chosen_4.get("acc", 0.0)),
            f1=_f(chosen_4.get("macro_f1", 0.0)),
            dt=_f(chosen_4.get("mean_decision_time_s", 0.0)),
            itr=_f(chosen_4.get("itr_bpm", 0.0)),
        ),
        "| 二分类 control vs idle | {acc:.4f} | {f1:.4f} | {dt:.4f} | {itr:.4f} |".format(
            acc=_f(chosen_2.get("acc", 0.0)),
            f1=_f(chosen_2.get("macro_f1", 0.0)),
            dt=_f(chosen_2.get("mean_decision_time_s", 0.0)),
            itr=_f(chosen_2.get("itr_bpm", 0.0)),
        ),
        "",
        "## 6.2.2 异步可用性",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| idle_fp_per_min | {_f(chosen_async.get('idle_fp_per_min', 0.0)):.4f} |",
        f"| control_recall | {_f(chosen_async.get('control_recall', 0.0)):.4f} |",
        f"| switch_latency_s | {_f(chosen_async.get('switch_latency_s', 0.0)):.4f} |",
        f"| release_latency_s | {_f(chosen_async.get('release_latency_s', 0.0)):.4f} |",
        f"| inference_ms | {_f(chosen_async.get('inference_ms', 0.0)):.4f} |",
        "",
        "## FBCCA 权重增益表",
        "",
        "| Rank | Model | 权重策略 | Acc4 | MacroF1_4 | ITR4 | idleFP/min | Recall | 通道权重数 | 子带权重 | Accepted |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    if fbcca_weight_table:
        for row in fbcca_weight_table:
            metrics = dict(row.get("metrics", {}) or {})
            channel_weights = list(row.get("channel_weights", []) or [])
            subband_weights = [round(_f(value), 6) for value in list(row.get("subband_weights", []) or [])]
            lines.append(
                "| {rank} | {model} | {strategy} | {acc:.4f} | {f1:.4f} | {itr:.4f} | {idle:.4f} | {recall:.4f} | {cw_count} | `{sw}` | {ok} |".format(
                    rank=int(row.get("rank", 0) or 0),
                    model=str(row.get("model_name", "")),
                    strategy=str(row.get("weight_strategy", "")),
                    acc=_f(row.get("acc_4class", 0.0)),
                    f1=_f(row.get("macro_f1_4class", 0.0)),
                    itr=_f(row.get("itr_bpm_4class", 0.0)),
                    idle=_f(metrics.get("idle_fp_per_min", 0.0)),
                    recall=_f(metrics.get("control_recall", 0.0)),
                    cw_count=len(channel_weights),
                    sw=subband_weights,
                    ok="Y" if bool(row.get("meets_acceptance", False)) else "N",
                )
            )
    else:
        lines.append("|  | 无 FBCCA 权重候选 |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | `[]` | N |")

    lines.extend(
        [
            "",
            "## 加权提升归因",
            "",
            "| 对比 | 可用 | idleFP变化 | Recall变化 | Switch时延变化(s) | 是否可接受提升 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    if ab_rows:
        for row in ab_rows:
            lines.append(
                "| {name} | {available} | {idle:.4f} | {recall:.4f} | {switch:.4f} | {gain} |".format(
                    name=str(row.get("comparison", "")),
                    available="Y" if bool(row.get("available", False)) else "N",
                    idle=_f(row.get("idle_fp_delta_per_min", 0.0)),
                    recall=_f(row.get("control_recall_delta", 0.0)),
                    switch=_f(row.get("switch_latency_delta_s", 0.0)),
                    gain="Y" if bool(row.get("acceptable_gain", False)) else "N",
                )
            )
    else:
        lines.append("| 无归因对比 | N | 0.0000 | 0.0000 | 0.0000 | N |")

    lines.extend(
        [
            "",
            "## 全模型识别效果对比",
            "",
            "| Rank | Model | Impl | Acc4 | MacroF1_4 | ITR4 | idleFP/min | Recall | Switch(s) | Release(s) | Inference(ms) | Accepted |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if model_compare_table:
        for row in model_compare_table:
            lines.append(
                "| {rank} | {model} | {impl} | {acc:.4f} | {f1:.4f} | {itr:.4f} | {idle:.4f} | {recall:.4f} | {switch:.4f} | {release:.4f} | {infer:.4f} | {ok} |".format(
                    rank=int(row.get("rank", 0) or 0),
                    model=str(row.get("model_name", "")),
                    impl=str(row.get("implementation_level", "")),
                    acc=_f(row.get("acc_4class", 0.0)),
                    f1=_f(row.get("macro_f1_4class", 0.0)),
                    itr=_f(row.get("itr_bpm_4class", 0.0)),
                    idle=_f(row.get("idle_fp_per_min", 0.0)),
                    recall=_f(row.get("control_recall", 0.0)),
                    switch=_f(row.get("switch_latency_s", 0.0)),
                    release=_f(row.get("release_latency_s", 0.0)),
                    infer=_f(row.get("inference_ms", 0.0)),
                    ok="Y" if bool(row.get("meets_acceptance", False)) else "N",
                )
            )
    else:
        lines.append("|  | 无模型对比结果 |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N |")

    lines.extend(
        [
            "",
            "## 图表文件",
            "",
            f"- 四分类混淆矩阵：`{figures.get('confusion_4class', '')}`",
            f"- 二分类混淆矩阵：`{figures.get('confusion_2class', '')}`",
            f"- 决策时间直方图：`{figures.get('decision_time_hist', '')}`",
            f"- 模型雷达图：`{figures.get('model_radar_async_vs_cls', '')}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _render_training_eval_markdown(report_payload: dict[str, Any]) -> str:
    """Render the current Chinese report schema.

    This final definition intentionally overrides older report renderers kept above
    for backward import compatibility.
    """

    def _f(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        return out

    chosen_async = dict(report_payload.get("chosen_async_metrics", {}) or {})
    chosen_4 = dict(report_payload.get("chosen_metrics_4class", {}) or {})
    chosen_2 = dict(report_payload.get("chosen_metrics_2class", {}) or {})
    quality_filter = dict(report_payload.get("quality_filter", {}) or {})
    figures = dict(report_payload.get("figures", {}) or {})
    fbcca_weight_table = [
        row for row in list(report_payload.get("fbcca_weight_table", []) or []) if isinstance(row, dict)
    ]
    model_compare_table = [
        row for row in list(report_payload.get("model_compare_table", []) or []) if isinstance(row, dict)
    ]
    weight_notes = dict(report_payload.get("weight_definition_notes", {}) or {})

    lines: list[str] = [
        "# SSVEP训练评测报告",
        "",
        f"- 生成时间：`{report_payload.get('generated_at', '')}`",
        f"- 任务模式：`{report_payload.get('task', '')}`",
        f"- Session1：`{report_payload.get('dataset_manifest_session1', '')}`",
        f"- Session2：`{report_payload.get('dataset_manifest_session2', '')}`",
        f"- 推荐模型族：`{report_payload.get('recommended_model', '')}`",
        f"- 部署候选模型：`{report_payload.get('chosen_model', '')}`",
        f"- 是否达到验收阈值：`{report_payload.get('chosen_meets_acceptance', False)}`",
        f"- 默认Profile是否覆盖：`{report_payload.get('default_profile_saved', report_payload.get('profile_saved', False))}`",
        f"- 最佳候选Profile：`{report_payload.get('best_candidate_profile_path', '')}`",
        f"- 最佳FBCCA权重Profile：`{report_payload.get('best_fbcca_weighted_profile_path', '')}`",
        "",
        "## 数据质量与协议一致性",
        "",
        f"- 最小样本比例：`{_f(quality_filter.get('min_sample_ratio', 0.0)):.3f}`",
        f"- 最大重采次数：`{int(quality_filter.get('max_retry_count', 0) or 0)}`",
        f"- Session1保留trial：`{int(report_payload.get('quality_kept_trials_session1', 0) or 0)}`/`{int(report_payload.get('quality_total_trials_session1', 0) or 0)}`",
        f"- 数据策略：`{report_payload.get('data_policy', '')}`",
        f"- 期望协议签名：`{report_payload.get('protocol_signature_expected', '')}`",
        "",
        "## 6.2.1 SSVEP分类准确率",
        "",
        "| 口径 | Acc_SSVEP | Macro-F1 | 平均决策时间(s) | ITR(bits/min) |",
        "|---|---:|---:|---:|---:|",
        "| 四分类(8/10/12/15Hz) | {acc:.4f} | {f1:.4f} | {dt:.4f} | {itr:.4f} |".format(
            acc=_f(chosen_4.get("acc", 0.0)),
            f1=_f(chosen_4.get("macro_f1", 0.0)),
            dt=_f(chosen_4.get("mean_decision_time_s", 0.0)),
            itr=_f(chosen_4.get("itr_bpm", 0.0)),
        ),
        "| 二分类(control vs idle) | {acc:.4f} | {f1:.4f} | {dt:.4f} | {itr:.4f} |".format(
            acc=_f(chosen_2.get("acc", 0.0)),
            f1=_f(chosen_2.get("macro_f1", 0.0)),
            dt=_f(chosen_2.get("mean_decision_time_s", 0.0)),
            itr=_f(chosen_2.get("itr_bpm", 0.0)),
        ),
        "",
        "## 6.2.2 异步可用性",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        "| idle_fp_per_min | {value:.4f} |".format(value=_f(chosen_async.get("idle_fp_per_min", float("inf")))),
        "| control_recall | {value:.4f} |".format(value=_f(chosen_async.get("control_recall", 0.0))),
        "| switch_latency_s | {value:.4f} |".format(value=_f(chosen_async.get("switch_latency_s", float("inf")))),
        "| release_latency_s | {value:.4f} |".format(value=_f(chosen_async.get("release_latency_s", float("inf")))),
        "| inference_ms | {value:.4f} |".format(value=_f(chosen_async.get("inference_ms", float("inf")))),
        "",
        "## FBCCA通道权重结果",
        "",
        f"- 定义：{weight_notes.get('channel_weights', '')}",
        "",
        "| Rank | Model | Mode | Seed | Channel Weight Mode | Count | Min | Max | Mean | Accepted |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in fbcca_weight_table:
        stats = dict(row.get("channel_weight_stats", {}) or {})
        weights = list(row.get("channel_weights", []) or [])
        lines.append(
            "| {rank} | {model} | {mode} | {seed} | {cw_mode} | {count} | {minv:.4f} | {maxv:.4f} | {meanv:.4f} | {ok} |".format(
                rank=int(row.get("rank", 0) or 0),
                model=str(row.get("model_name", "")),
                mode=str(row.get("channel_mode", "")),
                seed=int(row.get("eval_seed", 0) or 0),
                cw_mode=str(row.get("channel_weight_mode", "")),
                count=int(stats.get("count", len(weights)) or 0),
                minv=_f(stats.get("min", 0.0)),
                maxv=_f(stats.get("max", 0.0)),
                meanv=_f(stats.get("mean", 0.0)),
                ok="Y" if bool(row.get("meets_acceptance", False)) else "N",
            )
        )
    if not fbcca_weight_table:
        lines.append("|  | 无FBCCA权重候选 |  |  |  | 0 | 0.0000 | 0.0000 | 0.0000 | N |")

    lines.extend(
        [
            "",
            "## FBCCA子带权重结果",
            "",
            f"- 定义：{weight_notes.get('subband_weights', '')}",
            "",
            "| Rank | Model | Subband Mode | Subband Weights |",
            "|---:|---|---|---|",
        ]
    )
    for row in fbcca_weight_table:
        subband_weights = list(row.get("subband_weights", []) or [])
        lines.append(
            "| {rank} | {model} | {mode} | `{weights}` |".format(
                rank=int(row.get("rank", 0) or 0),
                model=str(row.get("model_name", "")),
                mode=str(row.get("subband_weight_mode", "")),
                weights=[round(float(value), 6) for value in subband_weights],
            )
        )
    if not fbcca_weight_table:
        lines.append("|  | 无FBCCA权重候选 |  | `[]` |")

    lines.extend(
        [
            "",
            "## 实时Profile可用性",
            "",
            f"- `profile_best_candidate.json`：`{report_payload.get('best_candidate_profile_path', '')}`",
            f"- `profile_best_fbcca_weighted.json`：`{report_payload.get('best_fbcca_weighted_profile_path', '')}`",
            f"- `default_profile.json`是否被覆盖：`{report_payload.get('default_profile_saved', False)}`",
            "- 规则：报告目录内的最佳候选和最佳FBCCA权重Profile总会保留；只有达标时才覆盖默认实时Profile。",
            "",
            "## 全模型识别准确度对比",
            "",
            "| Rank | Model | Impl | Acc4 | MacroF1_4 | ITR4 | idleFP/min | Recall | Switch(s) | Release(s) | Accepted |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in model_compare_table:
        lines.append(
            "| {rank} | {model} | {impl} | {acc:.4f} | {f1:.4f} | {itr:.4f} | {idle:.4f} | {recall:.4f} | {switch:.4f} | {release:.4f} | {ok} |".format(
                rank=int(row.get("rank", 0) or 0),
                model=str(row.get("model_name", "")),
                impl=str(row.get("implementation_level", "")),
                acc=_f(row.get("acc_4class", 0.0)),
                f1=_f(row.get("macro_f1_4class", 0.0)),
                itr=_f(row.get("itr_bpm_4class", 0.0)),
                idle=_f(row.get("idle_fp_per_min", float("inf"))),
                recall=_f(row.get("control_recall", 0.0)),
                switch=_f(row.get("switch_latency_s", float("inf"))),
                release=_f(row.get("release_latency_s", float("inf"))),
                ok="Y" if bool(row.get("meets_acceptance", False)) else "N",
            )
        )
    if not model_compare_table:
        lines.append("|  | 无模型对比结果 |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N |")

    lines.extend(
        [
            "",
            "## 图表文件",
            "",
            f"- 四分类混淆矩阵：`{figures.get('confusion_4class', '')}`",
            f"- 二分类混淆矩阵：`{figures.get('confusion_2class', '')}`",
            f"- 决策时间直方图：`{figures.get('decision_time_hist', '')}`",
            f"- 模型雷达图：`{figures.get('model_radar_async_vs_cls', '')}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


_render_training_eval_markdown_v2 = _render_training_eval_markdown


def _legacy_run_offline_train_eval_unused(
    config: OfflineTrainEvalConfig,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    sink_log = log_fn if log_fn is not None else (lambda _msg: None)
    run_log_lines: list[str] = []
    run_log_path: Optional[Path] = None

    def log(message: str) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{stamp}] {message}"
        run_log_lines.append(line)
        sink_log(message)
        if run_log_path is not None:
            with run_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    metric_scope = parse_metric_scope(config.metric_scope)
    decision_time_mode = parse_decision_time_mode(config.decision_time_mode)
    async_decision_time_mode = parse_decision_time_mode(config.async_decision_time_mode)
    data_policy = parse_data_policy(config.data_policy)
    ranking_policy = parse_ranking_policy(config.ranking_policy)
    export_figures = bool(config.export_figures)
    (
        dataset_session1,
        selected_session1_manifest_paths,
        session1_stages,
        session1_quality_rows,
        protocol_consistency_summary,
    ) = _load_session1_dataset(config)
    dataset_session2_raw = (
        load_collection_dataset(config.dataset_manifest_session2)
        if config.dataset_manifest_session2 is not None
        else None
    )
    session2_quality_row: Optional[dict[str, Any]] = None
    dataset_session2 = dataset_session2_raw
    if dataset_session2_raw is not None:
        dataset_session2, session2_quality_row = _apply_trial_quality_filter(
            dataset_session2_raw,
            min_sample_ratio=float(config.quality_min_sample_ratio),
            max_retry_count=int(config.quality_max_retry_count),
        )
        if not dataset_session2.trial_segments:
            raise RuntimeError(
                "session2 has zero trials after quality filtering; "
                f"manifest={config.dataset_manifest_session2} "
                f"min_sample_ratio={config.quality_min_sample_ratio} "
                f"max_retry_count={config.quality_max_retry_count}"
            )
    if session1_stages and session1_stages != {"collection"}:
        raise RuntimeError(
            "train-eval only accepts independent collection datasets (stage=collection); "
            f"got stages={sorted(session1_stages)}"
        )
    if dataset_session2 is not None and int(dataset_session2.sampling_rate) != int(dataset_session1.sampling_rate):
        raise RuntimeError("session2 sampling rate differs from session1")
    if dataset_session2 is not None and tuple(dataset_session2.freqs) != tuple(dataset_session1.freqs):
        raise RuntimeError("session2 freqs differ from session1")
    if dataset_session2 is not None and tuple(dataset_session2.board_eeg_channels) != tuple(dataset_session1.board_eeg_channels):
        raise RuntimeError("session2 EEG channel mapping differs from session1")
    if dataset_session2 is not None and bool(config.strict_subject_consistency):
        if str(dataset_session2.subject_id) != str(dataset_session1.subject_id):
            raise RuntimeError(
                "session2 subject_id differs from session1 under strict subject mode; "
                f"session1={dataset_session1.subject_id}, session2={dataset_session2.subject_id}"
            )
    if dataset_session2 is not None and bool(config.strict_protocol_consistency):
        protocol_s1 = _canonical_protocol_signature(dataset_session1)
        protocol_s2 = _canonical_protocol_signature(dataset_session2)
        protocol_diff = _protocol_mismatch_details(protocol_s1, protocol_s2)
        if protocol_diff:
            raise RuntimeError(
                "session2 protocol_config differs from session1 under strict protocol mode; "
                f"mismatches={protocol_diff}"
            )
    expected_protocol_signature = str(protocol_consistency_summary.get("protocol_signature_expected", ""))
    if dataset_session2 is not None and data_policy == "new-only":
        session2_signature = _manifest_protocol_signature(dataset_session2)
        if not session2_signature:
            raise RuntimeError(
                "data-policy=new-only requires session2 protocol_signature in manifest; "
                f"manifest={config.dataset_manifest_session2}"
            )
        if session2_signature != expected_protocol_signature:
            raise RuntimeError(
                "session2 protocol_signature differs from session1 under data-policy=new-only; "
                f"expected={expected_protocol_signature}, got={session2_signature}"
            )
    if dataset_session2 is not None:
        session2_stages = {
            str(row.get("stage", "")).strip().lower()
            for row in list(dataset_session2.manifest.get("trials", []))
            if isinstance(row, dict)
        }
        if session2_stages and session2_stages != {"collection"}:
            raise RuntimeError(
                "session2 manifest is not an independent collection dataset; "
                f"got stages={sorted(session2_stages)}"
            )

    for row in session1_quality_rows:
        log(
            "Session1 quality filter: "
            f"session={row.get('session_id','')} kept={row.get('kept_trials',0)}/{row.get('total_trials',0)} "
            f"drop={row.get('dropped_trials',0)} shortfall_drop={row.get('dropped_shortfall',0)} "
            f"retry_drop={row.get('dropped_retry',0)}"
        )
    if session2_quality_row is not None:
        log(
            "Session2 quality filter: "
            f"session={session2_quality_row.get('session_id','')} "
            f"kept={session2_quality_row.get('kept_trials',0)}/{session2_quality_row.get('total_trials',0)} "
            f"drop={session2_quality_row.get('dropped_trials',0)}"
        )

    resolved_report_paths = _resolve_report_paths(config, subject_id=str(dataset_session1.subject_id))
    report_json_path = Path(resolved_report_paths["report_json"]).expanduser().resolve()
    report_dir = Path(resolved_report_paths["report_dir"]).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    selection_snapshot_path = Path(resolved_report_paths["selection_snapshot"]).expanduser().resolve()
    run_config_path = Path(resolved_report_paths["run_config"]).expanduser().resolve()
    run_log_path = Path(resolved_report_paths["run_log"]).expanduser().resolve()
    if run_log_lines:
        run_log_path.write_text("\n".join(run_log_lines).strip() + "\n", encoding="utf-8")
    log(f"Report directory prepared: {report_dir}")

    fs = int(dataset_session1.sampling_rate)
    eeg_channels = tuple(int(channel) for channel in dataset_session1.board_eeg_channels)
    train_base, gate_base, holdout_base = _split_session_for_train_eval(dataset_session1, seed=config.seed)
    log(
        "Offline split ready: "
        f"train={len(train_base)} gate={len(gate_base)} holdout={len(holdout_base)}"
    )

    helper = BenchmarkRunner(
        serial_port="offline",
        board_id=0,
        freqs=dataset_session1.freqs,
        output_profile_path=config.output_profile_path,
        report_path=report_json_path,
        dataset_dir=report_dir,
        sampling_rate=fs,
        prepare_sec=float(dataset_session1.protocol_config.get("prepare_sec", 1.0)),
        active_sec=float(dataset_session1.protocol_config.get("active_sec", 4.0)),
        rest_sec=float(dataset_session1.protocol_config.get("rest_sec", 1.0)),
        calibration_target_repeats=1,
        calibration_idle_repeats=1,
        eval_target_repeats=1,
        eval_idle_repeats=1,
        eval_switch_trials=1,
        step_sec=float(dataset_session1.protocol_config.get("step_sec", 0.25)),
        model_names=parse_model_list(",".join(config.model_names)),
        channel_modes=parse_channel_mode_list(",".join(config.channel_modes)),
        multi_seed_count=max(1, int(config.multi_seed_count)),
        seed_step=max(1, int(config.seed_step)),
        win_candidates=tuple(float(value) for value in config.win_candidates),
        gate_policy=parse_gate_policy(config.gate_policy),
        channel_weight_mode=config.channel_weight_mode,
        subband_weight_mode=config.subband_weight_mode,
        spatial_filter_mode=config.spatial_filter_mode,
        spatial_rank_candidates=tuple(int(value) for value in config.spatial_rank_candidates),
        joint_weight_iters=max(1, int(config.joint_weight_iters)),
        weight_cv_folds=max(2, int(config.weight_cv_folds)),
        spatial_source_model=str(config.spatial_source_model),
        metric_scope=metric_scope,
        decision_time_mode=decision_time_mode,
        async_decision_time_mode=async_decision_time_mode,
        ranking_policy=ranking_policy,
        dynamic_stop_enabled=bool(config.dynamic_stop_enabled),
        dynamic_stop_alpha=float(config.dynamic_stop_alpha),
        seed=int(config.seed),
    )

    model_results: list[dict[str, Any]] = []
    best_profiles: dict[str, Any] = {}
    robustness_runs: list[dict[str, Any]] = []
    primary_mode = "auto" if "auto" in helper.channel_modes else str(helper.channel_modes[0])
    primary_seed = int(helper.eval_seeds[0]) if helper.eval_seeds else int(config.seed)
    total_runs = int(len(helper.channel_modes) * len(helper.eval_seeds) * len(helper.model_names))
    current_run = 0

    for channel_mode in helper.channel_modes:
        for eval_seed in helper.eval_seeds:
            seed_mode_success: list[dict[str, Any]] = []
            seed_mode_all: list[dict[str, Any]] = []
            for model_name in helper.model_names:
                model_name = normalize_model_name(model_name)
                current_run += 1
                include_details = bool(channel_mode == primary_mode and int(eval_seed) == int(primary_seed))
                try:
                    log(
                        f"Model start: {current_run}/{total_runs} "
                        f"mode={channel_mode} seed={eval_seed} model={model_name}"
                    )
                    if channel_mode == "auto":
                        selected_channels, channel_scores = select_auto_eeg_channels_for_model(
                            train_base,
                            model_name=model_name,
                            available_board_channels=eeg_channels,
                            sampling_rate=fs,
                            freqs=dataset_session1.freqs,
                            win_sec=max(helper.win_candidates),
                            step_sec=helper.step_sec,
                            model_params={
                                "Nh": 3,
                                "compute_backend": str(config.compute_backend),
                                "gpu_device": int(config.gpu_device),
                                "gpu_precision": str(config.gpu_precision),
                                "gpu_cache_policy": str(config.gpu_cache_policy),
                                "gpu_warmup": bool(config.gpu_warmup),
                            },
                            compute_backend=str(config.compute_backend),
                            gpu_device=int(config.gpu_device),
                            gpu_precision=str(config.gpu_precision),
                            gpu_warmup=bool(config.gpu_warmup),
                            gpu_cache_policy=str(config.gpu_cache_policy),
                            seed=int(eval_seed),
                        )
                    else:
                        selected_channels = tuple(int(channel) for channel in eeg_channels)
                        channel_scores = []

                    selected_positions = [list(eeg_channels).index(channel) for channel in selected_channels]
                    train_segments = _subset_trial_segments_by_positions(train_base, selected_positions)
                    gate_segments = _subset_trial_segments_by_positions(gate_base, selected_positions)
                    holdout_segments = _subset_trial_segments_by_positions(holdout_base, selected_positions)
                    if model_name == "fbcca":
                        log(
                            "FBCCA 当前会额外执行通道权重、空间滤波和门控拟合，"
                            "CPU 下耗时会明显高于前面的 CCA 家族模型。"
                        )

                    profile, result = helper._benchmark_single_model(
                        model_name=model_name,
                        fs=fs,
                        train_segments=train_segments,
                        gate_segments=gate_segments,
                        eval_segments=holdout_segments,
                        eeg_channels=selected_channels,
                        log_fn=log,
                    )
                    result["selected_eeg_channels"] = [int(channel) for channel in selected_channels]
                    result["channel_mode"] = str(channel_mode)
                    result["eval_seed"] = int(eval_seed)
                    result["split_counts"] = {
                        "train_segments": int(len(train_segments)),
                        "gate_segments": int(len(gate_segments)),
                        "holdout_segments": int(len(holdout_segments)),
                    }
                    if include_details:
                        result["channel_selection"] = channel_scores
                    if dataset_session2 is not None:
                        positions_s2 = [list(dataset_session2.board_eeg_channels).index(channel) for channel in selected_channels]
                        session2_subset = _subset_trial_segments_by_positions(dataset_session2.trial_segments, positions_s2)
                        decoder_s2 = load_decoder_from_profile(profile, sampling_rate=dataset_session2.sampling_rate)
                        cross_session_bundle = evaluate_decoder_on_trials_v2(
                            decoder_s2,
                            profile,
                            session2_subset,
                            metric_scope=metric_scope,
                            paper_decision_time_mode=decision_time_mode,
                            async_decision_time_mode=async_decision_time_mode,
                        )
                        result["cross_session_metrics"] = pack_evaluation_metrics_for_ranking(
                            cross_session_bundle,
                            metric_scope=metric_scope,
                        )
                        result["cross_session_async_metrics"] = dict(cross_session_bundle.get("async_metrics", {}))
                        result["cross_session_metrics_4class"] = dict(cross_session_bundle.get("metrics_4class", {}))
                        result["cross_session_metrics_2class"] = dict(cross_session_bundle.get("metrics_2class", {}))
                        result["cross_session_metrics_5class"] = (
                            None
                            if cross_session_bundle.get("metrics_5class") is None
                            else dict(cross_session_bundle.get("metrics_5class", {}))
                        )
                        result["cross_session_paper_lens_metrics"] = {
                            "metrics_4class": dict(cross_session_bundle.get("paper_lens_metrics_4class", {})),
                            "metrics_2class": dict(cross_session_bundle.get("paper_lens_metrics_2class", {})),
                            "metrics_5class": (
                                None
                                if cross_session_bundle.get("paper_lens_metrics_5class") is None
                                else dict(cross_session_bundle.get("paper_lens_metrics_5class", {}))
                            ),
                        }
                        result["cross_session_async_lens_metrics"] = {
                            "metrics_4class": dict(cross_session_bundle.get("async_lens_metrics_4class", {})),
                            "metrics_2class": dict(cross_session_bundle.get("async_lens_metrics_2class", {})),
                            "metrics_5class": (
                                None
                                if cross_session_bundle.get("async_lens_metrics_5class") is None
                                else dict(cross_session_bundle.get("async_lens_metrics_5class", {}))
                            ),
                        }

                    if include_details:
                        best_profiles[str(model_name)] = profile
                        model_results.append(dict(result))
                    seed_mode_all.append(dict(result))
                    seed_mode_success.append(dict(result))
                    log(
                        f"Model done: mode={channel_mode} seed={eval_seed} model={model_name} "
                        f"idle_fp={result['metrics'].get('idle_fp_per_min', float('inf')):.4f} "
                        f"recall={result['metrics'].get('control_recall', 0.0):.4f}"
                    )
                except Exception as exc:
                    failed = {
                        "model_name": str(model_name),
                        "implementation_level": model_implementation_level(model_name),
                        "method_note": model_method_note(model_name),
                        "channel_mode": str(channel_mode),
                        "eval_seed": int(eval_seed),
                        "error": str(exc),
                        "meets_acceptance": False,
                    }
                    if include_details:
                        model_results.append(dict(failed))
                    seed_mode_all.append(dict(failed))
                    log(f"Model failed: mode={channel_mode} seed={eval_seed} model={model_name} error={exc}")

            seed_mode_success.sort(
                key=lambda item: benchmark_rank_key(dict(item["metrics"]), ranking_policy=ranking_policy)
            )
            rank_by_model = {str(item["model_name"]): int(rank) for rank, item in enumerate(seed_mode_success, start=1)}
            for item in seed_mode_all:
                item["run_rank"] = rank_by_model.get(str(item.get("model_name", "")))
                robustness_runs.append(item)

    successful = [item for item in model_results if "metrics" in item]
    if not successful:
        raise RuntimeError("offline train-eval failed: all models failed")

    successful.sort(key=lambda item: benchmark_rank_key(dict(item["metrics"]), ranking_policy=ranking_policy))
    for rank, item in enumerate(successful, start=1):
        item["rank_end_to_end"] = int(rank)
        item["rank"] = int(rank)
        item["meets_acceptance"] = bool(profile_meets_acceptance(dict(item["metrics"])))

    classifier_only_ranked = sorted(
        successful,
        key=lambda item: _classifier_only_rank_key(
            dict(item.get("classifier_only_metrics") or item.get("metrics") or {})
        ),
    )
    for rank, item in enumerate(classifier_only_ranked, start=1):
        item["rank_classifier_only"] = int(rank)

    accepted = [item for item in successful if bool(item.get("meets_acceptance"))]
    recommended = successful[0]
    chosen = accepted[0] if accepted else recommended

    stats_baseline = next(
        (item for item in successful if str(item.get("model_name", "")) == "legacy_fbcca_202603"),
        successful[0],
    )
    baseline_true, baseline_pred, baseline_times, _baseline_labels = _extract_4class_vectors(stats_baseline)
    for index, item in enumerate(successful):
        y_true, y_pred, decision_times, labels = _extract_4class_vectors(item)
        item["bootstrap_4class"] = _bootstrap_4class_summary(
            y_true=y_true,
            y_pred=y_pred,
            decision_time_samples_s=decision_times,
            labels=labels if labels else ("8Hz", "10Hz", "12Hz", "15Hz"),
            seed=int(config.seed) + int(index) * 17,
        )
        paired_stats = {
            "baseline_model": str(stats_baseline.get("model_name", "")),
            "mcnemar_4class_acc": _paired_mcnemar(
                baseline_true=baseline_true,
                baseline_pred=baseline_pred,
                candidate_true=y_true,
                candidate_pred=y_pred,
            ),
            "wilcoxon_decision_time": _paired_wilcoxon(
                baseline_times=baseline_times,
                candidate_times=decision_times,
            ),
        }
        item["paired_vs_baseline"] = paired_stats

    ranking_end_to_end = [
        {
            "rank": int(item.get("rank_end_to_end", 0) or 0),
            "model_name": str(item.get("model_name", "")),
            "channel_mode": str(item.get("channel_mode", "")),
            "eval_seed": int(item.get("eval_seed", 0) or 0),
            "metrics": dict(item.get("metrics", {})),
            "meets_acceptance": bool(item.get("meets_acceptance", False)),
        }
        for item in successful
    ]
    ranking_classifier_only = [
        {
            "rank": int(item.get("rank_classifier_only", 0) or 0),
            "model_name": str(item.get("model_name", "")),
            "channel_mode": str(item.get("channel_mode", "")),
            "eval_seed": int(item.get("eval_seed", 0) or 0),
            "metrics": dict(item.get("classifier_only_metrics") or item.get("metrics") or {}),
        }
        for item in classifier_only_ranked
    ]
    classifier_only_top = classifier_only_ranked[0]

    profile_saved = False
    chosen_profile_path: Optional[str] = None
    if accepted:
        chosen_profile = best_profiles[str(chosen["model_name"])]
        chosen_profile = replace(
            chosen_profile,
            benchmark_metrics={
                key: float(value)
                for key, value in dict(chosen["metrics"]).items()
                if isinstance(value, (float, int)) and np.isfinite(float(value))
            },
        )
        save_profile(chosen_profile, config.output_profile_path)
        profile_saved = True
        chosen_profile_path = str(config.output_profile_path)
    else:
        log(
            "No model meets acceptance thresholds. "
            f"Top candidate={recommended.get('model_name')} (rank=1) is kept for reference only; profile not saved."
        )

    robustness_summary = summarize_benchmark_robustness(robustness_runs, ranking_policy=ranking_policy)
    metric_definition = benchmark_metric_definition_payload(
        ranking_policy=ranking_policy,
        metric_scope=metric_scope,
        decision_time_mode=decision_time_mode,
        async_decision_time_mode=async_decision_time_mode,
    )
    quality_total_trials_session1 = int(sum(int(row.get("total_trials", 0) or 0) for row in session1_quality_rows))
    quality_kept_trials_session1 = int(sum(int(row.get("kept_trials", 0) or 0) for row in session1_quality_rows))
    quality_dropped_trials_session1 = int(max(quality_total_trials_session1 - quality_kept_trials_session1, 0))
    report_payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "offline_train_eval",
        "report_path": str(report_json_path),
        "report_dir": str(report_dir),
        "selection_snapshot_path": str(selection_snapshot_path),
        "run_config_path": str(run_config_path),
        "run_log_path": str(run_log_path),
        "dataset_manifest_session1": str(config.dataset_manifest_session1),
        "selected_dataset_manifests_session1": [str(path) for path in selected_session1_manifest_paths],
        "selected_dataset_count_session1": int(len(selected_session1_manifest_paths)),
        "dataset_manifest_session2": None if config.dataset_manifest_session2 is None else str(config.dataset_manifest_session2),
        "quality_filter": {
            "min_sample_ratio": float(config.quality_min_sample_ratio),
            "max_retry_count": int(config.quality_max_retry_count),
        },
        "quality_summary_session1": list(session1_quality_rows),
        "quality_summary_session2": None if session2_quality_row is None else dict(session2_quality_row),
        "quality_total_trials_session1": int(quality_total_trials_session1),
        "quality_kept_trials_session1": int(quality_kept_trials_session1),
        "quality_dropped_trials_session1": int(quality_dropped_trials_session1),
        "protocol_consistency": dict(protocol_consistency_summary),
        "session1_id": dataset_session1.session_id,
        "session2_id": None if dataset_session2 is None else dataset_session2.session_id,
        "sampling_rate": fs,
        "freqs": list(dataset_session1.freqs),
        "board_eeg_channels": list(dataset_session1.board_eeg_channels),
        "split_counts_session1": {
            "train": int(len(train_base)),
            "gate": int(len(gate_base)),
            "holdout": int(len(holdout_base)),
        },
        "model_results": model_results,
        "chosen_model": str(chosen["model_name"]),
        "chosen_rank": int(chosen.get("rank_end_to_end", chosen.get("rank", 0)) or 0),
        "chosen_metrics": dict(chosen.get("metrics", {})),
        "chosen_async_metrics": dict(chosen.get("async_metrics", {})),
        "chosen_metrics_4class": dict(chosen.get("metrics_4class", {})),
        "chosen_metrics_2class": dict(chosen.get("metrics_2class", {})),
        "chosen_metrics_5class": (
            None
            if chosen.get("metrics_5class") is None
            else dict(chosen.get("metrics_5class", {}))
        ),
        "chosen_fixed_window_metrics": dict(chosen.get("fixed_window_metrics", {})),
        "chosen_fixed_window_async_metrics": dict(chosen.get("fixed_window_async_metrics", {})),
        "chosen_fixed_window_metrics_4class": dict(chosen.get("fixed_window_metrics_4class", {})),
        "chosen_fixed_window_metrics_2class": dict(chosen.get("fixed_window_metrics_2class", {})),
        "chosen_fixed_window_metrics_5class": (
            None
            if chosen.get("fixed_window_metrics_5class") is None
            else dict(chosen.get("fixed_window_metrics_5class", {}))
        ),
        "chosen_dynamic_delta": dict(chosen.get("dynamic_delta", {})),
        "chosen_profile_path": chosen_profile_path,
        "chosen_meets_acceptance": bool(chosen.get("meets_acceptance", False)),
        "accepted_model_count": int(len(accepted)),
        "recommended_model": str(recommended.get("model_name", "")),
        "recommended_rank": int(recommended.get("rank_end_to_end", recommended.get("rank", 0)) or 0),
        "recommended_metrics": dict(recommended.get("metrics", {})),
        "classifier_only_top_model": str(classifier_only_top.get("model_name", "")),
        "classifier_only_top_rank": int(classifier_only_top.get("rank_classifier_only", 0) or 0),
        "classifier_only_top_metrics": dict(
            classifier_only_top.get("classifier_only_metrics") or classifier_only_top.get("metrics") or {}
        ),
        "ranking_boards": {
            "end_to_end": ranking_end_to_end,
            "classifier_only": ranking_classifier_only,
        },
        "stats_baseline_model": str(stats_baseline.get("model_name", "")),
        "ab_comparisons": _build_ab_comparisons(successful),
        "profile_saved": bool(profile_saved),
        "metric_scope": str(metric_scope),
        "decision_time_mode": str(decision_time_mode),
        "async_decision_time_mode": str(async_decision_time_mode),
        "data_policy": str(data_policy),
        "protocol_signature_expected": str(expected_protocol_signature),
        "excluded_sessions": list(protocol_consistency_summary.get("excluded_sessions", [])),
        "ranking_policy": str(ranking_policy),
        "export_figures": bool(export_figures),
        "gate_policy": str(getattr(helper, "gate_policy", config.gate_policy)),
        "channel_weight_mode": getattr(helper, "channel_weight_mode", config.channel_weight_mode),
        "spatial_filter_mode": getattr(helper, "spatial_filter_mode", config.spatial_filter_mode),
        "spatial_rank_candidates": [
            int(value)
            for value in getattr(helper, "spatial_rank_candidates", config.spatial_rank_candidates)
        ],
        "joint_weight_iters": int(getattr(helper, "joint_weight_iters", config.joint_weight_iters)),
        "spatial_source_model": str(getattr(helper, "spatial_source_model", config.spatial_source_model)),
        "dynamic_stop_enabled": bool(getattr(helper, "dynamic_stop_enabled", config.dynamic_stop_enabled)),
        "dynamic_stop_alpha": float(getattr(helper, "dynamic_stop_alpha", config.dynamic_stop_alpha)),
        "metric_definition": metric_definition,
        "formula_definitions": dict(metric_definition.get("formula_definitions", {})),
        "method_references": list(metric_definition.get("method_references", [])),
        "robustness": robustness_summary,
    }
    report_payload["async_metrics"] = dict(report_payload.get("chosen_async_metrics", {}))
    report_payload["metrics_4class"] = dict(report_payload.get("chosen_metrics_4class", {}))
    report_payload["metrics_2class"] = dict(report_payload.get("chosen_metrics_2class", {}))
    report_payload["metrics_5class"] = report_payload.get("chosen_metrics_5class")
    report_payload["paper_lens_metrics"] = {
        "metrics_4class": dict(chosen.get("paper_lens_metrics_4class", chosen.get("metrics_4class", {}))),
        "metrics_2class": dict(chosen.get("paper_lens_metrics_2class", chosen.get("metrics_2class", {}))),
        "metrics_5class": (
            None
            if chosen.get("paper_lens_metrics_5class", chosen.get("metrics_5class")) is None
            else dict(chosen.get("paper_lens_metrics_5class", chosen.get("metrics_5class", {})))
        ),
        "decision_time_mode": str(decision_time_mode),
    }
    report_payload["async_lens_metrics"] = {
        "metrics_4class": dict(chosen.get("async_lens_metrics_4class", chosen.get("metrics_4class", {}))),
        "metrics_2class": dict(chosen.get("async_lens_metrics_2class", chosen.get("metrics_2class", {}))),
        "metrics_5class": (
            None
            if chosen.get("async_lens_metrics_5class", chosen.get("metrics_5class")) is None
            else dict(chosen.get("async_lens_metrics_5class", chosen.get("metrics_5class", {})))
        ),
        "decision_time_mode": str(async_decision_time_mode),
    }
    if export_figures:
        try:
            report_payload["figures"] = export_evaluation_figures(report_payload, output_dir=report_dir)
        except Exception as exc:
            report_payload["figures"] = {}
            report_payload["figure_export_error"] = str(exc)
            log(f"Figure export skipped: {exc}")
    else:
        report_payload["figures"] = {}

    config_payload = asdict(config)
    config_payload["dataset_manifest_session1"] = str(Path(config.dataset_manifest_session1).expanduser().resolve())
    config_payload["dataset_manifest_session2"] = (
        None
        if config.dataset_manifest_session2 is None
        else str(Path(config.dataset_manifest_session2).expanduser().resolve())
    )
    config_payload["output_profile_path"] = str(Path(config.output_profile_path).expanduser().resolve())
    config_payload["report_path"] = str(Path(config.report_path).expanduser().resolve())
    config_payload["dataset_manifests"] = [str(Path(path).expanduser().resolve()) for path in config.dataset_manifests]
    config_payload["report_root_dir"] = (
        None if config.report_root_dir is None else str(Path(config.report_root_dir).expanduser().resolve())
    )
    config_payload["resolved_report_path"] = str(report_json_path)
    config_payload["resolved_report_dir"] = str(report_dir)

    selection_snapshot_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_dataset_manifests_session1": [str(path) for path in selected_session1_manifest_paths],
        "selected_dataset_count_session1": int(len(selected_session1_manifest_paths)),
        "dataset_manifest_session2": (
            None
            if config.dataset_manifest_session2 is None
            else str(Path(config.dataset_manifest_session2).expanduser().resolve())
        ),
        "quality_filter": {
            "min_sample_ratio": float(config.quality_min_sample_ratio),
            "max_retry_count": int(config.quality_max_retry_count),
        },
        "strict_modes": {
            "strict_protocol_consistency": bool(config.strict_protocol_consistency),
            "strict_subject_consistency": bool(config.strict_subject_consistency),
        },
        "data_policy": str(data_policy),
        "decision_time_mode": str(decision_time_mode),
        "async_decision_time_mode": str(async_decision_time_mode),
        "protocol_signature_expected": str(expected_protocol_signature),
        "quality_summary_session1": list(session1_quality_rows),
        "quality_summary_session2": None if session2_quality_row is None else dict(session2_quality_row),
        "dataset_selection_snapshot": dict(config.dataset_selection_snapshot or {}),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    run_config_path.write_text(json_dumps(json_safe(config_payload)) + "\n", encoding="utf-8")
    selection_snapshot_path.write_text(json_dumps(json_safe(selection_snapshot_payload)) + "\n", encoding="utf-8")
    report_json_path.write_text(json_dumps(json_safe(report_payload)) + "\n", encoding="utf-8")
    markdown_path = report_json_path.with_suffix(".md")
    markdown_path.write_text(_render_training_eval_markdown(report_payload), encoding="utf-8")
    log(f"Report saved: {report_json_path}")
    log(f"Markdown saved: {markdown_path}")
    log(f"Selection snapshot saved: {selection_snapshot_path}")
    log(f"Run config saved: {run_config_path}")
    log(f"Run log saved: {run_log_path}")
    if profile_saved:
        log(f"Profile saved: {config.output_profile_path}")
    else:
        log("Profile not saved because no model met acceptance thresholds.")
    run_log_path.write_text("\n".join(run_log_lines).strip() + ("\n" if run_log_lines else ""), encoding="utf-8")
    return report_payload


def run_offline_train_eval(
    config: OfflineTrainEvalConfig,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_fn: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    from ._train_eval_staged import run_offline_train_eval as _staged_impl

    return _staged_impl(config, log_fn=log_fn, progress_fn=progress_fn)


def _render_training_eval_markdown(report_payload: dict[str, Any]) -> str:
    chosen_async = dict(report_payload.get("chosen_async_metrics", {}))
    chosen_4 = dict(report_payload.get("chosen_metrics_4class", {}))
    chosen_2 = dict(report_payload.get("chosen_metrics_2class", {}))
    chosen_frontend = dict(report_payload.get("chosen_frontend_summary", {}))
    quality_filter = dict(report_payload.get("quality_filter", {}))
    quality_rows_s1 = [
        row for row in list(report_payload.get("quality_summary_session1", [])) if isinstance(row, dict)
    ]
    lines = [
        "# SSVEP Offline Training-Evaluation Report",
        "",
        f"- Generated at: `{report_payload.get('generated_at', '')}`",
        f"- Session1 manifest: `{report_payload.get('dataset_manifest_session1', '')}`",
        f"- Session2 manifest: `{report_payload.get('dataset_manifest_session2', '')}`",
        f"- Chosen model: `{report_payload.get('chosen_model', '')}`",
        f"- Chosen rank: `{report_payload.get('chosen_rank', '')}`",
        f"- Chosen meets acceptance: `{report_payload.get('chosen_meets_acceptance', False)}`",
        f"- Profile saved: `{report_payload.get('profile_saved', False)}`",
        f"- Profile path: `{report_payload.get('chosen_profile_path', '')}`",
        "",
        "## 数据质量过滤",
        "",
        f"- min_sample_ratio: `{float(quality_filter.get('min_sample_ratio', 0.0)):.3f}`",
        f"- max_retry_count: `{int(quality_filter.get('max_retry_count', 0) or 0)}`",
        f"- Session1 kept/total: `{int(report_payload.get('quality_kept_trials_session1', 0) or 0)}`/`{int(report_payload.get('quality_total_trials_session1', 0) or 0)}`",
        "",
        "| Session | Kept/Total | Drop Ratio | Drop by Shortfall | Drop by Retry |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in quality_rows_s1:
        lines.append(
            "| {sid} | {kept}/{total} | {drop:.3f} | {shortfall} | {retry} |".format(
                sid=str(row.get("session_id", "")),
                kept=int(row.get("kept_trials", 0) or 0),
                total=int(row.get("total_trials", 0) or 0),
                drop=float(row.get("drop_ratio", 0.0) or 0.0),
                shortfall=int(row.get("dropped_shortfall", 0) or 0),
                retry=int(row.get("dropped_retry", 0) or 0),
            )
        )
    lines.extend(
        [
            "",
            "## 6.2.1 SSVEP 分类准确率",
            "",
            "| 口径 | Acc_SSVEP | Macro-F1 | Mean Decision Time(s) | ITR(bits/min) |",
            "|---|---:|---:|---:|---:|",
            "| 四分类(8/10/12/15Hz) | {acc4:.4f} | {f14:.4f} | {t4:.4f} | {itr4:.4f} |".format(
                acc4=float(chosen_4.get("acc", 0.0)),
                f14=float(chosen_4.get("macro_f1", 0.0)),
                t4=float(chosen_4.get("mean_decision_time_s", float("inf"))),
                itr4=float(chosen_4.get("itr_bpm", 0.0)),
            ),
            "| 二分类(control vs idle) | {acc2:.4f} | {f12:.4f} | {t2:.4f} | {itr2:.4f} |".format(
                acc2=float(chosen_2.get("acc", 0.0)),
                f12=float(chosen_2.get("macro_f1", 0.0)),
                t2=float(chosen_2.get("mean_decision_time_s", float("inf"))),
                itr2=float(chosen_2.get("itr_bpm", 0.0)),
            ),
            "",
            "## 6.2.2 异步可用性评测",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            "| idle_fp_per_min | {idle:.4f} |".format(idle=float(chosen_async.get("idle_fp_per_min", float("inf")))),
            "| control_recall | {recall:.4f} |".format(recall=float(chosen_async.get("control_recall", 0.0))),
            "| switch_latency_s | {switch_lat:.4f} |".format(
                switch_lat=float(chosen_async.get("switch_latency_s", float("inf")))
            ),
            "| release_latency_s | {release_lat:.4f} |".format(
                release_lat=float(chosen_async.get("release_latency_s", float("inf")))
            ),
            "| inference_ms | {infer:.4f} |".format(
                infer=float(chosen_async.get("inference_ms", float("inf")))
            ),
        ]
    )
    frontend_weights = list(chosen_frontend.get("channel_weights", []) or [])
    frontend_weight_stats = dict(chosen_frontend.get("channel_weight_stats", {}) or {})
    joint_summary = dict(chosen_frontend.get("joint_weight_training", {}) or {})
    top_channels = list(chosen_frontend.get("top_weight_channels", []) or [])
    if chosen_frontend:
        lines.extend(
            [
                "",
                "## FBCCA 前端与权重摘要",
                "",
                f"- channel_weight_mode: `{chosen_frontend.get('channel_weight_mode', None)}`",
                f"- spatial_filter_mode: `{chosen_frontend.get('spatial_filter_mode', None)}`",
                f"- spatial_filter_rank: `{chosen_frontend.get('spatial_filter_rank', None)}`",
            ]
        )
        if frontend_weights:
            lines.append(f"- channel_weights: `{frontend_weights}`")
        if frontend_weight_stats:
            lines.append(
                "- channel_weight_stats: "
                f"`count={int(frontend_weight_stats.get('count', 0) or 0)}, "
                f"min={float(frontend_weight_stats.get('min', 0.0)):.4f}, "
                f"max={float(frontend_weight_stats.get('max', 0.0)):.4f}, "
                f"mean={float(frontend_weight_stats.get('mean', 0.0)):.4f}, "
                f"std={float(frontend_weight_stats.get('std', 0.0)):.4f}`"
            )
        if top_channels:
            pairs = ", ".join(
                f"ch{int(item.get('board_channel', 0))}={float(item.get('weight', 0.0)):.4f}"
                for item in top_channels
                if isinstance(item, dict)
            )
            lines.append(f"- top_weight_channels: `{pairs}`")
        if joint_summary:
            lines.append(
                "- joint_weight_training: "
                f"`mode={joint_summary.get('mode', '')}, "
                f"iters={int(joint_summary.get('joint_weight_iters', 0) or 0)}, "
                f"iteration_count={int(joint_summary.get('iteration_count', 0) or 0)}, "
                f"improved_iteration_count={int(joint_summary.get('improved_iteration_count', 0) or 0)}, "
                f"selected_spatial_rank={joint_summary.get('selected_spatial_rank', None)}`"
            )
            lines.append(f"- joint_objective: `{list(joint_summary.get('objective', []))}`")
    lines.extend(
        [
            "",
            "## 图表文件",
            "",
            f"- confusion_4class: `{dict(report_payload.get('figures', {})).get('confusion_4class', '')}`",
            f"- confusion_2class: `{dict(report_payload.get('figures', {})).get('confusion_2class', '')}`",
            f"- decision_time_hist: `{dict(report_payload.get('figures', {})).get('decision_time_hist', '')}`",
            f"- model_radar_async_vs_cls: `{dict(report_payload.get('figures', {})).get('model_radar_async_vs_cls', '')}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _render_training_eval_markdown(report_payload: dict[str, Any]) -> str:
    return _render_training_eval_markdown_clean(report_payload)
