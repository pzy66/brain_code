from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .train_eval import (
    AsyncDecisionGate,
    DEFAULT_NH,
    OfflineTrainEvalConfig,
    _aggregate_family_results,
    _apply_trial_quality_filter,
    _bootstrap_4class_summary,
    _build_ab_comparisons,
    _canonical_protocol_signature,
    _classifier_only_rank_key,
    _evaluate_classifier_only_fixed_window,
    _extract_4class_vectors,
    _load_session1_dataset,
    _manifest_protocol_signature,
    _metrics_source_name,
    _paired_mcnemar,
    _paired_wilcoxon,
    _parse_evaluation_mode,
    _protocol_mismatch_details,
    _quick_screen_rank_key,
    _quick_screen_models,
    _ranking_metrics_from_result,
    _render_training_eval_markdown,
    _resolve_report_paths,
    _sanitize_path_token,
    _split_session_for_train_eval,
    _subset_trial_segments_by_positions,
    benchmark_metric_definition_payload,
    benchmark_rank_key,
    create_decoder,
    evaluate_decoder_on_trials_v2,
    export_evaluation_figures,
    atomic_copy_text_file,
    atomic_write_text,
    json_dumps,
    json_safe,
    load_collection_dataset,
    load_decoder_from_profile,
    load_profile,
    model_implementation_level,
    model_method_note,
    normalize_model_name,
    pack_evaluation_metrics_for_ranking,
    parse_channel_mode_list,
    parse_data_policy,
    parse_decision_time_mode,
    parse_gate_policy,
    parse_metric_scope,
    parse_model_list,
    parse_ranking_policy,
    profile_meets_acceptance,
    save_profile,
    select_auto_eeg_channels_for_model,
    summarize_benchmark_robustness,
    BenchmarkRunner,
)


def _fbcca_weight_learning_requires_all8_for_config(model_name: str, config: OfflineTrainEvalConfig) -> bool:
    name = normalize_model_name(model_name)
    if name in {"fbcca_fixed_all8", "fbcca_cw_all8", "fbcca_sw_all8", "fbcca_cw_sw_all8", "fbcca_cw_sw_trca_shared"}:
        return True
    if name != "fbcca":
        return False
    channel_mode = str(config.channel_weight_mode or "").strip().lower()
    subband_mode = str(config.subband_weight_mode or "").strip().lower()
    return channel_mode not in {"", "none"} or subband_mode not in {"", "none", "chen_fixed"}


def _best_successful_by_model(
    successful_runs: Sequence[dict[str, Any]],
    *,
    prefer_cross_session: bool,
    ranking_policy: str,
) -> list[dict[str, Any]]:
    sorted_runs = sorted(
        (dict(item) for item in successful_runs),
        key=lambda item: benchmark_rank_key(
            _ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session),
            ranking_policy=ranking_policy,
        ),
    )
    best_by_model: dict[str, dict[str, Any]] = {}
    for item in sorted_runs:
        model_name = str(item.get("model_name", "")).strip()
        if model_name and model_name not in best_by_model:
            best_by_model[model_name] = item
    return [best_by_model[name] for name in best_by_model]


def _build_classifier_only_board(
    *,
    quick_screen_ranked: Sequence[dict[str, Any]],
    deployment_ranked_by_model: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if quick_screen_ranked:
        board: list[dict[str, Any]] = []
        for item in quick_screen_ranked:
            board.append(
                {
                    "rank": int(item.get("quick_screen_rank", 0) or 0),
                    "model_name": str(item.get("model_name", "")),
                    "channel_mode": str(item.get("channel_mode", "all8")),
                    "eval_seed": int(item.get("eval_seed", 0) or 0),
                    "metrics": dict(item.get("classifier_only_metrics", {})),
                }
            )
        return board

    ranked = sorted(
        (dict(item) for item in deployment_ranked_by_model if "classifier_only_metrics" in item),
        key=lambda item: _classifier_only_rank_key(dict(item.get("classifier_only_metrics", {}))),
    )
    board = []
    for rank, item in enumerate(ranked, start=1):
        board.append(
            {
                "rank": int(rank),
                "model_name": str(item.get("model_name", "")),
                "channel_mode": str(item.get("channel_mode", "")),
                "eval_seed": int(item.get("eval_seed", 0) or 0),
                "metrics": dict(item.get("classifier_only_metrics", {})),
            }
        )
    return board


def _dataset_has_long_idle(dataset: Any) -> bool:
    try:
        long_idle_sec = float(dict(getattr(dataset, "protocol_config", {}) or {}).get("long_idle_sec", 0.0) or 0.0)
    except Exception:
        long_idle_sec = 0.0
    if long_idle_sec > 0.0:
        return True
    for row in list(getattr(dataset, "manifest", {}).get("trials", []) or []):
        if not isinstance(row, dict):
            continue
        label = str(row.get("label", "")).strip().lower()
        stage = str(row.get("stage", "")).strip().lower()
        if "long_idle" in label or "long idle" in label or "long_idle" in stage or "long idle" in stage:
            return True
    return False


def _numeric_metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys = sorted({str(key) for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    summary: dict[str, dict[str, Any]] = {}
    for key in keys:
        values = np.asarray(
            [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and np.isfinite(float(row[key]))],
            dtype=float,
        )
        if values.size == 0:
            continue
        lo, hi = np.percentile(values, [2.5, 97.5]) if values.size > 1 else (values[0], values[0])
        summary[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=0)),
            "ci95": [float(lo), float(hi)],
            "n": int(values.size),
        }
    return summary


def _mean_metric_row(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {key: float(value["mean"]) for key, value in _numeric_metric_summary(rows).items()}


def _seed_summary_from_runs(run_items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    seeds = sorted({int(item.get("eval_seed", -1)) for item in run_items if int(item.get("eval_seed", -1)) >= 0})
    models = sorted({str(item.get("model_name", "")) for item in run_items if str(item.get("model_name", "")).strip()})
    return {
        "multi_seed_count": int(len(seeds)),
        "eval_seeds": seeds,
        "model_count": int(len(models)),
        "models": models,
        "note": "Multiple seeds repeat trial-level splits to estimate stability; they do not create new EEG samples.",
    }


def _aggregate_classifier_seed_results(seed_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in seed_results:
        model_name = str(item.get("model_name", "")).strip()
        if model_name:
            grouped.setdefault(model_name, []).append(dict(item))
    aggregated: list[dict[str, Any]] = []
    for model_name, rows in grouped.items():
        metric_rows = [dict(row.get("classifier_only_metrics", {}) or {}) for row in rows]
        metric_summary = _numeric_metric_summary(metric_rows)
        metrics_mean = _mean_metric_row(metric_rows)
        best_seed_run = sorted(rows, key=lambda row: _classifier_only_rank_key(dict(row.get("classifier_only_metrics", {}))))[0]
        result = dict(best_seed_run)
        result["seed_level_results"] = rows
        result["seed_summary"] = _seed_summary_from_runs(rows)
        result["metric_summary"] = metric_summary
        result["classifier_only_metrics"] = metrics_mean
        result["metrics"] = metrics_mean
        result["metrics_4class"] = dict(best_seed_run.get("metrics_4class", {}))
        result["eval_seed"] = int(best_seed_run.get("eval_seed", 0) or 0)
        result["best_seed_for_artifacts"] = int(best_seed_run.get("eval_seed", 0) or 0)
        aggregated.append(result)
    aggregated.sort(key=lambda item: _classifier_only_rank_key(dict(item.get("classifier_only_metrics", {}))))
    for rank, item in enumerate(aggregated, start=1):
        item["rank_end_to_end"] = int(rank)
        item["deployment_rank"] = int(rank)
        item["quick_screen_rank"] = int(rank)
    return aggregated


def _source_views(
    result: dict[str, Any],
    *,
    prefer_cross_session: bool,
    decision_time_mode: str,
    async_decision_time_mode: str,
) -> dict[str, Any]:
    if prefer_cross_session and isinstance(result.get("cross_session_metrics"), dict):
        paper = dict(result.get("cross_session_paper_lens_metrics", {}))
        async_lens = dict(result.get("cross_session_async_lens_metrics", {}))
        return {
            "source": "cross_session",
            "metrics": dict(result.get("cross_session_metrics", {})),
            "async_metrics": dict(result.get("cross_session_async_metrics", {})),
            "metrics_4class": dict(result.get("cross_session_metrics_4class", {})),
            "metrics_2class": dict(result.get("cross_session_metrics_2class", {})),
            "metrics_5class": (
                None
                if result.get("cross_session_metrics_5class") is None
                else dict(result.get("cross_session_metrics_5class", {}))
            ),
            "paper_lens_metrics": {
                "metrics_4class": dict(paper.get("metrics_4class", result.get("cross_session_metrics_4class", {}))),
                "metrics_2class": dict(paper.get("metrics_2class", result.get("cross_session_metrics_2class", {}))),
                "metrics_5class": (
                    None
                    if paper.get("metrics_5class", result.get("cross_session_metrics_5class")) is None
                    else dict(paper.get("metrics_5class", result.get("cross_session_metrics_5class", {})))
                ),
                "decision_time_mode": str(decision_time_mode),
            },
            "async_lens_metrics": {
                "metrics_4class": dict(async_lens.get("metrics_4class", result.get("cross_session_metrics_4class", {}))),
                "metrics_2class": dict(async_lens.get("metrics_2class", result.get("cross_session_metrics_2class", {}))),
                "metrics_5class": (
                    None
                    if async_lens.get("metrics_5class", result.get("cross_session_metrics_5class")) is None
                    else dict(async_lens.get("metrics_5class", result.get("cross_session_metrics_5class", {})))
                ),
                "decision_time_mode": str(async_decision_time_mode),
            },
        }

    return {
        "source": "session1_holdout",
        "metrics": dict(result.get("metrics", {})),
        "async_metrics": dict(result.get("async_metrics", {})),
        "metrics_4class": dict(result.get("metrics_4class", {})),
        "metrics_2class": dict(result.get("metrics_2class", {})),
        "metrics_5class": (
            None if result.get("metrics_5class") is None else dict(result.get("metrics_5class", {}))
        ),
        "paper_lens_metrics": {
            "metrics_4class": dict(result.get("paper_lens_metrics_4class", result.get("metrics_4class", {}))),
            "metrics_2class": dict(result.get("paper_lens_metrics_2class", result.get("metrics_2class", {}))),
            "metrics_5class": (
                None
                if result.get("paper_lens_metrics_5class", result.get("metrics_5class")) is None
                else dict(result.get("paper_lens_metrics_5class", result.get("metrics_5class", {})))
            ),
            "decision_time_mode": str(decision_time_mode),
        },
        "async_lens_metrics": {
            "metrics_4class": dict(result.get("async_lens_metrics_4class", result.get("metrics_4class", {}))),
            "metrics_2class": dict(result.get("async_lens_metrics_2class", result.get("metrics_2class", {}))),
            "metrics_5class": (
                None
                if result.get("async_lens_metrics_5class", result.get("metrics_5class")) is None
                else dict(result.get("async_lens_metrics_5class", result.get("metrics_5class", {})))
            ),
            "decision_time_mode": str(async_decision_time_mode),
        },
    }


FBCCA_WEIGHT_MODEL_NAMES = {
    "fbcca_cw_all8",
    "fbcca_sw_all8",
    "fbcca_cw_sw_all8",
    "fbcca_cw_sw_trca_shared",
}
FBCCA_PROFILE_MODEL_NAMES = FBCCA_WEIGHT_MODEL_NAMES | {"fbcca", "legacy_fbcca_202603", "fbcca_plain_all8"}


def _run_key(result: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(result.get("model_name", "")),
        str(result.get("channel_mode", "")),
        int(result.get("eval_seed", 0) or 0),
    )


def _finite_benchmark_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in dict(metrics or {}).items():
        if isinstance(value, (float, int)) and np.isfinite(float(value)):
            payload[str(key)] = float(value)
    return payload


def _profile_for_result(
    profile: Any,
    result: dict[str, Any],
    *,
    prefer_cross_session: bool,
    decision_time_mode: str,
    async_decision_time_mode: str,
) -> Any:
    views = _source_views(
        result,
        prefer_cross_session=prefer_cross_session,
        decision_time_mode=decision_time_mode,
        async_decision_time_mode=async_decision_time_mode,
    )
    try:
        metrics = _finite_benchmark_metrics(dict(views.get("metrics", {})))
        validation_status = {
            "metrics_source": str(views.get("source", "")),
            "meets_acceptance": bool(profile_meets_acceptance(metrics)),
            "idle_fp_per_min": float(metrics.get("idle_fp_per_min", float("inf"))),
            "control_recall": float(metrics.get("control_recall", 0.0)),
        }
        return replace(
            profile,
            benchmark_metrics=metrics,
            profile_validation_status=json_safe(validation_status),
            recommended_for_realtime=bool(validation_status["meets_acceptance"]),
        )
    except TypeError:
        return profile


def _is_fbcca_weight_result(result: dict[str, Any]) -> bool:
    model_name = str(result.get("model_name", "")).strip()
    frontend = dict(result.get("frontend_summary", {}) or {})
    active = dict(frontend.get("active_frontend", {}) or {})
    channel_mode = str(
        active.get("channel_weight_mode")
        or frontend.get("channel_weight_mode")
        or result.get("profile_channel_weight_mode")
        or ""
    ).strip().lower()
    subband_mode = str(
        active.get("subband_weight_mode")
        or frontend.get("subband_weight_mode")
        or result.get("profile_subband_weight_mode")
        or ""
    ).strip().lower()
    has_channel_weights = bool(frontend.get("channel_weights") or result.get("profile_channel_weights"))
    has_subband_weights = bool(frontend.get("subband_weights") or result.get("profile_subband_weights"))
    return (
        model_name in FBCCA_WEIGHT_MODEL_NAMES
        or channel_mode == "fbcca_diag"
        or has_channel_weights
        or (has_subband_weights and subband_mode not in {"", "none", "chen_fixed"})
        or subband_mode not in {"", "none", "chen_fixed"}
    )


def _is_fbcca_profile_model(model_name: str) -> bool:
    normalized = normalize_model_name(model_name)
    return normalized in FBCCA_PROFILE_MODEL_NAMES or normalized.startswith("fbcca")


def _profile_frontend_weight_flags(profile: Any) -> dict[str, Any]:
    profile_model_name = str(getattr(profile, "model_name", "") or "")
    channel_weights = getattr(profile, "channel_weights", None)
    subband_weights = getattr(profile, "subband_weights", None)
    subband_weight_mode = str(getattr(profile, "subband_weight_mode", "") or "").strip().lower()
    has_channel_weights = isinstance(channel_weights, (list, tuple)) and len(channel_weights) > 0
    has_subband_weights = isinstance(subband_weights, (list, tuple)) and len(subband_weights) > 0
    has_weighted_subband_mode = subband_weight_mode not in {"", "none", "chen_fixed"}
    return {
        "profile_model_name": profile_model_name,
        "is_fbcca_family": bool(_is_fbcca_profile_model(profile_model_name)),
        "has_channel_weights": bool(has_channel_weights),
        "has_subband_weights": bool(has_subband_weights),
        "has_weighted_subband_mode": bool(has_weighted_subband_mode),
        "is_weighted_fbcca_profile": bool(
            _is_fbcca_profile_model(profile_model_name)
            and (has_channel_weights or has_weighted_subband_mode)
        ),
    }


def _aggregate_weight_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    mode: str,
    kind: str,
) -> Optional[list[float]]:
    clean_rows: list[np.ndarray] = []
    for row in vectors:
        array = np.asarray(list(row), dtype=float).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)):
            continue
        clean_rows.append(array)
    if not clean_rows:
        return None
    lengths = {int(row.size) for row in clean_rows}
    if len(lengths) != 1:
        return None
    stack = np.vstack(clean_rows)
    resolved_mode = str(mode).strip().lower()
    if resolved_mode == "mean":
        aggregate = np.mean(stack, axis=0)
    elif resolved_mode == "trimmed-mean" and int(stack.shape[0]) >= 3:
        trim = max(1, int(math.floor(stack.shape[0] * 0.2)))
        if trim * 2 >= int(stack.shape[0]):
            aggregate = np.mean(stack, axis=0)
        else:
            ordered = np.sort(stack, axis=0)
            aggregate = np.mean(ordered[trim:-trim], axis=0)
    else:
        aggregate = np.median(stack, axis=0)
    if kind == "channel":
        aggregate = np.clip(aggregate, 0.5, 1.8)
        mean_value = float(np.mean(aggregate))
        if mean_value > 0.0:
            aggregate = aggregate / mean_value
    elif kind == "subband":
        aggregate = np.clip(aggregate, 0.0, None)
        sum_value = float(np.sum(aggregate))
        if sum_value > 0.0:
            aggregate = aggregate / sum_value
    return [float(value) for value in aggregate.tolist()]


def _build_aggregated_fbcca_profile(
    *,
    weighted_runs: Sequence[dict[str, Any]],
    profile_by_run: dict[tuple[str, str, int], Any],
    prefer_cross_session: bool,
    decision_time_mode: str,
    async_decision_time_mode: str,
    aggregation_mode: str,
) -> tuple[Optional[Any], Optional[dict[str, Any]]]:
    if not weighted_runs:
        return None, None
    representative_run = sorted(
        (dict(item) for item in weighted_runs),
        key=lambda item: benchmark_rank_key(
            _ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session)
        ),
    )[0]
    representative_profile = profile_by_run.get(_run_key(representative_run))
    if representative_profile is None:
        return None, None
    channel_vectors = [
        list(profile.channel_weights)
        for item in weighted_runs
        for profile in [profile_by_run.get(_run_key(item))]
        if profile is not None and isinstance(getattr(profile, "channel_weights", None), tuple)
    ]
    subband_vectors = [
        list(profile.subband_weights)
        for item in weighted_runs
        for profile in [profile_by_run.get(_run_key(item))]
        if profile is not None and isinstance(getattr(profile, "subband_weights", None), tuple)
    ]
    aggregated_channel_weights = _aggregate_weight_vectors(
        channel_vectors,
        mode=aggregation_mode,
        kind="channel",
    )
    aggregated_subband_weights = _aggregate_weight_vectors(
        subband_vectors,
        mode=aggregation_mode,
        kind="subband",
    )
    if aggregated_channel_weights is None and aggregated_subband_weights is None:
        return None, None
    views = _source_views(
        representative_run,
        prefer_cross_session=prefer_cross_session,
        decision_time_mode=decision_time_mode,
        async_decision_time_mode=async_decision_time_mode,
    )
    metrics = _finite_benchmark_metrics(dict(views.get("metrics", {})))
    validation_status = {
        "metrics_source": str(views.get("source", "")),
        "meets_acceptance": bool(profile_meets_acceptance(metrics)),
        "idle_fp_per_min": float(metrics.get("idle_fp_per_min", float("inf"))),
        "control_recall": float(metrics.get("control_recall", 0.0)),
        "threshold_source": "representative_seed_profile",
    }
    seed_summary = {
        "aggregation_mode": str(aggregation_mode),
        "aggregated_from_runs": int(len(weighted_runs)),
        "eval_seeds": sorted({int(item.get("eval_seed", 0) or 0) for item in weighted_runs}),
        "representative_run": {
            "model_name": str(representative_run.get("model_name", "")),
            "channel_mode": str(representative_run.get("channel_mode", "")),
            "eval_seed": int(representative_run.get("eval_seed", 0) or 0),
        },
    }
    joint_training = dict(getattr(representative_profile, "joint_weight_training", {}) or {})
    joint_training["aggregation_mode"] = str(aggregation_mode)
    joint_training["aggregated_run_count"] = int(len(weighted_runs))
    aggregated_profile = replace(
        representative_profile,
        channel_weights=(
            None if aggregated_channel_weights is None else tuple(float(value) for value in aggregated_channel_weights)
        ),
        subband_weights=(
            None if aggregated_subband_weights is None else tuple(float(value) for value in aggregated_subband_weights)
        ),
        weight_training_seed_summary=json_safe(seed_summary),
        benchmark_metrics=metrics,
        joint_weight_training=json_safe(joint_training),
        profile_validation_status=json_safe(validation_status),
        recommended_for_realtime=bool(validation_status["meets_acceptance"]),
    )
    summary = {
        "aggregation_mode": str(aggregation_mode),
        "aggregated_from_runs": int(len(weighted_runs)),
        "weighted_model_family": str(representative_run.get("model_name", "")),
        "channel_weight_count": 0 if aggregated_channel_weights is None else int(len(aggregated_channel_weights)),
        "subband_weight_count": 0 if aggregated_subband_weights is None else int(len(aggregated_subband_weights)),
        "representative_eval_seed": int(representative_run.get("eval_seed", 0) or 0),
    }
    return aggregated_profile, summary


def _weight_table_row(
    result: dict[str, Any],
    *,
    prefer_cross_session: bool,
) -> dict[str, Any]:
    metrics = _ranking_metrics_from_result(result, prefer_cross_session=prefer_cross_session)
    metrics_4 = dict(result.get("metrics_4class", {}) or {})
    classifier_metrics = dict(result.get("classifier_only_metrics", {}) or {})
    if prefer_cross_session and isinstance(result.get("cross_session_metrics_4class"), dict):
        metrics_4 = dict(result.get("cross_session_metrics_4class", {}) or {})
    frontend = dict(result.get("frontend_summary", {}) or {})
    channel_weights = list(frontend.get("channel_weights", []) or [])
    subband_weights = list(frontend.get("subband_weights", []) or [])
    model_name = str(result.get("model_name", ""))
    if model_name == "fbcca_fixed_all8":
        weight_strategy = "plain_fbcca_all8"
    elif model_name == "fbcca_cw_all8":
        weight_strategy = "channel_weighted"
    elif model_name == "fbcca_sw_all8":
        weight_strategy = "subband_weighted"
    elif model_name == "fbcca_cw_sw_all8":
        weight_strategy = "separable_channel_x_subband"
    elif model_name == "fbcca_cw_sw_trca_shared":
        weight_strategy = "separable_channel_x_subband_plus_trca_shared"
    else:
        weight_strategy = "profile_or_config_weighted"
    return {
        "rank": int(result.get("deployment_rank", 0) or 0),
        "model_name": model_name,
        "weight_strategy": weight_strategy,
        "separable_weighting": bool(channel_weights or subband_weights),
        "channel_mode": str(result.get("channel_mode", "")),
        "eval_seed": int(result.get("eval_seed", 0) or 0),
        "meets_acceptance": bool(
            profile_meets_acceptance(_ranking_metrics_from_result(result, prefer_cross_session=prefer_cross_session))
        ),
        "metrics": dict(metrics),
        "acc_4class": float(metrics_4.get("acc", classifier_metrics.get("acc", 0.0)) or 0.0),
        "macro_f1_4class": float(metrics_4.get("macro_f1", classifier_metrics.get("macro_f1", 0.0)) or 0.0),
        "itr_bpm_4class": float(metrics_4.get("itr_bpm", metrics.get("itr_bpm", 0.0)) or 0.0),
        "mean_decision_time_s": float(metrics_4.get("mean_decision_time_s", 0.0) or 0.0),
        "channel_weight_mode": frontend.get("channel_weight_mode"),
        "channel_weights": channel_weights,
        "channel_weight_stats": frontend.get("channel_weight_stats"),
        "subband_weight_mode": frontend.get("subband_weight_mode"),
        "subband_weights": subband_weights,
        "spatial_filter_mode": frontend.get("spatial_filter_mode"),
        "spatial_filter_rank": frontend.get("spatial_filter_rank"),
        "joint_weight_training": frontend.get("joint_weight_training"),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(Path(path), json_dumps(json_safe(payload)) + "\n")


def _write_text_atomic(path: Path, body: str) -> None:
    atomic_write_text(Path(path), str(body), encoding="utf-8")


def _profile_channel_positions(profile: Any, dataset_channels: Sequence[int]) -> tuple[tuple[int, ...], list[int]]:
    available = [int(channel) for channel in dataset_channels]
    profile_channels_raw = getattr(profile, "eeg_channels", None)
    profile_channels = (
        tuple(int(channel) for channel in profile_channels_raw)
        if profile_channels_raw is not None
        else tuple(available)
    )
    missing = [int(channel) for channel in profile_channels if int(channel) not in available]
    if missing:
        raise ValueError(
            "pretrained profile EEG channels are not present in dataset channels: "
            f"missing={missing}, dataset_channels={available}"
        )
    channel_weights = getattr(profile, "channel_weights", None)
    if channel_weights is not None and len(channel_weights) != len(profile_channels):
        raise ValueError(
            "pretrained profile channel_weights length mismatch: "
            f"weights={len(channel_weights)} eeg_channels={len(profile_channels)}"
        )
    positions = [available.index(int(channel)) for channel in profile_channels]
    return profile_channels, positions


def _evaluate_pretrained_profile_result(
    *,
    profile_path: Path,
    eval_segments: Sequence[tuple[Any, np.ndarray]],
    dataset_channels: Sequence[int],
    sampling_rate: int,
    metric_scope: str,
    decision_time_mode: str,
    async_decision_time_mode: str,
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
    gpu_warmup: bool,
    gpu_cache_policy: str,
    prefer_cross_session: bool,
) -> tuple[Any, dict[str, Any]]:
    profile = load_profile(Path(profile_path), require_exists=True)
    weight_flags = _profile_frontend_weight_flags(profile)
    if not bool(weight_flags.get("is_fbcca_family", False)):
        raise ValueError(
            "profile-eval requires an FBCCA-family profile; "
            f"got model_name={weight_flags.get('profile_model_name', '')}"
        )
    if not bool(weight_flags.get("is_weighted_fbcca_profile", False)):
        raise ValueError(
            "profile-eval requires a weighted FBCCA profile with channel_weights or subband_weights; "
            f"got model_name={weight_flags.get('profile_model_name', '')}"
        )
    profile_channels, positions = _profile_channel_positions(profile, dataset_channels)
    eval_subset = _subset_trial_segments_by_positions(eval_segments, positions)
    decoder = load_decoder_from_profile(
        profile,
        sampling_rate=int(sampling_rate),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=bool(gpu_warmup),
        gpu_cache_policy=str(gpu_cache_policy),
    )
    bundle = evaluate_decoder_on_trials_v2(
        decoder,
        profile,
        eval_subset,
        metric_scope=metric_scope,
        paper_decision_time_mode=decision_time_mode,
        async_decision_time_mode=async_decision_time_mode,
    )
    metrics = pack_evaluation_metrics_for_ranking(bundle, metric_scope=metric_scope)
    result: dict[str, Any] = {
        "model_name": "fbcca_profile_weighted",
        "profile_model_name": str(getattr(profile, "model_name", "")),
        "implementation_level": "engineering-approx",
        "method_note": "Loaded frozen FBCCA profile weights; no weight retraining during profile-eval.",
        "metrics": dict(metrics),
        "async_metrics": dict(bundle.get("async_metrics", {})),
        "metrics_4class": dict(bundle.get("metrics_4class", {})),
        "metrics_2class": dict(bundle.get("metrics_2class", {})),
        "metrics_5class": (
            None if bundle.get("metrics_5class") is None else dict(bundle.get("metrics_5class", {}))
        ),
        "paper_lens_metrics_4class": dict(bundle.get("paper_lens_metrics_4class", bundle.get("metrics_4class", {}))),
        "paper_lens_metrics_2class": dict(bundle.get("paper_lens_metrics_2class", bundle.get("metrics_2class", {}))),
        "paper_lens_metrics_5class": (
            None
            if bundle.get("paper_lens_metrics_5class", bundle.get("metrics_5class")) is None
            else dict(bundle.get("paper_lens_metrics_5class", bundle.get("metrics_5class", {})))
        ),
        "async_lens_metrics_4class": dict(bundle.get("async_lens_metrics_4class", bundle.get("metrics_4class", {}))),
        "async_lens_metrics_2class": dict(bundle.get("async_lens_metrics_2class", bundle.get("metrics_2class", {}))),
        "async_lens_metrics_5class": (
            None
            if bundle.get("async_lens_metrics_5class", bundle.get("metrics_5class")) is None
            else dict(bundle.get("async_lens_metrics_5class", bundle.get("metrics_5class", {})))
        ),
        "classifier_only_metrics": dict(bundle.get("metrics_4class", {})),
        "fixed_window_metrics": dict(metrics),
        "fixed_window_async_metrics": dict(bundle.get("async_metrics", {})),
        "fixed_window_metrics_4class": dict(bundle.get("metrics_4class", {})),
        "fixed_window_metrics_2class": dict(bundle.get("metrics_2class", {})),
        "fixed_window_metrics_5class": (
            None if bundle.get("metrics_5class") is None else dict(bundle.get("metrics_5class", {}))
        ),
        "trial_events": list(bundle.get("trial_events", [])),
        "selected_eeg_channels": [int(channel) for channel in profile_channels],
        "channel_mode": "profile",
        "eval_seed": 0,
        "ranking_source": "cross_session" if prefer_cross_session else "session1_holdout",
        "meets_acceptance": bool(profile_meets_acceptance(metrics)),
        "profile_path": str(Path(profile_path).expanduser().resolve()),
        "profile_weighted_valid": bool(weight_flags.get("is_weighted_fbcca_profile", False)),
        "pretrained_profile_has_channel_weights": bool(weight_flags.get("has_channel_weights", False)),
        "pretrained_profile_has_subband_weights": bool(weight_flags.get("has_subband_weights", False)),
        "profile_channel_weight_mode": (
            None if getattr(profile, "channel_weight_mode", None) is None else str(profile.channel_weight_mode)
        ),
        "profile_channel_weights": (
            None
            if getattr(profile, "channel_weights", None) is None
            else [float(value) for value in profile.channel_weights]
        ),
        "profile_subband_weight_mode": (
            None if getattr(profile, "subband_weight_mode", None) is None else str(profile.subband_weight_mode)
        ),
        "profile_subband_weights": (
            None
            if getattr(profile, "subband_weights", None) is None
            else [float(value) for value in profile.subband_weights]
        ),
        "profile_subband_weight_params": (
            None
            if getattr(profile, "subband_weight_params", None) is None
            else json_safe(dict(profile.subband_weight_params))
        ),
        "profile_spatial_filter_mode": (
            None if getattr(profile, "spatial_filter_mode", None) is None else str(profile.spatial_filter_mode)
        ),
        "profile_spatial_filter_rank": (
            None if getattr(profile, "spatial_filter_rank", None) is None else int(profile.spatial_filter_rank)
        ),
        "profile_joint_weight_training": (
            None
            if getattr(profile, "joint_weight_training", None) is None
            else json_safe(dict(profile.joint_weight_training))
        ),
    }
    result = _normalize_frontend_result_fields(result)
    result["frontend_summary"] = _build_frontend_summary(result)
    if prefer_cross_session:
        result["cross_session_metrics"] = dict(metrics)
        result["cross_session_async_metrics"] = dict(bundle.get("async_metrics", {}))
        result["cross_session_metrics_4class"] = dict(bundle.get("metrics_4class", {}))
        result["cross_session_metrics_2class"] = dict(bundle.get("metrics_2class", {}))
        result["cross_session_metrics_5class"] = (
            None if bundle.get("metrics_5class") is None else dict(bundle.get("metrics_5class", {}))
        )
        result["cross_session_paper_lens_metrics_4class"] = dict(result.get("paper_lens_metrics_4class", {}))
        result["cross_session_paper_lens_metrics_2class"] = dict(result.get("paper_lens_metrics_2class", {}))
        result["cross_session_paper_lens_metrics_5class"] = result.get("paper_lens_metrics_5class")
        result["cross_session_async_lens_metrics_4class"] = dict(result.get("async_lens_metrics_4class", {}))
        result["cross_session_async_lens_metrics_2class"] = dict(result.get("async_lens_metrics_2class", {}))
        result["cross_session_async_lens_metrics_5class"] = result.get("async_lens_metrics_5class")
    return profile, result


def _build_roundtrip_window(*, win_samples: int, freqs: Sequence[float], channels: int, fs: int) -> np.ndarray:
    t = np.arange(int(win_samples), dtype=np.float64) / float(max(int(fs), 1))
    channel_count = max(1, int(channels))
    matrix = np.zeros((int(win_samples), channel_count), dtype=np.float64)
    for channel_index in range(channel_count):
        freq = float(freqs[channel_index % max(len(freqs), 1)])
        matrix[:, channel_index] = (
            np.sin(2.0 * np.pi * freq * t + channel_index * 0.17)
            + 0.15 * np.cos(2.0 * np.pi * (freq * 2.0) * t)
        )
    return np.asarray(matrix, dtype=np.float64)


def _run_profile_roundtrip_check(
    *,
    saved_profile_path: Path,
    source_profile: Any,
    sampling_rate: int,
    gpu_device: int,
    gpu_precision: str,
    gpu_warmup: bool,
    gpu_cache_policy: str,
) -> dict[str, Any]:
    saved_profile = load_profile(Path(saved_profile_path), require_exists=True)
    channel_count = (
        len(saved_profile.eeg_channels)
        if getattr(saved_profile, "eeg_channels", None) is not None
        else len(saved_profile.channel_weights)
        if getattr(saved_profile, "channel_weights", None) is not None
        else 8
    )
    source_decoder = load_decoder_from_profile(
        source_profile,
        sampling_rate=int(sampling_rate),
        compute_backend="cpu",
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=bool(gpu_warmup),
        gpu_cache_policy=str(gpu_cache_policy),
    )
    saved_decoder = load_decoder_from_profile(
        saved_profile,
        sampling_rate=int(sampling_rate),
        compute_backend="cpu",
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=bool(gpu_warmup),
        gpu_cache_policy=str(gpu_cache_policy),
    )
    probe_window = _build_roundtrip_window(
        win_samples=int(source_decoder.win_samples),
        freqs=tuple(float(value) for value in saved_profile.freqs),
        channels=int(channel_count),
        fs=int(sampling_rate),
    )
    source_analysis = dict(source_decoder.analyze_window(probe_window))
    saved_analysis = dict(saved_decoder.analyze_window(probe_window))
    source_gate = AsyncDecisionGate.from_profile(source_profile)
    saved_gate = AsyncDecisionGate.from_profile(saved_profile)
    source_decision = dict(source_gate.update(source_analysis))
    saved_decision = dict(saved_gate.update(saved_analysis))
    score_delta = {
        "top1_score_abs": abs(float(source_analysis.get("top1_score", 0.0)) - float(saved_analysis.get("top1_score", 0.0))),
        "top2_score_abs": abs(float(source_analysis.get("top2_score", 0.0)) - float(saved_analysis.get("top2_score", 0.0))),
        "margin_abs": abs(float(source_analysis.get("margin", 0.0)) - float(saved_analysis.get("margin", 0.0))),
        "ratio_abs": abs(float(source_analysis.get("ratio", 0.0)) - float(saved_analysis.get("ratio", 0.0))),
    }
    return {
        "saved_profile_path": str(Path(saved_profile_path).expanduser().resolve()),
        "ready": bool(
            source_analysis.get("pred_freq") == saved_analysis.get("pred_freq")
            and source_decision.get("selected_freq") == saved_decision.get("selected_freq")
            and max(score_delta.values(), default=0.0) <= 1e-6
        ),
        "channel_count": int(channel_count),
        "source_pred_freq": source_analysis.get("pred_freq"),
        "saved_pred_freq": saved_analysis.get("pred_freq"),
        "source_selected_freq": source_decision.get("selected_freq"),
        "saved_selected_freq": saved_decision.get("selected_freq"),
        "score_delta": score_delta,
    }


def _model_compare_row(
    result: dict[str, Any],
    *,
    prefer_cross_session: bool,
) -> dict[str, Any]:
    metrics = dict(_ranking_metrics_from_result(result, prefer_cross_session=prefer_cross_session))
    metrics_4 = dict(result.get("metrics_4class", {}) or {})
    metrics_2 = dict(result.get("metrics_2class", {}) or {})
    classifier_metrics = dict(result.get("classifier_only_metrics", {}) or {})
    if prefer_cross_session and isinstance(result.get("cross_session_metrics_4class"), dict):
        metrics_4 = dict(result.get("cross_session_metrics_4class", {}) or {})
        metrics_2 = dict(result.get("cross_session_metrics_2class", {}) or {})
    return {
        "rank": int(result.get("rank_end_to_end", result.get("deployment_rank", 0)) or 0),
        "model_name": str(result.get("model_name", "")),
        "implementation_level": str(result.get("implementation_level", "")),
        "channel_mode": str(result.get("channel_mode", "")),
        "eval_seed": int(result.get("eval_seed", 0) or 0),
        "meets_acceptance": bool(
            profile_meets_acceptance(_ranking_metrics_from_result(result, prefer_cross_session=prefer_cross_session))
        ),
        "acc_4class": float(metrics_4.get("acc", classifier_metrics.get("acc", 0.0)) or 0.0),
        "macro_f1_4class": float(metrics_4.get("macro_f1", classifier_metrics.get("macro_f1", 0.0)) or 0.0),
        "itr_bpm_4class": float(metrics_4.get("itr_bpm", metrics.get("itr_bpm", 0.0)) or 0.0),
        "mean_decision_time_s": float(metrics_4.get("mean_decision_time_s", 0.0) or 0.0),
        "acc_2class": float(metrics_2.get("acc", 0.0) or 0.0),
        "macro_f1_2class": float(metrics_2.get("macro_f1", 0.0) or 0.0),
        "idle_fp_per_min": float(metrics.get("idle_fp_per_min", float("inf"))),
        "control_recall": float(metrics.get("control_recall", 0.0) or 0.0),
        "switch_latency_s": float(metrics.get("switch_latency_s", float("inf"))),
        "release_latency_s": float(metrics.get("release_latency_s", float("inf"))),
        "inference_ms": float(metrics.get("inference_ms", float("inf"))),
    }


def _weight_definition_notes() -> dict[str, str]:
    return {
        "channel_weights": (
            "Eight-channel diagonal EEG weights for all8 FBCCA. The realtime window is transformed as "
            "X_weighted[:, c] = X[:, c] * channel_weights[c] before FBCCA scoring; len(w) must match "
            "the realtime EEG channel count."
        ),
        "subband_weights": (
            "Five global filter-bank fusion weights shared by all EEG channels. FBCCA combines scores as "
            "score[f] = sum_b subband_weights[b] * rho[b, f]^2. These weights are not per-channel weights "
            "and they are not learned cutoff frequencies."
        ),
        "separable_weighting": (
            "The default trainable FBCCA frontend is separable: channel_weights[8] and subband_weights[5] "
            "are learned separately and applied as channel scaling plus global subband-score fusion. "
            "A full 8x5 channel-by-subband matrix is intentionally not trained by default to reduce overfitting."
        ),
        "spatial_filter_state": (
            "Optional TRCA/shared spatial frontend state. It is a separate spatial projection and is not mixed "
            "into the pure FBCCA channel/subband-weight mainline unless the chosen profile enables it."
        ),
    }


def _summarize_joint_weight_training(metadata: Any) -> Optional[dict[str, Any]]:
    if not isinstance(metadata, dict) or not metadata:
        return None
    initial_weights = metadata.get("initial_channel_weights", metadata.get("initial_weights"))
    optimized_weights = metadata.get("optimized_channel_weights", metadata.get("optimized_weights"))
    initial_subband_weights = metadata.get("initial_subband_weights")
    optimized_subband_weights = metadata.get("optimized_subband_weights")
    iterations = list(metadata.get("iterations", [])) if isinstance(metadata.get("iterations"), list) else []

    def _stats(values: Any) -> Optional[dict[str, float]]:
        if not isinstance(values, (list, tuple)) or not values:
            return None
        arr = np.asarray([float(v) for v in values], dtype=float)
        if arr.size == 0:
            return None
        return {
            "count": int(arr.size),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    return {
        "mode": str(metadata.get("mode", "")),
        "objective": [float(value) for value in list(metadata.get("objective", []))],
        "subband_weight_mode": (
            None if metadata.get("subband_weight_mode") is None else str(metadata.get("subband_weight_mode"))
        ),
        "joint_weight_iters": int(metadata.get("joint_weight_iters", 0) or 0),
        "weight_cv_folds": int(metadata.get("weight_cv_folds", 0) or 0),
        "iteration_count": int(len(iterations)),
        "improved_iteration_count": int(
            sum(
                1
                for row in iterations
                if isinstance(row, dict)
                and (
                    bool(row.get("improved_by_weight", False))
                    or bool(row.get("channel_improved", False))
                    or bool(row.get("subband_improved", False))
                    or bool(row.get("basis_updated", False))
                )
            )
        ),
        "spatial_filter_mode": (
            None if metadata.get("spatial_filter_mode") is None else str(metadata.get("spatial_filter_mode"))
        ),
        "spatial_source_model": (
            None if metadata.get("spatial_source_model") is None else str(metadata.get("spatial_source_model"))
        ),
        "selected_spatial_rank": (
            None if metadata.get("selected_spatial_rank") is None else int(metadata.get("selected_spatial_rank"))
        ),
        "initial_weights": (
            [float(value) for value in initial_weights] if isinstance(initial_weights, (list, tuple)) else None
        ),
        "optimized_weights": (
            [float(value) for value in optimized_weights] if isinstance(optimized_weights, (list, tuple)) else None
        ),
        "initial_subband_weights": (
            [float(value) for value in initial_subband_weights]
            if isinstance(initial_subband_weights, (list, tuple))
            else None
        ),
        "optimized_subband_weights": (
            [float(value) for value in optimized_subband_weights]
            if isinstance(optimized_subband_weights, (list, tuple))
            else None
        ),
        "optimized_subband_params": (
            None
            if metadata.get("optimized_subband_params") is None
            else dict(metadata.get("optimized_subband_params"))
        ),
        "initial_weight_stats": _stats(initial_weights),
        "optimized_weight_stats": _stats(optimized_weights),
        "initial_subband_weight_stats": _stats(initial_subband_weights),
        "optimized_subband_weight_stats": _stats(optimized_subband_weights),
        "gate_metrics": (
            {
                str(key): float(value)
                for key, value in dict(
                    metadata.get("gate_metrics", metadata.get("gate_cv_metrics", {}))
                ).items()
                if isinstance(value, (int, float))
            }
            if isinstance(metadata.get("gate_metrics"), dict) or isinstance(metadata.get("gate_cv_metrics"), dict)
            else {}
        ),
    }


def _build_frontend_summary(result: dict[str, Any]) -> dict[str, Any]:
    configured_channel_mode = (
        None
        if result.get("configured_channel_weight_mode") is None
        else str(result.get("configured_channel_weight_mode"))
    )
    configured_subband_mode = (
        None
        if result.get("configured_subband_weight_mode") is None
        else str(result.get("configured_subband_weight_mode"))
    )
    configured_spatial_mode = (
        None
        if result.get("configured_spatial_filter_mode") is None
        else str(result.get("configured_spatial_filter_mode"))
    )
    active_channel_mode = (
        None
        if result.get("active_channel_weight_mode") is None
        else str(result.get("active_channel_weight_mode"))
    )
    active_subband_mode = (
        None
        if result.get("active_subband_weight_mode") is None
        else str(result.get("active_subband_weight_mode"))
    )
    active_spatial_mode = (
        None
        if result.get("active_spatial_filter_mode") is None
        else str(result.get("active_spatial_filter_mode"))
    )
    active_spatial_rank = (
        None
        if result.get("active_spatial_filter_rank") is None
        else int(result.get("active_spatial_filter_rank"))
    )
    channel_weights_raw = result.get("profile_channel_weights")
    channel_weights = (
        [float(value) for value in channel_weights_raw]
        if isinstance(channel_weights_raw, (list, tuple))
        else []
    )
    subband_weights_raw = result.get("profile_subband_weights", result.get("runtime_subband_weights"))
    subband_weights = (
        [float(value) for value in subband_weights_raw]
        if isinstance(subband_weights_raw, (list, tuple))
        else []
    )
    selected_channels = [
        int(value) for value in list(result.get("selected_eeg_channels", [])) if isinstance(value, (int, float))
    ]
    weight_pairs = []
    if channel_weights and len(channel_weights) == len(selected_channels):
        weight_pairs = [
            {"board_channel": int(ch), "weight": float(weight)}
            for ch, weight in sorted(
                zip(selected_channels, channel_weights),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ]
    arr = np.asarray(channel_weights, dtype=float) if channel_weights else np.asarray([], dtype=float)
    summary: dict[str, Any] = {
        "channel_weight_mode": active_channel_mode,
        "subband_weight_mode": active_subband_mode,
        "spatial_filter_mode": active_spatial_mode,
        "spatial_filter_rank": active_spatial_rank,
        "configured_frontend": {
            "channel_weight_mode": configured_channel_mode,
            "subband_weight_mode": configured_subband_mode,
            "spatial_filter_mode": configured_spatial_mode,
            "spatial_rank_candidates": list(result.get("configured_spatial_rank_candidates", []) or []),
            "joint_weight_iters": (
                None
                if result.get("configured_joint_weight_iters") is None
                else int(result.get("configured_joint_weight_iters"))
            ),
            "spatial_source_model": (
                None
                if result.get("configured_spatial_source_model") is None
                else str(result.get("configured_spatial_source_model"))
            ),
        },
        "active_frontend": {
            "channel_weight_mode": active_channel_mode,
            "subband_weight_mode": active_subband_mode,
            "spatial_filter_mode": active_spatial_mode,
            "spatial_filter_rank": active_spatial_rank,
        },
        "channel_weights": list(channel_weights),
        "subband_weights": list(subband_weights),
        "top_weight_channels": weight_pairs[: min(4, len(weight_pairs))],
        "joint_weight_training": _summarize_joint_weight_training(
            result.get("profile_joint_weight_training")
            if result.get("profile_joint_weight_training") is not None
            else result.get("channel_weight_training")
        ),
    }
    if arr.size > 0:
        summary["channel_weight_stats"] = {
            "count": int(arr.size),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    else:
        summary["channel_weight_stats"] = None
    return summary


def _normalize_frontend_result_fields(result: dict[str, Any]) -> dict[str, Any]:
    configured_channel_mode = result.get("configured_channel_weight_mode", result.get("channel_weight_mode"))
    configured_subband_mode = result.get("configured_subband_weight_mode", result.get("subband_weight_mode"))
    configured_spatial_mode = result.get("configured_spatial_filter_mode", result.get("spatial_filter_mode"))
    configured_spatial_rank_candidates = result.get(
        "configured_spatial_rank_candidates",
        result.get("spatial_rank_candidates"),
    )
    configured_joint_weight_iters = result.get(
        "configured_joint_weight_iters",
        result.get("joint_weight_iters"),
    )
    configured_spatial_source_model = result.get(
        "configured_spatial_source_model",
        result.get("spatial_source_model"),
    )

    active_channel_mode = result.get("profile_channel_weight_mode")
    if active_channel_mode is None:
        active_channel_mode = result.get("runtime_channel_weight_mode")
    active_subband_mode = result.get("profile_subband_weight_mode")
    if active_subband_mode is None:
        active_subband_mode = result.get("runtime_subband_weight_mode")
    active_spatial_mode = result.get("profile_spatial_filter_mode")
    if active_spatial_mode is None:
        active_spatial_mode = result.get("runtime_spatial_filter_mode")
    active_spatial_rank = result.get("profile_spatial_filter_rank")
    if active_spatial_rank is None:
        active_spatial_rank = result.get("runtime_spatial_filter_rank")

    result["configured_channel_weight_mode"] = (
        None if configured_channel_mode is None else str(configured_channel_mode)
    )
    result["configured_subband_weight_mode"] = (
        None if configured_subband_mode is None else str(configured_subband_mode)
    )
    result["configured_spatial_filter_mode"] = (
        None if configured_spatial_mode is None else str(configured_spatial_mode)
    )
    result["configured_spatial_rank_candidates"] = [
        int(value) for value in list(configured_spatial_rank_candidates or []) if isinstance(value, (int, float))
    ]
    result["configured_joint_weight_iters"] = (
        None if configured_joint_weight_iters is None else int(configured_joint_weight_iters)
    )
    result["configured_spatial_source_model"] = (
        None if configured_spatial_source_model is None else str(configured_spatial_source_model)
    )
    result["active_channel_weight_mode"] = None if active_channel_mode is None else str(active_channel_mode)
    result["active_subband_weight_mode"] = None if active_subband_mode is None else str(active_subband_mode)
    result["active_spatial_filter_mode"] = None if active_spatial_mode is None else str(active_spatial_mode)
    result["active_spatial_filter_rank"] = None if active_spatial_rank is None else int(active_spatial_rank)
    result["channel_weight_mode"] = result.get("active_channel_weight_mode")
    result["subband_weight_mode"] = result.get("active_subband_weight_mode")
    result["spatial_filter_mode"] = result.get("active_spatial_filter_mode")
    return result


def _named_report_stem(
    *,
    subject_id: str,
    recommended_model: str,
    deployed_model: str,
    channel_mode: str,
    eval_seed: int,
) -> str:
    _ = subject_id
    rec = _sanitize_path_token(recommended_model, fallback="model")[:18]
    dep = _sanitize_path_token(deployed_model, fallback="model")[:18]
    mode = _sanitize_path_token(channel_mode, fallback="mode")[:12]
    return (
        "summary"
        f"__rec-{rec}"
        f"__dep-{dep}"
        f"__m-{mode}"
        f"__s-{int(eval_seed)}"
    )


def run_offline_train_eval(
    config: OfflineTrainEvalConfig,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_fn: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    sink_log = log_fn if log_fn is not None else (lambda _msg: None)
    sink_progress = progress_fn if progress_fn is not None else (lambda _payload: None)
    run_log_lines: list[str] = []
    run_log_path: Optional[Path] = None
    progress_snapshot_path: Optional[Path] = None
    progress_started_at = time.perf_counter()
    progress_last_emit = 0.0
    heartbeat_sec = max(0.5, float(config.progress_heartbeat_sec))
    progress_state: dict[str, Any] = {
        "stage": "prepare",
        "model_name": "",
        "run_index": 0,
        "run_total": 0,
        "config_index": 0,
        "config_total": 0,
        "elapsed_s": 0.0,
        "eta_s": None,
        "last_message": "",
    }

    model_start_re = re.compile(
        r"^Model start: (?P<run>\d+)/(?P<total>\d+) mode=(?P<mode>\S+) seed=(?P<seed>\d+) model=(?P<model>\S+)$"
    )
    config_start_re = re.compile(r"^Config start: model=(?P<model>\S+) (?P<run>\d+)/(?P<total>\d+) ")
    stage_a_model_start_re = re.compile(r"^Stage A model start: (?P<run>\d+)/(?P<total>\d+) model=(?P<model>\S+)$")
    stage_a_config_start_re = re.compile(r"^Stage A config start: model=(?P<model>\S+) (?P<run>\d+)/(?P<total>\d+) ")

    def emit_progress(*, force: bool = False, **updates: Any) -> None:
        nonlocal progress_last_emit
        for key, value in updates.items():
            progress_state[key] = value
        elapsed_s = max(time.perf_counter() - progress_started_at, 0.0)
        progress_state["elapsed_s"] = float(elapsed_s)
        run_index = int(progress_state.get("run_index", 0) or 0)
        run_total = int(progress_state.get("run_total", 0) or 0)
        if run_total > 0 and run_index > 0:
            progress_state["eta_s"] = float(max((elapsed_s / run_index) * max(run_total - run_index, 0), 0.0))
        else:
            progress_state["eta_s"] = None
        now = time.perf_counter()
        if not force and (now - progress_last_emit) < heartbeat_sec:
            return
        if progress_snapshot_path is not None:
            atomic_write_text(progress_snapshot_path, json_dumps(json_safe(progress_state)) + "\n")
        sink_progress(dict(progress_state))
        progress_last_emit = now

    def log(message: str) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{stamp}] {message}"
        run_log_lines.append(line)
        sink_log(message)
        if run_log_path is not None:
            with run_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

        model_match = model_start_re.match(message)
        if model_match is not None:
            emit_progress(
                force=True,
                stage="stage_b",
                model_name=str(model_match.group("model")),
                run_index=int(model_match.group("run")),
                run_total=int(model_match.group("total")),
                config_index=0,
                config_total=0,
                last_message=message,
            )
            return
        config_match = config_start_re.match(message)
        if config_match is not None:
            emit_progress(
                force=True,
                stage="stage_b",
                model_name=str(config_match.group("model")),
                config_index=int(config_match.group("run")),
                config_total=int(config_match.group("total")),
                last_message=message,
            )
            return
        stage_a_model_match = stage_a_model_start_re.match(message)
        if stage_a_model_match is not None:
            emit_progress(
                force=True,
                stage="stage_a",
                model_name=str(stage_a_model_match.group("model")),
                run_index=int(stage_a_model_match.group("run")),
                run_total=int(stage_a_model_match.group("total")),
                config_index=0,
                config_total=0,
                last_message=message,
            )
            return
        stage_a_config_match = stage_a_config_start_re.match(message)
        if stage_a_config_match is not None:
            emit_progress(
                force=True,
                stage="stage_a",
                model_name=str(stage_a_config_match.group("model")),
                config_index=int(stage_a_config_match.group("run")),
                config_total=int(stage_a_config_match.group("total")),
                last_message=message,
            )
            return
        emit_progress(last_message=message)

    metric_scope = parse_metric_scope(config.metric_scope)
    decision_time_mode = parse_decision_time_mode(config.decision_time_mode)
    async_decision_time_mode = parse_decision_time_mode(config.async_decision_time_mode)
    data_policy = parse_data_policy(config.data_policy)
    ranking_policy = parse_ranking_policy(config.ranking_policy)
    export_figures = bool(config.export_figures)
    evaluation_mode = _parse_evaluation_mode(config.evaluation_mode)
    quick_screen_top_k = max(1, int(config.quick_screen_top_k))
    requested_models = tuple(normalize_model_name(name) for name in config.model_names)
    force_include_models = tuple(
        name
        for name in (normalize_model_name(item) for item in config.force_include_models)
        if name in set(requested_models)
    )
    if str(getattr(config, "task", "")).strip().lower() == "fbcca-weighted-compare":
        weighted_compare_prefix = (
            "legacy_fbcca_202603",
            "fbcca_fixed_all8",
            "fbcca_cw_all8",
            "fbcca_sw_all8",
            "fbcca_cw_sw_all8",
        )
        requested_models = tuple(
            dict.fromkeys(
                weighted_compare_prefix
                + tuple(
                    name
                    for name in requested_models
                    if name not in {"legacy_fbcca_202603", "fbcca", *weighted_compare_prefix}
                )
            )
        )
        force_include_models = requested_models
        quick_screen_top_k = max(quick_screen_top_k, len(requested_models))

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
    if dataset_session2 is not None and tuple(dataset_session2.board_eeg_channels) != tuple(
        dataset_session1.board_eeg_channels
    ):
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
    session1_has_long_idle = _dataset_has_long_idle(dataset_session1)
    session2_has_long_idle = False if dataset_session2 is None else _dataset_has_long_idle(dataset_session2)
    if bool(config.long_idle_required) and not session1_has_long_idle:
        raise RuntimeError(
            "long-idle evaluation was requested but session1 dataset does not contain long_idle data"
        )
    if bool(config.long_idle_required) and dataset_session2 is not None and not session2_has_long_idle:
        raise RuntimeError(
            "long-idle evaluation was requested but session2 dataset does not contain long_idle data"
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
    progress_snapshot_path = Path(resolved_report_paths["progress_snapshot"]).expanduser().resolve()
    log(f"Report directory prepared: {report_dir}")
    log(
        "Compute backend prepared: "
        f"requested={config.compute_backend} device={int(config.gpu_device)} "
        f"precision={config.gpu_precision} warmup={int(bool(config.gpu_warmup))} "
        f"cache={config.gpu_cache_policy}"
    )
    emit_progress(force=True, stage="prepare", last_message=f"Report directory prepared: {report_dir}")

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
        model_names=parse_model_list(",".join(requested_models)),
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
        idle_fp_hard_th=float(config.idle_fp_hard_th),
        channel_weight_l2=float(config.channel_weight_l2),
        subband_prior_strength=float(config.subband_prior_strength),
        control_state_mode=str(config.control_state_mode),
        metric_scope=metric_scope,
        decision_time_mode=decision_time_mode,
        async_decision_time_mode=async_decision_time_mode,
        ranking_policy=ranking_policy,
        dynamic_stop_enabled=bool(config.dynamic_stop_enabled),
        dynamic_stop_alpha=float(config.dynamic_stop_alpha),
        seed=int(config.seed),
        compute_backend=str(config.compute_backend),
        gpu_device=int(config.gpu_device),
        gpu_precision=str(config.gpu_precision),
        gpu_warmup=bool(config.gpu_warmup),
        gpu_cache_policy=str(config.gpu_cache_policy),
    )

    prefer_cross_session = dataset_session2 is not None
    ranking_source = "cross_session" if prefer_cross_session else "session1_holdout"

    if str(getattr(config, "task", "")).strip().lower() == "classifier-compare":
        eval_relation = "cross_session_eval" if dataset_session2 is not None else "session1_holdout_classifier_only"
        log(
            "Classifier compare start: "
            f"models={','.join(requested_models)} relation={eval_relation} "
            f"seeds={','.join(str(int(seed)) for seed in helper.eval_seeds)} "
            f"win_candidates={','.join(f'{float(v):g}' for v in config.win_candidates)}"
        )
        emit_progress(
            force=True,
            stage="classifier_compare",
            run_index=0,
            run_total=len(requested_models) * len(helper.eval_seeds),
            last_message="Classifier compare start",
        )

        classifier_seed_results: list[dict[str, Any]] = []
        failed_runs: list[dict[str, Any]] = []
        run_index = 0
        for raw_model_name in requested_models:
            model_name = normalize_model_name(raw_model_name)
            display_name = "fbcca_plain_all8" if model_name == "fbcca_fixed_all8" else model_name
            for eval_seed in helper.eval_seeds:
                run_index += 1
                seed_train, _seed_gate, seed_holdout = _split_session_for_train_eval(dataset_session1, seed=int(eval_seed))
                eval_segments = dataset_session2.trial_segments if dataset_session2 is not None else seed_holdout
                log(f"Stage A model start: {run_index}/{len(requested_models) * len(helper.eval_seeds)} model={display_name} seed={int(eval_seed)}")
                best_result: Optional[dict[str, Any]] = None
                for config_index, win_sec in enumerate(config.win_candidates, start=1):
                    try:
                        log(
                            f"Stage A config start: model={display_name} seed={int(eval_seed)} "
                            f"{config_index}/{len(config.win_candidates)} win={float(win_sec):g}s"
                        )
                        decoder = create_decoder(
                            model_name,
                            sampling_rate=fs,
                            freqs=dataset_session1.freqs,
                            win_sec=float(win_sec),
                            step_sec=float(helper.step_sec),
                            model_params={"Nh": DEFAULT_NH},
                            compute_backend=str(config.compute_backend),
                            gpu_device=int(config.gpu_device),
                            gpu_precision=str(config.gpu_precision),
                            gpu_warmup=bool(config.gpu_warmup),
                            gpu_cache_policy=str(config.gpu_cache_policy),
                        )
                        if getattr(decoder, "requires_fit", False):
                            decoder.fit(seed_train)
                        metrics_bundle = _evaluate_classifier_only_fixed_window(
                            decoder=decoder,
                            trial_segments=eval_segments,
                        )
                        metrics_4 = dict(metrics_bundle.get("metrics_4class", {}))
                        classifier_metrics = dict(metrics_bundle.get("classifier_only_metrics", {}))
                        result = {
                            "stage": "classifier_compare",
                            "model_name": display_name,
                            "source_model_name": model_name,
                            "implementation_level": model_implementation_level(model_name),
                            "method_note": model_method_note(model_name),
                            "channel_mode": "all8",
                            "eval_seed": int(eval_seed),
                            "selected_eeg_channels": [int(channel) for channel in eeg_channels],
                            "best_win_sec": float(win_sec),
                            "metrics": dict(classifier_metrics),
                            "metrics_4class": metrics_4,
                            "classifier_only_metrics": classifier_metrics,
                            "ranking_source": ranking_source,
                            "analysis_scope": "classifier_only_fixed_window",
                            "split_counts": {
                                "train": int(len(seed_train)),
                                "holdout": int(len(seed_holdout)),
                                "eval": int(len(eval_segments)),
                            },
                        }
                        if prefer_cross_session:
                            result["cross_session_metrics"] = dict(classifier_metrics)
                            result["cross_session_metrics_4class"] = dict(metrics_4)
                        if best_result is None or _classifier_only_rank_key(classifier_metrics) < _classifier_only_rank_key(
                            dict(best_result.get("classifier_only_metrics", {}))
                        ):
                            best_result = result
                        log(
                            "Classifier config done: "
                            f"model={display_name} seed={int(eval_seed)} win={float(win_sec):g}s "
                            f"acc={float(classifier_metrics.get('acc_4class', 0.0)):.4f} "
                            f"macro_f1={float(classifier_metrics.get('macro_f1_4class', 0.0)):.4f}"
                        )
                    except Exception as exc:
                        log(f"Classifier config failed: model={display_name} seed={int(eval_seed)} win={float(win_sec):g}s error={exc}")
                if best_result is None:
                    failed_runs.append({"model_name": display_name, "eval_seed": int(eval_seed), "error": "all classifier configs failed"})
                    continue
                classifier_seed_results.append(best_result)

        classifier_results = _aggregate_classifier_seed_results(classifier_seed_results)
        deployment_run = classifier_results[0] if classifier_results else {
            "model_name": "",
            "metrics": {},
            "metrics_4class": {},
            "classifier_only_metrics": {},
        }
        classifier_only_ranked = _build_classifier_only_board(
            quick_screen_ranked=classifier_results,
            deployment_ranked_by_model=classifier_results,
        )
        classifier_only_top = classifier_only_ranked[0] if classifier_only_ranked else {
            "model_name": "",
            "rank": 0,
            "metrics": {},
        }
        named_stem = _named_report_stem(
            subject_id=str(dataset_session1.subject_id),
            recommended_model=str(deployment_run.get("model_name", "")),
            deployed_model=str(deployment_run.get("model_name", "")),
            channel_mode="all8",
            eval_seed=int(config.seed),
        )
        named_report_json_path = report_dir / f"{named_stem}.json"
        named_report_md_path = report_dir / f"{named_stem}.md"
        report_payload: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "offline_train_eval",
            "task": "classifier-compare",
            "analysis_mode": "all_candidates_classifier_only",
            "report_path": str(report_json_path),
            "report_dir": str(report_dir),
            "selection_snapshot_path": str(selection_snapshot_path),
            "run_config_path": str(run_config_path),
            "run_log_path": str(run_log_path),
            "progress_snapshot_path": str(progress_snapshot_path),
            "dataset_manifest_session1": str(Path(config.dataset_manifest_session1).expanduser().resolve()),
            "dataset_manifest_session2": (
                None
                if config.dataset_manifest_session2 is None
                else str(Path(config.dataset_manifest_session2).expanduser().resolve())
            ),
            "classifier_compare": {
                "evaluation_data_relation": str(eval_relation),
                "ranking_rule": "Acc_4class -> Macro-F1 -> ITR -> decision_time -> inference_ms",
                "note": "This task compares classifier fixed-window performance only; it does not fit async gates.",
            },
            "seed_summary": _seed_summary_from_runs(classifier_seed_results),
            "seed_level_results": list(classifier_seed_results),
            "aggregated_model_ranking": list(classifier_results),
            "classifier_only_ranking": list(classifier_only_ranked),
            "end_to_end_ranking": [],
            "model_results": list(classifier_results),
            "stage_b_run_results": list(classifier_results) + list(failed_runs),
            "failed_model_results": list(failed_runs),
            "chosen_model": str(deployment_run.get("model_name", "")),
            "chosen_metrics": dict(deployment_run.get("classifier_only_metrics", {})),
            "chosen_async_metrics": {},
            "chosen_metrics_4class": dict(deployment_run.get("metrics_4class", {})),
            "chosen_metrics_2class": {},
            "chosen_metrics_5class": None,
            "async_metrics": {},
            "metrics_4class": dict(deployment_run.get("metrics_4class", {})),
            "metrics_2class": {},
            "metrics_5class": None,
            "ranking_boards": {
                "classifier_only": list(classifier_only_ranked),
                "end_to_end": [],
            },
            "classifier_only_top_model": str(classifier_only_top.get("model_name", "")),
            "classifier_only_top_rank": int(classifier_only_top.get("rank", 0) or 0),
            "classifier_only_top_metrics": dict(classifier_only_top.get("metrics", {})),
            "model_compare_table": [
                _model_compare_row(item, prefer_cross_session=prefer_cross_session)
                for item in classifier_results
            ],
            "fbcca_weight_table": [],
            "ab_comparisons": [],
            "fbcca_weight_aggregation": {
                "mode": str(config.weight_aggregation),
                "note": "No FBCCA weight aggregation is performed in classifier-compare.",
            },
            "idle_false_positive_events": [],
            "control_state_mode": str(config.control_state_mode),
            "realtime_profile_recommendation": {
                "recommended": False,
                "reason": "classifier-compare does not produce a realtime profile",
            },
            "profile_saved": False,
            "default_profile_saved": False,
            "profile_for_realtime_path": None,
            "profile_for_realtime_type": "none_classifier_only_report",
            "roundtrip_ready": False,
            "atomic_write_completed": False,
            "weight_definition_notes": _weight_definition_notes(),
            "metric_definition": benchmark_metric_definition_payload(),
            "metric_scope": str(metric_scope),
            "decision_time_mode": str(decision_time_mode),
            "async_decision_time_mode": str(async_decision_time_mode),
            "ranking_policy": "classifier-only",
            "ranking_source": str(ranking_source),
            "quality_filter": {
                "min_sample_ratio": float(config.quality_min_sample_ratio),
                "max_retry_count": int(config.quality_max_retry_count),
            },
            "quality_summary_session1": list(session1_quality_rows),
            "quality_summary_session2": [] if session2_quality_row is None else [dict(session2_quality_row)],
            "protocol_consistency": dict(protocol_consistency_summary),
            "named_artifacts": {
                "report_json": str(named_report_json_path),
                "report_md": str(named_report_md_path),
                "profile_json": "",
                "best_candidate_profile": "",
                "best_fbcca_weighted_profile": "",
            },
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
        selection_snapshot_payload = {
            "task": "classifier-compare",
            "analysis_mode": "all_candidates_classifier_only",
            "chosen_model": str(deployment_run.get("model_name", "")),
            "model_results": [
                {
                    "rank": int(item.get("rank_end_to_end", 0) or 0),
                    "model_name": str(item.get("model_name", "")),
                    "classifier_only_metrics": dict(item.get("classifier_only_metrics", {})),
                }
                for item in classifier_results
            ],
            "named_artifacts": dict(report_payload.get("named_artifacts", {})),
        }
        _write_json_atomic(run_config_path, config_payload)
        _write_json_atomic(selection_snapshot_path, selection_snapshot_payload)
        _write_json_atomic(report_json_path, report_payload)
        markdown_body = _render_training_eval_markdown(report_payload)
        _write_text_atomic(report_json_path.with_suffix(".md"), markdown_body)
        _write_json_atomic(named_report_json_path, report_payload)
        _write_text_atomic(named_report_md_path, markdown_body)
        report_payload["atomic_write_completed"] = True
        _write_json_atomic(report_json_path, report_payload)
        _write_json_atomic(named_report_json_path, report_payload)
        log(f"Classifier compare report written: {report_json_path}")
        emit_progress(
            force=True,
            stage="done",
            run_index=len(requested_models),
            run_total=len(requested_models),
            last_message=f"Classifier compare report written: {report_json_path}",
        )
        return report_payload

    if str(getattr(config, "task", "")).strip().lower() == "profile-eval":
        pretrained_profile_path = getattr(config, "pretrained_profile_path", None)
        if pretrained_profile_path is None:
            raise ValueError("--pretrained-profile is required when --task profile-eval")
        pretrained_profile_path = Path(pretrained_profile_path).expanduser().resolve()
        if not pretrained_profile_path.exists():
            raise FileNotFoundError(f"pretrained profile not found: {pretrained_profile_path}")

        eval_segments = dataset_session2.trial_segments if dataset_session2 is not None else holdout_base
        eval_dataset_channels = (
            tuple(int(channel) for channel in dataset_session2.board_eeg_channels)
            if dataset_session2 is not None
            else tuple(int(channel) for channel in eeg_channels)
        )
        eval_relation = (
            "cross_session_eval"
            if dataset_session2 is not None
            else "internal_holdout_or_replay"
        )
        log(
            "Profile eval start: "
            f"profile={pretrained_profile_path} relation={eval_relation} "
            f"mode={getattr(config, 'profile_eval_mode', 'fbcca-vs-all')}"
        )
        emit_progress(
            force=True,
            stage="profile_eval",
            run_index=0,
            run_total=1 + len(requested_models),
            last_message="Profile eval start",
        )
        compare_models = tuple(
            dict.fromkeys(
                ("fbcca_fixed_all8",)
                + tuple(
                    normalize_model_name(name)
                    for name in requested_models
                    if normalize_model_name(name) not in {"fbcca_profile_weighted", "fbcca_cw_all8", "fbcca_sw_all8", "fbcca_cw_sw_all8"}
                )
            )
        )
        if str(getattr(config, "profile_eval_mode", "fbcca-vs-all")).strip().lower() == "fbcca-only":
            compare_models = ("fbcca_fixed_all8",)

        total_runs = 1 + len(compare_models)
        log(f"Profile eval compare set: {','.join(compare_models)}")
        log(f"Model start: 1/{total_runs} mode=profile seed=0 model=fbcca_profile_weighted")
        pretrained_profile, profile_result = _evaluate_pretrained_profile_result(
            profile_path=pretrained_profile_path,
            eval_segments=eval_segments,
            dataset_channels=eval_dataset_channels,
            sampling_rate=fs if dataset_session2 is None else int(dataset_session2.sampling_rate),
            metric_scope=metric_scope,
            decision_time_mode=decision_time_mode,
            async_decision_time_mode=async_decision_time_mode,
            compute_backend=str(config.compute_backend),
            gpu_device=int(config.gpu_device),
            gpu_precision=str(config.gpu_precision),
            gpu_warmup=bool(config.gpu_warmup),
            gpu_cache_policy=str(config.gpu_cache_policy),
            prefer_cross_session=prefer_cross_session,
        )
        log(
            "Model done: "
            f"model=fbcca_profile_weighted idle_fp={float(profile_result.get('metrics', {}).get('idle_fp_per_min', 0.0)):.4f} "
            f"recall={float(profile_result.get('metrics', {}).get('control_recall', 0.0)):.4f}"
        )

        successful_runs: list[dict[str, Any]] = [profile_result]
        failed_runs: list[dict[str, Any]] = []
        for run_index, model_name in enumerate(compare_models, start=2):
            model_name = normalize_model_name(model_name)
            display_name = "fbcca_plain_all8" if model_name == "fbcca_fixed_all8" else model_name
            try:
                log(f"Model start: {run_index}/{total_runs} mode=all8 seed={int(config.seed)} model={display_name}")
                selected_channels = tuple(int(channel) for channel in eeg_channels)
                train_segments = _subset_trial_segments_by_positions(train_base, list(range(len(selected_channels))))
                gate_segments = _subset_trial_segments_by_positions(gate_base, list(range(len(selected_channels))))
                eval_subset = eval_segments
                profile, result = helper._benchmark_single_model(
                    model_name=model_name,
                    fs=fs,
                    train_segments=train_segments,
                    gate_segments=gate_segments,
                    eval_segments=eval_subset,
                    eeg_channels=selected_channels,
                    log_fn=log,
                )
                if model_name == "fbcca_fixed_all8":
                    result["model_name"] = "fbcca_plain_all8"
                result["selected_eeg_channels"] = [int(channel) for channel in selected_channels]
                result["channel_mode"] = "all8"
                result["eval_seed"] = int(config.seed)
                result["split_counts"] = {
                    "train_segments": int(len(train_segments)),
                    "gate_segments": int(len(gate_segments)),
                    "eval_segments": int(len(eval_subset)),
                }
                result["ranking_source"] = ranking_source
                result["profile_channel_weight_mode"] = (
                    None if getattr(profile, "channel_weight_mode", None) is None else str(profile.channel_weight_mode)
                )
                result["profile_channel_weights"] = (
                    None
                    if getattr(profile, "channel_weights", None) is None
                    else [float(value) for value in profile.channel_weights]
                )
                result["profile_subband_weight_mode"] = (
                    None if getattr(profile, "subband_weight_mode", None) is None else str(profile.subband_weight_mode)
                )
                result["profile_subband_weights"] = (
                    None
                    if getattr(profile, "subband_weights", None) is None
                    else [float(value) for value in profile.subband_weights]
                )
                result["profile_subband_weight_params"] = (
                    None
                    if getattr(profile, "subband_weight_params", None) is None
                    else json_safe(dict(profile.subband_weight_params))
                )
                result["profile_spatial_filter_mode"] = (
                    None if getattr(profile, "spatial_filter_mode", None) is None else str(profile.spatial_filter_mode)
                )
                result["profile_spatial_filter_rank"] = (
                    None if getattr(profile, "spatial_filter_rank", None) is None else int(profile.spatial_filter_rank)
                )
                result["profile_joint_weight_training"] = (
                    None
                    if getattr(profile, "joint_weight_training", None) is None
                    else json_safe(dict(profile.joint_weight_training))
                )
                result = _normalize_frontend_result_fields(result)
                result["frontend_summary"] = _build_frontend_summary(result)
                if prefer_cross_session:
                    result["cross_session_metrics"] = dict(result.get("metrics", {}))
                    result["cross_session_async_metrics"] = dict(result.get("async_metrics", {}))
                    result["cross_session_metrics_4class"] = dict(result.get("metrics_4class", {}))
                    result["cross_session_metrics_2class"] = dict(result.get("metrics_2class", {}))
                    result["cross_session_metrics_5class"] = result.get("metrics_5class")
                    result["cross_session_paper_lens_metrics_4class"] = dict(result.get("paper_lens_metrics_4class", {}))
                    result["cross_session_paper_lens_metrics_2class"] = dict(result.get("paper_lens_metrics_2class", {}))
                    result["cross_session_paper_lens_metrics_5class"] = result.get("paper_lens_metrics_5class")
                    result["cross_session_async_lens_metrics_4class"] = dict(result.get("async_lens_metrics_4class", {}))
                    result["cross_session_async_lens_metrics_2class"] = dict(result.get("async_lens_metrics_2class", {}))
                    result["cross_session_async_lens_metrics_5class"] = result.get("async_lens_metrics_5class")
                successful_runs.append(result)
                log(
                    "Model done: "
                    f"model={display_name} idle_fp={float(result.get('metrics', {}).get('idle_fp_per_min', 0.0)):.4f} "
                    f"recall={float(result.get('metrics', {}).get('control_recall', 0.0)):.4f}"
                )
            except Exception as exc:
                failed_runs.append({"model_name": display_name, "error": str(exc)})
                log(f"Model failed: model={display_name} error={exc}")

        ranked_runs = sorted(
            successful_runs,
            key=lambda item: benchmark_rank_key(
                _ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session)
            ),
        )
        for rank, item in enumerate(ranked_runs, start=1):
            item["rank_end_to_end"] = int(rank)
            item["deployment_rank"] = int(rank)
            item["meets_acceptance"] = bool(
                profile_meets_acceptance(_ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session))
            )
        deployment_run = ranked_runs[0] if ranked_runs else profile_result
        deployment_views = _source_views(
            deployment_run,
            prefer_cross_session=prefer_cross_session,
            decision_time_mode=decision_time_mode,
            async_decision_time_mode=async_decision_time_mode,
        )
        classifier_only_ranked = _build_classifier_only_board(
            quick_screen_ranked=[],
            deployment_ranked_by_model=ranked_runs,
        )
        classifier_only_top = classifier_only_ranked[0] if classifier_only_ranked else {"model_name": "", "rank": 0, "metrics": {}}
        for index, item in enumerate(ranked_runs):
            y_true, y_pred, decision_times, labels = _extract_4class_vectors(item)
            item["bootstrap_4class"] = _bootstrap_4class_summary(
                y_true=y_true,
                y_pred=y_pred,
                decision_time_samples_s=decision_times,
                labels=labels if labels else ("8Hz", "10Hz", "12Hz", "15Hz"),
                seed=int(config.seed) + int(index) * 17,
            )

        named_stem = _named_report_stem(
            subject_id=str(dataset_session1.subject_id),
            recommended_model=str(deployment_run.get("model_name", "")),
            deployed_model=str(deployment_run.get("model_name", "")),
            channel_mode=str(deployment_run.get("channel_mode", "")),
            eval_seed=int(deployment_run.get("eval_seed", 0) or 0),
        )
        named_report_json_path = report_dir / f"{named_stem}.json"
        named_report_md_path = report_dir / f"{named_stem}.md"
        named_profile_path = report_dir / f"{named_stem}__profile.json"
        atomic_copy_text_file(pretrained_profile_path, named_profile_path)

        report_payload: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "offline_train_eval",
            "task": "profile-eval",
            "report_path": str(report_json_path),
            "report_dir": str(report_dir),
            "selection_snapshot_path": str(selection_snapshot_path),
            "run_config_path": str(run_config_path),
            "run_log_path": str(run_log_path),
            "progress_snapshot_path": str(progress_snapshot_path),
            "dataset_manifest_session1": str(Path(config.dataset_manifest_session1).expanduser().resolve()),
            "dataset_manifest_session2": (
                None if config.dataset_manifest_session2 is None else str(Path(config.dataset_manifest_session2).expanduser().resolve())
            ),
            "data_policy": str(data_policy),
            "profile_eval": {
                "pretrained_profile_path": str(pretrained_profile_path),
                "profile_eval_mode": str(getattr(config, "profile_eval_mode", "fbcca-vs-all")),
                "freeze_profile_weights": bool(getattr(config, "freeze_profile_weights", True)),
                "evaluation_data_relation": str(eval_relation),
                "training_data_same_as_eval_data": bool(dataset_session2 is None),
                "same_session_profile_eval": bool(dataset_session2 is None),
                "cross_session_eval": bool(dataset_session2 is not None),
                "long_idle_used": bool(session2_has_long_idle if dataset_session2 is not None else session1_has_long_idle),
                "pretrained_profile_validation": _profile_frontend_weight_flags(pretrained_profile),
            },
            "model_results": list(ranked_runs),
            "stage_b_run_results": list(ranked_runs) + list(failed_runs),
            "failed_model_results": list(failed_runs),
            "chosen_model": str(deployment_run.get("model_name", "")),
            "chosen_metrics": dict(deployment_views.get("metrics", {})),
            "chosen_async_metrics": dict(deployment_views.get("async_metrics", {})),
            "chosen_metrics_4class": dict(deployment_views.get("metrics_4class", {})),
            "chosen_metrics_2class": dict(deployment_views.get("metrics_2class", {})),
            "chosen_metrics_5class": deployment_views.get("metrics_5class"),
            "chosen_frontend_summary": dict(deployment_run.get("frontend_summary", {})),
            "async_metrics": dict(deployment_views.get("async_metrics", {})),
            "metrics_4class": dict(deployment_views.get("metrics_4class", {})),
            "metrics_2class": dict(deployment_views.get("metrics_2class", {})),
            "metrics_5class": deployment_views.get("metrics_5class"),
            "paper_lens_metrics": dict(deployment_views.get("paper_lens_metrics", {})),
            "async_lens_metrics": dict(deployment_views.get("async_lens_metrics", {})),
            "ranking_boards": {
                "end_to_end": [
                    {
                        "rank": int(item.get("rank_end_to_end", 0) or 0),
                        "model_name": str(item.get("model_name", "")),
                        "metrics": dict(_ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session)),
                    }
                    for item in ranked_runs
                ],
                "classifier_only": list(classifier_only_ranked),
            },
            "classifier_only_top_model": str(classifier_only_top.get("model_name", "")),
            "classifier_only_top_rank": int(classifier_only_top.get("rank", 0) or 0),
            "classifier_only_top_metrics": dict(classifier_only_top.get("metrics", {})),
            "fbcca_weight_table": [
                _weight_table_row(item, prefer_cross_session=prefer_cross_session)
                for item in ranked_runs
                if _is_fbcca_weight_result(item) or str(item.get("model_name", "")).startswith("fbcca")
            ],
            "model_compare_table": [
                _model_compare_row(item, prefer_cross_session=prefer_cross_session)
                for item in ranked_runs
            ],
            "ab_comparisons": _build_ab_comparisons(ranked_runs),
            "profile_saved": False,
            "default_profile_saved": False,
            "best_candidate_profile_path": (
                str(named_profile_path) if str(deployment_run.get("model_name", "")) == "fbcca_profile_weighted" else None
            ),
            "best_candidate_profile_saved": bool(str(deployment_run.get("model_name", "")) == "fbcca_profile_weighted"),
            "best_fbcca_weighted_profile_path": str(named_profile_path),
            "best_fbcca_weighted_profile_saved": True,
            "profile_for_realtime_path": str(named_profile_path),
            "profile_for_realtime_type": "pretrained_weighted_profile_copy",
            "profile_for_realtime_source": str(pretrained_profile_path),
            "roundtrip_ready": True,
            "atomic_write_completed": False,
            "weight_definition_notes": _weight_definition_notes(),
            "metric_definition": benchmark_metric_definition_payload(),
            "metric_scope": str(metric_scope),
            "decision_time_mode": str(decision_time_mode),
            "async_decision_time_mode": str(async_decision_time_mode),
            "ranking_policy": str(ranking_policy),
            "ranking_source": str(ranking_source),
            "quality_filter": {
                "min_sample_ratio": float(config.quality_min_sample_ratio),
                "max_retry_count": int(config.quality_max_retry_count),
            },
            "quality_summary_session1": list(session1_quality_rows),
            "quality_summary_session2": [] if session2_quality_row is None else [dict(session2_quality_row)],
            "protocol_consistency": dict(protocol_consistency_summary),
            "named_artifacts": {
                "report_json": str(named_report_json_path),
                "report_md": str(named_report_md_path),
                "profile_json": str(named_profile_path),
                "best_candidate_profile": (
                    str(named_profile_path) if str(deployment_run.get("model_name", "")) == "fbcca_profile_weighted" else ""
                ),
                "best_fbcca_weighted_profile": str(named_profile_path),
            },
            "realtime_profile_recommendation": {
                "recommended": True,
                "profile_path": str(named_profile_path),
                "profile_type": "pretrained_weighted_profile_copy",
                "reason": "profile_eval_uses_frozen_weighted_fbcca_profile",
            },
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
        config_payload["pretrained_profile_path"] = str(pretrained_profile_path)
        selection_snapshot_payload = {
            "task": "profile-eval",
            "chosen_model": str(deployment_run.get("model_name", "")),
            "profile_for_realtime_path": str(named_profile_path),
            "model_results": [
                {
                    "rank": int(item.get("rank_end_to_end", 0) or 0),
                    "model_name": str(item.get("model_name", "")),
                    "metrics": dict(_ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session)),
                }
                for item in ranked_runs
            ],
            "named_artifacts": dict(report_payload.get("named_artifacts", {})),
        }
        _write_json_atomic(run_config_path, config_payload)
        _write_json_atomic(selection_snapshot_path, selection_snapshot_payload)
        _write_json_atomic(report_json_path, report_payload)
        markdown_body = _render_training_eval_markdown(report_payload)
        _write_text_atomic(report_json_path.with_suffix(".md"), markdown_body)
        _write_json_atomic(named_report_json_path, report_payload)
        _write_text_atomic(named_report_md_path, markdown_body)
        report_payload["atomic_write_completed"] = True
        _write_json_atomic(report_json_path, report_payload)
        _write_json_atomic(named_report_json_path, report_payload)
        log(f"Profile eval report written: {report_json_path}")
        emit_progress(
            force=True,
            stage="done",
            run_index=total_runs,
            run_total=total_runs,
            last_message=f"Profile eval report written: {report_json_path}",
        )
        return report_payload

    stage_a_candidates = tuple(normalize_model_name(name) for name in helper.model_names)
    quick_screen_rows: list[dict[str, Any]] = []
    quick_screen_ranked: list[dict[str, Any]] = []
    stage_a_only_models: list[str] = []

    stage_a_started_at = time.perf_counter()
    if evaluation_mode == "staged":
        log("Stage start: stage_a quick_screen")
        emit_progress(
            force=True,
            stage="stage_a",
            run_index=0,
            run_total=len(stage_a_candidates),
            config_index=0,
            config_total=0,
            last_message="Stage start: stage_a quick_screen",
        )
        try:
            quick_screen_rows = _quick_screen_models(
                model_names=stage_a_candidates,
                train_segments=train_base,
                gate_segments=gate_base,
                sampling_rate=fs,
                freqs=dataset_session1.freqs,
                step_sec=helper.step_sec,
                win_candidates=helper.win_candidates,
                seed=int(config.seed),
                log_fn=log,
                compute_backend=str(config.compute_backend),
                gpu_device=int(config.gpu_device),
                gpu_precision=str(config.gpu_precision),
                gpu_warmup=bool(config.gpu_warmup),
                gpu_cache_policy=str(config.gpu_cache_policy),
            )
        except Exception as exc:
            quick_screen_rows = []
            log(f"Stage A quick_screen skipped: {exc}; using requested models for Stage B.")
        quick_screen_ranked = [
            dict(item)
            for item in quick_screen_rows
            if isinstance(item.get("classifier_only_metrics"), dict)
        ]
        quick_screen_ranked.sort(
            key=lambda item: _quick_screen_rank_key(
                dict(item.get("quick_screen_metrics", {})) | dict(item.get("classifier_only_metrics", {}))
            )
        )
        for rank, item in enumerate(quick_screen_ranked, start=1):
            item["quick_screen_rank"] = int(rank)

        candidate_names: list[str] = []
        for item in quick_screen_ranked:
            if not bool(item.get("quick_screen_pass", False)):
                continue
            model_name = normalize_model_name(item.get("model_name", ""))
            if model_name not in candidate_names:
                candidate_names.append(model_name)
            if len(candidate_names) >= quick_screen_top_k:
                break
        for forced in force_include_models:
            if forced not in candidate_names:
                candidate_names.append(forced)
        if not candidate_names:
            candidate_names = list(stage_a_candidates)
        stage_b_candidates = tuple(candidate_names)
        stage_a_only_models = [name for name in stage_a_candidates if name not in stage_b_candidates]
        log(
            "Stage done: stage_a quick_screen "
            f"candidates={','.join(stage_b_candidates)} screened={len(quick_screen_rows)}"
        )
    else:
        stage_b_candidates = tuple(stage_a_candidates)
        log(
            "Stage start: stage_b full "
            f"candidate_models={','.join(stage_b_candidates)}"
        )
    stage_a_duration_s = float(max(time.perf_counter() - stage_a_started_at, 0.0))

    stage_b_started_at = time.perf_counter()
    successful_stage_b_runs: list[dict[str, Any]] = []
    failed_stage_b_runs: list[dict[str, Any]] = []
    robustness_runs: list[dict[str, Any]] = []
    profile_by_run: dict[tuple[str, str, int], Any] = {}
    total_runs = int(
        sum(
            1
            for channel_mode in helper.channel_modes
            for _eval_seed in helper.eval_seeds
            for model_name in stage_b_candidates
            if not (
                str(channel_mode) == "auto"
                and _fbcca_weight_learning_requires_all8_for_config(str(model_name), config)
            )
        )
    )
    current_run = 0

    for channel_mode in helper.channel_modes:
        for eval_seed in helper.eval_seeds:
            seed_mode_success: list[dict[str, Any]] = []
            seed_mode_all: list[dict[str, Any]] = []
            for model_name in stage_b_candidates:
                model_name = normalize_model_name(model_name)
                if str(channel_mode) == "auto" and _fbcca_weight_learning_requires_all8_for_config(model_name, config):
                    log(
                        f"Skip run: mode={channel_mode} seed={eval_seed} model={model_name} "
                        "requires all8 for FBCCA weight learning"
                    )
                    continue
                current_run += 1
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
                                "Nh": DEFAULT_NH,
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
                    if model_name in {"fbcca", "fbcca_cw_all8", "fbcca_sw_all8", "fbcca_cw_sw_all8", "fbcca_cw_sw_trca_shared"}:
                        log(
                            "FBCCA 当前会额外执行通道权重、空间前端和门控拟合，"
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
                    result["channel_selection"] = list(channel_scores)
                    result["ranking_source"] = _metrics_source_name(
                        result,
                        prefer_cross_session=prefer_cross_session,
                    )
                    profile_channel_weight_mode = getattr(profile, "channel_weight_mode", None)
                    profile_channel_weights = getattr(profile, "channel_weights", None)
                    profile_subband_weight_mode = getattr(profile, "subband_weight_mode", None)
                    profile_subband_weights = getattr(profile, "subband_weights", None)
                    profile_subband_weight_params = getattr(profile, "subband_weight_params", None)
                    profile_spatial_filter_mode = getattr(profile, "spatial_filter_mode", None)
                    profile_spatial_filter_rank = getattr(profile, "spatial_filter_rank", None)
                    profile_joint_weight_training = getattr(profile, "joint_weight_training", None)
                    result["profile_channel_weight_mode"] = (
                        None if profile_channel_weight_mode is None else str(profile_channel_weight_mode)
                    )
                    result["profile_channel_weights"] = (
                        None
                        if profile_channel_weights is None
                        else [float(value) for value in profile_channel_weights]
                    )
                    result["profile_subband_weight_mode"] = (
                        None if profile_subband_weight_mode is None else str(profile_subband_weight_mode)
                    )
                    result["profile_subband_weights"] = (
                        None
                        if profile_subband_weights is None
                        else [float(value) for value in profile_subband_weights]
                    )
                    result["profile_subband_weight_params"] = (
                        None
                        if profile_subband_weight_params is None
                        else json_safe(dict(profile_subband_weight_params))
                    )
                    result["profile_spatial_filter_mode"] = (
                        None if profile_spatial_filter_mode is None else str(profile_spatial_filter_mode)
                    )
                    result["profile_spatial_filter_rank"] = (
                        None if profile_spatial_filter_rank is None else int(profile_spatial_filter_rank)
                    )
                    result["profile_joint_weight_training"] = (
                        None
                        if profile_joint_weight_training is None
                        else json_safe(dict(profile_joint_weight_training))
                    )

                    if dataset_session2 is not None:
                        positions_s2 = [list(dataset_session2.board_eeg_channels).index(channel) for channel in selected_channels]
                        session2_subset = _subset_trial_segments_by_positions(dataset_session2.trial_segments, positions_s2)
                        decoder_s2 = load_decoder_from_profile(
                            profile,
                            sampling_rate=dataset_session2.sampling_rate,
                            compute_backend=str(config.compute_backend),
                            gpu_device=int(config.gpu_device),
                            gpu_precision=str(config.gpu_precision),
                            gpu_warmup=bool(config.gpu_warmup),
                            gpu_cache_policy=str(config.gpu_cache_policy),
                        )
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

                    source_metrics = _ranking_metrics_from_result(
                        result,
                        prefer_cross_session=prefer_cross_session,
                    )
                    result["source_metrics"] = dict(source_metrics)
                    result["meets_acceptance"] = bool(profile_meets_acceptance(source_metrics))
                    result = _normalize_frontend_result_fields(result)
                    result["frontend_summary"] = _build_frontend_summary(result)

                    profile_key = (str(model_name), str(channel_mode), int(eval_seed))
                    profile_by_run[profile_key] = profile
                    successful_stage_b_runs.append(dict(result))
                    seed_mode_all.append(dict(result))
                    seed_mode_success.append(dict(result))
                    log(
                        f"Model done: mode={channel_mode} seed={eval_seed} model={model_name} "
                        f"backend={result.get('compute_backend_used', config.compute_backend)} "
                        f"idle_fp={float(source_metrics.get('idle_fp_per_min', float('inf'))):.4f} "
                        f"recall={float(source_metrics.get('control_recall', 0.0)):.4f}"
                    )
                except Exception as exc:
                    failed = {
                        "model_name": str(model_name),
                        "implementation_level": model_implementation_level(model_name),
                        "method_note": model_method_note(model_name),
                        "channel_mode": str(channel_mode),
                        "eval_seed": int(eval_seed),
                        "error": str(exc),
                        "ranking_source": str(ranking_source),
                        "meets_acceptance": False,
                    }
                    failed_stage_b_runs.append(dict(failed))
                    seed_mode_all.append(dict(failed))
                    log(f"Model failed: mode={channel_mode} seed={eval_seed} model={model_name} error={exc}")

            seed_mode_success.sort(
                key=lambda item: benchmark_rank_key(
                    _ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session),
                    ranking_policy=ranking_policy,
                )
            )
            rank_by_model = {
                str(item["model_name"]): int(rank)
                for rank, item in enumerate(seed_mode_success, start=1)
            }
            for item in seed_mode_all:
                item["run_rank"] = rank_by_model.get(str(item.get("model_name", "")))
                robustness_runs.append(dict(item))

    stage_b_duration_s = float(max(time.perf_counter() - stage_b_started_at, 0.0))
    if not successful_stage_b_runs:
        raise RuntimeError("offline train-eval failed: all stage_b models failed")

    all_successful_ranked = sorted(
        (dict(item) for item in successful_stage_b_runs),
        key=lambda item: benchmark_rank_key(
            _ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session),
            ranking_policy=ranking_policy,
        ),
    )
    for rank, item in enumerate(all_successful_ranked, start=1):
        item["deployment_rank"] = int(rank)

    best_runs_by_model = _best_successful_by_model(
        all_successful_ranked,
        prefer_cross_session=prefer_cross_session,
        ranking_policy=ranking_policy,
    )
    for rank, item in enumerate(best_runs_by_model, start=1):
        item["rank_end_to_end"] = int(rank)
        item["rank"] = int(rank)
        item["meets_acceptance"] = bool(
            profile_meets_acceptance(_ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session))
        )

    classifier_only_ranked = _build_classifier_only_board(
        quick_screen_ranked=quick_screen_ranked,
        deployment_ranked_by_model=best_runs_by_model,
    )
    classifier_only_top = classifier_only_ranked[0] if classifier_only_ranked else {
        "model_name": "",
        "rank": 0,
        "metrics": {},
    }

    family_summary = _aggregate_family_results(
        all_successful_ranked,
        ranking_policy=ranking_policy,
        prefer_cross_session=prefer_cross_session,
    )
    family_rank_end_to_end = list(family_summary.get("families", []))
    winning_family = family_rank_end_to_end[0] if family_rank_end_to_end else {
        "model_name": str(best_runs_by_model[0].get("model_name", "")),
        "rank": 1,
        "metrics_mean": dict(_ranking_metrics_from_result(best_runs_by_model[0], prefer_cross_session=prefer_cross_session)),
    }
    winning_model_family = str(winning_family.get("model_name", ""))

    family_runs = [
        dict(item)
        for item in all_successful_ranked
        if str(item.get("model_name", "")) == winning_model_family
    ]
    accepted_family_runs = [
        item
        for item in family_runs
        if profile_meets_acceptance(_ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session))
    ]
    deployment_run = accepted_family_runs[0] if accepted_family_runs else family_runs[0]
    deployment_run_key = (
        str(deployment_run.get("model_name", "")),
        str(deployment_run.get("channel_mode", "")),
        int(deployment_run.get("eval_seed", 0) or 0),
    )
    deployment_views = _source_views(
        deployment_run,
        prefer_cross_session=prefer_cross_session,
        decision_time_mode=decision_time_mode,
        async_decision_time_mode=async_decision_time_mode,
    )
    deployment_run_rank = [
        {
            "rank": int(item.get("deployment_rank", 0) or 0),
            "model_name": str(item.get("model_name", "")),
            "channel_mode": str(item.get("channel_mode", "")),
            "eval_seed": int(item.get("eval_seed", 0) or 0),
            "metrics": dict(_ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session)),
            "ranking_source": str(_metrics_source_name(item, prefer_cross_session=prefer_cross_session)),
            "meets_acceptance": bool(
                profile_meets_acceptance(_ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session))
            ),
        }
        for item in all_successful_ranked
    ]
    ranking_end_to_end = [
        {
            "rank": int(item.get("rank_end_to_end", 0) or 0),
            "model_name": str(item.get("model_name", "")),
            "channel_mode": str(item.get("channel_mode", "")),
            "eval_seed": int(item.get("eval_seed", 0) or 0),
            "metrics": dict(_ranking_metrics_from_result(item, prefer_cross_session=prefer_cross_session)),
            "meets_acceptance": bool(item.get("meets_acceptance", False)),
        }
        for item in best_runs_by_model
    ]

    stats_baseline = next(
        (item for item in best_runs_by_model if str(item.get("model_name", "")) == "legacy_fbcca_202603"),
        best_runs_by_model[0],
    )
    baseline_true, baseline_pred, baseline_times, _baseline_labels = _extract_4class_vectors(stats_baseline)
    for index, item in enumerate(best_runs_by_model):
        y_true, y_pred, decision_times, labels = _extract_4class_vectors(item)
        item["bootstrap_4class"] = _bootstrap_4class_summary(
            y_true=y_true,
            y_pred=y_pred,
            decision_time_samples_s=decision_times,
            labels=labels if labels else ("8Hz", "10Hz", "12Hz", "15Hz"),
            seed=int(config.seed) + int(index) * 17,
        )
        item["paired_vs_baseline"] = {
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

    report_dir.mkdir(parents=True, exist_ok=True)
    best_candidate_profile_path = report_dir / "profile_best_candidate.json"
    best_fbcca_weighted_profile_path = report_dir / "profile_best_fbcca_weighted.json"
    best_candidate_profile_saved = False
    best_fbcca_weighted_profile_saved = False
    best_weighted_profile = None
    weighted_profile_aggregation_summary: dict[str, Any] = {
        "mode": str(config.weight_aggregation),
        "source": "no_weighted_runs",
        "note": "No weighted FBCCA runs were available for aggregation.",
    }
    best_candidate_profile = profile_by_run.get(deployment_run_key)
    if best_candidate_profile is not None:
        best_candidate_profile = _profile_for_result(
            best_candidate_profile,
            deployment_run,
            prefer_cross_session=prefer_cross_session,
            decision_time_mode=decision_time_mode,
            async_decision_time_mode=async_decision_time_mode,
        )
        save_profile(best_candidate_profile, best_candidate_profile_path)
        best_candidate_profile_saved = True
        log(f"Best candidate profile saved: {best_candidate_profile_path}")

    fbcca_weighted_runs = [dict(item) for item in all_successful_ranked if _is_fbcca_weight_result(dict(item))]
    weighted_family_summary = (
        _aggregate_family_results(
            fbcca_weighted_runs,
            ranking_policy=ranking_policy,
            prefer_cross_session=prefer_cross_session,
        )
        if fbcca_weighted_runs
        else {"families": []}
    )
    weighted_family_rows = list(weighted_family_summary.get("families", []))
    best_weighted_family_name = (
        str(weighted_family_rows[0].get("model_name", "")) if weighted_family_rows else ""
    )
    best_weighted_family_runs = [
        dict(item)
        for item in fbcca_weighted_runs
        if str(item.get("model_name", "")) == best_weighted_family_name
    ]
    aggregated_weighted_profile, aggregated_weight_summary = _build_aggregated_fbcca_profile(
        weighted_runs=best_weighted_family_runs,
        profile_by_run=profile_by_run,
        prefer_cross_session=prefer_cross_session,
        decision_time_mode=decision_time_mode,
        async_decision_time_mode=async_decision_time_mode,
        aggregation_mode=str(config.weight_aggregation),
    )
    if aggregated_weighted_profile is not None:
        best_weighted_profile = aggregated_weighted_profile
        save_profile(best_weighted_profile, best_fbcca_weighted_profile_path)
        best_fbcca_weighted_profile_saved = True
        weighted_profile_aggregation_summary = {
            "mode": str(config.weight_aggregation),
            "source": "aggregated_weighted_family",
            "weighted_family_ranking": list(weighted_family_rows),
            "selected_family": str(best_weighted_family_name),
            "aggregated_profile_summary": dict(aggregated_weight_summary or {}),
        }
        log(
            "Best FBCCA weighted profile saved: "
            f"{best_fbcca_weighted_profile_path} family={best_weighted_family_name} "
            f"runs={len(best_weighted_family_runs)} aggregation={config.weight_aggregation}"
        )
    elif fbcca_weighted_runs:
        best_fbcca_weighted_run = fbcca_weighted_runs[0]
        best_weighted_profile = profile_by_run.get(_run_key(best_fbcca_weighted_run))
        if best_weighted_profile is not None:
            best_weighted_profile = _profile_for_result(
                best_weighted_profile,
                best_fbcca_weighted_run,
                prefer_cross_session=prefer_cross_session,
                decision_time_mode=decision_time_mode,
                async_decision_time_mode=async_decision_time_mode,
            )
            save_profile(best_weighted_profile, best_fbcca_weighted_profile_path)
            best_fbcca_weighted_profile_saved = True
            weighted_profile_aggregation_summary = {
                "mode": str(config.weight_aggregation),
                "source": "single_best_weighted_run_fallback",
                "selected_family": str(best_fbcca_weighted_run.get("model_name", "")),
                "aggregated_profile_summary": None,
            }
            log(f"Best FBCCA weighted profile saved: {best_fbcca_weighted_profile_path}")

    profile_saved = False
    chosen_profile_path: Optional[str] = None
    if bool(deployment_run.get("meets_acceptance", False)):
        chosen_profile = best_candidate_profile
        if chosen_profile is not None:
            save_profile(chosen_profile, config.output_profile_path)
            profile_saved = True
            chosen_profile_path = str(config.output_profile_path)
    if not profile_saved:
        log(
            "No deployment profile saved. "
            f"Winning family={winning_model_family} deployment_run="
            f"{deployment_run.get('channel_mode')}/{deployment_run.get('eval_seed')} "
            "did not meet acceptance thresholds."
        )
    profile_for_realtime_source: Optional[Path] = None
    profile_for_realtime_type = "none"
    profile_for_realtime_object = None
    if (
        str(getattr(config, "task", "")).strip().lower() in {"fbcca-weighted-compare", "fbcca-weights"}
        and best_fbcca_weighted_profile_saved
    ):
        profile_for_realtime_source = best_fbcca_weighted_profile_path
        profile_for_realtime_type = "best_fbcca_weighted"
        profile_for_realtime_object = best_weighted_profile
    elif best_candidate_profile_saved:
        profile_for_realtime_source = best_candidate_profile_path
        profile_for_realtime_type = "best_candidate"
        profile_for_realtime_object = best_candidate_profile
    elif profile_saved and chosen_profile_path:
        profile_for_realtime_source = Path(chosen_profile_path).expanduser().resolve()
        profile_for_realtime_type = "deployment_profile"
        profile_for_realtime_object = best_candidate_profile
    roundtrip_summary: dict[str, Any] = {}
    roundtrip_ready = False
    if profile_for_realtime_source is not None and profile_for_realtime_object is not None:
        try:
            roundtrip_summary = _run_profile_roundtrip_check(
                saved_profile_path=profile_for_realtime_source,
                source_profile=profile_for_realtime_object,
                sampling_rate=int(fs),
                gpu_device=int(config.gpu_device),
                gpu_precision=str(config.gpu_precision),
                gpu_warmup=bool(config.gpu_warmup),
                gpu_cache_policy=str(config.gpu_cache_policy),
            )
            roundtrip_ready = bool(roundtrip_summary.get("ready", False))
            log(
                f"Profile roundtrip check: source={profile_for_realtime_type} ready={roundtrip_ready} "
                f"path={profile_for_realtime_source}"
            )
        except Exception as exc:
            roundtrip_summary = {
                "ready": False,
                "error": str(exc),
                "saved_profile_path": str(profile_for_realtime_source),
            }
            log(f"Profile roundtrip check failed: {exc}")

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
    total_duration_s = float(max(time.perf_counter() - progress_started_at, 0.0))
    runtime_backend_summary = {
        "requested_backend": str(config.compute_backend),
        "used_backend": str(deployment_run.get("compute_backend_used", config.compute_backend)),
        "gpu_device": int(config.gpu_device),
        "precision": str(config.gpu_precision),
        "gpu_warmup": bool(config.gpu_warmup),
        "gpu_cache_policy": str(config.gpu_cache_policy),
        "backend_timing_summary": dict(deployment_run.get("backend_timing_summary", {})),
        "stage_a_duration_s": float(stage_a_duration_s),
        "stage_b_duration_s": float(stage_b_duration_s),
        "total_duration_s": float(total_duration_s),
    }
    compute_cost_summary = {
        "requested_model_count": int(len(requested_models)),
        "stage_a_model_count": int(len(stage_a_candidates)),
        "stage_b_model_count": int(len(stage_b_candidates)),
        "stage_b_total_runs": int(total_runs),
        "stage_b_successful_runs": int(len(all_successful_ranked)),
        "stage_b_failed_runs": int(len(failed_stage_b_runs)),
        "stage_a_duration_s": float(stage_a_duration_s),
        "stage_b_duration_s": float(stage_b_duration_s),
        "total_duration_s": float(total_duration_s),
    }
    fbcca_weight_table = [
        _weight_table_row(item, prefer_cross_session=prefer_cross_session)
        for item in fbcca_weighted_runs
    ]
    model_compare_table = [
        _model_compare_row(item, prefer_cross_session=prefer_cross_session)
        for item in best_runs_by_model
    ]

    report_payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "offline_train_eval",
        "task": str(getattr(config, "task", "")),
        "report_path": str(report_json_path),
        "report_dir": str(report_dir),
        "selection_snapshot_path": str(selection_snapshot_path),
        "run_config_path": str(run_config_path),
        "run_log_path": str(run_log_path),
        "progress_snapshot_path": str(progress_snapshot_path),
        "compute_backend_requested": str(config.compute_backend),
        "compute_backend_used": str(deployment_run.get("compute_backend_used", config.compute_backend)),
        "gpu_device": int(config.gpu_device),
        "precision": str(config.gpu_precision),
        "runtime_backend_summary": dict(runtime_backend_summary),
        "backend_timing_summary": dict(deployment_run.get("backend_timing_summary", {})),
        "compute_cost_summary": dict(compute_cost_summary),
        "compute_summary": {
            "runtime_backend_summary": dict(runtime_backend_summary),
            "compute_cost_summary": dict(compute_cost_summary),
        },
        "dataset_manifest_session1": str(config.dataset_manifest_session1),
        "selected_dataset_manifests_session1": [str(path) for path in selected_session1_manifest_paths],
        "selected_dataset_count_session1": int(len(selected_session1_manifest_paths)),
        "dataset_manifest_session2": (
            None if config.dataset_manifest_session2 is None else str(config.dataset_manifest_session2)
        ),
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
        "evaluation_mode": str(evaluation_mode),
        "quick_screen_top_k": int(quick_screen_top_k),
        "force_include_models": list(force_include_models),
        "quick_screen_ranking": [
            {
                "rank": int(item.get("quick_screen_rank", 0) or 0),
                "model_name": str(item.get("model_name", "")),
                "metrics": dict(item.get("classifier_only_metrics", {})),
                "classifier_only_metrics": dict(item.get("classifier_only_metrics", {})),
                "quick_screen_metrics": dict(item.get("quick_screen_metrics", {})),
                "quick_screen_threshold_fit": dict(item.get("quick_screen_threshold_fit", {})),
                "best_win_sec": float(item.get("best_win_sec", 0.0) or 0.0),
                "quick_screen_pass": bool(item.get("quick_screen_pass", False)),
            }
            for item in quick_screen_ranked
        ],
        "stage_b_candidates": list(stage_b_candidates),
        "stage_a_only_models": list(stage_a_only_models),
        "model_results": best_runs_by_model,
        "stage_b_run_results": all_successful_ranked + failed_stage_b_runs,
        "winning_model_family": str(winning_model_family),
        "family_rank_end_to_end": list(family_rank_end_to_end),
        "deployment_run": {
            "model_name": str(deployment_run.get("model_name", "")),
            "channel_mode": str(deployment_run.get("channel_mode", "")),
            "eval_seed": int(deployment_run.get("eval_seed", 0) or 0),
            "rank": int(deployment_run.get("deployment_rank", 0) or 0),
            "ranking_source": str(deployment_views.get("source", ranking_source)),
            "metrics": dict(deployment_views.get("metrics", {})),
            "meets_acceptance": bool(deployment_run.get("meets_acceptance", False)),
            "frontend_summary": dict(deployment_run.get("frontend_summary", {})),
        },
        "deployment_run_rank": list(deployment_run_rank),
        "chosen_model": str(deployment_run.get("model_name", "")),
        "chosen_rank": int(deployment_run.get("deployment_rank", 0) or 0),
        "chosen_metrics": dict(deployment_views.get("metrics", {})),
        "chosen_async_metrics": dict(deployment_views.get("async_metrics", {})),
        "chosen_metrics_4class": dict(deployment_views.get("metrics_4class", {})),
        "chosen_metrics_2class": dict(deployment_views.get("metrics_2class", {})),
        "chosen_metrics_5class": deployment_views.get("metrics_5class"),
        "chosen_fixed_window_metrics": dict(deployment_run.get("fixed_window_metrics", {})),
        "chosen_fixed_window_async_metrics": dict(deployment_run.get("fixed_window_async_metrics", {})),
        "chosen_fixed_window_metrics_4class": dict(deployment_run.get("fixed_window_metrics_4class", {})),
        "chosen_fixed_window_metrics_2class": dict(deployment_run.get("fixed_window_metrics_2class", {})),
        "chosen_fixed_window_metrics_5class": (
            None
            if deployment_run.get("fixed_window_metrics_5class") is None
            else dict(deployment_run.get("fixed_window_metrics_5class", {}))
        ),
        "chosen_dynamic_delta": dict(deployment_run.get("dynamic_delta", {})),
        "chosen_frontend_summary": dict(deployment_run.get("frontend_summary", {})),
        "timing_breakdown": dict(deployment_run.get("timing_breakdown", {})),
        "warmup_overhead_ms": float(deployment_run.get("warmup_overhead_ms", 0.0) or 0.0),
        "chosen_profile_path": chosen_profile_path,
        "profile_for_realtime_path": (
            None if profile_for_realtime_source is None else str(Path(profile_for_realtime_source).expanduser().resolve())
        ),
        "profile_for_realtime_type": str(profile_for_realtime_type),
        "roundtrip_ready": bool(roundtrip_ready),
        "roundtrip_summary": dict(roundtrip_summary),
        "chosen_meets_acceptance": bool(deployment_run.get("meets_acceptance", False)),
        "accepted_model_count": int(sum(1 for item in best_runs_by_model if bool(item.get("meets_acceptance", False)))),
        "recommended_model": str(winning_model_family),
        "recommended_rank": int(winning_family.get("rank", 0) or 0),
        "recommended_metrics": dict(winning_family.get("metrics_mean", {})),
        "classifier_only_top_model": str(classifier_only_top.get("model_name", "")),
        "classifier_only_top_rank": int(classifier_only_top.get("rank", 0) or 0),
        "classifier_only_top_metrics": dict(classifier_only_top.get("metrics", {})),
        "ranking_boards": {
            "end_to_end": list(ranking_end_to_end),
            "classifier_only": list(classifier_only_ranked),
        },
        "stats_baseline_model": str(stats_baseline.get("model_name", "")),
        "ab_comparisons": _build_ab_comparisons(best_runs_by_model),
        "profile_saved": bool(profile_saved),
        "best_candidate_profile_path": str(best_candidate_profile_path) if best_candidate_profile_saved else None,
        "best_candidate_profile_saved": bool(best_candidate_profile_saved),
        "best_fbcca_weighted_profile_path": (
            str(best_fbcca_weighted_profile_path) if best_fbcca_weighted_profile_saved else None
        ),
        "best_fbcca_weighted_profile_saved": bool(best_fbcca_weighted_profile_saved),
        "default_profile_saved": bool(profile_saved),
        "atomic_write_completed": True,
        "fbcca_weight_table": list(fbcca_weight_table),
        "model_compare_table": list(model_compare_table),
        "fbcca_weighting_scheme": "separable_channel_weights_8_plus_global_subband_weights_5",
        "weight_definition_notes": _weight_definition_notes(),
        "metric_scope": str(metric_scope),
        "decision_time_mode": str(decision_time_mode),
        "async_decision_time_mode": str(async_decision_time_mode),
        "data_policy": str(data_policy),
        "protocol_signature_expected": str(expected_protocol_signature),
        "excluded_sessions": list(protocol_consistency_summary.get("excluded_sessions", [])),
        "evaluation_data_relation": "cross_session_eval" if dataset_session2 is not None else "internal_holdout",
        "same_session_profile_eval": False,
        "cross_session_eval": bool(dataset_session2 is not None),
        "long_idle_used": bool(session1_has_long_idle or session2_has_long_idle),
        "ranking_policy": str(ranking_policy),
        "ranking_source": str(ranking_source),
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
        "seed_summary": _seed_summary_from_runs(all_successful_ranked),
        "seed_level_results": all_successful_ranked + failed_stage_b_runs,
        "aggregated_model_ranking": list(family_rank_end_to_end),
        "classifier_only_ranking": list(classifier_only_ranked),
        "end_to_end_ranking": list(ranking_end_to_end),
        "fbcca_weight_aggregation": dict(weighted_profile_aggregation_summary),
        "idle_false_positive_events": list(deployment_views.get("async_metrics", {}).get("idle_false_positive_events", []) or []),
        "control_state_mode": str(config.control_state_mode),
        "realtime_profile_recommendation": {
            "recommended": bool(
                roundtrip_ready
                and profile_for_realtime_object is not None
                and bool(
                    getattr(
                        profile_for_realtime_object,
                        "recommended_for_realtime",
                        deployment_run.get("meets_acceptance", False),
                    )
                )
            ),
            "profile_path": (
                None
                if profile_for_realtime_source is None
                else str(Path(profile_for_realtime_source).expanduser().resolve())
            ),
            "profile_type": str(profile_for_realtime_type),
            "reason": (
                "accepted_and_roundtrip_ready"
                if bool(
                    roundtrip_ready
                    and profile_for_realtime_object is not None
                    and bool(
                        getattr(
                            profile_for_realtime_object,
                            "recommended_for_realtime",
                            deployment_run.get("meets_acceptance", False),
                        )
                    )
                )
                else "not_accepted_or_roundtrip_not_ready"
            ),
        },
    }
    report_payload["async_metrics"] = dict(report_payload.get("chosen_async_metrics", {}))
    report_payload["metrics_4class"] = dict(report_payload.get("chosen_metrics_4class", {}))
    report_payload["metrics_2class"] = dict(report_payload.get("chosen_metrics_2class", {}))
    report_payload["metrics_5class"] = report_payload.get("chosen_metrics_5class")
    report_payload["paper_lens_metrics"] = dict(deployment_views.get("paper_lens_metrics", {}))
    report_payload["async_lens_metrics"] = dict(deployment_views.get("async_lens_metrics", {}))
    named_stem = _named_report_stem(
        subject_id=str(dataset_session1.subject_id),
        recommended_model=str(winning_model_family),
        deployed_model=str(deployment_run.get("model_name", "")),
        channel_mode=str(deployment_run.get("channel_mode", "")),
        eval_seed=int(deployment_run.get("eval_seed", 0) or 0),
    )
    named_report_json_path = report_dir / f"{named_stem}.json"
    named_report_md_path = report_dir / f"{named_stem}.md"
    named_profile_path = report_dir / (
        "profile"
        f"__model-{_sanitize_path_token(str(deployment_run.get('model_name', '')), fallback='model')}"
        f"__m-{_sanitize_path_token(str(deployment_run.get('channel_mode', '')), fallback='mode')}"
        f"__s-{int(deployment_run.get('eval_seed', 0) or 0)}.json"
    )
    report_payload["named_artifacts"] = {
        "report_json": str(named_report_json_path),
        "report_md": str(named_report_md_path),
        "profile_json": str(named_profile_path),
        "best_candidate_profile": str(best_candidate_profile_path) if best_candidate_profile_saved else "",
        "best_fbcca_weighted_profile": (
            str(best_fbcca_weighted_profile_path) if best_fbcca_weighted_profile_saved else ""
        ),
        "profile_for_realtime": (
            "" if profile_for_realtime_source is None else str(Path(profile_for_realtime_source).expanduser().resolve())
        ),
        "default_profile": str(config.output_profile_path) if profile_saved else "",
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
        "task": str(getattr(config, "task", "")),
        "evaluation_mode": str(evaluation_mode),
        "stage_b_candidates": list(stage_b_candidates),
        "stage_a_only_models": list(stage_a_only_models),
        "named_artifacts": dict(report_payload.get("named_artifacts", {})),
    }

    _write_json_atomic(run_config_path, config_payload)
    _write_json_atomic(selection_snapshot_path, selection_snapshot_payload)
    _write_json_atomic(report_json_path, report_payload)
    markdown_path = report_json_path.with_suffix(".md")
    markdown_body = _render_training_eval_markdown(report_payload)
    _write_text_atomic(markdown_path, markdown_body)
    _write_json_atomic(named_report_json_path, report_payload)
    _write_text_atomic(named_report_md_path, markdown_body)
    source_profile_file: Optional[Path] = None
    if profile_for_realtime_source is not None and profile_for_realtime_source.exists():
        source_profile_file = Path(profile_for_realtime_source).expanduser().resolve()
    elif best_candidate_profile_saved:
        source_profile_file = best_candidate_profile_path
    elif profile_saved and chosen_profile_path:
        source_profile_file = Path(chosen_profile_path).expanduser().resolve()
    if source_profile_file is not None and source_profile_file.exists():
        atomic_copy_text_file(source_profile_file, named_profile_path)
    emit_progress(
        force=True,
        stage="complete",
        model_name=str(deployment_run.get("model_name", "")),
        run_index=int(total_runs),
        run_total=int(total_runs),
        config_index=0,
        config_total=0,
        last_message="Training evaluation completed",
    )
    log(f"Report saved: {report_json_path}")
    log(f"Named report saved: {named_report_json_path}")
    log(f"Markdown saved: {markdown_path}")
    log(f"Named markdown saved: {named_report_md_path}")
    log(f"Selection snapshot saved: {selection_snapshot_path}")
    log(f"Run config saved: {run_config_path}")
    log(f"Run log saved: {run_log_path}")
    log(f"Progress snapshot saved: {progress_snapshot_path}")
    if profile_saved:
        log(f"Profile saved: {config.output_profile_path}")
    if named_profile_path.exists():
        log(f"Named profile saved: {named_profile_path}")
    if profile_saved:
        log(
            f"Realtime profile source: {profile_for_realtime_type} -> "
            f"{profile_for_realtime_source if profile_for_realtime_source is not None else config.output_profile_path}"
        )
    else:
        log("Profile not saved because winning deployment run did not meet acceptance thresholds.")
        if best_candidate_profile_saved:
            log(f"Reusable best candidate profile kept in report dir: {best_candidate_profile_path}")
        if best_fbcca_weighted_profile_saved:
            log(f"Reusable best FBCCA weighted profile kept in report dir: {best_fbcca_weighted_profile_path}")
        if profile_for_realtime_source is not None:
            log(f"Realtime profile source kept in report dir: {profile_for_realtime_source}")
    atomic_write_text(run_log_path, "\n".join(run_log_lines).strip() + ("\n" if run_log_lines else ""))
    return report_payload
