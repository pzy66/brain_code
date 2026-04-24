from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import product
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from . import tdca_local_opt as _tdca
from .async_fbcca_idle_standalone import (
    DEFAULT_ASYNC_DECISION_TIME_MODE,
    DEFAULT_CHANNEL_WEIGHT_L2,
    DEFAULT_CONTROL_STATE_MODE,
    DEFAULT_DYNAMIC_STOP_ALPHA,
    DEFAULT_GATE_POLICY,
    DEFAULT_GPU_CACHE_MODE,
    DEFAULT_GPU_DEVICE_ID,
    DEFAULT_GPU_PRECISION_NAME,
    DEFAULT_IDLE_FP_HARD_TH,
    DEFAULT_NH,
    DEFAULT_PAPER_DECISION_TIME_MODE,
    DEFAULT_SPATIAL_RANK_CANDIDATES,
    DEFAULT_SPATIAL_SOURCE_MODEL,
    DEFAULT_SUBBAND_PRIOR_STRENGTH,
    FBCCA_VARIANT_SPECS,
    ThresholdProfile,
    atomic_write_text,
    json_dumps,
    json_safe,
    model_implementation_level,
    model_method_note,
    optimize_fbcca_frontend_weights,
    save_profile,
)
from .profile_v2 import DEFAULT_GATE_FEATURES, build_profile_v2
from .run_artifacts import make_run_tag, publish_deployed_profile, resolve_ssvep_run_artifacts


DEFAULT_FBCCA_LOCAL_MODEL = "fbcca"
DEFAULT_FBCCA_LOCAL_CHANNEL_MODE = "all8"
DEFAULT_FBCCA_LOCAL_WIN_CANDIDATES = (2.0, 2.5, 3.0, 3.5)
DEFAULT_FBCCA_LOCAL_STEP_SEC = 0.25
DEFAULT_FBCCA_LOCAL_REPEAT_COUNT = 5
DEFAULT_FBCCA_LOCAL_TOP_K = 8
DEFAULT_FBCCA_LOCAL_SEARCH_PRESET = "reduced40"
FBCCA_LOCAL_SEARCH_PRESETS = ("smoke20", "reduced40")
DEFAULT_FBCCA_LOCAL_VARIANTS = (
    "fbcca_fixed_all8",
    "fbcca_cw_all8",
    "fbcca_sw_all8",
    "fbcca_cw_sw_all8",
    "fbcca_cw_sw_trca_shared",
)
DEFAULT_BASELINE_ROWS = (
    "legacy_fbcca_202603",
    "fbcca_fixed_all8",
    "trca_r",
)
DEFAULT_BASELINE_WIN_SEC = 3.0
DEFAULT_FBCCA_LOCAL_DEPLOYMENT_GRADE = "provisional_single_session"
DEFAULT_FRONTEND_WEIGHT_CV_FOLDS = 3
DEFAULT_FRONTEND_JOINT_WEIGHT_ITERS = 2
DEFAULT_FRONTEND_DYNAMIC_STOP_ENABLED = False
DEFAULT_PROMOTION_IDLE_FP_PER_MIN = 1.0
DEFAULT_PROMOTION_CONTROL_RECALL = 0.75
DEFAULT_PROMOTION_RELEASE_LATENCY_S = 4.5
DEFAULT_PROMOTION_SWITCH_LATENCY_S = 4.5

FBCCA_VARIANT_METADATA: dict[str, dict[str, Any]] = {
    "fbcca_fixed_all8": {
        "algorithm_alignment": "paper-faithful",
        "channel_weight_mode": None,
        "subband_weight_mode": "chen_fixed",
        "spatial_filter_mode": None,
    },
    "fbcca_cw_all8": {
        "algorithm_alignment": "engineering-approx",
        "channel_weight_mode": "fbcca_diag",
        "subband_weight_mode": "chen_fixed",
        "spatial_filter_mode": None,
    },
    "fbcca_sw_all8": {
        "algorithm_alignment": "engineering-approx",
        "channel_weight_mode": None,
        "subband_weight_mode": "chen_ab_subject",
        "spatial_filter_mode": None,
    },
    "fbcca_cw_sw_all8": {
        "algorithm_alignment": "engineering-approx",
        "channel_weight_mode": "fbcca_diag",
        "subband_weight_mode": "chen_ab_subject",
        "spatial_filter_mode": None,
    },
    "fbcca_cw_sw_trca_shared": {
        "algorithm_alignment": "engineering-approx",
        "channel_weight_mode": "fbcca_diag",
        "subband_weight_mode": "chen_ab_subject",
        "spatial_filter_mode": "trca_shared",
    },
}

FBCCA_VARIANT_TIEBREAK_PRIORITY = {
    "fbcca_fixed_all8": 0,
    "fbcca_sw_all8": 1,
    "fbcca_cw_all8": 2,
    "fbcca_cw_sw_all8": 3,
    "fbcca_cw_sw_trca_shared": 4,
}


RepeatedGroupSplit = _tdca.RepeatedGroupSplit
MergedLocalDataset = _tdca.MergedLocalDataset
backfill_manifest_trial_roles = _tdca.backfill_manifest_trial_roles
build_repeated_group_splits = _tdca.build_repeated_group_splits
preflight_fbcca_local_env = _tdca.preflight_tdca_local_env

_safe_float = _tdca._safe_float
_safe_int = _tdca._safe_int
_median = _tdca._median
_rank_metrics_key = _tdca._rank_metrics_key
_candidate_rank_tuple = _tdca._candidate_rank_tuple
_sanitize_report_rows = _tdca._sanitize_report_rows
_aggregate_metric_bundle = _tdca._aggregate_metric_bundle
_resolved_confidence_variant = _tdca._resolved_confidence_variant
_build_confidence_diagnostics_board_raw = _tdca._build_confidence_diagnostics_board
_build_decision_bottleneck_summary_raw = _tdca._build_decision_bottleneck_summary
_build_error_attribution_board_raw = _tdca._build_error_attribution_board
_decision_param_grid = _tdca._decision_param_grid
_make_decision_params_key = _tdca._make_decision_params_key


@dataclass(frozen=True)
class FBCCALocalOptConfig:
    dataset_manifest_session1: Path
    output_profile_path: Path
    report_path: Path
    dataset_manifests: tuple[Path, ...] = ()
    report_root_dir: Optional[Path] = None
    organize_report_dir: bool = False
    model_names: tuple[str, ...] = (DEFAULT_FBCCA_LOCAL_MODEL,)
    channel_modes: tuple[str, ...] = (DEFAULT_FBCCA_LOCAL_CHANNEL_MODE,)
    multi_seed_count: int = DEFAULT_FBCCA_LOCAL_REPEAT_COUNT
    win_candidates: tuple[float, ...] = DEFAULT_FBCCA_LOCAL_WIN_CANDIDATES
    search_preset: str = DEFAULT_FBCCA_LOCAL_SEARCH_PRESET
    step_sec: float = DEFAULT_FBCCA_LOCAL_STEP_SEC
    Nh: int = DEFAULT_NH
    seed: int = 20260410
    compute_backend: str = "auto"
    gpu_device: int = DEFAULT_GPU_DEVICE_ID
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME
    gpu_warmup: bool = True
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE
    control_state_mode: str = DEFAULT_CONTROL_STATE_MODE
    decision_time_mode: str = DEFAULT_PAPER_DECISION_TIME_MODE
    async_decision_time_mode: str = DEFAULT_ASYNC_DECISION_TIME_MODE
    progress_heartbeat_sec: float = 5.0


def _fbcca_variant_metadata(fbcca_variant: Optional[str]) -> dict[str, Any]:
    variant = str(fbcca_variant or DEFAULT_FBCCA_LOCAL_VARIANTS[0]).strip().lower()
    if variant not in FBCCA_VARIANT_METADATA:
        raise ValueError(f"unsupported fbcca variant: {fbcca_variant}")
    payload = dict(FBCCA_VARIANT_METADATA[variant])
    payload["fbcca_variant"] = str(variant)
    payload["method_note"] = str(model_method_note(variant))
    return payload


def _fbcca_variant_priority(fbcca_variant: Optional[str]) -> int:
    variant = str(fbcca_variant or DEFAULT_FBCCA_LOCAL_VARIANTS[0]).strip().lower()
    return int(FBCCA_VARIANT_TIEBREAK_PRIORITY.get(variant, len(FBCCA_VARIANT_TIEBREAK_PRIORITY)))


def _candidate_key(*, fbcca_variant: str, win_sec: float, confidence_variant: str) -> str:
    return (
        f"variant={str(fbcca_variant).strip().lower()}|"
        f"win={float(win_sec):g}|"
        f"confidence={str(confidence_variant).strip().lower()}"
    )


def _normalize_name_list(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(item).strip().lower() for item in values if str(item).strip())


def _validate_fbcca_local_config(config: FBCCALocalOptConfig) -> None:
    model_names = _normalize_name_list(config.model_names)
    channel_modes = _normalize_name_list(config.channel_modes)
    if model_names != (DEFAULT_FBCCA_LOCAL_MODEL,):
        raise ValueError(f"fbcca-local-opt only supports model_names=('fbcca',); got {config.model_names}")
    if channel_modes != (DEFAULT_FBCCA_LOCAL_CHANNEL_MODE,):
        raise ValueError(f"fbcca-local-opt only supports channel_modes=('all8',); got {config.channel_modes}")
    preset = str(config.search_preset or DEFAULT_FBCCA_LOCAL_SEARCH_PRESET).strip().lower()
    if preset not in FBCCA_LOCAL_SEARCH_PRESETS:
        raise ValueError(
            f"fbcca-local-opt only supports search_preset in {FBCCA_LOCAL_SEARCH_PRESETS}; got {config.search_preset}"
        )


def _default_model_params(
    *,
    Nh: int,
    fbcca_variant: str,
    frontend_model_params: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    metadata = _fbcca_variant_metadata(fbcca_variant)
    params: dict[str, Any] = {
        "Nh": int(Nh),
        "fbcca_variant": str(metadata["fbcca_variant"]),
        "_decoder_model_name": str(metadata["fbcca_variant"]),
    }
    if metadata.get("channel_weight_mode") is not None:
        params["channel_weight_mode"] = str(metadata["channel_weight_mode"])
    if metadata.get("subband_weight_mode") is not None:
        params["subband_weight_mode"] = str(metadata["subband_weight_mode"])
    if metadata.get("spatial_filter_mode") is not None:
        params["spatial_filter_mode"] = str(metadata["spatial_filter_mode"])
    if frontend_model_params:
        params.update(dict(frontend_model_params))
    params["fbcca_variant"] = str(metadata["fbcca_variant"])
    params["_decoder_model_name"] = str(metadata["fbcca_variant"])
    return params


def _baseline_candidate(model_name: str, *, Nh: int) -> dict[str, Any]:
    normalized = str(model_name).strip().lower()
    if normalized == "fbcca_fixed_all8":
        return {
            "baseline_name": "fbcca_fixed_all8 @ win=3.0",
            "model_name": DEFAULT_FBCCA_LOCAL_MODEL,
            "fbcca_variant": "fbcca_fixed_all8",
            "win_sec": float(DEFAULT_BASELINE_WIN_SEC),
            "model_params": _default_model_params(Nh=int(Nh), fbcca_variant="fbcca_fixed_all8"),
        }
    if normalized == "legacy_fbcca_202603":
        return {
            "baseline_name": "legacy_fbcca_202603 @ win=3.0",
            "model_name": "legacy_fbcca_202603",
            "fbcca_variant": "legacy_fbcca_202603",
            "win_sec": float(DEFAULT_BASELINE_WIN_SEC),
            "model_params": {"Nh": int(Nh)},
        }
    if normalized == "trca_r":
        return {
            "baseline_name": "trca_r @ win=3.0",
            "model_name": "trca_r",
            "fbcca_variant": "n/a",
            "win_sec": float(DEFAULT_BASELINE_WIN_SEC),
            "model_params": {"Nh": int(Nh)},
        }
    raise ValueError(f"unsupported FBCCA baseline: {model_name}")


def _resolve_search_plan(config: FBCCALocalOptConfig) -> dict[str, Any]:
    preset = str(config.search_preset or DEFAULT_FBCCA_LOCAL_SEARCH_PRESET).strip().lower()
    custom_override = bool(
        tuple(float(item) for item in config.win_candidates)
        != tuple(float(item) for item in DEFAULT_FBCCA_LOCAL_WIN_CANDIDATES)
        or int(config.multi_seed_count) != int(DEFAULT_FBCCA_LOCAL_REPEAT_COUNT)
    )
    if custom_override:
        return {
            "search_preset": "custom",
            "repeats": int(config.multi_seed_count),
            "candidate_grid": [
                {
                    "fbcca_variant": str(fbcca_variant),
                    "confidence_variant": str(confidence_variant),
                    "win_sec": float(win_sec),
                }
                for fbcca_variant, confidence_variant, win_sec in product(
                    DEFAULT_FBCCA_LOCAL_VARIANTS,
                    _tdca.DEFAULT_CONFIDENCE_VARIANTS,
                    tuple(float(item) for item in config.win_candidates),
                )
            ],
        }
    if preset == "smoke20":
        win_values = (2.5, 3.0)
        repeats = 1
    else:
        win_values = DEFAULT_FBCCA_LOCAL_WIN_CANDIDATES
        repeats = DEFAULT_FBCCA_LOCAL_REPEAT_COUNT
    return {
        "search_preset": str(preset),
        "repeats": int(repeats),
        "candidate_grid": [
            {
                "fbcca_variant": str(fbcca_variant),
                "confidence_variant": str(confidence_variant),
                "win_sec": float(win_sec),
            }
            for fbcca_variant, confidence_variant, win_sec in product(
                DEFAULT_FBCCA_LOCAL_VARIANTS,
                _tdca.DEFAULT_CONFIDENCE_VARIANTS,
                tuple(float(item) for item in win_values),
            )
        ],
    }


def _fbcca_board_sort_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    fbcca_variant = row.get("fbcca_variant") or dict(row.get("candidate", {})).get("fbcca_variant") or DEFAULT_FBCCA_LOCAL_VARIANTS[0]
    confidence_variant = row.get("confidence_variant") or dict(row.get("candidate", {})).get("confidence_variant") or _tdca.DEFAULT_CONFIDENCE_VARIANT
    gate_valid = bool(row.get("gate_calibration_valid", True))
    return (
        0.0 if gate_valid else 1.0,
        *_candidate_rank_tuple(row),
        float(_fbcca_variant_priority(str(fbcca_variant))),
        float(_tdca._confidence_variant_priority(str(confidence_variant))),
    )


def _resolve_report_paths(config: FBCCALocalOptConfig) -> dict[str, Path]:
    artifacts = resolve_ssvep_run_artifacts(
        task="fbcca-local-opt",
        report_path=Path(config.report_path).expanduser().resolve(),
        output_profile_path=Path(config.output_profile_path).expanduser().resolve(),
        organize_report_dir=bool(config.organize_report_dir),
        report_root_dir=(Path(config.report_root_dir).expanduser().resolve() if config.report_root_dir is not None else None),
        run_tag=make_run_tag(task="fbcca-local-opt"),
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


def _fbcca_progress_label(stage: str) -> str:
    return {
        "prepare": "准备",
        "baseline_opening": "基线开场对比",
        "candidate_search": "FBCCA 候选搜索",
        "decision_search": "异步决策搜索",
        "baseline_seal": "基线封板对比",
        "finalize": "保存产物",
        "complete": "完成",
    }.get(str(stage).strip().lower(), str(stage or "处理中"))


def _fbcca_progress_percent(stage: str, *, run_index: int = 0, run_total: int = 0) -> int:
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
    if run_total <= 0:
        return int(round(end))
    ratio = min(max(float(run_index) / float(max(run_total, 1)), 0.0), 1.0)
    return int(round(start + (end - start) * ratio))


def _frontend_requires_learning(fbcca_variant: str) -> bool:
    return str(fbcca_variant).strip().lower() != "fbcca_fixed_all8"


def _frontend_optimization_summary(
    *,
    fbcca_variant: str,
    success: bool,
    metadata: Optional[Mapping[str, Any]],
    error: Optional[str] = None,
) -> dict[str, Any]:
    variant_metadata = _fbcca_variant_metadata(fbcca_variant)
    payload = {
        "fbcca_variant": str(fbcca_variant),
        "train_only": True,
        "dynamic_stop_enabled": False,
        "weight_cv_folds": int(DEFAULT_FRONTEND_WEIGHT_CV_FOLDS),
        "joint_weight_iters": int(DEFAULT_FRONTEND_JOINT_WEIGHT_ITERS),
        "spatial_rank_candidates": [int(value) for value in DEFAULT_SPATIAL_RANK_CANDIDATES],
        "spatial_source_model": str(DEFAULT_SPATIAL_SOURCE_MODEL),
        "channel_weight_l2": float(DEFAULT_CHANNEL_WEIGHT_L2),
        "subband_prior_strength": float(DEFAULT_SUBBAND_PRIOR_STRENGTH),
        "idle_fp_hard_th": float(DEFAULT_IDLE_FP_HARD_TH),
        "channel_weight_mode": variant_metadata.get("channel_weight_mode"),
        "subband_weight_mode": variant_metadata.get("subband_weight_mode"),
        "spatial_filter_mode": variant_metadata.get("spatial_filter_mode"),
        "status": "optimized" if success else ("fixed_default" if not _frontend_requires_learning(fbcca_variant) else "fallback_default"),
        "error": None if error is None else str(error),
    }
    if metadata:
        payload.update(json_safe(dict(metadata)))
    return payload


def _optimize_variant_frontend(
    *,
    merged_dataset: MergedLocalDataset,
    split: RepeatedGroupSplit,
    fbcca_variant: str,
    win_sec: float,
    config: FBCCALocalOptConfig,
    log: Callable[[str], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_params = _default_model_params(Nh=int(config.Nh), fbcca_variant=str(fbcca_variant))
    if not _frontend_requires_learning(fbcca_variant):
        return base_params, _frontend_optimization_summary(
            fbcca_variant=str(fbcca_variant),
            success=False,
            metadata={"mode": "fixed_default", "optimized_model_params": json_safe(base_params)},
        )
    train_segments = _tdca._subset_segments(merged_dataset.trial_segments, split.train_indices)
    if not train_segments:
        raise ValueError("FBCCA frontend optimization requires a non-empty train split")
    try:
        result = optimize_fbcca_frontend_weights(
            train_segments=train_segments,
            gate_segments=train_segments,
            sampling_rate=int(merged_dataset.sampling_rate),
            freqs=merged_dataset.freqs,
            win_sec=float(win_sec),
            step_sec=float(DEFAULT_FBCCA_LOCAL_STEP_SEC),
            min_enter_windows=1,
            min_exit_windows=1,
            gate_policy=DEFAULT_GATE_POLICY,
            dynamic_stop_enabled=bool(DEFAULT_FRONTEND_DYNAMIC_STOP_ENABLED),
            dynamic_stop_alpha=float(DEFAULT_DYNAMIC_STOP_ALPHA),
            channel_weight_mode=base_params.get("channel_weight_mode"),
            subband_weight_mode=base_params.get("subband_weight_mode"),
            spatial_filter_mode=base_params.get("spatial_filter_mode"),
            spatial_rank_candidates=DEFAULT_SPATIAL_RANK_CANDIDATES,
            joint_weight_iters=int(DEFAULT_FRONTEND_JOINT_WEIGHT_ITERS),
            weight_cv_folds=int(DEFAULT_FRONTEND_WEIGHT_CV_FOLDS),
            spatial_source_model=str(DEFAULT_SPATIAL_SOURCE_MODEL),
            idle_fp_hard_th=float(DEFAULT_IDLE_FP_HARD_TH),
            channel_weight_l2=float(DEFAULT_CHANNEL_WEIGHT_L2),
            subband_prior_strength=float(DEFAULT_SUBBAND_PRIOR_STRENGTH),
            control_state_mode=str(config.control_state_mode),
            compute_backend=str(config.compute_backend),
            gpu_device=int(config.gpu_device),
            gpu_precision=str(config.gpu_precision),
            gpu_warmup=bool(config.gpu_warmup),
            gpu_cache_policy=str(config.gpu_cache_policy),
            log_fn=log,
            log_prefix=f"FBCCA[{fbcca_variant}]",
        )
        metadata = dict(result.get("metadata", {}))
        optimized_params = _default_model_params(
            Nh=int(config.Nh),
            fbcca_variant=str(fbcca_variant),
            frontend_model_params=dict(metadata.get("optimized_model_params", {})),
        )
        return optimized_params, _frontend_optimization_summary(
            fbcca_variant=str(fbcca_variant),
            success=True,
            metadata={**metadata, "optimized_model_params": json_safe(optimized_params)},
        )
    except Exception as exc:
        return base_params, _frontend_optimization_summary(
            fbcca_variant=str(fbcca_variant),
            success=False,
            metadata={"optimized_model_params": json_safe(base_params)},
            error=str(exc),
        )


def _build_fbcca_candidate_context(
    *,
    merged_dataset: MergedLocalDataset,
    split: RepeatedGroupSplit,
    fbcca_variant: str,
    win_sec: float,
    confidence_variant: str,
    config: FBCCALocalOptConfig,
    replay_policy: Optional[Mapping[str, Any]] = None,
    log: Optional[Callable[[str], None]] = None,
    frontend_cache: Optional[dict[tuple[int, str, float], tuple[dict[str, Any], dict[str, Any]]]] = None,
) -> dict[str, Any]:
    cache_key = (int(split.repeat_index), str(fbcca_variant), float(win_sec))
    if frontend_cache is not None and cache_key in frontend_cache:
        frontend_params, frontend_summary = frontend_cache[cache_key]
    else:
        frontend_params, frontend_summary = _optimize_variant_frontend(
            merged_dataset=merged_dataset,
            split=split,
            fbcca_variant=str(fbcca_variant),
            win_sec=float(win_sec),
            config=config,
            log=(log if log is not None else (lambda _msg: None)),
        )
        if frontend_cache is not None:
            frontend_cache[cache_key] = (dict(frontend_params), dict(frontend_summary))
    context = _tdca._build_candidate_context(
        merged_dataset=merged_dataset,
        split=split,
        model_name=DEFAULT_FBCCA_LOCAL_MODEL,
        win_sec=float(win_sec),
        model_params=dict(frontend_params),
        confidence_variant=str(confidence_variant),
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
    variant_metadata = _fbcca_variant_metadata(str(fbcca_variant))
    context["model_name"] = DEFAULT_FBCCA_LOCAL_MODEL
    context["fbcca_variant"] = str(fbcca_variant)
    context["algorithm_alignment"] = str(variant_metadata["algorithm_alignment"])
    context["channel_weight_mode"] = variant_metadata.get("channel_weight_mode")
    context["subband_weight_mode"] = variant_metadata.get("subband_weight_mode")
    context["spatial_filter_mode"] = variant_metadata.get("spatial_filter_mode")
    context["method_note"] = str(variant_metadata["method_note"])
    context["frontend_optimization_summary"] = dict(frontend_summary)
    context["model_params"] = {
        **dict(context.get("model_params", {})),
        "fbcca_variant": str(fbcca_variant),
    }
    return context


def _rename_decoder_variant_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    fbcca_variant: str,
    confidence_variant: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload.pop("decoder_variant", None)
        payload["fbcca_variant"] = str(fbcca_variant)
        payload["confidence_variant"] = str(payload.get("confidence_variant", confidence_variant))
        output.append(payload)
    return output


def _build_confidence_diagnostics_board(
    *,
    candidate_row: Mapping[str, Any],
    tune_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = _build_confidence_diagnostics_board_raw(
        candidate_row={
            "candidate_key": str(candidate_row.get("candidate_key", "")),
            "decoder_variant": str(candidate_row.get("fbcca_variant", "")),
            "confidence_variant": str(candidate_row.get("confidence_variant", _tdca.DEFAULT_CONFIDENCE_VARIANT)),
        },
        tune_rows=tune_rows,
        holdout_rows=holdout_rows,
    )
    return _rename_decoder_variant_rows(
        rows,
        fbcca_variant=str(candidate_row.get("fbcca_variant", "")),
        confidence_variant=str(candidate_row.get("confidence_variant", _tdca.DEFAULT_CONFIDENCE_VARIANT)),
    )


def _build_decision_bottleneck_summary(
    *,
    candidate_row: Mapping[str, Any],
    holdout_bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = _build_decision_bottleneck_summary_raw(
        candidate_row={
            "candidate_key": str(candidate_row.get("candidate_key", "")),
            "decoder_variant": str(candidate_row.get("fbcca_variant", "")),
            "confidence_variant": str(candidate_row.get("confidence_variant", _tdca.DEFAULT_CONFIDENCE_VARIANT)),
        },
        holdout_bundles=holdout_bundles,
    )
    payload.pop("decoder_variant", None)
    payload["fbcca_variant"] = str(candidate_row.get("fbcca_variant", ""))
    return payload


def _build_error_attribution_board(
    *,
    candidate_row: Mapping[str, Any],
    holdout_bundles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = _build_error_attribution_board_raw(
        candidate_row={
            "candidate_key": str(candidate_row.get("candidate_key", "")),
            "decoder_variant": str(candidate_row.get("fbcca_variant", "")),
            "confidence_variant": str(candidate_row.get("confidence_variant", _tdca.DEFAULT_CONFIDENCE_VARIANT)),
        },
        holdout_bundles=holdout_bundles,
    )
    return _rename_decoder_variant_rows(
        rows,
        fbcca_variant=str(candidate_row.get("fbcca_variant", "")),
        confidence_variant=str(candidate_row.get("confidence_variant", _tdca.DEFAULT_CONFIDENCE_VARIANT)),
    )


def _build_fbcca_contrast_error_board(
    *,
    candidate_row: Mapping[str, Any],
    candidate_holdout_bundles: Sequence[Mapping[str, Any]],
    reference_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidate_index = _tdca._trial_event_index_by_repeat(candidate_holdout_bundles)
    reference_index = _tdca._trial_event_index_by_repeat(reference_row.get("_holdout_bundles", []) or [])
    output: list[dict[str, Any]] = []
    for key, reference_event in reference_index.items():
        expected = reference_event.get("expected_freq")
        if expected is None:
            continue
        reference_success = reference_event.get("first_correct_latency_s") is not None
        if not reference_success:
            continue
        candidate_event = dict(candidate_index.get(key, {}))
        candidate_success = candidate_event.get("first_correct_latency_s") is not None
        if candidate_success:
            continue
        output.append(
            {
                "repeat_index": int(key[0]),
                "label": str(reference_event.get("label", "")),
                "trial_id": int(_safe_int(reference_event.get("trial_id", -1), -1)),
                "expected_freq": float(_safe_float(expected, 0.0)),
                "switch_trial": bool(reference_event.get("switch_trial", False)),
                "reference_model_name": str(reference_row.get("model_name", "")),
                "reference_fbcca_variant": str(reference_row.get("fbcca_variant", "n/a")),
                "reference_first_correct_latency_s": float(_safe_float(reference_event.get("first_correct_latency_s"), 0.0)),
                "candidate_first_correct_latency_s": None
                if candidate_event.get("first_correct_latency_s") is None
                else float(_safe_float(candidate_event.get("first_correct_latency_s"), 0.0)),
                "candidate_error_type": _tdca._error_type_for_control_event(candidate_event)
                if candidate_event
                else "missing_trial_event",
                "candidate_raw_correct_seen": bool(candidate_event.get("raw_correct_seen", False)),
                "candidate_gate_pass_correct_seen": bool(candidate_event.get("gate_pass_correct_seen", False)),
                "candidate_key": str(candidate_row.get("candidate_key", "")),
                "fbcca_variant": str(candidate_row.get("fbcca_variant", "")),
                "confidence_variant": str(candidate_row.get("confidence_variant", _tdca.DEFAULT_CONFIDENCE_VARIANT)),
            }
        )
    return output


def _run_baseline_suite(
    *,
    merged_dataset: MergedLocalDataset,
    splits: Sequence[RepeatedGroupSplit],
    env_preflight: Mapping[str, Any],
    config: FBCCALocalOptConfig,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    decision_params = _tdca._default_decision_params()
    for baseline_name in DEFAULT_BASELINE_ROWS:
        candidate = _baseline_candidate(baseline_name, Nh=int(config.Nh))
        holdout_bundles: list[dict[str, Any]] = []
        inference_values: list[float] = []
        replay_policies: list[dict[str, Any]] = []
        for split in splits:
            replay_policy = _tdca._split_replay_policy(
                merged_dataset=merged_dataset,
                split=split,
                win_sec=float(candidate["win_sec"]),
                env_preflight=env_preflight,
            )
            context = _tdca._build_candidate_context(
                merged_dataset=merged_dataset,
                split=split,
                model_name=str(candidate["model_name"]),
                win_sec=float(candidate["win_sec"]),
                model_params=dict(candidate["model_params"]),
                confidence_variant=str(_tdca.DEFAULT_CONFIDENCE_VARIANT),
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
            bundle = _tdca._evaluate_structured_rows(
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
        rank_key = [float(item) for item in _rank_metrics_key(async_metrics)]
        policy = replay_policies[0] if replay_policies else {}
        fbcca_variant = str(candidate.get("fbcca_variant", "n/a"))
        if fbcca_variant in FBCCA_VARIANT_METADATA:
            variant_metadata = _fbcca_variant_metadata(fbcca_variant)
            algorithm_alignment = str(variant_metadata["algorithm_alignment"])
            channel_weight_mode = variant_metadata.get("channel_weight_mode")
            subband_weight_mode = variant_metadata.get("subband_weight_mode")
            spatial_filter_mode = variant_metadata.get("spatial_filter_mode")
        else:
            algorithm_alignment = str(model_implementation_level(str(candidate["model_name"])))
            channel_weight_mode = None
            subband_weight_mode = None
            spatial_filter_mode = None
        results.append(
            {
                "baseline_name": str(candidate["baseline_name"]),
                "model_name": str(candidate["model_name"]),
                "fbcca_variant": str(fbcca_variant),
                "algorithm_alignment": str(algorithm_alignment),
                "channel_weight_mode": channel_weight_mode,
                "subband_weight_mode": subband_weight_mode,
                "spatial_filter_mode": spatial_filter_mode,
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
                "rank_key": rank_key,
            }
        )
    results.sort(key=_candidate_rank_tuple)
    return results


def _fbcca_meets_promotion_thresholds(metrics: Mapping[str, Any]) -> bool:
    return bool(
        _safe_float(metrics.get("idle_fp_per_min"), float("inf")) <= float(DEFAULT_PROMOTION_IDLE_FP_PER_MIN)
        and _safe_float(metrics.get("control_recall"), 0.0) >= float(DEFAULT_PROMOTION_CONTROL_RECALL)
        and _safe_float(metrics.get("release_latency_s"), float("inf")) <= float(DEFAULT_PROMOTION_RELEASE_LATENCY_S)
        and _safe_float(metrics.get("switch_latency_s"), float("inf")) <= float(DEFAULT_PROMOTION_SWITCH_LATENCY_S)
    )


def _render_markdown(report_payload: Mapping[str, Any]) -> str:
    async_metrics = dict(report_payload.get("chosen_async_metrics", {}) or {})
    status_reasons = [str(item) for item in report_payload.get("status_reasons", [])]
    lines = [
        "# FBCCA Local Opt",
        "",
        f"- Generated at: `{report_payload.get('generated_at', '')}`",
        f"- Task: `{report_payload.get('task', '')}`",
        f"- Search preset: `{report_payload.get('search_preset', '')}`",
        f"- Chosen model: `{report_payload.get('chosen_model', '')}`",
        f"- FBCCA variant: `{report_payload.get('fbcca_variant', '')}`",
        f"- Confidence variant: `{report_payload.get('confidence_variant', '')}`",
        f"- Decision evidence variant: `{report_payload.get('decision_evidence_variant', '')}`",
        f"- Algorithm alignment: `{report_payload.get('algorithm_alignment', '')}`",
        f"- Channel weight mode: `{report_payload.get('channel_weight_mode', '')}`",
        f"- Subband weight mode: `{report_payload.get('subband_weight_mode', '')}`",
        f"- Spatial filter mode: `{report_payload.get('spatial_filter_mode', '')}`",
        f"- Deployment grade: `{report_payload.get('deployment_grade', '')}`",
        f"- Profile saved: `{report_payload.get('profile_saved', False)}`",
        f"- Run valid for deployment: `{bool(report_payload.get('run_valid_for_deployment', False))}`",
        f"- Status: `{report_payload.get('status', '')}`",
        f"- Status reasons: `{', '.join(status_reasons) if status_reasons else 'none'}`",
        f"- Chosen rationale: `{report_payload.get('chosen_model_rationale', '')}`",
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
        "## Diagnostics",
        "",
        f"- search rows: `{len(report_payload.get('fbcca_search_board', []) or [])}`",
        f"- decision rows: `{len(report_payload.get('decision_search_board', []) or [])}`",
        f"- holdout rows: `{len(report_payload.get('holdout_selection_board', []) or [])}`",
        f"- confidence diagnostics rows: `{len(report_payload.get('confidence_diagnostics_board', []) or [])}`",
        f"- error attribution rows: `{len(report_payload.get('error_attribution_board', []) or [])}`",
        f"- contrast rows: `{len(report_payload.get('contrast_error_board', []) or [])}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_fbcca_local_opt(
    config: FBCCALocalOptConfig,
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
        "task": "fbcca-local-opt",
        "stage": "prepare",
        "stage_label": _fbcca_progress_label("prepare"),
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
            progress_percent = _fbcca_progress_percent(stage_name, run_index=run_index, run_total=run_total)
        progress_state["stage_label"] = str(progress_state.get("stage_label") or _fbcca_progress_label(stage_name))
        progress_state["progress_percent"] = int(max(0, min(100, int(progress_percent))))
        progress_state["elapsed_s"] = float(elapsed_s)
        if 0 < progress_state["progress_percent"] < 100:
            eta_s = elapsed_s * ((100.0 / float(progress_state["progress_percent"])) - 1.0)
            progress_state["eta_s"] = float(max(eta_s, 0.0))
        elif progress_state["progress_percent"] >= 100:
            progress_state["eta_s"] = 0.0
        else:
            progress_state["eta_s"] = None
        atomic_write_text(progress_snapshot_path, json_dumps(json_safe(progress_state)) + "\n")
        if progress_fn is not None:
            progress_fn(dict(progress_state))

    _validate_fbcca_local_config(config)
    run_config_payload = dict(asdict(config))
    run_config_payload["output_profile_path"] = str(resolved_profile_path)
    run_config_payload["resolved_profile_v2_path"] = str(resolved_profile_v2_path)
    run_config_payload["report_path"] = str(Path(report_paths["report_json"]).expanduser().resolve())
    run_config_payload["report_dir"] = str(report_dir)
    run_config_payload["run_tag"] = str(run_tag)
    run_config_payload["dynamic_stop_enabled"] = False
    atomic_write_text(
        Path(report_paths["run_config"]).expanduser().resolve(),
        json_dumps(json_safe(run_config_payload)) + "\n",
    )
    log(f"[fbcca-local-opt] report directory prepared: {report_dir}")
    emit_progress(force=True, stage="prepare", detail=f"运行目录已准备：{report_dir}", run_index=1, run_total=4)

    env_preflight = preflight_fbcca_local_env(
        compute_backend=str(config.compute_backend),
        gpu_device=int(config.gpu_device),
        gpu_precision=str(config.gpu_precision),
    )
    log(
        "[fbcca-local-opt] compute backend prepared: "
        f"requested={config.compute_backend} effective={env_preflight.get('effective_backend', 'cpu')}"
    )
    emit_progress(force=True, stage="prepare", detail="后端预检完成，开始载入数据", run_index=2, run_total=4)

    merged_dataset = _tdca._load_merged_dataset(config)
    emit_progress(force=True, stage="prepare", detail=f"数据已载入：trial={len(merged_dataset.trial_segments)}", run_index=3, run_total=4)

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

    emit_progress(force=True, stage="prepare", detail=f"分组完成：preset={search_plan['search_preset']} repeat={len(splits)}", run_index=4, run_total=4)
    emit_progress(force=True, stage="baseline_opening", detail="开始基线开场对比", run_index=0, run_total=len(DEFAULT_BASELINE_ROWS))
    baseline_opening = _run_baseline_suite(
        merged_dataset=merged_dataset,
        splits=splits,
        env_preflight=env_preflight,
        config=config,
    )
    emit_progress(force=True, stage="baseline_opening", detail="基线开场对比完成", run_index=len(DEFAULT_BASELINE_ROWS), run_total=len(DEFAULT_BASELINE_ROWS))

    candidate_results: dict[str, list[dict[str, Any]]] = {}
    gate_search_by_candidate: dict[str, list[dict[str, Any]]] = {}
    gate_exit_by_candidate: dict[str, list[dict[str, Any]]] = {}
    candidate_context_cache: dict[tuple[str, int], dict[str, Any]] = {}
    frontend_cache: dict[tuple[int, str, float], tuple[dict[str, Any], dict[str, Any]]] = {}
    candidate_grid = [dict(item) for item in search_plan["candidate_grid"]]
    total_candidates = len(candidate_grid) * max(len(splits), 1)
    progress_index = 0
    for split in splits:
        for candidate in candidate_grid:
            progress_index += 1
            key = _candidate_key(
                fbcca_variant=str(candidate["fbcca_variant"]),
                win_sec=float(candidate["win_sec"]),
                confidence_variant=str(candidate.get("confidence_variant", _tdca.DEFAULT_CONFIDENCE_VARIANT)),
            )
            log(f"[fbcca-local-opt] repeat={split.repeat_index + 1}/{len(splits)} candidate={progress_index}/{total_candidates} {key}")
            emit_progress(
                stage="candidate_search",
                detail=f"repeat {split.repeat_index + 1}/{len(splits)} | candidate {progress_index}/{total_candidates} | {key}",
                model_name="fbcca",
                fbcca_variant=str(candidate["fbcca_variant"]),
                confidence_variant=str(candidate.get("confidence_variant", _tdca.DEFAULT_CONFIDENCE_VARIANT)),
                run_index=progress_index,
                run_total=total_candidates,
            )
            replay_policy = _tdca._split_replay_policy(
                merged_dataset=merged_dataset,
                split=split,
                win_sec=float(candidate["win_sec"]),
                env_preflight=env_preflight,
            )
            context = _build_fbcca_candidate_context(
                merged_dataset=merged_dataset,
                split=split,
                fbcca_variant=str(candidate["fbcca_variant"]),
                win_sec=float(candidate["win_sec"]),
                confidence_variant=str(candidate.get("confidence_variant", _tdca.DEFAULT_CONFIDENCE_VARIANT)),
                config=config,
                replay_policy=replay_policy,
                log=log,
                frontend_cache=frontend_cache,
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
                    "training_latency_sec": float(context.get("training_latency_sec", 0.0) or 0.0),
                    "analysis_latency_sec": float(context.get("analysis_latency_sec", 0.0) or 0.0),
                    "effective_raw_window_sec": float(context.get("effective_raw_window_sec", candidate["win_sec"]) or candidate["win_sec"]),
                    "confidence_training_scheme": str(context.get("confidence_training_scheme", _tdca.DEFAULT_CONFIDENCE_TRAINING_SCHEME)),
                    "oof_group_key": str(context.get("oof_group_key", "")),
                    "oof_group_count": int(context.get("oof_group_count", 0) or 0),
                    "sample_weight_mode": str(context.get("sample_weight_mode", "")),
                    "positive_trials": int(context.get("positive_trials", 0) or 0),
                    "negative_trials": int(context.get("negative_trials", 0) or 0),
                    "frontend_optimization_summary": dict(context.get("frontend_optimization_summary", {})),
                }
            )
            gate_search_by_candidate.setdefault(key, []).extend(
                _rename_decoder_variant_rows(
                    [{"repeat_index": int(split.repeat_index), **dict(row)} for row in list(context["gate_search_board"])],
                    fbcca_variant=str(candidate["fbcca_variant"]),
                    confidence_variant=str(candidate.get("confidence_variant", _tdca.DEFAULT_CONFIDENCE_VARIANT)),
                )
            )
            gate_exit_by_candidate.setdefault(key, []).extend(
                _rename_decoder_variant_rows(
                    [{"repeat_index": int(split.repeat_index), **dict(row)} for row in list(context["gate_exit_threshold_board"])],
                    fbcca_variant=str(candidate["fbcca_variant"]),
                    confidence_variant=str(candidate.get("confidence_variant", _tdca.DEFAULT_CONFIDENCE_VARIANT)),
                )
            )

    fbcca_search_board: list[dict[str, Any]] = []
    for key, rows in candidate_results.items():
        bundles = [dict(item.get("holdout_bundle", {})) for item in rows]
        aggregated = _aggregate_metric_bundle(bundles)
        async_metrics = dict(aggregated.get("async_metrics", {}))
        async_metrics["inference_ms"] = _median([item.get("inference_ms") for item in rows], default=float("inf"))
        rank_key = _rank_metrics_key(async_metrics)
        sample_candidate = dict(rows[0].get("candidate", {}))
        variant_metadata = _fbcca_variant_metadata(sample_candidate.get("fbcca_variant"))
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
        fbcca_search_board.append(
            {
                "candidate_key": str(key),
                "candidate": sample_candidate,
                "fbcca_variant": str(variant_metadata["fbcca_variant"]),
                "algorithm_alignment": str(variant_metadata["algorithm_alignment"]),
                "channel_weight_mode": variant_metadata.get("channel_weight_mode"),
                "subband_weight_mode": variant_metadata.get("subband_weight_mode"),
                "spatial_filter_mode": variant_metadata.get("spatial_filter_mode"),
                "frontend_optimization_summary": dict(rows[0].get("frontend_optimization_summary", {})),
                "confidence_variant": str(confidence_variant),
                "confidence_training_scheme": str(rows[0].get("confidence_training_scheme", _tdca.DEFAULT_CONFIDENCE_TRAINING_SCHEME)),
                "decision_evidence_variant": _tdca.DEFAULT_DECISION_EVIDENCE_VARIANT,
                "oof_group_key": str(rows[0].get("oof_group_key", "")),
                "oof_group_count": int(rows[0].get("oof_group_count", 0) or 0),
                "sample_weight_mode": str(rows[0].get("sample_weight_mode", "")),
                "positive_trials": int(rows[0].get("positive_trials", 0) or 0),
                "negative_trials": int(rows[0].get("negative_trials", 0) or 0),
                "training_window_policy": str(rows[0].get("training_window_policy", "last_window_only")),
                "training_latency_sec": float(rows[0].get("training_latency_sec", 0.0) or 0.0),
                "analysis_latency_sec": float(rows[0].get("analysis_latency_sec", 0.0) or 0.0),
                "effective_raw_window_sec": float(rows[0].get("effective_raw_window_sec", sample_candidate.get("win_sec", 0.0)) or 0.0),
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
    fbcca_search_board.sort(key=_fbcca_board_sort_key)

    top_candidates = [dict(item) for item in fbcca_search_board if bool(item.get("gate_calibration_valid", False))][:DEFAULT_FBCCA_LOCAL_TOP_K]
    if fbcca_search_board and not top_candidates:
        status = "invalid"
        status_reasons.append("gate_calibration_invalid_all_candidates")
        if any(
            "tune_rows_insufficient" in dict(row.get("gate_calibration_summary", {})).get("invalid_reasons", [])
            for row in fbcca_search_board
        ):
            status_reasons.append("tune_rows_insufficient")

    variant_summary: list[dict[str, Any]] = []
    selected_for_decision = {str(row.get("candidate_key", "")) for row in top_candidates}
    for fbcca_variant in DEFAULT_FBCCA_LOCAL_VARIANTS:
        variant_rows = [dict(item) for item in fbcca_search_board if str(item.get("fbcca_variant", "")) == str(fbcca_variant)]
        if not variant_rows:
            continue
        best_variant_row = dict(variant_rows[0])
        variant_summary.append(
            {
                "fbcca_variant": str(fbcca_variant),
                "algorithm_alignment": str(best_variant_row.get("algorithm_alignment", "")),
                "channel_weight_mode": best_variant_row.get("channel_weight_mode"),
                "subband_weight_mode": best_variant_row.get("subband_weight_mode"),
                "spatial_filter_mode": best_variant_row.get("spatial_filter_mode"),
                "confidence_variant": str(best_variant_row.get("confidence_variant", _tdca.DEFAULT_CONFIDENCE_VARIANT)),
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
                bundle = _tdca._evaluate_structured_rows(
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
            emit_progress(
                stage="decision_search",
                detail=f"candidate {decision_progress_index}/{total_decision_runs} | {candidate_key} | params={param_key}",
                model_name="fbcca",
                fbcca_variant=str(candidate_row.get("fbcca_variant", "")),
                confidence_variant=str(candidate_confidence_variant),
                run_index=decision_progress_index,
                run_total=total_decision_runs,
            )
            row = {
                "candidate_key": str(candidate_key),
                "candidate": dict(candidate_row.get("candidate", {})),
                "fbcca_variant": str(candidate_row.get("fbcca_variant", "")),
                "algorithm_alignment": str(candidate_row.get("algorithm_alignment", "")),
                "channel_weight_mode": candidate_row.get("channel_weight_mode"),
                "subband_weight_mode": candidate_row.get("subband_weight_mode"),
                "spatial_filter_mode": candidate_row.get("spatial_filter_mode"),
                "frontend_optimization_summary": dict(candidate_row.get("frontend_optimization_summary", {})),
                "confidence_variant": str(candidate_confidence_variant),
                "confidence_training_scheme": str(candidate_row.get("confidence_training_scheme", _tdca.DEFAULT_CONFIDENCE_TRAINING_SCHEME)),
                "decision_evidence_variant": _tdca.DEFAULT_DECISION_EVIDENCE_VARIANT,
                "oof_group_key": str(candidate_row.get("oof_group_key", "")),
                "oof_group_count": int(candidate_row.get("oof_group_count", 0) or 0),
                "sample_weight_mode": str(candidate_row.get("sample_weight_mode", "")),
                "positive_trials": int(candidate_row.get("positive_trials", 0) or 0),
                "negative_trials": int(candidate_row.get("negative_trials", 0) or 0),
                "training_window_policy": str(candidate_row.get("training_window_policy", "last_window_only")),
                "training_latency_sec": float(candidate_row.get("training_latency_sec", 0.0) or 0.0),
                "analysis_latency_sec": float(candidate_row.get("analysis_latency_sec", 0.0) or 0.0),
                "effective_raw_window_sec": float(candidate_row.get("effective_raw_window_sec", dict(candidate_row.get("candidate", {})).get("win_sec", 0.0)) or 0.0),
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
        candidate_decision_board.sort(key=_fbcca_board_sort_key)
        best_global_params = dict(candidate_decision_board[0]["decision_params"])
        holdout_bundles: list[dict[str, Any]] = []
        candidate_holdout_rows: list[dict[str, Any]] = []
        for split in splits:
            context = candidate_context_cache[(candidate_key, int(split.repeat_index))]
            candidate_holdout_rows.extend([dict(row) for row in context.get("scored_holdout_rows", []) or []])
            holdout_bundles.append(
                _tdca._evaluate_structured_rows(
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
        final_candidate_rows.append(
            {
                "candidate_key": str(candidate_key),
                "candidate": dict(candidate_row.get("candidate", {})),
                "fbcca_variant": str(candidate_row.get("fbcca_variant", "")),
                "algorithm_alignment": str(candidate_row.get("algorithm_alignment", "")),
                "channel_weight_mode": candidate_row.get("channel_weight_mode"),
                "subband_weight_mode": candidate_row.get("subband_weight_mode"),
                "spatial_filter_mode": candidate_row.get("spatial_filter_mode"),
                "frontend_optimization_summary": dict(candidate_row.get("frontend_optimization_summary", {})),
                "confidence_variant": str(candidate_confidence_variant),
                "confidence_training_scheme": str(candidate_row.get("confidence_training_scheme", _tdca.DEFAULT_CONFIDENCE_TRAINING_SCHEME)),
                "decision_evidence_variant": _tdca.DEFAULT_DECISION_EVIDENCE_VARIANT,
                "confidence_diagnostics_board": _build_confidence_diagnostics_board(
                    candidate_row=candidate_row,
                    tune_rows=candidate_tune_rows,
                    holdout_rows=candidate_holdout_rows,
                ),
                "decision_bottleneck_summary": _build_decision_bottleneck_summary(
                    candidate_row=candidate_row,
                    holdout_bundles=holdout_bundles,
                ),
                "oof_group_key": str(candidate_row.get("oof_group_key", "")),
                "oof_group_count": int(candidate_row.get("oof_group_count", 0) or 0),
                "sample_weight_mode": str(candidate_row.get("sample_weight_mode", "")),
                "positive_trials": int(candidate_row.get("positive_trials", 0) or 0),
                "negative_trials": int(candidate_row.get("negative_trials", 0) or 0),
                "training_window_policy": str(candidate_row.get("training_window_policy", "last_window_only")),
                "training_latency_sec": float(candidate_row.get("training_latency_sec", 0.0) or 0.0),
                "analysis_latency_sec": float(candidate_row.get("analysis_latency_sec", 0.0) or 0.0),
                "effective_raw_window_sec": float(candidate_row.get("effective_raw_window_sec", dict(candidate_row.get("candidate", {})).get("win_sec", 0.0)) or 0.0),
                "tune_summary": dict(candidate_row.get("tune_summary", {})),
                "gate_calibration_valid": bool(candidate_row.get("gate_calibration_valid", False)),
                "gate_calibration_summary": dict(candidate_row.get("gate_calibration_summary", {})),
                "selection_target": "holdout_split",
                "decision_params": best_global_params,
                "repeat_count": int(len(holdout_bundles)),
                "metrics_median": async_metrics,
                "metrics_4class_median": dict(aggregated_holdout.get("metrics_4class", {})),
                "metrics_2class_median": dict(aggregated_holdout.get("metrics_2class", {})),
                "rank_key": [float(item) for item in _rank_metrics_key(async_metrics)],
            }
        )

    decision_aggregate_rows.sort(key=_fbcca_board_sort_key)
    final_candidate_rows.sort(key=_fbcca_board_sort_key)
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

    chosen_row = dict(final_candidate_rows[0]) if final_candidate_rows else (dict(fbcca_search_board[0]) if fbcca_search_board else {})
    emit_progress(force=True, stage="baseline_seal", detail="开始基线封板对比", run_index=0, run_total=len(DEFAULT_BASELINE_ROWS))
    baseline_seal = _run_baseline_suite(
        merged_dataset=merged_dataset,
        splits=splits,
        env_preflight=env_preflight,
        config=config,
    )
    emit_progress(force=True, stage="baseline_seal", detail="基线封板对比完成", run_index=len(DEFAULT_BASELINE_ROWS), run_total=len(DEFAULT_BASELINE_ROWS))

    fixed_baseline_row = next((dict(row) for row in baseline_seal if str(row.get("fbcca_variant", "")) == "fbcca_fixed_all8"), None)
    trca_baseline_row = next((dict(row) for row in baseline_seal if str(row.get("model_name", "")) == "trca_r"), None)
    if status != "ok":
        chosen_model_rationale = "invalid_run_not_comparable"
    elif not chosen_row or fixed_baseline_row is None:
        chosen_model_rationale = "fbcca_not_clearly_improved"
    else:
        chosen_rank = _candidate_rank_tuple(chosen_row)
        fixed_rank = _candidate_rank_tuple(fixed_baseline_row)
        trca_rank = _candidate_rank_tuple(trca_baseline_row) if trca_baseline_row is not None else tuple()
        if chosen_rank >= fixed_rank or (trca_rank and chosen_rank > trca_rank):
            chosen_model_rationale = "fbcca_not_clearly_improved"
        else:
            chosen_model_rationale = "fbcca_improved_on_primary_ranking"

    error_attribution_board: list[dict[str, Any]] = []
    for candidate_row in final_candidate_rows:
        candidate_key = str(candidate_row.get("candidate_key", ""))
        holdout_bundles: list[dict[str, Any]] = []
        for split in splits:
            context = candidate_context_cache.get((candidate_key, int(split.repeat_index)))
            if context is None:
                continue
            holdout_bundles.append(
                _tdca._evaluate_structured_rows(
                    scored_rows=context["scored_holdout_rows"],
                    profile=context["gate_profile"],
                    freqs=merged_dataset.freqs,
                    decision_params=dict(candidate_row.get("decision_params", {})),
                    inference_ms=float(context["inference_ms"]),
                    decision_time_mode=str(config.decision_time_mode),
                    async_decision_time_mode=str(config.async_decision_time_mode),
                )
            )
        error_attribution_board.extend(_build_error_attribution_board(candidate_row=candidate_row, holdout_bundles=holdout_bundles))

    contrast_error_board: list[dict[str, Any]] = []
    if chosen_row and fixed_baseline_row is not None:
        chosen_holdout_bundles = [
            _tdca._evaluate_structured_rows(
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
        ]
        contrast_error_board = _build_fbcca_contrast_error_board(
            candidate_row=chosen_row,
            candidate_holdout_bundles=chosen_holdout_bundles,
            reference_row=fixed_baseline_row,
        )

    baseline_opening_report = _sanitize_report_rows(baseline_opening)
    baseline_seal_report = _sanitize_report_rows(baseline_seal)
    chosen_confidence_variant = _resolved_confidence_variant(chosen_row)
    chosen_confidence_diagnostics_board = [dict(item) for item in chosen_row.get("confidence_diagnostics_board", []) or []]
    chosen_decision_bottleneck_summary = dict(chosen_row.get("decision_bottleneck_summary", {}))
    chosen_gate_calibration_summary: dict[str, Any] = dict(chosen_row.get("gate_calibration_summary", {}))

    profile_saved = False
    profile_v2_saved = False
    deployed_profile_published = False
    chosen_profile_path: Optional[str] = None
    profile_v2_path: Optional[str] = None
    chosen_profile: Optional[ThresholdProfile] = None
    chosen_replay_policy: dict[str, Any] = {}

    data_sufficiency_summary = {
        "session_count": int(len(merged_dataset.manifest_paths)),
        "trial_count": int(len(merged_dataset.trial_segments)),
        "unique_split_fingerprints": int(seed_effective.get("unique_split_fingerprints", 0)),
        "minimum_sessions_for_deployment": 1,
        "current_sessions_sufficient_for_deployment": True,
        "deployment_grade": DEFAULT_FBCCA_LOCAL_DEPLOYMENT_GRADE,
    }

    if chosen_row and status == "ok":
        emit_progress(force=True, stage="finalize", detail="基于全量数据重建最终 profile", run_index=1, run_total=3)
        chosen_candidate = dict(chosen_row.get("candidate", {}))
        full_split = RepeatedGroupSplit(
            repeat_index=0,
            train_indices=tuple(range(len(merged_dataset.trial_segments))),
            gate_indices=tuple(range(len(merged_dataset.trial_segments))),
            holdout_indices=tuple(range(len(merged_dataset.trial_segments))),
            fingerprint="full-data",
        )
        full_replay_policy = _tdca._split_replay_policy(
            merged_dataset=merged_dataset,
            split=full_split,
            win_sec=float(chosen_candidate.get("win_sec", DEFAULT_BASELINE_WIN_SEC)),
            env_preflight=env_preflight,
        )
        chosen_replay_policy = dict(full_replay_policy)
        full_context = _build_fbcca_candidate_context(
            merged_dataset=merged_dataset,
            split=full_split,
            fbcca_variant=str(chosen_candidate.get("fbcca_variant", DEFAULT_FBCCA_LOCAL_VARIANTS[0])),
            win_sec=float(chosen_candidate.get("win_sec", DEFAULT_BASELINE_WIN_SEC)),
            confidence_variant=str(chosen_confidence_variant),
            config=config,
            replay_policy=full_replay_policy,
            log=log,
        )
        variant_metadata = _fbcca_variant_metadata(chosen_candidate.get("fbcca_variant"))
        final_metrics = dict(chosen_row.get("metrics_median", {}))
        candidate_run_valid_for_deployment = _fbcca_meets_promotion_thresholds(final_metrics)
        final_profile = replace(
            full_context["gate_profile"],
            model_name=DEFAULT_FBCCA_LOCAL_MODEL,
            model_params={
                **dict(full_context.get("model_params", {})),
                "state": full_context.get("state_payload"),
                "fbcca_variant": str(variant_metadata["fbcca_variant"]),
            },
            benchmark_metrics=final_metrics,
            confidence_variant=str(chosen_confidence_variant),
            training_window_policy=str(full_context.get("training_window_policy", "last_window_only")),
            metadata={
                "task": "fbcca-local-opt",
                "env_preflight": env_preflight,
                "split_fingerprints": split_fingerprints,
                "seed_effective": seed_effective,
                "search_preset": str(search_plan.get("search_preset", DEFAULT_FBCCA_LOCAL_SEARCH_PRESET)),
                "decision_params": dict(chosen_row.get("decision_params", {})),
                "candidate": dict(chosen_candidate),
                "fbcca_variant": str(variant_metadata["fbcca_variant"]),
                "algorithm_alignment": str(variant_metadata["algorithm_alignment"]),
                "channel_weight_mode": variant_metadata.get("channel_weight_mode"),
                "subband_weight_mode": variant_metadata.get("subband_weight_mode"),
                "spatial_filter_mode": variant_metadata.get("spatial_filter_mode"),
                "frontend_optimization_summary": dict(full_context.get("frontend_optimization_summary", {})),
                "confidence_variant": str(chosen_confidence_variant),
                "confidence_training_scheme": str(full_context.get("confidence_training_scheme", _tdca.DEFAULT_CONFIDENCE_TRAINING_SCHEME)),
                "decision_evidence_variant": _tdca.DEFAULT_DECISION_EVIDENCE_VARIANT,
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
                "tune_summary": dict(full_context.get("tune_summary", {})),
                "gate_calibration_summary": dict(full_context.get("gate_calibration_summary", {})),
                "confidence_diagnostics_board": [dict(item) for item in chosen_confidence_diagnostics_board],
                "decision_bottleneck_summary": dict(chosen_decision_bottleneck_summary),
                "data_sufficiency_summary": dict(data_sufficiency_summary),
                "deployment_grade": DEFAULT_FBCCA_LOCAL_DEPLOYMENT_GRADE,
                "run_valid_for_deployment": bool(candidate_run_valid_for_deployment),
                "decision_search_target": "tune_split",
                "final_selection_target": "holdout_split",
                "effective_replay_backend": str(full_replay_policy.get("effective_replay_backend", "cpu")),
                "gpu_replay_speedup": float(full_replay_policy.get("gpu_replay_speedup", 0.0)),
                "gpu_replay_eligible": bool(full_replay_policy.get("gpu_replay_eligible", False)),
                "gpu_replay_reason": str(full_replay_policy.get("gpu_replay_reason", "")),
                "chosen_model_rationale": str(chosen_model_rationale),
                "dynamic_stop_enabled": False,
            },
            recommended_for_realtime=True,
        )
        save_profile(final_profile, resolved_profile_path)
        chosen_profile = final_profile
        chosen_profile_path = str(resolved_profile_path)
        profile_saved = True
        chosen_gate_calibration_summary = dict(full_context.get("gate_calibration_summary", {}))
        emit_progress(force=True, stage="finalize", detail="主 profile 已保存，开始导出 profile_v2", run_index=2, run_total=3, profile_path=chosen_profile_path)

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
        atomic_write_text(resolved_profile_v2_path, json_dumps(json_safe(profile_v2.to_payload())) + "\n")
        profile_v2_saved = True
        profile_v2_path = str(resolved_profile_v2_path)
        if candidate_run_valid_for_deployment:
            publish_deployed_profile(
                source_profile=resolved_profile_path,
                source_profile_v2=resolved_profile_v2_path,
                run_dir=report_dir,
                task="fbcca-local-opt",
                run_tag=run_tag,
                report_json=Path(report_paths["report_json"]).expanduser().resolve(),
                extra_metadata={
                    "chosen_model": DEFAULT_FBCCA_LOCAL_MODEL,
                    "fbcca_variant": str(variant_metadata["fbcca_variant"]),
                    "deployment_grade": DEFAULT_FBCCA_LOCAL_DEPLOYMENT_GRADE,
                },
            )
            deployed_profile_published = True
        emit_progress(force=True, stage="finalize", detail="profile_v2 已保存，开始生成报告", run_index=3, run_total=3, profile_path=chosen_profile_path)
    else:
        emit_progress(force=True, stage="finalize", detail="本次运行未保存 profile，仍将输出报告", run_index=3, run_total=3)

    chosen_async_metrics = dict(chosen_row.get("metrics_median", {}))
    chosen_metrics_4class = dict(chosen_row.get("metrics_4class_median", {}))
    chosen_metrics_2class = dict(chosen_row.get("metrics_2class_median", {}))
    chosen_variant_metadata = _fbcca_variant_metadata(chosen_row.get("fbcca_variant", DEFAULT_FBCCA_LOCAL_VARIANTS[0])) if chosen_row else _fbcca_variant_metadata(DEFAULT_FBCCA_LOCAL_VARIANTS[0])
    run_valid_for_deployment = bool(status == "ok" and _fbcca_meets_promotion_thresholds(chosen_async_metrics))
    report_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "fbcca_local_opt",
        "task": "fbcca-local-opt",
        "search_preset": str(search_plan.get("search_preset", DEFAULT_FBCCA_LOCAL_SEARCH_PRESET)),
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
        "fbcca_variant": str(chosen_variant_metadata["fbcca_variant"]) if chosen_row else "",
        "algorithm_alignment": str(chosen_variant_metadata["algorithm_alignment"]) if chosen_row else "",
        "channel_weight_mode": chosen_variant_metadata.get("channel_weight_mode") if chosen_row else None,
        "subband_weight_mode": chosen_variant_metadata.get("subband_weight_mode") if chosen_row else None,
        "spatial_filter_mode": chosen_variant_metadata.get("spatial_filter_mode") if chosen_row else None,
        "frontend_optimization_summary": dict(chosen_row.get("frontend_optimization_summary", {})) if chosen_row else {},
        "confidence_variant": str(chosen_confidence_variant),
        "confidence_training_scheme": str(chosen_row.get("confidence_training_scheme", _tdca.DEFAULT_CONFIDENCE_TRAINING_SCHEME)),
        "decision_evidence_variant": _tdca.DEFAULT_DECISION_EVIDENCE_VARIANT,
        "decision_evidence_raw": "correctness_logit",
        "decision_evidence_reference": "logit(enter_p_th_for_pred_freq)",
        "oof_group_key": str(chosen_row.get("oof_group_key", "")),
        "oof_group_count": int(chosen_row.get("oof_group_count", 0) or 0),
        "sample_weight_mode": str(chosen_row.get("sample_weight_mode", "")),
        "positive_trials": int(chosen_row.get("positive_trials", 0) or 0),
        "negative_trials": int(chosen_row.get("negative_trials", 0) or 0),
        "training_window_policy": str(chosen_row.get("training_window_policy", "last_window_only")),
        "training_latency_sec": float(chosen_row.get("training_latency_sec", 0.0) or 0.0),
        "analysis_latency_sec": float(chosen_row.get("analysis_latency_sec", 0.0) or 0.0),
        "effective_raw_window_sec": float(chosen_row.get("effective_raw_window_sec", dict(chosen_row.get("candidate", {})).get("win_sec", 0.0)) or 0.0),
        "effective_replay_backend": str(chosen_replay_policy.get("effective_replay_backend", env_preflight.get("effective_backend", "cpu"))),
        "gpu_replay_speedup": float(chosen_replay_policy.get("gpu_replay_speedup", env_preflight.get("gpu_replay_speedup", 0.0))),
        "gpu_replay_eligible": bool(chosen_replay_policy.get("gpu_replay_eligible", False)),
        "gpu_replay_reason": str(chosen_replay_policy.get("gpu_replay_reason", "")),
        "decision_search_target": "tune_split",
        "final_selection_target": "holdout_split",
        "dynamic_stop_enabled": False,
        "fbcca_search_board": fbcca_search_board,
        "gate_exit_search_board": gate_exit_by_candidate.get(str(chosen_row.get("candidate_key", "")), []),
        "decision_search_board": decision_aggregate_rows,
        "holdout_selection_board": final_candidate_rows,
        "variant_summary": variant_summary,
        "tune_summary": dict(chosen_row.get("tune_summary", {})),
        "tune_rows_valid": bool(dict(chosen_row.get("tune_summary", {})).get("valid", False)),
        "confidence_diagnostics_board": [dict(item) for item in chosen_confidence_diagnostics_board],
        "decision_bottleneck_summary": dict(chosen_decision_bottleneck_summary),
        "ranking_boards": {"end_to_end": final_candidate_rows, "classifier_only": []},
        "model_results": final_candidate_rows,
        "chosen_model": DEFAULT_FBCCA_LOCAL_MODEL if chosen_row else "",
        "recommended_model": DEFAULT_FBCCA_LOCAL_MODEL if chosen_row else "",
        "chosen_rank": 1 if chosen_row else 0,
        "async_metrics": chosen_async_metrics,
        "chosen_metrics": chosen_async_metrics,
        "chosen_async_metrics": chosen_async_metrics,
        "chosen_metrics_4class": chosen_metrics_4class,
        "chosen_metrics_2class": chosen_metrics_2class,
        "gate_calibration_valid": bool(chosen_row.get("gate_calibration_valid", False)),
        "gate_calibration_summary": dict(chosen_gate_calibration_summary),
        "min_gate_control_rows": int(chosen_row.get("min_gate_control_rows", 0) or 0),
        "min_gate_idle_rows": int(chosen_row.get("min_gate_idle_rows", 0) or 0),
        "enter_p_th_median": float(chosen_row.get("enter_p_th_median", 0.65) or 0.65),
        "exit_p_th_median": float(chosen_row.get("exit_p_th_median", 0.30) or 0.30),
        "enter_logit_th_median": float(chosen_row.get("enter_logit_th_median", 0.0) or 0.0),
        "exit_logit_th_median": float(chosen_row.get("exit_logit_th_median", 0.0) or 0.0),
        "chosen_profile_path": chosen_profile_path,
        "profile_saved": bool(profile_saved),
        "profile_v2_saved": bool(profile_v2_saved),
        "profile_v2_path": profile_v2_path,
        "run_valid_for_deployment": bool(run_valid_for_deployment),
        "deployment_grade": DEFAULT_FBCCA_LOCAL_DEPLOYMENT_GRADE,
        "deployed_profile_published": bool(deployed_profile_published),
        "data_sufficiency_summary": dict(data_sufficiency_summary),
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
                    "search_preset": str(search_plan.get("search_preset", DEFAULT_FBCCA_LOCAL_SEARCH_PRESET)),
                    "split_fingerprints": split_fingerprints,
                    "seed_effective": seed_effective,
                    "fbcca_variant": report_payload["fbcca_variant"],
                    "algorithm_alignment": report_payload["algorithm_alignment"],
                    "channel_weight_mode": report_payload["channel_weight_mode"],
                    "subband_weight_mode": report_payload["subband_weight_mode"],
                    "spatial_filter_mode": report_payload["spatial_filter_mode"],
                    "frontend_optimization_summary": report_payload["frontend_optimization_summary"],
                    "confidence_variant": report_payload["confidence_variant"],
                    "confidence_training_scheme": report_payload["confidence_training_scheme"],
                    "decision_evidence_variant": report_payload["decision_evidence_variant"],
                    "decision_evidence_raw": report_payload["decision_evidence_raw"],
                    "decision_evidence_reference": report_payload["decision_evidence_reference"],
                    "oof_group_key": report_payload["oof_group_key"],
                    "oof_group_count": report_payload["oof_group_count"],
                    "sample_weight_mode": report_payload["sample_weight_mode"],
                    "positive_trials": report_payload["positive_trials"],
                    "negative_trials": report_payload["negative_trials"],
                    "training_window_policy": report_payload["training_window_policy"],
                    "training_latency_sec": report_payload["training_latency_sec"],
                    "analysis_latency_sec": report_payload["analysis_latency_sec"],
                    "effective_raw_window_sec": report_payload["effective_raw_window_sec"],
                    "decision_search_target": "tune_split",
                    "final_selection_target": "holdout_split",
                    "variant_summary": variant_summary,
                    "holdout_selection_board": final_candidate_rows,
                    "confidence_diagnostics_board": [dict(item) for item in chosen_confidence_diagnostics_board],
                    "decision_bottleneck_summary": dict(chosen_decision_bottleneck_summary),
                    "error_attribution_board": error_attribution_board,
                    "contrast_error_board": contrast_error_board,
                    "run_valid_for_deployment": bool(run_valid_for_deployment),
                    "deployment_grade": DEFAULT_FBCCA_LOCAL_DEPLOYMENT_GRADE,
                    "chosen_model_rationale": str(chosen_model_rationale),
                }
            )
        )
        + "\n",
    )
    atomic_write_text(report_paths["report_json"], json_dumps(json_safe(report_payload)) + "\n")
    atomic_write_text(report_paths["report_md"], _render_markdown(report_payload))
    log(f"[fbcca-local-opt] report saved: {report_paths['report_json']}")
    if chosen_profile is not None:
        log(f"[fbcca-local-opt] profile saved: {chosen_profile_path}")

    emit_progress(
        force=True,
        stage="complete",
        detail="FBCCA 本地异步优化完成",
        run_index=1,
        run_total=1,
        report_path=str(report_paths["report_json"]),
        profile_path=str(chosen_profile_path or resolved_profile_path),
        progress_percent=100,
    )
    return report_payload
