from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

from async_fbcca_idle_standalone import (
    DEFAULT_ASYNC_DECISION_TIME_MODE,
    DEFAULT_BENCHMARK_CHANNEL_MODES,
    DEFAULT_BENCHMARK_MODELS,
    DEFAULT_BENCHMARK_MULTI_SEED_COUNT,
    DEFAULT_CHANNEL_WEIGHT_L2,
    DEFAULT_CHANNEL_WEIGHT_MODE,
    DEFAULT_CONTROL_STATE_MODE,
    DEFAULT_DATA_POLICY,
    DEFAULT_DYNAMIC_STOP_ALPHA,
    DEFAULT_GATE_POLICY,
    DEFAULT_GPU_CACHE_MODE,
    DEFAULT_GPU_DEVICE_ID,
    DEFAULT_GPU_PRECISION_NAME,
    DEFAULT_METRIC_SCOPE,
    DEFAULT_PAPER_DECISION_TIME_MODE,
    DEFAULT_PROFILE_PATH,
    DEFAULT_RANKING_POLICY,
    DEFAULT_SPATIAL_FILTER_MODE,
    DEFAULT_SPATIAL_RANK_CANDIDATES,
    DEFAULT_SPATIAL_SOURCE_MODEL,
    DEFAULT_IDLE_FP_HARD_TH,
    DEFAULT_SUBBAND_WEIGHT_MODE,
    DEFAULT_SUBBAND_PRIOR_STRENGTH,
    DEFAULT_WEIGHT_AGGREGATION,
    DEFAULT_WIN_SEC_CANDIDATES,
    parse_channel_mode_list,
    parse_channel_weight_mode,
    parse_control_state_mode,
    parse_compute_backend_name,
    parse_data_policy,
    parse_decision_time_mode,
    parse_gate_policy,
    parse_gpu_cache_policy,
    parse_gpu_precision,
    parse_metric_scope,
    parse_model_list,
    parse_ranking_policy,
    parse_spatial_filter_mode,
    parse_spatial_rank_candidates,
    parse_spatial_source_model,
    parse_subband_weight_mode,
    parse_weight_aggregation,
)
from ssvep_core.train_eval import (
    DEFAULT_EVALUATION_MODE,
    DEFAULT_FORCE_INCLUDE_MODELS,
    DEFAULT_PROGRESS_HEARTBEAT_SEC,
    DEFAULT_QUICK_SCREEN_TOP_K,
    DEFAULT_TRAIN_EVAL_TASK,
    OfflineTrainEvalConfig,
    run_offline_train_eval,
)
from ssvep_core.registry import ModelRegistry

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = THIS_DIR / "profiles" / "datasets"
DEFAULT_REPORT_ROOT = THIS_DIR / "profiles" / "reports" / "train_eval"
TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND = "cuda"
TRAIN_EVAL_DEFAULT_GPU_PRECISION = "float32"

QUICK_MODE_MODELS = (
    "fbcca_fixed_all8",
    "fbcca_cw_all8",
    "fbcca_sw_all8",
    "fbcca_cw_sw_all8",
)
QUICK_MODE_CHANNEL_MODES = ("all8",)
QUICK_MODE_MULTI_SEED_COUNT = 5
QUICK_MODE_WIN_CANDIDATES = (1.5,)
QUICK_MODE_JOINT_WEIGHT_ITERS = 1
QUICK_MODE_WEIGHT_CV_FOLDS = 2
QUICK_MODE_QUICK_SCREEN_TOP_K = 2
QUICK_MODE_FORCE_INCLUDE_MODELS = ("fbcca_fixed_all8", "fbcca_cw_sw_all8")
QUICK_MODE_CHANNEL_WEIGHT_MODE = "fbcca_diag"
QUICK_MODE_SUBBAND_WEIGHT_MODE = "chen_ab_subject"
QUICK_MODE_SPATIAL_FILTER_MODE = "none"

MODEL_COMPARE_MODELS = tuple(ModelRegistry.list_models(task="benchmark"))
MODEL_COMPARE_CHANNEL_MODES = ("all8",)
MODEL_COMPARE_MULTI_SEED_COUNT = 5
MODEL_COMPARE_WIN_CANDIDATES = (1.5, 2.0)
MODEL_COMPARE_JOINT_WEIGHT_ITERS = 1
MODEL_COMPARE_WEIGHT_CV_FOLDS = 2
MODEL_COMPARE_QUICK_SCREEN_TOP_K = len(MODEL_COMPARE_MODELS)
MODEL_COMPARE_FORCE_INCLUDE_MODELS = MODEL_COMPARE_MODELS
MODEL_COMPARE_CHANNEL_WEIGHT_MODE = "none"
MODEL_COMPARE_SUBBAND_WEIGHT_MODE = "chen_fixed"
MODEL_COMPARE_SPATIAL_FILTER_MODE = "none"

WEIGHTED_COMPARE_MODELS = tuple(
    dict.fromkeys(
        (
            "legacy_fbcca_202603",
            "fbcca_fixed_all8",
            "fbcca_cw_all8",
            "fbcca_sw_all8",
            "fbcca_cw_sw_all8",
        )
        + tuple(
            name
            for name in ModelRegistry.list_models(task="benchmark")
            if str(name) not in {"legacy_fbcca_202603", "fbcca"}
        )
    )
)
WEIGHTED_COMPARE_CHANNEL_MODES = ("all8",)
WEIGHTED_COMPARE_MULTI_SEED_COUNT = 5
WEIGHTED_COMPARE_WIN_CANDIDATES = (1.5, 2.0)
WEIGHTED_COMPARE_JOINT_WEIGHT_ITERS = 1
WEIGHTED_COMPARE_WEIGHT_CV_FOLDS = 2
WEIGHTED_COMPARE_QUICK_SCREEN_TOP_K = len(WEIGHTED_COMPARE_MODELS)
WEIGHTED_COMPARE_FORCE_INCLUDE_MODELS = WEIGHTED_COMPARE_MODELS
WEIGHTED_COMPARE_CHANNEL_WEIGHT_MODE = "none"
WEIGHTED_COMPARE_SUBBAND_WEIGHT_MODE = "chen_fixed"
WEIGHTED_COMPARE_SPATIAL_FILTER_MODE = "none"

FOCUSED_COMPARE_MODELS = WEIGHTED_COMPARE_MODELS
FOCUSED_COMPARE_CHANNEL_MODES = ("all8",)
FOCUSED_COMPARE_MULTI_SEED_COUNT = 5
FOCUSED_COMPARE_WIN_CANDIDATES = (1.5, 2.0)
FOCUSED_COMPARE_JOINT_WEIGHT_ITERS = 1
FOCUSED_COMPARE_WEIGHT_CV_FOLDS = 2
FOCUSED_COMPARE_QUICK_SCREEN_TOP_K = 4
FOCUSED_COMPARE_FORCE_INCLUDE_MODELS = (
    "legacy_fbcca_202603",
    "fbcca_fixed_all8",
    "fbcca_cw_sw_all8",
)
FOCUSED_COMPARE_CHANNEL_WEIGHT_MODE = "none"
FOCUSED_COMPARE_SUBBAND_WEIGHT_MODE = "chen_fixed"
FOCUSED_COMPARE_SPATIAL_FILTER_MODE = "none"

CLASSIFIER_COMPARE_MODELS = tuple(
    dict.fromkeys(
        (
            "legacy_fbcca_202603",
            "fbcca_fixed_all8",
            *tuple(ModelRegistry.list_models(task="benchmark")),
            "oacca",
        )
    )
)
CLASSIFIER_COMPARE_CHANNEL_MODES = ("all8",)
CLASSIFIER_COMPARE_MULTI_SEED_COUNT = 10
CLASSIFIER_COMPARE_WIN_CANDIDATES = (1.5, 2.0)
CLASSIFIER_COMPARE_JOINT_WEIGHT_ITERS = 1
CLASSIFIER_COMPARE_WEIGHT_CV_FOLDS = 2
CLASSIFIER_COMPARE_QUICK_SCREEN_TOP_K = len(CLASSIFIER_COMPARE_MODELS)
CLASSIFIER_COMPARE_FORCE_INCLUDE_MODELS = CLASSIFIER_COMPARE_MODELS
CLASSIFIER_COMPARE_CHANNEL_WEIGHT_MODE = "none"
CLASSIFIER_COMPARE_SUBBAND_WEIGHT_MODE = "chen_fixed"
CLASSIFIER_COMPARE_SPATIAL_FILTER_MODE = "none"

PROFILE_EVAL_MODELS = (
    "fbcca_fixed_all8",
    "legacy_fbcca_202603",
    *tuple(ModelRegistry.list_models(task="benchmark")),
    "oacca",
)
PROFILE_EVAL_CHANNEL_MODES = ("all8",)
PROFILE_EVAL_MULTI_SEED_COUNT = 5
PROFILE_EVAL_WIN_CANDIDATES = (1.5, 2.0)
PROFILE_EVAL_JOINT_WEIGHT_ITERS = 1
PROFILE_EVAL_WEIGHT_CV_FOLDS = 2
PROFILE_EVAL_QUICK_SCREEN_TOP_K = len(PROFILE_EVAL_MODELS)
PROFILE_EVAL_FORCE_INCLUDE_MODELS = PROFILE_EVAL_MODELS
PROFILE_EVAL_CHANNEL_WEIGHT_MODE = "none"
PROFILE_EVAL_SUBBAND_WEIGHT_MODE = "chen_fixed"
PROFILE_EVAL_SPATIAL_FILTER_MODE = "none"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_task(raw: str) -> str:
    value = str(raw or DEFAULT_TRAIN_EVAL_TASK).strip().lower()
    aliases = {
        "fbcca": "fbcca-weights",
        "fbcca_weight": "fbcca-weights",
        "fbcca_weights": "fbcca-weights",
        "weights": "fbcca-weights",
        "compare": "model-compare",
        "model_compare": "model-compare",
        "models": "model-compare",
        "weighted_compare": "fbcca-weighted-compare",
        "fbcca_weighted_compare": "fbcca-weighted-compare",
        "weights_compare": "fbcca-weighted-compare",
        "focused": "focused-compare",
        "focused_compare": "focused-compare",
        "best_focus": "focused-compare",
        "best-analysis": "focused-compare",
        "classifier": "classifier-compare",
        "classifier_compare": "classifier-compare",
        "classification_compare": "classifier-compare",
        "all_classifier": "classifier-compare",
        "profile_eval": "profile-eval",
        "profile-eval": "profile-eval",
        "eval_profile": "profile-eval",
        "pretrained_eval": "profile-eval",
    }
    value = aliases.get(value, value)
    if value not in {
        "fbcca-weights",
        "model-compare",
        "fbcca-weighted-compare",
        "focused-compare",
        "classifier-compare",
        "profile-eval",
    }:
        raise ValueError(f"unsupported task: {raw}")
    return value


def _parse_manifest_csv(raw: str) -> tuple[Path, ...]:
    values: list[Path] = []
    for part in str(raw or "").split(","):
        item = str(part).strip()
        if not item:
            continue
        values.append(Path(item).expanduser().resolve())
    return tuple(values)


def _collect_provided_option_names(argv: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for token in argv:
        if not isinstance(token, str) or not token.startswith("--"):
            continue
        names.add(token.split("=", 1)[0])
    return names


def _set_if_missing(
    args: argparse.Namespace,
    provided_options: set[str],
    option_name: str,
    attr_name: str,
    value: Any,
) -> None:
    if option_name not in provided_options:
        setattr(args, attr_name, value)


def _apply_quick_mode_args(args: argparse.Namespace, provided_options: set[str]) -> None:
    args.task = "fbcca-weights"
    _set_if_missing(args, provided_options, "--models", "models", ",".join(QUICK_MODE_MODELS))
    _set_if_missing(args, provided_options, "--channel-modes", "channel_modes", ",".join(QUICK_MODE_CHANNEL_MODES))
    _set_if_missing(args, provided_options, "--multi-seed-count", "multi_seed_count", int(QUICK_MODE_MULTI_SEED_COUNT))
    _set_if_missing(args, provided_options, "--channel-weight-mode", "channel_weight_mode", str(QUICK_MODE_CHANNEL_WEIGHT_MODE))
    _set_if_missing(args, provided_options, "--subband-weight-mode", "subband_weight_mode", str(QUICK_MODE_SUBBAND_WEIGHT_MODE))
    _set_if_missing(args, provided_options, "--spatial-filter-mode", "spatial_filter_mode", str(QUICK_MODE_SPATIAL_FILTER_MODE))
    _set_if_missing(args, provided_options, "--joint-weight-iters", "joint_weight_iters", int(QUICK_MODE_JOINT_WEIGHT_ITERS))
    _set_if_missing(args, provided_options, "--weight-cv-folds", "weight_cv_folds", int(QUICK_MODE_WEIGHT_CV_FOLDS))
    _set_if_missing(args, provided_options, "--win-candidates", "win_candidates", ",".join(f"{float(value):g}" for value in QUICK_MODE_WIN_CANDIDATES))
    _set_if_missing(args, provided_options, "--quick-screen-top-k", "quick_screen_top_k", int(QUICK_MODE_QUICK_SCREEN_TOP_K))
    _set_if_missing(args, provided_options, "--force-include-models", "force_include_models", ",".join(QUICK_MODE_FORCE_INCLUDE_MODELS))
    _set_if_missing(args, provided_options, "--compute-backend", "compute_backend", str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND))
    _set_if_missing(args, provided_options, "--gpu-precision", "gpu_precision", str(TRAIN_EVAL_DEFAULT_GPU_PRECISION))


def _apply_model_compare_args(args: argparse.Namespace, provided_options: set[str]) -> None:
    args.task = "model-compare"
    _set_if_missing(args, provided_options, "--models", "models", ",".join(MODEL_COMPARE_MODELS))
    _set_if_missing(args, provided_options, "--channel-modes", "channel_modes", ",".join(MODEL_COMPARE_CHANNEL_MODES))
    _set_if_missing(args, provided_options, "--multi-seed-count", "multi_seed_count", int(MODEL_COMPARE_MULTI_SEED_COUNT))
    _set_if_missing(args, provided_options, "--channel-weight-mode", "channel_weight_mode", str(MODEL_COMPARE_CHANNEL_WEIGHT_MODE))
    _set_if_missing(args, provided_options, "--subband-weight-mode", "subband_weight_mode", str(MODEL_COMPARE_SUBBAND_WEIGHT_MODE))
    _set_if_missing(args, provided_options, "--spatial-filter-mode", "spatial_filter_mode", str(MODEL_COMPARE_SPATIAL_FILTER_MODE))
    _set_if_missing(args, provided_options, "--joint-weight-iters", "joint_weight_iters", int(MODEL_COMPARE_JOINT_WEIGHT_ITERS))
    _set_if_missing(args, provided_options, "--weight-cv-folds", "weight_cv_folds", int(MODEL_COMPARE_WEIGHT_CV_FOLDS))
    _set_if_missing(args, provided_options, "--win-candidates", "win_candidates", ",".join(f"{float(value):g}" for value in MODEL_COMPARE_WIN_CANDIDATES))
    _set_if_missing(args, provided_options, "--quick-screen-top-k", "quick_screen_top_k", int(MODEL_COMPARE_QUICK_SCREEN_TOP_K))
    _set_if_missing(args, provided_options, "--force-include-models", "force_include_models", ",".join(MODEL_COMPARE_FORCE_INCLUDE_MODELS))
    _set_if_missing(args, provided_options, "--compute-backend", "compute_backend", str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND))
    _set_if_missing(args, provided_options, "--gpu-precision", "gpu_precision", str(TRAIN_EVAL_DEFAULT_GPU_PRECISION))


def _apply_weighted_compare_args(args: argparse.Namespace, provided_options: set[str]) -> None:
    args.task = "fbcca-weighted-compare"
    _set_if_missing(args, provided_options, "--models", "models", ",".join(WEIGHTED_COMPARE_MODELS))
    _set_if_missing(args, provided_options, "--channel-modes", "channel_modes", ",".join(WEIGHTED_COMPARE_CHANNEL_MODES))
    _set_if_missing(args, provided_options, "--multi-seed-count", "multi_seed_count", int(WEIGHTED_COMPARE_MULTI_SEED_COUNT))
    _set_if_missing(args, provided_options, "--channel-weight-mode", "channel_weight_mode", str(WEIGHTED_COMPARE_CHANNEL_WEIGHT_MODE))
    _set_if_missing(args, provided_options, "--subband-weight-mode", "subband_weight_mode", str(WEIGHTED_COMPARE_SUBBAND_WEIGHT_MODE))
    _set_if_missing(args, provided_options, "--spatial-filter-mode", "spatial_filter_mode", str(WEIGHTED_COMPARE_SPATIAL_FILTER_MODE))
    _set_if_missing(args, provided_options, "--joint-weight-iters", "joint_weight_iters", int(WEIGHTED_COMPARE_JOINT_WEIGHT_ITERS))
    _set_if_missing(args, provided_options, "--weight-cv-folds", "weight_cv_folds", int(WEIGHTED_COMPARE_WEIGHT_CV_FOLDS))
    _set_if_missing(args, provided_options, "--win-candidates", "win_candidates", ",".join(f"{float(value):g}" for value in WEIGHTED_COMPARE_WIN_CANDIDATES))
    _set_if_missing(args, provided_options, "--quick-screen-top-k", "quick_screen_top_k", int(WEIGHTED_COMPARE_QUICK_SCREEN_TOP_K))
    _set_if_missing(args, provided_options, "--force-include-models", "force_include_models", ",".join(WEIGHTED_COMPARE_FORCE_INCLUDE_MODELS))
    _set_if_missing(args, provided_options, "--compute-backend", "compute_backend", str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND))
    _set_if_missing(args, provided_options, "--gpu-precision", "gpu_precision", str(TRAIN_EVAL_DEFAULT_GPU_PRECISION))


def _apply_focused_compare_args(args: argparse.Namespace, provided_options: set[str]) -> None:
    args.task = "focused-compare"
    _set_if_missing(args, provided_options, "--models", "models", ",".join(FOCUSED_COMPARE_MODELS))
    _set_if_missing(args, provided_options, "--channel-modes", "channel_modes", ",".join(FOCUSED_COMPARE_CHANNEL_MODES))
    _set_if_missing(args, provided_options, "--multi-seed-count", "multi_seed_count", int(FOCUSED_COMPARE_MULTI_SEED_COUNT))
    _set_if_missing(args, provided_options, "--channel-weight-mode", "channel_weight_mode", str(FOCUSED_COMPARE_CHANNEL_WEIGHT_MODE))
    _set_if_missing(args, provided_options, "--subband-weight-mode", "subband_weight_mode", str(FOCUSED_COMPARE_SUBBAND_WEIGHT_MODE))
    _set_if_missing(args, provided_options, "--spatial-filter-mode", "spatial_filter_mode", str(FOCUSED_COMPARE_SPATIAL_FILTER_MODE))
    _set_if_missing(args, provided_options, "--joint-weight-iters", "joint_weight_iters", int(FOCUSED_COMPARE_JOINT_WEIGHT_ITERS))
    _set_if_missing(args, provided_options, "--weight-cv-folds", "weight_cv_folds", int(FOCUSED_COMPARE_WEIGHT_CV_FOLDS))
    _set_if_missing(args, provided_options, "--win-candidates", "win_candidates", ",".join(f"{float(value):g}" for value in FOCUSED_COMPARE_WIN_CANDIDATES))
    _set_if_missing(args, provided_options, "--quick-screen-top-k", "quick_screen_top_k", int(FOCUSED_COMPARE_QUICK_SCREEN_TOP_K))
    _set_if_missing(args, provided_options, "--force-include-models", "force_include_models", ",".join(FOCUSED_COMPARE_FORCE_INCLUDE_MODELS))
    _set_if_missing(args, provided_options, "--compute-backend", "compute_backend", str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND))
    _set_if_missing(args, provided_options, "--gpu-precision", "gpu_precision", str(TRAIN_EVAL_DEFAULT_GPU_PRECISION))


def _apply_classifier_compare_args(args: argparse.Namespace, provided_options: set[str]) -> None:
    args.task = "classifier-compare"
    _set_if_missing(args, provided_options, "--models", "models", ",".join(CLASSIFIER_COMPARE_MODELS))
    _set_if_missing(args, provided_options, "--channel-modes", "channel_modes", ",".join(CLASSIFIER_COMPARE_CHANNEL_MODES))
    _set_if_missing(args, provided_options, "--multi-seed-count", "multi_seed_count", int(CLASSIFIER_COMPARE_MULTI_SEED_COUNT))
    _set_if_missing(args, provided_options, "--channel-weight-mode", "channel_weight_mode", str(CLASSIFIER_COMPARE_CHANNEL_WEIGHT_MODE))
    _set_if_missing(args, provided_options, "--subband-weight-mode", "subband_weight_mode", str(CLASSIFIER_COMPARE_SUBBAND_WEIGHT_MODE))
    _set_if_missing(args, provided_options, "--spatial-filter-mode", "spatial_filter_mode", str(CLASSIFIER_COMPARE_SPATIAL_FILTER_MODE))
    _set_if_missing(args, provided_options, "--joint-weight-iters", "joint_weight_iters", int(CLASSIFIER_COMPARE_JOINT_WEIGHT_ITERS))
    _set_if_missing(args, provided_options, "--weight-cv-folds", "weight_cv_folds", int(CLASSIFIER_COMPARE_WEIGHT_CV_FOLDS))
    _set_if_missing(args, provided_options, "--win-candidates", "win_candidates", ",".join(f"{float(value):g}" for value in CLASSIFIER_COMPARE_WIN_CANDIDATES))
    _set_if_missing(args, provided_options, "--quick-screen-top-k", "quick_screen_top_k", int(CLASSIFIER_COMPARE_QUICK_SCREEN_TOP_K))
    _set_if_missing(args, provided_options, "--force-include-models", "force_include_models", ",".join(CLASSIFIER_COMPARE_FORCE_INCLUDE_MODELS))
    _set_if_missing(args, provided_options, "--compute-backend", "compute_backend", str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND))
    _set_if_missing(args, provided_options, "--gpu-precision", "gpu_precision", str(TRAIN_EVAL_DEFAULT_GPU_PRECISION))


def _apply_profile_eval_args(args: argparse.Namespace, provided_options: set[str]) -> None:
    args.task = "profile-eval"
    if str(args.profile_eval_mode).strip().lower() == "fbcca-only":
        models = ("fbcca_fixed_all8",)
    else:
        models = PROFILE_EVAL_MODELS
    _set_if_missing(args, provided_options, "--models", "models", ",".join(models))
    _set_if_missing(args, provided_options, "--channel-modes", "channel_modes", ",".join(PROFILE_EVAL_CHANNEL_MODES))
    _set_if_missing(args, provided_options, "--multi-seed-count", "multi_seed_count", int(PROFILE_EVAL_MULTI_SEED_COUNT))
    _set_if_missing(args, provided_options, "--channel-weight-mode", "channel_weight_mode", str(PROFILE_EVAL_CHANNEL_WEIGHT_MODE))
    _set_if_missing(args, provided_options, "--subband-weight-mode", "subband_weight_mode", str(PROFILE_EVAL_SUBBAND_WEIGHT_MODE))
    _set_if_missing(args, provided_options, "--spatial-filter-mode", "spatial_filter_mode", str(PROFILE_EVAL_SPATIAL_FILTER_MODE))
    _set_if_missing(args, provided_options, "--joint-weight-iters", "joint_weight_iters", int(PROFILE_EVAL_JOINT_WEIGHT_ITERS))
    _set_if_missing(args, provided_options, "--weight-cv-folds", "weight_cv_folds", int(PROFILE_EVAL_WEIGHT_CV_FOLDS))
    _set_if_missing(args, provided_options, "--win-candidates", "win_candidates", ",".join(f"{float(value):g}" for value in PROFILE_EVAL_WIN_CANDIDATES))
    _set_if_missing(args, provided_options, "--quick-screen-top-k", "quick_screen_top_k", int(PROFILE_EVAL_QUICK_SCREEN_TOP_K))
    _set_if_missing(args, provided_options, "--force-include-models", "force_include_models", ",".join(PROFILE_EVAL_FORCE_INCLUDE_MODELS))
    _set_if_missing(args, provided_options, "--compute-backend", "compute_backend", str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND))
    _set_if_missing(args, provided_options, "--gpu-precision", "gpu_precision", str(TRAIN_EVAL_DEFAULT_GPU_PRECISION))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSVEP 训练评测纯 CLI（服务器友好）")
    parser.add_argument("--dataset-manifest", type=Path, default=None, help="session1 manifest path")
    parser.add_argument("--dataset-manifest-session2", type=Path, default=None, help="session2 manifest path")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--include-manifests", type=str, default="", help="comma-separated manifest paths")
    parser.add_argument("--output-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_ROOT / f"offline_train_eval_{_now_stamp()}.json")
    parser.add_argument("--report-root-dir", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--organize-report-dir", type=int, default=1)
    parser.add_argument("--quality-min-sample-ratio", type=float, default=0.9)
    parser.add_argument("--quality-max-retry-count", type=int, default=3)
    parser.add_argument("--strict-protocol-consistency", type=int, default=1)
    parser.add_argument("--strict-subject-consistency", type=int, default=1)
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_BENCHMARK_MODELS))
    parser.add_argument("--channel-modes", type=str, default=",".join(DEFAULT_BENCHMARK_CHANNEL_MODES))
    parser.add_argument("--multi-seed-count", type=int, default=DEFAULT_BENCHMARK_MULTI_SEED_COUNT)
    parser.add_argument("--gate-policy", type=str, default=DEFAULT_GATE_POLICY)
    parser.add_argument("--channel-weight-mode", type=str, default=str(DEFAULT_CHANNEL_WEIGHT_MODE))
    parser.add_argument("--subband-weight-mode", type=str, default=str(DEFAULT_SUBBAND_WEIGHT_MODE))
    parser.add_argument("--spatial-filter-mode", type=str, default=str(DEFAULT_SPATIAL_FILTER_MODE))
    parser.add_argument("--spatial-rank-candidates", type=str, default=",".join(str(v) for v in DEFAULT_SPATIAL_RANK_CANDIDATES))
    parser.add_argument("--joint-weight-iters", type=int, default=1)
    parser.add_argument("--weight-cv-folds", type=int, default=2)
    parser.add_argument("--spatial-source-model", type=str, default=str(DEFAULT_SPATIAL_SOURCE_MODEL))
    parser.add_argument("--metric-scope", type=str, default=DEFAULT_METRIC_SCOPE)
    parser.add_argument("--decision-time-mode", type=str, default=DEFAULT_PAPER_DECISION_TIME_MODE)
    parser.add_argument("--async-decision-time-mode", type=str, default=DEFAULT_ASYNC_DECISION_TIME_MODE)
    parser.add_argument("--data-policy", type=str, default=DEFAULT_DATA_POLICY)
    parser.add_argument("--export-figures", type=int, default=1)
    parser.add_argument("--ranking-policy", type=str, default=DEFAULT_RANKING_POLICY)
    parser.add_argument("--dynamic-stop-enabled", type=int, default=1)
    parser.add_argument("--dynamic-stop-alpha", type=float, default=DEFAULT_DYNAMIC_STOP_ALPHA)
    parser.add_argument("--win-candidates", type=str, default=",".join(f"{item:g}" for item in DEFAULT_WIN_SEC_CANDIDATES))
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument("--evaluation-mode", type=str, default=DEFAULT_EVALUATION_MODE)
    parser.add_argument("--quick-screen-top-k", type=int, default=DEFAULT_QUICK_SCREEN_TOP_K)
    parser.add_argument("--force-include-models", type=str, default=",".join(DEFAULT_FORCE_INCLUDE_MODELS))
    parser.add_argument("--progress-heartbeat-sec", type=float, default=DEFAULT_PROGRESS_HEARTBEAT_SEC)
    parser.add_argument("--compute-backend", type=str, default=TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-precision", type=str, default=TRAIN_EVAL_DEFAULT_GPU_PRECISION)
    parser.add_argument("--gpu-warmup", type=int, default=1)
    parser.add_argument("--gpu-cache-policy", type=str, default=DEFAULT_GPU_CACHE_MODE)
    parser.add_argument("--quick-mode", type=int, default=0)
    parser.add_argument("--pretrained-profile", type=Path, default=None)
    parser.add_argument("--weight-aggregation", type=str, default=DEFAULT_WEIGHT_AGGREGATION)
    parser.add_argument("--idle-fp-hard-th", type=float, default=DEFAULT_IDLE_FP_HARD_TH)
    parser.add_argument("--channel-weight-l2", type=float, default=DEFAULT_CHANNEL_WEIGHT_L2)
    parser.add_argument("--subband-prior-strength", type=float, default=DEFAULT_SUBBAND_PRIOR_STRENGTH)
    parser.add_argument("--control-state-mode", type=str, default=DEFAULT_CONTROL_STATE_MODE)
    parser.add_argument("--long-idle-required", type=int, default=0)
    parser.add_argument(
        "--profile-eval-mode",
        type=str,
        default="fbcca-vs-all",
        choices=["fbcca-vs-all", "fbcca-only"],
    )
    parser.add_argument("--freeze-profile-weights", type=int, default=1)
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TRAIN_EVAL_TASK,
        choices=[
            "fbcca-weights",
            "model-compare",
            "fbcca-weighted-compare",
            "focused-compare",
            "classifier-compare",
            "profile-eval",
        ],
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    provided_options = _collect_provided_option_names(argv_list)
    args = build_parser().parse_args(argv_list)
    task = _parse_task(args.task)
    if bool(int(args.quick_mode)) or task == "fbcca-weights":
        _apply_quick_mode_args(args, provided_options)
    elif task == "fbcca-weighted-compare":
        _apply_weighted_compare_args(args, provided_options)
    elif task == "focused-compare":
        _apply_focused_compare_args(args, provided_options)
    elif task == "classifier-compare":
        _apply_classifier_compare_args(args, provided_options)
    elif task == "model-compare":
        _apply_model_compare_args(args, provided_options)
    elif task == "profile-eval":
        _apply_profile_eval_args(args, provided_options)

    include_manifests = _parse_manifest_csv(args.include_manifests)
    session1 = include_manifests[0] if include_manifests else args.dataset_manifest
    if session1 is None:
        raise ValueError("--dataset-manifest or --include-manifests is required")

    config = OfflineTrainEvalConfig(
        dataset_manifest_session1=Path(session1).expanduser().resolve(),
        dataset_manifest_session2=(
            None
            if args.dataset_manifest_session2 is None
            else Path(args.dataset_manifest_session2).expanduser().resolve()
        ),
        dataset_manifests=include_manifests,
        output_profile_path=Path(args.output_profile).expanduser().resolve(),
        report_path=Path(args.report_path).expanduser().resolve(),
        report_root_dir=Path(args.report_root_dir).expanduser().resolve(),
        organize_report_dir=bool(int(args.organize_report_dir)),
        dataset_selection_snapshot={
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "selected_manifests": [str(path) for path in include_manifests],
            "selected_manifest_count": int(len(include_manifests)),
            "quality_min_sample_ratio": float(args.quality_min_sample_ratio),
            "quality_max_retry_count": int(args.quality_max_retry_count),
            "strict_protocol_consistency": bool(int(args.strict_protocol_consistency)),
            "strict_subject_consistency": bool(int(args.strict_subject_consistency)),
        },
        quality_min_sample_ratio=float(args.quality_min_sample_ratio),
        quality_max_retry_count=int(args.quality_max_retry_count),
        strict_protocol_consistency=bool(int(args.strict_protocol_consistency)),
        strict_subject_consistency=bool(int(args.strict_subject_consistency)),
        model_names=tuple(parse_model_list(args.models)),
        channel_modes=tuple(parse_channel_mode_list(args.channel_modes)),
        multi_seed_count=int(args.multi_seed_count),
        gate_policy=parse_gate_policy(args.gate_policy),
        channel_weight_mode=parse_channel_weight_mode(args.channel_weight_mode),
        subband_weight_mode=parse_subband_weight_mode(str(args.subband_weight_mode).strip()),
        spatial_filter_mode=parse_spatial_filter_mode(str(args.spatial_filter_mode).strip()),
        spatial_rank_candidates=tuple(parse_spatial_rank_candidates(str(args.spatial_rank_candidates))),
        joint_weight_iters=max(1, int(args.joint_weight_iters)),
        weight_cv_folds=max(2, int(args.weight_cv_folds)),
        spatial_source_model=parse_spatial_source_model(str(args.spatial_source_model).strip()),
        metric_scope=parse_metric_scope(args.metric_scope),
        decision_time_mode=parse_decision_time_mode(args.decision_time_mode),
        async_decision_time_mode=parse_decision_time_mode(args.async_decision_time_mode),
        data_policy=parse_data_policy(args.data_policy),
        export_figures=bool(int(args.export_figures)),
        ranking_policy=parse_ranking_policy(args.ranking_policy),
        dynamic_stop_enabled=bool(int(args.dynamic_stop_enabled)),
        dynamic_stop_alpha=float(args.dynamic_stop_alpha),
        win_candidates=tuple(float(item.strip()) for item in str(args.win_candidates).split(",") if item.strip()),
        seed=int(args.seed),
        task=_parse_task(args.task),
        evaluation_mode=str(args.evaluation_mode),
        quick_screen_top_k=max(1, int(args.quick_screen_top_k)),
        force_include_models=tuple(parse_model_list(str(args.force_include_models))),
        progress_heartbeat_sec=float(args.progress_heartbeat_sec),
        compute_backend=parse_compute_backend_name(str(args.compute_backend).strip()),
        gpu_device=int(args.gpu_device),
        gpu_precision=parse_gpu_precision(str(args.gpu_precision).strip()),
        gpu_warmup=bool(int(args.gpu_warmup)),
        gpu_cache_policy=parse_gpu_cache_policy(str(args.gpu_cache_policy).strip()),
        pretrained_profile_path=(
            None if args.pretrained_profile is None else Path(args.pretrained_profile).expanduser().resolve()
        ),
        profile_eval_mode=str(args.profile_eval_mode).strip().lower(),
        freeze_profile_weights=bool(int(args.freeze_profile_weights)),
        weight_aggregation=parse_weight_aggregation(str(args.weight_aggregation).strip()),
        idle_fp_hard_th=float(args.idle_fp_hard_th),
        channel_weight_l2=max(0.0, float(args.channel_weight_l2)),
        subband_prior_strength=max(0.0, float(args.subband_prior_strength)),
        control_state_mode=parse_control_state_mode(str(args.control_state_mode).strip()),
        long_idle_required=bool(int(args.long_idle_required)),
    )
    run_offline_train_eval(config, log_fn=lambda text: print(text, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
