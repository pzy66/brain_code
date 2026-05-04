from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import product
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from brain_workspace.paths import SSVEP_DATASET_DIR

from . import fbcca_local_opt as _fbcca
from . import tdca_local_opt as _tdca
from .async_fbcca_idle_standalone import (
    DEFAULT_ASYNC_DECISION_TIME_MODE,
    DEFAULT_CONTROL_STATE_MODE,
    DEFAULT_GATE_POLICY,
    DEFAULT_GPU_CACHE_MODE,
    DEFAULT_GPU_DEVICE_ID,
    DEFAULT_GPU_PRECISION_NAME,
    DEFAULT_NH,
    DEFAULT_PAPER_DECISION_TIME_MODE,
    ThresholdProfile,
    TrialSpec,
    atomic_write_text,
    compute_classification_metrics,
    fit_threshold_profile,
    json_dumps,
    json_safe,
    save_profile,
    create_decoder,
)
from .decision import DecisionEngine
from .external_replay_dataset import (
    EXTERNAL_LED_CHANNELS,
    EXTERNAL_LED_FREQS,
    ExternalReplayDataset,
    ExternalReplaySession,
    ExternalReplayTrial,
    collect_trial_segments,
    discover_external_replay_subjects,
    load_external_replay_dataset,
)
from .gating import CorrectnessCalibrator, PerFrequencyLogRegGate, RollingFeatureHistory
from .gating.correctness_calibrator import (
    BAYESIAN_GAP_GMM,
    GLOBAL_CORRECTNESS_LOGISTIC,
)
from .profile_v2 import DEFAULT_GATE_FEATURES, build_profile_v2
from .run_artifacts import make_run_tag, resolve_ssvep_run_artifacts
from .trial_roles import resolve_trial_role


DEFAULT_FBCCA_EXTERNAL_TASK = "fbcca-external-replay-opt"
DEFAULT_FBCCA_EXTERNAL_MODEL = "fbcca"
DEFAULT_FBCCA_EXTERNAL_CHANNEL_MODE = "all8"
DEFAULT_FBCCA_EXTERNAL_DATASET_ROOT = (
    SSVEP_DATASET_DIR
    / "external"
    / "dataset_ssvep_led_github"
)
DEFAULT_FBCCA_EXTERNAL_SEARCH_PRESET = "reduced24"
FBCCA_EXTERNAL_SEARCH_PRESETS = ("smoke8", "reduced24")
DEFAULT_FBCCA_EXTERNAL_OUTER_EVAL = "loso4"
FBCCA_EXTERNAL_OUTER_EVALS = ("chronological-last", "loso4")
DEFAULT_FBCCA_EXTERNAL_REPLAY_SPEED = "1x"
FBCCA_EXTERNAL_REPLAY_SPEEDS = ("1x", "2x", "5x", "max")
DEFAULT_FBCCA_EXTERNAL_STEP_SEC = 0.25
DEFAULT_FBCCA_EXTERNAL_CCA_FAMILY_SPECS = (
    {
        "decoder_variant": "fbcca_fixed_all8",
        "model_name": "fbcca",
        "fbcca_variant": "fbcca_fixed_all8",
        "decoder_family_variant": "fbcca",
        "template_usage": "none",
    },
    {
        "decoder_variant": "fbcca_sw_all8",
        "model_name": "fbcca",
        "fbcca_variant": "fbcca_sw_all8",
        "decoder_family_variant": "fbcca",
        "template_usage": "none",
    },
    {
        "decoder_variant": "itcca_all8",
        "model_name": "itcca",
        "fbcca_variant": "itcca_all8",
        "decoder_family_variant": "itcca",
        "template_usage": "individual_template",
    },
    {
        "decoder_variant": "ecca_all8",
        "model_name": "ecca",
        "fbcca_variant": "ecca_all8",
        "decoder_family_variant": "ecca",
        "template_usage": "extended_template",
    },
)
DEFAULT_FBCCA_EXTERNAL_CONFIDENCE_VARIANTS = (
    GLOBAL_CORRECTNESS_LOGISTIC,
    BAYESIAN_GAP_GMM,
)
DEFAULT_FBCCA_EXTERNAL_WIN_CANDIDATES = (2.0, 2.5, 3.0)
DEFAULT_FBCCA_EXTERNAL_STAGE_A_WIN_CANDIDATES = (2.5,)
DEFAULT_FBCCA_EXTERNAL_TRANSFER_WIN_CANDIDATES = (2.5, 3.0)
DEFAULT_EXTERNAL_CHANNEL_MONTAGE_NAME = "external_led_8ch"
DEFAULT_SIMULATION_PROTOCOL = "continuous_session_replay"
DEFAULT_DEPLOYMENT_VIEW = "chronological_last_session"
DEFAULT_DECISION_SEARCH_TARGET = "tune_split"
DEFAULT_FINAL_SELECTION_TARGET = "holdout_split"
DEFAULT_SIMULATION_ONLY_PROFILE = True
DEFAULT_EXTERNAL_TRANSFER_MODE = "none"
EXTERNAL_TRANSFER_MODES = ("none", "sd_lst")
DEFAULT_EXTERNAL_SOURCE_SUBJECT_SELECTOR = "similarity_top2"
DEFAULT_EXTERNAL_TRANSFER_DECODER_VARIANTS = frozenset(
    {"cca_itcca_combo_all8", "ecca_paper_all8", "trca_all8", "etrca_r_all8"}
)
DEFAULT_TDCA_SANITY_VARIANT = "tdca_like_legacy"
DEFAULT_TDCA_SANITY_WIN_CANDIDATES = (2.0, 3.0)
DEFAULT_EXTERNAL_DECODER_PRIORITY = {
    "fbcca_fixed_all8": 0,
    "cca_itcca_combo_all8": 1,
    "ecca_paper_all8": 2,
    "trca_all8": 3,
    "etrca_r_all8": 4,
    "fbcca_sw_all8": 5,
    "itcca_all8": 6,
    "ecca_all8": 7,
}
DEFAULT_EXTERNAL_FREQUENCY_MIN_RAW_CORRECT_RATE = 0.70
DEFAULT_EXTERNAL_FREQUENCY_MIN_GATE_PASS_RATE = 0.20
DEFAULT_EXTERNAL_MAX_CONFIDENCE_REJECT_RATIO = 0.60
DEFAULT_EXTERNAL_MIN_REFERENCE_CONTROL_TRIALS = 4
DEFAULT_EXTERNAL_MIN_REFERENCE_POSITIVE_WINDOWS = 24
DEFAULT_EXTERNAL_MIN_REFERENCE_NEGATIVE_WINDOWS = 24
DEFAULT_EXTERNAL_MIN_REFERENCE_POSITIVE_TRIALS = 8
DEFAULT_EXTERNAL_MIN_REFERENCE_NEGATIVE_TRIALS = 8
DEFAULT_EXTERNAL_REFERENCE_ENTER_FLOOR = 0.40
DEFAULT_EXTERNAL_REFERENCE_MAX_GLOBAL_SHIFT = 0.10
DEFAULT_EXTERNAL_REFERENCE_MAX_ENTER = 0.90
DEFAULT_EXTERNAL_DIAGNOSTIC_ONLY_VARIANTS = frozenset({"fbcca_sw_all8", "itcca_all8", "ecca_all8"})


@dataclass(frozen=True)
class FBCCAExternalReplayOptConfig:
    external_dataset_root: Path
    subject: str
    output_profile_path: Path
    report_path: Path
    report_root_dir: Optional[Path] = None
    organize_report_dir: bool = False
    model_names: tuple[str, ...] = (DEFAULT_FBCCA_EXTERNAL_MODEL,)
    channel_modes: tuple[str, ...] = (DEFAULT_FBCCA_EXTERNAL_CHANNEL_MODE,)
    search_preset: str = DEFAULT_FBCCA_EXTERNAL_SEARCH_PRESET
    outer_eval: str = DEFAULT_FBCCA_EXTERNAL_OUTER_EVAL
    replay_speed: str = DEFAULT_FBCCA_EXTERNAL_REPLAY_SPEED
    step_sec: float = DEFAULT_FBCCA_EXTERNAL_STEP_SEC
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


@dataclass(frozen=True)
class ExternalReplayFold:
    fold_index: int
    holdout_session_index: int
    train_session_indices: tuple[int, ...]
    view_name: str
    fingerprint: str


@dataclass(frozen=True)
class ExternalTrialSegment:
    trial_spec: TrialSpec
    segment: np.ndarray
    session_index: int
    session_id: str
    trial_index: int
    label: str
    expected_freq: Optional[float]
    stim_start_sample: int
    stim_stop_sample: int


def _safe_float(value: Any, default: float = 0.0) -> float:
    return _tdca._safe_float(value, default)


def _safe_int(value: Any, default: int = 0) -> int:
    return _tdca._safe_int(value, default)


def _median(values: Sequence[Any], default: float = 0.0) -> float:
    return _tdca._median(values, default)


def _normalize_name_list(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(item).strip().lower() for item in values if str(item).strip())


def _candidate_key(
    *,
    fbcca_variant: str,
    win_sec: float,
    confidence_variant: str,
    transfer_mode: str = DEFAULT_EXTERNAL_TRANSFER_MODE,
) -> str:
    return (
        f"variant={str(fbcca_variant).strip().lower()}|"
        f"win={float(win_sec):g}|"
        f"confidence={str(confidence_variant).strip().lower()}|"
        f"transfer={str(transfer_mode or DEFAULT_EXTERNAL_TRANSFER_MODE).strip().lower()}"
    )


def _validate_config(config: FBCCAExternalReplayOptConfig) -> None:
    model_names = _normalize_name_list(config.model_names)
    channel_modes = _normalize_name_list(config.channel_modes)
    if model_names != (DEFAULT_FBCCA_EXTERNAL_MODEL,):
        raise ValueError(
            f"{DEFAULT_FBCCA_EXTERNAL_TASK} only supports model_names=('fbcca',); got {config.model_names}"
        )
    if channel_modes != (DEFAULT_FBCCA_EXTERNAL_CHANNEL_MODE,):
        raise ValueError(
            f"{DEFAULT_FBCCA_EXTERNAL_TASK} only supports channel_modes=('all8',); got {config.channel_modes}"
        )
    preset = str(config.search_preset or DEFAULT_FBCCA_EXTERNAL_SEARCH_PRESET).strip().lower()
    if preset not in FBCCA_EXTERNAL_SEARCH_PRESETS:
        raise ValueError(
            f"{DEFAULT_FBCCA_EXTERNAL_TASK} only supports search_preset in {FBCCA_EXTERNAL_SEARCH_PRESETS}; got {config.search_preset}"
        )
    outer_eval = str(config.outer_eval or DEFAULT_FBCCA_EXTERNAL_OUTER_EVAL).strip().lower()
    if outer_eval not in FBCCA_EXTERNAL_OUTER_EVALS:
        raise ValueError(
            f"{DEFAULT_FBCCA_EXTERNAL_TASK} only supports outer_eval in {FBCCA_EXTERNAL_OUTER_EVALS}; got {config.outer_eval}"
        )
    replay_speed = str(config.replay_speed or DEFAULT_FBCCA_EXTERNAL_REPLAY_SPEED).strip().lower()
    if replay_speed not in FBCCA_EXTERNAL_REPLAY_SPEEDS:
        raise ValueError(
            f"{DEFAULT_FBCCA_EXTERNAL_TASK} only supports replay_speed in {FBCCA_EXTERNAL_REPLAY_SPEEDS}; got {config.replay_speed}"
        )


def _resolve_search_plan(config: FBCCAExternalReplayOptConfig) -> dict[str, Any]:
    preset = str(config.search_preset or DEFAULT_FBCCA_EXTERNAL_SEARCH_PRESET).strip().lower()
    if preset == "smoke8":
        decoder_specs = tuple(dict(item) for item in DEFAULT_FBCCA_EXTERNAL_CCA_FAMILY_SPECS)
        win_candidates = DEFAULT_FBCCA_EXTERNAL_STAGE_A_WIN_CANDIDATES
        confidence_variants = DEFAULT_FBCCA_EXTERNAL_CONFIDENCE_VARIANTS
        transfer_modes = (DEFAULT_EXTERNAL_TRANSFER_MODE,)
    elif preset == "reduced24":
        decoder_specs = tuple(dict(item) for item in DEFAULT_FBCCA_EXTERNAL_CCA_FAMILY_SPECS)
        win_candidates = DEFAULT_FBCCA_EXTERNAL_WIN_CANDIDATES
        confidence_variants = DEFAULT_FBCCA_EXTERNAL_CONFIDENCE_VARIANTS
        transfer_modes = (DEFAULT_EXTERNAL_TRANSFER_MODE,)
    else:
        raise ValueError(
            f"{DEFAULT_FBCCA_EXTERNAL_TASK} only supports search_preset in {FBCCA_EXTERNAL_SEARCH_PRESETS}; got {config.search_preset}"
        )
    candidate_grid = [
        {
            "model_name": str(decoder_spec["model_name"]),
            "decoder_variant": str(decoder_spec["decoder_variant"]),
            "fbcca_variant": str(decoder_spec["fbcca_variant"]),
            "decoder_family_variant": str(decoder_spec["decoder_family_variant"]),
            "template_usage": str(decoder_spec["template_usage"]),
            "confidence_variant": str(confidence_variant),
            "win_sec": float(win_sec),
            "transfer_mode": str(transfer_mode),
        }
        for decoder_spec, confidence_variant, win_sec, transfer_mode in product(
            decoder_specs,
            confidence_variants,
            win_candidates,
            transfer_modes,
        )
    ]
    return {
        "task": DEFAULT_FBCCA_EXTERNAL_TASK,
        "search_preset": str(preset),
        "candidate_grid": candidate_grid,
        "variant_names": tuple(str(item["decoder_variant"]) for item in decoder_specs),
        "confidence_variants": tuple(str(item) for item in confidence_variants),
        "win_candidates": tuple(float(item) for item in win_candidates),
        "transfer_modes": tuple(str(item) for item in transfer_modes),
    }


def _resolve_report_paths(config: FBCCAExternalReplayOptConfig) -> dict[str, Path | str]:
    artifacts = resolve_ssvep_run_artifacts(
        task=DEFAULT_FBCCA_EXTERNAL_TASK,
        report_path=Path(config.report_path).expanduser().resolve(),
        output_profile_path=Path(config.output_profile_path).expanduser().resolve(),
        organize_report_dir=bool(config.organize_report_dir),
        report_root_dir=(
            None
            if config.report_root_dir is None
            else Path(config.report_root_dir).expanduser().resolve()
        ),
        run_tag=make_run_tag(task=DEFAULT_FBCCA_EXTERNAL_TASK),
    )
    return {
        "report_dir": artifacts.run_dir,
        "report_json": artifacts.report_json,
        "report_md": artifacts.report_md,
        "output_profile": artifacts.output_profile,
        "profile_v2": artifacts.profile_v2,
        "canonical_profile": artifacts.run_dir / "profile.json",
        "canonical_profile_v2": artifacts.run_dir / "profile_v2.json",
        "selection_snapshot": artifacts.selection_snapshot,
        "run_config": artifacts.run_config,
        "run_log": artifacts.run_log,
        "progress_snapshot": artifacts.progress_snapshot,
        "figures_dir": artifacts.figures_dir,
        "run_tag": artifacts.run_tag,
        "root_dir": artifacts.root_dir,
    }


def _copy_artifact_alias(*, source: Path, destination: Path) -> None:
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    if src == dst:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _append_run_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(str(text).rstrip() + "\n")


def _progress_label(stage: str) -> str:
    labels = {
        "prepare": "准备",
        "candidate_search": "候选搜索",
        "decision_search": "Decision 搜索",
        "holdout_replay": "回放评估",
        "sanity_compare": "TDCA 对照",
        "finalize": "整理结果",
        "complete": "完成",
    }
    return str(labels.get(str(stage), stage))


def _progress_percent(stage: str, *, run_index: float, run_total: float) -> int:
    total = max(float(run_total), 1.0)
    frac = float(max(min(float(run_index), total), 0.0)) / float(total)
    base = {
        "prepare": 5.0,
        "candidate_search": 45.0,
        "decision_search": 75.0,
        "holdout_replay": 90.0,
        "sanity_compare": 95.0,
        "finalize": 98.0,
        "complete": 100.0,
    }.get(str(stage), 0.0)
    span = {
        "prepare": 5.0,
        "candidate_search": 40.0,
        "decision_search": 30.0,
        "holdout_replay": 15.0,
        "sanity_compare": 3.0,
        "finalize": 2.0,
        "complete": 0.0,
    }.get(str(stage), 0.0)
    return int(max(0, min(100, round(base + span * frac))))


def _resolve_parallel_fold_workers(config: FBCCAExternalReplayOptConfig, *, fold_count: int) -> int:
    if int(fold_count) <= 1:
        return 1
    backend = str(config.compute_backend or "auto").strip().lower()
    if backend in {"gpu", "cuda", "cupy"}:
        return 1
    cpu_count = max(int(os.cpu_count() or 1), 1)
    return max(1, min(int(fold_count), min(cpu_count, 4)))


def _external_gate_grid(search_preset: str) -> list[tuple[float, float, int, int, int]]:
    preset = str(search_preset or DEFAULT_FBCCA_EXTERNAL_SEARCH_PRESET).strip().lower()
    if preset == "smoke8":
        return [
            (float(enter_p_th), float(exit_p_th), int(min_enter_windows), 1, 1)
            for enter_p_th, exit_p_th, min_enter_windows in product(
                (0.50, 0.70),
                (0.15, 0.35),
                (1, 2),
            )
        ]
    return [
        (float(enter_p_th), float(exit_p_th), int(min_enter_windows), int(min_exit_windows), int(min_switch_windows))
        for enter_p_th, exit_p_th, min_enter_windows, min_exit_windows, min_switch_windows in product(
            _tdca.DEFAULT_GATE_ENTER_P_GRID,
            _tdca.DEFAULT_GATE_EXIT_P_GRID,
            (1, 2),
            (1, 2),
            (1, 2),
        )
    ]


def _external_decision_param_grid(search_preset: str) -> list[dict[str, Any]]:
    preset = str(search_preset or DEFAULT_FBCCA_EXTERNAL_SEARCH_PRESET).strip().lower()
    if preset == "smoke8":
        return [
            {
                "candidate_min_windows": int(candidate_min_windows),
                "armed_min_windows": int(armed_min_windows),
                "lambda_decay": float(lambda_decay),
                "upper_commit_th": float(upper_commit_th),
                "lower_idle_th": float(lower_idle_th),
                "refractory_sec": float(refractory_sec),
            }
            for candidate_min_windows, armed_min_windows, lambda_decay, upper_commit_th, lower_idle_th, refractory_sec in product(
                (1, 2),
                (1, 2),
                (0.75, 0.85),
                (0.0, 0.4, 0.8),
                (-0.8, -0.4, 0.0),
                (0.0, 0.4),
            )
        ]
    return [dict(item) for item in _tdca._decision_param_grid()]


def _variant_priority(fbcca_variant: Optional[str]) -> int:
    variant = str(fbcca_variant or "").strip().lower()
    return int(DEFAULT_EXTERNAL_DECODER_PRIORITY.get(variant, len(DEFAULT_EXTERNAL_DECODER_PRIORITY)))


def _fbcca_metadata(fbcca_variant: Optional[str]) -> dict[str, Any]:
    return _fbcca._fbcca_variant_metadata(fbcca_variant)


def _is_diagnostic_only_variant(decoder_variant: Optional[str]) -> bool:
    return str(decoder_variant or "").strip().lower() in DEFAULT_EXTERNAL_DIAGNOSTIC_ONLY_VARIANTS


def _count_tune_origins(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        origin = str(row.get("tune_origin", "") or "").strip().lower()
        if not origin:
            continue
        counts[origin] = int(counts.get(origin, 0) or 0) + 1
    return counts


def _assert_external_tune_rows_oof_only(rows: Sequence[Mapping[str, Any]]) -> None:
    counts = _count_tune_origins(rows)
    invalid_origins = sorted(origin for origin in counts if origin != "train_oof")
    if invalid_origins:
        raise ValueError(
            "external replay tune rows must be OOF-only; "
            f"found disallowed tune_origin values: {invalid_origins}"
        )


def _trial_level_max_probability(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    grouped: dict[int, float] = {}
    for row in rows:
        trial_id = _safe_int(row.get("trial_id", -1), -1)
        if trial_id < 0:
            continue
        value = _tdca._row_correctness_probability(row)
        if not np.isfinite(value):
            continue
        grouped[trial_id] = float(max(grouped.get(trial_id, float("-inf")), float(value)))
    if not grouped:
        return np.asarray([], dtype=float)
    return np.asarray([grouped[key] for key in sorted(grouped.keys())], dtype=float)


def _quantile_or_nan(values: Sequence[float] | np.ndarray, q: float) -> float:
    array = np.asarray(values, dtype=float)
    if array.size <= 0:
        return float("nan")
    return float(np.quantile(array, float(q)))


def _median_or_nan(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size <= 0:
        return float("nan")
    return float(np.median(array))


def _diagnostic_rank_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, float, float, int, str]:
    invalid_reasons = [str(item) for item in row.get("selection_invalid_reasons", []) or [] if str(item)]
    return (
        float(len(invalid_reasons)),
        _safe_float(dict(row.get("metrics_median", {})).get("idle_fp_per_min"), float("inf")),
        -_safe_float(row.get("min_gate_pass_rate_by_freq"), 0.0),
        _safe_float(dict(row.get("metrics_median", {})).get("release_latency_s"), float("inf")),
        -_safe_float(dict(row.get("metrics_median", {})).get("control_recall_at_3s"), 0.0),
        -_safe_float(dict(row.get("metrics_median", {})).get("control_recall"), 0.0),
        _safe_float(dict(row.get("metrics_median", {})).get("inference_ms"), float("inf")),
        _variant_priority(row.get("fbcca_variant")),
        str(row.get("confidence_variant", "")),
    )


def _strict_selection_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if bool(row.get("selection_eligible", False)) and not bool(row.get("diagnostic_only", False))
    ]


def _invalid_reason_histogram(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reasons = [str(item) for item in row.get("selection_invalid_reasons", []) or [] if str(item)]
        if not reasons:
            counts["eligible"] = int(counts.get("eligible", 0) or 0) + 1
            continue
        for reason in reasons:
            counts[str(reason)] = int(counts.get(str(reason), 0) or 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _diagnostic_best_row(rows: Sequence[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    candidates = [
        {
            **dict(row),
            "diagnostic_rank_key": [float(item) if isinstance(item, (int, float)) else item for item in _diagnostic_rank_key(row)],
        }
        for row in rows
        if bool(row.get("gate_calibration_valid", False))
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            tuple(_diagnostic_rank_key(row)),
            str(row.get("candidate_key", "")),
        )
    )
    return dict(candidates[0])


def _candidate_metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    model_name = str(candidate.get("model_name", DEFAULT_FBCCA_EXTERNAL_MODEL)).strip().lower()
    if model_name == DEFAULT_FBCCA_EXTERNAL_MODEL:
        payload = dict(_fbcca_metadata(candidate.get("fbcca_variant")))
        payload["decoder_family_variant"] = "fbcca"
        payload["template_usage"] = "none"
        payload["decoder_variant"] = str(candidate.get("decoder_variant", candidate.get("fbcca_variant", "")))
        return payload
    if model_name == "cca_itcca_combo":
        return {
            "decoder_variant": str(candidate.get("decoder_variant", "cca_itcca_combo_all8")),
            "decoder_family_variant": "cca_itcca_combo",
            "algorithm_alignment": "paper-faithful",
            "channel_weight_mode": None,
            "subband_weight_mode": None,
            "spatial_filter_mode": None,
            "template_usage": "individual_template",
        }
    if model_name == "itcca":
        return {
            "decoder_variant": str(candidate.get("decoder_variant", "itcca_all8")),
            "decoder_family_variant": "itcca",
            "algorithm_alignment": "engineering-approx",
            "channel_weight_mode": None,
            "subband_weight_mode": None,
            "spatial_filter_mode": None,
            "template_usage": "individual_template",
        }
    if model_name == "ecca":
        return {
            "decoder_variant": str(candidate.get("decoder_variant", "ecca_all8")),
            "decoder_family_variant": "ecca",
            "algorithm_alignment": "engineering-approx",
            "channel_weight_mode": None,
            "subband_weight_mode": None,
            "spatial_filter_mode": None,
            "template_usage": "extended_template",
        }
    if model_name == "ecca_paper":
        return {
            "decoder_variant": str(candidate.get("decoder_variant", "ecca_paper_all8")),
            "decoder_family_variant": "ecca_paper",
            "algorithm_alignment": "paper-faithful",
            "channel_weight_mode": None,
            "subband_weight_mode": None,
            "spatial_filter_mode": None,
            "template_usage": "extended_template",
        }
    if model_name == "trca":
        return {
            "decoder_variant": str(candidate.get("decoder_variant", "trca_all8")),
            "decoder_family_variant": "trca",
            "algorithm_alignment": "paper-faithful",
            "channel_weight_mode": None,
            "subband_weight_mode": None,
            "spatial_filter_mode": None,
            "template_usage": "individual_template",
        }
    if model_name == "trca_r":
        return {
            "decoder_variant": str(candidate.get("decoder_variant", "etrca_r_all8")),
            "decoder_family_variant": "etrca_r",
            "algorithm_alignment": "engineering-approx",
            "channel_weight_mode": None,
            "subband_weight_mode": None,
            "spatial_filter_mode": None,
            "template_usage": "individual_template",
        }
    if model_name == "tdca":
        return {
            "decoder_variant": str(candidate.get("decoder_variant", DEFAULT_TDCA_SANITY_VARIANT)),
            "decoder_family_variant": "tdca",
            "algorithm_alignment": "sanity-tdca-like-legacy",
            "channel_weight_mode": None,
            "subband_weight_mode": None,
            "spatial_filter_mode": None,
            "template_usage": "none",
        }
    return {
        "decoder_variant": str(candidate.get("decoder_variant", model_name)),
        "decoder_family_variant": str(model_name),
        "algorithm_alignment": str(model_name),
        "channel_weight_mode": None,
        "subband_weight_mode": None,
        "spatial_filter_mode": None,
        "template_usage": "none",
    }


def _outer_folds(dataset: ExternalReplayDataset, *, mode: str) -> tuple[ExternalReplayFold, ...]:
    session_count = int(len(dataset.sessions))
    if session_count < 2:
        raise ValueError("external replay dataset requires at least 2 sessions")
    mode_value = str(mode or DEFAULT_FBCCA_EXTERNAL_OUTER_EVAL).strip().lower()
    if mode_value == "chronological-last":
        holdout = int(session_count - 1)
        return (
            ExternalReplayFold(
                fold_index=0,
                holdout_session_index=holdout,
                train_session_indices=tuple(range(holdout)),
                view_name=DEFAULT_DEPLOYMENT_VIEW,
                fingerprint=f"chronological-last:{holdout}",
            ),
        )
    folds: list[ExternalReplayFold] = []
    for holdout in range(session_count):
        folds.append(
            ExternalReplayFold(
                fold_index=int(holdout),
                holdout_session_index=int(holdout),
                train_session_indices=tuple(
                    session_index for session_index in range(session_count) if int(session_index) != int(holdout)
                ),
                view_name="loso4",
                fingerprint=f"loso4:{holdout}",
            )
        )
    return tuple(folds)


def _build_external_segments(
    dataset: ExternalReplayDataset,
    *,
    session_indices: Sequence[int],
    include_rest: bool = True,
) -> list[ExternalTrialSegment]:
    keep = {int(index) for index in session_indices}
    output: list[ExternalTrialSegment] = []
    global_trial_id = 0
    for session in dataset.sessions:
        if int(session.session_index) not in keep:
            continue
        for trial in session.trials:
            if trial.expected_freq is None and not include_rest:
                continue
            segment = np.ascontiguousarray(
                session.data[trial.stim_start_sample : trial.stim_stop_sample, :],
                dtype=np.float64,
            )
            output.append(
                ExternalTrialSegment(
                    trial_spec=TrialSpec(
                        label=str(trial.label_name),
                        expected_freq=None if trial.expected_freq is None else float(trial.expected_freq),
                        trial_id=int(global_trial_id),
                        block_index=int(session.session_index),
                    ),
                    segment=segment,
                    session_index=int(session.session_index),
                    session_id=str(session.session_id),
                    trial_index=int(trial.trial_index),
                    label=str(trial.label_name),
                    expected_freq=None if trial.expected_freq is None else float(trial.expected_freq),
                    stim_start_sample=int(trial.stim_start_sample),
                    stim_stop_sample=int(trial.stim_stop_sample),
                )
            )
            global_trial_id += 1
    return output


def _segment_pairs(segments: Sequence[ExternalTrialSegment]) -> list[tuple[TrialSpec, np.ndarray]]:
    return [(item.trial_spec, np.asarray(item.segment, dtype=np.float64)) for item in segments]


def _dataset_summary_rows(dataset: ExternalReplayDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in dataset.sessions:
        active_trials = sum(1 for trial in session.trials if trial.expected_freq is not None)
        rest_trials = sum(1 for trial in session.trials if trial.expected_freq is None)
        rows.append(
            {
                "subject_id": str(dataset.subject_id),
                "session_index": int(session.session_index),
                "session_id": str(session.session_id),
                "file_path": str(session.file_path),
                "sampling_rate": int(session.sampling_rate),
                "duration_sec": float(session.duration_sec),
                "trial_count": int(len(session.trials)),
                "active_trials": int(active_trials),
                "rest_trials": int(rest_trials),
            }
        )
    return rows


def _make_external_merged_dataset(
    dataset: ExternalReplayDataset,
    *,
    segments: Sequence[ExternalTrialSegment],
    session_indices: Sequence[int],
) -> _tdca.MergedLocalDataset:
    trial_role_counts: dict[str, int] = {}
    for item in segments:
        role = "control" if item.expected_freq is not None else "clean_idle"
        trial_role_counts[role] = int(trial_role_counts.get(role, 0) or 0) + 1
    quality_rows = tuple(
        row
        for row in _dataset_summary_rows(dataset)
        if int(row.get("session_index", -1)) in {int(index) for index in session_indices}
    )
    session_ids = tuple(
        str(session.session_id)
        for session in dataset.sessions
        if int(session.session_index) in {int(index) for index in session_indices}
    )
    return _tdca.MergedLocalDataset(
        manifest_paths=tuple(),
        datasets=tuple(),
        trial_segments=tuple(_segment_pairs(segments)),
        sampling_rate=int(dataset.sampling_rate),
        freqs=tuple(float(freq) for freq in dataset.freqs),
        board_eeg_channels=tuple(range(len(dataset.channel_names))),
        subject_id=str(dataset.subject_id),
        session_ids=session_ids,
        trial_role_counts=trial_role_counts,
        quality_rows=quality_rows,
    )


def _build_feature_rows_for_segments(decoder: Any, segments: Sequence[ExternalTrialSegment]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in segments:
        feature_rows = decoder.iter_window_features(
            np.asarray(item.segment, dtype=np.float64),
            expected_freq=item.expected_freq,
            label=item.label,
            trial_id=int(item.trial_spec.trial_id),
            block_index=int(item.session_index),
        )
        for row_raw in feature_rows:
            row = dict(row_raw)
            row["label"] = str(item.label)
            row["expected_freq"] = None if item.expected_freq is None else float(item.expected_freq)
            row["trial_id"] = int(item.trial_spec.trial_id)
            row["block_index"] = int(item.session_index)
            row["session_index"] = int(item.session_index)
            row["session_id"] = str(item.session_id)
            row["trial_index_in_session"] = int(item.trial_index)
            row["stim_start_sample"] = int(item.stim_start_sample)
            row["stim_stop_sample"] = int(item.stim_stop_sample)
            row["trial_role"] = resolve_trial_role(
                {
                    "trial_role": row.get("trial_role"),
                    "label": str(item.label),
                    "expected_freq": item.expected_freq,
                }
            )
            rows.append(row)
    return _tdca._attach_history_features(rows)


def _default_fbcca_model_params(
    *,
    Nh: int,
    fbcca_variant: str,
    frontend_model_params: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    return _fbcca._default_model_params(
        Nh=int(Nh),
        fbcca_variant=str(fbcca_variant),
        frontend_model_params=frontend_model_params,
    )


def _default_candidate_model_params(
    *,
    model_name: str,
    Nh: int,
    decoder_variant: str,
    fbcca_variant: str,
    transfer_mode: str = DEFAULT_EXTERNAL_TRANSFER_MODE,
) -> dict[str, Any]:
    normalized_model = str(model_name).strip().lower()
    if normalized_model == DEFAULT_FBCCA_EXTERNAL_MODEL:
        payload = _default_fbcca_model_params(Nh=int(Nh), fbcca_variant=str(fbcca_variant))
    else:
        payload = {"Nh": int(Nh)}
    payload["decoder_variant"] = str(decoder_variant)
    payload["transfer_mode"] = str(transfer_mode or DEFAULT_EXTERNAL_TRANSFER_MODE)
    return payload


def _optimize_fbcca_frontend_external(
    *,
    dataset: ExternalReplayDataset,
    train_segments: Sequence[ExternalTrialSegment],
    train_session_indices: Sequence[int],
    fbcca_variant: str,
    win_sec: float,
    config: FBCCAExternalReplayOptConfig,
    log: Callable[[str], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    merged = _make_external_merged_dataset(
        dataset,
        segments=train_segments,
        session_indices=train_session_indices,
    )
    full_indices = tuple(range(len(train_segments)))
    split = _tdca.RepeatedGroupSplit(
        repeat_index=0,
        train_indices=full_indices,
        gate_indices=full_indices,
        holdout_indices=full_indices,
        fingerprint="external-train-full",
    )
    local_cfg = _fbcca.FBCCALocalOptConfig(
        dataset_manifest_session1=Path("external://dataset"),
        dataset_manifests=tuple(),
        output_profile_path=Path("external://profile.json"),
        report_path=Path("external://report.json"),
        model_names=(DEFAULT_FBCCA_EXTERNAL_MODEL,),
        channel_modes=(DEFAULT_FBCCA_EXTERNAL_CHANNEL_MODE,),
        search_preset="smoke20",
        compute_backend=str(config.compute_backend),
        gpu_device=int(config.gpu_device),
        gpu_precision=str(config.gpu_precision),
        gpu_warmup=bool(config.gpu_warmup),
        gpu_cache_policy=str(config.gpu_cache_policy),
        control_state_mode=str(config.control_state_mode),
        decision_time_mode=str(config.decision_time_mode),
        async_decision_time_mode=str(config.async_decision_time_mode),
    )
    return _fbcca._optimize_variant_frontend(
        merged_dataset=merged,
        split=split,
        fbcca_variant=str(fbcca_variant),
        win_sec=float(win_sec),
        config=local_cfg,
        log=log,
    )


def _train_rows_valid_for_tuning(tune_summary: Mapping[str, Any]) -> bool:
    return bool(tune_summary.get("valid", False))


def _build_base_profile(
    *,
    model_name: str,
    model_params: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    freqs: Sequence[float],
    effective_raw_window_sec: float,
    dataset: ExternalReplayDataset,
    control_state_mode: str,
    confidence_variant: str,
    training_window_policy: str,
) -> ThresholdProfile:
    control_rows = [dict(row) for row in train_rows if resolve_trial_role(row) == "control"]
    idle_rows = [dict(row) for row in train_rows if resolve_trial_role(row) != "control"]
    control_feature_means, control_feature_stds = _tdca._feature_stats(control_rows)
    idle_feature_means, idle_feature_stds = _tdca._feature_stats(idle_rows)
    return ThresholdProfile(
        freqs=tuple(float(freq) for freq in freqs),  # type: ignore[arg-type]
        win_sec=float(effective_raw_window_sec),
        step_sec=float(DEFAULT_FBCCA_EXTERNAL_STEP_SEC),
        enter_score_th=0.5,
        enter_ratio_th=1.4,
        enter_margin_th=0.2,
        exit_score_th=0.4,
        exit_ratio_th=1.1,
        min_enter_windows=1,
        min_exit_windows=1,
        model_name=str(model_name),
        model_params=dict(model_params),
        eeg_channels=tuple(range(len(dataset.channel_names))),
        control_feature_means=control_feature_means,
        control_feature_stds=control_feature_stds,
        idle_feature_means=idle_feature_means,
        idle_feature_stds=idle_feature_stds,
        control_state_mode=str(control_state_mode),
        enter_p_th=0.65,
        exit_p_th=0.30,
        confidence_variant=str(confidence_variant),
        training_window_policy=str(training_window_policy),
        gate_policy=DEFAULT_GATE_POLICY,
        min_switch_windows=1,
        switch_enter_score_th=0.5,
        switch_enter_ratio_th=1.4,
        switch_enter_margin_th=0.2,
        metadata={
            "channel_montage_name": DEFAULT_EXTERNAL_CHANNEL_MONTAGE_NAME,
            "simulation_only_profile": True,
        },
    )


def _build_candidate_fold_context(
    *,
    dataset: ExternalReplayDataset,
    fold: ExternalReplayFold,
    model_name: str,
    model_params: Mapping[str, Any],
    confidence_variant: str,
    win_sec: float,
    config: FBCCAExternalReplayOptConfig,
    log: Callable[[str], None],
    frontend_cache: Optional[dict[tuple[str, tuple[int, ...], float], tuple[dict[str, Any], dict[str, Any]]]] = None,
) -> dict[str, Any]:
    train_segments = _build_external_segments(
        dataset,
        session_indices=fold.train_session_indices,
        include_rest=True,
    )
    holdout_segments = _build_external_segments(
        dataset,
        session_indices=(fold.holdout_session_index,),
        include_rest=True,
    )
    if not train_segments:
        raise ValueError("external replay train split is empty")
    resolved_model_params = dict(model_params)
    frontend_summary: dict[str, Any] = {}
    fbcca_variant = str(model_params.get("fbcca_variant", "") or "")
    if str(model_name) == DEFAULT_FBCCA_EXTERNAL_MODEL and fbcca_variant:
        cache_key = (str(fbcca_variant), tuple(int(item) for item in fold.train_session_indices), float(win_sec))
        if frontend_cache is not None and cache_key in frontend_cache:
            resolved_model_params, frontend_summary = frontend_cache[cache_key]
        else:
            resolved_model_params, frontend_summary = _optimize_fbcca_frontend_external(
                dataset=dataset,
                train_segments=train_segments,
                train_session_indices=fold.train_session_indices,
                fbcca_variant=str(fbcca_variant),
                win_sec=float(win_sec),
                config=config,
                log=log,
            )
            if frontend_cache is not None:
                frontend_cache[cache_key] = (dict(resolved_model_params), dict(frontend_summary))
    decoder = create_decoder(
        str(model_name),
        sampling_rate=int(dataset.sampling_rate),
        freqs=tuple(float(freq) for freq in dataset.freqs),
        win_sec=float(win_sec),
        step_sec=float(config.step_sec),
        model_params=dict(resolved_model_params),
        decoder_compute_backend=str(config.compute_backend),
        gpu_device=int(config.gpu_device),
        gpu_precision=str(config.gpu_precision),
        gpu_warmup=bool(config.gpu_warmup),
        gpu_cache_policy=str(config.gpu_cache_policy),
    )
    fit_pairs = _segment_pairs(train_segments)
    if bool(getattr(decoder, "requires_fit", False)):
        decoder.fit(fit_pairs)
    training_window_policy = str(
        getattr(
            decoder,
            "training_window_policy",
            dict(resolved_model_params).get("training_window_policy", "last_window_only"),
        )
    )
    analysis_latency_sec = _safe_float(
        getattr(decoder, "analysis_latency_sec", dict(resolved_model_params).get("analysis_latency_sec", 0.0)),
        0.0,
    )
    effective_raw_window_sec = _safe_float(
        getattr(decoder, "effective_raw_window_sec", float(win_sec) + float(analysis_latency_sec)),
        float(win_sec) + float(analysis_latency_sec),
    )
    train_rows = _build_feature_rows_for_segments(decoder, train_segments)
    holdout_trial_rows = _build_feature_rows_for_segments(decoder, holdout_segments)
    gate_model = PerFrequencyLogRegGate()
    gate_fit_summary = gate_model.fit(
        rows=train_rows,
        freqs=tuple(float(freq) for freq in dataset.freqs),
        fit_config=_tdca.DEFAULT_LOGREG_FIT_CONFIG,
    )
    oof_train_scored_rows, oof_summary = _tdca._build_oof_train_scored_rows(
        train_rows=train_rows,
        freqs=tuple(float(freq) for freq in dataset.freqs),
        fit_config=_tdca.DEFAULT_LOGREG_FIT_CONFIG,
    )
    scored_train_rows = _tdca._score_rows_with_gate(train_rows, gate=gate_model)
    scored_holdout_trial_rows = _tdca._score_rows_with_gate(holdout_trial_rows, gate=gate_model)
    correctness_calibrator = CorrectnessCalibrator()
    correctness_fit_summary = correctness_calibrator.fit(
        rows=oof_train_scored_rows,
        freqs=tuple(float(freq) for freq in dataset.freqs),
        config=replace(
            _tdca.DEFAULT_CORRECTNESS_CALIBRATOR_CONFIG,
            variant=str(confidence_variant),
        ),
    )
    correctness_fit_summary = {
        **dict(correctness_fit_summary),
        **dict(oof_summary),
    }
    oof_train_scored_rows = _tdca._score_rows_with_correctness(
        oof_train_scored_rows,
        calibrator=correctness_calibrator,
    )
    scored_train_rows = _tdca._score_rows_with_correctness(
        scored_train_rows,
        calibrator=correctness_calibrator,
    )
    scored_holdout_trial_rows = _tdca._score_rows_with_correctness(
        scored_holdout_trial_rows,
        calibrator=correctness_calibrator,
    )
    scored_tune_rows = _tdca._tag_tune_rows(oof_train_scored_rows, origin="train_oof")
    _assert_external_tune_rows_oof_only(scored_tune_rows)
    tune_summary = _tdca._tune_summary(
        scored_tune_rows,
        freqs=tuple(float(freq) for freq in dataset.freqs),
    )
    gate_calibration_summary = _tdca._gate_calibration_summary(
        scored_rows=scored_tune_rows,
        freqs=tuple(float(freq) for freq in dataset.freqs),
        calibrator_summary=correctness_fit_summary,
    )
    gate_calibration_summary = {
        **dict(gate_calibration_summary),
        **dict(tune_summary),
    }
    invalid_reasons = [str(item) for item in gate_calibration_summary.get("invalid_reasons", []) if str(item)]
    if not _train_rows_valid_for_tuning(tune_summary):
        gate_calibration_summary["gate_calibration_valid"] = False
        invalid_reasons.extend(str(item) for item in tune_summary.get("invalid_reasons", []) if str(item))
        invalid_reasons.append("tune_rows_insufficient")
    gate_calibration_summary["invalid_reasons"] = sorted(set(invalid_reasons))
    base_profile = _build_base_profile(
        model_name=str(model_name),
        model_params={
            **dict(resolved_model_params),
            "training_window_policy": str(training_window_policy),
            "analysis_latency_sec": float(analysis_latency_sec),
            "effective_raw_window_sec": float(effective_raw_window_sec),
        },
        train_rows=train_rows,
        freqs=tuple(float(freq) for freq in dataset.freqs),
        effective_raw_window_sec=float(effective_raw_window_sec),
        dataset=dataset,
        control_state_mode=str(config.control_state_mode),
        confidence_variant=str(confidence_variant),
        training_window_policy=str(training_window_policy),
    )
    inference_ms = _tdca._measure_decoder_inference_ms(decoder, fit_pairs)
    try:
        state_payload = decoder.get_state()
    except Exception:
        state_payload = None
    return {
        "fold": fold,
        "decoder": decoder,
        "model_name": str(model_name),
        "decoder_variant": str(model_params.get("decoder_variant", fbcca_variant or model_name)),
        "model_params": dict(resolved_model_params),
        "fbcca_variant": str(fbcca_variant),
        "confidence_variant": str(confidence_variant),
        "frontend_optimization_summary": dict(frontend_summary),
        "train_segments": train_segments,
        "holdout_segments": holdout_segments,
        "train_rows": train_rows,
        "holdout_trial_rows": holdout_trial_rows,
        "oof_train_scored_rows": oof_train_scored_rows,
        "scored_train_rows": scored_train_rows,
        "scored_tune_rows": scored_tune_rows,
        "scored_holdout_trial_rows": scored_holdout_trial_rows,
        "gate_model": gate_model,
        "gate_fit_summary": gate_fit_summary,
        "correctness_calibrator": correctness_calibrator,
        "correctness_fit_summary": correctness_fit_summary,
        "gate_calibration_summary": gate_calibration_summary,
        "tune_summary": tune_summary,
        "base_profile": base_profile,
        "state_payload": state_payload,
        "inference_ms": float(inference_ms),
        "training_window_policy": str(training_window_policy),
        "analysis_latency_sec": float(analysis_latency_sec),
        "effective_raw_window_sec": float(effective_raw_window_sec),
        "confidence_training_scheme": str(
            correctness_fit_summary.get("confidence_training_scheme", _tdca.DEFAULT_CONFIDENCE_TRAINING_SCHEME)
        ),
        "oof_group_key": str(correctness_fit_summary.get("oof_group_key", "")),
        "oof_group_count": int(correctness_fit_summary.get("oof_group_count", 0) or 0),
        "sample_weight_mode": str(correctness_fit_summary.get("sample_weight_mode", "")),
        "positive_trials": int(correctness_fit_summary.get("positive_trials", 0) or 0),
        "negative_trials": int(correctness_fit_summary.get("negative_trials", 0) or 0),
    }


def _external_rank_metrics_key(metrics: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        _safe_float(metrics.get("idle_fp_per_min"), float("inf")),
        _safe_float(metrics.get("release_latency_s"), float("inf")),
        -_safe_float(metrics.get("control_recall_at_3s"), 0.0),
        -_safe_float(metrics.get("control_recall"), 0.0),
        _safe_float(metrics.get("inference_ms"), float("inf")),
    )


def _build_tune_frequency_breakdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    freqs: Sequence[float],
) -> list[dict[str, Any]]:
    _assert_external_tune_rows_oof_only(rows)
    breakdown: list[dict[str, Any]] = []
    for freq in freqs:
        freq_value = float(freq)
        freq_rows = [
            dict(row)
            for row in rows
            if row.get("pred_freq") is not None
            and abs(_safe_float(row.get("pred_freq"), float("nan")) - freq_value) <= 1e-8
        ]
        positive_rows = [row for row in freq_rows if float(_tdca._row_correctness_label(row)) > 0.5]
        negative_rows = [row for row in freq_rows if float(_tdca._row_correctness_label(row)) <= 0.5]
        positive_trial_max = _trial_level_max_probability(positive_rows)
        negative_trial_max = _trial_level_max_probability(negative_rows)
        control_trials = {
            _safe_int(row.get("trial_id", -1), -1)
            for row in rows
            if resolve_trial_role(row) == "control"
            and row.get("expected_freq") is not None
            and abs(_safe_float(row.get("expected_freq"), float("nan")) - freq_value) <= 1e-8
        }
        p_values = [_tdca._row_correctness_probability(row) for row in freq_rows]
        logits = [
            _safe_float(row.get("correctness_logit"), float("nan"))
            for row in freq_rows
            if np.isfinite(_safe_float(row.get("correctness_logit"), float("nan")))
        ]
        breakdown.append(
            {
                "freq": float(freq_value),
                "freq_label": _tdca._freq_label(float(freq_value)),
                "row_count": int(len(freq_rows)),
                "control_trial_count": int(len({item for item in control_trials if item >= 0})),
                "positive_windows": int(len(positive_rows)),
                "negative_windows": int(len(negative_rows)),
                "positive_trials": int(positive_trial_max.size),
                "negative_trials": int(negative_trial_max.size),
                "positive_rate": float(len(positive_rows) / float(max(len(freq_rows), 1))),
                "median_p_correct": None if not p_values else float(np.median(np.asarray(p_values, dtype=float))),
                "median_correctness_logit": None if not logits else float(np.median(np.asarray(logits, dtype=float))),
                "positive_trial_max_p50": None if positive_trial_max.size <= 0 else float(np.median(positive_trial_max)),
                "positive_trial_max_p75": None if positive_trial_max.size <= 0 else float(np.quantile(positive_trial_max, 0.75)),
                "negative_trial_max_p90": None if negative_trial_max.size <= 0 else float(np.quantile(negative_trial_max, 0.90)),
            }
        )
    return breakdown


def _select_probability_threshold(
    *,
    positive_probs: np.ndarray,
    negative_probs: np.ndarray,
    global_reference: float,
    mode: str,
) -> float:
    positive = np.asarray(positive_probs, dtype=float).reshape(-1)
    negative = np.asarray(negative_probs, dtype=float).reshape(-1)
    global_value = float(np.clip(_safe_float(global_reference, 0.5), 1e-6, 1.0 - 1e-6))
    if positive.size <= 0 or negative.size <= 0:
        return float(global_value)
    if str(mode) == "exit":
        quantiles = (
            *(float(np.quantile(positive, q)) for q in (0.02, 0.05, 0.10, 0.20, 0.30)),
            *(float(np.quantile(negative, q)) for q in (0.50, 0.60, 0.70, 0.80, 0.90)),
            float(global_value),
        )
    else:
        quantiles = (
            *(float(np.quantile(positive, q)) for q in (0.05, 0.10, 0.20, 0.30, 0.40)),
            *(float(np.quantile(negative, q)) for q in (0.80, 0.85, 0.90, 0.95)),
            float(global_value),
        )
    candidates = sorted(
        {
            float(np.clip(_safe_float(item, global_value), 1e-6, 1.0 - 1e-6))
            for item in quantiles
            if np.isfinite(_safe_float(item, float("nan")))
        }
    )
    if not candidates:
        return float(global_value)
    best_threshold = float(global_value)
    best_objective: Optional[tuple[float, float, float]] = None
    for candidate in candidates:
        if str(mode) == "exit":
            control_drop_rate = float(np.mean(positive < candidate))
            idle_clear_rate = float(np.mean(negative < candidate))
            objective = (
                float(control_drop_rate),
                -float(idle_clear_rate),
                float(abs(candidate - global_value)),
            )
        else:
            idle_fp_rate = float(np.mean(negative >= candidate))
            control_recall = float(np.mean(positive >= candidate))
            objective = (
                float(idle_fp_rate),
                -float(control_recall),
                float(abs(candidate - global_value)),
            )
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_threshold = float(candidate)
    return float(best_threshold)


def _reference_shrinkage_weight(
    *,
    control_trials: int,
    positive_windows: int,
    negative_windows: int,
    positive_trials: int = 0,
    negative_trials: int = 0,
) -> float:
    trial_cap = 1.0
    if positive_trials > 0 or negative_trials > 0:
        trial_cap = min(
            1.0,
            max(float(control_trials), 0.0) / 8.0,
            max(float(positive_trials), 0.0) / 12.0,
            max(float(negative_trials), 0.0) / 12.0,
        )
    return float(
        min(
            1.0,
            float(trial_cap),
            max(float(control_trials), 0.0) / 8.0,
            max(float(positive_windows), 0.0) / 96.0,
            max(float(negative_windows), 0.0) / 96.0,
        )
    )


def _compute_per_frequency_reference_overrides(
    *,
    scored_rows: Sequence[Mapping[str, Any]],
    freqs: Sequence[float],
    global_enter_p_th: float,
    global_exit_p_th: float,
    base_payloads: Optional[Mapping[str, Any]] = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    _assert_external_tune_rows_oof_only(scored_rows)
    payloads: dict[str, dict[str, Any]] = {}
    board: list[dict[str, Any]] = []
    base_map = dict(base_payloads or {})
    for freq in freqs:
        freq_value = float(freq)
        freq_key = _tdca._freq_label(freq_value)
        freq_rows = [
            dict(row)
            for row in scored_rows
            if row.get("pred_freq") is not None
            and abs(_safe_float(row.get("pred_freq"), float("nan")) - freq_value) <= 1e-8
        ]
        positive_rows = [row for row in freq_rows if float(_tdca._row_correctness_label(row)) > 0.5]
        negative_rows = [row for row in freq_rows if float(_tdca._row_correctness_label(row)) <= 0.5]
        control_trials = {
            _safe_int(row.get("trial_id", -1), -1)
            for row in scored_rows
            if resolve_trial_role(row) == "control"
            and row.get("expected_freq") is not None
            and abs(_safe_float(row.get("expected_freq"), float("nan")) - freq_value) <= 1e-8
        }
        control_trial_count = int(len({item for item in control_trials if item >= 0}))
        positive_probs = np.asarray([_tdca._row_correctness_probability(row) for row in positive_rows], dtype=float)
        negative_probs = np.asarray([_tdca._row_correctness_probability(row) for row in negative_rows], dtype=float)
        positive_trial_max = _trial_level_max_probability(positive_rows)
        negative_trial_max = _trial_level_max_probability(negative_rows)
        positive_trial_count = int(positive_trial_max.size)
        negative_trial_count = int(negative_trial_max.size)
        positive_trial_max_p50 = _median_or_nan(positive_trial_max)
        positive_trial_max_p75 = _quantile_or_nan(positive_trial_max, 0.75)
        negative_trial_max_p90 = _quantile_or_nan(negative_trial_max, 0.90)
        valid = bool(
            control_trial_count >= int(DEFAULT_EXTERNAL_MIN_REFERENCE_CONTROL_TRIALS)
            and positive_trial_count >= int(DEFAULT_EXTERNAL_MIN_REFERENCE_POSITIVE_TRIALS)
            and negative_trial_count >= int(DEFAULT_EXTERNAL_MIN_REFERENCE_NEGATIVE_TRIALS)
        )
        raw_enter = float(global_enter_p_th)
        raw_exit = float(global_exit_p_th)
        shrinkage = 0.0
        adaptation_mode = "global_fallback"
        bound_low = float(
            max(
                float(DEFAULT_EXTERNAL_REFERENCE_ENTER_FLOOR),
                float(global_enter_p_th) - float(DEFAULT_EXTERNAL_REFERENCE_MAX_GLOBAL_SHIFT),
            )
        )
        bound_high = float(
            min(
                float(DEFAULT_EXTERNAL_REFERENCE_MAX_ENTER),
                float(global_enter_p_th) + float(DEFAULT_EXTERNAL_REFERENCE_MAX_GLOBAL_SHIFT),
                positive_trial_max_p75 if np.isfinite(positive_trial_max_p75) else float(global_enter_p_th),
            )
        )
        reference_bound_applied = False
        if valid:
            raw_enter = _select_probability_threshold(
                positive_probs=positive_probs,
                negative_probs=negative_probs,
                global_reference=float(global_enter_p_th),
                mode="enter",
            )
            shrinkage = _reference_shrinkage_weight(
                control_trials=control_trial_count,
                positive_windows=int(positive_probs.size),
                negative_windows=int(negative_probs.size),
                positive_trials=int(positive_trial_count),
                negative_trials=int(negative_trial_count),
            )
            adaptation_mode = "shrunk_freq_specific"
        shrunk_raw_enter = float(
            np.clip(
                float(global_enter_p_th) + float(shrinkage) * (float(raw_enter) - float(global_enter_p_th)),
                1e-6,
                1.0 - 1e-6,
            )
        )
        if valid and np.isfinite(bound_high) and bound_high >= bound_low:
            enter_reference = float(np.clip(float(shrunk_raw_enter), float(bound_low), float(bound_high)))
            reference_bound_applied = bool(abs(float(enter_reference) - float(shrunk_raw_enter)) > 1e-9)
            if reference_bound_applied:
                adaptation_mode = "shrunk_freq_specific_bounded"
        else:
            enter_reference = float(global_enter_p_th)
            if valid:
                adaptation_mode = "global_fallback_bound"
                valid = False
        exit_reference = float(
            np.clip(
                float(global_exit_p_th),
                1e-6,
                1.0 - 1e-6,
            )
        )
        reference_headroom_p50 = (
            float(positive_trial_max_p50 - float(enter_reference))
            if np.isfinite(positive_trial_max_p50)
            else float("nan")
        )
        payload = dict(base_map.get(freq_key, {}))
        payload["enter_p_th"] = float(enter_reference)
        payload["exit_p_th"] = float(exit_reference)
        payload["enter_log_lr_th"] = float(_tdca._p_to_logit(enter_reference, global_enter_p_th))
        payload["exit_log_lr_th"] = float(_tdca._p_to_logit(exit_reference, global_exit_p_th))
        payload["decision_evidence_reference"] = float(_tdca._p_to_logit(enter_reference, global_enter_p_th))
        payload["per_frequency_reference_valid"] = bool(valid)
        payload["per_frequency_enter_reference"] = float(enter_reference)
        payload["per_frequency_exit_reference"] = float(exit_reference)
        payload["per_frequency_reference_shrinkage"] = float(shrinkage)
        payload["per_frequency_reference_mode"] = str(adaptation_mode)
        payload["global_enter_p_th"] = float(global_enter_p_th)
        payload["global_exit_p_th"] = float(global_exit_p_th)
        payload["control_trial_count"] = int(control_trial_count)
        payload["positive_trials"] = int(positive_trial_count)
        payload["negative_trials"] = int(negative_trial_count)
        payload["positive_trial_max_p50"] = None if not np.isfinite(positive_trial_max_p50) else float(positive_trial_max_p50)
        payload["positive_trial_max_p75"] = None if not np.isfinite(positive_trial_max_p75) else float(positive_trial_max_p75)
        payload["negative_trial_max_p90"] = None if not np.isfinite(negative_trial_max_p90) else float(negative_trial_max_p90)
        payload["enter_reference_bound_low"] = float(bound_low)
        payload["enter_reference_bound_high"] = None if not np.isfinite(bound_high) else float(bound_high)
        payload["reference_bound_applied"] = bool(reference_bound_applied)
        payload["reference_headroom_p50"] = None if not np.isfinite(reference_headroom_p50) else float(reference_headroom_p50)
        payloads[freq_key] = payload
        board.append(
            {
                "freq": float(freq_value),
                "control_trial_count": int(control_trial_count),
                "positive_windows": int(positive_probs.size),
                "negative_windows": int(negative_probs.size),
                "positive_trials": int(positive_trial_count),
                "negative_trials": int(negative_trial_count),
                "reference_valid": bool(valid),
                "adaptation_mode": str(adaptation_mode),
                "shrinkage_weight": float(shrinkage),
                "global_enter_p_th": float(global_enter_p_th),
                "global_exit_p_th": float(global_exit_p_th),
                "raw_enter_reference": float(raw_enter),
                "raw_exit_reference": float(raw_exit),
                "positive_trial_max_p50": None if not np.isfinite(positive_trial_max_p50) else float(positive_trial_max_p50),
                "positive_trial_max_p75": None if not np.isfinite(positive_trial_max_p75) else float(positive_trial_max_p75),
                "negative_trial_max_p90": None if not np.isfinite(negative_trial_max_p90) else float(negative_trial_max_p90),
                "enter_reference_bound_low": float(bound_low),
                "enter_reference_bound_high": None if not np.isfinite(bound_high) else float(bound_high),
                "reference_bound_applied": bool(reference_bound_applied),
                "reference_headroom_p50": None if not np.isfinite(reference_headroom_p50) else float(reference_headroom_p50),
                "enter_reference": float(enter_reference),
                "exit_reference": float(exit_reference),
            }
        )
    return payloads, board


def _reference_maps_from_thresholds(
    frequency_specific_thresholds: Optional[Mapping[str, Any]],
    *,
    freqs: Sequence[float],
    global_enter_p_th: float,
    global_exit_p_th: float,
) -> tuple[dict[str, float], dict[str, float]]:
    payloads = dict(frequency_specific_thresholds or {})
    enter_map: dict[str, float] = {}
    exit_map: dict[str, float] = {}
    for freq in freqs:
        freq_key = _tdca._freq_label(float(freq))
        payload = dict(payloads.get(freq_key, {}))
        enter_map[freq_key] = float(
            np.clip(
                _safe_float(payload.get("enter_p_th"), float(global_enter_p_th)),
                1e-6,
                1.0 - 1e-6,
            )
        )
        exit_map[freq_key] = float(
            np.clip(
                _safe_float(payload.get("exit_p_th"), float(global_exit_p_th)),
                1e-6,
                1.0 - 1e-6,
            )
        )
    return enter_map, exit_map


def _aggregate_reference_maps(
    contexts: Sequence[Mapping[str, Any]],
    *,
    freqs: Sequence[float],
) -> tuple[dict[str, float], dict[str, float]]:
    enter_values: dict[str, list[float]] = {_tdca._freq_label(float(freq)): [] for freq in freqs}
    exit_values: dict[str, list[float]] = {_tdca._freq_label(float(freq)): [] for freq in freqs}
    for context in contexts:
        profile = context.get("gate_profile")
        if profile is None:
            continue
        enter_map, exit_map = _reference_maps_from_thresholds(
            getattr(profile, "frequency_specific_thresholds", None),
            freqs=freqs,
            global_enter_p_th=float(getattr(profile, "enter_p_th", 0.65) or 0.65),
            global_exit_p_th=float(getattr(profile, "exit_p_th", 0.30) or 0.30),
        )
        for freq_key, value in enter_map.items():
            enter_values.setdefault(str(freq_key), []).append(float(value))
        for freq_key, value in exit_map.items():
            exit_values.setdefault(str(freq_key), []).append(float(value))
    enter_map = {
        str(freq_key): float(np.median(np.asarray(values, dtype=float))) if values else 0.65
        for freq_key, values in enter_values.items()
    }
    exit_map = {
        str(freq_key): float(np.median(np.asarray(values, dtype=float))) if values else 0.30
        for freq_key, values in exit_values.items()
    }
    return enter_map, exit_map


def _aggregate_reference_diagnostics(
    contexts: Sequence[Mapping[str, Any]],
    *,
    freqs: Sequence[float],
) -> list[dict[str, Any]]:
    payloads_by_freq: dict[str, list[dict[str, Any]]] = {_tdca._freq_label(float(freq)): [] for freq in freqs}
    for context in contexts:
        profile = context.get("gate_profile")
        if profile is None:
            continue
        threshold_payloads = dict(getattr(profile, "frequency_specific_thresholds", None) or {})
        for freq in freqs:
            freq_key = _tdca._freq_label(float(freq))
            payload = dict(threshold_payloads.get(freq_key, {}) or {})
            if payload:
                payloads_by_freq.setdefault(freq_key, []).append(payload)

    def _median_payload_value(entries: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
        values = [
            _safe_float(entry.get(key), float("nan"))
            for entry in entries
            if np.isfinite(_safe_float(entry.get(key), float("nan")))
        ]
        if not values:
            return None
        return float(np.median(np.asarray(values, dtype=float)))

    board: list[dict[str, Any]] = []
    for freq in freqs:
        freq_value = float(freq)
        freq_key = _tdca._freq_label(freq_value)
        entries = list(payloads_by_freq.get(freq_key, []) or [])
        board.append(
            {
                "freq": float(freq_value),
                "freq_label": str(freq_key),
                "fold_count": int(len(entries)),
                "reference_valid_any": bool(any(bool(entry.get("per_frequency_reference_valid", False)) for entry in entries)),
                "reference_valid_all": bool(entries) and all(bool(entry.get("per_frequency_reference_valid", False)) for entry in entries),
                "global_enter_p_th": _median_payload_value(entries, "global_enter_p_th"),
                "enter_reference": _median_payload_value(entries, "per_frequency_enter_reference"),
                "exit_reference": _median_payload_value(entries, "per_frequency_exit_reference"),
                "positive_trial_max_p50": _median_payload_value(entries, "positive_trial_max_p50"),
                "positive_trial_max_p75": _median_payload_value(entries, "positive_trial_max_p75"),
                "negative_trial_max_p90": _median_payload_value(entries, "negative_trial_max_p90"),
                "enter_reference_bound_low": _median_payload_value(entries, "enter_reference_bound_low"),
                "enter_reference_bound_high": _median_payload_value(entries, "enter_reference_bound_high"),
                "reference_bound_applied": bool(any(bool(entry.get("reference_bound_applied", False)) for entry in entries)),
                "reference_headroom_p50": _median_payload_value(entries, "reference_headroom_p50"),
                "positive_trials": int(round(_median_payload_value(entries, "positive_trials") or 0.0)),
                "negative_trials": int(round(_median_payload_value(entries, "negative_trials") or 0.0)),
                "control_trial_count": int(round(_median_payload_value(entries, "control_trial_count") or 0.0)),
            }
        )
    return board


def _selection_validity_summary(
    *,
    replay_frequency_breakdown: Sequence[Mapping[str, Any]],
    decision_bottleneck_summary: Mapping[str, Any],
) -> dict[str, Any]:
    frequency_balance_valid = bool(
        replay_frequency_breakdown
        and all(
            _safe_float(row.get("raw_correct_rate"), 0.0) >= float(DEFAULT_EXTERNAL_FREQUENCY_MIN_RAW_CORRECT_RATE)
            and _safe_float(row.get("gate_pass_rate"), 0.0) >= float(DEFAULT_EXTERNAL_FREQUENCY_MIN_GATE_PASS_RATE)
            for row in replay_frequency_breakdown
        )
    )
    failure_breakdown = dict(decision_bottleneck_summary.get("failure_breakdown", {}) or {})
    decoder_miss = int(failure_breakdown.get("decoder_miss", 0) or 0)
    confidence_reject_miss = int(failure_breakdown.get("confidence_reject_miss", 0) or 0)
    decision_miss = int(failure_breakdown.get("decision_miss", 0) or 0)
    total_failures = int(decoder_miss + confidence_reject_miss + decision_miss)
    confidence_reject_ratio = (
        float(confidence_reject_miss) / float(total_failures)
        if total_failures > 0
        else 0.0
    )
    confidence_dominance_valid = bool(
        total_failures <= 0
        or confidence_reject_ratio <= float(DEFAULT_EXTERNAL_MAX_CONFIDENCE_REJECT_RATIO)
    )
    invalid_reasons: list[str] = []
    if not frequency_balance_valid:
        invalid_reasons.append("frequency_balance_invalid")
    if not confidence_dominance_valid:
        invalid_reasons.append("confidence_dominance_invalid")
    return {
        "frequency_balance_valid": bool(frequency_balance_valid),
        "confidence_dominance_valid": bool(confidence_dominance_valid),
        "confidence_reject_failure_ratio": float(confidence_reject_ratio),
        "selection_eligible": bool(frequency_balance_valid and confidence_dominance_valid),
        "invalid_reasons": list(invalid_reasons),
    }


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    rank_key = tuple(float(item) for item in row.get("rank_key", []))
    return (
        rank_key,
        _variant_priority(row.get("fbcca_variant")),
        _safe_float(dict(row.get("candidate", {})).get("win_sec"), float("inf")),
        str(row.get("confidence_variant", "")),
    )


def _group_trial_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[tuple[int, int], list[dict[str, Any]]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            _safe_int(row.get("session_index", row.get("block_index", -1)), -1),
            _safe_int(row.get("trial_index_in_session", row.get("trial_id", -1)), -1),
        )
        grouped.setdefault(key, []).append(dict(row))
    return sorted(
        [
            (
                key,
                sorted(items, key=lambda item: _safe_int(item.get("window_index", 0), 0)),
            )
            for key, items in grouped.items()
        ],
        key=lambda item: (int(item[0][0]), int(item[0][1])),
    )


def _evaluate_external_trial_rows(
    *,
    scored_rows: Sequence[Mapping[str, Any]],
    profile: ThresholdProfile,
    freqs: Sequence[float],
    decision_params: Mapping[str, Any],
    inference_ms: float,
) -> dict[str, Any]:
    rows = [dict(row) for row in scored_rows]
    if not rows:
        empty_metrics = {
            "idle_fp_per_min": 0.0,
            "control_recall": 0.0,
            "control_recall_at_3s": 0.0,
            "release_detect_rate": 0.0,
            "release_latency_s": float("inf"),
            "detection_latency_s": float("inf"),
            "active_recall": 0.0,
            "first_detection_latency_s": float("inf"),
            "inference_ms": float(inference_ms),
        }
        return {
            "async_metrics": empty_metrics,
            "metrics_4class": {},
            "metrics_2class": {},
            "trial_events": [],
        }

    gate = _tdca.GateReplayState(profile)
    history = RollingFeatureHistory(window_size=4)
    engine = DecisionEngine(_tdca._make_decision_engine_config(profile=profile, decision_params=decision_params))
    grouped_rows = _group_trial_rows(rows)

    control_trials = 0
    control_detected_trials = 0
    control_detected_trials_at_3s = 0
    release_trials = 0
    release_detected_trials = 0
    idle_selected_events = 0
    idle_duration_sec = 0.0
    detection_latencies: list[float] = []
    release_latencies: list[float] = []
    y_active_true: list[str] = []
    y_active_pred: list[str] = []
    y_binary_true: list[str] = []
    y_binary_pred: list[str] = []
    times_active: list[float] = []
    times_binary: list[float] = []
    trial_events: list[dict[str, Any]] = []

    current_session_index: Optional[int] = None
    session_stream_index = 0
    previous_trial_expected: Optional[float] = None
    selected_active_prev = False

    for (session_index, _trial_index), trial_rows in grouped_rows:
        if current_session_index is None or int(session_index) != int(current_session_index):
            current_session_index = int(session_index)
            session_stream_index = 0
            previous_trial_expected = None
            selected_active_prev = False
            gate.reset()
            history.reset()
            engine.reset()
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
        first_gate_pass_latency: Optional[float] = None
        tracked_freq_first_seen: Optional[float] = None
        commit_freq_first_seen: Optional[float] = None
        last_pred_freq: Optional[float] = None
        raw_correct_seen = False
        gate_pass_correct_seen = False
        idle_commit_seen = False
        max_p_correct = 0.0
        max_decision_evidence: Optional[float] = None

        for row in trial_rows:
            pred_freq_raw = row.get("pred_freq")
            pred_freq_value = None if pred_freq_raw is None else _safe_float(pred_freq_raw, float("nan"))
            if pred_freq_value is not None and np.isfinite(pred_freq_value):
                last_pred_freq = float(pred_freq_value)
                if expected_freq is not None and abs(float(pred_freq_value) - float(expected_freq)) <= 1e-8:
                    raw_correct_seen = True
            gate_row = gate.update(dict(row))
            gate_row = _tdca._decision_evidence_row(row=gate_row, profile=profile)
            hist = history.update(
                pred_freq=pred_freq_value,
                margin=_safe_float(gate_row.get("margin", 0.0), 0.0),
                ratio=_safe_float(gate_row.get("ratio", 1.0), 1.0),
            )
            timestamp_s = float(session_stream_index) * float(profile.step_sec)
            decision = engine.step(
                pred_freq_value,
                _safe_float(gate_row.get("decision_evidence_centered", 0.0), 0.0),
                float(hist["consistency"]),
                gate_open_freq=gate_row.get("gate_open_freq"),
                timestamp_s=timestamp_s,
            )
            session_stream_index += 1
            commit = bool(decision.get("commit", False))
            committed_freq = decision.get("commit_freq")
            tracked_freq = decision.get("tracked_freq")
            selected_freq = decision.get("selected_freq")
            window_index = _safe_int(row.get("window_index", 0), 0)
            latency_value = float(profile.win_sec + window_index * profile.step_sec)
            gate_open_freq = gate_row.get("gate_open_freq")
            gate_open_freq_value = None if gate_open_freq is None else _safe_float(gate_open_freq, float("nan"))
            p_correct = _tdca._row_correctness_probability(gate_row)
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
                if previous_trial_expected is not None and first_release_latency is None and selected_active_prev and selected_freq is None:
                    first_release_latency = float(latency_value)
            else:
                if (
                    commit
                    and committed_freq is not None
                    and abs(_safe_float(committed_freq, float("nan")) - float(expected_freq)) <= 1e-8
                    and first_correct_latency is None
                ):
                    first_correct_latency = float(latency_value)
            selected_active_prev = selected_freq is not None

        if expected_freq is None:
            idle_duration_sec += float(max(trial_duration, 0.0))
            release_trial = previous_trial_expected is not None
            y_binary_true.append("idle")
            y_binary_pred.append("control" if first_any_latency is not None else "idle")
            times_binary.append(float(first_any_latency if first_any_latency is not None else penalty_latency))
            if release_trial:
                release_trials += 1
                if first_release_latency is not None:
                    release_detected_trials += 1
                    release_latencies.append(float(first_release_latency))
                else:
                    release_latencies.append(float(penalty_latency))
            previous_trial_expected = None
            trial_events.append(
                {
                    "label": label_text,
                    "trial_id": int(_safe_int(trial_rows[0].get("trial_id", -1), -1)),
                    "expected_freq": None,
                    "session_index": int(session_index),
                    "release_trial": bool(release_trial),
                    "first_any_latency_s": first_any_latency,
                    "first_release_latency_s": first_release_latency,
                    "tracked_freq_first_seen_s": tracked_freq_first_seen,
                    "commit_freq_first_seen_s": commit_freq_first_seen,
                    "first_gate_pass_latency_s": first_gate_pass_latency,
                    "trial_duration_s": float(trial_duration),
                    "commit_seen": bool(first_any_latency is not None),
                    "raw_correct_seen": False,
                    "gate_pass_correct_seen": False,
                    "max_p_correct": float(max_p_correct),
                    "max_decision_evidence": float(max_decision_evidence if max_decision_evidence is not None else 0.0),
                    "switch_trial": False,
                }
            )
            continue

        control_trials += 1
        if first_correct_latency is not None:
            control_detected_trials += 1
            detection_latencies.append(float(first_correct_latency))
            if float(first_correct_latency) <= 3.0:
                control_detected_trials_at_3s += 1
        y_binary_true.append("control")
        y_binary_pred.append("control" if first_any_latency is not None else "idle")
        times_binary.append(float(first_any_latency if first_any_latency is not None else penalty_latency))
        pred_active_freq = first_any_freq
        if pred_active_freq is None and last_pred_freq is not None:
            pred_active_freq = last_pred_freq
        if pred_active_freq is None:
            pred_active_freq = float(expected_freq)
        y_active_true.append(_tdca._freq_label(float(expected_freq)))
        y_active_pred.append(_tdca._freq_label(_tdca._nearest_freq(float(pred_active_freq), freqs)))
        times_active.append(float(first_any_latency if first_any_latency is not None else penalty_latency))
        previous_trial_expected = float(expected_freq)
        trial_events.append(
            {
                "label": label_text,
                "trial_id": int(_safe_int(trial_rows[0].get("trial_id", -1), -1)),
                "expected_freq": float(expected_freq),
                "session_index": int(session_index),
                "first_correct_latency_s": first_correct_latency,
                "first_any_latency_s": first_any_latency,
                "tracked_freq_first_seen_s": tracked_freq_first_seen,
                "commit_freq_first_seen_s": commit_freq_first_seen,
                "first_gate_pass_latency_s": first_gate_pass_latency,
                "trial_duration_s": float(trial_duration),
                "commit_seen": bool(first_any_latency is not None),
                "raw_correct_seen": bool(raw_correct_seen),
                "gate_pass_correct_seen": bool(gate_pass_correct_seen),
                "max_p_correct": float(max_p_correct),
                "max_decision_evidence": float(max_decision_evidence if max_decision_evidence is not None else 0.0),
                "switch_trial": False,
            }
        )

    idle_minutes = float(max(idle_duration_sec, 0.0)) / 60.0
    async_metrics = {
        "idle_fp_per_min": float(idle_selected_events / idle_minutes) if idle_minutes > 1e-12 else 0.0,
        "control_recall": float(control_detected_trials / control_trials) if control_trials else 0.0,
        "control_recall_at_3s": float(control_detected_trials_at_3s / control_trials) if control_trials else 0.0,
        "release_detect_rate": float(release_detected_trials / release_trials) if release_trials else 0.0,
        "release_latency_s": _median(release_latencies, default=float("inf")),
        "detection_latency_s": _median(detection_latencies, default=float("inf")),
        "first_detection_latency_s": _median(detection_latencies, default=float("inf")),
        "active_recall": float(control_detected_trials / control_trials) if control_trials else 0.0,
        "control_trials": float(control_trials),
        "release_trials": float(release_trials),
        "idle_fp_event_count": float(idle_selected_events),
        "idle_time_sec": float(idle_duration_sec),
        "inference_ms": float(inference_ms),
    }
    metrics_active = compute_classification_metrics(
        y_true=y_active_true,
        y_pred=y_active_pred,
        labels=[_tdca._freq_label(float(freq)) for freq in freqs],
        decision_time_samples_s=times_active,
        itr_class_count=max(int(len(freqs)), 2),
        decision_time_fallback_s=float(profile.win_sec),
    )
    metrics_binary = compute_classification_metrics(
        y_true=y_binary_true,
        y_pred=y_binary_pred,
        labels=["idle", "control"],
        decision_time_samples_s=times_binary,
        itr_class_count=2,
        decision_time_fallback_s=float(profile.win_sec),
    )
    return {
        "async_metrics": async_metrics,
        "metrics_4class": metrics_active,
        "metrics_2class": metrics_binary,
        "trial_events": trial_events,
        "decision_params": dict(decision_params),
    }


def _search_gate_profile_external(
    *,
    base_profile: ThresholdProfile,
    scored_tune_rows: Sequence[Mapping[str, Any]],
    freqs: Sequence[float],
    inference_ms: float,
    gate_calibration_summary: Mapping[str, Any],
    confidence_variant: str,
    search_preset: str,
) -> tuple[ThresholdProfile, list[dict[str, Any]], list[dict[str, Any]], bool]:
    best_profile = base_profile
    best_objective: Optional[tuple[float, float, float, float, float]] = None
    gate_board: list[dict[str, Any]] = []
    threshold_board: list[dict[str, Any]] = []
    any_valid = bool(gate_calibration_summary.get("gate_calibration_valid", False))
    for enter_p_th, exit_p_th, min_enter_windows, min_exit_windows, min_switch_windows in _external_gate_grid(
        search_preset
    ):
        per_freq_thresholds, freq_board, gate_valid = _tdca._build_frequency_specific_thresholds(
            base_profile=base_profile,
            freqs=freqs,
            gate_calibration_summary=gate_calibration_summary,
            enter_p_th=float(enter_p_th),
            exit_p_th=float(exit_p_th),
            min_enter_windows=int(min_enter_windows),
            min_exit_windows=int(min_exit_windows),
            min_switch_windows=int(min_switch_windows),
        )
        reference_payloads, reference_board = _compute_per_frequency_reference_overrides(
            scored_rows=scored_tune_rows,
            freqs=freqs,
            global_enter_p_th=float(enter_p_th),
            global_exit_p_th=float(exit_p_th),
            base_payloads=per_freq_thresholds,
        )
        per_freq_thresholds = {
            str(freq_key): {
                **dict(per_freq_thresholds.get(str(freq_key), {})),
                **dict(reference_payloads.get(str(freq_key), {})),
            }
            for freq_key in {*(str(key) for key in per_freq_thresholds.keys()), *(str(key) for key in reference_payloads.keys())}
        }
        any_valid = bool(any_valid or gate_valid)
        candidate_profile = replace(
            base_profile,
            min_enter_windows=int(min_enter_windows),
            min_exit_windows=int(min_exit_windows),
            min_switch_windows=int(min_switch_windows),
            enter_p_th=float(enter_p_th),
            exit_p_th=float(exit_p_th),
            enter_log_lr_th=float(_tdca._p_to_logit(enter_p_th, 0.65)),
            exit_log_lr_th=float(_tdca._p_to_logit(exit_p_th, 0.30)),
            frequency_specific_thresholds=per_freq_thresholds,
            confidence_variant=str(confidence_variant),
            control_state_mode=str(DEFAULT_CONTROL_STATE_MODE),
        )
        if gate_valid:
            metrics = dict(
                _evaluate_external_trial_rows(
                    scored_rows=scored_tune_rows,
                    profile=candidate_profile,
                    freqs=freqs,
                    decision_params=_tdca._default_decision_params(),
                    inference_ms=float(inference_ms),
                ).get("async_metrics", {})
            )
        else:
            metrics = {
                "idle_fp_per_min": float("inf"),
                "control_recall": 0.0,
                "control_recall_at_3s": 0.0,
                "release_detect_rate": 0.0,
                "release_latency_s": float("inf"),
                "detection_latency_s": float("inf"),
                "active_recall": 0.0,
                "inference_ms": float(inference_ms),
            }
        objective = _external_rank_metrics_key(metrics)
        gate_board.append(
            {
                "enter_p_th": float(enter_p_th),
                "exit_p_th": float(exit_p_th),
                "min_enter_windows": int(min_enter_windows),
                "min_exit_windows": int(min_exit_windows),
                "min_switch_windows": int(min_switch_windows),
                "gate_calibration_valid": bool(gate_valid),
                "min_gate_control_rows": int(gate_calibration_summary.get("min_control_trials_by_freq", 0) or 0),
                "min_gate_idle_rows": int(gate_calibration_summary.get("idle_trial_count", 0) or 0),
                "metrics": dict(metrics),
                "rank_key": [float(item) for item in objective],
            }
        )
        for row in freq_board:
            reference_row = next(
                (
                    dict(item)
                    for item in reference_board
                    if abs(_safe_float(item.get("freq"), float("nan")) - _safe_float(row.get("freq"), float("nan"))) <= 1e-8
                ),
                {},
            )
            threshold_board.append(
                {
                    "min_enter_windows": int(min_enter_windows),
                    "min_exit_windows": int(min_exit_windows),
                    "min_switch_windows": int(min_switch_windows),
                    **dict(row),
                    **reference_row,
                }
            )
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_profile = candidate_profile
    gate_board.sort(key=lambda item: tuple(float(value) for value in item.get("rank_key", [])))
    return best_profile, gate_board, threshold_board, any_valid


def _trial_covering_sample(session: ExternalReplaySession, sample: int) -> Optional[ExternalReplayTrial]:
    sample_value = int(sample)
    for trial in session.trials:
        if int(trial.stim_start_sample) <= sample_value < int(trial.stim_stop_sample):
            return trial
    return None


def _next_active_start_sample(session: ExternalReplaySession, *, after_trial_index: int) -> int:
    for trial in session.trials:
        if int(trial.trial_index) <= int(after_trial_index):
            continue
        if trial.expected_freq is not None:
            return int(trial.stim_start_sample)
    return int(session.sample_count)


def _continuous_replay_session(
    *,
    session: ExternalReplaySession,
    decoder: Any,
    gate_model: PerFrequencyLogRegGate,
    correctness_calibrator: CorrectnessCalibrator,
    profile: ThresholdProfile,
    decision_params: Mapping[str, Any],
    replay_speed: str,
    freqs: Sequence[float],
    inference_ms: float,
) -> dict[str, Any]:
    gate = _tdca.GateReplayState(profile)
    history = RollingFeatureHistory(window_size=4)
    engine = DecisionEngine(_tdca._make_decision_engine_config(profile=profile, decision_params=decision_params))
    gate.reset()
    history.reset()
    engine.reset()

    win_samples = int(getattr(decoder, "win_samples", 0) or 0)
    step_samples = int(getattr(decoder, "step_samples", 0) or 0)
    if win_samples <= 0 or step_samples <= 0:
        raise RuntimeError("decoder runtime is not configured for continuous replay")

    active_trials = [trial for trial in session.trials if trial.expected_freq is not None]
    active_event_by_index: dict[int, dict[str, Any]] = {}
    for trial in active_trials:
        active_event_by_index[int(trial.trial_index)] = {
            "label": str(trial.label_name),
            "trial_id": int(trial.trial_index),
            "expected_freq": float(trial.expected_freq),
            "session_index": int(session.session_index),
            "stim_start_sample": int(trial.stim_start_sample),
            "stim_stop_sample": int(trial.stim_stop_sample),
            "stim_start_sec": float(trial.stim_start_sample) / float(session.sampling_rate),
            "stim_stop_sec": float(trial.stim_stop_sample) / float(session.sampling_rate),
            "next_active_start_sample": int(
                _next_active_start_sample(session, after_trial_index=int(trial.trial_index))
            ),
            "first_correct_latency_s": None,
            "first_any_latency_s": None,
            "first_any_freq": None,
            "first_gate_pass_latency_s": None,
            "first_release_latency_s": None,
            "raw_correct_seen": False,
            "gate_pass_correct_seen": False,
            "commit_seen": False,
            "max_p_correct": 0.0,
            "max_decision_evidence": 0.0,
            "last_pred_freq": None,
            "selected_active_at_stop": False,
            "release_trial": True,
            "switch_trial": False,
        }

    timeline_board: list[dict[str, Any]] = []
    idle_commit_count = 0
    idle_duration_sec = 0.0
    selected_prev = False
    last_active_trial_index: Optional[int] = None
    pending_release_trial_index: Optional[int] = None
    stream_index = 0

    for stop_sample in range(win_samples, int(session.sample_count) + 1, step_samples):
        start_sample = int(stop_sample - win_samples)
        window = np.ascontiguousarray(session.data[start_sample:stop_sample, :], dtype=np.float64)
        scored = dict(decoder.analyze_window(window))
        pred_freq_value = _safe_float(scored.get("pred_freq"), float("nan"))
        pred_freq = None if not np.isfinite(pred_freq_value) else float(pred_freq_value)
        hist = history.update(
            pred_freq=pred_freq,
            margin=_safe_float(scored.get("margin", 0.0), 0.0),
            ratio=_safe_float(scored.get("ratio", 1.0), 1.0),
        )
        row = {
            **dict(scored),
            "margin_mean_k": float(hist.get("margin_mean_k", scored.get("margin", 0.0))),
            "ratio_mean_k": float(hist.get("ratio_mean_k", scored.get("ratio", 1.0))),
            "consistency": float(hist.get("consistency", 0.0)),
            "session_index": int(session.session_index),
            "session_id": str(session.session_id),
            "window_stop_sample": int(stop_sample),
            "window_start_sample": int(start_sample),
            "window_stop_sec": float(stop_sample) / float(session.sampling_rate),
            "window_start_sec": float(start_sample) / float(session.sampling_rate),
        }
        trial = _trial_covering_sample(session, int(stop_sample - 1))
        current_active_trial = trial if trial is not None and trial.expected_freq is not None else None
        current_expected_freq = None if current_active_trial is None else float(current_active_trial.expected_freq)
        row["label"] = str(trial.label_name) if trial is not None else "gap_rest"
        row["expected_freq"] = current_expected_freq
        row["trial_role"] = "control" if current_active_trial is not None else "clean_idle"
        row["trial_id"] = -1 if trial is None else int(trial.trial_index)
        row["trial_index_in_session"] = -1 if trial is None else int(trial.trial_index)
        row["block_index"] = int(session.session_index)

        gate_row = _tdca._score_rows_with_gate([row], gate=gate_model)[0]
        gate_row = _tdca._score_rows_with_correctness([gate_row], calibrator=correctness_calibrator)[0]
        gate_row = gate.update(gate_row)
        gate_row = _tdca._decision_evidence_row(row=gate_row, profile=profile)
        timestamp_s = float(stop_sample) / float(session.sampling_rate)
        decision = engine.step(
            pred_freq,
            _safe_float(gate_row.get("decision_evidence_centered", 0.0), 0.0),
            float(gate_row.get("consistency", 0.0)),
            gate_open_freq=gate_row.get("gate_open_freq"),
            timestamp_s=timestamp_s,
        )
        stream_index += 1
        commit = bool(decision.get("commit", False))
        committed_freq = decision.get("commit_freq")
        selected_freq = decision.get("selected_freq")
        tracked_freq = decision.get("tracked_freq")

        if current_active_trial is None:
            idle_duration_sec += float(step_samples) / float(session.sampling_rate)
            if commit and committed_freq is not None:
                idle_commit_count += 1
        else:
            event = active_event_by_index[int(current_active_trial.trial_index)]
            if pred_freq is not None and abs(float(pred_freq) - float(current_active_trial.expected_freq)) <= 1e-8:
                event["raw_correct_seen"] = True
            if pred_freq is not None:
                event["last_pred_freq"] = float(pred_freq)
            if (
                gate_row.get("gate_open_freq") is not None
                and abs(_safe_float(gate_row.get("gate_open_freq"), float("nan")) - float(current_active_trial.expected_freq)) <= 1e-8
            ):
                event["gate_pass_correct_seen"] = True
                if event["first_gate_pass_latency_s"] is None:
                    event["first_gate_pass_latency_s"] = float(timestamp_s - event["stim_start_sec"])
            if commit and event["first_any_latency_s"] is None:
                event["first_any_latency_s"] = float(timestamp_s - event["stim_start_sec"])
                event["first_any_freq"] = None if committed_freq is None else float(_safe_float(committed_freq, float("nan")))
            if commit and committed_freq is not None:
                event["commit_seen"] = True
                if (
                    abs(_safe_float(committed_freq, float("nan")) - float(current_active_trial.expected_freq)) <= 1e-8
                    and event["first_correct_latency_s"] is None
                ):
                    event["first_correct_latency_s"] = float(timestamp_s - event["stim_start_sec"])
            event["max_p_correct"] = float(max(_safe_float(event.get("max_p_correct"), 0.0), _tdca._row_correctness_probability(gate_row)))
            event["max_decision_evidence"] = float(
                max(
                    _safe_float(event.get("max_decision_evidence"), 0.0),
                    _safe_float(gate_row.get("decision_evidence_centered", 0.0), 0.0),
                )
            )

        current_active_trial_index = None if current_active_trial is None else int(current_active_trial.trial_index)
        if last_active_trial_index is not None and current_active_trial_index != int(last_active_trial_index):
            ended_event = active_event_by_index.get(int(last_active_trial_index))
            if ended_event is not None:
                ended_event["selected_active_at_stop"] = bool(selected_prev)
                pending_release_trial_index = int(last_active_trial_index)
        if current_active_trial_index is not None:
            last_active_trial_index = int(current_active_trial_index)
        elif current_active_trial is None and last_active_trial_index is None:
            pending_release_trial_index = None

        if pending_release_trial_index is not None:
            release_event = active_event_by_index.get(int(pending_release_trial_index))
            if release_event is not None:
                next_active_start_sample = int(release_event.get("next_active_start_sample", session.sample_count))
                stop_sec = float(release_event["stim_stop_sec"])
                if bool(release_event.get("selected_active_at_stop", False)) and not bool(selected_freq):
                    if release_event.get("first_release_latency_s") is None and float(timestamp_s) >= stop_sec:
                        release_event["first_release_latency_s"] = float(timestamp_s - stop_sec)
                if int(stop_sample) >= int(next_active_start_sample):
                    pending_release_trial_index = None

        timeline_board.append(
            {
                "session_id": str(session.session_id),
                "session_index": int(session.session_index),
                "time_sec": float(timestamp_s),
                "window_stop_sec": float(gate_row.get("window_stop_sec", 0.0)),
                "current_trial_type": str(row["label"]),
                "current_expected_freq": row["expected_freq"],
                "pred_freq": pred_freq,
                "top1_score": float(_safe_float(gate_row.get("top1_score"), 0.0)),
                "p_correct": float(_tdca._row_correctness_probability(gate_row)),
                "correctness_logit": float(_safe_float(gate_row.get("correctness_logit"), 0.0)),
                "decision_evidence_centered": float(_safe_float(gate_row.get("decision_evidence_centered"), 0.0)),
                "selected_freq": None if selected_freq is None else float(_safe_float(selected_freq, float("nan"))),
                "tracked_freq": None if tracked_freq is None else float(_safe_float(tracked_freq, float("nan"))),
                "commit_freq": None if committed_freq is None else float(_safe_float(committed_freq, float("nan"))),
                "commit": bool(commit),
                "gate_open_freq": gate_row.get("gate_open_freq"),
                "gate_event": str(gate_row.get("gate_event", "hold")),
                "gate_is_open": bool(gate_row.get("gate_is_open", False)),
            }
        )
        selected_prev = bool(selected_freq is not None)

    trial_events = [dict(active_event_by_index[key]) for key in sorted(active_event_by_index.keys())]
    control_trials = int(len(trial_events))
    control_latencies = [
        _safe_float(item.get("first_correct_latency_s"), float("nan"))
        for item in trial_events
        if item.get("first_correct_latency_s") is not None
    ]
    release_latencies = [
        _safe_float(item.get("first_release_latency_s"), float("nan"))
        for item in trial_events
        if item.get("first_release_latency_s") is not None
    ]
    replay_metrics = {
        "idle_fp_per_min": float(idle_commit_count / max(idle_duration_sec / 60.0, 1e-12)),
        "control_recall": float(
            sum(1 for item in trial_events if item.get("first_correct_latency_s") is not None) / max(control_trials, 1)
        ),
        "control_recall_at_3s": float(
            sum(
                1
                for item in trial_events
                if item.get("first_correct_latency_s") is not None
                and float(_safe_float(item.get("first_correct_latency_s"), float("inf"))) <= 3.0
            )
            / max(control_trials, 1)
        ),
        "release_detect_rate": float(
            sum(1 for item in trial_events if item.get("first_release_latency_s") is not None) / max(control_trials, 1)
        ),
        "release_latency_s": _median(release_latencies, default=float("inf")),
        "first_detection_latency_s": _median(control_latencies, default=float("inf")),
        "detection_latency_s": _median(control_latencies, default=float("inf")),
        "active_recall": float(
            sum(1 for item in trial_events if item.get("first_correct_latency_s") is not None) / max(control_trials, 1)
        ),
        "inference_ms": float(inference_ms),
        "replay_speed": str(replay_speed),
        "active_trial_count": int(control_trials),
        "idle_time_sec": float(idle_duration_sec),
    }
    y_true = [_tdca._freq_label(float(item["expected_freq"])) for item in trial_events]
    y_pred = []
    decision_times = []
    for item in trial_events:
        fallback_freq = item.get("first_any_freq")
        if fallback_freq is None:
            fallback_freq = item.get("last_pred_freq")
        if fallback_freq is None:
            fallback_freq = item["expected_freq"]
        if item.get("first_any_latency_s") is None:
            y_pred.append(_tdca._freq_label(_tdca._nearest_freq(float(fallback_freq), freqs)))
            decision_times.append(float(item["stim_stop_sec"] - item["stim_start_sec"]))
        else:
            y_pred.append(_tdca._freq_label(_tdca._nearest_freq(float(fallback_freq), freqs)))
            decision_times.append(float(_safe_float(item.get("first_any_latency_s"), item["stim_stop_sec"] - item["stim_start_sec"])))
    metrics_active = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=[_tdca._freq_label(float(freq)) for freq in freqs],
        decision_time_samples_s=decision_times,
        itr_class_count=max(int(len(freqs)), 2),
        decision_time_fallback_s=float(profile.win_sec),
    )
    return {
        "async_metrics": replay_metrics,
        "metrics_4class": metrics_active,
        "metrics_2class": {},
        "trial_events": trial_events,
        "replay_timeline_board": timeline_board,
        "replay_trial_events": trial_events,
    }


def _reference_baseline_row(rows: Sequence[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    preferred = [
        row
        for row in rows
        if str(row.get("fbcca_variant", "")) == "fbcca_fixed_all8"
        and abs(_safe_float(dict(row.get("candidate", {})).get("win_sec"), -1.0) - 3.0) <= 1e-8
        and str(row.get("confidence_variant", "")) == GLOBAL_CORRECTNESS_LOGISTIC
    ]
    if preferred:
        return dict(preferred[0])
    fallback = [
        row
        for row in rows
        if str(row.get("fbcca_variant", "")) == "fbcca_fixed_all8"
        and abs(_safe_float(dict(row.get("candidate", {})).get("win_sec"), -1.0) - 3.0) <= 1e-8
    ]
    if fallback:
        return dict(fallback[0])
    for row in rows:
        if str(row.get("fbcca_variant", "")) == "fbcca_fixed_all8":
            return dict(row)
    return None


def _chosen_model_rationale(
    *,
    status: str,
    chosen_row: Optional[Mapping[str, Any]],
    baseline_row: Optional[Mapping[str, Any]],
    sanity_row: Optional[Mapping[str, Any]],
) -> str:
    if str(status).strip().lower() != "ok":
        return "invalid_run_not_comparable"
    if chosen_row is None:
        return "fbcca_not_clearly_improved"
    chosen_key = tuple(float(item) for item in chosen_row.get("rank_key", []))
    baseline_key = None if baseline_row is None else tuple(float(item) for item in baseline_row.get("rank_key", []))
    sanity_key = None if sanity_row is None else tuple(float(item) for item in sanity_row.get("rank_key", []))
    better_than_baseline = baseline_key is None or chosen_key < baseline_key
    not_worse_than_sanity = sanity_key is None or chosen_key <= sanity_key
    return (
        "fbcca_improved_on_primary_ranking"
        if better_than_baseline and not_worse_than_sanity
        else "fbcca_not_clearly_improved"
    )


def _candidate_run_across_folds(
    *,
    dataset: ExternalReplayDataset,
    folds: Sequence[ExternalReplayFold],
    candidate: Mapping[str, Any],
    config: FBCCAExternalReplayOptConfig,
    log: Callable[[str], None],
    frontend_cache: Optional[dict[tuple[str, tuple[int, ...], float], tuple[dict[str, Any], dict[str, Any]]]] = None,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    candidate_index: int = 1,
    candidate_total: int = 1,
) -> dict[str, Any]:
    metadata = _candidate_metadata(candidate)
    candidate_key = _candidate_key(
        fbcca_variant=str(candidate.get("fbcca_variant", candidate.get("decoder_variant", candidate.get("model_name", "")))),
        win_sec=float(candidate.get("win_sec", 0.0)),
        confidence_variant=str(candidate.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
    )
    fold_total = max(int(len(folds)), 1)
    parallel_workers = _resolve_parallel_fold_workers(config, fold_count=fold_total)

    def emit_candidate_progress(
        phase: str,
        *,
        completed: float = 0.0,
        fold: Optional[ExternalReplayFold] = None,
        decision_index: Optional[int] = None,
        decision_total: Optional[int] = None,
    ) -> None:
        if progress_callback is None:
            return
        if phase == "build_context":
            fraction = 0.60 * (float(completed) / float(fold_total))
            detail = (
                f"candidate {candidate_index}/{candidate_total} | "
                f"{candidate.get('decoder_variant', candidate.get('fbcca_variant', candidate.get('model_name', '')))} | "
                f"conf={candidate.get('confidence_variant', GLOBAL_CORRECTNESS_LOGISTIC)} | "
                f"win={float(candidate.get('win_sec', 0.0)):g}s | "
                f"build folds {int(max(completed, 0.0))}/{fold_total}"
                if fold is None
                else f"build fold {int(fold.fold_index) + 1}/{fold_total}"
            )
        elif phase == "decision_search":
            fraction = 0.60 + 0.20 * float(max(min(completed, 1.0), 0.0))
            detail = (
                f"candidate {candidate_index}/{candidate_total} | "
                f"{candidate.get('decoder_variant', candidate.get('fbcca_variant', candidate.get('model_name', '')))} | "
                f"conf={candidate.get('confidence_variant', GLOBAL_CORRECTNESS_LOGISTIC)} | "
                f"win={float(candidate.get('win_sec', 0.0)):g}s | "
                f"decision search {int(decision_index or 0)}/{int(decision_total or 0)}"
            )
        else:
            fraction = 0.80 + 0.20 * (float(completed) / float(fold_total))
            detail = (
                f"candidate {candidate_index}/{candidate_total} | "
                f"{candidate.get('decoder_variant', candidate.get('fbcca_variant', candidate.get('model_name', '')))} | "
                f"conf={candidate.get('confidence_variant', GLOBAL_CORRECTNESS_LOGISTIC)} | "
                f"win={float(candidate.get('win_sec', 0.0)):g}s | "
                f"replay folds {int(max(completed, 0.0))}/{fold_total}"
                if fold is None
                else f"replay fold {int(fold.fold_index) + 1}/{fold_total}"
            )
        progress_callback(
            {
                "stage": "candidate_search",
                "run_index": float(candidate_index - 1) + float(max(min(fraction, 0.999), 0.0)),
                "run_total": float(candidate_total),
                "detail": detail,
                "fbcca_variant": str(candidate.get("decoder_variant", candidate.get("fbcca_variant", ""))),
                "confidence_variant": str(candidate.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
                "current_phase": str(phase),
                "candidate_index": int(candidate_index),
                "candidate_total": int(candidate_total),
                "fold_index": None if fold is None else int(fold.fold_index) + 1,
                "fold_total": int(fold_total),
                "parallel_fold_workers": int(parallel_workers),
            }
        )

    contexts: list[dict[str, Any]] = []
    gate_search_board: list[dict[str, Any]] = []
    gate_threshold_board: list[dict[str, Any]] = []
    default_tune_bundles: list[dict[str, Any]] = []
    scored_holdout_rows_all: list[dict[str, Any]] = []

    emit_candidate_progress("build_context", completed=0.0)
    built_contexts: dict[int, dict[str, Any]] = {}
    if parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers, thread_name_prefix="fbcca-ext-fold") as executor:
            future_to_fold = {
                executor.submit(
                    _build_candidate_fold_context,
                    dataset=dataset,
                    fold=fold,
                    model_name=str(candidate.get("model_name", DEFAULT_FBCCA_EXTERNAL_MODEL)),
                    model_params=dict(candidate.get("model_params", {})),
                    confidence_variant=str(candidate.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
                    win_sec=float(candidate.get("win_sec", 0.0)),
                    config=config,
                    log=log,
                    frontend_cache=frontend_cache,
                ): fold
                for fold in folds
            }
            completed_contexts = 0
            pending_futures = set(future_to_fold.keys())
            heartbeat_sec = max(float(config.progress_heartbeat_sec), 0.5)
            while pending_futures:
                done_futures, pending_futures = wait(
                    pending_futures,
                    timeout=heartbeat_sec,
                    return_when=FIRST_COMPLETED,
                )
                if not done_futures:
                    emit_candidate_progress("build_context", completed=float(completed_contexts))
                    continue
                for future in done_futures:
                    fold = future_to_fold[future]
                    built_contexts[int(fold.fold_index)] = future.result()
                    completed_contexts += 1
                    emit_candidate_progress("build_context", completed=float(completed_contexts), fold=fold)
    else:
        for fold in folds:
            built_contexts[int(fold.fold_index)] = _build_candidate_fold_context(
                dataset=dataset,
                fold=fold,
                model_name=str(candidate.get("model_name", DEFAULT_FBCCA_EXTERNAL_MODEL)),
                model_params=dict(candidate.get("model_params", {})),
                confidence_variant=str(candidate.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
                win_sec=float(candidate.get("win_sec", 0.0)),
                config=config,
                log=log,
                frontend_cache=frontend_cache,
            )
            emit_candidate_progress(
                "build_context",
                completed=float(len(built_contexts)),
                fold=fold,
            )

    for fold in folds:
        context = built_contexts[int(fold.fold_index)]
        gate_profile, gate_rows, threshold_rows, gate_valid = _search_gate_profile_external(
            base_profile=context["base_profile"],
            scored_tune_rows=context["scored_tune_rows"],
            freqs=tuple(float(freq) for freq in dataset.freqs),
            inference_ms=float(context["inference_ms"]),
            gate_calibration_summary=context["gate_calibration_summary"],
            confidence_variant=str(candidate.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
            search_preset=str(config.search_preset),
        )
        context["gate_profile"] = gate_profile
        context["gate_search_board"] = gate_rows
        context["gate_threshold_board"] = threshold_rows
        context["gate_calibration_valid"] = bool(gate_valid)
        contexts.append(context)
        default_tune_bundles.append(
            _evaluate_external_trial_rows(
                scored_rows=context["scored_tune_rows"],
                profile=gate_profile,
                freqs=tuple(float(freq) for freq in dataset.freqs),
                decision_params=_tdca._default_decision_params(),
                inference_ms=float(context["inference_ms"]),
            )
        )
        scored_holdout_rows_all.extend([dict(row) for row in context.get("scored_holdout_trial_rows", []) or []])
        gate_search_board.extend(
            [
                {
                    "candidate_key": str(candidate_key),
                    "outer_fold": int(fold.fold_index),
                    "decoder_variant": str(candidate.get("decoder_variant", candidate.get("fbcca_variant", ""))),
                    "decoder_family_variant": str(metadata.get("decoder_family_variant", "")),
                    "fbcca_variant": str(candidate.get("fbcca_variant", "")),
                    "template_usage": str(metadata.get("template_usage", "none")),
                    "confidence_variant": str(candidate.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
                    **dict(row),
                }
                for row in gate_rows
            ]
        )
        gate_threshold_board.extend(
            [
                {
                    "candidate_key": str(candidate_key),
                    "outer_fold": int(fold.fold_index),
                    "decoder_variant": str(candidate.get("decoder_variant", candidate.get("fbcca_variant", ""))),
                    "decoder_family_variant": str(metadata.get("decoder_family_variant", "")),
                    "fbcca_variant": str(candidate.get("fbcca_variant", "")),
                    "template_usage": str(metadata.get("template_usage", "none")),
                    "confidence_variant": str(candidate.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
                    **dict(row),
                }
                for row in threshold_rows
            ]
        )
    aggregated_tune = _tdca._aggregate_metric_bundle(default_tune_bundles)
    async_metrics_tune = dict(aggregated_tune.get("async_metrics", {}))
    async_metrics_tune["inference_ms"] = _median(
        [context.get("inference_ms") for context in contexts],
        default=float("inf"),
    )
    gate_summary = dict(contexts[0].get("gate_calibration_summary", {})) if contexts else {}
    gate_valid_any = any(bool(context.get("gate_calibration_valid", False)) for context in contexts)
    diagnostic_only = bool(_is_diagnostic_only_variant(candidate.get("decoder_variant", candidate.get("fbcca_variant", ""))))
    gate_row = {
        "candidate_key": str(candidate_key),
        "candidate": dict(candidate),
        "decoder_variant": str(candidate.get("decoder_variant", candidate.get("fbcca_variant", ""))),
        "decoder_family_variant": str(metadata.get("decoder_family_variant", "")),
        "template_usage": str(metadata.get("template_usage", "none")),
        "fbcca_variant": str(candidate.get("fbcca_variant", "")),
        "algorithm_alignment": str(metadata.get("algorithm_alignment", "")),
        "channel_weight_mode": metadata.get("channel_weight_mode"),
        "subband_weight_mode": metadata.get("subband_weight_mode"),
        "spatial_filter_mode": metadata.get("spatial_filter_mode"),
        "frontend_optimization_summary": dict(contexts[0].get("frontend_optimization_summary", {})) if contexts else {},
        "confidence_variant": str(candidate.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
        "decision_evidence_variant": _tdca.DEFAULT_DECISION_EVIDENCE_VARIANT,
        "metrics_median": async_metrics_tune,
        "rank_key": [float(item) for item in _external_rank_metrics_key(async_metrics_tune)],
        "gate_calibration_valid": bool(gate_valid_any),
        "gate_calibration_summary": gate_summary,
        "tune_summary": dict(contexts[0].get("tune_summary", {})) if contexts else {},
        "training_window_policy": str(contexts[0].get("training_window_policy", "last_window_only")) if contexts else "last_window_only",
        "analysis_latency_sec": float(contexts[0].get("analysis_latency_sec", 0.0) or 0.0) if contexts else 0.0,
        "effective_raw_window_sec": float(contexts[0].get("effective_raw_window_sec", candidate.get("win_sec", 0.0)) or 0.0) if contexts else float(candidate.get("win_sec", 0.0)),
        "confidence_training_scheme": str(contexts[0].get("confidence_training_scheme", _tdca.DEFAULT_CONFIDENCE_TRAINING_SCHEME)) if contexts else _tdca.DEFAULT_CONFIDENCE_TRAINING_SCHEME,
        "oof_group_key": str(contexts[0].get("oof_group_key", "")) if contexts else "",
        "oof_group_count": int(contexts[0].get("oof_group_count", 0) or 0) if contexts else 0,
        "sample_weight_mode": str(contexts[0].get("sample_weight_mode", "")) if contexts else "",
        "positive_trials": int(contexts[0].get("positive_trials", 0) or 0) if contexts else 0,
        "negative_trials": int(contexts[0].get("negative_trials", 0) or 0) if contexts else 0,
        "diagnostic_only": bool(diagnostic_only),
    }
    decision_rows: list[dict[str, Any]] = []
    best_decision_params = dict(_tdca._default_decision_params())
    if gate_valid_any:
        param_grid = _external_decision_param_grid(str(config.search_preset))
        emit_candidate_progress("decision_search", completed=0.0, decision_index=0, decision_total=len(param_grid))
        progress_stride = max(1, len(param_grid) // 8)
        for param_index, params in enumerate(param_grid, start=1):
            bundles = [
                _evaluate_external_trial_rows(
                    scored_rows=context["scored_tune_rows"],
                    profile=context["gate_profile"],
                    freqs=tuple(float(freq) for freq in dataset.freqs),
                    decision_params=params,
                    inference_ms=float(context["inference_ms"]),
                )
                for context in contexts
            ]
            aggregated = _tdca._aggregate_metric_bundle(bundles)
            async_metrics = dict(aggregated.get("async_metrics", {}))
            async_metrics["inference_ms"] = _median(
                [context.get("inference_ms") for context in contexts],
                default=float("inf"),
            )
            decision_rows.append(
                {
                    "candidate_key": str(candidate_key),
                    "candidate": dict(candidate),
                    "decoder_variant": str(candidate.get("decoder_variant", candidate.get("fbcca_variant", ""))),
                    "decoder_family_variant": str(metadata.get("decoder_family_variant", "")),
                    "fbcca_variant": str(candidate.get("fbcca_variant", "")),
                    "template_usage": str(metadata.get("template_usage", "none")),
                    "confidence_variant": str(candidate.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
                    "selection_target": DEFAULT_DECISION_SEARCH_TARGET,
                    "decision_params": dict(params),
                    "metrics_median": async_metrics,
                    "rank_key": [float(item) for item in _external_rank_metrics_key(async_metrics)],
                    "diagnostic_only": bool(diagnostic_only),
                }
            )
            if param_index == 1 or param_index == len(param_grid) or param_index % progress_stride == 0:
                emit_candidate_progress(
                    "decision_search",
                    completed=float(param_index) / float(max(len(param_grid), 1)),
                    decision_index=param_index,
                    decision_total=len(param_grid),
                )
        decision_rows.sort(key=_candidate_sort_key)
        if decision_rows:
            best_decision_params = dict(decision_rows[0].get("decision_params", {}))
    holdout_bundles: list[dict[str, Any]] = []
    chronological_bundle: Optional[dict[str, Any]] = None
    emit_candidate_progress("holdout_replay", completed=0.0)
    replay_by_fold: dict[int, dict[str, Any]] = {}
    if parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers, thread_name_prefix="fbcca-ext-replay") as executor:
            future_to_context = {
                executor.submit(
                    _continuous_replay_session,
                    session=dataset.sessions[int(context["fold"].holdout_session_index)],
                    decoder=context["decoder"],
                    gate_model=context["gate_model"],
                    correctness_calibrator=context["correctness_calibrator"],
                    profile=context["gate_profile"],
                    decision_params=best_decision_params,
                    replay_speed=str(config.replay_speed),
                    freqs=tuple(float(freq) for freq in dataset.freqs),
                    inference_ms=float(context["inference_ms"]),
                ): context
                for context in contexts
            }
            completed_replays = 0
            pending_futures = set(future_to_context.keys())
            heartbeat_sec = max(float(config.progress_heartbeat_sec), 0.5)
            while pending_futures:
                done_futures, pending_futures = wait(
                    pending_futures,
                    timeout=heartbeat_sec,
                    return_when=FIRST_COMPLETED,
                )
                if not done_futures:
                    emit_candidate_progress("holdout_replay", completed=float(completed_replays))
                    continue
                for future in done_futures:
                    context = future_to_context[future]
                    fold = context["fold"]
                    replay_bundle = future.result()
                    replay_bundle["outer_fold"] = int(fold.fold_index)
                    replay_bundle["fingerprint"] = str(fold.fingerprint)
                    replay_by_fold[int(fold.fold_index)] = replay_bundle
                    completed_replays += 1
                    emit_candidate_progress("holdout_replay", completed=float(completed_replays), fold=fold)
    else:
        for context in contexts:
            fold = context["fold"]
            replay_bundle = _continuous_replay_session(
                session=dataset.sessions[int(fold.holdout_session_index)],
                decoder=context["decoder"],
                gate_model=context["gate_model"],
                correctness_calibrator=context["correctness_calibrator"],
                profile=context["gate_profile"],
                decision_params=best_decision_params,
                replay_speed=str(config.replay_speed),
                freqs=tuple(float(freq) for freq in dataset.freqs),
                inference_ms=float(context["inference_ms"]),
            )
            replay_bundle["outer_fold"] = int(fold.fold_index)
            replay_bundle["fingerprint"] = str(fold.fingerprint)
            replay_by_fold[int(fold.fold_index)] = replay_bundle
            emit_candidate_progress(
                "holdout_replay",
                completed=float(len(replay_by_fold)),
                fold=fold,
            )

    for fold in folds:
        replay_bundle = replay_by_fold[int(fold.fold_index)]
        holdout_bundles.append(replay_bundle)
        if int(fold.holdout_session_index) == int(len(dataset.sessions) - 1):
            chronological_bundle = dict(replay_bundle)
    aggregated_holdout = _tdca._aggregate_metric_bundle(holdout_bundles)
    holdout_metrics = dict(aggregated_holdout.get("async_metrics", {}))
    holdout_metrics["inference_ms"] = _median(
        [context.get("inference_ms") for context in contexts],
        default=float("inf"),
    )
    holdout_metrics["replay_speed"] = str(config.replay_speed)
    holdout_trial_events = [
        dict(item)
        for bundle in holdout_bundles
        for item in list(bundle.get("replay_trial_events", []) or [])
    ]
    replay_frequency_breakdown = _build_replay_frequency_breakdown(holdout_trial_events)
    decision_bottleneck_summary = _build_external_decision_bottleneck_summary(
        candidate_row={
            "candidate_key": str(candidate_key),
            "fbcca_variant": str(candidate.get("fbcca_variant", "")),
            "decoder_variant": str(candidate.get("decoder_variant", candidate.get("fbcca_variant", ""))),
            "confidence_variant": str(candidate.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
        },
        holdout_bundles=holdout_bundles,
    )
    validity_summary = _selection_validity_summary(
        replay_frequency_breakdown=replay_frequency_breakdown,
        decision_bottleneck_summary=decision_bottleneck_summary,
    )
    per_frequency_enter_reference, per_frequency_exit_reference = _aggregate_reference_maps(
        contexts,
        freqs=tuple(float(freq) for freq in dataset.freqs),
    )
    reference_diagnostics_board = _aggregate_reference_diagnostics(
        contexts,
        freqs=tuple(float(freq) for freq in dataset.freqs),
    )
    min_gate_pass_rate_by_freq = min(
        (_safe_float(row.get("gate_pass_rate"), 0.0) for row in replay_frequency_breakdown),
        default=0.0,
    )
    final_row = {
        "candidate_key": str(candidate_key),
        "candidate": dict(candidate),
        "model_name": str(candidate.get("model_name", DEFAULT_FBCCA_EXTERNAL_MODEL)),
        "decoder_variant": str(candidate.get("decoder_variant", candidate.get("fbcca_variant", ""))),
        "decoder_family_variant": str(metadata.get("decoder_family_variant", "")),
        "template_usage": str(metadata.get("template_usage", "none")),
        "fbcca_variant": str(candidate.get("fbcca_variant", "")),
        "algorithm_alignment": str(metadata.get("algorithm_alignment", "")),
        "channel_weight_mode": metadata.get("channel_weight_mode"),
        "subband_weight_mode": metadata.get("subband_weight_mode"),
        "spatial_filter_mode": metadata.get("spatial_filter_mode"),
        "frontend_optimization_summary": dict(contexts[0].get("frontend_optimization_summary", {})) if contexts else {},
        "confidence_variant": str(candidate.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
        "decision_evidence_variant": _tdca.DEFAULT_DECISION_EVIDENCE_VARIANT,
        "decision_params": dict(best_decision_params),
        "metrics_median": holdout_metrics,
        "rank_key": [float(item) for item in _external_rank_metrics_key(holdout_metrics)],
        "training_window_policy": str(contexts[0].get("training_window_policy", "last_window_only")) if contexts else "last_window_only",
        "analysis_latency_sec": float(contexts[0].get("analysis_latency_sec", 0.0) or 0.0) if contexts else 0.0,
        "effective_raw_window_sec": float(contexts[0].get("effective_raw_window_sec", candidate.get("win_sec", 0.0)) or 0.0) if contexts else float(candidate.get("win_sec", 0.0)),
        "gate_calibration_valid": bool(gate_valid_any),
        "gate_calibration_summary": gate_summary,
        "tune_summary": dict(contexts[0].get("tune_summary", {})) if contexts else {},
        "per_frequency_enter_reference": dict(per_frequency_enter_reference),
        "per_frequency_exit_reference": dict(per_frequency_exit_reference),
        "replay_frequency_breakdown": replay_frequency_breakdown,
        "reference_diagnostics_board": reference_diagnostics_board,
        "decision_bottleneck_summary": dict(decision_bottleneck_summary),
        "min_gate_pass_rate_by_freq": float(min_gate_pass_rate_by_freq),
        "frequency_balance_valid": bool(validity_summary.get("frequency_balance_valid", False)),
        "confidence_dominance_valid": bool(validity_summary.get("confidence_dominance_valid", False)),
        "confidence_reject_failure_ratio": float(validity_summary.get("confidence_reject_failure_ratio", 0.0)),
        "selection_eligible": bool(gate_valid_any and validity_summary.get("selection_eligible", False)),
        "selection_invalid_reasons": [
            *([] if gate_valid_any else ["gate_calibration_invalid"]),
            *list(validity_summary.get("invalid_reasons", []) or []),
        ],
        "diagnostic_only": bool(diagnostic_only),
        "_holdout_bundles": holdout_bundles,
    }
    final_row["diagnostic_rank_key"] = [
        float(item) if isinstance(item, (int, float)) else item
        for item in _diagnostic_rank_key(final_row)
    ]
    return {
        "candidate_key": str(candidate_key),
        "contexts": contexts,
        "gate_row": gate_row,
        "gate_search_board": gate_search_board,
        "gate_threshold_board": gate_threshold_board,
        "decision_rows": decision_rows,
        "final_row": final_row,
        "holdout_bundles": holdout_bundles,
        "chronological_bundle": chronological_bundle,
        "scored_holdout_trial_rows": scored_holdout_rows_all,
    }


def _render_markdown(report_payload: Mapping[str, Any]) -> str:
    chosen_metrics = dict(report_payload.get("chosen_async_metrics", {}) or {})
    diagnostic_metrics = dict(report_payload.get("diagnostic_best_async_metrics", {}) or {})
    diagnostic_row = dict(report_payload.get("diagnostic_best_row", {}) or {})
    replay_metrics = dict(report_payload.get("replay_metrics", {}) or {})
    replay_frequency_breakdown = [dict(item) for item in report_payload.get("replay_frequency_breakdown", []) or []]
    tune_frequency_breakdown = [dict(item) for item in report_payload.get("tune_frequency_breakdown", []) or []]
    reference_diagnostics_board = [dict(item) for item in report_payload.get("reference_diagnostics_board", []) or []]
    lines = [
        "# FBCCA External Replay Opt",
        "",
        f"- Generated at: `{report_payload.get('generated_at', '')}`",
        f"- Task: `{report_payload.get('task', '')}`",
        f"- Subject: `{report_payload.get('subject', '')}`",
        f"- Outer eval mode: `{report_payload.get('outer_eval_mode', '')}`",
        f"- Deployment view: `{report_payload.get('deployment_view', '')}`",
        f"- Replay speed: `{report_payload.get('replay_speed', '')}`",
        f"- Search preset: `{report_payload.get('search_preset', '')}`",
        f"- Chosen model: `{report_payload.get('chosen_model', '')}`",
        f"- Decoder variant: `{report_payload.get('decoder_variant', '')}`",
        f"- Decoder family: `{report_payload.get('decoder_family_variant', '')}`",
        f"- FBCCA variant: `{report_payload.get('fbcca_variant', '')}`",
        f"- Template usage: `{report_payload.get('template_usage', '')}`",
        f"- Confidence variant: `{report_payload.get('confidence_variant', '')}`",
        f"- Status: `{report_payload.get('status', '')}`",
        f"- Status reasons: `{', '.join(report_payload.get('status_reasons', []) or []) or 'none'}`",
        f"- Chosen rationale: `{report_payload.get('chosen_model_rationale', '')}`",
        f"- frequency_balance_valid: `{bool(report_payload.get('frequency_balance_valid', False))}`",
        f"- confidence_dominance_valid: `{bool(report_payload.get('confidence_dominance_valid', False))}`",
        f"- Simulation only profile: `{bool(report_payload.get('simulation_only_profile', True))}`",
        f"- strict_eligible_candidate_count: `{report_payload.get('strict_eligible_candidate_count', 0)}`",
        f"- gate_valid_candidate_count: `{report_payload.get('gate_valid_candidate_count', 0)}`",
        "",
        "## Holdout Selection",
        "",
        f"- idle_fp_per_min: `{chosen_metrics.get('idle_fp_per_min', '')}`",
        f"- control_recall: `{chosen_metrics.get('control_recall', '')}`",
        f"- control_recall_at_3s: `{chosen_metrics.get('control_recall_at_3s', '')}`",
        f"- release_latency_s: `{chosen_metrics.get('release_latency_s', '')}`",
        f"- inference_ms: `{chosen_metrics.get('inference_ms', '')}`",
        "",
        "## Diagnostic Best Row",
        "",
        f"- candidate_key: `{diagnostic_row.get('candidate_key', '')}`",
        f"- decoder_variant: `{diagnostic_row.get('decoder_variant', '')}`",
        f"- confidence_variant: `{diagnostic_row.get('confidence_variant', '')}`",
        f"- diagnostic_only: `{bool(diagnostic_row.get('diagnostic_only', False))}`",
        f"- invalid_reasons: `{', '.join(report_payload.get('diagnostic_invalid_reasons', []) or []) or 'none'}`",
        f"- idle_fp_per_min: `{diagnostic_metrics.get('idle_fp_per_min', '')}`",
        f"- control_recall: `{diagnostic_metrics.get('control_recall', '')}`",
        f"- control_recall_at_3s: `{diagnostic_metrics.get('control_recall_at_3s', '')}`",
        f"- release_latency_s: `{diagnostic_metrics.get('release_latency_s', '')}`",
        "",
        "## Deployment View Replay",
        "",
        f"- idle_fp_per_min: `{replay_metrics.get('idle_fp_per_min', '')}`",
        f"- control_recall: `{replay_metrics.get('control_recall', '')}`",
        f"- control_recall_at_3s: `{replay_metrics.get('control_recall_at_3s', '')}`",
        f"- release_latency_s: `{replay_metrics.get('release_latency_s', '')}`",
        f"- first_detection_latency_s: `{replay_metrics.get('first_detection_latency_s', '')}`",
        "",
        "## Replay Frequency Breakdown",
        "",
    ]
    if replay_frequency_breakdown:
        for row in replay_frequency_breakdown:
            lines.append(
                "- "
                f"freq=`{row.get('freq_label', row.get('freq', ''))}` "
                f"trials=`{row.get('trial_count', '')}` "
                f"raw=`{row.get('raw_correct_rate', '')}` "
                f"gate=`{row.get('gate_pass_rate', '')}` "
                f"commit=`{row.get('commit_rate', '')}` "
                f"release=`{row.get('release_seen_rate', '')}`"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Tune Frequency Breakdown",
            "",
        ]
    )
    if tune_frequency_breakdown:
        for row in tune_frequency_breakdown:
            lines.append(
                "- "
                f"freq=`{row.get('freq', '')}`Hz "
                f"rows=`{row.get('row_count', '')}` "
                f"positive_windows=`{row.get('positive_windows', '')}` "
                f"negative_windows=`{row.get('negative_windows', '')}` "
                f"median_p=`{row.get('median_p_correct', '')}` "
                f"median_logit=`{row.get('median_correctness_logit', '')}`"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Reference Diagnostics",
            "",
        ]
    )
    if reference_diagnostics_board:
        for row in reference_diagnostics_board:
            lines.append(
                "- "
                f"freq=`{row.get('freq_label', row.get('freq', ''))}` "
                f"enter=`{row.get('enter_reference', '')}` "
                f"p50=`{row.get('positive_trial_max_p50', '')}` "
                f"p75=`{row.get('positive_trial_max_p75', '')}` "
                f"neg_p90=`{row.get('negative_trial_max_p90', '')}` "
                f"headroom_p50=`{row.get('reference_headroom_p50', '')}` "
                f"bound=`{bool(row.get('reference_bound_applied', False))}`"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
        "## Diagnostics",
        "",
        f"- fbcca_search_board rows: `{len(report_payload.get('fbcca_search_board', []) or [])}`",
        f"- decision_search_board rows: `{len(report_payload.get('decision_search_board', []) or [])}`",
        f"- holdout_selection_board rows: `{len(report_payload.get('holdout_selection_board', []) or [])}`",
        f"- replay_timeline rows: `{len(report_payload.get('replay_timeline_board', []) or [])}`",
        f"- confidence_diagnostics rows: `{len(report_payload.get('confidence_diagnostics_board', []) or [])}`",
        f"- error_attribution rows: `{len(report_payload.get('error_attribution_board', []) or [])}`",
        f"- replay_frequency_breakdown rows: `{len(replay_frequency_breakdown)}`",
        f"- reference_diagnostics rows: `{len(reference_diagnostics_board)}`",
        f"- sanity_compare rows: `{len(report_payload.get('sanity_compare_board', []) or [])}`",
        f"- invalid_reason_histogram: `{dict(report_payload.get('invalid_reason_histogram', {}) or {})}`",
        "",
        ]
    )
    return "\n".join(lines) + "\n"


def _bundle_replay_trial_events(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    replay_events = bundle.get("replay_trial_events", None)
    if replay_events is None:
        replay_events = bundle.get("trial_events", []) or []
    return [dict(item) for item in replay_events or []]


def _build_external_decision_bottleneck_summary(
    *,
    candidate_row: Mapping[str, Any],
    holdout_bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    control_events: list[dict[str, Any]] = []
    switch_events: list[dict[str, Any]] = []
    release_events: list[dict[str, Any]] = []
    for bundle in holdout_bundles:
        for event in _bundle_replay_trial_events(bundle):
            expected = event.get("expected_freq")
            if expected is not None:
                control_events.append(event)
                if bool(event.get("switch_trial", False)):
                    switch_events.append(event)
            if bool(event.get("release_trial", False)):
                release_events.append(event)
    failure_breakdown = {
        "decoder_miss": 0,
        "confidence_reject_miss": 0,
        "decision_miss": 0,
    }
    for event in control_events:
        if event.get("first_correct_latency_s") is not None:
            continue
        failure_breakdown[_tdca._error_type_for_control_event(event)] += 1
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
    release_latencies = [
        _safe_float(event.get("first_release_latency_s"), float("nan"))
        for event in release_events
        if event.get("first_release_latency_s") is not None
    ]
    return {
        "candidate_key": str(candidate_row.get("candidate_key", "")),
        "decoder_variant": str(candidate_row.get("decoder_variant", candidate_row.get("fbcca_variant", ""))),
        "decoder_family_variant": str(candidate_row.get("decoder_family_variant", "")),
        "fbcca_variant": str(candidate_row.get("fbcca_variant", "")),
        "template_usage": str(candidate_row.get("template_usage", "none")),
        "confidence_variant": str(candidate_row.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
        "control_trials": int(len(control_events)),
        "switch_trials": int(len(switch_events)),
        "release_trials": int(len(release_events)),
        "raw_correct_seen_count": int(sum(1 for event in control_events if bool(event.get("raw_correct_seen", False)))),
        "gate_pass_correct_seen_count": int(sum(1 for event in control_events if bool(event.get("gate_pass_correct_seen", False)))),
        "commit_seen_count": int(sum(1 for event in control_events if bool(event.get("commit_seen", False)))),
        "release_seen_count": int(sum(1 for event in release_events if event.get("first_release_latency_s") is not None)),
        "median_first_gate_pass_latency_s": None if not gate_pass_latencies else float(np.median(np.asarray(gate_pass_latencies, dtype=float))),
        "median_max_p_correct": None if not max_p_correct_values else float(np.median(np.asarray(max_p_correct_values, dtype=float))),
        "median_max_decision_evidence": None if not max_decision_values else float(np.median(np.asarray(max_decision_values, dtype=float))),
        "median_release_latency_s": None if not release_latencies else float(np.median(np.asarray(release_latencies, dtype=float))),
        "failure_breakdown": {str(key): int(value) for key, value in failure_breakdown.items()},
    }


def _build_external_error_attribution_board(
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
        for event in _bundle_replay_trial_events(bundle):
            event_dict = dict(event)
            expected = event_dict.get("expected_freq")
            if expected is not None:
                totals["control"]["total"] += 1
                first_correct = event_dict.get("first_correct_latency_s")
                if first_correct is not None:
                    totals["control"]["success"] += 1
                else:
                    totals["control"][_tdca._error_type_for_control_event(event_dict)] += 1
                if bool(event_dict.get("switch_trial", False)):
                    totals["switch"]["total"] += 1
                    if first_correct is not None and float(_safe_float(first_correct, float("inf"))) <= 2.8:
                        totals["switch"]["success"] += 1
                    else:
                        totals["switch"][_tdca._error_type_for_control_event(event_dict)] += 1
            if bool(event_dict.get("release_trial", False)):
                totals["release"]["total"] += 1
                if event_dict.get("first_release_latency_s") is not None:
                    totals["release"]["success"] += 1
                else:
                    totals["release"]["decision_miss"] += 1
    board: list[dict[str, Any]] = []
    for event_type in ("control", "switch", "release"):
        payload = dict(totals[event_type])
        board.append(
            {
                "candidate_key": str(candidate_row.get("candidate_key", "")),
                "decoder_variant": str(candidate_row.get("decoder_variant", candidate_row.get("fbcca_variant", ""))),
                "decoder_family_variant": str(candidate_row.get("decoder_family_variant", "")),
                "fbcca_variant": str(candidate_row.get("fbcca_variant", "")),
                "template_usage": str(candidate_row.get("template_usage", "none")),
                "confidence_variant": str(candidate_row.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
                "event_type": str(event_type),
                **{str(key): int(value) for key, value in payload.items()},
            }
        )
    return board


def _build_replay_frequency_breakdown(trial_events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for event in trial_events:
        event_dict = dict(event)
        expected = event_dict.get("expected_freq")
        if expected is None:
            continue
        freq = float(_safe_float(expected, float("nan")))
        if not np.isfinite(freq):
            continue
        grouped.setdefault(freq, []).append(event_dict)
    output: list[dict[str, Any]] = []
    for freq in sorted(grouped.keys()):
        items = grouped[freq]
        release_latencies = [
            _safe_float(item.get("first_release_latency_s"), float("nan"))
            for item in items
            if item.get("first_release_latency_s") is not None
        ]
        correct_latencies = [
            _safe_float(item.get("first_correct_latency_s"), float("nan"))
            for item in items
            if item.get("first_correct_latency_s") is not None
        ]
        gate_pass_latencies = [
            _safe_float(item.get("first_gate_pass_latency_s"), float("nan"))
            for item in items
            if item.get("first_gate_pass_latency_s") is not None
        ]
        max_p_values = [
            _safe_float(item.get("max_p_correct"), float("nan"))
            for item in items
            if np.isfinite(_safe_float(item.get("max_p_correct"), float("nan")))
        ]
        max_decision_values = [
            _safe_float(item.get("max_decision_evidence"), float("nan"))
            for item in items
            if np.isfinite(_safe_float(item.get("max_decision_evidence"), float("nan")))
        ]
        trial_count = max(int(len(items)), 1)
        output.append(
            {
                "freq": float(freq),
                "freq_label": _tdca._freq_label(float(freq)),
                "trial_count": int(len(items)),
                "raw_correct_rate": float(sum(1 for item in items if bool(item.get("raw_correct_seen", False))) / float(trial_count)),
                "gate_pass_rate": float(sum(1 for item in items if bool(item.get("gate_pass_correct_seen", False))) / float(trial_count)),
                "commit_rate": float(sum(1 for item in items if bool(item.get("commit_seen", False))) / float(trial_count)),
                "release_seen_rate": float(sum(1 for item in items if item.get("first_release_latency_s") is not None) / float(trial_count)),
                "median_first_correct_latency_s": None if not correct_latencies else float(np.median(np.asarray(correct_latencies, dtype=float))),
                "median_first_gate_pass_latency_s": None if not gate_pass_latencies else float(np.median(np.asarray(gate_pass_latencies, dtype=float))),
                "median_first_release_latency_s": None if not release_latencies else float(np.median(np.asarray(release_latencies, dtype=float))),
                "median_max_p_correct": None if not max_p_values else float(np.median(np.asarray(max_p_values, dtype=float))),
                "median_max_decision_evidence": None if not max_decision_values else float(np.median(np.asarray(max_decision_values, dtype=float))),
            }
        )
    return output


def run_fbcca_external_replay_opt(
    config: FBCCAExternalReplayOptConfig,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_fn: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    _validate_config(config)
    report_paths = _resolve_report_paths(config)
    report_dir = Path(report_paths["report_dir"]).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = Path(report_paths["run_log"]).expanduser().resolve()
    progress_snapshot_path = Path(report_paths["progress_snapshot"]).expanduser().resolve()
    started_at = time.perf_counter()
    progress_state: dict[str, Any] = {
        "task": DEFAULT_FBCCA_EXTERNAL_TASK,
        "stage": "prepare",
        "stage_label": _progress_label("prepare"),
        "detail": "准备外部数据回放优化",
        "report_dir": str(report_dir),
        "report_path": str(Path(report_paths["report_json"]).expanduser().resolve()),
        "profile_path": str(Path(report_paths["canonical_profile"]).expanduser().resolve()),
        "progress_percent": 0,
        "elapsed_s": 0.0,
        "eta_s": None,
    }

    def log(text: str) -> None:
        _append_run_log(run_log_path, text)
        if log_fn is not None:
            log_fn(text)

    def emit_progress(*, force: bool = False, **updates: Any) -> None:
        del force
        progress_state.update(updates)
        elapsed_s = float(max(time.perf_counter() - started_at, 0.0))
        run_index = float(progress_state.get("run_index", 0) or 0.0)
        run_total = float(progress_state.get("run_total", 0) or 0.0)
        stage = str(progress_state.get("stage", "prepare"))
        progress_state["stage_label"] = _progress_label(stage)
        progress_state["elapsed_s"] = elapsed_s
        progress_state["progress_percent"] = _progress_percent(stage, run_index=run_index, run_total=run_total)
        if 0 < int(progress_state["progress_percent"]) < 100:
            eta_s = elapsed_s * ((100.0 / float(progress_state["progress_percent"])) - 1.0)
            progress_state["eta_s"] = float(max(eta_s, 0.0))
        elif int(progress_state["progress_percent"]) >= 100:
            progress_state["eta_s"] = 0.0
        else:
            progress_state["eta_s"] = None
        atomic_write_text(progress_snapshot_path, json_dumps(json_safe(progress_state)) + "\n")
        if progress_fn is not None:
            progress_fn(dict(progress_state))

    run_config_payload = dict(asdict(config))
    run_config_payload["task"] = DEFAULT_FBCCA_EXTERNAL_TASK
    run_config_payload["output_profile_path"] = str(Path(report_paths["output_profile"]).expanduser().resolve())
    run_config_payload["report_path"] = str(Path(report_paths["report_json"]).expanduser().resolve())
    run_config_payload["report_dir"] = str(report_dir)
    atomic_write_text(Path(report_paths["run_config"]).expanduser().resolve(), json_dumps(json_safe(run_config_payload)) + "\n")

    dataset_root = Path(config.external_dataset_root).expanduser().resolve()
    subjects = discover_external_replay_subjects(dataset_root)
    if str(config.subject) not in subjects:
        raise FileNotFoundError(
            f"subject '{config.subject}' not found under {dataset_root}; available={sorted(subjects.keys())}"
        )
    emit_progress(stage="prepare", detail="加载外部数据集", run_index=1, run_total=3)
    dataset = load_external_replay_dataset(dataset_root, subject=str(config.subject))
    folds = _outer_folds(dataset, mode=str(config.outer_eval))
    search_plan = _resolve_search_plan(config)
    emit_progress(stage="prepare", detail="生成 outer folds", run_index=2, run_total=3)
    log(
        f"[{DEFAULT_FBCCA_EXTERNAL_TASK}] dataset loaded: subject={dataset.subject_id} "
        f"sessions={len(dataset.sessions)} preset={search_plan['search_preset']} outer_eval={config.outer_eval}"
    )
    frontend_cache: dict[tuple[str, tuple[int, ...], float], tuple[dict[str, Any], dict[str, Any]]] = {}
    parallel_fold_workers = _resolve_parallel_fold_workers(config, fold_count=len(folds))
    progress_state["parallel_fold_workers"] = int(parallel_fold_workers)
    log(
        f"[{DEFAULT_FBCCA_EXTERNAL_TASK}] execution plan: folds={len(folds)} "
        f"parallel_fold_workers={parallel_fold_workers} compute_backend={config.compute_backend}"
    )

    candidate_specs = [
        {
            "model_name": str(item["model_name"]),
            "decoder_variant": str(item["decoder_variant"]),
            "fbcca_variant": str(item["fbcca_variant"]),
            "decoder_family_variant": str(item["decoder_family_variant"]),
            "template_usage": str(item["template_usage"]),
            "confidence_variant": str(item["confidence_variant"]),
            "win_sec": float(item["win_sec"]),
            "model_params": _default_candidate_model_params(
                model_name=str(item["model_name"]),
                Nh=int(config.Nh),
                decoder_variant=str(item["decoder_variant"]),
                fbcca_variant=str(item["fbcca_variant"]),
            ),
        }
        for item in search_plan["candidate_grid"]
    ]
    candidate_runs: list[dict[str, Any]] = []
    total_candidates = max(len(candidate_specs), 1)
    for index, candidate in enumerate(candidate_specs, start=1):
        emit_progress(
            stage="candidate_search",
            detail=(
                f"candidate {index}/{total_candidates} | "
                f"{candidate['decoder_variant']} | conf={candidate['confidence_variant']} | win={candidate['win_sec']:g}s"
            ),
            run_index=index,
            run_total=total_candidates,
            fbcca_variant=str(candidate["decoder_variant"]),
            confidence_variant=str(candidate["confidence_variant"]),
        )
        candidate_runs.append(
            _candidate_run_across_folds(
                dataset=dataset,
                folds=folds,
                candidate=candidate,
                config=config,
                log=log,
                frontend_cache=frontend_cache,
                progress_callback=lambda updates: emit_progress(**updates),
                candidate_index=index,
                candidate_total=total_candidates,
            )
        )

    fbcca_search_board = [dict(item["gate_row"]) for item in candidate_runs]
    fbcca_search_board.sort(key=_candidate_sort_key)
    decision_search_board = [
        dict(row)
        for candidate_run in candidate_runs
        for row in list(candidate_run.get("decision_rows", []) or [])
    ]
    decision_search_board.sort(key=_candidate_sort_key)
    holdout_selection_board = [dict(item["final_row"]) for item in candidate_runs]
    holdout_selection_board.sort(key=_candidate_sort_key)
    eligible_holdout_rows = _strict_selection_rows(holdout_selection_board)

    sanity_specs = [
        {
            "model_name": "tdca",
            "decoder_variant": DEFAULT_TDCA_SANITY_VARIANT,
            "fbcca_variant": "",
            "confidence_variant": GLOBAL_CORRECTNESS_LOGISTIC,
            "win_sec": float(win_sec),
            "model_params": {
                "Nh": int(config.Nh),
                "delay_steps": 3,
                "n_components": 2,
                "decoder_variant": DEFAULT_TDCA_SANITY_VARIANT,
            },
        }
        for win_sec in DEFAULT_TDCA_SANITY_WIN_CANDIDATES
    ]
    sanity_runs: list[dict[str, Any]] = []
    sanity_total = max(len(sanity_specs), 1)

    def emit_sanity_progress(
        *,
        spec: Mapping[str, Any],
        spec_index: int,
        detail: str,
        fraction: float = 0.0,
        updates: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload: dict[str, Any] = {
            "stage": "sanity_compare",
            "detail": str(detail),
            "run_index": float(spec_index - 1) + float(max(min(fraction, 0.999), 0.0)),
            "run_total": int(sanity_total),
            "fbcca_variant": str(spec.get("decoder_variant", "")),
            "confidence_variant": str(spec.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
            "current_phase": "sanity_compare",
            "candidate_index": int(spec_index),
            "candidate_total": int(sanity_total),
            "fold_index": None,
            "fold_total": int(len(folds)),
            "parallel_fold_workers": int(parallel_fold_workers),
        }
        if updates:
            payload["current_phase"] = str(updates.get("current_phase", "sanity_compare"))
            payload["fold_index"] = updates.get("fold_index")
            payload["fold_total"] = updates.get("fold_total", int(len(folds)))
            payload["parallel_fold_workers"] = int(updates.get("parallel_fold_workers", parallel_fold_workers) or parallel_fold_workers)
        emit_progress(**payload)

    for spec_index, spec in enumerate(sanity_specs, start=1):
        emit_sanity_progress(
            spec=spec,
            spec_index=spec_index,
            detail=f"TDCA sanity {spec_index}/{sanity_total} | {spec['decoder_variant']} | win={float(spec['win_sec']):g}s",
            fraction=0.0,
        )
        sanity_runs.append(
            _candidate_run_across_folds(
                dataset=dataset,
                folds=folds,
                candidate=spec,
                config=config,
                log=log,
                frontend_cache=None,
                progress_callback=lambda updates, spec=dict(spec), spec_index=int(spec_index): emit_sanity_progress(
                    spec=spec,
                    spec_index=spec_index,
                    detail=f"TDCA sanity {spec_index}/{sanity_total} | {updates.get('detail', '')}",
                    fraction=(
                        float(updates.get("run_index", 0.0) or 0.0)
                        / float(max(float(updates.get("run_total", 1.0) or 1.0), 1.0))
                    ),
                    updates=updates,
                ),
                candidate_index=1,
                candidate_total=1,
            )
        )
    emit_progress(
        stage="sanity_compare",
        detail="TDCA sanity done",
        run_index=sanity_total,
        run_total=sanity_total,
        current_phase="sanity_compare",
        candidate_index=sanity_total,
        candidate_total=sanity_total,
        fold_index=None,
        fold_total=int(len(folds)),
        parallel_fold_workers=int(parallel_fold_workers),
    )
    sanity_compare_board = [dict(item["final_row"]) for item in sanity_runs]
    sanity_compare_board.sort(key=lambda row: tuple(float(item) for item in row.get("rank_key", [])))

    status = "ok"
    status_reasons: list[str] = []
    if not fbcca_search_board:
        status = "invalid"
        status_reasons.append("no_fbcca_candidates")
    elif not any(bool(row.get("gate_calibration_valid", False)) for row in fbcca_search_board):
        status = "invalid"
        status_reasons.append("gate_calibration_invalid_all_candidates")
    if not decision_search_board:
        status = "invalid"
        status_reasons.append("decision_search_not_effective")
    if not holdout_selection_board:
        status = "invalid"
        status_reasons.append("no_holdout_candidates")
    elif not eligible_holdout_rows:
        status = "invalid"
        status_reasons.append("no_selection_eligible_candidates")

    chosen_row = dict(eligible_holdout_rows[0]) if eligible_holdout_rows else None
    chosen_run = None
    for item in candidate_runs:
        if chosen_row is not None and str(item.get("candidate_key", "")) == str(chosen_row.get("candidate_key", "")):
            chosen_run = item
            break
    diagnostic_best_row = _diagnostic_best_row(holdout_selection_board)
    diagnostic_run = None
    for item in candidate_runs:
        if diagnostic_best_row is not None and str(item.get("candidate_key", "")) == str(diagnostic_best_row.get("candidate_key", "")):
            diagnostic_run = item
            break
    baseline_row = _reference_baseline_row(holdout_selection_board)
    best_sanity_row = dict(sanity_compare_board[0]) if sanity_compare_board else None
    contrast_reference_row = baseline_row if baseline_row is not None else best_sanity_row
    chosen_model_rationale = _chosen_model_rationale(
        status=status,
        chosen_row=chosen_row,
        baseline_row=baseline_row,
        sanity_row=best_sanity_row,
    )
    analysis_row = chosen_row if chosen_row is not None else diagnostic_best_row
    analysis_run = chosen_run if chosen_run is not None else diagnostic_run
    if chosen_run is None and analysis_run is not None:
        chosen_run = analysis_run
    strict_eligible_candidate_count = int(len(eligible_holdout_rows))
    gate_valid_candidate_count = int(sum(1 for row in holdout_selection_board if bool(row.get("gate_calibration_valid", False))))
    invalid_reason_histogram = _invalid_reason_histogram(holdout_selection_board)

    variant_summary: list[dict[str, Any]] = []
    for decoder_variant in search_plan["variant_names"]:
        variant_rows = [
            dict(row)
            for row in holdout_selection_board
            if str(row.get("decoder_variant", row.get("fbcca_variant", ""))) == str(decoder_variant)
        ]
        if not variant_rows:
            continue
        best_row = dict(variant_rows[0])
        variant_summary.append(
            {
                "decoder_variant": str(decoder_variant),
                "fbcca_variant": str(best_row.get("fbcca_variant", "")),
                "decoder_family_variant": str(best_row.get("decoder_family_variant", "")),
                "template_usage": str(best_row.get("template_usage", "none")),
                "confidence_variant": str(best_row.get("confidence_variant", "")),
                "best_candidate": dict(best_row.get("candidate", {})),
                "metrics_median": dict(best_row.get("metrics_median", {})),
                "diagnostic_only": bool(best_row.get("diagnostic_only", False)),
                "frequency_balance_valid": bool(best_row.get("frequency_balance_valid", False)),
                "confidence_dominance_valid": bool(best_row.get("confidence_dominance_valid", False)),
                "min_gate_pass_rate_by_freq": _safe_float(best_row.get("min_gate_pass_rate_by_freq"), 0.0),
                "rank_key": [float(item) for item in best_row.get("rank_key", [])],
            }
        )

    chosen_confidence_diagnostics_board: list[dict[str, Any]] = []
    chosen_decision_bottleneck_summary: dict[str, Any] = {}
    error_attribution_board: list[dict[str, Any]] = []
    contrast_error_board: list[dict[str, Any]] = []
    replay_metrics: dict[str, Any] = {}
    replay_timeline_board: list[dict[str, Any]] = []
    replay_trial_events: list[dict[str, Any]] = []
    replay_frequency_breakdown: list[dict[str, Any]] = []
    tune_frequency_breakdown: list[dict[str, Any]] = []
    reference_diagnostics_board: list[dict[str, Any]] = []
    profile_saved = False
    profile_v2_saved = False
    chosen_profile_path = ""
    profile_v2_path = ""

    if analysis_row is not None and analysis_run is not None:
        tune_rows = [
            dict(row)
            for context in analysis_run.get("contexts", [])
            for row in list(context.get("scored_tune_rows", []) or [])
        ]
        chosen_confidence_diagnostics_board = _fbcca._build_confidence_diagnostics_board(
            candidate_row=analysis_row,
            tune_rows=tune_rows,
            holdout_rows=analysis_run.get("scored_holdout_trial_rows", []),
        )
        tune_frequency_breakdown = _build_tune_frequency_breakdown(
            tune_rows,
            freqs=tuple(float(freq) for freq in dataset.freqs),
        )
        chosen_decision_bottleneck_summary = dict(analysis_row.get("decision_bottleneck_summary", {}))
        error_attribution_board = _build_external_error_attribution_board(
            candidate_row=analysis_row,
            holdout_bundles=analysis_run.get("holdout_bundles", []),
        )
        reference_diagnostics_board = [dict(item) for item in analysis_row.get("reference_diagnostics_board", []) or []]
        if contrast_reference_row is not None:
            contrast_error_board = _fbcca._build_fbcca_contrast_error_board(
                candidate_row=analysis_row,
                candidate_holdout_bundles=analysis_run.get("holdout_bundles", []),
                reference_row=contrast_reference_row,
            )
        chronological_bundle = dict(analysis_run.get("chronological_bundle", {}) or {})
        replay_metrics = dict(chronological_bundle.get("async_metrics", {}) or {})
        replay_timeline_board = [dict(item) for item in chronological_bundle.get("replay_timeline_board", []) or []]
        replay_trial_events = [dict(item) for item in chronological_bundle.get("replay_trial_events", []) or []]
        replay_frequency_breakdown = _build_replay_frequency_breakdown(replay_trial_events)

        emit_progress(stage="finalize", detail="保存 simulation-only profile", run_index=1, run_total=2)
        profile_context = None
        for context in chosen_run.get("contexts", []):
            fold = context["fold"]
            if int(fold.holdout_session_index) == int(len(dataset.sessions) - 1):
                profile_context = context
                break
        if profile_context is None and chosen_run.get("contexts"):
            profile_context = chosen_run["contexts"][0]
        if profile_context is not None and status == "ok":
            metadata = _candidate_metadata(dict(chosen_row.get("candidate", {})))
            final_profile = replace(
                profile_context["gate_profile"],
                model_name=str(chosen_row.get("model_name", DEFAULT_FBCCA_EXTERNAL_MODEL)),
                model_params={
                    **dict(profile_context.get("model_params", {})),
                    "state": profile_context.get("state_payload"),
                    "decoder_variant": str(chosen_row.get("decoder_variant", "")),
                    "fbcca_variant": str(chosen_row.get("fbcca_variant", "")),
                },
                benchmark_metrics=dict(chosen_row.get("metrics_median", {})),
                confidence_variant=str(chosen_row.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
                training_window_policy=str(profile_context.get("training_window_policy", "last_window_only")),
                recommended_for_realtime=False,
                metadata={
                    "task": DEFAULT_FBCCA_EXTERNAL_TASK,
                    "external_dataset_root": str(dataset_root),
                    "subject": str(dataset.subject_id),
                    "channel_montage_name": DEFAULT_EXTERNAL_CHANNEL_MONTAGE_NAME,
                    "simulation_only_profile": True,
                    "simulation_protocol": DEFAULT_SIMULATION_PROTOCOL,
                    "outer_eval_mode": str(config.outer_eval),
                    "deployment_view": DEFAULT_DEPLOYMENT_VIEW,
                    "candidate": dict(chosen_row.get("candidate", {})),
                    "decoder_variant": str(chosen_row.get("decoder_variant", "")),
                    "decoder_family_variant": str(chosen_row.get("decoder_family_variant", "")),
                    "fbcca_variant": str(chosen_row.get("fbcca_variant", "")),
                    "template_usage": str(chosen_row.get("template_usage", "none")),
                    "confidence_variant": str(chosen_row.get("confidence_variant", GLOBAL_CORRECTNESS_LOGISTIC)),
                    "algorithm_alignment": str(metadata.get("algorithm_alignment", "")),
                    "channel_weight_mode": metadata.get("channel_weight_mode"),
                    "subband_weight_mode": metadata.get("subband_weight_mode"),
                    "spatial_filter_mode": metadata.get("spatial_filter_mode"),
                    "per_frequency_enter_reference": dict(chosen_row.get("per_frequency_enter_reference", {})),
                    "per_frequency_exit_reference": dict(chosen_row.get("per_frequency_exit_reference", {})),
                    "frequency_balance_valid": bool(chosen_row.get("frequency_balance_valid", False)),
                    "confidence_dominance_valid": bool(chosen_row.get("confidence_dominance_valid", False)),
                    "frontend_optimization_summary": dict(profile_context.get("frontend_optimization_summary", {})),
                    "decision_params": dict(chosen_row.get("decision_params", {})),
                    "replay_metrics": dict(replay_metrics),
                    "chosen_model_rationale": str(chosen_model_rationale),
                    "run_valid_for_deployment": False,
                },
            )
            save_profile(final_profile, Path(report_paths["output_profile"]).expanduser().resolve())
            _copy_artifact_alias(
                source=Path(report_paths["output_profile"]).expanduser().resolve(),
                destination=Path(report_paths["canonical_profile"]).expanduser().resolve(),
            )
            profile_saved = True
            chosen_profile_path = str(Path(report_paths["canonical_profile"]).expanduser().resolve())
            gate_payload = dict(profile_context["gate_model"].to_payload())
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
                feature_names=tuple(getattr(profile_context["gate_model"], "feature_names", DEFAULT_GATE_FEATURES)),
                evidence=dict(chosen_row.get("decision_params", {})),
                refractory_sec=float(dict(chosen_row.get("decision_params", {})).get("refractory_sec", 0.0)),
            )
            atomic_write_text(Path(report_paths["profile_v2"]).expanduser().resolve(), json_dumps(json_safe(profile_v2.to_payload())) + "\n")
            _copy_artifact_alias(
                source=Path(report_paths["profile_v2"]).expanduser().resolve(),
                destination=Path(report_paths["canonical_profile_v2"]).expanduser().resolve(),
            )
            profile_v2_saved = True
            profile_v2_path = str(Path(report_paths["canonical_profile_v2"]).expanduser().resolve())

    emit_progress(stage="finalize", detail="写出报告与快照", run_index=2, run_total=2)
    chosen_metrics = {} if chosen_row is None else dict(chosen_row.get("metrics_median", {}))
    chosen_candidate_payload = {} if chosen_row is None else dict(chosen_row.get("candidate", {}))
    chosen_metadata = _candidate_metadata(chosen_candidate_payload) if chosen_row is not None else {}
    diagnostic_metrics = {} if diagnostic_best_row is None else dict(diagnostic_best_row.get("metrics_median", {}))
    diagnostic_best_row_public = (
        {}
        if diagnostic_best_row is None
        else {str(key): value for key, value in diagnostic_best_row.items() if not str(key).startswith("_")}
    )
    data_sufficiency_summary = {
        "subject": str(dataset.subject_id),
        "session_count": int(len(dataset.sessions)),
        "session_ids": [str(session.session_id) for session in dataset.sessions],
        "sampling_rate": int(dataset.sampling_rate),
        "trial_count": int(sum(len(session.trials) for session in dataset.sessions)),
        "active_trial_count": int(
            sum(1 for session in dataset.sessions for trial in session.trials if trial.expected_freq is not None)
        ),
        "rest_trial_count": int(
            sum(1 for session in dataset.sessions for trial in session.trials if trial.expected_freq is None)
        ),
        "current_sessions_sufficient_for_deployment": False,
    }
    report_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task": DEFAULT_FBCCA_EXTERNAL_TASK,
        "mode": "fbcca_external_replay_opt",
        "status": str(status),
        "status_reasons": list(dict.fromkeys(status_reasons)),
        "external_dataset_root": str(dataset_root),
        "subject": str(dataset.subject_id),
        "available_subjects": sorted(subjects.keys()),
        "session_count": int(len(dataset.sessions)),
        "session_ids": [str(session.session_id) for session in dataset.sessions],
        "sampling_rate": int(dataset.sampling_rate),
        "channel_names": [str(item) for item in dataset.channel_names],
        "channel_montage_name": DEFAULT_EXTERNAL_CHANNEL_MONTAGE_NAME,
        "freqs": [float(item) for item in dataset.freqs],
        "dataset_summary_rows": _dataset_summary_rows(dataset),
        "search_preset": str(search_plan["search_preset"]),
        "outer_eval_mode": str(config.outer_eval),
        "deployment_view": DEFAULT_DEPLOYMENT_VIEW,
        "replay_speed": str(config.replay_speed),
        "parallel_fold_workers": int(parallel_fold_workers),
        "simulation_protocol": DEFAULT_SIMULATION_PROTOCOL,
        "simulation_only_profile": True,
        "decision_search_target": DEFAULT_DECISION_SEARCH_TARGET,
        "final_selection_target": DEFAULT_FINAL_SELECTION_TARGET,
        "chosen_model": "" if chosen_row is None else str(chosen_row.get("model_name", DEFAULT_FBCCA_EXTERNAL_MODEL)),
        "decoder_variant": str(chosen_row.get("decoder_variant", "")) if chosen_row is not None else "",
        "decoder_family_variant": str(chosen_row.get("decoder_family_variant", "")) if chosen_row is not None else "",
        "fbcca_variant": str(chosen_row.get("fbcca_variant", "")) if chosen_row is not None else "",
        "template_usage": str(chosen_row.get("template_usage", "none")) if chosen_row is not None else "",
        "confidence_variant": str(chosen_row.get("confidence_variant", "")) if chosen_row is not None else "",
        "algorithm_alignment": str(chosen_metadata.get("algorithm_alignment", "")),
        "channel_weight_mode": chosen_metadata.get("channel_weight_mode"),
        "subband_weight_mode": chosen_metadata.get("subband_weight_mode"),
        "spatial_filter_mode": chosen_metadata.get("spatial_filter_mode"),
        "decision_evidence_variant": _tdca.DEFAULT_DECISION_EVIDENCE_VARIANT,
        "per_frequency_enter_reference": {} if chosen_row is None else dict(chosen_row.get("per_frequency_enter_reference", {})),
        "per_frequency_exit_reference": {} if chosen_row is None else dict(chosen_row.get("per_frequency_exit_reference", {})),
        "frequency_balance_valid": bool(chosen_row.get("frequency_balance_valid", False)) if chosen_row is not None else False,
        "confidence_dominance_valid": bool(chosen_row.get("confidence_dominance_valid", False)) if chosen_row is not None else False,
        "confidence_reject_failure_ratio": _safe_float(chosen_row.get("confidence_reject_failure_ratio"), 0.0) if chosen_row is not None else 0.0,
        "chosen_async_metrics": dict(chosen_metrics),
        "async_metrics": dict(chosen_metrics),
        "diagnostic_best_row": dict(diagnostic_best_row_public),
        "diagnostic_best_async_metrics": dict(diagnostic_metrics),
        "diagnostic_invalid_reasons": [] if diagnostic_best_row is None else list(diagnostic_best_row.get("selection_invalid_reasons", []) or []),
        "strict_eligible_candidate_count": int(strict_eligible_candidate_count),
        "gate_valid_candidate_count": int(gate_valid_candidate_count),
        "invalid_reason_histogram": dict(invalid_reason_histogram),
        "replay_metrics": dict(replay_metrics),
        "replay_timeline_board": replay_timeline_board,
        "replay_trial_events": replay_trial_events,
        "replay_frequency_breakdown": replay_frequency_breakdown,
        "tune_frequency_breakdown": tune_frequency_breakdown,
        "reference_diagnostics_board": reference_diagnostics_board,
        "fbcca_search_board": fbcca_search_board,
        "decision_search_board": decision_search_board,
        "holdout_selection_board": holdout_selection_board,
        "variant_summary": variant_summary,
        "confidence_diagnostics_board": chosen_confidence_diagnostics_board,
        "decision_bottleneck_summary": chosen_decision_bottleneck_summary,
        "error_attribution_board": error_attribution_board,
        "contrast_error_board": contrast_error_board,
        "sanity_compare_board": sanity_compare_board,
        "data_sufficiency_summary": data_sufficiency_summary,
        "chosen_model_rationale": str(chosen_model_rationale),
        "profile_saved": bool(profile_saved),
        "chosen_profile_path": str(chosen_profile_path),
        "profile_v2_saved": bool(profile_v2_saved),
        "profile_v2_path": str(profile_v2_path),
        "run_valid_for_deployment": False,
        "selection_snapshot_path": str(Path(report_paths["selection_snapshot"]).expanduser().resolve()),
        "report_path": str(Path(report_paths["report_json"]).expanduser().resolve()),
        "report_dir": str(report_dir),
        "run_log_path": str(run_log_path),
        "progress_snapshot_path": str(progress_snapshot_path),
    }
    atomic_write_text(Path(report_paths["selection_snapshot"]).expanduser().resolve(), json_dumps(json_safe(report_payload)) + "\n")
    atomic_write_text(Path(report_paths["report_json"]).expanduser().resolve(), json_dumps(json_safe(report_payload)) + "\n")
    atomic_write_text(Path(report_paths["report_md"]).expanduser().resolve(), _render_markdown(report_payload))
    log(f"[{DEFAULT_FBCCA_EXTERNAL_TASK}] report saved: {report_paths['report_json']}")
    emit_progress(
        stage="complete",
        detail="外部数据 FBCCA 回放优化完成",
        run_index=1,
        run_total=1,
        current_phase="complete",
        fbcca_variant="" if chosen_row is None else str(chosen_row.get("fbcca_variant", "")),
        confidence_variant="" if chosen_row is None else str(chosen_row.get("confidence_variant", "")),
        candidate_index=int(len(search_plan["candidate_grid"])),
        candidate_total=int(len(search_plan["candidate_grid"])),
        fold_index=None,
        fold_total=int(len(folds)),
        parallel_fold_workers=int(parallel_fold_workers),
        report_path=str(Path(report_paths["report_json"]).expanduser().resolve()),
        profile_path=str(chosen_profile_path or Path(report_paths["output_profile"]).expanduser().resolve()),
    )
    return report_payload
