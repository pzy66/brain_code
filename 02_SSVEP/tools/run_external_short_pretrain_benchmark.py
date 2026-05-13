from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from dataclasses import replace
from datetime import datetime
from itertools import combinations, product
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import sys
import traceback
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
from ssvep_core.external_ysuan_dataset import (
    YSUAN_DEFAULT_NS_CALIBRATION_TRIALS_PER_SUBTYPE,
    YSUAN_FREQS,
    YSUAN_REQUIRED_CHANNELS,
    YSUAN_TARGET_FREQUENCIES,
    build_ysuan_cs_segments,
    build_ysuan_ns_segments,
    build_ysuan_segments,
    load_ysuan_subject,
    resolve_ysuan_command_frequencies,
)
from ssvep_core.fast_fbcca_pretrain import FastFBCCAPretrainConfig, run_fast_fbcca_personalization
from ssvep_core.fbcca_threshold_pretrain import FBCCAThresholdPretrainConfig, run_fbcca_threshold_pretrain
from ssvep_core.score_classifier_runtime import (
    CLASSIFIER_DERIVED_FEATURE_NAMES as RUNTIME_CLASSIFIER_DERIVED_FEATURE_NAMES,
    CLASSIFIER_GATE_VARIANTS as RUNTIME_CLASSIFIER_GATE_VARIANTS,
    CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
    CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC,
    CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
    CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
    CLASSIFIER_GATE_VARIANT_LRTMW_ENTROPY,
    CLASSIFIER_GATE_VARIANT_LRTMW_MARGIN,
    CLASSIFIER_GATE_VARIANT_NS2_AWARE,
    CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE,
    CLASSIFIER_GATE_VARIANT_SUBJECT_THRESHOLD_FLOOR,
    CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO,
    CLASSIFIER_GATE_VARIANT_WEAK_SUBJECT_GUARD,
    FULL_REFERENCE_BANK_FEATURE_NAMES as RUNTIME_FULL_REFERENCE_BANK_FEATURE_NAMES,
    classifier_feature_names as runtime_classifier_feature_names,
    lrt_window_evidence_from_state,
    normalize_frequency_specific_control_state_gates,
    parse_classifier_gate_variant,
    ridge5_predict_windows_from_state,
    score_matrices_to_features as runtime_score_matrices_to_features,
    smooth_classifier_probabilities as runtime_smooth_classifier_probabilities,
)
from ssvep_core.stimulus_profiles import frame_lock_frequency_report


DEFAULT_FREQS = (9.8, 12.0, 14.8, 15.8)
DEFAULT_DATASETS = ("wang2016", "beta", "ysu_an")
DEFAULT_METHODS = ("zero_shot_default", "fast_fbcca", "threshold_pretrain", "fbcca_lda5", "fbcca_ridge5")
SUPPORTED_SHORT_PRETRAIN_METHODS = ("itcca5", "ecca5", "trca5", "trca_r5", "tdca5")
SUPPORTED_METHODS = DEFAULT_METHODS + SUPPORTED_SHORT_PRETRAIN_METHODS
METHOD_ALIASES = {
    "fbcca_lda5_fullbank": "fbcca_lda5",
    "fbcca_ridge5_fullbank": "fbcca_ridge5",
}
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
DEFAULT_CLASSIFIER_SMOOTHING_WINDOWS_CANDIDATES = (1,)
DEFAULT_CLASSIFIER_THRESHOLD_POLICY = "balanced"
CLASSIFIER_CONFIDENCE_GATE_POLICY = "confidence_threshold"
CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY = "adaptive_evidence_gate"
CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY = "lrt_multiwindow_reject_gate"
CLASSIFIER_SUBJECT_ADAPTIVE_THRESHOLD_POLICY = "subject_adaptive_threshold"
CLASSIFIER_THRESHOLD_POLICIES = (
    "balanced",
    "balanced_recall_guard",
    CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY,
    CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
    CLASSIFIER_SUBJECT_ADAPTIVE_THRESHOLD_POLICY,
)
DEFAULT_SCORE_BANK_MODE = "command_only"
SCORE_BANK_MODES = ("command_only", "full_reference_bank")
DEFAULT_FREQ_SEARCH_MODE = "none"
FREQ_SEARCH_MODES = ("none", "shared_fixed4", "personalized_upper_bound", "both")
DEFAULT_FREQ_CANDIDATE_SOURCE = "frame_locked_240"
FREQ_CANDIDATE_SOURCES = ("frame_locked_240", "beta_all40", "wang_all40", "ysu_an_all8")
DEFAULT_IDLE_EVAL_MODE = "hard_noncommand"
IDLE_EVAL_MODES = ("hard_noncommand", "clean_idle_proxy", "both")
DEFAULT_PRETRAIN_BUDGET_SEC = 120.0
DEFAULT_PERSONALIZED_CANDIDATE_COUNT = (8, 12, 40)
DEFAULT_FRAME_LOCKED_240_FREQS = (8.0, 9.6, 10.0, 12.0, 15.0)
PROJECT_POSTERIOR_8_CHANNELS = ("Oz", "O1", "O2", "PO3", "POz", "PO7", "PO8", "PO4")
WEAK_SUBJECT_AUDIT_SUBJECTS = ("S2", "S11", "S65", "S33", "S44", "S55", "S59", "S6")
DEFAULT_CLASSIFIER_THRESHOLD_MIN_CONTROL_RECALL = 0.80
DEFAULT_CLASSIFIER_IDLE_FP_BUDGET_PER_MIN = 1.0
DEFAULT_CLASSIFIER_IDLE_SELECTED_WINDOWS_BUDGET_PER_MIN = 6.0
DEFAULT_DEPLOYABLE_MIN_CONTROL_RECALL_AT_2P5S = 0.75
DEFAULT_DEPLOYABLE_MAX_DETECTION_LATENCY_SEC = 2.5
DEFAULT_RIDGE_L2_CANDIDATES = (0.03, 0.1, 0.3, 1.0, 3.0)
DEFAULT_MAX_SPLITS_PER_SUBJECT = 6
DEFAULT_STEP_SEC = 0.25
DEFAULT_DECISION_START_SEC = 0.5
DEFAULT_DECISION_DEADLINE_SEC = 2.5
DEFAULT_MIN_RELEASE_WINDOWS = 2
DEFAULT_TIMEOUT_SEC = 86400.0
DEFAULT_CASE_LIMIT = 0
DEPLOYABLE_CANDIDATE_PROFILE_SCHEMA_VERSION = "external_fbcca_classifier_candidate_profile_v1"
DEPLOYABLE_CANDIDATE_PROFILE_FILENAME = "external_fbcca_classifier_candidate_v1.json"
SERVER_SSVEP_WRITABLE_ROOT = PurePosixPath("/data1/zkx/brain/ssvep")
SERVER_SSVEP_LOG_ROOT = SERVER_SSVEP_WRITABLE_ROOT / "logs"
CLASSIFIER_DERIVED_FEATURE_NAMES = tuple(RUNTIME_CLASSIFIER_DERIVED_FEATURE_NAMES)
FULL_REFERENCE_BANK_FEATURE_NAMES = tuple(RUNTIME_FULL_REFERENCE_BANK_FEATURE_NAMES)
CLASSIFIER_GATE_VARIANTS = tuple(RUNTIME_CLASSIFIER_GATE_VARIANTS)
DEFAULT_CLASSIFIER_GATE_VARIANTS = (CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,)
DEFAULT_MARGIN_CONTROL_QUANTILES = (0.05, 0.10, 0.15)
DEFAULT_MARGIN_IDLE_QUANTILES = (0.90, 0.95, 0.975)
DEFAULT_RATIO_IDLE_QUANTILES = (0.90, 0.95, 0.975)
DEFAULT_ENTROPY_CONTROL_QUANTILES = (0.80, 0.85, 0.90)
DEFAULT_ENTROPY_IDLE_QUANTILES = (0.05, 0.10, 0.15)
DEFAULT_GLOBAL_FLOOR_QUANTILES = (0.90, 0.95, 0.975)
DEFAULT_SUBJECT_IDLE_QUANTILES = (0.90, 0.95, 0.975)
DEFAULT_NS2_SAFETY_FACTORS = (1.0, 1.1, 1.2, 1.3)
DEFAULT_FREQSPEC_MARGIN_IDLE_QUANTILES = (0.90, 0.95, 0.975)
DEFAULT_FREQSPEC_RATIO_IDLE_QUANTILES = (0.90, 0.95, 0.975)
DEFAULT_FREQSPEC_ENTROPY_CONTROL_QUANTILES = (0.80, 0.85, 0.90)
DEFAULT_FREQSPEC_LOGISTIC_PROB_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
DEFAULT_FREQSPEC_LOGISTIC_NS2_WEIGHTS = (1.0, 2.0, 3.0, 5.0)
DEFAULT_TENP5_VETO_THRESHOLDS = (0.5, 0.55, 0.6, 0.65)
DEFAULT_TENP5_NS2_WEIGHTS = (1.0, 1.5, 2.0, 3.0)
TENP5_NS2_VETO_FREQ_KEY = "10.5"
DEFAULT_NC_CALIBRATION_SECONDS = (0.0, 15.0, 30.0, 60.0, 90.0, 120.0)
NC_CALIBRATION_SOURCE_NS1 = "ns1"
NC_CALIBRATION_SOURCE_NS2 = "ns2"
NC_CALIBRATION_SOURCE_NS3 = "ns3"
NC_CALIBRATION_SOURCE_MIXED = "mixed"
NC_CALIBRATION_SOURCE_NS2_HEAVY = "ns2_heavy"
NC_CALIBRATION_SOURCES = (
    NC_CALIBRATION_SOURCE_NS1,
    NC_CALIBRATION_SOURCE_NS2,
    NC_CALIBRATION_SOURCE_NS3,
    NC_CALIBRATION_SOURCE_MIXED,
    NC_CALIBRATION_SOURCE_NS2_HEAVY,
)
NC_GATE_BASELINE_LRT_THRESHOLD = "baseline_lrt_with_nc_calibrated_threshold"
NC_GATE_SESSION_LOGISTIC = "session_specific_logistic_csns_detector"
NC_GATE_CONDITIONAL_SESSION_LOGISTIC = "conditional_baseline_plus_session_csns_detector"
NC_CALIBRATION_GATE_TYPES = (
    NC_GATE_BASELINE_LRT_THRESHOLD,
    NC_GATE_SESSION_LOGISTIC,
    NC_GATE_CONDITIONAL_SESSION_LOGISTIC,
)
NC_CSNS_PROB_THRESHOLD = 0.50
NC_CONDITIONAL_LOW_RISK_MARGIN_QUANTILE = 0.25
NC_CONDITIONAL_LOW_RISK_RATIO_QUANTILE = 0.25
NC_CONDITIONAL_LOW_RISK_ENTROPY_QUANTILE = 0.75
NC_CONDITIONAL_LOW_RISK_LRT_QUANTILE = 0.25
NC_CONDITIONAL_LOW_RISK_SAME_FREQ_COUNT = 2.0
FREQSPEC_THRESHOLD_COMBO_SET_NONE = ""
FREQSPEC_THRESHOLD_COMBO_SET_PRIORITY6 = "priority6"
FREQSPEC_THRESHOLD_COMBO_SETS = (
    FREQSPEC_THRESHOLD_COMBO_SET_NONE,
    FREQSPEC_THRESHOLD_COMBO_SET_PRIORITY6,
)
FREQSPEC_THRESHOLD_PRIORITY6_COMBOS = (
    {
        "combo_name": "mild",
        "margin_idle_quantile": 0.90,
        "ratio_idle_quantile": 0.90,
        "entropy_control_quantile": 0.90,
        "ns2_safety_factor": 1.0,
    },
    {
        "combo_name": "balanced",
        "margin_idle_quantile": 0.95,
        "ratio_idle_quantile": 0.90,
        "entropy_control_quantile": 0.90,
        "ns2_safety_factor": 1.1,
    },
    {
        "combo_name": "ns2_strict",
        "margin_idle_quantile": 0.95,
        "ratio_idle_quantile": 0.95,
        "entropy_control_quantile": 0.85,
        "ns2_safety_factor": 1.2,
    },
    {
        "combo_name": "recall_safe",
        "margin_idle_quantile": 0.90,
        "ratio_idle_quantile": 0.90,
        "entropy_control_quantile": 0.85,
        "ns2_safety_factor": 1.1,
    },
    {
        "combo_name": "margin_only-ish",
        "margin_idle_quantile": 0.95,
        "ratio_idle_quantile": 0.90,
        "entropy_control_quantile": 0.90,
        "ns2_safety_factor": 1.0,
    },
    {
        "combo_name": "ratio_ns2",
        "margin_idle_quantile": 0.90,
        "ratio_idle_quantile": 0.95,
        "entropy_control_quantile": 0.90,
        "ns2_safety_factor": 1.1,
    },
)
HIGH_FP_SUBGROUP_SUBJECTS = ("S22", "S19", "S24", "S12", "S14", "S06")
LOW_RECALL_SUBGROUP_SUBJECTS = ("S11", "S01", "S18")
OVERLAP_WATCH_SUBJECTS = ("S22", "S14")
HIGH_RISK_VALIDATION_SUBJECTS = tuple(
    dict.fromkeys(HIGH_FP_SUBGROUP_SUBJECTS + LOW_RECALL_SUBGROUP_SUBJECTS)
)
FREQSPEC_GRID_SELECTION_POLICY = "calibration_internal_validation_first"
CONDITIONAL_GATE_CONFIGS = (
    {
        "conditional_policy": "conservative",
        "conditional_low_risk_margin_quantile": 0.05,
        "conditional_low_risk_ratio_quantile": 0.05,
        "conditional_low_risk_entropy_quantile": 0.95,
        "conditional_low_risk_lrt_quantile": 0.05,
        "conditional_high_risk_margin_quantile": 0.10,
        "conditional_high_risk_ratio_quantile": 0.10,
        "conditional_high_risk_entropy_quantile": 0.90,
        "conditional_high_risk_lrt_quantile": 0.10,
        "conditional_low_risk_same_freq_count": 2.0,
        "conditional_high_risk_same_freq_count": 1.0,
        "conditional_extra_windows": 0,
    },
    {
        "conditional_policy": "balanced",
        "conditional_low_risk_margin_quantile": 0.10,
        "conditional_low_risk_ratio_quantile": 0.10,
        "conditional_low_risk_entropy_quantile": 0.90,
        "conditional_low_risk_lrt_quantile": 0.10,
        "conditional_high_risk_margin_quantile": 0.15,
        "conditional_high_risk_ratio_quantile": 0.15,
        "conditional_high_risk_entropy_quantile": 0.85,
        "conditional_high_risk_lrt_quantile": 0.15,
        "conditional_low_risk_same_freq_count": 2.0,
        "conditional_high_risk_same_freq_count": 1.0,
        "conditional_extra_windows": 1,
    },
    {
        "conditional_policy": "recall_safe",
        "conditional_low_risk_margin_quantile": 0.15,
        "conditional_low_risk_ratio_quantile": 0.15,
        "conditional_low_risk_entropy_quantile": 0.85,
        "conditional_low_risk_lrt_quantile": 0.15,
        "conditional_high_risk_margin_quantile": 0.20,
        "conditional_high_risk_ratio_quantile": 0.20,
        "conditional_high_risk_entropy_quantile": 0.80,
        "conditional_high_risk_lrt_quantile": 0.20,
        "conditional_low_risk_same_freq_count": 1.0,
        "conditional_high_risk_same_freq_count": 1.0,
        "conditional_extra_windows": 0,
        "conditional_risk_freqs": "10.5",
    },
)
FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES = (
    "selected_freq_score",
    "top1_score",
    "top2_score",
    "margin",
    "ratio",
    "normalized_top1",
    "score_entropy",
    "lrt_evidence",
    "multiwindow_same_freq_count",
    "multiwindow_margin_mean",
    "multiwindow_entropy_mean",
)
TENP5_NS2_VETO_FEATURE_NAMES = (
    "selected_freq_score",
    "top1_score",
    "lrt_evidence",
    "margin",
    "ratio",
    "score_entropy",
    "multiwindow_same_freq_count",
    "multiwindow_margin_mean",
)
NC_CSNS_FEATURE_NAMES = (
    "top1_score",
    "top2_score",
    "selected_freq_score",
    "margin",
    "ratio",
    "score_entropy",
    "lrt_evidence",
    "multiwindow_same_freq_count",
    "multiwindow_margin_mean",
    "multiwindow_entropy_mean",
    "selected_freq_is_8",
    "selected_freq_is_10.5",
    "selected_freq_is_12",
    "selected_freq_is_15",
)
ADAPTIVE_EVIDENCE_FEATURE_NAMES = (
    "command_probability",
    "top_command_probability",
    "command_vs_idle_probability_margin",
    "top_command_probability_margin",
    "top_command_probability_ratio",
    "probability_entropy",
    "same_top_command_as_previous",
    "causal_top_command_streak",
    "full_bank_command_to_all_ratio",
    "full_bank_nearest_noncommand_margin",
    "full_bank_inverse_command_rank",
    "full_bank_entropy",
)
DEFAULT_ADAPTIVE_EVIDENCE_ENTER_CANDIDATES = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
DEFAULT_ADAPTIVE_EVIDENCE_DECAY = 0.50
DEFAULT_LRT_MULTIWINDOW_ENTER_CANDIDATES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
DEFAULT_LRT_MULTIWINDOW_DECAY = 0.65


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
    all_score_matrix: Optional[np.ndarray] = None
    all_freqs: tuple[float, ...] = ()


@dataclass(frozen=True)
class FrequencyEvalCase:
    mode: str
    frequency_set_id: str
    freqs: tuple[float, float, float, float]
    candidate_freqs: tuple[float, ...] = ()
    personalized_candidate_count: int = 0
    selected_by_calibration: bool = False


@dataclass(frozen=True)
class CaseContext:
    dataset: str
    subject: str
    frequency_profile: str
    frequency_set_id: str
    selected_freqs: tuple[float, float, float, float]
    method: str
    calibration_blocks: tuple[int, ...]
    holdout_blocks: tuple[int, ...]
    split_index: int
    window_length_s: float
    min_enter_windows: int
    reject_gate: str
    implementation_level: str


@dataclass
class CaseTracker:
    expected_subject_count: int
    entries: list[dict[str, Any]]
    failed_cases: list[dict[str, Any]]

    def __init__(self, *, expected_subject_count: int) -> None:
        self.expected_subject_count = int(expected_subject_count)
        self.entries = []
        self.failed_cases = []

    def planned(self, ctx: CaseContext) -> None:
        self.entries.append({**asdict(ctx), "status": "planned"})

    def completed(self, ctx: CaseContext, *, row: Mapping[str, Any]) -> None:
        self.entries.append(
            {
                **asdict(ctx),
                "status": "completed",
                "recipe_id": str(row.get("recipe_id", "")),
                "artifact_paths": _artifact_paths_from_row(row),
            }
        )

    def skipped(self, ctx: CaseContext, *, reason: str, detail: str = "") -> dict[str, Any]:
        payload = _case_failure_payload(ctx, skip_or_fail="skip", reason=reason, detail=detail)
        self.entries.append({**asdict(ctx), "status": "skipped", "skip_reason": str(reason), "detail": str(detail)})
        self.failed_cases.append(payload)
        return payload

    def skipped_cases(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.failed_cases if str(item.get("skip_or_fail", "")) == "skip"]

    def failed_cases_only(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.failed_cases if str(item.get("skip_or_fail", "")) == "fail"]

    def status_count(self, status: str) -> int:
        normalized = str(status).strip().lower()
        return int(sum(1 for item in self.entries if str(item.get("status", "")).strip().lower() == normalized))

    def failed(self, ctx: CaseContext, *, exc: BaseException) -> dict[str, Any]:
        payload = _case_failure_payload(ctx, skip_or_fail="fail", reason=type(exc).__name__, detail=str(exc))
        payload["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
        self.entries.append({**asdict(ctx), "status": "failed", "error_type": type(exc).__name__, "detail": str(exc)})
        self.failed_cases.append(payload)
        return payload

    def report(self) -> dict[str, Any]:
        grouped: dict[str, Any] = {}
        for item in self.entries:
            dataset = str(item.get("dataset", ""))
            profile = str(item.get("frequency_profile", ""))
            method = str(item.get("method", ""))
            subject = str(item.get("subject", ""))
            cal_key = ",".join(str(block) for block in item.get("calibration_blocks", []) or [])
            split_key = f"split{int(item.get('split_index', 0)):02d}"
            leaf = grouped.setdefault(dataset, {}).setdefault(profile, {}).setdefault(method, {}).setdefault(subject, {}).setdefault(cal_key, {}).setdefault(split_key, {})
            status = str(item.get("status", "planned"))
            leaf[status] = int(leaf.get(status, 0)) + 1
            leaf["selected_freqs"] = [float(freq) for freq in item.get("selected_freqs", []) or []]
            leaf["frequency_set_id"] = str(item.get("frequency_set_id", ""))
        by_method: dict[str, dict[str, int]] = {}
        subjects_completed_by_method: dict[str, set[str]] = {}
        for item in self.entries:
            key = f"{item.get('dataset','')}|{item.get('frequency_profile','')}|{item.get('method','')}"
            status = str(item.get("status", "planned"))
            by_method.setdefault(key, {"planned": 0, "completed": 0, "skipped": 0, "failed": 0})
            if status in by_method[key]:
                by_method[key][status] += 1
            if status == "completed":
                subjects_completed_by_method.setdefault(key, set()).add(f"{item.get('dataset','')}:{item.get('subject','')}")
        flat = []
        for key, counts in sorted(by_method.items()):
            dataset, profile, method = key.split("|", 2)
            completed_subjects = sorted(subjects_completed_by_method.get(key, set()))
            flat.append(
                {
                    "dataset": dataset,
                    "frequency_profile": profile,
                    "method": method,
                    **counts,
                    "subjects_completed": len(completed_subjects),
                    "subjects_expected": int(self.expected_subject_count),
                    "shared_eligible": bool(
                        self.expected_subject_count > 0 and len(completed_subjects) == int(self.expected_subject_count)
                    ),
                    "completed_subject_ids": completed_subjects,
                }
            )
        return {
            "schema_version": "ssvep_external_short_pretrain_coverage_v1",
            "subjects_expected": int(self.expected_subject_count),
            "event_count": int(len(self.entries)),
            "case_count": int(self.status_count("planned")),
            "planned_case_count": int(self.status_count("planned")),
            "completed_case_count": int(self.status_count("completed")),
            "skipped_case_count": int(self.status_count("skipped")),
            "failed_case_count": int(len(self.failed_cases)),
            "hard_failed_case_count": int(self.status_count("failed")),
            "by_dataset_frequency_profile_method_subject": grouped,
            "by_dataset_frequency_profile_method": flat,
        }


def _case_failure_payload(ctx: CaseContext, *, skip_or_fail: str, reason: str, detail: str = "") -> dict[str, Any]:
    return {
        "dataset": str(ctx.dataset),
        "subject": str(ctx.subject),
        "frequency_profile": str(ctx.frequency_profile),
        "frequency_set_id": str(ctx.frequency_set_id),
        "selected_freqs": [float(freq) for freq in ctx.selected_freqs],
        "method": str(ctx.method),
        "calibration_blocks": [int(block) for block in ctx.calibration_blocks],
        "excluded_test_blocks": [int(block) for block in ctx.holdout_blocks],
        "split_index": int(ctx.split_index),
        "window_length_s": float(ctx.window_length_s),
        "min_enter_windows": int(ctx.min_enter_windows),
        "reject_gate": str(ctx.reject_gate),
        "implementation_level": str(ctx.implementation_level),
        "skip_or_fail": str(skip_or_fail),
        "skip_reason": str(reason) if str(skip_or_fail) == "skip" else "",
        "error_type": str(reason) if str(skip_or_fail) == "fail" else "",
        "detail": str(detail),
    }


def _exception_is_insufficient_training(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "insufficient",
        "not enough",
        "too few",
        "requires at least",
        "requires >= ",
        "no calibration",
        "empty calibration",
        "singular",
        "not invertible",
        "sample",
        "trial",
    )
    return any(needle in message for needle in needles)


def _is_tdca_insufficient_case(ctx: CaseContext, exc: BaseException) -> bool:
    return str(ctx.method).strip().lower() == "tdca5" and _exception_is_insufficient_training(exc)


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
    smoothing_windows: int = 1
    gate_policy: str = CLASSIFIER_CONFIDENCE_GATE_POLICY
    evidence_weights: Optional[np.ndarray] = None
    evidence_feature_mean: Optional[np.ndarray] = None
    evidence_feature_std: Optional[np.ndarray] = None
    evidence_decision_th: float = 0.0
    evidence_enter_th: float = 0.0
    evidence_decay: float = DEFAULT_ADAPTIVE_EVIDENCE_DECAY
    lrt_feature_indices: tuple[int, ...] = ()
    lrt_feature_mean_control: Optional[np.ndarray] = None
    lrt_feature_std_control: Optional[np.ndarray] = None
    lrt_feature_mean_idle: Optional[np.ndarray] = None
    lrt_feature_std_idle: Optional[np.ndarray] = None
    lrt_window_th: float = 0.0
    lrt_enter_th: float = 0.0
    lrt_decay: float = DEFAULT_LRT_MULTIWINDOW_DECAY
    gate_variant: str = CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW
    score_shape_margin_index: Optional[int] = None
    score_shape_ratio_index: Optional[int] = None
    score_shape_entropy_index: Optional[int] = None
    score_shape_margin_th: Optional[float] = None
    score_shape_ratio_th: Optional[float] = None
    score_shape_entropy_th: Optional[float] = None
    lrt_window_floor_th: Optional[float] = None
    weak_subject_guard_active: bool = False
    weak_subject_guard_reasons: tuple[str, ...] = ()
    frequency_specific_control_state_gates: Optional[dict[str, dict[str, Any]]] = None


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
    smoothing_windows: int = 1
    gate_policy: str = CLASSIFIER_CONFIDENCE_GATE_POLICY
    evidence_weights: Optional[np.ndarray] = None
    evidence_feature_mean: Optional[np.ndarray] = None
    evidence_feature_std: Optional[np.ndarray] = None
    evidence_decision_th: float = 0.0
    evidence_enter_th: float = 0.0
    evidence_decay: float = DEFAULT_ADAPTIVE_EVIDENCE_DECAY
    lrt_feature_indices: tuple[int, ...] = ()
    lrt_feature_mean_control: Optional[np.ndarray] = None
    lrt_feature_std_control: Optional[np.ndarray] = None
    lrt_feature_mean_idle: Optional[np.ndarray] = None
    lrt_feature_std_idle: Optional[np.ndarray] = None
    lrt_window_th: float = 0.0
    lrt_enter_th: float = 0.0
    lrt_decay: float = DEFAULT_LRT_MULTIWINDOW_DECAY
    gate_variant: str = CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW
    score_shape_margin_index: Optional[int] = None
    score_shape_ratio_index: Optional[int] = None
    score_shape_entropy_index: Optional[int] = None
    score_shape_margin_th: Optional[float] = None
    score_shape_ratio_th: Optional[float] = None
    score_shape_entropy_th: Optional[float] = None
    lrt_window_floor_th: Optional[float] = None
    weak_subject_guard_active: bool = False
    weak_subject_guard_reasons: tuple[str, ...] = ()
    frequency_specific_control_state_gates: Optional[dict[str, dict[str, Any]]] = None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(json_safe(payload)) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flattened = {}
            for key, value in dict(row).items():
                flattened[key] = (
                    json.dumps(json_safe(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
            writer.writerow(flattened)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
        if not np.isfinite(result):
            return float(default)
        return result
    except Exception:
        return float(default)


def _safe_mean(values: Sequence[Any], default: float = float("nan")) -> float:
    parsed = [
        float(value)
        for value in (_safe_float(item, float("nan")) for item in values)
        if np.isfinite(float(value))
    ]
    if not parsed:
        return float(default)
    return float(np.mean(np.asarray(parsed, dtype=np.float64)))


def _finite_or_none(value: Any) -> Optional[float]:
    parsed = _safe_float(value, float("nan"))
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def _safe_quantile(values: Sequence[Any] | np.ndarray, quantile: float, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float(default)
    return float(np.quantile(arr, min(max(float(quantile), 0.0), 1.0)))


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    den = float(denominator)
    if abs(den) <= 1e-12:
        return float(default)
    return float(float(numerator) / den)


def _first_finite_metric(metrics: Mapping[str, Any], keys: Sequence[str], default: float = float("nan")) -> float:
    for key in keys:
        value = _safe_float(metrics.get(key), float("nan"))
        if np.isfinite(value):
            return float(value)
    return float(default)


def _frequency_profile_name(freqs: Sequence[float]) -> str:
    key = tuple(round(float(freq), 6) for freq in freqs)
    if key == (8.0, 10.0, 12.0, 15.0):
        return "deploy_current_profile"
    if key == (9.8, 12.0, 14.8, 15.8):
        return "exploratory_profile"
    return "custom_" + "_".join(f"{float(freq):g}".replace(".", "p") for freq in freqs)


def _method_implementation_level(method_name: str, calibration_block_count: int) -> str:
    name = str(method_name).strip().lower()
    cal_count = int(calibration_block_count)
    if name in {"fbcca_lda5", "fbcca_ridge5", "fbcca_ridge5_nc_calibration", "zero_shot_default"}:
        return "paper-faithful"
    if name == "trca5":
        return "paper-faithful" if cal_count >= 2 else "engineering-approx"
    if name == "trca_r5":
        return "paper-faithful" if cal_count >= 2 else "engineering-approx"
    return "engineering-approx"


def _idle_source_policy_for_dataset(dataset: str) -> dict[str, Any]:
    normalized = str(dataset).strip().lower()
    if normalized == "ysu_an":
        return {
            "primary_idle_source": "real_idle",
            "idle_source_policy": "real_no_control_trials",
            "real_idle_available": True,
            "approx_idle_allowed": False,
            "note": "YSU-an uses explicit NS1/NS2/NS3 no-control trials for reject evaluation.",
        }
    return {
        "primary_idle_source": "engineering_approx",
        "idle_source_policy": "engineering_approx_noncommand_targets",
        "real_idle_available": False,
        "approx_idle_allowed": True,
        "note": "This external dataset path lacks reliable rest/blank holdout by default; non-command target trials are marked engineering-approx idle.",
    }


def _metric_definitions_payload(
    *,
    step_sec: float,
    decision_start_sec: float,
    decision_deadline_sec: float,
    min_release_windows: int,
) -> dict[str, Any]:
    return {
        "state_definition": "async5 means 4 command frequencies plus reject/no-control; classifiers train only command frequencies and reject is gate-derived.",
        "step_size_s": float(step_sec),
        "decision_start_s": float(decision_start_sec),
        "decision_deadline_s": float(decision_deadline_sec),
        "min_enter_definition": "A control output is emitted only after min_enter consecutive windows pass the enter gate.",
        "min_release_windows": int(min_release_windows),
        "min_release_definition": "Release requires consecutive windows below the control gate; replay support is metric-schema only in this benchmark.",
        "detection_latency_s": "stimulus onset to the first correct control output; missed control trials use the trial duration plus window length as fallback in async decision metrics.",
        "switch_latency_s": "target switch to first correct new-target output; unsupported external fixed-trial replay reports support=false.",
        "release_latency_s": "entry into no-control to stable release; unsupported external fixed-trial replay reports support=false.",
    }


def _reject_gate_name(*, method_name: str, threshold_policy: str, score_bank_mode: str) -> str:
    normalized = _parse_classifier_threshold_policy(threshold_policy)
    if str(method_name).strip().lower() in {"zero_shot_default", "fast_fbcca", "threshold_pretrain"}:
        return "profile_gate"
    if normalized == "balanced":
        return "confidence_threshold"
    if normalized == "balanced_recall_guard":
        return "confidence_threshold_recall_guard"
    return f"{normalized}:{_parse_score_bank_mode(score_bank_mode)}"


def _artifact_paths_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    calibration_profile = dict(row.get("calibration_profile", {}) or {})
    paths: dict[str, Any] = {
        "profile_path": str(calibration_profile.get("profile_path", "")),
        "profile_v2_path": str(calibration_profile.get("profile_v2_path", "")),
        "candidate_artifact_path": str(calibration_profile.get("candidate_artifact_path", "")),
    }
    return {key: value for key, value in paths.items() if value}


def _nc_gate_type_token(gate_type: str) -> str:
    resolved = _parse_nc_gate_type(gate_type)
    if resolved == NC_GATE_BASELINE_LRT_THRESHOLD:
        return "nclrt"
    if resolved == NC_GATE_SESSION_LOGISTIC:
        return "nclog"
    if resolved == NC_GATE_CONDITIONAL_SESSION_LOGISTIC:
        return "nccond"
    return str(resolved).replace("_", "")


def _seconds_token(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p")
    return text.replace("-", "m")


def _nc_recipe_id(
    *,
    base_recipe_id: str,
    seconds: float,
    source: str,
    gate_type: str,
) -> str:
    return (
        f"{str(base_recipe_id)}_nc{_seconds_token(float(seconds))}s_"
        f"{_parse_nc_calibration_source(source)}_{_nc_gate_type_token(gate_type)}"
    )


def _candidate_artifact_paths_for_recipe(
    rows: Sequence[Mapping[str, Any]],
    recipe: Mapping[str, Any],
) -> list[str]:
    if not recipe:
        return []
    embedded = list(dict(dict(recipe).get("artifact_paths", {}) or {}).get("candidate_artifacts", []) or [])
    row_paths = [
        str(dict(row.get("calibration_profile", {}) or {}).get("candidate_artifact_path", ""))
        for row in _deployable_recipe_rows(rows, recipe)
        if str(dict(row.get("calibration_profile", {}) or {}).get("candidate_artifact_path", "")).strip()
    ]
    return sorted(dict.fromkeys(str(path) for path in [*embedded, *row_paths] if str(path).strip()))


def _enrich_result_row(
    row: Mapping[str, Any],
    *,
    frequency_profile: str,
    frequency_case: FrequencyEvalCase,
    step_sec: float,
    decision_start_sec: float,
    decision_deadline_sec: float,
    min_release_windows: int,
    threshold_policy: str,
    score_bank_mode: str,
) -> dict[str, Any]:
    enriched = dict(row)
    method_name = str(enriched.get("method", ""))
    calibration_blocks = [int(block) for block in enriched.get("calibration_blocks", []) or []]
    calibration_block_count = int(len(calibration_blocks))
    summary_metrics = dict(enriched.get("summary_metrics", {}) or {})
    idle_policy = _idle_source_policy_for_dataset(str(enriched.get("dataset", "")))
    if not np.isfinite(_safe_float(summary_metrics.get("real_idle_fp_per_min"), float("nan"))):
        if bool(idle_policy.get("real_idle_available", False)):
            summary_metrics["real_idle_fp_per_min"] = _safe_float(summary_metrics.get("mixed_idle_fp_per_min"), float("nan"))
        else:
            summary_metrics["real_idle_fp_per_min"] = float("nan")
    if not np.isfinite(_safe_float(summary_metrics.get("approx_idle_fp_per_min"), float("nan"))):
        summary_metrics["approx_idle_fp_per_min"] = _safe_float(summary_metrics.get("idle_fp_per_min"), float("inf"))
    if not np.isfinite(_safe_float(summary_metrics.get("mixed_idle_fp_per_min"), float("nan"))):
        summary_metrics["mixed_idle_fp_per_min"] = _first_finite_metric(
            summary_metrics,
            ("real_idle_fp_per_min", "approx_idle_fp_per_min", "idle_fp_per_min"),
            default=float("inf"),
        )
    implementation_level = _method_implementation_level(method_name, calibration_block_count)
    reject_gate = _reject_gate_name(
        method_name=method_name,
        threshold_policy=threshold_policy,
        score_bank_mode=score_bank_mode,
    )
    enriched["frequency_profile"] = str(frequency_profile)
    enriched["frequency_set_id"] = str(frequency_case.frequency_set_id)
    enriched["frequency_selection_mode"] = str(frequency_case.mode)
    enriched["selected_freqs"] = [float(freq) for freq in frequency_case.freqs]
    enriched["window_length_s"] = _safe_float(_window_length_from_row(enriched), 0.0)
    enriched["step_size_s"] = float(step_sec)
    enriched["decision_start_s"] = float(decision_start_sec)
    enriched["decision_deadline_s"] = float(decision_deadline_sec)
    enriched["min_release_windows"] = int(min_release_windows)
    enriched["reject_gate"] = reject_gate
    calibration_profile = dict(enriched.get("calibration_profile", {}) or {})
    fit_summary = dict(calibration_profile.get("fit_summary", {}) or {})
    gate_variant = parse_classifier_gate_variant(
        calibration_profile.get("gate_variant", fit_summary.get("gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW))
    )
    gate_variant_params = dict(
        calibration_profile.get(
            "gate_variant_params",
            fit_summary.get("gate_variant_params", {"gate_variant": gate_variant}),
        )
        or {"gate_variant": gate_variant}
    )
    enriched["gate_variant"] = gate_variant
    enriched["gate_variant_params"] = gate_variant_params
    enriched["implementation_level"] = implementation_level
    enriched["paper_faithful"] = bool(implementation_level == "paper-faithful")
    enriched["engineering_approx"] = bool(implementation_level != "paper-faithful")
    enriched["idle_source_policy"] = idle_policy
    enriched["artifact_paths"] = _artifact_paths_from_row(enriched)
    enriched["summary_metrics"] = summary_metrics
    diagnostics = dict(enriched.get("ns2_selected_freq_diagnostics", {}) or {})
    for diag_key in ("ns2_by_selected_freq", "ns2_by_subject_freq", "selected_freq_confusion"):
        if diag_key in diagnostics:
            diagnostics[diag_key] = [
                {
                    **dict(item),
                    "frequency_profile": str(frequency_profile),
                    "frequency_set_id": str(frequency_case.frequency_set_id),
                    "selected_freqs": [float(freq) for freq in frequency_case.freqs],
                    "method": method_name,
                    "calibration_blocks": [int(block) for block in calibration_blocks],
                    "holdout_blocks": [int(block) for block in enriched.get("holdout_blocks", []) or []],
                    "win_sec": _safe_float(enriched.get("window_length_s"), 0.0),
                    "step_sec": float(step_sec),
                    "min_enter_windows": int(
                        dict(enriched.get("calibration_profile", {}) or {}).get("min_enter_windows", 0) or 0
                    ),
                    "smoothing_windows": int(
                        dict(enriched.get("calibration_profile", {}) or {}).get("smoothing_windows", 1) or 1
                    ),
                }
                for item in list(diagnostics.get(diag_key, []) or [])
            ]
    enriched["ns2_selected_freq_diagnostics"] = diagnostics
    trace_diagnostics = dict(enriched.get("logistic_trace_diagnostics", {}) or {})
    for trace_key in (
        "logistic_trace_windows",
        "logistic_trace_trial_summary",
        "logistic_transition_counts_by_subject",
        "logistic_transition_counts_by_frequency",
        "logistic_feature_summary_tp_fp",
    ):
        if trace_key in trace_diagnostics:
            trace_diagnostics[trace_key] = [
                {
                    **dict(item),
                    "dataset": str(enriched.get("dataset", "")) or str(dict(item).get("dataset", "")),
                    "subject": str(enriched.get("subject", "")) or str(dict(item).get("subject", "")),
                    "split_index": int(enriched.get("split_index", dict(item).get("split_id", 0)) or 0),
                    "method": method_name,
                    "recipe_id": str(enriched.get("recipe_id", dict(item).get("recipe_id", ""))),
                    "gate_variant": gate_variant,
                    "frequency_profile": str(frequency_profile),
                    "frequency_set_id": str(frequency_case.frequency_set_id),
                    "calibration_blocks": [int(block) for block in calibration_blocks],
                    "holdout_blocks": [int(block) for block in enriched.get("holdout_blocks", []) or []],
                    "win_sec": _safe_float(enriched.get("window_length_s"), 0.0),
                    "step_sec": float(step_sec),
                    "min_enter_windows": int(
                        dict(enriched.get("calibration_profile", {}) or {}).get("min_enter_windows", 0) or 0
                    ),
                    "smoothing_windows": int(
                        dict(enriched.get("calibration_profile", {}) or {}).get("smoothing_windows", 1) or 1
                    ),
                }
                for item in list(trace_diagnostics.get(trace_key, []) or [])
            ]
    enriched["logistic_trace_diagnostics"] = trace_diagnostics
    tenp5_diagnostics = dict(enriched.get("tenp5_ns2_veto_diagnostics", {}) or {})
    for diag_key in ("tenp5_ns2_veto_diagnostics", "tenp5_ns2_veto_summary_rows"):
        if diag_key in tenp5_diagnostics:
            tenp5_diagnostics[diag_key] = [
                {
                    **dict(item),
                    "dataset": str(enriched.get("dataset", "")) or str(dict(item).get("dataset", "")),
                    "subject": str(enriched.get("subject", "")) or str(dict(item).get("subject", "")),
                    "split_index": int(enriched.get("split_index", dict(item).get("split_id", 0)) or 0),
                    "method": method_name,
                    "recipe_id": str(enriched.get("recipe_id", dict(item).get("recipe_id", ""))),
                    "gate_variant": gate_variant,
                    "frequency_profile": str(frequency_profile),
                    "frequency_set_id": str(frequency_case.frequency_set_id),
                    "calibration_blocks": [int(block) for block in calibration_blocks],
                    "holdout_blocks": [int(block) for block in enriched.get("holdout_blocks", []) or []],
                    "win_sec": _safe_float(enriched.get("window_length_s"), 0.0),
                    "step_sec": float(step_sec),
                    "min_enter_windows": int(
                        dict(enriched.get("calibration_profile", {}) or {}).get("min_enter_windows", 0) or 0
                    ),
                    "smoothing_windows": int(
                        dict(enriched.get("calibration_profile", {}) or {}).get("smoothing_windows", 1) or 1
                    ),
                }
                for item in list(tenp5_diagnostics.get(diag_key, []) or [])
            ]
    enriched["tenp5_ns2_veto_diagnostics"] = tenp5_diagnostics
    enriched["per_frequency_metrics"] = [
        {
            **dict(item),
            "dataset": str(enriched.get("dataset", "")),
            "subject": str(enriched.get("subject", "")),
            "split_index": int(enriched.get("split_index", 0) or 0),
            "method": method_name,
            "recipe_id": str(enriched.get("recipe_id", "")),
            "gate_variant": gate_variant,
            "frequency_profile": str(frequency_profile),
            "frequency_set_id": str(frequency_case.frequency_set_id),
            "calibration_blocks": [int(block) for block in calibration_blocks],
            "win_sec": _safe_float(enriched.get("window_length_s"), 0.0),
            "step_sec": float(step_sec),
            "min_enter_windows": int(dict(enriched.get("calibration_profile", {}) or {}).get("min_enter_windows", 0) or 0),
            "smoothing_windows": int(dict(enriched.get("calibration_profile", {}) or {}).get("smoothing_windows", 1) or 1),
        }
        for item in list(enriched.get("per_frequency_metrics", []) or [])
    ]
    return enriched


def _window_length_from_row(row: Mapping[str, Any]) -> float:
    calibration_profile = dict(row.get("calibration_profile", {}) or {})
    for key in ("win_sec", "window_length_s"):
        if key in calibration_profile:
            return _safe_float(calibration_profile.get(key), 0.0)
        if key in row:
            return _safe_float(row.get(key), 0.0)
    fit_summary = dict(calibration_profile.get("fit_summary", {}) or {})
    if "win_sec" in fit_summary:
        return _safe_float(fit_summary.get("win_sec"), 0.0)
    recipe = str(row.get("recipe_id", ""))
    match = re.search(r"win([0-9]+(?:p[0-9]+)?)", recipe)
    if match:
        return float(match.group(1).replace("p", "."))
    return 0.0


def _resource_limits_payload(timeout_sec: float, *, case_limit: int = DEFAULT_CASE_LIMIT) -> dict[str, Any]:
    cpu_count = int(os.cpu_count() or 1)
    return {
        "gpu_max_concurrent_tasks": 1,
        "cpu_max_workers": int(min(8, max(1, cpu_count // 2))),
        "timeout_sec": float(timeout_sec),
        "timeout_enabled": bool(float(timeout_sec) > 0.0),
        "case_limit": int(max(0, int(case_limit))),
        "case_limit_enabled": bool(int(max(0, int(case_limit))) > 0),
        "explicit_run_id_required": True,
    }


def _artifact_manifest_paths(
    *,
    report_root: Path,
    run_id: str,
    log_path: Path,
    failed_cases_path: Path,
    coverage_report_path: Path,
    server_log_path: Optional[Path | PurePosixPath] = None,
) -> dict[str, Any]:
    resolved_server_log = server_log_path or (SERVER_SSVEP_LOG_ROOT / f"{run_id}.log")
    return {
        "summary_json": str(report_root / "summary.json"),
        "summary_md": str(report_root / "summary.md"),
        "partial_summary_json": str(report_root / "partial_summary.json"),
        "failed_cases_json": str(failed_cases_path),
        "coverage_report_json": str(coverage_report_path),
        "subject_breakdown_csv": str(report_root / "subject_breakdown.csv"),
        "subtype_breakdown_csv": str(report_root / "subtype_breakdown.csv"),
        "ns2_by_selected_freq_csv": str(report_root / "ns2_by_selected_freq.csv"),
        "ns2_by_subject_freq_csv": str(report_root / "ns2_by_subject_freq.csv"),
        "selected_freq_confusion_csv": str(report_root / "selected_freq_confusion.csv"),
        "per_frequency_metrics_csv": str(report_root / "per_frequency_metrics.csv"),
        "candidate_comparison_csv": str(report_root / "candidate_comparison.csv"),
        "gate_params_json": str(report_root / "gate_params.json"),
        "gate_params_by_frequency_json": str(report_root / "gate_params_by_frequency.json"),
        "logistic_trace_windows_csv": str(report_root / "logistic_trace_windows.csv"),
        "logistic_trace_trial_summary_csv": str(report_root / "logistic_trace_trial_summary.csv"),
        "tenp5_ns2_veto_diagnostics_csv": str(report_root / "tenp5_ns2_veto_diagnostics.csv"),
        "tenp5_ns2_veto_summary_json": str(report_root / "tenp5_ns2_veto_summary.json"),
        "logistic_transition_counts_by_subject_csv": str(report_root / "logistic_transition_counts_by_subject.csv"),
        "logistic_transition_counts_by_frequency_csv": str(report_root / "logistic_transition_counts_by_frequency.csv"),
        "logistic_feature_summary_tp_fp_csv": str(report_root / "logistic_feature_summary_tp_fp.csv"),
        "nc_calibration_budget_curve_csv": str(report_root / "nc_calibration_budget_curve.csv"),
        "csns_feature_summary_csv": str(report_root / "csns_feature_summary.csv"),
        "trace_separability_summary_json": str(report_root / "trace_separability_summary.json"),
        "trace_separability_summary_md": str(report_root / "trace_separability_summary.md"),
        "transition_by_subject_freq_csv": str(report_root / "transition_by_subject_freq.csv"),
        "feature_separability_by_subject_freq_csv": str(report_root / "feature_separability_by_subject_freq.csv"),
        "risk_rule_candidates_csv": str(report_root / "risk_rule_candidates.csv"),
        "run_log_copy": str(report_root / "logs" / f"{run_id}.log"),
        "deployable_candidate_profile_json": str(report_root / DEPLOYABLE_CANDIDATE_PROFILE_FILENAME),
        "local_log": str(log_path),
        "server_log_contract": str(resolved_server_log),
    }


def _recipe_token_value(recipe_id: str, prefix: str, default: float) -> float:
    match = re.search(rf"(?:^|_){re.escape(prefix)}([0-9]+(?:p[0-9]+)?)", str(recipe_id))
    if not match:
        return float(default)
    return float(str(match.group(1)).replace("p", "."))


def _recipe_has_token(recipe_id: str, token: str) -> bool:
    return bool(re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", str(recipe_id)))


def _deployable_recipe_rows(rows: Sequence[Mapping[str, Any]], recipe: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not recipe:
        return []
    method = str(recipe.get("method", ""))
    recipe_id = str(recipe.get("recipe_id", ""))
    frequency_set_id = str(recipe.get("frequency_set_id", ""))
    idle_multiplier = _safe_float(recipe.get("idle_multiplier"), float("nan"))
    calibration_block_count = int(_safe_float(recipe.get("calibration_block_count"), -1.0))
    matches: list[dict[str, Any]] = []
    for row in rows:
        row_recipe_id = str(row.get("aggregate_recipe_id") or row.get("recipe_id", ""))
        if method and str(row.get("method", "")) != method:
            continue
        if recipe_id and row_recipe_id != recipe_id:
            continue
        if frequency_set_id and str(row.get("frequency_set_id", "")) != frequency_set_id:
            continue
        row_idle_multiplier = _safe_float(dict(row.get("split_summary", {}) or {}).get("idle_multiplier"), float("nan"))
        if np.isfinite(idle_multiplier) and (
            not np.isfinite(row_idle_multiplier) or abs(float(row_idle_multiplier) - float(idle_multiplier)) > 1e-9
        ):
            continue
        calibration_blocks = [int(block) for block in row.get("calibration_blocks", []) or []]
        if calibration_block_count >= 0 and len(calibration_blocks) != calibration_block_count:
            continue
        matches.append(dict(row))
    return matches


def _deployable_candidate_profile_payload(
    *,
    run_id: str,
    best_deployable_shared_recipe: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    channel_compatibility: Mapping[str, Any],
    artifact_paths: Mapping[str, Any],
) -> dict[str, Any]:
    recipe = dict(best_deployable_shared_recipe or {})
    candidate_rows = _deployable_recipe_rows(rows, recipe)
    recipe_id = str(recipe.get("recipe_id", ""))
    calibration_profile = dict(recipe.get("calibration_profile", {}) or {})
    min_enter_windows = int(_recipe_token_value(recipe_id, "me", calibration_profile.get("min_enter_windows", 1)))
    smoothing_windows = int(_recipe_token_value(recipe_id, "sm", 1.0))
    max_gap_windows = int(_recipe_token_value(recipe_id, "gap", 0.0))
    threshold_policy = (
        CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY
        if _recipe_has_token(recipe_id, "lrtmw")
        else CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY
        if _recipe_has_token(recipe_id, "aeg")
        else CLASSIFIER_SUBJECT_ADAPTIVE_THRESHOLD_POLICY
        if _recipe_has_token(recipe_id, "sat")
        else DEFAULT_CLASSIFIER_THRESHOLD_POLICY
    )
    win_sec = _recipe_token_value(recipe_id, "win", recipe.get("mean_detection_latency_s", DEFAULT_CLASSIFIER_WIN_SEC_CANDIDATES[0]))
    candidate_paths = [
        str(dict(row.get("calibration_profile", {}) or {}).get("candidate_artifact_path", ""))
        for row in candidate_rows
        if str(dict(row.get("calibration_profile", {}) or {}).get("candidate_artifact_path", "")).strip()
    ]
    unique_candidate_paths = sorted(dict.fromkeys(candidate_paths))
    return {
        "schema_version": DEPLOYABLE_CANDIDATE_PROFILE_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": str(run_id),
        "status": "deployable_candidate_found" if recipe else "no_deployable_candidate",
        "runtime_loadable": bool(recipe) and False,
        "runtime_load_note": (
            "Shared recipe manifest only. The runtime bridge exists for per-subject profiles that "
            "include model_params.state; do not copy this JSON or recipe manifest over "
            "datasets/profiles/ssvep/default_profile.json."
        ),
        "candidate_name": "external_fbcca_classifier_candidate_v1",
        "recommended_short_pretrain_recipe": {
            "method": str(recipe.get("method", "")),
            "recipe_id": recipe_id,
            "decoder": "fbcca_fixed_all8",
            "classifier": "fbcca_score_ridge_5class",
            "score_bank_mode": "full_reference_bank",
            "threshold_policy": threshold_policy,
            "win_sec": float(win_sec),
            "step_sec": float(DEFAULT_STEP_SEC),
            "min_enter_windows": int(min_enter_windows),
            "max_gap_windows": int(max_gap_windows),
            "smoothing_windows": int(smoothing_windows),
            "idle_multiplier": _safe_float(recipe.get("idle_multiplier"), float("nan")),
            "calibration_block_count": int(_safe_float(recipe.get("calibration_block_count"), 0.0)),
            "selected_freqs": [float(freq) for freq in recipe.get("selected_freqs", []) or []],
            "frequency_set_id": str(recipe.get("frequency_set_id", "")),
            "channel_weight_mode": None,
            "channel_weights": None,
        },
        "deployment_budget": dict(recipe.get("deployable_budget", {}) or {}),
        "budget_checks": dict(recipe.get("deployable_budget_checks", {}) or {}),
        "budget_pass": bool(recipe.get("deployable_budget_pass", False)),
        "validation_metrics": {
            "mean_idle_fp_per_min": _finite_or_none(recipe.get("mean_idle_fp_per_min")),
            "mean_mixed_idle_fp_per_min": _finite_or_none(recipe.get("mean_mixed_idle_fp_per_min")),
            "mean_control_recall": _finite_or_none(recipe.get("mean_control_recall")),
            "mean_control_recall_at_2.5s": _finite_or_none(recipe.get("mean_control_recall_at_2.5s")),
            "mean_detection_latency_s": _finite_or_none(recipe.get("mean_detection_latency_s")),
            "mean_ns1_fp_per_min": _finite_or_none(recipe.get("mean_ns1_fp_per_min")),
            "mean_ns2_fp_per_min": _finite_or_none(recipe.get("mean_ns2_fp_per_min")),
            "mean_ns3_fp_per_min": _finite_or_none(recipe.get("mean_ns3_fp_per_min")),
            "expected_subject_count": int(_safe_float(recipe.get("expected_subject_count"), 0.0)),
            "coverage_subject_count": int(_safe_float(recipe.get("coverage_subject_count"), 0.0)),
            "split_count": int(_safe_float(recipe.get("split_count"), 0.0)),
        },
        "channel_contract": {
            "contract": "strict_required_8_posterior",
            "project_channel_names": list(PROJECT_POSTERIOR_8_CHANNELS),
            "all_loaded_subjects_match_project_channel_contract": bool(
                dict(channel_compatibility or {}).get("all_loaded_subjects_match_project_channel_contract", False)
            ),
            "only_required_channels_used": True,
        },
        "candidate_artifacts": {
            "matching_row_count": int(len(candidate_rows)),
            "candidate_artifact_count": int(len(unique_candidate_paths)),
            "candidate_artifact_paths": unique_candidate_paths,
        },
        "artifact_paths": dict(artifact_paths),
        "next_steps": [
            "Use this recipe as the short-pretrain default grid for live calibration.",
            "Keep channel weights disabled for the first deployable candidate.",
            "Tune NS2-specific LRT rejection before changing the decoder if no-control FP rises in live data.",
            "Convert per-subject runtime_loadable candidate artifacts to ThresholdProfile JSON before online loading.",
        ],
    }


def _run_metadata_payload(
    *,
    run_id: str,
    datasets: Sequence[str],
    freqs: Sequence[float],
    methods: Sequence[str],
    subjects_expected: int,
    calibration_blocks: Sequence[int],
    window_lengths: Sequence[float],
    score_bank_mode: str,
    classifier_gate_variants: Sequence[str] = DEFAULT_CLASSIFIER_GATE_VARIANTS,
    idle_eval_mode: str,
    timeout_sec: float,
    artifact_paths: Mapping[str, Any],
) -> dict[str, Any]:
    frequency_profile = _frequency_profile_name(freqs)
    return {
        "run_id": str(run_id),
        "dataset": ",".join(str(item) for item in datasets),
        "frequency_profile": frequency_profile,
        "selected_freqs": [float(freq) for freq in freqs],
        "method": ",".join(str(item) for item in methods),
        "subjects_expected": int(subjects_expected),
        "calibration_blocks": [int(block) for block in calibration_blocks],
        "window_length_s": [float(win_sec) for win_sec in window_lengths],
        "score_bank_mode": str(score_bank_mode),
        "classifier_gate_variants": [parse_classifier_gate_variant(item) for item in classifier_gate_variants],
        "idle_eval_mode": str(idle_eval_mode),
        "implementation_scope": "pseudo-online/offline replay only; no runtime profile writeback",
        "candidate_artifacts_only": True,
        "server_writable_root": str(SERVER_SSVEP_WRITABLE_ROOT),
        "server_log_contract": str(SERVER_SSVEP_LOG_ROOT / f"{run_id}.log"),
        "timeout_sec": float(timeout_sec),
        "artifact_paths": dict(artifact_paths),
    }


def _evaluation_contract_payload(
    *,
    datasets: Sequence[str],
    freqs: Sequence[float],
    methods: Sequence[str],
    subjects_expected: int,
    subjects_completed: int,
    calibration_blocks: Sequence[int],
    window_lengths: Sequence[float],
    step_sec: float,
    decision_start_sec: float,
    decision_deadline_sec: float,
    min_release_windows: int,
    reject_gate: str,
    artifact_paths: Mapping[str, Any],
    implementation_level: str,
) -> dict[str, Any]:
    payload = _metric_definitions_payload(
        step_sec=float(step_sec),
        decision_start_sec=float(decision_start_sec),
        decision_deadline_sec=float(decision_deadline_sec),
        min_release_windows=int(min_release_windows),
    )
    payload.update(
        {
            "dataset": ",".join(str(item) for item in datasets),
            "frequency_profile": _frequency_profile_name(freqs),
            "selected_freqs": [float(freq) for freq in freqs],
            "method": ",".join(str(item) for item in methods),
            "subjects_expected": int(subjects_expected),
            "subjects_completed": int(subjects_completed),
            "calibration_blocks": [int(block) for block in calibration_blocks],
            "window_length_s": [float(win_sec) for win_sec in window_lengths],
            "reject_gate": str(reject_gate),
            "artifact_paths": dict(artifact_paths),
            "implementation_level": str(implementation_level),
            "paper_faithful": bool(str(implementation_level) == "paper-faithful"),
            "engineering_approx": bool(str(implementation_level) != "paper-faithful"),
            "trial_block_split_guard": (
                "calibration, validation/threshold fitting, templates, reference banks, and LDA/Ridge fits "
                "are block/trial split; test/holdout blocks are excluded from fitting."
            ),
            "no_control_policy": (
                "4 command frequencies are trained as supervised classes; no-control is produced only by the reject gate."
            ),
        }
    )
    return payload


def _parse_choice(raw: str | None, *, default: str, choices: Sequence[str], label: str) -> str:
    value = str(raw or default).strip().lower()
    if not value:
        value = str(default)
    if value not in set(str(choice) for choice in choices):
        raise ValueError(f"{label} must be one of {','.join(choices)}; got {raw}")
    return value


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
    idle_fp = _first_finite_metric(
        metrics,
        ("mixed_idle_fp_per_min", "real_idle_fp_per_min", "approx_idle_fp_per_min", "idle_fp_per_min"),
        float("inf"),
    )
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


def _deployable_budget_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    expected_subject_count = int(_safe_float(summary.get("expected_subject_count"), 0.0))
    coverage_subject_count = int(
        _safe_float(
            summary.get("coverage_subject_count", summary.get("subject_count", 0)),
            0.0,
        )
    )
    idle_fp = _first_finite_metric(
        summary,
        ("mean_mixed_idle_fp_per_min", "mean_real_idle_fp_per_min", "mean_approx_idle_fp_per_min", "mean_idle_fp_per_min"),
        float("inf"),
    )
    control_recall = _safe_float(summary.get("mean_control_recall"), 0.0)
    control_recall_at_2p5s = _safe_float(summary.get("mean_control_recall_at_2.5s"), 0.0)
    detection_latency_s = _safe_float(summary.get("mean_detection_latency_s"), float("inf"))
    checks = {
        "full_subject_coverage": bool(
            expected_subject_count > 0 and coverage_subject_count == expected_subject_count
        ),
        "idle_fp_budget": bool(idle_fp <= float(DEFAULT_CLASSIFIER_IDLE_FP_BUDGET_PER_MIN) + 1e-12),
        "control_recall_budget": bool(
            control_recall >= float(DEFAULT_CLASSIFIER_THRESHOLD_MIN_CONTROL_RECALL) - 1e-12
        ),
        "control_recall_at_2.5s_budget": bool(
            control_recall_at_2p5s >= float(DEFAULT_DEPLOYABLE_MIN_CONTROL_RECALL_AT_2P5S) - 1e-12
        ),
        "detection_latency_budget": bool(
            detection_latency_s <= float(DEFAULT_DEPLOYABLE_MAX_DETECTION_LATENCY_SEC) + 1e-12
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "deployable_budget_pass": bool(not failed),
        "deployable_budget_failed_reasons": failed,
        "deployable_budget_checks": checks,
        "deployable_budget": {
            "max_idle_fp_per_min": float(DEFAULT_CLASSIFIER_IDLE_FP_BUDGET_PER_MIN),
            "min_control_recall": float(DEFAULT_CLASSIFIER_THRESHOLD_MIN_CONTROL_RECALL),
            "min_control_recall_at_2.5s": float(DEFAULT_DEPLOYABLE_MIN_CONTROL_RECALL_AT_2P5S),
            "max_detection_latency_s": float(DEFAULT_DEPLOYABLE_MAX_DETECTION_LATENCY_SEC),
            "require_full_subject_coverage": True,
        },
    }


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
    if normalized in {
        "balanced_recall_guard",
        CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY,
        CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        CLASSIFIER_SUBJECT_ADAPTIVE_THRESHOLD_POLICY,
    }:
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


def _csv_gate_variant_tuple(raw: str | None) -> tuple[str, ...]:
    values = _csv_str_tuple(raw, default=DEFAULT_CLASSIFIER_GATE_VARIANTS)
    parsed: list[str] = []
    for item in values:
        variant = parse_classifier_gate_variant(item)
        if variant not in parsed:
            parsed.append(variant)
    return tuple(parsed)


def _parse_nc_calibration_source(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace("+", "_").replace(" ", "_")
    aliases = {
        "ns1_only": NC_CALIBRATION_SOURCE_NS1,
        "ns2_only": NC_CALIBRATION_SOURCE_NS2,
        "ns3_only": NC_CALIBRATION_SOURCE_NS3,
        "ns_all": NC_CALIBRATION_SOURCE_MIXED,
        "ns1_ns2_ns3": NC_CALIBRATION_SOURCE_MIXED,
        "ns1_ns2_ns3_mixed": NC_CALIBRATION_SOURCE_MIXED,
        "mixed_all": NC_CALIBRATION_SOURCE_MIXED,
        "ns2heavy": NC_CALIBRATION_SOURCE_NS2_HEAVY,
        "ns2_heavy_mixed": NC_CALIBRATION_SOURCE_NS2_HEAVY,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in NC_CALIBRATION_SOURCES:
        raise ValueError(f"unsupported no-control calibration source: {value!r}")
    return normalized


def _csv_nc_calibration_sources(raw: str | None) -> tuple[str, ...]:
    values: list[str] = []
    for item in _csv_str_tuple(raw, default=NC_CALIBRATION_SOURCES):
        source = _parse_nc_calibration_source(item)
        if source not in values:
            values.append(source)
    return tuple(values)


def _parse_nc_gate_type(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "baseline": NC_GATE_BASELINE_LRT_THRESHOLD,
        "lrt": NC_GATE_BASELINE_LRT_THRESHOLD,
        "nc_lrt": NC_GATE_BASELINE_LRT_THRESHOLD,
        "nc_threshold": NC_GATE_BASELINE_LRT_THRESHOLD,
        "session_logistic": NC_GATE_SESSION_LOGISTIC,
        "csns_logistic": NC_GATE_SESSION_LOGISTIC,
        "logistic": NC_GATE_SESSION_LOGISTIC,
        "conditional": NC_GATE_CONDITIONAL_SESSION_LOGISTIC,
        "conditional_csns": NC_GATE_CONDITIONAL_SESSION_LOGISTIC,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in NC_CALIBRATION_GATE_TYPES:
        raise ValueError(f"unsupported no-control calibration gate type: {value!r}")
    return normalized


def _csv_nc_gate_types(raw: str | None) -> tuple[str, ...]:
    values: list[str] = []
    for item in _csv_str_tuple(raw, default=NC_CALIBRATION_GATE_TYPES):
        gate_type = _parse_nc_gate_type(item)
        if gate_type not in values:
            values.append(gate_type)
    return tuple(values)


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
    values = tuple(
        METHOD_ALIASES.get(str(item).strip().lower(), str(item).strip().lower())
        for item in str(raw or "").split(",")
        if str(item).strip()
    )
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
    return _parse_choice(
        raw,
        default=DEFAULT_CLASSIFIER_THRESHOLD_POLICY,
        choices=CLASSIFIER_THRESHOLD_POLICIES,
        label="classifier threshold policy",
    )


def _parse_score_bank_mode(raw: str | None) -> str:
    return _parse_choice(raw, default=DEFAULT_SCORE_BANK_MODE, choices=SCORE_BANK_MODES, label="score bank mode")


def _parse_freq_search_mode(raw: str | None) -> str:
    return _parse_choice(raw, default=DEFAULT_FREQ_SEARCH_MODE, choices=FREQ_SEARCH_MODES, label="freq search mode")


def _parse_freq_candidate_source(raw: str | None) -> str:
    return _parse_choice(
        raw,
        default=DEFAULT_FREQ_CANDIDATE_SOURCE,
        choices=FREQ_CANDIDATE_SOURCES,
        label="freq candidate source",
    )


def _parse_idle_eval_mode(raw: str | None) -> str:
    return _parse_choice(raw, default=DEFAULT_IDLE_EVAL_MODE, choices=IDLE_EVAL_MODES, label="idle eval mode")


def _score_method_spec(method_name: str) -> ScoreMethodSpec:
    normalized = str(method_name).strip().lower()
    spec = SCORE_METHOD_SPECS.get(normalized)
    if spec is None:
        raise ValueError(f"unsupported score method: {method_name}")
    return spec


def _method_score_bank_skip_reason(method_name: str, score_bank_mode: str) -> str:
    normalized_method = str(method_name).strip().lower()
    if normalized_method not in SCORE_METHOD_SPECS:
        return ""
    mode = _parse_score_bank_mode(score_bank_mode)
    spec = _score_method_spec(normalized_method)
    if mode == "full_reference_bank" and bool(spec.fit_decoder):
        return "unsupported_score_bank_mode"
    return ""


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


def _classifier_recipe_id_with_smoothing(
    *,
    win_sec: float,
    min_enter_windows: int,
    max_gap_windows: int = 0,
    smoothing_windows: int = 1,
    gate_policy: str = CLASSIFIER_CONFIDENCE_GATE_POLICY,
    gate_variant: str = CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
    variant_token: str = "",
) -> str:
    base = _classifier_recipe_id(
        win_sec=float(win_sec),
        min_enter_windows=int(min_enter_windows),
        max_gap_windows=max(0, int(max_gap_windows)),
    )
    if int(smoothing_windows) > 1:
        base = f"{base}_sm{int(smoothing_windows)}"
    normalized_gate = str(gate_policy or CLASSIFIER_CONFIDENCE_GATE_POLICY).strip().lower()
    if normalized_gate == CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY:
        base = f"{base}_aeg"
    elif normalized_gate == CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY:
        base = f"{base}_lrtmw"
    elif normalized_gate == CLASSIFIER_SUBJECT_ADAPTIVE_THRESHOLD_POLICY:
        base = f"{base}_sat"
    resolved_variant = parse_classifier_gate_variant(gate_variant)
    if resolved_variant != CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW:
        token = (str(variant_token).strip() or resolved_variant).replace(".", "p")
        base = f"{base}_{token}"
    return base


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


def _dataset_all_target_freqs(dataset: str) -> tuple[float, ...]:
    normalized = str(dataset).strip().lower()
    if normalized == "beta":
        return tuple(round(8.0 + 0.2 * index, 10) for index in range(40))
    if normalized == "wang2016":
        from ssvep_core.external_wang2016_dataset import WANG2016_TARGET_FREQUENCIES

        return tuple(float(freq) for freq in WANG2016_TARGET_FREQUENCIES)
    if normalized == "ysu_an":
        return tuple(float(freq) for freq in YSUAN_TARGET_FREQUENCIES)
    return tuple(float(freq) for freq in DEFAULT_FRAME_LOCKED_240_FREQS)


def _available_freqs_for_subject(spec: ExternalSubjectSpec, source_metadata: Mapping[str, Any]) -> tuple[float, ...]:
    values = list(source_metadata.get("all_target_frequencies", []) or [])
    if values:
        return tuple(float(freq) for freq in values)
    if str(spec.dataset).strip().lower() == "beta":
        try:
            subject = load_beta_subject(spec.mat_path)
            return tuple(float(freq) for freq in subject.target_frequencies)
        except Exception:
            return _dataset_all_target_freqs("beta")
    return _dataset_all_target_freqs(str(spec.dataset))


def _freq_lookup_key(freq: float) -> float:
    return round(float(freq), 10)


def _freqs_available(freqs: Sequence[float], available_freqs: Sequence[float]) -> bool:
    available = {_freq_lookup_key(freq) for freq in available_freqs}
    return all(_freq_lookup_key(freq) in available for freq in freqs)


def _canonical_freq_tuple(freqs: Sequence[float]) -> tuple[float, float, float, float]:
    values = tuple(float(freq) for freq in freqs)
    if len(values) != 4:
        raise ValueError(f"frequency set must contain exactly 4 values; got {len(values)}")
    if len({_freq_lookup_key(freq) for freq in values}) != 4:
        raise ValueError(f"frequency set must contain 4 unique values; got {values}")
    return values  # type: ignore[return-value]


def _parse_freq_from_label(label: str) -> Optional[float]:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*Hz", str(label))
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _segment_source_freq(item: tuple[TrialSpec, np.ndarray]) -> Optional[float]:
    trial, _segment = item
    if trial.expected_freq is not None:
        return float(trial.expected_freq)
    return _parse_freq_from_label(str(trial.label))


def _relabel_segments_for_command_freqs(
    segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    command_freqs: Sequence[float],
) -> list[tuple[TrialSpec, np.ndarray]]:
    command_map = {_freq_lookup_key(freq): float(freq) for freq in command_freqs}
    relabeled: list[tuple[TrialSpec, np.ndarray]] = []
    for trial, segment in segments:
        source_freq = _segment_source_freq((trial, segment))
        expected_freq = None
        label = str(trial.label)
        if source_freq is not None:
            matched = command_map.get(_freq_lookup_key(source_freq))
            if matched is not None:
                expected_freq = float(matched)
                label = f"{float(matched):g}Hz"
            else:
                label = f"hard_idle_{float(source_freq):g}Hz"
        relabeled.append(
            (
                TrialSpec(
                    label=label,
                    expected_freq=expected_freq,
                    trial_id=int(trial.trial_id),
                    block_index=int(trial.block_index),
                ),
                segment,
            )
        )
    return relabeled


def _filter_segments_by_freqs(
    segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    freqs: Sequence[float],
) -> list[tuple[TrialSpec, np.ndarray]]:
    allowed = {_freq_lookup_key(freq) for freq in freqs}
    return [
        item
        for item in segments
        if (source_freq := _segment_source_freq(item)) is not None and _freq_lookup_key(source_freq) in allowed
    ]


def _control_segments_for_freqs(
    all_target_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    freqs: Sequence[float],
) -> list[tuple[TrialSpec, np.ndarray]]:
    return _relabel_segments_for_command_freqs(
        _filter_segments_by_freqs(all_target_segments, freqs=freqs),
        command_freqs=freqs,
    )


def _build_all_target_segments_for_spec(
    spec: ExternalSubjectSpec,
    *,
    available_freqs: Sequence[float],
) -> list[tuple[TrialSpec, np.ndarray]]:
    candidate = tuple(float(freq) for freq in available_freqs)
    if len(candidate) < 4:
        raise ValueError(f"need at least 4 available frequencies for {spec.dataset}:{spec.subject}")
    command_seed = candidate[:4]
    if str(spec.dataset).strip().lower() == "beta":
        subject = load_beta_subject(spec.mat_path)
        return _relabel_segments_for_command_freqs(
            build_beta_segments(
                subject,
                freqs=command_seed,
                include_hard_idle=True,
                include_pre_stim_idle=False,
            ),
            command_freqs=candidate,
        )
    if str(spec.dataset).strip().lower() == "wang2016":
        if spec.channel_loc_path is None:
            raise ValueError("wang2016 requires channel_loc_path")
        subject = load_wang2016_subject(spec.mat_path, spec.channel_loc_path)
        return _relabel_segments_for_command_freqs(
            build_wang2016_segments(
                subject,
                freqs=command_seed,
                include_hard_idle=True,
                include_pre_stim_idle=False,
            ),
            command_freqs=candidate,
        )
    if str(spec.dataset).strip().lower() == "ysu_an":
        subject = load_ysuan_subject(spec.mat_path, channel_loc_path=spec.channel_loc_path)
        segments = [
            *build_ysuan_cs_segments(subject, freqs=candidate),
            *build_ysuan_ns_segments(subject),
        ]
        return _relabel_segments_for_command_freqs(
            segments,
            command_freqs=candidate,
        )
    raise ValueError(f"unsupported dataset: {spec.dataset}")


def _build_clean_idle_segments_for_spec(
    spec: ExternalSubjectSpec,
    *,
    command_freqs: Sequence[float],
) -> list[tuple[TrialSpec, np.ndarray]]:
    if str(spec.dataset).strip().lower() == "beta":
        subject = load_beta_subject(spec.mat_path)
        return [
            item for item in build_beta_segments(
                subject,
                freqs=command_freqs,
                include_hard_idle=False,
                include_pre_stim_idle=True,
            )
            if item[0].expected_freq is None
        ]
    if str(spec.dataset).strip().lower() == "wang2016":
        if spec.channel_loc_path is None:
            raise ValueError("wang2016 requires channel_loc_path")
        subject = load_wang2016_subject(spec.mat_path, spec.channel_loc_path)
        return [
            item for item in build_wang2016_segments(
                subject,
                freqs=command_freqs,
                include_hard_idle=False,
                include_pre_stim_idle=True,
            )
            if item[0].expected_freq is None
        ]
    if str(spec.dataset).strip().lower() == "ysu_an":
        subject = load_ysuan_subject(spec.mat_path, channel_loc_path=spec.channel_loc_path)
        return build_ysuan_ns_segments(subject)
    raise ValueError(f"unsupported dataset: {spec.dataset}")


def _clean_idle_proxy_support_payload(
    *,
    clean_idle_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    sampling_rate: int,
    win_sec: float,
) -> dict[str, Any]:
    durations = [
        float(np.asarray(segment).shape[0]) / float(max(int(sampling_rate), 1))
        for _trial, segment in clean_idle_segments
        if np.asarray(segment).ndim == 2
    ]
    max_duration = max(durations, default=0.0)
    return {
        "available": bool(clean_idle_segments),
        "supported": bool(clean_idle_segments and max_duration + 1e-9 >= float(win_sec)),
        "segment_count": int(len(clean_idle_segments)),
        "max_segment_duration_sec": float(max_duration),
        "requested_win_sec": float(win_sec),
        "note": (
            "Clean idle proxy uses pre-stimulus baseline segments when they are long enough for the classifier window; "
            "otherwise it is marked unsupported and must not be interpreted as real no-target evidence."
        ),
    }


def _evaluate_clean_idle_proxy_from_cache(
    model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int,
) -> dict[str, Any]:
    if not scored_trials:
        return {
            "supported": False,
            "reason": "no clean idle windows were long enough for this classifier window",
            "idle_fp_per_min": None,
            "idle_trial_fp_rate": None,
            "idle_trials": 0,
        }
    bundle = _evaluate_fbcca_lda5_model(
        model,
        scored_trials,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
    )
    metrics = dict(bundle.get("async_metrics", {}) or {})
    return {
        "supported": True,
        "idle_fp_per_min": _safe_float(metrics.get("idle_fp_per_min"), 0.0),
        "idle_selected_windows_per_min": _safe_float(metrics.get("idle_selected_windows_per_min"), 0.0),
        "idle_trial_fp_rate": _safe_float(metrics.get("idle_trial_fp_rate"), 0.0),
        "idle_trials": int(_safe_float(metrics.get("idle_trials"), 0.0)),
        "idle_fp_trials": int(_safe_float(metrics.get("idle_fp_trials"), 0.0)),
    }


def _ysuan_ns_subtype_from_label(label: str) -> Optional[str]:
    text = str(label).strip().lower()
    match = re.search(r"(?:^|[_-])(ns[123])(?:[_-]|$)", text)
    if match:
        return str(match.group(1))
    return None


def _evaluate_no_control_subtypes_from_cache(
    model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int,
) -> dict[str, Any]:
    grouped: dict[str, list[ScoredTrial]] = defaultdict(list)
    for item in scored_trials:
        subtype = _ysuan_ns_subtype_from_label(str(item.trial.label))
        if subtype:
            grouped[subtype].append(item)
    subtype_metrics: dict[str, dict[str, Any]] = {}
    fp_values: list[float] = []
    for subtype in ("ns1", "ns2", "ns3"):
        metrics = _evaluate_clean_idle_proxy_from_cache(
            model,
            grouped.get(subtype, []),
            win_sec=float(win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
        )
        subtype_metrics[subtype] = metrics
        if bool(metrics.get("supported", False)):
            fp_values.append(_safe_float(metrics.get("idle_fp_per_min"), 0.0))
    pooled = _evaluate_clean_idle_proxy_from_cache(
        model,
        scored_trials,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
    )
    return {
        "supported": bool(pooled.get("supported", False)),
        "ns1": subtype_metrics["ns1"],
        "ns2": subtype_metrics["ns2"],
        "ns3": subtype_metrics["ns3"],
        "ns_all_fp_per_min": _safe_float(pooled.get("idle_fp_per_min"), float("nan")),
        "ns_all_trial_fp_rate": _safe_float(pooled.get("idle_trial_fp_rate"), float("nan")),
        "pooled": pooled,
    }


def _no_control_subtype_by_frequency_from_cache(
    model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
    step_sec: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {
        _freq_label(freq): {
            subtype: {"fp_windows": 0, "duration_sec": 0.0, "fp_per_min": None}
            for subtype in ("ns1", "ns2", "ns3")
        }
        for freq in model.freqs
    }
    cache = _build_classifier_probability_cache(model, scored_trials)
    for item, probs, labels in cache:
        subtype = _ysuan_ns_subtype_from_label(str(item.trial.label))
        if subtype not in {"ns1", "ns2", "ns3"}:
            continue
        probs = _smooth_classifier_probabilities(probs, smoothing_windows=int(getattr(model, "smoothing_windows", 1)))
        lrt_evidence = (
            _lrt_window_evidence_from_features(model, item.feature_matrix)
            if str(getattr(model, "gate_policy", "")) == CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY
            else np.zeros(int(probs.shape[0]), dtype=np.float64)
        )
        label_values = np.asarray(labels, dtype=object)
        idle_index = int(np.where(label_values == "idle")[0][0])
        pred_indices = np.argmax(probs, axis=1)
        if str(getattr(model, "gate_policy", "")) == CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY:
            floor_th = getattr(model, "lrt_window_floor_th", None)
            effective_window_th = max(
                float(getattr(model, "lrt_window_th", 0.0)),
                float(floor_th) if floor_th is not None else float(getattr(model, "lrt_window_th", 0.0)),
            )
            gate_mask = (lrt_evidence >= effective_window_th) & _score_shape_gate_mask_for_model(model, item.feature_matrix)
            if getattr(model, "frequency_specific_control_state_gates", None):
                gate_mask &= _frequency_specific_gate_mask_for_model(
                    model,
                    probs=probs,
                    labels=labels,
                    feature_matrix=item.feature_matrix,
                    lrt_evidence=lrt_evidence,
                )
        else:
            gate_mask = (1.0 - probs[:, idle_index]) >= float(getattr(model, "command_confidence_th", 0.0))
        for freq in model.freqs:
            freq_key = _freq_label(freq)
            result[freq_key][subtype]["duration_sec"] = float(result[freq_key][subtype]["duration_sec"]) + float(item.duration_sec)
        for index, pred_index in enumerate(pred_indices):
            if int(pred_index) == idle_index or not bool(gate_mask[index]):
                continue
            freq_key = _label_to_freq_key(str(label_values[int(pred_index)]))
            if freq_key in result:
                result[freq_key][subtype]["fp_windows"] = int(result[freq_key][subtype]["fp_windows"]) + 1
    for freq_payload in result.values():
        for subtype_payload in freq_payload.values():
            minutes = float(subtype_payload["duration_sec"]) / 60.0
            subtype_payload["fp_per_min"] = (
                float(int(subtype_payload["fp_windows"]) / minutes)
                if minutes > 1e-12
                else None
            )
    return result


def _ysuan_holdout_no_control_scored(
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
) -> tuple[list[ScoredTrial], dict[str, Any]]:
    ns_scored = [
        item
        for item in scored_trials
        if item.trial.expected_freq is None and _ysuan_ns_subtype_from_label(str(item.trial.label))
    ]
    durations = [float(item.duration_sec) for item in ns_scored]
    max_duration = max(durations, default=0.0)
    return ns_scored, {
        "available": bool(ns_scored),
        "supported": bool(ns_scored),
        "segment_count": int(len(ns_scored)),
        "max_segment_duration_sec": float(max_duration),
        "requested_win_sec": float(win_sec),
        "note": "YSU-an no-control metrics use holdout NS1/NS2/NS3 trials only.",
    }


def _candidate_freqs_for_source(*, source: str, datasets: Sequence[str]) -> tuple[float, ...]:
    normalized = _parse_freq_candidate_source(source)
    if normalized == "frame_locked_240":
        return tuple(float(freq) for freq in DEFAULT_FRAME_LOCKED_240_FREQS)
    if normalized == "beta_all40":
        return _dataset_all_target_freqs("beta")
    if normalized == "wang_all40":
        return _dataset_all_target_freqs("wang2016")
    if normalized == "ysu_an_all8":
        return _dataset_all_target_freqs("ysu_an")
    return _dataset_all_target_freqs(str(next(iter(datasets), "beta")))


def _full_bank_freqs_for_dataset(*, dataset: str, score_bank_mode: str, fallback_freqs: Sequence[float]) -> tuple[float, ...]:
    if _parse_score_bank_mode(score_bank_mode) != "full_reference_bank":
        return tuple(float(freq) for freq in fallback_freqs)
    return _dataset_all_target_freqs(dataset)


def _frequency_search_plan(*, mode: str, candidate_source: str, datasets: Sequence[str]) -> dict[str, Any]:
    search_mode = _parse_freq_search_mode(mode)
    source = _parse_freq_candidate_source(candidate_source)
    candidate_freqs = _candidate_freqs_for_source(source=source, datasets=datasets)
    shared_sets: list[list[float]] = []
    if search_mode in {"shared_fixed4", "both"}:
        shared_sets = [[float(freq) for freq in combo] for combo in combinations(candidate_freqs, 4)]
    return {
        "frequency_selection_mode": search_mode,
        "freq_candidate_source": source,
        "candidate_freqs": [float(freq) for freq in candidate_freqs],
        "shared_candidate_set_count": int(len(shared_sets)),
        "shared_candidate_sets_preview": shared_sets[:20],
        "personalized_upper_bound_enabled": bool(search_mode in {"personalized_upper_bound", "both"}),
    }


def _shared_frequency_sets_for_plan(freq_plan: Mapping[str, Any], *, fallback_freqs: Sequence[float]) -> tuple[tuple[float, float, float, float], ...]:
    mode = _parse_freq_search_mode(str(freq_plan.get("frequency_selection_mode", DEFAULT_FREQ_SEARCH_MODE)))
    if mode not in {"shared_fixed4", "both"}:
        return (_canonical_freq_tuple(fallback_freqs),)
    raw_sets = list(freq_plan.get("shared_candidate_sets_preview", []) or [])
    if int(freq_plan.get("shared_candidate_set_count", 0) or 0) > len(raw_sets):
        candidate_freqs = tuple(float(freq) for freq in freq_plan.get("candidate_freqs", []) or [])
        raw_sets = [list(combo) for combo in combinations(candidate_freqs, 4)]
    if not raw_sets:
        raw_sets = [list(fallback_freqs)]
    return tuple(_canonical_freq_tuple(freq_set) for freq_set in raw_sets)


def _frequency_set_id(mode: str, freqs: Sequence[float]) -> str:
    return f"{str(mode).strip().lower()}_{_freq_token(freqs)}"


def _frequency_case_payload(case: FrequencyEvalCase) -> dict[str, Any]:
    return {
        "frequency_selection_mode": str(case.mode),
        "frequency_set_id": str(case.frequency_set_id),
        "selected_freqs": [float(freq) for freq in case.freqs],
        "candidate_freqs": [float(freq) for freq in case.candidate_freqs],
        "personalized_candidate_count": int(case.personalized_candidate_count),
        "selected_by_calibration": bool(case.selected_by_calibration),
    }


def _candidate_freqs_for_subject(
    *,
    candidate_freqs: Sequence[float],
    available_freqs: Sequence[float],
    count: int,
) -> tuple[float, ...]:
    available_by_key = {_freq_lookup_key(freq): float(freq) for freq in available_freqs}
    selected: list[float] = []
    for freq in candidate_freqs:
        key = _freq_lookup_key(freq)
        if key not in available_by_key:
            continue
        if key in {_freq_lookup_key(item) for item in selected}:
            continue
        selected.append(float(available_by_key[key]))
        if int(count) > 0 and len(selected) >= int(count):
            break
    if len(selected) < max(4, int(count or 0)):
        for freq in available_freqs:
            key = _freq_lookup_key(freq)
            if key in {_freq_lookup_key(item) for item in selected}:
                continue
            selected.append(float(freq))
            if int(count) > 0 and len(selected) >= int(count):
                break
    return tuple(float(freq) for freq in selected)


def _score_personalized_frequency_candidates(
    *,
    all_target_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    candidate_freqs: Sequence[float],
    calibration_blocks: Sequence[int],
    sampling_rate: int,
    win_sec: float,
    max_supported_win_sec: float,
    step_sec: float,
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    candidate_tuple = tuple(float(freq) for freq in candidate_freqs)
    if len(candidate_tuple) < 4:
        raise ValueError("personalized frequency selection needs at least 4 candidate frequencies")
    calibration_block_set = {int(block) for block in calibration_blocks}
    calibration_segments = [
        item for item in all_target_segments if int(item[0].block_index) in calibration_block_set
    ]
    filtered = _filter_segments_by_freqs(calibration_segments, freqs=candidate_tuple)
    if not filtered:
        raise ValueError("personalized frequency selection found no calibration candidate segments")
    selection_win_sec = min(float(win_sec), float(max_supported_win_sec))
    if selection_win_sec <= 0.0:
        raise ValueError("personalized frequency selection needs a positive supported window")
    decoder = _build_fbcca_decoder_for_scoring(
        freqs=candidate_tuple,
        sampling_rate=int(sampling_rate),
        win_sec=float(selection_win_sec),
        step_sec=float(step_sec),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
    )

    def score_windows(windows: np.ndarray) -> np.ndarray:
        if hasattr(decoder, "score_windows_batch"):
            return np.asarray(decoder.score_windows_batch(windows), dtype=np.float64)
        if hasattr(decoder, "analyze_windows_batch"):
            payloads = decoder.analyze_windows_batch(windows)
            return np.vstack(
                [np.asarray(dict(item)["scores"], dtype=np.float64).reshape(1, -1) for item in payloads]
            )
        rows: list[np.ndarray] = []
        for window in windows:
            if hasattr(decoder, "score_window"):
                row = np.asarray(decoder.score_window(window), dtype=np.float64)
            elif hasattr(decoder, "analyze_window"):
                row = np.asarray(dict(decoder.analyze_window(window))["scores"], dtype=np.float64)
            else:
                raise TypeError("decoder does not expose a supported scoring API")
            rows.append(row.reshape(1, -1))
        return np.vstack(rows)

    rows: dict[float, list[dict[str, float]]] = {float(freq): [] for freq in candidate_tuple}
    for trial, segment in filtered:
        source_freq = _segment_source_freq((trial, segment))
        if source_freq is None:
            continue
        canonical = None
        for freq in candidate_tuple:
            if _freq_lookup_key(freq) == _freq_lookup_key(source_freq):
                canonical = float(freq)
                break
        if canonical is None:
            continue
        matrix = np.ascontiguousarray(np.asarray(segment, dtype=np.float64))
        if matrix.ndim != 2 or matrix.shape[0] < int(decoder.win_samples):
            continue
        windows = extract_window_batch(
            matrix,
            win_samples=int(decoder.win_samples),
            step_samples=int(decoder.step_samples),
        )
        scores = score_windows(windows)
        freq_index = int([_freq_lookup_key(freq) for freq in candidate_tuple].index(_freq_lookup_key(canonical)))
        own_scores = scores[:, freq_index]
        other_scores = np.delete(scores, freq_index, axis=1)
        top_other = np.max(other_scores, axis=1) if other_scores.size else np.zeros_like(own_scores)
        pred = np.argmax(scores, axis=1)
        rows[canonical].append(
            {
                "self_score_median": float(np.median(own_scores)),
                "margin_median": float(np.median(own_scores - top_other)),
                "top1_rate": float(np.mean(pred == freq_index)),
                "window_count": float(scores.shape[0]),
            }
        )
    summaries: list[dict[str, Any]] = []
    for freq in candidate_tuple:
        stats = rows.get(float(freq), [])
        if not stats:
            summaries.append(
                {
                    "freq": float(freq),
                    "self_score_median": 0.0,
                    "margin_median": -float("inf"),
                    "top1_rate": 0.0,
                    "trial_count": 0,
                }
            )
            continue
        summaries.append(
            {
                "freq": float(freq),
                "self_score_median": float(np.mean([item["self_score_median"] for item in stats])),
                "margin_median": float(np.mean([item["margin_median"] for item in stats])),
                "top1_rate": float(np.mean([item["top1_rate"] for item in stats])),
                "trial_count": int(len(stats)),
            }
        )
    ranked = sorted(
        summaries,
        key=lambda row: (
            -_safe_float(row.get("top1_rate"), 0.0),
            -_safe_float(row.get("margin_median"), -float("inf")),
            -_safe_float(row.get("self_score_median"), 0.0),
            _safe_float(row.get("freq"), 99.0),
        ),
    )
    selected = tuple(float(row["freq"]) for row in ranked[:4])
    return _canonical_freq_tuple(selected), {
        "selection_policy": "calibration_only_fbcca_self_top1_margin",
        "selection_win_sec": float(selection_win_sec),
        "candidate_freqs": [float(freq) for freq in candidate_tuple],
        "selected_freqs": [float(freq) for freq in selected],
        "ranked_candidates": ranked,
    }


def _row_with_frequency_case(row: dict[str, Any], case: FrequencyEvalCase) -> dict[str, Any]:
    payload = dict(row)
    freq_payload = _frequency_case_payload(case)
    payload["selected_freqs"] = [float(freq) for freq in case.freqs]
    payload["frequency_selection_mode"] = str(case.mode)
    payload["frequency_set_id"] = str(case.frequency_set_id)
    payload["frequency_case"] = freq_payload
    split_summary = dict(payload.get("split_summary", {}) or {})
    split_summary["selected_freqs"] = [float(freq) for freq in case.freqs]
    split_summary["frequency_selection_mode"] = str(case.mode)
    split_summary["frequency_set_id"] = str(case.frequency_set_id)
    payload["split_summary"] = split_summary
    return payload


def _estimate_pretrain_duration_sec(
    *,
    freq_selection_mode: str,
    command_freq_count: int = 4,
    command_repeats: int = 3,
    command_trial_sec: float = 6.0,
    clean_idle_sec: float = 30.0,
    personalized_candidate_count: int = 0,
    personalized_screen_trial_sec: float = 4.0,
) -> float:
    mode = _parse_freq_search_mode(freq_selection_mode)
    base = float(command_freq_count) * float(command_repeats) * float(command_trial_sec) + float(clean_idle_sec)
    if mode in {"personalized_upper_bound", "both"}:
        base += float(max(0, personalized_candidate_count)) * float(personalized_screen_trial_sec)
    return float(base)


def _budget_payload(*, freq_selection_mode: str, pretrain_budget_sec: float, personalized_candidate_count: int) -> dict[str, Any]:
    estimated = _estimate_pretrain_duration_sec(
        freq_selection_mode=freq_selection_mode,
        personalized_candidate_count=int(personalized_candidate_count),
    )
    return {
        "pretrain_budget_sec": float(pretrain_budget_sec),
        "estimated_pretrain_duration_sec": float(estimated),
        "pretrain_budget_pass": bool(float(estimated) <= float(pretrain_budget_sec) + 1e-9),
        "duration_model": "4 commands * 3 repeats * 6s + 30s clean idle; personalized adds candidate_count * 4s",
    }


def _weak_subject_audit(summary: Mapping[str, Any], *, weak_subjects: Sequence[str] = WEAK_SUBJECT_AUDIT_SUBJECTS) -> dict[str, Any]:
    rows = list(summary.get("subjects", []) or []) if isinstance(summary, Mapping) else []
    weak_set = {str(item).upper() for item in weak_subjects}
    compact = []
    for row in rows:
        subject = str(row.get("subject", "")).upper()
        if subject not in weak_set:
            continue
        compact.append(
            {
                "dataset": row.get("dataset"),
                "subject": row.get("subject"),
                "mean_control_recall": row.get("mean_control_recall"),
                "mean_idle_fp_per_min": row.get("mean_idle_fp_per_min"),
                "mean_async_macro_f1_5class": row.get("mean_async_macro_f1_5class"),
                "mean_detection_latency_s": row.get("mean_detection_latency_s"),
            }
        )
    base_rows = [
        {
            "dataset": row.get("dataset"),
            "subject": row.get("subject"),
            "mean_control_recall": row.get("mean_control_recall"),
            "mean_idle_fp_per_min": row.get("mean_idle_fp_per_min"),
            "mean_async_macro_f1_5class": row.get("mean_async_macro_f1_5class"),
            "mean_detection_latency_s": row.get("mean_detection_latency_s"),
        }
        for row in rows
    ]
    return {
        "tracked_weak_subjects": compact,
        "weakest_recall_subjects": sorted(
            base_rows,
            key=lambda row: _safe_float(row.get("mean_control_recall"), 0.0),
        )[:10],
        "highest_idle_fp_subjects": sorted(
            base_rows,
            key=lambda row: _safe_float(row.get("mean_idle_fp_per_min"), 0.0),
            reverse=True,
        )[:10],
    }


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
    if str(dataset) == "ysu_an":
        return tuple(str(name) for name in YSUAN_REQUIRED_CHANNELS)
    raise ValueError(f"unsupported dataset: {dataset}")


def _normalized_channel_names(names: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(name).strip().lower() for name in names)


def _channel_compatibility_payload(dataset: str, source_metadata: Mapping[str, Any]) -> dict[str, Any]:
    required = list(_required_channel_names(dataset))
    selected = [
        str(name)
        for name in list(source_metadata.get("selected_channel_names", []) or [])
    ]
    selected_indices = [
        int(index)
        for index in list(source_metadata.get("selected_channel_indices_zero_based", []) or [])
        if isinstance(index, (int, float, np.integer, np.floating))
    ]
    required_norm = _normalized_channel_names(required)
    selected_norm = _normalized_channel_names(selected)
    project_norm = _normalized_channel_names(PROJECT_POSTERIOR_8_CHANNELS)
    dataset_matches_required = bool(selected_norm == required_norm)
    matches_project = bool(dataset_matches_required and required_norm == project_norm)
    only_required = bool(
        source_metadata.get("only_required_channels_used", False)
        or source_metadata.get("selected_channel_policy") == "strict_required_8_channels_only"
        or dataset_matches_required
    )
    return {
        "channel_contract": "strict_required_8_posterior",
        "project_channel_names": list(PROJECT_POSTERIOR_8_CHANNELS),
        "dataset_required_channel_names": required,
        "dataset_selected_channel_names": selected,
        "selected_channel_indices_zero_based": selected_indices,
        "selected_channel_indices_one_based": [int(index) + 1 for index in selected_indices],
        "source_channel_count": int(source_metadata.get("all_channel_count", len(selected)) or len(selected)),
        "selected_channel_count": int(len(selected)),
        "only_required_channels_used": only_required,
        "dataset_matches_required_order": dataset_matches_required,
        "matches_project_channel_contract": matches_project,
        "online_mapping_assumption": (
            "Deployed BrainFlow profiles store numeric board eeg_channels; they are compatible only if "
            "those board channels are wired to Oz,O1,O2,PO3,POz,PO7,PO8,PO4 in this order."
        ),
    }


def _channel_compatibility_summary(subject_manifest: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    subjects = list(subject_manifest)
    mismatches: list[dict[str, Any]] = []
    for item in subjects:
        payload = dict(item.get("channel_compatibility", {}) or {})
        if bool(payload.get("matches_project_channel_contract", False)):
            continue
        mismatches.append(
            {
                "dataset": str(item.get("dataset", "")),
                "subject": str(item.get("subject", "")),
                "dataset_selected_channel_names": list(payload.get("dataset_selected_channel_names", []) or []),
                "dataset_required_channel_names": list(payload.get("dataset_required_channel_names", []) or []),
            }
        )
    loaded_count = int(len(subjects))
    return {
        "channel_contract": "strict_required_8_posterior",
        "project_channel_names": list(PROJECT_POSTERIOR_8_CHANNELS),
        "loaded_subject_count": loaded_count,
        "all_loaded_subjects_match_project_channel_contract": bool(loaded_count > 0 and not mismatches),
        "mismatched_subjects": mismatches,
        "note": (
            "External datasets are evaluated after selecting only the posterior 8-channel subset. "
            "This validates the algorithmic channel contract, not arbitrary 63/64-channel full-head input."
        ),
    }


def enumerate_external_subjects(
    *,
    datasets: Sequence[str],
    freqs: Sequence[float],
    wang_raw_dir: Path,
    wang_channels_loc: Path,
    beta_raw_dir: Path,
    ysu_an_raw_dir: Optional[Path] = None,
    ysu_an_channel_loc: Optional[Path] = None,
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
    if "ysu_an" in datasets:
        if ysu_an_raw_dir is None:
            raise ValueError("ysu_an requires ysu_an_raw_dir")
        raw_dir = Path(ysu_an_raw_dir).expanduser().resolve()
        subject_paths: dict[str, Path] = {}
        for path in sorted(raw_dir.glob("S*")):
            if not path.is_dir() and path.suffix.lower() != ".mat":
                continue
            match = re.search(r"S\d{2}", path.stem.upper())
            if not match:
                continue
            subject_key = match.group(0)
            if subject_key not in subject_paths or path.is_dir():
                subject_paths[subject_key] = path
        paths = [subject_paths[key] for key in sorted(subject_paths)]
        filtered_paths = [path for path in paths if _subject_allowed("ysu_an", path.stem.upper(), subject_whitelist)]
        if dataset_limit > 0:
            filtered_paths = filtered_paths[:dataset_limit]
        for path in filtered_paths:
            rows.append(
                ExternalSubjectSpec(
                    dataset="ysu_an",
                    subject=path.stem.upper(),
                    mat_path=path,
                    channel_loc_path=Path(ysu_an_channel_loc).expanduser().resolve() if ysu_an_channel_loc else None,
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
            "all_channel_count": int(len(subject.channel_names)),
            "selected_channel_policy": "strict_required_8_channels_only",
            "only_required_channels_used": True,
            "target_index_by_freq": {f"{float(freq):g}": int(index) for freq, index in target_index.items()},
            "all_target_frequencies": [float(freq) for freq in WANG2016_TARGET_FREQUENCIES],
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
            "all_channel_count": int(len(subject.channel_names)),
            "selected_channel_policy": "strict_required_8_channels_only",
            "only_required_channels_used": True,
            "target_index_by_freq": {f"{float(freq):g}": int(index) for freq, index in target_index.items()},
            "all_target_frequencies": [float(freq) for freq in subject.target_frequencies],
            "idle_proxy_note": (
                "Idle/no-control is proxied with non-command target stimulus trials from the external benchmark."
            ),
        }
        return int(subject.sampling_rate), segments, metadata
    if spec.dataset == "ysu_an":
        subject = load_ysuan_subject(spec.mat_path, channel_loc_path=spec.channel_loc_path)
        resolved_freqs, target_index = resolve_ysuan_command_frequencies(freqs)
        segments = build_ysuan_segments(
            subject,
            freqs=resolved_freqs,
            include_ns_idle=True,
        )
        metadata = {
            "dataset": "ysu_an",
            "subject": subject.subject,
            "mat_path": str(subject.root_path),
            "channel_loc_path": "" if spec.channel_loc_path is None else str(spec.channel_loc_path),
            "sampling_rate": int(subject.sampling_rate),
            "raw_sampling_rate": int(subject.raw_sampling_rate),
            "required_channel_names": list(YSUAN_REQUIRED_CHANNELS),
            "selected_channel_names": list(subject.selected_channel_names),
            "selected_channel_indices_zero_based": list(subject.selected_channel_indices),
            "all_channel_count": int(len(subject.channel_names)),
            "selected_channel_policy": "strict_required_8_channels_only",
            "only_required_channels_used": True,
            "target_index_by_freq": {f"{float(freq):g}": int(index) for freq, index in target_index.items()},
            "all_target_frequencies": [float(freq) for freq in YSUAN_TARGET_FREQUENCIES],
            "no_control_subtypes": ["ns1", "ns2", "ns3"],
            "idle_proxy_note": (
                "YSU-an idle/no-control uses explicit NS1/NS2/NS3 trials, not non-command target proxy."
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


def select_ysuan_split_segments(
    segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    freqs: Sequence[float],
    calibration_blocks: Sequence[int],
    holdout_blocks: Sequence[int],
    idle_multiplier: float,
    seed: int,
    ns_calibration_trials_per_subtype: int = YSUAN_DEFAULT_NS_CALIBRATION_TRIALS_PER_SUBTYPE,
) -> tuple[list[tuple[TrialSpec, np.ndarray]], list[tuple[TrialSpec, np.ndarray]], dict[str, Any]]:
    calibration_block_set = {int(block) for block in calibration_blocks}
    holdout_block_set = {int(block) for block in holdout_blocks}
    calibration_control = [
        item
        for item in segments
        if int(item[0].block_index) in calibration_block_set and item[0].expected_freq is not None
    ]
    holdout_control = [
        item
        for item in segments
        if int(item[0].block_index) in holdout_block_set and item[0].expected_freq is not None
    ]
    ns_groups: dict[str, list[tuple[TrialSpec, np.ndarray]]] = defaultdict(list)
    for item in segments:
        trial, _segment = item
        if trial.expected_freq is not None:
            continue
        subtype = _ysuan_ns_subtype_from_label(str(trial.label))
        if subtype:
            ns_groups[subtype].append(item)
    calibration_idle_pool: list[tuple[TrialSpec, np.ndarray]] = []
    holdout_idle: list[tuple[TrialSpec, np.ndarray]] = []
    ns_cal_count = max(0, int(ns_calibration_trials_per_subtype))
    for subtype in ("ns1", "ns2", "ns3"):
        group = sorted(ns_groups.get(subtype, []), key=_trial_sort_key)
        calibration_idle_pool.extend(group[:ns_cal_count])
        holdout_idle.extend(group[ns_cal_count:])
    control_count = len(calibration_control)
    if control_count <= 0:
        raise ValueError("YSU-an calibration split produced no CS control trials")
    holdout_segments = sorted([*holdout_control, *holdout_idle], key=_trial_sort_key)
    if not holdout_segments or not holdout_control:
        raise ValueError("YSU-an holdout split produced no CS holdout trials")
    idle_budget = int(round(float(control_count) * max(float(idle_multiplier), 0.0)))
    if idle_budget > len(calibration_idle_pool):
        idle_budget = len(calibration_idle_pool)
    rng = random.Random(int(seed))
    sampled_idle = list(calibration_idle_pool)
    if idle_budget < len(sampled_idle):
        sampled_idle = rng.sample(sampled_idle, idle_budget)
    calibration_segments = sorted([*calibration_control, *sampled_idle], key=_trial_sort_key)
    subtype_counts = {
        subtype: int(
            sum(1 for item in sampled_idle if _ysuan_ns_subtype_from_label(str(item[0].label)) == subtype)
        )
        for subtype in ("ns1", "ns2", "ns3")
    }
    holdout_subtype_counts = {
        subtype: int(
            sum(1 for item in holdout_idle if _ysuan_ns_subtype_from_label(str(item[0].label)) == subtype)
        )
        for subtype in ("ns1", "ns2", "ns3")
    }
    summary = {
        "seed": int(seed),
        "idle_multiplier": float(idle_multiplier),
        "idle_pool_count": int(len(calibration_idle_pool)),
        "idle_selected_count": int(len(sampled_idle)),
        "ysu_an_ns_calibration_trials_per_subtype": int(ns_cal_count),
        "ysu_an_ns_calibration_counts": subtype_counts,
        "ysu_an_ns_holdout_counts": holdout_subtype_counts,
        "calibration_blocks": [int(block) for block in calibration_blocks],
        "holdout_blocks": [int(block) for block in holdout_blocks],
        "calibration_counts": _count_segments(calibration_segments, freqs),
        "holdout_counts": _count_segments(holdout_segments, freqs),
    }
    return calibration_segments, holdout_segments, summary


def _ysuan_ns_calibration_pool_from_segments(
    segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    ns_calibration_trials_per_subtype: Optional[int] = YSUAN_DEFAULT_NS_CALIBRATION_TRIALS_PER_SUBTYPE,
) -> dict[str, list[tuple[TrialSpec, np.ndarray]]]:
    ns_groups: dict[str, list[tuple[TrialSpec, np.ndarray]]] = defaultdict(list)
    for item in segments:
        trial, _segment = item
        if trial.expected_freq is not None:
            continue
        subtype = _ysuan_ns_subtype_from_label(str(trial.label))
        if subtype:
            ns_groups[subtype].append(item)
    ns_cal_count = (
        None
        if ns_calibration_trials_per_subtype is None
        else max(0, int(ns_calibration_trials_per_subtype))
    )
    return {
        subtype: (
            sorted(list(ns_groups.get(subtype, [])), key=_trial_sort_key)
            if ns_cal_count is None
            else sorted(list(ns_groups.get(subtype, [])), key=_trial_sort_key)[:ns_cal_count]
        )
        for subtype in ("ns1", "ns2", "ns3")
    }


def _select_nc_calibration_segments(
    pool_by_subtype: Mapping[str, Sequence[tuple[TrialSpec, np.ndarray]]],
    *,
    source: str,
    seconds: float,
    sampling_rate: int,
) -> tuple[list[tuple[TrialSpec, np.ndarray]], dict[str, Any]]:
    parsed_source = _parse_nc_calibration_source(source)
    budget_sec = max(0.0, float(seconds))
    if budget_sec <= 1e-12:
        return [], {
            "source": parsed_source,
            "requested_seconds": float(budget_sec),
            "selected_seconds": 0.0,
            "trial_ids": [],
            "segment_ids": [],
            "counts": {"ns1": 0, "ns2": 0, "ns3": 0},
            "selection_policy": "zero_seconds_baseline_no_extra_no_control",
            "fit_split": "calibration_no_control_pool",
            "test_split": "holdout_blocks",
        }

    subtype_weights = {
        NC_CALIBRATION_SOURCE_NS1: {"ns1": 1.0},
        NC_CALIBRATION_SOURCE_NS2: {"ns2": 1.0},
        NC_CALIBRATION_SOURCE_NS3: {"ns3": 1.0},
        NC_CALIBRATION_SOURCE_MIXED: {"ns1": 1.0, "ns2": 1.0, "ns3": 1.0},
        NC_CALIBRATION_SOURCE_NS2_HEAVY: {"ns1": 1.0, "ns2": 2.0, "ns3": 1.0},
    }[parsed_source]
    selected: list[tuple[TrialSpec, np.ndarray]] = []
    selected_keys: set[tuple[str, int, int]] = set()
    elapsed = 0.0
    counts = {"ns1": 0, "ns2": 0, "ns3": 0}
    fs = max(1, int(sampling_rate))

    def add_item(item: tuple[TrialSpec, np.ndarray]) -> None:
        nonlocal elapsed
        trial, segment = item
        subtype = _ysuan_ns_subtype_from_label(str(trial.label)) or ""
        key = (subtype, int(trial.trial_id), int(trial.block_index))
        if subtype not in counts or key in selected_keys:
            return
        selected.append(item)
        selected_keys.add(key)
        counts[subtype] += 1
        elapsed += float(np.asarray(segment).shape[0]) / float(fs)

    indices = {subtype: 0 for subtype in ("ns1", "ns2", "ns3")}
    schedule: list[str] = []
    for subtype, weight in subtype_weights.items():
        schedule.extend([subtype] * max(1, int(round(float(weight)))))
    if not schedule:
        schedule = list(subtype_weights)
    while elapsed < budget_sec - 1e-12:
        made_progress = False
        for subtype in schedule:
            if elapsed >= budget_sec - 1e-12:
                break
            group = list(pool_by_subtype.get(subtype, []) or [])
            idx = int(indices.get(subtype, 0))
            if idx >= len(group):
                continue
            add_item(group[idx])
            indices[subtype] = idx + 1
            made_progress = True
        if not made_progress:
            break
    return sorted(selected, key=_trial_sort_key), {
        "source": parsed_source,
        "requested_seconds": float(budget_sec),
        "selected_seconds": float(elapsed),
        "trial_ids": [int(item[0].trial_id) for item in selected],
        "segment_ids": [
            f"{_ysuan_ns_subtype_from_label(str(item[0].label))}:trial{int(item[0].trial_id)}:block{int(item[0].block_index)}"
            for item in selected
        ],
        "counts": counts,
        "available_counts": {
            subtype: int(len(list(pool_by_subtype.get(subtype, []) or [])))
            for subtype in ("ns1", "ns2", "ns3")
        },
        "selection_policy": (
            "calibration_only_whole_trials_round_robin_ns2_weighted"
            if parsed_source == NC_CALIBRATION_SOURCE_NS2_HEAVY
            else "calibration_only_whole_trials_round_robin"
        ),
        "fit_split": "calibration_no_control_pool",
        "test_split": "holdout_blocks",
    }


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


def _full_reference_bank_features(
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


def _score_matrices_to_features(
    *,
    command_score_matrix: np.ndarray,
    command_freqs: Sequence[float],
    score_bank_mode: str = DEFAULT_SCORE_BANK_MODE,
    all_score_matrix: Optional[np.ndarray] = None,
    all_freqs: Sequence[float] = (),
) -> np.ndarray:
    return runtime_score_matrices_to_features(
        command_score_matrix=command_score_matrix,
        command_freqs=command_freqs,
        all_freqs=all_freqs,
        score_bank_mode=score_bank_mode,
        all_score_matrix=all_score_matrix,
    )


def _classifier_feature_names(
    freqs: Sequence[float],
    *,
    score_source_name: str = "fbcca",
    score_bank_mode: str = DEFAULT_SCORE_BANK_MODE,
) -> list[str]:
    return runtime_classifier_feature_names(
        freqs,
        score_source_name=score_source_name,
        score_bank_mode=score_bank_mode,
    )


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
    freqs: Sequence[float] = DEFAULT_FREQS,
    score_bank_mode: str = DEFAULT_SCORE_BANK_MODE,
    full_bank_decoder: Any = None,
    full_bank_freqs: Sequence[float] = (),
) -> list[ScoredTrial]:
    mode = _parse_score_bank_mode(score_bank_mode)
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
        all_score_matrix: Optional[np.ndarray] = None
        if mode == "full_reference_bank":
            if full_bank_decoder is None:
                raise ValueError("full_reference_bank scoring requires full_bank_decoder")
            if hasattr(full_bank_decoder, "score_windows_batch"):
                all_score_matrix = np.asarray(full_bank_decoder.score_windows_batch(windows), dtype=np.float64)
            else:
                rows = []
                for window in windows:
                    if hasattr(full_bank_decoder, "score_window"):
                        row = np.asarray(full_bank_decoder.score_window(window), dtype=np.float64)
                    elif hasattr(full_bank_decoder, "analyze_window"):
                        row = np.asarray(dict(full_bank_decoder.analyze_window(window))["scores"], dtype=np.float64)
                    else:
                        raise TypeError("full-bank decoder does not expose a supported scoring API")
                    rows.append(row.reshape(1, -1))
                all_score_matrix = np.vstack(rows)
        feature_matrix = _score_matrices_to_features(
            command_score_matrix=score_matrix,
            command_freqs=freqs,
            score_bank_mode=mode,
            all_score_matrix=all_score_matrix,
            all_freqs=tuple(float(freq) for freq in full_bank_freqs),
        )
        scored.append(
            ScoredTrial(
                trial=trial,
                score_matrix=score_matrix,
                feature_matrix=feature_matrix,
                duration_sec=float(matrix.shape[0]) / float(max(int(decoder.fs), 1)),
                all_score_matrix=all_score_matrix,
                all_freqs=tuple(float(freq) for freq in full_bank_freqs),
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
    score_bank_mode: str = DEFAULT_SCORE_BANK_MODE,
    full_bank_freqs: Sequence[float] = (),
    segments: Sequence[tuple[TrialSpec, np.ndarray]],
    context: str,
    decoder_cache: dict[tuple[Any, ...], Any],
    scored_cache: dict[tuple[Any, ...], dict[tuple[Any, ...], ScoredTrial]],
    require_control: bool = True,
) -> list[ScoredTrial]:
    mode = _parse_score_bank_mode(score_bank_mode)
    full_freqs = tuple(float(freq) for freq in full_bank_freqs)
    runtime_key = _scoring_cache_key(
        freqs=freqs,
        sampling_rate=int(sampling_rate),
        step_sec=float(step_sec),
        win_sec=float(win_sec),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
    ) + (mode, tuple(round(float(freq), 10) for freq in full_freqs))
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
        scoreable_missing_segments = [
            item
            for item in missing_segments
            if (
                np.asarray(item[1]).ndim == 2
                and np.asarray(item[1]).shape[0] >= int(getattr(decoder, "win_samples", 0))
            )
        ]
        full_bank_decoder = None
        if mode == "full_reference_bank":
            full_bank_key = runtime_key + ("full_reference_bank_decoder",)
            full_bank_decoder = decoder_cache.get(full_bank_key)
            if full_bank_decoder is None:
                full_bank_decoder = _build_fbcca_decoder_for_scoring(
                    freqs=full_freqs,
                    sampling_rate=int(sampling_rate),
                    win_sec=float(win_sec),
                    step_sec=float(step_sec),
                    compute_backend=str(compute_backend),
                    gpu_device=int(gpu_device),
                    gpu_precision=str(gpu_precision),
                )
                decoder_cache[full_bank_key] = full_bank_decoder
        if scoreable_missing_segments:
            scored_missing = _score_trials_for_classifier(
                trial_segments=scoreable_missing_segments,
                decoder=decoder,
                freqs=freqs,
                score_bank_mode=mode,
                full_bank_decoder=full_bank_decoder,
                full_bank_freqs=full_freqs,
            )
            if len(scored_missing) != len(scoreable_missing_segments):
                raise RuntimeError(
                    f"{context} scored {len(scored_missing)} of {len(scoreable_missing_segments)} "
                    "scoreable trial segments"
                )
            for item, scored in zip(scoreable_missing_segments, scored_missing):
                runtime_scored_cache[_trial_segment_cache_key(item)] = scored

    scored_subset = [
        runtime_scored_cache[segment_key]
        for segment_key in segment_keys
        if segment_key in runtime_scored_cache
    ]
    _validate_scored_trial_coverage(
        scored_subset,
        freqs=freqs,
        context=context,
        require_control=bool(require_control),
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
    return ridge5_predict_windows_from_state(_classifier_state_payload(model), feature_matrix)


def _softmax_2class_logit(logit: np.ndarray) -> np.ndarray:
    values = np.asarray(logit, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _safe_probability_entropy(probs: np.ndarray) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("probability matrix must be 2D")
    safe = np.clip(values, 1e-12, 1.0)
    safe = safe / np.maximum(np.sum(safe, axis=1, keepdims=True), 1e-12)
    return (-np.sum(safe * np.log(safe), axis=1) / np.log(float(max(values.shape[1], 2)))).astype(
        np.float64,
        copy=False,
    )


def _adaptive_gate_feature_matrix(
    *,
    probs: np.ndarray,
    labels: np.ndarray,
    scored_trial: Optional[ScoredTrial] = None,
) -> np.ndarray:
    probability = np.asarray(probs, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] <= 0:
        return np.zeros((0, len(ADAPTIVE_EVIDENCE_FEATURE_NAMES)), dtype=np.float64)
    label_values = np.asarray(labels, dtype=object)
    idle_matches = np.where(label_values == "idle")[0]
    if idle_matches.size <= 0:
        raise ValueError("adaptive evidence gate requires an idle label")
    idle_index = int(idle_matches[0])
    command_indices = np.asarray([index for index in range(int(label_values.shape[0])) if index != idle_index], dtype=int)
    if command_indices.size <= 0:
        raise ValueError("adaptive evidence gate requires command labels")
    command_probs = probability[:, command_indices]
    top_command_local = np.argmax(command_probs, axis=1)
    top_command_prob = np.take_along_axis(command_probs, top_command_local[:, None], axis=1)[:, 0]
    if command_probs.shape[1] >= 2:
        sorted_command = np.sort(command_probs, axis=1)[:, ::-1]
        top2_command_prob = sorted_command[:, 1]
    else:
        top2_command_prob = np.zeros(int(command_probs.shape[0]), dtype=np.float64)
    idle_prob = probability[:, idle_index]
    command_probability = 1.0 - idle_prob
    top_command_label_index = command_indices[top_command_local]
    same_previous = np.zeros(int(probability.shape[0]), dtype=np.float64)
    streak = np.ones(int(probability.shape[0]), dtype=np.float64)
    for index in range(1, int(probability.shape[0])):
        if int(top_command_label_index[index]) == int(top_command_label_index[index - 1]):
            same_previous[index] = 1.0
            streak[index] = streak[index - 1] + 1.0
    entropy = _safe_probability_entropy(probability)

    full_ratio = np.ones(int(probability.shape[0]), dtype=np.float64)
    full_margin = np.zeros(int(probability.shape[0]), dtype=np.float64)
    inverse_rank = np.ones(int(probability.shape[0]), dtype=np.float64)
    full_entropy = entropy.copy()
    if scored_trial is not None:
        features = np.asarray(scored_trial.feature_matrix, dtype=np.float64)
        full_start = int(command_indices.size + len(CLASSIFIER_DERIVED_FEATURE_NAMES))
        if features.ndim == 2 and features.shape[1] >= full_start + len(FULL_REFERENCE_BANK_FEATURE_NAMES):
            full = features[:, full_start : full_start + len(FULL_REFERENCE_BANK_FEATURE_NAMES)]
            full_ratio = np.asarray(full[:, 3], dtype=np.float64)
            full_margin = np.asarray(full[:, 4], dtype=np.float64)
            inverse_rank = 1.0 / np.maximum(np.asarray(full[:, 2], dtype=np.float64), 1.0)
            full_entropy = np.asarray(full[:, 5], dtype=np.float64)

    return np.column_stack(
        [
            command_probability,
            top_command_prob,
            command_probability - idle_prob,
            top_command_prob - top2_command_prob,
            top_command_prob / np.maximum(top2_command_prob, 1e-12),
            entropy,
            same_previous,
            streak,
            full_ratio,
            full_margin,
            inverse_rank,
            full_entropy,
        ]
    ).astype(np.float64, copy=False)


def _adaptive_gate_feature_matrix_for_trial(
    model: FBCCALDA5Model | FBCCARidge5Model,
    item: ScoredTrial,
    probs: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    return _adaptive_gate_feature_matrix(
        probs=probs,
        labels=labels,
        scored_trial=item,
    )


def _adaptive_gate_window_probabilities(
    model: FBCCALDA5Model | FBCCARidge5Model,
    features: np.ndarray,
) -> np.ndarray:
    if model.evidence_weights is None:
        raise ValueError("adaptive evidence gate model is missing evidence weights")
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("adaptive evidence gate features must be 2D")
    mean = (
        np.asarray(model.evidence_feature_mean, dtype=np.float64)
        if model.evidence_feature_mean is not None
        else np.zeros(int(values.shape[1]), dtype=np.float64)
    )
    std = (
        np.asarray(model.evidence_feature_std, dtype=np.float64)
        if model.evidence_feature_std is not None
        else np.ones(int(values.shape[1]), dtype=np.float64)
    )
    z = (values - mean) / np.maximum(std, 1e-9)
    design = np.column_stack([np.ones(int(z.shape[0]), dtype=np.float64), z])
    logits = design @ np.asarray(model.evidence_weights, dtype=np.float64)
    return _softmax_2class_logit(logits)


def _fit_logistic_binary_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 80,
    sample_weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or target.shape[0] != features.shape[0]:
        raise ValueError("binary logistic fit expects x=(n, d), y=(n,)")
    if features.shape[0] <= 0 or len(set(float(value) for value in target.tolist())) < 2:
        raise ValueError("binary logistic fit requires both classes")
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    z = (features - mean) / std
    design = np.column_stack([np.ones(int(z.shape[0]), dtype=np.float64), z])
    weights = np.zeros(int(design.shape[1]), dtype=np.float64)
    pos = max(float(np.sum(target >= 0.5)), 1.0)
    neg = max(float(np.sum(target < 0.5)), 1.0)
    balanced_weights = np.where(target >= 0.5, 0.5 / pos, 0.5 / neg)
    if sample_weights is None:
        resolved_weights = balanced_weights
    else:
        resolved_weights = balanced_weights * np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if resolved_weights.shape[0] != target.shape[0]:
            raise ValueError("sample_weights must match y")
    resolved_weights *= float(target.shape[0]) / max(float(np.sum(resolved_weights)), 1e-12)
    reg = np.eye(int(design.shape[1]), dtype=np.float64) * max(float(l2), 0.0)
    reg[0, 0] = 0.0
    for _ in range(max(1, int(max_iter))):
        logits = design @ weights
        probs = _softmax_2class_logit(logits)
        grad = design.T @ ((probs - target) * resolved_weights) + reg @ weights
        curvature = np.maximum(probs * (1.0 - probs), 1e-6) * resolved_weights
        hessian = design.T @ (design * curvature[:, None]) + reg
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ grad
        weights -= step
        if float(np.linalg.norm(step)) < 1e-6:
            break
    return weights.astype(np.float64, copy=False), mean.astype(np.float64, copy=False), std.astype(np.float64, copy=False)


def _command_confidence_from_probs(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    idle_index = int(np.where(np.asarray(labels, dtype=object) == "idle")[0][0])
    return 1.0 - values[:, idle_index]


def _label_to_freq_key(label: Any) -> Optional[str]:
    text = str(label)
    if not text or text == "idle":
        return None
    try:
        return _freq_label(float(text))
    except Exception:
        return None


def _feature_name_indices(feature_names: Sequence[str]) -> dict[str, int]:
    return {str(name): int(index) for index, name in enumerate(feature_names)}


def _selected_freq_score_from_row(
    row: np.ndarray,
    pred_label: str,
    feature_indices: Mapping[str, int],
    *,
    score_source_name: str = "fbcca",
) -> float:
    key = f"{str(score_source_name).strip().lower()}_score_{pred_label}"
    if key in feature_indices:
        return float(row[int(feature_indices[key])])
    if "top1_score" in feature_indices:
        return float(row[int(feature_indices["top1_score"])])
    return 0.0


def _frequency_specific_gate_features_for_trial(
    *,
    model: FBCCALDA5Model | FBCCARidge5Model,
    item: ScoredTrial,
    probs: np.ndarray,
    labels: np.ndarray,
    lrt_evidence: np.ndarray,
    feature_names: Sequence[str],
    smoothing_windows: int,
    score_source_name: str = "fbcca",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    probability = np.asarray(probs, dtype=np.float64)
    features = np.asarray(item.feature_matrix, dtype=np.float64)
    evidence = np.asarray(lrt_evidence, dtype=np.float64).reshape(-1)
    if probability.ndim != 2 or features.ndim != 2:
        raise ValueError("frequency-specific gate expects 2D probabilities and features")
    if probability.shape[0] != features.shape[0] or evidence.shape[0] != probability.shape[0]:
        raise ValueError("frequency-specific gate rows must align")
    label_values = np.asarray(labels, dtype=object)
    pred_indices = np.argmax(probability, axis=1)
    pred_labels = [str(label_values[int(index)]) for index in pred_indices]
    indices = _feature_name_indices(feature_names)
    margin_index = _feature_index(feature_names, "margin")
    ratio_index = _feature_index(feature_names, "ratio")
    entropy_index = _feature_index(feature_names, "score_entropy")
    top1_index = _feature_index(feature_names, "top1_score")
    top2_index = _feature_index(feature_names, "top2_score")
    normalized_top1_index = _feature_index(feature_names, "normalized_top1")
    window_count = max(1, int(smoothing_windows))
    gate_rows: list[list[float]] = []
    meta_rows: list[dict[str, Any]] = []
    for index, pred_label in enumerate(pred_labels):
        freq_key = _label_to_freq_key(pred_label)
        start = max(0, int(index) - window_count + 1)
        window_slice = slice(start, int(index) + 1)
        trailing_labels = pred_labels[window_slice]
        same_freq_count = int(sum(1 for item_label in trailing_labels if item_label == pred_label and freq_key is not None))
        margin_values = features[window_slice, margin_index]
        entropy_values = features[window_slice, entropy_index]
        selected_score = (
            _selected_freq_score_from_row(
                features[index],
                pred_label,
                indices,
                score_source_name=score_source_name,
            )
            if freq_key is not None
            else 0.0
        )
        gate_rows.append(
            [
                float(selected_score),
                float(features[index, top1_index]),
                float(features[index, top2_index]),
                float(features[index, margin_index]),
                float(features[index, ratio_index]),
                float(features[index, normalized_top1_index]),
                float(features[index, entropy_index]),
                float(evidence[index]),
                float(same_freq_count),
                float(np.mean(margin_values)) if margin_values.size else 0.0,
                float(np.mean(entropy_values)) if entropy_values.size else 0.0,
            ]
        )
        meta_rows.append(
            {
                "pred_label": pred_label,
                "freq_key": freq_key,
                "window_index": int(index),
                "true_label": _trial_true_label(item.trial),
                "subtype": _ysuan_ns_subtype_from_label(str(item.trial.label)),
            }
        )
    return (
        np.asarray(gate_rows, dtype=np.float64),
        pred_indices.astype(int, copy=False),
        meta_rows,
    )


def _frequency_specific_gate_payload_for_freq(
    model: FBCCALDA5Model | FBCCARidge5Model,
    freq_key: str,
) -> dict[str, Any]:
    payloads = normalize_frequency_specific_control_state_gates(
        getattr(model, "frequency_specific_control_state_gates", None)
    )
    return dict(payloads.get(str(freq_key), {}) or {})


def _tenp5_veto_feature_row(row: np.ndarray) -> np.ndarray:
    source = np.asarray(row, dtype=np.float64).reshape(-1)
    name_to_index = {name: int(index) for index, name in enumerate(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES)}
    return np.asarray(
        [source[name_to_index[name]] for name in TENP5_NS2_VETO_FEATURE_NAMES],
        dtype=np.float64,
    )


def _logistic_payload_probability(payload: Mapping[str, Any], row: np.ndarray) -> Optional[float]:
    if str(dict(payload).get("type", "")) not in {"logistic", "ns2_hard_negative_veto"}:
        return None
    weights = np.asarray(dict(payload).get("weights", []), dtype=np.float64).reshape(-1)
    mean = np.asarray(dict(payload).get("feature_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(dict(payload).get("feature_std", []), dtype=np.float64).reshape(-1)
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    if weights.size != values.size + 1 or mean.size != values.size or std.size != values.size:
        return None
    z = (values - mean) / np.maximum(std, 1e-9)
    return float(_softmax_2class_logit(np.asarray([weights[0] + z @ weights[1:]], dtype=np.float64))[0])


def _frequency_specific_payload_pass(payload: Mapping[str, Any], row: np.ndarray) -> bool:
    payload_dict = dict(payload or {})
    gate_type = str(payload_dict.get("type", "threshold"))
    if gate_type == "ns2_hard_negative_veto":
        prob = _logistic_payload_probability(payload_dict, row)
        if prob is None:
            return True
        return bool(prob < float(payload_dict.get("veto_threshold", 0.5)) - 1e-12)
    if gate_type == "logistic":
        prob = _logistic_payload_probability(payload_dict, row)
        if prob is None:
            return True
        return bool(prob + 1e-12 >= float(payload_dict.get("prob_threshold", 0.5)))
    name_to_index = {name: int(index) for index, name in enumerate(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES)}
    return bool(
        row[name_to_index["lrt_evidence"]] + 1e-12 >= float(payload_dict.get("theta_lrt_f", payload_dict.get("theta_lrt", 0.0)))
        and row[name_to_index["selected_freq_score"]] + 1e-12 >= float(payload_dict.get("theta_score_f", payload_dict.get("theta_score", 0.0)))
        and row[name_to_index["margin"]] + 1e-12 >= float(payload_dict.get("theta_margin_f", payload_dict.get("theta_margin", 0.0)))
        and row[name_to_index["ratio"]] + 1e-12 >= float(payload_dict.get("theta_ratio_f", payload_dict.get("theta_ratio", 0.0)))
        and row[name_to_index["score_entropy"]] <= float(payload_dict.get("theta_entropy_f", payload_dict.get("theta_entropy", 1.0))) + 1e-12
        and row[name_to_index["multiwindow_same_freq_count"]] + 1e-12
        >= float(payload_dict.get("theta_multiwindow_same_freq_count", 1.0))
    )


def _conditional_frequency_specific_risk_level(payload: Mapping[str, Any], row: np.ndarray) -> str:
    payload_dict = dict(payload or {})
    if not bool(payload_dict.get("conditional_applies", True)):
        return "low"
    name_to_index = {name: int(index) for index, name in enumerate(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES)}
    lrt = float(row[name_to_index["lrt_evidence"]])
    margin = float(row[name_to_index["margin"]])
    ratio = float(row[name_to_index["ratio"]])
    entropy = float(row[name_to_index["score_entropy"]])
    same_freq = float(row[name_to_index["multiwindow_same_freq_count"]])
    low_risk = (
        lrt + 1e-12 >= float(payload_dict.get("conditional_low_risk_lrt_th", -float("inf")))
        and margin + 1e-12 >= float(payload_dict.get("conditional_low_risk_margin_th", 0.0))
        and ratio + 1e-12 >= float(payload_dict.get("conditional_low_risk_ratio_th", 1.0))
        and entropy <= float(payload_dict.get("conditional_low_risk_entropy_th", 1.0)) + 1e-12
        and same_freq + 1e-12 >= float(payload_dict.get("conditional_low_risk_same_freq_count", 1.0))
    )
    if low_risk:
        return "low"
    high_risk = (
        lrt + 1e-12 < float(payload_dict.get("conditional_high_risk_lrt_th", -float("inf")))
        or margin + 1e-12 < float(payload_dict.get("conditional_high_risk_margin_th", 0.0))
        or ratio + 1e-12 < float(payload_dict.get("conditional_high_risk_ratio_th", 1.0))
        or entropy > float(payload_dict.get("conditional_high_risk_entropy_th", 1.0)) + 1e-12
        or same_freq + 1e-12 < float(payload_dict.get("conditional_high_risk_same_freq_count", 1.0))
    )
    return "high" if high_risk else "medium"


def _frequency_specific_gate_mask_for_model(
    model: FBCCALDA5Model | FBCCARidge5Model,
    *,
    probs: np.ndarray,
    labels: np.ndarray,
    feature_matrix: np.ndarray,
    lrt_evidence: np.ndarray,
) -> np.ndarray:
    payloads = normalize_frequency_specific_control_state_gates(
        getattr(model, "frequency_specific_control_state_gates", None)
    )
    values = np.asarray(probs, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("probability matrix must be 2D")
    if not payloads:
        return np.ones(int(values.shape[0]), dtype=bool)
    gate_variant = parse_classifier_gate_variant(getattr(model, "gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW))
    feature_names = _classifier_feature_names(
        model.freqs,
        score_bank_mode=str(getattr(model, "fit_summary", {}).get("score_bank_mode", "full_reference_bank")),
    )
    gate_features, pred_indices, meta_rows = _frequency_specific_gate_features_for_trial(
        model=model,
        item=ScoredTrial(
            trial=TrialSpec(label="", expected_freq=None, trial_id=0, block_index=0),
            score_matrix=np.zeros((int(values.shape[0]), len(model.freqs)), dtype=np.float64),
            feature_matrix=feature_matrix,
            duration_sec=0.0,
        ),
        probs=values,
        labels=labels,
        lrt_evidence=lrt_evidence,
        feature_names=feature_names,
        smoothing_windows=int(getattr(model, "smoothing_windows", 1)),
    )
    if gate_variant == CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO:
        mask = np.ones(int(values.shape[0]), dtype=bool)
        payload = dict(payloads.get(TENP5_NS2_VETO_FREQ_KEY, {}) or {})
        if not payload or str(payload.get("status", "")) != "ok":
            return mask
        for row_index, meta in enumerate(meta_rows):
            if str(meta.get("freq_key")) != TENP5_NS2_VETO_FREQ_KEY:
                continue
            mask[row_index] = _frequency_specific_payload_pass(
                payload,
                _tenp5_veto_feature_row(gate_features[row_index]),
            )
        return mask
    mask = np.zeros(int(values.shape[0]), dtype=bool)
    idle_index = int(np.where(np.asarray(labels, dtype=object) == "idle")[0][0])
    for row_index, meta in enumerate(meta_rows):
        pred_index = int(pred_indices[row_index])
        if pred_index == idle_index:
            continue
        freq_key = meta.get("freq_key")
        payload = dict(payloads.get(str(freq_key), {}) or {})
        if not payload:
            continue
        if gate_variant == CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC:
            risk_level = _conditional_frequency_specific_risk_level(payload, gate_features[row_index])
            if risk_level == "low":
                mask[row_index] = True
                continue
            if risk_level == "high":
                mask[row_index] = False
                continue
            mask[row_index] = _frequency_specific_payload_pass(payload, gate_features[row_index])
            continue
        row = gate_features[row_index]
        mask[row_index] = _frequency_specific_payload_pass(payload, row)
    return mask


def _predict_fbcca_lda5_trial(
    model: FBCCALDA5Model | FBCCARidge5Model,
    item: ScoredTrial,
    *,
    min_enter_windows: int,
    max_gap_windows: int = 0,
) -> tuple[str, float, float]:
    probs, labels = _predict_classifier_windows(model, item.feature_matrix)
    probs = _smooth_classifier_probabilities(
        probs,
        smoothing_windows=int(getattr(model, "smoothing_windows", 1)),
    )
    if str(getattr(model, "gate_policy", CLASSIFIER_CONFIDENCE_GATE_POLICY)) == CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY:
        features = _adaptive_gate_feature_matrix_for_trial(model, item, probs, labels)
        gate_probs = _adaptive_gate_window_probabilities(model, features)
        return _predict_adaptive_evidence_trial_from_probs(
            model,
            probs,
            labels,
            gate_probs,
            min_enter_windows=min_enter_windows,
            max_gap_windows=max(0, int(max_gap_windows)),
        )
    if str(getattr(model, "gate_policy", CLASSIFIER_CONFIDENCE_GATE_POLICY)) == CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY:
        window_evidence = _lrt_window_evidence_from_features(model, item.feature_matrix)
        return _predict_lrt_multiwindow_reject_trial_from_probs(
            model,
            probs,
            labels,
            window_evidence,
            min_enter_windows=min_enter_windows,
            max_gap_windows=max(0, int(max_gap_windows)),
            feature_matrix=item.feature_matrix,
        )
    return _predict_fbcca_lda5_trial_from_probs(
        model,
        probs,
        labels,
        min_enter_windows=min_enter_windows,
        max_gap_windows=max(0, int(max_gap_windows)),
    )


def _smooth_classifier_probabilities(probs: np.ndarray, smoothing_windows: int = 1) -> np.ndarray:
    return runtime_smooth_classifier_probabilities(probs, smoothing_windows=smoothing_windows)


def _feature_index(feature_names: Sequence[str], name: str) -> int:
    if name not in feature_names:
        raise ValueError(f"missing classifier feature for gate variant: {name}")
    return int(list(feature_names).index(name))


def _classifier_gate_variant_token(
    variant: str,
    params: Optional[Mapping[str, Any]] = None,
) -> str:
    resolved = parse_classifier_gate_variant(variant)
    payload = dict(params or {})
    if resolved == CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW:
        return "base"
    if resolved == CLASSIFIER_GATE_VARIANT_LRTMW_MARGIN:
        return "mg" + "_".join(
            str(payload.get(key, "")).replace(".", "p")
            for key in ("margin_control_quantile", "margin_idle_quantile", "ratio_idle_quantile")
        )
    if resolved == CLASSIFIER_GATE_VARIANT_LRTMW_ENTROPY:
        return "ent" + "_".join(
            str(payload.get(key, "")).replace(".", "p")
            for key in ("entropy_control_quantile", "entropy_idle_quantile")
        )
    if resolved == CLASSIFIER_GATE_VARIANT_SUBJECT_THRESHOLD_FLOOR:
        return "floor" + "_".join(
            str(payload.get(key, "")).replace(".", "p")
            for key in ("global_floor_quantile", "subject_idle_quantile")
        )
    if resolved == CLASSIFIER_GATE_VARIANT_NS2_AWARE:
        return f"ns2sf{str(payload.get('ns2_safety_factor', '')).replace('.', 'p')}"
    if resolved == CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE:
        return "floorns2" + "_".join(
            str(payload.get(key, "")).replace(".", "p")
            for key in ("global_floor_quantile", "subject_idle_quantile", "ns2_safety_factor")
        )
    if resolved == CLASSIFIER_GATE_VARIANT_WEAK_SUBJECT_GUARD:
        return "weakguard"
    if resolved == CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD:
        return "fsth" + "_".join(
            str(payload.get(key, "")).replace(".", "p")
            for key in (
                "margin_idle_quantile",
                "ratio_idle_quantile",
                "entropy_control_quantile",
                "ns2_safety_factor",
            )
        )
    if resolved == CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC:
        return "fslog" + "_".join(
            str(payload.get(key, "")).replace(".", "p")
            for key in ("prob_threshold", "ns2_sample_weight")
        )
    if resolved == CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC:
        return "cfslog" + "_".join(
            str(payload.get(key, "")).replace(".", "p")
            for key in ("conditional_policy", "prob_threshold", "ns2_sample_weight")
        )
    if resolved == CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO:
        return (
            "t105ns2veto_"
            f"th{str(payload.get('veto_threshold', '')).replace('.', 'p')}_"
            f"w{str(payload.get('ns2_weight', '')).replace('.', 'p')}"
        )
    return resolved


def _gate_variant_param_grid(
    variant: str,
    *,
    freqspec_threshold_combo_set: str = FREQSPEC_THRESHOLD_COMBO_SET_NONE,
    freqspec_margin_idle_quantiles: Sequence[float] = DEFAULT_FREQSPEC_MARGIN_IDLE_QUANTILES,
    freqspec_ratio_idle_quantiles: Sequence[float] = DEFAULT_FREQSPEC_RATIO_IDLE_QUANTILES,
    freqspec_entropy_control_quantiles: Sequence[float] = DEFAULT_FREQSPEC_ENTROPY_CONTROL_QUANTILES,
    ns2_safety_factors: Sequence[float] = DEFAULT_NS2_SAFETY_FACTORS,
    subject_floor_global_quantiles: Sequence[float] = DEFAULT_GLOBAL_FLOOR_QUANTILES,
    subject_floor_idle_quantiles: Sequence[float] = DEFAULT_SUBJECT_IDLE_QUANTILES,
    freqspec_ns2_safety_factors: Sequence[float] = DEFAULT_NS2_SAFETY_FACTORS,
    freqspec_logistic_prob_thresholds: Sequence[float] = DEFAULT_FREQSPEC_LOGISTIC_PROB_THRESHOLDS,
    freqspec_logistic_ns2_weights: Sequence[float] = DEFAULT_FREQSPEC_LOGISTIC_NS2_WEIGHTS,
    tenp5_veto_thresholds: Sequence[float] = DEFAULT_TENP5_VETO_THRESHOLDS,
    tenp5_ns2_weights: Sequence[float] = DEFAULT_TENP5_NS2_WEIGHTS,
) -> list[dict[str, Any]]:
    resolved = parse_classifier_gate_variant(variant)
    if resolved == CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW:
        return [{"gate_variant": resolved}]
    if resolved == CLASSIFIER_GATE_VARIANT_LRTMW_MARGIN:
        return [
            {
                "gate_variant": resolved,
                "margin_control_quantile": float(mc),
                "margin_idle_quantile": float(mi),
                "ratio_idle_quantile": float(ri),
            }
            for mc, mi, ri in product(
                DEFAULT_MARGIN_CONTROL_QUANTILES,
                DEFAULT_MARGIN_IDLE_QUANTILES,
                DEFAULT_RATIO_IDLE_QUANTILES,
            )
        ]
    if resolved == CLASSIFIER_GATE_VARIANT_LRTMW_ENTROPY:
        return [
            {
                "gate_variant": resolved,
                "entropy_control_quantile": float(ec),
                "entropy_idle_quantile": float(ei),
            }
            for ec, ei in product(DEFAULT_ENTROPY_CONTROL_QUANTILES, DEFAULT_ENTROPY_IDLE_QUANTILES)
        ]
    if resolved == CLASSIFIER_GATE_VARIANT_SUBJECT_THRESHOLD_FLOOR:
        return [
            {
                "gate_variant": resolved,
                "global_floor_quantile": float(gq),
                "subject_idle_quantile": float(sq),
            }
            for gq, sq in product(
                tuple(float(value) for value in subject_floor_global_quantiles),
                tuple(float(value) for value in subject_floor_idle_quantiles),
            )
        ]
    if resolved == CLASSIFIER_GATE_VARIANT_NS2_AWARE:
        return [
            {
                "gate_variant": resolved,
                "ns2_safety_factor": float(value),
            }
            for value in tuple(float(value) for value in ns2_safety_factors)
        ]
    if resolved == CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE:
        return [
            {
                "gate_variant": resolved,
                "global_floor_quantile": float(gq),
                "subject_idle_quantile": float(sq),
                "ns2_safety_factor": float(sf),
            }
            for gq, sq, sf in product(
                tuple(float(value) for value in subject_floor_global_quantiles),
                tuple(float(value) for value in subject_floor_idle_quantiles),
                tuple(float(value) for value in ns2_safety_factors),
            )
        ]
    if resolved == CLASSIFIER_GATE_VARIANT_WEAK_SUBJECT_GUARD:
        return [{"gate_variant": resolved}]
    if resolved == CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD:
        combo_set = str(freqspec_threshold_combo_set or "").strip().lower()
        if combo_set == FREQSPEC_THRESHOLD_COMBO_SET_PRIORITY6:
            return [
                {
                    "gate_variant": resolved,
                    **{
                        key: (str(value) if key == "combo_name" else float(value))
                        for key, value in combo.items()
                    },
                }
                for combo in FREQSPEC_THRESHOLD_PRIORITY6_COMBOS
            ]
        if combo_set:
            raise ValueError(
                "unsupported frequency-specific threshold combo set "
                f"{freqspec_threshold_combo_set!r}"
            )
        return [
            {
                "gate_variant": resolved,
                "margin_idle_quantile": float(mq),
                "ratio_idle_quantile": float(rq),
                "entropy_control_quantile": float(eq),
                "ns2_safety_factor": float(sf),
            }
            for mq, rq, eq, sf in product(
                tuple(float(value) for value in freqspec_margin_idle_quantiles),
                tuple(float(value) for value in freqspec_ratio_idle_quantiles),
                tuple(float(value) for value in freqspec_entropy_control_quantiles),
                tuple(float(value) for value in freqspec_ns2_safety_factors),
            )
        ]
    if resolved == CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC:
        return [
            {
                "gate_variant": resolved,
                "prob_threshold": float(th),
                "ns2_sample_weight": float(weight),
            }
            for th, weight in product(
                tuple(float(value) for value in freqspec_logistic_prob_thresholds),
                tuple(float(value) for value in freqspec_logistic_ns2_weights),
            )
        ]
    if resolved == CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC:
        return [
            {
                "gate_variant": resolved,
                "prob_threshold": float(th),
                "ns2_sample_weight": float(weight),
                **dict(config),
            }
            for config, th, weight in product(
                CONDITIONAL_GATE_CONFIGS,
                tuple(float(value) for value in freqspec_logistic_prob_thresholds),
                tuple(float(value) for value in freqspec_logistic_ns2_weights),
            )
        ]
    if resolved == CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO:
        return [
            {
                "gate_variant": resolved,
                "veto_threshold": float(th),
                "ns2_weight": float(weight),
            }
            for th, weight in product(
                tuple(float(value) for value in tenp5_veto_thresholds),
                tuple(float(value) for value in tenp5_ns2_weights),
            )
        ]
    return [{"gate_variant": resolved}]


def _trial_group_arrays_for_gate(
    base_model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    smoothing_windows: int,
) -> dict[str, np.ndarray]:
    probability_cache = _build_classifier_probability_cache(base_model, scored_trials)
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    for item, probs, labels in probability_cache:
        if item.feature_matrix.shape[0] <= 0:
            continue
        smoothed = _smooth_classifier_probabilities(probs, smoothing_windows=max(1, int(smoothing_windows)))
        pred_indices = np.argmax(smoothed, axis=1)
        true_label = _trial_true_label(item.trial)
        features = np.asarray(item.feature_matrix, dtype=np.float64)
        if true_label == "idle":
            rows["idle"].append(features)
            subtype = _ysuan_ns_subtype_from_label(str(item.trial.label))
            if subtype:
                rows[subtype].append(features)
            continue
        command_rows = [
            features[row_index : row_index + 1]
            for row_index, pred_index in enumerate(pred_indices)
            if str(base_model.labels[int(pred_index)]) == true_label
        ]
        if command_rows:
            rows["control"].append(np.vstack(command_rows).astype(np.float64, copy=False))
    return {
        key: np.vstack(value).astype(np.float64, copy=False) if value else np.zeros((0, 0), dtype=np.float64)
        for key, value in rows.items()
    }


def _trial_frequency_specific_rows_for_gate(
    base_model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    feature_names: Sequence[str],
    smoothing_windows: int,
) -> dict[str, list[dict[str, Any]]]:
    probability_cache = _build_classifier_probability_cache(base_model, scored_trials)
    rows_by_freq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baseline_model = replace(
        base_model,
        gate_variant=CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
        frequency_specific_control_state_gates=None,
    )
    for item, probs, labels in probability_cache:
        if item.feature_matrix.shape[0] <= 0:
            continue
        smoothed = _smooth_classifier_probabilities(probs, smoothing_windows=max(1, int(smoothing_windows)))
        evidence = _lrt_window_evidence_from_features(base_model, item.feature_matrix)
        baseline_label, _baseline_conf, _baseline_index = _predict_lrt_multiwindow_reject_trial_from_probs(
            baseline_model,
            smoothed,
            labels,
            evidence,
            min_enter_windows=max(1, int(base_model.fit_summary.get("min_enter_windows", 1))),
            max_gap_windows=0,
            feature_matrix=item.feature_matrix,
        )
        baseline_pass = _window_pass_mask_for_lrt_model(
            baseline_model,
            probs=smoothed,
            labels=labels,
            lrt_evidence=evidence,
            feature_matrix=item.feature_matrix,
        )
        gate_features, pred_indices, meta_rows = _frequency_specific_gate_features_for_trial(
            model=base_model,
            item=item,
            probs=smoothed,
            labels=labels,
            lrt_evidence=evidence,
            feature_names=feature_names,
            smoothing_windows=max(1, int(smoothing_windows)),
        )
        true_label = _trial_true_label(item.trial)
        subtype = _ysuan_ns_subtype_from_label(str(item.trial.label))
        for row_index, meta in enumerate(meta_rows):
            freq_key = meta.get("freq_key")
            if freq_key is None:
                continue
            pred_label = str(meta.get("pred_label", ""))
            is_positive = bool(true_label != "idle" and true_label == pred_label)
            is_idle_negative = bool(true_label == "idle")
            is_optional_negative = bool(true_label != "idle" and true_label != pred_label)
            is_hard_negative = bool(subtype == "ns2" and baseline_label == pred_label)
            rows_by_freq[str(freq_key)].append(
                {
                    "x": gate_features[row_index].astype(np.float64, copy=False),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "subtype": subtype,
                    "positive": is_positive,
                    "negative": bool(is_idle_negative or is_optional_negative),
                    "idle_negative": is_idle_negative,
                    "optional_negative": is_optional_negative,
                    "hard_negative": is_hard_negative,
                    "baseline_trial_label": baseline_label,
                    "baseline_window_pass": bool(row_index < baseline_pass.shape[0] and baseline_pass[row_index]),
                    "trial_id": int(item.trial.trial_id),
                    "block_index": int(item.trial.block_index),
                    "window_index": int(row_index),
                }
            )
    return {key: list(value) for key, value in rows_by_freq.items()}


def _nc_csns_feature_row(row: np.ndarray, selected_freq: Any) -> np.ndarray:
    source = np.asarray(row, dtype=np.float64).reshape(-1)
    name_to_index = {name: int(index) for index, name in enumerate(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES)}
    freq_key = ""
    if selected_freq not in (None, ""):
        try:
            freq_key = _freq_label(float(selected_freq))
        except Exception:
            freq_key = str(selected_freq)
    values = [
        float(source[name_to_index["top1_score"]]),
        float(source[name_to_index["top2_score"]]),
        float(source[name_to_index["selected_freq_score"]]),
        float(source[name_to_index["margin"]]),
        float(source[name_to_index["ratio"]]),
        float(source[name_to_index["score_entropy"]]),
        float(source[name_to_index["lrt_evidence"]]),
        float(source[name_to_index["multiwindow_same_freq_count"]]),
        float(source[name_to_index["multiwindow_margin_mean"]]),
        float(source[name_to_index["multiwindow_entropy_mean"]]),
    ]
    for freq in (8.0, 10.5, 12.0, 15.0):
        values.append(1.0 if freq_key == _freq_label(freq) else 0.0)
    return np.asarray(values, dtype=np.float64)


def _nc_csns_feature_matrix_for_trial(
    model: FBCCALDA5Model | FBCCARidge5Model,
    item: ScoredTrial,
    probs: np.ndarray,
    labels: np.ndarray,
    lrt_evidence: np.ndarray,
    *,
    feature_names: Sequence[str],
    smoothing_windows: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    gate_features, pred_indices, meta_rows = _frequency_specific_gate_features_for_trial(
        model=model,
        item=item,
        probs=probs,
        labels=labels,
        lrt_evidence=lrt_evidence,
        feature_names=feature_names,
        smoothing_windows=max(1, int(smoothing_windows)),
        score_source_name=str(dict(getattr(model, "fit_summary", {}) or {}).get("score_source_name", "fbcca")),
    )
    rows = [
        _nc_csns_feature_row(gate_row, dict(meta).get("freq_key", ""))
        for gate_row, meta in zip(gate_features, meta_rows)
    ]
    matrix = (
        np.vstack(rows).astype(np.float64, copy=False)
        if rows
        else np.zeros((0, len(NC_CSNS_FEATURE_NAMES)), dtype=np.float64)
    )
    return matrix, pred_indices, meta_rows


def _nc_csns_rows_for_trials(
    model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    feature_names: Sequence[str],
    smoothing_windows: int,
    target: int,
) -> list[dict[str, Any]]:
    baseline_model = replace(
        model,
        gate_variant=CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
        frequency_specific_control_state_gates=None,
        lrt_window_floor_th=None,
    )
    rows: list[dict[str, Any]] = []
    probability_cache = _build_classifier_probability_cache(baseline_model, scored_trials)
    for item, probs, labels in probability_cache:
        if item.feature_matrix.shape[0] <= 0:
            continue
        smoothed = _smooth_classifier_probabilities(probs, smoothing_windows=max(1, int(smoothing_windows)))
        evidence = _lrt_window_evidence_from_features(baseline_model, item.feature_matrix)
        baseline_pass = _window_pass_mask_for_lrt_model(
            baseline_model,
            probs=smoothed,
            labels=labels,
            lrt_evidence=evidence,
            feature_matrix=item.feature_matrix,
        )
        feature_matrix, pred_indices, meta_rows = _nc_csns_feature_matrix_for_trial(
            baseline_model,
            item,
            smoothed,
            labels,
            evidence,
            feature_names=feature_names,
            smoothing_windows=max(1, int(smoothing_windows)),
        )
        label_values = np.asarray(labels, dtype=object)
        true_label = _trial_true_label(item.trial)
        subtype = _ysuan_ns_subtype_from_label(str(item.trial.label))
        for row_index, row in enumerate(feature_matrix):
            pred_label = str(label_values[int(pred_indices[row_index])])
            rows.append(
                {
                    "x": np.asarray(row, dtype=np.float64),
                    "target": int(target),
                    "true_label": true_label,
                    "subtype": subtype,
                    "selected_freq": str(dict(meta_rows[row_index]).get("freq_key") or "idle"),
                    "pred_label": pred_label,
                    "baseline_window_pass": bool(
                        row_index < baseline_pass.shape[0] and baseline_pass[row_index]
                    ),
                    "trial_id": int(item.trial.trial_id),
                    "block_index": int(item.trial.block_index),
                    "window_index": int(row_index),
                }
            )
    return rows


def _feature_auc_positive_greater(positive: Sequence[Any], negative: Sequence[Any]) -> float:
    pos = np.asarray([_safe_float(value, float("nan")) for value in positive], dtype=np.float64)
    neg = np.asarray([_safe_float(value, float("nan")) for value in negative], dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size <= 0 or neg.size <= 0:
        return float("nan")
    greater = 0.0
    ties = 0.0
    for value in pos:
        greater += float(np.sum(value > neg))
        ties += float(np.sum(value == neg))
    return float((greater + 0.5 * ties) / max(float(pos.size * neg.size), 1.0))


def _nc_calibration_feature_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str = "",
    subject: str = "",
    split_index: int = 0,
    nc_seconds: float = 0.0,
    nc_source: str = "",
    nc_gate_type: str = "",
    recipe_id: str = "",
) -> list[dict[str, Any]]:
    positive = [np.asarray(row.get("x"), dtype=np.float64) for row in rows if int(row.get("target", 0)) == 1]
    negative = [np.asarray(row.get("x"), dtype=np.float64) for row in rows if int(row.get("target", 0)) == 0]
    if not positive or not negative:
        return []
    pos = np.vstack(positive).astype(np.float64, copy=False)
    neg = np.vstack(negative).astype(np.float64, copy=False)
    payload: list[dict[str, Any]] = []
    for index, feature in enumerate(NC_CSNS_FEATURE_NAMES):
        pos_values = pos[:, index]
        neg_values = neg[:, index]
        auc = _feature_auc_positive_greater(pos_values, neg_values)
        payload.append(
            {
                "dataset": str(dataset),
                "subject": str(subject),
                "split_index": int(split_index),
                "recipe_id": str(recipe_id),
                "nc_seconds": float(nc_seconds),
                "nc_source": str(nc_source),
                "nc_gate_type": str(nc_gate_type),
                "feature": str(feature),
                "positive_count": int(pos_values.size),
                "negative_count": int(neg_values.size),
                "mean_positive": _finite_or_none(np.mean(pos_values)),
                "mean_negative": _finite_or_none(np.mean(neg_values)),
                "median_positive": _finite_or_none(np.median(pos_values)),
                "median_negative": _finite_or_none(np.median(neg_values)),
                "delta_mean": _finite_or_none(float(np.mean(pos_values) - np.mean(neg_values))),
                "auc_positive_greater": _finite_or_none(auc),
                "auc_best_direction": _finite_or_none(max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan")),
            }
        )
    return payload


def _fit_nc_session_csns_payload(
    model: FBCCALDA5Model | FBCCARidge5Model,
    *,
    command_scored: Sequence[ScoredTrial],
    nc_scored: Sequence[ScoredTrial],
    feature_names: Sequence[str],
    smoothing_windows: int,
    nc_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    command_only_scored = [item for item in command_scored if item.trial.expected_freq is not None]
    positive_rows = _nc_csns_rows_for_trials(
        model,
        command_only_scored,
        feature_names=feature_names,
        smoothing_windows=max(1, int(smoothing_windows)),
        target=1,
    )
    negative_rows = _nc_csns_rows_for_trials(
        model,
        nc_scored,
        feature_names=feature_names,
        smoothing_windows=max(1, int(smoothing_windows)),
        target=0,
    )
    rows = [*positive_rows, *negative_rows]
    payload: dict[str, Any] = {
        "type": "session_specific_logistic_csns_detector",
        "status": "unsupported_missing_training_class",
        "feature_names": list(NC_CSNS_FEATURE_NAMES),
        "prob_threshold": float(NC_CSNS_PROB_THRESHOLD),
        "positive_windows": int(len(positive_rows)),
        "negative_windows": int(len(negative_rows)),
        "command_calibration_trial_count": int(len(command_only_scored)),
        "ignored_idle_calibration_trial_count": int(len(command_scored) - len(command_only_scored)),
        "fit_split": "command_calibration_blocks_plus_no_control_calibration_fit_split",
        "test_split": "holdout_blocks",
        "no_control_calibration_provenance": dict(nc_provenance),
        "training_window_policy": "all_scored_calibration_windows_score_space_only",
        "target_definition": "1=command_state,0=no_control_state",
    }
    if not positive_rows:
        payload["reason"] = "no_command_calibration_windows"
        return payload, rows, _nc_calibration_feature_summary_rows(rows)
    if not negative_rows:
        payload["reason"] = "no_no_control_calibration_windows"
        return payload, rows, _nc_calibration_feature_summary_rows(rows)
    x = np.vstack([np.asarray(row["x"], dtype=np.float64) for row in rows])
    y = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    if len(set(float(value) for value in y.tolist())) < 2:
        payload["reason"] = "single_training_class"
        return payload, rows, _nc_calibration_feature_summary_rows(rows)
    weights, mean, std = _fit_logistic_binary_ridge(
        x,
        y,
        l2=1.0,
        max_iter=100,
    )
    payload.update(
        {
            "status": "ok",
            "weights": _array_payload(weights),
            "feature_mean": _array_payload(mean),
            "feature_std": _array_payload(std),
        }
    )
    return payload, rows, _nc_calibration_feature_summary_rows(rows)


def _nc_csns_probability(payload: Mapping[str, Any], feature_matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(feature_matrix, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.size <= 0:
        return np.zeros((int(values.shape[0]) if values.ndim == 2 else 0,), dtype=np.float64)
    payload_dict = dict(payload or {})
    if str(payload_dict.get("status", "")) != "ok":
        return np.ones(int(values.shape[0]), dtype=np.float64)
    weights = np.asarray(payload_dict.get("weights", []), dtype=np.float64).reshape(-1)
    mean = np.asarray(payload_dict.get("feature_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(payload_dict.get("feature_std", []), dtype=np.float64).reshape(-1)
    if weights.size != values.shape[1] + 1 or mean.size != values.shape[1] or std.size != values.shape[1]:
        return np.ones(int(values.shape[0]), dtype=np.float64)
    z = (values - mean) / np.maximum(std, 1e-9)
    return _softmax_2class_logit(weights[0] + z @ weights[1:])


def _nc_conditional_thresholds(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    positive = [np.asarray(row.get("x"), dtype=np.float64) for row in rows if int(row.get("target", 0)) == 1]
    if not positive:
        return {
            "margin": 0.0,
            "ratio": 1.0,
            "score_entropy": 1.0,
            "lrt_evidence": -float("inf"),
            "multiwindow_same_freq_count": float(NC_CONDITIONAL_LOW_RISK_SAME_FREQ_COUNT),
        }
    matrix = np.vstack(positive).astype(np.float64, copy=False)
    indices = {name: int(index) for index, name in enumerate(NC_CSNS_FEATURE_NAMES)}
    return {
        "margin": _safe_quantile(
            matrix[:, indices["margin"]],
            NC_CONDITIONAL_LOW_RISK_MARGIN_QUANTILE,
            0.0,
        ),
        "ratio": _safe_quantile(
            matrix[:, indices["ratio"]],
            NC_CONDITIONAL_LOW_RISK_RATIO_QUANTILE,
            1.0,
        ),
        "score_entropy": _safe_quantile(
            matrix[:, indices["score_entropy"]],
            NC_CONDITIONAL_LOW_RISK_ENTROPY_QUANTILE,
            1.0,
        ),
        "lrt_evidence": _safe_quantile(
            matrix[:, indices["lrt_evidence"]],
            NC_CONDITIONAL_LOW_RISK_LRT_QUANTILE,
            -float("inf"),
        ),
        "multiwindow_same_freq_count": float(NC_CONDITIONAL_LOW_RISK_SAME_FREQ_COUNT),
    }


def _nc_conditional_low_risk(feature_matrix: np.ndarray, thresholds: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(feature_matrix, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.size <= 0:
        return np.zeros((int(values.shape[0]) if values.ndim == 2 else 0,), dtype=bool)
    indices = {name: int(index) for index, name in enumerate(NC_CSNS_FEATURE_NAMES)}
    return (
        (values[:, indices["margin"]] + 1e-12 >= float(thresholds.get("margin", 0.0)))
        & (values[:, indices["ratio"]] + 1e-12 >= float(thresholds.get("ratio", 1.0)))
        & (values[:, indices["score_entropy"]] <= float(thresholds.get("score_entropy", 1.0)) + 1e-12)
        & (values[:, indices["lrt_evidence"]] + 1e-12 >= float(thresholds.get("lrt_evidence", -float("inf"))))
        & (
            values[:, indices["multiwindow_same_freq_count"]] + 1e-12
            >= float(thresholds.get("multiwindow_same_freq_count", NC_CONDITIONAL_LOW_RISK_SAME_FREQ_COUNT))
        )
    )


def _split_gate_fit_validation_trials(
    scored_trials: Sequence[ScoredTrial],
) -> tuple[list[ScoredTrial], list[ScoredTrial], dict[str, Any]]:
    trials = sorted(list(scored_trials), key=lambda item: (int(item.trial.block_index), int(item.trial.trial_id)))
    if len(trials) <= 1:
        return trials, [], {
            "policy": "all_fit_no_validation",
            "fit_trial_ids": [int(item.trial.trial_id) for item in trials],
            "validation_trial_ids": [],
            "fit_blocks": sorted({int(item.trial.block_index) for item in trials}),
            "validation_blocks": [],
        }
    blocks = sorted({int(item.trial.block_index) for item in trials})
    if len(blocks) >= 2:
        validation_blocks = {int(blocks[-1])}
        fit_trials = [item for item in trials if int(item.trial.block_index) not in validation_blocks]
        validation_trials = [item for item in trials if int(item.trial.block_index) in validation_blocks]
        policy = "calibration_block_holdout_last_block"
    else:
        fit_trials = [item for index, item in enumerate(trials) if index % 2 == 0]
        validation_trials = [item for index, item in enumerate(trials) if index % 2 == 1]
        policy = "calibration_trial_holdout_alternating"
    if not fit_trials:
        fit_trials, validation_trials = trials, []
        policy = "all_fit_no_validation_fallback"
    return fit_trials, validation_trials, {
        "policy": policy,
        "fit_trial_ids": [int(item.trial.trial_id) for item in fit_trials],
        "validation_trial_ids": [int(item.trial.trial_id) for item in validation_trials],
        "fit_blocks": sorted({int(item.trial.block_index) for item in fit_trials}),
        "validation_blocks": sorted({int(item.trial.block_index) for item in validation_trials}),
    }


def _fit_frequency_specific_threshold_gate_payload(
    model: FBCCALDA5Model | FBCCARidge5Model,
    *,
    feature_names: Sequence[str],
    scored_trials: Sequence[ScoredTrial],
    params: Mapping[str, Any],
    smoothing_windows: int,
) -> dict[str, dict[str, Any]]:
    fit_trials, validation_trials, split_payload = _split_gate_fit_validation_trials(scored_trials)
    rows_by_freq = _trial_frequency_specific_rows_for_gate(
        model,
        fit_trials,
        feature_names=feature_names,
        smoothing_windows=max(1, int(smoothing_windows)),
    )
    feature_index = {name: int(index) for index, name in enumerate(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES)}
    payloads: dict[str, dict[str, Any]] = {}
    for freq in model.freqs:
        freq_key = _freq_label(freq)
        freq_rows = rows_by_freq.get(freq_key, [])
        positives = [row["x"] for row in freq_rows if bool(row.get("positive"))]
        idle_negatives = [row["x"] for row in freq_rows if bool(row.get("idle_negative"))]
        hard_negatives = [row["x"] for row in freq_rows if bool(row.get("hard_negative"))]
        negatives = [row["x"] for row in freq_rows if bool(row.get("negative"))]
        if not positives or not negatives:
            payloads[freq_key] = {
                "type": "threshold",
                "status": "fallback_global_lrt",
                "theta_lrt_f": float(model.lrt_window_floor_th or model.lrt_window_th),
                "theta_score_f": 0.0,
                "theta_margin_f": 0.0,
                "theta_ratio_f": 0.0,
                "theta_entropy_f": 1.0,
                "theta_multiwindow_same_freq_count": 1.0,
                "positive_windows": int(len(positives)),
                "negative_windows": int(len(negatives)),
                "hard_negative_windows": int(len(hard_negatives)),
                "fit_split": "calibration_gate_fit_trials",
                "validation_split": "calibration_gate_validation_trials",
                "gate_fit_validation": dict(split_payload),
            }
            continue
        positive = np.vstack(positives).astype(np.float64, copy=False)
        negative = np.vstack(negatives).astype(np.float64, copy=False)
        idle_negative = np.vstack(idle_negatives).astype(np.float64, copy=False) if idle_negatives else negative
        hard_negative = np.vstack(hard_negatives).astype(np.float64, copy=False) if hard_negatives else idle_negative
        margin_q = float(params.get("margin_idle_quantile", 0.95))
        ratio_q = float(params.get("ratio_idle_quantile", 0.95))
        entropy_q = float(params.get("entropy_control_quantile", 0.85))
        ns2_factor = float(params.get("ns2_safety_factor", 1.0))
        theta_general_idle = max(
            float(model.lrt_window_th),
            _safe_quantile(idle_negative[:, feature_index["lrt_evidence"]], 0.95, float(model.lrt_window_th)),
        )
        theta_ns2 = _safe_quantile(
            hard_negative[:, feature_index["lrt_evidence"]],
            0.95,
            theta_general_idle,
        )
        theta_lrt = max(theta_general_idle, theta_ns2 * ns2_factor)
        theta_score = max(
            _safe_quantile(positive[:, feature_index["selected_freq_score"]], 0.05, 0.0),
            _safe_quantile(idle_negative[:, feature_index["selected_freq_score"]], 0.90, 0.0),
        )
        theta_margin = max(
            _safe_quantile(positive[:, feature_index["margin"]], 0.05, 0.0),
            _safe_quantile(idle_negative[:, feature_index["margin"]], margin_q, 0.0),
        )
        theta_ratio = max(
            1.0,
            _safe_quantile(idle_negative[:, feature_index["ratio"]], ratio_q, 1.0),
        )
        theta_entropy = min(
            _safe_quantile(positive[:, feature_index["score_entropy"]], entropy_q, 1.0),
            _safe_quantile(idle_negative[:, feature_index["score_entropy"]], 0.50, 1.0),
        )
        payloads[freq_key] = {
            "type": "threshold",
            "status": "ok",
            "feature_names": list(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES),
            "theta_lrt_f": float(theta_lrt),
            "theta_general_idle_f": float(theta_general_idle),
            "theta_ns2_f": float(theta_ns2),
            "theta_score_f": float(theta_score),
            "theta_margin_f": float(theta_margin),
            "theta_ratio_f": float(theta_ratio),
            "theta_entropy_f": float(theta_entropy),
            "theta_multiwindow_same_freq_count": 1.0,
            "margin_idle_quantile": margin_q,
            "ratio_idle_quantile": ratio_q,
            "entropy_control_quantile": entropy_q,
            "ns2_safety_factor": ns2_factor,
            "positive_windows": int(len(positives)),
            "negative_windows": int(len(negatives)),
            "idle_negative_windows": int(len(idle_negatives)),
            "hard_negative_windows": int(len(hard_negatives)),
            "fit_split": "calibration_gate_fit_trials",
            "validation_split": "calibration_gate_validation_trials",
            "gate_fit_validation": dict(split_payload),
        }
    return payloads


def _frequency_specific_validation_metrics_payload(
    model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int,
) -> dict[str, Any]:
    if not scored_trials:
        return {
            "supported": False,
            "reason": "no_calibration_gate_validation_trials",
            "rank_key": [],
        }
    bundle = _evaluate_fbcca_lda5_model(
        model,
        scored_trials,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
    )
    payload = _evaluation_payload(bundle)
    payload["no_control_subtype_metrics"] = _evaluate_no_control_subtypes_from_cache(
        model,
        scored_trials,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
    )
    metrics = _extract_row_metrics(payload)
    rank_key = _classifier_threshold_rank_key(
        metrics,
        policy=str(getattr(model, "gate_policy", DEFAULT_CLASSIFIER_THRESHOLD_POLICY)),
    )
    return {
        "supported": True,
        "split": "calibration_gate_validation_trials",
        "trial_count": int(len(scored_trials)),
        "metrics": metrics,
        "rank_key": [float(value) for value in rank_key],
    }


def _fit_frequency_specific_logistic_gate_payload(
    model: FBCCALDA5Model | FBCCARidge5Model,
    *,
    feature_names: Sequence[str],
    scored_trials: Sequence[ScoredTrial],
    params: Mapping[str, Any],
    smoothing_windows: int,
) -> dict[str, dict[str, Any]]:
    fit_trials, validation_trials, split_payload = _split_gate_fit_validation_trials(scored_trials)
    rows_by_freq = _trial_frequency_specific_rows_for_gate(
        model,
        fit_trials,
        feature_names=feature_names,
        smoothing_windows=max(1, int(smoothing_windows)),
    )
    payloads: dict[str, dict[str, Any]] = {}
    for freq in model.freqs:
        freq_key = _freq_label(freq)
        freq_rows = rows_by_freq.get(freq_key, [])
        usable = [row for row in freq_rows if bool(row.get("positive")) or bool(row.get("negative"))]
        positive_count = int(sum(1 for row in usable if bool(row.get("positive"))))
        negative_count = int(sum(1 for row in usable if bool(row.get("negative"))))
        if positive_count <= 0 or negative_count <= 0:
            payloads[freq_key] = {
                "type": "logistic",
                "status": "fallback_global_lrt",
                "prob_threshold": float(params.get("prob_threshold", 0.5)),
                "positive_windows": positive_count,
                "negative_windows": negative_count,
                "fit_split": "calibration_gate_fit_trials",
                "validation_split": "calibration_gate_validation_trials",
                "gate_fit_validation": dict(split_payload),
            }
            continue
        x = np.vstack([np.asarray(row["x"], dtype=np.float64) for row in usable])
        y = np.asarray([1.0 if bool(row.get("positive")) else 0.0 for row in usable], dtype=np.float64)
        sample_weights = np.ones(int(y.shape[0]), dtype=np.float64)
        ns2_weight = float(params.get("ns2_sample_weight", 1.0))
        for index, row in enumerate(usable):
            if row.get("subtype") == "ns2":
                sample_weights[index] *= ns2_weight
            if bool(row.get("hard_negative")):
                sample_weights[index] *= max(ns2_weight, 1.0)
        weights, mean, std = _fit_logistic_binary_ridge(
            x,
            y,
            l2=1.0,
            max_iter=100,
            sample_weights=sample_weights,
        )
        payloads[freq_key] = {
            "type": "logistic",
            "status": "ok",
            "feature_names": list(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES),
            "weights": _array_payload(weights),
            "feature_mean": _array_payload(mean),
            "feature_std": _array_payload(std),
            "prob_threshold": float(params.get("prob_threshold", 0.5)),
            "ns2_sample_weight": ns2_weight,
            "positive_windows": positive_count,
            "negative_windows": negative_count,
            "hard_negative_windows": int(sum(1 for row in usable if bool(row.get("hard_negative")))),
            "fit_split": "calibration_gate_fit_trials",
            "validation_split": "calibration_gate_validation_trials",
            "gate_fit_validation": dict(split_payload),
        }
    return payloads


def _fit_tenp5_ns2_hard_negative_veto_payload(
    model: FBCCALDA5Model | FBCCARidge5Model,
    *,
    feature_names: Sequence[str],
    scored_trials: Sequence[ScoredTrial],
    params: Mapping[str, Any],
    smoothing_windows: int,
) -> dict[str, dict[str, Any]]:
    fit_trials, _validation_trials, split_payload = _split_gate_fit_validation_trials(scored_trials)
    rows_by_freq = _trial_frequency_specific_rows_for_gate(
        model,
        fit_trials,
        feature_names=feature_names,
        smoothing_windows=max(1, int(smoothing_windows)),
    )
    freq_key = TENP5_NS2_VETO_FREQ_KEY
    freq_rows = list(rows_by_freq.get(freq_key, []) or [])
    usable = [
        row
        for row in freq_rows
        if bool(row.get("baseline_window_pass"))
        and (bool(row.get("positive")) or bool(row.get("hard_negative")))
    ]
    positive_rows = [row for row in usable if bool(row.get("positive"))]
    ns2_rows = [row for row in usable if bool(row.get("hard_negative"))]
    payload: dict[str, Any] = {
        "type": "ns2_hard_negative_veto",
        "status": "unsupported_missing_training_class",
        "feature_names": list(TENP5_NS2_VETO_FEATURE_NAMES),
        "source_feature_names": list(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES),
        "target_frequency": freq_key,
        "veto_threshold": float(params.get("veto_threshold", 0.5)),
        "ns2_weight": float(params.get("ns2_weight", 1.0)),
        "positive_windows": int(len(positive_rows)),
        "ns2_hard_negative_windows": int(len(ns2_rows)),
        "optional_negative_windows": int(sum(1 for row in freq_rows if bool(row.get("optional_negative")))),
        "baseline_10p5_pass_windows": int(sum(1 for row in freq_rows if bool(row.get("baseline_window_pass")))),
        "ns1_ns3_excluded": True,
        "fit_split": "calibration_gate_fit_trials",
        "validation_split": "calibration_gate_validation_trials",
        "gate_fit_validation": dict(split_payload),
    }
    if not positive_rows or not ns2_rows:
        return {freq_key: payload}
    x = np.vstack([_tenp5_veto_feature_row(np.asarray(row["x"], dtype=np.float64)) for row in usable])
    y = np.asarray([1.0 if bool(row.get("hard_negative")) else 0.0 for row in usable], dtype=np.float64)
    sample_weights = np.where(y >= 0.5, float(params.get("ns2_weight", 1.0)), 1.0).astype(np.float64, copy=False)
    weights, mean, std = _fit_logistic_binary_ridge(
        x,
        y,
        l2=1.0,
        max_iter=100,
        sample_weights=sample_weights,
    )
    payload.update(
        {
            "status": "ok",
            "weights": _array_payload(weights),
            "feature_mean": _array_payload(mean),
            "feature_std": _array_payload(std),
            "training_target": "1=NS2_hard_negative,0=10.5_command_TP",
        }
    )
    return {freq_key: payload}


def _with_conditional_frequency_specific_payload(
    payloads: Mapping[str, Mapping[str, Any]],
    model: FBCCALDA5Model | FBCCARidge5Model,
    *,
    feature_names: Sequence[str],
    scored_trials: Sequence[ScoredTrial],
    params: Mapping[str, Any],
    smoothing_windows: int,
) -> dict[str, dict[str, Any]]:
    fit_trials, _validation_trials, split_payload = _split_gate_fit_validation_trials(scored_trials)
    rows_by_freq = _trial_frequency_specific_rows_for_gate(
        model,
        fit_trials,
        feature_names=feature_names,
        smoothing_windows=max(1, int(smoothing_windows)),
    )
    feature_index = {name: int(index) for index, name in enumerate(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES)}
    risk_freq_raw = str(params.get("conditional_risk_freqs", "") or "").strip()
    risk_freqs = {
        _freq_label(float(item))
        for item in re.split(r"[,;]", risk_freq_raw)
        if str(item).strip()
    } if risk_freq_raw else set()
    updated: dict[str, dict[str, Any]] = {}
    for freq in model.freqs:
        freq_key = _freq_label(freq)
        payload = dict(dict(payloads).get(freq_key, {}) or {})
        positives = [
            np.asarray(row.get("x"), dtype=np.float64)
            for row in rows_by_freq.get(freq_key, [])
            if bool(row.get("positive"))
        ]
        positive = np.vstack(positives).astype(np.float64, copy=False) if positives else np.zeros((0, 0), dtype=np.float64)

        def q(feature_name: str, quantile_key: str, fallback: float) -> float:
            index = feature_index[feature_name]
            if positive.size <= 0 or positive.shape[1] <= index:
                return float(fallback)
            return _safe_quantile(positive[:, index], float(params.get(quantile_key, 0.5)), float(fallback))

        lrt_fallback = float(model.lrt_window_th)
        payload.update(
            {
                "conditional_policy": str(params.get("conditional_policy", "balanced")),
                "conditional_fit_policy": "calibration_only_positive_command_windows",
                "conditional_fit_split": "calibration_gate_fit_trials",
                "conditional_gate_fit_validation": dict(split_payload),
                "conditional_positive_windows": int(len(positives)),
                "conditional_low_risk_lrt_th": float(
                    max(
                        float(model.lrt_window_th),
                        q("lrt_evidence", "conditional_low_risk_lrt_quantile", lrt_fallback),
                    )
                ),
                "conditional_low_risk_margin_th": float(q("margin", "conditional_low_risk_margin_quantile", 0.0)),
                "conditional_low_risk_ratio_th": float(q("ratio", "conditional_low_risk_ratio_quantile", 1.0)),
                "conditional_low_risk_entropy_th": float(q("score_entropy", "conditional_low_risk_entropy_quantile", 1.0)),
                "conditional_low_risk_same_freq_count": float(params.get("conditional_low_risk_same_freq_count", 2.0)),
                "conditional_high_risk_lrt_th": float(
                    max(
                        float(model.lrt_window_th),
                        q("lrt_evidence", "conditional_high_risk_lrt_quantile", lrt_fallback),
                    )
                ),
                "conditional_high_risk_margin_th": float(q("margin", "conditional_high_risk_margin_quantile", 0.0)),
                "conditional_high_risk_ratio_th": float(q("ratio", "conditional_high_risk_ratio_quantile", 1.0)),
                "conditional_high_risk_entropy_th": float(q("score_entropy", "conditional_high_risk_entropy_quantile", 1.0)),
                "conditional_high_risk_same_freq_count": float(params.get("conditional_high_risk_same_freq_count", 1.0)),
                "conditional_extra_windows": max(0, int(params.get("conditional_extra_windows", 0))),
                "conditional_applies": bool(not risk_freqs or freq_key in risk_freqs),
                "conditional_risk_freqs": sorted(risk_freqs),
            }
        )
        updated[freq_key] = payload
    return updated


def _apply_gate_variant_to_model(
    model: FBCCALDA5Model | FBCCARidge5Model,
    *,
    feature_names: Sequence[str],
    grouped_features: Mapping[str, np.ndarray],
    scored_trials: Sequence[ScoredTrial] = (),
    params: Mapping[str, Any],
    win_sec: float = DEFAULT_CLASSIFIER_WIN_SEC_CANDIDATES[0],
    step_sec: float = DEFAULT_STEP_SEC,
    min_enter_windows: int = 1,
    max_gap_windows: int = 0,
) -> FBCCALDA5Model | FBCCARidge5Model:
    variant = parse_classifier_gate_variant(params.get("gate_variant"))
    if variant == CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW:
        return replace(model, gate_variant=variant)
    control = np.asarray(grouped_features.get("control", np.zeros((0, 0))), dtype=np.float64)
    idle = np.asarray(grouped_features.get("idle", np.zeros((0, 0))), dtype=np.float64)
    if control.size <= 0 or idle.size <= 0:
        raise ValueError(f"{variant} requires calibration control and idle features")
    if variant == CLASSIFIER_GATE_VARIANT_LRTMW_MARGIN:
        margin_index = _feature_index(feature_names, "margin")
        ratio_index = _feature_index(feature_names, "ratio")
        margin_th = max(
            float(np.quantile(control[:, margin_index], float(params["margin_control_quantile"]))),
            float(np.quantile(idle[:, margin_index], float(params["margin_idle_quantile"]))),
        )
        ratio_th = float(np.quantile(idle[:, ratio_index], float(params["ratio_idle_quantile"])))
        return replace(
            model,
            gate_variant=variant,
            score_shape_margin_index=margin_index,
            score_shape_ratio_index=ratio_index,
            score_shape_margin_th=float(margin_th),
            score_shape_ratio_th=float(ratio_th),
            fit_summary={
                **model.fit_summary,
                "gate_variant": variant,
                "gate_variant_params": dict(params),
                "score_shape_margin_th": float(margin_th),
                "score_shape_ratio_th": float(ratio_th),
            },
        )
    if variant == CLASSIFIER_GATE_VARIANT_LRTMW_ENTROPY:
        entropy_index = _feature_index(feature_names, "score_entropy")
        entropy_th = min(
            float(np.quantile(control[:, entropy_index], float(params["entropy_control_quantile"]))),
            float(np.quantile(idle[:, entropy_index], float(params["entropy_idle_quantile"]))),
        )
        return replace(
            model,
            gate_variant=variant,
            score_shape_entropy_index=entropy_index,
            score_shape_entropy_th=float(entropy_th),
            fit_summary={
                **model.fit_summary,
                "gate_variant": variant,
                "gate_variant_params": dict(params),
                "score_shape_entropy_th": float(entropy_th),
            },
        )
    if variant == CLASSIFIER_GATE_VARIANT_SUBJECT_THRESHOLD_FLOOR:
        evidence = _lrt_window_evidence_from_features(model, idle)
        global_floor = float(np.quantile(evidence, float(params["global_floor_quantile"])))
        subject_floor = float(np.quantile(evidence, float(params["subject_idle_quantile"])))
        floor_th = max(global_floor, subject_floor, float(model.lrt_window_th))
        return replace(
            model,
            gate_variant=variant,
            lrt_window_floor_th=float(floor_th),
            fit_summary={
                **model.fit_summary,
                "gate_variant": variant,
                "gate_variant_params": dict(params),
                "subject_floor_global_lrt_quantile": float(params["global_floor_quantile"]),
                "subject_floor_idle_lrt_quantile": float(params["subject_idle_quantile"]),
                "subject_floor_global_lrt_th": float(global_floor),
                "subject_floor_idle_lrt_th": float(subject_floor),
                "lrt_window_floor_th": float(floor_th),
                "fit_split": "calibration_blocks",
                "threshold_fit_split": "calibration_blocks",
                "test_split": "holdout_blocks",
            },
        )
    if variant == CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE:
        idle_evidence = _lrt_window_evidence_from_features(model, idle)
        global_floor = float(np.quantile(idle_evidence, float(params["global_floor_quantile"])))
        subject_floor = float(np.quantile(idle_evidence, float(params["subject_idle_quantile"])))
        ns2 = np.asarray(grouped_features.get("ns2", np.zeros((0, 0))), dtype=np.float64)
        ns2_source = ns2 if ns2.size > 0 else idle
        ns2_scores = _lrt_window_evidence_from_features(model, ns2_source)
        ns2_threshold = float(np.quantile(ns2_scores, 0.95)) if ns2_scores.size else float(model.lrt_window_th)
        ns2_floor = ns2_threshold * float(params["ns2_safety_factor"])
        floor_th = max(float(model.lrt_window_th), global_floor, subject_floor, ns2_floor)
        return replace(
            model,
            gate_variant=variant,
            lrt_window_floor_th=float(floor_th),
            fit_summary={
                **model.fit_summary,
                "gate_variant": variant,
                "gate_variant_params": dict(params),
                "subject_floor_global_lrt_quantile": float(params["global_floor_quantile"]),
                "subject_floor_idle_lrt_quantile": float(params["subject_idle_quantile"]),
                "subject_floor_global_lrt_th": float(global_floor),
                "subject_floor_idle_lrt_th": float(subject_floor),
                "ns2_threshold_source": "calibration_ns2" if ns2.size > 0 else "calibration_idle",
                "ns2_lrt_window_p95": float(ns2_threshold),
                "ns2_safety_factor": float(params["ns2_safety_factor"]),
                "ns2_lrt_window_floor_th": float(ns2_floor),
                "lrt_window_floor_th": float(floor_th),
                "fit_split": "calibration_blocks",
                "threshold_fit_split": "calibration_blocks",
                "test_split": "holdout_blocks",
            },
        )
    if variant == CLASSIFIER_GATE_VARIANT_NS2_AWARE:
        ns2 = np.asarray(grouped_features.get("ns2", np.zeros((0, 0))), dtype=np.float64)
        source = ns2 if ns2.size > 0 else idle
        ns2_scores = _lrt_window_evidence_from_features(model, source)
        ns2_threshold = float(np.quantile(ns2_scores, 0.95)) if ns2_scores.size else float(model.lrt_window_th)
        floor_th = max(float(model.lrt_window_th), ns2_threshold * float(params["ns2_safety_factor"]))
        return replace(
            model,
            gate_variant=variant,
            lrt_window_floor_th=float(floor_th),
            fit_summary={
                **model.fit_summary,
                "gate_variant": variant,
                "gate_variant_params": dict(params),
                "ns2_threshold_source": "calibration_ns2" if ns2.size > 0 else "calibration_idle",
                "ns2_lrt_window_p95": float(ns2_threshold),
                "lrt_window_floor_th": float(floor_th),
                "fit_split": "calibration_blocks",
                "threshold_fit_split": "calibration_blocks",
                "test_split": "holdout_blocks",
            },
        )
    if variant == CLASSIFIER_GATE_VARIANT_WEAK_SUBJECT_GUARD:
        control_scores = _lrt_window_evidence_from_features(model, control)
        idle_scores = _lrt_window_evidence_from_features(model, idle)
        reasons: list[str] = []
        separation = float(np.median(control_scores) - np.quantile(idle_scores, 0.95))
        if separation < 0.25:
            reasons.append("poor_lrt_separation")
        if float(np.mean(idle_scores >= float(model.lrt_window_th))) > 0.10:
            reasons.append("calibration_idle_fp_estimate_high")
        margin_index = _feature_index(feature_names, "margin")
        if float(np.quantile(control[:, margin_index], 0.10)) <= float(np.quantile(idle[:, margin_index], 0.95)):
            reasons.append("margin_overlap")
        if not reasons:
            return replace(
                model,
                gate_variant=variant,
                weak_subject_guard_active=False,
                weak_subject_guard_reasons=(),
                fit_summary={**model.fit_summary, "gate_variant": variant, "weak_subject_guard_active": False},
            )
        floor_th = max(float(model.lrt_window_th), float(np.quantile(idle_scores, 0.975)))
        margin_th = max(
            float(np.quantile(control[:, margin_index], 0.15)),
            float(np.quantile(idle[:, margin_index], 0.95)),
        )
        return replace(
            model,
            gate_variant=variant,
            lrt_window_floor_th=float(floor_th),
            score_shape_margin_index=margin_index,
            score_shape_margin_th=float(margin_th),
            weak_subject_guard_active=True,
            weak_subject_guard_reasons=tuple(reasons),
            fit_summary={
                **model.fit_summary,
                "gate_variant": variant,
                "weak_subject_guard_active": True,
                "weak_subject_guard_reasons": list(reasons),
                "lrt_window_floor_th": float(floor_th),
                "score_shape_margin_th": float(margin_th),
            },
        )
    if variant == CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD:
        _fit_trials, validation_trials, validation_split = _split_gate_fit_validation_trials(scored_trials)
        payloads = _fit_frequency_specific_threshold_gate_payload(
            model,
            feature_names=feature_names,
            scored_trials=scored_trials,
            params=params,
            smoothing_windows=int(getattr(model, "smoothing_windows", 1)),
        )
        updated = replace(
            model,
            gate_variant=variant,
            frequency_specific_control_state_gates=payloads,
        )
        validation_metrics = _frequency_specific_validation_metrics_payload(
            updated,
            validation_trials,
            win_sec=float(win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
        )
        return replace(
            updated,
            fit_summary={
                **model.fit_summary,
                "gate_variant": variant,
                "gate_variant_params": dict(params),
                "frequency_specific_control_state_gates": payloads,
                "frequency_specific_gate_feature_names": list(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES),
                "frequency_specific_fit_policy": "calibration_only_selected_freq_windows",
                "frequency_specific_grid_selection_policy": FREQSPEC_GRID_SELECTION_POLICY,
                "fit_split": "calibration_blocks",
                "validation_split": "calibration_gate_validation_trials",
                "gate_fit_validation": dict(validation_split),
                "gate_validation_metrics": validation_metrics,
            },
        )
    if variant in {
        CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
        CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC,
    }:
        _fit_trials, validation_trials, validation_split = _split_gate_fit_validation_trials(scored_trials)
        payloads = _fit_frequency_specific_logistic_gate_payload(
            model,
            feature_names=feature_names,
            scored_trials=scored_trials,
            params=params,
            smoothing_windows=int(getattr(model, "smoothing_windows", 1)),
        )
        if variant == CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC:
            payloads = _with_conditional_frequency_specific_payload(
                payloads,
                model,
                feature_names=feature_names,
                scored_trials=scored_trials,
                params=params,
                smoothing_windows=int(getattr(model, "smoothing_windows", 1)),
            )
        updated = replace(
            model,
            gate_variant=variant,
            frequency_specific_control_state_gates=payloads,
        )
        validation_metrics = _frequency_specific_validation_metrics_payload(
            updated,
            validation_trials,
            win_sec=float(win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
        )
        return replace(
            updated,
            fit_summary={
                **model.fit_summary,
                "gate_variant": variant,
                "gate_variant_params": dict(params),
                "frequency_specific_control_state_gates": payloads,
                "frequency_specific_gate_feature_names": list(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES),
                "frequency_specific_fit_policy": (
                    "calibration_only_selected_freq_windows_conditional_selective"
                    if variant == CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC
                    else "calibration_only_selected_freq_windows"
                ),
                "frequency_specific_grid_selection_policy": FREQSPEC_GRID_SELECTION_POLICY,
                "fit_split": "calibration_blocks",
                "validation_split": "calibration_gate_validation_trials",
                "gate_fit_validation": dict(validation_split),
                "gate_validation_metrics": validation_metrics,
            },
        )
    if variant == CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO:
        _fit_trials, validation_trials, validation_split = _split_gate_fit_validation_trials(scored_trials)
        payloads = _fit_tenp5_ns2_hard_negative_veto_payload(
            model,
            feature_names=feature_names,
            scored_trials=scored_trials,
            params=params,
            smoothing_windows=int(getattr(model, "smoothing_windows", 1)),
        )
        updated = replace(
            model,
            gate_variant=variant,
            frequency_specific_control_state_gates=payloads,
        )
        validation_metrics = _frequency_specific_validation_metrics_payload(
            updated,
            validation_trials,
            win_sec=float(win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
        )
        return replace(
            updated,
            fit_summary={
                **model.fit_summary,
                "gate_variant": variant,
                "gate_variant_params": dict(params),
                "frequency_specific_control_state_gates": payloads,
                "frequency_specific_gate_feature_names": list(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES),
                "tenp5_ns2_veto_feature_names": list(TENP5_NS2_VETO_FEATURE_NAMES),
                "tenp5_ns2_veto_fit_policy": "calibration_only_baseline_10p5_tp_vs_ns2_windows",
                "fit_split": "calibration_blocks",
                "validation_split": "calibration_gate_validation_trials",
                "gate_fit_validation": dict(validation_split),
                "gate_validation_metrics": validation_metrics,
            },
        )
    return replace(model, gate_variant=variant)


def _lrt_window_evidence_from_features(
    model: FBCCALDA5Model | FBCCARidge5Model,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    return lrt_window_evidence_from_state(_classifier_state_payload(model), feature_matrix)


def _score_shape_gate_mask_for_model(
    model: FBCCALDA5Model | FBCCARidge5Model,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    features = np.asarray(feature_matrix, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("feature_matrix must be 2D")
    mask = np.ones(int(features.shape[0]), dtype=bool)
    margin_index = getattr(model, "score_shape_margin_index", None)
    margin_th = getattr(model, "score_shape_margin_th", None)
    if margin_index is not None and margin_th is not None:
        mask &= features[:, int(margin_index)] + 1e-12 >= float(margin_th)
    ratio_index = getattr(model, "score_shape_ratio_index", None)
    ratio_th = getattr(model, "score_shape_ratio_th", None)
    if ratio_index is not None and ratio_th is not None:
        mask &= features[:, int(ratio_index)] + 1e-12 >= float(ratio_th)
    entropy_index = getattr(model, "score_shape_entropy_index", None)
    entropy_th = getattr(model, "score_shape_entropy_th", None)
    if entropy_index is not None and entropy_th is not None:
        mask &= features[:, int(entropy_index)] <= float(entropy_th) + 1e-12
    return mask


def _predict_tenp5_ns2_hard_negative_veto_trial_from_probs(
    model: FBCCALDA5Model | FBCCARidge5Model,
    probs: np.ndarray,
    labels: np.ndarray,
    window_evidence: np.ndarray,
    *,
    min_enter_windows: int,
    max_gap_windows: int = 0,
    feature_matrix: Optional[np.ndarray] = None,
) -> tuple[str, float, float]:
    baseline_model = replace(
        model,
        gate_variant=CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
        frequency_specific_control_state_gates=None,
        lrt_window_floor_th=None,
    )
    baseline_label, baseline_confidence, baseline_index = _predict_lrt_multiwindow_reject_trial_from_probs(
        baseline_model,
        probs,
        labels,
        window_evidence,
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
        feature_matrix=feature_matrix,
    )
    if _label_to_freq_key(baseline_label) != TENP5_NS2_VETO_FREQ_KEY:
        return baseline_label, baseline_confidence, baseline_index
    if feature_matrix is None:
        return baseline_label, baseline_confidence, baseline_index
    payload = _frequency_specific_gate_payload_for_freq(model, TENP5_NS2_VETO_FREQ_KEY)
    if (
        str(payload.get("type", "")) != "ns2_hard_negative_veto"
        or str(payload.get("status", "")) != "ok"
    ):
        return baseline_label, baseline_confidence, baseline_index

    values = np.asarray(probs, dtype=np.float64)
    evidence = np.asarray(window_evidence, dtype=np.float64).reshape(-1)
    label_values = np.asarray(labels, dtype=object)
    idle_index = int(np.where(label_values == "idle")[0][0])
    window_th = float(getattr(model, "lrt_window_th", 0.0))
    floor_th = getattr(model, "lrt_window_floor_th", None)
    effective_window_th = max(window_th, float(floor_th) if floor_th is not None else window_th)
    enter_th = float(getattr(model, "lrt_enter_th", 0.0))
    decay = min(max(float(getattr(model, "lrt_decay", DEFAULT_LRT_MULTIWINDOW_DECAY)), 0.0), 0.99)
    needed = max(1, int(min_enter_windows))
    max_gap = max(0, int(max_gap_windows))
    feature_names = _classifier_feature_names(
        model.freqs,
        score_source_name=str(dict(model.fit_summary).get("score_source_name", "fbcca")),
        score_bank_mode=str(dict(model.fit_summary).get("score_bank_mode", "full_reference_bank")),
    )
    gate_features, pred_indices, _meta_rows = _frequency_specific_gate_features_for_trial(
        model=model,
        item=ScoredTrial(
            trial=TrialSpec(label="", expected_freq=None, trial_id=0, block_index=0),
            score_matrix=np.zeros((int(values.shape[0]), len(model.freqs)), dtype=np.float64),
            feature_matrix=np.asarray(feature_matrix, dtype=np.float64),
            duration_sec=0.0,
        ),
        probs=values,
        labels=label_values,
        lrt_evidence=evidence,
        feature_names=feature_names,
        smoothing_windows=int(getattr(model, "smoothing_windows", 1)),
        score_source_name=str(dict(model.fit_summary).get("score_source_name", "fbcca")),
    )
    shape_mask = _score_shape_gate_mask_for_model(baseline_model, np.asarray(feature_matrix, dtype=np.float64))
    target_label = TENP5_NS2_VETO_FREQ_KEY
    accumulated = 0.0
    streak_count = 0
    gap_count = 0
    best_score = 0.0
    best_index = 0
    for index, row in enumerate(values):
        pred_index = int(pred_indices[index])
        pred_label = str(label_values[pred_index])
        command_confidence = float(1.0 - row[idle_index])
        evidence_value = float(evidence[index])
        passes_baseline_10p5 = (
            pred_label == target_label
            and command_confidence + 1e-12 >= float(model.command_confidence_th)
            and evidence_value + 1e-12 >= effective_window_th
            and bool(shape_mask[index])
        )
        passes_veto = False
        if passes_baseline_10p5:
            veto_row = _tenp5_veto_feature_row(gate_features[index])
            veto_probability = _logistic_payload_probability(payload, veto_row)
            passes_veto = bool(
                veto_probability is None
                or veto_probability < float(payload.get("veto_threshold", 0.5)) - 1e-12
            )
        if passes_baseline_10p5 and passes_veto:
            increment = max(0.0, evidence_value - window_th)
            if enter_th <= 1e-12:
                increment = max(increment, evidence_value)
            accumulated = float(accumulated + increment)
            streak_count += 1
            gap_count = 0
            if evidence_value > best_score:
                best_score = evidence_value
                best_index = int(index)
            if streak_count >= needed and (enter_th <= 1e-12 or accumulated + 1e-12 >= enter_th):
                return target_label, evidence_value, float(index)
        elif streak_count and gap_count < max_gap:
            accumulated *= decay
            gap_count += 1
        else:
            accumulated *= decay
            streak_count = 0
            gap_count = 0
    if needed <= 1 and best_score + 1e-12 >= effective_window_th:
        return target_label, best_score, float(best_index)
    return "idle", 0.0, 0.0


def _predict_lrt_multiwindow_reject_trial_from_probs(
    model: FBCCALDA5Model | FBCCARidge5Model,
    probs: np.ndarray,
    labels: np.ndarray,
    window_evidence: np.ndarray,
    *,
    min_enter_windows: int,
    max_gap_windows: int = 0,
    feature_matrix: Optional[np.ndarray] = None,
) -> tuple[str, float, float]:
    if probs.size <= 0:
        return "idle", 0.0, float("inf")
    if (
        parse_classifier_gate_variant(getattr(model, "gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW))
        == CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO
    ):
        return _predict_tenp5_ns2_hard_negative_veto_trial_from_probs(
            model,
            probs,
            labels,
            window_evidence,
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            feature_matrix=feature_matrix,
        )
    values = np.asarray(probs, dtype=np.float64)
    evidence = np.asarray(window_evidence, dtype=np.float64).reshape(-1)
    if evidence.shape[0] != values.shape[0]:
        raise ValueError("lrt evidence count must match classifier windows")
    label_values = np.asarray(labels, dtype=object)
    idle_index = int(np.where(label_values == "idle")[0][0])
    needed = max(1, int(min_enter_windows))
    max_gap = max(0, int(max_gap_windows))
    window_th = float(getattr(model, "lrt_window_th", 0.0))
    floor_th = getattr(model, "lrt_window_floor_th", None)
    effective_window_th = max(window_th, float(floor_th) if floor_th is not None else window_th)
    enter_th = float(getattr(model, "lrt_enter_th", 0.0))
    decay = min(max(float(getattr(model, "lrt_decay", DEFAULT_LRT_MULTIWINDOW_DECAY)), 0.0), 0.99)
    shape_mask = np.ones(int(values.shape[0]), dtype=bool)
    if feature_matrix is not None:
        shape_mask = _score_shape_gate_mask_for_model(model, feature_matrix)
        if getattr(model, "frequency_specific_control_state_gates", None):
            shape_mask &= _frequency_specific_gate_mask_for_model(
                model,
                probs=values,
                labels=label_values,
                feature_matrix=feature_matrix,
                lrt_evidence=evidence,
            )
    conditional_extra_by_window = np.zeros(int(values.shape[0]), dtype=int)
    if (
        feature_matrix is not None
        and getattr(model, "frequency_specific_control_state_gates", None)
        and parse_classifier_gate_variant(getattr(model, "gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW))
        == CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC
    ):
        feature_names = _classifier_feature_names(
            model.freqs,
            score_bank_mode=str(getattr(model, "fit_summary", {}).get("score_bank_mode", "full_reference_bank")),
        )
        gate_features, pred_indices_for_gate, meta_rows = _frequency_specific_gate_features_for_trial(
            model=model,
            item=ScoredTrial(
                trial=TrialSpec(label="", expected_freq=None, trial_id=0, block_index=0),
                score_matrix=np.zeros((int(values.shape[0]), len(model.freqs)), dtype=np.float64),
                feature_matrix=feature_matrix,
                duration_sec=0.0,
            ),
            probs=values,
            labels=label_values,
            lrt_evidence=evidence,
            feature_names=feature_names,
            smoothing_windows=int(getattr(model, "smoothing_windows", 1)),
        )
        payloads = normalize_frequency_specific_control_state_gates(
            getattr(model, "frequency_specific_control_state_gates", None)
        )
        for row_index, meta in enumerate(meta_rows):
            if int(pred_indices_for_gate[row_index]) == idle_index:
                continue
            payload = dict(payloads.get(str(meta.get("freq_key")), {}) or {})
            if not payload:
                continue
            risk_level = _conditional_frequency_specific_risk_level(payload, gate_features[row_index])
            if risk_level == "medium" and _frequency_specific_payload_pass(payload, gate_features[row_index]):
                conditional_extra_by_window[row_index] = max(0, int(payload.get("conditional_extra_windows", 0)))
    accumulated_by_label = {str(label): 0.0 for label in label_values if str(label) != "idle"}
    streak_label = ""
    streak_count = 0
    gap_count = 0
    best_label = "idle"
    best_score = 0.0
    best_index = 0
    for index, row in enumerate(values):
        pred_index = int(np.argmax(row))
        pred_label = str(label_values[pred_index])
        command_confidence = float(1.0 - row[idle_index])
        evidence_value = float(evidence[index])
        passes_command = (
            pred_label != "idle"
            and command_confidence + 1e-12 >= float(model.command_confidence_th)
        )
        passes_lrt = evidence_value + 1e-12 >= effective_window_th and bool(shape_mask[index])
        if passes_command and passes_lrt:
            for label in list(accumulated_by_label):
                if label != pred_label:
                    accumulated_by_label[label] *= decay
            increment = max(0.0, evidence_value - window_th)
            if enter_th <= 1e-12:
                increment = max(increment, evidence_value)
            accumulated_by_label[pred_label] = float(accumulated_by_label.get(pred_label, 0.0) + increment)
            if evidence_value > best_score:
                best_label = pred_label
                best_score = evidence_value
                best_index = int(index)
            if pred_label == streak_label:
                streak_count += 1
            else:
                streak_label = pred_label
                streak_count = 1
            gap_count = 0
            needed_for_window = needed + int(conditional_extra_by_window[index])
            if streak_count >= needed_for_window and (
                enter_th <= 1e-12 or accumulated_by_label[pred_label] + 1e-12 >= enter_th
            ):
                return pred_label, evidence_value, float(index)
        elif streak_label and gap_count < max_gap:
            for label in list(accumulated_by_label):
                accumulated_by_label[label] *= decay
            gap_count += 1
        else:
            for label in list(accumulated_by_label):
                accumulated_by_label[label] *= decay
            streak_label = ""
            streak_count = 0
            gap_count = 0
    best_extra_windows = (
        int(conditional_extra_by_window[int(best_index)])
        if 0 <= int(best_index) < int(conditional_extra_by_window.shape[0])
        else 0
    )
    if needed + best_extra_windows <= 1 and best_label != "idle" and best_score + 1e-12 >= effective_window_th:
        return best_label, best_score, float(best_index)
    return "idle", 0.0, 0.0


def _window_pass_mask_for_lrt_model(
    model: FBCCALDA5Model | FBCCARidge5Model,
    *,
    probs: np.ndarray,
    labels: np.ndarray,
    lrt_evidence: np.ndarray,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    label_values = np.asarray(labels, dtype=object)
    if values.ndim != 2:
        return np.zeros(0, dtype=bool)
    idle_index = int(np.where(label_values == "idle")[0][0])
    window_th = float(getattr(model, "lrt_window_th", 0.0))
    floor_th = getattr(model, "lrt_window_floor_th", None)
    effective_window_th = max(window_th, float(floor_th) if floor_th is not None else window_th)
    shape_mask = _score_shape_gate_mask_for_model(model, feature_matrix)
    if getattr(model, "frequency_specific_control_state_gates", None):
        shape_mask &= _frequency_specific_gate_mask_for_model(
            model,
            probs=values,
            labels=label_values,
            feature_matrix=feature_matrix,
            lrt_evidence=lrt_evidence,
        )
    return (
        (np.argmax(values, axis=1) != idle_index)
        & (np.asarray(lrt_evidence, dtype=np.float64).reshape(-1) >= effective_window_th)
        & shape_mask
    )


def _window_output_labels_from_pass_mask(
    probs: np.ndarray,
    labels: np.ndarray,
    pass_mask: np.ndarray,
) -> list[str]:
    label_values = np.asarray(labels, dtype=object)
    pred_indices = np.argmax(np.asarray(probs, dtype=np.float64), axis=1)
    mask = np.asarray(pass_mask, dtype=bool).reshape(-1)
    return [
        str(label_values[int(index)]) if row_index < mask.shape[0] and bool(mask[row_index]) else "idle"
        for row_index, index in enumerate(pred_indices)
    ]


def _predict_lrt_trial_with_pass_mask(
    model: FBCCALDA5Model | FBCCARidge5Model,
    probs: np.ndarray,
    labels: np.ndarray,
    lrt_evidence: np.ndarray,
    pass_mask: np.ndarray,
    *,
    min_enter_windows: int,
    max_gap_windows: int = 0,
) -> tuple[str, float, float]:
    values = np.asarray(probs, dtype=np.float64)
    if values.size <= 0:
        return "idle", 0.0, float("inf")
    label_values = np.asarray(labels, dtype=object)
    idle_index = int(np.where(label_values == "idle")[0][0])
    evidence = np.asarray(lrt_evidence, dtype=np.float64).reshape(-1)
    mask = np.asarray(pass_mask, dtype=bool).reshape(-1)
    needed = max(1, int(min_enter_windows))
    max_gap = max(0, int(max_gap_windows))
    window_th = float(getattr(model, "lrt_window_th", 0.0))
    enter_th = float(getattr(model, "lrt_enter_th", 0.0))
    decay = min(max(float(getattr(model, "lrt_decay", DEFAULT_LRT_MULTIWINDOW_DECAY)), 0.0), 0.99)
    accumulated_by_label = {str(label): 0.0 for label in label_values if str(label) != "idle"}
    streak_label = ""
    streak_count = 0
    gap_count = 0
    best_label = "idle"
    best_score = 0.0
    best_index = 0
    for index, row in enumerate(values):
        pred_index = int(np.argmax(row))
        pred_label = str(label_values[pred_index])
        command_confidence = float(1.0 - row[idle_index])
        passes_command = (
            pred_label != "idle"
            and command_confidence + 1e-12 >= float(model.command_confidence_th)
        )
        passes_lrt = bool(index < mask.shape[0] and mask[index])
        evidence_value = float(evidence[index]) if index < evidence.shape[0] else 0.0
        if passes_command and passes_lrt:
            for label in list(accumulated_by_label):
                if label != pred_label:
                    accumulated_by_label[label] *= decay
            increment = max(0.0, evidence_value - window_th)
            if enter_th <= 1e-12:
                increment = max(increment, evidence_value)
            accumulated_by_label[pred_label] = float(accumulated_by_label.get(pred_label, 0.0) + increment)
            if evidence_value > best_score:
                best_label = pred_label
                best_score = evidence_value
                best_index = int(index)
            if pred_label == streak_label:
                streak_count += 1
            else:
                streak_label = pred_label
                streak_count = 1
            gap_count = 0
            if streak_count >= needed and (
                enter_th <= 1e-12 or accumulated_by_label[pred_label] + 1e-12 >= enter_th
            ):
                return pred_label, evidence_value, float(index)
        elif streak_label and gap_count < max_gap:
            for label in list(accumulated_by_label):
                accumulated_by_label[label] *= decay
            gap_count += 1
        else:
            for label in list(accumulated_by_label):
                accumulated_by_label[label] *= decay
            streak_label = ""
            streak_count = 0
            gap_count = 0
    if needed <= 1 and best_label != "idle":
        return best_label, best_score, float(best_index)
    return "idle", 0.0, 0.0


def _nc_calibrated_pass_mask(
    model: FBCCALDA5Model | FBCCARidge5Model,
    item: ScoredTrial,
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: Sequence[str],
    lrt_evidence: np.ndarray,
    nc_gate_type: str,
    nc_payload: Mapping[str, Any],
    nc_thresholds: Mapping[str, Any],
    min_enter_windows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    baseline_model = replace(
        model,
        gate_variant=CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
        frequency_specific_control_state_gates=None,
        lrt_window_floor_th=None,
    )
    baseline_mask = _window_pass_mask_for_lrt_model(
        baseline_model,
        probs=probs,
        labels=labels,
        lrt_evidence=lrt_evidence,
        feature_matrix=item.feature_matrix,
    )
    resolved_gate = _parse_nc_gate_type(nc_gate_type)
    if resolved_gate == NC_GATE_BASELINE_LRT_THRESHOLD:
        candidate_mask = _window_pass_mask_for_lrt_model(
            model,
            probs=probs,
            labels=labels,
            lrt_evidence=lrt_evidence,
            feature_matrix=item.feature_matrix,
        )
        veto_count = int(np.sum(baseline_mask & ~candidate_mask))
        return candidate_mask.astype(bool, copy=False), np.ones(int(candidate_mask.shape[0]), dtype=np.float64), {
            "baseline_pass_windows": float(np.sum(baseline_mask)),
            "candidate_pass_windows": float(np.sum(candidate_mask)),
            "detector_veto_windows": float(veto_count),
            "low_risk_bypass_windows": 0.0,
        }
    nc_features, _pred_indices, _meta_rows = _nc_csns_feature_matrix_for_trial(
        baseline_model,
        item,
        probs,
        labels,
        lrt_evidence,
        feature_names=feature_names,
        smoothing_windows=max(1, int(getattr(model, "smoothing_windows", 1))),
    )
    cs_prob = _nc_csns_probability(nc_payload, nc_features)
    detector_pass = cs_prob + 1e-12 >= float(dict(nc_payload).get("prob_threshold", NC_CSNS_PROB_THRESHOLD))
    if resolved_gate == NC_GATE_CONDITIONAL_SESSION_LOGISTIC:
        low_risk = _nc_conditional_low_risk(nc_features, nc_thresholds)
        candidate_mask = baseline_mask & (low_risk | detector_pass)
        low_risk_count = int(np.sum(baseline_mask & low_risk))
    else:
        candidate_mask = baseline_mask & detector_pass
        low_risk_count = 0
    veto_count = int(np.sum(baseline_mask & ~candidate_mask))
    return candidate_mask.astype(bool, copy=False), cs_prob, {
        "baseline_pass_windows": float(np.sum(baseline_mask)),
        "candidate_pass_windows": float(np.sum(candidate_mask)),
        "detector_veto_windows": float(veto_count),
        "low_risk_bypass_windows": float(low_risk_count),
    }


def _cs_probability_for_frequency_specific_row(
    payload: Mapping[str, Any],
    row: np.ndarray,
) -> Optional[float]:
    if str(dict(payload).get("type", "")) != "logistic":
        return None
    weights = np.asarray(dict(payload).get("weights", []), dtype=np.float64).reshape(-1)
    mean = np.asarray(dict(payload).get("feature_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(dict(payload).get("feature_std", []), dtype=np.float64).reshape(-1)
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    if weights.size != values.size + 1 or mean.size != values.size or std.size != values.size:
        return None
    z = (values - mean) / np.maximum(std, 1e-9)
    return float(_softmax_2class_logit(np.asarray([weights[0] + z @ weights[1:]], dtype=np.float64))[0])


def _transition_type_for_trace(true_label: str, baseline_pred: str, candidate_pred: str) -> str:
    true_label = str(true_label)
    baseline_pred = str(baseline_pred)
    candidate_pred = str(candidate_pred)
    if true_label != "idle" and baseline_pred == true_label and candidate_pred == true_label:
        return "baseline_TP_candidate_TP"
    if true_label != "idle" and baseline_pred == true_label and candidate_pred == "idle":
        return "baseline_TP_candidate_idle"
    if true_label == "idle" and baseline_pred != "idle" and candidate_pred == "idle":
        return "baseline_FP_candidate_idle"
    if true_label == "idle" and baseline_pred != "idle" and candidate_pred != "idle":
        return "baseline_FP_candidate_FP"
    if true_label == "idle" and baseline_pred == "idle" and candidate_pred != "idle":
        return "baseline_idle_candidate_command"
    if true_label != "idle" and baseline_pred != true_label and candidate_pred == true_label:
        return "baseline_miss_candidate_TP"
    if true_label != "idle" and candidate_pred == "idle":
        return "candidate_idle_command_miss"
    if true_label != "idle" and candidate_pred != true_label:
        return "candidate_wrong_command"
    return "unchanged_or_other"


def _trace_rows_for_frequency_specific_logistic_case(
    *,
    baseline_model: FBCCARidge5Model,
    candidate_model: FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    dataset: str,
    subject: str,
    split_index: int,
    recipe_id: str,
    frequency_profile: str,
    frequency_set_id: str,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    fit_summary = dict(getattr(candidate_model, "fit_summary", {}) or {})
    candidate_variant = parse_classifier_gate_variant(
        fit_summary.get(
            "gate_variant",
            getattr(candidate_model, "gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW),
        )
    )
    if candidate_variant not in {
        CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
        CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC,
    }:
        return {}
    feature_names = _classifier_feature_names(
        candidate_model.freqs,
        score_source_name=str(dict(candidate_model.fit_summary).get("score_source_name", "fbcca")),
        score_bank_mode=str(dict(candidate_model.fit_summary).get("score_bank_mode", "full_reference_bank")),
    )
    name_to_index = {name: int(index) for index, name in enumerate(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES)}
    gates = normalize_frequency_specific_control_state_gates(
        getattr(candidate_model, "frequency_specific_control_state_gates", None)
        or fit_summary.get("frequency_specific_control_state_gates")
    )
    window_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    transition_counts_by_subject: dict[tuple[str, str], int] = defaultdict(int)
    transition_counts_by_frequency: dict[tuple[str, str], int] = defaultdict(int)
    feature_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for item in scored_trials:
        true_label = _trial_true_label(item.trial)
        true_freq = "" if true_label == "idle" else true_label
        subtype = _ysuan_ns_subtype_from_label(str(item.trial.label))
        candidate_probs, labels = _predict_classifier_windows(candidate_model, item.feature_matrix)
        candidate_probs = _smooth_classifier_probabilities(
            candidate_probs,
            smoothing_windows=int(getattr(candidate_model, "smoothing_windows", 1)),
        )
        baseline_probs, baseline_labels = _predict_classifier_windows(baseline_model, item.feature_matrix)
        baseline_probs = _smooth_classifier_probabilities(
            baseline_probs,
            smoothing_windows=int(getattr(baseline_model, "smoothing_windows", 1)),
        )
        lrt_evidence = _lrt_window_evidence_from_features(candidate_model, item.feature_matrix)
        baseline_lrt_evidence = _lrt_window_evidence_from_features(baseline_model, item.feature_matrix)
        baseline_pass = _window_pass_mask_for_lrt_model(
            baseline_model,
            probs=baseline_probs,
            labels=baseline_labels,
            lrt_evidence=baseline_lrt_evidence,
            feature_matrix=item.feature_matrix,
        )
        candidate_pass = _window_pass_mask_for_lrt_model(
            candidate_model,
            probs=candidate_probs,
            labels=labels,
            lrt_evidence=lrt_evidence,
            feature_matrix=item.feature_matrix,
        )
        baseline_window_preds = _window_output_labels_from_pass_mask(baseline_probs, baseline_labels, baseline_pass)
        candidate_window_preds = _window_output_labels_from_pass_mask(candidate_probs, labels, candidate_pass)
        gate_features, pred_indices, meta_rows = _frequency_specific_gate_features_for_trial(
            model=candidate_model,
            item=item,
            probs=candidate_probs,
            labels=labels,
            lrt_evidence=lrt_evidence,
            feature_names=feature_names,
            smoothing_windows=int(getattr(candidate_model, "smoothing_windows", 1)),
            score_source_name=str(dict(candidate_model.fit_summary).get("score_source_name", "fbcca")),
        )
        label_values = np.asarray(labels, dtype=object)
        baseline_trial_pred, _base_conf, baseline_first_index = _predict_lrt_multiwindow_reject_trial_from_probs(
            baseline_model,
            baseline_probs,
            baseline_labels,
            baseline_lrt_evidence,
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            feature_matrix=item.feature_matrix,
        )
        candidate_trial_pred, _cand_conf, candidate_first_index = _predict_lrt_multiwindow_reject_trial_from_probs(
            candidate_model,
            candidate_probs,
            labels,
            lrt_evidence,
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            feature_matrix=item.feature_matrix,
        )
        trial_transition = _transition_type_for_trace(true_label, baseline_trial_pred, candidate_trial_pred)
        baseline_decision_time = (
            float(win_sec) + float(baseline_first_index) * float(step_sec)
            if baseline_trial_pred != "idle"
            else float(item.duration_sec + win_sec)
            if true_label != "idle"
            else float(win_sec)
        )
        candidate_decision_time = (
            float(win_sec) + float(candidate_first_index) * float(step_sec)
            if candidate_trial_pred != "idle"
            else float(item.duration_sec + win_sec)
            if true_label != "idle"
            else float(win_sec)
        )
        trial_rows.append(
            {
                "dataset": str(dataset),
                "subject": str(subject),
                "split_id": int(split_index),
                "trial_id": int(item.trial.trial_id),
                "block_index": int(item.trial.block_index),
                "true_state": subtype or true_label,
                "true_freq": true_freq,
                "baseline_pred": baseline_trial_pred,
                "candidate_pred": candidate_trial_pred,
                "baseline_decision_time_s": float(baseline_decision_time),
                "candidate_decision_time_s": float(candidate_decision_time),
                "transition_type": trial_transition,
                "recipe_id": str(recipe_id),
                "frequency_profile": str(frequency_profile),
                "frequency_set_id": str(frequency_set_id),
            }
        )
        transition_counts_by_subject[(str(subject), trial_transition)] += 1
        transition_counts_by_frequency[(true_label if true_label != "idle" else subtype or "idle", trial_transition)] += 1

        for row_index, row in enumerate(gate_features):
            pred_label = str(label_values[int(pred_indices[row_index])])
            freq_key = _label_to_freq_key(pred_label)
            payload = dict(gates.get(str(freq_key), {}) or {})
            cs_probability = _cs_probability_for_frequency_specific_row(payload, row)
            selected_scores = np.sort(np.asarray(item.score_matrix[row_index], dtype=np.float64))[::-1]
            top3_score = float(selected_scores[2]) if selected_scores.size >= 3 else None
            baseline_pred = baseline_window_preds[row_index] if row_index < len(baseline_window_preds) else "idle"
            candidate_pred = candidate_window_preds[row_index] if row_index < len(candidate_window_preds) else "idle"
            transition_type = _transition_type_for_trace(true_label, baseline_pred, candidate_pred)
            payload_row = {
                "dataset": str(dataset),
                "subject": str(subject),
                "split_id": int(split_index),
                "trial_id": int(item.trial.trial_id),
                "block_index": int(item.trial.block_index),
                "window_idx": int(row_index),
                "time_from_onset": float(row_index) * float(step_sec),
                "true_state": subtype or true_label,
                "true_freq": true_freq,
                "selected_freq": str(freq_key or "idle"),
                "baseline_pred": baseline_pred,
                "candidate_pred": candidate_pred,
                "top1_score": float(row[name_to_index["top1_score"]]),
                "top2_score": float(row[name_to_index["top2_score"]]),
                "top3_score": top3_score,
                "selected_freq_score": float(row[name_to_index["selected_freq_score"]]),
                "margin": float(row[name_to_index["margin"]]),
                "ratio": float(row[name_to_index["ratio"]]),
                "normalized_top1": float(row[name_to_index["normalized_top1"]]),
                "score_entropy": float(row[name_to_index["score_entropy"]]),
                "lrt_evidence": float(row[name_to_index["lrt_evidence"]]),
                "multiwindow_same_freq_count": float(row[name_to_index["multiwindow_same_freq_count"]]),
                "multiwindow_margin_mean": float(row[name_to_index["multiwindow_margin_mean"]]),
                "multiwindow_entropy_mean": float(row[name_to_index["multiwindow_entropy_mean"]]),
                "cs_probability": cs_probability,
                "gate_pass": bool(row_index < candidate_pass.shape[0] and candidate_pass[row_index]),
                "transition_type": transition_type,
                "recipe_id": str(recipe_id),
                "frequency_profile": str(frequency_profile),
                "frequency_set_id": str(frequency_set_id),
            }
            window_rows.append(payload_row)
            if transition_type in {
                "baseline_TP_candidate_TP",
                "baseline_TP_candidate_idle",
                "baseline_FP_candidate_idle",
                "baseline_FP_candidate_FP",
                "baseline_idle_candidate_command",
                "baseline_miss_candidate_TP",
            }:
                feature_groups[(transition_type, str(freq_key or "idle"))].append(payload_row)

    subject_rows = [
        {"subject": subject_key, "transition_type": transition, "count": count}
        for (subject_key, transition), count in sorted(transition_counts_by_subject.items())
    ]
    frequency_rows = [
        {"frequency_or_state": freq, "transition_type": transition, "count": count}
        for (freq, transition), count in sorted(transition_counts_by_frequency.items())
    ]
    feature_summary_rows: list[dict[str, Any]] = []
    for (transition, freq), items in sorted(feature_groups.items()):
        feature_summary_rows.append(
            {
                "transition_type": transition,
                "selected_freq": freq,
                "count": int(len(items)),
                "mean_selected_freq_score": _safe_mean([item.get("selected_freq_score") for item in items]),
                "mean_top1_score": _safe_mean([item.get("top1_score") for item in items]),
                "mean_top2_score": _safe_mean([item.get("top2_score") for item in items]),
                "mean_top3_score": _safe_mean([item.get("top3_score") for item in items]),
                "mean_margin": _safe_mean([item.get("margin") for item in items]),
                "mean_ratio": _safe_mean([item.get("ratio") for item in items]),
                "mean_normalized_top1": _safe_mean([item.get("normalized_top1") for item in items]),
                "mean_score_entropy": _safe_mean([item.get("score_entropy") for item in items]),
                "mean_lrt_evidence": _safe_mean([item.get("lrt_evidence") for item in items]),
                "mean_multiwindow_same_freq_count": _safe_mean(
                    [item.get("multiwindow_same_freq_count") for item in items]
                ),
                "mean_multiwindow_margin_mean": _safe_mean(
                    [item.get("multiwindow_margin_mean") for item in items]
                ),
                "mean_multiwindow_entropy_mean": _safe_mean(
                    [item.get("multiwindow_entropy_mean") for item in items]
                ),
                "mean_cs_probability": _safe_mean([item.get("cs_probability") for item in items]),
                "gate_pass_rate": _safe_mean([1.0 if item.get("gate_pass") else 0.0 for item in items]),
            }
        )
    return {
        "logistic_trace_windows": window_rows,
        "logistic_trace_trial_summary": trial_rows,
        "logistic_transition_counts_by_subject": subject_rows,
        "logistic_transition_counts_by_frequency": frequency_rows,
        "logistic_feature_summary_tp_fp": feature_summary_rows,
    }


def _trace_rows_for_tenp5_ns2_veto_case(
    *,
    baseline_model: FBCCARidge5Model,
    candidate_model: FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    dataset: str,
    subject: str,
    split_index: int,
    recipe_id: str,
    frequency_profile: str,
    frequency_set_id: str,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int = 0,
) -> dict[str, Any]:
    candidate_variant = parse_classifier_gate_variant(
        getattr(candidate_model, "gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW)
    )
    if candidate_variant != CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO:
        return {}
    fit_summary = dict(getattr(candidate_model, "fit_summary", {}) or {})
    gates = normalize_frequency_specific_control_state_gates(
        getattr(candidate_model, "frequency_specific_control_state_gates", None)
        or fit_summary.get("frequency_specific_control_state_gates")
    )
    payload = dict(gates.get(TENP5_NS2_VETO_FREQ_KEY, {}) or {})
    if str(payload.get("type", "")) != "ns2_hard_negative_veto":
        return {}
    feature_names = _classifier_feature_names(
        candidate_model.freqs,
        score_source_name=str(dict(candidate_model.fit_summary).get("score_source_name", "fbcca")),
        score_bank_mode=str(dict(candidate_model.fit_summary).get("score_bank_mode", "full_reference_bank")),
    )
    name_to_index = {name: int(index) for index, name in enumerate(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES)}
    diagnostics: list[dict[str, Any]] = []
    fixed_ns2_fp_count = 0
    lost_command_tp_count = 0
    baseline_tenp5_ns2_fp_count = 0
    candidate_tenp5_ns2_fp_count = 0
    baseline_tenp5_command_tp_count = 0
    candidate_tenp5_command_tp_count = 0

    for item in scored_trials:
        true_label = _trial_true_label(item.trial)
        subtype = _ysuan_ns_subtype_from_label(str(item.trial.label))
        candidate_probs, labels = _predict_classifier_windows(candidate_model, item.feature_matrix)
        candidate_probs = _smooth_classifier_probabilities(
            candidate_probs,
            smoothing_windows=int(getattr(candidate_model, "smoothing_windows", 1)),
        )
        baseline_probs, baseline_labels = _predict_classifier_windows(baseline_model, item.feature_matrix)
        baseline_probs = _smooth_classifier_probabilities(
            baseline_probs,
            smoothing_windows=int(getattr(baseline_model, "smoothing_windows", 1)),
        )
        evidence = _lrt_window_evidence_from_features(candidate_model, item.feature_matrix)
        baseline_evidence = _lrt_window_evidence_from_features(baseline_model, item.feature_matrix)
        baseline_pass = _window_pass_mask_for_lrt_model(
            baseline_model,
            probs=baseline_probs,
            labels=baseline_labels,
            lrt_evidence=baseline_evidence,
            feature_matrix=item.feature_matrix,
        )
        candidate_pass = _window_pass_mask_for_lrt_model(
            candidate_model,
            probs=candidate_probs,
            labels=labels,
            lrt_evidence=evidence,
            feature_matrix=item.feature_matrix,
        )
        label_values = np.asarray(labels, dtype=object)
        gate_features, pred_indices, _meta_rows = _frequency_specific_gate_features_for_trial(
            model=candidate_model,
            item=item,
            probs=candidate_probs,
            labels=labels,
            lrt_evidence=evidence,
            feature_names=feature_names,
            smoothing_windows=int(getattr(candidate_model, "smoothing_windows", 1)),
            score_source_name=str(dict(candidate_model.fit_summary).get("score_source_name", "fbcca")),
        )
        for row_index, row in enumerate(gate_features):
            pred_label = str(label_values[int(pred_indices[row_index])])
            selected_freq = _label_to_freq_key(pred_label)
            baseline_10p5_pass = bool(
                row_index < baseline_pass.shape[0]
                and baseline_pass[row_index]
                and selected_freq == TENP5_NS2_VETO_FREQ_KEY
            )
            if not baseline_10p5_pass:
                continue
            veto_row = _tenp5_veto_feature_row(row)
            veto_probability = _logistic_payload_probability(payload, veto_row)
            vetoed = bool(
                str(payload.get("status", "")) == "ok"
                and veto_probability is not None
                and veto_probability + 1e-12 >= float(payload.get("veto_threshold", 0.5))
            )
            candidate_10p5_pass = bool(
                row_index < candidate_pass.shape[0]
                and candidate_pass[row_index]
                and selected_freq == TENP5_NS2_VETO_FREQ_KEY
            )
            if subtype == "ns2":
                baseline_tenp5_ns2_fp_count += 1
                if candidate_10p5_pass:
                    candidate_tenp5_ns2_fp_count += 1
                if vetoed:
                    fixed_ns2_fp_count += 1
            if true_label == TENP5_NS2_VETO_FREQ_KEY:
                baseline_tenp5_command_tp_count += 1
                if candidate_10p5_pass:
                    candidate_tenp5_command_tp_count += 1
                if vetoed:
                    lost_command_tp_count += 1
            selected_scores = np.sort(np.asarray(item.score_matrix[row_index], dtype=np.float64))[::-1]
            diagnostics.append(
                {
                    "dataset": str(dataset),
                    "subject": str(subject),
                    "split_id": int(split_index),
                    "trial_id": int(item.trial.trial_id),
                    "block_index": int(item.trial.block_index),
                    "window_idx": int(row_index),
                    "time_from_onset": float(row_index) * float(step_sec),
                    "true_state": subtype or true_label,
                    "true_freq": "" if true_label == "idle" else true_label,
                    "selected_freq": str(selected_freq or "idle"),
                    "baseline_10p5_pass": baseline_10p5_pass,
                    "candidate_10p5_pass": candidate_10p5_pass,
                    "veto_probability": _finite_or_none(veto_probability),
                    "veto_threshold": _finite_or_none(payload.get("veto_threshold")),
                    "vetoed": vetoed,
                    "fixed_ns2_fp": bool(subtype == "ns2" and vetoed),
                    "lost_command_tp": bool(true_label == TENP5_NS2_VETO_FREQ_KEY and vetoed),
                    "top1_score": float(row[name_to_index["top1_score"]]),
                    "top2_score": float(row[name_to_index["top2_score"]]),
                    "top3_score": float(selected_scores[2]) if selected_scores.size >= 3 else None,
                    "selected_freq_score": float(row[name_to_index["selected_freq_score"]]),
                    "margin": float(row[name_to_index["margin"]]),
                    "ratio": float(row[name_to_index["ratio"]]),
                    "score_entropy": float(row[name_to_index["score_entropy"]]),
                    "lrt_evidence": float(row[name_to_index["lrt_evidence"]]),
                    "multiwindow_same_freq_count": float(row[name_to_index["multiwindow_same_freq_count"]]),
                    "multiwindow_margin_mean": float(row[name_to_index["multiwindow_margin_mean"]]),
                    "recipe_id": str(recipe_id),
                    "frequency_profile": str(frequency_profile),
                    "frequency_set_id": str(frequency_set_id),
                }
            )
    precision_den = fixed_ns2_fp_count + lost_command_tp_count
    summary_row = {
        "dataset": str(dataset),
        "subject": str(subject),
        "split_id": int(split_index),
        "recipe_id": str(recipe_id),
        "frequency_profile": str(frequency_profile),
        "frequency_set_id": str(frequency_set_id),
        "veto_status": str(payload.get("status", "")),
        "positive_windows": int(payload.get("positive_windows", 0) or 0),
        "ns2_hard_negative_windows": int(payload.get("ns2_hard_negative_windows", 0) or 0),
        "veto_threshold": _finite_or_none(payload.get("veto_threshold")),
        "ns2_weight": _finite_or_none(payload.get("ns2_weight")),
        "fixed_ns2_fp_count": int(fixed_ns2_fp_count),
        "lost_command_tp_count": int(lost_command_tp_count),
        "baseline_tenp5_ns2_fp_count": int(baseline_tenp5_ns2_fp_count),
        "candidate_tenp5_ns2_fp_count": int(candidate_tenp5_ns2_fp_count),
        "baseline_tenp5_command_tp_count": int(baseline_tenp5_command_tp_count),
        "candidate_tenp5_command_tp_count": int(candidate_tenp5_command_tp_count),
        "veto_precision": float(fixed_ns2_fp_count / precision_den) if precision_den else None,
        "tp_loss_per_fixed_fp": float(lost_command_tp_count / max(fixed_ns2_fp_count, 1)),
    }
    return {
        "tenp5_ns2_veto_diagnostics": diagnostics,
        "tenp5_ns2_veto_summary_rows": [summary_row],
    }


def _predict_adaptive_evidence_trial_from_probs(
    model: FBCCALDA5Model | FBCCARidge5Model,
    probs: np.ndarray,
    labels: np.ndarray,
    gate_probs: np.ndarray,
    *,
    min_enter_windows: int,
    max_gap_windows: int = 0,
) -> tuple[str, float, float]:
    if probs.size <= 0:
        return "idle", 0.0, float("inf")
    values = np.asarray(probs, dtype=np.float64)
    gates = np.asarray(gate_probs, dtype=np.float64).reshape(-1)
    if gates.shape[0] != values.shape[0]:
        raise ValueError("adaptive gate probability count must match classifier windows")
    label_values = np.asarray(labels, dtype=object)
    idle_index = int(np.where(label_values == "idle")[0][0])
    needed = max(1, int(min_enter_windows))
    max_gap = max(0, int(max_gap_windows))
    decay = min(max(float(getattr(model, "evidence_decay", DEFAULT_ADAPTIVE_EVIDENCE_DECAY)), 0.0), 0.99)
    decision_th = float(getattr(model, "evidence_decision_th", 0.5))
    enter_th = float(getattr(model, "evidence_enter_th", 0.0))
    evidence_by_label = {str(label): 0.0 for label in label_values if str(label) != "idle"}
    streak_label = ""
    streak_count = 0
    gap_count = 0
    best_label = "idle"
    best_conf = 0.0
    best_index = 0
    for index, row in enumerate(values):
        pred_index = int(np.argmax(row))
        pred_label = str(label_values[pred_index])
        command_confidence = float(1.0 - row[idle_index])
        gate_prob = float(gates[index])
        passes_command = (
            pred_label != "idle"
            and command_confidence + 1e-12 >= float(model.command_confidence_th)
        )
        passes_gate = gate_prob + 1e-12 >= decision_th
        if passes_command and passes_gate:
            for label in list(evidence_by_label):
                if label != pred_label:
                    evidence_by_label[label] *= decay
            increment = max(0.0, gate_prob - decision_th)
            evidence_by_label[pred_label] = float(evidence_by_label.get(pred_label, 0.0) + increment)
            if gate_prob > best_conf:
                best_label = pred_label
                best_conf = gate_prob
                best_index = int(index)
            if pred_label == streak_label:
                streak_count += 1
            else:
                streak_label = pred_label
                streak_count = 1
            gap_count = 0
            if streak_count >= needed and (
                enter_th <= 1e-12 or evidence_by_label[pred_label] + 1e-12 >= enter_th
            ):
                return pred_label, gate_prob, float(index)
        elif streak_label and gap_count < max_gap:
            for label in list(evidence_by_label):
                evidence_by_label[label] *= decay
            gap_count += 1
        else:
            for label in list(evidence_by_label):
                evidence_by_label[label] *= decay
            streak_label = ""
            streak_count = 0
            gap_count = 0
    if needed <= 1 and best_label != "idle" and best_conf >= decision_th:
        return best_label, best_conf, float(best_index)
    return "idle", 0.0, 0.0


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
        if pred_label != "idle" and command_confidence + 1e-12 >= float(model.command_confidence_th):
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
    probs = _smooth_classifier_probabilities(
        probs,
        smoothing_windows=int(getattr(model, "smoothing_windows", 1)),
    )
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
    require_control: bool = True,
) -> None:
    expected_labels = set(_classifier_labels(freqs) if require_control else ("idle",))
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
    score_bank_mode: str = DEFAULT_SCORE_BANK_MODE,
    full_bank_freqs: Sequence[float] = (),
    validate_holdout_control: bool = True,
    decoder_cache: Optional[dict[tuple[Any, ...], Any]] = None,
    scored_cache: Optional[dict[tuple[Any, ...], dict[tuple[Any, ...], ScoredTrial]]] = None,
) -> tuple[list[ScoredTrial], list[ScoredTrial]]:
    mode = _parse_score_bank_mode(score_bank_mode)
    if decoder_cache is not None and scored_cache is not None:
        calibration_scored = _score_segment_subset_cached(
            freqs=freqs,
            sampling_rate=int(sampling_rate),
            step_sec=float(step_sec),
            compute_backend=str(compute_backend),
            gpu_device=int(gpu_device),
            gpu_precision=str(gpu_precision),
            win_sec=float(win_sec),
            score_bank_mode=mode,
            full_bank_freqs=full_bank_freqs,
            segments=calibration_segments,
            context=f"{context} calibration",
            decoder_cache=decoder_cache,
            scored_cache=scored_cache,
        )
        holdout_scored = _score_segment_subset_cached(
            freqs=freqs,
            sampling_rate=int(sampling_rate),
            step_sec=float(step_sec),
            compute_backend=str(compute_backend),
            gpu_device=int(gpu_device),
            gpu_precision=str(gpu_precision),
            win_sec=float(win_sec),
            score_bank_mode=mode,
            full_bank_freqs=full_bank_freqs,
            segments=holdout_segments,
            context=f"{context} holdout",
            decoder_cache=decoder_cache,
            scored_cache=scored_cache,
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
            require_control=bool(validate_holdout_control),
        )
        return calibration_scored, holdout_scored
    decoder = _build_fbcca_decoder_for_scoring(
        freqs=freqs,
        sampling_rate=int(sampling_rate),
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
    )
    full_bank_decoder = None
    full_freqs = tuple(float(freq) for freq in full_bank_freqs)
    if mode == "full_reference_bank":
        full_bank_decoder = _build_fbcca_decoder_for_scoring(
            freqs=full_freqs,
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
        freqs=freqs,
        score_bank_mode=mode,
        full_bank_decoder=full_bank_decoder,
        full_bank_freqs=full_freqs,
    )
    holdout_scored = _score_trials_for_classifier(
        trial_segments=holdout_segments,
        decoder=decoder,
        freqs=freqs,
        score_bank_mode=mode,
        full_bank_decoder=full_bank_decoder,
        full_bank_freqs=full_freqs,
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
        require_control=bool(validate_holdout_control),
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
    score_bank_mode: str = DEFAULT_SCORE_BANK_MODE,
    full_bank_freqs: Sequence[float] = (),
    validate_holdout_control: bool = True,
    decoder_cache: Optional[dict[tuple[Any, ...], Any]] = None,
    scored_cache: Optional[dict[tuple[Any, ...], dict[tuple[Any, ...], ScoredTrial]]] = None,
) -> tuple[list[ScoredTrial], list[ScoredTrial]]:
    spec = _score_method_spec(method_name)
    mode = _parse_score_bank_mode(score_bank_mode)
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
            score_bank_mode=score_bank_mode,
            full_bank_freqs=full_bank_freqs,
            validate_holdout_control=bool(validate_holdout_control),
            decoder_cache=decoder_cache,
            scored_cache=scored_cache,
        )
    if mode != DEFAULT_SCORE_BANK_MODE:
        raise ValueError(
            f"{method_name} does not support score_bank_mode={mode}; "
            "use command_only for template/spatial decoders or fbcca_lda5/fbcca_ridge5 for full_reference_bank"
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
        freqs=freqs,
        score_bank_mode=DEFAULT_SCORE_BANK_MODE,
    )
    holdout_scored = _score_trials_for_classifier(
        trial_segments=holdout_segments,
        decoder=decoder,
        freqs=freqs,
        score_bank_mode=DEFAULT_SCORE_BANK_MODE,
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
        require_control=bool(validate_holdout_control),
    )
    return calibration_scored, holdout_scored


def _fbcca_lda5_threshold_candidates(
    model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    smoothing_windows: int = 1,
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
        probs = _smooth_classifier_probabilities(
            probs,
            smoothing_windows=max(1, int(smoothing_windows)),
        )
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


def _fit_adaptive_evidence_gate(
    base_model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    smoothing_windows: int = 1,
    l2: float = 1.0,
) -> dict[str, Any]:
    probability_cache = _build_classifier_probability_cache(base_model, scored_trials)
    rows: list[np.ndarray] = []
    y: list[np.ndarray] = []
    trial_counts = {"control": 0, "idle": 0}
    window_counts = {"control": 0, "idle": 0}
    for item, probs, labels in probability_cache:
        smoothed = _smooth_classifier_probabilities(
            probs,
            smoothing_windows=max(1, int(smoothing_windows)),
        )
        features = _adaptive_gate_feature_matrix_for_trial(base_model, item, smoothed, labels)
        if features.shape[0] <= 0:
            continue
        label = _trial_true_label(item.trial)
        target_value = 0.0 if label == "idle" else 1.0
        rows.append(features)
        y.append(np.full(int(features.shape[0]), float(target_value), dtype=np.float64))
        key = "idle" if label == "idle" else "control"
        trial_counts[key] += 1
        window_counts[key] += int(features.shape[0])
    if not rows:
        raise ValueError("adaptive evidence gate calibration has no windows")
    x = np.vstack(rows).astype(np.float64, copy=False)
    y_array = np.concatenate(y).astype(np.float64, copy=False)
    if len(set(float(value) for value in y_array.tolist())) < 2:
        raise ValueError("adaptive evidence gate requires control and idle calibration windows")
    weights, mean, std = _fit_logistic_binary_ridge(x, y_array, l2=float(l2))
    logits = np.column_stack([np.ones(int(x.shape[0]), dtype=np.float64), (x - mean) / std]) @ weights
    p = _softmax_2class_logit(logits)
    control_scores = p[y_array >= 0.5]
    idle_scores = p[y_array < 0.5]
    decision_th = float(np.quantile(idle_scores, 0.95)) if idle_scores.size else 0.5
    if control_scores.size:
        decision_th = min(decision_th, float(np.quantile(control_scores, 0.75)))
    decision_th = float(max(min(decision_th, 0.95), 0.05))
    return {
        "evidence_weights": weights,
        "evidence_feature_mean": mean,
        "evidence_feature_std": std,
        "evidence_decision_th": decision_th,
        "fit_summary": {
            "adaptive_gate": CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY,
            "adaptive_gate_feature_names": list(ADAPTIVE_EVIDENCE_FEATURE_NAMES),
            "adaptive_gate_l2": float(l2),
            "adaptive_gate_calibration_windows": int(x.shape[0]),
            "adaptive_gate_trial_counts": trial_counts,
            "adaptive_gate_window_counts": window_counts,
            "adaptive_gate_idle_score_p95": float(np.quantile(idle_scores, 0.95)) if idle_scores.size else None,
            "adaptive_gate_control_score_p25": float(np.quantile(control_scores, 0.25)) if control_scores.size else None,
            "adaptive_gate_control_score_p50": float(np.quantile(control_scores, 0.50)) if control_scores.size else None,
            "adaptive_gate_decision_th_seed": float(decision_th),
        },
    }


def _adaptive_evidence_threshold_candidates(
    gate_payload: Mapping[str, Any],
) -> tuple[float, ...]:
    seed = _safe_float(gate_payload.get("evidence_decision_th"), 0.5)
    values = [float(value) for value in DEFAULT_ADAPTIVE_EVIDENCE_ENTER_CANDIDATES]
    values.extend(
        [
            0.0,
            float(max(seed - 0.20, 0.0)),
            float(max(seed, 0.0)),
            float(max(seed + 0.20, 0.0)),
            float(max(seed * 2.0, 0.0)),
        ]
    )
    rounded = np.unique(np.round(np.asarray(values, dtype=np.float64), 6))
    return tuple(float(value) for value in rounded.tolist())


def _lrt_feature_indices_for_model(model: FBCCALDA5Model | FBCCARidge5Model) -> tuple[int, ...]:
    feature_names = _classifier_feature_names(
        model.freqs,
        score_bank_mode="full_reference_bank",
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
    indices: list[int] = []
    for name in wanted:
        if name in feature_names:
            indices.append(int(feature_names.index(name)))
    if not indices:
        raise ValueError("lrt multi-window gate requires full-reference-bank features")
    return tuple(indices)


def _fit_lrt_multiwindow_reject_gate(
    base_model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    smoothing_windows: int = 1,
) -> dict[str, Any]:
    probability_cache = _build_classifier_probability_cache(base_model, scored_trials)
    feature_indices = _lrt_feature_indices_for_model(base_model)
    control_rows: list[np.ndarray] = []
    idle_rows: list[np.ndarray] = []
    trial_counts = {"control": 0, "idle": 0}
    window_counts = {"control": 0, "idle": 0}
    for item, probs, _labels in probability_cache:
        if item.feature_matrix.shape[0] <= 0:
            continue
        smoothed = _smooth_classifier_probabilities(
            probs,
            smoothing_windows=max(1, int(smoothing_windows)),
        )
        pred_indices = np.argmax(smoothed, axis=1)
        rows = np.asarray(item.feature_matrix, dtype=np.float64)[:, np.asarray(feature_indices, dtype=int)]
        command_rows: list[np.ndarray] = []
        for row_index, pred_index in enumerate(pred_indices):
            true_label = _trial_true_label(item.trial)
            if true_label != "idle" and str(base_model.labels[int(pred_index)]) == true_label:
                command_rows.append(rows[row_index])
        label = _trial_true_label(item.trial)
        if label == "idle":
            idle_rows.append(rows)
            trial_counts["idle"] += 1
            window_counts["idle"] += int(rows.shape[0])
        elif command_rows:
            command_array = np.vstack(command_rows).astype(np.float64, copy=False)
            control_rows.append(command_array)
            trial_counts["control"] += 1
            window_counts["control"] += int(command_array.shape[0])
    if not control_rows or not idle_rows:
        raise ValueError("lrt multi-window gate requires calibration command and idle windows")
    control = np.vstack(control_rows).astype(np.float64, copy=False)
    idle = np.vstack(idle_rows).astype(np.float64, copy=False)
    control_mean = np.mean(control, axis=0)
    idle_mean = np.mean(idle, axis=0)
    control_std = np.maximum(np.std(control, axis=0), 1e-6)
    idle_std = np.maximum(np.std(idle, axis=0), 1e-6)
    seed_model = replace(
        base_model,
        gate_policy=CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        lrt_feature_indices=feature_indices,
        lrt_feature_mean_control=control_mean,
        lrt_feature_std_control=control_std,
        lrt_feature_mean_idle=idle_mean,
        lrt_feature_std_idle=idle_std,
    )
    control_scores = _lrt_window_evidence_from_features(
        replace(seed_model, lrt_feature_indices=tuple(range(len(feature_indices)))),
        control,
    )
    idle_scores = _lrt_window_evidence_from_features(
        replace(seed_model, lrt_feature_indices=tuple(range(len(feature_indices)))),
        idle,
    )
    window_th = float(np.quantile(idle_scores, 0.95)) if idle_scores.size else 0.0
    if control_scores.size:
        window_th = min(window_th, float(np.quantile(control_scores, 0.50)))
    window_th = float(max(window_th, 0.0))
    return {
        "lrt_feature_indices": feature_indices,
        "lrt_feature_names": [
            _classifier_feature_names(base_model.freqs, score_bank_mode="full_reference_bank")[index]
            for index in feature_indices
        ],
        "lrt_feature_mean_control": control_mean,
        "lrt_feature_std_control": control_std,
        "lrt_feature_mean_idle": idle_mean,
        "lrt_feature_std_idle": idle_std,
        "lrt_window_th": window_th,
        "fit_summary": {
            "lrt_multiwindow_gate": CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
            "lrt_feature_names": [
                _classifier_feature_names(base_model.freqs, score_bank_mode="full_reference_bank")[index]
                for index in feature_indices
            ],
            "lrt_calibration_trial_counts": trial_counts,
            "lrt_calibration_window_counts": window_counts,
            "lrt_control_score_p25": float(np.quantile(control_scores, 0.25)) if control_scores.size else None,
            "lrt_control_score_p50": float(np.quantile(control_scores, 0.50)) if control_scores.size else None,
            "lrt_idle_score_p95": float(np.quantile(idle_scores, 0.95)) if idle_scores.size else None,
            "lrt_window_th_seed": float(window_th),
        },
    }


def _lrt_enter_threshold_candidates(gate_payload: Mapping[str, Any]) -> tuple[float, ...]:
    seed = _safe_float(gate_payload.get("lrt_window_th"), 0.0)
    values = [float(value) for value in DEFAULT_LRT_MULTIWINDOW_ENTER_CANDIDATES]
    values.extend(
        [
            0.0,
            float(max(seed, 0.0)),
            float(max(seed * 1.5, 0.0)),
            float(max(seed * 2.0, 0.0)),
        ]
    )
    rounded = np.unique(np.round(np.asarray(values, dtype=np.float64), 6))
    return tuple(float(value) for value in rounded.tolist())


def _select_lrt_multiwindow_reject_gate(
    base_model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int = 0,
    smoothing_windows: int = 1,
    threshold_policy: str = CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
) -> dict[str, Any]:
    gate_payload = _fit_lrt_multiwindow_reject_gate(
        base_model,
        scored_trials,
        smoothing_windows=max(1, int(smoothing_windows)),
    )
    base_window_th = float(gate_payload["lrt_window_th"])
    window_candidates = np.unique(
        np.round(
            np.asarray(
                [
                    0.0,
                    max(base_window_th - 1.0, 0.0),
                    max(base_window_th - 0.5, 0.0),
                    base_window_th,
                    base_window_th + 0.5,
                    base_window_th + 1.0,
                ],
                dtype=np.float64,
            ),
            6,
        )
    )
    enter_candidates = _lrt_enter_threshold_candidates(gate_payload)
    best_model: Optional[FBCCALDA5Model | FBCCARidge5Model] = None
    best_rank: Optional[tuple[float, ...]] = None
    best_bundle: dict[str, Any] = {}
    best_candidate = (0.0, 0.0)
    for window_th in window_candidates:
        for enter_th in enter_candidates:
            model = replace(
                base_model,
                command_confidence_th=0.0,
                smoothing_windows=max(1, int(smoothing_windows)),
                gate_policy=CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
                lrt_feature_indices=tuple(gate_payload["lrt_feature_indices"]),
                lrt_feature_mean_control=np.asarray(gate_payload["lrt_feature_mean_control"], dtype=np.float64),
                lrt_feature_std_control=np.asarray(gate_payload["lrt_feature_std_control"], dtype=np.float64),
                lrt_feature_mean_idle=np.asarray(gate_payload["lrt_feature_mean_idle"], dtype=np.float64),
                lrt_feature_std_idle=np.asarray(gate_payload["lrt_feature_std_idle"], dtype=np.float64),
                lrt_window_th=float(window_th),
                lrt_enter_th=float(enter_th),
                lrt_decay=float(DEFAULT_LRT_MULTIWINDOW_DECAY),
            )
            bundle = _evaluate_fbcca_lda5_model(
                model,
                scored_trials,
                win_sec=float(win_sec),
                step_sec=float(step_sec),
                min_enter_windows=max(1, int(min_enter_windows)),
                max_gap_windows=max(0, int(max_gap_windows)),
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
                tie_breaker=float(window_th + enter_th),
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_model = model
                best_bundle = bundle
                best_candidate = (float(window_th), float(enter_th))
    if best_model is None:
        raise ValueError("lrt multi-window gate could not select a candidate")
    best_async = dict(best_bundle.get("async_metrics") or {})
    best_async5 = dict(best_bundle.get("async_lens_metrics_5class") or {})
    best_fixed5 = dict(best_bundle.get("fixed_window_metrics_5class") or {})
    return {
        "gate_policy": CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        "threshold_policy": _parse_classifier_threshold_policy(threshold_policy),
        "command_confidence_th": 0.0,
        "smoothing_windows": max(1, int(smoothing_windows)),
        "max_gap_windows": max(0, int(max_gap_windows)),
        "candidate_count": int(len(window_candidates) * len(enter_candidates)),
        "candidate_lrt_window_thresholds_preview": [float(value) for value in window_candidates[:12]],
        "candidate_lrt_enter_thresholds_preview": [float(value) for value in enter_candidates[:12]],
        "lrt_feature_indices": tuple(gate_payload["lrt_feature_indices"]),
        "lrt_feature_mean_control": np.asarray(gate_payload["lrt_feature_mean_control"], dtype=np.float64),
        "lrt_feature_std_control": np.asarray(gate_payload["lrt_feature_std_control"], dtype=np.float64),
        "lrt_feature_mean_idle": np.asarray(gate_payload["lrt_feature_mean_idle"], dtype=np.float64),
        "lrt_feature_std_idle": np.asarray(gate_payload["lrt_feature_std_idle"], dtype=np.float64),
        "lrt_window_th": float(best_candidate[0]),
        "lrt_enter_th": float(best_candidate[1]),
        "lrt_decay": float(DEFAULT_LRT_MULTIWINDOW_DECAY),
        "fit_summary": dict(gate_payload.get("fit_summary") or {}),
        "selected_metrics": {
            "idle_fp_per_min": _safe_float(best_async.get("idle_fp_per_min"), float("inf")),
            "idle_selected_windows_per_min": _safe_float(best_async.get("idle_selected_windows_per_min"), float("inf")),
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


def _select_adaptive_evidence_gate(
    base_model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int = 0,
    smoothing_windows: int = 1,
    threshold_policy: str = CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY,
) -> dict[str, Any]:
    gate_payload = _fit_adaptive_evidence_gate(
        base_model,
        scored_trials,
        smoothing_windows=max(1, int(smoothing_windows)),
    )
    best_model: Optional[FBCCALDA5Model | FBCCARidge5Model] = None
    best_rank: Optional[tuple[float, ...]] = None
    best_bundle: dict[str, Any] = {}
    candidates = _adaptive_evidence_threshold_candidates(gate_payload)
    for threshold in candidates:
        model = replace(
            base_model,
            command_confidence_th=0.0,
            smoothing_windows=max(1, int(smoothing_windows)),
            gate_policy=CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY,
            evidence_weights=np.asarray(gate_payload["evidence_weights"], dtype=np.float64),
            evidence_feature_mean=np.asarray(gate_payload["evidence_feature_mean"], dtype=np.float64),
            evidence_feature_std=np.asarray(gate_payload["evidence_feature_std"], dtype=np.float64),
            evidence_decision_th=float(gate_payload["evidence_decision_th"]),
            evidence_enter_th=float(threshold),
            evidence_decay=float(DEFAULT_ADAPTIVE_EVIDENCE_DECAY),
        )
        bundle = _evaluate_fbcca_lda5_model(
            model,
            scored_trials,
            win_sec=float(win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
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
            best_model = model
            best_bundle = bundle
    if best_model is None:
        raise ValueError("adaptive evidence gate could not select a candidate")
    best_async = dict(best_bundle.get("async_metrics") or {})
    best_async5 = dict(best_bundle.get("async_lens_metrics_5class") or {})
    best_fixed5 = dict(best_bundle.get("fixed_window_metrics_5class") or {})
    return {
        "gate_policy": CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY,
        "threshold_policy": _parse_classifier_threshold_policy(threshold_policy),
        "command_confidence_th": 0.0,
        "smoothing_windows": max(1, int(smoothing_windows)),
        "max_gap_windows": max(0, int(max_gap_windows)),
        "candidate_count": int(len(candidates)),
        "candidate_evidence_enter_thresholds_preview": [float(value) for value in candidates[:12]],
        "evidence_weights": np.asarray(best_model.evidence_weights, dtype=np.float64),
        "evidence_feature_mean": np.asarray(best_model.evidence_feature_mean, dtype=np.float64),
        "evidence_feature_std": np.asarray(best_model.evidence_feature_std, dtype=np.float64),
        "evidence_decision_th": float(best_model.evidence_decision_th),
        "evidence_enter_th": float(best_model.evidence_enter_th),
        "evidence_decay": float(best_model.evidence_decay),
        "fit_summary": dict(gate_payload.get("fit_summary") or {}),
        "selected_metrics": {
            "idle_fp_per_min": _safe_float(best_async.get("idle_fp_per_min"), float("inf")),
            "idle_selected_windows_per_min": _safe_float(best_async.get("idle_selected_windows_per_min"), float("inf")),
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


def _subject_adaptive_command_threshold(
    base_model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    smoothing_windows: int = 1,
) -> dict[str, Any]:
    probability_cache = _build_classifier_probability_cache(base_model, scored_trials)
    idle_values: list[float] = []
    control_values: list[float] = []
    for item, probs, labels in probability_cache:
        smoothed = _smooth_classifier_probabilities(
            probs,
            smoothing_windows=max(1, int(smoothing_windows)),
        )
        command_conf = _command_confidence_from_probs(smoothed, labels)
        label = _trial_true_label(item.trial)
        if label == "idle":
            idle_values.extend(float(value) for value in command_conf.tolist())
        else:
            control_values.extend(float(value) for value in command_conf.tolist())
    if not idle_values or not control_values:
        raise ValueError("subject adaptive threshold requires control and idle calibration windows")
    idle_q = float(np.quantile(np.asarray(idle_values, dtype=np.float64), 0.95))
    control_q = float(np.quantile(np.asarray(control_values, dtype=np.float64), 0.25))
    threshold = float(max(0.0, min(1.0, min(max(idle_q, 0.0), max(control_q, idle_q)))))
    return {
        "command_confidence_th": threshold,
        "threshold_policy": CLASSIFIER_SUBJECT_ADAPTIVE_THRESHOLD_POLICY,
        "smoothing_windows": max(1, int(smoothing_windows)),
        "candidate_count": 1,
        "selected_metrics": {},
        "fit_summary": {
            "subject_adaptive_threshold_idle_p95": idle_q,
            "subject_adaptive_threshold_control_p25": control_q,
            "subject_adaptive_threshold": threshold,
            "subject_adaptive_threshold_idle_windows": int(len(idle_values)),
            "subject_adaptive_threshold_control_windows": int(len(control_values)),
        },
    }


def _select_fbcca_lda5_confidence_threshold(
    base_model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int = 0,
    smoothing_windows: int = 1,
    threshold_policy: str = DEFAULT_CLASSIFIER_THRESHOLD_POLICY,
) -> dict[str, Any]:
    normalized_policy = _parse_classifier_threshold_policy(threshold_policy)
    if normalized_policy == CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY:
        return _select_adaptive_evidence_gate(
            base_model,
            scored_trials,
            win_sec=float(win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            smoothing_windows=max(1, int(smoothing_windows)),
            threshold_policy=normalized_policy,
        )
    if normalized_policy == CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY:
        return _select_lrt_multiwindow_reject_gate(
            base_model,
            scored_trials,
            win_sec=float(win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            smoothing_windows=max(1, int(smoothing_windows)),
            threshold_policy=normalized_policy,
        )
    if normalized_policy == CLASSIFIER_SUBJECT_ADAPTIVE_THRESHOLD_POLICY:
        threshold_payload = _subject_adaptive_command_threshold(
            base_model,
            scored_trials,
            smoothing_windows=max(1, int(smoothing_windows)),
        )
        model = replace(
            base_model,
            command_confidence_th=float(threshold_payload.get("command_confidence_th", 0.0)),
            smoothing_windows=max(1, int(smoothing_windows)),
            gate_policy=CLASSIFIER_SUBJECT_ADAPTIVE_THRESHOLD_POLICY,
        )
        bundle = _evaluate_fbcca_lda5_model(
            model,
            scored_trials,
            win_sec=float(win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
        )
        fixed5 = dict(bundle.get("fixed_window_metrics_5class") or {})
        async5 = dict(bundle.get("async_lens_metrics_5class") or {})
        async_metrics = dict(bundle.get("async_metrics") or {})
        threshold_payload["selected_metrics"] = {
            **async_metrics,
            "async_acc_5class": _safe_float(async5.get("acc"), 0.0),
            "async_macro_f1_5class": _safe_float(async5.get("macro_f1"), 0.0),
            "fixed_acc_5class": _safe_float(fixed5.get("acc"), 0.0),
            "fixed_macro_f1_5class": _safe_float(fixed5.get("macro_f1"), 0.0),
        }
        return threshold_payload
    best_threshold = 0.0
    best_rank: Optional[tuple[float, ...]] = None
    best_bundle: dict[str, Any] = {}
    probability_cache = _build_classifier_probability_cache(base_model, scored_trials)
    candidates = _fbcca_lda5_threshold_candidates(
        base_model,
        scored_trials,
        smoothing_windows=max(1, int(smoothing_windows)),
        probability_cache=probability_cache,
    )
    for threshold in candidates:
        model = replace(
            base_model,
            command_confidence_th=float(threshold),
            smoothing_windows=max(1, int(smoothing_windows)),
        )
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
            policy=str(normalized_policy),
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
        "threshold_policy": normalized_policy,
        "max_gap_windows": max(0, int(max_gap_windows)),
        "smoothing_windows": max(1, int(smoothing_windows)),
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
        smoothing_windows=1,
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
                    smoothing_windows=1,
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
    smoothing_windows: int = 1,
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
        smoothing_windows=max(1, int(smoothing_windows)),
        threshold_policy=str(threshold_policy),
    )
    confidence_th = float(threshold_selection.get("command_confidence_th", 0.0))
    gate_policy = str(threshold_selection.get("gate_policy", CLASSIFIER_CONFIDENCE_GATE_POLICY))
    lrt_feature_indices = tuple(
        int(index) for index in threshold_selection.get("lrt_feature_indices", ()) or ()
    )
    return replace(
        model,
        command_confidence_th=max(float(confidence_th), 0.0),
        smoothing_windows=max(1, int(smoothing_windows)),
        gate_policy=gate_policy,
        evidence_weights=(
            np.asarray(threshold_selection["evidence_weights"], dtype=np.float64)
            if "evidence_weights" in threshold_selection
            else None
        ),
        evidence_feature_mean=(
            np.asarray(threshold_selection["evidence_feature_mean"], dtype=np.float64)
            if "evidence_feature_mean" in threshold_selection
            else None
        ),
        evidence_feature_std=(
            np.asarray(threshold_selection["evidence_feature_std"], dtype=np.float64)
            if "evidence_feature_std" in threshold_selection
            else None
        ),
        evidence_decision_th=_safe_float(threshold_selection.get("evidence_decision_th"), 0.0),
        evidence_enter_th=_safe_float(threshold_selection.get("evidence_enter_th"), 0.0),
        evidence_decay=_safe_float(threshold_selection.get("evidence_decay"), DEFAULT_ADAPTIVE_EVIDENCE_DECAY),
        lrt_feature_indices=lrt_feature_indices,
        lrt_feature_mean_control=(
            np.asarray(threshold_selection["lrt_feature_mean_control"], dtype=np.float64)
            if "lrt_feature_mean_control" in threshold_selection
            else None
        ),
        lrt_feature_std_control=(
            np.asarray(threshold_selection["lrt_feature_std_control"], dtype=np.float64)
            if "lrt_feature_std_control" in threshold_selection
            else None
        ),
        lrt_feature_mean_idle=(
            np.asarray(threshold_selection["lrt_feature_mean_idle"], dtype=np.float64)
            if "lrt_feature_mean_idle" in threshold_selection
            else None
        ),
        lrt_feature_std_idle=(
            np.asarray(threshold_selection["lrt_feature_std_idle"], dtype=np.float64)
            if "lrt_feature_std_idle" in threshold_selection
            else None
        ),
        lrt_window_th=_safe_float(threshold_selection.get("lrt_window_th"), 0.0),
        lrt_enter_th=_safe_float(threshold_selection.get("lrt_enter_th"), 0.0),
        lrt_decay=_safe_float(threshold_selection.get("lrt_decay"), DEFAULT_LRT_MULTIWINDOW_DECAY),
        fit_summary={
            **model.fit_summary,
            **dict(threshold_selection.get("fit_summary") or {}),
            "score_source_name": str(score_source_name).strip().lower(),
            "classifier": _classifier_name_for_model(model, score_source_name=score_source_name),
            "gate_policy": gate_policy,
            "command_confidence_th": max(float(confidence_th), 0.0),
            "evidence_decision_th": _safe_float(threshold_selection.get("evidence_decision_th"), 0.0),
            "evidence_enter_th": _safe_float(threshold_selection.get("evidence_enter_th"), 0.0),
            "evidence_decay": _safe_float(threshold_selection.get("evidence_decay"), DEFAULT_ADAPTIVE_EVIDENCE_DECAY),
            "lrt_feature_indices": [int(index) for index in lrt_feature_indices],
            "lrt_window_th": _safe_float(threshold_selection.get("lrt_window_th"), 0.0),
            "lrt_enter_th": _safe_float(threshold_selection.get("lrt_enter_th"), 0.0),
            "lrt_decay": _safe_float(threshold_selection.get("lrt_decay"), DEFAULT_LRT_MULTIWINDOW_DECAY),
            "min_enter_windows": max(1, int(min_enter_windows)),
            "max_gap_windows": max(0, int(max_gap_windows)),
            "smoothing_windows": max(1, int(smoothing_windows)),
            "threshold_policy": _parse_classifier_threshold_policy(threshold_policy),
            "threshold_selection": dict(threshold_selection),
            "score_bank_mode": DEFAULT_SCORE_BANK_MODE,
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
        smoothing_windows=1,
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
                    smoothing_windows=1,
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
    smoothing_windows: int = 1,
    threshold_policy: str = DEFAULT_CLASSIFIER_THRESHOLD_POLICY,
    l2_candidates: Sequence[float] = DEFAULT_RIDGE_L2_CANDIDATES,
    base_models: Optional[Sequence[FBCCARidge5Model]] = None,
    score_source_name: str = "fbcca",
    score_bank_mode: str = DEFAULT_SCORE_BANK_MODE,
    gate_variant_params: Optional[Mapping[str, Any]] = None,
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
            smoothing_windows=max(1, int(smoothing_windows)),
            threshold_policy=str(threshold_policy),
        )
        lrt_feature_indices = tuple(
            int(index) for index in threshold_selection.get("lrt_feature_indices", ()) or ()
        )
        model = replace(
            base_model,
            command_confidence_th=max(float(threshold_selection.get("command_confidence_th", 0.0)), 0.0),
            smoothing_windows=max(1, int(smoothing_windows)),
            gate_policy=str(threshold_selection.get("gate_policy", CLASSIFIER_CONFIDENCE_GATE_POLICY)),
            evidence_weights=(
                np.asarray(threshold_selection["evidence_weights"], dtype=np.float64)
                if "evidence_weights" in threshold_selection
                else None
            ),
            evidence_feature_mean=(
                np.asarray(threshold_selection["evidence_feature_mean"], dtype=np.float64)
                if "evidence_feature_mean" in threshold_selection
                else None
            ),
            evidence_feature_std=(
                np.asarray(threshold_selection["evidence_feature_std"], dtype=np.float64)
                if "evidence_feature_std" in threshold_selection
                else None
            ),
            evidence_decision_th=_safe_float(threshold_selection.get("evidence_decision_th"), 0.0),
            evidence_enter_th=_safe_float(threshold_selection.get("evidence_enter_th"), 0.0),
            evidence_decay=_safe_float(threshold_selection.get("evidence_decay"), DEFAULT_ADAPTIVE_EVIDENCE_DECAY),
            lrt_feature_indices=lrt_feature_indices,
            lrt_feature_mean_control=(
                np.asarray(threshold_selection["lrt_feature_mean_control"], dtype=np.float64)
                if "lrt_feature_mean_control" in threshold_selection
                else None
            ),
            lrt_feature_std_control=(
                np.asarray(threshold_selection["lrt_feature_std_control"], dtype=np.float64)
                if "lrt_feature_std_control" in threshold_selection
                else None
            ),
            lrt_feature_mean_idle=(
                np.asarray(threshold_selection["lrt_feature_mean_idle"], dtype=np.float64)
                if "lrt_feature_mean_idle" in threshold_selection
                else None
            ),
            lrt_feature_std_idle=(
                np.asarray(threshold_selection["lrt_feature_std_idle"], dtype=np.float64)
                if "lrt_feature_std_idle" in threshold_selection
                else None
            ),
            lrt_window_th=_safe_float(threshold_selection.get("lrt_window_th"), 0.0),
            lrt_enter_th=_safe_float(threshold_selection.get("lrt_enter_th"), 0.0),
            lrt_decay=_safe_float(threshold_selection.get("lrt_decay"), DEFAULT_LRT_MULTIWINDOW_DECAY),
            fit_summary={
                **base_model.fit_summary,
                **dict(threshold_selection.get("fit_summary") or {}),
                "score_source_name": str(score_source_name).strip().lower(),
                "classifier": _classifier_name_for_model(base_model, score_source_name=score_source_name),
                "gate_policy": str(threshold_selection.get("gate_policy", CLASSIFIER_CONFIDENCE_GATE_POLICY)),
                "command_confidence_th": max(float(threshold_selection.get("command_confidence_th", 0.0)), 0.0),
                "evidence_decision_th": _safe_float(threshold_selection.get("evidence_decision_th"), 0.0),
                "evidence_enter_th": _safe_float(threshold_selection.get("evidence_enter_th"), 0.0),
                "evidence_decay": _safe_float(threshold_selection.get("evidence_decay"), DEFAULT_ADAPTIVE_EVIDENCE_DECAY),
                "lrt_feature_indices": [int(index) for index in lrt_feature_indices],
                "lrt_window_th": _safe_float(threshold_selection.get("lrt_window_th"), 0.0),
                "lrt_enter_th": _safe_float(threshold_selection.get("lrt_enter_th"), 0.0),
                "lrt_decay": _safe_float(threshold_selection.get("lrt_decay"), DEFAULT_LRT_MULTIWINDOW_DECAY),
                "min_enter_windows": max(1, int(min_enter_windows)),
                "max_gap_windows": max(0, int(max_gap_windows)),
                "smoothing_windows": max(1, int(smoothing_windows)),
                "threshold_policy": _parse_classifier_threshold_policy(threshold_policy),
                "threshold_selection": dict(threshold_selection),
                "score_bank_mode": _parse_score_bank_mode(score_bank_mode),
            },
        )
        resolved_variant_params = dict(gate_variant_params or {"gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW})
        resolved_variant = parse_classifier_gate_variant(resolved_variant_params.get("gate_variant"))
        if resolved_variant != CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW:
            grouped_features = _trial_group_arrays_for_gate(
                model,
                scored_trials,
                smoothing_windows=max(1, int(smoothing_windows)),
            )
            feature_names = _classifier_feature_names(
                freqs,
                score_source_name=score_source_name,
                score_bank_mode=score_bank_mode,
            )
            model = _apply_gate_variant_to_model(
                model,
                feature_names=feature_names,
                grouped_features=grouped_features,
                scored_trials=scored_trials,
                params=resolved_variant_params,
                win_sec=float(win_sec),
                step_sec=float(step_sec),
                min_enter_windows=max(1, int(min_enter_windows)),
                max_gap_windows=max(0, int(max_gap_windows)),
            )
            fit_summary = dict(getattr(model, "fit_summary", {}) or {})
            validation_payload = dict(fit_summary.get("gate_validation_metrics", {}) or {})
            selected = dict(validation_payload.get("metrics", {}) or {})
            if not selected:
                selected_eval = _evaluate_fbcca_lda5_model(
                    model,
                    scored_trials,
                    win_sec=float(win_sec),
                    step_sec=float(step_sec),
                    min_enter_windows=max(1, int(min_enter_windows)),
                    max_gap_windows=max(0, int(max_gap_windows)),
                )
                selected = _extract_row_metrics(_evaluation_payload(selected_eval))
        else:
            model = replace(
                model,
                gate_variant=resolved_variant,
                fit_summary={
                    **model.fit_summary,
                    "gate_variant": resolved_variant,
                    "gate_variant_params": resolved_variant_params,
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
    per_freq_correct_2p5 = {_freq_label(freq): 0 for freq in model.freqs}
    per_freq_latencies: dict[str, list[float]] = {_freq_label(freq): [] for freq in model.freqs}
    per_freq_gate_windows = {_freq_label(freq): 0 for freq in model.freqs}
    per_freq_total_windows = {_freq_label(freq): 0 for freq in model.freqs}
    command_labels = list(labels5[1:])
    cache = (
        tuple(probability_cache)
        if probability_cache is not None
        else _build_classifier_probability_cache(model, scored_trials)
    )
    for item, probs, labels in cache:
        probs = _smooth_classifier_probabilities(
            probs,
            smoothing_windows=int(getattr(model, "smoothing_windows", 1)),
        )
        true_label = _trial_true_label(item.trial)
        fixed_pred_5, fixed_pred_4, _fixed_confidence = _predict_fbcca_lda5_fixed_from_probs(model, probs, labels)
        fixed_y5_true.append(true_label)
        fixed_y5_pred.append(fixed_pred_5)
        fixed_times5.append(float(win_sec))
        if true_label != "idle":
            fixed_y4_true.append(true_label)
            fixed_y4_pred.append(fixed_pred_4)
            fixed_times4.append(float(win_sec))

        use_adaptive_gate = (
            str(getattr(model, "gate_policy", CLASSIFIER_CONFIDENCE_GATE_POLICY))
            == CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY
        )
        use_lrt_gate = (
            str(getattr(model, "gate_policy", CLASSIFIER_CONFIDENCE_GATE_POLICY))
            == CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY
        )
        if use_adaptive_gate:
            gate_features = _adaptive_gate_feature_matrix_for_trial(model, item, probs, labels)
            gate_probs = _adaptive_gate_window_probabilities(model, gate_features)
            lrt_evidence = np.asarray([], dtype=np.float64)
            async_pred_label, confidence, first_index = _predict_adaptive_evidence_trial_from_probs(
                model,
                probs,
                labels,
                gate_probs,
                min_enter_windows=max(1, int(min_enter_windows)),
                max_gap_windows=max(0, int(max_gap_windows)),
            )
        elif use_lrt_gate:
            gate_probs = np.asarray([], dtype=np.float64)
            lrt_evidence = _lrt_window_evidence_from_features(model, item.feature_matrix)
            async_pred_label, confidence, first_index = _predict_lrt_multiwindow_reject_trial_from_probs(
                model,
                probs,
                labels,
                lrt_evidence,
                min_enter_windows=max(1, int(min_enter_windows)),
                max_gap_windows=max(0, int(max_gap_windows)),
                feature_matrix=item.feature_matrix,
            )
        else:
            gate_probs = np.asarray([], dtype=np.float64)
            lrt_evidence = np.asarray([], dtype=np.float64)
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
            if use_adaptive_gate:
                selected_mask = (np.argmax(probs, axis=1) != idle_index) & (
                    gate_probs >= float(getattr(model, "evidence_decision_th", 0.5))
                )
            elif use_lrt_gate:
                floor_th = getattr(model, "lrt_window_floor_th", None)
                effective_window_th = max(
                    float(getattr(model, "lrt_window_th", 0.0)),
                    float(floor_th) if floor_th is not None else float(getattr(model, "lrt_window_th", 0.0)),
                )
                shape_mask = _score_shape_gate_mask_for_model(model, item.feature_matrix)
                if getattr(model, "frequency_specific_control_state_gates", None):
                    shape_mask &= _frequency_specific_gate_mask_for_model(
                        model,
                        probs=probs,
                        labels=labels,
                        feature_matrix=item.feature_matrix,
                        lrt_evidence=lrt_evidence,
                    )
                selected_mask = (np.argmax(probs, axis=1) != idle_index) & (
                    lrt_evidence >= effective_window_th
                ) & (
                    shape_mask
                )
            else:
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
            pred_labels_for_gate = [str(labels[int(index)]) for index in np.argmax(probs, axis=1)]
            for freq_label_value in per_freq_total_windows:
                per_freq_total_windows[freq_label_value] += int(sum(1 for label in pred_labels_for_gate if label == freq_label_value))
            if use_lrt_gate:
                gate_mask_for_trial = _score_shape_gate_mask_for_model(model, item.feature_matrix)
                if getattr(model, "frequency_specific_control_state_gates", None):
                    gate_mask_for_trial &= _frequency_specific_gate_mask_for_model(
                        model,
                        probs=probs,
                        labels=labels,
                        feature_matrix=item.feature_matrix,
                        lrt_evidence=lrt_evidence,
                    )
                for freq_label_value in per_freq_gate_windows:
                    per_freq_gate_windows[freq_label_value] += int(
                        sum(
                            1
                            for idx, label in enumerate(pred_labels_for_gate)
                            if label == freq_label_value and bool(gate_mask_for_trial[idx])
                        )
                    )
            if async_pred_label == true_label:
                control_correct += 1
                per_freq_correct[true_label] += 1
                detection_latencies.append(float(async_latency))
                per_freq_latencies[true_label].append(float(async_latency))
                if float(async_latency) <= 2.0:
                    control_correct_at_2s += 1
                if float(async_latency) <= 2.5:
                    control_correct_at_2p5s += 1
                    per_freq_correct_2p5[true_label] += 1
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
        "per_frequency_recall_at_2.5s": {
            label: float(per_freq_correct_2p5[label] / per_freq_total[label]) if per_freq_total[label] else 0.0
            for label in per_freq_total
        },
        "per_frequency_detection_latency_s": {
            label: float(np.mean(values)) if values else float("inf")
            for label, values in per_freq_latencies.items()
        },
        "per_frequency_gate_pass_rate": {
            label: float(per_freq_gate_windows[label] / per_freq_total_windows[label])
            if per_freq_total_windows[label]
            else 0.0
            for label in per_freq_total_windows
        },
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
        "smoothing_windows": int(getattr(model, "smoothing_windows", 1)),
    }


def _evaluate_nc_calibrated_model(
    model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int,
    nc_gate_type: str,
    nc_payload: Mapping[str, Any],
    nc_thresholds: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    labels5 = list(model.labels)
    command_labels = list(labels5[1:])
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
    per_freq_correct_2p5 = {_freq_label(freq): 0 for freq in model.freqs}
    per_freq_latencies: dict[str, list[float]] = {_freq_label(freq): [] for freq in model.freqs}
    per_freq_gate_windows = {_freq_label(freq): 0 for freq in model.freqs}
    per_freq_total_windows = {_freq_label(freq): 0 for freq in model.freqs}
    nc_gate_stats = {
        "baseline_pass_windows": 0.0,
        "candidate_pass_windows": 0.0,
        "detector_veto_windows": 0.0,
        "low_risk_bypass_windows": 0.0,
    }
    cache = _build_classifier_probability_cache(model, scored_trials)
    for item, probs, labels in cache:
        probs = _smooth_classifier_probabilities(
            probs,
            smoothing_windows=max(1, int(getattr(model, "smoothing_windows", 1))),
        )
        true_label = _trial_true_label(item.trial)
        fixed_pred_5, fixed_pred_4, _fixed_confidence = _predict_fbcca_lda5_fixed_from_probs(model, probs, labels)
        fixed_y5_true.append(true_label)
        fixed_y5_pred.append(fixed_pred_5)
        fixed_times5.append(float(win_sec))
        if true_label != "idle":
            fixed_y4_true.append(true_label)
            fixed_y4_pred.append(fixed_pred_4)
            fixed_times4.append(float(win_sec))

        lrt_evidence = _lrt_window_evidence_from_features(model, item.feature_matrix)
        pass_mask, _cs_prob, stats = _nc_calibrated_pass_mask(
            model,
            item,
            probs,
            labels,
            feature_names=feature_names,
            lrt_evidence=lrt_evidence,
            nc_gate_type=nc_gate_type,
            nc_payload=nc_payload,
            nc_thresholds=nc_thresholds,
            min_enter_windows=max(1, int(min_enter_windows)),
        )
        for key in nc_gate_stats:
            nc_gate_stats[key] += _safe_float(stats.get(key), 0.0)
        async_pred_label, confidence, first_index = _predict_lrt_trial_with_pass_mask(
            model,
            probs,
            labels,
            lrt_evidence,
            pass_mask,
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
        pred_labels_for_gate = [str(labels[int(index)]) for index in np.argmax(probs, axis=1)]
        if true_label == "idle":
            idle_total += 1
            idle_selected_windows += int(np.sum(pass_mask))
            if async_pred_label != "idle":
                idle_selected_events += 1
        else:
            async_y4_true.append(true_label)
            missed_pred = next((label for label in command_labels if label != true_label), command_labels[0])
            async_y4_pred.append(async_pred_label if async_pred_label != "idle" else missed_pred)
            async_times4.append(float(async_latency))
            control_total += 1
            per_freq_total[true_label] += 1
            for freq_label_value in per_freq_total_windows:
                per_freq_total_windows[freq_label_value] += int(
                    sum(1 for label in pred_labels_for_gate if label == freq_label_value)
                )
            for freq_label_value in per_freq_gate_windows:
                per_freq_gate_windows[freq_label_value] += int(
                    sum(
                        1
                        for idx, label in enumerate(pred_labels_for_gate)
                        if label == freq_label_value and bool(idx < pass_mask.shape[0] and pass_mask[idx])
                    )
                )
            if async_pred_label == true_label:
                control_correct += 1
                per_freq_correct[true_label] += 1
                detection_latencies.append(float(async_latency))
                per_freq_latencies[true_label].append(float(async_latency))
                if float(async_latency) <= 2.0:
                    control_correct_at_2s += 1
                if float(async_latency) <= 2.5:
                    control_correct_at_2p5s += 1
                    per_freq_correct_2p5[true_label] += 1
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
    async_metrics = {
        "idle_fp_per_min": float(idle_selected_events / idle_minutes) if idle_minutes > 1e-12 else 0.0,
        "idle_selected_windows_per_min": (
            float(idle_selected_windows / idle_minutes) if idle_minutes > 1e-12 else 0.0
        ),
        "control_recall": _safe_div(control_correct, control_total, 0.0),
        "control_recall_at_2s": _safe_div(control_correct_at_2s, control_total, 0.0),
        "control_recall_at_2.5s": _safe_div(control_correct_at_2p5s, control_total, 0.0),
        "control_recall_at_3s": _safe_div(control_correct_at_3s, control_total, 0.0),
        "switch_latency_s": float("inf"),
        "release_latency_s": float("inf"),
        "switch_latency_supported": False,
        "release_latency_supported": False,
        "detection_latency_s": (
            float(np.median(np.asarray(detection_latencies, dtype=np.float64)))
            if detection_latencies
            else float("inf")
        ),
        "idle_trial_fp_rate": _safe_div(idle_selected_events, idle_total, 0.0),
        "idle_trials": float(idle_total),
        "idle_fp_trials": float(idle_selected_events),
        "idle_selected_windows": float(idle_selected_windows),
        "control_trials": float(control_total),
        "per_frequency_recall_at_2.5s": {
            label: _safe_div(per_freq_correct_2p5[label], per_freq_total[label], 0.0)
            for label in per_freq_total
        },
        "per_frequency_detection_latency_s": {
            label: float(np.mean(values)) if values else float("inf")
            for label, values in per_freq_latencies.items()
        },
        "per_frequency_gate_pass_rate": {
            label: _safe_div(per_freq_gate_windows[label], per_freq_total_windows[label], 0.0)
            for label in per_freq_total_windows
        },
        "nc_gate_stats": dict(nc_gate_stats),
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
            label: _safe_div(per_freq_correct[label], per_freq_total[label], 0.0)
            for label in per_freq_total
        },
        "model_summary": dict(model.fit_summary),
        "command_confidence_th": float(model.command_confidence_th),
        "min_enter_windows": int(min_enter_windows),
        "max_gap_windows": max(0, int(max_gap_windows)),
        "smoothing_windows": int(getattr(model, "smoothing_windows", 1)),
    }


def _nc_event_fix_metrics(
    baseline_model: FBCCALDA5Model | FBCCARidge5Model,
    candidate_model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int,
    nc_gate_type: str,
    nc_payload: Mapping[str, Any],
    nc_thresholds: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    fixed_ns2_fp_count = 0
    lost_command_tp_count = 0
    for item in scored_trials:
        true_label = _trial_true_label(item.trial)
        subtype = _ysuan_ns_subtype_from_label(str(item.trial.label))
        probs, labels = _predict_classifier_windows(baseline_model, item.feature_matrix)
        probs = _smooth_classifier_probabilities(
            probs,
            smoothing_windows=max(1, int(getattr(baseline_model, "smoothing_windows", 1))),
        )
        evidence = _lrt_window_evidence_from_features(baseline_model, item.feature_matrix)
        baseline_label, _base_score, _base_index = _predict_lrt_multiwindow_reject_trial_from_probs(
            baseline_model,
            probs,
            labels,
            evidence,
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            feature_matrix=item.feature_matrix,
        )
        candidate_mask, _cs_prob, _stats = _nc_calibrated_pass_mask(
            candidate_model,
            item,
            probs,
            labels,
            feature_names=feature_names,
            lrt_evidence=evidence,
            nc_gate_type=nc_gate_type,
            nc_payload=nc_payload,
            nc_thresholds=nc_thresholds,
            min_enter_windows=max(1, int(min_enter_windows)),
        )
        candidate_label, _cand_score, _cand_index = _predict_lrt_trial_with_pass_mask(
            candidate_model,
            probs,
            labels,
            evidence,
            candidate_mask,
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
        )
        if subtype == "ns2" and baseline_label != "idle" and candidate_label == "idle":
            fixed_ns2_fp_count += 1
        if true_label != "idle" and baseline_label == true_label and candidate_label == "idle":
            lost_command_tp_count += 1
    return {
        "fixed_ns2_fp_count": int(fixed_ns2_fp_count),
        "lost_command_tp_count": int(lost_command_tp_count),
        "tp_loss_per_fixed_fp": float(lost_command_tp_count / max(fixed_ns2_fp_count, 1)),
    }


def _evaluate_nc_clean_idle_from_cache(
    model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int,
    nc_gate_type: str,
    nc_payload: Mapping[str, Any],
    nc_thresholds: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    if not scored_trials:
        return {
            "supported": False,
            "reason": "no no-control scored trials",
            "idle_fp_per_min": None,
            "idle_trial_fp_rate": None,
            "idle_trials": 0,
        }
    bundle = _evaluate_nc_calibrated_model(
        model,
        scored_trials,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
        nc_gate_type=nc_gate_type,
        nc_payload=nc_payload,
        nc_thresholds=nc_thresholds,
        feature_names=feature_names,
    )
    metrics = dict(bundle.get("async_metrics", {}) or {})
    return {
        "supported": True,
        "idle_fp_per_min": _safe_float(metrics.get("idle_fp_per_min"), 0.0),
        "idle_selected_windows_per_min": _safe_float(metrics.get("idle_selected_windows_per_min"), 0.0),
        "idle_trial_fp_rate": _safe_float(metrics.get("idle_trial_fp_rate"), 0.0),
        "idle_trials": int(_safe_float(metrics.get("idle_trials"), 0.0)),
        "idle_fp_trials": int(_safe_float(metrics.get("idle_fp_trials"), 0.0)),
    }


def _evaluate_nc_no_control_subtypes_from_cache(
    model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int,
    nc_gate_type: str,
    nc_payload: Mapping[str, Any],
    nc_thresholds: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    grouped: dict[str, list[ScoredTrial]] = defaultdict(list)
    for item in scored_trials:
        subtype = _ysuan_ns_subtype_from_label(str(item.trial.label))
        if subtype:
            grouped[subtype].append(item)
    subtype_metrics: dict[str, dict[str, Any]] = {}
    for subtype in ("ns1", "ns2", "ns3"):
        subtype_metrics[subtype] = _evaluate_nc_clean_idle_from_cache(
            model,
            grouped.get(subtype, []),
            win_sec=float(win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            nc_gate_type=nc_gate_type,
            nc_payload=nc_payload,
            nc_thresholds=nc_thresholds,
            feature_names=feature_names,
        )
    pooled = _evaluate_nc_clean_idle_from_cache(
        model,
        scored_trials,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
        nc_gate_type=nc_gate_type,
        nc_payload=nc_payload,
        nc_thresholds=nc_thresholds,
        feature_names=feature_names,
    )
    return {
        "supported": bool(pooled.get("supported", False)),
        "ns1": subtype_metrics["ns1"],
        "ns2": subtype_metrics["ns2"],
        "ns3": subtype_metrics["ns3"],
        "ns_all_fp_per_min": _safe_float(pooled.get("idle_fp_per_min"), float("nan")),
        "ns_all_trial_fp_rate": _safe_float(pooled.get("idle_trial_fp_rate"), float("nan")),
        "pooled": pooled,
    }


def run_fbcca_ridge5_nc_calibration_method(
    *,
    spec: ExternalSubjectSpec,
    split_plan: SplitPlan,
    split_summary: dict[str, Any],
    sampling_rate: int,
    freqs: Sequence[float],
    step_sec: float,
    win_sec: float,
    min_enter_windows: int,
    max_gap_windows: int = 0,
    smoothing_windows: int = 1,
    threshold_policy: str = DEFAULT_CLASSIFIER_THRESHOLD_POLICY,
    calibration_scored: Sequence[ScoredTrial],
    holdout_scored: Sequence[ScoredTrial],
    nc_calibration_scored: Sequence[ScoredTrial],
    clean_idle_scored: Optional[Sequence[ScoredTrial]],
    clean_idle_support: Optional[Mapping[str, Any]],
    base_models: Optional[Sequence[FBCCARidge5Model]],
    baseline_model: Optional[FBCCARidge5Model] = None,
    score_bank_mode: str,
    nc_seconds: float,
    nc_source: str,
    nc_gate_type: str,
    nc_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    calibration_scored = list(calibration_scored)
    holdout_scored = list(holdout_scored)
    nc_calibration_scored = list(nc_calibration_scored)
    clean_idle_scored = list(clean_idle_scored or [])
    resolved_gate = _parse_nc_gate_type(nc_gate_type)
    base_recipe_id = _classifier_recipe_id_with_smoothing(
        win_sec=float(win_sec),
        min_enter_windows=int(min_enter_windows),
        max_gap_windows=max(0, int(max_gap_windows)),
        smoothing_windows=max(1, int(smoothing_windows)),
        gate_policy=str(threshold_policy),
        gate_variant=CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
        variant_token=_classifier_gate_variant_token(CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW),
    )
    recipe_id = _nc_recipe_id(
        base_recipe_id=base_recipe_id,
        seconds=float(nc_seconds),
        source=nc_source,
        gate_type=resolved_gate,
    )
    latency_win_sec = _method_latency_window_sec(
        method_name="fbcca_ridge5",
        win_sec=float(win_sec),
        sampling_rate=int(sampling_rate),
    )
    base_model = baseline_model or _fit_fbcca_ridge5_model(
        calibration_scored,
        freqs=freqs,
        win_sec=float(latency_win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
        smoothing_windows=max(1, int(smoothing_windows)),
        threshold_policy=str(threshold_policy),
        base_models=base_models,
        score_source_name="fbcca",
        score_bank_mode=score_bank_mode,
        gate_variant_params={"gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW},
    )
    feature_names = _classifier_feature_names(freqs, score_bank_mode=score_bank_mode)
    nc_support_status = "baseline_no_extra_no_control"
    nc_payload: dict[str, Any] = {
        "type": "none",
        "status": "not_required",
        "prob_threshold": float(NC_CSNS_PROB_THRESHOLD),
    }
    training_rows: list[dict[str, Any]] = []
    feature_summary_rows: list[dict[str, Any]] = []
    nc_thresholds: dict[str, Any] = {}
    candidate_model = base_model
    if resolved_gate == NC_GATE_BASELINE_LRT_THRESHOLD:
        evidence = _lrt_window_evidence_from_features(
            base_model,
            np.vstack([np.asarray(item.feature_matrix, dtype=np.float64) for item in nc_calibration_scored])
            if nc_calibration_scored
            else np.zeros((0, int(base_model.feature_mean.shape[0])), dtype=np.float64),
        )
        nc_floor = (
            _safe_quantile(evidence, 0.95, float(base_model.lrt_window_th))
            if np.asarray(evidence).size
            else float(base_model.lrt_window_th)
        )
        floor_th = max(float(base_model.lrt_window_th), float(nc_floor))
        nc_support_status = "ok" if nc_calibration_scored else "baseline_fallback_no_no_control_calibration"
        candidate_model = replace(
            base_model,
            lrt_window_floor_th=float(floor_th),
            fit_summary={
                **base_model.fit_summary,
                "nc_calibration_simulation": True,
                "nc_gate_type": resolved_gate,
                "nc_seconds": float(nc_seconds),
                "nc_source": _parse_nc_calibration_source(nc_source),
                "nc_lrt_window_floor_th": float(floor_th),
                "nc_lrt_window_p95": float(nc_floor),
                "no_control_calibration_provenance": dict(nc_provenance),
                "fit_split": "command_calibration_blocks_plus_no_control_calibration_fit_split",
                "test_split": "holdout_blocks",
            },
        )
        nc_payload = {
            "type": "lrt_window_floor",
            "status": nc_support_status,
            "lrt_window_floor_th": float(floor_th),
            "nc_lrt_window_p95": float(nc_floor),
            "fit_split": "no_control_calibration_fit_split",
            "test_split": "holdout_blocks",
            "no_control_calibration_provenance": dict(nc_provenance),
        }
    else:
        nc_payload, training_rows, feature_summary_rows = _fit_nc_session_csns_payload(
            base_model,
            command_scored=calibration_scored,
            nc_scored=nc_calibration_scored,
            feature_names=feature_names,
            smoothing_windows=max(1, int(smoothing_windows)),
            nc_provenance=nc_provenance,
        )
        nc_support_status = str(nc_payload.get("status", "unsupported"))
        nc_thresholds = _nc_conditional_thresholds(training_rows)
        nc_payload = {**nc_payload, "prob_threshold": float(NC_CSNS_PROB_THRESHOLD)}
        feature_summary_rows = [
            {
                **dict(item),
                "dataset": str(spec.dataset),
                "subject": str(spec.subject),
                "split_index": int(split_plan.split_index),
                "recipe_id": recipe_id,
                "nc_seconds": float(nc_seconds),
                "nc_source": _parse_nc_calibration_source(nc_source),
                "nc_gate_type": resolved_gate,
            }
            for item in feature_summary_rows
        ]
        candidate_model = replace(
            base_model,
            fit_summary={
                **base_model.fit_summary,
                "nc_calibration_simulation": True,
                "nc_gate_type": resolved_gate,
                "nc_seconds": float(nc_seconds),
                "nc_source": _parse_nc_calibration_source(nc_source),
                "nc_csns_payload": dict(nc_payload),
                "nc_conditional_low_risk_thresholds": dict(nc_thresholds),
                "no_control_calibration_provenance": dict(nc_provenance),
                "fit_split": "command_calibration_blocks_plus_no_control_calibration_fit_split",
                "test_split": "holdout_blocks",
            },
        )
    if resolved_gate == NC_GATE_BASELINE_LRT_THRESHOLD:
        bundle = _evaluate_fbcca_lda5_model(
            candidate_model,
            holdout_scored,
            win_sec=float(latency_win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
        )
    else:
        bundle = _evaluate_nc_calibrated_model(
            candidate_model,
            holdout_scored,
            win_sec=float(latency_win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            nc_gate_type=resolved_gate,
            nc_payload=nc_payload,
            nc_thresholds=nc_thresholds,
            feature_names=feature_names,
        )
    eval_payload = _evaluation_payload(bundle)
    if clean_idle_support is not None:
        support_payload = dict(clean_idle_support)
        if support_payload.get("supported") and clean_idle_scored:
            if resolved_gate == NC_GATE_BASELINE_LRT_THRESHOLD:
                eval_payload["clean_idle_proxy_metrics"] = _evaluate_clean_idle_proxy_from_cache(
                    candidate_model,
                    clean_idle_scored,
                    win_sec=float(latency_win_sec),
                    step_sec=float(step_sec),
                    min_enter_windows=max(1, int(min_enter_windows)),
                    max_gap_windows=max(0, int(max_gap_windows)),
                )
                eval_payload["no_control_subtype_metrics"] = _evaluate_no_control_subtypes_from_cache(
                    candidate_model,
                    clean_idle_scored,
                    win_sec=float(latency_win_sec),
                    step_sec=float(step_sec),
                    min_enter_windows=max(1, int(min_enter_windows)),
                    max_gap_windows=max(0, int(max_gap_windows)),
                )
                eval_payload["no_control_subtype_by_frequency"] = _no_control_subtype_by_frequency_from_cache(
                    candidate_model,
                    clean_idle_scored,
                    win_sec=float(latency_win_sec),
                    step_sec=float(step_sec),
                )
            else:
                clean_bundle = _evaluate_nc_calibrated_model(
                    candidate_model,
                    clean_idle_scored,
                    win_sec=float(latency_win_sec),
                    step_sec=float(step_sec),
                    min_enter_windows=max(1, int(min_enter_windows)),
                    max_gap_windows=max(0, int(max_gap_windows)),
                    nc_gate_type=resolved_gate,
                    nc_payload=nc_payload,
                    nc_thresholds=nc_thresholds,
                    feature_names=feature_names,
                )
                clean_metrics = dict(clean_bundle.get("async_metrics", {}) or {})
                eval_payload["clean_idle_proxy_metrics"] = {
                    "supported": True,
                    "idle_fp_per_min": _safe_float(clean_metrics.get("idle_fp_per_min"), 0.0),
                    "idle_selected_windows_per_min": _safe_float(
                        clean_metrics.get("idle_selected_windows_per_min"), 0.0
                    ),
                    "idle_trial_fp_rate": _safe_float(clean_metrics.get("idle_trial_fp_rate"), 0.0),
                    "idle_trials": int(_safe_float(clean_metrics.get("idle_trials"), 0.0)),
                    "idle_fp_trials": int(_safe_float(clean_metrics.get("idle_fp_trials"), 0.0)),
                }
                eval_payload["no_control_subtype_metrics"] = _evaluate_nc_no_control_subtypes_from_cache(
                    candidate_model,
                    clean_idle_scored,
                    win_sec=float(latency_win_sec),
                    step_sec=float(step_sec),
                    min_enter_windows=max(1, int(min_enter_windows)),
                    max_gap_windows=max(0, int(max_gap_windows)),
                    nc_gate_type=resolved_gate,
                    nc_payload=nc_payload,
                    nc_thresholds=nc_thresholds,
                    feature_names=feature_names,
                )
                eval_payload["no_control_subtype_by_frequency"] = {}
    fix_metrics = _nc_event_fix_metrics(
        base_model,
        candidate_model,
        holdout_scored,
        win_sec=float(latency_win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
        nc_gate_type=resolved_gate,
        nc_payload=nc_payload,
        nc_thresholds=nc_thresholds,
        feature_names=feature_names,
    )
    eval_payload["nc_calibration_metrics"] = fix_metrics
    diagnostics = _ns2_selected_freq_diagnostic_rows(
        candidate_model,
        list(holdout_scored),
        dataset=str(spec.dataset),
        subject=str(spec.subject),
        split_index=int(split_plan.split_index),
        recipe_id=recipe_id,
        gate_variant=CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
        win_sec=float(latency_win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
    )
    calibration_profile = {
        "status": "ok",
        "classifier": _classifier_name_for_model(candidate_model, score_source_name="fbcca"),
        "fit_summary": dict(candidate_model.fit_summary),
        "command_confidence_th": float(candidate_model.command_confidence_th),
        "min_enter_windows": int(min_enter_windows),
        "max_gap_windows": max(0, int(max_gap_windows)),
        "smoothing_windows": max(1, int(smoothing_windows)),
        "gate_policy": str(getattr(candidate_model, "gate_policy", CLASSIFIER_CONFIDENCE_GATE_POLICY)),
        "gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
        "gate_variant_params": {"gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW},
        "threshold_policy": _parse_classifier_threshold_policy(threshold_policy),
        "feature_count": int(candidate_model.feature_mean.shape[0]),
        "feature_names": feature_names,
        "score_bank_mode": _parse_score_bank_mode(score_bank_mode),
        "runtime_loadable": False,
        "no_control_calibration_simulation": {
            "enabled": True,
            "nc_seconds": float(nc_seconds),
            "nc_source": _parse_nc_calibration_source(nc_source),
            "nc_gate_type": resolved_gate,
            "support_status": nc_support_status,
            "payload": dict(nc_payload),
            "conditional_low_risk_thresholds": dict(nc_thresholds),
            "provenance": dict(nc_provenance),
        },
    }
    metrics = _extract_row_metrics(eval_payload)
    metrics.update(
        {
            "fixed_ns2_fp_count": float(fix_metrics.get("fixed_ns2_fp_count", 0)),
            "lost_command_tp_count": float(fix_metrics.get("lost_command_tp_count", 0)),
            "tp_loss_per_fixed_fp": _safe_float(fix_metrics.get("tp_loss_per_fixed_fp"), float("inf")),
        }
    )
    return {
        "method": "fbcca_ridge5_nc_calibration",
        "aggregate_recipe_id": recipe_id,
        "recipe_id": recipe_id,
        "dataset": str(spec.dataset),
        "subject": str(spec.subject),
        "split_index": int(split_plan.split_index),
        "calibration_blocks": [int(block) for block in split_plan.calibration_blocks],
        "holdout_blocks": [int(block) for block in split_plan.holdout_blocks],
        "selected_freqs": [float(freq) for freq in freqs],
        "gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
        "gate_variant_params": {"gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW},
        "split_summary": {
            **dict(split_summary),
            "no_control_calibration": dict(nc_provenance),
        },
        "nc_calibration_simulation": True,
        "nc_seconds": float(nc_seconds),
        "nc_source": _parse_nc_calibration_source(nc_source),
        "nc_gate_type": resolved_gate,
        "nc_calibration_provenance": dict(nc_provenance),
        "nc_calibration_support_status": nc_support_status,
        "nc_csns_feature_summary": feature_summary_rows,
        "calibration_profile": calibration_profile,
        "holdout_eval": eval_payload,
        "per_frequency_metrics": _per_frequency_metrics_from_eval(eval_payload, candidate_model),
        "ns2_selected_freq_diagnostics": diagnostics,
        "summary_metrics": metrics,
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
            "clean_idle_proxy_metrics": dict(bundle.get("clean_idle_proxy_metrics") or {}),
            "no_control_subtype_metrics": dict(bundle.get("no_control_subtype_metrics") or {}),
            "no_control_subtype_by_frequency": dict(bundle.get("no_control_subtype_by_frequency") or {}),
            "per_frequency_recall": dict(bundle.get("per_frequency_recall") or {}),
        }
    return {
        "metric_scope": str(bundle.get("metric_scope", "")),
        "fixed_window_metrics_5class": dict(bundle.get("metrics_5class") or {}),
        "async_lens_metrics_5class": dict(bundle.get("async_lens_metrics_5class") or {}),
        "fixed_window_metrics_4class": dict(bundle.get("metrics_4class") or {}),
        "async_lens_metrics_4class": dict(bundle.get("async_lens_metrics_4class") or {}),
        "async_metrics": dict(bundle.get("async_metrics") or {}),
        "clean_idle_proxy_metrics": dict(bundle.get("clean_idle_proxy_metrics") or {}),
        "no_control_subtype_metrics": dict(bundle.get("no_control_subtype_metrics") or {}),
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
        "smoothing_windows": int(getattr(model, "smoothing_windows", 1)),
        "gate_policy": str(getattr(model, "gate_policy", CLASSIFIER_CONFIDENCE_GATE_POLICY)),
        "evidence_decision_th": float(getattr(model, "evidence_decision_th", 0.0)),
        "evidence_enter_th": float(getattr(model, "evidence_enter_th", 0.0)),
        "evidence_decay": float(getattr(model, "evidence_decay", DEFAULT_ADAPTIVE_EVIDENCE_DECAY)),
        "lrt_feature_indices": [int(index) for index in getattr(model, "lrt_feature_indices", ())],
        "lrt_window_th": float(getattr(model, "lrt_window_th", 0.0)),
        "lrt_enter_th": float(getattr(model, "lrt_enter_th", 0.0)),
        "lrt_decay": float(getattr(model, "lrt_decay", DEFAULT_LRT_MULTIWINDOW_DECAY)),
        "gate_variant": parse_classifier_gate_variant(getattr(model, "gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW)),
        "score_shape_margin_index": (
            None
            if getattr(model, "score_shape_margin_index", None) is None
            else int(getattr(model, "score_shape_margin_index"))
        ),
        "score_shape_ratio_index": (
            None
            if getattr(model, "score_shape_ratio_index", None) is None
            else int(getattr(model, "score_shape_ratio_index"))
        ),
        "score_shape_entropy_index": (
            None
            if getattr(model, "score_shape_entropy_index", None) is None
            else int(getattr(model, "score_shape_entropy_index"))
        ),
        "score_shape_margin_th": (
            None
            if getattr(model, "score_shape_margin_th", None) is None
            else float(getattr(model, "score_shape_margin_th"))
        ),
        "score_shape_ratio_th": (
            None
            if getattr(model, "score_shape_ratio_th", None) is None
            else float(getattr(model, "score_shape_ratio_th"))
        ),
        "score_shape_entropy_th": (
            None
            if getattr(model, "score_shape_entropy_th", None) is None
            else float(getattr(model, "score_shape_entropy_th"))
        ),
        "lrt_window_floor_th": (
            None
            if getattr(model, "lrt_window_floor_th", None) is None
            else float(getattr(model, "lrt_window_floor_th"))
        ),
        "weak_subject_guard_active": bool(getattr(model, "weak_subject_guard_active", False)),
        "weak_subject_guard_reasons": list(getattr(model, "weak_subject_guard_reasons", ()) or ()),
        "frequency_specific_control_state_gates": normalize_frequency_specific_control_state_gates(
            getattr(model, "frequency_specific_control_state_gates", None)
        ),
        "fit_summary": dict(model.fit_summary),
    }
    if getattr(model, "evidence_weights", None) is not None:
        payload["evidence_weights"] = _array_payload(np.asarray(model.evidence_weights, dtype=np.float64))
    if getattr(model, "evidence_feature_mean", None) is not None:
        payload["evidence_feature_mean"] = _array_payload(np.asarray(model.evidence_feature_mean, dtype=np.float64))
    if getattr(model, "evidence_feature_std", None) is not None:
        payload["evidence_feature_std"] = _array_payload(np.asarray(model.evidence_feature_std, dtype=np.float64))
    if getattr(model, "lrt_feature_mean_control", None) is not None:
        payload["lrt_feature_mean_control"] = _array_payload(
            np.asarray(model.lrt_feature_mean_control, dtype=np.float64)
        )
    if getattr(model, "lrt_feature_std_control", None) is not None:
        payload["lrt_feature_std_control"] = _array_payload(
            np.asarray(model.lrt_feature_std_control, dtype=np.float64)
        )
    if getattr(model, "lrt_feature_mean_idle", None) is not None:
        payload["lrt_feature_mean_idle"] = _array_payload(
            np.asarray(model.lrt_feature_mean_idle, dtype=np.float64)
        )
    if getattr(model, "lrt_feature_std_idle", None) is not None:
        payload["lrt_feature_std_idle"] = _array_payload(
            np.asarray(model.lrt_feature_std_idle, dtype=np.float64)
        )
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


def _full_reference_bank_freqs_from_scored(
    scored_trials: Sequence[ScoredTrial],
    *,
    fallback_freqs: Sequence[float],
) -> list[float]:
    for item in scored_trials:
        values = tuple(float(freq) for freq in getattr(item, "all_freqs", ()) or ())
        if values:
            return [float(freq) for freq in values]
    return [float(freq) for freq in fallback_freqs]


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
    score_bank_mode: str = DEFAULT_SCORE_BANK_MODE,
    decoder_name: Optional[str] = None,
    decoder_model_params: Optional[Mapping[str, Any]] = None,
    full_reference_bank_freqs: Sequence[float] = (),
) -> dict[str, Any]:
    normalized_source_name = str(score_source_name).strip().lower()
    full_bank_freqs = [float(freq) for freq in full_reference_bank_freqs]
    classifier_name = _classifier_name_for_model(model, score_source_name=normalized_source_name)
    runtime_loadable = False
    feature_names = _classifier_feature_names(
        freqs,
        score_source_name=normalized_source_name,
        score_bank_mode=score_bank_mode,
    )
    return {
        "artifact_schema_version": "external_fbcca_classifier_candidate_v1",
        "status": "candidate_only",
        "runtime_loadable": runtime_loadable,
        "runtime_load_note": (
            "Pseudo-online/offline replay candidate only; this run must not be loaded directly "
            "or written back to deployed/default runtime profiles."
        ),
        "model_name": classifier_name,
        "model_family": f"{normalized_source_name}_score_classifier_5class",
        "feature_contract": {
            "feature_source": f"{normalized_source_name}_score_matrix",
            "feature_builder": "_score_matrices_to_features",
            "score_bank_mode": _parse_score_bank_mode(score_bank_mode),
            "feature_names": feature_names,
            "feature_count": int(len(feature_names)),
            "command_score_count": int(len(freqs)),
            "derived_feature_names": list(CLASSIFIER_DERIVED_FEATURE_NAMES),
            "full_reference_bank_feature_names": (
                list(FULL_REFERENCE_BANK_FEATURE_NAMES)
                if _parse_score_bank_mode(score_bank_mode) == "full_reference_bank"
                else []
            ),
        },
        "windowing": {
            "win_sec": float(win_sec),
            "step_sec": float(step_sec),
            "min_enter_windows": int(min_enter_windows),
            "smoothing_windows": int(getattr(model, "smoothing_windows", 1)),
        },
        "gate": {
            "policy": str(getattr(model, "gate_policy", CLASSIFIER_CONFIDENCE_GATE_POLICY)),
            "variant": parse_classifier_gate_variant(getattr(model, "gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW)),
            "params": dict(dict(model.fit_summary).get("gate_variant_params", {}) or {}),
            "frequency_specific_control_state_gates": normalize_frequency_specific_control_state_gates(
                getattr(model, "frequency_specific_control_state_gates", None)
            ),
            "lrt_window_th": _finite_or_none(getattr(model, "lrt_window_th", None)),
            "lrt_window_floor_th": _finite_or_none(getattr(model, "lrt_window_floor_th", None)),
            "score_shape_margin_th": _finite_or_none(getattr(model, "score_shape_margin_th", None)),
            "score_shape_ratio_th": _finite_or_none(getattr(model, "score_shape_ratio_th", None)),
            "score_shape_entropy_th": _finite_or_none(getattr(model, "score_shape_entropy_th", None)),
            "weak_subject_guard_active": bool(getattr(model, "weak_subject_guard_active", False)),
            "weak_subject_guard_reasons": list(getattr(model, "weak_subject_guard_reasons", ()) or ()),
        },
        "training_provenance": {
            "dataset": str(spec.dataset),
            "subject": str(spec.subject),
            "source_subjects": [str(spec.subject)],
            "target_subject": str(spec.subject),
            "source_mat_path": str(spec.mat_path),
            "channel_loc_path": "" if spec.channel_loc_path is None else str(spec.channel_loc_path),
            "sampling_rate": int(sampling_rate),
            "score_source_name": normalized_source_name,
            "decoder_name": str(decoder_name or normalized_source_name),
            "decoder_model_params": json_safe(dict(decoder_model_params or {})),
            "full_reference_bank_freqs": list(full_bank_freqs),
            "required_channel_names": list(_required_channel_names(spec.dataset)),
            "only_required_channels_used": True,
            "command_freqs": [float(freq) for freq in freqs],
            "split_index": int(split_plan.split_index),
            "seed": int(split_plan.seed),
            "calibration_blocks": [int(block) for block in split_plan.calibration_blocks],
            "holdout_blocks": [int(block) for block in split_plan.holdout_blocks],
            "excluded_test_blocks": [int(block) for block in split_plan.holdout_blocks],
            "fit_split": [int(block) for block in split_plan.calibration_blocks],
            "test_split": [int(block) for block in split_plan.holdout_blocks],
            "ysu_an_ns_calibration_counts": dict(dict(split_summary).get("ysu_an_ns_calibration_counts", {}) or {}),
            "ysu_an_ns_holdout_counts": dict(dict(split_summary).get("ysu_an_ns_holdout_counts", {}) or {}),
            "idle_multiplier": _safe_float(dict(split_summary).get("idle_multiplier"), 1.0),
            "data_leakage_guard": "reference bank, threshold fitting, templates, and LDA/Ridge models use calibration blocks only; holdout/test blocks are excluded from fitting.",
            "idle_proxy_note": (
                "Idle/no-control is proxied with non-command target stimulus trials from external benchmarks."
            ),
        },
        "state": _classifier_state_payload(model),
        "runtime_profile_model_params": {
            "state": _classifier_state_payload(model),
            "score_source_name": normalized_source_name,
            "score_bank_mode": _parse_score_bank_mode(score_bank_mode),
            "feature_names": feature_names,
            "decoder_name": str(decoder_name or normalized_source_name),
            "decoder_model_params": json_safe(dict(decoder_model_params or {})),
            "full_reference_bank_freqs": list(full_bank_freqs),
            "max_gap_windows": int(dict(model.fit_summary).get("max_gap_windows", 0) or 0),
        },
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
    clean_idle = dict(eval_payload.get("clean_idle_proxy_metrics") or {})
    subtype_metrics = dict(eval_payload.get("no_control_subtype_metrics") or {})
    ns1 = dict(subtype_metrics.get("ns1") or {})
    ns2 = dict(subtype_metrics.get("ns2") or {})
    ns3 = dict(subtype_metrics.get("ns3") or {})
    real_idle_fp = _safe_float(subtype_metrics.get("ns_all_fp_per_min"), float("nan"))
    clean_idle_fp = _safe_float(clean_idle.get("idle_fp_per_min"), float("nan"))
    approx_idle_fp = _safe_float(async_metrics.get("idle_fp_per_min"), float("inf"))
    real_candidates = [value for value in (real_idle_fp, clean_idle_fp) if np.isfinite(value)]
    real_idle_fp = float(real_candidates[0]) if real_candidates else float("nan")
    mixed_idle_fp = float(real_candidates[0]) if real_candidates else approx_idle_fp
    metrics = {
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
        "clean_idle_proxy_supported": float(1.0 if bool(clean_idle.get("supported", False)) else 0.0),
        "clean_idle_proxy_fp_per_min": _safe_float(clean_idle.get("idle_fp_per_min"), float("nan")),
        "ns1_fp_per_min": _safe_float(ns1.get("idle_fp_per_min"), float("nan")),
        "ns2_fp_per_min": _safe_float(ns2.get("idle_fp_per_min"), float("nan")),
        "ns3_fp_per_min": _safe_float(ns3.get("idle_fp_per_min"), float("nan")),
        "ns_all_fp_per_min": _safe_float(subtype_metrics.get("ns_all_fp_per_min"), float("nan")),
        "cs_control_recall": _safe_float(async_metrics.get("control_recall"), 0.0),
    }
    metrics.update(
        {
            "real_idle_fp_per_min": real_idle_fp,
            "approx_idle_fp_per_min": approx_idle_fp,
            "mixed_idle_fp_per_min": mixed_idle_fp,
        }
    )
    tenp5 = dict(eval_payload.get("tenp5_ns2_veto_metrics") or {})
    if tenp5:
        for key in (
            "fixed_ns2_fp_count",
            "lost_command_tp_count",
            "veto_precision",
            "tp_loss_per_fixed_fp",
        ):
            metrics[key] = _safe_float(tenp5.get(key), float("nan"))
    nc_metrics = dict(eval_payload.get("nc_calibration_metrics") or {})
    if nc_metrics:
        for key in (
            "fixed_ns2_fp_count",
            "lost_command_tp_count",
            "tp_loss_per_fixed_fp",
        ):
            metrics[key] = _safe_float(nc_metrics.get(key), float("nan"))
    return metrics


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


def _per_frequency_metrics_from_eval(eval_payload: Mapping[str, Any], model: FBCCALDA5Model | FBCCARidge5Model) -> list[dict[str, Any]]:
    async_metrics = dict(eval_payload.get("async_metrics", {}) or {})
    per_freq_recall = dict(eval_payload.get("per_frequency_recall", {}) or {})
    per_freq_latency = dict(async_metrics.get("per_frequency_detection_latency_s", {}) or {})
    gate_pass_rates = dict(async_metrics.get("per_frequency_gate_pass_rate", {}) or {})
    subtype_by_freq = dict(eval_payload.get("no_control_subtype_by_frequency", {}) or {})
    rows: list[dict[str, Any]] = []
    for freq in model.freqs:
        key = _freq_label(freq)
        ns_payload = dict(subtype_by_freq.get(key, {}) or {})
        rows.append(
            {
                "freq": float(freq),
                "command_recall": _finite_or_none(per_freq_recall.get(key)),
                "recall_at_2.5s": _finite_or_none(dict(async_metrics.get("per_frequency_recall_at_2.5s", {}) or {}).get(key)),
                "fp_per_min_ns1": _finite_or_none(dict(ns_payload.get("ns1", {}) or {}).get("fp_per_min")),
                "fp_per_min_ns2": _finite_or_none(dict(ns_payload.get("ns2", {}) or {}).get("fp_per_min")),
                "fp_per_min_ns3": _finite_or_none(dict(ns_payload.get("ns3", {}) or {}).get("fp_per_min")),
                "gate_pass_rate": _finite_or_none(gate_pass_rates.get(key)),
                "mean_latency": _finite_or_none(per_freq_latency.get(key)),
            }
        )
    return rows


def _ns2_selected_freq_diagnostic_rows(
    model: FBCCALDA5Model | FBCCARidge5Model,
    scored_trials: Sequence[ScoredTrial],
    *,
    dataset: str,
    subject: str,
    split_index: int,
    recipe_id: str,
    gate_variant: str,
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    max_gap_windows: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    feature_names = _classifier_feature_names(
        model.freqs,
        score_source_name=str(dict(model.fit_summary).get("score_source_name", "fbcca")),
        score_bank_mode=str(dict(model.fit_summary).get("score_bank_mode", DEFAULT_SCORE_BANK_MODE)),
    )
    name_to_index = {name: int(index) for index, name in enumerate(FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES)}
    by_freq: dict[str, dict[str, Any]] = {}
    by_subject_freq: dict[str, dict[str, Any]] = {}
    ns2_event_fp_by_freq: dict[str, int] = defaultdict(int)
    ns2_duration_total_sec = 0.0
    confusion: dict[tuple[str, str, str], int] = defaultdict(int)
    command_trials_by_label: dict[str, list[ScoredTrial]] = defaultdict(list)
    cache = _build_classifier_probability_cache(model, scored_trials)
    for item, probs, labels in cache:
        true_label = _trial_true_label(item.trial)
        subtype = _ysuan_ns_subtype_from_label(str(item.trial.label))
        if true_label != "idle":
            command_trials_by_label[true_label].append(item)
        smoothed = _smooth_classifier_probabilities(probs, smoothing_windows=int(getattr(model, "smoothing_windows", 1)))
        lrt_evidence = (
            _lrt_window_evidence_from_features(model, item.feature_matrix)
            if str(getattr(model, "gate_policy", "")) == CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY
            else np.zeros(int(smoothed.shape[0]), dtype=np.float64)
        )
        gate_features, pred_indices, meta_rows = _frequency_specific_gate_features_for_trial(
            model=model,
            item=item,
            probs=smoothed,
            labels=labels,
            lrt_evidence=lrt_evidence,
            feature_names=feature_names,
            smoothing_windows=int(getattr(model, "smoothing_windows", 1)),
            score_source_name=str(dict(model.fit_summary).get("score_source_name", "fbcca")),
        )
        label_values = np.asarray(labels, dtype=object)
        idle_index = int(np.where(label_values == "idle")[0][0])
        if str(getattr(model, "gate_policy", "")) == CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY:
            floor_th = getattr(model, "lrt_window_floor_th", None)
            effective_window_th = max(
                float(getattr(model, "lrt_window_th", 0.0)),
                float(floor_th) if floor_th is not None else float(getattr(model, "lrt_window_th", 0.0)),
            )
            gate_mask = (lrt_evidence >= effective_window_th) & _score_shape_gate_mask_for_model(model, item.feature_matrix)
            if getattr(model, "frequency_specific_control_state_gates", None):
                gate_mask &= _frequency_specific_gate_mask_for_model(
                    model,
                    probs=smoothed,
                    labels=labels,
                    feature_matrix=item.feature_matrix,
                    lrt_evidence=lrt_evidence,
                )
        else:
            gate_mask = (1.0 - smoothed[:, idle_index]) >= float(getattr(model, "command_confidence_th", 0.0))
        subtype_or_true = subtype if subtype else true_label
        if subtype == "ns2":
            ns2_duration_total_sec += float(item.duration_sec)
            if str(getattr(model, "gate_policy", "")) == CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY:
                async_pred_label, _confidence, _first_index = _predict_lrt_multiwindow_reject_trial_from_probs(
                    model,
                    smoothed,
                    labels,
                    lrt_evidence,
                    min_enter_windows=max(1, int(min_enter_windows)),
                    max_gap_windows=max(0, int(max_gap_windows)),
                    feature_matrix=item.feature_matrix,
                )
            else:
                async_pred_label, _confidence, _first_index = _predict_fbcca_lda5_trial_from_probs(
                    model,
                    smoothed,
                    labels,
                    min_enter_windows=max(1, int(min_enter_windows)),
                    max_gap_windows=max(0, int(max_gap_windows)),
                )
            if async_pred_label != "idle":
                event_freq_key = _label_to_freq_key(async_pred_label)
                if event_freq_key is not None:
                    ns2_event_fp_by_freq[event_freq_key] += 1
        for row_index, pred_index in enumerate(pred_indices):
            pred_label = str(label_values[int(pred_index)])
            selected_freq = _label_to_freq_key(pred_label)
            confusion[(subtype_or_true, selected_freq or "idle", "window")] += 1
            if subtype != "ns2" or selected_freq is None:
                continue
            payload = by_freq.setdefault(
                selected_freq,
                {
                    "dataset": str(dataset),
                    "subject": "ALL",
                    "split_index": int(split_index),
                    "recipe_id": str(recipe_id),
                    "gate_variant": parse_classifier_gate_variant(gate_variant),
                    "freq": selected_freq,
                    "NS2_total_windows": 0,
                    "NS2_FP_windows": 0,
                    "_top1": [],
                    "_margin": [],
                    "_ratio": [],
                    "_entropy": [],
                    "_lrt": [],
                },
            )
            subject_payload = by_subject_freq.setdefault(
                selected_freq,
                {
                    "dataset": str(dataset),
                    "subject": str(subject),
                    "split_index": int(split_index),
                    "recipe_id": str(recipe_id),
                    "gate_variant": parse_classifier_gate_variant(gate_variant),
                    "freq": selected_freq,
                    "NS2_total_windows": 0,
                    "NS2_FP_windows": 0,
                },
            )
            payload["NS2_total_windows"] = int(payload["NS2_total_windows"]) + 1
            subject_payload["NS2_total_windows"] = int(subject_payload["NS2_total_windows"]) + 1
            if bool(gate_mask[row_index]):
                payload["NS2_FP_windows"] = int(payload["NS2_FP_windows"]) + 1
                subject_payload["NS2_FP_windows"] = int(subject_payload["NS2_FP_windows"]) + 1
            feature_row = gate_features[row_index]
            payload["_top1"].append(float(feature_row[name_to_index["top1_score"]]))
            payload["_margin"].append(float(feature_row[name_to_index["margin"]]))
            payload["_ratio"].append(float(feature_row[name_to_index["ratio"]]))
            payload["_entropy"].append(float(feature_row[name_to_index["score_entropy"]]))
            payload["_lrt"].append(float(feature_row[name_to_index["lrt_evidence"]]))

    per_freq_eval = _evaluate_fbcca_lda5_model(
        model,
        scored_trials,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
        probability_cache=cache,
    )
    eval_payload = _evaluation_payload(per_freq_eval)
    per_freq_recall = dict(eval_payload.get("per_frequency_recall", {}) or {})
    async_metrics = dict(eval_payload.get("async_metrics", {}) or {})
    per_freq_recall_2p5 = dict(async_metrics.get("per_frequency_recall_at_2.5s", {}) or {})
    per_freq_latency = dict(async_metrics.get("per_frequency_detection_latency_s", {}) or {})

    by_freq_rows: list[dict[str, Any]] = []
    ns2_minutes = float(ns2_duration_total_sec) / 60.0
    for freq_key, payload in sorted(by_freq.items()):
        total = int(payload.get("NS2_total_windows", 0) or 0)
        fp = int(payload.get("NS2_FP_windows", 0) or 0)
        event_fp = int(ns2_event_fp_by_freq.get(freq_key, 0))
        row = {
            **payload,
            "NS2_FP_events": int(event_fp),
            "NS2_FP_per_min": float(event_fp / ns2_minutes) if ns2_minutes > 1e-12 else None,
            "FP_share": float(fp / total) if total else None,
            "mean_top1_score": _finite_or_none(np.mean(payload.pop("_top1")) if payload.get("_top1") else float("nan")),
            "mean_margin": _finite_or_none(np.mean(payload.pop("_margin")) if payload.get("_margin") else float("nan")),
            "mean_ratio": _finite_or_none(np.mean(payload.pop("_ratio")) if payload.get("_ratio") else float("nan")),
            "mean_entropy": _finite_or_none(np.mean(payload.pop("_entropy")) if payload.get("_entropy") else float("nan")),
            "mean_lrt_evidence": _finite_or_none(np.mean(payload.pop("_lrt")) if payload.get("_lrt") else float("nan")),
        }
        by_freq_rows.append(row)

    by_subject_rows: list[dict[str, Any]] = []
    for freq_key, payload in sorted(by_subject_freq.items()):
        event_fp = int(ns2_event_fp_by_freq.get(freq_key, 0))
        by_subject_rows.append(
            {
                **payload,
                "NS2_FP_events": int(event_fp),
                "NS2_FP_per_min": float(event_fp / ns2_minutes) if ns2_minutes > 1e-12 else None,
                "command_recall_for_freq": _finite_or_none(per_freq_recall.get(freq_key)),
                "recall_at_2.5_for_freq": _finite_or_none(per_freq_recall_2p5.get(freq_key)),
                "detection_latency_for_freq": _finite_or_none(per_freq_latency.get(freq_key)),
            }
        )

    confusion_rows = [
        {
            "dataset": str(dataset),
            "subject": str(subject),
            "split_index": int(split_index),
            "recipe_id": str(recipe_id),
            "gate_variant": parse_classifier_gate_variant(gate_variant),
            "true_state": str(true_state),
            "selected_freq": str(selected_freq),
            "unit": str(unit),
            "count": int(count),
        }
        for (true_state, selected_freq, unit), count in sorted(confusion.items())
    ]
    return {
        "ns2_by_selected_freq": by_freq_rows,
        "ns2_by_subject_freq": by_subject_rows,
        "selected_freq_confusion": confusion_rows,
    }


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
        "selected_freqs": [float(freq) for freq in freqs],
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
        "selected_freqs": [float(freq) for freq in freqs],
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
    smoothing_windows: int = 1,
    threshold_policy: str = DEFAULT_CLASSIFIER_THRESHOLD_POLICY,
    score_bank_mode: str = DEFAULT_SCORE_BANK_MODE,
    calibration_scored: Optional[Sequence[ScoredTrial]] = None,
    holdout_scored: Optional[Sequence[ScoredTrial]] = None,
    base_model: Optional[FBCCALDA5Model] = None,
    clean_idle_scored: Optional[Sequence[ScoredTrial]] = None,
    clean_idle_support: Optional[Mapping[str, Any]] = None,
    gate_variant_params: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    resolved_gate_variant_params = dict(gate_variant_params or {"gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW})
    resolved_gate_variant = parse_classifier_gate_variant(resolved_gate_variant_params.get("gate_variant"))
    gate_variant_token = _classifier_gate_variant_token(resolved_gate_variant, resolved_gate_variant_params)
    recipe_id = _classifier_recipe_id_with_smoothing(
        win_sec=float(win_sec),
        min_enter_windows=int(min_enter_windows),
        max_gap_windows=max(0, int(max_gap_windows)),
        smoothing_windows=max(1, int(smoothing_windows)),
        gate_policy=str(threshold_policy),
        gate_variant=resolved_gate_variant,
        variant_token=gate_variant_token,
    )
    if calibration_scored is None or holdout_scored is None:
        full_bank_freqs = _full_bank_freqs_for_dataset(
            dataset=spec.dataset,
            score_bank_mode=score_bank_mode,
            fallback_freqs=freqs,
        )
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
            score_bank_mode=score_bank_mode,
            full_bank_freqs=full_bank_freqs,
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
        smoothing_windows=max(1, int(smoothing_windows)),
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
    if clean_idle_support is not None:
        support_payload = dict(clean_idle_support)
        if support_payload.get("supported") and clean_idle_scored is not None:
            eval_payload["clean_idle_proxy_metrics"] = _evaluate_clean_idle_proxy_from_cache(
                model,
                list(clean_idle_scored),
                win_sec=float(win_sec),
                step_sec=float(step_sec),
                min_enter_windows=max(1, int(min_enter_windows)),
                max_gap_windows=max(0, int(max_gap_windows)),
            )
            eval_payload["no_control_subtype_metrics"] = _evaluate_no_control_subtypes_from_cache(
                model,
                list(clean_idle_scored),
                win_sec=float(win_sec),
                step_sec=float(step_sec),
                min_enter_windows=max(1, int(min_enter_windows)),
                max_gap_windows=max(0, int(max_gap_windows)),
            )
            eval_payload["no_control_subtype_by_frequency"] = _no_control_subtype_by_frequency_from_cache(
                model,
                list(clean_idle_scored),
                win_sec=float(win_sec),
                step_sec=float(step_sec),
            )
        else:
            eval_payload["clean_idle_proxy_metrics"] = {
                **support_payload,
                "idle_fp_per_min": None,
                "idle_trial_fp_rate": None,
            }
    diagnostics = _ns2_selected_freq_diagnostic_rows(
        model,
        list(holdout_scored),
        dataset=str(spec.dataset),
        subject=str(spec.subject),
        split_index=int(split_plan.split_index),
        recipe_id=recipe_id,
        gate_variant=resolved_gate_variant,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
    )
    logistic_trace_diagnostics: dict[str, list[dict[str, Any]]] = {}
    tenp5_veto_diagnostics: dict[str, Any] = {}
    if resolved_gate_variant in {
        CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
        CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC,
    }:
        baseline_trace_model = _fit_fbcca_ridge5_model(
            calibration_scored,
            freqs=freqs,
            win_sec=float(latency_win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            smoothing_windows=max(1, int(smoothing_windows)),
            threshold_policy=str(threshold_policy),
            base_models=base_models,
            score_source_name=str(score_source_name),
            score_bank_mode=score_bank_mode,
            gate_variant_params={"gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW},
        )
        logistic_trace_diagnostics = _trace_rows_for_frequency_specific_logistic_case(
            baseline_model=baseline_trace_model,
            candidate_model=model,
            scored_trials=list(holdout_scored),
            dataset=str(spec.dataset),
            subject=str(spec.subject),
            split_index=int(split_plan.split_index),
            recipe_id=recipe_id,
            frequency_profile=_frequency_profile_name(freqs),
            frequency_set_id="",
            win_sec=float(latency_win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
        )
    if resolved_gate_variant == CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO:
        baseline_trace_model = _fit_fbcca_ridge5_model(
            calibration_scored,
            freqs=freqs,
            win_sec=float(latency_win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            smoothing_windows=max(1, int(smoothing_windows)),
            threshold_policy=str(threshold_policy),
            base_models=base_models,
            score_source_name=str(score_source_name),
            score_bank_mode=score_bank_mode,
            gate_variant_params={"gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW},
        )
        tenp5_veto_diagnostics = _trace_rows_for_tenp5_ns2_veto_case(
            baseline_model=baseline_trace_model,
            candidate_model=model,
            scored_trials=list(holdout_scored),
            dataset=str(spec.dataset),
            subject=str(spec.subject),
            split_index=int(split_plan.split_index),
            recipe_id=recipe_id,
            frequency_profile=_frequency_profile_name(freqs),
            frequency_set_id="",
            win_sec=float(latency_win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
        )
        summary_rows = list(dict(tenp5_veto_diagnostics).get("tenp5_ns2_veto_summary_rows", []) or [])
        if summary_rows:
            row0 = dict(summary_rows[0])
            metrics = dict(_extract_row_metrics(eval_payload))
            metrics.update(
                {
                    "fixed_ns2_fp_count": float(row0.get("fixed_ns2_fp_count", 0) or 0),
                    "lost_command_tp_count": float(row0.get("lost_command_tp_count", 0) or 0),
                    "veto_precision": _safe_float(row0.get("veto_precision"), float("nan")),
                    "tp_loss_per_fixed_fp": _safe_float(row0.get("tp_loss_per_fixed_fp"), float("inf")),
                }
            )
            eval_payload["tenp5_ns2_veto_metrics"] = metrics
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
        score_bank_mode=score_bank_mode,
        decoder_name="fbcca",
        decoder_model_params={"Nh": 5, "subband_weight_mode": "chen_fixed"},
        full_reference_bank_freqs=(
            _full_reference_bank_freqs_from_scored(calibration_scored, fallback_freqs=freqs)
            if _parse_score_bank_mode(score_bank_mode) == "full_reference_bank"
            else ()
        ),
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
        "selected_freqs": [float(freq) for freq in freqs],
        "split_summary": dict(split_summary),
        "calibration_profile": {
            "status": "ok",
            "classifier": "fbcca_score_lda_5class",
            "fit_summary": dict(model.fit_summary),
            "command_confidence_th": float(model.command_confidence_th),
            "min_enter_windows": int(min_enter_windows),
            "max_gap_windows": max(0, int(max_gap_windows)),
            "smoothing_windows": max(1, int(smoothing_windows)),
            "gate_policy": str(getattr(model, "gate_policy", CLASSIFIER_CONFIDENCE_GATE_POLICY)),
            "evidence_decision_th": float(getattr(model, "evidence_decision_th", 0.0)),
            "evidence_enter_th": float(getattr(model, "evidence_enter_th", 0.0)),
            "evidence_decay": float(getattr(model, "evidence_decay", DEFAULT_ADAPTIVE_EVIDENCE_DECAY)),
            "threshold_policy": _parse_classifier_threshold_policy(threshold_policy),
            "feature_count": int(model.feature_mean.shape[0]),
            "feature_names": _classifier_feature_names(freqs, score_bank_mode=score_bank_mode),
            "adaptive_gate_feature_names": list(ADAPTIVE_EVIDENCE_FEATURE_NAMES)
            if str(getattr(model, "gate_policy", "")) == CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY
            else [],
            "frequency_specific_control_state_gates": normalize_frequency_specific_control_state_gates(
                getattr(model, "frequency_specific_control_state_gates", None)
            ),
            "score_bank_mode": _parse_score_bank_mode(score_bank_mode),
            "candidate_artifact": candidate_artifact,
            "candidate_artifact_path": candidate_artifact_path,
        },
        "holdout_eval": eval_payload,
        "per_frequency_metrics": _per_frequency_metrics_from_eval(eval_payload, model),
        "ns2_selected_freq_diagnostics": diagnostics,
        "logistic_trace_diagnostics": logistic_trace_diagnostics,
        "tenp5_ns2_veto_diagnostics": tenp5_veto_diagnostics,
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
    smoothing_windows: int = 1,
    threshold_policy: str = DEFAULT_CLASSIFIER_THRESHOLD_POLICY,
    calibration_scored: Optional[Sequence[ScoredTrial]] = None,
    holdout_scored: Optional[Sequence[ScoredTrial]] = None,
    base_models: Optional[Sequence[FBCCARidge5Model]] = None,
    method_name: str = "fbcca_ridge5",
    score_source_name: str = "fbcca",
    score_bank_mode: str = DEFAULT_SCORE_BANK_MODE,
    decoder_name: Optional[str] = None,
    decoder_model_params: Optional[Mapping[str, Any]] = None,
    clean_idle_scored: Optional[Sequence[ScoredTrial]] = None,
    clean_idle_support: Optional[Mapping[str, Any]] = None,
    gate_variant_params: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    resolved_gate_variant_params = dict(gate_variant_params or {"gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW})
    resolved_gate_variant = parse_classifier_gate_variant(resolved_gate_variant_params.get("gate_variant"))
    gate_variant_token = _classifier_gate_variant_token(resolved_gate_variant, resolved_gate_variant_params)
    recipe_id = _classifier_recipe_id_with_smoothing(
        win_sec=float(win_sec),
        min_enter_windows=int(min_enter_windows),
        max_gap_windows=max(0, int(max_gap_windows)),
        smoothing_windows=max(1, int(smoothing_windows)),
        gate_policy=str(threshold_policy),
        gate_variant=resolved_gate_variant,
        variant_token=gate_variant_token,
    )
    latency_win_sec = _method_latency_window_sec(
        method_name=str(method_name),
        win_sec=float(win_sec),
        sampling_rate=int(sampling_rate),
    )
    if calibration_scored is None or holdout_scored is None:
        full_bank_freqs = _full_bank_freqs_for_dataset(
            dataset=spec.dataset,
            score_bank_mode=score_bank_mode,
            fallback_freqs=freqs,
        )
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
            score_bank_mode=score_bank_mode,
            full_bank_freqs=full_bank_freqs,
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
        smoothing_windows=max(1, int(smoothing_windows)),
        threshold_policy=str(threshold_policy),
        base_models=base_models,
        score_source_name=str(score_source_name),
        score_bank_mode=score_bank_mode,
        gate_variant_params=resolved_gate_variant_params,
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
    if clean_idle_support is not None:
        support_payload = dict(clean_idle_support)
        if support_payload.get("supported") and clean_idle_scored is not None:
            eval_payload["clean_idle_proxy_metrics"] = _evaluate_clean_idle_proxy_from_cache(
                model,
                list(clean_idle_scored),
                win_sec=float(latency_win_sec),
                step_sec=float(step_sec),
                min_enter_windows=max(1, int(min_enter_windows)),
                max_gap_windows=max(0, int(max_gap_windows)),
            )
            eval_payload["no_control_subtype_metrics"] = _evaluate_no_control_subtypes_from_cache(
                model,
                list(clean_idle_scored),
                win_sec=float(latency_win_sec),
                step_sec=float(step_sec),
                min_enter_windows=max(1, int(min_enter_windows)),
                max_gap_windows=max(0, int(max_gap_windows)),
            )
            eval_payload["no_control_subtype_by_frequency"] = _no_control_subtype_by_frequency_from_cache(
                model,
                list(clean_idle_scored),
                win_sec=float(latency_win_sec),
                step_sec=float(step_sec),
            )
        else:
            eval_payload["clean_idle_proxy_metrics"] = {
                **support_payload,
                "idle_fp_per_min": None,
                "idle_trial_fp_rate": None,
            }
    diagnostics = _ns2_selected_freq_diagnostic_rows(
        model,
        list(holdout_scored),
        dataset=str(spec.dataset),
        subject=str(spec.subject),
        split_index=int(split_plan.split_index),
        recipe_id=recipe_id,
        gate_variant=resolved_gate_variant,
        win_sec=float(latency_win_sec),
        step_sec=float(step_sec),
        min_enter_windows=max(1, int(min_enter_windows)),
        max_gap_windows=max(0, int(max_gap_windows)),
    )
    logistic_trace_diagnostics: dict[str, list[dict[str, Any]]] = {}
    tenp5_veto_diagnostics: dict[str, Any] = {}
    if resolved_gate_variant in {
        CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
        CLASSIFIER_GATE_VARIANT_CONDITIONAL_FREQUENCY_SPECIFIC_LOGISTIC,
    }:
        baseline_trace_model = _fit_fbcca_ridge5_model(
            calibration_scored,
            freqs=freqs,
            win_sec=float(latency_win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            smoothing_windows=max(1, int(smoothing_windows)),
            threshold_policy=str(threshold_policy),
            base_models=base_models,
            score_source_name=str(score_source_name),
            score_bank_mode=score_bank_mode,
            gate_variant_params={"gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW},
        )
        logistic_trace_diagnostics = _trace_rows_for_frequency_specific_logistic_case(
            baseline_model=baseline_trace_model,
            candidate_model=model,
            scored_trials=list(holdout_scored),
            dataset=str(spec.dataset),
            subject=str(spec.subject),
            split_index=int(split_plan.split_index),
            recipe_id=recipe_id,
            frequency_profile=_frequency_profile_name(freqs),
            frequency_set_id="",
            win_sec=float(latency_win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
        )
    if resolved_gate_variant == CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO:
        baseline_trace_model = _fit_fbcca_ridge5_model(
            calibration_scored,
            freqs=freqs,
            win_sec=float(latency_win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
            smoothing_windows=max(1, int(smoothing_windows)),
            threshold_policy=str(threshold_policy),
            base_models=base_models,
            score_source_name=str(score_source_name),
            score_bank_mode=score_bank_mode,
            gate_variant_params={"gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW},
        )
        tenp5_veto_diagnostics = _trace_rows_for_tenp5_ns2_veto_case(
            baseline_model=baseline_trace_model,
            candidate_model=model,
            scored_trials=list(holdout_scored),
            dataset=str(spec.dataset),
            subject=str(spec.subject),
            split_index=int(split_plan.split_index),
            recipe_id=recipe_id,
            frequency_profile=_frequency_profile_name(freqs),
            frequency_set_id="",
            win_sec=float(latency_win_sec),
            step_sec=float(step_sec),
            min_enter_windows=max(1, int(min_enter_windows)),
            max_gap_windows=max(0, int(max_gap_windows)),
        )
        summary_rows = list(dict(tenp5_veto_diagnostics).get("tenp5_ns2_veto_summary_rows", []) or [])
        if summary_rows:
            row0 = dict(summary_rows[0])
            metrics = dict(_extract_row_metrics(eval_payload))
            metrics.update(
                {
                    "fixed_ns2_fp_count": float(row0.get("fixed_ns2_fp_count", 0) or 0),
                    "lost_command_tp_count": float(row0.get("lost_command_tp_count", 0) or 0),
                    "veto_precision": _safe_float(row0.get("veto_precision"), float("nan")),
                    "tp_loss_per_fixed_fp": _safe_float(row0.get("tp_loss_per_fixed_fp"), float("inf")),
                }
            )
            eval_payload["tenp5_ns2_veto_metrics"] = metrics
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
        score_bank_mode=score_bank_mode,
        decoder_name=str(decoder_name or _score_method_spec(method_name).decoder_name),
        decoder_model_params=dict(decoder_model_params or _score_method_spec(method_name).decoder_model_params),
        full_reference_bank_freqs=(
            _full_reference_bank_freqs_from_scored(calibration_scored, fallback_freqs=freqs)
            if _parse_score_bank_mode(score_bank_mode) == "full_reference_bank"
            else ()
        ),
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
        "selected_freqs": [float(freq) for freq in freqs],
        "gate_variant": resolved_gate_variant,
        "gate_variant_params": resolved_gate_variant_params,
        "split_summary": dict(split_summary),
        "calibration_profile": {
            "status": "ok",
            "classifier": _classifier_name_for_model(model, score_source_name=score_source_name),
            "fit_summary": dict(model.fit_summary),
            "command_confidence_th": float(model.command_confidence_th),
            "min_enter_windows": int(min_enter_windows),
            "max_gap_windows": max(0, int(max_gap_windows)),
            "smoothing_windows": max(1, int(smoothing_windows)),
            "gate_policy": str(getattr(model, "gate_policy", CLASSIFIER_CONFIDENCE_GATE_POLICY)),
            "gate_variant": resolved_gate_variant,
            "gate_variant_params": resolved_gate_variant_params,
            "lrt_window_th": float(getattr(model, "lrt_window_th", 0.0)),
            "lrt_window_floor_th": (
                None
                if getattr(model, "lrt_window_floor_th", None) is None
                else float(getattr(model, "lrt_window_floor_th"))
            ),
            "score_shape_margin_th": (
                None
                if getattr(model, "score_shape_margin_th", None) is None
                else float(getattr(model, "score_shape_margin_th"))
            ),
            "score_shape_ratio_th": (
                None
                if getattr(model, "score_shape_ratio_th", None) is None
                else float(getattr(model, "score_shape_ratio_th"))
            ),
            "score_shape_entropy_th": (
                None
                if getattr(model, "score_shape_entropy_th", None) is None
                else float(getattr(model, "score_shape_entropy_th"))
            ),
            "weak_subject_guard_active": bool(getattr(model, "weak_subject_guard_active", False)),
            "weak_subject_guard_reasons": list(getattr(model, "weak_subject_guard_reasons", ()) or ()),
            "evidence_decision_th": float(getattr(model, "evidence_decision_th", 0.0)),
            "evidence_enter_th": float(getattr(model, "evidence_enter_th", 0.0)),
            "evidence_decay": float(getattr(model, "evidence_decay", DEFAULT_ADAPTIVE_EVIDENCE_DECAY)),
            "threshold_policy": _parse_classifier_threshold_policy(threshold_policy),
            "feature_count": int(model.feature_mean.shape[0]),
            "feature_names": _classifier_feature_names(
                freqs,
                score_source_name=score_source_name,
                score_bank_mode=score_bank_mode,
            ),
            "adaptive_gate_feature_names": list(ADAPTIVE_EVIDENCE_FEATURE_NAMES)
            if str(getattr(model, "gate_policy", "")) == CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY
            else [],
            "frequency_specific_control_state_gates": normalize_frequency_specific_control_state_gates(
                getattr(model, "frequency_specific_control_state_gates", None)
            ),
            "score_source_name": str(score_source_name).strip().lower(),
            "score_bank_mode": _parse_score_bank_mode(score_bank_mode),
            "decoder_name": str(decoder_name or _score_method_spec(method_name).decoder_name),
            "decoder_model_params": dict(decoder_model_params or _score_method_spec(method_name).decoder_model_params),
            "l2": float(model.l2),
            "candidate_artifact": candidate_artifact,
            "candidate_artifact_path": candidate_artifact_path,
        },
        "holdout_eval": eval_payload,
        "per_frequency_metrics": _per_frequency_metrics_from_eval(eval_payload, model),
        "ns2_selected_freq_diagnostics": diagnostics,
        "logistic_trace_diagnostics": logistic_trace_diagnostics,
        "tenp5_ns2_veto_diagnostics": tenp5_veto_diagnostics,
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
        "selected_freqs": [float(freq) for freq in freqs],
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
    grouped: dict[tuple[str, str, int, float, str, str, bool, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        calibration_blocks = tuple(int(item) for item in row.get("calibration_blocks", []))
        aggregate_recipe_id = str(row.get("aggregate_recipe_id") or row.get("recipe_id", ""))
        selected_freqs = row.get("selected_freqs", None)
        if selected_freqs is None:
            selected_freqs = dict(row.get("split_summary", {}) or {}).get("selected_freqs", [])
        frequency_set_id = str(row.get("frequency_set_id") or "")
        if not frequency_set_id:
            frequency_set_id = f"freqs_{_freq_token(selected_freqs)}" if selected_freqs else ""
        gate_variant = parse_classifier_gate_variant(row.get("gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW))
        key = (
            str(row.get("method", "")),
            aggregate_recipe_id,
            int(len(calibration_blocks)),
            float(dict(row.get("split_summary", {}) or {}).get("idle_multiplier", 0.0)),
            frequency_set_id,
            gate_variant,
            bool(row.get("nc_calibration_simulation", False)),
            _safe_float(row.get("nc_seconds"), -1.0),
            str(row.get("nc_source", "")),
            str(row.get("nc_gate_type", "")),
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
                    "nc_calibration_simulation": bool(subject_rows[0].get("nc_calibration_simulation", False)),
                    "nc_seconds": _finite_or_none(subject_rows[0].get("nc_seconds")),
                    "nc_source": str(subject_rows[0].get("nc_source", "")),
                    "nc_gate_type": str(subject_rows[0].get("nc_gate_type", "")),
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
                    "mean_real_idle_fp_per_min": _safe_mean(
                        [m.get("real_idle_fp_per_min") for m in metrics],
                        default=float("nan"),
                    ),
                    "mean_approx_idle_fp_per_min": _safe_mean(
                        [_first_finite_metric(m, ("approx_idle_fp_per_min", "idle_fp_per_min")) for m in metrics],
                        default=float("nan"),
                    ),
                    "mean_mixed_idle_fp_per_min": _safe_mean(
                        [
                            _first_finite_metric(
                                m,
                                ("mixed_idle_fp_per_min", "real_idle_fp_per_min", "approx_idle_fp_per_min", "idle_fp_per_min"),
                            )
                            for m in metrics
                        ],
                        default=float("inf"),
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
                    "mean_clean_idle_proxy_supported": float(
                        np.mean([_safe_float(m.get("clean_idle_proxy_supported"), 0.0) for m in metrics])
                    ),
                    "mean_clean_idle_proxy_fp_per_min": float(
                        np.mean([_safe_float(m.get("clean_idle_proxy_fp_per_min"), float("nan")) for m in metrics])
                    ),
                    "mean_ns1_fp_per_min": float(
                        np.mean([_safe_float(m.get("ns1_fp_per_min"), float("nan")) for m in metrics])
                    ),
                    "mean_ns2_fp_per_min": float(
                        np.mean([_safe_float(m.get("ns2_fp_per_min"), float("nan")) for m in metrics])
                    ),
                    "mean_ns3_fp_per_min": float(
                        np.mean([_safe_float(m.get("ns3_fp_per_min"), float("nan")) for m in metrics])
                    ),
                    "mean_ns_all_fp_per_min": float(
                        np.mean([_safe_float(m.get("ns_all_fp_per_min"), float("nan")) for m in metrics])
                    ),
                    "mean_cs_control_recall": float(
                        np.mean([_safe_float(m.get("cs_control_recall"), 0.0) for m in metrics])
                    ),
                    "sum_fixed_ns2_fp_count": float(
                        np.sum([_safe_float(m.get("fixed_ns2_fp_count"), 0.0) for m in metrics])
                    ),
                    "sum_lost_command_tp_count": float(
                        np.sum([_safe_float(m.get("lost_command_tp_count"), 0.0) for m in metrics])
                    ),
                }
            )

        def subject_metric(field: str, *, default: float = 0.0, missing_default: Optional[float] = None) -> float:
            if not subject_summaries:
                return float(default)
            if missing_default is None:
                return float(np.mean([_safe_float(item.get(field), default) for item in subject_summaries]))
            return _safe_mean(
                [item.get(field) for item in subject_summaries],
                default=float(missing_default),
            )
        calibration_patterns = sorted(
            {
                tuple(int(item) for item in row.get("calibration_blocks", []))
                for row in key_rows
            }
        )
        selected_freq_patterns = sorted(
            {
                tuple(float(freq) for freq in (row.get("selected_freqs") or dict(row.get("split_summary", {}) or {}).get("selected_freqs", []) or []))
                for row in key_rows
            }
        )
        per_subject_selected_freqs: dict[str, list[float]] = {}
        for row in key_rows:
            selected_freqs = row.get("selected_freqs") or dict(row.get("split_summary", {}) or {}).get("selected_freqs", [])
            if not selected_freqs:
                continue
            subject_key = f"{str(row.get('dataset', ''))}:{str(row.get('subject', ''))}"
            per_subject_selected_freqs[subject_key] = [float(freq) for freq in selected_freqs]
        frequency_modes = sorted(
            {
                str(row.get("frequency_selection_mode") or dict(row.get("split_summary", {}) or {}).get("frequency_selection_mode", "") or "")
                for row in key_rows
            }
        )
        selected_recipe_counts: dict[str, int] = {}
        gate_param_rows: list[dict[str, Any]] = []
        validation_metrics_by_row: list[dict[str, Any]] = []
        weak_guard_subjects: set[str] = set()
        for row in key_rows:
            selected_recipe = str(row.get("selected_recipe_id") or row.get("recipe_id", ""))
            if not selected_recipe:
                continue
            selected_recipe_counts[selected_recipe] = int(selected_recipe_counts.get(selected_recipe, 0)) + 1
            calibration_profile = dict(row.get("calibration_profile", {}) or {})
            fit_summary = dict(calibration_profile.get("fit_summary", {}) or {})
            validation_payload = dict(fit_summary.get("gate_validation_metrics", {}) or {})
            validation_metrics = dict(validation_payload.get("metrics", {}) or {})
            if validation_metrics:
                validation_metrics_by_row.append(validation_metrics)
            gate_param_rows.append(
                {
                    "subject": f"{row.get('dataset','')}:{row.get('subject','')}",
                    "split_index": int(row.get("split_index", 0) or 0),
                    "gate_variant": parse_classifier_gate_variant(row.get("gate_variant", calibration_profile.get("gate_variant", fit_summary.get("gate_variant")))),
                    "gate_variant_params": dict(row.get("gate_variant_params", calibration_profile.get("gate_variant_params", fit_summary.get("gate_variant_params", {}))) or {}),
                    "fit_split": str(fit_summary.get("threshold_fit_split", fit_summary.get("fit_split", "calibration_blocks"))),
                    "test_split": str(fit_summary.get("test_split", "holdout_blocks")),
                    "lrt_window_th": _finite_or_none(calibration_profile.get("lrt_window_th", fit_summary.get("lrt_window_th"))),
                    "lrt_window_floor_th": _finite_or_none(calibration_profile.get("lrt_window_floor_th", fit_summary.get("lrt_window_floor_th"))),
                    "subject_floor_global_lrt_th": _finite_or_none(fit_summary.get("subject_floor_global_lrt_th")),
                    "subject_floor_idle_lrt_th": _finite_or_none(fit_summary.get("subject_floor_idle_lrt_th")),
                    "ns2_lrt_window_p95": _finite_or_none(fit_summary.get("ns2_lrt_window_p95")),
                    "ns2_lrt_window_floor_th": _finite_or_none(fit_summary.get("ns2_lrt_window_floor_th")),
                    "ns2_threshold_source": str(fit_summary.get("ns2_threshold_source", "")),
                    "score_shape_margin_th": _finite_or_none(calibration_profile.get("score_shape_margin_th", fit_summary.get("score_shape_margin_th"))),
                    "score_shape_ratio_th": _finite_or_none(calibration_profile.get("score_shape_ratio_th", fit_summary.get("score_shape_ratio_th"))),
                    "score_shape_entropy_th": _finite_or_none(calibration_profile.get("score_shape_entropy_th", fit_summary.get("score_shape_entropy_th"))),
                    "frequency_specific_control_state_gates": normalize_frequency_specific_control_state_gates(
                        calibration_profile.get(
                            "frequency_specific_control_state_gates",
                            fit_summary.get("frequency_specific_control_state_gates"),
                        )
                    ),
                    "frequency_specific_grid_selection_policy": str(
                        fit_summary.get("frequency_specific_grid_selection_policy", "")
                    ),
                    "tenp5_ns2_veto_fit_policy": str(fit_summary.get("tenp5_ns2_veto_fit_policy", "")),
                    "tenp5_ns2_veto_feature_names": list(fit_summary.get("tenp5_ns2_veto_feature_names", []) or []),
                    "gate_validation_metrics": validation_payload,
                    "weak_subject_guard_active": bool(calibration_profile.get("weak_subject_guard_active", fit_summary.get("weak_subject_guard_active", False))),
                    "weak_subject_guard_reasons": list(calibration_profile.get("weak_subject_guard_reasons", fit_summary.get("weak_subject_guard_reasons", [])) or []),
                    "nc_calibration_simulation": bool(row.get("nc_calibration_simulation", False)),
                    "nc_seconds": _finite_or_none(row.get("nc_seconds")),
                    "nc_source": str(row.get("nc_source", "")),
                    "nc_gate_type": str(row.get("nc_gate_type", "")),
                    "nc_calibration_support_status": str(row.get("nc_calibration_support_status", "")),
                }
            )
            if bool(gate_param_rows[-1].get("weak_subject_guard_active", False)):
                weak_guard_subjects.add(f"{row.get('dataset','')}:{row.get('subject','')}")
        validation_summary = {}
        if validation_metrics_by_row:
            validation_summary = {
                "supported": True,
                "split": "calibration_gate_validation_trials",
                "mean_idle_fp_per_min": _safe_mean(
                    [_first_finite_metric(item, ("mixed_idle_fp_per_min", "idle_fp_per_min")) for item in validation_metrics_by_row],
                    default=float("inf"),
                ),
                "mean_ns2_fp_per_min": _safe_mean(
                    [item.get("ns2_fp_per_min") for item in validation_metrics_by_row],
                    default=float("nan"),
                ),
                "mean_control_recall": _safe_mean(
                    [item.get("control_recall") for item in validation_metrics_by_row],
                    default=0.0,
                ),
                "mean_control_recall_at_2.5s": _safe_mean(
                    [item.get("control_recall_at_2.5s") for item in validation_metrics_by_row],
                    default=0.0,
                ),
                "mean_async_macro_f1_5class": _safe_mean(
                    [item.get("async_macro_f1_5class") for item in validation_metrics_by_row],
                    default=0.0,
                ),
                "mean_detection_latency_s": _safe_mean(
                    [item.get("detection_latency_s") for item in validation_metrics_by_row],
                    default=float("inf"),
                ),
                "row_count": int(len(validation_metrics_by_row)),
            }
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
            "frequency_set_id": str(key[4]),
            "gate_variant": str(key[5]),
            "nc_calibration_simulation": bool(key[6]),
            "nc_seconds": None if _safe_float(key[7], -1.0) < 0 else float(key[7]),
            "nc_source": str(key[8]),
            "nc_gate_type": str(key[9]),
            "nc_calibration_support_statuses": sorted(
                {
                    str(row.get("nc_calibration_support_status", ""))
                    for row in key_rows
                    if str(row.get("nc_calibration_support_status", "")).strip()
                }
            ),
            "nc_calibration_provenance": [
                dict(row.get("nc_calibration_provenance", {}) or {})
                for row in key_rows
                if row.get("nc_calibration_provenance")
            ],
            "selected_freqs": (
                [float(freq) for freq in selected_freq_patterns[0]]
                if len(selected_freq_patterns) == 1
                else []
            ),
            "selected_freq_patterns": [[float(freq) for freq in pattern] for pattern in selected_freq_patterns],
            "per_subject_selected_freqs": per_subject_selected_freqs,
            "frequency_selection_mode": (
                str(frequency_modes[0])
                if len(frequency_modes) == 1
                else "mixed"
            ),
            "frequency_set_coverage_subject_count": int(len(subject_summaries)),
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
            "mean_real_idle_fp_per_min": subject_metric("mean_real_idle_fp_per_min", missing_default=float("nan")),
            "mean_approx_idle_fp_per_min": _safe_mean(
                [
                    _first_finite_metric(item, ("mean_approx_idle_fp_per_min", "mean_idle_fp_per_min"))
                    for item in subject_summaries
                ],
                default=float("nan"),
            ),
            "mean_mixed_idle_fp_per_min": _safe_mean(
                [
                    _first_finite_metric(
                        item,
                        (
                            "mean_mixed_idle_fp_per_min",
                            "mean_real_idle_fp_per_min",
                            "mean_approx_idle_fp_per_min",
                            "mean_idle_fp_per_min",
                        ),
                    )
                    for item in subject_summaries
                ],
                default=float("inf"),
            ),
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
            "mean_clean_idle_proxy_supported": subject_metric("mean_clean_idle_proxy_supported"),
            "mean_clean_idle_proxy_fp_per_min": subject_metric("mean_clean_idle_proxy_fp_per_min"),
            "mean_ns1_fp_per_min": subject_metric("mean_ns1_fp_per_min"),
            "mean_ns2_fp_per_min": subject_metric("mean_ns2_fp_per_min"),
            "mean_ns3_fp_per_min": subject_metric("mean_ns3_fp_per_min"),
            "mean_ns_all_fp_per_min": subject_metric("mean_ns_all_fp_per_min"),
            "mean_cs_control_recall": subject_metric("mean_cs_control_recall"),
            "fixed_ns2_fp_count": float(
                np.sum([_safe_float(item.get("sum_fixed_ns2_fp_count"), 0.0) for item in subject_summaries])
            ),
            "lost_command_tp_count": float(
                np.sum([_safe_float(item.get("sum_lost_command_tp_count"), 0.0) for item in subject_summaries])
            ),
            "selected_recipe_counts": selected_recipe_counts,
            "gate_params": gate_param_rows,
            "gate_validation_summary": validation_summary,
            "per_frequency_summary": _per_frequency_summary_from_recipe_rows(key_rows),
            "weak_guard_subjects": sorted(weak_guard_subjects),
            "frequency_profile": (
                str(key_rows[0].get("frequency_profile", ""))
                if key_rows
                and len({str(row.get("frequency_profile", "")) for row in key_rows}) == 1
                else "mixed"
            ),
            "reject_gate": (
                str(key_rows[0].get("reject_gate", ""))
                if key_rows
                and len({str(row.get("reject_gate", "")) for row in key_rows}) == 1
                else "mixed"
            ),
            "implementation_level": (
                str(key_rows[0].get("implementation_level", ""))
                if key_rows
                and len({str(row.get("implementation_level", "")) for row in key_rows}) == 1
                else "mixed"
            ),
            "paper_faithful": bool(
                key_rows
                and all(bool(row.get("paper_faithful", False)) for row in key_rows)
            ),
            "engineering_approx": bool(
                key_rows
                and any(bool(row.get("engineering_approx", False)) for row in key_rows)
            ),
            "subjects": subject_summaries,
        }
        fixed_count = _safe_float(summary.get("fixed_ns2_fp_count"), 0.0)
        lost_count = _safe_float(summary.get("lost_command_tp_count"), 0.0)
        summary["veto_precision"] = (
            float(fixed_count / (fixed_count + lost_count))
            if fixed_count + lost_count > 1e-12
            else None
        )
        summary["tp_loss_per_fixed_fp"] = float(lost_count / max(fixed_count, 1.0))
        summary.update(_deployable_budget_payload(summary))
        summaries.append(summary)
    summaries.sort(key=_summary_rank_key)
    return summaries


def _summary_rank_key(summary: dict[str, Any]) -> tuple[float, ...]:
    gate_variant = parse_classifier_gate_variant(summary.get("gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW))
    validation_summary = dict(summary.get("gate_validation_summary", {}) or {})
    if (
        gate_variant
        in {
            CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
            CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
        }
        and bool(validation_summary.get("supported", False))
    ):
        return _classifier_rank_key(
            {
                "idle_fp_per_min": validation_summary.get("mean_idle_fp_per_min"),
                "mixed_idle_fp_per_min": validation_summary.get("mean_idle_fp_per_min"),
                "control_recall": validation_summary.get("mean_control_recall"),
                "control_recall_at_2.5s": validation_summary.get("mean_control_recall_at_2.5s"),
                "async_macro_f1_5class": validation_summary.get("mean_async_macro_f1_5class"),
                "detection_latency_s": validation_summary.get("mean_detection_latency_s"),
            },
            tie_breaker=_safe_float(summary.get("mean_idle_fp_per_min"), float("inf")),
        )
    return _classifier_rank_key(
        {
            "idle_fp_per_min": summary.get("mean_idle_fp_per_min"),
            "real_idle_fp_per_min": summary.get("mean_real_idle_fp_per_min"),
            "mixed_idle_fp_per_min": summary.get("mean_mixed_idle_fp_per_min"),
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


def _deployable_recipe_summaries(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(summary) for summary in summaries if bool(summary.get("deployable_budget_pass", False))]


def _ns2_safety_label(summary: Mapping[str, Any]) -> str:
    ns2 = _safe_float(summary.get("mean_ns2_fp_per_min"), float("inf"))
    if ns2 <= 1.0 + 1e-12:
        return "ns2_safe"
    if ns2 <= 1.25 + 1e-12:
        return "ns2_reduced_tradeoff"
    return "ns2_risk"


def _subject_id_token(subject: Any) -> str:
    text = str(subject or "").strip()
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    match = re.search(r"(\d+)$", text)
    if not match:
        return text.upper()
    return f"S{int(match.group(1)):02d}"


def _subgroup_definitions_payload() -> dict[str, Any]:
    return {
        "high_fp_group": list(HIGH_FP_SUBGROUP_SUBJECTS),
        "low_recall_group": list(LOW_RECALL_SUBGROUP_SUBJECTS),
        "overlap_watch": list(OVERLAP_WATCH_SUBJECTS),
        "high_risk_validation_subjects": list(HIGH_RISK_VALIDATION_SUBJECTS),
        "comparison_baseline": "high-risk subset baseline_lrtmw within the same run when available",
        "ns2_improved_rule": "high-risk NS2 FP/min reduction >= 25% vs high-risk baseline",
        "recall_degraded_rule": "low-recall recall@2.5s drop > 0.03; relaxed to > 0.05 when high-risk NS2 reduction >= 40%",
        "full24_candidate_eligible_rule": "failed=0, hard_failed=0, ns2_improved=true, recall_degraded=false, control_recall_delta>=-0.03",
    }


def _subject_subgroup_metric_payload(
    summary: Mapping[str, Any],
    subjects: Sequence[str],
) -> dict[str, Any]:
    wanted = {_subject_id_token(subject) for subject in subjects}
    matched_by_subject: dict[str, Mapping[str, Any]] = {}
    for item in list(summary.get("subjects", []) or []):
        row = dict(item)
        subject_token = _subject_id_token(row.get("subject", ""))
        if subject_token in wanted and subject_token not in matched_by_subject:
            matched_by_subject[subject_token] = row
    matched = list(matched_by_subject.values())
    return {
        "subjects_expected": list(subjects),
        "subjects_completed": sorted(matched_by_subject),
        "subject_count": int(len(matched_by_subject)),
        "idle_fp_per_min": _finite_or_none(
            _safe_mean(
                [
                    _first_finite_metric(
                        item,
                        (
                            "mean_mixed_idle_fp_per_min",
                            "mean_real_idle_fp_per_min",
                            "mean_approx_idle_fp_per_min",
                            "mean_idle_fp_per_min",
                        ),
                    )
                    for item in matched
                ],
                default=float("nan"),
            )
        ),
        "ns2_fp_per_min": _finite_or_none(
            _safe_mean([item.get("mean_ns2_fp_per_min") for item in matched], default=float("nan"))
        ),
        "control_recall": _finite_or_none(
            _safe_mean([item.get("mean_control_recall") for item in matched], default=float("nan"))
        ),
        "control_recall_at_2.5s": _finite_or_none(
            _safe_mean([item.get("mean_control_recall_at_2.5s") for item in matched], default=float("nan"))
        ),
        "detection_latency_s": _finite_or_none(
            _safe_mean([item.get("mean_detection_latency_s") for item in matched], default=float("nan"))
        ),
    }


def _candidate_subgroup_comparison_payload(
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    item = dict(summary)
    base = dict(baseline)
    high_risk = _subject_subgroup_metric_payload(item, HIGH_FP_SUBGROUP_SUBJECTS)
    high_risk_base = _subject_subgroup_metric_payload(base, HIGH_FP_SUBGROUP_SUBJECTS)
    low_recall = _subject_subgroup_metric_payload(item, LOW_RECALL_SUBGROUP_SUBJECTS)
    low_recall_base = _subject_subgroup_metric_payload(base, LOW_RECALL_SUBGROUP_SUBJECTS)
    high_ns2 = _safe_float(high_risk.get("ns2_fp_per_min"), float("nan"))
    base_high_ns2 = _safe_float(high_risk_base.get("ns2_fp_per_min"), float("nan"))
    low_recall_2p5 = _safe_float(low_recall.get("control_recall_at_2.5s"), float("nan"))
    base_low_recall_2p5 = _safe_float(low_recall_base.get("control_recall_at_2.5s"), float("nan"))
    control_recall = _safe_float(item.get("mean_control_recall"), float("nan"))
    base_control_recall = _safe_float(base.get("mean_control_recall"), float("nan"))
    high_delta = (
        float(high_ns2 - base_high_ns2)
        if np.isfinite(high_ns2) and np.isfinite(base_high_ns2)
        else float("nan")
    )
    reduction_ratio = (
        max(0.0, float((base_high_ns2 - high_ns2) / base_high_ns2))
        if np.isfinite(high_ns2) and np.isfinite(base_high_ns2) and base_high_ns2 > 1e-12
        else float("nan")
    )
    low_delta = (
        float(low_recall_2p5 - base_low_recall_2p5)
        if np.isfinite(low_recall_2p5) and np.isfinite(base_low_recall_2p5)
        else float("nan")
    )
    control_delta = (
        float(control_recall - base_control_recall)
        if np.isfinite(control_recall) and np.isfinite(base_control_recall)
        else float("nan")
    )
    ns2_improved = bool(np.isfinite(reduction_ratio) and reduction_ratio >= 0.25 - 1e-12)
    allowed_recall_drop = 0.05 if np.isfinite(reduction_ratio) and reduction_ratio >= 0.40 - 1e-12 else 0.03
    recall_degraded = bool(np.isfinite(low_delta) and low_delta < -float(allowed_recall_drop) - 1e-12)
    failed_count = int(item.get("failed_case_count", item.get("failed_count", 0)) or 0)
    hard_failed_count = int(item.get("hard_failed_case_count", item.get("hard_failed_count", 0)) or 0)
    full24_candidate_eligible = bool(
        failed_count == 0
        and hard_failed_count == 0
        and ns2_improved
        and not recall_degraded
        and (not np.isfinite(control_delta) or control_delta >= -0.03 - 1e-12)
    )
    return {
        "high_fp_group": high_risk,
        "high_fp_baseline": high_risk_base,
        "low_recall_group": low_recall,
        "low_recall_baseline": low_recall_base,
        "high_risk_ns2_fp_per_min": _finite_or_none(high_ns2),
        "high_risk_delta_ns2_fp_per_min": _finite_or_none(high_delta),
        "high_risk_ns2_reduction_ratio": _finite_or_none(reduction_ratio),
        "low_recall_recall_at_2.5s": _finite_or_none(low_recall_2p5),
        "low_recall_delta_recall_at_2.5s": _finite_or_none(low_delta),
        "control_recall_delta": _finite_or_none(control_delta),
        "allowed_low_recall_drop": float(allowed_recall_drop),
        "ns2_improved": ns2_improved,
        "recall_degraded": recall_degraded,
        "full24_candidate_eligible": full24_candidate_eligible,
    }


def _gate_params_first_value(summary: Mapping[str, Any], key: str, default: Any = None) -> Any:
    for item in list(summary.get("gate_params", []) or []):
        row = dict(item)
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
        params = dict(row.get("gate_variant_params", {}) or {})
        if key in params and params.get(key) not in (None, ""):
            return params.get(key)
    return default


def _gate_combo_name_for_summary(summary: Mapping[str, Any]) -> str:
    value = _gate_params_first_value(summary, "combo_name", "")
    return str(value or "")


def _subject_risk_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    high_fp_focus = set(HIGH_FP_SUBGROUP_SUBJECTS)
    low_recall_focus = set(LOW_RECALL_SUBGROUP_SUBJECTS) | set(OVERLAP_WATCH_SUBJECTS)
    high_idle_fp_subjects: list[dict[str, Any]] = []
    low_recall_subjects: list[dict[str, Any]] = []
    for item in list(summary.get("subjects", []) or []):
        row = dict(item)
        subject = str(row.get("subject", ""))
        idle_fp = _first_finite_metric(
            row,
            ("mean_mixed_idle_fp_per_min", "mean_real_idle_fp_per_min", "mean_approx_idle_fp_per_min", "mean_idle_fp_per_min"),
            float("nan"),
        )
        recall = _safe_float(row.get("mean_control_recall"), float("nan"))
        if subject in high_fp_focus or (np.isfinite(idle_fp) and idle_fp > float(DEFAULT_CLASSIFIER_IDLE_FP_BUDGET_PER_MIN)):
            high_idle_fp_subjects.append(
                {
                    "dataset": str(row.get("dataset", "")),
                    "subject": subject,
                    "idle_fp_per_min": _finite_or_none(idle_fp),
                    "ns2_fp_per_min": _finite_or_none(row.get("mean_ns2_fp_per_min")),
                }
            )
        if subject in low_recall_focus or (np.isfinite(recall) and recall < 0.80):
            low_recall_subjects.append(
                {
                    "dataset": str(row.get("dataset", "")),
                    "subject": subject,
                    "control_recall": _finite_or_none(recall),
                    "control_recall_at_2.5s": _finite_or_none(row.get("mean_control_recall_at_2.5s")),
                }
            )
    return {
        "high_idle_fp_subjects": high_idle_fp_subjects,
        "low_recall_subjects": low_recall_subjects,
        "weak_guard_subjects": list(summary.get("weak_guard_subjects", []) or []),
    }


def _subject_watch_metric_map(summary: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for item in list(summary.get("subjects", []) or []):
        row = dict(item)
        subject = _subject_id_token(row.get("subject", ""))
        if not subject:
            continue
        payload[subject] = {
            "ns2_fp_per_min": _safe_float(row.get("mean_ns2_fp_per_min"), float("nan")),
            "control_recall": _safe_float(row.get("mean_control_recall"), float("nan")),
            "control_recall_at_2.5s": _safe_float(row.get("mean_control_recall_at_2.5s"), float("nan")),
        }
    return payload


def _candidate_subject_watch_risk(summary: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    current = _subject_watch_metric_map(summary)
    base = _subject_watch_metric_map(baseline)
    for subject in LOW_RECALL_SUBGROUP_SUBJECTS:
        now = current.get(_subject_id_token(subject), {})
        before = base.get(_subject_id_token(subject), {})
        for key in ("control_recall", "control_recall_at_2.5s"):
            now_value = _safe_float(now.get(key), float("nan"))
            before_value = _safe_float(before.get(key), float("nan"))
            if np.isfinite(now_value) and np.isfinite(before_value) and now_value < before_value - 0.05:
                return True
    for subject in ("S22", "S24"):
        now_ns2 = _safe_float(current.get(_subject_id_token(subject), {}).get("ns2_fp_per_min"), float("nan"))
        before_ns2 = _safe_float(base.get(_subject_id_token(subject), {}).get("ns2_fp_per_min"), float("nan"))
        if np.isfinite(now_ns2) and np.isfinite(before_ns2) and now_ns2 >= before_ns2 - 1e-12:
            return True
    return False


def _summary_metric_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "idle_fp_per_min": _finite_or_none(
            _first_finite_metric(
                summary,
                ("mean_mixed_idle_fp_per_min", "mean_real_idle_fp_per_min", "mean_approx_idle_fp_per_min", "mean_idle_fp_per_min"),
                float("nan"),
            )
        ),
        "ns1_fp_per_min": _finite_or_none(summary.get("mean_ns1_fp_per_min")),
        "ns2_fp_per_min": _finite_or_none(summary.get("mean_ns2_fp_per_min")),
        "ns3_fp_per_min": _finite_or_none(summary.get("mean_ns3_fp_per_min")),
        "control_recall": _finite_or_none(summary.get("mean_control_recall")),
        "control_recall_at_2.5s": _finite_or_none(summary.get("mean_control_recall_at_2.5s")),
        "detection_latency_s": _finite_or_none(summary.get("mean_detection_latency_s")),
        "async_macro_f1_5state": _finite_or_none(summary.get("mean_async_macro_f1_5class")),
        "async_macro_f1_5class": _finite_or_none(summary.get("mean_async_macro_f1_5class")),
        "fixed_ns2_fp_count": _finite_or_none(summary.get("fixed_ns2_fp_count")),
        "lost_command_tp_count": _finite_or_none(summary.get("lost_command_tp_count")),
        "veto_precision": _finite_or_none(summary.get("veto_precision")),
        "tp_loss_per_fixed_fp": _finite_or_none(summary.get("tp_loss_per_fixed_fp")),
    }


def _summary_frequency_metric(summary: Mapping[str, Any], freq: str, key: str) -> Optional[float]:
    target = _freq_label(_safe_float(freq, float("nan")))
    for item in list(dict(summary).get("per_frequency_summary", []) or []):
        row = dict(item)
        if _freq_label(_safe_float(row.get("freq"), float("nan"))) == target:
            return _finite_or_none(row.get(key))
    return None


def _comparison_baseline_summary(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for summary in summaries:
        if (
            parse_classifier_gate_variant(summary.get("gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW))
            == CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW
            and str(summary.get("recipe_id", "")) == "win2_me2_sm3_lrtmw"
        ):
            return dict(summary)
    for summary in summaries:
        if parse_classifier_gate_variant(summary.get("gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW)) == CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW:
            return dict(summary)
    return dict(summaries[0]) if summaries else {}


def _decision_table_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baseline = _comparison_baseline_summary(summaries)
    base_metrics = _summary_metric_payload(baseline)
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        item = dict(summary)
        metrics = _summary_metric_payload(item)
        freq_10p5_recall = _summary_frequency_metric(item, "10.5", "command_recall")
        base_10p5_recall = _summary_frequency_metric(baseline, "10.5", "command_recall")
        freq_10p5_recall_2p5 = _summary_frequency_metric(item, "10.5", "recall_at_2.5s")
        base_10p5_recall_2p5 = _summary_frequency_metric(baseline, "10.5", "recall_at_2.5s")
        freq_10p5_ns2 = _summary_frequency_metric(item, "10.5", "fp_per_min_ns2")
        base_10p5_ns2 = _summary_frequency_metric(baseline, "10.5", "fp_per_min_ns2")
        freq_8_recall = _summary_frequency_metric(item, "8", "command_recall")
        base_8_recall = _summary_frequency_metric(baseline, "8", "command_recall")
        freq_8_recall_2p5 = _summary_frequency_metric(item, "8", "recall_at_2.5s")
        base_8_recall_2p5 = _summary_frequency_metric(baseline, "8", "recall_at_2.5s")
        freq_8_ns2 = _summary_frequency_metric(item, "8", "fp_per_min_ns2")
        base_8_ns2 = _summary_frequency_metric(baseline, "8", "fp_per_min_ns2")
        subject_watch = _subject_watch_metric_map(item)
        baseline_subject_watch = _subject_watch_metric_map(baseline)
        def subject_delta(subject: str, metric_key: str) -> Optional[float]:
            now = _safe_float(subject_watch.get(_subject_id_token(subject), {}).get(metric_key), float("nan"))
            before = _safe_float(baseline_subject_watch.get(_subject_id_token(subject), {}).get(metric_key), float("nan"))
            if np.isfinite(now) and np.isfinite(before):
                return float(now - before)
            return None
        ns2_label = _ns2_safety_label(item)
        subject_risk = _subject_risk_payload(item)
        validation_summary = dict(item.get("gate_validation_summary", {}) or {})
        subgroup_comparison = _candidate_subgroup_comparison_payload(item, baseline)
        risk_flag = bool(
            _candidate_subject_watch_risk(item, baseline)
            or subgroup_comparison["recall_degraded"]
        )
        row = {
            "method": str(item.get("method", "")),
            "recipe_id": str(item.get("recipe_id", "")),
            "gate_variant": parse_classifier_gate_variant(item.get("gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW)),
            "combo_name": _gate_combo_name_for_summary(item),
            "nc_calibration_simulation": bool(item.get("nc_calibration_simulation", False)),
            "nc_seconds": _finite_or_none(item.get("nc_seconds")),
            "nc_source": str(item.get("nc_source", "")),
            "nc_gate_type": str(item.get("nc_gate_type", "")),
            "gate_type": str(item.get("nc_gate_type", "")),
            "coverage_subject_count": int(item.get("coverage_subject_count", 0) or 0),
            "expected_subject_count": int(item.get("expected_subject_count", 0) or 0),
            "split_count": int(item.get("split_count", 0) or 0),
            "idle_fp_per_min": metrics["idle_fp_per_min"],
            "delta_idle_fp_per_min": None if metrics["idle_fp_per_min"] is None or base_metrics["idle_fp_per_min"] is None else float(metrics["idle_fp_per_min"] - base_metrics["idle_fp_per_min"]),
            "ns1_fp_per_min": metrics["ns1_fp_per_min"],
            "ns2_fp_per_min": metrics["ns2_fp_per_min"],
            "delta_ns2_fp_per_min": None if metrics["ns2_fp_per_min"] is None or base_metrics["ns2_fp_per_min"] is None else float(metrics["ns2_fp_per_min"] - base_metrics["ns2_fp_per_min"]),
            "ns3_fp_per_min": metrics["ns3_fp_per_min"],
            "control_recall": metrics["control_recall"],
            "delta_control_recall": None if metrics["control_recall"] is None or base_metrics["control_recall"] is None else float(metrics["control_recall"] - base_metrics["control_recall"]),
            "control_recall_at_2.5s": metrics["control_recall_at_2.5s"],
            "delta_control_recall_at_2.5s": None if metrics["control_recall_at_2.5s"] is None or base_metrics["control_recall_at_2.5s"] is None else float(metrics["control_recall_at_2.5s"] - base_metrics["control_recall_at_2.5s"]),
            "detection_latency_s": metrics["detection_latency_s"],
            "delta_detection_latency_s": None if metrics["detection_latency_s"] is None or base_metrics["detection_latency_s"] is None else float(metrics["detection_latency_s"] - base_metrics["detection_latency_s"]),
            "deployable": bool(item.get("deployable_budget_pass", False)),
            "ns2_safe": ns2_label == "ns2_safe",
            "ns2_status": ns2_label,
            "subject_risk": risk_flag,
            "gate_validation_supported": bool(validation_summary.get("supported", False)),
            "gate_validation_idle_fp_per_min": _finite_or_none(validation_summary.get("mean_idle_fp_per_min")),
            "gate_validation_ns2_fp_per_min": _finite_or_none(validation_summary.get("mean_ns2_fp_per_min")),
            "gate_validation_control_recall": _finite_or_none(validation_summary.get("mean_control_recall")),
            "gate_validation_control_recall_at_2.5s": _finite_or_none(
                validation_summary.get("mean_control_recall_at_2.5s")
            ),
            "frequency_specific_grid_selection_policy": str(
                item.get("frequency_specific_grid_selection_policy", "")
                or _gate_params_first_value(item, "frequency_specific_grid_selection_policy", "")
                or (FREQSPEC_GRID_SELECTION_POLICY if validation_summary else "")
            ),
            "high_risk_ns2_fp_per_min": subgroup_comparison["high_risk_ns2_fp_per_min"],
            "high_risk_delta_ns2_fp_per_min": subgroup_comparison["high_risk_delta_ns2_fp_per_min"],
            "high_risk_ns2_reduction_ratio": subgroup_comparison["high_risk_ns2_reduction_ratio"],
            "low_recall_recall_at_2.5s": subgroup_comparison["low_recall_recall_at_2.5s"],
            "low_recall_delta_recall_at_2.5s": subgroup_comparison["low_recall_delta_recall_at_2.5s"],
            "control_recall_delta_for_eligibility": subgroup_comparison["control_recall_delta"],
            "ns2_improved": subgroup_comparison["ns2_improved"],
            "recall_degraded": subgroup_comparison["recall_degraded"],
            "full24_candidate_eligible": subgroup_comparison["full24_candidate_eligible"],
            "freq_10p5_command_recall": freq_10p5_recall,
            "freq_10p5_delta_command_recall": None
            if freq_10p5_recall is None or base_10p5_recall is None
            else float(freq_10p5_recall - base_10p5_recall),
            "freq_10p5_recall_at_2.5s": freq_10p5_recall_2p5,
            "freq_10p5_delta_recall_at_2.5s": None
            if freq_10p5_recall_2p5 is None or base_10p5_recall_2p5 is None
            else float(freq_10p5_recall_2p5 - base_10p5_recall_2p5),
            "freq_10p5_ns2_fp_per_min": freq_10p5_ns2,
            "freq_10p5_delta_ns2_fp_per_min": None
            if freq_10p5_ns2 is None or base_10p5_ns2 is None
            else float(freq_10p5_ns2 - base_10p5_ns2),
            "freq_8_command_recall": freq_8_recall,
            "freq_8_delta_command_recall": None
            if freq_8_recall is None or base_8_recall is None
            else float(freq_8_recall - base_8_recall),
            "freq_8_recall_at_2.5s": freq_8_recall_2p5,
            "freq_8_delta_recall_at_2.5s": None
            if freq_8_recall_2p5 is None or base_8_recall_2p5 is None
            else float(freq_8_recall_2p5 - base_8_recall_2p5),
            "freq_8_ns2_fp_per_min": freq_8_ns2,
            "freq_8_delta_ns2_fp_per_min": None
            if freq_8_ns2 is None or base_8_ns2 is None
            else float(freq_8_ns2 - base_8_ns2),
            "fixed_ns2_fp_count": metrics["fixed_ns2_fp_count"],
            "lost_command_tp_count": metrics["lost_command_tp_count"],
            "veto_precision": metrics["veto_precision"],
            "tp_loss_per_fixed_fp": metrics["tp_loss_per_fixed_fp"],
            "tenp5_ns2_fp_delta": None
            if freq_10p5_ns2 is None or base_10p5_ns2 is None
            else float(freq_10p5_ns2 - base_10p5_ns2),
            "tenp5_command_recall_delta": None
            if freq_10p5_recall is None or base_10p5_recall is None
            else float(freq_10p5_recall - base_10p5_recall),
            "overall_recall_at_2.5_delta": None
            if metrics["control_recall_at_2.5s"] is None or base_metrics["control_recall_at_2.5s"] is None
            else float(metrics["control_recall_at_2.5s"] - base_metrics["control_recall_at_2.5s"]),
            "S11_recall_delta": subject_delta("S11", "control_recall"),
            "S19_recall_delta": subject_delta("S19", "control_recall"),
            "S24_recall_delta": subject_delta("S24", "control_recall"),
            "tenp5_full24_entry_eligible": bool(
                parse_classifier_gate_variant(item.get("gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW))
                == CLASSIFIER_GATE_VARIANT_TENP5_NS2_HARD_NEGATIVE_VETO
                and subgroup_comparison["ns2_improved"]
                and (metrics["control_recall_at_2.5s"] is None or base_metrics["control_recall_at_2.5s"] is None
                     or float(metrics["control_recall_at_2.5s"] - base_metrics["control_recall_at_2.5s"]) >= -0.03 - 1e-12)
                and (metrics["control_recall"] is None or base_metrics["control_recall"] is None
                     or float(metrics["control_recall"] - base_metrics["control_recall"]) >= -0.03 - 1e-12)
                and (freq_10p5_recall is None or base_10p5_recall is None
                     or float(freq_10p5_recall - base_10p5_recall) >= -0.05 - 1e-12)
                and not risk_flag
                and _safe_float(metrics["tp_loss_per_fixed_fp"], float("inf")) <= 1.0 + 1e-12
            ),
            "recommended_profile_export": bool(
                item.get("deployable_budget_pass", False)
                and ns2_label in {"ns2_safe", "ns2_reduced_tradeoff"}
                and not risk_flag
            ),
        }
        rows.append(row)
    return rows


def _subject_breakdown_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        gate_variant = parse_classifier_gate_variant(summary.get("gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW))
        for subject in list(summary.get("subjects", []) or []):
            item = dict(subject)
            rows.append(
                {
                    "method": str(summary.get("method", "")),
                    "recipe_id": str(summary.get("recipe_id", "")),
                    "gate_variant": gate_variant,
                    "dataset": str(item.get("dataset", "")),
                    "subject": str(item.get("subject", "")),
                    "split_count": int(item.get("split_count", 0) or 0),
                    "idle_fp_per_min": _finite_or_none(_first_finite_metric(item, ("mean_mixed_idle_fp_per_min", "mean_real_idle_fp_per_min", "mean_approx_idle_fp_per_min", "mean_idle_fp_per_min"), float("nan"))),
                    "ns1_fp_per_min": _finite_or_none(item.get("mean_ns1_fp_per_min")),
                    "ns2_fp_per_min": _finite_or_none(item.get("mean_ns2_fp_per_min")),
                    "ns3_fp_per_min": _finite_or_none(item.get("mean_ns3_fp_per_min")),
                    "control_recall": _finite_or_none(item.get("mean_control_recall")),
                    "control_recall_at_2.5s": _finite_or_none(item.get("mean_control_recall_at_2.5s")),
                    "detection_latency_s": _finite_or_none(item.get("mean_detection_latency_s")),
                    "async_macro_f1_5state": _finite_or_none(item.get("mean_async_macro_f1_5class")),
                }
            )
    return rows


def _subtype_breakdown_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        rows.append(
            {
                "method": str(summary.get("method", "")),
                "recipe_id": str(summary.get("recipe_id", "")),
                "gate_variant": parse_classifier_gate_variant(summary.get("gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW)),
                "coverage_subject_count": int(summary.get("coverage_subject_count", 0) or 0),
                "split_count": int(summary.get("split_count", 0) or 0),
                "overall_idle_fp_per_min": _finite_or_none(_first_finite_metric(summary, ("mean_mixed_idle_fp_per_min", "mean_real_idle_fp_per_min", "mean_approx_idle_fp_per_min", "mean_idle_fp_per_min"), float("nan"))),
                "ns1_fp_per_min": _finite_or_none(summary.get("mean_ns1_fp_per_min")),
                "ns2_fp_per_min": _finite_or_none(summary.get("mean_ns2_fp_per_min")),
                "ns3_fp_per_min": _finite_or_none(summary.get("mean_ns3_fp_per_min")),
                "ns_all_fp_per_min": _finite_or_none(summary.get("mean_ns_all_fp_per_min")),
                "control_recall": _finite_or_none(summary.get("mean_control_recall")),
                "control_recall_at_2.5s": _finite_or_none(summary.get("mean_control_recall_at_2.5s")),
                "detection_latency_s": _finite_or_none(summary.get("mean_detection_latency_s")),
            }
        )
    return rows


def _nc_budget_curve_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        if not bool(summary.get("nc_calibration_simulation", False)):
            continue
        subjects = list(summary.get("subjects", []) or [])
        s11 = next((dict(item) for item in subjects if _subject_id_token(dict(item).get("subject")) == "S11"), {})
        s19 = next((dict(item) for item in subjects if _subject_id_token(dict(item).get("subject")) == "S19"), {})
        s24 = next((dict(item) for item in subjects if _subject_id_token(dict(item).get("subject")) == "S24"), {})
        idle_fp = _first_finite_metric(
            summary,
            ("mean_mixed_idle_fp_per_min", "mean_real_idle_fp_per_min", "mean_approx_idle_fp_per_min", "mean_idle_fp_per_min"),
            float("nan"),
        )
        ns2 = _safe_float(summary.get("mean_ns2_fp_per_min"), float("nan"))
        recall_2p5 = _safe_float(summary.get("mean_control_recall_at_2.5s"), 0.0)
        control_recall = _safe_float(summary.get("mean_control_recall"), 0.0)
        latency = _safe_float(summary.get("mean_detection_latency_s"), float("inf"))
        deployable = bool(
            np.isfinite(idle_fp)
            and idle_fp <= 1.0 + 1e-12
            and np.isfinite(ns2)
            and ns2 <= 1.25 + 1e-12
            and recall_2p5 >= 0.75 - 1e-12
            and control_recall >= 0.85 - 1e-12
            and latency <= 2.5 + 1e-12
        )
        rows.append(
            {
                "nc_seconds": _finite_or_none(summary.get("nc_seconds")),
                "nc_source": str(summary.get("nc_source", "")),
                "gate_type": str(summary.get("nc_gate_type", "")),
                "idle_fp_per_min": _finite_or_none(idle_fp),
                "NS1_fp_per_min": _finite_or_none(summary.get("mean_ns1_fp_per_min")),
                "NS2_fp_per_min": _finite_or_none(ns2),
                "NS3_fp_per_min": _finite_or_none(summary.get("mean_ns3_fp_per_min")),
                "control_recall": _finite_or_none(control_recall),
                "control_recall_at_2.5s": _finite_or_none(recall_2p5),
                "detection_latency_s": _finite_or_none(latency),
                "S11_recall": _finite_or_none(s11.get("mean_control_recall")),
                "S19_recall": _finite_or_none(s19.get("mean_control_recall")),
                "S24_NS2_fp": _finite_or_none(s24.get("mean_ns2_fp_per_min")),
                "tp_loss_per_fixed_fp": _finite_or_none(summary.get("tp_loss_per_fixed_fp")),
                "deployable": deployable,
                "ns2_safe": bool(np.isfinite(ns2) and ns2 <= 1.25 + 1e-12),
            }
        )
    rows.sort(
        key=lambda row: (
            _safe_float(row.get("nc_seconds"), 0.0),
            str(row.get("nc_source", "")),
            str(row.get("gate_type", "")),
        )
    )
    return rows


def _flatten_nc_feature_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        for item in list(row.get("nc_csns_feature_summary", []) or []):
            payload = {
                **dict(item),
                "dataset": str(row.get("dataset", dict(item).get("dataset", ""))),
                "subject": str(row.get("subject", dict(item).get("subject", ""))),
                "split_index": int(row.get("split_index", dict(item).get("split_index", 0)) or 0),
                "recipe_id": str(row.get("recipe_id", dict(item).get("recipe_id", ""))),
                "method": str(row.get("method", "")),
            }
            flattened.append(payload)
    return flattened


def _flatten_split_diagnostic_rows(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        diagnostics = dict(row.get("ns2_selected_freq_diagnostics", {}) or {})
        for item in list(diagnostics.get(key, []) or []):
            flattened.append(dict(item))
    return flattened


def _flatten_logistic_trace_rows(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        diagnostics = dict(row.get("logistic_trace_diagnostics", {}) or {})
        for item in list(diagnostics.get(key, []) or []):
            flattened.append(dict(item))
    return flattened


def _flatten_tenp5_veto_rows(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        diagnostics = dict(row.get("tenp5_ns2_veto_diagnostics", {}) or {})
        for item in list(diagnostics.get(key, []) or []):
            flattened.append(dict(item))
    return flattened


def _tenp5_veto_summary_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary_rows = _flatten_tenp5_veto_rows(rows, "tenp5_ns2_veto_summary_rows")
    by_recipe: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in summary_rows:
        item = dict(row)
        key = (
            str(item.get("method", "")),
            str(item.get("recipe_id", "")),
            str(item.get("gate_variant", "")),
        )
        payload = by_recipe.setdefault(
            key,
            {
                "method": key[0],
                "recipe_id": key[1],
                "gate_variant": key[2],
                "fixed_ns2_fp_count": 0,
                "lost_command_tp_count": 0,
                "baseline_tenp5_ns2_fp_count": 0,
                "candidate_tenp5_ns2_fp_count": 0,
                "baseline_tenp5_command_tp_count": 0,
                "candidate_tenp5_command_tp_count": 0,
                "unsupported_split_count": 0,
                "split_count": 0,
                "subjects": set(),
            },
        )
        payload["split_count"] = int(payload["split_count"]) + 1
        payload["subjects"].add(str(item.get("subject", "")))
        if str(item.get("veto_status", "")) != "ok":
            payload["unsupported_split_count"] = int(payload["unsupported_split_count"]) + 1
        for field in (
            "fixed_ns2_fp_count",
            "lost_command_tp_count",
            "baseline_tenp5_ns2_fp_count",
            "candidate_tenp5_ns2_fp_count",
            "baseline_tenp5_command_tp_count",
            "candidate_tenp5_command_tp_count",
        ):
            payload[field] = int(payload[field]) + int(_safe_float(item.get(field), 0.0))
    recipes: list[dict[str, Any]] = []
    for payload in by_recipe.values():
        fixed = int(payload.get("fixed_ns2_fp_count", 0))
        lost = int(payload.get("lost_command_tp_count", 0))
        baseline_ns2 = int(payload.get("baseline_tenp5_ns2_fp_count", 0))
        candidate_ns2 = int(payload.get("candidate_tenp5_ns2_fp_count", 0))
        baseline_tp = int(payload.get("baseline_tenp5_command_tp_count", 0))
        candidate_tp = int(payload.get("candidate_tenp5_command_tp_count", 0))
        recipes.append(
            {
                **{key: value for key, value in payload.items() if key != "subjects"},
                "subject_count": int(len(payload.get("subjects", set()))),
                "veto_precision": float(fixed / (fixed + lost)) if fixed + lost > 0 else None,
                "tp_loss_per_fixed_fp": float(lost / max(fixed, 1)),
                "tenp5_ns2_fp_delta_count": int(candidate_ns2 - baseline_ns2),
                "tenp5_command_tp_delta_count": int(candidate_tp - baseline_tp),
            }
        )
    recipes.sort(key=lambda item: (str(item.get("method", "")), str(item.get("recipe_id", ""))))
    return {
        "schema_version": "ssvep_tenp5_ns2_hard_negative_veto_summary_v1",
        "recipes": recipes,
    }


TRACE_SEPARABILITY_FEATURES = (
    "selected_freq_score",
    "top1_score",
    "top2_score",
    "margin",
    "ratio",
    "normalized_top1",
    "score_entropy",
    "lrt_evidence",
    "multiwindow_same_freq_count",
    "multiwindow_margin_mean",
    "multiwindow_entropy_mean",
    "cs_probability",
)


def _trace_positive_negative_groups(rows: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    positives = [
        row
        for row in rows
        if str(row.get("transition_type", "")) in {"baseline_TP_candidate_TP", "baseline_TP_candidate_idle"}
    ]
    negatives = [
        row
        for row in rows
        if str(row.get("true_state", "")).lower() == "ns2"
        and str(row.get("baseline_pred", "")) != "idle"
    ]
    return positives, negatives


def _auc_positive_greater(positive: Sequence[float], negative: Sequence[float]) -> float:
    pos = np.asarray(list(positive), dtype=np.float64)
    neg = np.asarray(list(negative), dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size <= 0 or neg.size <= 0:
        return float("nan")
    greater = 0.0
    ties = 0.0
    for value in pos:
        greater += float(np.sum(value > neg))
        ties += float(np.sum(value == neg))
    return float((greater + 0.5 * ties) / float(pos.size * neg.size))


def _feature_separability_row(
    *,
    group: Mapping[str, Any],
    feature: str,
    positives: Sequence[Mapping[str, Any]],
    negatives: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pos = np.asarray(
        [
            _safe_float(row.get(feature), float("nan"))
            for row in positives
            if np.isfinite(_safe_float(row.get(feature), float("nan")))
        ],
        dtype=np.float64,
    )
    neg = np.asarray(
        [
            _safe_float(row.get(feature), float("nan"))
            for row in negatives
            if np.isfinite(_safe_float(row.get(feature), float("nan")))
        ],
        dtype=np.float64,
    )
    if pos.size <= 0 or neg.size <= 0:
        return {
            **dict(group),
            "feature": str(feature),
            "positive_count": int(pos.size),
            "negative_count": int(neg.size),
            "mean_positive": None,
            "mean_negative": None,
            "delta_mean": None,
            "effect_size": None,
            "auc_positive_greater": None,
            "auc_best_direction": None,
            "overlap_rate": None,
        }
    mean_pos = float(np.mean(pos))
    mean_neg = float(np.mean(neg))
    pooled = float(np.sqrt(max((float(np.var(pos)) + float(np.var(neg))) / 2.0, 1e-12)))
    auc = _auc_positive_greater(pos, neg)
    pos_lo, pos_hi = float(np.quantile(pos, 0.05)), float(np.quantile(pos, 0.95))
    neg_lo, neg_hi = float(np.quantile(neg, 0.05)), float(np.quantile(neg, 0.95))
    overlap = max(0.0, min(pos_hi, neg_hi) - max(pos_lo, neg_lo))
    span = max(max(pos_hi, neg_hi) - min(pos_lo, neg_lo), 1e-12)
    return {
        **dict(group),
        "feature": str(feature),
        "positive_count": int(pos.size),
        "negative_count": int(neg.size),
        "mean_positive": mean_pos,
        "mean_negative": mean_neg,
        "median_positive": float(np.median(pos)),
        "median_negative": float(np.median(neg)),
        "delta_mean": float(mean_pos - mean_neg),
        "effect_size": float((mean_pos - mean_neg) / pooled),
        "auc_positive_greater": float(auc),
        "auc_best_direction": float(max(auc, 1.0 - auc)),
        "overlap_rate": float(overlap / span),
    }


def _trace_separability_outputs(window_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in window_rows]
    group_defs: list[dict[str, Any]] = [{"scope": "all"}]
    for key in ("subject", "selected_freq"):
        group_defs.extend(
            {"scope": key, key: value}
            for value in sorted({str(row.get(key, "")) for row in rows if str(row.get(key, ""))})
        )
    group_defs.extend(
        {"scope": "subject_selected_freq", "subject": subject, "selected_freq": freq}
        for subject, freq in sorted(
            {
                (str(row.get("subject", "")), str(row.get("selected_freq", "")))
                for row in rows
                if str(row.get("subject", "")) and str(row.get("selected_freq", ""))
            }
        )
    )

    feature_rows: list[dict[str, Any]] = []
    for group in group_defs:
        scope = str(group.get("scope", ""))
        if scope == "all":
            selected = rows
        elif scope == "subject":
            selected = [row for row in rows if str(row.get("subject", "")) == str(group.get("subject", ""))]
        elif scope == "selected_freq":
            selected = [row for row in rows if str(row.get("selected_freq", "")) == str(group.get("selected_freq", ""))]
        else:
            selected = [
                row
                for row in rows
                if str(row.get("subject", "")) == str(group.get("subject", ""))
                and str(row.get("selected_freq", "")) == str(group.get("selected_freq", ""))
            ]
        positives, negatives = _trace_positive_negative_groups(selected)
        for feature in TRACE_SEPARABILITY_FEATURES:
            feature_rows.append(
                _feature_separability_row(
                    group=group,
                    feature=feature,
                    positives=positives,
                    negatives=negatives,
                )
            )

    transition_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        transition_counts[
            (
                str(row.get("subject", "")),
                str(row.get("selected_freq", "")),
                str(row.get("transition_type", "")),
            )
        ] += 1
    transition_rows = [
        {
            "subject": subject,
            "selected_freq": selected_freq,
            "transition_type": transition_type,
            "count": int(count),
        }
        for (subject, selected_freq, transition_type), count in sorted(transition_counts.items())
    ]

    risk_rows: list[dict[str, Any]] = []
    for row in feature_rows:
        if str(row.get("scope", "")) not in {"selected_freq", "subject_selected_freq"}:
            continue
        if str(row.get("feature", "")) not in {
            "margin",
            "ratio",
            "score_entropy",
            "lrt_evidence",
            "multiwindow_same_freq_count",
            "cs_probability",
        }:
            continue
        auc_best = _safe_float(row.get("auc_best_direction"), float("nan"))
        effect = abs(_safe_float(row.get("effect_size"), float("nan")))
        if np.isfinite(auc_best) and np.isfinite(effect) and (auc_best >= 0.65 or effect >= 0.5):
            risk_rows.append(
                {
                    "scope": str(row.get("scope", "")),
                    "subject": str(row.get("subject", "")),
                    "selected_freq": str(row.get("selected_freq", "")),
                    "feature": str(row.get("feature", "")),
                    "auc_best_direction": float(auc_best),
                    "effect_size_abs": float(effect),
                    "positive_count": int(row.get("positive_count", 0) or 0),
                    "negative_count": int(row.get("negative_count", 0) or 0),
                    "candidate_rule": "candidate_selective_gate_feature",
                }
            )

    positives, negatives = _trace_positive_negative_groups(rows)
    best_rows = [
        row
        for row in feature_rows
        if str(row.get("scope", "")) == "all" and row.get("auc_best_direction") is not None
    ]
    best_rows.sort(key=lambda item: _safe_float(item.get("auc_best_direction"), -1.0), reverse=True)
    ns2_by_freq: dict[str, int] = defaultdict(int)
    for row in negatives:
        ns2_by_freq[str(row.get("selected_freq", ""))] += 1
    total_ns2 = sum(ns2_by_freq.values())
    top_ns2_freq = max(ns2_by_freq.items(), key=lambda item: item[1])[0] if ns2_by_freq else ""
    summary = {
        "schema_version": "ssvep_logistic_trace_separability_v1",
        "window_row_count": int(len(rows)),
        "positive_tp_window_count": int(len(positives)),
        "ns2_baseline_fp_window_count": int(len(negatives)),
        "ns2_baseline_fp_by_selected_freq": dict(sorted(ns2_by_freq.items())),
        "ns2_top_selected_freq": top_ns2_freq,
        "ns2_top_selected_freq_share": float(ns2_by_freq[top_ns2_freq] / total_ns2) if total_ns2 else 0.0,
        "best_global_features": best_rows[:8],
        "trace_based_recommendation": (
            "conditional_gate_candidate"
            if best_rows and _safe_float(best_rows[0].get("auc_best_direction"), 0.0) >= 0.65
            else "trace_overlap_high_consider_hard_negative_or_session_calibration"
        ),
    }
    return {
        "summary": summary,
        "transition_by_subject_freq": transition_rows,
        "feature_separability_by_subject_freq": feature_rows,
        "risk_rule_candidates": risk_rows,
    }


def _render_trace_separability_markdown(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("summary", {}) or {})
    lines = [
        "# Trace Separability Summary",
        "",
        f"- window_row_count: {int(summary.get('window_row_count', 0) or 0)}",
        f"- positive_tp_window_count: {int(summary.get('positive_tp_window_count', 0) or 0)}",
        f"- ns2_baseline_fp_window_count: {int(summary.get('ns2_baseline_fp_window_count', 0) or 0)}",
        f"- ns2_top_selected_freq: {summary.get('ns2_top_selected_freq', '')}",
        f"- ns2_top_selected_freq_share: {_safe_float(summary.get('ns2_top_selected_freq_share'), 0.0):.4f}",
        f"- recommendation: {summary.get('trace_based_recommendation', '')}",
        "",
        "## Best Global Features",
        "",
        "| feature | auc_best_direction | effect_size | positive_count | negative_count |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in list(summary.get("best_global_features", []) or [])[:8]:
        lines.append(
            "| {feature} | {auc:.4f} | {effect:.4f} | {pos} | {neg} |".format(
                feature=str(row.get("feature", "")),
                auc=_safe_float(row.get("auc_best_direction"), 0.0),
                effect=_safe_float(row.get("effect_size"), 0.0),
                pos=int(row.get("positive_count", 0) or 0),
                neg=int(row.get("negative_count", 0) or 0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _split_per_frequency_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        for item in list(row.get("per_frequency_metrics", []) or []):
            flattened.append(dict(item))
    return flattened


def _per_frequency_summary_from_recipe_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for item in list(dict(row).get("per_frequency_metrics", []) or []):
            freq = _freq_label(_safe_float(dict(item).get("freq"), 0.0))
            grouped[freq].append(dict(item))
    summary_rows: list[dict[str, Any]] = []
    for freq, items in sorted(grouped.items(), key=lambda item: _safe_float(item[0], float("inf"))):
        summary_rows.append(
            {
                "freq": str(freq),
                "command_recall": _mean_field(items, "command_recall"),
                "recall_at_2.5s": _mean_field(items, "recall_at_2.5s"),
                "fp_per_min_ns1": _mean_field(items, "fp_per_min_ns1"),
                "fp_per_min_ns2": _mean_field(items, "fp_per_min_ns2"),
                "fp_per_min_ns3": _mean_field(items, "fp_per_min_ns3"),
                "gate_pass_rate": _mean_field(items, "gate_pass_rate"),
                "mean_latency": _mean_field(items, "mean_latency"),
                "row_count": int(len(items)),
            }
        )
    return summary_rows


def _mean_field(items: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = [_safe_float(dict(item).get(key), float("nan")) for item in items]
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _per_frequency_summary_payload(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in _split_per_frequency_rows(rows):
        grouped[
            (
                str(item.get("method", "")),
                str(item.get("recipe_id", "")),
                str(item.get("gate_variant", "")),
                _freq_label(_safe_float(item.get("freq"), 0.0)),
            )
        ].append(item)
    payload: list[dict[str, Any]] = []
    for (method, recipe_id, gate_variant, freq), items in sorted(grouped.items()):
        payload.append(
            {
                "method": method,
                "recipe_id": recipe_id,
                "gate_variant": gate_variant,
                "freq": freq,
                "split_count": int(len(items)),
                "command_recall": _mean_field(items, "command_recall"),
                "recall_at_2.5s": _mean_field(items, "recall_at_2.5s"),
                "fp_per_min_ns1": _mean_field(items, "fp_per_min_ns1"),
                "fp_per_min_ns2": _mean_field(items, "fp_per_min_ns2"),
                "fp_per_min_ns3": _mean_field(items, "fp_per_min_ns3"),
                "gate_pass_rate": _mean_field(items, "gate_pass_rate"),
                "mean_latency": _mean_field(items, "mean_latency"),
            }
        )
    return payload


def _gate_params_by_frequency_payload(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recipes: list[dict[str, Any]] = []
    for summary in summaries:
        per_freq: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in list(summary.get("gate_params", []) or []):
            gates = normalize_frequency_specific_control_state_gates(
                dict(item).get("frequency_specific_control_state_gates")
            )
            for freq, payload in gates.items():
                per_freq[str(freq)].append(
                    {
                        "subject": str(dict(item).get("subject", "")),
                        "split_index": int(dict(item).get("split_index", 0) or 0),
                        **dict(payload),
                    }
                )
        recipes.append(
            {
                "method": str(summary.get("method", "")),
                "recipe_id": str(summary.get("recipe_id", "")),
                "gate_variant": parse_classifier_gate_variant(summary.get("gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW)),
                "frequency_specific_control_state_gates": {
                    freq: values for freq, values in sorted(per_freq.items())
                },
            }
        )
    return {
        "schema_version": "ssvep_frequency_specific_gate_params_v1",
        "recipes": recipes,
    }


def _gate_params_payload(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recipes = []
    for summary in summaries:
        recipes.append(
            {
                "method": str(summary.get("method", "")),
                "recipe_id": str(summary.get("recipe_id", "")),
                "gate_variant": parse_classifier_gate_variant(summary.get("gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW)),
                "gate_params": list(summary.get("gate_params", []) or []),
                "weak_guard_subjects": list(summary.get("weak_guard_subjects", []) or []),
            }
        )
    return {
        "schema_version": "ssvep_gate_params_v1",
        "recipes": recipes,
    }


def render_markdown_summary(
    *,
    run_id: str,
    freqs: Sequence[float],
    subjects: Sequence[ExternalSubjectSpec],
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    shared_summaries: Optional[Sequence[dict[str, Any]]] = None,
    score_bank_mode: str = DEFAULT_SCORE_BANK_MODE,
    frequency_search_plan: Optional[Mapping[str, Any]] = None,
    idle_eval_mode: str = DEFAULT_IDLE_EVAL_MODE,
    budget: Optional[Mapping[str, Any]] = None,
    weak_subject_audit: Optional[Mapping[str, Any]] = None,
    metric_definitions: Optional[Mapping[str, Any]] = None,
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
        f"- score_bank_mode: `{score_bank_mode}`",
        f"- frequency_selection_mode: `{dict(frequency_search_plan or {}).get('frequency_selection_mode', DEFAULT_FREQ_SEARCH_MODE)}`",
        f"- idle_eval_mode: `{idle_eval_mode}`",
        f"- pretrain_budget_sec: `{float(dict(budget or {}).get('pretrain_budget_sec', DEFAULT_PRETRAIN_BUDGET_SEC)):.1f}`",
        f"- estimated_pretrain_duration_sec: `{float(dict(budget or {}).get('estimated_pretrain_duration_sec', 0.0)):.1f}`",
        f"- pretrain_budget_pass: `{bool(dict(budget or {}).get('pretrain_budget_pass', True))}`",
        f"- channel_contract: `strict_required_8_posterior`",
        f"- project_channel_names: `{','.join(PROJECT_POSTERIOR_8_CHANNELS)}`",
        f"- decision_start_s: `{_safe_float(dict(metric_definitions or {}).get('decision_start_s'), DEFAULT_DECISION_START_SEC):.3f}`",
        f"- decision_deadline_s: `{_safe_float(dict(metric_definitions or {}).get('decision_deadline_s'), DEFAULT_DECISION_DEADLINE_SEC):.3f}`",
        f"- min_release_windows: `{int(_safe_float(dict(metric_definitions or {}).get('min_release_windows'), DEFAULT_MIN_RELEASE_WINDOWS))}`",
        "",
        "> Idle/no-control is proxied with non-command target stimulus trials for Wang/BETA; YSU-an uses explicit NS1/NS2/NS3 no-control trials.",
        "> External datasets are evaluated after selecting only this posterior 8-channel subset; deployed numeric board channels must be wired to the same electrode order.",
        "> Metric definitions: detection_latency_s is stimulus onset to first correct control output; switch/release latency are reported only when replay supports those transitions.",
        "",
    ]

    def append_recipe_table(title: str, table_summaries: Sequence[dict[str, Any]]) -> None:
        lines.extend(
            [
                f"## {title}",
                "",
                "| Rank | Deployable | Profile | Method | Recipe | Freqs | Coverage | Cal Blocks | Idle Mult | Mixed Idle FP/min | Real Idle FP/min | Approx Idle FP/min | Mean Async 5c Macro-F1 | Mean Control Recall | Recall <=2.5s | Mean Detection Latency s |",
                "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if not table_summaries:
            lines.append("| - | - | - | - | - | - | 0/0 | - | - | - | - | - | - | - | - | - |")
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
            freq_text = ",".join(f"{float(freq):g}" for freq in summary.get("selected_freqs", []) or [])
            if not freq_text and summary.get("frequency_selection_mode") == "personalized_upper_bound":
                freq_text = "per-subject"
            lines.append(
                "| {rank} | {deployable} | {profile} | {method} | `{recipe}` | `{freqs}` | {coverage} | {cal_blocks} | {idle_mult:.2f} | {mixed_idle:.4f} | {real_idle:.4f} | {approx_idle:.4f} | {async_f1:.4f} | {recall:.4f} | {recall_2p5:.4f} | {latency:.4f} |".format(
                    rank=index,
                    deployable="yes" if bool(summary.get("deployable_budget_pass", False)) else "no",
                    profile=str(summary.get("frequency_profile", "")),
                    method=str(summary.get("method", "")),
                    recipe=str(summary.get("recipe_id", "")),
                    freqs=freq_text,
                    coverage=coverage,
                    cal_blocks=int(summary.get("calibration_block_count", 0)),
                    idle_mult=float(summary.get("idle_multiplier", 0.0)),
                    async_f1=float(summary.get("mean_async_macro_f1_5class", 0.0)),
                    mixed_idle=_safe_float(summary.get("mean_mixed_idle_fp_per_min"), float("inf")),
                    real_idle=_safe_float(summary.get("mean_real_idle_fp_per_min"), float("nan")),
                    approx_idle=_safe_float(summary.get("mean_approx_idle_fp_per_min"), float("nan")),
                    recall=float(summary.get("mean_control_recall", 0.0)),
                    recall_2p5=float(summary.get("mean_control_recall_at_2.5s", 0.0)),
                    latency=float(summary.get("mean_detection_latency_s", float("inf"))),
                )
            )
        lines.append("")

    append_recipe_table("Top Shared Recipes", resolved_shared_summaries)
    if any("mean_ns_all_fp_per_min" in dict(summary) for summary in resolved_shared_summaries):
        lines.extend(
            [
                "",
                "## YSU-an No-Control Subtype FP",
                "",
                "| Rank | Method | Recipe | NS1 FP/min | NS2 FP/min | NS3 FP/min | NS All FP/min | CS Control Recall |",
                "|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for index, summary in enumerate(list(resolved_shared_summaries)[:10], start=1):
            lines.append(
                "| {rank} | {method} | `{recipe}` | {ns1:.4f} | {ns2:.4f} | {ns3:.4f} | {nsall:.4f} | {cs:.4f} |".format(
                    rank=index,
                    method=str(summary.get("method", "")),
                    recipe=str(summary.get("recipe_id", "")),
                    ns1=_safe_float(summary.get("mean_ns1_fp_per_min"), float("nan")),
                    ns2=_safe_float(summary.get("mean_ns2_fp_per_min"), float("nan")),
                    ns3=_safe_float(summary.get("mean_ns3_fp_per_min"), float("nan")),
                    nsall=_safe_float(summary.get("mean_ns_all_fp_per_min"), float("nan")),
                    cs=_safe_float(summary.get("mean_cs_control_recall"), 0.0),
                )
            )
    lines.append("")
    decision_rows = _decision_table_rows(resolved_shared_summaries)
    if decision_rows:
        lines.extend(
            [
                "",
                "## Decision Table",
                "",
                "| Method | Recipe | Gate Variant | Deployable | NS2 Status | Subject Risk | Idle FP/min | NS2 FP/min | Recall <=2.5s | Control Recall | Latency s |",
                "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in decision_rows[:12]:
            lines.append(
                "| {method} | `{recipe}` | `{variant}` | {deployable} | {ns2_status} | {risk} | {idle:.4f} | {ns2:.4f} | {recall2p5:.4f} | {recall:.4f} | {latency:.4f} |".format(
                    method=str(row.get("method", "")),
                    recipe=str(row.get("recipe_id", "")),
                    variant=str(row.get("gate_variant", "")),
                    deployable="yes" if bool(row.get("deployable", False)) else "no",
                    ns2_status=str(row.get("ns2_status", "")),
                    risk="yes" if bool(row.get("subject_risk", False)) else "no",
                    idle=_safe_float(row.get("idle_fp_per_min"), float("nan")),
                    ns2=_safe_float(row.get("ns2_fp_per_min"), float("nan")),
                    recall2p5=_safe_float(row.get("control_recall_at_2.5s"), float("nan")),
                    recall=_safe_float(row.get("control_recall"), float("nan")),
                    latency=_safe_float(row.get("detection_latency_s"), float("nan")),
                )
            )
    nc_rows = _nc_budget_curve_rows(resolved_shared_summaries)
    if nc_rows:
        lines.extend(
            [
                "",
                "## No-Control Calibration Simulation",
                "",
                "These rows are benchmark-only session-specific no-control calibration simulations. They are not runtime profiles and must not be copied to `default_profile.json`.",
                "",
                "| Seconds | Source | Gate | Idle FP/min | NS2 FP/min | Recall <=2.5s | Control Recall | Latency s | TP Loss / Fixed FP | Deployable |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in nc_rows[:20]:
            lines.append(
                "| {seconds:.0f} | {source} | `{gate}` | {idle:.4f} | {ns2:.4f} | {recall2p5:.4f} | {recall:.4f} | {latency:.4f} | {loss:.4f} | {deployable} |".format(
                    seconds=_safe_float(row.get("nc_seconds"), 0.0),
                    source=str(row.get("nc_source", "")),
                    gate=str(row.get("gate_type", "")),
                    idle=_safe_float(row.get("idle_fp_per_min"), float("nan")),
                    ns2=_safe_float(row.get("NS2_fp_per_min"), float("nan")),
                    recall2p5=_safe_float(row.get("control_recall_at_2.5s"), float("nan")),
                    recall=_safe_float(row.get("control_recall"), float("nan")),
                    latency=_safe_float(row.get("detection_latency_s"), float("nan")),
                    loss=_safe_float(row.get("tp_loss_per_fixed_fp"), float("nan")),
                    deployable="yes" if bool(row.get("deployable", False)) else "no",
                )
            )
    lines.extend(
        [
            "",
            "## Metric Note",
            "",
            "`async_macro_f1_5class` is kept for historical compatibility; here `5class` means the five output states `idle + 4 command`, not five ordinary SSVEP frequency classes. New summaries also expose `async_macro_f1_5state`.",
            "",
        ]
    )
    append_recipe_table("Top Recipes", summaries)
    if weak_subject_audit:
        lines.extend(["", "## Weak Subject Audit", ""])
        tracked = list(dict(weak_subject_audit).get("tracked_weak_subjects", []) or [])
        if tracked:
            lines.extend(
                [
                    "| Subject | Control Recall | Idle FP/min | Async 5c Macro-F1 | Detection Latency s |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for row in tracked:
                lines.append(
                    "| {subject} | {recall:.4f} | {idle:.4f} | {f1:.4f} | {latency:.4f} |".format(
                        subject=str(row.get("subject", "")),
                        recall=_safe_float(row.get("mean_control_recall"), 0.0),
                        idle=_safe_float(row.get("mean_idle_fp_per_min"), 0.0),
                        f1=_safe_float(row.get("mean_async_macro_f1_5class"), 0.0),
                        latency=_safe_float(row.get("mean_detection_latency_s"), float("inf")),
                    )
                )
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
    if not str(args.run_id or "").strip():
        raise ValueError("--run-id is required for pseudo-online recipe screening; nohup/server runs must be traceable")
    output_root = Path(args.output_root).expanduser().resolve() / run_id
    report_root = output_root / "reports"
    dataset_root = Path(args.dataset_root).expanduser().resolve() / run_id
    report_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    log_path = report_root / "benchmark.log"
    server_log_contract_path = SERVER_SSVEP_LOG_ROOT / f"{run_id}.log"
    server_log_path = Path(str(server_log_contract_path))
    mirror_server_log = bool(
        os.name != "nt"
        and str(output_root).startswith(str(SERVER_SSVEP_WRITABLE_ROOT))
    )
    if mirror_server_log:
        server_log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = report_root / "progress_snapshot.json"
    partial_summary_path = report_root / "partial_summary.json"
    failed_cases_path = report_root / "failed_cases.json"
    coverage_report_path = report_root / "coverage_report.json"

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if mirror_server_log:
            with server_log_path.open("a", encoding="utf-8") as handle:
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
    classifier_smoothing_candidates = _csv_int_tuple(
        getattr(args, "classifier_smoothing_windows_candidates", ""),
        default=DEFAULT_CLASSIFIER_SMOOTHING_WINDOWS_CANDIDATES,
    )
    classifier_gate_variants = _csv_gate_variant_tuple(
        getattr(args, "classifier_gate_variants", ",".join(DEFAULT_CLASSIFIER_GATE_VARIANTS))
    )
    freqspec_threshold_combo_set = str(
        getattr(args, "freqspec_threshold_combo_set", FREQSPEC_THRESHOLD_COMBO_SET_NONE) or ""
    ).strip().lower()
    if freqspec_threshold_combo_set not in FREQSPEC_THRESHOLD_COMBO_SETS:
        raise ValueError(f"unsupported --freqspec-threshold-combo-set {freqspec_threshold_combo_set!r}")
    freqspec_margin_idle_quantiles = _csv_float_tuple(
        getattr(args, "freqspec_margin_idle_quantiles", ""),
        default=DEFAULT_FREQSPEC_MARGIN_IDLE_QUANTILES,
    )
    ns2_safety_factors = _csv_float_tuple(
        getattr(args, "ns2_safety_factors", ""),
        default=DEFAULT_NS2_SAFETY_FACTORS,
    )
    subject_floor_global_quantiles = _csv_float_tuple(
        getattr(args, "subject-floor-global-quantiles", getattr(args, "subject_floor_global_quantiles", "")),
        default=DEFAULT_GLOBAL_FLOOR_QUANTILES,
    )
    subject_floor_idle_quantiles = _csv_float_tuple(
        getattr(args, "subject-floor-idle-quantiles", getattr(args, "subject_floor_idle_quantiles", "")),
        default=DEFAULT_SUBJECT_IDLE_QUANTILES,
    )
    freqspec_ratio_idle_quantiles = _csv_float_tuple(
        getattr(args, "freqspec_ratio_idle_quantiles", ""),
        default=DEFAULT_FREQSPEC_RATIO_IDLE_QUANTILES,
    )
    freqspec_entropy_control_quantiles = _csv_float_tuple(
        getattr(args, "freqspec_entropy_control_quantiles", ""),
        default=DEFAULT_FREQSPEC_ENTROPY_CONTROL_QUANTILES,
    )
    freqspec_ns2_safety_factors = _csv_float_tuple(
        getattr(args, "freqspec_ns2_safety_factors", ""),
        default=DEFAULT_NS2_SAFETY_FACTORS,
    )
    freqspec_logistic_prob_thresholds = _csv_float_tuple(
        getattr(args, "freqspec_logistic_prob_thresholds", ""),
        default=DEFAULT_FREQSPEC_LOGISTIC_PROB_THRESHOLDS,
    )
    freqspec_logistic_ns2_weights = _csv_float_tuple(
        getattr(args, "freqspec_logistic_ns2_weights", ""),
        default=DEFAULT_FREQSPEC_LOGISTIC_NS2_WEIGHTS,
    )
    tenp5_veto_thresholds = _csv_float_tuple(
        getattr(args, "tenp5_veto_thresholds", ""),
        default=DEFAULT_TENP5_VETO_THRESHOLDS,
    )
    tenp5_ns2_weights = _csv_float_tuple(
        getattr(args, "tenp5_ns2_weights", ""),
        default=DEFAULT_TENP5_NS2_WEIGHTS,
    )
    enable_nc_calibration_simulation = bool(getattr(args, "enable_nc_calibration_simulation", False))
    nc_calibration_seconds = _csv_float_tuple(
        getattr(args, "nc_calibration_seconds", ""),
        default=DEFAULT_NC_CALIBRATION_SECONDS,
    )
    nc_calibration_sources = _csv_nc_calibration_sources(getattr(args, "nc_calibration_sources", ""))
    nc_calibration_gate_types = _csv_nc_gate_types(getattr(args, "nc_calibration_gate_types", ""))
    classifier_gate_variant_param_grid = [
        params
        for variant in classifier_gate_variants
        for params in _gate_variant_param_grid(
            variant,
            freqspec_threshold_combo_set=freqspec_threshold_combo_set,
            freqspec_margin_idle_quantiles=freqspec_margin_idle_quantiles,
            freqspec_ratio_idle_quantiles=freqspec_ratio_idle_quantiles,
            freqspec_entropy_control_quantiles=freqspec_entropy_control_quantiles,
            ns2_safety_factors=ns2_safety_factors,
            subject_floor_global_quantiles=subject_floor_global_quantiles,
            subject_floor_idle_quantiles=subject_floor_idle_quantiles,
            freqspec_ns2_safety_factors=freqspec_ns2_safety_factors,
            freqspec_logistic_prob_thresholds=freqspec_logistic_prob_thresholds,
            freqspec_logistic_ns2_weights=freqspec_logistic_ns2_weights,
            tenp5_veto_thresholds=tenp5_veto_thresholds,
            tenp5_ns2_weights=tenp5_ns2_weights,
        )
    ]
    classifier_threshold_policy = _parse_classifier_threshold_policy(args.classifier_threshold_policy)
    score_bank_mode = _parse_score_bank_mode(getattr(args, "score_bank_mode", DEFAULT_SCORE_BANK_MODE))
    freq_search_mode = _parse_freq_search_mode(getattr(args, "freq_search_mode", DEFAULT_FREQ_SEARCH_MODE))
    freq_candidate_source = _parse_freq_candidate_source(
        getattr(args, "freq_candidate_source", DEFAULT_FREQ_CANDIDATE_SOURCE)
    )
    idle_eval_mode = _parse_idle_eval_mode(getattr(args, "idle_eval_mode", DEFAULT_IDLE_EVAL_MODE))
    personalized_candidate_counts = _csv_int_tuple(
        getattr(args, "personalized_candidate_count", ""),
        default=DEFAULT_PERSONALIZED_CANDIDATE_COUNT,
    )
    personalized_candidate_count = int(personalized_candidate_counts[0]) if personalized_candidate_counts else 0
    pretrain_budget_sec = float(getattr(args, "pretrain_budget_sec", DEFAULT_PRETRAIN_BUDGET_SEC))
    decision_start_sec = float(getattr(args, "decision_start_sec", DEFAULT_DECISION_START_SEC))
    decision_deadline_sec = float(getattr(args, "decision_deadline_sec", DEFAULT_DECISION_DEADLINE_SEC))
    min_release_windows = int(getattr(args, "min_release_windows", DEFAULT_MIN_RELEASE_WINDOWS))
    timeout_sec = float(getattr(args, "timeout_sec", DEFAULT_TIMEOUT_SEC))
    case_limit = max(0, int(getattr(args, "case_limit", DEFAULT_CASE_LIMIT)))
    started_monotonic = __import__("time").monotonic()
    freq_plan = _frequency_search_plan(
        mode=freq_search_mode,
        candidate_source=freq_candidate_source,
        datasets=datasets,
    )
    shared_frequency_sets = _shared_frequency_sets_for_plan(freq_plan, fallback_freqs=freqs)
    budget = _budget_payload(
        freq_selection_mode=freq_search_mode,
        pretrain_budget_sec=pretrain_budget_sec,
        personalized_candidate_count=personalized_candidate_count,
    )
    subject_whitelist = _parse_subject_whitelist(getattr(args, "subject_whitelist", ""))
    if enable_nc_calibration_simulation:
        if tuple(datasets) != ("ysu_an",):
            raise ValueError("no-control calibration simulation is only supported for --datasets ysu_an")
        if "fbcca_ridge5" not in methods:
            raise ValueError("no-control calibration simulation requires --methods to include fbcca_ridge5")

    subjects = enumerate_external_subjects(
        datasets=datasets,
        freqs=freqs,
        wang_raw_dir=Path(args.wang_raw_dir),
        wang_channels_loc=Path(args.wang_channels_loc),
        beta_raw_dir=Path(args.beta_raw_dir),
        ysu_an_raw_dir=Path(args.ysu_an_raw_dir) if getattr(args, "ysu_an_raw_dir", None) else None,
        ysu_an_channel_loc=Path(args.ysu_an_channel_loc) if getattr(args, "ysu_an_channel_loc", None) else None,
        subject_limit_per_dataset=int(args.subject_limit_per_dataset),
        subject_whitelist=subject_whitelist,
    )
    if not subjects:
        raise RuntimeError("no external subjects found for the requested datasets")

    rows: list[dict[str, Any]] = []
    subject_manifest: list[dict[str, Any]] = []
    completed_rows_total = 0
    subject_count_total = int(len(subjects))
    case_tracker = CaseTracker(expected_subject_count=len(subjects))
    metric_definitions = _metric_definitions_payload(
        step_sec=float(args.step_sec),
        decision_start_sec=decision_start_sec,
        decision_deadline_sec=decision_deadline_sec,
        min_release_windows=min_release_windows,
    )

    def timeout_exceeded() -> bool:
        return bool(timeout_sec > 0.0 and (__import__("time").monotonic() - started_monotonic) >= timeout_sec)

    def case_limit_reached() -> bool:
        return bool(case_limit > 0 and case_tracker.status_count("planned") >= case_limit)

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
        partial_deployable_shared_summaries = _deployable_recipe_summaries(partial_shared_summaries)
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
                "score_bank_mode": score_bank_mode,
                "idle_eval_mode": idle_eval_mode,
                "frequency_selection_mode": freq_search_mode,
                "frequency_search_plan": freq_plan,
                "shared_frequency_sets": [[float(freq) for freq in item] for item in shared_frequency_sets],
                "pretrain_budget": budget,
                "pretrain_budget_sec": float(pretrain_budget_sec),
                "estimated_pretrain_duration_sec": float(budget.get("estimated_pretrain_duration_sec", 0.0)),
                "expected_subject_count": int(len(subjects)),
                "row_count": int(len(rows)),
                "completed_recipe_count": int(completed_rows_total),
                "current_stage": str(stage),
                "current_detail": str(detail),
                "best_recipe": dict(partial_summaries[0]) if partial_summaries else {},
                "best_shared_recipe": dict(partial_shared_summaries[0]) if partial_shared_summaries else {},
                "best_deployable_shared_recipe": (
                    dict(partial_deployable_shared_summaries[0])
                    if partial_deployable_shared_summaries
                    else {}
                ),
                "recipe_summaries": partial_summaries,
                "shared_recipe_summaries": partial_shared_summaries,
                "deployable_shared_recipe_summaries": partial_deployable_shared_summaries,
                "subjects": subject_manifest,
                "evaluation_contract": metric_definitions,
                "resource_limits": _resource_limits_payload(timeout_sec, case_limit=case_limit),
                "case_limit": int(case_limit),
                "failed_cases_path": str(failed_cases_path),
                "coverage_report_path": str(coverage_report_path),
                "failed_case_count": int(len(case_tracker.failed_cases)),
                "idle_proxy_note": (
                    "Idle/no-control is proxied with non-command target stimulus trials for Wang/BETA; "
                    "YSU-an rows use explicit NS1/NS2/NS3 no-control trials."
                ),
            },
        )
        _write_json(failed_cases_path, {"failed_cases": case_tracker.failed_cases})
        _write_json(coverage_report_path, case_tracker.report())

    emit_progress(stage="start", detail="benchmark initialized", percent=0.0)
    emit_partial("start", "benchmark initialized")

    for subject_index, spec in enumerate(subjects, start=1):
        if timeout_exceeded():
            log(f"timeout reached before loading subject dataset={spec.dataset} subject={spec.subject}")
            break
        log(f"load subject dataset={spec.dataset} subject={spec.subject} path={spec.mat_path}")
        sampling_rate, segments, source_metadata = load_external_subject_segments(spec)
        available_freqs = _available_freqs_for_subject(spec, source_metadata)
        all_target_segments = _build_all_target_segments_for_spec(spec, available_freqs=available_freqs)
        subject_shared_frequency_sets = tuple(
            freq_set for freq_set in shared_frequency_sets if _freqs_available(freq_set, available_freqs)
        )
        if freq_search_mode in {"shared_fixed4", "both"} and not subject_shared_frequency_sets:
            raise RuntimeError(f"no shared frequency set is available for {spec.dataset}:{spec.subject}")
        personalized_subject_candidates = _candidate_freqs_for_subject(
            candidate_freqs=tuple(float(freq) for freq in freq_plan.get("candidate_freqs", []) or ()),
            available_freqs=available_freqs,
            count=int(personalized_candidate_count),
        )
        active_segments_for_window = [item for item in segments if item[0].expected_freq is not None]
        if not active_segments_for_window:
            active_segments_for_window = list(segments)
        counts = _count_segments(segments, freqs)
        planning_counts = _count_segments(active_segments_for_window, freqs)
        max_supported_win_sec = _max_supported_win_sec(segments, sampling_rate)
        planning_max_supported_win_sec = _max_supported_win_sec(active_segments_for_window, sampling_rate)
        channel_compatibility = _channel_compatibility_payload(spec.dataset, source_metadata)
        subject_manifest.append(
            {
                "dataset": str(spec.dataset),
                "subject": str(spec.subject),
                "mat_path": str(spec.mat_path),
                "channel_loc_path": "" if spec.channel_loc_path is None else str(spec.channel_loc_path),
                "sampling_rate": int(sampling_rate),
                "max_supported_win_sec": float(max_supported_win_sec),
                "counts": counts,
                "planning_max_supported_win_sec": float(planning_max_supported_win_sec),
                "planning_counts": planning_counts,
                "available_freqs": [float(freq) for freq in available_freqs],
                "shared_frequency_set_count": int(len(subject_shared_frequency_sets)),
                "personalized_candidate_freqs": [float(freq) for freq in personalized_subject_candidates],
                "source_metadata": source_metadata,
                "channel_compatibility": channel_compatibility,
            }
        )
        blocks = list(planning_counts["blocks"])
        fast_candidates = [
            (float(win_sec), float(template_weight))
            for win_sec, template_weight in product(
                fast_win_sec_candidates,
                fast_template_weight_candidates,
            )
            if float(win_sec) <= float(planning_max_supported_win_sec) + 1e-9
        ]
        threshold_supported_wins = tuple(
            float(win_sec)
            for win_sec in threshold_win_sec_candidates
            if float(win_sec) <= float(planning_max_supported_win_sec) + 1e-9
        )
        score_method_candidate_pairs_by_method: dict[str, list[tuple[float, int]]] = {}
        for method_name in methods:
            if method_name not in SCORE_METHOD_SPECS:
                continue
            score_method_candidate_pairs_by_method[method_name] = _score_method_candidate_pairs(
                method_name=method_name,
                win_sec_candidates=classifier_win_sec_candidates,
                min_enter_candidates=classifier_min_enter_candidates,
                max_supported_win_sec=float(planning_max_supported_win_sec),
                sampling_rate=int(sampling_rate),
            )
        plans_by_calibration: list[tuple[int, list[SplitPlan]]] = []
        subject_planned_rows = 0
        frequency_case_multiplier = 1
        if freq_search_mode == "shared_fixed4":
            frequency_case_multiplier = int(len(subject_shared_frequency_sets))
        elif freq_search_mode == "personalized_upper_bound":
            frequency_case_multiplier = 1
        elif freq_search_mode == "both":
            frequency_case_multiplier = int(len(subject_shared_frequency_sets)) + 1
        rows_per_split = 0
        if "zero_shot_default" in methods:
            rows_per_split += 1
        if "fast_fbcca" in methods:
            rows_per_split += len(fast_candidates)
        for method_name, candidate_pairs in score_method_candidate_pairs_by_method.items():
            rows_per_split += int(
                len(candidate_pairs)
                * len(classifier_max_gap_candidates)
                * len(classifier_smoothing_candidates)
                * (len(classifier_gate_variant_param_grid) if str(method_name) == "fbcca_ridge5" else 1)
            )
            if enable_nc_calibration_simulation and str(method_name) == "fbcca_ridge5":
                rows_per_split += int(
                    len(candidate_pairs)
                    * len(classifier_max_gap_candidates)
                    * len(classifier_smoothing_candidates)
                    * len(nc_calibration_seconds)
                    * len(nc_calibration_sources)
                    * len(nc_calibration_gate_types)
                )
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
            subject_planned_rows += int(
                len(plans) * len(idle_multipliers) * rows_per_split * max(1, frequency_case_multiplier)
            )
        subject_completed_rows = 0

        emit_progress(
            stage="load_subject",
            detail=f"loaded dataset={spec.dataset} subject={spec.subject}",
            percent=100.0 * float(subject_index - 1) / float(max(subject_count_total, 1)),
            current_dataset=str(spec.dataset),
            current_subject=str(spec.subject),
        )
        emit_partial("load_subject", f"loaded dataset={spec.dataset} subject={spec.subject}")
        subject_decoder_cache: dict[tuple[Any, ...], Any] = {}
        subject_scored_cache: dict[tuple[Any, ...], dict[tuple[Any, ...], ScoredTrial]] = {}
        for calibration_block_count, plans in plans_by_calibration:
            if not plans:
                log(
                    f"skip subject={spec.subject} dataset={spec.dataset} calibration_blocks={int(calibration_block_count)}"
                )
                continue
            for plan in plans:
                for idle_multiplier in idle_multipliers:
                    frequency_cases: list[FrequencyEvalCase] = []
                    if freq_search_mode in {"none", "shared_fixed4", "both"}:
                        shared_mode_name = "none" if freq_search_mode == "none" else "shared_fixed4"
                        for shared_freqs in (
                            (_canonical_freq_tuple(freqs),)
                            if freq_search_mode == "none"
                            else subject_shared_frequency_sets
                        ):
                            frequency_cases.append(
                                FrequencyEvalCase(
                                    mode=shared_mode_name,
                                    frequency_set_id=_frequency_set_id(shared_mode_name, shared_freqs),
                                    freqs=_canonical_freq_tuple(shared_freqs),
                                    candidate_freqs=tuple(float(freq) for freq in freq_plan.get("candidate_freqs", []) or ()),
                                    personalized_candidate_count=0,
                                    selected_by_calibration=False,
                                )
                            )
                    if freq_search_mode in {"personalized_upper_bound", "both"}:
                        selected_personalized_freqs, selection_summary = _score_personalized_frequency_candidates(
                            all_target_segments=all_target_segments,
                            candidate_freqs=personalized_subject_candidates,
                            calibration_blocks=plan.calibration_blocks,
                            sampling_rate=int(sampling_rate),
                            win_sec=float(classifier_win_sec_candidates[0] if classifier_win_sec_candidates else 1.25),
                            max_supported_win_sec=float(max_supported_win_sec),
                            step_sec=float(args.step_sec),
                            compute_backend=str(args.compute_backend),
                            gpu_device=int(args.gpu_device),
                            gpu_precision=str(args.gpu_precision),
                        )
                        frequency_cases.append(
                            FrequencyEvalCase(
                                mode="personalized_upper_bound",
                                frequency_set_id=(
                                    f"personalized_upper_bound_calibration_only_c"
                                    f"{int(len(personalized_subject_candidates))}"
                                ),
                                freqs=selected_personalized_freqs,
                                candidate_freqs=personalized_subject_candidates,
                                personalized_candidate_count=int(len(personalized_subject_candidates)),
                                selected_by_calibration=True,
                            )
                        )
                    else:
                        selection_summary = {}
                    for frequency_case in frequency_cases:
                        case_freqs = frequency_case.freqs
                        case_segments = _relabel_segments_for_command_freqs(
                            all_target_segments,
                            command_freqs=case_freqs,
                        )
                        clean_idle_segments = (
                            _build_clean_idle_segments_for_spec(spec, command_freqs=case_freqs)
                            if idle_eval_mode in {"clean_idle_proxy", "both"}
                            else []
                        )
                        if str(spec.dataset).strip().lower() == "ysu_an":
                            calibration_segments, holdout_segments, split_summary = select_ysuan_split_segments(
                                case_segments,
                                freqs=case_freqs,
                                calibration_blocks=plan.calibration_blocks,
                                holdout_blocks=plan.holdout_blocks,
                                idle_multiplier=float(idle_multiplier),
                                seed=int(plan.seed),
                            )
                        else:
                            calibration_segments, holdout_segments, split_summary = select_split_segments(
                                case_segments,
                                freqs=case_freqs,
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
                        split_summary.update(_frequency_case_payload(frequency_case))
                        if frequency_case.mode == "personalized_upper_bound":
                            split_summary["frequency_selection_summary"] = selection_summary
                        method_root = report_root / spec.dataset / spec.subject / (
                            f"cb{int(calibration_block_count)}_idle{float(idle_multiplier):g}_"
                            f"split{int(plan.split_index):02d}_{frequency_case.frequency_set_id}"
                        )
                        score_bundle_cache: dict[tuple[str, float, str], tuple[list[ScoredTrial], list[ScoredTrial]]] = {}
                        clean_idle_score_cache: dict[tuple[str, float, str], list[ScoredTrial]] = {}
                        nc_calibration_score_cache: dict[tuple[str, float, str, float, str], tuple[list[ScoredTrial], dict[str, Any]]] = {}
                        lda_base_cache: dict[tuple[str, float], FBCCALDA5Model] = {}
                        ridge_base_cache: dict[tuple[str, float], list[FBCCARidge5Model]] = {}
                        ridge_baseline_model_cache: dict[tuple[str, float, str, int, int, int], FBCCARidge5Model] = {}

                        def _method_cache_key(method_name: str, win_sec: float) -> tuple[str, float, str]:
                            namespace = _score_method_cache_namespace(method_name)
                            return (namespace, round(float(win_sec), 9), score_bank_mode)

                        def make_case_context(
                            *,
                            method_name: str,
                            win_sec: float,
                            min_enter: int = 1,
                        ) -> CaseContext:
                            return CaseContext(
                                dataset=str(spec.dataset),
                                subject=str(spec.subject),
                                frequency_profile=_frequency_profile_name(case_freqs),
                                frequency_set_id=str(frequency_case.frequency_set_id),
                                selected_freqs=_canonical_freq_tuple(case_freqs),
                                method=str(method_name),
                                calibration_blocks=tuple(int(block) for block in plan.calibration_blocks),
                                holdout_blocks=tuple(int(block) for block in plan.holdout_blocks),
                                split_index=int(plan.split_index),
                                window_length_s=float(win_sec),
                                min_enter_windows=max(1, int(min_enter)),
                                reject_gate=_reject_gate_name(
                                    method_name=str(method_name),
                                    threshold_policy=classifier_threshold_policy,
                                    score_bank_mode=score_bank_mode,
                                ),
                                implementation_level=_method_implementation_level(
                                    str(method_name),
                                    len(plan.calibration_blocks),
                                ),
                            )

                        def run_case(
                            ctx: CaseContext,
                            *,
                            fn: Any,
                            method_name: str,
                            detail: str,
                        ) -> None:
                            nonlocal completed_rows_total, subject_completed_rows
                            skip_reason = _method_score_bank_skip_reason(ctx.method, score_bank_mode)
                            if skip_reason:
                                skip_case(
                                    ctx,
                                    reason=skip_reason,
                                    detail=(
                                        f"{ctx.method} is incompatible with score_bank_mode={score_bank_mode}; "
                                        "use command_only for template/spatial decoders"
                                    ),
                                )
                                return
                            if case_limit_reached():
                                return
                            case_tracker.planned(ctx)
                            if timeout_exceeded():
                                case_tracker.skipped(ctx, reason="timeout", detail="run timeout reached before case start")
                                emit_partial("timeout", detail)
                                return
                            try:
                                raw_row = fn()
                                row = _enrich_result_row(
                                    raw_row,
                                    frequency_profile=ctx.frequency_profile,
                                    frequency_case=frequency_case,
                                    step_sec=float(args.step_sec),
                                    decision_start_sec=decision_start_sec,
                                    decision_deadline_sec=decision_deadline_sec,
                                    min_release_windows=min_release_windows,
                                    threshold_policy=classifier_threshold_policy,
                                    score_bank_mode=score_bank_mode,
                                )
                            except Exception as exc:
                                log(f"case {ctx.method} dataset={ctx.dataset} subject={ctx.subject} failed: {exc}")
                                if _is_tdca_insufficient_case(ctx, exc):
                                    case_tracker.skipped(
                                        ctx,
                                        reason="insufficient_training_trials",
                                        detail=str(exc),
                                    )
                                    emit_partial("case_skipped", detail)
                                else:
                                    case_tracker.failed(ctx, exc=exc)
                                    emit_partial("case_failed", detail)
                                return
                            rows.append(row)
                            case_tracker.completed(ctx, row=row)
                            completed_rows_total += 1
                            subject_completed_rows += 1
                            subject_fraction = float(subject_completed_rows) / float(max(subject_planned_rows, 1))
                            percent = 100.0 * (
                                (float(subject_index - 1) + min(max(subject_fraction, 0.0), 1.0))
                                / float(max(subject_count_total, 1))
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

                        def skip_case(ctx: CaseContext, *, reason: str, detail: str) -> None:
                            case_tracker.planned(ctx)
                            case_tracker.skipped(ctx, reason=reason, detail=detail)
                            emit_partial("case_skipped", detail)

                        def scored_for_method_win(method_name: str, win_sec: float) -> tuple[list[ScoredTrial], list[ScoredTrial]]:
                            cache_key = _method_cache_key(method_name, float(win_sec))
                            if cache_key not in score_bundle_cache:
                                full_bank_freqs = _full_bank_freqs_for_dataset(
                                    dataset=spec.dataset,
                                    score_bank_mode=score_bank_mode,
                                    fallback_freqs=case_freqs,
                                )
                                score_bundle_cache[cache_key] = _score_split_once_for_method(
                                    method_name=str(method_name),
                                    freqs=case_freqs,
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
                                    score_bank_mode=score_bank_mode,
                                    full_bank_freqs=full_bank_freqs,
                                    decoder_cache=subject_decoder_cache,
                                    scored_cache=subject_scored_cache,
                                )
                            return score_bundle_cache[cache_key]

                        def clean_idle_for_method_win(method_name: str, win_sec: float) -> tuple[list[ScoredTrial], dict[str, Any]]:
                            support = _clean_idle_proxy_support_payload(
                                clean_idle_segments=clean_idle_segments,
                                sampling_rate=int(sampling_rate),
                                win_sec=float(win_sec),
                            )
                            if not support.get("supported"):
                                return [], support
                            cache_key = _method_cache_key(method_name, float(win_sec))
                            if cache_key not in clean_idle_score_cache:
                                full_bank_freqs = _full_bank_freqs_for_dataset(
                                    dataset=spec.dataset,
                                    score_bank_mode=score_bank_mode,
                                    fallback_freqs=case_freqs,
                                )
                                _unused_cal, clean_scored = _score_split_once_for_method(
                                    method_name=str(method_name),
                                    freqs=case_freqs,
                                    sampling_rate=sampling_rate,
                                    step_sec=float(args.step_sec),
                                    compute_backend=str(args.compute_backend),
                                    gpu_device=int(args.gpu_device),
                                    gpu_precision=str(args.gpu_precision),
                                    calibration_segments=calibration_segments,
                                    holdout_segments=clean_idle_segments,
                                    win_sec=float(win_sec),
                                    context=(
                                        f"{str(method_name)} clean-idle dataset={spec.dataset} subject={spec.subject} "
                                        f"split={int(plan.split_index)} win={float(win_sec):g}"
                                    ),
                                    score_bank_mode=score_bank_mode,
                                    full_bank_freqs=full_bank_freqs,
                                    validate_holdout_control=False,
                                    decoder_cache=subject_decoder_cache,
                                    scored_cache=subject_scored_cache,
                                )
                                clean_idle_score_cache[cache_key] = clean_scored
                            return clean_idle_score_cache[cache_key], support

                        def clean_idle_payload_for(method_name: str, win_sec: float) -> tuple[Optional[list[ScoredTrial]], Optional[dict[str, Any]]]:
                            if str(spec.dataset).strip().lower() == "ysu_an":
                                _calibration_scored, holdout_scored = scored_for_method_win(method_name, float(win_sec))
                                return _ysuan_holdout_no_control_scored(
                                    holdout_scored,
                                    win_sec=float(win_sec),
                                )
                            if idle_eval_mode not in {"clean_idle_proxy", "both"}:
                                return None, None
                            clean_scored, support = clean_idle_for_method_win(method_name, float(win_sec))
                            return clean_scored, support

                        def nc_calibration_payload_for(
                            method_name: str,
                            win_sec: float,
                            *,
                            seconds: float,
                            source: str,
                        ) -> tuple[list[ScoredTrial], dict[str, Any]]:
                            if str(spec.dataset).strip().lower() != "ysu_an":
                                return [], {
                                    "source": _parse_nc_calibration_source(source),
                                    "requested_seconds": float(seconds),
                                    "selected_seconds": 0.0,
                                    "supported": False,
                                    "reason": "dataset_not_ysu_an",
                                    "fit_split": "unsupported",
                                    "test_split": "holdout_blocks",
                                }
                            parsed_source = _parse_nc_calibration_source(source)
                            cache_key = (
                                _score_method_cache_namespace(method_name),
                                round(float(win_sec), 9),
                                score_bank_mode,
                                round(float(seconds), 9),
                                parsed_source,
                            )
                            if cache_key in nc_calibration_score_cache:
                                return nc_calibration_score_cache[cache_key]
                            pool = _ysuan_ns_calibration_pool_from_segments(
                                calibration_segments,
                                ns_calibration_trials_per_subtype=None,
                            )
                            selected_segments, provenance = _select_nc_calibration_segments(
                                pool,
                                source=parsed_source,
                                seconds=float(seconds),
                                sampling_rate=int(sampling_rate),
                            )
                            provenance = {
                                **dict(provenance),
                                "fit_split": "current_split_calibration_blocks_no_control_only",
                                "test_split": "holdout_blocks",
                                "calibration_blocks": [int(block) for block in plan.calibration_blocks],
                                "holdout_blocks": [int(block) for block in plan.holdout_blocks],
                                "available_counts": {
                                    subtype: int(len(list(pool.get(subtype, []) or [])))
                                    for subtype in ("ns1", "ns2", "ns3")
                                },
                                "leakage_guard": "pool_built_from_current_split_calibration_segments_only",
                            }
                            if not selected_segments:
                                payload = (
                                    [],
                                    {
                                        **dict(provenance),
                                        "supported": False,
                                        "reason": "zero_or_unavailable_no_control_calibration",
                                    },
                                )
                                nc_calibration_score_cache[cache_key] = payload
                                return payload
                            full_bank_freqs = _full_bank_freqs_for_dataset(
                                dataset=spec.dataset,
                                score_bank_mode=score_bank_mode,
                                fallback_freqs=case_freqs,
                            )
                            scored = _score_segment_subset_cached(
                                freqs=case_freqs,
                                sampling_rate=int(sampling_rate),
                                step_sec=float(args.step_sec),
                                compute_backend=str(args.compute_backend),
                                gpu_device=int(args.gpu_device),
                                gpu_precision=str(args.gpu_precision),
                                win_sec=float(win_sec),
                                score_bank_mode=score_bank_mode,
                                full_bank_freqs=full_bank_freqs,
                                segments=selected_segments,
                                context=(
                                    f"{str(method_name)} nc-calibration dataset={spec.dataset} "
                                    f"subject={spec.subject} split={int(plan.split_index)} "
                                    f"win={float(win_sec):g} source={parsed_source} seconds={float(seconds):g}"
                                ),
                                decoder_cache=subject_decoder_cache,
                                scored_cache=subject_scored_cache,
                                require_control=False,
                            )
                            payload = (
                                scored,
                                {
                                    **dict(provenance),
                                    "supported": bool(scored),
                                    "scored_trial_count": int(len(scored)),
                                    "score_space_only": True,
                                },
                            )
                            nc_calibration_score_cache[cache_key] = payload
                            return payload

                        def lda_base_for_win(method_name: str, win_sec: float) -> FBCCALDA5Model:
                            cache_key = _method_cache_key(method_name, float(win_sec))
                            if cache_key not in lda_base_cache:
                                calibration_scored, _holdout_scored = scored_for_method_win(method_name, float(win_sec))
                                try:
                                    lda_base_cache[cache_key] = _fit_fbcca_lda5_base_model(
                                        calibration_scored,
                                        freqs=case_freqs,
                                        score_source_name=_score_method_spec(method_name).score_source_name,
                                    )
                                except TypeError as exc:
                                    if "score_source_name" not in str(exc):
                                        raise
                                    lda_base_cache[cache_key] = _fit_fbcca_lda5_base_model(
                                        calibration_scored,
                                        freqs=case_freqs,
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
                                            freqs=case_freqs,
                                            l2=float(l2),
                                            score_source_name=_score_method_spec(method_name).score_source_name,
                                        )
                                    except TypeError as exc:
                                        if "score_source_name" not in str(exc):
                                            raise
                                        base_model = _fit_fbcca_ridge5_base_model(
                                            calibration_scored,
                                            freqs=case_freqs,
                                            l2=float(l2),
                                        )
                                    ridge_base_models.append(base_model)
                                ridge_base_cache[cache_key] = ridge_base_models
                            return ridge_base_cache[cache_key]

                        def ridge_baseline_model_for_win(
                            method_name: str,
                            win_sec: float,
                            *,
                            min_enter: int,
                            max_gap: int,
                            smoothing_windows: int,
                        ) -> FBCCARidge5Model:
                            cache_key = (
                                str(method_name),
                                round(float(win_sec), 9),
                                score_bank_mode,
                                int(min_enter),
                                int(max_gap),
                                int(smoothing_windows),
                            )
                            if cache_key not in ridge_baseline_model_cache:
                                calibration_scored, _holdout_scored = scored_for_method_win(method_name, float(win_sec))
                                ridge_baseline_model_cache[cache_key] = _fit_fbcca_ridge5_model(
                                    calibration_scored,
                                    freqs=case_freqs,
                                    win_sec=float(
                                        _method_latency_window_sec(
                                            method_name=str(method_name),
                                            win_sec=float(win_sec),
                                            sampling_rate=int(sampling_rate),
                                        )
                                    ),
                                    step_sec=float(args.step_sec),
                                    min_enter_windows=max(1, int(min_enter)),
                                    max_gap_windows=max(0, int(max_gap)),
                                    smoothing_windows=max(1, int(smoothing_windows)),
                                    threshold_policy=classifier_threshold_policy,
                                    base_models=ridge_bases_for_win(method_name, float(win_sec)),
                                    score_source_name=str(_score_method_spec(method_name).score_source_name),
                                    score_bank_mode=score_bank_mode,
                                    gate_variant_params={"gate_variant": CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW},
                                )
                            return ridge_baseline_model_cache[cache_key]

                        if "zero_shot_default" in methods:
                            detail = (
                                "zero_shot_default "
                                f"dataset={spec.dataset} subject={spec.subject} split={int(plan.split_index)} "
                                f"cal_blocks={len(plan.calibration_blocks)} idle_mult={float(idle_multiplier):g}"
                            )
                            log(detail)
                            ctx = make_case_context(method_name="zero_shot_default", win_sec=max_supported_win_sec, min_enter=1)
                            run_case(
                                ctx,
                                fn=lambda: run_zero_shot_default(
                                    spec=spec,
                                    split_plan=plan,
                                    split_summary=split_summary,
                                    sampling_rate=sampling_rate,
                                    freqs=case_freqs,
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
                                detail = (
                                    f"skip fast_fbcca dataset={spec.dataset} subject={spec.subject} "
                                    f"because no win candidate fits max_supported_win_sec={float(planning_max_supported_win_sec):g}s"
                                )
                                log(detail)
                                ctx = make_case_context(method_name="fast_fbcca", win_sec=0.0, min_enter=1)
                                skip_case(ctx, reason="insufficient_window_length", detail=detail)
                            for win_sec, template_weight in fast_candidates:
                                detail = (
                                    "fast_fbcca "
                                    f"dataset={spec.dataset} subject={spec.subject} split={int(plan.split_index)} "
                                    f"cal_blocks={len(plan.calibration_blocks)} idle_mult={float(idle_multiplier):g} "
                                    f"win={float(win_sec):g} tw={float(template_weight):g}"
                                )
                                log(detail)
                                ctx = make_case_context(method_name="fast_fbcca", win_sec=float(win_sec), min_enter=1)
                                run_case(
                                    ctx,
                                    fn=lambda win_sec=win_sec, template_weight=template_weight: run_fast_fbcca_method(
                                        method_dir=method_root / "fast_fbcca",
                                        spec=spec,
                                        split_plan=plan,
                                        split_summary=split_summary,
                                        sampling_rate=sampling_rate,
                                        freqs=case_freqs,
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
                                detail = (
                                    f"skip fbcca_lda5 dataset={spec.dataset} subject={spec.subject} "
                                    f"because no win candidate fits max_supported_win_sec={float(planning_max_supported_win_sec):g}s"
                                )
                                log(detail)
                                ctx = make_case_context(method_name="fbcca_lda5", win_sec=0.0, min_enter=1)
                                skip_case(ctx, reason="insufficient_window_length", detail=detail)
                            for win_sec, min_enter in lda_candidates:
                                for max_gap in classifier_max_gap_candidates:
                                    for smoothing_windows in classifier_smoothing_candidates:
                                        detail = (
                                            "fbcca_lda5 "
                                            f"dataset={spec.dataset} subject={spec.subject} split={int(plan.split_index)} "
                                            f"cal_blocks={len(plan.calibration_blocks)} idle_mult={float(idle_multiplier):g} "
                                            f"win={float(win_sec):g} min_enter={int(min_enter)} max_gap={int(max_gap)} "
                                            f"smooth={int(smoothing_windows)} threshold_policy={classifier_threshold_policy}"
                                        )
                                        log(detail)
                                        ctx = make_case_context(
                                            method_name="fbcca_lda5",
                                            win_sec=float(win_sec),
                                            min_enter=int(min_enter),
                                        )
                                        run_case(
                                            ctx,
                                            fn=lambda win_sec=win_sec, min_enter=min_enter, max_gap=max_gap, smoothing_windows=smoothing_windows: run_fbcca_lda5_method(
                                                artifact_dir=method_root / "fbcca_lda5",
                                                spec=spec,
                                                split_plan=plan,
                                                split_summary=split_summary,
                                                sampling_rate=sampling_rate,
                                                freqs=case_freqs,
                                                calibration_segments=calibration_segments,
                                                holdout_segments=holdout_segments,
                                                compute_backend=str(args.compute_backend),
                                                gpu_device=int(args.gpu_device),
                                                gpu_precision=str(args.gpu_precision),
                                                step_sec=float(args.step_sec),
                                                win_sec=float(win_sec),
                                                min_enter_windows=int(min_enter),
                                                max_gap_windows=int(max_gap),
                                                smoothing_windows=int(smoothing_windows),
                                                threshold_policy=classifier_threshold_policy,
                                                score_bank_mode=score_bank_mode,
                                                calibration_scored=scored_for_method_win("fbcca_lda5", float(win_sec))[0],
                                                holdout_scored=scored_for_method_win("fbcca_lda5", float(win_sec))[1],
                                                base_model=lda_base_for_win("fbcca_lda5", float(win_sec)),
                                                clean_idle_scored=clean_idle_payload_for("fbcca_lda5", float(win_sec))[0],
                                                clean_idle_support=clean_idle_payload_for("fbcca_lda5", float(win_sec))[1],
                                            ),
                                            method_name="fbcca_lda5",
                                            detail=detail,
                                        )
                        if "fbcca_ridge5" in methods:
                            ridge_candidates = score_method_candidate_pairs_by_method.get("fbcca_ridge5", [])
                            if not ridge_candidates:
                                detail = (
                                    f"skip fbcca_ridge5 dataset={spec.dataset} subject={spec.subject} "
                                    f"because no win candidate fits max_supported_win_sec={float(planning_max_supported_win_sec):g}s"
                                )
                                log(detail)
                                ctx = make_case_context(method_name="fbcca_ridge5", win_sec=0.0, min_enter=1)
                                skip_case(ctx, reason="insufficient_window_length", detail=detail)
                            for win_sec, min_enter in ridge_candidates:
                                for max_gap in classifier_max_gap_candidates:
                                    for smoothing_windows in classifier_smoothing_candidates:
                                        for gate_variant_params in classifier_gate_variant_param_grid:
                                            gate_variant = parse_classifier_gate_variant(gate_variant_params.get("gate_variant"))
                                            gate_token = _classifier_gate_variant_token(gate_variant, gate_variant_params)
                                            detail = (
                                                "fbcca_ridge5 "
                                                f"dataset={spec.dataset} subject={spec.subject} split={int(plan.split_index)} "
                                                f"cal_blocks={len(plan.calibration_blocks)} idle_mult={float(idle_multiplier):g} "
                                                f"win={float(win_sec):g} min_enter={int(min_enter)} max_gap={int(max_gap)} "
                                                f"smooth={int(smoothing_windows)} threshold_policy={classifier_threshold_policy} "
                                                f"gate_variant={gate_variant} gate_token={gate_token}"
                                            )
                                            log(detail)
                                            ctx = make_case_context(
                                                method_name="fbcca_ridge5",
                                                win_sec=float(win_sec),
                                                min_enter=int(min_enter),
                                            )
                                            run_case(
                                                ctx,
                                                fn=lambda win_sec=win_sec, min_enter=min_enter, max_gap=max_gap, smoothing_windows=smoothing_windows, gate_variant_params=gate_variant_params: run_fbcca_ridge5_method(
                                                    artifact_dir=method_root / "fbcca_ridge5",
                                                    spec=spec,
                                                    split_plan=plan,
                                                    split_summary=split_summary,
                                                    sampling_rate=sampling_rate,
                                                    freqs=case_freqs,
                                                    calibration_segments=calibration_segments,
                                                    holdout_segments=holdout_segments,
                                                    compute_backend=str(args.compute_backend),
                                                    gpu_device=int(args.gpu_device),
                                                    gpu_precision=str(args.gpu_precision),
                                                    step_sec=float(args.step_sec),
                                                    win_sec=float(win_sec),
                                                    min_enter_windows=int(min_enter),
                                                    max_gap_windows=int(max_gap),
                                                    smoothing_windows=int(smoothing_windows),
                                                    threshold_policy=classifier_threshold_policy,
                                                    score_bank_mode=score_bank_mode,
                                                    calibration_scored=scored_for_method_win("fbcca_ridge5", float(win_sec))[0],
                                                    holdout_scored=scored_for_method_win("fbcca_ridge5", float(win_sec))[1],
                                                    base_models=ridge_bases_for_win("fbcca_ridge5", float(win_sec)),
                                                    clean_idle_scored=clean_idle_payload_for("fbcca_ridge5", float(win_sec))[0],
                                                    clean_idle_support=clean_idle_payload_for("fbcca_ridge5", float(win_sec))[1],
                                                    gate_variant_params=gate_variant_params,
                                                ),
                                                method_name="fbcca_ridge5",
                                                detail=detail,
                                            )
                                        if enable_nc_calibration_simulation:
                                            for nc_seconds in nc_calibration_seconds:
                                                for nc_source in nc_calibration_sources:
                                                    for nc_gate_type in nc_calibration_gate_types:
                                                        nc_detail = (
                                                            "fbcca_ridge5_nc_calibration "
                                                            f"dataset={spec.dataset} subject={spec.subject} "
                                                            f"split={int(plan.split_index)} "
                                                            f"cal_blocks={len(plan.calibration_blocks)} "
                                                            f"idle_mult={float(idle_multiplier):g} "
                                                            f"win={float(win_sec):g} min_enter={int(min_enter)} "
                                                            f"max_gap={int(max_gap)} smooth={int(smoothing_windows)} "
                                                            f"nc_seconds={float(nc_seconds):g} "
                                                            f"nc_source={nc_source} nc_gate_type={nc_gate_type}"
                                                        )
                                                        log(nc_detail)
                                                        ctx = make_case_context(
                                                            method_name="fbcca_ridge5_nc_calibration",
                                                            win_sec=float(win_sec),
                                                            min_enter=int(min_enter),
                                                        )
                                                        run_case(
                                                            ctx,
                                                            fn=lambda win_sec=win_sec, min_enter=min_enter, max_gap=max_gap, smoothing_windows=smoothing_windows, nc_seconds=nc_seconds, nc_source=nc_source, nc_gate_type=nc_gate_type: run_fbcca_ridge5_nc_calibration_method(
                                                                spec=spec,
                                                                split_plan=plan,
                                                                split_summary=split_summary,
                                                                sampling_rate=sampling_rate,
                                                                freqs=case_freqs,
                                                                step_sec=float(args.step_sec),
                                                                win_sec=float(win_sec),
                                                                min_enter_windows=int(min_enter),
                                                                max_gap_windows=int(max_gap),
                                                                smoothing_windows=int(smoothing_windows),
                                                                threshold_policy=classifier_threshold_policy,
                                                                calibration_scored=scored_for_method_win("fbcca_ridge5", float(win_sec))[0],
                                                                holdout_scored=scored_for_method_win("fbcca_ridge5", float(win_sec))[1],
                                                                nc_calibration_scored=nc_calibration_payload_for(
                                                                    "fbcca_ridge5",
                                                                    float(win_sec),
                                                                    seconds=float(nc_seconds),
                                                                    source=str(nc_source),
                                                                )[0],
                                                                clean_idle_scored=clean_idle_payload_for("fbcca_ridge5", float(win_sec))[0],
                                                                clean_idle_support=clean_idle_payload_for("fbcca_ridge5", float(win_sec))[1],
                                                                base_models=ridge_bases_for_win("fbcca_ridge5", float(win_sec)),
                                                                baseline_model=ridge_baseline_model_for_win(
                                                                    "fbcca_ridge5",
                                                                    float(win_sec),
                                                                    min_enter=int(min_enter),
                                                                    max_gap=int(max_gap),
                                                                    smoothing_windows=int(smoothing_windows),
                                                                ),
                                                                score_bank_mode=score_bank_mode,
                                                                nc_seconds=float(nc_seconds),
                                                                nc_source=str(nc_source),
                                                                nc_gate_type=str(nc_gate_type),
                                                                nc_provenance=nc_calibration_payload_for(
                                                                    "fbcca_ridge5",
                                                                    float(win_sec),
                                                                    seconds=float(nc_seconds),
                                                                    source=str(nc_source),
                                                                )[1],
                                                            ),
                                                            method_name="fbcca_ridge5_nc_calibration",
                                                            detail=nc_detail,
                                                        )
                        for method_name in SUPPORTED_SHORT_PRETRAIN_METHODS:
                            if method_name not in methods or method_name in {"fbcca_lda5", "fbcca_ridge5"}:
                                continue
                            method_candidates = score_method_candidate_pairs_by_method.get(method_name, [])
                            if not method_candidates:
                                detail = (
                                    f"skip {method_name} dataset={spec.dataset} subject={spec.subject} "
                                    f"because no win candidate fits max_supported_win_sec={float(planning_max_supported_win_sec):g}s"
                                )
                                log(detail)
                                ctx = make_case_context(method_name=method_name, win_sec=0.0, min_enter=1)
                                reason = (
                                    "insufficient_training_trials"
                                    if str(method_name) == "tdca5"
                                    else "insufficient_window_length"
                                )
                                skip_case(ctx, reason=reason, detail=detail)
                                continue
                            method_spec = _score_method_spec(method_name)
                            for win_sec, min_enter in method_candidates:
                                for max_gap in classifier_max_gap_candidates:
                                    for smoothing_windows in classifier_smoothing_candidates:
                                        detail = (
                                            f"{method_name} "
                                            f"dataset={spec.dataset} subject={spec.subject} split={int(plan.split_index)} "
                                            f"cal_blocks={len(plan.calibration_blocks)} idle_mult={float(idle_multiplier):g} "
                                            f"win={float(win_sec):g} min_enter={int(min_enter)} max_gap={int(max_gap)} "
                                            f"smooth={int(smoothing_windows)} threshold_policy={classifier_threshold_policy}"
                                        )
                                        log(detail)
                                        ctx = make_case_context(
                                            method_name=str(method_name),
                                            win_sec=float(win_sec),
                                            min_enter=int(min_enter),
                                        )
                                        run_case(
                                            ctx,
                                            fn=lambda method_name=method_name, method_spec=method_spec, win_sec=win_sec, min_enter=min_enter, max_gap=max_gap, smoothing_windows=smoothing_windows: run_fbcca_ridge5_method(
                                                artifact_dir=method_root / method_name,
                                                spec=spec,
                                                split_plan=plan,
                                                split_summary=split_summary,
                                                sampling_rate=sampling_rate,
                                                freqs=case_freqs,
                                                calibration_segments=calibration_segments,
                                                holdout_segments=holdout_segments,
                                                compute_backend=str(args.compute_backend),
                                                gpu_device=int(args.gpu_device),
                                                gpu_precision=str(args.gpu_precision),
                                                step_sec=float(args.step_sec),
                                                win_sec=float(win_sec),
                                                min_enter_windows=int(min_enter),
                                                max_gap_windows=int(max_gap),
                                                smoothing_windows=int(smoothing_windows),
                                                threshold_policy=classifier_threshold_policy,
                                                calibration_scored=scored_for_method_win(method_name, float(win_sec))[0],
                                                holdout_scored=scored_for_method_win(method_name, float(win_sec))[1],
                                                base_models=ridge_bases_for_win(method_name, float(win_sec)),
                                                method_name=str(method_name),
                                                score_source_name=str(method_spec.score_source_name),
                                                score_bank_mode=score_bank_mode,
                                                decoder_name=str(method_spec.decoder_name),
                                                decoder_model_params=dict(method_spec.decoder_model_params),
                                                clean_idle_scored=clean_idle_payload_for(method_name, float(win_sec))[0],
                                                clean_idle_support=clean_idle_payload_for(method_name, float(win_sec))[1],
                                            ),
                                            method_name=str(method_name),
                                            detail=detail,
                                        )
                        if "threshold_pretrain" in methods:
                            if not threshold_supported_wins:
                                detail = (
                                    f"skip threshold_pretrain dataset={spec.dataset} subject={spec.subject} "
                                    f"because no win candidate fits max_supported_win_sec={float(planning_max_supported_win_sec):g}s"
                                )
                                log(detail)
                                ctx = make_case_context(method_name="threshold_pretrain", win_sec=0.0, min_enter=1)
                                skip_case(ctx, reason="insufficient_window_length", detail=detail)
                            else:
                                detail = (
                                    "threshold_pretrain "
                                    f"dataset={spec.dataset} subject={spec.subject} split={int(plan.split_index)} "
                                    f"cal_blocks={len(plan.calibration_blocks)} idle_mult={float(idle_multiplier):g}"
                                )
                                log(detail)
                                ctx = make_case_context(
                                    method_name="threshold_pretrain",
                                    win_sec=float(max(threshold_supported_wins)),
                                    min_enter=int(threshold_min_enter_candidates[0] if threshold_min_enter_candidates else 1),
                                )
                                run_case(
                                    ctx,
                                    fn=lambda: run_threshold_pretrain_method(
                                        method_dir=method_root / "threshold_pretrain",
                                        dataset_root=dataset_root / "threshold_pretrain",
                                        spec=spec,
                                        split_plan=plan,
                                        split_summary=split_summary,
                                        sampling_rate=sampling_rate,
                                        freqs=case_freqs,
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
    deployable_shared_summaries = _deployable_recipe_summaries(shared_summaries)
    best_recipe = dict(summaries[0]) if summaries else {}
    best_shared_recipe = dict(shared_summaries[0]) if shared_summaries else {}
    best_deployable_shared_recipe = (
        dict(deployable_shared_summaries[0])
        if deployable_shared_summaries
        else {}
    )
    weak_audit = _weak_subject_audit(best_shared_recipe or best_recipe)
    frequency_set_coverage_subject_count = int(
        (best_shared_recipe or best_recipe).get("coverage_subject_count", 0)
        if (best_shared_recipe or best_recipe)
        else 0
    )
    per_subject_selected_freqs = dict((best_shared_recipe or best_recipe).get("per_subject_selected_freqs", {}) or {})
    candidate_comparison_rows = _decision_table_rows(shared_summaries or summaries)
    subject_breakdown_rows = _subject_breakdown_rows(shared_summaries or summaries)
    subtype_breakdown_rows = _subtype_breakdown_rows(shared_summaries or summaries)
    comparison_baseline_recipe = _comparison_baseline_summary(shared_summaries or summaries)
    recipe_subgroup_comparisons = [
        {
            "method": str(item.get("method", "")),
            "recipe_id": str(item.get("recipe_id", "")),
            "gate_variant": parse_classifier_gate_variant(
                item.get("gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW)
            ),
            "combo_name": _gate_combo_name_for_summary(item),
            **_candidate_subgroup_comparison_payload(item, comparison_baseline_recipe),
        }
        for item in (shared_summaries or summaries)
    ]
    ns2_by_selected_freq_rows = _flatten_split_diagnostic_rows(rows, "ns2_by_selected_freq")
    ns2_by_subject_freq_rows = _flatten_split_diagnostic_rows(rows, "ns2_by_subject_freq")
    selected_freq_confusion_rows = _flatten_split_diagnostic_rows(rows, "selected_freq_confusion")
    logistic_trace_window_rows = _flatten_logistic_trace_rows(rows, "logistic_trace_windows")
    logistic_trace_trial_rows = _flatten_logistic_trace_rows(rows, "logistic_trace_trial_summary")
    logistic_transition_subject_rows = _flatten_logistic_trace_rows(rows, "logistic_transition_counts_by_subject")
    logistic_transition_frequency_rows = _flatten_logistic_trace_rows(rows, "logistic_transition_counts_by_frequency")
    logistic_feature_summary_rows = _flatten_logistic_trace_rows(rows, "logistic_feature_summary_tp_fp")
    tenp5_veto_diagnostic_rows = _flatten_tenp5_veto_rows(rows, "tenp5_ns2_veto_diagnostics")
    tenp5_veto_summary_payload = _tenp5_veto_summary_payload(rows)
    nc_budget_curve_rows = _nc_budget_curve_rows(shared_summaries or summaries)
    csns_feature_summary_rows = _flatten_nc_feature_summary_rows(rows)
    trace_separability = _trace_separability_outputs(logistic_trace_window_rows)
    per_frequency_metric_rows = _split_per_frequency_rows(rows)
    per_frequency_summary = _per_frequency_summary_payload(rows)
    gate_params_payload = _gate_params_payload(shared_summaries or summaries)
    gate_params_by_frequency_payload = _gate_params_by_frequency_payload(shared_summaries or summaries)
    chosen_recipe_for_fields = best_deployable_shared_recipe or best_shared_recipe or best_recipe
    chosen_metrics_for_fields = _summary_metric_payload(chosen_recipe_for_fields)
    chosen_subject_risk = _subject_risk_payload(chosen_recipe_for_fields)
    chosen_subgroup_comparison = (
        _candidate_subgroup_comparison_payload(chosen_recipe_for_fields, comparison_baseline_recipe)
        if chosen_recipe_for_fields
        else {}
    )
    chosen_gate_variant = (
        parse_classifier_gate_variant(chosen_recipe_for_fields.get("gate_variant", CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW))
        if chosen_recipe_for_fields
        else CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW
    )
    chosen_ns2_label = _ns2_safety_label(chosen_recipe_for_fields) if chosen_recipe_for_fields else "not_run"
    chosen_coverage_subject_count = int(
        chosen_recipe_for_fields.get(
            "coverage_subject_count",
            chosen_recipe_for_fields.get("subject_count", 0),
        )
        or 0
    )
    chosen_subject_count = int(
        chosen_recipe_for_fields.get("subject_count", chosen_coverage_subject_count)
        or chosen_coverage_subject_count
    )
    chosen_split_count = int(
        chosen_recipe_for_fields.get("split_count", len(rows))
        if chosen_recipe_for_fields
        else len(rows)
    )
    chosen_calibration_block_count = int(
        chosen_recipe_for_fields.get("calibration_block_count", len(calibration_blocks))
        if chosen_recipe_for_fields
        else len(calibration_blocks)
    )
    chosen_idle_multiplier = _safe_float(
        chosen_recipe_for_fields.get("idle_multiplier"),
        float(idle_multipliers[0]) if idle_multipliers else 1.0,
    )
    chosen_candidate_artifact_paths = _candidate_artifact_paths_for_recipe(rows, chosen_recipe_for_fields)
    channel_compatibility = _channel_compatibility_summary(subject_manifest)
    coverage_report = case_tracker.report()
    failure_payload = {
        "schema_version": "ssvep_external_short_pretrain_failed_cases_v1",
        "run_id": run_id,
        "failed_cases": case_tracker.failed_cases,
        "skipped_cases": case_tracker.skipped_cases(),
        "hard_failed_cases": case_tracker.failed_cases_only(),
    }
    artifact_paths = _artifact_manifest_paths(
        report_root=report_root,
        run_id=run_id,
        log_path=log_path,
        failed_cases_path=failed_cases_path,
        coverage_report_path=coverage_report_path,
        server_log_path=server_log_contract_path,
    )
    subjects_completed = len(
        {
            str(_subject)
            for item in coverage_report.get("by_dataset_frequency_profile_method", [])
            for _subject in item.get("completed_subject_ids", []) or []
        }
    )
    if subjects_completed == 0:
        subjects_completed = len({f"{row.get('dataset','')}:{row.get('subject','')}" for row in rows})
    observed_implementation_levels = {str(row.get("implementation_level", "")) for row in rows if row.get("implementation_level")}
    implementation_level = (
        "paper-faithful"
        if observed_implementation_levels == {"paper-faithful"}
        else "engineering-approx"
        if observed_implementation_levels
        else "not-run"
    )
    observed_reject_gates = sorted({str(row.get("reject_gate", "")) for row in rows if row.get("reject_gate")})
    reject_gate_contract = ",".join(observed_reject_gates) if observed_reject_gates else _reject_gate_name(
        method_name=str(methods[0] if methods else ""),
        threshold_policy=classifier_threshold_policy,
        score_bank_mode=score_bank_mode,
    )
    run_metadata = _run_metadata_payload(
        run_id=run_id,
        datasets=datasets,
        freqs=freqs,
        methods=methods,
        subjects_expected=len(subjects),
        calibration_blocks=calibration_blocks,
        window_lengths=classifier_win_sec_candidates,
        score_bank_mode=score_bank_mode,
        classifier_gate_variants=classifier_gate_variants,
        idle_eval_mode=idle_eval_mode,
        timeout_sec=timeout_sec,
        artifact_paths=artifact_paths,
    )
    evaluation_contract = _evaluation_contract_payload(
        datasets=datasets,
        freqs=freqs,
        methods=methods,
        subjects_expected=len(subjects),
        subjects_completed=subjects_completed,
        calibration_blocks=calibration_blocks,
        window_lengths=classifier_win_sec_candidates,
        step_sec=float(args.step_sec),
        decision_start_sec=decision_start_sec,
        decision_deadline_sec=decision_deadline_sec,
        min_release_windows=min_release_windows,
        reject_gate=reject_gate_contract,
        artifact_paths=artifact_paths,
        implementation_level=implementation_level,
    )
    idle_source_policies = {
        str(dataset): _idle_source_policy_for_dataset(str(dataset))
        for dataset in datasets
    }
    summary = {
        "task": "external-short-pretrain-benchmark",
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "run_metadata": run_metadata,
        "evaluation_contract": evaluation_contract,
        "resource_limits": _resource_limits_payload(timeout_sec, case_limit=case_limit),
        "config": json_safe(vars(args)),
        "dataset": ",".join(str(item) for item in datasets),
        "frequency_profile": [float(freq) for freq in freqs],
        "frequency_profile_id": _frequency_profile_name(freqs),
        "frequency_profile_hz": [float(freq) for freq in freqs],
        "method": ",".join(str(item) for item in methods),
        "subjects_expected": int(len(subjects)),
        "subjects_completed": int(subjects_completed),
        "calibration_blocks": [int(block) for block in calibration_blocks],
        "calibration_block_count": int(chosen_calibration_block_count),
        "holdout_blocks": sorted(
            {
                int(block)
                for row in rows
                for block in (dict(row.get("split_summary", {}) or {}).get("holdout_blocks", []) or [])
            }
        ),
        "window_length_s": [float(win_sec) for win_sec in classifier_win_sec_candidates],
        "win_sec": (
            float(classifier_win_sec_candidates[0])
            if len(classifier_win_sec_candidates) == 1
            else [float(win_sec) for win_sec in classifier_win_sec_candidates]
        ),
        "step_size_s": float(args.step_sec),
        "step_sec": float(args.step_sec),
        "decision_start_s": float(decision_start_sec),
        "decision_deadline_s": float(decision_deadline_sec),
        "reject_gate": reject_gate_contract,
        "gate": str(classifier_threshold_policy),
        "artifact_paths": artifact_paths,
        "implementation_level": implementation_level,
        "paper_faithful": bool(implementation_level == "paper-faithful"),
        "engineering_approx": bool(implementation_level != "paper-faithful"),
        "freqs": [float(freq) for freq in freqs],
        "selected_freqs": [float(freq) for freq in freqs],
        "per_subject_selected_freqs": per_subject_selected_freqs,
        "frequency_selection_mode": freq_search_mode,
        "frequency_search_plan": freq_plan,
        "shared_frequency_sets": [[float(freq) for freq in item] for item in shared_frequency_sets],
        "frequency_set_coverage_subject_count": int(frequency_set_coverage_subject_count),
        "coverage_subject_count": int(chosen_coverage_subject_count),
        "subject_count": int(chosen_subject_count),
        "score_bank_mode": score_bank_mode,
        "score_bank": score_bank_mode,
        "scorer": "fbcca_fixed_all8" if "fbcca_ridge5" in methods else "",
        "gate_variant": chosen_gate_variant,
        "classifier_gate_variants": list(classifier_gate_variants),
        "split_count": int(chosen_split_count),
        "min_enter_windows": (
            int(classifier_min_enter_candidates[0])
            if len(classifier_min_enter_candidates) == 1
            else [int(value) for value in classifier_min_enter_candidates]
        ),
        "smoothing_windows": (
            int(classifier_smoothing_candidates[0])
            if len(classifier_smoothing_candidates) == 1
            else [int(value) for value in classifier_smoothing_candidates]
        ),
        "idle_multiplier": float(chosen_idle_multiplier),
        "idle_eval_mode": idle_eval_mode,
        "case_limit": int(case_limit),
        "pretrain_budget_sec": float(pretrain_budget_sec),
        "estimated_pretrain_duration_sec": float(budget.get("estimated_pretrain_duration_sec", 0.0)),
        "pretrain_budget": budget,
        "weak_subject_audit": weak_audit,
        "datasets": list(datasets),
        "methods": list(methods),
        "subject_whitelist": [[dataset, subject] for dataset, subject in subject_whitelist],
        "subjects": subject_manifest,
        "channel_compatibility": channel_compatibility,
        "expected_subject_count": int(len(subjects)),
        "row_count": int(len(rows)),
        "rows": rows,
        "metrics": {
            **chosen_metrics_for_fields,
            "best_shared_recipe": dict(best_shared_recipe),
            "best_deployable_shared_recipe": dict(best_deployable_shared_recipe),
            "best_recipe_partial_coverage": dict(best_recipe),
        },
        "no_control_subtype_metrics": {
            "ns1": {"fp_per_min": chosen_metrics_for_fields.get("ns1_fp_per_min")},
            "ns2": {"fp_per_min": chosen_metrics_for_fields.get("ns2_fp_per_min")},
            "ns3": {"fp_per_min": chosen_metrics_for_fields.get("ns3_fp_per_min")},
            "mixed_idle_fp_per_min": chosen_metrics_for_fields.get("idle_fp_per_min"),
        },
        "per_frequency_summary": per_frequency_summary,
        "subject_risk": chosen_subject_risk,
        "subject_risk_summary": chosen_subject_risk,
        "subgroup_definitions": _subgroup_definitions_payload(),
        "subgroup_metrics": chosen_subgroup_comparison,
        "recipe_subgroup_comparisons": recipe_subgroup_comparisons,
        "frequency_specific_grid_selection_policy": FREQSPEC_GRID_SELECTION_POLICY,
        "gate_params_by_frequency": gate_params_by_frequency_payload,
        "tenp5_ns2_veto_summary": tenp5_veto_summary_payload,
        "nc_calibration_simulation": {
            "enabled": bool(getattr(args, "enable_nc_calibration_simulation", False)),
            "seconds": [float(value) for value in _csv_float_tuple(
                getattr(args, "nc_calibration_seconds", ""),
                default=DEFAULT_NC_CALIBRATION_SECONDS,
            )],
            "sources": list(_csv_nc_calibration_sources(getattr(args, "nc_calibration_sources", ""))),
            "gate_types": list(_csv_nc_gate_types(getattr(args, "nc_calibration_gate_types", ""))),
            "budget_curve_row_count": int(len(nc_budget_curve_rows)),
            "full24_auto_started": False,
            "runtime_profile_exported": False,
        },
        "trace_separability": dict(trace_separability.get("summary", {}) or {}),
        "subject_risk_flag": bool(
            chosen_recipe_for_fields
            and (
                _candidate_subject_watch_risk(chosen_recipe_for_fields, comparison_baseline_recipe)
                or chosen_subgroup_comparison.get("recall_degraded", False)
            )
        ),
        "decision_table": candidate_comparison_rows,
        "deployable": bool(chosen_recipe_for_fields.get("deployable_budget_pass", False)) if chosen_recipe_for_fields else False,
        "ns2_safe": bool(chosen_ns2_label == "ns2_safe"),
        "ns2_status": chosen_ns2_label,
        "ns2_improved": bool(chosen_subgroup_comparison.get("ns2_improved", False)),
        "recall_degraded": bool(chosen_subgroup_comparison.get("recall_degraded", False)),
        "full24_candidate_eligible": bool(chosen_subgroup_comparison.get("full24_candidate_eligible", False)),
        "runtime_loadable": False,
        "candidate_artifact_paths": chosen_candidate_artifact_paths,
        "recipe_summaries": summaries,
        "shared_recipe_summaries": shared_summaries,
        "deployable_shared_recipe_summaries": deployable_shared_summaries,
        "best_recipe": best_recipe,
        "best_shared_recipe": best_shared_recipe,
        "best_deployable_shared_recipe": best_deployable_shared_recipe,
        "deployable_budget": {
            "max_idle_fp_per_min": float(DEFAULT_CLASSIFIER_IDLE_FP_BUDGET_PER_MIN),
            "min_control_recall": float(DEFAULT_CLASSIFIER_THRESHOLD_MIN_CONTROL_RECALL),
            "min_control_recall_at_2.5s": float(DEFAULT_DEPLOYABLE_MIN_CONTROL_RECALL_AT_2P5S),
            "max_detection_latency_s": float(DEFAULT_DEPLOYABLE_MAX_DETECTION_LATENCY_SEC),
            "require_full_subject_coverage": True,
        },
        "coverage_report_path": str(coverage_report_path),
        "failed_cases_path": str(failed_cases_path),
        "failed_case_count": int(len(case_tracker.failed_cases)),
        "hard_failed_case_count": int(len(case_tracker.failed_cases_only())),
        "skipped_case_count": int(len(case_tracker.skipped_cases())),
        "coverage_report": coverage_report,
        "idle_source_policy": idle_source_policies,
        "idle_proxy_note": "Idle/no-control is proxied with non-command target stimulus trials from external benchmarks.",
    }
    deployable_candidate_profile = _deployable_candidate_profile_payload(
        run_id=run_id,
        best_deployable_shared_recipe=best_deployable_shared_recipe,
        rows=rows,
        channel_compatibility=channel_compatibility,
        artifact_paths=artifact_paths,
    )
    summary["deployable_candidate_profile"] = deployable_candidate_profile
    summary["deployable_candidate_profile_path"] = str(report_root / DEPLOYABLE_CANDIDATE_PROFILE_FILENAME)
    _write_csv(
        report_root / "subject_breakdown.csv",
        subject_breakdown_rows,
        (
            "method",
            "recipe_id",
            "gate_variant",
            "dataset",
            "subject",
            "split_count",
            "idle_fp_per_min",
            "ns1_fp_per_min",
            "ns2_fp_per_min",
            "ns3_fp_per_min",
            "control_recall",
            "control_recall_at_2.5s",
            "detection_latency_s",
            "async_macro_f1_5state",
        ),
    )
    _write_csv(
        report_root / "subtype_breakdown.csv",
        subtype_breakdown_rows,
        (
            "method",
            "recipe_id",
            "gate_variant",
            "coverage_subject_count",
            "split_count",
            "overall_idle_fp_per_min",
            "ns1_fp_per_min",
            "ns2_fp_per_min",
            "ns3_fp_per_min",
            "ns_all_fp_per_min",
            "control_recall",
            "control_recall_at_2.5s",
            "detection_latency_s",
        ),
    )
    _write_csv(
        report_root / "ns2_by_selected_freq.csv",
        ns2_by_selected_freq_rows,
        (
            "dataset",
            "subject",
            "split_index",
            "method",
            "recipe_id",
            "gate_variant",
            "frequency_profile",
            "frequency_set_id",
            "freq",
            "NS2_total_windows",
            "NS2_FP_windows",
            "NS2_FP_events",
            "NS2_FP_per_min",
            "FP_share",
            "mean_top1_score",
            "mean_margin",
            "mean_ratio",
            "mean_entropy",
            "mean_lrt_evidence",
            "calibration_blocks",
            "holdout_blocks",
            "win_sec",
            "step_sec",
            "min_enter_windows",
            "smoothing_windows",
        ),
    )
    _write_csv(
        report_root / "ns2_by_subject_freq.csv",
        ns2_by_subject_freq_rows,
        (
            "dataset",
            "subject",
            "split_index",
            "method",
            "recipe_id",
            "gate_variant",
            "frequency_profile",
            "frequency_set_id",
            "freq",
            "NS2_total_windows",
            "NS2_FP_windows",
            "NS2_FP_events",
            "NS2_FP_per_min",
            "command_recall_for_freq",
            "recall_at_2.5_for_freq",
            "detection_latency_for_freq",
            "calibration_blocks",
            "holdout_blocks",
            "win_sec",
            "step_sec",
            "min_enter_windows",
            "smoothing_windows",
        ),
    )
    _write_csv(
        report_root / "selected_freq_confusion.csv",
        selected_freq_confusion_rows,
        (
            "dataset",
            "subject",
            "split_index",
            "method",
            "recipe_id",
            "gate_variant",
            "frequency_profile",
            "frequency_set_id",
            "true_state",
            "selected_freq",
            "unit",
            "count",
            "calibration_blocks",
            "holdout_blocks",
            "win_sec",
            "step_sec",
            "min_enter_windows",
            "smoothing_windows",
        ),
    )
    _write_csv(
        report_root / "per_frequency_metrics.csv",
        per_frequency_metric_rows,
        (
            "dataset",
            "subject",
            "split_index",
            "method",
            "recipe_id",
            "gate_variant",
            "frequency_profile",
            "frequency_set_id",
            "freq",
            "command_recall",
            "recall_at_2.5s",
            "fp_per_min_ns1",
            "fp_per_min_ns2",
            "fp_per_min_ns3",
            "gate_pass_rate",
            "mean_latency",
            "calibration_blocks",
            "win_sec",
            "step_sec",
            "min_enter_windows",
            "smoothing_windows",
        ),
    )
    _write_csv(
        report_root / "candidate_comparison.csv",
        candidate_comparison_rows,
        (
            "method",
            "recipe_id",
            "gate_variant",
            "combo_name",
            "nc_calibration_simulation",
            "nc_seconds",
            "nc_source",
            "nc_gate_type",
            "gate_type",
            "coverage_subject_count",
            "expected_subject_count",
            "split_count",
            "idle_fp_per_min",
            "delta_idle_fp_per_min",
            "ns1_fp_per_min",
            "ns2_fp_per_min",
            "delta_ns2_fp_per_min",
            "ns3_fp_per_min",
            "control_recall",
            "delta_control_recall",
            "control_recall_at_2.5s",
            "delta_control_recall_at_2.5s",
            "detection_latency_s",
            "delta_detection_latency_s",
            "deployable",
            "ns2_safe",
            "ns2_status",
            "subject_risk",
            "gate_validation_supported",
            "gate_validation_idle_fp_per_min",
            "gate_validation_ns2_fp_per_min",
            "gate_validation_control_recall",
            "gate_validation_control_recall_at_2.5s",
            "frequency_specific_grid_selection_policy",
            "high_risk_ns2_fp_per_min",
            "high_risk_delta_ns2_fp_per_min",
            "high_risk_ns2_reduction_ratio",
            "low_recall_recall_at_2.5s",
            "low_recall_delta_recall_at_2.5s",
            "control_recall_delta_for_eligibility",
            "ns2_improved",
            "recall_degraded",
            "full24_candidate_eligible",
            "freq_10p5_command_recall",
            "freq_10p5_delta_command_recall",
            "freq_10p5_recall_at_2.5s",
            "freq_10p5_delta_recall_at_2.5s",
            "freq_10p5_ns2_fp_per_min",
            "freq_10p5_delta_ns2_fp_per_min",
            "freq_8_command_recall",
            "freq_8_delta_command_recall",
            "freq_8_recall_at_2.5s",
            "freq_8_delta_recall_at_2.5s",
            "freq_8_ns2_fp_per_min",
            "freq_8_delta_ns2_fp_per_min",
            "fixed_ns2_fp_count",
            "lost_command_tp_count",
            "veto_precision",
            "tp_loss_per_fixed_fp",
            "tenp5_ns2_fp_delta",
            "tenp5_command_recall_delta",
            "overall_recall_at_2.5_delta",
            "S11_recall_delta",
            "S19_recall_delta",
            "S24_recall_delta",
            "tenp5_full24_entry_eligible",
            "recommended_profile_export",
        ),
    )
    _write_csv(
        report_root / "nc_calibration_budget_curve.csv",
        nc_budget_curve_rows,
        (
            "nc_seconds",
            "nc_source",
            "gate_type",
            "idle_fp_per_min",
            "NS1_fp_per_min",
            "NS2_fp_per_min",
            "NS3_fp_per_min",
            "control_recall",
            "control_recall_at_2.5s",
            "detection_latency_s",
            "S11_recall",
            "S19_recall",
            "S24_NS2_fp",
            "tp_loss_per_fixed_fp",
            "deployable",
            "ns2_safe",
        ),
    )
    _write_csv(
        report_root / "csns_feature_summary.csv",
        csns_feature_summary_rows,
        (
            "dataset",
            "subject",
            "split_index",
            "method",
            "recipe_id",
            "nc_seconds",
            "nc_source",
            "nc_gate_type",
            "feature",
            "positive_count",
            "negative_count",
            "mean_positive",
            "mean_negative",
            "median_positive",
            "median_negative",
            "delta_mean",
            "auc_positive_greater",
            "auc_best_direction",
        ),
    )
    _write_csv(
        report_root / "tenp5_ns2_veto_diagnostics.csv",
        tenp5_veto_diagnostic_rows,
        (
            "dataset",
            "subject",
            "split_id",
            "split_index",
            "trial_id",
            "block_index",
            "window_idx",
            "time_from_onset",
            "true_state",
            "true_freq",
            "selected_freq",
            "baseline_10p5_pass",
            "candidate_10p5_pass",
            "veto_probability",
            "veto_threshold",
            "vetoed",
            "fixed_ns2_fp",
            "lost_command_tp",
            "top1_score",
            "top2_score",
            "top3_score",
            "selected_freq_score",
            "margin",
            "ratio",
            "score_entropy",
            "lrt_evidence",
            "multiwindow_same_freq_count",
            "multiwindow_margin_mean",
            "method",
            "recipe_id",
            "gate_variant",
            "frequency_profile",
            "frequency_set_id",
            "calibration_blocks",
            "holdout_blocks",
            "win_sec",
            "step_sec",
            "min_enter_windows",
            "smoothing_windows",
        ),
    )
    _write_json(report_root / "tenp5_ns2_veto_summary.json", tenp5_veto_summary_payload)
    _write_csv(
        report_root / "logistic_trace_windows.csv",
        logistic_trace_window_rows,
        (
            "dataset",
            "subject",
            "split_id",
            "split_index",
            "trial_id",
            "block_index",
            "window_idx",
            "time_from_onset",
            "true_state",
            "true_freq",
            "selected_freq",
            "baseline_pred",
            "candidate_pred",
            "top1_score",
            "top2_score",
            "top3_score",
            "selected_freq_score",
            "margin",
            "ratio",
            "normalized_top1",
            "score_entropy",
            "lrt_evidence",
            "multiwindow_same_freq_count",
            "multiwindow_margin_mean",
            "multiwindow_entropy_mean",
            "cs_probability",
            "gate_pass",
            "transition_type",
            "method",
            "recipe_id",
            "gate_variant",
            "frequency_profile",
            "frequency_set_id",
            "calibration_blocks",
            "holdout_blocks",
            "win_sec",
            "step_sec",
            "min_enter_windows",
            "smoothing_windows",
        ),
    )
    _write_csv(
        report_root / "logistic_trace_trial_summary.csv",
        logistic_trace_trial_rows,
        (
            "dataset",
            "subject",
            "split_id",
            "split_index",
            "trial_id",
            "block_index",
            "true_state",
            "true_freq",
            "baseline_pred",
            "candidate_pred",
            "baseline_decision_time_s",
            "candidate_decision_time_s",
            "transition_type",
            "method",
            "recipe_id",
            "gate_variant",
            "frequency_profile",
            "frequency_set_id",
            "calibration_blocks",
            "holdout_blocks",
            "win_sec",
            "step_sec",
            "min_enter_windows",
            "smoothing_windows",
        ),
    )
    _write_csv(
        report_root / "logistic_transition_counts_by_subject.csv",
        logistic_transition_subject_rows,
        (
            "dataset",
            "subject",
            "split_index",
            "transition_type",
            "count",
            "method",
            "recipe_id",
            "gate_variant",
            "frequency_profile",
            "frequency_set_id",
            "calibration_blocks",
            "holdout_blocks",
            "win_sec",
            "step_sec",
            "min_enter_windows",
            "smoothing_windows",
        ),
    )
    _write_csv(
        report_root / "logistic_transition_counts_by_frequency.csv",
        logistic_transition_frequency_rows,
        (
            "dataset",
            "subject",
            "split_index",
            "frequency_or_state",
            "transition_type",
            "count",
            "method",
            "recipe_id",
            "gate_variant",
            "frequency_profile",
            "frequency_set_id",
            "calibration_blocks",
            "holdout_blocks",
            "win_sec",
            "step_sec",
            "min_enter_windows",
            "smoothing_windows",
        ),
    )
    _write_csv(
        report_root / "logistic_feature_summary_tp_fp.csv",
        logistic_feature_summary_rows,
        (
            "dataset",
            "subject",
            "split_index",
            "transition_type",
            "selected_freq",
            "count",
            "mean_selected_freq_score",
            "mean_top1_score",
            "mean_top2_score",
            "mean_top3_score",
            "mean_margin",
            "mean_ratio",
            "mean_normalized_top1",
            "mean_score_entropy",
            "mean_lrt_evidence",
            "mean_multiwindow_same_freq_count",
            "mean_multiwindow_margin_mean",
            "mean_multiwindow_entropy_mean",
            "mean_cs_probability",
            "gate_pass_rate",
            "method",
            "recipe_id",
            "gate_variant",
            "frequency_profile",
            "frequency_set_id",
            "calibration_blocks",
            "holdout_blocks",
            "win_sec",
            "step_sec",
            "min_enter_windows",
            "smoothing_windows",
        ),
    )
    _write_json(report_root / "trace_separability_summary.json", dict(trace_separability.get("summary", {}) or {}))
    (report_root / "trace_separability_summary.md").write_text(
        _render_trace_separability_markdown(trace_separability),
        encoding="utf-8",
    )
    _write_csv(
        report_root / "transition_by_subject_freq.csv",
        list(trace_separability.get("transition_by_subject_freq", []) or []),
        ("subject", "selected_freq", "transition_type", "count"),
    )
    _write_csv(
        report_root / "feature_separability_by_subject_freq.csv",
        list(trace_separability.get("feature_separability_by_subject_freq", []) or []),
        (
            "scope",
            "subject",
            "selected_freq",
            "feature",
            "positive_count",
            "negative_count",
            "mean_positive",
            "mean_negative",
            "median_positive",
            "median_negative",
            "delta_mean",
            "effect_size",
            "auc_positive_greater",
            "auc_best_direction",
            "overlap_rate",
        ),
    )
    _write_csv(
        report_root / "risk_rule_candidates.csv",
        list(trace_separability.get("risk_rule_candidates", []) or []),
        (
            "scope",
            "subject",
            "selected_freq",
            "feature",
            "auc_best_direction",
            "effect_size_abs",
            "positive_count",
            "negative_count",
            "candidate_rule",
        ),
    )
    _write_json(report_root / "gate_params.json", gate_params_payload)
    _write_json(report_root / "gate_params_by_frequency.json", gate_params_by_frequency_payload)
    _write_json(failed_cases_path, failure_payload)
    _write_json(coverage_report_path, coverage_report)
    _write_json(report_root / DEPLOYABLE_CANDIDATE_PROFILE_FILENAME, deployable_candidate_profile)
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
            score_bank_mode=score_bank_mode,
            frequency_search_plan=freq_plan,
            idle_eval_mode=idle_eval_mode,
            budget=budget,
            weak_subject_audit=weak_audit,
            metric_definitions=metric_definitions,
        ),
        encoding="utf-8",
    )
    _write_json(partial_summary_path, {**summary, "status": "ok"})
    logs_dir = report_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        (logs_dir / f"{run_id}.log").write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
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
    parser.add_argument("--ysu-an-raw-dir", type=Path, default=None)
    parser.add_argument("--ysu-an-channel-loc", type=Path, default=None)
    parser.add_argument("--subject-limit-per-dataset", type=int, default=0)
    parser.add_argument("--subject-whitelist", type=str, default="")
    parser.add_argument("--calibration-blocks", type=str, default="1,2,3")
    parser.add_argument("--idle-multipliers", type=str, default="1.0,2.0")
    parser.add_argument("--max-splits-per-subject", type=int, default=DEFAULT_MAX_SPLITS_PER_SUBJECT)
    parser.add_argument("--step-sec", type=float, default=DEFAULT_STEP_SEC)
    parser.add_argument("--decision-start-sec", type=float, default=DEFAULT_DECISION_START_SEC)
    parser.add_argument("--decision-deadline-sec", type=float, default=DEFAULT_DECISION_DEADLINE_SEC)
    parser.add_argument("--min-release-windows", type=int, default=DEFAULT_MIN_RELEASE_WINDOWS)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--case-limit", type=int, default=DEFAULT_CASE_LIMIT)
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
        "--classifier-smoothing-windows-candidates",
        type=str,
        default=",".join(str(item) for item in DEFAULT_CLASSIFIER_SMOOTHING_WINDOWS_CANDIDATES),
    )
    parser.add_argument(
        "--classifier-gate-variants",
        type=str,
        default=",".join(DEFAULT_CLASSIFIER_GATE_VARIANTS),
        help="Comma-separated fbcca_ridge5 reject-gate variants to evaluate.",
    )
    parser.add_argument(
        "--ns2-safety-factors",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_NS2_SAFETY_FACTORS),
        help="Comma-separated NS2 safety factors for ns2_aware_gate and subject_floor_ns2_aware_gate.",
    )
    parser.add_argument(
        "--subject-floor-global-quantiles",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_GLOBAL_FLOOR_QUANTILES),
        help="Comma-separated pooled idle evidence quantiles for subject_threshold_floor variants.",
    )
    parser.add_argument(
        "--subject-floor-idle-quantiles",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_SUBJECT_IDLE_QUANTILES),
        help="Comma-separated per-subject idle evidence quantiles for subject_threshold_floor variants.",
    )
    parser.add_argument(
        "--freqspec-margin-idle-quantiles",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_FREQSPEC_MARGIN_IDLE_QUANTILES),
        help="Comma-separated margin idle quantiles for frequency-specific threshold gate.",
    )
    parser.add_argument(
        "--freqspec-threshold-combo-set",
        type=str,
        default=FREQSPEC_THRESHOLD_COMBO_SET_NONE,
        choices=FREQSPEC_THRESHOLD_COMBO_SETS,
        help="Named exact parameter set for frequency-specific threshold gate, e.g. priority6.",
    )
    parser.add_argument(
        "--freqspec-ratio-idle-quantiles",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_FREQSPEC_RATIO_IDLE_QUANTILES),
        help="Comma-separated ratio idle quantiles for frequency-specific threshold gate.",
    )
    parser.add_argument(
        "--freqspec-entropy-control-quantiles",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_FREQSPEC_ENTROPY_CONTROL_QUANTILES),
        help="Comma-separated control entropy quantiles for frequency-specific threshold gate.",
    )
    parser.add_argument(
        "--freqspec-ns2-safety-factors",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_NS2_SAFETY_FACTORS),
        help="Comma-separated NS2 safety factors for frequency-specific threshold gate.",
    )
    parser.add_argument(
        "--freqspec-logistic-prob-thresholds",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_FREQSPEC_LOGISTIC_PROB_THRESHOLDS),
        help="Comma-separated probability thresholds for frequency-specific logistic gate.",
    )
    parser.add_argument(
        "--freqspec-logistic-ns2-weights",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_FREQSPEC_LOGISTIC_NS2_WEIGHTS),
        help="Comma-separated NS2 sample weights for frequency-specific logistic gate.",
    )
    parser.add_argument(
        "--tenp5-veto-thresholds",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_TENP5_VETO_THRESHOLDS),
        help="Comma-separated P(NS2 hard-negative) veto thresholds for tenp5_ns2_hard_negative_veto.",
    )
    parser.add_argument(
        "--tenp5-ns2-weights",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_TENP5_NS2_WEIGHTS),
        help="Comma-separated NS2 class weights for tenp5_ns2_hard_negative_veto.",
    )
    parser.add_argument(
        "--enable-nc-calibration-simulation",
        action="store_true",
        help="Run YSU-an session-specific no-control calibration simulation rows; benchmark-only, no runtime export.",
    )
    parser.add_argument(
        "--nc-calibration-seconds",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_NC_CALIBRATION_SECONDS),
        help="Comma-separated extra no-control calibration budgets in seconds.",
    )
    parser.add_argument(
        "--nc-calibration-sources",
        type=str,
        default=",".join(NC_CALIBRATION_SOURCES),
        help="Comma-separated no-control sources: ns1,ns2,ns3,mixed,ns2_heavy.",
    )
    parser.add_argument(
        "--nc-calibration-gate-types",
        type=str,
        default=",".join(NC_CALIBRATION_GATE_TYPES),
        help="Comma-separated NC gate schemes: baseline_lrt_with_nc_calibrated_threshold, session_specific_logistic_csns_detector, conditional_baseline_plus_session_csns_detector.",
    )
    parser.add_argument(
        "--classifier-threshold-policy",
        type=str,
        default=DEFAULT_CLASSIFIER_THRESHOLD_POLICY,
        choices=CLASSIFIER_THRESHOLD_POLICIES,
    )
    parser.add_argument("--score-bank-mode", type=str, default=DEFAULT_SCORE_BANK_MODE, choices=SCORE_BANK_MODES)
    parser.add_argument("--freq-search-mode", type=str, default=DEFAULT_FREQ_SEARCH_MODE, choices=FREQ_SEARCH_MODES)
    parser.add_argument(
        "--freq-candidate-source",
        type=str,
        default=DEFAULT_FREQ_CANDIDATE_SOURCE,
        choices=FREQ_CANDIDATE_SOURCES,
    )
    parser.add_argument("--idle-eval-mode", type=str, default=DEFAULT_IDLE_EVAL_MODE, choices=IDLE_EVAL_MODES)
    parser.add_argument("--pretrain-budget-sec", type=float, default=DEFAULT_PRETRAIN_BUDGET_SEC)
    parser.add_argument(
        "--personalized-candidate-count",
        type=str,
        default=",".join(str(item) for item in DEFAULT_PERSONALIZED_CANDIDATE_COUNT),
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
