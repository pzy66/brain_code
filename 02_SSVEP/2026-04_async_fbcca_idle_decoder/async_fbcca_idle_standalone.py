from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import scipy.linalg
from scipy.signal import sosfiltfilt

from ssvep_core.compute_backend import (
    DEFAULT_COMPUTE_BACKEND,
    DEFAULT_GPU_CACHE_POLICY,
    DEFAULT_GPU_DEVICE,
    DEFAULT_GPU_PRECISION,
    parse_compute_backend_name,
    parse_gpu_cache_policy,
    parse_gpu_precision,
    resolve_compute_backend,
)
from ssvep_core.compute_kernels import (
    benchmark_fbcca_batch_path,
    build_reference_tensor,
    cca_scores_batch,
    design_sos_bandpass,
    design_sos_lowpass,
    design_sos_notch,
    fbcca_subband_scores_batch,
    fbcca_scores_batch,
    preprocess_windows_batch,
)

try:
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
except Exception:
    BoardIds = None
    BoardShim = None
    BrainFlowInputParams = None

try:
    from serial.tools import list_ports as serial_list_ports
except Exception:
    serial_list_ports = None


DEFAULT_FREQS = (8.0, 10.0, 12.0, 15.0)
DEFAULT_WIN_SEC = 3.0
DEFAULT_STEP_SEC = 0.25
DEFAULT_SUBBANDS = ((6.0, 50.0), (10.0, 50.0), (14.0, 50.0), (18.0, 50.0), (22.0, 50.0))
DEFAULT_NH = 3
DEFAULT_BOARD_ID = BoardIds.CYTON_BOARD.value if BoardIds is not None else 0
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent / "profiles" / "default_profile.json"
MODEL_FEATURE_NAMES = ("top1_score", "ratio", "margin", "normalized_top1", "score_entropy")
DEFAULT_STREAM_WARMUP_SEC = 1.0
DEFAULT_STREAM_READY_SEC = 0.5
DEFAULT_SAMPLE_GRACE_SEC = 1.0
DEFAULT_MAX_TRANSIENT_READ_ERRORS = 5
DEFAULT_MAX_CALIBRATION_TRIAL_ERRORS = 3
DEFAULT_CALIBRATION_SEED = 20260407
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_AUTO_CHANNEL_COUNT = 4
DEFAULT_WIN_SEC_CANDIDATES = (1.5, 2.0, 2.5, 3.0)
DEFAULT_MIN_ENTER_CANDIDATES = (1, 2)
DEFAULT_MIN_EXIT_CANDIDATES = (1, 2, 3)
DEFAULT_SPEED_WIN_SEC_CANDIDATES = (1.0, 1.25, 1.5, 2.0, 2.5)
DEFAULT_SPEED_MIN_ENTER_CANDIDATES = (1, 2)
DEFAULT_SPEED_MIN_EXIT_CANDIDATES = (1, 2, 3)
DEFAULT_MODEL_NAME = "fbcca"
DEFAULT_SERIAL_PORT = "auto"
DEFAULT_GATE_POLICY = "balanced"
DEFAULT_GATE_POLICIES = ("conservative", "balanced", "speed")
DEFAULT_CHANNEL_WEIGHT_MODE = "fbcca_diag"
DEFAULT_CHANNEL_WEIGHT_RANGE = (0.5, 1.8)
DEFAULT_SUBBAND_WEIGHT_MODE = "chen_fixed"
DEFAULT_SUBBAND_WEIGHT_MODES = ("chen_fixed", "chen_ab_subject", "simplex_subject")
DEFAULT_SUBBAND_WEIGHT_RANGE = (0.02, 1.0)
DEFAULT_WEIGHT_AGGREGATION = "median"
DEFAULT_WEIGHT_AGGREGATION_MODES = ("median", "mean", "trimmed-mean")
DEFAULT_IDLE_FP_HARD_TH = 1.5
DEFAULT_CHANNEL_WEIGHT_L2 = 0.03
DEFAULT_SUBBAND_PRIOR_STRENGTH = 0.10
DEFAULT_CONTROL_STATE_MODE = "unified"
DEFAULT_CONTROL_STATE_MODES = ("unified", "frequency-specific-threshold", "frequency-specific-logistic")
DEFAULT_FBCCA_SUBBAND_WEIGHT_A = 1.25
DEFAULT_FBCCA_SUBBAND_WEIGHT_B = 0.25
DEFAULT_FBCCA_SUBBAND_WEIGHT_A_GRID = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50)
DEFAULT_FBCCA_SUBBAND_WEIGHT_B_GRID = (0.00, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
DEFAULT_FBCCA_WEIGHT_CV_FOLDS = 3
DEFAULT_SPATIAL_FILTER_MODE = "trca_shared"
DEFAULT_SPATIAL_FILTER_MODES = ("none", "trca_shared")
DEFAULT_SPATIAL_RANK_CANDIDATES = (1, 2, 3)
DEFAULT_JOINT_WEIGHT_ITERS = 2
DEFAULT_SPATIAL_SOURCE_MODEL = "trca"
DEFAULT_SPATIAL_SOURCE_MODELS = ("trca",)
DEFAULT_DYNAMIC_STOP_ALPHA = 0.7
DEFAULT_DYNAMIC_STOP_ENABLED = True
DEFAULT_COMPUTE_BACKEND_NAME = DEFAULT_COMPUTE_BACKEND
DEFAULT_GPU_DEVICE_ID = DEFAULT_GPU_DEVICE
DEFAULT_GPU_PRECISION_NAME = DEFAULT_GPU_PRECISION
DEFAULT_GPU_CACHE_MODE = DEFAULT_GPU_CACHE_POLICY
DEFAULT_BALANCED_IDLE_FP_MAX = 3.0
DEFAULT_BALANCED_MIN_CONTROL_RECALL = 0.60
DEFAULT_BALANCED_OBJECTIVE_IDLE_WEIGHT = 0.45
DEFAULT_BALANCED_OBJECTIVE_RECALL_WEIGHT = 0.35
DEFAULT_BALANCED_OBJECTIVE_SWITCH_WEIGHT = 0.10
DEFAULT_BALANCED_OBJECTIVE_RELEASE_WEIGHT = 0.10
DEFAULT_SPEED_IDLE_FP_MAX = 1.5
DEFAULT_SPEED_MIN_CONTROL_RECALL = 0.65
DEFAULT_SPEED_OBJECTIVE_IDLE_WEIGHT = 0.35
DEFAULT_SPEED_OBJECTIVE_RECALL_WEIGHT = 0.25
DEFAULT_SPEED_OBJECTIVE_SWITCH_WEIGHT = 0.25
DEFAULT_SPEED_OBJECTIVE_RELEASE_WEIGHT = 0.15
DEFAULT_CONTROL_RECALL_AT_2S_DEADLINE = 2.0
DEFAULT_CONTROL_RECALL_AT_3S_DEADLINE = 3.0
DEFAULT_SWITCH_DETECT_AT_2P8S_DEADLINE = 2.8
DEFAULT_ACCEPTANCE_IDLE_FP_PER_MIN = 1.5
DEFAULT_ACCEPTANCE_CONTROL_RECALL = 0.75
DEFAULT_ACCEPTANCE_SWITCH_LATENCY_S = 2.8
DEFAULT_ACCEPTANCE_RELEASE_LATENCY_S = 1.5
DEFAULT_ACCEPTANCE_INFERENCE_MS = 40.0
DEFAULT_BENCHMARK_CAL_TARGET_REPEATS = 24
DEFAULT_BENCHMARK_CAL_IDLE_REPEATS = 48
DEFAULT_BENCHMARK_EVAL_TARGET_REPEATS = 24
DEFAULT_BENCHMARK_EVAL_IDLE_REPEATS = 48
DEFAULT_BENCHMARK_MODELS = (
    "cca",
    "itcca",
    "ecca",
    "msetcca",
    "fbcca",
    "legacy_fbcca_202603",
    "trca",
    "trca_r",
    "sscor",
    "tdca",
    "oacca",
)
DEFAULT_BENCHMARK_SWITCH_TRIALS = 32
DEFAULT_BENCHMARK_CHANNEL_MODES = ("auto", "all8")
DEFAULT_BENCHMARK_MULTI_SEED_COUNT = 5
DEFAULT_BENCHMARK_SEED_STEP = 1
DEFAULT_DECODER_UPDATE_ALPHA = 0.08
DEFAULT_SSCOR_COMPONENTS = 3
DEFAULT_TDCA_COMPONENTS = 3
DEFAULT_BENCHMARK_DATASET_ROOT = Path(__file__).resolve().parent / "profiles" / "datasets"
DATA_SCHEMA_VERSION = "1.0"
SWITCH_LATENCY_MODE = "penalized_median"
DEFAULT_BENCHMARK_RANK_MIN_CONTROL_RECALL = DEFAULT_BALANCED_MIN_CONTROL_RECALL
DEFAULT_CALIBRATION_MIN_DETECTION_RECALL = 0.05
DEFAULT_METRIC_SCOPE = "dual"
DEFAULT_METRIC_SCOPES = ("dual", "4class", "5class")
DEFAULT_DECISION_TIME_MODE = "first-correct"
DEFAULT_DECISION_TIME_MODES = ("first-correct", "first-any", "fixed-window")
DEFAULT_PAPER_DECISION_TIME_MODE = "fixed-window"
DEFAULT_ASYNC_DECISION_TIME_MODE = "first-correct"
DEFAULT_DATA_POLICY = "new-only"
DEFAULT_DATA_POLICIES = ("new-only", "legacy-compatible")
DEFAULT_RANKING_POLICY = "async-first"
DEFAULT_RANKING_POLICIES = ("async-first", "paper-first", "dual-board", "async-speed")
DEFAULT_EXPORT_FIGURES = True
MODEL_IMPLEMENTATION_LEVELS = {
    "cca": "paper-faithful",
    "fbcca": "paper-faithful",
    "legacy_fbcca_202603": "paper-faithful",
    "fbcca_fixed_all8": "paper-faithful",
    "fbcca_cw_all8": "engineering-approx",
    "fbcca_sw_all8": "engineering-approx",
    "fbcca_cw_sw_all8": "engineering-approx",
    "fbcca_cw_sw_trca_shared": "engineering-approx",
    "itcca": "engineering-approx",
    "ecca": "engineering-approx",
    "msetcca": "engineering-approx",
    "trca": "paper-faithful",
    "trca_r": "engineering-approx",
    "sscor": "engineering-approx",
    "tdca": "engineering-approx",
    "oacca": "engineering-approx",
}
MODEL_METHOD_NOTES = {
    "cca": "Reference-signal CCA baseline consistent with the canonical SSVEP formulation.",
    "fbcca": "Filterbank CCA with harmonic fusion follows the FBCCA baseline formulation.",
    "legacy_fbcca_202603": (
        "Legacy FBCCA scorer compatible with the 2026-03 implementation "
        "(fixed FBCCA score path without channel/spatial weighting frontend)."
    ),
    "fbcca_fixed_all8": "Pure FBCCA on all EEG channels with fixed Chen-style subband weights.",
    "fbcca_cw_all8": "Pure FBCCA on all EEG channels with learned diagonal channel weights and fixed subband weights.",
    "fbcca_sw_all8": (
        "Pure FBCCA on all EEG channels with learned subject-specific global subband fusion weights "
        "shared by all channels."
    ),
    "fbcca_cw_sw_all8": (
        "Pure FBCCA on all EEG channels with separable learned weights: 8 channel weights plus "
        "5 global subband fusion weights, not a full channel-by-subband matrix."
    ),
    "fbcca_cw_sw_trca_shared": "FBCCA with jointly learned channel/subband weights plus TRCA-shared spatial frontend.",
    "itcca": "Template-assisted CCA approximation for personalized reference matching.",
    "ecca": "Extended CCA approximation using template + reference correlation fusion.",
    "msetcca": "Template-based multiset CCA approximation with simplified fusion weights.",
    "trca": "TRCA spatial-filter implementation with class-wise templates.",
    "trca_r": "Filterbank ensemble TRCA approximation of TRCA-R / eTRCA family.",
    "sscor": "SSCOR-inspired multi-component correlation scoring built on TRCA-style filters.",
    "tdca": "TDCA-inspired delay-embedding + discriminant projection approximation.",
    "oacca": "Online adaptive CCA approximation with selected-class template updates.",
}
FBCCA_VARIANT_SPECS = {
    "fbcca_fixed_all8": {
        "channel_weight_mode": None,
        "subband_weight_mode": "chen_fixed",
        "spatial_filter_mode": None,
    },
    "fbcca_cw_all8": {
        "channel_weight_mode": "fbcca_diag",
        "subband_weight_mode": "chen_fixed",
        "spatial_filter_mode": None,
    },
    "fbcca_sw_all8": {
        "channel_weight_mode": None,
        "subband_weight_mode": "chen_ab_subject",
        "spatial_filter_mode": None,
    },
    "fbcca_cw_sw_all8": {
        "channel_weight_mode": "fbcca_diag",
        "subband_weight_mode": "chen_ab_subject",
        "spatial_filter_mode": None,
    },
    "fbcca_cw_sw_trca_shared": {
        "channel_weight_mode": "fbcca_diag",
        "subband_weight_mode": "chen_ab_subject",
        "spatial_filter_mode": "trca_shared",
    },
}


def parse_freqs(raw: str) -> tuple[float, float, float, float]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if len(values) != 4:
        raise ValueError("freqs must contain exactly four comma-separated values")
    return values


def require_brainflow() -> None:
    if BoardShim is None or BrainFlowInputParams is None:
        raise RuntimeError("BrainFlow is required for calibrate/online mode.")


def normalize_serial_port(serial_port: Optional[str]) -> str:
    value = str(serial_port).strip() if serial_port is not None else ""
    return value if value else DEFAULT_SERIAL_PORT


def serial_port_is_auto(serial_port: Optional[str]) -> bool:
    normalized = normalize_serial_port(serial_port).lower()
    return normalized in {"auto", "detect", "auto-detect", "autodetect", "*"}


def _serial_port_sort_key(device: str) -> tuple[int, Any]:
    match = re.match(r"^COM(\d+)$", str(device).upper())
    if match:
        return (0, int(match.group(1)))
    return (1, str(device).upper())


def list_serial_port_candidates() -> list[str]:
    candidates: list[tuple[int, tuple[int, Any], str]] = []
    usb_keywords = (
        "usb",
        "serial",
        "uart",
        "cp210",
        "ch340",
        "ftdi",
        "silabs",
        "arduino",
        "openbci",
        "dongle",
        "brain",
    )
    if serial_list_ports is not None:
        try:
            for info in serial_list_ports.comports():
                device = str(getattr(info, "device", "")).strip()
                if not device:
                    continue
                description = str(getattr(info, "description", "")).lower()
                hwid = str(getattr(info, "hwid", "")).lower()
                score = 0
                if any(keyword in description or keyword in hwid for keyword in usb_keywords):
                    score += 8
                match = re.match(r"^COM(\d+)$", device.upper())
                if match:
                    idx = int(match.group(1))
                    if idx >= 3:
                        score += 3
                    if idx == 4:
                        score += 2
                candidates.append((-score, _serial_port_sort_key(device), device))
        except Exception:
            candidates = []
    if not candidates and sys.platform.startswith("win"):
        return ["COM4", "COM3", "COM5", "COM6", "COM7", "COM8", "COM9", "COM10"]
    candidates.sort()
    ordered: list[str] = []
    seen: set[str] = set()
    for _, _, device in candidates:
        upper = str(device).upper()
        if upper in seen:
            continue
        seen.add(upper)
        ordered.append(str(device))
    return ordered


def prepare_board_session(board_id: int, serial_port: Optional[str]) -> tuple[Any, str, list[str]]:
    require_brainflow()
    requested = normalize_serial_port(serial_port)
    candidates = [requested] if not serial_port_is_auto(requested) else list_serial_port_candidates()
    if not candidates:
        raise RuntimeError(
            "No serial ports detected in auto mode. Connect device and retry, or pass --serial-port COMx explicitly."
        )
    attempted: list[str] = []
    last_exc: Optional[Exception] = None
    for candidate in candidates:
        params = BrainFlowInputParams()
        params.serial_port = candidate
        board = BoardShim(int(board_id), params)
        try:
            board.prepare_session()
            attempted.append(candidate)
            return board, candidate, attempted
        except Exception as exc:
            attempted.append(candidate)
            last_exc = exc
            try:
                board.release_session()
            except Exception:
                pass
            if not serial_port_is_auto(requested):
                raise RuntimeError(describe_runtime_error(exc, serial_port=candidate)) from exc
    attempted_hint = ", ".join(attempted[:8])
    if len(attempted) > 8:
        attempted_hint = f"{attempted_hint}, ..."
    if last_exc is None:
        raise RuntimeError(f"Failed to open serial port in auto mode. attempted=[{attempted_hint}]")
    raise RuntimeError(
        f"Failed to auto-open serial port. attempted=[{attempted_hint}]. "
        f"Last error: {describe_runtime_error(last_exc, serial_port=attempted[-1] if attempted else None)}"
    ) from last_exc


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def model_implementation_level(model_name: str) -> str:
    name = normalize_model_name(model_name)
    return str(MODEL_IMPLEMENTATION_LEVELS.get(name, "engineering-approx"))


def model_method_note(model_name: str) -> str:
    name = normalize_model_name(model_name)
    return str(MODEL_METHOD_NOTES.get(name, "No method note available."))


def model_method_mapping_payload(model_names: Sequence[str]) -> dict[str, dict[str, str]]:
    payload: dict[str, dict[str, str]] = {}
    for name in model_names:
        normalized = normalize_model_name(name)
        payload[normalized] = {
            "implementation_level": model_implementation_level(normalized),
            "method_note": model_method_note(normalized),
        }
    return payload


def describe_runtime_error(exc: Exception, *, serial_port: Optional[str] = None) -> str:
    raw = str(exc).strip() or exc.__class__.__name__
    lower = raw.lower()
    port_note = f" {serial_port}" if serial_port else ""
    if "unable_to_open_port_error" in lower or "unable to prepare streaming session" in lower:
        return (
            f"Unable to open serial port{port_note}. The port is likely occupied by another process, "
            f"or device connection/permission is invalid. Raw error: {raw}"
        )
    if "not enough buffered samples" in lower or "stream did not stabilize" in lower:
        return (
            f"Sampling stream is not stable yet or buffered samples are insufficient "
            f"(common right after startup or unstable USB/serial link). Raw error: {raw}"
        )
    if "repeated read failures" in lower:
        return f"Repeated read failures: data stream unstable or connection occupied. Raw error: {raw}"
    return raw

def wait_for_buffer_samples(
    board: Any,
    target_samples: int,
    *,
    timeout_sec: float,
    poll_sec: float = 0.02,
) -> int:
    target = max(int(target_samples), 0)
    if target <= 0:
        return 0
    deadline = time.perf_counter() + max(float(timeout_sec), 0.0)
    available = 0
    while True:
        available = int(board.get_board_data_count())
        if available >= target or time.perf_counter() >= deadline:
            return available
        time.sleep(min(float(poll_sec), max(deadline - time.perf_counter(), 0.0)))


def read_recent_eeg_segment(
    board: Any,
    eeg_channels: Sequence[int],
    *,
    target_samples: int,
    minimum_samples: int,
    grace_sec: float = DEFAULT_SAMPLE_GRACE_SEC,
) -> tuple[np.ndarray, int, int]:
    target = max(int(target_samples), 0)
    minimum = max(int(minimum_samples), 0)
    wait_for_buffer_samples(board, target, timeout_sec=grace_sec)
    data = board.get_current_board_data(max(target, minimum))
    available = int(data.shape[1])
    if available < minimum:
        raise RuntimeError(f"not enough buffered samples: {available}/{minimum}")
    used = min(available, target) if target > 0 else available
    segment = np.ascontiguousarray(data[eeg_channels, -used:].T, dtype=np.float64)
    return segment, used, available


def ensure_stream_ready(
    board: Any,
    sampling_rate: int,
    *,
    minimum_sec: float = DEFAULT_STREAM_READY_SEC,
    timeout_sec: float = 3.0,
) -> int:
    target = max(1, int(round(float(sampling_rate) * max(float(minimum_sec), 0.1))))
    available = wait_for_buffer_samples(board, target, timeout_sec=timeout_sec)
    if available < target:
        raise RuntimeError(f"stream did not stabilize: {available}/{target} buffered samples")
    return available


def merge_idle_transition_segments(
    rest_segment: Optional[np.ndarray],
    prepare_segment: Optional[np.ndarray],
    *,
    minimum_samples: int,
) -> Optional[np.ndarray]:
    required = max(1, int(minimum_samples))
    parts: list[np.ndarray] = []
    for candidate in (rest_segment, prepare_segment):
        if candidate is None:
            continue
        matrix = np.asarray(candidate, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] <= 0:
            continue
        parts.append(matrix)
    if not parts:
        return None
    channel_counts = {int(part.shape[1]) for part in parts}
    if len(channel_counts) != 1:
        return None
    if len(parts) == 1:
        merged = np.ascontiguousarray(parts[0], dtype=np.float64)
    else:
        merged = np.ascontiguousarray(np.vstack(parts), dtype=np.float64)
    if int(merged.shape[0]) < required:
        return None
    return merged


@dataclass(frozen=True)
class ThresholdProfile:
    freqs: tuple[float, float, float, float]
    win_sec: float
    step_sec: float
    enter_score_th: float
    enter_ratio_th: float
    enter_margin_th: float
    exit_score_th: float
    exit_ratio_th: float
    min_enter_windows: int
    min_exit_windows: int
    model_name: str = DEFAULT_MODEL_NAME
    model_params: Optional[dict[str, Any]] = None
    calibration_split_seed: Optional[int] = None
    benchmark_metrics: Optional[dict[str, float]] = None
    enter_log_lr_th: Optional[float] = None
    exit_log_lr_th: Optional[float] = None
    control_feature_means: Optional[dict[str, float]] = None
    control_feature_stds: Optional[dict[str, float]] = None
    idle_feature_means: Optional[dict[str, float]] = None
    idle_feature_stds: Optional[dict[str, float]] = None
    eeg_channels: Optional[tuple[int, ...]] = None
    gate_policy: str = DEFAULT_GATE_POLICY
    min_switch_windows: int = 1
    switch_enter_score_th: Optional[float] = None
    switch_enter_ratio_th: Optional[float] = None
    switch_enter_margin_th: Optional[float] = None
    speed_objective_weights: Optional[dict[str, float]] = None
    dynamic_stop: Optional[dict[str, Any]] = None
    channel_weight_mode: Optional[str] = None
    channel_weights: Optional[tuple[float, ...]] = None
    subband_weight_mode: Optional[str] = None
    subband_weights: Optional[tuple[float, ...]] = None
    subband_weight_params: Optional[dict[str, Any]] = None
    spatial_filter_mode: Optional[str] = None
    spatial_filter_rank: Optional[int] = None
    spatial_filter_state: Optional[dict[str, Any]] = None
    joint_weight_training: Optional[dict[str, Any]] = None
    weight_training_seed_summary: Optional[dict[str, Any]] = None
    control_state_mode: str = DEFAULT_CONTROL_STATE_MODE
    frequency_specific_thresholds: Optional[dict[str, Any]] = None
    profile_validation_status: Optional[dict[str, Any]] = None
    recommended_for_realtime: Optional[bool] = None
    runtime_backend_preference: Optional[str] = None
    runtime_precision_preference: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ThresholdProfile":
        names = {item.name for item in fields(cls)}
        data = {key: payload[key] for key in names if key in payload}
        if "freqs" in data:
            data["freqs"] = tuple(float(value) for value in data["freqs"])
        if "eeg_channels" in data and data["eeg_channels"] is not None:
            data["eeg_channels"] = tuple(int(value) for value in data["eeg_channels"])
        if "gate_policy" in data and data["gate_policy"] is not None:
            data["gate_policy"] = str(data["gate_policy"]).strip().lower()
        if "min_switch_windows" in data and data["min_switch_windows"] is not None:
            data["min_switch_windows"] = max(1, int(data["min_switch_windows"]))
        for name in ("switch_enter_score_th", "switch_enter_ratio_th", "switch_enter_margin_th"):
            if name in data and data[name] is not None:
                data[name] = float(data[name])
        if "speed_objective_weights" in data and data["speed_objective_weights"] is not None:
            raw = dict(data["speed_objective_weights"])
            data["speed_objective_weights"] = {str(key): float(value) for key, value in raw.items()}
        if "channel_weight_mode" in data and data["channel_weight_mode"] is not None:
            data["channel_weight_mode"] = str(data["channel_weight_mode"]).strip().lower()
        if "channel_weights" in data and data["channel_weights"] is not None:
            data["channel_weights"] = tuple(float(value) for value in data["channel_weights"])
        if "subband_weight_mode" in data and data["subband_weight_mode"] is not None:
            data["subband_weight_mode"] = str(data["subband_weight_mode"]).strip().lower()
        if "subband_weights" in data and data["subband_weights"] is not None:
            data["subband_weights"] = tuple(float(value) for value in data["subband_weights"])
        if "subband_weight_params" in data and data["subband_weight_params"] is not None:
            raw = dict(data["subband_weight_params"])
            data["subband_weight_params"] = {
                str(key): float(value) if isinstance(value, (int, float)) else value for key, value in raw.items()
            }
        if "spatial_filter_mode" in data and data["spatial_filter_mode"] is not None:
            data["spatial_filter_mode"] = str(data["spatial_filter_mode"]).strip().lower()
        if "spatial_filter_rank" in data and data["spatial_filter_rank"] is not None:
            data["spatial_filter_rank"] = max(1, int(data["spatial_filter_rank"]))
        if "spatial_filter_state" in data and data["spatial_filter_state"] is not None:
            data["spatial_filter_state"] = dict(data["spatial_filter_state"])
        if "joint_weight_training" in data and data["joint_weight_training"] is not None:
            data["joint_weight_training"] = dict(data["joint_weight_training"])
        if "weight_training_seed_summary" in data and data["weight_training_seed_summary"] is not None:
            data["weight_training_seed_summary"] = dict(data["weight_training_seed_summary"])
        if "control_state_mode" in data and data["control_state_mode"] is not None:
            data["control_state_mode"] = parse_control_state_mode(str(data["control_state_mode"]))
        if "frequency_specific_thresholds" in data and data["frequency_specific_thresholds"] is not None:
            data["frequency_specific_thresholds"] = normalize_frequency_specific_thresholds(
                data["frequency_specific_thresholds"]
            )
        if "profile_validation_status" in data and data["profile_validation_status"] is not None:
            data["profile_validation_status"] = dict(data["profile_validation_status"])
        if "recommended_for_realtime" in data and data["recommended_for_realtime"] is not None:
            data["recommended_for_realtime"] = bool(data["recommended_for_realtime"])
        if "runtime_backend_preference" in data and data["runtime_backend_preference"] is not None:
            data["runtime_backend_preference"] = str(data["runtime_backend_preference"]).strip().lower()
        if "runtime_precision_preference" in data and data["runtime_precision_preference"] is not None:
            data["runtime_precision_preference"] = str(data["runtime_precision_preference"]).strip().lower()
        if "dynamic_stop" in data and data["dynamic_stop"] is not None:
            payload = dict(data["dynamic_stop"])
            if "enabled" in payload:
                payload["enabled"] = bool(payload["enabled"])
            if "alpha" in payload and payload["alpha"] is not None:
                payload["alpha"] = float(payload["alpha"])
            if "enter_acc_th" in payload and payload["enter_acc_th"] is not None:
                payload["enter_acc_th"] = float(payload["enter_acc_th"])
            if "exit_acc_th" in payload and payload["exit_acc_th"] is not None:
                payload["exit_acc_th"] = float(payload["exit_acc_th"])
            data["dynamic_stop"] = payload
        return cls(**data)


def _frequency_key(value: Any) -> Optional[str]:
    try:
        numeric = float(value)
    except Exception:
        text = str(value).strip()
        return text or None
    if not np.isfinite(numeric):
        return None
    return f"{float(numeric):g}"


def normalize_frequency_specific_thresholds(
    payload: Optional[dict[str, Any]],
) -> Optional[dict[str, dict[str, Any]]]:
    if payload is None:
        return None
    normalized: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in dict(payload).items():
        key = _frequency_key(raw_key)
        if key is None or not isinstance(raw_value, dict):
            continue
        item = dict(raw_value)
        coerced: dict[str, Any] = {}
        float_fields = {
            "enter_score_th",
            "enter_ratio_th",
            "enter_margin_th",
            "exit_score_th",
            "exit_ratio_th",
            "switch_enter_score_th",
            "switch_enter_ratio_th",
            "switch_enter_margin_th",
            "enter_log_lr_th",
            "exit_log_lr_th",
            "freq",
        }
        int_fields = {"min_enter_windows", "min_exit_windows", "min_switch_windows"}
        for field_name, field_value in item.items():
            if field_value is None:
                continue
            if field_name in float_fields:
                coerced[str(field_name)] = float(field_value)
            elif field_name in int_fields:
                coerced[str(field_name)] = int(field_value)
            else:
                coerced[str(field_name)] = field_value
        normalized[key] = coerced
    return normalized or None


def default_profile(freqs: Sequence[float] = DEFAULT_FREQS) -> ThresholdProfile:
    freq_tuple = tuple(float(freq) for freq in freqs)
    return ThresholdProfile(
        freqs=(freq_tuple[0], freq_tuple[1], freq_tuple[2], freq_tuple[3]),
        win_sec=DEFAULT_WIN_SEC,
        step_sec=DEFAULT_STEP_SEC,
        enter_score_th=0.0200,
        enter_ratio_th=1.1500,
        enter_margin_th=0.0030,
        exit_score_th=0.0170,
        exit_ratio_th=1.0925,
        min_enter_windows=2,
        min_exit_windows=2,
        model_name=DEFAULT_MODEL_NAME,
        model_params=None,
        calibration_split_seed=None,
        benchmark_metrics=None,
        enter_log_lr_th=None,
        exit_log_lr_th=None,
        control_feature_means=None,
        control_feature_stds=None,
        idle_feature_means=None,
        idle_feature_stds=None,
        eeg_channels=None,
        gate_policy=DEFAULT_GATE_POLICY,
        min_switch_windows=1,
        switch_enter_score_th=None,
        switch_enter_ratio_th=None,
        switch_enter_margin_th=None,
        speed_objective_weights=None,
        dynamic_stop={
            "enabled": bool(DEFAULT_DYNAMIC_STOP_ENABLED),
            "alpha": float(DEFAULT_DYNAMIC_STOP_ALPHA),
            "enter_acc_th": None,
            "exit_acc_th": None,
        },
        channel_weight_mode=None,
        channel_weights=None,
        subband_weight_mode=None,
        subband_weights=None,
        subband_weight_params=None,
        spatial_filter_mode=None,
        spatial_filter_rank=None,
        spatial_filter_state=None,
        joint_weight_training=None,
        metadata={
            "source": "default_fallback",
            "has_stat_model": False,
            "requires_calibration": True,
        },
    )

def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    def _native_path(raw_path: Path) -> str:
        resolved = str(raw_path)
        if os.name != "nt":
            return resolved
        normalized = os.path.abspath(resolved)
        if normalized.startswith("\\\\?\\"):
            return normalized
        # Win32 APIs hit MAX_PATH for long report paths; use extended-length prefix.
        if len(normalized) >= 240:
            if normalized.startswith("\\\\"):
                return "\\\\?\\UNC\\" + normalized[2:]
            return "\\\\?\\" + normalized
        return normalized

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".atomic_{uuid.uuid4().hex[:8]}.tmp"
    native_tmp = _native_path(tmp_path)
    native_target = _native_path(target)
    try:
        with open(native_tmp, "w", encoding=encoding) as handle:
            handle.write(str(text))
        os.replace(native_tmp, native_target)
    except Exception:
        try:
            os.remove(native_tmp)
        except Exception:
            pass
        raise


def atomic_copy_text_file(source: Path, destination: Path, *, encoding: str = "utf-8") -> None:
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    atomic_write_text(dst, src.read_text(encoding=encoding), encoding=encoding)


def save_profile(profile: ThresholdProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(asdict(profile))
    payload["freqs"] = list(profile.freqs)
    if profile.eeg_channels is not None:
        payload["eeg_channels"] = list(profile.eeg_channels)
    if profile.channel_weights is not None:
        payload["channel_weights"] = [float(value) for value in profile.channel_weights]
    if profile.subband_weights is not None:
        payload["subband_weights"] = [float(value) for value in profile.subband_weights]
    payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_text(path, json_dumps(payload) + "\n")


def load_profile(
    path: Path,
    *,
    fallback_freqs: Sequence[float] = DEFAULT_FREQS,
    require_exists: bool = False,
) -> ThresholdProfile:
    if path.exists():
        return ThresholdProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if require_exists:
        raise FileNotFoundError(f"profile not found: {path}")
    return default_profile(fallback_freqs)


def row_feature_vector(row: dict[str, Any]) -> np.ndarray:
    return np.array([float(row[name]) for name in MODEL_FEATURE_NAMES], dtype=float)


def _safe_feature_stats(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.stack([row_feature_vector(row) for row in rows], axis=0)
    means = matrix.mean(axis=0)
    stds = np.maximum(matrix.std(axis=0, ddof=0), 1e-6)
    return means.astype(float), stds.astype(float)


def gaussian_log_likelihood_ratio(
    row: dict[str, Any] | np.ndarray,
    *,
    control_means: np.ndarray,
    control_stds: np.ndarray,
    idle_means: np.ndarray,
    idle_stds: np.ndarray,
) -> float:
    vector = row_feature_vector(row) if isinstance(row, dict) else np.asarray(row, dtype=float)

    def _gaussian_logpdf(x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> float:
        var = np.square(np.maximum(stds, 1e-6))
        return float(-0.5 * np.sum(np.log(2.0 * np.pi * var) + np.square(x - means) / var))

    control_logp = _gaussian_logpdf(vector, control_means, control_stds)
    idle_logp = _gaussian_logpdf(vector, idle_means, idle_stds)
    return float(control_logp - idle_logp)


def profile_has_stat_model(profile: ThresholdProfile) -> bool:
    return all(
        item is not None
        for item in (
            profile.control_feature_means,
            profile.control_feature_stds,
            profile.idle_feature_means,
            profile.idle_feature_stds,
            profile.enter_log_lr_th,
            profile.exit_log_lr_th,
        )
    )


def profile_is_default_fallback(profile: ThresholdProfile) -> bool:
    metadata = profile.metadata or {}
    return str(metadata.get("source", "")) == "default_fallback"


def profile_log_lr(profile: ThresholdProfile, row: dict[str, Any]) -> Optional[float]:
    if not profile_has_stat_model(profile):
        return None
    control_means = np.array([float(profile.control_feature_means[name]) for name in MODEL_FEATURE_NAMES], dtype=float)
    control_stds = np.array([float(profile.control_feature_stds[name]) for name in MODEL_FEATURE_NAMES], dtype=float)
    idle_means = np.array([float(profile.idle_feature_means[name]) for name in MODEL_FEATURE_NAMES], dtype=float)
    idle_stds = np.array([float(profile.idle_feature_stds[name]) for name in MODEL_FEATURE_NAMES], dtype=float)
    return gaussian_log_likelihood_ratio(
        row,
        control_means=control_means,
        control_stds=control_stds,
        idle_means=idle_means,
        idle_stds=idle_stds,
    )


@dataclass(frozen=True)
class TrialSpec:
    label: str
    expected_freq: Optional[float]
    trial_id: int = -1
    block_index: int = -1


def build_calibration_trials(
    freqs: Sequence[float],
    *,
    target_repeats: int,
    idle_repeats: int,
    shuffle: bool = True,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> list[TrialSpec]:
    freq_list = [float(freq) for freq in freqs]
    block_count = max(int(target_repeats), 1)
    idle_distribution = [0 for _ in range(block_count)]
    for index in range(max(int(idle_repeats), 0)):
        idle_distribution[index % block_count] += 1

    rng = random.Random(int(seed))
    trials: list[TrialSpec] = []
    trial_id = 0
    for block_index in range(block_count):
        block_trials = [
            TrialSpec(
                label=f"{freq:g}Hz",
                expected_freq=freq,
                trial_id=trial_id + local_index,
                block_index=block_index,
            )
            for local_index, freq in enumerate(freq_list)
        ]
        trial_id += len(block_trials)
        for _ in range(idle_distribution[block_index]):
            block_trials.append(
                TrialSpec(
                    label="idle",
                    expected_freq=None,
                    trial_id=trial_id,
                    block_index=block_index,
                )
            )
            trial_id += 1
        if shuffle:
            rng.shuffle(block_trials)
        trials.extend(block_trials)
    return trials


def calibration_search_win_candidates(
    active_sec: float,
    preferred_win_sec: float,
    *,
    gate_policy: str = DEFAULT_GATE_POLICY,
) -> list[float]:
    policy = parse_gate_policy(gate_policy)
    if policy == "speed":
        candidates = tuple(DEFAULT_SPEED_WIN_SEC_CANDIDATES)
    else:
        candidates = (*DEFAULT_WIN_SEC_CANDIDATES, float(preferred_win_sec))
    return sorted(
        {
            round(float(candidate), 3)
            for candidate in candidates
            if 0.5 < float(candidate) <= float(active_sec)
        }
    )


def calibration_window_count(active_sec: float, win_sec: float, step_sec: float) -> int:
    active = float(active_sec)
    win = float(win_sec)
    step = max(float(step_sec), 1e-6)
    if active < win:
        return 0
    return 1 + int(math.floor((active - win) / step + 1e-9))


def validate_calibration_plan(
    *,
    target_repeats: int,
    idle_repeats: int,
    active_sec: float,
    preferred_win_sec: float,
    step_sec: float,
    gate_policy: str = DEFAULT_GATE_POLICY,
) -> None:
    problems: list[str] = []
    if int(target_repeats) < 2:
        problems.append("target_repeats must be at least 2 so each target has both train and validation trials")
    if int(idle_repeats) < 2:
        problems.append("idle_repeats must be at least 2 so idle has both train and validation trials")

    policy = parse_gate_policy(gate_policy)
    win_candidates = calibration_search_win_candidates(active_sec, preferred_win_sec, gate_policy=policy)
    min_required_windows = min(DEFAULT_MIN_ENTER_CANDIDATES)
    if not any(calibration_window_count(active_sec, win_sec, step_sec) >= min_required_windows for win_sec in win_candidates):
        problems.append(
            f"active_sec={active_sec:g} is too short for the current search space; "
            f"need at least {min_required_windows} windows per trial with win_sec in {win_candidates or [preferred_win_sec]}"
        )

    if problems:
        raise ValueError("; ".join(problems))


class FBCCAEngine:
    def __init__(
        self,
        sampling_rate: int = 250,
        freqs: Sequence[float] = DEFAULT_FREQS,
        win_sec: float = DEFAULT_WIN_SEC,
        step_sec: float = DEFAULT_STEP_SEC,
        notch_freq: float = 50.0,
        notch_q: float = 30.0,
        Nh: int = DEFAULT_NH,
        subbands: Sequence[tuple[float, float]] = DEFAULT_SUBBANDS,
        subband_weight_mode: Optional[str] = DEFAULT_SUBBAND_WEIGHT_MODE,
        subband_weights: Optional[Sequence[float]] = None,
        subband_weight_params: Optional[dict[str, Any]] = None,
        compute_backend: str = DEFAULT_COMPUTE_BACKEND_NAME,
        gpu_device: int = DEFAULT_GPU_DEVICE_ID,
        gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
        gpu_warmup: bool = True,
    ) -> None:
        self.freqs = tuple(float(freq) for freq in freqs)
        if len(self.freqs) != 4:
            raise ValueError("FBCCAEngine expects exactly four target frequencies")
        self.win_sec = float(win_sec)
        self.step_sec = float(step_sec)
        self.notch_freq = float(notch_freq)
        self.notch_q = float(notch_q)
        self.Nh = int(Nh)
        self.base_subbands = tuple((float(low), float(high)) for low, high in subbands)
        self.subband_weight_mode = parse_subband_weight_mode(subband_weight_mode) or DEFAULT_SUBBAND_WEIGHT_MODE
        self._requested_subband_weights = None if subband_weights is None else tuple(float(value) for value in subband_weights)
        self.subband_weight_params = dict(subband_weight_params or {})
        self.compute_backend_requested = parse_compute_backend_name(compute_backend)
        self.gpu_device = int(gpu_device)
        self.gpu_precision = parse_gpu_precision(gpu_precision)
        self.gpu_warmup = bool(gpu_warmup)
        self.backend = resolve_compute_backend(
            self.compute_backend_requested,
            gpu_device=self.gpu_device,
            precision=self.gpu_precision,
        )
        self.backend_name = str(self.backend.backend_name)

        self.fs = 0
        self.win_samples = 0
        self.step_samples = 0
        self.subbands: list[tuple[float, float]] = []
        self.subband_sos: list[np.ndarray] = []
        self.subband_filters: list[np.ndarray] = []
        self.weights: np.ndarray | None = None
        self.Y_refs: dict[float, np.ndarray] = {}
        self._baseline_sos: Optional[np.ndarray] = None
        self._notch_sos: Optional[np.ndarray] = None
        self._y_refs_host: Optional[np.ndarray] = None
        self._y_refs_device: Any = None
        self.last_timing_breakdown: dict[str, float] = {
            "host_to_device_ms": 0.0,
            "preprocess_ms": 0.0,
            "score_ms": 0.0,
            "device_to_host_ms": 0.0,
            "synchronize_ms": 0.0,
            "warmup_overhead_ms": 0.0,
        }
        self.backend_timing_summary: dict[str, Any] = {}
        self.configure_runtime(sampling_rate)

    def configure_runtime(self, sampling_rate: int) -> None:
        self.fs = int(sampling_rate)
        if self.fs <= 0:
            raise ValueError("sampling_rate must be positive")
        self.win_samples = int(round(self.win_sec * self.fs))
        self.step_samples = max(1, int(round(self.step_sec * self.fs)))
        if self.win_samples < 64:
            raise ValueError("win_sec is too short for stable FBCCA")

        self._baseline_sos = design_sos_lowpass(self.fs, 3.0, order=1)
        self._notch_sos = design_sos_notch(self.fs, self.notch_freq, self.notch_q)
        self.subbands = []
        self.subband_sos = []
        self.subband_filters = []
        for low, high in self.base_subbands:
            sos = design_sos_bandpass(self.fs, low, high, order=2)
            if sos is None:
                continue
            clipped_high = min(float(high), self.fs / 2.0 - 1e-3)
            self.subbands.append((float(low), clipped_high))
            coeffs = np.asarray(sos, dtype=np.float64)
            self.subband_sos.append(coeffs)
            self.subband_filters.append(coeffs)
        if not self.subband_sos:
            raise ValueError(f"sampling_rate={self.fs} is too low for configured subbands")
        weights, resolved_mode, resolved_params = resolve_fbcca_subband_weight_spec(
            len(self.subband_sos),
            mode=self.subband_weight_mode,
            explicit_weights=self._requested_subband_weights,
            params=self.subband_weight_params,
        )
        self.weights = np.asarray(weights, dtype=np.float64)
        self.subband_weight_mode = str(resolved_mode)
        self.subband_weight_params = None if resolved_params is None else dict(resolved_params)
        self._y_refs_host = build_reference_tensor(
            self.fs,
            self.win_samples,
            self.freqs,
            self.Nh,
            dtype=np.float64,
        )
        self.Y_refs = {
            float(freq): np.asarray(self._y_refs_host[index], dtype=np.float64)
            for index, freq in enumerate(self.freqs)
        }
        self._y_refs_device = None
        warmup_ms = 0.0
        if self.backend.uses_cuda:
            self._y_refs_device, _ = self.backend.to_device(self._y_refs_host)
            if self.gpu_warmup:
                warmup_ms = float(self.backend.benchmark_warmup())
        self.last_timing_breakdown = {
            "host_to_device_ms": 0.0,
            "preprocess_ms": 0.0,
            "score_ms": 0.0,
            "device_to_host_ms": 0.0,
            "synchronize_ms": 0.0,
            "warmup_overhead_ms": float(warmup_ms),
        }
        self.backend_timing_summary = {}

    def benchmark_backend_path(
        self,
        sample_window: Optional[np.ndarray] = None,
        *,
        repeats: int = 3,
        batch_size: int = 2,
        default_channels: int = 8,
    ) -> dict[str, Any]:
        if self._baseline_sos is None or self.weights is None:
            raise RuntimeError("FBCCAEngine runtime is not configured")
        refs = self._y_refs_device if self.backend.uses_cuda and self._y_refs_device is not None else self._y_refs_host
        if refs is None:
            raise RuntimeError("FBCCAEngine reference tensors are not configured")
        if sample_window is not None:
            window = np.asarray(sample_window, dtype=np.float64)
            if window.ndim != 2:
                raise ValueError("sample_window must have shape (samples, channels)")
            if int(window.shape[0]) != int(self.win_samples):
                raise ValueError(
                    f"sample_window length mismatch: expected {self.win_samples}, got {window.shape[0]}"
                )
            channel_count = int(window.shape[1])
        else:
            channel_count = max(1, int(default_channels))
            template = self.backend.alloc_pinned_host_array(
                (self.win_samples, channel_count),
                dtype=np.float64,
            )
            template.fill(0.0)
            t = np.arange(self.win_samples, dtype=np.float64) / float(self.fs)
            for channel_index in range(channel_count):
                freq = float(self.freqs[channel_index % len(self.freqs)])
                template[:, channel_index] = np.sin(2.0 * np.pi * freq * t + float(channel_index) * 0.1)
            window = np.asarray(template, dtype=np.float64)
        batch_count = max(1, int(batch_size))
        windows = self.backend.alloc_pinned_host_array(
            (batch_count, self.win_samples, channel_count),
            dtype=np.float64,
        )
        for batch_index in range(batch_count):
            windows[batch_index, :, :] = window
        transfer_summary = self.backend.microbenchmark_transfer(
            sample_shape=tuple(int(value) for value in windows.shape),
            repeats=max(1, int(repeats)),
        )
        kernel_summary = benchmark_fbcca_batch_path(
            self.backend,
            windows,
            y_refs=refs,
            baseline_sos=np.asarray(self._baseline_sos, dtype=np.float64),
            notch_sos=None if self._notch_sos is None else np.asarray(self._notch_sos, dtype=np.float64),
            subband_sos=[np.asarray(item, dtype=np.float64) for item in self.subband_sos],
            subband_weights=np.asarray(self.weights, dtype=np.float64),
            reg=1e-6 if self.gpu_precision == "float32" else 1e-8,
            repeats=max(1, int(repeats)),
        )
        self.backend_timing_summary = {
            "requested_backend": str(self.compute_backend_requested),
            "used_backend": str(self.backend_name),
            "precision": str(self.gpu_precision),
            "gpu_device": int(self.gpu_device),
            "uses_cuda": bool(self.backend.uses_cuda),
            "transfer_benchmark": dict(transfer_summary),
            "kernel_benchmark": dict(kernel_summary),
        }
        return dict(self.backend_timing_summary)

    @staticmethod
    def build_ref_matrix(fs: int, T: int, freq: float, Nh: int) -> np.ndarray:
        return np.asarray(build_reference_tensor(fs, T, (float(freq),), Nh, dtype=np.float64)[0], dtype=np.float64)

    def detrend_and_notch(self, x_raw: np.ndarray) -> np.ndarray:
        if self._baseline_sos is None:
            raise RuntimeError("FBCCAEngine runtime is not configured")
        filtered = np.asarray(x_raw, dtype=np.float64)
        baseline = sosfiltfilt(np.asarray(self._baseline_sos, dtype=np.float64), filtered, axis=0)
        filtered = filtered - baseline
        if self._notch_sos is not None:
            filtered = sosfiltfilt(np.asarray(self._notch_sos, dtype=np.float64), filtered, axis=0)
        return np.asarray(filtered, dtype=np.float64)

    def bandpass_filter_multichannel(self, x_raw: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x_raw, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("bandpass_filter_multichannel expects shape (samples, channels)")
        return np.asarray(
            sosfiltfilt(np.asarray(coeffs, dtype=np.float64), matrix, axis=0),
            dtype=np.float64,
        )

    def preprocess_window(self, X_raw: np.ndarray) -> np.ndarray:
        return self.preprocess_windows_batch(np.asarray(X_raw, dtype=np.float64)[None, :, :])[0]

    def preprocess_windows_batch(self, windows: np.ndarray) -> np.ndarray:
        if self._baseline_sos is None:
            raise RuntimeError("FBCCAEngine runtime is not configured")
        values = preprocess_windows_batch(
            self.backend,
            np.asarray(windows, dtype=np.float64),
            baseline_sos=np.asarray(self._baseline_sos, dtype=np.float64),
            notch_sos=None if self._notch_sos is None else np.asarray(self._notch_sos, dtype=np.float64),
        )
        host_values, device_to_host_ms = self.backend.to_host(values)
        self.last_timing_breakdown = {
            "host_to_device_ms": 0.0,
            "preprocess_ms": 0.0,
            "score_ms": 0.0,
            "device_to_host_ms": float(device_to_host_ms),
            "synchronize_ms": float(self.backend.synchronize()),
            "warmup_overhead_ms": float(self.last_timing_breakdown.get("warmup_overhead_ms", 0.0)),
        }
        return np.asarray(host_values, dtype=np.float64)

    @staticmethod
    def cca_multi_channel_svd(X: np.ndarray, Y: np.ndarray, reg: float = 1e-8) -> float:
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        denom = max(X.shape[0] - 1, 1)
        Sxx = X.T @ X / denom
        Syy = Y.T @ Y / denom
        Sxy = X.T @ Y / denom
        Sxx += reg * np.eye(Sxx.shape[0])
        Syy += reg * np.eye(Syy.shape[0])

        ex, vx = np.linalg.eigh(Sxx)
        ey, vy = np.linalg.eigh(Syy)
        ex = np.maximum(ex, reg)
        ey = np.maximum(ey, reg)
        Sxx_inv_sqrt = vx @ np.diag(1.0 / np.sqrt(ex)) @ vx.T
        Syy_inv_sqrt = vy @ np.diag(1.0 / np.sqrt(ey)) @ vy.T
        Tmat = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
        singular_values = np.linalg.svd(Tmat, compute_uv=False)
        return float(np.max(singular_values))

    def score_windows_batch(self, windows: np.ndarray) -> np.ndarray:
        if self._baseline_sos is None or self.weights is None:
            raise RuntimeError("FBCCAEngine runtime is not configured")
        refs = self._y_refs_device if self.backend.uses_cuda and self._y_refs_device is not None else self._y_refs_host
        if refs is None:
            raise RuntimeError("FBCCAEngine reference tensors are not configured")
        score_matrix, timings = fbcca_scores_batch(
            self.backend,
            np.asarray(windows, dtype=np.float64),
            y_refs=refs,
            baseline_sos=np.asarray(self._baseline_sos, dtype=np.float64),
            notch_sos=None if self._notch_sos is None else np.asarray(self._notch_sos, dtype=np.float64),
            subband_sos=[np.asarray(item, dtype=np.float64) for item in self.subband_sos],
            subband_weights=np.asarray(self.weights, dtype=np.float64),
            reg=1e-6 if self.gpu_precision == "float32" else 1e-8,
        )
        scores_host, device_to_host_ms = self.backend.to_host(score_matrix)
        timings["device_to_host_ms"] = float(device_to_host_ms)
        self.last_timing_breakdown = {str(key): float(value) for key, value in timings.items()}
        return np.asarray(scores_host, dtype=np.float64)

    def score_subbands_batch(self, windows: np.ndarray) -> np.ndarray:
        if self._baseline_sos is None:
            raise RuntimeError("FBCCAEngine runtime is not configured")
        refs = self._y_refs_device if self.backend.uses_cuda and self._y_refs_device is not None else self._y_refs_host
        if refs is None:
            raise RuntimeError("FBCCAEngine reference tensors are not configured")
        tensor, timings = fbcca_subband_scores_batch(
            self.backend,
            np.asarray(windows, dtype=np.float64),
            y_refs=refs,
            baseline_sos=np.asarray(self._baseline_sos, dtype=np.float64),
            notch_sos=None if self._notch_sos is None else np.asarray(self._notch_sos, dtype=np.float64),
            subband_sos=[np.asarray(item, dtype=np.float64) for item in self.subband_sos],
            reg=1e-6 if self.gpu_precision == "float32" else 1e-8,
        )
        tensor_host, device_to_host_ms = self.backend.to_host(tensor)
        timings["device_to_host_ms"] = float(device_to_host_ms)
        self.last_timing_breakdown = {str(key): float(value) for key, value in timings.items()}
        return np.asarray(tensor_host, dtype=np.float64)

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        window = np.asarray(X_window, dtype=np.float64)
        if window.shape[0] != self.win_samples:
            raise ValueError(f"expected {self.win_samples} samples, got {window.shape[0]}")
        return self.score_windows_batch(window[None, :, :])[0]

    def set_subband_weights(
        self,
        weights: Optional[Sequence[float]],
        *,
        mode: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        resolved_weights, resolved_mode, resolved_params = resolve_fbcca_subband_weight_spec(
            len(self.subband_sos),
            mode=mode or self.subband_weight_mode,
            explicit_weights=weights,
            params=params or self.subband_weight_params,
        )
        self.weights = np.asarray(resolved_weights, dtype=np.float64)
        self.subband_weight_mode = str(resolved_mode)
        self.subband_weight_params = None if resolved_params is None else dict(resolved_params)

    def get_subband_weights(self) -> Optional[np.ndarray]:
        if self.weights is None:
            return None
        return np.asarray(self.weights, dtype=float)

    def analyze_window(self, X_window: np.ndarray) -> dict[str, Any]:
        return scores_to_feature_dict(self.score_window(X_window), self.freqs)

    def analyze_windows_batch(self, windows: np.ndarray) -> list[dict[str, Any]]:
        score_matrix = self.score_windows_batch(np.asarray(windows, dtype=np.float64))
        return [scores_to_feature_dict(row, self.freqs) for row in score_matrix]

    def iter_window_features(
        self,
        segment: np.ndarray,
        *,
        expected_freq: Optional[float],
        label: str,
        trial_id: int = -1,
        block_index: int = -1,
    ) -> list[dict[str, Any]]:
        window_batch = extract_window_batch(
            np.asarray(segment, dtype=np.float64),
            win_samples=self.win_samples,
            step_samples=self.step_samples,
        )
        score_matrix = self.score_windows_batch(window_batch)
        return build_feature_rows_from_score_matrix(
            score_matrix,
            freqs=self.freqs,
            expected_freq=expected_freq,
            label=label,
            trial_id=trial_id,
            block_index=block_index,
        )


def scores_to_feature_dict(scores: np.ndarray, freqs: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be a 1D vector")
    if values.size != len(freqs):
        raise ValueError(f"scores length {values.size} does not match freqs length {len(freqs)}")
    order = np.argsort(values)[::-1]
    top1_index = int(order[0])
    top2_index = int(order[1]) if len(order) > 1 else top1_index
    top1_score = float(values[top1_index])
    top2_score = float(values[top2_index]) if len(order) > 1 else 0.0
    margin = top1_score - top2_score
    ratio = float(top1_score / max(top2_score, 1e-12))
    score_sum = float(np.sum(values))
    safe_sum = max(score_sum, 1e-12)
    probs = np.clip(values / safe_sum, 1e-12, None)
    probs = probs / probs.sum()
    entropy = float(-np.sum(probs * np.log(probs)) / np.log(len(probs)))
    return {
        "scores": values,
        "pred_freq": float(freqs[top1_index]),
        "top1_score": top1_score,
        "top2_score": top2_score,
        "margin": float(margin),
        "ratio": float(ratio),
        "score_sum": score_sum,
        "normalized_top1": float(top1_score / safe_sum),
        "score_entropy": entropy,
    }


def extract_window_batch(segment: np.ndarray, *, win_samples: int, step_samples: int) -> np.ndarray:
    matrix = np.ascontiguousarray(np.asarray(segment, dtype=np.float64))
    if matrix.ndim != 2:
        raise ValueError("segment must have shape (samples, channels)")
    if int(matrix.shape[0]) < int(win_samples):
        raise ValueError("segment is shorter than the analysis window")
    windows = np.lib.stride_tricks.sliding_window_view(matrix, window_shape=int(win_samples), axis=0)
    windows = np.asarray(windows[:: max(1, int(step_samples))], dtype=np.float64)
    if windows.ndim != 3:
        raise ValueError(f"unexpected window batch shape: {windows.shape}")
    return np.ascontiguousarray(np.swapaxes(windows, 1, 2), dtype=np.float64)


def build_feature_rows_from_score_matrix(
    scores: np.ndarray,
    *,
    freqs: Sequence[float],
    expected_freq: Optional[float],
    label: str,
    trial_id: int,
    block_index: int,
) -> list[dict[str, Any]]:
    score_matrix = np.asarray(scores, dtype=np.float64)
    if score_matrix.ndim != 2:
        raise ValueError("scores must have shape (windows, freqs)")
    rows: list[dict[str, Any]] = []
    for window_index, score_row in enumerate(score_matrix):
        result = scores_to_feature_dict(np.asarray(score_row, dtype=np.float64), freqs)
        correct = expected_freq is not None and abs(float(result["pred_freq"]) - float(expected_freq)) < 1e-8
        rows.append(
            {
                "label": label,
                "expected_freq": expected_freq,
                "pred_freq": result["pred_freq"],
                "top1_score": result["top1_score"],
                "top2_score": result["top2_score"],
                "margin": result["margin"],
                "ratio": result["ratio"],
                "normalized_top1": result["normalized_top1"],
                "score_entropy": result["score_entropy"],
                "correct": bool(correct),
                "trial_id": int(trial_id),
                "block_index": int(block_index),
                "window_index": int(window_index),
            }
        )
    return rows


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _extract_last_window(segment: np.ndarray, win_samples: int) -> np.ndarray:
    matrix = np.asarray(segment, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("segment must have shape (samples, channels)")
    if matrix.shape[0] < int(win_samples):
        raise ValueError(f"segment too short: {matrix.shape[0]} < {win_samples}")
    return np.ascontiguousarray(matrix[-int(win_samples) :, :], dtype=np.float64)


def _augment_delay_matrix(matrix: np.ndarray, delay_steps: int) -> np.ndarray:
    data = np.asarray(matrix, dtype=float)
    steps = max(1, int(delay_steps))
    if steps <= 1:
        return data
    if data.shape[0] <= steps:
        raise ValueError(f"matrix rows {data.shape[0]} too short for delay_steps={steps}")
    rows: list[np.ndarray] = []
    target_length = data.shape[0] - steps + 1
    for delay in range(steps):
        rows.append(data[delay : delay + target_length, :])
    return np.ascontiguousarray(np.concatenate(rows, axis=1), dtype=np.float64)


def _subset_trial_segments_by_positions(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    channel_positions: Sequence[int],
) -> list[tuple[TrialSpec, np.ndarray]]:
    positions = [int(item) for item in channel_positions]
    subset: list[tuple[TrialSpec, np.ndarray]] = []
    for trial, segment in trial_segments:
        segment_matrix = np.asarray(segment, dtype=float)
        subset.append((trial, np.ascontiguousarray(segment_matrix[:, positions], dtype=np.float64)))
    return subset


class BaseSSVEPDecoder(ABC):
    model_name: str

    def __init__(
        self,
        *,
        model_name: str,
        sampling_rate: int,
        freqs: Sequence[float],
        win_sec: float,
        step_sec: float,
        model_params: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model_name = str(model_name)
        self.freqs = tuple(float(freq) for freq in freqs)
        if len(self.freqs) != 4:
            raise ValueError(f"{self.model_name} expects exactly four target frequencies")
        self.win_sec = float(win_sec)
        self.step_sec = float(step_sec)
        self.model_params = dict(model_params or {})
        self.fs = int(sampling_rate)
        self.win_samples = max(1, int(round(self.win_sec * self.fs)))
        self.step_samples = max(1, int(round(self.step_sec * self.fs)))
        self._channel_weights = normalize_channel_weights(self.model_params.get("channel_weights"))
        self.compute_backend_requested = parse_compute_backend_name(
            self.model_params.get("compute_backend", DEFAULT_COMPUTE_BACKEND_NAME)
        )
        self.gpu_device = int(self.model_params.get("gpu_device", DEFAULT_GPU_DEVICE_ID) or DEFAULT_GPU_DEVICE_ID)
        self.gpu_precision = parse_gpu_precision(
            self.model_params.get("gpu_precision", DEFAULT_GPU_PRECISION_NAME)
        )
        self.gpu_cache_policy = parse_gpu_cache_policy(
            self.model_params.get("gpu_cache_policy", DEFAULT_GPU_CACHE_MODE)
        )
        self.gpu_warmup = bool(int(self.model_params.get("gpu_warmup", 1)))
        self.compute_backend_used = "cpu"
        self._runtime_timing_breakdown: dict[str, float] = {
            "host_to_device_ms": 0.0,
            "preprocess_ms": 0.0,
            "score_ms": 0.0,
            "gate_ms": 0.0,
            "device_to_host_ms": 0.0,
            "synchronize_ms": 0.0,
            "warmup_overhead_ms": 0.0,
        }
        self._backend_timing_summary: dict[str, Any] = {}

    @property
    def requires_fit(self) -> bool:
        return False

    def configure_runtime(self, sampling_rate: int) -> None:
        self.fs = int(sampling_rate)
        self.win_samples = max(1, int(round(self.win_sec * self.fs)))
        self.step_samples = max(1, int(round(self.step_sec * self.fs)))

    def fit(self, trial_segments: Sequence[tuple[TrialSpec, np.ndarray]]) -> None:
        _ = trial_segments

    def _apply_channel_weights(self, matrix: np.ndarray) -> np.ndarray:
        signal = np.asarray(matrix, dtype=float)
        if signal.ndim != 2:
            raise ValueError("window must have shape (samples, channels)")
        if self._channel_weights is None:
            return np.asarray(signal, dtype=float)
        if signal.shape[1] != int(self._channel_weights.size):
            raise ValueError(
                f"{self.model_name} channel_weights mismatch: window has {signal.shape[1]} channels, "
                f"weights has {self._channel_weights.size}"
            )
        return np.asarray(signal * self._channel_weights.reshape(1, -1), dtype=float)

    def set_channel_weights(self, weights: Optional[Sequence[float]]) -> None:
        normalized = normalize_channel_weights(weights)
        self._channel_weights = normalized
        if normalized is None:
            self.model_params.pop("channel_weights", None)
        else:
            self.model_params["channel_weights"] = [float(value) for value in normalized]

    def get_channel_weights(self) -> Optional[np.ndarray]:
        if self._channel_weights is None:
            return None
        return np.asarray(self._channel_weights, dtype=float)

    @abstractmethod
    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def analyze_window(self, X_window: np.ndarray) -> dict[str, Any]:
        weighted = self._apply_channel_weights(np.asarray(X_window, dtype=float))
        return scores_to_feature_dict(self.score_window(weighted), self.freqs)

    def iter_window_features(
        self,
        segment: np.ndarray,
        *,
        expected_freq: Optional[float],
        label: str,
        trial_id: int = -1,
        block_index: int = -1,
    ) -> list[dict[str, Any]]:
        segment_matrix = np.asarray(segment, dtype=float)
        if segment_matrix.ndim != 2:
            raise ValueError("segment must have shape (samples, channels)")
        if segment_matrix.shape[0] < self.win_samples:
            raise ValueError("segment is shorter than the analysis window")
        rows: list[dict[str, Any]] = []
        for window_index, start in enumerate(
            range(0, segment_matrix.shape[0] - self.win_samples + 1, self.step_samples)
        ):
            window = np.ascontiguousarray(segment_matrix[start : start + self.win_samples], dtype=np.float64)
            result = self.analyze_window(window)
            correct = expected_freq is not None and abs(float(result["pred_freq"]) - float(expected_freq)) < 1e-8
            rows.append(
                {
                    "label": label,
                    "expected_freq": expected_freq,
                    "pred_freq": result["pred_freq"],
                    "top1_score": result["top1_score"],
                    "top2_score": result["top2_score"],
                    "margin": result["margin"],
                    "ratio": result["ratio"],
                    "normalized_top1": result["normalized_top1"],
                    "score_entropy": result["score_entropy"],
                    "correct": bool(correct),
                    "trial_id": int(trial_id),
                    "block_index": int(block_index),
                    "window_index": int(window_index),
                }
            )
        return rows

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: Optional[dict[str, Any]]) -> None:
        _ = state

    def update_online(self, decision: dict[str, Any], window: np.ndarray) -> None:
        _ = decision
        _ = window

    def run_backend_microbenchmark(
        self,
        sample_window: Optional[np.ndarray] = None,
        *,
        repeats: int = 3,
    ) -> dict[str, Any]:
        self._backend_timing_summary = {}
        return {}

    def get_compute_backend_summary(self) -> dict[str, Any]:
        return {
            "requested_backend": str(self.compute_backend_requested),
            "used_backend": str(self.compute_backend_used),
            "compute_backend_requested": str(self.compute_backend_requested),
            "compute_backend_used": str(self.compute_backend_used),
            "gpu_device": int(self.gpu_device),
            "precision": str(self.gpu_precision),
            "gpu_cache_policy": str(self.gpu_cache_policy),
            "gpu_warmup": bool(self.gpu_warmup),
            "timing_breakdown": {str(key): float(value) for key, value in self._runtime_timing_breakdown.items()},
            "backend_timing_summary": dict(self._backend_timing_summary),
        }


class FBCCADecoder(BaseSSVEPDecoder):
    def __init__(
        self,
        *,
        sampling_rate: int,
        freqs: Sequence[float],
        win_sec: float,
        step_sec: float,
        model_params: Optional[dict[str, Any]] = None,
    ) -> None:
        params = dict(model_params or {})
        decoder_model_name = str(params.pop("_decoder_model_name", "fbcca"))
        nh = int(params.get("Nh", DEFAULT_NH))
        subbands = tuple(params.get("subbands", DEFAULT_SUBBANDS))
        super().__init__(
            model_name=decoder_model_name,
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
        )
        self.engine = FBCCAEngine(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            Nh=nh,
            subbands=subbands,
            subband_weight_mode=parse_subband_weight_mode(params.get("subband_weight_mode")) or DEFAULT_SUBBAND_WEIGHT_MODE,
            subband_weights=params.get("subband_weights"),
            subband_weight_params=(
                dict(params.get("subband_weight_params"))
                if isinstance(params.get("subband_weight_params"), dict)
                else None
            ),
            compute_backend=self.compute_backend_requested,
            gpu_device=self.gpu_device,
            gpu_precision=self.gpu_precision,
            gpu_warmup=self.gpu_warmup,
        )
        self.compute_backend_used = str(self.engine.backend_name)
        self._backend_timing_summary = dict(self.engine.backend_timing_summary)
        self._spatial_filter_mode: Optional[str] = None
        self._spatial_filter_rank: Optional[int] = None
        self._spatial_source_model: str = DEFAULT_SPATIAL_SOURCE_MODEL
        self._spatial_projection: Optional[np.ndarray] = None
        self._sync_subband_model_params()
        self._load_spatial_frontend_from_params()

    def _sync_subband_model_params(self) -> None:
        self.model_params["subband_weight_mode"] = str(self.engine.subband_weight_mode)
        weights = self.engine.get_subband_weights()
        if weights is None:
            self.model_params.pop("subband_weights", None)
        else:
            self.model_params["subband_weights"] = [float(value) for value in weights]
        if self.engine.subband_weight_params is None:
            self.model_params.pop("subband_weight_params", None)
        else:
            self.model_params["subband_weight_params"] = json_safe(dict(self.engine.subband_weight_params))

    def run_backend_microbenchmark(
        self,
        sample_window: Optional[np.ndarray] = None,
        *,
        repeats: int = 3,
    ) -> dict[str, Any]:
        summary = self.engine.benchmark_backend_path(sample_window=sample_window, repeats=max(1, int(repeats)))
        self._backend_timing_summary = dict(summary)
        return dict(summary)

    def _load_spatial_frontend_from_params(self) -> None:
        mode = parse_spatial_filter_mode(self.model_params.get("spatial_filter_mode"))
        self._spatial_filter_mode = mode
        self._spatial_filter_rank = None
        self._spatial_source_model = parse_spatial_source_model(
            self.model_params.get("spatial_source_model", DEFAULT_SPATIAL_SOURCE_MODEL)
        )
        self._spatial_projection = None
        if mode != "trca_shared":
            self.model_params.pop("spatial_filter_mode", None)
            self.model_params.pop("spatial_filter_rank", None)
            self.model_params.pop("spatial_filter_state", None)
            return
        rank_raw = self.model_params.get("spatial_filter_rank")
        if rank_raw is not None:
            self._spatial_filter_rank = max(1, int(rank_raw))
        state_raw = dict(self.model_params.get("spatial_filter_state") or {})
        source_raw = state_raw.get("source_model")
        if source_raw is not None:
            self._spatial_source_model = parse_spatial_source_model(source_raw)
        projection_raw = (
            state_raw.get("projection")
            if "projection" in state_raw
            else state_raw.get("projection_matrix", state_raw.get("basis"))
        )
        if projection_raw is not None:
            projection = np.asarray(projection_raw, dtype=float)
            if projection.ndim != 2:
                raise ValueError("spatial_filter_state.projection must be a 2D matrix")
            rank = int(self._spatial_filter_rank or projection.shape[1])
            rank = max(1, min(rank, int(projection.shape[1])))
            self._spatial_filter_rank = rank
            self._spatial_projection = np.asarray(projection[:, :rank], dtype=float)
            state_payload = dict(state_raw)
            state_payload["source_model"] = str(self._spatial_source_model)
            state_payload["projection"] = np.asarray(self._spatial_projection, dtype=float).tolist()
            state_payload["rank"] = int(rank)
            self.model_params["spatial_filter_state"] = state_payload
            self.model_params["spatial_filter_mode"] = str(mode)
            self.model_params["spatial_filter_rank"] = int(rank)
            self.model_params["spatial_source_model"] = str(self._spatial_source_model)
        else:
            self.model_params["spatial_filter_mode"] = str(mode)
            if self._spatial_filter_rank is not None:
                self.model_params["spatial_filter_rank"] = int(self._spatial_filter_rank)
            self.model_params["spatial_source_model"] = str(self._spatial_source_model)

    def _apply_frontend(self, matrix: np.ndarray) -> np.ndarray:
        weighted = self._apply_channel_weights(np.asarray(matrix, dtype=float))
        if self._spatial_projection is None:
            return np.asarray(weighted, dtype=float)
        if weighted.shape[1] != int(self._spatial_projection.shape[0]):
            raise ValueError(
                "spatial projection shape mismatch: "
                f"window channels={weighted.shape[1]}, projection input={self._spatial_projection.shape[0]}"
            )
        transformed = weighted @ np.asarray(self._spatial_projection, dtype=float)
        return np.asarray(transformed, dtype=float)

    def configure_runtime(self, sampling_rate: int) -> None:
        super().configure_runtime(sampling_rate)
        self.engine.configure_runtime(self.fs)
        self._sync_subband_model_params()
        self.compute_backend_used = str(self.engine.backend_name)

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        front = self._apply_frontend(np.asarray(X_window, dtype=float))
        scores = self.engine.score_window(front)
        self._runtime_timing_breakdown = dict(self.engine.last_timing_breakdown)
        return scores

    def analyze_window(self, X_window: np.ndarray) -> dict[str, Any]:
        front = self._apply_frontend(np.asarray(X_window, dtype=float))
        result = self.engine.analyze_window(front)
        self._runtime_timing_breakdown = dict(self.engine.last_timing_breakdown)
        return result

    def iter_window_features(
        self,
        segment: np.ndarray,
        *,
        expected_freq: Optional[float],
        label: str,
        trial_id: int = -1,
        block_index: int = -1,
    ) -> list[dict[str, Any]]:
        front = self._apply_frontend(np.asarray(segment, dtype=np.float64))
        rows = self.engine.iter_window_features(
            front,
            expected_freq=expected_freq,
            label=label,
            trial_id=trial_id,
            block_index=block_index,
        )
        self._runtime_timing_breakdown = dict(self.engine.last_timing_breakdown)
        return rows

    def set_subband_weights(
        self,
        weights: Optional[Sequence[float]],
        *,
        mode: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        self.engine.set_subband_weights(weights, mode=mode, params=params)
        self._sync_subband_model_params()


class LegacyFBCCA202603Decoder(BaseSSVEPDecoder):
    def __init__(
        self,
        *,
        sampling_rate: int,
        freqs: Sequence[float],
        win_sec: float,
        step_sec: float,
        model_params: Optional[dict[str, Any]] = None,
    ) -> None:
        params = dict(model_params or {})
        params.pop("channel_weights", None)
        params.pop("channel_weight_mode", None)
        params.pop("spatial_filter_mode", None)
        params.pop("spatial_filter_rank", None)
        params.pop("spatial_filter_state", None)
        params.pop("joint_weight_training", None)
        nh = int(params.get("Nh", DEFAULT_NH))
        subbands = tuple(params.get("subbands", DEFAULT_SUBBANDS))
        super().__init__(
            model_name="legacy_fbcca_202603",
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
        )
        self.engine = FBCCAEngine(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            Nh=nh,
            subbands=subbands,
            compute_backend=self.compute_backend_requested,
            gpu_device=self.gpu_device,
            gpu_precision=self.gpu_precision,
            gpu_warmup=self.gpu_warmup,
        )
        self.compute_backend_used = str(self.engine.backend_name)

    def configure_runtime(self, sampling_rate: int) -> None:
        super().configure_runtime(sampling_rate)
        self.engine.configure_runtime(self.fs)
        self.compute_backend_used = str(self.engine.backend_name)

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        scores = self.engine.score_window(np.asarray(X_window, dtype=float))
        self._runtime_timing_breakdown = dict(self.engine.last_timing_breakdown)
        return scores

    def analyze_window(self, X_window: np.ndarray) -> dict[str, Any]:
        result = self.engine.analyze_window(np.asarray(X_window, dtype=float))
        self._runtime_timing_breakdown = dict(self.engine.last_timing_breakdown)
        return result

    def iter_window_features(
        self,
        segment: np.ndarray,
        *,
        expected_freq: Optional[float],
        label: str,
        trial_id: int = -1,
        block_index: int = -1,
    ) -> list[dict[str, Any]]:
        rows = self.engine.iter_window_features(
            np.asarray(segment, dtype=np.float64),
            expected_freq=expected_freq,
            label=label,
            trial_id=trial_id,
            block_index=block_index,
        )
        self._runtime_timing_breakdown = dict(self.engine.last_timing_breakdown)
        return rows


class CCADecoder(BaseSSVEPDecoder):
    def __init__(
        self,
        *,
        sampling_rate: int,
        freqs: Sequence[float],
        win_sec: float,
        step_sec: float,
        model_params: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_name="cca",
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=model_params,
        )
        nh = int(self.model_params.get("Nh", DEFAULT_NH))
        self._core = FBCCAEngine(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            Nh=nh,
            subbands=((6.0, 50.0),),
            compute_backend=self.compute_backend_requested,
            gpu_device=self.gpu_device,
            gpu_precision=self.gpu_precision,
            gpu_warmup=self.gpu_warmup,
        )
        self.compute_backend_used = str(self._core.backend_name)

    def configure_runtime(self, sampling_rate: int) -> None:
        super().configure_runtime(sampling_rate)
        self._core.configure_runtime(self.fs)
        self.compute_backend_used = str(self._core.backend_name)

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        windows = np.asarray(X_window, dtype=np.float64)[None, :, :]
        refs = self._core._y_refs_device if self._core.backend.uses_cuda and self._core._y_refs_device is not None else self._core._y_refs_host
        if refs is None or self._core._baseline_sos is None:
            raise RuntimeError("CCA runtime is not configured")
        score_matrix, timings = cca_scores_batch(
            self._core.backend,
            windows,
            y_refs=refs,
            baseline_sos=np.asarray(self._core._baseline_sos, dtype=np.float64),
            notch_sos=None if self._core._notch_sos is None else np.asarray(self._core._notch_sos, dtype=np.float64),
            reg=1e-6 if self.gpu_precision == "float32" else 1e-8,
        )
        scores, device_to_host_ms = self._core.backend.to_host(score_matrix)
        timings["device_to_host_ms"] = float(device_to_host_ms)
        self._runtime_timing_breakdown = {str(key): float(value) for key, value in timings.items()}
        return np.asarray(scores[0], dtype=np.float64)

    def iter_window_features(
        self,
        segment: np.ndarray,
        *,
        expected_freq: Optional[float],
        label: str,
        trial_id: int = -1,
        block_index: int = -1,
    ) -> list[dict[str, Any]]:
        window_batch = extract_window_batch(
            np.asarray(segment, dtype=np.float64),
            win_samples=self.win_samples,
            step_samples=self.step_samples,
        )
        refs = self._core._y_refs_device if self._core.backend.uses_cuda and self._core._y_refs_device is not None else self._core._y_refs_host
        if refs is None or self._core._baseline_sos is None:
            raise RuntimeError("CCA runtime is not configured")
        score_matrix, timings = cca_scores_batch(
            self._core.backend,
            window_batch,
            y_refs=refs,
            baseline_sos=np.asarray(self._core._baseline_sos, dtype=np.float64),
            notch_sos=None if self._core._notch_sos is None else np.asarray(self._core._notch_sos, dtype=np.float64),
            reg=1e-6 if self.gpu_precision == "float32" else 1e-8,
        )
        scores, device_to_host_ms = self._core.backend.to_host(score_matrix)
        timings["device_to_host_ms"] = float(device_to_host_ms)
        self._runtime_timing_breakdown = {str(key): float(value) for key, value in timings.items()}
        return build_feature_rows_from_score_matrix(
            np.asarray(scores, dtype=np.float64),
            freqs=self.freqs,
            expected_freq=expected_freq,
            label=label,
            trial_id=trial_id,
            block_index=block_index,
        )


class TemplateCCADecoder(BaseSSVEPDecoder):
    def __init__(
        self,
        *,
        variant: str,
        sampling_rate: int,
        freqs: Sequence[float],
        win_sec: float,
        step_sec: float,
        model_params: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_name=str(variant),
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=model_params,
        )
        nh = int(self.model_params.get("Nh", DEFAULT_NH))
        self.variant = str(variant)
        self._core = FBCCAEngine(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            Nh=nh,
            subbands=((6.0, 50.0),),
            compute_backend=self.compute_backend_requested,
            gpu_device=self.gpu_device,
            gpu_precision=self.gpu_precision,
            gpu_warmup=self.gpu_warmup,
        )
        self.compute_backend_used = str(self._core.backend_name)
        self.templates: dict[float, np.ndarray] = {}
        self.template_weights: dict[float, np.ndarray] = {}

    @property
    def requires_fit(self) -> bool:
        return True

    def configure_runtime(self, sampling_rate: int) -> None:
        super().configure_runtime(sampling_rate)
        self._core.configure_runtime(self.fs)
        self.compute_backend_used = str(self._core.backend_name)

    def fit(self, trial_segments: Sequence[tuple[TrialSpec, np.ndarray]]) -> None:
        grouped: dict[float, list[np.ndarray]] = defaultdict(list)
        for trial, segment in trial_segments:
            if trial.expected_freq is None:
                continue
            weighted_segment = self._apply_channel_weights(np.asarray(segment, dtype=float))
            window = _extract_last_window(weighted_segment, self._core.win_samples)
            grouped[float(trial.expected_freq)].append(self._core.preprocess_window(window))
        templates: dict[float, np.ndarray] = {}
        weights: dict[float, np.ndarray] = {}
        for freq in self.freqs:
            entries = grouped.get(float(freq), [])
            if not entries:
                continue
            template = np.mean(np.stack(entries, axis=0), axis=0)
            templates[float(freq)] = np.asarray(template, dtype=float)
            try:
                X = template - template.mean(axis=0, keepdims=True)
                Y = self._core.Y_refs[float(freq)] - self._core.Y_refs[float(freq)].mean(axis=0, keepdims=True)
                denom = max(X.shape[0] - 1, 1)
                Sxx = X.T @ X / denom + 1e-8 * np.eye(X.shape[1])
                Syy = Y.T @ Y / denom + 1e-8 * np.eye(Y.shape[1])
                Sxy = X.T @ Y / denom
                ex, vx = np.linalg.eigh(Sxx)
                ey, vy = np.linalg.eigh(Syy)
                ex = np.maximum(ex, 1e-8)
                ey = np.maximum(ey, 1e-8)
                tx = vx @ np.diag(1.0 / np.sqrt(ex)) @ vx.T
                ty = vy @ np.diag(1.0 / np.sqrt(ey)) @ vy.T
                matrix = tx @ Sxy @ ty
                u, _s, _vt = np.linalg.svd(matrix, full_matrices=False)
                weights[float(freq)] = np.asarray(tx @ u[:, :1], dtype=float)
            except Exception:
                weights[float(freq)] = np.ones((template.shape[1], 1), dtype=float)
        if len(templates) != len(self.freqs):
            missing = [freq for freq in self.freqs if float(freq) not in templates]
            raise ValueError(f"{self.model_name} fit missing templates for frequencies: {missing}")
        self.templates = templates
        self.template_weights = weights

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        if not self.templates:
            raise RuntimeError(f"{self.model_name} requires fit() before score_window()")
        X0 = self._core.preprocess_window(np.asarray(X_window, dtype=float))
        scores = np.zeros(len(self.freqs), dtype=float)
        for index, freq in enumerate(self.freqs):
            ref = self._core.Y_refs[float(freq)]
            template = self.templates[float(freq)]
            r_ref = self._core.cca_multi_channel_svd(X0, ref)
            r_template = self._core.cca_multi_channel_svd(X0, template)
            if self.variant == "itcca":
                scores[index] = float(max(r_template, 0.0) ** 2)
                continue
            if self.variant == "msetcca":
                scores[index] = float(0.5 * max(r_ref, 0.0) ** 2 + 0.5 * max(r_template, 0.0) ** 2)
                continue
            template_weight = self.template_weights.get(float(freq))
            if template_weight is None:
                template_weight = np.ones((X0.shape[1], 1), dtype=float)
            w = np.asarray(template_weight, dtype=float).reshape(-1)
            x_proj = X0 @ w
            t_proj = template @ w
            r_proj = _safe_corrcoef(x_proj, t_proj)
            scores[index] = float(max(r_ref, 0.0) ** 2 + max(r_template, 0.0) ** 2 + max(r_proj, 0.0) ** 2)
        return scores

    def get_state(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "templates": {f"{freq:g}": matrix.tolist() for freq, matrix in self.templates.items()},
            "template_weights": {f"{freq:g}": matrix.tolist() for freq, matrix in self.template_weights.items()},
        }

    def set_state(self, state: Optional[dict[str, Any]]) -> None:
        payload = dict(state or {})
        templates_raw = payload.get("templates", {})
        weights_raw = payload.get("template_weights", {})
        self.templates = {
            float(key.replace("Hz", "")): np.asarray(value, dtype=float) for key, value in templates_raw.items()
        }
        self.template_weights = {
            float(key.replace("Hz", "")): np.asarray(value, dtype=float) for key, value in weights_raw.items()
        }


class TRCABasedDecoder(BaseSSVEPDecoder):
    def __init__(
        self,
        *,
        model_name: str,
        sampling_rate: int,
        freqs: Sequence[float],
        win_sec: float,
        step_sec: float,
        model_params: Optional[dict[str, Any]] = None,
        use_filterbank: bool = False,
        ensemble: bool = False,
        delay_steps: int = 1,
    ) -> None:
        super().__init__(
            model_name=model_name,
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=model_params,
        )
        self.use_filterbank = bool(use_filterbank)
        self.ensemble = bool(ensemble)
        self.delay_steps = max(1, int(delay_steps))
        nh = int(self.model_params.get("Nh", DEFAULT_NH))
        self._core = FBCCAEngine(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            Nh=nh,
            subbands=DEFAULT_SUBBANDS if self.use_filterbank else ((6.0, 50.0),),
            compute_backend=self.compute_backend_requested,
            gpu_device=self.gpu_device,
            gpu_precision=self.gpu_precision,
            gpu_warmup=self.gpu_warmup,
        )
        self.compute_backend_used = str(self._core.backend_name)
        self.band_templates: list[dict[float, np.ndarray]] = []
        self.band_filters: list[dict[float, np.ndarray]] = []

    @property
    def requires_fit(self) -> bool:
        return True

    def configure_runtime(self, sampling_rate: int) -> None:
        super().configure_runtime(sampling_rate)
        self._core.configure_runtime(self.fs)
        self.compute_backend_used = str(self._core.backend_name)

    def _augment_delay(self, matrix: np.ndarray) -> np.ndarray:
        return _augment_delay_matrix(matrix, self.delay_steps)

    @staticmethod
    def _train_trca_filter(trials: Sequence[np.ndarray]) -> np.ndarray:
        filters = TRCABasedDecoder._train_trca_filters(trials, n_components=1)
        return np.asarray(filters[:, 0], dtype=float)

    @staticmethod
    def _train_trca_filters(trials: Sequence[np.ndarray], *, n_components: int) -> np.ndarray:
        if not trials:
            raise ValueError("TRCA training trials are empty")
        channels = int(trials[0].shape[0])
        if len(trials) == 1:
            u, _s, _vt = np.linalg.svd(trials[0], full_matrices=False)
            keep = max(1, min(int(n_components), u.shape[1]))
            return np.asarray(u[:, :keep], dtype=float)
        q_mat = np.zeros((channels, channels), dtype=float)
        s_mat = np.zeros((channels, channels), dtype=float)
        for Xi in trials:
            q_mat += Xi @ Xi.T
        for i in range(len(trials)):
            for j in range(len(trials)):
                if i == j:
                    continue
                s_mat += trials[i] @ trials[j].T
        reg = 1e-6 * np.trace(q_mat) / max(channels, 1)
        q_mat = q_mat + reg * np.eye(channels)
        eigvals, eigvecs = scipy.linalg.eigh(s_mat, q_mat)
        order = np.argsort(eigvals)[::-1]
        keep = max(1, min(int(n_components), eigvecs.shape[1]))
        selected = np.asarray(eigvecs[:, order[:keep]], dtype=float)
        for index in range(selected.shape[1]):
            norm = np.linalg.norm(selected[:, index])
            if norm <= 1e-12:
                selected[:, index] = 1.0 / math.sqrt(max(channels, 1))
            else:
                selected[:, index] = selected[:, index] / norm
        return selected

    def fit(self, trial_segments: Sequence[tuple[TrialSpec, np.ndarray]]) -> None:
        grouped: dict[float, list[np.ndarray]] = defaultdict(list)
        for trial, segment in trial_segments:
            if trial.expected_freq is None:
                continue
            weighted_segment = self._apply_channel_weights(np.asarray(segment, dtype=float))
            window = _extract_last_window(weighted_segment, self._core.win_samples)
            base = self._core.preprocess_window(window)
            grouped[float(trial.expected_freq)].append(base)
        if not all(grouped.get(float(freq)) for freq in self.freqs):
            missing = [freq for freq in self.freqs if not grouped.get(float(freq))]
            raise ValueError(f"{self.model_name} fit missing training trials for frequencies: {missing}")

        band_templates: list[dict[float, np.ndarray]] = []
        band_filters: list[dict[float, np.ndarray]] = []
        for band_index, coeffs in enumerate(self._core.subband_filters):
            templates_for_band: dict[float, np.ndarray] = {}
            filters_for_band: dict[float, np.ndarray] = {}
            for freq in self.freqs:
                freq_trials = [self._core.bandpass_filter_multichannel(item, coeffs) for item in grouped[float(freq)]]
                delayed_trials = [self._augment_delay(item).T for item in freq_trials]
                templates_for_band[float(freq)] = np.mean(np.stack(delayed_trials, axis=0), axis=0)
                filters_for_band[float(freq)] = self._train_trca_filter(delayed_trials)
            band_templates.append(templates_for_band)
            band_filters.append(filters_for_band)
            _ = band_index
        self.band_templates = band_templates
        self.band_filters = band_filters

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        if not self.band_templates or not self.band_filters:
            raise RuntimeError(f"{self.model_name} requires fit() before score_window()")
        base = self._core.preprocess_window(np.asarray(X_window, dtype=float))
        scores = np.zeros(len(self.freqs), dtype=float)
        for band_index, coeffs in enumerate(self._core.subband_filters):
            band_signal = self._core.bandpass_filter_multichannel(base, coeffs)
            delayed = self._augment_delay(band_signal).T
            templates = self.band_templates[band_index]
            filters = self.band_filters[band_index]
            for class_index, freq in enumerate(self.freqs):
                if self.ensemble:
                    class_score = 0.0
                    for filter_freq in self.freqs:
                        w = np.asarray(filters[float(filter_freq)], dtype=float).reshape(-1)
                        x_proj = w @ delayed
                        y_proj = w @ templates[float(freq)]
                        class_score += _safe_corrcoef(x_proj, y_proj) ** 2
                else:
                    w = np.asarray(filters[float(freq)], dtype=float).reshape(-1)
                    x_proj = w @ delayed
                    y_proj = w @ templates[float(freq)]
                    class_score = _safe_corrcoef(x_proj, y_proj) ** 2
                scores[class_index] += float(self._core.weights[band_index]) * float(class_score)
        return scores

    def get_state(self) -> dict[str, Any]:
        return {
            "use_filterbank": bool(self.use_filterbank),
            "ensemble": bool(self.ensemble),
            "delay_steps": int(self.delay_steps),
            "band_templates": [
                {f"{freq:g}": np.asarray(matrix, dtype=float).tolist() for freq, matrix in entry.items()}
                for entry in self.band_templates
            ],
            "band_filters": [
                {f"{freq:g}": np.asarray(matrix, dtype=float).tolist() for freq, matrix in entry.items()}
                for entry in self.band_filters
            ],
        }

    def set_state(self, state: Optional[dict[str, Any]]) -> None:
        payload = dict(state or {})
        self.use_filterbank = bool(payload.get("use_filterbank", self.use_filterbank))
        self.ensemble = bool(payload.get("ensemble", self.ensemble))
        self.delay_steps = int(payload.get("delay_steps", self.delay_steps))
        templates_raw = list(payload.get("band_templates", []))
        filters_raw = list(payload.get("band_filters", []))
        self.band_templates = [
            {float(key.replace("Hz", "")): np.asarray(value, dtype=float) for key, value in entry.items()}
            for entry in templates_raw
        ]
        self.band_filters = [
            {float(key.replace("Hz", "")): np.asarray(value, dtype=float) for key, value in entry.items()}
            for entry in filters_raw
        ]


class SSCORDecoder(TRCABasedDecoder):
    def __init__(
        self,
        *,
        sampling_rate: int,
        freqs: Sequence[float],
        win_sec: float,
        step_sec: float,
        model_params: Optional[dict[str, Any]] = None,
    ) -> None:
        params = dict(model_params or {})
        self.n_components = max(1, int(params.get("n_components", DEFAULT_SSCOR_COMPONENTS)))
        super().__init__(
            model_name="sscor",
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
            use_filterbank=bool(params.get("use_filterbank", True)),
            ensemble=False,
            delay_steps=int(params.get("delay_steps", 1)),
        )

    def fit(self, trial_segments: Sequence[tuple[TrialSpec, np.ndarray]]) -> None:
        grouped: dict[float, list[np.ndarray]] = defaultdict(list)
        for trial, segment in trial_segments:
            if trial.expected_freq is None:
                continue
            weighted_segment = self._apply_channel_weights(np.asarray(segment, dtype=float))
            window = _extract_last_window(weighted_segment, self._core.win_samples)
            base = self._core.preprocess_window(window)
            grouped[float(trial.expected_freq)].append(base)
        if not all(grouped.get(float(freq)) for freq in self.freqs):
            missing = [freq for freq in self.freqs if not grouped.get(float(freq))]
            raise ValueError(f"{self.model_name} fit missing training trials for frequencies: {missing}")

        band_templates: list[dict[float, np.ndarray]] = []
        band_filters: list[dict[float, np.ndarray]] = []
        for coeffs in self._core.subband_filters:
            templates_for_band: dict[float, np.ndarray] = {}
            filters_for_band: dict[float, np.ndarray] = {}
            for freq in self.freqs:
                freq_trials = [self._core.bandpass_filter_multichannel(item, coeffs) for item in grouped[float(freq)]]
                delayed_trials = [self._augment_delay(item).T for item in freq_trials]
                templates_for_band[float(freq)] = np.mean(np.stack(delayed_trials, axis=0), axis=0)
                filters_for_band[float(freq)] = self._train_trca_filters(
                    delayed_trials,
                    n_components=self.n_components,
                )
            band_templates.append(templates_for_band)
            band_filters.append(filters_for_band)
        self.band_templates = band_templates
        self.band_filters = band_filters

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        if not self.band_templates or not self.band_filters:
            raise RuntimeError(f"{self.model_name} requires fit() before score_window()")
        base = self._core.preprocess_window(np.asarray(X_window, dtype=float))
        scores = np.zeros(len(self.freqs), dtype=float)
        for band_index, coeffs in enumerate(self._core.subband_filters):
            band_signal = self._core.bandpass_filter_multichannel(base, coeffs)
            delayed = self._augment_delay(band_signal).T
            templates = self.band_templates[band_index]
            filters = self.band_filters[band_index]
            for class_index, freq in enumerate(self.freqs):
                W = np.asarray(filters[float(freq)], dtype=float)
                if W.ndim == 1:
                    W = W.reshape(-1, 1)
                x_proj = W.T @ delayed
                y_proj = W.T @ templates[float(freq)]
                component_scores = [
                    _safe_corrcoef(x_proj[idx], y_proj[idx]) ** 2 for idx in range(min(x_proj.shape[0], y_proj.shape[0]))
                ]
                class_score = float(np.sum(component_scores))
                scores[class_index] += float(self._core.weights[band_index]) * class_score
        return scores

    def get_state(self) -> dict[str, Any]:
        payload = super().get_state()
        payload["n_components"] = int(self.n_components)
        payload["use_filterbank"] = bool(self.use_filterbank)
        return payload

    def set_state(self, state: Optional[dict[str, Any]]) -> None:
        super().set_state(state)
        payload = dict(state or {})
        self.n_components = max(1, int(payload.get("n_components", self.n_components)))


class TDCADecoder(BaseSSVEPDecoder):
    def __init__(
        self,
        *,
        sampling_rate: int,
        freqs: Sequence[float],
        win_sec: float,
        step_sec: float,
        model_params: Optional[dict[str, Any]] = None,
    ) -> None:
        params = dict(model_params or {})
        super().__init__(
            model_name="tdca",
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
        )
        self.delay_steps = max(1, int(params.get("delay_steps", 3)))
        self.n_components = max(1, int(params.get("n_components", DEFAULT_TDCA_COMPONENTS)))
        nh = int(self.model_params.get("Nh", DEFAULT_NH))
        self._core = FBCCAEngine(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            Nh=nh,
            subbands=DEFAULT_SUBBANDS,
            compute_backend=self.compute_backend_requested,
            gpu_device=self.gpu_device,
            gpu_precision=self.gpu_precision,
            gpu_warmup=self.gpu_warmup,
        )
        self.compute_backend_used = str(self._core.backend_name)
        self.band_projection: list[np.ndarray] = []
        self.band_templates: list[dict[float, np.ndarray]] = []

    @property
    def requires_fit(self) -> bool:
        return True

    def configure_runtime(self, sampling_rate: int) -> None:
        super().configure_runtime(sampling_rate)
        self._core.configure_runtime(self.fs)
        self.compute_backend_used = str(self._core.backend_name)

    def _augment_delay(self, matrix: np.ndarray) -> np.ndarray:
        return _augment_delay_matrix(matrix, self.delay_steps)

    @staticmethod
    def _safe_projector(
        between_cov: np.ndarray,
        within_cov: np.ndarray,
        *,
        n_components: int,
    ) -> np.ndarray:
        dim = int(within_cov.shape[0])
        reg = 1e-6 * np.trace(within_cov) / max(dim, 1)
        within_cov = within_cov + reg * np.eye(dim)
        eigvals, eigvecs = scipy.linalg.eigh(between_cov, within_cov)
        order = np.argsort(eigvals)[::-1]
        keep = max(1, min(int(n_components), eigvecs.shape[1]))
        selected = np.asarray(eigvecs[:, order[:keep]], dtype=float)
        for index in range(selected.shape[1]):
            norm = np.linalg.norm(selected[:, index])
            if norm <= 1e-12:
                selected[:, index] = 1.0 / math.sqrt(max(dim, 1))
            else:
                selected[:, index] = selected[:, index] / norm
        return selected

    def fit(self, trial_segments: Sequence[tuple[TrialSpec, np.ndarray]]) -> None:
        grouped: dict[float, list[np.ndarray]] = defaultdict(list)
        for trial, segment in trial_segments:
            if trial.expected_freq is None:
                continue
            weighted_segment = self._apply_channel_weights(np.asarray(segment, dtype=float))
            window = _extract_last_window(weighted_segment, self._core.win_samples)
            grouped[float(trial.expected_freq)].append(self._core.preprocess_window(window))
        if not all(grouped.get(float(freq)) for freq in self.freqs):
            missing = [freq for freq in self.freqs if not grouped.get(float(freq))]
            raise ValueError(f"{self.model_name} fit missing training trials for frequencies: {missing}")

        band_projection: list[np.ndarray] = []
        band_templates: list[dict[float, np.ndarray]] = []
        for coeffs in self._core.subband_filters:
            class_trials: dict[float, list[np.ndarray]] = {}
            class_means: dict[float, np.ndarray] = {}
            for freq in self.freqs:
                filtered = [self._core.bandpass_filter_multichannel(item, coeffs) for item in grouped[float(freq)]]
                delayed_trials = [self._augment_delay(item).T for item in filtered]
                class_trials[float(freq)] = delayed_trials
                class_means[float(freq)] = np.mean(np.stack(delayed_trials, axis=0), axis=0)

            all_means = np.stack([class_means[float(freq)] for freq in self.freqs], axis=0)
            global_mean = np.mean(all_means, axis=0)
            feat_dim = int(global_mean.shape[0])
            between_cov = np.zeros((feat_dim, feat_dim), dtype=float)
            within_cov = np.zeros((feat_dim, feat_dim), dtype=float)
            for freq in self.freqs:
                mean_mat = class_means[float(freq)]
                mean_centered = mean_mat - global_mean
                between_cov += float(len(class_trials[float(freq)])) * (mean_centered @ mean_centered.T)
                for trial_mat in class_trials[float(freq)]:
                    diff = trial_mat - mean_mat
                    within_cov += diff @ diff.T

            projector = self._safe_projector(
                between_cov,
                within_cov,
                n_components=self.n_components,
            )
            band_projection.append(projector)
            band_templates.append(class_means)

        self.band_projection = band_projection
        self.band_templates = band_templates

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        if not self.band_projection or not self.band_templates:
            raise RuntimeError(f"{self.model_name} requires fit() before score_window()")
        base = self._core.preprocess_window(np.asarray(X_window, dtype=float))
        scores = np.zeros(len(self.freqs), dtype=float)
        for band_index, coeffs in enumerate(self._core.subband_filters):
            band_signal = self._core.bandpass_filter_multichannel(base, coeffs)
            delayed = self._augment_delay(band_signal).T
            projector = self.band_projection[band_index]
            projected_window = projector.T @ delayed
            templates = self.band_templates[band_index]
            for class_index, freq in enumerate(self.freqs):
                projected_template = projector.T @ templates[float(freq)]
                component_scores = [
                    _safe_corrcoef(projected_window[idx], projected_template[idx]) ** 2
                    for idx in range(min(projected_window.shape[0], projected_template.shape[0]))
                ]
                class_score = float(np.sum(component_scores))
                scores[class_index] += float(self._core.weights[band_index]) * class_score
        return scores

    def get_state(self) -> dict[str, Any]:
        return {
            "delay_steps": int(self.delay_steps),
            "n_components": int(self.n_components),
            "band_projection": [np.asarray(matrix, dtype=float).tolist() for matrix in self.band_projection],
            "band_templates": [
                {f"{freq:g}": np.asarray(matrix, dtype=float).tolist() for freq, matrix in entry.items()}
                for entry in self.band_templates
            ],
        }

    def set_state(self, state: Optional[dict[str, Any]]) -> None:
        payload = dict(state or {})
        self.delay_steps = max(1, int(payload.get("delay_steps", self.delay_steps)))
        self.n_components = max(1, int(payload.get("n_components", self.n_components)))
        projection_raw = list(payload.get("band_projection", []))
        templates_raw = list(payload.get("band_templates", []))
        self.band_projection = [np.asarray(matrix, dtype=float) for matrix in projection_raw]
        self.band_templates = [
            {float(key.replace("Hz", "")): np.asarray(value, dtype=float) for key, value in entry.items()}
            for entry in templates_raw
        ]


class OACCADecoder(CCADecoder):
    def __init__(
        self,
        *,
        sampling_rate: int,
        freqs: Sequence[float],
        win_sec: float,
        step_sec: float,
        model_params: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=model_params,
        )
        self.model_name = "oacca"
        self.templates: dict[float, np.ndarray] = {}
        self.update_alpha = float(self.model_params.get("update_alpha", DEFAULT_DECODER_UPDATE_ALPHA))

    @property
    def requires_fit(self) -> bool:
        return True

    def fit(self, trial_segments: Sequence[tuple[TrialSpec, np.ndarray]]) -> None:
        grouped: dict[float, list[np.ndarray]] = defaultdict(list)
        for trial, segment in trial_segments:
            if trial.expected_freq is None:
                continue
            weighted_segment = self._apply_channel_weights(np.asarray(segment, dtype=float))
            window = _extract_last_window(weighted_segment, self._core.win_samples)
            grouped[float(trial.expected_freq)].append(self._core.preprocess_window(window))
        templates: dict[float, np.ndarray] = {}
        for freq in self.freqs:
            entries = grouped.get(float(freq), [])
            if not entries:
                continue
            templates[float(freq)] = np.mean(np.stack(entries, axis=0), axis=0)
        if len(templates) != len(self.freqs):
            missing = [freq for freq in self.freqs if float(freq) not in templates]
            raise ValueError(f"oacca fit missing templates for frequencies: {missing}")
        self.templates = templates

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        if not self.templates:
            raise RuntimeError("oacca requires fit() before score_window()")
        base = self._core.preprocess_window(np.asarray(X_window, dtype=float))
        scores = np.zeros(len(self.freqs), dtype=float)
        for index, freq in enumerate(self.freqs):
            ref = self._core.Y_refs[float(freq)]
            template = self.templates[float(freq)]
            r_ref = self._core.cca_multi_channel_svd(base, ref)
            r_template = self._core.cca_multi_channel_svd(base, template)
            scores[index] = float(0.6 * max(r_ref, 0.0) ** 2 + 0.4 * max(r_template, 0.0) ** 2)
        return scores

    def update_online(self, decision: dict[str, Any], window: np.ndarray) -> None:
        selected = decision.get("selected_freq")
        if selected is None or float(selected) not in self.templates:
            return
        weighted = self._apply_channel_weights(np.asarray(window, dtype=float))
        candidate = self._core.preprocess_window(weighted)
        alpha = min(max(float(self.update_alpha), 0.0), 1.0)
        old_template = self.templates[float(selected)]
        self.templates[float(selected)] = (1.0 - alpha) * old_template + alpha * candidate

    def get_state(self) -> dict[str, Any]:
        return {
            "update_alpha": float(self.update_alpha),
            "templates": {f"{freq:g}": matrix.tolist() for freq, matrix in self.templates.items()},
        }

    def set_state(self, state: Optional[dict[str, Any]]) -> None:
        payload = dict(state or {})
        self.update_alpha = float(payload.get("update_alpha", self.update_alpha))
        templates_raw = payload.get("templates", {})
        self.templates = {
            float(key.replace("Hz", "")): np.asarray(value, dtype=float) for key, value in templates_raw.items()
        }


MODEL_ALIASES = {
    "fbcca": "fbcca",
    "legacy_fbcca_202603": "legacy_fbcca_202603",
    "legacy-fbcca-202603": "legacy_fbcca_202603",
    "legacy_fbcca": "legacy_fbcca_202603",
    "fbcca_202603": "legacy_fbcca_202603",
    "fbcca_plain_all8": "fbcca_fixed_all8",
    "fbcca-plain-all8": "fbcca_fixed_all8",
    "fbcca_fixed_all8": "fbcca_fixed_all8",
    "fbcca_cw_all8": "fbcca_cw_all8",
    "fbcca_sw_all8": "fbcca_sw_all8",
    "fbcca_cw_sw_all8": "fbcca_cw_sw_all8",
    "fbcca_cw_sw_trca_shared": "fbcca_cw_sw_trca_shared",
    "cca": "cca",
    "itcca": "itcca",
    "ecca": "ecca",
    "msetcca": "msetcca",
    "trca": "trca",
    "trca_r": "trca_r",
    "trca-r": "trca_r",
    "etrca": "trca_r",
    "etrca_r": "trca_r",
    "etrca-r": "trca_r",
    "sscor": "sscor",
    "tdca": "tdca",
    "tdca_v2": "tdca",
    "tdca-v2": "tdca",
    "fbcca_v1": "fbcca",
    "fbcca-v1": "fbcca",
    "oacca": "oacca",
}


def normalize_model_name(model_name: str) -> str:
    key = str(model_name).strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"unsupported model: {model_name}")
    return MODEL_ALIASES[key]


def parse_model_list(raw: str) -> tuple[str, ...]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("models must contain at least one model name")
    normalized = [normalize_model_name(item) for item in values]
    return tuple(dict.fromkeys(normalized))


def resolve_fbcca_variant_modes(
    model_name: str,
    *,
    channel_weight_mode: Optional[str],
    subband_weight_mode: Optional[str],
    spatial_filter_mode: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    name = normalize_model_name(model_name)
    if name in FBCCA_VARIANT_SPECS:
        spec = FBCCA_VARIANT_SPECS[name]
        return (
            parse_channel_weight_mode(spec.get("channel_weight_mode")),
            parse_subband_weight_mode(spec.get("subband_weight_mode")),
            parse_spatial_filter_mode(spec.get("spatial_filter_mode")),
        )
    if name != "fbcca":
        return None, None, None
    return (
        parse_channel_weight_mode(channel_weight_mode),
        parse_subband_weight_mode(subband_weight_mode),
        parse_spatial_filter_mode(spatial_filter_mode),
    )


def fbcca_weight_learning_requires_all8(
    model_name: str,
    *,
    channel_weight_mode: Optional[str],
    subband_weight_mode: Optional[str],
) -> bool:
    name = normalize_model_name(model_name)
    if name in FBCCA_VARIANT_SPECS:
        return True
    if name != "fbcca":
        return False
    return parse_channel_weight_mode(channel_weight_mode) is not None or (
        parse_subband_weight_mode(subband_weight_mode) not in {None, "chen_fixed"}
    )


def parse_channel_mode_list(raw: str) -> tuple[str, ...]:
    values = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("channel modes must contain at least one value")
    allowed = {"auto", "all8"}
    normalized: list[str] = []
    for value in values:
        if value not in allowed:
            raise ValueError(f"unsupported channel mode: {value}")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def parse_gate_policy(raw: str) -> str:
    value = str(raw).strip().lower()
    if value not in set(DEFAULT_GATE_POLICIES):
        raise ValueError(f"unsupported gate policy: {raw}")
    return value


def parse_metric_scope(raw: str) -> str:
    value = str(raw).strip().lower()
    if value not in set(DEFAULT_METRIC_SCOPES):
        raise ValueError(f"unsupported metric scope: {raw}")
    return value


def parse_decision_time_mode(raw: str) -> str:
    value = str(raw).strip().lower()
    if value not in set(DEFAULT_DECISION_TIME_MODES):
        raise ValueError(f"unsupported decision time mode: {raw}")
    return value


def parse_ranking_policy(raw: str) -> str:
    value = str(raw).strip().lower()
    if value not in set(DEFAULT_RANKING_POLICIES):
        raise ValueError(f"unsupported ranking policy: {raw}")
    return value


def parse_data_policy(raw: str) -> str:
    value = str(raw).strip().lower()
    if value not in set(DEFAULT_DATA_POLICIES):
        raise ValueError(f"unsupported data policy: {raw}")
    return value


def parse_weight_aggregation(raw: str) -> str:
    value = str(raw).strip().lower()
    if value not in set(DEFAULT_WEIGHT_AGGREGATION_MODES):
        raise ValueError(f"unsupported weight aggregation: {raw}")
    return value


def parse_control_state_mode(raw: str) -> str:
    value = str(raw).strip().lower()
    if value not in set(DEFAULT_CONTROL_STATE_MODES):
        raise ValueError(f"unsupported control state mode: {raw}")
    return value


def parse_channel_weight_mode(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"", "none", "off", "disabled"}:
        return None
    if value != "fbcca_diag":
        raise ValueError(f"unsupported channel weight mode: {raw}")
    return value


def parse_subband_weight_mode(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"", "none", "off", "disabled"}:
        return None
    if value not in set(DEFAULT_SUBBAND_WEIGHT_MODES):
        raise ValueError(f"unsupported subband weight mode: {raw}")
    return value


def parse_spatial_filter_mode(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"", "none", "off", "disabled"}:
        return None
    if value not in set(DEFAULT_SPATIAL_FILTER_MODES):
        raise ValueError(f"unsupported spatial filter mode: {raw}")
    if value == "none":
        return None
    return value


def parse_spatial_source_model(raw: Optional[str]) -> str:
    if raw is None:
        return DEFAULT_SPATIAL_SOURCE_MODEL
    value = str(raw).strip().lower()
    if value not in set(DEFAULT_SPATIAL_SOURCE_MODELS):
        raise ValueError(f"unsupported spatial source model: {raw}")
    return value


def parse_spatial_rank_candidates(raw: Optional[str]) -> tuple[int, ...]:
    if raw is None:
        return tuple(DEFAULT_SPATIAL_RANK_CANDIDATES)
    values = [
        int(float(item.strip()))
        for item in str(raw).split(",")
        if str(item).strip()
    ]
    cleaned = sorted({max(1, int(value)) for value in values})
    if not cleaned:
        return tuple(DEFAULT_SPATIAL_RANK_CANDIDATES)
    return tuple(cleaned)


def normalize_channel_weights(
    weights: Optional[Sequence[float]],
    *,
    channels: Optional[int] = None,
    min_value: float = DEFAULT_CHANNEL_WEIGHT_RANGE[0],
    max_value: float = DEFAULT_CHANNEL_WEIGHT_RANGE[1],
) -> Optional[np.ndarray]:
    if weights is None:
        return None
    vector = np.asarray(list(weights), dtype=float).reshape(-1)
    if vector.size == 0:
        return None
    if channels is not None and int(vector.size) != int(channels):
        raise ValueError(f"channel weights length mismatch: got {vector.size}, expected {channels}")
    clipped = np.clip(vector, float(min_value), float(max_value))
    mean_value = float(np.mean(clipped))
    if mean_value <= 1e-12:
        clipped = np.ones_like(clipped, dtype=float)
    else:
        clipped = clipped / mean_value
    return np.asarray(clipped, dtype=float)


def normalize_subband_weights(
    weights: Optional[Sequence[float]],
    *,
    subbands: Optional[int] = None,
    min_value: float = DEFAULT_SUBBAND_WEIGHT_RANGE[0],
) -> Optional[np.ndarray]:
    if weights is None:
        return None
    vector = np.asarray(list(weights), dtype=float).reshape(-1)
    if vector.size == 0:
        return None
    if subbands is not None and int(vector.size) != int(subbands):
        raise ValueError(f"subband weights length mismatch: got {vector.size}, expected {subbands}")
    clipped = np.clip(vector, float(min_value), None)
    total = float(np.sum(clipped))
    if total <= 1e-12:
        clipped = np.ones_like(clipped, dtype=float)
        total = float(np.sum(clipped))
    return np.asarray(clipped / total, dtype=float)


def fbcca_subband_weights_from_ab(
    subband_count: int,
    *,
    a: float = DEFAULT_FBCCA_SUBBAND_WEIGHT_A,
    b: float = DEFAULT_FBCCA_SUBBAND_WEIGHT_B,
) -> np.ndarray:
    weights = np.asarray(
        [(index + 1) ** (-float(a)) + float(b) for index in range(int(subband_count))],
        dtype=float,
    )
    normalized = normalize_subband_weights(weights, subbands=int(subband_count), min_value=0.0)
    if normalized is None:
        return np.ones(int(subband_count), dtype=float) / max(int(subband_count), 1)
    return np.asarray(normalized, dtype=float)


def resolve_fbcca_subband_weight_spec(
    subband_count: int,
    *,
    mode: Optional[str],
    explicit_weights: Optional[Sequence[float]] = None,
    params: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, str, Optional[dict[str, Any]]]:
    resolved_mode = parse_subband_weight_mode(mode) or DEFAULT_SUBBAND_WEIGHT_MODE
    param_payload = dict(params or {})
    if explicit_weights is not None:
        normalized = normalize_subband_weights(explicit_weights, subbands=int(subband_count))
        if normalized is None:
            normalized = fbcca_subband_weights_from_ab(int(subband_count))
        return np.asarray(normalized, dtype=float), str(resolved_mode), (dict(param_payload) or None)
    if resolved_mode == "chen_fixed":
        return (
            fbcca_subband_weights_from_ab(int(subband_count)),
            "chen_fixed",
            {"a": float(DEFAULT_FBCCA_SUBBAND_WEIGHT_A), "b": float(DEFAULT_FBCCA_SUBBAND_WEIGHT_B)},
        )
    if resolved_mode == "chen_ab_subject":
        a_value = float(param_payload.get("a", DEFAULT_FBCCA_SUBBAND_WEIGHT_A))
        b_value = float(param_payload.get("b", DEFAULT_FBCCA_SUBBAND_WEIGHT_B))
        return (
            fbcca_subband_weights_from_ab(int(subband_count), a=a_value, b=b_value),
            "chen_ab_subject",
            {"a": float(a_value), "b": float(b_value)},
        )
    if resolved_mode == "simplex_subject":
        fallback = normalize_subband_weights(
            param_payload.get("weights"),
            subbands=int(subband_count),
        )
        if fallback is None:
            fallback = fbcca_subband_weights_from_ab(int(subband_count))
        return np.asarray(fallback, dtype=float), "simplex_subject", None
    raise ValueError(f"unsupported subband weight mode: {mode}")


def softmax(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0:
        return vector
    shifted = vector - float(np.max(vector))
    exp_values = np.exp(shifted)
    denom = float(np.sum(exp_values))
    if denom <= 1e-12:
        return np.ones_like(vector, dtype=float) / max(vector.size, 1)
    return np.asarray(exp_values / denom, dtype=float)


def create_decoder(
    model_name: str,
    *,
    sampling_rate: int,
    freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    model_params: Optional[dict[str, Any]] = None,
    compute_backend: Optional[str] = None,
    gpu_device: int = DEFAULT_GPU_DEVICE_ID,
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
    gpu_warmup: bool = True,
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE,
) -> BaseSSVEPDecoder:
    name = normalize_model_name(model_name)
    params = dict(model_params or {})
    if name in {
        "fbcca_fixed_all8",
        "fbcca_cw_all8",
        "fbcca_sw_all8",
        "fbcca_cw_sw_all8",
        "fbcca_cw_sw_trca_shared",
    }:
        params["_decoder_model_name"] = name
        if name == "fbcca_fixed_all8":
            params.pop("channel_weight_mode", None)
            params.pop("channel_weights", None)
            params.pop("spatial_filter_mode", None)
            params.pop("spatial_filter_rank", None)
            params.pop("spatial_filter_state", None)
            params["subband_weight_mode"] = "chen_fixed"
            params.pop("subband_weights", None)
            params.pop("subband_weight_params", None)
        elif name == "fbcca_cw_all8":
            params["channel_weight_mode"] = "fbcca_diag"
            params["subband_weight_mode"] = "chen_fixed"
            params.pop("spatial_filter_mode", None)
            params.pop("spatial_filter_rank", None)
            params.pop("spatial_filter_state", None)
            params.pop("subband_weights", None)
            params.pop("subband_weight_params", None)
        elif name == "fbcca_sw_all8":
            params.pop("channel_weight_mode", None)
            params.pop("channel_weights", None)
            params["subband_weight_mode"] = "chen_ab_subject"
            params.pop("spatial_filter_mode", None)
            params.pop("spatial_filter_rank", None)
            params.pop("spatial_filter_state", None)
        elif name == "fbcca_cw_sw_all8":
            params["channel_weight_mode"] = "fbcca_diag"
            params["subband_weight_mode"] = "chen_ab_subject"
            params.pop("spatial_filter_mode", None)
            params.pop("spatial_filter_rank", None)
            params.pop("spatial_filter_state", None)
        elif name == "fbcca_cw_sw_trca_shared":
            params["channel_weight_mode"] = "fbcca_diag"
            params["subband_weight_mode"] = "chen_ab_subject"
            params["spatial_filter_mode"] = "trca_shared"
        name = "fbcca"
    if compute_backend is not None:
        params["compute_backend"] = parse_compute_backend_name(compute_backend)
    params.setdefault("compute_backend", DEFAULT_COMPUTE_BACKEND_NAME)
    params.setdefault("gpu_device", int(gpu_device))
    params.setdefault("gpu_precision", parse_gpu_precision(gpu_precision))
    params.setdefault("gpu_warmup", bool(gpu_warmup))
    params.setdefault("gpu_cache_policy", parse_gpu_cache_policy(gpu_cache_policy))
    if name == "fbcca":
        return FBCCADecoder(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
        )
    if name == "legacy_fbcca_202603":
        return LegacyFBCCA202603Decoder(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
        )
    if name == "cca":
        return CCADecoder(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
        )
    if name in {"itcca", "ecca", "msetcca"}:
        return TemplateCCADecoder(
            variant=name,
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
        )
    if name == "trca":
        return TRCABasedDecoder(
            model_name=name,
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
            use_filterbank=False,
            ensemble=False,
            delay_steps=1,
        )
    if name == "trca_r":
        return TRCABasedDecoder(
            model_name=name,
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
            use_filterbank=True,
            ensemble=True,
            delay_steps=1,
        )
    if name == "sscor":
        return SSCORDecoder(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
        )
    if name == "tdca":
        return TDCADecoder(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
        )
    if name == "oacca":
        return OACCADecoder(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
        )
    raise ValueError(f"unsupported model: {model_name}")


def load_decoder_from_profile(
    profile: ThresholdProfile,
    *,
    sampling_rate: int,
    compute_backend: Optional[str] = None,
    gpu_device: int = DEFAULT_GPU_DEVICE_ID,
    gpu_precision: Optional[str] = None,
    gpu_warmup: bool = True,
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE,
) -> BaseSSVEPDecoder:
    if profile.channel_weights is not None and profile.eeg_channels is not None:
        if len(profile.channel_weights) != len(profile.eeg_channels):
            raise ValueError(
                "profile channel_weights mismatch: "
                f"weights={len(profile.channel_weights)} eeg_channels={len(profile.eeg_channels)}"
            )
    model_params = dict(profile.model_params or {})
    if profile.channel_weight_mode is not None and "channel_weight_mode" not in model_params:
        model_params["channel_weight_mode"] = str(profile.channel_weight_mode)
    if profile.channel_weights is not None and "channel_weights" not in model_params:
        model_params["channel_weights"] = [float(value) for value in profile.channel_weights]
    if profile.subband_weight_mode is not None and "subband_weight_mode" not in model_params:
        model_params["subband_weight_mode"] = str(profile.subband_weight_mode)
    if profile.subband_weights is not None and "subband_weights" not in model_params:
        model_params["subband_weights"] = [float(value) for value in profile.subband_weights]
    if profile.subband_weight_params is not None and "subband_weight_params" not in model_params:
        model_params["subband_weight_params"] = dict(profile.subband_weight_params)
    if profile.spatial_filter_mode is not None and "spatial_filter_mode" not in model_params:
        model_params["spatial_filter_mode"] = str(profile.spatial_filter_mode)
    if profile.spatial_filter_rank is not None and "spatial_filter_rank" not in model_params:
        model_params["spatial_filter_rank"] = int(profile.spatial_filter_rank)
    if profile.spatial_filter_state is not None and "spatial_filter_state" not in model_params:
        model_params["spatial_filter_state"] = dict(profile.spatial_filter_state)
    if profile.joint_weight_training is not None and "joint_weight_training" not in model_params:
        model_params["joint_weight_training"] = dict(profile.joint_weight_training)
    requested_backend = (
        parse_compute_backend_name(compute_backend)
        if compute_backend is not None
        else parse_compute_backend_name(profile.runtime_backend_preference or model_params.get("compute_backend"))
    )
    requested_precision = parse_gpu_precision(
        gpu_precision or profile.runtime_precision_preference or model_params.get("gpu_precision")
    )
    decoder = create_decoder(
        profile.model_name,
        sampling_rate=sampling_rate,
        freqs=profile.freqs,
        win_sec=profile.win_sec,
        step_sec=profile.step_sec,
        model_params=model_params,
        compute_backend=requested_backend,
        gpu_device=int(gpu_device),
        gpu_precision=requested_precision,
        gpu_warmup=bool(gpu_warmup),
        gpu_cache_policy=parse_gpu_cache_policy(gpu_cache_policy),
    )
    state = None
    if isinstance(profile.model_params, dict):
        state = profile.model_params.get("state")
    if state is not None:
        decoder.set_state(state)
    if decoder.requires_fit and state is None:
        raise RuntimeError(
            f"profile model '{profile.model_name}' requires fitted state; run benchmark and save a profile "
            f"that includes model_params.state"
        )
    if profile.subband_weights is not None and hasattr(decoder, "engine") and hasattr(decoder.engine, "subband_sos"):
        subband_count = len(getattr(decoder.engine, "subband_sos", []) or [])
        if subband_count and len(profile.subband_weights) != int(subband_count):
            raise ValueError(
                "profile subband_weights mismatch: "
                f"weights={len(profile.subband_weights)} subbands={int(subband_count)}"
            )
    return decoder


class AsyncDecisionGate:
    def __init__(
        self,
        *,
        enter_score_th: float,
        enter_ratio_th: float,
        enter_margin_th: float,
        exit_score_th: float,
        exit_ratio_th: float,
        min_enter_windows: int = 2,
        min_exit_windows: int = 2,
        gate_policy: str = DEFAULT_GATE_POLICY,
        min_switch_windows: int = 1,
        switch_enter_score_th: Optional[float] = None,
        switch_enter_ratio_th: Optional[float] = None,
        switch_enter_margin_th: Optional[float] = None,
        enter_log_lr_th: Optional[float] = None,
        exit_log_lr_th: Optional[float] = None,
        control_feature_means: Optional[dict[str, float]] = None,
        control_feature_stds: Optional[dict[str, float]] = None,
        idle_feature_means: Optional[dict[str, float]] = None,
        idle_feature_stds: Optional[dict[str, float]] = None,
        dynamic_stop_enabled: bool = False,
        dynamic_stop_alpha: float = DEFAULT_DYNAMIC_STOP_ALPHA,
        enter_acc_th: Optional[float] = None,
        exit_acc_th: Optional[float] = None,
        control_state_mode: str = DEFAULT_CONTROL_STATE_MODE,
        frequency_specific_thresholds: Optional[dict[str, Any]] = None,
    ) -> None:
        self.enter_score_th = float(enter_score_th)
        self.enter_ratio_th = float(enter_ratio_th)
        self.enter_margin_th = float(enter_margin_th)
        self.exit_score_th = float(exit_score_th)
        self.exit_ratio_th = float(exit_ratio_th)
        self.min_enter_windows = max(1, int(min_enter_windows))
        self.min_exit_windows = max(1, int(min_exit_windows))
        self.gate_policy = parse_gate_policy(gate_policy)
        self.min_switch_windows = max(1, int(min_switch_windows))
        self.switch_enter_score_th = (
            0.95 * self.enter_score_th
            if switch_enter_score_th is None
            else float(switch_enter_score_th)
        )
        self.switch_enter_ratio_th = (
            self.enter_ratio_th
            if switch_enter_ratio_th is None
            else float(switch_enter_ratio_th)
        )
        self.switch_enter_margin_th = (
            0.80 * self.enter_margin_th
            if switch_enter_margin_th is None
            else float(switch_enter_margin_th)
        )
        self.enter_log_lr_th = None if enter_log_lr_th is None else float(enter_log_lr_th)
        self.exit_log_lr_th = None if exit_log_lr_th is None else float(exit_log_lr_th)
        self.control_feature_means = control_feature_means
        self.control_feature_stds = control_feature_stds
        self.idle_feature_means = idle_feature_means
        self.idle_feature_stds = idle_feature_stds
        self.dynamic_stop_enabled = bool(dynamic_stop_enabled)
        self.dynamic_stop_alpha = min(max(float(dynamic_stop_alpha), 0.0), 1.0)
        self.enter_acc_th = None if enter_acc_th is None else float(enter_acc_th)
        self.exit_acc_th = None if exit_acc_th is None else float(exit_acc_th)
        self.control_state_mode = parse_control_state_mode(control_state_mode)
        self.frequency_specific_thresholds = normalize_frequency_specific_thresholds(frequency_specific_thresholds)
        self._control_means_vec = (
            np.array([float(control_feature_means[name]) for name in MODEL_FEATURE_NAMES], dtype=float)
            if control_feature_means is not None
            else None
        )
        self._control_stds_vec = (
            np.array([float(control_feature_stds[name]) for name in MODEL_FEATURE_NAMES], dtype=float)
            if control_feature_stds is not None
            else None
        )
        self._idle_means_vec = (
            np.array([float(idle_feature_means[name]) for name in MODEL_FEATURE_NAMES], dtype=float)
            if idle_feature_means is not None
            else None
        )
        self._idle_stds_vec = (
            np.array([float(idle_feature_stds[name]) for name in MODEL_FEATURE_NAMES], dtype=float)
            if idle_feature_stds is not None
            else None
        )
        self.reset()

    @classmethod
    def from_profile(cls, profile: ThresholdProfile) -> "AsyncDecisionGate":
        dynamic_payload = dict(profile.dynamic_stop or {})
        return cls(
            enter_score_th=profile.enter_score_th,
            enter_ratio_th=profile.enter_ratio_th,
            enter_margin_th=profile.enter_margin_th,
            exit_score_th=profile.exit_score_th,
            exit_ratio_th=profile.exit_ratio_th,
            min_enter_windows=profile.min_enter_windows,
            min_exit_windows=profile.min_exit_windows,
            gate_policy=profile.gate_policy,
            min_switch_windows=profile.min_switch_windows,
            switch_enter_score_th=profile.switch_enter_score_th,
            switch_enter_ratio_th=profile.switch_enter_ratio_th,
            switch_enter_margin_th=profile.switch_enter_margin_th,
            enter_log_lr_th=profile.enter_log_lr_th,
            exit_log_lr_th=profile.exit_log_lr_th,
            control_feature_means=profile.control_feature_means,
            control_feature_stds=profile.control_feature_stds,
            idle_feature_means=profile.idle_feature_means,
            idle_feature_stds=profile.idle_feature_stds,
            dynamic_stop_enabled=bool(dynamic_payload.get("enabled", False)),
            dynamic_stop_alpha=float(dynamic_payload.get("alpha", DEFAULT_DYNAMIC_STOP_ALPHA)),
            enter_acc_th=dynamic_payload.get("enter_acc_th"),
            exit_acc_th=dynamic_payload.get("exit_acc_th"),
            control_state_mode=profile.control_state_mode,
            frequency_specific_thresholds=profile.frequency_specific_thresholds,
        )

    def reset(self) -> None:
        self.state = "idle"
        self._candidate_freq: Optional[float] = None
        self._selected_freq: Optional[float] = None
        self._switch_candidate_freq: Optional[float] = None
        self._candidate_windows = 0
        self._switch_candidate_windows = 0
        self._support_windows = 0
        self._exit_windows = 0
        self._acc_log_lr = 0.0

    def _threshold_payload_for_features(
        self,
        features: dict[str, Any],
        *,
        selected_fallback: bool = False,
    ) -> dict[str, Any]:
        if not self.frequency_specific_thresholds:
            return {}
        if self.control_state_mode not in {"frequency-specific-threshold", "frequency-specific-logistic"}:
            return {}
        freq_value = features.get("pred_freq")
        if selected_fallback and self._selected_freq is not None:
            freq_value = self._selected_freq
        if freq_value is None:
            return {}
        key = _frequency_key(freq_value)
        if key is None:
            return {}
        payload = self.frequency_specific_thresholds.get(key)
        return dict(payload) if isinstance(payload, dict) else {}

    def _enter_pass(self, features: dict[str, Any]) -> bool:
        payload = self._threshold_payload_for_features(features)
        score_th = float(payload.get("enter_score_th", self.enter_score_th))
        ratio_th = float(payload.get("enter_ratio_th", self.enter_ratio_th))
        margin_th = float(payload.get("enter_margin_th", self.enter_margin_th))
        enter_log_lr_th = payload.get("enter_log_lr_th", self.enter_log_lr_th)
        legacy_pass = (
            float(features["top1_score"]) >= score_th
            and float(features["ratio"]) >= ratio_th
            and float(features["margin"]) >= margin_th
        )
        if enter_log_lr_th is None:
            pass_by_log_lr = True
        else:
            control_log_lr = features.get("control_log_lr")
            pass_by_log_lr = control_log_lr is not None and float(control_log_lr) >= float(enter_log_lr_th)
        if not (legacy_pass and pass_by_log_lr):
            return False
        if self.dynamic_stop_enabled and self.enter_acc_th is not None:
            acc_log_lr = features.get("acc_log_lr")
            return acc_log_lr is not None and float(acc_log_lr) >= self.enter_acc_th
        return True

    def _switch_pass(self, features: dict[str, Any]) -> bool:
        if self.gate_policy != "speed":
            return False
        if self._selected_freq is None:
            return False
        pred_freq = features.get("pred_freq")
        if pred_freq is None:
            return False
        if abs(float(pred_freq) - float(self._selected_freq)) <= 1e-8:
            return False
        payload = self._threshold_payload_for_features(features)
        switch_score_th = float(payload.get("switch_enter_score_th", self.switch_enter_score_th))
        switch_ratio_th = float(payload.get("switch_enter_ratio_th", self.switch_enter_ratio_th))
        switch_margin_th = float(payload.get("switch_enter_margin_th", self.switch_enter_margin_th))
        return (
            float(features["top1_score"]) >= switch_score_th
            and float(features["ratio"]) >= switch_ratio_th
            and float(features["margin"]) >= switch_margin_th
        )

    def _exit_fail(self, features: dict[str, Any]) -> bool:
        if self._selected_freq is None:
            return True
        payload = self._threshold_payload_for_features(features, selected_fallback=True)
        exit_score_th = float(payload.get("exit_score_th", self.exit_score_th))
        exit_ratio_th = float(payload.get("exit_ratio_th", self.exit_ratio_th))
        exit_log_lr_th = payload.get("exit_log_lr_th", self.exit_log_lr_th)
        pred_freq = features.get("pred_freq")
        if pred_freq is None or abs(float(pred_freq) - float(self._selected_freq)) > 1e-8:
            return True
        if float(features["top1_score"]) < exit_score_th:
            return True
        if float(features["ratio"]) < exit_ratio_th:
            return True
        if exit_log_lr_th is not None:
            control_log_lr = features.get("control_log_lr")
            if control_log_lr is None or float(control_log_lr) < float(exit_log_lr_th):
                return True
        if self.dynamic_stop_enabled and self.exit_acc_th is not None:
            acc_log_lr = features.get("acc_log_lr")
            if acc_log_lr is None or float(acc_log_lr) <= self.exit_acc_th:
                return True
        return False

    def update(self, features: dict[str, Any]) -> dict[str, Any]:
        features = dict(features)
        if (
            features.get("control_log_lr") is None
            and self._control_means_vec is not None
            and self._control_stds_vec is not None
            and self._idle_means_vec is not None
            and self._idle_stds_vec is not None
        ):
            features["control_log_lr"] = gaussian_log_likelihood_ratio(
                features,
                control_means=self._control_means_vec,
                control_stds=self._control_stds_vec,
                idle_means=self._idle_means_vec,
                idle_stds=self._idle_stds_vec,
            )
        control_log_lr = features.get("control_log_lr")
        if control_log_lr is not None:
            value = float(control_log_lr)
            if self.dynamic_stop_enabled:
                self._acc_log_lr = self.dynamic_stop_alpha * float(self._acc_log_lr) + value
                features["acc_log_lr"] = float(self._acc_log_lr)
            else:
                self._acc_log_lr = value
                features["acc_log_lr"] = float(self._acc_log_lr)
            if value >= 0.0:
                exp_neg = math.exp(-value)
                features["control_confidence"] = float(1.0 / (1.0 + exp_neg))
            else:
                exp_pos = math.exp(value)
                features["control_confidence"] = float(exp_pos / (1.0 + exp_pos))
        else:
            features["control_confidence"] = None
            features["acc_log_lr"] = None
            self._acc_log_lr = 0.0

        pred_freq = features.get("pred_freq")
        enter_pass = pred_freq is not None and self._enter_pass(features)
        switch_pass = pred_freq is not None and self._switch_pass(features)

        if self.state == "idle":
            if enter_pass:
                self.state = "candidate"
                self._candidate_freq = float(pred_freq)
                self._candidate_windows = 1
                self._support_windows = 1
                if self.min_enter_windows <= 1:
                    self.state = "selected"
                    self._selected_freq = float(pred_freq)
                    self._exit_windows = 0
            else:
                self._candidate_freq = None
                self._candidate_windows = 0
                self._support_windows = 0
            self._switch_candidate_freq = None
            self._switch_candidate_windows = 0
        elif self.state == "candidate":
            if enter_pass and abs(float(pred_freq) - float(self._candidate_freq)) < 1e-8:
                self._candidate_windows += 1
                self._support_windows = self._candidate_windows
                if self._candidate_windows >= self.min_enter_windows:
                    self.state = "selected"
                    self._selected_freq = float(pred_freq)
                    self._exit_windows = 0
            elif enter_pass:
                self._candidate_freq = float(pred_freq)
                self._candidate_windows = 1
                self._support_windows = 1
            else:
                self.state = "idle"
                self._candidate_freq = None
                self._candidate_windows = 0
                self._support_windows = 0
            self._switch_candidate_freq = None
            self._switch_candidate_windows = 0
        else:
            if switch_pass:
                if self._switch_candidate_freq is None or abs(float(pred_freq) - float(self._switch_candidate_freq)) > 1e-8:
                    self._switch_candidate_freq = float(pred_freq)
                    self._switch_candidate_windows = 1
                else:
                    self._switch_candidate_windows += 1
                self._exit_windows = 0
                if self._switch_candidate_windows >= self.min_switch_windows:
                    self._selected_freq = float(pred_freq)
                    self._switch_candidate_freq = None
                    self._switch_candidate_windows = 0
                    self._support_windows = 1
            else:
                self._switch_candidate_freq = None
                self._switch_candidate_windows = 0
                if self._exit_fail(features):
                    self._exit_windows += 1
                    if self._exit_windows >= self.min_exit_windows:
                        self.state = "idle"
                        self._selected_freq = None
                        self._candidate_freq = None
                        self._candidate_windows = 0
                        self._support_windows = 0
                        self._exit_windows = 0
                else:
                    self._exit_windows = 0
                    self._support_windows += 1

        selected_freq = self._selected_freq if self.state == "selected" else None
        stable_windows = self._support_windows if self.state != "idle" else 0
        return {
            **features,
            "state": self.state,
            "selected_freq": selected_freq,
            "stable_windows": int(stable_windows),
            "enter_pass": bool(enter_pass),
            "switch_pass": bool(switch_pass),
            "switch_candidate_freq": self._switch_candidate_freq,
            "switch_candidate_windows": int(self._switch_candidate_windows),
            "exit_windows": int(self._exit_windows),
        }


def _trial_row_key(row: dict[str, Any], fallback_index: int) -> tuple[str, int]:
    trial_id = row.get("trial_id")
    if trial_id is None or int(trial_id) < 0:
        return str(row.get("label", "")), -1_000_000 - int(fallback_index)
    return str(row.get("label", "")), int(trial_id)


def split_feature_rows_by_trial(
    rows: Sequence[dict[str, Any]],
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trial_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        key = _trial_row_key(row, index)
        trial_groups.setdefault(key, []).append(row)

    label_to_trials: dict[str, list[int]] = {}
    for label, trial_id in trial_groups:
        label_to_trials.setdefault(label, []).append(trial_id)

    rng = random.Random(int(seed))
    validation_trial_ids: set[int] = set()
    split_counts: dict[str, dict[str, int]] = {}
    for label, trial_ids in label_to_trials.items():
        ids = sorted(set(int(item) for item in trial_ids))
        rng.shuffle(ids)
        if len(ids) <= 1:
            val_count = 0
        else:
            proposed = int(round(len(ids) * float(validation_fraction)))
            val_count = max(1, proposed)
            val_count = min(val_count, len(ids) - 1)
        validation_trial_ids.update(ids[:val_count])
        split_counts[label] = {
            "train_trials": int(len(ids) - val_count),
            "validation_trials": int(val_count),
        }

    fit_rows = [row for index, row in enumerate(rows) if _trial_row_key(row, index)[1] not in validation_trial_ids]
    validation_rows = [row for index, row in enumerate(rows) if _trial_row_key(row, index)[1] in validation_trial_ids]
    return fit_rows, validation_rows, {
        "validation_fraction": float(validation_fraction),
        "seed": int(seed),
        "split_counts": split_counts,
    }


def summarize_profile_quality(
    feature_rows: Sequence[dict[str, Any]],
    profile: ThresholdProfile,
) -> dict[str, float]:
    threshold_gate = AsyncDecisionGate.from_profile(profile)

    def is_idle_row(row: dict[str, Any]) -> bool:
        return row.get("expected_freq") is None

    def passes(row: dict[str, Any]) -> bool:
        control_log_lr = profile_log_lr(profile, row)
        payload = threshold_gate._threshold_payload_for_features(row)
        enter_score_th = float(payload.get("enter_score_th", profile.enter_score_th))
        enter_ratio_th = float(payload.get("enter_ratio_th", profile.enter_ratio_th))
        enter_margin_th = float(payload.get("enter_margin_th", profile.enter_margin_th))
        enter_log_lr_th = payload.get("enter_log_lr_th", profile.enter_log_lr_th)
        log_lr_pass = enter_log_lr_th is None or (
            control_log_lr is not None and float(control_log_lr) >= float(enter_log_lr_th)
        )
        return (
            float(row["top1_score"]) >= enter_score_th
            and float(row["ratio"]) >= enter_ratio_th
            and float(row["margin"]) >= enter_margin_th
            and log_lr_pass
        )

    rows = list(feature_rows)
    non_idle_rows = [row for row in rows if not is_idle_row(row)]
    control_rows = [row for row in non_idle_rows if bool(row.get("correct"))]
    idle_rows = [row for row in rows if is_idle_row(row)]
    raw_accuracy = (
        float(sum(1 for row in non_idle_rows if bool(row.get("correct")))) / float(len(non_idle_rows))
        if non_idle_rows
        else 0.0
    )
    control_recall = (
        float(sum(1 for row in control_rows if passes(row))) / float(len(control_rows))
        if control_rows
        else 0.0
    )
    idle_false_positive = (
        float(sum(1 for row in idle_rows if passes(row))) / float(len(idle_rows))
        if idle_rows
        else 0.0
    )

    grouped_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        grouped_rows.setdefault(_trial_row_key(row, index), []).append(dict(row))

    non_idle_trials = 0
    idle_trials = 0
    successful_trials = 0
    idle_fp_trials = 0
    latency_values: list[float] = []
    selected_correct_windows = 0
    selected_wrong_windows = 0
    selected_idle_windows = 0
    non_idle_window_count = 0
    idle_window_count = 0

    for (label, _trial_id), trial_rows in grouped_rows.items():
        sorted_rows = sorted(trial_rows, key=lambda row: int(row.get("window_index", 0)))
        gate = AsyncDecisionGate.from_profile(profile)
        expected_freq = sorted_rows[0].get("expected_freq")
        is_idle_trial = expected_freq is None
        any_correct = False
        any_idle_selected = False
        first_latency: Optional[float] = None

        for row in sorted_rows:
            decision = gate.update(row)
            selected_freq = decision.get("selected_freq")
            if is_idle_trial:
                idle_window_count += 1
                if selected_freq is not None:
                    selected_idle_windows += 1
                    any_idle_selected = True
                continue

            non_idle_window_count += 1
            if selected_freq is None:
                continue
            if expected_freq is not None and abs(float(selected_freq) - float(expected_freq)) < 1e-8:
                selected_correct_windows += 1
                any_correct = True
                if first_latency is None:
                    first_latency = profile.win_sec + int(row.get("window_index", 0)) * profile.step_sec
            else:
                selected_wrong_windows += 1

        if is_idle_trial:
            idle_trials += 1
            if any_idle_selected:
                idle_fp_trials += 1
        else:
            non_idle_trials += 1
            if any_correct:
                successful_trials += 1
                if first_latency is not None:
                    latency_values.append(float(first_latency))

    window_end_to_end_recall = (
        float(selected_correct_windows) / float(non_idle_window_count) if non_idle_window_count else 0.0
    )
    window_wrong_target_rate = (
        float(selected_wrong_windows) / float(non_idle_window_count) if non_idle_window_count else 0.0
    )
    idle_window_selected_rate = (
        float(selected_idle_windows) / float(idle_window_count) if idle_window_count else 0.0
    )
    trial_success_rate = float(successful_trials) / float(non_idle_trials) if non_idle_trials else 0.0
    idle_trial_false_positive = float(idle_fp_trials) / float(idle_trials) if idle_trials else 0.0
    mean_detection_latency_sec = float(np.mean(latency_values)) if latency_values else float("nan")

    return {
        "non_idle_windows": float(len(non_idle_rows)),
        "control_windows": float(len(control_rows)),
        "idle_windows": float(len(idle_rows)),
        "raw_accuracy": raw_accuracy,
        "control_recall": control_recall,
        "idle_false_positive": idle_false_positive,
        "window_end_to_end_recall": window_end_to_end_recall,
        "window_wrong_target_rate": window_wrong_target_rate,
        "idle_window_selected_rate": idle_window_selected_rate,
        "non_idle_trials": float(non_idle_trials),
        "idle_trials": float(idle_trials),
        "trial_success_rate": trial_success_rate,
        "idle_trial_false_positive": idle_trial_false_positive,
        "mean_detection_latency_sec": mean_detection_latency_sec,
    }


def format_profile_quality_summary(summary: dict[str, float]) -> str:
    latency = summary.get("mean_detection_latency_sec", float("nan"))
    latency_text = "n/a" if math.isnan(float(latency)) else f"{float(latency):.2f}s"
    return (
        f"Non-idle windows: {int(summary['non_idle_windows'])}\\n"
        f"Usable control windows: {int(summary['control_windows'])}\\n"
        f"Idle windows: {int(summary['idle_windows'])}\\n"
        f"Raw window accuracy: {summary['raw_accuracy'] * 100.0:.1f}%\\n"
        f"Control recall: {summary['control_recall'] * 100.0:.1f}%\\n"
        f"Idle false positive: {summary['idle_false_positive'] * 100.0:.1f}%\\n"
        f"End-to-end window recall: {summary['window_end_to_end_recall'] * 100.0:.1f}%\\n"
        f"Wrong-target activation: {summary['window_wrong_target_rate'] * 100.0:.1f}%\\n"
        f"Non-idle trial success: {summary['trial_success_rate'] * 100.0:.1f}%\\n"
        f"Idle trial false trigger: {summary['idle_trial_false_positive'] * 100.0:.1f}%\\n"
        f"Mean detection latency: {latency_text}"
    )


def _latency_or_penalty(value: float, *, penalty: float = 1_000_000.0) -> float:
    numeric = float(value)
    if math.isnan(numeric) or not np.isfinite(numeric):
        return float(penalty)
    return numeric


def evaluate_profile_on_feature_rows(
    feature_rows: Sequence[dict[str, Any]],
    profile: ThresholdProfile,
) -> dict[str, float]:
    rows = [dict(row) for row in feature_rows]
    if not rows:
        return {
            "idle_fp_per_min": 0.0,
            "control_recall": 0.0,
            "switch_detect_rate": 0.0,
            "switch_latency_s": float("inf"),
            "release_latency_s": float("inf"),
            "detection_latency_s": float("inf"),
            "idle_windows": 0.0,
            "control_trials": 0.0,
            "switch_trials": 0.0,
            "release_trials": 0.0,
        }

    grouped_rows: dict[tuple[str, int], list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        key = _trial_row_key(row, index)
        grouped_rows.setdefault(key, []).append((index, row))

    trial_entries: list[tuple[int, tuple[str, int], list[dict[str, Any]]]] = []
    for key, values in grouped_rows.items():
        first_index = min(int(item[0]) for item in values)
        sorted_rows = [dict(item[1]) for item in sorted(values, key=lambda item: int(item[1].get("window_index", 0)))]
        trial_entries.append((first_index, key, sorted_rows))
    trial_entries.sort(key=lambda item: item[0])

    gate = AsyncDecisionGate.from_profile(profile)
    gate.reset()
    idle_windows = 0
    idle_selected_windows = 0
    idle_selected_events = 0
    idle_duration_sec = 0.0
    control_trials = 0
    control_detected_trials = 0
    switch_trials = 0
    switch_detected_trials = 0
    switch_detected_trials_at_2p8s = 0
    release_trials = 0
    release_detected_trials = 0
    detection_latencies: list[float] = []
    switch_latencies: list[float] = []
    release_latencies: list[float] = []
    control_detected_trials_at_2s = 0
    control_detected_trials_at_3s = 0
    last_control_freq: Optional[float] = None
    previous_trial_expected_freq: Optional[float] = None
    has_explicit_switch = any(
        row.get("expected_freq") is not None and str(row.get("label", "")).startswith("switch_to_")
        for row in rows
    )
    selected_active_prev = False

    for _index, (_label, _trial_id), trial_rows in trial_entries:
        if not trial_rows:
            continue
        expected_freq = trial_rows[0].get("expected_freq")
        expected_numeric = None if expected_freq is None else float(expected_freq)
        first_correct_latency: Optional[float] = None
        first_release_latency: Optional[float] = None

        last_window_index = int(trial_rows[-1].get("window_index", max(len(trial_rows) - 1, 0)))
        trial_duration = float(profile.win_sec + max(last_window_index, 0) * profile.step_sec)
        penalty_latency = float(trial_duration + profile.win_sec)

        for row in trial_rows:
            decision = gate.update(row)
            selected = decision.get("selected_freq")
            selected_active = selected is not None
            window_index = int(row.get("window_index", 0))
            latency_value = float(profile.win_sec + window_index * profile.step_sec)

            if expected_numeric is None:
                idle_windows += 1
                if selected_active:
                    idle_selected_windows += 1
                    if not selected_active_prev:
                        idle_selected_events += 1
                if previous_trial_expected_freq is not None and first_release_latency is None and selected is None:
                    first_release_latency = latency_value
                selected_active_prev = selected_active
                continue

            if (
                first_correct_latency is None
                and selected is not None
                and abs(float(selected) - float(expected_numeric)) < 1e-8
            ):
                first_correct_latency = latency_value
            selected_active_prev = selected_active

        if expected_numeric is None:
            idle_duration_sec += float(max(trial_duration, 0.0))
            if previous_trial_expected_freq is not None:
                release_trials += 1
                if first_release_latency is None:
                    release_latencies.append(penalty_latency)
                else:
                    release_detected_trials += 1
                    release_latencies.append(float(first_release_latency))
            previous_trial_expected_freq = None
            continue

        control_trials += 1
        if first_correct_latency is not None:
            control_detected_trials += 1
            detection_latencies.append(float(first_correct_latency))
            if float(first_correct_latency) <= float(DEFAULT_CONTROL_RECALL_AT_2S_DEADLINE):
                control_detected_trials_at_2s += 1
            if float(first_correct_latency) <= float(DEFAULT_CONTROL_RECALL_AT_3S_DEADLINE):
                control_detected_trials_at_3s += 1

        is_switch_trial = False
        if has_explicit_switch:
            is_switch_trial = str(trial_rows[0].get("label", "")).startswith("switch_to_")
        elif last_control_freq is not None and abs(float(last_control_freq) - float(expected_numeric)) > 1e-8:
            is_switch_trial = True
        if is_switch_trial:
            switch_trials += 1
            if first_correct_latency is None:
                switch_latencies.append(penalty_latency)
            else:
                switch_detected_trials += 1
                switch_latencies.append(float(first_correct_latency))
                if float(first_correct_latency) <= float(DEFAULT_SWITCH_DETECT_AT_2P8S_DEADLINE):
                    switch_detected_trials_at_2p8s += 1
        last_control_freq = float(expected_numeric)
        previous_trial_expected_freq = float(expected_numeric)

    idle_minutes = float(max(idle_duration_sec, 0.0)) / 60.0
    idle_fp_per_min = float(idle_selected_events / idle_minutes) if idle_minutes > 1e-12 else 0.0
    idle_selected_windows_per_min = float(idle_selected_windows / idle_minutes) if idle_minutes > 1e-12 else 0.0
    control_recall = float(control_detected_trials / control_trials) if control_trials else 0.0
    switch_detect_rate = float(switch_detected_trials / switch_trials) if switch_trials else 0.0
    switch_detect_rate_at_2p8s = (
        float(switch_detected_trials_at_2p8s / switch_trials) if switch_trials else 0.0
    )
    release_detect_rate = float(release_detected_trials / release_trials) if release_trials else 0.0
    control_recall_at_2s = float(control_detected_trials_at_2s / control_trials) if control_trials else 0.0
    control_recall_at_3s = float(control_detected_trials_at_3s / control_trials) if control_trials else 0.0
    switch_latency = float(np.median(np.asarray(switch_latencies, dtype=float))) if switch_latencies else float("inf")
    release_latency = (
        float(np.median(np.asarray(release_latencies, dtype=float))) if release_latencies else float("inf")
    )
    detection_latency = (
        float(np.median(np.asarray(detection_latencies, dtype=float))) if detection_latencies else float("inf")
    )
    return {
        "idle_fp_per_min": idle_fp_per_min,
        "idle_selected_windows_per_min": idle_selected_windows_per_min,
        "control_recall": control_recall,
        "control_recall_at_2s": control_recall_at_2s,
        "control_recall_at_3s": control_recall_at_3s,
        "switch_detect_rate": switch_detect_rate,
        "switch_detect_rate_at_2.8s": switch_detect_rate_at_2p8s,
        "release_detect_rate": release_detect_rate,
        "switch_latency_s": switch_latency,
        "release_latency_s": release_latency,
        "detection_latency_s": detection_latency,
        "idle_windows": float(idle_windows),
        "idle_selected_windows": float(idle_selected_windows),
        "idle_selected_events": float(idle_selected_events),
        "idle_fp_event_count": float(idle_selected_events),
        "idle_time_sec": float(idle_duration_sec),
        "idle_time_min": float(idle_minutes),
        "control_trials": float(control_trials),
        "switch_trials": float(switch_trials),
        "switch_detected_trials_at_2.8s": float(switch_detected_trials_at_2p8s),
        "release_trials": float(release_trials),
    }


def _balanced_gate_objective(metrics: dict[str, float]) -> tuple[float, float, float, float, float, float, float, float]:
    idle_fp = float(metrics.get("idle_fp_per_min", float("inf")))
    control_recall = float(metrics.get("control_recall", 0.0))
    switch_latency = _latency_or_penalty(float(metrics.get("switch_latency_s", float("inf"))))
    release_latency = _latency_or_penalty(float(metrics.get("release_latency_s", float("inf"))))
    idle_violation = 1.0 if idle_fp > float(DEFAULT_BALANCED_IDLE_FP_MAX) else 0.0
    recall_violation = 1.0 if control_recall < float(DEFAULT_BALANCED_MIN_CONTROL_RECALL) else 0.0
    cost = (
        float(DEFAULT_BALANCED_OBJECTIVE_IDLE_WEIGHT) * idle_fp
        + float(DEFAULT_BALANCED_OBJECTIVE_RECALL_WEIGHT) * (1.0 - control_recall)
        + float(DEFAULT_BALANCED_OBJECTIVE_SWITCH_WEIGHT) * switch_latency
        + float(DEFAULT_BALANCED_OBJECTIVE_RELEASE_WEIGHT) * release_latency
    )
    return (
        idle_violation + recall_violation,
        idle_violation,
        recall_violation,
        float(cost),
        idle_fp,
        -control_recall,
        switch_latency,
        release_latency,
    )


def _speed_gate_objective(metrics: dict[str, float]) -> tuple[float, float, float, float, float, float, float, float]:
    idle_fp = float(metrics.get("idle_fp_per_min", float("inf")))
    control_recall = float(metrics.get("control_recall", 0.0))
    switch_latency = _latency_or_penalty(float(metrics.get("switch_latency_s", float("inf"))))
    release_latency = _latency_or_penalty(float(metrics.get("release_latency_s", float("inf"))))
    idle_violation = 1.0 if idle_fp > float(DEFAULT_SPEED_IDLE_FP_MAX) else 0.0
    recall_violation = 1.0 if control_recall < float(DEFAULT_SPEED_MIN_CONTROL_RECALL) else 0.0
    cost = (
        float(DEFAULT_SPEED_OBJECTIVE_IDLE_WEIGHT) * idle_fp
        + float(DEFAULT_SPEED_OBJECTIVE_RECALL_WEIGHT) * (1.0 - control_recall)
        + float(DEFAULT_SPEED_OBJECTIVE_SWITCH_WEIGHT) * switch_latency
        + float(DEFAULT_SPEED_OBJECTIVE_RELEASE_WEIGHT) * release_latency
    )
    return (
        idle_violation + recall_violation,
        idle_violation,
        recall_violation,
        float(cost),
        idle_fp,
        -control_recall,
        switch_latency,
        release_latency,
    )


def _calibration_objective(
    summary: dict[str, float],
    *,
    gate_policy: str = "conservative",
    gate_metrics: Optional[dict[str, float]] = None,
) -> tuple[float, ...]:
    policy = parse_gate_policy(gate_policy)
    if policy == "balanced":
        metrics = dict(gate_metrics or {})
        if not metrics:
            metrics = {
                "idle_fp_per_min": float("inf"),
                "control_recall": 0.0,
                "switch_latency_s": float("inf"),
                "release_latency_s": float("inf"),
            }
        return _balanced_gate_objective(metrics)
    if policy == "speed":
        metrics = dict(gate_metrics or {})
        if not metrics:
            metrics = {
                "idle_fp_per_min": float("inf"),
                "control_recall": 0.0,
                "switch_latency_s": float("inf"),
                "release_latency_s": float("inf"),
            }
        return _speed_gate_objective(metrics)

    latency = float(summary.get("mean_detection_latency_sec", float("nan")))
    latency_value = 1_000_000.0 if math.isnan(latency) else latency
    trial_success = float(summary.get("trial_success_rate", 0.0))
    window_recall = float(summary.get("window_end_to_end_recall", 0.0))
    detection_recall = max(trial_success, window_recall)
    low_recall_penalty = 1.0 if detection_recall < float(DEFAULT_CALIBRATION_MIN_DETECTION_RECALL) else 0.0
    return (
        low_recall_penalty,
        float(summary.get("idle_trial_false_positive", 1.0)),
        float(summary.get("idle_false_positive", 1.0)),
        -trial_success,
        -window_recall,
        latency_value,
    )


def resolve_selected_eeg_channels(
    available_board_channels: Sequence[int],
    requested: Optional[Sequence[int]],
) -> tuple[int, ...]:
    available = tuple(int(channel) for channel in available_board_channels)
    if requested is None:
        return available
    requested_set = {int(channel) for channel in requested}
    resolved = tuple(channel for channel in available if channel in requested_set)
    if not resolved:
        raise ValueError(f"requested EEG channels are not available: {list(requested)}")
    return resolved


def build_feature_rows_from_segments(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    available_board_channels: Sequence[int],
    selected_board_channels: Optional[Sequence[int]],
    sampling_rate: int,
    freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    compute_backend: str = DEFAULT_COMPUTE_BACKEND_NAME,
    gpu_device: int = DEFAULT_GPU_DEVICE_ID,
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
    gpu_warmup: bool = True,
) -> list[dict[str, Any]]:
    resolved_channels = resolve_selected_eeg_channels(available_board_channels, selected_board_channels)
    channel_positions = [tuple(int(value) for value in available_board_channels).index(channel) for channel in resolved_channels]
    engine = FBCCAEngine(
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=win_sec,
        step_sec=step_sec,
        compute_backend=compute_backend,
        gpu_device=int(gpu_device),
        gpu_precision=gpu_precision,
        gpu_warmup=bool(gpu_warmup),
    )
    rows: list[dict[str, Any]] = []
    for trial, segment in trial_segments:
        subset = np.ascontiguousarray(segment[:, channel_positions], dtype=np.float64)
        rows.extend(
            engine.iter_window_features(
                subset,
                expected_freq=trial.expected_freq,
                label=trial.label,
                trial_id=trial.trial_id,
                block_index=trial.block_index,
            )
        )
    return rows


def estimate_fbcca_diag_channel_weights(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    available_board_channels: Sequence[int],
    sampling_rate: int,
    freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    compute_backend: str = DEFAULT_COMPUTE_BACKEND_NAME,
    gpu_device: int = DEFAULT_GPU_DEVICE_ID,
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
    gpu_warmup: bool = True,
) -> np.ndarray:
    channels = tuple(int(channel) for channel in available_board_channels)
    if not channels:
        raise ValueError("no EEG channels available for channel-weight estimation")
    dprime_values: list[float] = []
    for channel in channels:
        rows = build_feature_rows_from_segments(
            trial_segments,
            available_board_channels=channels,
            selected_board_channels=(channel,),
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            compute_backend=compute_backend,
            gpu_device=int(gpu_device),
            gpu_precision=gpu_precision,
            gpu_warmup=bool(gpu_warmup),
        )
        control_rows = [row for row in rows if row.get("expected_freq") is not None and bool(row.get("correct"))]
        if not control_rows:
            control_rows = [row for row in rows if row.get("expected_freq") is not None]
        idle_rows = [row for row in rows if row.get("expected_freq") is None]
        if not control_rows or not idle_rows:
            dprime_values.append(0.0)
            continue
        control_scores = np.asarray([float(row.get("top1_score", 0.0)) for row in control_rows], dtype=float)
        idle_scores = np.asarray([float(row.get("top1_score", 0.0)) for row in idle_rows], dtype=float)
        control_mean = float(np.mean(control_scores))
        idle_mean = float(np.mean(idle_scores))
        control_var = float(np.var(control_scores))
        idle_var = float(np.var(idle_scores))
        pooled_std = math.sqrt(max(0.5 * (control_var + idle_var), 1e-6))
        dprime_values.append(float((control_mean - idle_mean) / pooled_std))
    weights = softmax(np.asarray(dprime_values, dtype=float))
    normalized = normalize_channel_weights(weights, channels=len(channels))
    if normalized is None:
        return np.ones(len(channels), dtype=float)
    return np.asarray(normalized, dtype=float)


def train_trca_shared_spatial_basis(
    train_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    sampling_rate: int,
    freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    channel_weights: Optional[Sequence[float]] = None,
    max_rank: int = 3,
    spatial_source_model: str = DEFAULT_SPATIAL_SOURCE_MODEL,
) -> np.ndarray:
    source_model = parse_spatial_source_model(spatial_source_model)
    if source_model != "trca":
        raise ValueError(f"unsupported spatial source model: {spatial_source_model}")
    if not train_segments:
        raise ValueError("spatial basis training requires non-empty train segments")
    core = FBCCAEngine(
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=win_sec,
        step_sec=step_sec,
        Nh=DEFAULT_NH,
        subbands=((6.0, 50.0),),
    )
    channels = int(np.asarray(train_segments[0][1]).shape[1])
    weights = normalize_channel_weights(channel_weights, channels=channels)
    grouped: dict[float, list[np.ndarray]] = defaultdict(list)
    for trial, segment in train_segments:
        if trial.expected_freq is None:
            continue
        matrix = np.asarray(segment, dtype=float)
        if matrix.ndim != 2:
            continue
        if weights is not None:
            matrix = np.asarray(matrix * weights.reshape(1, -1), dtype=float)
        window = _extract_last_window(matrix, core.win_samples)
        preprocessed = core.preprocess_window(window)
        grouped[float(trial.expected_freq)].append(np.asarray(preprocessed.T, dtype=float))
    if not all(grouped.get(float(freq)) for freq in freqs):
        missing = [float(freq) for freq in freqs if not grouped.get(float(freq))]
        raise ValueError(f"spatial basis training missing trials for frequencies: {missing}")

    per_class_filters: list[np.ndarray] = []
    keep = max(1, int(max_rank))
    for freq in freqs:
        trials = grouped[float(freq)]
        filters = TRCABasedDecoder._train_trca_filters(trials, n_components=keep)
        per_class_filters.append(np.asarray(filters, dtype=float))
    stacked = np.concatenate(per_class_filters, axis=1)
    q_mat, _ = np.linalg.qr(stacked)
    if q_mat.size == 0:
        raise ValueError("failed to build spatial projection basis")
    max_keep = max(1, min(int(keep), int(q_mat.shape[1]), int(q_mat.shape[0])))
    return np.asarray(q_mat[:, :max_keep], dtype=float)


def build_fbcca_frontend_model_params(
    *,
    channel_weights: Optional[Sequence[float]],
    subband_weight_mode: Optional[str] = None,
    subband_weights: Optional[Sequence[float]] = None,
    subband_weight_params: Optional[dict[str, Any]] = None,
    spatial_filter_mode: Optional[str] = None,
    spatial_projection_basis: Optional[np.ndarray] = None,
    spatial_rank: Optional[int] = None,
    spatial_source_model: str = DEFAULT_SPATIAL_SOURCE_MODEL,
) -> dict[str, Any]:
    model_params: dict[str, Any] = {"Nh": DEFAULT_NH}
    normalized_weights = normalize_channel_weights(channel_weights)
    if normalized_weights is not None:
        model_params["channel_weight_mode"] = "fbcca_diag"
        model_params["channel_weights"] = [float(value) for value in normalized_weights]
    resolved_subband_mode = parse_subband_weight_mode(subband_weight_mode) or DEFAULT_SUBBAND_WEIGHT_MODE
    resolved_subband_weights, _, resolved_subband_params = resolve_fbcca_subband_weight_spec(
        len(DEFAULT_SUBBANDS),
        mode=resolved_subband_mode,
        explicit_weights=subband_weights,
        params=subband_weight_params,
    )
    model_params["subband_weight_mode"] = str(resolved_subband_mode)
    model_params["subband_weights"] = [float(value) for value in resolved_subband_weights]
    if resolved_subband_params is not None:
        model_params["subband_weight_params"] = json_safe(dict(resolved_subband_params))
    mode = parse_spatial_filter_mode(spatial_filter_mode)
    if mode == "trca_shared" and spatial_projection_basis is not None:
        basis = np.asarray(spatial_projection_basis, dtype=float)
        if basis.ndim != 2:
            raise ValueError("spatial projection basis must be 2D")
        rank = int(spatial_rank or basis.shape[1])
        rank = max(1, min(rank, int(basis.shape[1])))
        projection = np.asarray(basis[:, :rank], dtype=float)
        state = {
            "source_model": str(parse_spatial_source_model(spatial_source_model)),
            "projection": projection.tolist(),
            "basis_rank": int(basis.shape[1]),
            "rank": int(rank),
        }
        model_params["spatial_filter_mode"] = str(mode)
        model_params["spatial_filter_rank"] = int(rank)
        model_params["spatial_filter_state"] = state
        model_params["spatial_source_model"] = str(parse_spatial_source_model(spatial_source_model))
    return model_params


def optimize_fbcca_diag_channel_weights(
    *,
    train_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    gate_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    sampling_rate: int,
    freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    min_exit_windows: int,
    gate_policy: str,
    dynamic_stop_enabled: bool,
    dynamic_stop_alpha: float,
    spatial_filter_mode: Optional[str] = None,
    spatial_rank_candidates: Sequence[int] = DEFAULT_SPATIAL_RANK_CANDIDATES,
    joint_weight_iters: int = DEFAULT_JOINT_WEIGHT_ITERS,
    spatial_source_model: str = DEFAULT_SPATIAL_SOURCE_MODEL,
    compute_backend: str = DEFAULT_COMPUTE_BACKEND_NAME,
    gpu_device: int = DEFAULT_GPU_DEVICE_ID,
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
    gpu_warmup: bool = True,
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE,
    log_fn: Optional[Callable[[str], None]] = None,
    log_prefix: str = "FBCCA",
) -> tuple[np.ndarray, dict[str, Any]]:
    if not train_segments or not gate_segments:
        raise ValueError("channel-weight optimization requires non-empty train and gate segments")
    logger = log_fn if log_fn is not None else (lambda _msg: None)
    mode = parse_spatial_filter_mode(spatial_filter_mode)
    source_model = parse_spatial_source_model(spatial_source_model)
    rank_candidates = tuple(sorted({max(1, int(value)) for value in spatial_rank_candidates}))
    if not rank_candidates:
        rank_candidates = tuple(DEFAULT_SPATIAL_RANK_CANDIDATES)
    joint_iters = max(1, int(joint_weight_iters))
    channels = int(np.asarray(train_segments[0][1]).shape[1])
    initial_weights = estimate_fbcca_diag_channel_weights(
        train_segments,
        available_board_channels=tuple(range(channels)),
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=win_sec,
        step_sec=step_sec,
        compute_backend=compute_backend,
        gpu_device=int(gpu_device),
        gpu_precision=gpu_precision,
        gpu_warmup=bool(gpu_warmup),
    )
    initial_weights = normalize_channel_weights(initial_weights, channels=channels)
    if initial_weights is None:
        initial_weights = np.ones(channels, dtype=float)
    logger(f"{log_prefix}: initial channel weights ready | channels={channels}")

    factors = (0.75, 0.90, 1.00, 1.10, 1.25)
    max_rank = max(rank_candidates)
    current_basis: Optional[np.ndarray] = None
    if mode == "trca_shared":
        logger(f"{log_prefix}: training shared spatial basis | max_rank={max_rank}")
        current_basis = train_trca_shared_spatial_basis(
            train_segments,
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            channel_weights=initial_weights,
            max_rank=max_rank,
            spatial_source_model=source_model,
        )
        logger(f"{log_prefix}: shared spatial basis ready")

    def evaluate_weights(
        candidate_weights: np.ndarray,
        candidate_basis: Optional[np.ndarray],
    ) -> tuple[tuple[float, ...], ThresholdProfile, dict[str, float], Optional[int], dict[str, Any]]:
        best_eval: Optional[
            tuple[tuple[float, ...], ThresholdProfile, dict[str, float], Optional[int], dict[str, Any]]
        ] = None
        candidate_ranks: Sequence[Optional[int]]
        if mode == "trca_shared" and candidate_basis is not None:
            candidate_ranks = tuple(
                rank for rank in rank_candidates if int(rank) <= int(np.asarray(candidate_basis).shape[1])
            )
            if not candidate_ranks:
                candidate_ranks = (int(np.asarray(candidate_basis).shape[1]),)
        else:
            candidate_ranks = (None,)
        for rank in candidate_ranks:
            params = build_fbcca_frontend_model_params(
                channel_weights=candidate_weights,
                spatial_filter_mode=mode,
                spatial_projection_basis=candidate_basis,
                spatial_rank=rank,
                spatial_source_model=source_model,
            )
            decoder = create_decoder(
                "fbcca",
                sampling_rate=sampling_rate,
                freqs=freqs,
                win_sec=win_sec,
                step_sec=step_sec,
                model_params=params,
                compute_backend=compute_backend,
                gpu_device=gpu_device,
                gpu_precision=gpu_precision,
                gpu_warmup=gpu_warmup,
                gpu_cache_policy=gpu_cache_policy,
            )
            gate_rows = build_feature_rows_with_decoder(decoder, gate_segments)
            if not gate_rows:
                continue
            profile = fit_threshold_profile(
                gate_rows,
                freqs=freqs,
                win_sec=win_sec,
                step_sec=step_sec,
                min_enter_windows=min_enter_windows,
                min_exit_windows=min_exit_windows,
                gate_policy=gate_policy,
                evaluation_rows=gate_rows,
                dynamic_stop_enabled=dynamic_stop_enabled,
                dynamic_stop_alpha=dynamic_stop_alpha,
            )
            metrics = evaluate_profile_on_feature_rows(gate_rows, profile)
            objective = _calibration_objective({}, gate_policy=gate_policy, gate_metrics=metrics) + (
                int(rank or 0),
            )
            if best_eval is None or objective < best_eval[0]:
                best_eval = (objective, profile, metrics, rank, params)
        if best_eval is None:
            raise RuntimeError("no valid FBCCA frontend candidate produced gate rows")
        return best_eval

    best_weights = np.asarray(initial_weights, dtype=float)
    best_objective, best_profile, best_metrics, best_rank, best_model_params = evaluate_weights(
        best_weights,
        current_basis,
    )
    best_basis = None if current_basis is None else np.asarray(current_basis, dtype=float)
    iteration_logs: list[dict[str, Any]] = []
    logger(f"{log_prefix}: joint optimization start | iterations={joint_iters}")

    for iteration in range(joint_iters):
        improved = False
        before_objective = tuple(best_objective)
        logger(f"{log_prefix}: iteration {iteration + 1}/{joint_iters} start")
        for ch in range(channels):
            logger(f"{log_prefix}: iteration {iteration + 1}/{joint_iters} channel {ch + 1}/{channels}")
            local_best_weights = np.asarray(best_weights, dtype=float)
            local_best_objective = tuple(best_objective)
            local_profile = best_profile
            local_metrics = dict(best_metrics)
            local_rank = best_rank
            local_model_params = dict(best_model_params)
            for factor in factors:
                candidate = np.asarray(best_weights, dtype=float)
                candidate[ch] *= float(factor)
                normalized = normalize_channel_weights(candidate, channels=channels)
                if normalized is None:
                    continue
                try:
                    objective, profile, metrics, rank, params = evaluate_weights(
                        np.asarray(normalized, dtype=float),
                        best_basis,
                    )
                except Exception:
                    continue
                if objective < local_best_objective:
                    local_best_objective = tuple(objective)
                    local_best_weights = np.asarray(normalized, dtype=float)
                    local_profile = profile
                    local_metrics = dict(metrics)
                    local_rank = rank
                    local_model_params = dict(params)
            if local_best_objective < best_objective:
                best_objective = local_best_objective
                best_weights = local_best_weights
                best_profile = local_profile
                best_metrics = local_metrics
                best_rank = local_rank
                best_model_params = local_model_params
                improved = True
        basis_updated = False
        if mode == "trca_shared":
            try:
                logger(f"{log_prefix}: iteration {iteration + 1}/{joint_iters} refreshing spatial basis")
                refreshed_basis = train_trca_shared_spatial_basis(
                    train_segments,
                    sampling_rate=sampling_rate,
                    freqs=freqs,
                    win_sec=win_sec,
                    step_sec=step_sec,
                    channel_weights=best_weights,
                    max_rank=max_rank,
                    spatial_source_model=source_model,
                )
                refreshed_eval = evaluate_weights(best_weights, refreshed_basis)
                if refreshed_eval[0] <= best_objective:
                    best_objective, best_profile, best_metrics, best_rank, best_model_params = refreshed_eval
                    best_basis = np.asarray(refreshed_basis, dtype=float)
                    basis_updated = True
            except Exception:
                pass
        iteration_logs.append(
            {
                "iteration": int(iteration + 1),
                "before_objective": [float(value) for value in before_objective],
                "after_objective": [float(value) for value in best_objective],
                "improved_by_weight": bool(improved),
                "basis_updated": bool(basis_updated),
                "selected_rank": None if best_rank is None else int(best_rank),
            }
        )
        logger(
            f"{log_prefix}: iteration {iteration + 1}/{joint_iters} done | "
            f"improved={int(improved)} basis_updated={int(basis_updated)} "
            f"rank={0 if best_rank is None else int(best_rank)}"
        )
        if not improved and not basis_updated:
            break

    metadata = {
        "mode": "fbcca_diag_joint",
        "objective": [float(value) for value in best_objective],
        "initial_weights": [float(value) for value in initial_weights],
        "optimized_weights": [float(value) for value in best_weights],
        "spatial_filter_mode": mode,
        "spatial_source_model": source_model,
        "spatial_rank_candidates": [int(value) for value in rank_candidates],
        "selected_spatial_rank": None if best_rank is None else int(best_rank),
        "joint_weight_iters": int(joint_iters),
        "iterations": json_safe(iteration_logs),
        "gate_metrics": {key: float(value) for key, value in best_metrics.items()},
        "fit_profile": json_safe(asdict(best_profile)),
        "optimized_model_params": json_safe(dict(best_model_params)),
        "optimized_spatial_projection": (
            None if best_basis is None else np.asarray(best_basis[:, : int(best_rank or 1)], dtype=float).tolist()
        ),
    }
    logger(
        f"{log_prefix}: optimization done | "
        f"selected_rank={0 if best_rank is None else int(best_rank)} "
        f"objective={[float(value) for value in best_objective]}"
    )
    return np.asarray(best_weights, dtype=float), metadata


def _split_trial_segments_kfold(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    folds: int,
    seed: int,
) -> list[tuple[list[tuple[TrialSpec, np.ndarray]], list[tuple[TrialSpec, np.ndarray]]]]:
    indexed = list(enumerate(trial_segments))
    if len(indexed) < 2:
        return [(list(trial_segments), list(trial_segments))]
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, (trial, _segment) in indexed:
        key = "idle" if trial.expected_freq is None else f"{float(trial.expected_freq):g}"
        grouped[key].append(int(index))
    rng = random.Random(int(seed))
    fold_count = max(2, min(int(folds), len(indexed)))
    fold_ids: list[set[int]] = [set() for _ in range(fold_count)]
    for indices in grouped.values():
        block = [int(item) for item in indices]
        rng.shuffle(block)
        for position, value in enumerate(block):
            fold_ids[position % fold_count].add(int(value))
    results: list[tuple[list[tuple[TrialSpec, np.ndarray]], list[tuple[TrialSpec, np.ndarray]]]] = []
    for fold_index in range(fold_count):
        val_ids = fold_ids[fold_index]
        fit = [item for index, item in indexed if index not in val_ids]
        val = [item for index, item in indexed if index in val_ids]
        if not fit or not val:
            continue
        results.append((fit, val))
    return results or [(list(trial_segments), list(trial_segments))]


def _aggregate_numeric_metric_dicts(metric_rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted({str(key) for row in metric_rows for key in row.keys()})
    aggregated: dict[str, float] = {}
    for key in keys:
        values = []
        for row in metric_rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                values.append(float(value))
        if values:
            aggregated[str(key)] = float(np.mean(np.asarray(values, dtype=float)))
    return aggregated


def _evaluate_fbcca_frontend_candidate_cv(
    *,
    model_params: dict[str, Any],
    train_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    gate_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    sampling_rate: int,
    freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    min_exit_windows: int,
    gate_policy: str,
    dynamic_stop_enabled: bool,
    dynamic_stop_alpha: float,
    control_state_mode: str,
    weight_cv_folds: int,
    seed: int,
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
    gpu_warmup: bool,
    gpu_cache_policy: str,
) -> dict[str, Any]:
    folds = _split_trial_segments_kfold(gate_segments, folds=weight_cv_folds, seed=seed)
    fold_metrics: list[dict[str, float]] = []
    for fold_index, (profile_segments, eval_segments) in enumerate(folds, start=1):
        decoder = create_decoder(
            "fbcca",
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=model_params,
            compute_backend=compute_backend,
            gpu_device=gpu_device,
            gpu_precision=gpu_precision,
            gpu_warmup=gpu_warmup,
            gpu_cache_policy=gpu_cache_policy,
        )
        if decoder.requires_fit:
            decoder.fit(train_segments)
        fit_rows = build_feature_rows_with_decoder(decoder, profile_segments)
        val_rows = build_feature_rows_with_decoder(decoder, eval_segments)
        if not fit_rows or not val_rows:
            continue
        profile = fit_threshold_profile(
            fit_rows,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            min_enter_windows=min_enter_windows,
            min_exit_windows=min_exit_windows,
            gate_policy=gate_policy,
            evaluation_rows=fit_rows,
            dynamic_stop_enabled=dynamic_stop_enabled,
            dynamic_stop_alpha=dynamic_stop_alpha,
            control_state_mode=control_state_mode,
        )
        metrics = evaluate_profile_on_feature_rows(val_rows, profile)
        metrics["cv_fold_index"] = float(fold_index)
        fold_metrics.append(metrics)
    if not fold_metrics:
        raise RuntimeError("no valid gate fold metrics for FBCCA frontend candidate")
    aggregated = _aggregate_numeric_metric_dicts(fold_metrics)
    objective = _calibration_objective({}, gate_policy=gate_policy, gate_metrics=aggregated)
    return {
        "objective": tuple(float(value) for value in objective),
        "gate_cv_metrics": aggregated,
        "fold_metrics": fold_metrics,
        "fold_count": int(len(fold_metrics)),
    }


def optimize_fbcca_frontend_weights(
    *,
    train_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    gate_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    sampling_rate: int,
    freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    min_enter_windows: int,
    min_exit_windows: int,
    gate_policy: str,
    dynamic_stop_enabled: bool,
    dynamic_stop_alpha: float,
    channel_weight_mode: Optional[str] = None,
    subband_weight_mode: Optional[str] = None,
    spatial_filter_mode: Optional[str] = None,
    spatial_rank_candidates: Sequence[int] = DEFAULT_SPATIAL_RANK_CANDIDATES,
    joint_weight_iters: int = DEFAULT_JOINT_WEIGHT_ITERS,
    weight_cv_folds: int = DEFAULT_FBCCA_WEIGHT_CV_FOLDS,
    spatial_source_model: str = DEFAULT_SPATIAL_SOURCE_MODEL,
    idle_fp_hard_th: float = DEFAULT_IDLE_FP_HARD_TH,
    channel_weight_l2: float = DEFAULT_CHANNEL_WEIGHT_L2,
    subband_prior_strength: float = DEFAULT_SUBBAND_PRIOR_STRENGTH,
    control_state_mode: str = DEFAULT_CONTROL_STATE_MODE,
    compute_backend: str = DEFAULT_COMPUTE_BACKEND_NAME,
    gpu_device: int = DEFAULT_GPU_DEVICE_ID,
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
    gpu_warmup: bool = True,
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE,
    log_fn: Optional[Callable[[str], None]] = None,
    log_prefix: str = "FBCCA",
) -> dict[str, Any]:
    if not train_segments or not gate_segments:
        raise ValueError("FBCCA frontend optimization requires non-empty train and gate segments")
    logger = log_fn if log_fn is not None else (lambda _msg: None)
    resolved_channel_mode = parse_channel_weight_mode(channel_weight_mode)
    resolved_subband_mode = parse_subband_weight_mode(subband_weight_mode) or DEFAULT_SUBBAND_WEIGHT_MODE
    resolved_spatial_mode = parse_spatial_filter_mode(spatial_filter_mode)
    resolved_spatial_source = parse_spatial_source_model(spatial_source_model)
    rank_candidates = tuple(sorted({max(1, int(value)) for value in spatial_rank_candidates})) or tuple(
        DEFAULT_SPATIAL_RANK_CANDIDATES
    )
    channels = int(np.asarray(train_segments[0][1]).shape[1])
    joint_iters = max(1, int(joint_weight_iters))
    weight_folds = max(2, int(weight_cv_folds))
    idle_fp_limit = float(idle_fp_hard_th)
    channel_l2 = max(0.0, float(channel_weight_l2))
    subband_prior = max(0.0, float(subband_prior_strength))
    resolved_control_state_mode = parse_control_state_mode(control_state_mode)

    initial_channel_weights = None
    if resolved_channel_mode == "fbcca_diag":
        initial_channel_weights = estimate_fbcca_diag_channel_weights(
            train_segments,
            available_board_channels=tuple(range(channels)),
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            compute_backend=compute_backend,
            gpu_device=int(gpu_device),
            gpu_precision=gpu_precision,
            gpu_warmup=bool(gpu_warmup),
        )
        initial_channel_weights = normalize_channel_weights(initial_channel_weights, channels=channels)
    initial_subband_weights, _, initial_subband_params = resolve_fbcca_subband_weight_spec(
        len(DEFAULT_SUBBANDS),
        mode=resolved_subband_mode,
    )
    chen_prior_weights = fbcca_subband_weights_from_ab(
        len(DEFAULT_SUBBANDS),
        a=DEFAULT_FBCCA_SUBBAND_WEIGHT_A,
        b=DEFAULT_FBCCA_SUBBAND_WEIGHT_B,
    )
    current_basis = None
    if resolved_spatial_mode == "trca_shared":
        seed_weights = initial_channel_weights if initial_channel_weights is not None else None
        current_basis = train_trca_shared_spatial_basis(
            train_segments,
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            channel_weights=seed_weights,
            max_rank=max(rank_candidates),
            spatial_source_model=resolved_spatial_source,
        )

    def evaluate_candidate(
        candidate_channel_weights: Optional[np.ndarray],
        candidate_subband_weights: np.ndarray,
        candidate_subband_params: Optional[dict[str, Any]],
        candidate_basis: Optional[np.ndarray],
    ) -> dict[str, Any]:
        best_eval: Optional[dict[str, Any]] = None
        candidate_ranks: Sequence[Optional[int]]
        if resolved_spatial_mode == "trca_shared" and candidate_basis is not None:
            candidate_ranks = tuple(
                rank for rank in rank_candidates if int(rank) <= int(np.asarray(candidate_basis).shape[1])
            ) or (int(np.asarray(candidate_basis).shape[1]),)
        else:
            candidate_ranks = (None,)
        for rank in candidate_ranks:
            params = build_fbcca_frontend_model_params(
                channel_weights=candidate_channel_weights,
                subband_weight_mode=resolved_subband_mode,
                subband_weights=candidate_subband_weights,
                subband_weight_params=candidate_subband_params,
                spatial_filter_mode=resolved_spatial_mode,
                spatial_projection_basis=candidate_basis,
                spatial_rank=rank,
                spatial_source_model=resolved_spatial_source,
            )
            evaluation = _evaluate_fbcca_frontend_candidate_cv(
                model_params=params,
                train_segments=train_segments,
                gate_segments=gate_segments,
                sampling_rate=sampling_rate,
                freqs=freqs,
                win_sec=win_sec,
                step_sec=step_sec,
                min_enter_windows=min_enter_windows,
                min_exit_windows=min_exit_windows,
                gate_policy=gate_policy,
                dynamic_stop_enabled=dynamic_stop_enabled,
                dynamic_stop_alpha=dynamic_stop_alpha,
                control_state_mode=resolved_control_state_mode,
                weight_cv_folds=weight_folds,
                seed=DEFAULT_CALIBRATION_SEED,
                compute_backend=compute_backend,
                gpu_device=gpu_device,
                gpu_precision=gpu_precision,
                gpu_warmup=gpu_warmup,
                gpu_cache_policy=gpu_cache_policy,
            )
            candidate = {
                "objective": tuple(evaluation["objective"]) + (int(rank or 0),),
                "metrics": dict(evaluation["gate_cv_metrics"]),
                "fold_metrics": list(evaluation["fold_metrics"]),
                "fold_count": int(evaluation["fold_count"]),
                "rank": None if rank is None else int(rank),
                "model_params": params,
            }
            idle_fp = float(candidate["metrics"].get("idle_fp_per_min", float("inf")))
            hard_idle_violation = 1.0 if idle_fp > idle_fp_limit else 0.0
            channel_penalty = 0.0
            if candidate_channel_weights is not None:
                cw = np.asarray(candidate_channel_weights, dtype=float).reshape(-1)
                if cw.size:
                    channel_penalty = float(np.mean((cw - 1.0) ** 2))
            subband_penalty = 0.0
            sw = np.asarray(candidate_subband_weights, dtype=float).reshape(-1)
            if sw.size == chen_prior_weights.size:
                subband_penalty = float(np.mean((sw - chen_prior_weights) ** 2))
            base_objective = tuple(float(value) for value in candidate["objective"])
            candidate["objective"] = (
                hard_idle_violation,
                base_objective[0] if base_objective else 0.0,
                base_objective[1] if len(base_objective) > 1 else 0.0,
                base_objective[2] if len(base_objective) > 2 else 0.0,
                (base_objective[3] if len(base_objective) > 3 else 0.0)
                + channel_l2 * channel_penalty
                + subband_prior * subband_penalty,
            ) + base_objective[4:]
            candidate["regularization"] = {
                "idle_fp_hard_th": float(idle_fp_limit),
                "hard_idle_violation": bool(hard_idle_violation > 0.0),
                "channel_weight_l2": float(channel_l2),
                "channel_penalty": float(channel_penalty),
                "subband_prior_strength": float(subband_prior),
                "subband_prior_penalty": float(subband_penalty),
                "control_state_mode": str(resolved_control_state_mode),
            }
            if best_eval is None or tuple(candidate["objective"]) < tuple(best_eval["objective"]):
                best_eval = candidate
        if best_eval is None:
            raise RuntimeError("no valid FBCCA frontend candidate produced fold metrics")
        return dict(best_eval)

    best_channel_weights = None if initial_channel_weights is None else np.asarray(initial_channel_weights, dtype=float)
    best_subband_weights = np.asarray(initial_subband_weights, dtype=float)
    best_subband_params = None if initial_subband_params is None else dict(initial_subband_params)
    best_basis = None if current_basis is None else np.asarray(current_basis, dtype=float)
    best_eval = evaluate_candidate(best_channel_weights, best_subband_weights, best_subband_params, best_basis)
    iteration_logs: list[dict[str, Any]] = []
    logger(
        f"{log_prefix}: frontend optimization start | channel_mode={resolved_channel_mode or 'none'} "
        f"subband_mode={resolved_subband_mode} spatial_mode={resolved_spatial_mode or 'none'} "
        f"iters={joint_iters} folds={weight_folds}"
    )
    channel_factors = (0.75, 0.90, 1.00, 1.10, 1.25)
    simplex_factors = (0.75, 0.90, 1.00, 1.10, 1.25)
    for iteration in range(joint_iters):
        weight_improved = False
        subband_improved = False
        basis_updated = False
        before_objective = tuple(best_eval["objective"])
        if resolved_channel_mode == "fbcca_diag" and best_channel_weights is not None:
            for channel_index in range(channels):
                logger(f"{log_prefix}: iteration {iteration + 1}/{joint_iters} channel {channel_index + 1}/{channels}")
                local_best = dict(best_eval)
                local_weights = np.asarray(best_channel_weights, dtype=float)
                for factor in channel_factors:
                    candidate = np.asarray(best_channel_weights, dtype=float)
                    candidate[channel_index] *= float(factor)
                    normalized = normalize_channel_weights(candidate, channels=channels)
                    if normalized is None:
                        continue
                    try:
                        evaluation = evaluate_candidate(np.asarray(normalized, dtype=float), best_subband_weights, best_subband_params, best_basis)
                    except Exception as exc:
                        logger(f"{log_prefix}: channel candidate failed channel={channel_index + 1} factor={float(factor):g} error={exc}")
                        continue
                    if tuple(evaluation["objective"]) < tuple(local_best["objective"]):
                        local_best = dict(evaluation)
                        local_weights = np.asarray(normalized, dtype=float)
                if tuple(local_best["objective"]) < tuple(best_eval["objective"]):
                    best_eval = local_best
                    best_channel_weights = np.asarray(local_weights, dtype=float)
                    weight_improved = True
        if resolved_subband_mode == "chen_ab_subject":
            logger(f"{log_prefix}: iteration {iteration + 1}/{joint_iters} subband grid search")
            local_best = dict(best_eval)
            local_weights = np.asarray(best_subband_weights, dtype=float)
            local_params = None if best_subband_params is None else dict(best_subband_params)
            for a_value in DEFAULT_FBCCA_SUBBAND_WEIGHT_A_GRID:
                for b_value in DEFAULT_FBCCA_SUBBAND_WEIGHT_B_GRID:
                    candidate_weights = fbcca_subband_weights_from_ab(
                        len(DEFAULT_SUBBANDS),
                        a=float(a_value),
                        b=float(b_value),
                    )
                    try:
                        evaluation = evaluate_candidate(
                            best_channel_weights,
                            candidate_weights,
                            {"a": float(a_value), "b": float(b_value)},
                            best_basis,
                        )
                    except Exception:
                        continue
                    if tuple(evaluation["objective"]) < tuple(local_best["objective"]):
                        local_best = dict(evaluation)
                        local_weights = np.asarray(candidate_weights, dtype=float)
                        local_params = {"a": float(a_value), "b": float(b_value)}
            if tuple(local_best["objective"]) < tuple(best_eval["objective"]):
                best_eval = local_best
                best_subband_weights = np.asarray(local_weights, dtype=float)
                best_subband_params = local_params
                subband_improved = True
        elif resolved_subband_mode == "simplex_subject":
            for subband_index in range(int(best_subband_weights.size)):
                logger(
                    f"{log_prefix}: iteration {iteration + 1}/{joint_iters} subband {subband_index + 1}/{int(best_subband_weights.size)}"
                )
                local_best = dict(best_eval)
                local_weights = np.asarray(best_subband_weights, dtype=float)
                for factor in simplex_factors:
                    candidate = np.asarray(best_subband_weights, dtype=float)
                    candidate[subband_index] *= float(factor)
                    normalized = normalize_subband_weights(candidate, subbands=int(candidate.size))
                    if normalized is None:
                        continue
                    try:
                        evaluation = evaluate_candidate(best_channel_weights, np.asarray(normalized, dtype=float), None, best_basis)
                    except Exception:
                        continue
                    if tuple(evaluation["objective"]) < tuple(local_best["objective"]):
                        local_best = dict(evaluation)
                        local_weights = np.asarray(normalized, dtype=float)
                if tuple(local_best["objective"]) < tuple(best_eval["objective"]):
                    best_eval = local_best
                    best_subband_weights = np.asarray(local_weights, dtype=float)
                    best_subband_params = None
                    subband_improved = True
        if resolved_spatial_mode == "trca_shared":
            try:
                logger(f"{log_prefix}: iteration {iteration + 1}/{joint_iters} refreshing spatial basis")
                refreshed_basis = train_trca_shared_spatial_basis(
                    train_segments,
                    sampling_rate=sampling_rate,
                    freqs=freqs,
                    win_sec=win_sec,
                    step_sec=step_sec,
                    channel_weights=best_channel_weights,
                    max_rank=max(rank_candidates),
                    spatial_source_model=resolved_spatial_source,
                )
                refreshed = evaluate_candidate(best_channel_weights, best_subband_weights, best_subband_params, refreshed_basis)
                if tuple(refreshed["objective"]) <= tuple(best_eval["objective"]):
                    best_eval = dict(refreshed)
                    best_basis = np.asarray(refreshed_basis, dtype=float)
                    basis_updated = True
            except Exception:
                pass
        iteration_logs.append(
            {
                "iteration": int(iteration + 1),
                "before_objective": [float(value) for value in before_objective],
                "after_objective": [float(value) for value in best_eval["objective"]],
                "channel_improved": bool(weight_improved),
                "subband_improved": bool(subband_improved),
                "basis_updated": bool(basis_updated),
                "selected_rank": best_eval["rank"],
            }
        )
        logger(
            f"{log_prefix}: iteration {iteration + 1}/{joint_iters} done | "
            f"channel_improved={int(weight_improved)} subband_improved={int(subband_improved)} "
            f"basis_updated={int(basis_updated)}"
        )
        if not weight_improved and not subband_improved and not basis_updated:
            break
    optimized_params = dict(best_eval["model_params"])
    metadata = {
        "mode": "fbcca_frontend_joint",
        "channel_weight_mode": resolved_channel_mode,
        "subband_weight_mode": resolved_subband_mode,
        "spatial_filter_mode": resolved_spatial_mode,
        "spatial_source_model": resolved_spatial_source,
        "objective": [float(value) for value in best_eval["objective"]],
        "initial_channel_weights": None if initial_channel_weights is None else [float(value) for value in initial_channel_weights],
        "optimized_channel_weights": None if best_channel_weights is None else [float(value) for value in best_channel_weights],
        "initial_subband_weights": [float(value) for value in initial_subband_weights],
        "optimized_subband_weights": [float(value) for value in best_subband_weights],
        "optimized_subband_params": None if best_subband_params is None else json_safe(dict(best_subband_params)),
        "selected_spatial_rank": best_eval["rank"],
        "spatial_rank_candidates": [int(value) for value in rank_candidates],
        "joint_weight_iters": int(joint_iters),
        "weight_cv_folds": int(weight_folds),
        "iterations": json_safe(iteration_logs),
        "gate_cv_metrics": dict(best_eval["metrics"]),
        "gate_cv_fold_metrics": json_safe(list(best_eval["fold_metrics"])),
        "regularization": json_safe(dict(best_eval.get("regularization", {}))),
        "optimized_model_params": json_safe(optimized_params),
        "optimized_spatial_projection": (
            None
            if best_basis is None
            else np.asarray(best_basis[:, : int(best_eval["rank"] or 1)], dtype=float).tolist()
        ),
    }
    return {
        "channel_weights": None if best_channel_weights is None else np.asarray(best_channel_weights, dtype=float),
        "subband_weights": np.asarray(best_subband_weights, dtype=float),
        "subband_weight_params": None if best_subband_params is None else dict(best_subband_params),
        "metadata": metadata,
    }


def select_auto_eeg_channels(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    available_board_channels: Sequence[int],
    sampling_rate: int,
    freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    compute_backend: str = DEFAULT_COMPUTE_BACKEND_NAME,
    gpu_device: int = DEFAULT_GPU_DEVICE_ID,
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
    gpu_warmup: bool = True,
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE,
    seed: int = DEFAULT_CALIBRATION_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    target_count: int = DEFAULT_AUTO_CHANNEL_COUNT,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    available = tuple(int(channel) for channel in available_board_channels)
    if len(available) <= max(1, int(target_count)):
        return available, []

    channel_scores: list[dict[str, Any]] = []
    for channel in available:
        rows = build_feature_rows_from_segments(
            trial_segments,
            available_board_channels=available,
            selected_board_channels=(channel,),
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            compute_backend=compute_backend,
            gpu_device=int(gpu_device),
            gpu_precision=gpu_precision,
            gpu_warmup=bool(gpu_warmup),
        )
        fit_rows, validation_rows, _ = split_feature_rows_by_trial(
            rows,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        if not fit_rows or not validation_rows:
            continue
        try:
            profile = fit_threshold_profile(
                fit_rows,
                freqs=freqs,
                win_sec=win_sec,
                step_sec=step_sec,
                min_enter_windows=2,
                min_exit_windows=2,
                gate_policy=DEFAULT_GATE_POLICY,
                evaluation_rows=validation_rows,
            )
            summary = summarize_profile_quality(validation_rows, profile)
            gate_metrics = evaluate_profile_on_feature_rows(validation_rows, profile)
            objective = _calibration_objective(summary, gate_policy=DEFAULT_GATE_POLICY, gate_metrics=gate_metrics)
            channel_scores.append(
                {
                    "channel": int(channel),
                    "objective": objective,
                    "summary": summary,
                    "gate_metrics": gate_metrics,
                }
            )
        except Exception:
            continue

    if not channel_scores:
        return available, []

    channel_scores.sort(key=lambda item: (item["objective"], item["channel"]))
    keep_count = max(1, min(int(target_count), len(channel_scores)))
    selected = tuple(sorted(int(item["channel"]) for item in channel_scores[:keep_count]))
    return selected, channel_scores


def select_auto_eeg_channels_for_model(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    model_name: str,
    available_board_channels: Sequence[int],
    sampling_rate: int,
    freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    model_params: Optional[dict[str, Any]] = None,
    compute_backend: str = DEFAULT_COMPUTE_BACKEND_NAME,
    gpu_device: int = DEFAULT_GPU_DEVICE_ID,
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
    gpu_warmup: bool = True,
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE,
    seed: int = DEFAULT_CALIBRATION_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    target_count: int = DEFAULT_AUTO_CHANNEL_COUNT,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    available = tuple(int(channel) for channel in available_board_channels)
    if len(available) <= max(1, int(target_count)):
        return available, []

    # Keep trial-level split and preserve temporal order in each split.
    train_ratio = min(0.9, max(0.5, 1.0 - float(validation_fraction)))
    gate_ratio = min(0.4, max(0.1, float(validation_fraction) * 0.5))
    model = normalize_model_name(model_name)
    params = dict(model_params or {})

    channel_scores: list[dict[str, Any]] = []
    for channel in available:
        position = list(available).index(int(channel))
        subset = _subset_trial_segments_by_positions(trial_segments, [position])
        fit_segments, threshold_segments, score_segments = split_trial_segments_for_benchmark(
            subset,
            seed=seed,
            train_ratio=train_ratio,
            gate_ratio=gate_ratio,
        )
        if not fit_segments or not threshold_segments:
            continue
        if not score_segments:
            score_segments = list(threshold_segments)
        try:
            decoder = create_decoder(
                model,
                sampling_rate=sampling_rate,
                freqs=freqs,
                win_sec=win_sec,
                step_sec=step_sec,
                model_params=params,
                compute_backend=compute_backend,
                gpu_device=gpu_device,
                gpu_precision=gpu_precision,
                gpu_warmup=gpu_warmup,
                gpu_cache_policy=gpu_cache_policy,
            )
            if decoder.requires_fit:
                decoder.fit(fit_segments)
            threshold_rows = build_feature_rows_with_decoder(decoder, threshold_segments)
            score_rows = build_feature_rows_with_decoder(decoder, score_segments)
            profile = fit_threshold_profile(
                threshold_rows,
                freqs=freqs,
                win_sec=win_sec,
                step_sec=step_sec,
                min_enter_windows=2,
                min_exit_windows=2,
                gate_policy=DEFAULT_GATE_POLICY,
                evaluation_rows=score_rows,
            )
            summary = summarize_profile_quality(score_rows, profile)
            gate_metrics = evaluate_profile_on_feature_rows(score_rows, profile)
            objective = _calibration_objective(
                summary,
                gate_policy=DEFAULT_GATE_POLICY,
                gate_metrics=gate_metrics,
            )
            channel_scores.append(
                {
                    "channel": int(channel),
                    "objective": objective,
                    "summary": summary,
                    "gate_metrics": gate_metrics,
                    "split_counts": {
                        "fit_segments": int(len(fit_segments)),
                        "threshold_segments": int(len(threshold_segments)),
                        "score_segments": int(len(score_segments)),
                    },
                }
            )
        except Exception:
            continue

    if not channel_scores:
        return available, []

    channel_scores.sort(key=lambda item: (item["objective"], item["channel"]))
    keep_count = max(1, min(int(target_count), len(channel_scores)))
    selected = tuple(sorted(int(item["channel"]) for item in channel_scores[:keep_count]))
    return selected, channel_scores


def optimize_profile_from_segments(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    available_board_channels: Sequence[int],
    sampling_rate: int,
    freqs: Sequence[float],
    active_sec: float,
    preferred_win_sec: float,
    step_sec: float,
    seed: int = DEFAULT_CALIBRATION_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    gate_policy: str = DEFAULT_GATE_POLICY,
    channel_weight_mode: Optional[str] = DEFAULT_CHANNEL_WEIGHT_MODE,
    subband_weight_mode: Optional[str] = DEFAULT_SUBBAND_WEIGHT_MODE,
    spatial_filter_mode: Optional[str] = DEFAULT_SPATIAL_FILTER_MODE,
    spatial_rank_candidates: Sequence[int] = DEFAULT_SPATIAL_RANK_CANDIDATES,
    joint_weight_iters: int = DEFAULT_JOINT_WEIGHT_ITERS,
    weight_cv_folds: int = DEFAULT_FBCCA_WEIGHT_CV_FOLDS,
    spatial_source_model: str = DEFAULT_SPATIAL_SOURCE_MODEL,
    dynamic_stop_enabled: bool = DEFAULT_DYNAMIC_STOP_ENABLED,
    dynamic_stop_alpha: float = DEFAULT_DYNAMIC_STOP_ALPHA,
    compute_backend: str = DEFAULT_COMPUTE_BACKEND_NAME,
    gpu_device: int = DEFAULT_GPU_DEVICE_ID,
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
    gpu_warmup: bool = True,
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE,
) -> tuple[ThresholdProfile, dict[str, Any]]:
    policy = parse_gate_policy(gate_policy)
    weight_mode = parse_channel_weight_mode(channel_weight_mode)
    resolved_subband_mode = parse_subband_weight_mode(subband_weight_mode) or DEFAULT_SUBBAND_WEIGHT_MODE
    resolved_spatial_mode = parse_spatial_filter_mode(spatial_filter_mode)
    resolved_spatial_ranks = tuple(sorted({max(1, int(value)) for value in spatial_rank_candidates}))
    if not resolved_spatial_ranks:
        resolved_spatial_ranks = tuple(DEFAULT_SPATIAL_RANK_CANDIDATES)
    resolved_joint_iters = max(1, int(joint_weight_iters))
    resolved_weight_cv_folds = max(2, int(weight_cv_folds))
    resolved_spatial_source = parse_spatial_source_model(spatial_source_model)
    if fbcca_weight_learning_requires_all8(
        "fbcca",
        channel_weight_mode=weight_mode,
        subband_weight_mode=resolved_subband_mode,
    ):
        selected_channels = tuple(int(channel) for channel in available_board_channels)
        channel_scores = []
    else:
        selected_channels, channel_scores = select_auto_eeg_channels(
            trial_segments,
            available_board_channels=available_board_channels,
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=preferred_win_sec,
            step_sec=step_sec,
            compute_backend=compute_backend,
            gpu_device=int(gpu_device),
            gpu_precision=gpu_precision,
            gpu_warmup=bool(gpu_warmup),
            gpu_cache_policy=gpu_cache_policy,
            seed=seed,
            validation_fraction=validation_fraction,
        )

    available = tuple(int(channel) for channel in available_board_channels)
    selected_positions = [available.index(int(channel)) for channel in selected_channels]
    selected_segments = _subset_trial_segments_by_positions(trial_segments, selected_positions)
    train_ratio = min(0.9, max(0.5, 1.0 - float(validation_fraction)))
    gate_ratio = min(0.4, max(0.1, float(validation_fraction) * 0.5))
    fit_segments, gate_segments, holdout_segments = split_trial_segments_for_benchmark(
        selected_segments,
        seed=seed,
        train_ratio=train_ratio,
        gate_ratio=gate_ratio,
    )
    if not fit_segments:
        fit_segments = list(gate_segments)
    if not fit_segments:
        fit_segments = list(selected_segments)
    if not gate_segments:
        gate_segments = list(fit_segments)
    eval_segments = list(holdout_segments) if holdout_segments else list(gate_segments)

    win_candidates = calibration_search_win_candidates(active_sec, preferred_win_sec, gate_policy=policy)
    if policy == "speed":
        min_enter_candidates = tuple(DEFAULT_SPEED_MIN_ENTER_CANDIDATES)
        min_exit_candidates = tuple(DEFAULT_SPEED_MIN_EXIT_CANDIDATES)
    else:
        min_enter_candidates = tuple(DEFAULT_MIN_ENTER_CANDIDATES)
        min_exit_candidates = tuple(DEFAULT_MIN_EXIT_CANDIDATES)
    best_profile: Optional[ThresholdProfile] = None
    best_summary: Optional[dict[str, float]] = None
    best_search: Optional[dict[str, Any]] = None
    best_objective: Optional[tuple[float, ...]] = None
    best_gate_metrics: Optional[dict[str, float]] = None
    best_weight_metadata: Optional[dict[str, Any]] = None

    for win_sec in win_candidates:
        available_windows = calibration_window_count(active_sec, win_sec, step_sec)
        valid_enter_candidates = [int(candidate) for candidate in min_enter_candidates if int(candidate) <= int(available_windows)]
        if not valid_enter_candidates:
            continue
        for min_enter in valid_enter_candidates:
            for min_exit in min_exit_candidates:
                model_params: dict[str, Any] = {"Nh": DEFAULT_NH}
                weight_metadata: Optional[dict[str, Any]] = None
                if weight_mode == "fbcca_diag" or resolved_subband_mode != "chen_fixed":
                    try:
                        frontend_result = optimize_fbcca_frontend_weights(
                            train_segments=fit_segments,
                            gate_segments=gate_segments,
                            sampling_rate=sampling_rate,
                            freqs=freqs,
                            win_sec=win_sec,
                            step_sec=step_sec,
                            min_enter_windows=min_enter,
                            min_exit_windows=min_exit,
                            gate_policy=policy,
                            dynamic_stop_enabled=dynamic_stop_enabled,
                            dynamic_stop_alpha=dynamic_stop_alpha,
                            channel_weight_mode=weight_mode,
                            subband_weight_mode=resolved_subband_mode,
                            spatial_filter_mode=resolved_spatial_mode,
                            spatial_rank_candidates=resolved_spatial_ranks,
                            joint_weight_iters=resolved_joint_iters,
                            weight_cv_folds=resolved_weight_cv_folds,
                            spatial_source_model=resolved_spatial_source,
                            compute_backend=compute_backend,
                            gpu_device=gpu_device,
                            gpu_precision=gpu_precision,
                            gpu_warmup=gpu_warmup,
                            gpu_cache_policy=gpu_cache_policy,
                        )
                        weight_metadata = (
                            dict(frontend_result.get("metadata"))
                            if isinstance(frontend_result.get("metadata"), dict)
                            else None
                        )
                        optimized_params = (
                            weight_metadata.get("optimized_model_params")
                            if isinstance(weight_metadata, dict)
                            else None
                        )
                        if isinstance(optimized_params, dict) and optimized_params:
                            model_params = dict(optimized_params)
                            model_params.setdefault("Nh", DEFAULT_NH)
                        else:
                            optimized_weights = frontend_result.get("channel_weights")
                            optimized_subband_weights = frontend_result.get("subband_weights")
                            optimized_subband_params = frontend_result.get("subband_weight_params")
                            if optimized_weights is not None:
                                model_params["channel_weight_mode"] = "fbcca_diag"
                                model_params["channel_weights"] = [float(value) for value in optimized_weights]
                            model_params["subband_weight_mode"] = str(resolved_subband_mode)
                            if optimized_subband_weights is not None:
                                model_params["subband_weights"] = [float(value) for value in optimized_subband_weights]
                            if optimized_subband_params is not None:
                                model_params["subband_weight_params"] = json_safe(dict(optimized_subband_params))
                    except Exception:
                        continue
                try:
                    decoder = create_decoder(
                        DEFAULT_MODEL_NAME,
                        sampling_rate=sampling_rate,
                        freqs=freqs,
                        win_sec=win_sec,
                        step_sec=step_sec,
                        model_params=model_params,
                        compute_backend=compute_backend,
                        gpu_device=gpu_device,
                        gpu_precision=gpu_precision,
                        gpu_warmup=gpu_warmup,
                        gpu_cache_policy=gpu_cache_policy,
                    )
                    gate_rows = build_feature_rows_with_decoder(decoder, gate_segments)
                    if not gate_rows:
                        continue
                    candidate_profile = fit_threshold_profile(
                        gate_rows,
                        freqs=freqs,
                        win_sec=win_sec,
                        step_sec=step_sec,
                        min_enter_windows=min_enter,
                        min_exit_windows=min_exit,
                        gate_policy=policy,
                        evaluation_rows=gate_rows,
                        dynamic_stop_enabled=dynamic_stop_enabled,
                        dynamic_stop_alpha=dynamic_stop_alpha,
                        control_state_mode=control_state_mode,
                    )
                    eval_rows = build_feature_rows_with_decoder(decoder, eval_segments)
                    summary = summarize_profile_quality(eval_rows, candidate_profile)
                    gate_metrics = evaluate_profile_on_feature_rows(eval_rows, candidate_profile)
                except Exception:
                    continue
                objective = _calibration_objective(
                    summary,
                    gate_policy=policy,
                    gate_metrics=gate_metrics,
                ) + (float(win_sec), int(min_enter), int(min_exit))
                if best_objective is None or objective < best_objective:
                    best_objective = objective
                    best_profile = candidate_profile
                    best_summary = summary
                    best_gate_metrics = gate_metrics
                    best_weight_metadata = weight_metadata
                    best_search = {
                        "win_sec": float(win_sec),
                        "min_enter_windows": int(min_enter),
                        "min_exit_windows": int(min_exit),
                    }

    if best_profile is None or best_summary is None or best_search is None or best_gate_metrics is None:
        raise RuntimeError("no valid calibration configuration found during validation search")

    final_model_params: dict[str, Any] = {"Nh": DEFAULT_NH, "state": {}}
    if (weight_mode == "fbcca_diag" or resolved_subband_mode != "chen_fixed") and best_weight_metadata is not None:
        optimized_model_params = best_weight_metadata.get("optimized_model_params")
        if isinstance(optimized_model_params, dict) and optimized_model_params:
            final_model_params.update({key: value for key, value in optimized_model_params.items() if key != "state"})
        else:
            optimized_weights = best_weight_metadata.get("optimized_channel_weights", best_weight_metadata.get("optimized_weights"))
            normalized = normalize_channel_weights(optimized_weights, channels=len(selected_channels))
            if normalized is not None and weight_mode == "fbcca_diag":
                final_model_params["channel_weight_mode"] = "fbcca_diag"
                final_model_params["channel_weights"] = [float(value) for value in normalized]
            optimized_subband_weights = best_weight_metadata.get("optimized_subband_weights")
            if optimized_subband_weights is not None:
                final_model_params["subband_weight_mode"] = str(resolved_subband_mode)
                final_model_params["subband_weights"] = [float(value) for value in optimized_subband_weights]
            optimized_subband_params = best_weight_metadata.get("optimized_subband_params")
            if isinstance(optimized_subband_params, dict):
                final_model_params["subband_weight_params"] = json_safe(dict(optimized_subband_params))

    final_decoder = create_decoder(
        DEFAULT_MODEL_NAME,
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=float(best_search["win_sec"]),
        step_sec=step_sec,
        model_params=final_model_params,
        compute_backend=compute_backend,
        gpu_device=gpu_device,
        gpu_precision=gpu_precision,
        gpu_warmup=gpu_warmup,
        gpu_cache_policy=gpu_cache_policy,
    )
    all_rows = build_feature_rows_with_decoder(final_decoder, selected_segments)
    refit_profile = fit_threshold_profile(
        all_rows,
        freqs=freqs,
        win_sec=float(best_search["win_sec"]),
        step_sec=step_sec,
        min_enter_windows=int(best_search["min_enter_windows"]),
        min_exit_windows=int(best_search["min_exit_windows"]),
        gate_policy=policy,
        evaluation_rows=all_rows,
        dynamic_stop_enabled=dynamic_stop_enabled,
        dynamic_stop_alpha=dynamic_stop_alpha,
        control_state_mode=control_state_mode,
    )
    refit_summary = summarize_profile_quality(all_rows, refit_profile)
    refit_gate_metrics = evaluate_profile_on_feature_rows(all_rows, refit_profile)

    channel_weights = final_model_params.get("channel_weights")
    profile_channel_weights = None
    if channel_weights is not None:
        profile_channel_weights = tuple(float(value) for value in channel_weights)
    subband_weights = final_model_params.get("subband_weights")
    profile_subband_weights = None
    if subband_weights is not None:
        profile_subband_weights = tuple(float(value) for value in subband_weights)
    spatial_filter_rank = final_model_params.get("spatial_filter_rank")
    profile_spatial_rank = None if spatial_filter_rank is None else int(spatial_filter_rank)
    profile_spatial_mode = (
        str(final_model_params.get("spatial_filter_mode"))
        if final_model_params.get("spatial_filter_mode") is not None
        else resolved_spatial_mode
    )
    profile_spatial_state = (
        dict(final_model_params.get("spatial_filter_state"))
        if isinstance(final_model_params.get("spatial_filter_state"), dict)
        else None
    )
    profile_joint_training = dict(best_weight_metadata) if isinstance(best_weight_metadata, dict) else None
    metadata = {
        "source": "calibration",
        "calibration_seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "gate_policy": policy,
        "channel_weight_mode": weight_mode,
        "subband_weight_mode": resolved_subband_mode,
        "spatial_filter_mode": resolved_spatial_mode,
        "spatial_rank_candidates": [int(value) for value in resolved_spatial_ranks],
        "joint_weight_iters": int(resolved_joint_iters),
        "weight_cv_folds": int(resolved_weight_cv_folds),
        "spatial_source_model": resolved_spatial_source,
        "selected_eeg_channels": [int(channel) for channel in selected_channels],
        "channel_selection": channel_scores,
        "validation_search": best_search,
        "validation_summary": best_summary,
        "validation_gate_metrics": best_gate_metrics,
        "refit_summary": refit_summary,
        "refit_gate_metrics": refit_gate_metrics,
        "split_metadata": {
            "fit_segments": int(len(fit_segments)),
            "gate_segments": int(len(gate_segments)),
            "holdout_segments": int(len(holdout_segments)),
        },
        "channel_weight_training": best_weight_metadata,
        "has_stat_model": profile_has_stat_model(refit_profile),
    }
    return replace(
        refit_profile,
        eeg_channels=selected_channels,
        model_name=DEFAULT_MODEL_NAME,
        model_params=final_model_params,
        calibration_split_seed=int(seed),
        benchmark_metrics=None,
        gate_policy=policy,
        dynamic_stop=dict(refit_profile.dynamic_stop or {}),
        channel_weight_mode=weight_mode,
        channel_weights=profile_channel_weights,
        subband_weight_mode=resolved_subband_mode,
        subband_weights=profile_subband_weights,
        subband_weight_params=(
            dict(final_model_params.get("subband_weight_params"))
            if isinstance(final_model_params.get("subband_weight_params"), dict)
            else None
        ),
        spatial_filter_mode=profile_spatial_mode,
        spatial_filter_rank=profile_spatial_rank,
        spatial_filter_state=profile_spatial_state,
        joint_weight_training=profile_joint_training,
        metadata=metadata,
    ), metadata


def _quantile_candidates(values: np.ndarray, quantiles: Sequence[float], *, floor: float = 0.0) -> list[float]:
    if values.size == 0:
        return [floor]
    candidates = [max(float(item), floor) for item in np.quantile(values, quantiles)]
    return [float(value) for value in sorted({round(item, 6) for item in candidates})]


def select_control_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    non_idle_rows = [row for row in rows if row.get("expected_freq") is not None]
    correct_rows = [row for row in non_idle_rows if bool(row.get("correct"))]
    if correct_rows:
        return correct_rows, "correct_only"
    if not non_idle_rows:
        return [], "empty"
    ranked = sorted(
        non_idle_rows,
        key=lambda row: (
            float(row.get("top1_score", 0.0)),
            float(row.get("margin", 0.0)),
            float(row.get("ratio", 1.0)),
        ),
        reverse=True,
    )
    keep_count = max(1, min(len(ranked), max(8, int(math.ceil(len(ranked) * 0.5)))))
    return ranked[:keep_count], "top_non_idle_fallback"


def _fit_frequency_specific_thresholds(
    rows: Sequence[dict[str, Any]],
    *,
    freqs: Sequence[float],
    global_profile: ThresholdProfile,
    control_means: np.ndarray,
    control_stds: np.ndarray,
    idle_means: np.ndarray,
    idle_stds: np.ndarray,
) -> Optional[dict[str, dict[str, Any]]]:
    control_rows, _ = select_control_rows(rows)
    idle_rows_global = [dict(row) for row in rows if row.get("expected_freq") is None]
    if not control_rows or not idle_rows_global:
        return None

    output: dict[str, dict[str, Any]] = {}
    per_freq_idle_min = max(4, int(math.ceil(len(idle_rows_global) / max(len(freqs), 1))))
    for freq in tuple(float(value) for value in freqs):
        freq_key = _frequency_key(freq)
        if freq_key is None:
            continue
        control_rows_freq = [
            dict(row)
            for row in control_rows
            if row.get("expected_freq") is not None
            and abs(float(row.get("expected_freq")) - float(freq)) < 1e-8
            and row.get("pred_freq") is not None
            and abs(float(row.get("pred_freq")) - float(freq)) < 1e-8
        ]
        if len(control_rows_freq) < 4:
            control_rows_freq = [
                dict(row)
                for row in control_rows
                if row.get("expected_freq") is not None
                and abs(float(row.get("expected_freq")) - float(freq)) < 1e-8
            ]
        idle_rows_freq = [
            dict(row)
            for row in idle_rows_global
            if row.get("pred_freq") is not None and abs(float(row.get("pred_freq")) - float(freq)) < 1e-8
        ]
        if len(control_rows_freq) < 4 or len(idle_rows_freq) < per_freq_idle_min:
            idle_rows_freq = list(idle_rows_global)
        if len(control_rows_freq) < 4 or len(idle_rows_freq) < 4:
            continue

        control_scores = np.asarray([float(row["top1_score"]) for row in control_rows_freq], dtype=float)
        control_ratios = np.asarray([float(row["ratio"]) for row in control_rows_freq], dtype=float)
        control_margins = np.asarray([float(row["margin"]) for row in control_rows_freq], dtype=float)
        idle_scores = np.asarray([float(row["top1_score"]) for row in idle_rows_freq], dtype=float)
        idle_ratios = np.asarray([float(row["ratio"]) for row in idle_rows_freq], dtype=float)
        idle_margins = np.asarray([float(row["margin"]) for row in idle_rows_freq], dtype=float)
        score_grid = sorted(
            {
                *(_quantile_candidates(control_scores, (0.10, 0.20, 0.30, 0.40, 0.50), floor=0.0)),
                *(_quantile_candidates(idle_scores, (0.80, 0.90, 0.95, 0.98), floor=0.0)),
            }
        )
        ratio_grid = sorted(
            {
                *(_quantile_candidates(control_ratios, (0.10, 0.20, 0.30, 0.40, 0.50), floor=1.0)),
                *(_quantile_candidates(idle_ratios, (0.80, 0.90, 0.95, 0.98), floor=1.0)),
            }
        )
        margin_grid = sorted(
            {
                *(_quantile_candidates(control_margins, (0.10, 0.20, 0.30, 0.40, 0.50), floor=0.0)),
                *(_quantile_candidates(idle_margins, (0.80, 0.90, 0.95, 0.98), floor=0.0)),
            }
        )

        best_local = (
            float(global_profile.enter_score_th),
            float(global_profile.enter_ratio_th),
            float(global_profile.enter_margin_th),
        )
        best_objective: Optional[tuple[float, float, float, float, float]] = None
        for score_th, ratio_th, margin_th in product(score_grid, ratio_grid, margin_grid):
            idle_fp = float(
                np.mean((idle_scores >= score_th) & (idle_ratios >= ratio_th) & (idle_margins >= margin_th))
            )
            control_recall = float(
                np.mean(
                    (control_scores >= score_th)
                    & (control_ratios >= ratio_th)
                    & (control_margins >= margin_th)
                )
            )
            objective = (
                idle_fp,
                -control_recall,
                float(score_th),
                float(ratio_th),
                float(margin_th),
            )
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_local = (float(score_th), float(ratio_th), float(margin_th))

        control_log_lrs = np.asarray(
            [
                gaussian_log_likelihood_ratio(
                    row,
                    control_means=control_means,
                    control_stds=control_stds,
                    idle_means=idle_means,
                    idle_stds=idle_stds,
                )
                for row in control_rows_freq
            ],
            dtype=float,
        )
        idle_log_lrs = np.asarray(
            [
                gaussian_log_likelihood_ratio(
                    row,
                    control_means=control_means,
                    control_stds=control_stds,
                    idle_means=idle_means,
                    idle_stds=idle_stds,
                )
                for row in idle_rows_freq
            ],
            dtype=float,
        )
        best_log_lr_th = float(global_profile.enter_log_lr_th) if global_profile.enter_log_lr_th is not None else None
        exit_log_lr_th = float(global_profile.exit_log_lr_th) if global_profile.exit_log_lr_th is not None else None
        if control_log_lrs.size and idle_log_lrs.size:
            log_lr_grid = sorted(
                {
                    *(_quantile_candidates(control_log_lrs, (0.05, 0.10, 0.20, 0.30, 0.40), floor=-1_000_000.0)),
                    *(_quantile_candidates(idle_log_lrs, (0.80, 0.90, 0.95, 0.98), floor=-1_000_000.0)),
                }
            )
            best_log_objective: Optional[tuple[float, float, float]] = None
            for log_lr_th in log_lr_grid:
                idle_fp = float(np.mean(idle_log_lrs >= log_lr_th))
                control_recall = float(np.mean(control_log_lrs >= log_lr_th))
                objective = (idle_fp, -control_recall, float(log_lr_th))
                if best_log_objective is None or objective < best_log_objective:
                    best_log_objective = objective
                    best_log_lr_th = float(log_lr_th)
            exit_floor = float(np.quantile(control_log_lrs, 0.05))
            if best_log_lr_th is not None:
                exit_log_lr_th = min(float(best_log_lr_th), exit_floor)

        output[freq_key] = {
            "freq": float(freq),
            "enter_score_th": float(best_local[0]),
            "enter_ratio_th": float(best_local[1]),
            "enter_margin_th": float(best_local[2]),
            "exit_score_th": 0.85 * float(best_local[0]),
            "exit_ratio_th": 0.95 * float(best_local[1]),
            "switch_enter_score_th": 0.95 * float(best_local[0]),
            "switch_enter_ratio_th": float(best_local[1]),
            "switch_enter_margin_th": 0.80 * float(best_local[2]),
            "enter_log_lr_th": None if best_log_lr_th is None else float(best_log_lr_th),
            "exit_log_lr_th": None if exit_log_lr_th is None else float(exit_log_lr_th),
            "n_control_rows": int(len(control_rows_freq)),
            "n_idle_rows": int(len(idle_rows_freq)),
        }
    return output or None


def fit_threshold_profile(
    feature_rows: Iterable[dict[str, Any]],
    *,
    freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    min_enter_windows: int = 2,
    min_exit_windows: int = 2,
    gate_policy: str = DEFAULT_GATE_POLICY,
    evaluation_rows: Optional[Sequence[dict[str, Any]]] = None,
    dynamic_stop_enabled: bool = DEFAULT_DYNAMIC_STOP_ENABLED,
    dynamic_stop_alpha: float = DEFAULT_DYNAMIC_STOP_ALPHA,
    control_state_mode: str = DEFAULT_CONTROL_STATE_MODE,
) -> ThresholdProfile:
    policy = parse_gate_policy(gate_policy)
    resolved_control_state_mode = parse_control_state_mode(control_state_mode)
    rows = list(feature_rows)
    objective_rows = list(evaluation_rows) if evaluation_rows is not None else list(rows)
    control_rows, _control_strategy = select_control_rows(rows)
    idle_rows = [row for row in rows if row.get("expected_freq") is None]
    if not control_rows:
        raise ValueError("calibration did not produce any usable non-idle windows")
    if not idle_rows:
        raise ValueError("calibration did not produce any idle windows")

    control_scores = np.array([float(row["top1_score"]) for row in control_rows], dtype=float)
    control_ratios = np.array([float(row["ratio"]) for row in control_rows], dtype=float)
    control_margins = np.array([float(row["margin"]) for row in control_rows], dtype=float)
    idle_scores = np.array([float(row["top1_score"]) for row in idle_rows], dtype=float)
    idle_ratios = np.array([float(row["ratio"]) for row in idle_rows], dtype=float)
    idle_margins = np.array([float(row["margin"]) for row in idle_rows], dtype=float)

    score_grid = sorted(
        {
            *(_quantile_candidates(control_scores, (0.10, 0.20, 0.30, 0.40, 0.50), floor=0.0)),
            *(_quantile_candidates(idle_scores, (0.80, 0.90, 0.95, 0.98), floor=0.0)),
        }
    )
    ratio_grid = sorted(
        {
            *(_quantile_candidates(control_ratios, (0.10, 0.20, 0.30, 0.40, 0.50), floor=1.0)),
            *(_quantile_candidates(idle_ratios, (0.80, 0.90, 0.95, 0.98), floor=1.0)),
        }
    )
    margin_grid = sorted(
        {
            *(_quantile_candidates(control_margins, (0.10, 0.20, 0.30, 0.40, 0.50), floor=0.0)),
            *(_quantile_candidates(idle_margins, (0.80, 0.90, 0.95, 0.98), floor=0.0)),
        }
    )

    best: tuple[float, float, float] | None = None
    best_objective: Optional[tuple[float, ...]] = None
    for score_th, ratio_th, margin_th in product(score_grid, ratio_grid, margin_grid):
        if policy in {"balanced", "speed"}:
            candidate_profile = ThresholdProfile(
                freqs=(float(freqs[0]), float(freqs[1]), float(freqs[2]), float(freqs[3])),
                win_sec=float(win_sec),
                step_sec=float(step_sec),
                enter_score_th=float(score_th),
                enter_ratio_th=float(ratio_th),
                enter_margin_th=float(margin_th),
                exit_score_th=0.85 * float(score_th),
                exit_ratio_th=0.95 * float(ratio_th),
                min_enter_windows=int(min_enter_windows),
                min_exit_windows=int(min_exit_windows),
                gate_policy=policy,
                min_switch_windows=1,
                switch_enter_score_th=0.95 * float(score_th),
                switch_enter_ratio_th=float(ratio_th),
                switch_enter_margin_th=0.80 * float(margin_th),
                dynamic_stop={
                    "enabled": False,
                    "alpha": float(dynamic_stop_alpha),
                    "enter_acc_th": None,
                    "exit_acc_th": None,
                },
            )
            metrics = evaluate_profile_on_feature_rows(objective_rows, candidate_profile)
            objective = _calibration_objective(
                {},
                gate_policy=policy,
                gate_metrics=metrics,
            ) + (float(score_th), float(ratio_th), float(margin_th))
        else:
            idle_pass = (idle_scores >= score_th) & (idle_ratios >= ratio_th) & (idle_margins >= margin_th)
            control_pass = (
                (control_scores >= score_th) & (control_ratios >= ratio_th) & (control_margins >= margin_th)
            )
            idle_false_positive = float(idle_pass.mean()) if idle_pass.size else 1.0
            control_recall = float(control_pass.mean()) if control_pass.size else 0.0
            objective = (
                idle_false_positive,
                -control_recall,
                float(score_th),
                float(ratio_th),
                float(margin_th),
            )
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best = (float(score_th), float(ratio_th), float(margin_th))

    if best is None:
        raise RuntimeError("threshold search failed")

    enter_score_th, enter_ratio_th, enter_margin_th = best
    control_means, control_stds = _safe_feature_stats(control_rows)
    idle_means, idle_stds = _safe_feature_stats(idle_rows)
    control_log_lrs = np.array(
        [
            gaussian_log_likelihood_ratio(
                row,
                control_means=control_means,
                control_stds=control_stds,
                idle_means=idle_means,
                idle_stds=idle_stds,
            )
            for row in control_rows
        ],
        dtype=float,
    )
    idle_log_lrs = np.array(
        [
            gaussian_log_likelihood_ratio(
                row,
                control_means=control_means,
                control_stds=control_stds,
                idle_means=idle_means,
                idle_stds=idle_stds,
            )
            for row in idle_rows
        ],
        dtype=float,
    )
    log_lr_grid = sorted(
        {
            *(_quantile_candidates(control_log_lrs, (0.05, 0.10, 0.20, 0.30, 0.40), floor=-1_000_000.0)),
            *(_quantile_candidates(idle_log_lrs, (0.80, 0.90, 0.95, 0.98), floor=-1_000_000.0)),
        }
    )
    best_log_lr_th: Optional[float] = None
    best_log_lr_objective: Optional[tuple[float, float, float]] = None
    for log_lr_th in log_lr_grid:
        idle_fp = float((idle_log_lrs >= log_lr_th).mean()) if idle_log_lrs.size else 1.0
        control_recall = float((control_log_lrs >= log_lr_th).mean()) if control_log_lrs.size else 0.0
        objective = (idle_fp, -control_recall, float(log_lr_th))
        if best_log_lr_objective is None or objective < best_log_lr_objective:
            best_log_lr_objective = objective
            best_log_lr_th = float(log_lr_th)

    if best_log_lr_th is None:
        raise RuntimeError("log-likelihood threshold search failed")

    exit_log_lr_floor = float(np.quantile(control_log_lrs, 0.05)) if control_log_lrs.size else best_log_lr_th
    exit_log_lr_th = min(best_log_lr_th, exit_log_lr_floor)
    freq_tuple = tuple(float(freq) for freq in freqs)
    dynamic_payload = {
        "enabled": bool(dynamic_stop_enabled),
        "alpha": float(dynamic_stop_alpha),
        "enter_acc_th": None,
        "exit_acc_th": None,
    }
    if policy == "speed":
        speed_weights = {
            "idle_fp_per_min": float(DEFAULT_SPEED_OBJECTIVE_IDLE_WEIGHT),
            "control_recall": float(DEFAULT_SPEED_OBJECTIVE_RECALL_WEIGHT),
            "switch_latency_s": float(DEFAULT_SPEED_OBJECTIVE_SWITCH_WEIGHT),
            "release_latency_s": float(DEFAULT_SPEED_OBJECTIVE_RELEASE_WEIGHT),
        }
    else:
        speed_weights = None
    base_profile = ThresholdProfile(
        freqs=(freq_tuple[0], freq_tuple[1], freq_tuple[2], freq_tuple[3]),
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        enter_score_th=enter_score_th,
        enter_ratio_th=enter_ratio_th,
        enter_margin_th=enter_margin_th,
        exit_score_th=0.85 * enter_score_th,
        exit_ratio_th=0.95 * enter_ratio_th,
        min_enter_windows=int(min_enter_windows),
        min_exit_windows=int(min_exit_windows),
        min_switch_windows=1,
        switch_enter_score_th=0.95 * float(enter_score_th),
        switch_enter_ratio_th=float(enter_ratio_th),
        switch_enter_margin_th=0.80 * float(enter_margin_th),
        speed_objective_weights=speed_weights,
        enter_log_lr_th=best_log_lr_th,
        exit_log_lr_th=exit_log_lr_th,
        control_feature_means={name: float(value) for name, value in zip(MODEL_FEATURE_NAMES, control_means)},
        control_feature_stds={name: float(value) for name, value in zip(MODEL_FEATURE_NAMES, control_stds)},
        idle_feature_means={name: float(value) for name, value in zip(MODEL_FEATURE_NAMES, idle_means)},
        idle_feature_stds={name: float(value) for name, value in zip(MODEL_FEATURE_NAMES, idle_stds)},
        gate_policy=policy,
        dynamic_stop=dynamic_payload,
        control_state_mode=resolved_control_state_mode,
    )
    if resolved_control_state_mode in {"frequency-specific-threshold", "frequency-specific-logistic"}:
        frequency_specific_thresholds = _fit_frequency_specific_thresholds(
            rows,
            freqs=freqs,
            global_profile=base_profile,
            control_means=control_means,
            control_stds=control_stds,
            idle_means=idle_means,
            idle_stds=idle_stds,
        )
        if frequency_specific_thresholds is not None:
            base_profile = replace(
                base_profile,
                frequency_specific_thresholds=frequency_specific_thresholds,
            )

    if not bool(dynamic_stop_enabled):
        return base_profile

    ordered_rows = sorted(
        [dict(row) for row in objective_rows],
        key=lambda row: (int(row.get("trial_id", -1_000_000)), int(row.get("window_index", 0))),
    )
    acc_values: list[float] = []
    accum = 0.0
    for row in ordered_rows:
        value = gaussian_log_likelihood_ratio(
            row,
            control_means=control_means,
            control_stds=control_stds,
            idle_means=idle_means,
            idle_stds=idle_stds,
        )
        accum = float(dynamic_stop_alpha) * float(accum) + float(value)
        acc_values.append(float(accum))
    if not acc_values:
        return base_profile

    acc_array = np.asarray(acc_values, dtype=float)
    enter_grid = _quantile_candidates(acc_array, (0.55, 0.65, 0.75, 0.85, 0.92), floor=-1_000_000.0)
    exit_grid = _quantile_candidates(acc_array, (0.05, 0.15, 0.25, 0.35, 0.45), floor=-1_000_000.0)
    best_dynamic_profile = base_profile
    baseline_metrics = evaluate_profile_on_feature_rows(objective_rows, base_profile)
    best_dynamic_objective = _calibration_objective(
        {},
        gate_policy=policy,
        gate_metrics=baseline_metrics,
    ) + (float("inf"), float("inf"))
    for enter_acc_th in enter_grid:
        for exit_acc_th in exit_grid:
            if float(exit_acc_th) > float(enter_acc_th):
                continue
            candidate_dynamic = {
                "enabled": True,
                "alpha": float(dynamic_stop_alpha),
                "enter_acc_th": float(enter_acc_th),
                "exit_acc_th": float(exit_acc_th),
            }
            candidate_profile = replace(base_profile, dynamic_stop=candidate_dynamic)
            candidate_metrics = evaluate_profile_on_feature_rows(objective_rows, candidate_profile)
            candidate_objective = _calibration_objective(
                {},
                gate_policy=policy,
                gate_metrics=candidate_metrics,
            ) + (float(enter_acc_th), float(exit_acc_th))
            if best_dynamic_objective is None or candidate_objective < best_dynamic_objective:
                best_dynamic_objective = candidate_objective
                best_dynamic_profile = candidate_profile
    return best_dynamic_profile


def build_benchmark_eval_trials(
    freqs: Sequence[float],
    *,
    target_repeats: int,
    idle_repeats: int,
    switch_trials: int,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> list[TrialSpec]:
    trials = build_calibration_trials(
        freqs,
        target_repeats=target_repeats,
        idle_repeats=idle_repeats,
        shuffle=True,
        seed=seed,
    )
    freq_list = [float(freq) for freq in freqs]
    rng = random.Random(int(seed) + 17)
    next_trial_id = max((trial.trial_id for trial in trials), default=-1) + 1
    last_freq = rng.choice(freq_list)
    for switch_index in range(max(int(switch_trials), 0)):
        choices = [freq for freq in freq_list if abs(freq - last_freq) > 1e-8]
        if not choices:
            break
        next_freq = rng.choice(choices)
        trials.append(
            TrialSpec(
                label=f"switch_to_{next_freq:g}Hz",
                expected_freq=float(next_freq),
                trial_id=next_trial_id,
                block_index=target_repeats + switch_index,
            )
        )
        next_trial_id += 1
        last_freq = next_freq
    return trials


def split_trial_segments_for_benchmark(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    seed: int = DEFAULT_CALIBRATION_SEED,
    train_ratio: float = 0.6,
    gate_ratio: float = 0.2,
) -> tuple[list[tuple[TrialSpec, np.ndarray]], list[tuple[TrialSpec, np.ndarray]], list[tuple[TrialSpec, np.ndarray]]]:
    indexed = list(enumerate(trial_segments))
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, (trial, _segment) in indexed:
        key = "idle" if trial.expected_freq is None else f"{float(trial.expected_freq):g}"
        grouped[key].append(int(index))
    rng = random.Random(int(seed))
    train_ids: set[int] = set()
    gate_ids: set[int] = set()
    eval_ids: set[int] = set()
    for entries in grouped.values():
        block = [int(item) for item in entries]
        rng.shuffle(block)
        count = len(block)
        if count <= 1:
            train_ids.update(block)
            continue
        if count == 2:
            train_ids.update(block[:1])
            gate_ids.update(block[1:])
            continue
        train_count = int(round(count * float(train_ratio)))
        gate_count = int(round(count * float(gate_ratio)))
        train_count = max(1, min(train_count, count - 2))
        gate_count = max(1, min(gate_count, count - train_count - 1))
        eval_count = count - train_count - gate_count
        if eval_count <= 0:
            eval_count = 1
            if gate_count > 1:
                gate_count -= 1
            else:
                train_count = max(1, train_count - 1)
        train_ids.update(block[:train_count])
        gate_ids.update(block[train_count : train_count + gate_count])
        eval_ids.update(block[train_count + gate_count :])

    train: list[tuple[TrialSpec, np.ndarray]] = []
    gate: list[tuple[TrialSpec, np.ndarray]] = []
    evaluation: list[tuple[TrialSpec, np.ndarray]] = []
    for index, item in indexed:
        if index in train_ids:
            train.append(item)
        elif index in gate_ids:
            gate.append(item)
        elif index in eval_ids:
            evaluation.append(item)
        else:
            train.append(item)
    return train, gate, evaluation


def build_feature_rows_with_decoder(
    decoder: BaseSSVEPDecoder,
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial, segment in trial_segments:
        rows.extend(
            decoder.iter_window_features(
                segment,
                expected_freq=trial.expected_freq,
                label=trial.label,
                trial_id=trial.trial_id,
                block_index=trial.block_index,
            )
        )
    return rows


def compute_itr_bits_per_minute(
    *,
    accuracy: float,
    class_count: int,
    decision_time_sec: float,
) -> float:
    n = max(int(class_count), 2)
    p = min(max(float(accuracy), 1e-6), 1.0 - 1e-6)
    t = max(float(decision_time_sec), 1e-3)
    bits_per_trial = math.log2(n) + p * math.log2(p) + (1.0 - p) * math.log2((1.0 - p) / (n - 1.0))
    return float(bits_per_trial * 60.0 / t)


def _freq_label(freq: float) -> str:
    return f"{float(freq):g}"


def _nearest_freq(freq: float, freqs: Sequence[float]) -> float:
    candidates = [float(item) for item in freqs]
    if not candidates:
        raise ValueError("freq list is empty")
    return min(candidates, key=lambda item: abs(float(item) - float(freq)))


def compute_confusion_matrix_counts(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> list[list[int]]:
    index = {str(label): idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, pred_label in zip(y_true, y_pred):
        true_key = str(true_label)
        pred_key = str(pred_label)
        if true_key not in index or pred_key not in index:
            continue
        matrix[index[true_key]][index[pred_key]] += 1
    return matrix


def compute_macro_f1_from_confusion(confusion: Sequence[Sequence[int]]) -> float:
    matrix = np.asarray(confusion, dtype=float)
    if matrix.size == 0 or matrix.shape[0] == 0:
        return 0.0
    f1_values: list[float] = []
    for idx in range(matrix.shape[0]):
        tp = float(matrix[idx, idx])
        fp = float(np.sum(matrix[:, idx]) - tp)
        fn = float(np.sum(matrix[idx, :]) - tp)
        precision = tp / (tp + fp) if (tp + fp) > 1e-12 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 1e-12 else 0.0
        if (precision + recall) <= 1e-12:
            f1_values.append(0.0)
        else:
            f1_values.append(float(2.0 * precision * recall / (precision + recall)))
    return float(np.mean(np.asarray(f1_values, dtype=float))) if f1_values else 0.0


def compute_classification_metrics(
    *,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    decision_time_samples_s: Sequence[float],
    itr_class_count: int,
    decision_time_fallback_s: float,
) -> dict[str, Any]:
    ordered_labels = [str(item) for item in labels]
    confusion = compute_confusion_matrix_counts(y_true, y_pred, ordered_labels)
    total = int(sum(sum(row) for row in confusion))
    correct = int(sum(confusion[idx][idx] for idx in range(len(confusion))))
    accuracy = float(correct / total) if total > 0 else 0.0
    macro_f1 = compute_macro_f1_from_confusion(confusion)
    times = [float(item) for item in decision_time_samples_s if np.isfinite(float(item))]
    mean_decision_time_s = float(np.mean(np.asarray(times, dtype=float))) if times else float("inf")
    itr_reference_time = (
        mean_decision_time_s if np.isfinite(mean_decision_time_s) else float(max(decision_time_fallback_s, 1e-3))
    )
    itr_bpm = compute_itr_bits_per_minute(
        accuracy=accuracy,
        class_count=max(int(itr_class_count), 2),
        decision_time_sec=itr_reference_time,
    )
    return {
        "acc": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": [[int(item) for item in row] for row in confusion],
        "labels": ordered_labels,
        "y_true": [str(item) for item in y_true],
        "y_pred": [str(item) for item in y_pred],
        "mean_decision_time_s": mean_decision_time_s,
        "itr_bpm": itr_bpm,
        "n_total": int(total),
        "n_correct": int(correct),
        "decision_time_samples_s": [float(item) for item in times],
    }


def _decision_latency_with_mode(
    *,
    mode: str,
    first_correct_latency: Optional[float],
    first_any_latency: Optional[float],
    trial_duration_sec: float,
    win_sec: float,
) -> float:
    penalty = float(trial_duration_sec + win_sec)
    if mode == "fixed-window":
        return float(win_sec)
    if mode == "first-any":
        return float(first_any_latency) if first_any_latency is not None else penalty
    return float(first_correct_latency) if first_correct_latency is not None else penalty


def evaluate_decoder_on_trials_v2(
    decoder: BaseSSVEPDecoder,
    profile: ThresholdProfile,
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    dynamic_stop_enabled: Optional[bool] = None,
    metric_scope: str = DEFAULT_METRIC_SCOPE,
    decision_time_mode: str = DEFAULT_DECISION_TIME_MODE,
    async_decision_time_mode: Optional[str] = None,
    paper_decision_time_mode: Optional[str] = None,
) -> dict[str, Any]:
    scope = parse_metric_scope(metric_scope)
    default_mode = parse_decision_time_mode(decision_time_mode)
    paper_mode = parse_decision_time_mode(
        paper_decision_time_mode if paper_decision_time_mode is not None else default_mode
    )
    async_mode = parse_decision_time_mode(
        async_decision_time_mode if async_decision_time_mode is not None else DEFAULT_ASYNC_DECISION_TIME_MODE
    )
    gate_profile = profile
    if dynamic_stop_enabled is not None:
        payload = dict(profile.dynamic_stop or {})
        payload["enabled"] = bool(dynamic_stop_enabled)
        gate_profile = replace(profile, dynamic_stop=payload)
    gate = AsyncDecisionGate.from_profile(gate_profile)
    gate.reset()
    idle_selected_windows = 0
    idle_selected_events = 0
    idle_windows = 0
    idle_duration_sec = 0.0
    control_trials = 0
    control_detected_trials = 0
    detection_latencies: list[float] = []
    switch_latencies: list[float] = []
    release_latencies: list[float] = []
    inference_latencies_ms: list[float] = []
    last_control_freq: Optional[float] = None
    prev_trial_expected_freq: Optional[float] = None
    switch_trials = 0
    switch_detected_trials = 0
    switch_detected_trials_at_2p8s = 0
    switch_penalty_trials = 0
    release_trials = 0
    release_detected_trials = 0
    release_penalty_trials = 0
    control_detected_trials_at_2s = 0
    control_detected_trials_at_3s = 0
    trial_rows: list[dict[str, Any]] = []
    has_explicit_switch_trials = any(
        trial.expected_freq is not None and str(trial.label).startswith("switch_to_")
        for trial, _segment in trial_segments
    )
    selected_active_prev = False

    for trial, segment in trial_segments:
        segment_matrix = np.asarray(segment, dtype=float)
        if segment_matrix.shape[0] < decoder.win_samples:
            continue
        first_correct_latency: Optional[float] = None
        first_selected_any_latency: Optional[float] = None
        first_selected_any_freq: Optional[float] = None
        first_release_latency: Optional[float] = None
        last_pred_freq: Optional[float] = None
        is_switch_trial = False
        if trial.expected_freq is not None:
            if has_explicit_switch_trials:
                is_switch_trial = str(trial.label).startswith("switch_to_")
            elif last_control_freq is not None and abs(last_control_freq - float(trial.expected_freq)) > 1e-8:
                is_switch_trial = True
            if is_switch_trial:
                switch_trials += 1
        window_index = 0
        for start in range(0, segment_matrix.shape[0] - decoder.win_samples + 1, decoder.step_samples):
            window = np.ascontiguousarray(segment_matrix[start : start + decoder.win_samples], dtype=np.float64)
            infer_t0 = time.perf_counter()
            features = decoder.analyze_window(window)
            infer_t1 = time.perf_counter()
            decision = gate.update(features)
            decoder.update_online(decision, window)
            inference_latencies_ms.append((infer_t1 - infer_t0) * 1000.0)
            predicted = decision.get("pred_freq")
            if predicted is not None:
                last_pred_freq = float(predicted)
            selected = decision.get("selected_freq")
            selected_active = selected is not None
            if selected is not None and first_selected_any_latency is None:
                first_selected_any_latency = float(decoder.win_sec + window_index * decoder.step_sec)
                first_selected_any_freq = float(selected)
            if trial.expected_freq is None:
                idle_windows += 1
                if selected_active:
                    idle_selected_windows += 1
                    if not selected_active_prev:
                        idle_selected_events += 1
                if prev_trial_expected_freq is not None and first_release_latency is None and selected is None:
                    first_release_latency = float(decoder.win_sec + window_index * decoder.step_sec)
            else:
                if (
                    first_correct_latency is None
                    and selected is not None
                    and abs(float(selected) - float(trial.expected_freq)) < 1e-8
                ):
                    first_correct_latency = float(decoder.win_sec + window_index * decoder.step_sec)
            selected_active_prev = selected_active
            window_index += 1

        trial_duration_sec = float(segment_matrix.shape[0]) / max(float(decoder.fs), 1.0)
        had_selected = bool(first_selected_any_latency is not None)
        trial_rows.append(
            {
                "label": str(trial.label),
                "expected_freq": None if trial.expected_freq is None else float(trial.expected_freq),
                "trial_duration_sec": float(trial_duration_sec),
                "first_selected_any_latency_s": None
                if first_selected_any_latency is None
                else float(first_selected_any_latency),
                "first_selected_any_freq": None if first_selected_any_freq is None else float(first_selected_any_freq),
                "first_correct_latency_s": None if first_correct_latency is None else float(first_correct_latency),
                "first_release_latency_s": None if first_release_latency is None else float(first_release_latency),
                "last_pred_freq": None if last_pred_freq is None else float(last_pred_freq),
                "had_selected": bool(had_selected),
                "is_switch_trial": bool(is_switch_trial),
            }
        )

        if trial.expected_freq is None:
            idle_duration_sec += float(max(trial_duration_sec, 0.0))
            if prev_trial_expected_freq is not None:
                release_trials += 1
                penalty_latency = float(trial_duration_sec + decoder.win_sec)
                if first_release_latency is None:
                    release_penalty_trials += 1
                    release_latencies.append(penalty_latency)
                else:
                    release_detected_trials += 1
                    release_latencies.append(float(first_release_latency))
            prev_trial_expected_freq = None
            continue
        control_trials += 1
        if first_correct_latency is not None:
            control_detected_trials += 1
            detection_latencies.append(float(first_correct_latency))
            if float(first_correct_latency) <= float(DEFAULT_CONTROL_RECALL_AT_2S_DEADLINE):
                control_detected_trials_at_2s += 1
            if float(first_correct_latency) <= float(DEFAULT_CONTROL_RECALL_AT_3S_DEADLINE):
                control_detected_trials_at_3s += 1
        if is_switch_trial:
            penalty_latency = float(trial_duration_sec + decoder.win_sec)
            if first_correct_latency is None:
                switch_penalty_trials += 1
                switch_latencies.append(penalty_latency)
            else:
                switch_detected_trials += 1
                switch_latencies.append(float(first_correct_latency))
                if float(first_correct_latency) <= float(DEFAULT_SWITCH_DETECT_AT_2P8S_DEADLINE):
                    switch_detected_trials_at_2p8s += 1
        last_control_freq = float(trial.expected_freq)
        prev_trial_expected_freq = float(trial.expected_freq)

    idle_minutes = float(max(idle_duration_sec, 0.0)) / 60.0
    idle_fp_per_min = float(idle_selected_events / idle_minutes) if idle_minutes > 1e-12 else 0.0
    idle_selected_windows_per_min = float(idle_selected_windows / idle_minutes) if idle_minutes > 1e-12 else 0.0
    control_recall = float(control_detected_trials / control_trials) if control_trials else 0.0
    control_miss_rate = float(1.0 - control_recall) if control_trials else 1.0
    median_detection_latency = (
        float(np.median(np.asarray(detection_latencies, dtype=float))) if detection_latencies else float("inf")
    )
    median_switch_latency = (
        float(np.median(np.asarray(switch_latencies, dtype=float))) if switch_latencies else float("inf")
    )
    median_release_latency = (
        float(np.median(np.asarray(release_latencies, dtype=float))) if release_latencies else float("inf")
    )
    switch_detect_rate = float(switch_detected_trials / switch_trials) if switch_trials else 0.0
    switch_detect_rate_at_2p8s = (
        float(switch_detected_trials_at_2p8s / switch_trials) if switch_trials else 0.0
    )
    release_detect_rate = float(release_detected_trials / release_trials) if release_trials else 0.0
    control_recall_at_2s = float(control_detected_trials_at_2s / control_trials) if control_trials else 0.0
    control_recall_at_3s = float(control_detected_trials_at_3s / control_trials) if control_trials else 0.0
    async_itr = compute_itr_bits_per_minute(
        accuracy=control_recall,
        class_count=len(decoder.freqs),
        decision_time_sec=median_detection_latency if np.isfinite(median_detection_latency) else decoder.win_sec,
    )
    async_metrics = {
        "idle_fp_per_min": idle_fp_per_min,
        "idle_selected_windows_per_min": idle_selected_windows_per_min,
        "control_recall": control_recall,
        "control_recall_at_2s": control_recall_at_2s,
        "control_recall_at_3s": control_recall_at_3s,
        "control_miss_rate": control_miss_rate,
        "switch_detect_rate": switch_detect_rate,
        "switch_detect_rate_at_2.8s": switch_detect_rate_at_2p8s,
        "switch_latency_s": median_switch_latency,
        "release_detect_rate": release_detect_rate,
        "release_latency_s": median_release_latency,
        "detection_latency_s": median_detection_latency,
        "itr_bpm": async_itr,
        "inference_ms": float(np.mean(inference_latencies_ms)) if inference_latencies_ms else float("inf"),
        "control_trials": float(control_trials),
        "idle_windows": float(idle_windows),
        "idle_selected_windows": float(idle_selected_windows),
        "idle_selected_events": float(idle_selected_events),
        "idle_fp_event_count": float(idle_selected_events),
        "idle_time_sec": float(idle_duration_sec),
        "idle_time_min": float(idle_minutes),
        "switch_trials": float(switch_trials),
        "switch_detected_trials": float(switch_detected_trials),
        "switch_detected_trials_at_2.8s": float(switch_detected_trials_at_2p8s),
        "switch_penalty_trials": float(switch_penalty_trials),
        "release_trials": float(release_trials),
        "release_detected_trials": float(release_detected_trials),
        "release_penalty_trials": float(release_penalty_trials),
    }

    freq_labels = [_freq_label(freq) for freq in decoder.freqs]
    labels2 = ("idle", "control")
    labels5 = ("idle", *freq_labels)

    def _build_lens_metrics(mode: str) -> tuple[dict[str, Any], dict[str, Any], Optional[dict[str, Any]]]:
        y4_true: list[str] = []
        y4_pred: list[str] = []
        times4: list[float] = []
        y2_true: list[str] = []
        y2_pred: list[str] = []
        times2: list[float] = []
        y5_true: list[str] = []
        y5_pred: list[str] = []
        times5: list[float] = []

        for row in trial_rows:
            expected = row["expected_freq"]
            trial_duration_sec = float(row["trial_duration_sec"])
            first_correct_latency = row["first_correct_latency_s"]
            first_any_latency = row["first_selected_any_latency_s"]
            first_any_freq = row["first_selected_any_freq"]
            last_pred_freq = row["last_pred_freq"]
            had_selected = bool(row["had_selected"])

            true_2 = "idle" if expected is None else "control"
            if expected is None:
                pred_2 = "control" if had_selected else "idle"
            elif mode == "first-correct":
                pred_2 = "control" if first_correct_latency is not None else "idle"
            elif mode == "first-any":
                pred_2 = "control" if first_any_latency is not None else "idle"
            else:
                pred_2 = "control" if had_selected else "idle"
            y2_true.append(true_2)
            y2_pred.append(pred_2)
            if expected is None:
                if mode == "first-any" and first_any_latency is not None:
                    times2.append(float(first_any_latency))
                else:
                    times2.append(float(decoder.win_sec))
            else:
                times2.append(
                    _decision_latency_with_mode(
                        mode=mode,
                        first_correct_latency=(None if first_correct_latency is None else float(first_correct_latency)),
                        first_any_latency=(None if first_any_latency is None else float(first_any_latency)),
                        trial_duration_sec=trial_duration_sec,
                        win_sec=decoder.win_sec,
                    )
                )

            if expected is not None:
                true_label = _freq_label(float(expected))
                if mode == "first-correct" and first_correct_latency is not None:
                    pred_label = true_label
                else:
                    if mode == "fixed-window":
                        candidate_freq = last_pred_freq if last_pred_freq is not None else first_any_freq
                    else:
                        candidate_freq = first_any_freq if first_any_freq is not None else last_pred_freq
                    if candidate_freq is None:
                        candidate_freq = float(decoder.freqs[0])
                    pred_label = _freq_label(_nearest_freq(float(candidate_freq), decoder.freqs))
                y4_true.append(true_label)
                y4_pred.append(pred_label)
                times4.append(
                    _decision_latency_with_mode(
                        mode=mode,
                        first_correct_latency=(None if first_correct_latency is None else float(first_correct_latency)),
                        first_any_latency=(None if first_any_latency is None else float(first_any_latency)),
                        trial_duration_sec=trial_duration_sec,
                        win_sec=decoder.win_sec,
                    )
                )

            if scope == "5class":
                if expected is None:
                    true_5 = "idle"
                    if mode == "first-any" and first_any_latency is not None:
                        times5.append(float(first_any_latency))
                    else:
                        times5.append(float(decoder.win_sec))
                else:
                    true_5 = _freq_label(float(expected))
                    times5.append(
                        _decision_latency_with_mode(
                            mode=mode,
                            first_correct_latency=(None if first_correct_latency is None else float(first_correct_latency)),
                            first_any_latency=(None if first_any_latency is None else float(first_any_latency)),
                            trial_duration_sec=trial_duration_sec,
                            win_sec=decoder.win_sec,
                        )
                    )
                if expected is not None and mode == "first-correct" and first_correct_latency is not None:
                    pred_5 = _freq_label(float(expected))
                else:
                    if mode == "fixed-window":
                        candidate_freq = last_pred_freq if last_pred_freq is not None else first_any_freq
                    else:
                        candidate_freq = first_any_freq if first_any_freq is not None else last_pred_freq
                    if candidate_freq is None:
                        pred_5 = "idle"
                    else:
                        pred_5 = _freq_label(_nearest_freq(float(candidate_freq), decoder.freqs))
                y5_true.append(true_5)
                y5_pred.append(pred_5)

        metrics_4class = compute_classification_metrics(
            y_true=y4_true,
            y_pred=y4_pred,
            labels=freq_labels,
            decision_time_samples_s=times4,
            itr_class_count=4,
            decision_time_fallback_s=float(decoder.win_sec),
        )
        metrics_2class = compute_classification_metrics(
            y_true=y2_true,
            y_pred=y2_pred,
            labels=labels2,
            decision_time_samples_s=times2,
            itr_class_count=2,
            decision_time_fallback_s=float(decoder.win_sec),
        )
        metrics_5class: Optional[dict[str, Any]]
        if scope == "5class":
            metrics_5class = compute_classification_metrics(
                y_true=y5_true,
                y_pred=y5_pred,
                labels=labels5,
                decision_time_samples_s=times5,
                itr_class_count=5,
                decision_time_fallback_s=float(decoder.win_sec),
            )
        else:
            metrics_5class = None
        return metrics_4class, metrics_2class, metrics_5class

    paper_metrics_4class, paper_metrics_2class, paper_metrics_5class = _build_lens_metrics(paper_mode)
    async_lens_metrics_4class, async_lens_metrics_2class, async_lens_metrics_5class = _build_lens_metrics(async_mode)

    return {
        "async_metrics": async_metrics,
        "metrics_4class": paper_metrics_4class,
        "metrics_2class": paper_metrics_2class,
        "metrics_5class": paper_metrics_5class,
        "paper_lens_metrics_4class": paper_metrics_4class,
        "paper_lens_metrics_2class": paper_metrics_2class,
        "paper_lens_metrics_5class": paper_metrics_5class,
        "paper_lens_decision_time_mode": paper_mode,
        "async_lens_metrics_4class": async_lens_metrics_4class,
        "async_lens_metrics_2class": async_lens_metrics_2class,
        "async_lens_metrics_5class": async_lens_metrics_5class,
        "async_lens_decision_time_mode": async_mode,
        "metric_scope": scope,
        "decision_time_mode": paper_mode,
        "async_decision_time_mode": async_mode,
        "trial_events": trial_rows,
    }


def pack_evaluation_metrics_for_ranking(
    evaluation_bundle: dict[str, Any],
    *,
    metric_scope: str = DEFAULT_METRIC_SCOPE,
) -> dict[str, float]:
    scope = parse_metric_scope(metric_scope)
    async_metrics = dict(evaluation_bundle.get("async_metrics", {}))
    metrics_4 = dict(evaluation_bundle.get("metrics_4class", {}))
    metrics_5 = dict(evaluation_bundle.get("metrics_5class") or {})
    ranking_metrics = dict(async_metrics)
    ranking_metrics["acc_4class"] = float(metrics_4.get("acc", 0.0))
    ranking_metrics["macro_f1_4class"] = float(metrics_4.get("macro_f1", 0.0))
    ranking_metrics["itr_bpm_4class"] = float(metrics_4.get("itr_bpm", 0.0))
    ranking_metrics["mean_decision_time_s_4class"] = float(metrics_4.get("mean_decision_time_s", float("inf")))
    if scope == "5class":
        ranking_metrics["acc_5class"] = float(metrics_5.get("acc", 0.0))
        ranking_metrics["macro_f1_5class"] = float(metrics_5.get("macro_f1", 0.0))
        ranking_metrics["itr_bpm_5class"] = float(metrics_5.get("itr_bpm", 0.0))
    return ranking_metrics


def evaluate_decoder_on_trials(
    decoder: BaseSSVEPDecoder,
    profile: ThresholdProfile,
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    dynamic_stop_enabled: Optional[bool] = None,
) -> dict[str, float]:
    bundle = evaluate_decoder_on_trials_v2(
        decoder,
        profile,
        trial_segments,
        dynamic_stop_enabled=dynamic_stop_enabled,
        metric_scope=DEFAULT_METRIC_SCOPE,
        decision_time_mode=DEFAULT_DECISION_TIME_MODE,
    )
    return dict(bundle.get("async_metrics", {}))


def benchmark_rank_key(
    metrics: dict[str, float],
    *,
    ranking_policy: str = DEFAULT_RANKING_POLICY,
) -> tuple[float, ...]:
    policy = parse_ranking_policy(ranking_policy)
    control_recall = float(metrics.get("control_recall", 0.0))
    recall_threshold = float(DEFAULT_BENCHMARK_RANK_MIN_CONTROL_RECALL)
    low_recall_penalty = max(0.0, recall_threshold - control_recall)
    idle_fp = float(metrics.get("idle_fp_per_min", float("inf")))
    switch_latency = float(metrics.get("switch_latency_s", float("inf")))
    release_latency = float(metrics.get("release_latency_s", float("inf")))
    control_recall_at_3s = float(metrics.get("control_recall_at_3s", control_recall))
    switch_detect_rate_at_2p8s = float(metrics.get("switch_detect_rate_at_2.8s", metrics.get("switch_detect_rate", 0.0)))
    acc_4class = float(metrics.get("acc_4class", 0.0))
    macro_f1_4class = float(metrics.get("macro_f1_4class", 0.0))
    itr_4class = float(metrics.get("itr_bpm_4class", 0.0))
    inference_ms = float(metrics.get("inference_ms", float("inf")))
    if policy == "async-speed":
        return (
            idle_fp,
            -control_recall_at_3s,
            -switch_detect_rate_at_2p8s,
            switch_latency,
            release_latency,
            -acc_4class,
            -macro_f1_4class,
            -itr_4class,
            inference_ms,
        )
    if policy == "paper-first":
        return (
            low_recall_penalty,
            -acc_4class,
            -macro_f1_4class,
            -itr_4class,
            idle_fp,
            -control_recall,
            switch_latency,
            release_latency,
            inference_ms,
        )
    if policy == "dual-board":
        return (
            low_recall_penalty,
            idle_fp,
            -control_recall,
            switch_latency,
            release_latency,
            -acc_4class,
            -macro_f1_4class,
            -itr_4class,
            inference_ms,
        )
    return (
        low_recall_penalty,
        idle_fp,
        -control_recall,
        switch_latency,
        release_latency,
        -acc_4class,
        -macro_f1_4class,
        -itr_4class,
        inference_ms,
    )


def profile_meets_acceptance(metrics: dict[str, float]) -> bool:
    return (
        float(metrics.get("idle_fp_per_min", float("inf"))) <= float(DEFAULT_ACCEPTANCE_IDLE_FP_PER_MIN)
        and float(metrics.get("control_recall", 0.0)) >= float(DEFAULT_ACCEPTANCE_CONTROL_RECALL)
        and float(metrics.get("switch_latency_s", float("inf"))) <= float(DEFAULT_ACCEPTANCE_SWITCH_LATENCY_S)
        and float(metrics.get("release_latency_s", float("inf"))) <= float(DEFAULT_ACCEPTANCE_RELEASE_LATENCY_S)
        and float(metrics.get("inference_ms", float("inf"))) < float(DEFAULT_ACCEPTANCE_INFERENCE_MS)
    )


def benchmark_metric_definition_payload(
    *,
    ranking_policy: str = DEFAULT_RANKING_POLICY,
    metric_scope: str = DEFAULT_METRIC_SCOPE,
    decision_time_mode: str = DEFAULT_DECISION_TIME_MODE,
    async_decision_time_mode: str = DEFAULT_ASYNC_DECISION_TIME_MODE,
) -> dict[str, Any]:
    policy = parse_ranking_policy(ranking_policy)
    scope = parse_metric_scope(metric_scope)
    time_mode = parse_decision_time_mode(decision_time_mode)
    async_time_mode = parse_decision_time_mode(async_decision_time_mode)
    if policy == "paper-first":
        ranking_priority = [
            "low_recall_penalty",
            "acc_4class",
            "macro_f1_4class",
            "itr_bpm_4class",
            "idle_fp_per_min",
            "control_recall",
            "switch_latency_s",
            "release_latency_s",
            "inference_ms",
        ]
        ranking_recall_field = "control_recall"
        ranking_recall_threshold = float(DEFAULT_BENCHMARK_RANK_MIN_CONTROL_RECALL)
    elif policy == "async-speed":
        ranking_priority = [
            "idle_fp_per_min",
            "control_recall_at_3s",
            "switch_detect_rate_at_2.8s",
            "switch_latency_s",
            "release_latency_s",
            "acc_4class",
            "macro_f1_4class",
            "itr_bpm_4class",
            "inference_ms",
        ]
        ranking_recall_field = "control_recall_at_3s"
        ranking_recall_threshold = float(DEFAULT_SPEED_MIN_CONTROL_RECALL)
    else:
        ranking_priority = [
            "low_recall_penalty",
            "idle_fp_per_min",
            "control_recall",
            "switch_latency_s",
            "release_latency_s",
            "acc_4class",
            "macro_f1_4class",
            "itr_bpm_4class",
            "inference_ms",
        ]
        ranking_recall_field = "control_recall"
        ranking_recall_threshold = float(DEFAULT_BENCHMARK_RANK_MIN_CONTROL_RECALL)
    return {
        "ranking_policy": {
            "description": (
                "Constrained lexicographic ranking for asynchronous control-state decoding with "
                "classification quality tie-breakers."
            ),
            "policy": policy,
            "min_control_recall_for_ranking": ranking_recall_threshold,
            "control_recall_field": ranking_recall_field,
            "priority": ranking_priority,
        },
        "evaluation_scope": {
            "metric_scope": scope,
            "paper_lens_decision_time_mode": time_mode,
            "async_lens_decision_time_mode": async_time_mode,
        },
        "acceptance_policy": {
            "idle_fp_per_min_max": float(DEFAULT_ACCEPTANCE_IDLE_FP_PER_MIN),
            "control_recall_min": float(DEFAULT_ACCEPTANCE_CONTROL_RECALL),
            "switch_latency_s_max": float(DEFAULT_ACCEPTANCE_SWITCH_LATENCY_S),
            "release_latency_s_max": float(DEFAULT_ACCEPTANCE_RELEASE_LATENCY_S),
            "inference_ms_max": float(DEFAULT_ACCEPTANCE_INFERENCE_MS),
        },
        "gate_search_policy": {
            "min_detection_recall_for_calibration": float(DEFAULT_CALIBRATION_MIN_DETECTION_RECALL),
            "win_candidates": [float(value) for value in DEFAULT_WIN_SEC_CANDIDATES],
            "min_enter_candidates": [int(value) for value in DEFAULT_MIN_ENTER_CANDIDATES],
            "min_exit_candidates": [int(value) for value in DEFAULT_MIN_EXIT_CANDIDATES],
            "default_policy": str(DEFAULT_GATE_POLICY),
            "balanced_objective": {
                "idle_fp_per_min_weight": float(DEFAULT_BALANCED_OBJECTIVE_IDLE_WEIGHT),
                "control_recall_weight": float(DEFAULT_BALANCED_OBJECTIVE_RECALL_WEIGHT),
                "switch_latency_weight": float(DEFAULT_BALANCED_OBJECTIVE_SWITCH_WEIGHT),
                "release_latency_weight": float(DEFAULT_BALANCED_OBJECTIVE_RELEASE_WEIGHT),
                "hard_constraints": {
                    "idle_fp_per_min_max": float(DEFAULT_BALANCED_IDLE_FP_MAX),
                    "control_recall_min": float(DEFAULT_BALANCED_MIN_CONTROL_RECALL),
                },
            },
            "speed_objective": {
                "idle_fp_per_min_weight": float(DEFAULT_SPEED_OBJECTIVE_IDLE_WEIGHT),
                "control_recall_weight": float(DEFAULT_SPEED_OBJECTIVE_RECALL_WEIGHT),
                "switch_latency_weight": float(DEFAULT_SPEED_OBJECTIVE_SWITCH_WEIGHT),
                "release_latency_weight": float(DEFAULT_SPEED_OBJECTIVE_RELEASE_WEIGHT),
                "hard_constraints": {
                    "idle_fp_per_min_max": float(DEFAULT_SPEED_IDLE_FP_MAX),
                    "control_recall_min": float(DEFAULT_SPEED_MIN_CONTROL_RECALL),
                },
                "fixed_search_space": {
                    "win_sec_candidates": [float(value) for value in DEFAULT_SPEED_WIN_SEC_CANDIDATES],
                    "min_enter_candidates": [int(value) for value in DEFAULT_SPEED_MIN_ENTER_CANDIDATES],
                    "min_exit_candidates": [int(value) for value in DEFAULT_SPEED_MIN_EXIT_CANDIDATES],
                },
            },
            "dynamic_stop_defaults": {
                "enabled": bool(DEFAULT_DYNAMIC_STOP_ENABLED),
                "alpha": float(DEFAULT_DYNAMIC_STOP_ALPHA),
            },
            "frontend_training_defaults": {
                "channel_weight_mode": str(DEFAULT_CHANNEL_WEIGHT_MODE),
                "channel_weight_range": [float(DEFAULT_CHANNEL_WEIGHT_RANGE[0]), float(DEFAULT_CHANNEL_WEIGHT_RANGE[1])],
                "spatial_filter_mode": str(DEFAULT_SPATIAL_FILTER_MODE),
                "spatial_rank_candidates": [int(value) for value in DEFAULT_SPATIAL_RANK_CANDIDATES],
                "joint_weight_iters": int(DEFAULT_JOINT_WEIGHT_ITERS),
                "spatial_source_model": str(DEFAULT_SPATIAL_SOURCE_MODEL),
            },
        },
        "robustness_policy": {
            "channel_modes_default": [str(mode) for mode in DEFAULT_BENCHMARK_CHANNEL_MODES],
            "multi_seed_count_default": int(DEFAULT_BENCHMARK_MULTI_SEED_COUNT),
            "seed_step_default": int(DEFAULT_BENCHMARK_SEED_STEP),
            "summary_fields": [
                "mean_rank",
                "std_rank",
                "metrics_mean",
                "metrics_std",
                "accept_rate",
                "recall_pass_rate",
            ],
        },
        "switch_latency_s": {
            "mode": SWITCH_LATENCY_MODE,
            "description": (
                "Median latency across switch trials. If a switch trial is not detected, "
                "a fixed penalty latency is used."
            ),
            "penalty_formula": "trial_duration_sec + win_sec",
        },
        "release_latency_s": {
            "mode": SWITCH_LATENCY_MODE,
            "description": (
                "Median latency across release trials. If release is not detected, "
                "a fixed penalty latency is used."
            ),
            "penalty_formula": "trial_duration_sec + win_sec",
        },
        "deadline_metrics": {
            "control_recall_at_2s": {
                "deadline_s": float(DEFAULT_CONTROL_RECALL_AT_2S_DEADLINE),
                "description": "Control-trial recall within 2.0 seconds.",
            },
            "control_recall_at_3s": {
                "deadline_s": float(DEFAULT_CONTROL_RECALL_AT_3S_DEADLINE),
                "description": "Control-trial recall within 3.0 seconds.",
            },
            "switch_detect_rate_at_2.8s": {
                "deadline_s": float(DEFAULT_SWITCH_DETECT_AT_2P8S_DEADLINE),
                "description": "Switch-trial detection rate within 2.8 seconds.",
            },
        },
        "idle_fp_per_min": {
            "description": (
                "Idle false-positive events per minute (event-based). "
                "One event is counted when selected_freq transitions from None to a target during idle."
            ),
            "count_rule": "count rising edges of selected state on idle windows",
            "time_base": "idle_window_count * step_sec / 60",
        },
        "formula_definitions": {
            "acc_ssvep": "Acc_SSVEP = N_correct / N_total * 100%",
            "macro_f1": "Macro-F1 = (1/C) * sum(F1_i)",
            "itr": "ITR = [log2(N)+P*log2(P)+(1-P)*log2((1-P)/(N-1))] * 60 / T",
        },
        "method_references": [
            {"name": "CCA baseline", "pmid": "17549911"},
            {"name": "FBCCA", "pmid": "26035476"},
            {"name": "TRCA", "pmid": "28436836"},
            {"name": "TRCA-R framework", "pmid": "32091986"},
            {"name": "TDCA", "pmid": "34543200"},
            {"name": "Async control-state detection", "pmid": "26246229"},
            {"name": "Dynamic stopping", "pmid": "26736447"},
            {"name": "Pseudo-online evaluation", "pmid": "38113535"},
        ],
    }


def summarize_benchmark_robustness(
    run_items: Sequence[dict[str, Any]],
    *,
    ranking_policy: str = DEFAULT_RANKING_POLICY,
) -> dict[str, Any]:
    policy = parse_ranking_policy(ranking_policy)
    recall_field = "control_recall_at_3s" if policy == "async-speed" else "control_recall"
    recall_threshold = (
        float(DEFAULT_SPEED_MIN_CONTROL_RECALL)
        if policy == "async-speed"
        else float(DEFAULT_BENCHMARK_RANK_MIN_CONTROL_RECALL)
    )
    runs: list[dict[str, Any]] = [dict(item) for item in run_items]
    grouped_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, item in enumerate(runs):
        mode = str(item.get("channel_mode", "auto"))
        seed = int(item.get("eval_seed", -1))
        grouped_indices[(mode, seed)].append(idx)

    for key, indices in grouped_indices.items():
        successful_indices = [idx for idx in indices if "metrics" in runs[idx]]
        successful_indices.sort(
            key=lambda idx: benchmark_rank_key(dict(runs[idx]["metrics"]), ranking_policy=policy)
        )
        for rank, idx in enumerate(successful_indices, start=1):
            runs[idx]["run_rank"] = int(rank)
        for idx in indices:
            if "metrics" not in runs[idx]:
                runs[idx]["run_rank"] = None

    metrics_keys = (
        "idle_fp_per_min",
        "control_recall",
        "control_recall_at_2s",
        "control_recall_at_3s",
        "control_miss_rate",
        "switch_detect_rate",
        "switch_detect_rate_at_2.8s",
        "switch_latency_s",
        "release_latency_s",
        "detection_latency_s",
        "itr_bpm",
        "acc_4class",
        "macro_f1_4class",
        "itr_bpm_4class",
        "inference_ms",
    )
    mode_model_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for item in runs:
        mode = str(item.get("channel_mode", "auto"))
        model_name = str(item.get("model_name", ""))
        key = (mode, model_name)
        row = mode_model_rows.get(key)
        if row is None:
            row = {
                "channel_mode": mode,
                "model_name": model_name,
                "runs_total": 0,
                "runs_success": 0,
                "runs_failed": 0,
                "rank_values": [],
                "accept_values": [],
                "recall_pass_values": [],
                "metric_values": {name: [] for name in metrics_keys},
                "channels_seen": [],
            }
            mode_model_rows[key] = row
        row["runs_total"] += 1
        selected_channels = item.get("selected_eeg_channels")
        if isinstance(selected_channels, (list, tuple)) and selected_channels:
            row["channels_seen"].append([int(value) for value in selected_channels])
        if "metrics" not in item:
            row["runs_failed"] += 1
            continue
        row["runs_success"] += 1
        run_rank = item.get("run_rank")
        if run_rank is not None:
            row["rank_values"].append(float(run_rank))
        row["accept_values"].append(1.0 if bool(item.get("meets_acceptance", False)) else 0.0)
        row["recall_pass_values"].append(
            1.0
            if float(item.get("metrics", {}).get(recall_field, 0.0)) >= recall_threshold
            else 0.0
        )
        metrics = dict(item.get("metrics", {}))
        for name in metrics_keys:
            value = metrics.get(name)
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                row["metric_values"][name].append(float(value))

    by_mode: dict[str, dict[str, Any]] = {}
    for (mode, _model_name), row in mode_model_rows.items():
        ranked_models = by_mode.setdefault(
            mode,
            {"channel_mode": mode, "ranked_models": [], "runs_total": 0, "runs_success": 0, "runs_failed": 0},
        )["ranked_models"]

        metrics_mean: dict[str, float] = {}
        metrics_std: dict[str, float] = {}
        for metric_name, values in row["metric_values"].items():
            if values:
                arr = np.asarray(values, dtype=float)
                metrics_mean[metric_name] = float(np.mean(arr))
                metrics_std[metric_name] = float(np.std(arr))
            else:
                if metric_name in {
                    "idle_fp_per_min",
                    "control_miss_rate",
                    "switch_latency_s",
                    "release_latency_s",
                    "detection_latency_s",
                    "inference_ms",
                }:
                    metrics_mean[metric_name] = float("inf")
                else:
                    metrics_mean[metric_name] = 0.0
                metrics_std[metric_name] = float("inf")

        rank_values = np.asarray(row["rank_values"], dtype=float) if row["rank_values"] else np.asarray([], dtype=float)
        mean_rank = float(np.mean(rank_values)) if rank_values.size else float("inf")
        std_rank = float(np.std(rank_values)) if rank_values.size else float("inf")
        accept_rate = float(np.mean(np.asarray(row["accept_values"], dtype=float))) if row["accept_values"] else 0.0
        recall_pass_rate = (
            float(np.mean(np.asarray(row["recall_pass_values"], dtype=float))) if row["recall_pass_values"] else 0.0
        )
        channels_mode = None
        if row["channels_seen"]:
            counter: dict[tuple[int, ...], int] = defaultdict(int)
            for channels in row["channels_seen"]:
                counter[tuple(int(value) for value in channels)] += 1
            channels_mode = list(max(counter.items(), key=lambda item: item[1])[0])

        ranked_models.append(
            {
                "channel_mode": mode,
                "model_name": row["model_name"],
                "runs_total": int(row["runs_total"]),
                "runs_success": int(row["runs_success"]),
                "runs_failed": int(row["runs_failed"]),
                "mean_rank": mean_rank,
                "std_rank": std_rank,
                "accept_rate": accept_rate,
                "recall_pass_rate": recall_pass_rate,
                "metrics_mean": metrics_mean,
                "metrics_std": metrics_std,
                "selected_eeg_channels_mode": channels_mode,
            }
        )

    for mode, payload in by_mode.items():
        ranked_models = [dict(item) for item in payload["ranked_models"]]
        ranked_models.sort(
            key=lambda item: (
                benchmark_rank_key(item.get("metrics_mean", {}), ranking_policy=policy),
                float(item.get("mean_rank", float("inf"))),
                float(item.get("std_rank", float("inf"))),
            )
        )
        for rank, item in enumerate(ranked_models, start=1):
            item["rank"] = int(rank)
        payload["ranked_models"] = ranked_models
        payload["runs_total"] = int(sum(int(item.get("runs_total", 0)) for item in ranked_models))
        payload["runs_success"] = int(sum(int(item.get("runs_success", 0)) for item in ranked_models))
        payload["runs_failed"] = int(sum(int(item.get("runs_failed", 0)) for item in ranked_models))

    modes = sorted(by_mode.keys())
    seeds = sorted({int(item.get("eval_seed", -1)) for item in runs})
    return {
        "ranking_policy": policy,
        "channel_modes": modes,
        "seeds": seeds,
        "runs": runs,
        "by_mode": by_mode,
    }


def _build_trial_dataset_entries(
    *,
    stage: str,
    segments: Sequence[tuple[TrialSpec, np.ndarray]],
    target_samples: int,
    npz_arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fallback_index, (trial, segment) in enumerate(segments):
        base_key = f"trial_{stage}_{int(trial.trial_id)}"
        npz_key = base_key
        suffix = 1
        while npz_key in npz_arrays:
            npz_key = f"{base_key}_{suffix}"
            suffix += 1
        segment_matrix = np.ascontiguousarray(np.asarray(segment, dtype=np.float32))
        npz_arrays[npz_key] = segment_matrix
        records.append(
            {
                "stage": str(stage),
                "label": str(trial.label),
                "expected_freq": None if trial.expected_freq is None else float(trial.expected_freq),
                "trial_id": int(trial.trial_id),
                "block_index": int(trial.block_index),
                "order_index": int(fallback_index),
                "used_samples": int(segment_matrix.shape[0]),
                "target_samples": int(target_samples),
                "channels": int(segment_matrix.shape[1]),
                "npz_key": npz_key,
            }
        )
    return records


def save_benchmark_dataset_bundle(
    *,
    dataset_root: Path,
    session_id: str,
    serial_port: str,
    board_id: int,
    sampling_rate: int,
    freqs: Sequence[float],
    board_eeg_channels: Sequence[int],
    protocol_config: dict[str, Any],
    calibration_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    evaluation_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    split_trial_ids: dict[str, list[int]],
    benchmark_summary: dict[str, Any],
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    session_dir = dataset_root / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    target_samples = int(round(float(protocol_config.get("active_sec", 0.0)) * float(sampling_rate)))
    npz_arrays: dict[str, np.ndarray] = {}
    trial_records = []
    trial_records.extend(
        _build_trial_dataset_entries(
            stage="calibration",
            segments=calibration_segments,
            target_samples=target_samples,
            npz_arrays=npz_arrays,
        )
    )
    trial_records.extend(
        _build_trial_dataset_entries(
            stage="evaluation",
            segments=evaluation_segments,
            target_samples=target_samples,
            npz_arrays=npz_arrays,
        )
    )

    npz_path = session_dir / "raw_trials.npz"
    np.savez_compressed(npz_path, **npz_arrays)

    manifest_payload = {
        "data_schema_version": DATA_SCHEMA_VERSION,
        "session_id": str(session_id),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "serial_port": str(serial_port),
        "board_id": int(board_id),
        "sampling_rate": int(sampling_rate),
        "freqs": [float(freq) for freq in freqs],
        "board_eeg_channels": [int(channel) for channel in board_eeg_channels],
        "protocol_config": json_safe(protocol_config),
        "trials": json_safe(trial_records),
        "splits": {
            "train": [int(item) for item in split_trial_ids.get("train", [])],
            "gate": [int(item) for item in split_trial_ids.get("gate", [])],
            "holdout": [int(item) for item in split_trial_ids.get("holdout", [])],
        },
        "benchmark_summary": json_safe(benchmark_summary),
        "files": {
            "raw_trials_npz": str(npz_path),
        },
    }
    manifest_path = session_dir / "session_manifest.json"
    manifest_path.write_text(json_dumps(json_safe(manifest_payload)) + "\n", encoding="utf-8")

    return {
        "dataset_dir": str(session_dir),
        "dataset_manifest": str(manifest_path),
        "dataset_npz": str(npz_path),
        "data_schema_version": DATA_SCHEMA_VERSION,
    }


def save_collection_dataset_bundle(
    *,
    dataset_root: Path,
    session_id: str,
    subject_id: str,
    serial_port: str,
    board_id: int,
    sampling_rate: int,
    freqs: Sequence[float],
    board_eeg_channels: Sequence[int],
    protocol_config: dict[str, Any],
    collection_segments: Sequence[tuple[TrialSpec, np.ndarray]],
) -> dict[str, Any]:
    # Compatibility wrapper: keep the legacy public entry point, but delegate
    # collection dataset persistence to ssvep_core.dataset so protocol-signature
    # and manifest schema stay consistent across collection and training.
    from ssvep_core.dataset import save_collection_dataset_bundle as _core_save_collection_dataset_bundle

    return _core_save_collection_dataset_bundle(
        dataset_root=dataset_root,
        session_id=session_id,
        subject_id=subject_id,
        serial_port=serial_port,
        board_id=board_id,
        sampling_rate=sampling_rate,
        freqs=freqs,
        board_eeg_channels=board_eeg_channels,
        protocol_config=protocol_config,
        trial_segments=collection_segments,
        quality_rows=None,
    )


class CalibrationRunner:
    def __init__(
        self,
        *,
        serial_port: str,
        board_id: int,
        freqs: Sequence[float],
        output_path: Path,
        sampling_rate: int = 250,
        prepare_sec: float = 1.0,
        active_sec: float = 4.0,
        rest_sec: float = 1.0,
        target_repeats: int = 5,
        idle_repeats: int = 10,
        win_sec: float = DEFAULT_WIN_SEC,
        step_sec: float = DEFAULT_STEP_SEC,
        gate_policy: str = DEFAULT_GATE_POLICY,
        channel_weight_mode: Optional[str] = DEFAULT_CHANNEL_WEIGHT_MODE,
        subband_weight_mode: Optional[str] = DEFAULT_SUBBAND_WEIGHT_MODE,
        spatial_filter_mode: Optional[str] = DEFAULT_SPATIAL_FILTER_MODE,
        spatial_rank_candidates: Sequence[int] = DEFAULT_SPATIAL_RANK_CANDIDATES,
        joint_weight_iters: int = DEFAULT_JOINT_WEIGHT_ITERS,
        weight_cv_folds: int = DEFAULT_FBCCA_WEIGHT_CV_FOLDS,
        spatial_source_model: str = DEFAULT_SPATIAL_SOURCE_MODEL,
        idle_fp_hard_th: float = DEFAULT_IDLE_FP_HARD_TH,
        channel_weight_l2: float = DEFAULT_CHANNEL_WEIGHT_L2,
        subband_prior_strength: float = DEFAULT_SUBBAND_PRIOR_STRENGTH,
        control_state_mode: str = DEFAULT_CONTROL_STATE_MODE,
        dynamic_stop_enabled: bool = DEFAULT_DYNAMIC_STOP_ENABLED,
        dynamic_stop_alpha: float = DEFAULT_DYNAMIC_STOP_ALPHA,
        compute_backend: str = DEFAULT_COMPUTE_BACKEND_NAME,
        gpu_device: int = DEFAULT_GPU_DEVICE_ID,
        gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
        gpu_warmup: bool = True,
        gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE,
    ) -> None:
        self.requested_serial_port = normalize_serial_port(serial_port)
        self.serial_port = self.requested_serial_port
        self.board_id = int(board_id)
        self.freqs = tuple(float(freq) for freq in freqs)
        self.output_path = Path(output_path)
        self.prepare_sec = float(prepare_sec)
        self.active_sec = float(active_sec)
        self.rest_sec = float(rest_sec)
        self.target_repeats = int(target_repeats)
        self.idle_repeats = int(idle_repeats)
        self.gate_policy = parse_gate_policy(gate_policy)
        self.channel_weight_mode = parse_channel_weight_mode(channel_weight_mode)
        self.subband_weight_mode = parse_subband_weight_mode(subband_weight_mode) or DEFAULT_SUBBAND_WEIGHT_MODE
        self.spatial_filter_mode = parse_spatial_filter_mode(spatial_filter_mode)
        self.spatial_rank_candidates = tuple(sorted({max(1, int(value)) for value in spatial_rank_candidates}))
        if not self.spatial_rank_candidates:
            self.spatial_rank_candidates = tuple(DEFAULT_SPATIAL_RANK_CANDIDATES)
        self.joint_weight_iters = max(1, int(joint_weight_iters))
        self.weight_cv_folds = max(2, int(weight_cv_folds))
        self.spatial_source_model = parse_spatial_source_model(spatial_source_model)
        self.idle_fp_hard_th = float(idle_fp_hard_th)
        self.channel_weight_l2 = max(0.0, float(channel_weight_l2))
        self.subband_prior_strength = max(0.0, float(subband_prior_strength))
        self.control_state_mode = parse_control_state_mode(control_state_mode)
        self.dynamic_stop_enabled = bool(dynamic_stop_enabled)
        self.dynamic_stop_alpha = float(dynamic_stop_alpha)
        self.compute_backend = parse_compute_backend_name(compute_backend)
        self.gpu_device = int(gpu_device)
        self.gpu_precision = parse_gpu_precision(gpu_precision)
        self.gpu_warmup = bool(gpu_warmup)
        self.gpu_cache_policy = parse_gpu_cache_policy(gpu_cache_policy)
        self.engine = FBCCAEngine(
            sampling_rate=sampling_rate,
            freqs=self.freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            compute_backend=self.compute_backend,
            gpu_device=self.gpu_device,
            gpu_precision=self.gpu_precision,
            gpu_warmup=self.gpu_warmup,
        )

    def _build_trials(self) -> list[TrialSpec]:
        return build_calibration_trials(
            self.freqs,
            target_repeats=self.target_repeats,
            idle_repeats=self.idle_repeats,
            shuffle=True,
            seed=DEFAULT_CALIBRATION_SEED,
        )

    def _countdown(self, message: str, duration_sec: float) -> None:
        remaining = max(0, int(round(duration_sec)))
        if remaining <= 0:
            print(message, flush=True)
            return
        for second in range(remaining, 0, -1):
            print(f"{message} ({second}s)", flush=True)
            time.sleep(1.0)

    def run(self) -> ThresholdProfile:
        validate_calibration_plan(
            target_repeats=self.target_repeats,
            idle_repeats=self.idle_repeats,
            active_sec=self.active_sec,
            preferred_win_sec=self.engine.win_sec,
            step_sec=self.engine.step_sec,
            gate_policy=self.gate_policy,
        )
        require_brainflow()
        board = None
        trial_segments: list[tuple[TrialSpec, np.ndarray]] = []
        failed_trials = 0

        try:
            board, resolved_port, attempted_ports = prepare_board_session(self.board_id, self.requested_serial_port)
            self.serial_port = resolved_port
            if serial_port_is_auto(self.requested_serial_port):
                attempts = ", ".join(attempted_ports)
                print(
                    f"Serial auto-select: requested={self.requested_serial_port} -> using {resolved_port} "
                    f"(attempted: {attempts})",
                    flush=True,
                )
            actual_fs = BoardShim.get_sampling_rate(self.board_id)
            eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            active_samples = int(round(self.active_sec * actual_fs))
            self.engine.configure_runtime(actual_fs)
            if active_samples < self.engine.win_samples:
                raise ValueError("active_sec must be at least as large as win_sec")

            try:
                board.start_stream(450000)
                ready_samples = ensure_stream_ready(board, actual_fs)
            except Exception as exc:
                raise RuntimeError(describe_runtime_error(exc, serial_port=self.serial_port)) from exc
            print("Calibration started. Keep your external four-target stimulus running.", flush=True)
            print(
                f"Sampling rate: {actual_fs}Hz | EEG channels: {eeg_channels} | "
                f"stream_ready={ready_samples}",
                flush=True,
            )
            time.sleep(max(2.0, DEFAULT_STREAM_WARMUP_SEC))
            board.get_board_data()

            for index, trial in enumerate(self._build_trials(), start=1):
                prompt = (
                    f"Trial {index}: focus {trial.label}"
                    if trial.expected_freq is not None
                    else f"Trial {index}: idle (look at center, avoid all flicker targets)"
                )
                self._countdown(prompt, self.prepare_sec)
                board.get_board_data()
                time.sleep(self.active_sec)
                try:
                    segment, used_samples, available_samples = read_recent_eeg_segment(
                        board,
                        eeg_channels,
                        target_samples=active_samples,
                        minimum_samples=self.engine.win_samples,
                    )
                    if used_samples < active_samples:
                        print(
                            f"Calibration warning: sample shortfall for {trial.label}: "
                            f"using {used_samples}/{active_samples} samples (buffer={available_samples})",
                            flush=True,
                        )
                    trial_segments.append((trial, segment))
                except Exception as exc:
                    failed_trials += 1
                    print(
                        f"Calibration warning: skipped trial {index} ({trial.label}) after acquisition failure: "
                        f"{describe_runtime_error(exc, serial_port=self.serial_port)}",
                        flush=True,
                    )
                    if failed_trials >= DEFAULT_MAX_CALIBRATION_TRIAL_ERRORS:
                        raise RuntimeError(
                            f"calibration aborted after {failed_trials} failed trials"
                        ) from exc
                self._countdown("Rest", self.rest_sec)
                board.get_board_data()

            try:
                profile, metadata = optimize_profile_from_segments(
                    trial_segments,
                    available_board_channels=eeg_channels,
                    sampling_rate=actual_fs,
                    freqs=self.freqs,
                    active_sec=self.active_sec,
                    preferred_win_sec=self.engine.win_sec,
                    step_sec=self.engine.step_sec,
                    gate_policy=self.gate_policy,
                    channel_weight_mode=self.channel_weight_mode,
                    subband_weight_mode=self.subband_weight_mode,
                    spatial_filter_mode=self.spatial_filter_mode,
                    spatial_rank_candidates=self.spatial_rank_candidates,
                    joint_weight_iters=self.joint_weight_iters,
                    weight_cv_folds=self.weight_cv_folds,
                    spatial_source_model=self.spatial_source_model,
                    dynamic_stop_enabled=self.dynamic_stop_enabled,
                    dynamic_stop_alpha=self.dynamic_stop_alpha,
                    compute_backend=self.compute_backend,
                    gpu_device=int(self.gpu_device),
                    gpu_precision=self.gpu_precision,
                    gpu_warmup=bool(self.gpu_warmup),
                    gpu_cache_policy=self.gpu_cache_policy,
                )
            except Exception as exc:
                raise RuntimeError(
                    "calibration profile fitting failed: "
                    f"usable_trials={len(trial_segments)}, failed_trials={failed_trials}. "
                    f"{describe_runtime_error(exc, serial_port=self.serial_port)}"
                ) from exc
            save_profile(profile, self.output_path)
            print(f"Calibration profile saved to: {self.output_path}", flush=True)
            print(
                "Calibration selection: "
                f"channels={metadata.get('selected_eeg_channels')} | "
                f"search={metadata.get('validation_search')}",
                flush=True,
            )
            print(format_profile_quality_summary(metadata["validation_summary"]), flush=True)
            print(json_dumps(json_safe(asdict(profile))), flush=True)
            return profile
        finally:
            try:
                board.stop_stream()
            except Exception:
                pass
            try:
                board.release_session()
            except Exception:
                pass


class CollectionRunner:
    def __init__(
        self,
        *,
        serial_port: str,
        board_id: int,
        freqs: Sequence[float],
        dataset_dir: Path,
        subject_id: str,
        session_id: str,
        protocol: str = "enhanced_45m",
        sampling_rate: int = 250,
        prepare_sec: float = 1.0,
        active_sec: float = 4.0,
        rest_sec: float = 1.0,
        target_repeats: int = 24,
        idle_repeats: int = 48,
        switch_trials: int = 32,
        seed: int = DEFAULT_CALIBRATION_SEED,
    ) -> None:
        self.requested_serial_port = normalize_serial_port(serial_port)
        self.serial_port = self.requested_serial_port
        self.board_id = int(board_id)
        self.freqs = tuple(float(freq) for freq in freqs)
        self.dataset_dir = Path(dataset_dir)
        self.subject_id = str(subject_id).strip() or "subject001"
        self.session_id = str(session_id).strip()
        self.protocol = str(protocol).strip().lower()
        self.sampling_rate = int(sampling_rate)
        self.prepare_sec = float(prepare_sec)
        self.active_sec = float(active_sec)
        self.rest_sec = float(rest_sec)
        self.target_repeats = int(target_repeats)
        self.idle_repeats = int(idle_repeats)
        self.switch_trials = int(switch_trials)
        self.seed = int(seed)

    def run(self) -> dict[str, Any]:
        if self.protocol != "enhanced_45m":
            raise ValueError(f"unsupported protocol: {self.protocol}")
        validate_calibration_plan(
            target_repeats=max(self.target_repeats, 2),
            idle_repeats=max(self.idle_repeats, 2),
            active_sec=self.active_sec,
            preferred_win_sec=min(float(DEFAULT_WIN_SEC), float(self.active_sec)),
            step_sec=float(DEFAULT_STEP_SEC),
        )
        require_brainflow()
        board = None
        try:
            board, resolved_port, attempted_ports = prepare_board_session(self.board_id, self.requested_serial_port)
            self.serial_port = resolved_port
            if serial_port_is_auto(self.requested_serial_port):
                attempts = ", ".join(attempted_ports)
                print(
                    f"Serial auto-select: requested={self.requested_serial_port} -> using {resolved_port} "
                    f"(attempted: {attempts})",
                    flush=True,
                )
            actual_fs = int(BoardShim.get_sampling_rate(self.board_id))
            eeg_channels = tuple(int(channel) for channel in BoardShim.get_eeg_channels(self.board_id))
            board.start_stream(450000)
            ready_samples = ensure_stream_ready(board, actual_fs)
            print(
                f"Collection started | fs={actual_fs}Hz | stream_ready={ready_samples} | eeg_channels={list(eeg_channels)}",
                flush=True,
            )
            time.sleep(max(2.0, DEFAULT_STREAM_WARMUP_SEC))
            board.get_board_data()

            helper = BenchmarkRunner(
                serial_port=self.serial_port,
                board_id=self.board_id,
                freqs=self.freqs,
                output_profile_path=self.dataset_dir / "_unused_profile.json",
                report_path=self.dataset_dir / "_unused_report.json",
                dataset_dir=self.dataset_dir,
                sampling_rate=self.sampling_rate,
                prepare_sec=self.prepare_sec,
                active_sec=self.active_sec,
                rest_sec=self.rest_sec,
                calibration_target_repeats=1,
                calibration_idle_repeats=1,
                eval_target_repeats=1,
                eval_idle_repeats=1,
                eval_switch_trials=1,
                step_sec=DEFAULT_STEP_SEC,
                model_names=(DEFAULT_MODEL_NAME,),
                channel_modes=("all8",),
                multi_seed_count=1,
                seed_step=1,
                win_candidates=(min(float(DEFAULT_WIN_SEC), float(self.active_sec)),),
                gate_policy=DEFAULT_GATE_POLICY,
                channel_weight_mode=DEFAULT_CHANNEL_WEIGHT_MODE,
                dynamic_stop_enabled=DEFAULT_DYNAMIC_STOP_ENABLED,
                dynamic_stop_alpha=DEFAULT_DYNAMIC_STOP_ALPHA,
                seed=self.seed,
            )

            trials = build_benchmark_eval_trials(
                self.freqs,
                target_repeats=self.target_repeats,
                idle_repeats=self.idle_repeats,
                switch_trials=self.switch_trials,
                seed=self.seed,
            )
            collection_segments = helper._collect_segments(
                board=board,
                eeg_channels=eeg_channels,
                actual_fs=actual_fs,
                trials=trials,
                title="Collection",
                include_transition_idle=False,
            )
            if not self.session_id:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.session_id = f"{self.subject_id}_session_{stamp}"
            protocol_config = {
                "protocol_name": str(self.protocol),
                "prepare_sec": float(self.prepare_sec),
                "active_sec": float(self.active_sec),
                "rest_sec": float(self.rest_sec),
                "target_repeats": int(self.target_repeats),
                "idle_repeats": int(self.idle_repeats),
                "switch_trials": int(self.switch_trials),
                "seed": int(self.seed),
            }
            metadata = save_collection_dataset_bundle(
                dataset_root=self.dataset_dir,
                session_id=self.session_id,
                subject_id=self.subject_id,
                serial_port=self.serial_port,
                board_id=self.board_id,
                sampling_rate=actual_fs,
                freqs=self.freqs,
                board_eeg_channels=eeg_channels,
                protocol_config=protocol_config,
                collection_segments=collection_segments,
            )
            print(f"Collection dataset manifest saved to: {metadata['dataset_manifest']}", flush=True)
            return {
                "session_id": self.session_id,
                "subject_id": self.subject_id,
                "collected_trials": int(len(collection_segments)),
                **metadata,
            }
        finally:
            try:
                board.stop_stream()
            except Exception:
                pass
            try:
                board.release_session()
            except Exception:
                pass


class BenchmarkRunner:
    def __init__(
        self,
        *,
        serial_port: str,
        board_id: int,
        freqs: Sequence[float],
        output_profile_path: Path,
        report_path: Optional[Path] = None,
        dataset_dir: Path = DEFAULT_BENCHMARK_DATASET_ROOT,
        sampling_rate: int = 250,
        prepare_sec: float = 1.0,
        active_sec: float = 4.0,
        rest_sec: float = 1.0,
        calibration_target_repeats: int = DEFAULT_BENCHMARK_CAL_TARGET_REPEATS,
        calibration_idle_repeats: int = DEFAULT_BENCHMARK_CAL_IDLE_REPEATS,
        eval_target_repeats: int = DEFAULT_BENCHMARK_EVAL_TARGET_REPEATS,
        eval_idle_repeats: int = DEFAULT_BENCHMARK_EVAL_IDLE_REPEATS,
        eval_switch_trials: int = DEFAULT_BENCHMARK_SWITCH_TRIALS,
        step_sec: float = DEFAULT_STEP_SEC,
        model_names: Sequence[str] = DEFAULT_BENCHMARK_MODELS,
        channel_modes: Sequence[str] = DEFAULT_BENCHMARK_CHANNEL_MODES,
        multi_seed_count: int = DEFAULT_BENCHMARK_MULTI_SEED_COUNT,
        seed_step: int = DEFAULT_BENCHMARK_SEED_STEP,
        win_candidates: Sequence[float] = DEFAULT_WIN_SEC_CANDIDATES,
        gate_policy: str = DEFAULT_GATE_POLICY,
        channel_weight_mode: Optional[str] = DEFAULT_CHANNEL_WEIGHT_MODE,
        subband_weight_mode: Optional[str] = DEFAULT_SUBBAND_WEIGHT_MODE,
        spatial_filter_mode: Optional[str] = DEFAULT_SPATIAL_FILTER_MODE,
        spatial_rank_candidates: Sequence[int] = DEFAULT_SPATIAL_RANK_CANDIDATES,
        joint_weight_iters: int = DEFAULT_JOINT_WEIGHT_ITERS,
        weight_cv_folds: int = DEFAULT_FBCCA_WEIGHT_CV_FOLDS,
        idle_fp_hard_th: float = DEFAULT_IDLE_FP_HARD_TH,
        channel_weight_l2: float = DEFAULT_CHANNEL_WEIGHT_L2,
        subband_prior_strength: float = DEFAULT_SUBBAND_PRIOR_STRENGTH,
        control_state_mode: str = DEFAULT_CONTROL_STATE_MODE,
        spatial_source_model: str = DEFAULT_SPATIAL_SOURCE_MODEL,
        dynamic_stop_enabled: bool = DEFAULT_DYNAMIC_STOP_ENABLED,
        dynamic_stop_alpha: float = DEFAULT_DYNAMIC_STOP_ALPHA,
        metric_scope: str = DEFAULT_METRIC_SCOPE,
        decision_time_mode: str = DEFAULT_DECISION_TIME_MODE,
        async_decision_time_mode: str = DEFAULT_ASYNC_DECISION_TIME_MODE,
        ranking_policy: str = DEFAULT_RANKING_POLICY,
        seed: int = DEFAULT_CALIBRATION_SEED,
        compute_backend: str = DEFAULT_COMPUTE_BACKEND_NAME,
        gpu_device: int = DEFAULT_GPU_DEVICE_ID,
        gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
        gpu_warmup: bool = True,
        gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE,
    ) -> None:
        self.requested_serial_port = normalize_serial_port(serial_port)
        self.serial_port = self.requested_serial_port
        self.board_id = int(board_id)
        self.freqs = tuple(float(freq) for freq in freqs)
        self.output_profile_path = Path(output_profile_path)
        self.report_path = Path(report_path) if report_path is not None else None
        self.dataset_root = Path(dataset_dir)
        self.prepare_sec = float(prepare_sec)
        self.active_sec = float(active_sec)
        self.rest_sec = float(rest_sec)
        self.calibration_target_repeats = int(calibration_target_repeats)
        self.calibration_idle_repeats = int(calibration_idle_repeats)
        self.eval_target_repeats = int(eval_target_repeats)
        self.eval_idle_repeats = int(eval_idle_repeats)
        self.eval_switch_trials = int(eval_switch_trials)
        self.step_sec = float(step_sec)
        self.model_names = tuple(normalize_model_name(name) for name in model_names)
        self.seed = int(seed)
        self.channel_modes = tuple(parse_channel_mode_list(",".join(str(item) for item in channel_modes)))
        self.multi_seed_count = max(1, int(multi_seed_count))
        self.seed_step = max(1, int(seed_step))
        self.eval_seeds = tuple(self.seed + idx * self.seed_step for idx in range(self.multi_seed_count))
        self.win_candidates = tuple(sorted({float(item) for item in win_candidates if float(item) <= self.active_sec}))
        if not self.win_candidates:
            self.win_candidates = (min(float(DEFAULT_WIN_SEC), float(self.active_sec)),)
        self.gate_policy = parse_gate_policy(gate_policy)
        self.channel_weight_mode = parse_channel_weight_mode(channel_weight_mode)
        self.subband_weight_mode = parse_subband_weight_mode(subband_weight_mode) or DEFAULT_SUBBAND_WEIGHT_MODE
        self.spatial_filter_mode = parse_spatial_filter_mode(spatial_filter_mode)
        self.spatial_rank_candidates = tuple(sorted({max(1, int(value)) for value in spatial_rank_candidates}))
        if not self.spatial_rank_candidates:
            self.spatial_rank_candidates = tuple(DEFAULT_SPATIAL_RANK_CANDIDATES)
        self.joint_weight_iters = max(1, int(joint_weight_iters))
        self.weight_cv_folds = max(2, int(weight_cv_folds))
        self.idle_fp_hard_th = float(idle_fp_hard_th)
        self.channel_weight_l2 = max(0.0, float(channel_weight_l2))
        self.subband_prior_strength = max(0.0, float(subband_prior_strength))
        self.control_state_mode = parse_control_state_mode(control_state_mode)
        self.spatial_source_model = parse_spatial_source_model(spatial_source_model)
        self.dynamic_stop_enabled = bool(dynamic_stop_enabled)
        self.dynamic_stop_alpha = float(dynamic_stop_alpha)
        self.metric_scope = parse_metric_scope(metric_scope)
        self.decision_time_mode = parse_decision_time_mode(decision_time_mode)
        self.async_decision_time_mode = parse_decision_time_mode(async_decision_time_mode)
        self.ranking_policy = parse_ranking_policy(ranking_policy)
        self._sampling_rate_hint = int(sampling_rate)
        self.compute_backend = parse_compute_backend_name(compute_backend)
        self.gpu_device = int(gpu_device)
        self.gpu_precision = parse_gpu_precision(gpu_precision)
        self.gpu_warmup = bool(gpu_warmup)
        self.gpu_cache_policy = parse_gpu_cache_policy(gpu_cache_policy)

    def _countdown(self, message: str, duration_sec: float) -> None:
        remaining = max(0, int(round(duration_sec)))
        if remaining <= 0:
            print(message, flush=True)
            return
        for second in range(remaining, 0, -1):
            print(f"{message} ({second}s)", flush=True)
            time.sleep(1.0)

    def _collect_segments(
        self,
        *,
        board: Any,
        eeg_channels: Sequence[int],
        actual_fs: int,
        trials: Sequence[TrialSpec],
        title: str,
        include_transition_idle: bool = False,
    ) -> list[tuple[TrialSpec, np.ndarray]]:
        active_samples = int(round(self.active_sec * actual_fs))
        target_min_samples = max(1, int(round(min(self.win_candidates) * actual_fs)))
        segments: list[tuple[TrialSpec, np.ndarray]] = []
        failed_trials = 0
        transition_enabled = bool(include_transition_idle)
        prepare_samples = max(0, int(round(self.prepare_sec * actual_fs)))
        rest_samples = max(0, int(round(self.rest_sec * actual_fs)))
        transition_trial_id = max((int(trial.trial_id) for trial in trials), default=-1) + 1
        pending_rest_segment: Optional[np.ndarray] = None
        print(f"{title}: total_trials={len(trials)}", flush=True)
        for index, trial in enumerate(trials, start=1):
            prompt = (
                f"{title} {index}/{len(trials)} focus {trial.label}"
                if trial.expected_freq is not None
                else f"{title} {index}/{len(trials)} idle (look center, avoid flicker targets)"
            )
            prepare_segment: Optional[np.ndarray] = None
            if transition_enabled:
                board.get_board_data()
            self._countdown(prompt, self.prepare_sec)
            if transition_enabled and prepare_samples > 0:
                try:
                    prepare_segment, _prepare_used, _prepare_available = read_recent_eeg_segment(
                        board,
                        eeg_channels,
                        target_samples=prepare_samples,
                        minimum_samples=max(1, min(prepare_samples, target_min_samples)),
                    )
                except Exception as exc:
                    print(
                        f"{title} warning: failed to capture prepare idle segment before trial {index}: "
                        f"{describe_runtime_error(exc, serial_port=self.serial_port)}",
                        flush=True,
                    )
                    prepare_segment = None
            if transition_enabled:
                transition_segment = merge_idle_transition_segments(
                    pending_rest_segment,
                    prepare_segment,
                    minimum_samples=target_min_samples,
                )
                if transition_segment is not None:
                    transition_trial = TrialSpec(
                        label="transition_idle",
                        expected_freq=None,
                        trial_id=int(transition_trial_id),
                        block_index=int(trial.block_index),
                    )
                    transition_trial_id += 1
                    segments.append((transition_trial, transition_segment))
                pending_rest_segment = None
            board.get_board_data()
            time.sleep(self.active_sec)
            try:
                segment, used_samples, available_samples = read_recent_eeg_segment(
                    board,
                    eeg_channels,
                    target_samples=active_samples,
                    minimum_samples=target_min_samples,
                )
                if used_samples < active_samples:
                    print(
                        f"{title} warning: sample shortfall for {trial.label}: "
                        f"using {used_samples}/{active_samples} (buffer={available_samples})",
                        flush=True,
                    )
                segments.append((trial, segment))
            except Exception as exc:
                failed_trials += 1
                print(
                    f"{title} warning: skipped trial {index} ({trial.label}) after acquisition failure: "
                    f"{describe_runtime_error(exc, serial_port=self.serial_port)}",
                    flush=True,
                )
                if failed_trials >= DEFAULT_MAX_CALIBRATION_TRIAL_ERRORS:
                    raise RuntimeError(f"{title} aborted after {failed_trials} failed trials") from exc
            if transition_enabled:
                board.get_board_data()
                self._countdown("Rest", self.rest_sec)
                if rest_samples > 0:
                    try:
                        rest_segment, _rest_used, _rest_available = read_recent_eeg_segment(
                            board,
                            eeg_channels,
                            target_samples=rest_samples,
                            minimum_samples=max(1, min(rest_samples, target_min_samples)),
                        )
                        pending_rest_segment = rest_segment
                    except Exception as exc:
                        pending_rest_segment = None
                        print(
                            f"{title} warning: failed to capture rest idle segment after trial {index}: "
                            f"{describe_runtime_error(exc, serial_port=self.serial_port)}",
                            flush=True,
                        )
                else:
                    pending_rest_segment = None
            else:
                self._countdown("Rest", self.rest_sec)
                board.get_board_data()
        if transition_enabled and pending_rest_segment is not None:
            tail_segment = merge_idle_transition_segments(
                pending_rest_segment,
                None,
                minimum_samples=target_min_samples,
            )
            if tail_segment is not None:
                tail_trial = TrialSpec(
                    label="transition_idle_tail",
                    expected_freq=None,
                    trial_id=int(transition_trial_id),
                    block_index=int(len(trials)),
                )
                segments.append((tail_trial, tail_segment))
        return segments

    def _benchmark_single_model(
        self,
        *,
        model_name: str,
        fs: int,
        train_segments: Sequence[tuple[TrialSpec, np.ndarray]],
        gate_segments: Sequence[tuple[TrialSpec, np.ndarray]],
        eval_segments: Sequence[tuple[TrialSpec, np.ndarray]],
        eeg_channels: Sequence[int],
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> tuple[ThresholdProfile, dict[str, Any]]:
        best_config: Optional[dict[str, Any]] = None
        best_objective: Optional[tuple[float, ...]] = None
        logger = log_fn if log_fn is not None else (lambda _msg: None)
        normalized_model_name = normalize_model_name(model_name)
        resolved_channel_mode, resolved_subband_mode, resolved_spatial_mode = resolve_fbcca_variant_modes(
            normalized_model_name,
            channel_weight_mode=self.channel_weight_mode,
            subband_weight_mode=self.subband_weight_mode,
            spatial_filter_mode=self.spatial_filter_mode,
        )

        if self.gate_policy == "speed":
            policy_win_candidates = tuple(
                sorted(
                    {
                        float(value)
                        for value in DEFAULT_SPEED_WIN_SEC_CANDIDATES
                        if 0.5 < float(value) <= float(self.active_sec)
                    }
                )
            )
            if not policy_win_candidates:
                policy_win_candidates = tuple(self.win_candidates)
            enter_candidates = tuple(DEFAULT_SPEED_MIN_ENTER_CANDIDATES)
            exit_candidates = tuple(DEFAULT_SPEED_MIN_EXIT_CANDIDATES)
        else:
            policy_win_candidates = tuple(self.win_candidates)
            enter_candidates = tuple(DEFAULT_MIN_ENTER_CANDIDATES)
            exit_candidates = tuple(DEFAULT_MIN_EXIT_CANDIDATES)

        config_candidates: list[tuple[float, int, int]] = []
        for win_sec in policy_win_candidates:
            available_windows = calibration_window_count(self.active_sec, win_sec, self.step_sec)
            valid_enter = [candidate for candidate in enter_candidates if int(candidate) <= available_windows]
            if not valid_enter:
                continue
            for min_enter in valid_enter:
                for min_exit in exit_candidates:
                    config_candidates.append((float(win_sec), int(min_enter), int(min_exit)))

        for config_index, (win_sec, min_enter, min_exit) in enumerate(config_candidates, start=1):
                    try:
                        logger(
                            f"Config start: model={model_name} {config_index}/{len(config_candidates)} "
                            f"win={float(win_sec):g}s enter={int(min_enter)} exit={int(min_exit)}"
                        )
                        model_params: dict[str, Any] = {
                            "Nh": DEFAULT_NH,
                            "compute_backend": self.compute_backend,
                            "gpu_device": int(self.gpu_device),
                            "gpu_precision": self.gpu_precision,
                            "gpu_cache_policy": self.gpu_cache_policy,
                        }
                        channel_weight_training = None
                        if normalized_model_name in {"fbcca", *FBCCA_VARIANT_SPECS.keys()} and (
                            resolved_channel_mode == "fbcca_diag" or resolved_subband_mode != "chen_fixed"
                        ):
                            logger(
                                f"FBCCA frontend start: config={config_index}/{len(config_candidates)} "
                                f"win={float(win_sec):g}s enter={int(min_enter)} exit={int(min_exit)}"
                            )
                            frontend_result = optimize_fbcca_frontend_weights(
                                train_segments=train_segments,
                                gate_segments=gate_segments,
                                sampling_rate=fs,
                                freqs=self.freqs,
                                win_sec=win_sec,
                                step_sec=self.step_sec,
                                min_enter_windows=min_enter,
                                min_exit_windows=min_exit,
                                gate_policy=self.gate_policy,
                                dynamic_stop_enabled=self.dynamic_stop_enabled,
                                dynamic_stop_alpha=self.dynamic_stop_alpha,
                                channel_weight_mode=resolved_channel_mode,
                                subband_weight_mode=resolved_subband_mode,
                                spatial_filter_mode=resolved_spatial_mode,
                                spatial_rank_candidates=self.spatial_rank_candidates,
                                joint_weight_iters=self.joint_weight_iters,
                                weight_cv_folds=self.weight_cv_folds,
                                spatial_source_model=self.spatial_source_model,
                                idle_fp_hard_th=self.idle_fp_hard_th,
                                channel_weight_l2=self.channel_weight_l2,
                                subband_prior_strength=self.subband_prior_strength,
                                control_state_mode=self.control_state_mode,
                                compute_backend=self.compute_backend,
                                gpu_device=int(self.gpu_device),
                                gpu_precision=self.gpu_precision,
                                gpu_warmup=bool(self.gpu_warmup),
                                gpu_cache_policy=self.gpu_cache_policy,
                                log_fn=logger,
                                log_prefix=(
                                    f"FBCCA config {config_index}/{len(config_candidates)} "
                                    f"(win={float(win_sec):g}s enter={int(min_enter)} exit={int(min_exit)})"
                                ),
                            )
                            channel_weight_training = (
                                dict(frontend_result.get("metadata"))
                                if isinstance(frontend_result.get("metadata"), dict)
                                else None
                            )
                            optimized_model_params = (
                                channel_weight_training.get("optimized_model_params")
                                if isinstance(channel_weight_training, dict)
                                else None
                            )
                            if isinstance(optimized_model_params, dict) and optimized_model_params:
                                model_params = dict(optimized_model_params)
                                model_params.setdefault("Nh", DEFAULT_NH)
                            else:
                                optimized_channel_weights = frontend_result.get("channel_weights")
                                optimized_subband_weights = frontend_result.get("subband_weights")
                                optimized_subband_params = frontend_result.get("subband_weight_params")
                                if optimized_channel_weights is not None:
                                    model_params["channel_weight_mode"] = "fbcca_diag"
                                    model_params["channel_weights"] = [float(value) for value in optimized_channel_weights]
                                model_params["subband_weight_mode"] = str(resolved_subband_mode)
                                if optimized_subband_weights is not None:
                                    model_params["subband_weights"] = [float(value) for value in optimized_subband_weights]
                                if optimized_subband_params is not None:
                                    model_params["subband_weight_params"] = json_safe(dict(optimized_subband_params))
                            model_params.setdefault("compute_backend", self.compute_backend)
                            model_params.setdefault("gpu_device", int(self.gpu_device))
                            model_params.setdefault("gpu_precision", self.gpu_precision)
                            model_params.setdefault("gpu_cache_policy", self.gpu_cache_policy)
                            logger(
                                f"FBCCA frontend done: config={config_index}/{len(config_candidates)} "
                                f"win={float(win_sec):g}s enter={int(min_enter)} exit={int(min_exit)}"
                            )
                        decoder = create_decoder(
                            model_name,
                            sampling_rate=fs,
                            freqs=self.freqs,
                            win_sec=win_sec,
                            step_sec=self.step_sec,
                            model_params=model_params,
                            compute_backend=self.compute_backend,
                            gpu_device=int(self.gpu_device),
                            gpu_precision=self.gpu_precision,
                            gpu_warmup=bool(self.gpu_warmup),
                            gpu_cache_policy=self.gpu_cache_policy,
                        )
                        if decoder.requires_fit:
                            decoder.fit(train_segments)
                        gate_rows = build_feature_rows_with_decoder(decoder, gate_segments)
                        if not gate_rows:
                            continue
                        candidate_profile = fit_threshold_profile(
                            gate_rows,
                            freqs=self.freqs,
                            win_sec=win_sec,
                            step_sec=self.step_sec,
                            min_enter_windows=min_enter,
                            min_exit_windows=min_exit,
                            gate_policy=self.gate_policy,
                            evaluation_rows=gate_rows,
                            dynamic_stop_enabled=self.dynamic_stop_enabled,
                            dynamic_stop_alpha=self.dynamic_stop_alpha,
                        )
                        gate_summary = summarize_profile_quality(gate_rows, candidate_profile)
                        gate_metrics = evaluate_profile_on_feature_rows(gate_rows, candidate_profile)
                    except Exception:
                        continue
                    logger(
                        f"Config done: model={model_name} {config_index}/{len(config_candidates)} "
                        f"idle_fp={float(gate_metrics.get('idle_fp_per_min', float('inf'))):.4f} "
                        f"recall={float(gate_metrics.get('control_recall', 0.0)):.4f}"
                    )
                    objective = _calibration_objective(
                        gate_summary,
                        gate_policy=self.gate_policy,
                        gate_metrics=gate_metrics,
                    ) + (float(win_sec), int(min_enter), int(min_exit))
                    if best_objective is None or objective < best_objective:
                        best_objective = objective
                        best_config = {
                            "win_sec": float(win_sec),
                            "min_enter_windows": int(min_enter),
                            "min_exit_windows": int(min_exit),
                            "gate_summary": gate_summary,
                            "gate_metrics": gate_metrics,
                            "model_params": model_params,
                            "channel_weight_training": channel_weight_training,
                        }

        if best_config is None:
            raise RuntimeError(f"{model_name}: no valid profile candidate found")
        logger(
            f"Best config: model={model_name} win={float(best_config['win_sec']):g}s "
            f"enter={int(best_config['min_enter_windows'])} exit={int(best_config['min_exit_windows'])}"
        )

        final_decoder = create_decoder(
            model_name,
            sampling_rate=fs,
            freqs=self.freqs,
            win_sec=float(best_config["win_sec"]),
            step_sec=self.step_sec,
            model_params=dict(best_config.get("model_params", {"Nh": DEFAULT_NH})),
            compute_backend=self.compute_backend,
            gpu_device=int(self.gpu_device),
            gpu_precision=self.gpu_precision,
            gpu_warmup=bool(self.gpu_warmup),
            gpu_cache_policy=self.gpu_cache_policy,
        )
        if final_decoder.requires_fit:
            logger(f"Final fit start: model={model_name}")
            final_decoder.fit([*train_segments, *gate_segments])
            logger(f"Final fit done: model={model_name}")
        backend_probe_window: Optional[np.ndarray] = None
        for _trial, segment in [*eval_segments, *gate_segments, *train_segments]:
            segment_matrix = np.asarray(segment, dtype=np.float64)
            if segment_matrix.ndim == 2 and int(segment_matrix.shape[0]) >= int(final_decoder.win_samples):
                backend_probe_window = np.ascontiguousarray(
                    segment_matrix[: int(final_decoder.win_samples), :],
                    dtype=np.float64,
                )
                break
        try:
            backend_timing_summary = final_decoder.run_backend_microbenchmark(
                sample_window=backend_probe_window,
                repeats=2,
            )
            logger(
                "Final fit backend benchmark: "
                f"model={model_name} requested={self.compute_backend} "
                f"used={backend_timing_summary.get('used_backend', final_decoder.compute_backend_used)} "
                f"total_ms={float(dict(backend_timing_summary.get('kernel_benchmark', {})).get('total_ms', 0.0)):.3f}"
            )
        except Exception as exc:
            backend_timing_summary = {
                "requested_backend": str(self.compute_backend),
                "used_backend": str(final_decoder.compute_backend_used),
                "error": str(exc),
            }
            logger(f"Final fit backend benchmark skipped: model={model_name} reason={exc}")
        final_gate_rows = build_feature_rows_with_decoder(final_decoder, gate_segments)
        final_profile = fit_threshold_profile(
            final_gate_rows,
            freqs=self.freqs,
            win_sec=float(best_config["win_sec"]),
            step_sec=self.step_sec,
            min_enter_windows=int(best_config["min_enter_windows"]),
            min_exit_windows=int(best_config["min_exit_windows"]),
            gate_policy=self.gate_policy,
            evaluation_rows=final_gate_rows,
            dynamic_stop_enabled=self.dynamic_stop_enabled,
            dynamic_stop_alpha=self.dynamic_stop_alpha,
            control_state_mode=self.control_state_mode,
        )
        trained_state = json_safe(final_decoder.get_state())
        base_model_params = dict(final_decoder.model_params)
        base_model_params["state"] = trained_state
        base_channel_weights = base_model_params.get("channel_weights")
        base_spatial_state = (
            dict(base_model_params.get("spatial_filter_state"))
            if isinstance(base_model_params.get("spatial_filter_state"), dict)
            else None
        )
        base_profile = replace(
            final_profile,
            model_name=normalize_model_name(model_name),
            model_params=base_model_params,
            channel_weight_mode=(
                str(base_model_params.get("channel_weight_mode"))
                if base_model_params.get("channel_weight_mode") is not None
                else None
            ),
            channel_weights=(
                tuple(float(value) for value in base_channel_weights)
                if isinstance(base_channel_weights, (list, tuple))
                else None
            ),
            subband_weight_mode=(
                str(base_model_params.get("subband_weight_mode"))
                if base_model_params.get("subband_weight_mode") is not None
                else None
            ),
            subband_weights=(
                tuple(float(value) for value in base_model_params.get("subband_weights"))
                if isinstance(base_model_params.get("subband_weights"), (list, tuple))
                else None
            ),
            subband_weight_params=(
                dict(base_model_params.get("subband_weight_params"))
                if isinstance(base_model_params.get("subband_weight_params"), dict)
                else None
            ),
            spatial_filter_mode=(
                str(base_model_params.get("spatial_filter_mode"))
                if base_model_params.get("spatial_filter_mode") is not None
                else None
            ),
            spatial_filter_rank=(
                None
                if base_model_params.get("spatial_filter_rank") is None
                else int(base_model_params.get("spatial_filter_rank"))
            ),
            spatial_filter_state=base_spatial_state,
            joint_weight_training=(
                dict(best_config.get("channel_weight_training"))
                if isinstance(best_config.get("channel_weight_training"), dict)
                else None
            ),
            control_state_mode=self.control_state_mode,
            frequency_specific_thresholds=final_profile.frequency_specific_thresholds,
            runtime_backend_preference=str(self.compute_backend),
            runtime_precision_preference=str(self.gpu_precision),
        )
        eval_decoder = load_decoder_from_profile(
            base_profile,
            sampling_rate=fs,
            compute_backend=self.compute_backend,
            gpu_device=int(self.gpu_device),
            gpu_precision=self.gpu_precision,
            gpu_warmup=bool(self.gpu_warmup),
            gpu_cache_policy=self.gpu_cache_policy,
        )
        fixed_decoder = load_decoder_from_profile(
            base_profile,
            sampling_rate=fs,
            compute_backend=self.compute_backend,
            gpu_device=int(self.gpu_device),
            gpu_precision=self.gpu_precision,
            gpu_warmup=bool(self.gpu_warmup),
            gpu_cache_policy=self.gpu_cache_policy,
        )
        classifier_only_decoder = load_decoder_from_profile(
            base_profile,
            sampling_rate=fs,
            compute_backend=self.compute_backend,
            gpu_device=int(self.gpu_device),
            gpu_precision=self.gpu_precision,
            gpu_warmup=bool(self.gpu_warmup),
            gpu_cache_policy=self.gpu_cache_policy,
        )
        eval_bundle = evaluate_decoder_on_trials_v2(
            eval_decoder,
            base_profile,
            eval_segments,
            metric_scope=self.metric_scope,
            paper_decision_time_mode=self.decision_time_mode,
            async_decision_time_mode=self.async_decision_time_mode,
        )
        fixed_bundle = evaluate_decoder_on_trials_v2(
            fixed_decoder,
            base_profile,
            eval_segments,
            dynamic_stop_enabled=False,
            metric_scope=self.metric_scope,
            paper_decision_time_mode=self.decision_time_mode,
            async_decision_time_mode=self.async_decision_time_mode,
        )
        classifier_only_bundle = evaluate_decoder_on_trials_v2(
            classifier_only_decoder,
            base_profile,
            eval_segments,
            dynamic_stop_enabled=False,
            metric_scope="4class",
            paper_decision_time_mode="fixed-window",
            async_decision_time_mode=self.async_decision_time_mode,
        )
        eval_async_metrics = dict(eval_bundle.get("async_metrics", {}))
        fixed_async_metrics = dict(fixed_bundle.get("async_metrics", {}))
        eval_metrics = pack_evaluation_metrics_for_ranking(
            eval_bundle,
            metric_scope=self.metric_scope,
        )
        fixed_metrics = pack_evaluation_metrics_for_ranking(
            fixed_bundle,
            metric_scope=self.metric_scope,
        )
        classifier_only_metrics = pack_evaluation_metrics_for_ranking(
            classifier_only_bundle,
            metric_scope="4class",
        )
        dynamic_delta: dict[str, float] = {}
        for metric_name in (
            "idle_fp_per_min",
            "control_recall",
            "control_recall_at_2s",
            "control_recall_at_3s",
            "switch_detect_rate",
            "switch_detect_rate_at_2.8s",
            "switch_latency_s",
            "release_latency_s",
            "detection_latency_s",
            "itr_bpm",
        ):
            dynamic_value = float(eval_async_metrics.get(metric_name, 0.0))
            fixed_value = float(fixed_async_metrics.get(metric_name, 0.0))
            if np.isfinite(dynamic_value) and np.isfinite(fixed_value):
                dynamic_delta[metric_name] = float(dynamic_value - fixed_value)
        model_params = dict(base_model_params)
        profile_channel_weights = model_params.get("channel_weights")
        profile_weight_tuple = (
            tuple(float(value) for value in profile_channel_weights)
            if isinstance(profile_channel_weights, (list, tuple))
            else None
        )
        profile_subband_weights = model_params.get("subband_weights")
        profile_subband_tuple = (
            tuple(float(value) for value in profile_subband_weights)
            if isinstance(profile_subband_weights, (list, tuple))
            else None
        )
        profile_spatial_state = (
            dict(model_params.get("spatial_filter_state"))
            if isinstance(model_params.get("spatial_filter_state"), dict)
            else None
        )
        final_profile = replace(
            final_profile,
            model_name=normalize_model_name(model_name),
            model_params=model_params,
            calibration_split_seed=int(self.seed),
            benchmark_metrics={key: float(value) for key, value in eval_metrics.items() if np.isfinite(value)},
            eeg_channels=tuple(int(channel) for channel in eeg_channels),
            gate_policy=self.gate_policy,
            dynamic_stop=dict(final_profile.dynamic_stop or {}),
            channel_weight_mode=(
                str(model_params.get("channel_weight_mode"))
                if model_params.get("channel_weight_mode") is not None
                else None
            ),
            channel_weights=profile_weight_tuple,
            subband_weight_mode=(
                str(model_params.get("subband_weight_mode"))
                if model_params.get("subband_weight_mode") is not None
                else None
            ),
            subband_weights=profile_subband_tuple,
            subband_weight_params=(
                dict(model_params.get("subband_weight_params"))
                if isinstance(model_params.get("subband_weight_params"), dict)
                else None
            ),
            spatial_filter_mode=(
                str(model_params.get("spatial_filter_mode"))
                if model_params.get("spatial_filter_mode") is not None
                else None
            ),
            spatial_filter_rank=(
                None
                if model_params.get("spatial_filter_rank") is None
                else int(model_params.get("spatial_filter_rank"))
            ),
            spatial_filter_state=profile_spatial_state,
            joint_weight_training=(
                dict(best_config.get("channel_weight_training"))
                if isinstance(best_config.get("channel_weight_training"), dict)
                else None
            ),
            runtime_backend_preference=str(self.compute_backend),
            runtime_precision_preference=str(self.gpu_precision),
            metadata={
                "source": "benchmark",
                "gate_summary": best_config["gate_summary"],
                "gate_metrics": best_config.get("gate_metrics", {}),
                "selection_search": {
                    "win_sec": float(best_config["win_sec"]),
                    "min_enter_windows": int(best_config["min_enter_windows"]),
                    "min_exit_windows": int(best_config["min_exit_windows"]),
                },
                "gate_policy": self.gate_policy,
                "channel_weight_mode": self.channel_weight_mode,
                "subband_weight_mode": self.subband_weight_mode,
                "spatial_filter_mode": self.spatial_filter_mode,
                "spatial_rank_candidates": [int(value) for value in self.spatial_rank_candidates],
                "joint_weight_iters": int(self.joint_weight_iters),
                "weight_cv_folds": int(self.weight_cv_folds),
                "idle_fp_hard_th": float(self.idle_fp_hard_th),
                "channel_weight_l2": float(self.channel_weight_l2),
                "subband_prior_strength": float(self.subband_prior_strength),
                "control_state_mode": str(self.control_state_mode),
                "spatial_source_model": self.spatial_source_model,
                "compute_backend_requested": str(self.compute_backend),
                "compute_backend_used": str(final_decoder.compute_backend_used),
                "gpu_device": int(self.gpu_device),
                "gpu_precision": str(self.gpu_precision),
                "gpu_cache_policy": str(self.gpu_cache_policy),
                "timing_breakdown": dict(final_decoder.get_compute_backend_summary().get("timing_breakdown", {})),
                "backend_timing_summary": dict(backend_timing_summary),
                "channel_weight_training": best_config.get("channel_weight_training"),
                "dynamic_comparison": {
                    "dynamic": eval_bundle,
                    "fixed": fixed_bundle,
                    "delta": dynamic_delta,
                },
                "has_stat_model": profile_has_stat_model(base_profile),
            },
        )
        normalized_model_name = normalize_model_name(model_name)
        runtime_implementation_level = model_implementation_level(normalized_model_name)
        runtime_method_note = model_method_note(normalized_model_name)
        if normalized_model_name == "fbcca":
            frontend_components: list[str] = []
            channel_mode_name = str(model_params.get("channel_weight_mode") or "").strip().lower()
            if channel_mode_name == "fbcca_diag":
                frontend_components.append("diag-channel weighting")
            spatial_mode_name = parse_spatial_filter_mode(model_params.get("spatial_filter_mode"))
            if spatial_mode_name == "trca_shared" and isinstance(model_params.get("spatial_filter_state"), dict):
                frontend_components.append("TRCA-shared spatial frontend")
            if frontend_components:
                runtime_implementation_level = "engineering-approx"
                runtime_method_note = (
                    f"{runtime_method_note} "
                    f"Runtime frontend combines {' + '.join(frontend_components)} before FBCCA scoring."
                )
        return final_profile, {
            "model_name": normalized_model_name,
            "implementation_level": runtime_implementation_level,
            "method_note": runtime_method_note,
            "metrics": eval_metrics,
            "async_metrics": eval_async_metrics,
            "metrics_4class": dict(eval_bundle.get("metrics_4class", {})),
            "metrics_2class": dict(eval_bundle.get("metrics_2class", {})),
            "metrics_5class": (
                None
                if eval_bundle.get("metrics_5class") is None
                else dict(eval_bundle.get("metrics_5class", {}))
            ),
            "paper_lens_metrics_4class": dict(eval_bundle.get("paper_lens_metrics_4class", {})),
            "paper_lens_metrics_2class": dict(eval_bundle.get("paper_lens_metrics_2class", {})),
            "paper_lens_metrics_5class": (
                None
                if eval_bundle.get("paper_lens_metrics_5class") is None
                else dict(eval_bundle.get("paper_lens_metrics_5class", {}))
            ),
            "async_lens_metrics_4class": dict(eval_bundle.get("async_lens_metrics_4class", {})),
            "async_lens_metrics_2class": dict(eval_bundle.get("async_lens_metrics_2class", {})),
            "async_lens_metrics_5class": (
                None
                if eval_bundle.get("async_lens_metrics_5class") is None
                else dict(eval_bundle.get("async_lens_metrics_5class", {}))
            ),
            "fixed_window_metrics": fixed_metrics,
            "fixed_window_async_metrics": fixed_async_metrics,
            "fixed_window_metrics_4class": dict(fixed_bundle.get("metrics_4class", {})),
            "fixed_window_metrics_2class": dict(fixed_bundle.get("metrics_2class", {})),
            "fixed_window_metrics_5class": (
                None
                if fixed_bundle.get("metrics_5class") is None
                else dict(fixed_bundle.get("metrics_5class", {}))
            ),
            "classifier_only_metrics": classifier_only_metrics,
            "classifier_only_metrics_4class": dict(classifier_only_bundle.get("metrics_4class", {})),
            "classifier_only_metrics_2class": dict(classifier_only_bundle.get("metrics_2class", {})),
            "dynamic_delta": dynamic_delta,
            "rank_key": benchmark_rank_key(eval_metrics, ranking_policy=self.ranking_policy),
            "rank_constraints": {
                "min_control_recall_for_ranking": float(
                    DEFAULT_SPEED_MIN_CONTROL_RECALL
                    if self.ranking_policy == "async-speed"
                    else DEFAULT_BENCHMARK_RANK_MIN_CONTROL_RECALL
                ),
                "control_recall_field": (
                    "control_recall_at_3s" if self.ranking_policy == "async-speed" else "control_recall"
                ),
                "control_recall_pass": bool(
                    float(
                        eval_metrics.get(
                            "control_recall_at_3s" if self.ranking_policy == "async-speed" else "control_recall",
                            0.0,
                        )
                    )
                    >= float(
                        DEFAULT_SPEED_MIN_CONTROL_RECALL
                        if self.ranking_policy == "async-speed"
                        else DEFAULT_BENCHMARK_RANK_MIN_CONTROL_RECALL
                    )
                ),
            },
            "gate_summary": best_config["gate_summary"],
            "gate_metrics": best_config.get("gate_metrics", {}),
            "selection_search": {
                "win_sec": float(best_config["win_sec"]),
                "min_enter_windows": int(best_config["min_enter_windows"]),
                "min_exit_windows": int(best_config["min_exit_windows"]),
            },
            "gate_policy": self.gate_policy,
            "channel_weight_mode": self.channel_weight_mode,
            "subband_weight_mode": self.subband_weight_mode,
            "runtime_channel_weight_mode": (
                None
                if model_params.get("channel_weight_mode") is None
                else str(model_params.get("channel_weight_mode"))
            ),
            "runtime_subband_weight_mode": (
                None
                if model_params.get("subband_weight_mode") is None
                else str(model_params.get("subband_weight_mode"))
            ),
            "runtime_subband_weights": (
                [float(value) for value in model_params.get("subband_weights", [])]
                if isinstance(model_params.get("subband_weights"), (list, tuple))
                else None
            ),
            "spatial_filter_mode": self.spatial_filter_mode,
            "runtime_spatial_filter_mode": parse_spatial_filter_mode(model_params.get("spatial_filter_mode")),
            "runtime_spatial_filter_rank": (
                None
                if model_params.get("spatial_filter_rank") is None
                else int(model_params.get("spatial_filter_rank"))
            ),
            "spatial_rank_candidates": [int(value) for value in self.spatial_rank_candidates],
            "joint_weight_iters": int(self.joint_weight_iters),
            "spatial_source_model": self.spatial_source_model,
            "compute_backend_requested": str(self.compute_backend),
            "compute_backend_used": str(final_decoder.compute_backend_used),
            "gpu_device": int(self.gpu_device),
            "precision": str(self.gpu_precision),
            "gpu_cache_policy": str(self.gpu_cache_policy),
            "timing_breakdown": dict(final_decoder.get_compute_backend_summary().get("timing_breakdown", {})),
            "backend_timing_summary": dict(backend_timing_summary),
            "warmup_overhead_ms": float(
                final_decoder.get_compute_backend_summary().get("timing_breakdown", {}).get("warmup_overhead_ms", 0.0)
            ),
            "channel_weight_training": best_config.get("channel_weight_training"),
            "meets_acceptance": profile_meets_acceptance(eval_metrics),
        }

    def _evaluate_model_once(
        self,
        *,
        model_name: str,
        channel_mode: str,
        eval_seed: int,
        fs: int,
        eeg_channels: Sequence[int],
        calibration_segments: Sequence[tuple[TrialSpec, np.ndarray]],
        eval_segments: Sequence[tuple[TrialSpec, np.ndarray]],
        include_channel_scores: bool = False,
    ) -> tuple[ThresholdProfile, dict[str, Any]]:
        mode = str(channel_mode).lower()
        if mode == "auto":
            selected_channels, channel_scores = select_auto_eeg_channels_for_model(
                calibration_segments,
                model_name=model_name,
                available_board_channels=eeg_channels,
                sampling_rate=fs,
                freqs=self.freqs,
                win_sec=max(self.win_candidates),
                step_sec=self.step_sec,
                model_params={
                    "Nh": DEFAULT_NH,
                    "compute_backend": self.compute_backend,
                    "gpu_device": int(self.gpu_device),
                    "gpu_precision": self.gpu_precision,
                    "gpu_cache_policy": self.gpu_cache_policy,
                },
                seed=eval_seed,
                validation_fraction=DEFAULT_VALIDATION_FRACTION,
            )
        elif mode == "all8":
            selected_channels = tuple(int(channel) for channel in eeg_channels)
            channel_scores = []
        else:
            raise ValueError(f"unsupported channel mode: {channel_mode}")

        selected_positions = [list(eeg_channels).index(channel) for channel in selected_channels]
        calibration_subset = _subset_trial_segments_by_positions(calibration_segments, selected_positions)
        eval_subset = _subset_trial_segments_by_positions(eval_segments, selected_positions)
        eval_train, eval_gate, eval_holdout = split_trial_segments_for_benchmark(eval_subset, seed=eval_seed)
        train_segments = [*calibration_subset, *eval_train]
        if not eval_gate or not eval_holdout:
            raise RuntimeError(
                f"benchmark split is invalid: gate={len(eval_gate)}, holdout={len(eval_holdout)}; "
                f"increase evaluation repeats"
            )

        profile, result = self._benchmark_single_model(
            model_name=model_name,
            fs=fs,
            train_segments=train_segments,
            gate_segments=eval_gate,
            eval_segments=eval_holdout,
            eeg_channels=selected_channels,
        )
        result["selected_eeg_channels"] = [int(channel) for channel in selected_channels]
        result["channel_mode"] = mode
        result["eval_seed"] = int(eval_seed)
        result["split_counts"] = {
            "train_segments": int(len(train_segments)),
            "gate_segments": int(len(eval_gate)),
            "holdout_segments": int(len(eval_holdout)),
        }
        if include_channel_scores:
            result["channel_selection"] = channel_scores
        return profile, result

    def run(self) -> dict[str, Any]:
        require_brainflow()
        validate_calibration_plan(
            target_repeats=max(self.calibration_target_repeats, 2),
            idle_repeats=max(self.calibration_idle_repeats, 2),
            active_sec=self.active_sec,
            preferred_win_sec=max(self.win_candidates),
            step_sec=self.step_sec,
            gate_policy=self.gate_policy,
        )
        board = None
        try:
            board, resolved_port, attempted_ports = prepare_board_session(self.board_id, self.requested_serial_port)
            self.serial_port = resolved_port
            if serial_port_is_auto(self.requested_serial_port):
                attempts = ", ".join(attempted_ports)
                print(
                    f"Serial auto-select: requested={self.requested_serial_port} -> using {resolved_port} "
                    f"(attempted: {attempts})",
                    flush=True,
                )
            actual_fs = BoardShim.get_sampling_rate(self.board_id)
            eeg_channels = tuple(int(channel) for channel in BoardShim.get_eeg_channels(self.board_id))
            try:
                board.start_stream(450000)
                ready_samples = ensure_stream_ready(board, actual_fs)
            except Exception as exc:
                raise RuntimeError(describe_runtime_error(exc, serial_port=self.serial_port)) from exc
            time.sleep(max(2.0, DEFAULT_STREAM_WARMUP_SEC))
            board.get_board_data()
            print(
                f"Benchmark started | fs={actual_fs}Hz | stream_ready={ready_samples} | eeg_channels={list(eeg_channels)}",
                flush=True,
            )

            calibration_trials = build_calibration_trials(
                self.freqs,
                target_repeats=self.calibration_target_repeats,
                idle_repeats=self.calibration_idle_repeats,
                shuffle=True,
                seed=self.seed,
            )
            calibration_segments = self._collect_segments(
                board=board,
                eeg_channels=eeg_channels,
                actual_fs=actual_fs,
                trials=calibration_trials,
                title="Calibration",
            )
            eval_trials = build_benchmark_eval_trials(
                self.freqs,
                target_repeats=self.eval_target_repeats,
                idle_repeats=self.eval_idle_repeats,
                switch_trials=self.eval_switch_trials,
                seed=self.seed + 101,
            )
            eval_segments = self._collect_segments(
                board=board,
                eeg_channels=eeg_channels,
                actual_fs=actual_fs,
                trials=eval_trials,
                title="Evaluation",
                include_transition_idle=True,
            )
            eval_train_base, eval_gate_base, eval_holdout_base = split_trial_segments_for_benchmark(
                eval_segments,
                seed=self.seed,
            )
            split_trial_ids = {
                "train": [int(trial.trial_id) for trial, _ in eval_train_base],
                "gate": [int(trial.trial_id) for trial, _ in eval_gate_base],
                "holdout": [int(trial.trial_id) for trial, _ in eval_holdout_base],
            }
            protocol_config = {
                "prepare_sec": float(self.prepare_sec),
                "active_sec": float(self.active_sec),
                "rest_sec": float(self.rest_sec),
                "calibration_target_repeats": int(self.calibration_target_repeats),
                "calibration_idle_repeats": int(self.calibration_idle_repeats),
                "eval_target_repeats": int(self.eval_target_repeats),
                "eval_idle_repeats": int(self.eval_idle_repeats),
                "eval_switch_trials": int(self.eval_switch_trials),
                "step_sec": float(self.step_sec),
                "win_candidates": [float(item) for item in self.win_candidates],
                "transition_idle_enabled": True,
                "seed": int(self.seed),
                "channel_modes": [str(mode) for mode in self.channel_modes],
                "multi_seed_count": int(self.multi_seed_count),
                "seed_step": int(self.seed_step),
                "eval_seeds": [int(seed) for seed in self.eval_seeds],
                "gate_policy": str(self.gate_policy),
                "channel_weight_mode": self.channel_weight_mode,
                "subband_weight_mode": self.subband_weight_mode,
                "spatial_filter_mode": self.spatial_filter_mode,
                "spatial_rank_candidates": [int(value) for value in self.spatial_rank_candidates],
                "joint_weight_iters": int(self.joint_weight_iters),
                "weight_cv_folds": int(self.weight_cv_folds),
                "spatial_source_model": self.spatial_source_model,
                "dynamic_stop_enabled": bool(self.dynamic_stop_enabled),
                "dynamic_stop_alpha": float(self.dynamic_stop_alpha),
                "metric_scope": str(self.metric_scope),
                "decision_time_mode": str(self.decision_time_mode),
                "ranking_policy": str(self.ranking_policy),
            }

            model_results: list[dict[str, Any]] = []
            best_profiles: dict[str, ThresholdProfile] = {}
            robustness_runs: list[dict[str, Any]] = []
            primary_mode = "auto" if "auto" in self.channel_modes else str(self.channel_modes[0])
            primary_seed = int(self.eval_seeds[0]) if self.eval_seeds else int(self.seed)
            for channel_mode in self.channel_modes:
                for eval_seed in self.eval_seeds:
                    print(f"Benchmark mode={channel_mode} seed={eval_seed}", flush=True)
                    seed_mode_success: list[dict[str, Any]] = []
                    seed_mode_all: list[dict[str, Any]] = []
                    for model_name in self.model_names:
                        print(f"Benchmark model: {model_name}", flush=True)
                        include_details = bool(channel_mode == primary_mode and int(eval_seed) == int(primary_seed))
                        try:
                            profile, result = self._evaluate_model_once(
                                model_name=model_name,
                                channel_mode=str(channel_mode),
                                eval_seed=int(eval_seed),
                                fs=actual_fs,
                                eeg_channels=eeg_channels,
                                calibration_segments=calibration_segments,
                                eval_segments=eval_segments,
                                include_channel_scores=include_details,
                            )
                            if include_details:
                                best_profiles[str(model_name)] = profile
                                model_results.append(dict(result))
                            run_item = dict(result)
                            seed_mode_all.append(run_item)
                            seed_mode_success.append(run_item)
                        except Exception as exc:
                            failed_item = {
                                "model_name": str(model_name),
                                "implementation_level": model_implementation_level(model_name),
                                "method_note": model_method_note(model_name),
                                "channel_mode": str(channel_mode),
                                "eval_seed": int(eval_seed),
                                "error": describe_runtime_error(exc, serial_port=self.serial_port),
                                "meets_acceptance": False,
                            }
                            seed_mode_all.append(failed_item)
                            if include_details:
                                model_results.append(dict(failed_item))

                    seed_mode_success.sort(
                        key=lambda item: benchmark_rank_key(item["metrics"], ranking_policy=self.ranking_policy)
                    )
                    rank_by_model = {
                        str(item["model_name"]): int(rank) for rank, item in enumerate(seed_mode_success, start=1)
                    }
                    for run_item in seed_mode_all:
                        model_name = str(run_item.get("model_name", ""))
                        run_item["run_rank"] = rank_by_model.get(model_name)
                        robustness_runs.append(run_item)

            successful = [item for item in model_results if "metrics" in item]
            if not successful:
                raise RuntimeError("all model benchmarks failed")
            successful.sort(key=lambda item: benchmark_rank_key(item["metrics"], ranking_policy=self.ranking_policy))
            accepted = [item for item in successful if bool(item.get("meets_acceptance"))]
            chosen_result = accepted[0] if accepted else successful[0]
            chosen_profile = best_profiles[str(chosen_result["model_name"])]
            chosen_profile = replace(
                chosen_profile,
                benchmark_metrics={
                    key: float(value)
                    for key, value in chosen_result["metrics"].items()
                    if isinstance(value, (int, float)) and np.isfinite(float(value))
                },
            )
            now_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"benchmark_session_{now_stamp}"
            robustness_summary = summarize_benchmark_robustness(
                robustness_runs,
                ranking_policy=self.ranking_policy,
            )
            robust_recommendation = None
            robust_auto = dict(robustness_summary.get("by_mode", {})).get("auto")
            if isinstance(robust_auto, dict):
                ranked_models = list(robust_auto.get("ranked_models", []))
                if ranked_models:
                    robust_recommendation = {
                        "channel_mode": "auto",
                        "model_name": str(ranked_models[0].get("model_name", "")),
                        "rank": int(ranked_models[0].get("rank", 1)),
                        "metrics_mean": dict(ranked_models[0].get("metrics_mean", {})),
                        "metrics_std": dict(ranked_models[0].get("metrics_std", {})),
                    }
            dataset_metadata = save_benchmark_dataset_bundle(
                dataset_root=self.dataset_root,
                session_id=session_id,
                serial_port=self.serial_port,
                board_id=self.board_id,
                sampling_rate=int(actual_fs),
                freqs=self.freqs,
                board_eeg_channels=eeg_channels,
                protocol_config=protocol_config,
                calibration_segments=calibration_segments,
                evaluation_segments=eval_segments,
                split_trial_ids=split_trial_ids,
                benchmark_summary={
                    "chosen_model": chosen_result["model_name"],
                    "chosen_rank_key": chosen_result.get("rank_key"),
                    "chosen_metrics": chosen_result.get("metrics", {}),
                    "chosen_fixed_window_metrics": chosen_result.get("fixed_window_metrics", {}),
                    "chosen_dynamic_delta": chosen_result.get("dynamic_delta", {}),
                    "model_results": model_results,
                    "robustness_summary": robustness_summary,
                },
            )
            save_profile(chosen_profile, self.output_profile_path)
            output_report_path = self.report_path
            if output_report_path is None:
                output_report_path = self.output_profile_path.parent / f"benchmark_report_{now_stamp}.json"
            output_report_path.parent.mkdir(parents=True, exist_ok=True)
            method_mapping = model_method_mapping_payload(self.model_names)
            report_payload = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "serial_port": self.serial_port,
                "board_id": self.board_id,
                "freqs": list(self.freqs),
                "board_eeg_channels": [int(channel) for channel in eeg_channels],
                "protocol_config": protocol_config,
                "selected_eeg_channels": list(chosen_result.get("selected_eeg_channels", [])),
                "channel_selection": chosen_result.get("channel_selection", []),
                "split_counts": dict(chosen_result.get("split_counts", {})),
                "splits": split_trial_ids,
                "model_results": model_results,
                "chosen_model": chosen_result["model_name"],
                "chosen_profile_path": str(self.output_profile_path),
                "chosen_meets_acceptance": bool(chosen_result.get("meets_acceptance", False)),
                "chosen_metrics": dict(chosen_result.get("metrics", {})),
                "chosen_async_metrics": dict(chosen_result.get("async_metrics", {})),
                "chosen_metrics_4class": dict(chosen_result.get("metrics_4class", {})),
                "chosen_metrics_2class": dict(chosen_result.get("metrics_2class", {})),
                "chosen_metrics_5class": (
                    None
                    if chosen_result.get("metrics_5class") is None
                    else dict(chosen_result.get("metrics_5class", {}))
                ),
                "chosen_fixed_window_metrics": dict(chosen_result.get("fixed_window_metrics", {})),
                "chosen_dynamic_delta": dict(chosen_result.get("dynamic_delta", {})),
                "gate_policy": str(self.gate_policy),
                "channel_weight_mode": self.channel_weight_mode,
                "subband_weight_mode": self.subband_weight_mode,
                "weight_cv_folds": int(self.weight_cv_folds),
                "dynamic_stop_enabled": bool(self.dynamic_stop_enabled),
                "dynamic_stop_alpha": float(self.dynamic_stop_alpha),
                "metric_scope": str(self.metric_scope),
                "decision_time_mode": str(self.decision_time_mode),
                "async_decision_time_mode": str(self.async_decision_time_mode),
                "ranking_policy": str(self.ranking_policy),
                "model_method_mapping": method_mapping,
                "metric_definition": benchmark_metric_definition_payload(
                    ranking_policy=self.ranking_policy,
                    metric_scope=self.metric_scope,
                    decision_time_mode=self.decision_time_mode,
                    async_decision_time_mode=self.async_decision_time_mode,
                ),
                "robustness": robustness_summary,
                "robust_recommendation": robust_recommendation,
                **dataset_metadata,
            }
            output_report_path.write_text(json_dumps(json_safe(report_payload)) + "\n", encoding="utf-8")
            print(f"Benchmark report saved to: {output_report_path}", flush=True)
            print(f"Best profile saved to: {self.output_profile_path}", flush=True)
            print(f"Dataset manifest saved to: {dataset_metadata['dataset_manifest']}", flush=True)
            print(f"Chosen model: {chosen_result['model_name']} | metrics={json_dumps(json_safe(chosen_result['metrics']))}", flush=True)
            return report_payload
        finally:
            try:
                board.stop_stream()
            except Exception:
                pass
            try:
                board.release_session()
            except Exception:
                pass


class OnlineRunner:
    def __init__(
        self,
        *,
        serial_port: str,
        board_id: int,
        freqs: Sequence[float],
        profile_path: Path,
        sampling_rate: int = 250,
        emit_all: bool = False,
        result_callback: Optional[callable] = None,
        allow_default_profile: bool = False,
        model_name: Optional[str] = None,
        stop_event: Optional[Any] = None,
        compute_backend: str = DEFAULT_COMPUTE_BACKEND_NAME,
        gpu_device: int = DEFAULT_GPU_DEVICE_ID,
        gpu_precision: str = DEFAULT_GPU_PRECISION_NAME,
        gpu_warmup: bool = True,
        gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE,
    ) -> None:
        self.requested_serial_port = normalize_serial_port(serial_port)
        self.serial_port = self.requested_serial_port
        self.board_id = int(board_id)
        self.freqs = tuple(float(freq) for freq in freqs)
        self.profile_path = Path(profile_path)
        self.emit_all = bool(emit_all)
        self.result_callback = result_callback
        self.stop_event = stop_event
        self.compute_backend = parse_compute_backend_name(compute_backend)
        self.gpu_device = int(gpu_device)
        self.gpu_precision = parse_gpu_precision(gpu_precision)
        self.gpu_warmup = bool(gpu_warmup)
        self.gpu_cache_policy = parse_gpu_cache_policy(gpu_cache_policy)
        loaded_profile = load_profile(
            self.profile_path,
            fallback_freqs=self.freqs,
            require_exists=not bool(allow_default_profile),
        )
        if profile_is_default_fallback(loaded_profile) and not bool(allow_default_profile):
            raise RuntimeError(
                f"profile '{self.profile_path}' is only the default fallback; run calibration first "
                f"or pass --allow-default-profile explicitly"
            )
        resolved_model = normalize_model_name(model_name or loaded_profile.model_name or DEFAULT_MODEL_NAME)
        if resolved_model != normalize_model_name(loaded_profile.model_name or DEFAULT_MODEL_NAME):
            loaded_profile = replace(loaded_profile, model_name=resolved_model)
        self.profile = loaded_profile
        decoder_params = dict(self.profile.model_params or {})
        if self.profile.channel_weight_mode is not None and "channel_weight_mode" not in decoder_params:
            decoder_params["channel_weight_mode"] = str(self.profile.channel_weight_mode)
        if self.profile.channel_weights is not None and "channel_weights" not in decoder_params:
            decoder_params["channel_weights"] = [float(value) for value in self.profile.channel_weights]
        self.decoder = create_decoder(
            self.profile.model_name,
            sampling_rate=sampling_rate,
            freqs=self.profile.freqs,
            win_sec=self.profile.win_sec,
            step_sec=self.profile.step_sec,
            model_params=decoder_params,
            compute_backend=self.compute_backend,
            gpu_device=self.gpu_device,
            gpu_precision=self.gpu_precision,
            gpu_warmup=bool(self.gpu_warmup),
            gpu_cache_policy=self.gpu_cache_policy,
        )
        state = None
        if isinstance(decoder_params, dict):
            state = decoder_params.get("state")
        if state is not None:
            self.decoder.set_state(state)
        if self.decoder.requires_fit and state is None:
            raise RuntimeError(
                f"profile model '{self.profile.model_name}' requires fitted state; run benchmark "
                f"and save profile with model_params.state"
            )
        self.gate = AsyncDecisionGate.from_profile(self.profile)

    def run(self, *, max_updates: Optional[int] = None) -> None:
        require_brainflow()
        board = None
        last_signature: tuple[str, Optional[float]] | None = None

        try:
            board, resolved_port, attempted_ports = prepare_board_session(self.board_id, self.requested_serial_port)
            self.serial_port = resolved_port
            if serial_port_is_auto(self.requested_serial_port):
                attempts = ", ".join(attempted_ports)
                print(
                    f"Serial auto-select: requested={self.requested_serial_port} -> using {resolved_port} "
                    f"(attempted: {attempts})",
                    flush=True,
                )
            actual_fs = BoardShim.get_sampling_rate(self.board_id)
            eeg_channels = resolve_selected_eeg_channels(
                BoardShim.get_eeg_channels(self.board_id),
                self.profile.eeg_channels,
            )
            self.decoder.configure_runtime(actual_fs)
            try:
                board.start_stream(450000)
                ready_samples = ensure_stream_ready(board, actual_fs)
            except Exception as exc:
                raise RuntimeError(describe_runtime_error(exc, serial_port=self.serial_port)) from exc
            time.sleep(DEFAULT_STREAM_WARMUP_SEC)
            print(
                f"Online decoder started | fs={actual_fs}Hz | stream_ready={ready_samples} | "
                f"profile='{self.profile_path}' | model={self.profile.model_name} | "
                f"eeg_channels={list(eeg_channels)}",
                flush=True,
            )

            emitted = 0
            consecutive_errors = 0
            while True:
                if self.stop_event is not None and bool(getattr(self.stop_event, "is_set", lambda: False)()):
                    break
                try:
                    if board.get_board_data_count() < self.decoder.win_samples:
                        time.sleep(0.05)
                        continue
                    data = board.get_current_board_data(self.decoder.win_samples)
                    if data.shape[1] < self.decoder.win_samples:
                        time.sleep(0.05)
                        continue

                    eeg = np.ascontiguousarray(data[eeg_channels, -self.decoder.win_samples :].T, dtype=np.float64)
                    infer_t0 = time.perf_counter()
                    decision = self.gate.update(self.decoder.analyze_window(eeg))
                    infer_t1 = time.perf_counter()
                    self.decoder.update_online(decision, eeg)
                    consecutive_errors = 0
                except Exception as exc:
                    consecutive_errors += 1
                    print(
                        f"Online decoder warning: transient read failure "
                        f"{consecutive_errors}/{DEFAULT_MAX_TRANSIENT_READ_ERRORS}: {exc}",
                        flush=True,
                    )
                    if consecutive_errors >= DEFAULT_MAX_TRANSIENT_READ_ERRORS:
                        raise RuntimeError(
                            "online decoder aborted after repeated read failures: "
                            f"{describe_runtime_error(exc, serial_port=self.serial_port)}"
                        ) from exc
                    time.sleep(0.2)
                    continue
                payload = {
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "model_name": self.profile.model_name,
                    "state": decision["state"],
                    "pred_freq": decision["pred_freq"],
                    "selected_freq": decision["selected_freq"],
                    "top1_score": round(float(decision["top1_score"]), 6),
                    "top2_score": round(float(decision["top2_score"]), 6),
                    "margin": round(float(decision["margin"]), 6),
                    "ratio": round(float(decision["ratio"]), 6),
                    "normalized_top1": round(float(decision["normalized_top1"]), 6),
                    "score_entropy": round(float(decision["score_entropy"]), 6),
                    "control_log_lr": None
                    if decision.get("control_log_lr") is None
                    else round(float(decision["control_log_lr"]), 6),
                    "control_confidence": None
                    if decision.get("control_confidence") is None
                    else round(float(decision["control_confidence"]), 6),
                    "acc_log_lr": None
                    if decision.get("acc_log_lr") is None
                    else round(float(decision["acc_log_lr"]), 6),
                    "stable_windows": int(decision["stable_windows"]),
                    "decision_latency_ms": round(float((infer_t1 - infer_t0) * 1000.0), 4),
                }

                signature = (payload["state"], payload["selected_freq"])
                if self.emit_all or signature != last_signature:
                    print(json_dumps(payload), flush=True)
                    last_signature = signature
                    if self.result_callback is not None:
                        self.result_callback(payload)
                    emitted += 1
                    if max_updates is not None and emitted >= max_updates:
                        break

                time.sleep(self.decoder.step_sec)
        finally:
            try:
                board.stop_stream()
            except Exception:
                pass
            try:
                board.release_session()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone asynchronous FBCCA decoder with idle suppression")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate", help="collect short calibration data and fit an idle-aware profile")
    calibrate.add_argument(
        "--serial-port",
        type=str,
        default=DEFAULT_SERIAL_PORT,
        help="serial port name (e.g., COM4). default=auto",
    )
    calibrate.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    calibrate.add_argument("--sampling-rate", type=int, default=250)
    calibrate.add_argument("--freqs", type=str, default="8,10,12,15")
    calibrate.add_argument("--output", type=Path, default=DEFAULT_PROFILE_PATH)
    calibrate.add_argument("--prepare-sec", type=float, default=1.0)
    calibrate.add_argument("--active-sec", type=float, default=4.0)
    calibrate.add_argument("--rest-sec", type=float, default=1.0)
    calibrate.add_argument("--target-repeats", type=int, default=5)
    calibrate.add_argument("--idle-repeats", type=int, default=10)
    calibrate.add_argument("--win-sec", type=float, default=DEFAULT_WIN_SEC)
    calibrate.add_argument("--step-sec", type=float, default=DEFAULT_STEP_SEC)
    calibrate.add_argument("--gate-policy", type=str, default=DEFAULT_GATE_POLICY)
    calibrate.add_argument("--channel-weight-mode", type=str, default=DEFAULT_CHANNEL_WEIGHT_MODE)
    calibrate.add_argument("--subband-weight-mode", type=str, default=DEFAULT_SUBBAND_WEIGHT_MODE)
    calibrate.add_argument("--spatial-filter-mode", type=str, default=DEFAULT_SPATIAL_FILTER_MODE)
    calibrate.add_argument(
        "--spatial-rank-candidates",
        type=str,
        default=",".join(str(value) for value in DEFAULT_SPATIAL_RANK_CANDIDATES),
    )
    calibrate.add_argument("--joint-weight-iters", type=int, default=DEFAULT_JOINT_WEIGHT_ITERS)
    calibrate.add_argument("--weight-cv-folds", type=int, default=DEFAULT_FBCCA_WEIGHT_CV_FOLDS)
    calibrate.add_argument("--spatial-source-model", type=str, default=DEFAULT_SPATIAL_SOURCE_MODEL)
    calibrate.add_argument(
        "--disable-dynamic-stop",
        action="store_true",
        help="disable accumulated-evidence dynamic stopping in gate fitting",
    )
    calibrate.add_argument("--dynamic-stop-alpha", type=float, default=DEFAULT_DYNAMIC_STOP_ALPHA)
    calibrate.add_argument("--compute-backend", type=str, default=DEFAULT_COMPUTE_BACKEND_NAME)
    calibrate.add_argument("--gpu-device", type=int, default=DEFAULT_GPU_DEVICE_ID)
    calibrate.add_argument("--gpu-precision", type=str, default=DEFAULT_GPU_PRECISION_NAME)
    calibrate.add_argument("--gpu-warmup", type=int, default=1)
    calibrate.add_argument("--gpu-cache-policy", type=str, default=DEFAULT_GPU_CACHE_MODE)

    online = subparsers.add_parser("online", help="run realtime asynchronous decoding")
    online.add_argument(
        "--serial-port",
        type=str,
        default=DEFAULT_SERIAL_PORT,
        help="serial port name (e.g., COM4). default=auto",
    )
    online.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    online.add_argument("--sampling-rate", type=int, default=250)
    online.add_argument("--freqs", type=str, default="8,10,12,15")
    online.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    online.add_argument(
        "--model",
        type=str,
        default=None,
        help="optional model override; defaults to model_name inside profile",
    )
    online.add_argument(
        "--allow-default-profile",
        action="store_true",
        help="allow running online with built-in fallback thresholds when the profile file is missing",
    )
    online.add_argument("--compute-backend", type=str, default=DEFAULT_COMPUTE_BACKEND_NAME)
    online.add_argument("--gpu-device", type=int, default=DEFAULT_GPU_DEVICE_ID)
    online.add_argument("--gpu-precision", type=str, default=DEFAULT_GPU_PRECISION_NAME)
    online.add_argument("--gpu-warmup", type=int, default=1)
    online.add_argument("--gpu-cache-policy", type=str, default=DEFAULT_GPU_CACHE_MODE)
    online.add_argument("--emit-all", action="store_true", help="emit every update instead of only state changes")
    online.add_argument("--max-updates", type=int, default=None)

    realtime = subparsers.add_parser("realtime", help="alias of online mode (model-select realtime decode)")
    realtime.add_argument(
        "--serial-port",
        type=str,
        default=DEFAULT_SERIAL_PORT,
        help="serial port name (e.g., COM4). default=auto",
    )
    realtime.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    realtime.add_argument("--sampling-rate", type=int, default=250)
    realtime.add_argument("--freqs", type=str, default="8,10,12,15")
    realtime.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    realtime.add_argument(
        "--model",
        type=str,
        default=None,
        help="optional model override; defaults to model_name inside profile",
    )
    realtime.add_argument(
        "--allow-default-profile",
        action="store_true",
        help="allow running realtime with built-in fallback thresholds when the profile file is missing",
    )
    realtime.add_argument("--compute-backend", type=str, default=DEFAULT_COMPUTE_BACKEND_NAME)
    realtime.add_argument("--gpu-device", type=int, default=DEFAULT_GPU_DEVICE_ID)
    realtime.add_argument("--gpu-precision", type=str, default=DEFAULT_GPU_PRECISION_NAME)
    realtime.add_argument("--gpu-warmup", type=int, default=1)
    realtime.add_argument("--gpu-cache-policy", type=str, default=DEFAULT_GPU_CACHE_MODE)
    realtime.add_argument("--emit-all", action="store_true", help="emit every update instead of only state changes")
    realtime.add_argument("--max-updates", type=int, default=None)

    collect = subparsers.add_parser(
        "collect",
        help="collect SSVEP dataset only (no model training), save NPZ + manifest",
    )
    collect.add_argument(
        "--serial-port",
        type=str,
        default=DEFAULT_SERIAL_PORT,
        help="serial port name (e.g., COM4). default=auto",
    )
    collect.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    collect.add_argument("--sampling-rate", type=int, default=250)
    collect.add_argument("--freqs", type=str, default="8,10,12,15")
    collect.add_argument("--dataset-dir", type=Path, default=DEFAULT_BENCHMARK_DATASET_ROOT)
    collect.add_argument("--subject-id", type=str, default="subject001")
    collect.add_argument("--session-id", type=str, default="")
    collect.add_argument("--protocol", type=str, default="enhanced_45m")
    collect.add_argument("--prepare-sec", type=float, default=1.0)
    collect.add_argument("--active-sec", type=float, default=4.0)
    collect.add_argument("--rest-sec", type=float, default=1.0)
    collect.add_argument("--target-repeats", type=int, default=24)
    collect.add_argument("--idle-repeats", type=int, default=48)
    collect.add_argument("--switch-trials", type=int, default=32)
    collect.add_argument("--seed", type=int, default=DEFAULT_CALIBRATION_SEED)

    train_eval = subparsers.add_parser(
        "train-eval",
        help="offline training/evaluation from collected dataset manifests",
    )
    train_eval.add_argument("--dataset-manifest", type=Path, required=True)
    train_eval.add_argument("--dataset-manifest-session2", type=Path, default=None)
    train_eval.add_argument("--output-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    train_eval.add_argument("--report-path", type=Path, default=None)
    train_eval.add_argument("--models", type=str, default=",".join(DEFAULT_BENCHMARK_MODELS))
    train_eval.add_argument("--channel-modes", type=str, default=",".join(DEFAULT_BENCHMARK_CHANNEL_MODES))
    train_eval.add_argument("--multi-seed-count", type=int, default=DEFAULT_BENCHMARK_MULTI_SEED_COUNT)
    train_eval.add_argument("--seed-step", type=int, default=DEFAULT_BENCHMARK_SEED_STEP)
    train_eval.add_argument("--win-candidates", type=str, default=",".join(f"{value:g}" for value in DEFAULT_WIN_SEC_CANDIDATES))
    train_eval.add_argument("--gate-policy", type=str, default=DEFAULT_GATE_POLICY)
    train_eval.add_argument("--channel-weight-mode", type=str, default=DEFAULT_CHANNEL_WEIGHT_MODE)
    train_eval.add_argument("--subband-weight-mode", type=str, default=DEFAULT_SUBBAND_WEIGHT_MODE)
    train_eval.add_argument("--spatial-filter-mode", type=str, default=DEFAULT_SPATIAL_FILTER_MODE)
    train_eval.add_argument(
        "--spatial-rank-candidates",
        type=str,
        default=",".join(str(value) for value in DEFAULT_SPATIAL_RANK_CANDIDATES),
    )
    train_eval.add_argument("--joint-weight-iters", type=int, default=DEFAULT_JOINT_WEIGHT_ITERS)
    train_eval.add_argument("--weight-cv-folds", type=int, default=DEFAULT_FBCCA_WEIGHT_CV_FOLDS)
    train_eval.add_argument("--spatial-source-model", type=str, default=DEFAULT_SPATIAL_SOURCE_MODEL)
    train_eval.add_argument("--metric-scope", type=str, default=DEFAULT_METRIC_SCOPE)
    train_eval.add_argument("--decision-time-mode", type=str, default=DEFAULT_PAPER_DECISION_TIME_MODE)
    train_eval.add_argument("--async-decision-time-mode", type=str, default=DEFAULT_ASYNC_DECISION_TIME_MODE)
    train_eval.add_argument("--data-policy", type=str, default=DEFAULT_DATA_POLICY)
    train_eval.add_argument("--export-figures", type=int, default=1)
    train_eval.add_argument("--ranking-policy", type=str, default=DEFAULT_RANKING_POLICY)
    train_eval.add_argument(
        "--disable-dynamic-stop",
        action="store_true",
        help="disable accumulated-evidence dynamic stopping in gate fitting",
    )
    train_eval.add_argument("--dynamic-stop-alpha", type=float, default=DEFAULT_DYNAMIC_STOP_ALPHA)
    train_eval.add_argument("--seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    train_eval.add_argument("--compute-backend", type=str, default=DEFAULT_COMPUTE_BACKEND_NAME)
    train_eval.add_argument("--gpu-device", type=int, default=DEFAULT_GPU_DEVICE_ID)
    train_eval.add_argument("--gpu-precision", type=str, default=DEFAULT_GPU_PRECISION_NAME)
    train_eval.add_argument("--gpu-warmup", type=int, default=1)
    train_eval.add_argument("--gpu-cache-policy", type=str, default=DEFAULT_GPU_CACHE_MODE)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="collect benchmark data, compare multiple decoders, and save best profile",
    )
    benchmark.add_argument(
        "--serial-port",
        type=str,
        default=DEFAULT_SERIAL_PORT,
        help="serial port name (e.g., COM4). default=auto",
    )
    benchmark.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    benchmark.add_argument("--sampling-rate", type=int, default=250)
    benchmark.add_argument("--freqs", type=str, default="8,10,12,15")
    benchmark.add_argument("--output-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    benchmark.add_argument("--report-path", type=Path, default=None)
    benchmark.add_argument("--dataset-dir", type=Path, default=DEFAULT_BENCHMARK_DATASET_ROOT)
    benchmark.add_argument("--prepare-sec", type=float, default=1.0)
    benchmark.add_argument("--active-sec", type=float, default=4.0)
    benchmark.add_argument("--rest-sec", type=float, default=1.0)
    benchmark.add_argument("--calibration-target-repeats", type=int, default=DEFAULT_BENCHMARK_CAL_TARGET_REPEATS)
    benchmark.add_argument("--calibration-idle-repeats", type=int, default=DEFAULT_BENCHMARK_CAL_IDLE_REPEATS)
    benchmark.add_argument("--eval-target-repeats", type=int, default=DEFAULT_BENCHMARK_EVAL_TARGET_REPEATS)
    benchmark.add_argument("--eval-idle-repeats", type=int, default=DEFAULT_BENCHMARK_EVAL_IDLE_REPEATS)
    benchmark.add_argument("--eval-switch-trials", type=int, default=DEFAULT_BENCHMARK_SWITCH_TRIALS)
    benchmark.add_argument("--step-sec", type=float, default=DEFAULT_STEP_SEC)
    benchmark.add_argument("--models", type=str, default=",".join(DEFAULT_BENCHMARK_MODELS))
    benchmark.add_argument(
        "--channel-modes",
        type=str,
        default=",".join(DEFAULT_BENCHMARK_CHANNEL_MODES),
        help="comma-separated channel modes: auto,all8",
    )
    benchmark.add_argument("--multi-seed-count", type=int, default=DEFAULT_BENCHMARK_MULTI_SEED_COUNT)
    benchmark.add_argument("--seed-step", type=int, default=DEFAULT_BENCHMARK_SEED_STEP)
    benchmark.add_argument(
        "--win-candidates",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_WIN_SEC_CANDIDATES),
        help="comma-separated candidate window lengths in seconds",
    )
    benchmark.add_argument("--gate-policy", type=str, default=DEFAULT_GATE_POLICY)
    benchmark.add_argument("--channel-weight-mode", type=str, default=DEFAULT_CHANNEL_WEIGHT_MODE)
    benchmark.add_argument("--subband-weight-mode", type=str, default=DEFAULT_SUBBAND_WEIGHT_MODE)
    benchmark.add_argument("--spatial-filter-mode", type=str, default=DEFAULT_SPATIAL_FILTER_MODE)
    benchmark.add_argument(
        "--spatial-rank-candidates",
        type=str,
        default=",".join(str(value) for value in DEFAULT_SPATIAL_RANK_CANDIDATES),
    )
    benchmark.add_argument("--joint-weight-iters", type=int, default=DEFAULT_JOINT_WEIGHT_ITERS)
    benchmark.add_argument("--weight-cv-folds", type=int, default=DEFAULT_FBCCA_WEIGHT_CV_FOLDS)
    benchmark.add_argument("--spatial-source-model", type=str, default=DEFAULT_SPATIAL_SOURCE_MODEL)
    benchmark.add_argument(
        "--disable-dynamic-stop",
        action="store_true",
        help="disable accumulated-evidence dynamic stopping in gate fitting",
    )
    benchmark.add_argument("--dynamic-stop-alpha", type=float, default=DEFAULT_DYNAMIC_STOP_ALPHA)
    benchmark.add_argument("--seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    benchmark.add_argument("--compute-backend", type=str, default=DEFAULT_COMPUTE_BACKEND_NAME)
    benchmark.add_argument("--gpu-device", type=int, default=DEFAULT_GPU_DEVICE_ID)
    benchmark.add_argument("--gpu-precision", type=str, default=DEFAULT_GPU_PRECISION_NAME)
    benchmark.add_argument("--gpu-warmup", type=int, default=1)
    benchmark.add_argument("--gpu-cache-policy", type=str, default=DEFAULT_GPU_CACHE_MODE)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    freqs = parse_freqs(args.freqs) if hasattr(args, "freqs") else DEFAULT_FREQS

    if args.command == "calibrate":
        try:
            runner = CalibrationRunner(
                serial_port=args.serial_port,
                board_id=args.board_id,
                freqs=freqs,
                output_path=args.output,
                sampling_rate=args.sampling_rate,
                prepare_sec=args.prepare_sec,
                active_sec=args.active_sec,
                rest_sec=args.rest_sec,
                target_repeats=args.target_repeats,
                idle_repeats=args.idle_repeats,
                win_sec=args.win_sec,
                step_sec=args.step_sec,
                gate_policy=args.gate_policy,
                channel_weight_mode=args.channel_weight_mode,
                subband_weight_mode=args.subband_weight_mode,
                spatial_filter_mode=args.spatial_filter_mode,
                spatial_rank_candidates=parse_spatial_rank_candidates(args.spatial_rank_candidates),
                joint_weight_iters=args.joint_weight_iters,
                weight_cv_folds=max(2, int(args.weight_cv_folds)),
                spatial_source_model=args.spatial_source_model,
                dynamic_stop_enabled=not bool(args.disable_dynamic_stop),
                dynamic_stop_alpha=args.dynamic_stop_alpha,
                compute_backend=parse_compute_backend_name(args.compute_backend),
                gpu_device=int(args.gpu_device),
                gpu_precision=parse_gpu_precision(args.gpu_precision),
                gpu_warmup=bool(int(args.gpu_warmup)),
                gpu_cache_policy=parse_gpu_cache_policy(args.gpu_cache_policy),
            )
            runner.run()
        except Exception as exc:
            print(
                f"Calibration failed: {describe_runtime_error(exc, serial_port=args.serial_port)}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        return 0

    if args.command in ("online", "realtime"):
        try:
            runner = OnlineRunner(
                serial_port=args.serial_port,
                board_id=args.board_id,
                freqs=freqs,
                profile_path=args.profile,
                sampling_rate=args.sampling_rate,
                emit_all=args.emit_all,
                allow_default_profile=args.allow_default_profile,
                model_name=args.model,
                compute_backend=parse_compute_backend_name(args.compute_backend),
                gpu_device=int(args.gpu_device),
                gpu_precision=parse_gpu_precision(args.gpu_precision),
                gpu_warmup=bool(int(args.gpu_warmup)),
                gpu_cache_policy=parse_gpu_cache_policy(args.gpu_cache_policy),
            )
            runner.run(max_updates=args.max_updates)
        except Exception as exc:
            print(
                f"Online decode failed: {describe_runtime_error(exc, serial_port=args.serial_port)}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        return 0

    if args.command == "collect":
        try:
            runner = CollectionRunner(
                serial_port=args.serial_port,
                board_id=args.board_id,
                freqs=freqs,
                dataset_dir=args.dataset_dir,
                subject_id=args.subject_id,
                session_id=args.session_id,
                protocol=args.protocol,
                sampling_rate=args.sampling_rate,
                prepare_sec=args.prepare_sec,
                active_sec=args.active_sec,
                rest_sec=args.rest_sec,
                target_repeats=args.target_repeats,
                idle_repeats=args.idle_repeats,
                switch_trials=args.switch_trials,
                seed=args.seed,
            )
            payload = runner.run()
            print(json_dumps(json_safe(payload)), flush=True)
        except Exception as exc:
            print(
                f"Collection failed: {describe_runtime_error(exc, serial_port=args.serial_port)}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        return 0

    if args.command == "train-eval":
        try:
            from ssvep_core.train_eval import OfflineTrainEvalConfig, run_offline_train_eval

            output_report = args.report_path
            if output_report is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_report = args.output_profile.parent / f"offline_train_eval_{stamp}.json"
            config = OfflineTrainEvalConfig(
                dataset_manifest_session1=Path(args.dataset_manifest).expanduser().resolve(),
                dataset_manifest_session2=(
                    None
                    if args.dataset_manifest_session2 is None
                    else Path(args.dataset_manifest_session2).expanduser().resolve()
                ),
                output_profile_path=Path(args.output_profile).expanduser().resolve(),
                report_path=Path(output_report).expanduser().resolve(),
                model_names=tuple(parse_model_list(args.models)),
                channel_modes=tuple(parse_channel_mode_list(args.channel_modes)),
                multi_seed_count=int(args.multi_seed_count),
                seed_step=int(args.seed_step),
                win_candidates=tuple(float(item.strip()) for item in str(args.win_candidates).split(",") if item.strip()),
                gate_policy=parse_gate_policy(args.gate_policy),
                channel_weight_mode=parse_channel_weight_mode(args.channel_weight_mode),
                subband_weight_mode=parse_subband_weight_mode(args.subband_weight_mode),
                spatial_filter_mode=parse_spatial_filter_mode(args.spatial_filter_mode),
                spatial_rank_candidates=parse_spatial_rank_candidates(args.spatial_rank_candidates),
                joint_weight_iters=max(1, int(args.joint_weight_iters)),
                weight_cv_folds=max(2, int(args.weight_cv_folds)),
                spatial_source_model=parse_spatial_source_model(args.spatial_source_model),
                metric_scope=parse_metric_scope(args.metric_scope),
                decision_time_mode=parse_decision_time_mode(args.decision_time_mode),
                async_decision_time_mode=parse_decision_time_mode(args.async_decision_time_mode),
                data_policy=parse_data_policy(args.data_policy),
                export_figures=bool(int(args.export_figures)),
                ranking_policy=parse_ranking_policy(args.ranking_policy),
                dynamic_stop_enabled=not bool(args.disable_dynamic_stop),
                dynamic_stop_alpha=float(args.dynamic_stop_alpha),
                seed=int(args.seed),
                compute_backend=parse_compute_backend_name(args.compute_backend),
                gpu_device=int(args.gpu_device),
                gpu_precision=parse_gpu_precision(args.gpu_precision),
                gpu_warmup=bool(int(args.gpu_warmup)),
                gpu_cache_policy=parse_gpu_cache_policy(args.gpu_cache_policy),
            )
            report = run_offline_train_eval(config, log_fn=lambda text: print(text, flush=True))
            print(json_dumps(json_safe(report)), flush=True)
        except Exception as exc:
            print(
                f"Train-eval failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        return 0

    if args.command == "benchmark":
        try:
            win_candidates = tuple(
                float(item.strip()) for item in str(args.win_candidates).split(",") if item.strip()
            )
            model_names = parse_model_list(args.models)
            channel_modes = parse_channel_mode_list(args.channel_modes)
        except Exception as exc:
            print(
                f"Benchmark argument error: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        try:
            runner = BenchmarkRunner(
                serial_port=args.serial_port,
                board_id=args.board_id,
                freqs=freqs,
                output_profile_path=args.output_profile,
                report_path=args.report_path,
                dataset_dir=args.dataset_dir,
                sampling_rate=args.sampling_rate,
                prepare_sec=args.prepare_sec,
                active_sec=args.active_sec,
                rest_sec=args.rest_sec,
                calibration_target_repeats=args.calibration_target_repeats,
                calibration_idle_repeats=args.calibration_idle_repeats,
                eval_target_repeats=args.eval_target_repeats,
                eval_idle_repeats=args.eval_idle_repeats,
                eval_switch_trials=args.eval_switch_trials,
                step_sec=args.step_sec,
                model_names=model_names,
                channel_modes=channel_modes,
                multi_seed_count=args.multi_seed_count,
                seed_step=args.seed_step,
                win_candidates=win_candidates,
                gate_policy=args.gate_policy,
                channel_weight_mode=args.channel_weight_mode,
                subband_weight_mode=args.subband_weight_mode,
                spatial_filter_mode=args.spatial_filter_mode,
                spatial_rank_candidates=parse_spatial_rank_candidates(args.spatial_rank_candidates),
                joint_weight_iters=max(1, int(args.joint_weight_iters)),
                weight_cv_folds=max(2, int(args.weight_cv_folds)),
                spatial_source_model=args.spatial_source_model,
                dynamic_stop_enabled=not bool(args.disable_dynamic_stop),
                dynamic_stop_alpha=args.dynamic_stop_alpha,
                seed=args.seed,
                compute_backend=parse_compute_backend_name(args.compute_backend),
                gpu_device=int(args.gpu_device),
                gpu_precision=parse_gpu_precision(args.gpu_precision),
                gpu_warmup=bool(int(args.gpu_warmup)),
                gpu_cache_policy=parse_gpu_cache_policy(args.gpu_cache_policy),
            )
            runner.run()
        except Exception as exc:
            print(
                f"Benchmark failed: {describe_runtime_error(exc, serial_port=args.serial_port)}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)


