from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import scipy.linalg
from scipy.signal import butter, filtfilt, iirnotch

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
DEFAULT_MODEL_NAME = "fbcca"
DEFAULT_SERIAL_PORT = "auto"
DEFAULT_GATE_POLICY = "balanced"
DEFAULT_GATE_POLICIES = ("conservative", "balanced")
DEFAULT_CHANNEL_WEIGHT_MODE = "fbcca_diag"
DEFAULT_CHANNEL_WEIGHT_RANGE = (0.3, 2.5)
DEFAULT_DYNAMIC_STOP_ALPHA = 0.7
DEFAULT_DYNAMIC_STOP_ENABLED = True
DEFAULT_BALANCED_IDLE_FP_MAX = 3.0
DEFAULT_BALANCED_MIN_CONTROL_RECALL = 0.60
DEFAULT_BALANCED_OBJECTIVE_IDLE_WEIGHT = 0.45
DEFAULT_BALANCED_OBJECTIVE_RECALL_WEIGHT = 0.35
DEFAULT_BALANCED_OBJECTIVE_SWITCH_WEIGHT = 0.10
DEFAULT_BALANCED_OBJECTIVE_RELEASE_WEIGHT = 0.10
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
DEFAULT_BENCHMARK_RANK_MIN_CONTROL_RECALL = 0.10
DEFAULT_CALIBRATION_MIN_DETECTION_RECALL = 0.05
MODEL_IMPLEMENTATION_LEVELS = {
    "cca": "paper-faithful",
    "fbcca": "paper-faithful",
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
    "itcca": "Template-assisted CCA approximation for personalized reference matching.",
    "ecca": "Extended CCA approximation using template + reference correlation fusion.",
    "msetcca": "Template-based multiset CCA approximation with simplified fusion weights.",
    "trca": "TRCA spatial-filter implementation with class-wise templates.",
    "trca_r": "Filterbank ensemble TRCA approximation of TRCA-R / eTRCA family.",
    "sscor": "SSCOR-inspired multi-component correlation scoring built on TRCA-style filters.",
    "tdca": "TDCA-inspired delay-embedding + discriminant projection approximation.",
    "oacca": "Online adaptive CCA approximation with selected-class template updates.",
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
    dynamic_stop: Optional[dict[str, Any]] = None
    channel_weight_mode: Optional[str] = None
    channel_weights: Optional[tuple[float, ...]] = None
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
        if "channel_weight_mode" in data and data["channel_weight_mode"] is not None:
            data["channel_weight_mode"] = str(data["channel_weight_mode"]).strip().lower()
        if "channel_weights" in data and data["channel_weights"] is not None:
            data["channel_weights"] = tuple(float(value) for value in data["channel_weights"])
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
        dynamic_stop={
            "enabled": bool(DEFAULT_DYNAMIC_STOP_ENABLED),
            "alpha": float(DEFAULT_DYNAMIC_STOP_ALPHA),
            "enter_acc_th": None,
            "exit_acc_th": None,
        },
        channel_weight_mode=None,
        channel_weights=None,
        metadata={
            "source": "default_fallback",
            "has_stat_model": False,
            "requires_calibration": True,
        },
    )


def save_profile(profile: ThresholdProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(asdict(profile))
    payload["freqs"] = list(profile.freqs)
    if profile.eeg_channels is not None:
        payload["eeg_channels"] = list(profile.eeg_channels)
    if profile.channel_weights is not None:
        payload["channel_weights"] = [float(value) for value in profile.channel_weights]
    payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json_dumps(payload) + "\n", encoding="utf-8")


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


def calibration_search_win_candidates(active_sec: float, preferred_win_sec: float) -> list[float]:
    return sorted(
        {
            round(float(candidate), 3)
            for candidate in (*DEFAULT_WIN_SEC_CANDIDATES, float(preferred_win_sec))
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
) -> None:
    problems: list[str] = []
    if int(target_repeats) < 2:
        problems.append("target_repeats must be at least 2 so each target has both train and validation trials")
    if int(idle_repeats) < 2:
        problems.append("idle_repeats must be at least 2 so idle has both train and validation trials")

    win_candidates = calibration_search_win_candidates(active_sec, preferred_win_sec)
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

        self.fs = 0
        self.win_samples = 0
        self.step_samples = 0
        self.subbands: list[tuple[float, float]] = []
        self.subband_filters: list[tuple[np.ndarray, np.ndarray]] = []
        self.weights: np.ndarray | None = None
        self.Y_refs: dict[float, np.ndarray] = {}
        self._baseline_b: Optional[np.ndarray] = None
        self._baseline_a: Optional[np.ndarray] = None
        self._notch_b: Optional[np.ndarray] = None
        self._notch_a: Optional[np.ndarray] = None
        self.configure_runtime(sampling_rate)

    def configure_runtime(self, sampling_rate: int) -> None:
        self.fs = int(sampling_rate)
        if self.fs <= 0:
            raise ValueError("sampling_rate must be positive")
        self.win_samples = int(round(self.win_sec * self.fs))
        self.step_samples = max(1, int(round(self.step_sec * self.fs)))
        if self.win_samples < 64:
            raise ValueError("win_sec is too short for stable FBCCA")

        nyq = self.fs / 2.0
        self._baseline_b, self._baseline_a = butter(1, 3.0 / nyq, btype="low")
        if self.notch_freq < nyq - 1e-6:
            self._notch_b, self._notch_a = iirnotch(self.notch_freq, self.notch_q, self.fs)
        else:
            self._notch_b, self._notch_a = None, None

        self.subbands = []
        self.subband_filters = []
        for low, high in self.base_subbands:
            clipped_high = min(high, nyq - 1e-3)
            if low >= clipped_high:
                continue
            coeff_b, coeff_a = butter(2, [low / nyq, clipped_high / nyq], btype="band")
            self.subbands.append((low, clipped_high))
            self.subband_filters.append((coeff_b, coeff_a))
        if not self.subband_filters:
            raise ValueError(f"sampling_rate={self.fs} is too low for configured subbands")

        a_w, b_w = 1.25, 0.25
        self.weights = np.array(
            [(index + 1) ** (-a_w) + b_w for index in range(len(self.subband_filters))],
            dtype=float,
        )
        self.weights = self.weights / self.weights.sum()
        self.Y_refs = {
            freq: self.build_ref_matrix(self.fs, self.win_samples, freq, self.Nh)
            for freq in self.freqs
        }

    @staticmethod
    def build_ref_matrix(fs: int, T: int, freq: float, Nh: int) -> np.ndarray:
        t = np.arange(T) / float(fs)
        cols = []
        for harmonic in range(1, Nh + 1):
            cols.append(np.sin(2.0 * np.pi * harmonic * freq * t))
            cols.append(np.cos(2.0 * np.pi * harmonic * freq * t))
        Y = np.stack(cols, axis=1)
        return Y - Y.mean(axis=0, keepdims=True)

    def detrend_and_notch(self, x_raw: np.ndarray) -> np.ndarray:
        if self._baseline_b is None or self._baseline_a is None:
            raise RuntimeError("FBCCAEngine runtime is not configured")
        base = filtfilt(self._baseline_b, self._baseline_a, x_raw)
        filtered = x_raw - base
        if self._notch_b is not None and self._notch_a is not None:
            filtered = filtfilt(self._notch_b, self._notch_a, filtered)
        return filtered

    def preprocess_window(self, X_raw: np.ndarray) -> np.ndarray:
        X0 = np.zeros_like(X_raw, dtype=float)
        for channel_index in range(X_raw.shape[1]):
            X0[:, channel_index] = self.detrend_and_notch(X_raw[:, channel_index])
        return X0 - X0.mean(axis=0, keepdims=True)

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

    @staticmethod
    def bandpass_filter_multichannel(X_in: np.ndarray, coeffs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        coeff_b, coeff_a = coeffs
        X_out = np.zeros_like(X_in, dtype=float)
        for channel_index in range(X_in.shape[1]):
            X_out[:, channel_index] = filtfilt(coeff_b, coeff_a, X_in[:, channel_index])
        return X_out - X_out.mean(axis=0, keepdims=True)

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        if X_window.shape[0] != self.win_samples:
            raise ValueError(f"expected {self.win_samples} samples, got {X_window.shape[0]}")
        X0 = self.preprocess_window(np.asarray(X_window, dtype=float))
        X_subbands = [self.bandpass_filter_multichannel(X0, coeffs) for coeffs in self.subband_filters]

        scores = np.zeros(len(self.freqs), dtype=float)
        for freq_index, freq in enumerate(self.freqs):
            ref = self.Y_refs[freq]
            score_value = 0.0
            for subband_index, X_subband in enumerate(X_subbands):
                rho = self.cca_multi_channel_svd(X_subband, ref)
                score_value += float(self.weights[subband_index]) * (rho ** 2)
            scores[freq_index] = score_value
        return scores

    def analyze_window(self, X_window: np.ndarray) -> dict[str, Any]:
        scores = self.score_window(X_window)
        order = np.argsort(scores)[::-1]
        top1_index = int(order[0])
        top2_index = int(order[1]) if len(order) > 1 else top1_index
        top1_score = float(scores[top1_index])
        top2_score = float(scores[top2_index]) if len(order) > 1 else 0.0
        margin = top1_score - top2_score
        ratio = float(top1_score / max(top2_score, 1e-12))
        score_sum = float(np.sum(scores))
        safe_sum = max(score_sum, 1e-12)
        score_probs = np.clip(scores / safe_sum, 1e-12, None)
        score_probs = score_probs / score_probs.sum()
        score_entropy = float(-np.sum(score_probs * np.log(score_probs)) / np.log(len(score_probs)))
        normalized_top1 = float(top1_score / safe_sum)
        return {
            "scores": scores,
            "pred_freq": float(self.freqs[top1_index]),
            "top1_score": top1_score,
            "top2_score": top2_score,
            "margin": float(margin),
            "ratio": float(ratio),
            "score_sum": score_sum,
            "normalized_top1": normalized_top1,
            "score_entropy": score_entropy,
        }

    def iter_window_features(
        self,
        segment: np.ndarray,
        *,
        expected_freq: Optional[float],
        label: str,
        trial_id: int = -1,
        block_index: int = -1,
    ) -> list[dict[str, Any]]:
        segment = np.asarray(segment, dtype=float)
        if segment.ndim != 2:
            raise ValueError("segment must have shape (samples, channels)")
        if segment.shape[0] < self.win_samples:
            raise ValueError("segment is shorter than the analysis window")

        rows: list[dict[str, Any]] = []
        for window_index, start in enumerate(range(0, segment.shape[0] - self.win_samples + 1, self.step_samples)):
            window = segment[start : start + self.win_samples]
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
        nh = int(params.get("Nh", DEFAULT_NH))
        subbands = tuple(params.get("subbands", DEFAULT_SUBBANDS))
        super().__init__(
            model_name="fbcca",
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
        )

    def configure_runtime(self, sampling_rate: int) -> None:
        super().configure_runtime(sampling_rate)
        self.engine.configure_runtime(self.fs)

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        return self.engine.score_window(X_window)

    def analyze_window(self, X_window: np.ndarray) -> dict[str, Any]:
        return self.engine.analyze_window(X_window)


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
        )

    def configure_runtime(self, sampling_rate: int) -> None:
        super().configure_runtime(sampling_rate)
        self._core.configure_runtime(self.fs)

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        X0 = self._core.preprocess_window(np.asarray(X_window, dtype=float))
        scores = np.zeros(len(self.freqs), dtype=float)
        for freq_index, freq in enumerate(self.freqs):
            rho = self._core.cca_multi_channel_svd(X0, self._core.Y_refs[freq])
            scores[freq_index] = float(max(rho, 0.0) ** 2)
        return scores


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
        )
        self.templates: dict[float, np.ndarray] = {}
        self.template_weights: dict[float, np.ndarray] = {}

    @property
    def requires_fit(self) -> bool:
        return True

    def configure_runtime(self, sampling_rate: int) -> None:
        super().configure_runtime(sampling_rate)
        self._core.configure_runtime(self.fs)

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
        )
        self.band_templates: list[dict[float, np.ndarray]] = []
        self.band_filters: list[dict[float, np.ndarray]] = []

    @property
    def requires_fit(self) -> bool:
        return True

    def configure_runtime(self, sampling_rate: int) -> None:
        super().configure_runtime(sampling_rate)
        self._core.configure_runtime(self.fs)

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
        )
        self.band_projection: list[np.ndarray] = []
        self.band_templates: list[dict[float, np.ndarray]] = []

    @property
    def requires_fit(self) -> bool:
        return True

    def configure_runtime(self, sampling_rate: int) -> None:
        super().configure_runtime(sampling_rate)
        self._core.configure_runtime(self.fs)

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
    "cca": "cca",
    "itcca": "itcca",
    "ecca": "ecca",
    "msetcca": "msetcca",
    "trca": "trca",
    "trca_r": "trca_r",
    "trca-r": "trca_r",
    "etrca": "trca_r",
    "sscor": "sscor",
    "tdca": "tdca",
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


def parse_channel_weight_mode(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"", "none", "off", "disabled"}:
        return None
    if value != "fbcca_diag":
        raise ValueError(f"unsupported channel weight mode: {raw}")
    return value


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
) -> BaseSSVEPDecoder:
    name = normalize_model_name(model_name)
    params = dict(model_params or {})
    if name == "fbcca":
        return FBCCADecoder(
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


def load_decoder_from_profile(profile: ThresholdProfile, *, sampling_rate: int) -> BaseSSVEPDecoder:
    model_params = dict(profile.model_params or {})
    if profile.channel_weight_mode is not None and "channel_weight_mode" not in model_params:
        model_params["channel_weight_mode"] = str(profile.channel_weight_mode)
    if profile.channel_weights is not None and "channel_weights" not in model_params:
        model_params["channel_weights"] = [float(value) for value in profile.channel_weights]
    decoder = create_decoder(
        profile.model_name,
        sampling_rate=sampling_rate,
        freqs=profile.freqs,
        win_sec=profile.win_sec,
        step_sec=profile.step_sec,
        model_params=model_params,
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
    ) -> None:
        self.enter_score_th = float(enter_score_th)
        self.enter_ratio_th = float(enter_ratio_th)
        self.enter_margin_th = float(enter_margin_th)
        self.exit_score_th = float(exit_score_th)
        self.exit_ratio_th = float(exit_ratio_th)
        self.min_enter_windows = max(1, int(min_enter_windows))
        self.min_exit_windows = max(1, int(min_exit_windows))
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
        )

    def reset(self) -> None:
        self.state = "idle"
        self._candidate_freq: Optional[float] = None
        self._selected_freq: Optional[float] = None
        self._candidate_windows = 0
        self._support_windows = 0
        self._exit_windows = 0
        self._acc_log_lr = 0.0

    def _enter_pass(self, features: dict[str, Any]) -> bool:
        legacy_pass = (
            float(features["top1_score"]) >= self.enter_score_th
            and float(features["ratio"]) >= self.enter_ratio_th
            and float(features["margin"]) >= self.enter_margin_th
        )
        if self.enter_log_lr_th is None:
            pass_by_log_lr = True
        else:
            control_log_lr = features.get("control_log_lr")
            pass_by_log_lr = control_log_lr is not None and float(control_log_lr) >= self.enter_log_lr_th
        if not (legacy_pass and pass_by_log_lr):
            return False
        if self.dynamic_stop_enabled and self.enter_acc_th is not None:
            acc_log_lr = features.get("acc_log_lr")
            return acc_log_lr is not None and float(acc_log_lr) >= self.enter_acc_th
        return True

    def _exit_fail(self, features: dict[str, Any]) -> bool:
        if self._selected_freq is None:
            return True
        pred_freq = features.get("pred_freq")
        if pred_freq is None or abs(float(pred_freq) - float(self._selected_freq)) > 1e-8:
            return True
        if float(features["top1_score"]) < self.exit_score_th:
            return True
        if float(features["ratio"]) < self.exit_ratio_th:
            return True
        if self.exit_log_lr_th is not None:
            control_log_lr = features.get("control_log_lr")
            if control_log_lr is None or float(control_log_lr) < self.exit_log_lr_th:
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

        if self.state == "idle":
            if enter_pass:
                self.state = "candidate"
                self._candidate_freq = float(pred_freq)
                self._candidate_windows = 1
                self._support_windows = 1
            else:
                self._candidate_freq = None
                self._candidate_windows = 0
                self._support_windows = 0
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
        else:
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
    def is_idle_row(row: dict[str, Any]) -> bool:
        return row.get("expected_freq") is None

    def passes(row: dict[str, Any]) -> bool:
        control_log_lr = profile_log_lr(profile, row)
        log_lr_pass = profile.enter_log_lr_th is None or (
            control_log_lr is not None and float(control_log_lr) >= float(profile.enter_log_lr_th)
        )
        return (
            float(row["top1_score"]) >= profile.enter_score_th
            and float(row["ratio"]) >= profile.enter_ratio_th
            and float(row["margin"]) >= profile.enter_margin_th
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
    control_trials = 0
    control_detected_trials = 0
    switch_trials = 0
    switch_detected_trials = 0
    release_trials = 0
    release_detected_trials = 0
    detection_latencies: list[float] = []
    switch_latencies: list[float] = []
    release_latencies: list[float] = []
    last_control_freq: Optional[float] = None
    previous_trial_expected_freq: Optional[float] = None
    has_explicit_switch = any(
        row.get("expected_freq") is not None and str(row.get("label", "")).startswith("switch_to_")
        for row in rows
    )

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
            window_index = int(row.get("window_index", 0))
            latency_value = float(profile.win_sec + window_index * profile.step_sec)

            if expected_numeric is None:
                idle_windows += 1
                if selected is not None:
                    idle_selected_windows += 1
                if previous_trial_expected_freq is not None and first_release_latency is None and selected is None:
                    first_release_latency = latency_value
                continue

            if (
                first_correct_latency is None
                and selected is not None
                and abs(float(selected) - float(expected_numeric)) < 1e-8
            ):
                first_correct_latency = latency_value

        if expected_numeric is None:
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
        last_control_freq = float(expected_numeric)
        previous_trial_expected_freq = float(expected_numeric)

    idle_minutes = float(idle_windows) * float(profile.step_sec) / 60.0
    idle_fp_per_min = float(idle_selected_windows / idle_minutes) if idle_minutes > 1e-12 else 0.0
    control_recall = float(control_detected_trials / control_trials) if control_trials else 0.0
    switch_detect_rate = float(switch_detected_trials / switch_trials) if switch_trials else 0.0
    release_detect_rate = float(release_detected_trials / release_trials) if release_trials else 0.0
    switch_latency = float(np.median(np.asarray(switch_latencies, dtype=float))) if switch_latencies else float("inf")
    release_latency = (
        float(np.median(np.asarray(release_latencies, dtype=float))) if release_latencies else float("inf")
    )
    detection_latency = (
        float(np.median(np.asarray(detection_latencies, dtype=float))) if detection_latencies else float("inf")
    )
    return {
        "idle_fp_per_min": idle_fp_per_min,
        "control_recall": control_recall,
        "switch_detect_rate": switch_detect_rate,
        "release_detect_rate": release_detect_rate,
        "switch_latency_s": switch_latency,
        "release_latency_s": release_latency,
        "detection_latency_s": detection_latency,
        "idle_windows": float(idle_windows),
        "control_trials": float(control_trials),
        "switch_trials": float(switch_trials),
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
) -> list[dict[str, Any]]:
    resolved_channels = resolve_selected_eeg_channels(available_board_channels, selected_board_channels)
    channel_positions = [tuple(int(value) for value in available_board_channels).index(channel) for channel in resolved_channels]
    engine = FBCCAEngine(
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=win_sec,
        step_sec=step_sec,
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
) -> tuple[np.ndarray, dict[str, Any]]:
    if not train_segments or not gate_segments:
        raise ValueError("channel-weight optimization requires non-empty train and gate segments")
    channels = int(np.asarray(train_segments[0][1]).shape[1])
    initial_weights = estimate_fbcca_diag_channel_weights(
        train_segments,
        available_board_channels=tuple(range(channels)),
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=win_sec,
        step_sec=step_sec,
    )
    initial_weights = normalize_channel_weights(initial_weights, channels=channels)
    if initial_weights is None:
        initial_weights = np.ones(channels, dtype=float)

    factors = (0.75, 0.90, 1.00, 1.10, 1.25)

    def evaluate_weights(candidate_weights: np.ndarray) -> tuple[tuple[float, ...], ThresholdProfile, dict[str, float]]:
        params = {
            "Nh": DEFAULT_NH,
            "channel_weight_mode": "fbcca_diag",
            "channel_weights": [float(value) for value in candidate_weights],
        }
        decoder = create_decoder(
            "fbcca",
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
        )
        gate_rows = build_feature_rows_with_decoder(decoder, gate_segments)
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
        objective = _calibration_objective({}, gate_policy=gate_policy, gate_metrics=metrics)
        return objective, profile, metrics

    best_weights = np.asarray(initial_weights, dtype=float)
    best_objective, best_profile, best_metrics = evaluate_weights(best_weights)

    for _pass in range(2):
        improved = False
        for ch in range(channels):
            local_best_weights = np.asarray(best_weights, dtype=float)
            local_best_objective = tuple(best_objective)
            local_profile = best_profile
            local_metrics = dict(best_metrics)
            for factor in factors:
                candidate = np.asarray(best_weights, dtype=float)
                candidate[ch] *= float(factor)
                normalized = normalize_channel_weights(candidate, channels=channels)
                if normalized is None:
                    continue
                try:
                    objective, profile, metrics = evaluate_weights(np.asarray(normalized, dtype=float))
                except Exception:
                    continue
                if objective < local_best_objective:
                    local_best_objective = tuple(objective)
                    local_best_weights = np.asarray(normalized, dtype=float)
                    local_profile = profile
                    local_metrics = dict(metrics)
            if local_best_objective < best_objective:
                best_objective = local_best_objective
                best_weights = local_best_weights
                best_profile = local_profile
                best_metrics = local_metrics
                improved = True
        if not improved:
            break

    metadata = {
        "mode": "fbcca_diag",
        "objective": [float(value) for value in best_objective],
        "initial_weights": [float(value) for value in initial_weights],
        "optimized_weights": [float(value) for value in best_weights],
        "gate_metrics": {key: float(value) for key, value in best_metrics.items()},
        "fit_profile": json_safe(asdict(best_profile)),
    }
    return np.asarray(best_weights, dtype=float), metadata


def select_auto_eeg_channels(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    available_board_channels: Sequence[int],
    sampling_rate: int,
    freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
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
    dynamic_stop_enabled: bool = DEFAULT_DYNAMIC_STOP_ENABLED,
    dynamic_stop_alpha: float = DEFAULT_DYNAMIC_STOP_ALPHA,
) -> tuple[ThresholdProfile, dict[str, Any]]:
    policy = parse_gate_policy(gate_policy)
    weight_mode = parse_channel_weight_mode(channel_weight_mode)
    selected_channels, channel_scores = select_auto_eeg_channels(
        trial_segments,
        available_board_channels=available_board_channels,
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=preferred_win_sec,
        step_sec=step_sec,
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

    win_candidates = calibration_search_win_candidates(active_sec, preferred_win_sec)
    best_profile: Optional[ThresholdProfile] = None
    best_summary: Optional[dict[str, float]] = None
    best_search: Optional[dict[str, Any]] = None
    best_objective: Optional[tuple[float, ...]] = None
    best_gate_metrics: Optional[dict[str, float]] = None
    best_weight_metadata: Optional[dict[str, Any]] = None

    for win_sec in win_candidates:
        available_windows = calibration_window_count(active_sec, win_sec, step_sec)
        valid_enter_candidates = [
            int(candidate) for candidate in DEFAULT_MIN_ENTER_CANDIDATES if int(candidate) <= int(available_windows)
        ]
        if not valid_enter_candidates:
            continue
        for min_enter in valid_enter_candidates:
            for min_exit in DEFAULT_MIN_EXIT_CANDIDATES:
                model_params: dict[str, Any] = {"Nh": DEFAULT_NH}
                weight_metadata: Optional[dict[str, Any]] = None
                if weight_mode == "fbcca_diag":
                    try:
                        weights, weight_metadata = optimize_fbcca_diag_channel_weights(
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
                        )
                        model_params["channel_weight_mode"] = "fbcca_diag"
                        model_params["channel_weights"] = [float(value) for value in weights]
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
    if weight_mode == "fbcca_diag" and best_weight_metadata is not None:
        optimized_weights = list(best_weight_metadata.get("optimized_weights", []))
        normalized = normalize_channel_weights(optimized_weights, channels=len(selected_channels))
        if normalized is not None:
            final_model_params["channel_weight_mode"] = "fbcca_diag"
            final_model_params["channel_weights"] = [float(value) for value in normalized]

    final_decoder = create_decoder(
        DEFAULT_MODEL_NAME,
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=float(best_search["win_sec"]),
        step_sec=step_sec,
        model_params=final_model_params,
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
    )
    refit_summary = summarize_profile_quality(all_rows, refit_profile)
    refit_gate_metrics = evaluate_profile_on_feature_rows(all_rows, refit_profile)

    channel_weights = final_model_params.get("channel_weights")
    profile_channel_weights = None
    if channel_weights is not None:
        profile_channel_weights = tuple(float(value) for value in channel_weights)
    metadata = {
        "source": "calibration",
        "calibration_seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "gate_policy": policy,
        "channel_weight_mode": weight_mode,
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
) -> ThresholdProfile:
    policy = parse_gate_policy(gate_policy)
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
        if policy == "balanced":
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
        enter_log_lr_th=best_log_lr_th,
        exit_log_lr_th=exit_log_lr_th,
        control_feature_means={name: float(value) for name, value in zip(MODEL_FEATURE_NAMES, control_means)},
        control_feature_stds={name: float(value) for name, value in zip(MODEL_FEATURE_NAMES, control_stds)},
        idle_feature_means={name: float(value) for name, value in zip(MODEL_FEATURE_NAMES, idle_means)},
        idle_feature_stds={name: float(value) for name, value in zip(MODEL_FEATURE_NAMES, idle_stds)},
        gate_policy=policy,
        dynamic_stop=dynamic_payload,
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


def evaluate_decoder_on_trials(
    decoder: BaseSSVEPDecoder,
    profile: ThresholdProfile,
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    dynamic_stop_enabled: Optional[bool] = None,
) -> dict[str, float]:
    gate_profile = profile
    if dynamic_stop_enabled is not None:
        payload = dict(profile.dynamic_stop or {})
        payload["enabled"] = bool(dynamic_stop_enabled)
        gate_profile = replace(profile, dynamic_stop=payload)
    gate = AsyncDecisionGate.from_profile(gate_profile)
    gate.reset()
    idle_selected_windows = 0
    idle_windows = 0
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
    switch_penalty_trials = 0
    release_trials = 0
    release_detected_trials = 0
    release_penalty_trials = 0
    has_explicit_switch_trials = any(
        trial.expected_freq is not None and str(trial.label).startswith("switch_to_")
        for trial, _segment in trial_segments
    )

    for trial, segment in trial_segments:
        segment_matrix = np.asarray(segment, dtype=float)
        if segment_matrix.shape[0] < decoder.win_samples:
            continue
        first_correct_latency: Optional[float] = None
        first_release_latency: Optional[float] = None
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
            if trial.expected_freq is None:
                idle_windows += 1
                if decision.get("selected_freq") is not None:
                    idle_selected_windows += 1
                if prev_trial_expected_freq is not None and first_release_latency is None and decision.get("selected_freq") is None:
                    first_release_latency = float(decoder.win_sec + window_index * decoder.step_sec)
            else:
                selected = decision.get("selected_freq")
                if (
                    first_correct_latency is None
                    and selected is not None
                    and abs(float(selected) - float(trial.expected_freq)) < 1e-8
                ):
                    first_correct_latency = float(decoder.win_sec + window_index * decoder.step_sec)
            window_index += 1

        if trial.expected_freq is None:
            if prev_trial_expected_freq is not None:
                release_trials += 1
                trial_duration_sec = float(segment_matrix.shape[0]) / max(float(decoder.fs), 1.0)
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
        if is_switch_trial:
            trial_duration_sec = float(segment_matrix.shape[0]) / max(float(decoder.fs), 1.0)
            penalty_latency = float(trial_duration_sec + decoder.win_sec)
            if first_correct_latency is None:
                switch_penalty_trials += 1
                switch_latencies.append(penalty_latency)
            else:
                switch_detected_trials += 1
                switch_latencies.append(float(first_correct_latency))
        last_control_freq = float(trial.expected_freq)
        prev_trial_expected_freq = float(trial.expected_freq)

    idle_minutes = float(idle_windows) * float(decoder.step_sec) / 60.0
    idle_fp_per_min = float(idle_selected_windows / idle_minutes) if idle_minutes > 1e-12 else 0.0
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
    release_detect_rate = float(release_detected_trials / release_trials) if release_trials else 0.0
    itr_bpm = compute_itr_bits_per_minute(
        accuracy=control_recall,
        class_count=len(decoder.freqs),
        decision_time_sec=median_detection_latency if np.isfinite(median_detection_latency) else decoder.win_sec,
    )
    return {
        "idle_fp_per_min": idle_fp_per_min,
        "control_recall": control_recall,
        "control_miss_rate": control_miss_rate,
        "switch_detect_rate": switch_detect_rate,
        "switch_latency_s": median_switch_latency,
        "release_detect_rate": release_detect_rate,
        "release_latency_s": median_release_latency,
        "detection_latency_s": median_detection_latency,
        "itr_bpm": itr_bpm,
        "inference_ms": float(np.mean(inference_latencies_ms)) if inference_latencies_ms else float("inf"),
        "control_trials": float(control_trials),
        "idle_windows": float(idle_windows),
        "switch_trials": float(switch_trials),
        "switch_detected_trials": float(switch_detected_trials),
        "switch_penalty_trials": float(switch_penalty_trials),
        "release_trials": float(release_trials),
        "release_detected_trials": float(release_detected_trials),
        "release_penalty_trials": float(release_penalty_trials),
    }


def benchmark_rank_key(metrics: dict[str, float]) -> tuple[float, float, float, float, float, float, float, float]:
    control_recall = float(metrics.get("control_recall", 0.0))
    low_recall_penalty = 1.0 if control_recall < float(DEFAULT_BENCHMARK_RANK_MIN_CONTROL_RECALL) else 0.0
    return (
        low_recall_penalty,
        float(metrics.get("idle_fp_per_min", float("inf"))),
        -control_recall,
        -float(metrics.get("switch_detect_rate", 0.0)),
        float(metrics.get("switch_latency_s", float("inf"))),
        float(metrics.get("release_latency_s", float("inf"))),
        -float(metrics.get("itr_bpm", 0.0)),
        float(metrics.get("inference_ms", float("inf"))),
    )


def profile_meets_acceptance(metrics: dict[str, float]) -> bool:
    return (
        float(metrics.get("idle_fp_per_min", float("inf"))) <= float(DEFAULT_ACCEPTANCE_IDLE_FP_PER_MIN)
        and float(metrics.get("control_recall", 0.0)) >= float(DEFAULT_ACCEPTANCE_CONTROL_RECALL)
        and float(metrics.get("switch_latency_s", float("inf"))) <= float(DEFAULT_ACCEPTANCE_SWITCH_LATENCY_S)
        and float(metrics.get("release_latency_s", float("inf"))) <= float(DEFAULT_ACCEPTANCE_RELEASE_LATENCY_S)
        and float(metrics.get("inference_ms", float("inf"))) < float(DEFAULT_ACCEPTANCE_INFERENCE_MS)
    )


def benchmark_metric_definition_payload() -> dict[str, Any]:
    return {
        "ranking_policy": {
            "description": (
                "Constrained lexicographic ranking for asynchronous control-state decoding."
                " Models below the minimum control recall are down-ranked before idle FP ordering."
            ),
            "min_control_recall_for_ranking": float(DEFAULT_BENCHMARK_RANK_MIN_CONTROL_RECALL),
            "priority": [
                "low_recall_penalty",
                "idle_fp_per_min",
                "control_recall",
                "switch_detect_rate",
                "switch_latency_s",
                "release_latency_s",
                "itr_bpm",
                "inference_ms",
            ],
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
            "dynamic_stop_defaults": {
                "enabled": bool(DEFAULT_DYNAMIC_STOP_ENABLED),
                "alpha": float(DEFAULT_DYNAMIC_STOP_ALPHA),
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
    }


def summarize_benchmark_robustness(
    run_items: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = [dict(item) for item in run_items]
    grouped_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, item in enumerate(runs):
        mode = str(item.get("channel_mode", "auto"))
        seed = int(item.get("eval_seed", -1))
        grouped_indices[(mode, seed)].append(idx)

    for key, indices in grouped_indices.items():
        successful_indices = [idx for idx in indices if "metrics" in runs[idx]]
        successful_indices.sort(key=lambda idx: benchmark_rank_key(dict(runs[idx]["metrics"])))
        for rank, idx in enumerate(successful_indices, start=1):
            runs[idx]["run_rank"] = int(rank)
        for idx in indices:
            if "metrics" not in runs[idx]:
                runs[idx]["run_rank"] = None

    metrics_keys = (
        "idle_fp_per_min",
        "control_recall",
        "control_miss_rate",
        "switch_detect_rate",
        "switch_latency_s",
        "release_latency_s",
        "detection_latency_s",
        "itr_bpm",
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
            if float(item.get("metrics", {}).get("control_recall", 0.0))
            >= float(DEFAULT_BENCHMARK_RANK_MIN_CONTROL_RECALL)
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
                benchmark_rank_key(item.get("metrics_mean", {})),
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
        dynamic_stop_enabled: bool = DEFAULT_DYNAMIC_STOP_ENABLED,
        dynamic_stop_alpha: float = DEFAULT_DYNAMIC_STOP_ALPHA,
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
        self.dynamic_stop_enabled = bool(dynamic_stop_enabled)
        self.dynamic_stop_alpha = float(dynamic_stop_alpha)
        self.engine = FBCCAEngine(sampling_rate=sampling_rate, freqs=self.freqs, win_sec=win_sec, step_sec=step_sec)

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
                    dynamic_stop_enabled=self.dynamic_stop_enabled,
                    dynamic_stop_alpha=self.dynamic_stop_alpha,
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
        dynamic_stop_enabled: bool = DEFAULT_DYNAMIC_STOP_ENABLED,
        dynamic_stop_alpha: float = DEFAULT_DYNAMIC_STOP_ALPHA,
        seed: int = DEFAULT_CALIBRATION_SEED,
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
        self.dynamic_stop_enabled = bool(dynamic_stop_enabled)
        self.dynamic_stop_alpha = float(dynamic_stop_alpha)
        self._sampling_rate_hint = int(sampling_rate)

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
    ) -> tuple[ThresholdProfile, dict[str, Any]]:
        best_config: Optional[dict[str, Any]] = None
        best_objective: Optional[tuple[float, ...]] = None

        for win_sec in self.win_candidates:
            available_windows = calibration_window_count(self.active_sec, win_sec, self.step_sec)
            valid_enter = [candidate for candidate in DEFAULT_MIN_ENTER_CANDIDATES if int(candidate) <= available_windows]
            if not valid_enter:
                continue
            for min_enter in valid_enter:
                for min_exit in DEFAULT_MIN_EXIT_CANDIDATES:
                    try:
                        model_params: dict[str, Any] = {"Nh": DEFAULT_NH}
                        channel_weight_training = None
                        if (
                            normalize_model_name(model_name) == "fbcca"
                            and self.channel_weight_mode == "fbcca_diag"
                        ):
                            optimized_weights, channel_weight_training = optimize_fbcca_diag_channel_weights(
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
                            )
                            model_params["channel_weight_mode"] = "fbcca_diag"
                            model_params["channel_weights"] = [float(value) for value in optimized_weights]
                        decoder = create_decoder(
                            model_name,
                            sampling_rate=fs,
                            freqs=self.freqs,
                            win_sec=win_sec,
                            step_sec=self.step_sec,
                            model_params=model_params,
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

        final_decoder = create_decoder(
            model_name,
            sampling_rate=fs,
            freqs=self.freqs,
            win_sec=float(best_config["win_sec"]),
            step_sec=self.step_sec,
            model_params=dict(best_config.get("model_params", {"Nh": DEFAULT_NH})),
        )
        if final_decoder.requires_fit:
            final_decoder.fit([*train_segments, *gate_segments])
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
        )
        eval_metrics = evaluate_decoder_on_trials(final_decoder, final_profile, eval_segments)
        fixed_metrics = evaluate_decoder_on_trials(
            final_decoder,
            final_profile,
            eval_segments,
            dynamic_stop_enabled=False,
        )
        dynamic_delta: dict[str, float] = {}
        for metric_name in (
            "idle_fp_per_min",
            "control_recall",
            "switch_detect_rate",
            "switch_latency_s",
            "release_latency_s",
            "detection_latency_s",
            "itr_bpm",
        ):
            dynamic_value = float(eval_metrics.get(metric_name, 0.0))
            fixed_value = float(fixed_metrics.get(metric_name, 0.0))
            if np.isfinite(dynamic_value) and np.isfinite(fixed_value):
                dynamic_delta[metric_name] = float(dynamic_value - fixed_value)
        state = json_safe(final_decoder.get_state())
        model_params = dict(final_decoder.model_params)
        model_params["state"] = state
        profile_channel_weights = model_params.get("channel_weights")
        profile_weight_tuple = (
            tuple(float(value) for value in profile_channel_weights)
            if isinstance(profile_channel_weights, (list, tuple))
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
                "channel_weight_training": best_config.get("channel_weight_training"),
                "dynamic_comparison": {
                    "dynamic": eval_metrics,
                    "fixed": fixed_metrics,
                    "delta": dynamic_delta,
                },
                "has_stat_model": profile_has_stat_model(final_profile),
            },
        )
        return final_profile, {
            "model_name": normalize_model_name(model_name),
            "implementation_level": model_implementation_level(model_name),
            "method_note": model_method_note(model_name),
            "metrics": eval_metrics,
            "fixed_window_metrics": fixed_metrics,
            "dynamic_delta": dynamic_delta,
            "rank_key": benchmark_rank_key(eval_metrics),
            "rank_constraints": {
                "min_control_recall_for_ranking": float(DEFAULT_BENCHMARK_RANK_MIN_CONTROL_RECALL),
                "control_recall_pass": bool(
                    float(eval_metrics.get("control_recall", 0.0)) >= float(DEFAULT_BENCHMARK_RANK_MIN_CONTROL_RECALL)
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
                model_params={"Nh": DEFAULT_NH},
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
                "dynamic_stop_enabled": bool(self.dynamic_stop_enabled),
                "dynamic_stop_alpha": float(self.dynamic_stop_alpha),
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

                    seed_mode_success.sort(key=lambda item: benchmark_rank_key(item["metrics"]))
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
            successful.sort(key=lambda item: benchmark_rank_key(item["metrics"]))
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
            robustness_summary = summarize_benchmark_robustness(robustness_runs)
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
                "chosen_fixed_window_metrics": dict(chosen_result.get("fixed_window_metrics", {})),
                "chosen_dynamic_delta": dict(chosen_result.get("dynamic_delta", {})),
                "gate_policy": str(self.gate_policy),
                "channel_weight_mode": self.channel_weight_mode,
                "dynamic_stop_enabled": bool(self.dynamic_stop_enabled),
                "dynamic_stop_alpha": float(self.dynamic_stop_alpha),
                "model_method_mapping": method_mapping,
                "metric_definition": benchmark_metric_definition_payload(),
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
    ) -> None:
        self.requested_serial_port = normalize_serial_port(serial_port)
        self.serial_port = self.requested_serial_port
        self.board_id = int(board_id)
        self.freqs = tuple(float(freq) for freq in freqs)
        self.profile_path = Path(profile_path)
        self.emit_all = bool(emit_all)
        self.result_callback = result_callback
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
    calibrate.add_argument(
        "--disable-dynamic-stop",
        action="store_true",
        help="disable accumulated-evidence dynamic stopping in gate fitting",
    )
    calibrate.add_argument("--dynamic-stop-alpha", type=float, default=DEFAULT_DYNAMIC_STOP_ALPHA)

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
    online.add_argument("--emit-all", action="store_true", help="emit every update instead of only state changes")
    online.add_argument("--max-updates", type=int, default=None)

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
    benchmark.add_argument(
        "--disable-dynamic-stop",
        action="store_true",
        help="disable accumulated-evidence dynamic stopping in gate fitting",
    )
    benchmark.add_argument("--dynamic-stop-alpha", type=float, default=DEFAULT_DYNAMIC_STOP_ALPHA)
    benchmark.add_argument("--seed", type=int, default=DEFAULT_CALIBRATION_SEED)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    freqs = parse_freqs(args.freqs)

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
                dynamic_stop_enabled=not bool(args.disable_dynamic_stop),
                dynamic_stop_alpha=args.dynamic_stop_alpha,
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

    if args.command == "online":
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
                dynamic_stop_enabled=not bool(args.disable_dynamic_stop),
                dynamic_stop_alpha=args.dynamic_stop_alpha,
                seed=args.seed,
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


