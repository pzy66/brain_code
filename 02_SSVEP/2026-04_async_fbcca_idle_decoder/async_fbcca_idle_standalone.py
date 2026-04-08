from __future__ import annotations

import argparse
import json
import math
import random
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
DEFAULT_WIN_SEC_CANDIDATES = (2.0, 2.5, 3.0)
DEFAULT_MIN_ENTER_CANDIDATES = (2, 3)
DEFAULT_MIN_EXIT_CANDIDATES = (2, 3)
DEFAULT_MODEL_NAME = "fbcca"
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
)
DEFAULT_BENCHMARK_SWITCH_TRIALS = 16
DEFAULT_DECODER_UPDATE_ALPHA = 0.08


def parse_freqs(raw: str) -> tuple[float, float, float, float]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if len(values) != 4:
        raise ValueError("freqs must contain exactly four comma-separated values")
    return values


def require_brainflow() -> None:
    if BoardShim is None or BrainFlowInputParams is None:
        raise RuntimeError("BrainFlow is required for calibrate/online mode.")


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
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
    metadata: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ThresholdProfile":
        names = {item.name for item in fields(cls)}
        data = {key: payload[key] for key in names if key in payload}
        if "freqs" in data:
            data["freqs"] = tuple(float(value) for value in data["freqs"])
        if "eeg_channels" in data and data["eeg_channels"] is not None:
            data["eeg_channels"] = tuple(int(value) for value in data["eeg_channels"])
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

    @property
    def requires_fit(self) -> bool:
        return False

    def configure_runtime(self, sampling_rate: int) -> None:
        self.fs = int(sampling_rate)
        self.win_samples = max(1, int(round(self.win_sec * self.fs)))
        self.step_samples = max(1, int(round(self.step_sec * self.fs)))

    def fit(self, trial_segments: Sequence[tuple[TrialSpec, np.ndarray]]) -> None:
        _ = trial_segments

    @abstractmethod
    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def analyze_window(self, X_window: np.ndarray) -> dict[str, Any]:
        return scores_to_feature_dict(self.score_window(X_window), self.freqs)

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
            window = _extract_last_window(segment, self._core.win_samples)
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
        if self.delay_steps <= 1:
            return matrix
        if matrix.shape[0] <= self.delay_steps:
            raise ValueError(
                f"{self.model_name}: win_samples={matrix.shape[0]} too short for delay_steps={self.delay_steps}"
            )
        rows: list[np.ndarray] = []
        target_length = matrix.shape[0] - self.delay_steps + 1
        for delay in range(self.delay_steps):
            rows.append(matrix[delay : delay + target_length, :])
        return np.concatenate(rows, axis=1)

    @staticmethod
    def _train_trca_filter(trials: Sequence[np.ndarray]) -> np.ndarray:
        if not trials:
            raise ValueError("TRCA training trials are empty")
        channels = int(trials[0].shape[0])
        if len(trials) == 1:
            u, _s, _vt = np.linalg.svd(trials[0], full_matrices=False)
            return np.asarray(u[:, 0], dtype=float)
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
        best = eigvecs[:, int(np.argmax(eigvals))]
        norm = np.linalg.norm(best)
        if norm <= 1e-12:
            return np.ones((channels,), dtype=float) / math.sqrt(max(channels, 1))
        return np.asarray(best / norm, dtype=float)

    def fit(self, trial_segments: Sequence[tuple[TrialSpec, np.ndarray]]) -> None:
        grouped: dict[float, list[np.ndarray]] = defaultdict(list)
        for trial, segment in trial_segments:
            if trial.expected_freq is None:
                continue
            window = _extract_last_window(segment, self._core.win_samples)
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
            window = _extract_last_window(segment, self._core.win_samples)
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
        candidate = self._core.preprocess_window(np.asarray(window, dtype=float))
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
    if name in {"trca_r", "sscor"}:
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
    if name == "tdca":
        return TRCABasedDecoder(
            model_name=name,
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
            use_filterbank=True,
            ensemble=True,
            delay_steps=int(params.get("delay_steps", 3)),
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
    decoder = create_decoder(
        profile.model_name,
        sampling_rate=sampling_rate,
        freqs=profile.freqs,
        win_sec=profile.win_sec,
        step_sec=profile.step_sec,
        model_params=profile.model_params,
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
        )

    def reset(self) -> None:
        self.state = "idle"
        self._candidate_freq: Optional[float] = None
        self._selected_freq: Optional[float] = None
        self._candidate_windows = 0
        self._support_windows = 0
        self._exit_windows = 0

    def _enter_pass(self, features: dict[str, Any]) -> bool:
        legacy_pass = (
            float(features["top1_score"]) >= self.enter_score_th
            and float(features["ratio"]) >= self.enter_ratio_th
            and float(features["margin"]) >= self.enter_margin_th
        )
        if self.enter_log_lr_th is None:
            return legacy_pass
        control_log_lr = features.get("control_log_lr")
        return legacy_pass and control_log_lr is not None and float(control_log_lr) >= self.enter_log_lr_th

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
            if value >= 0.0:
                exp_neg = math.exp(-value)
                features["control_confidence"] = float(1.0 / (1.0 + exp_neg))
            else:
                exp_pos = math.exp(value)
                features["control_confidence"] = float(exp_pos / (1.0 + exp_pos))
        else:
            features["control_confidence"] = None

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
    non_idle_rows = [row for row in rows if row.get("label") != "idle"]
    control_rows = [row for row in non_idle_rows if bool(row.get("correct"))]
    idle_rows = [row for row in rows if row.get("label") == "idle"]
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
        any_correct = False
        any_idle_selected = False
        first_latency: Optional[float] = None

        for row in sorted_rows:
            decision = gate.update(row)
            selected_freq = decision.get("selected_freq")
            if label == "idle":
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

        if label == "idle":
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

def _calibration_objective(summary: dict[str, float]) -> tuple[float, float, float, float, float]:
    latency = float(summary.get("mean_detection_latency_sec", float("nan")))
    latency_value = 1_000_000.0 if math.isnan(latency) else latency
    return (
        float(summary.get("idle_trial_false_positive", 1.0)),
        float(summary.get("idle_false_positive", 1.0)),
        -float(summary.get("trial_success_rate", 0.0)),
        -float(summary.get("window_end_to_end_recall", 0.0)),
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
            )
            summary = summarize_profile_quality(validation_rows, profile)
            objective = _calibration_objective(summary)
            channel_scores.append(
                {
                    "channel": int(channel),
                    "objective": objective,
                    "summary": summary,
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
) -> tuple[ThresholdProfile, dict[str, Any]]:
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

    win_candidates = calibration_search_win_candidates(active_sec, preferred_win_sec)
    best_profile: Optional[ThresholdProfile] = None
    best_summary: Optional[dict[str, float]] = None
    best_search: Optional[dict[str, Any]] = None
    best_objective: Optional[tuple[float, float, float, float, float, float, int, int]] = None
    split_metadata: Optional[dict[str, Any]] = None

    for win_sec in win_candidates:
        available_windows = calibration_window_count(active_sec, win_sec, step_sec)
        rows = build_feature_rows_from_segments(
            trial_segments,
            available_board_channels=available_board_channels,
            selected_board_channels=selected_channels,
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
        )
        fit_rows, validation_rows, split_meta = split_feature_rows_by_trial(
            rows,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        if not fit_rows or not validation_rows:
            continue
        split_metadata = split_meta
        valid_enter_candidates = [
            int(candidate) for candidate in DEFAULT_MIN_ENTER_CANDIDATES if int(candidate) <= int(available_windows)
        ]
        if not valid_enter_candidates:
            continue
        for min_enter in valid_enter_candidates:
            for min_exit in DEFAULT_MIN_EXIT_CANDIDATES:
                try:
                    candidate_profile = fit_threshold_profile(
                        fit_rows,
                        freqs=freqs,
                        win_sec=win_sec,
                        step_sec=step_sec,
                        min_enter_windows=min_enter,
                        min_exit_windows=min_exit,
                    )
                    summary = summarize_profile_quality(validation_rows, candidate_profile)
                except Exception:
                    continue
                objective = _calibration_objective(summary) + (float(win_sec), int(min_enter), int(min_exit))
                if best_objective is None or objective < best_objective:
                    best_objective = objective
                    best_profile = candidate_profile
                    best_summary = summary
                    best_search = {
                        "win_sec": float(win_sec),
                        "min_enter_windows": int(min_enter),
                        "min_exit_windows": int(min_exit),
                    }

    if best_profile is None or best_summary is None or best_search is None:
        raise RuntimeError("no valid calibration configuration found during validation search")

    all_rows = build_feature_rows_from_segments(
        trial_segments,
        available_board_channels=available_board_channels,
        selected_board_channels=selected_channels,
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=float(best_search["win_sec"]),
        step_sec=step_sec,
    )
    refit_profile = fit_threshold_profile(
        all_rows,
        freqs=freqs,
        win_sec=float(best_search["win_sec"]),
        step_sec=step_sec,
        min_enter_windows=int(best_search["min_enter_windows"]),
        min_exit_windows=int(best_search["min_exit_windows"]),
    )
    refit_summary = summarize_profile_quality(all_rows, refit_profile)
    metadata = {
        "source": "calibration",
        "calibration_seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "selected_eeg_channels": [int(channel) for channel in selected_channels],
        "channel_selection": channel_scores,
        "validation_search": best_search,
        "validation_summary": best_summary,
        "refit_summary": refit_summary,
        "split_metadata": split_metadata,
        "has_stat_model": profile_has_stat_model(refit_profile),
    }
    return replace(
        refit_profile,
        eeg_channels=selected_channels,
        model_name=DEFAULT_MODEL_NAME,
        model_params={"Nh": DEFAULT_NH, "state": {}},
        calibration_split_seed=int(seed),
        benchmark_metrics=None,
        metadata=metadata,
    ), metadata


def _quantile_candidates(values: np.ndarray, quantiles: Sequence[float], *, floor: float = 0.0) -> list[float]:
    if values.size == 0:
        return [floor]
    candidates = [max(float(item), floor) for item in np.quantile(values, quantiles)]
    return [float(value) for value in sorted({round(item, 6) for item in candidates})]


def select_control_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    non_idle_rows = [row for row in rows if row.get("label") != "idle"]
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
) -> ThresholdProfile:
    rows = list(feature_rows)
    control_rows, _control_strategy = select_control_rows(rows)
    idle_rows = [row for row in rows if row.get("label") == "idle"]
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
    best_objective: tuple[float, float, float, float, float] | None = None
    for score_th, ratio_th, margin_th in product(score_grid, ratio_grid, margin_grid):
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
    return ThresholdProfile(
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
    )


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
    grouped: dict[str, list[tuple[TrialSpec, np.ndarray]]] = defaultdict(list)
    for trial, segment in trial_segments:
        key = "idle" if trial.expected_freq is None else f"{float(trial.expected_freq):g}"
        grouped[key].append((trial, segment))
    rng = random.Random(int(seed))
    train: list[tuple[TrialSpec, np.ndarray]] = []
    gate: list[tuple[TrialSpec, np.ndarray]] = []
    evaluation: list[tuple[TrialSpec, np.ndarray]] = []
    for entries in grouped.values():
        block = list(entries)
        rng.shuffle(block)
        count = len(block)
        if count <= 1:
            train.extend(block)
            continue
        if count == 2:
            train.extend(block[:1])
            gate.extend(block[1:])
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
        train.extend(block[:train_count])
        gate.extend(block[train_count : train_count + gate_count])
        evaluation.extend(block[train_count + gate_count :])
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
) -> dict[str, float]:
    gate = AsyncDecisionGate.from_profile(profile)
    idle_selected_windows = 0
    idle_windows = 0
    control_trials = 0
    control_detected_trials = 0
    detection_latencies: list[float] = []
    switch_latencies: list[float] = []
    inference_latencies_ms: list[float] = []
    last_control_freq: Optional[float] = None

    for trial, segment in trial_segments:
        gate.reset()
        segment_matrix = np.asarray(segment, dtype=float)
        if segment_matrix.shape[0] < decoder.win_samples:
            continue
        first_correct_latency: Optional[float] = None
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
            continue
        control_trials += 1
        if first_correct_latency is not None:
            control_detected_trials += 1
            detection_latencies.append(float(first_correct_latency))
            if last_control_freq is not None and abs(last_control_freq - float(trial.expected_freq)) > 1e-8:
                switch_latencies.append(float(first_correct_latency))
        last_control_freq = float(trial.expected_freq)

    idle_minutes = float(idle_windows) * float(decoder.step_sec) / 60.0
    idle_fp_per_min = float(idle_selected_windows / idle_minutes) if idle_minutes > 1e-12 else 0.0
    control_recall = float(control_detected_trials / control_trials) if control_trials else 0.0
    median_detection_latency = (
        float(np.median(np.asarray(detection_latencies, dtype=float))) if detection_latencies else float("inf")
    )
    median_switch_latency = (
        float(np.median(np.asarray(switch_latencies, dtype=float))) if switch_latencies else float("inf")
    )
    itr_bpm = compute_itr_bits_per_minute(
        accuracy=control_recall,
        class_count=len(decoder.freqs),
        decision_time_sec=median_detection_latency if np.isfinite(median_detection_latency) else decoder.win_sec,
    )
    return {
        "idle_fp_per_min": idle_fp_per_min,
        "control_recall": control_recall,
        "switch_latency_s": median_switch_latency,
        "detection_latency_s": median_detection_latency,
        "itr_bpm": itr_bpm,
        "inference_ms": float(np.mean(inference_latencies_ms)) if inference_latencies_ms else float("inf"),
        "control_trials": float(control_trials),
        "idle_windows": float(idle_windows),
    }


def benchmark_rank_key(metrics: dict[str, float]) -> tuple[float, float, float, float, float]:
    return (
        float(metrics.get("idle_fp_per_min", float("inf"))),
        -float(metrics.get("control_recall", 0.0)),
        float(metrics.get("switch_latency_s", float("inf"))),
        -float(metrics.get("itr_bpm", 0.0)),
        float(metrics.get("inference_ms", float("inf"))),
    )


def profile_meets_acceptance(metrics: dict[str, float]) -> bool:
    return (
        float(metrics.get("idle_fp_per_min", float("inf"))) <= 0.5
        and float(metrics.get("control_recall", 0.0)) >= 0.85
        and float(metrics.get("switch_latency_s", float("inf"))) <= 2.8
        and float(metrics.get("inference_ms", float("inf"))) < 40.0
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
    ) -> None:
        self.serial_port = str(serial_port)
        self.board_id = int(board_id)
        self.freqs = tuple(float(freq) for freq in freqs)
        self.output_path = Path(output_path)
        self.prepare_sec = float(prepare_sec)
        self.active_sec = float(active_sec)
        self.rest_sec = float(rest_sec)
        self.target_repeats = int(target_repeats)
        self.idle_repeats = int(idle_repeats)
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
        params = BrainFlowInputParams()
        params.serial_port = self.serial_port
        board = BoardShim(self.board_id, params)
        trial_segments: list[tuple[TrialSpec, np.ndarray]] = []
        failed_trials = 0

        try:
            try:
                board.prepare_session()
            except Exception as exc:
                raise RuntimeError(describe_runtime_error(exc, serial_port=self.serial_port)) from exc
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
        sampling_rate: int = 250,
        prepare_sec: float = 1.0,
        active_sec: float = 4.0,
        rest_sec: float = 1.0,
        calibration_target_repeats: int = 5,
        calibration_idle_repeats: int = 10,
        eval_target_repeats: int = 8,
        eval_idle_repeats: int = 16,
        eval_switch_trials: int = DEFAULT_BENCHMARK_SWITCH_TRIALS,
        step_sec: float = DEFAULT_STEP_SEC,
        model_names: Sequence[str] = DEFAULT_BENCHMARK_MODELS,
        win_candidates: Sequence[float] = DEFAULT_WIN_SEC_CANDIDATES,
        seed: int = DEFAULT_CALIBRATION_SEED,
    ) -> None:
        self.serial_port = str(serial_port)
        self.board_id = int(board_id)
        self.freqs = tuple(float(freq) for freq in freqs)
        self.output_profile_path = Path(output_profile_path)
        self.report_path = Path(report_path) if report_path is not None else None
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
        self.win_candidates = tuple(sorted({float(item) for item in win_candidates if float(item) <= self.active_sec}))
        if not self.win_candidates:
            self.win_candidates = (min(float(DEFAULT_WIN_SEC), float(self.active_sec)),)
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
    ) -> list[tuple[TrialSpec, np.ndarray]]:
        active_samples = int(round(self.active_sec * actual_fs))
        target_min_samples = max(1, int(round(min(self.win_candidates) * actual_fs)))
        segments: list[tuple[TrialSpec, np.ndarray]] = []
        failed_trials = 0
        print(f"{title}: total_trials={len(trials)}", flush=True)
        for index, trial in enumerate(trials, start=1):
            prompt = (
                f"{title} {index}/{len(trials)} focus {trial.label}"
                if trial.expected_freq is not None
                else f"{title} {index}/{len(trials)} idle (look center, avoid flicker targets)"
            )
            self._countdown(prompt, self.prepare_sec)
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
            self._countdown("Rest", self.rest_sec)
            board.get_board_data()
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
        best_objective: Optional[tuple[float, float, float, float, float, float, int, int]] = None

        for win_sec in self.win_candidates:
            available_windows = calibration_window_count(self.active_sec, win_sec, self.step_sec)
            valid_enter = [candidate for candidate in DEFAULT_MIN_ENTER_CANDIDATES if int(candidate) <= available_windows]
            if not valid_enter:
                continue
            for min_enter in valid_enter:
                for min_exit in DEFAULT_MIN_EXIT_CANDIDATES:
                    try:
                        decoder = create_decoder(
                            model_name,
                            sampling_rate=fs,
                            freqs=self.freqs,
                            win_sec=win_sec,
                            step_sec=self.step_sec,
                            model_params={"Nh": DEFAULT_NH},
                        )
                        if decoder.requires_fit:
                            decoder.fit(train_segments)
                        gate_rows = build_feature_rows_with_decoder(decoder, gate_segments)
                        candidate_profile = fit_threshold_profile(
                            gate_rows,
                            freqs=self.freqs,
                            win_sec=win_sec,
                            step_sec=self.step_sec,
                            min_enter_windows=min_enter,
                            min_exit_windows=min_exit,
                        )
                        gate_summary = summarize_profile_quality(gate_rows, candidate_profile)
                    except Exception:
                        continue
                    objective = _calibration_objective(gate_summary) + (float(win_sec), int(min_enter), int(min_exit))
                    if best_objective is None or objective < best_objective:
                        best_objective = objective
                        best_config = {
                            "win_sec": float(win_sec),
                            "min_enter_windows": int(min_enter),
                            "min_exit_windows": int(min_exit),
                            "gate_summary": gate_summary,
                        }

        if best_config is None:
            raise RuntimeError(f"{model_name}: no valid profile candidate found")

        final_decoder = create_decoder(
            model_name,
            sampling_rate=fs,
            freqs=self.freqs,
            win_sec=float(best_config["win_sec"]),
            step_sec=self.step_sec,
            model_params={"Nh": DEFAULT_NH},
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
        )
        eval_metrics = evaluate_decoder_on_trials(final_decoder, final_profile, eval_segments)
        state = json_safe(final_decoder.get_state())
        model_params = dict(final_decoder.model_params)
        model_params["state"] = state
        final_profile = replace(
            final_profile,
            model_name=normalize_model_name(model_name),
            model_params=model_params,
            calibration_split_seed=int(self.seed),
            benchmark_metrics={key: float(value) for key, value in eval_metrics.items() if np.isfinite(value)},
            eeg_channels=tuple(int(channel) for channel in eeg_channels),
            metadata={
                "source": "benchmark",
                "gate_summary": best_config["gate_summary"],
                "selection_search": {
                    "win_sec": float(best_config["win_sec"]),
                    "min_enter_windows": int(best_config["min_enter_windows"]),
                    "min_exit_windows": int(best_config["min_exit_windows"]),
                },
                "has_stat_model": profile_has_stat_model(final_profile),
            },
        )
        return final_profile, {
            "model_name": normalize_model_name(model_name),
            "metrics": eval_metrics,
            "rank_key": benchmark_rank_key(eval_metrics),
            "gate_summary": best_config["gate_summary"],
            "selection_search": {
                "win_sec": float(best_config["win_sec"]),
                "min_enter_windows": int(best_config["min_enter_windows"]),
                "min_exit_windows": int(best_config["min_exit_windows"]),
            },
            "meets_acceptance": profile_meets_acceptance(eval_metrics),
        }

    def run(self) -> dict[str, Any]:
        require_brainflow()
        validate_calibration_plan(
            target_repeats=max(self.calibration_target_repeats, 2),
            idle_repeats=max(self.calibration_idle_repeats, 2),
            active_sec=self.active_sec,
            preferred_win_sec=max(self.win_candidates),
            step_sec=self.step_sec,
        )
        params = BrainFlowInputParams()
        params.serial_port = self.serial_port
        board = BoardShim(self.board_id, params)
        try:
            try:
                board.prepare_session()
            except Exception as exc:
                raise RuntimeError(describe_runtime_error(exc, serial_port=self.serial_port)) from exc
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
            )

            selected_channels, channel_scores = select_auto_eeg_channels(
                calibration_segments,
                available_board_channels=eeg_channels,
                sampling_rate=actual_fs,
                freqs=self.freqs,
                win_sec=max(self.win_candidates),
                step_sec=self.step_sec,
                seed=self.seed,
                validation_fraction=DEFAULT_VALIDATION_FRACTION,
            )
            selected_positions = [list(eeg_channels).index(channel) for channel in selected_channels]
            calibration_subset = _subset_trial_segments_by_positions(calibration_segments, selected_positions)
            eval_subset = _subset_trial_segments_by_positions(eval_segments, selected_positions)
            eval_train, eval_gate, eval_holdout = split_trial_segments_for_benchmark(eval_subset, seed=self.seed)
            train_segments = [*calibration_subset, *eval_train]
            if not eval_gate or not eval_holdout:
                raise RuntimeError(
                    f"benchmark split is invalid: gate={len(eval_gate)}, holdout={len(eval_holdout)}; "
                    f"increase evaluation repeats"
                )

            model_results: list[dict[str, Any]] = []
            best_profiles: dict[str, ThresholdProfile] = {}
            for model_name in self.model_names:
                print(f"Benchmark model: {model_name}", flush=True)
                try:
                    profile, result = self._benchmark_single_model(
                        model_name=model_name,
                        fs=actual_fs,
                        train_segments=train_segments,
                        gate_segments=eval_gate,
                        eval_segments=eval_holdout,
                        eeg_channels=selected_channels,
                    )
                    best_profiles[model_name] = profile
                    model_results.append(result)
                except Exception as exc:
                    model_results.append(
                        {
                            "model_name": model_name,
                            "error": describe_runtime_error(exc, serial_port=self.serial_port),
                            "meets_acceptance": False,
                        }
                    )

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
            save_profile(chosen_profile, self.output_profile_path)
            now_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_report_path = self.report_path
            if output_report_path is None:
                output_report_path = self.output_profile_path.parent / f"benchmark_report_{now_stamp}.json"
            output_report_path.parent.mkdir(parents=True, exist_ok=True)
            report_payload = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "serial_port": self.serial_port,
                "board_id": self.board_id,
                "freqs": list(self.freqs),
                "selected_eeg_channels": [int(channel) for channel in selected_channels],
                "channel_selection": channel_scores,
                "split_counts": {
                    "train_segments": len(train_segments),
                    "gate_segments": len(eval_gate),
                    "holdout_segments": len(eval_holdout),
                },
                "model_results": model_results,
                "chosen_model": chosen_result["model_name"],
                "chosen_profile_path": str(self.output_profile_path),
                "chosen_meets_acceptance": bool(chosen_result.get("meets_acceptance", False)),
            }
            output_report_path.write_text(json_dumps(json_safe(report_payload)) + "\n", encoding="utf-8")
            print(f"Benchmark report saved to: {output_report_path}", flush=True)
            print(f"Best profile saved to: {self.output_profile_path}", flush=True)
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
        self.serial_port = str(serial_port)
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
        self.decoder = create_decoder(
            self.profile.model_name,
            sampling_rate=sampling_rate,
            freqs=self.profile.freqs,
            win_sec=self.profile.win_sec,
            step_sec=self.profile.step_sec,
            model_params=self.profile.model_params,
        )
        state = None
        if isinstance(self.profile.model_params, dict):
            state = self.profile.model_params.get("state")
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
        params = BrainFlowInputParams()
        params.serial_port = self.serial_port
        board = BoardShim(self.board_id, params)
        last_signature: tuple[str, Optional[float]] | None = None

        try:
            try:
                board.prepare_session()
            except Exception as exc:
                raise RuntimeError(describe_runtime_error(exc, serial_port=self.serial_port)) from exc
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
    calibrate.add_argument("--serial-port", type=str, default="COM3")
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

    online = subparsers.add_parser("online", help="run realtime asynchronous decoding")
    online.add_argument("--serial-port", type=str, default="COM3")
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
    benchmark.add_argument("--serial-port", type=str, default="COM3")
    benchmark.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    benchmark.add_argument("--sampling-rate", type=int, default=250)
    benchmark.add_argument("--freqs", type=str, default="8,10,12,15")
    benchmark.add_argument("--output-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    benchmark.add_argument("--report-path", type=Path, default=None)
    benchmark.add_argument("--prepare-sec", type=float, default=1.0)
    benchmark.add_argument("--active-sec", type=float, default=4.0)
    benchmark.add_argument("--rest-sec", type=float, default=1.0)
    benchmark.add_argument("--calibration-target-repeats", type=int, default=5)
    benchmark.add_argument("--calibration-idle-repeats", type=int, default=10)
    benchmark.add_argument("--eval-target-repeats", type=int, default=8)
    benchmark.add_argument("--eval-idle-repeats", type=int, default=16)
    benchmark.add_argument("--eval-switch-trials", type=int, default=DEFAULT_BENCHMARK_SWITCH_TRIALS)
    benchmark.add_argument("--step-sec", type=float, default=DEFAULT_STEP_SEC)
    benchmark.add_argument("--models", type=str, default=",".join(DEFAULT_BENCHMARK_MODELS))
    benchmark.add_argument(
        "--win-candidates",
        type=str,
        default="2.0,2.5,3.0",
        help="comma-separated candidate window lengths in seconds",
    )
    benchmark.add_argument("--seed", type=int, default=DEFAULT_CALIBRATION_SEED)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    freqs = parse_freqs(args.freqs)

    if args.command == "calibrate":
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
        )
        try:
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
        try:
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
        except Exception as exc:
            print(
                f"Benchmark argument error: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        runner = BenchmarkRunner(
            serial_port=args.serial_port,
            board_id=args.board_id,
            freqs=freqs,
            output_profile_path=args.output_profile,
            report_path=args.report_path,
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
            win_candidates=win_candidates,
            seed=args.seed,
        )
        try:
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


