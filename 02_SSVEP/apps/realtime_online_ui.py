from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QCloseEvent, QColor, QFont, QKeyEvent
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from brain_workspace.paths import DATASETS_ROOT, HYBRID_SSVEP_PROFILE_DIR, SSVEP_PROFILE_DIR
from ssvep_core.async_fbcca_idle_standalone import (
    AsyncDecisionGate,
    DEFAULT_BENCHMARK_MODELS,
    DEFAULT_BOARD_ID,
    DEFAULT_CALIBRATION_SEED,
    DEFAULT_COMPUTE_BACKEND_NAME,
    DEFAULT_GPU_CACHE_MODE,
    DEFAULT_GPU_DEVICE_ID,
    DEFAULT_GPU_PRECISION_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_NH,
    DEFAULT_PROFILE_PATH,
    DEFAULT_STREAM_WARMUP_SEC,
    DEFAULT_MAX_TRANSIENT_READ_ERRORS,
    OnlineRunner,
    BoardShim,
    TrialSpec,
    build_calibration_trials,
    describe_runtime_error,
    default_profile,
    ensure_stream_ready,
    format_profile_quality_summary,
    load_decoder_from_profile,
    load_profile,
    normalize_model_name,
    optimize_profile_from_segments,
    normalize_serial_port,
    parse_compute_backend_name,
    parse_freqs,
    parse_gpu_cache_policy,
    parse_gpu_precision,
    prepare_board_session,
    profile_is_default_fallback,
    read_recent_eeg_segment,
    resolve_selected_eeg_channels,
    save_profile,
    validate_calibration_plan,
)
from ssvep_core.dataset import save_collection_dataset_bundle
from ssvep_core.fast_fbcca_pretrain import (
    DEFAULT_FAST_FBCCA_ACTIVE_SEC,
    DEFAULT_FAST_FBCCA_IDLE_REPEATS,
    DEFAULT_FAST_FBCCA_PREPARE_SEC,
    DEFAULT_FAST_FBCCA_REST_SEC,
    DEFAULT_FAST_FBCCA_STEP_SEC,
    DEFAULT_FAST_FBCCA_TARGET_REPEATS,
    DEFAULT_FAST_FBCCA_TEMPLATE_WEIGHT,
    DEFAULT_FAST_FBCCA_TEMPLATE_WIN_SEC,
    DEFAULT_FAST_FBCCA_WIN_SEC,
    DEFAULT_FBCCA_BASE_PROFILE_PATH,
    FastFBCCAPretrainConfig,
    build_fast_fbcca_history_profile_path,
    run_fast_fbcca_personalization,
    save_fast_fbcca_profile_bundle,
)
from ssvep_core.runtime_shadow import build_shadow_runtime_chain
from ssvep_core.stimulus_profiles import (
    DEFAULT_STIMULUS_PROFILE_ID,
    get_stimulus_profile,
    select_stimulus_mode_for_profile,
)
from apps.async_fbcca_validation_ui import (
    FourArrowStimWidget,
    PHASE_CAL_ACTIVE,
    PHASE_CAL_PREPARE,
    PHASE_CAL_REST,
    PHASE_ERROR,
    PHASE_IDLE,
    PHASE_STOPPED,
    PHASE_VALIDATION,
    validate_stimulus_mode,
    direction_label,
)


THIS_DIR = Path(__file__).resolve().parent
REALTIME_FBCCA_PROFILE_CANDIDATES = (
    SSVEP_PROFILE_DIR / "fbcca_profile.json",
    SSVEP_PROFILE_DIR / "fbcca_base_profile.json",
    SSVEP_PROFILE_DIR / "default_profile.json",
)
DEMO_EXPECTED_FREQS = (8.0, 10.0, 12.0, 15.0)
SSVEP_FBCCA_BASE_PROFILE_PATH = SSVEP_PROFILE_DIR / "fbcca_base_profile.json"
SSVEP_REALTIME_PROFILE_PATH = SSVEP_PROFILE_DIR / "fbcca_profile.json"
SSVEP_REALTIME_PROFILE_V2_PATH = SSVEP_PROFILE_DIR / "fbcca_profile_v2.json"
SSVEP_NO_TRAIN_FBCCA_PROFILE_PATH = SSVEP_PROFILE_DIR / "fbcca_no_train_profile.json"
HYBRID_PROFILE_DIR = HYBRID_SSVEP_PROFILE_DIR
HYBRID_CURRENT_PROFILE_PATH = HYBRID_PROFILE_DIR / "current_fbcca_profile.json"
ENABLE_HYBRID_PROFILE_PUBLISH = False


def _profile_model_name(path: Path) -> str:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    return normalize_model_name(str(payload.get("model_name", "")))


def resolve_default_realtime_profile_path() -> Path:
    for candidate in REALTIME_FBCCA_PROFILE_CANDIDATES:
        if candidate.exists() and "fbcca" in _profile_model_name(candidate):
            return candidate
    if "fbcca" in _profile_model_name(DEFAULT_PROFILE_PATH):
        return DEFAULT_PROFILE_PATH
    return DEFAULT_PROFILE_PATH


def build_no_train_fbcca_profile(freqs: Sequence[float] = DEMO_EXPECTED_FREQS) -> Any:
    freq_tuple = tuple(float(item) for item in freqs)
    if len(freq_tuple) != 4:
        raise ValueError("no-train FBCCA profile requires exactly 4 frequencies")
    profile = default_profile(freq_tuple)
    model_params = {
        "Nh": int(DEFAULT_NH),
        "fbcca_variant": "fbcca_fixed_all8",
        "_decoder_model_name": "fbcca_fixed_all8",
        "subband_weight_mode": "chen_fixed",
    }
    return replace(
        profile,
        model_name=DEFAULT_MODEL_NAME,
        model_params=model_params,
        metadata={
            "source": "no_train_fbcca_direct",
            "training_required": False,
            "fast_pretrain": {
                "status": "not_used",
                "template_enabled": False,
                "gate_calibration_enabled": False,
            },
        },
        recommended_for_realtime=True,
    )


def save_no_train_fbcca_profile(
    path: Path | None = None,
    *,
    freqs: Sequence[float] = DEMO_EXPECTED_FREQS,
) -> tuple[Path, Path]:
    resolved_path = SSVEP_NO_TRAIN_FBCCA_PROFILE_PATH if path is None else Path(path)
    return save_fast_fbcca_profile_bundle(build_no_train_fbcca_profile(freqs), resolved_path, {})


DEFAULT_REALTIME_PROFILE_PATH = resolve_default_realtime_profile_path()
MODEL_OPTIONS = (DEFAULT_MODEL_NAME,) + tuple(item for item in DEFAULT_BENCHMARK_MODELS if item != DEFAULT_MODEL_NAME)
_DEFAULT_REALTIME_STIMULUS_PROFILE = get_stimulus_profile(DEFAULT_STIMULUS_PROFILE_ID)
DEFAULT_REALTIME_STIMULUS_PROFILE_ID = DEFAULT_STIMULUS_PROFILE_ID
DEFAULT_STIM_REFRESH_RATE_HZ = float(_DEFAULT_REALTIME_STIMULUS_PROFILE.refresh_rate_hz)
DEFAULT_STIM_MEAN = float(_DEFAULT_REALTIME_STIMULUS_PROFILE.mean)
DEFAULT_STIM_AMP = float(_DEFAULT_REALTIME_STIMULUS_PROFILE.amp)
DEFAULT_STIM_PHI = float(_DEFAULT_REALTIME_STIMULUS_PROFILE.phi)
DEFAULT_STIM_RAMP_SEC = float(_DEFAULT_REALTIME_STIMULUS_PROFILE.ramp_sec)
DEFAULT_FAST_CONTROL_PRETRAIN_DATASET_DIR = DATASETS_ROOT
DEFAULT_PRETRAIN_PREPARE_SEC = DEFAULT_FAST_FBCCA_PREPARE_SEC
DEFAULT_PRETRAIN_ACTIVE_SEC = DEFAULT_FAST_FBCCA_ACTIVE_SEC
DEFAULT_PRETRAIN_REST_SEC = DEFAULT_FAST_FBCCA_REST_SEC
DEFAULT_PRETRAIN_TARGET_REPEATS = DEFAULT_FAST_FBCCA_TARGET_REPEATS
DEFAULT_PRETRAIN_IDLE_REPEATS = DEFAULT_FAST_FBCCA_IDLE_REPEATS
DEFAULT_PRETRAIN_WIN_SEC = DEFAULT_FAST_FBCCA_WIN_SEC
DEFAULT_PRETRAIN_STEP_SEC = DEFAULT_FAST_FBCCA_STEP_SEC
DEFAULT_FULL_PRETRAIN_PREPARE_SEC = 1.0
DEFAULT_FULL_PRETRAIN_ACTIVE_SEC = 4.0
DEFAULT_FULL_PRETRAIN_REST_SEC = 1.0
DEFAULT_FULL_PRETRAIN_TARGET_REPEATS = 8
DEFAULT_FULL_PRETRAIN_IDLE_REPEATS = 16
DEFAULT_FULL_PRETRAIN_WIN_SEC = 3.0
DEFAULT_FULL_PRETRAIN_STEP_SEC = 0.5
REALTIME_STIMULUS_PHASE_APPLY_TIMEOUT_SEC = 1.0
REALTIME_CONTROL_PANEL_WIDTH = 440
REALTIME_STIM_MIN_WIDTH = 760
REALTIME_STIM_MIN_HEIGHT = 560
REALTIME_SELECTED_BORDER_COLOR = QColor(80, 170, 255)
REALTIME_PRETRAIN_HISTORY_DIR = SSVEP_PROFILE_DIR / "pretrain_history"


@dataclass(frozen=True)
class RealtimeConfig:
    serial_port: str
    board_id: int
    freqs: tuple[float, float, float, float]
    profile_path: Path
    model_name: str
    compute_backend: str
    gpu_device: int
    gpu_precision: str
    gpu_warmup: bool
    gpu_cache_policy: str
    shadow_mode: bool = True
    stimulus_profile_id: str = DEFAULT_REALTIME_STIMULUS_PROFILE_ID
    stimulus_mode: str = ""
    stim_refresh_rate_hz: float = DEFAULT_STIM_REFRESH_RATE_HZ


@dataclass(frozen=True)
class RealtimePretrainConfig:
    serial_port: str
    board_id: int
    freqs: tuple[float, float, float, float]
    base_profile_path: Path
    fallback_profile_path: Path
    output_profile_path: Path
    history_profile_path: Path
    compute_backend: str
    gpu_device: int
    gpu_precision: str
    gpu_warmup: bool
    gpu_cache_policy: str
    prepare_sec: float = DEFAULT_PRETRAIN_PREPARE_SEC
    active_sec: float = DEFAULT_PRETRAIN_ACTIVE_SEC
    rest_sec: float = DEFAULT_PRETRAIN_REST_SEC
    target_repeats: int = DEFAULT_PRETRAIN_TARGET_REPEATS
    idle_repeats: int = DEFAULT_PRETRAIN_IDLE_REPEATS
    win_sec: float = DEFAULT_PRETRAIN_WIN_SEC
    step_sec: float = DEFAULT_PRETRAIN_STEP_SEC
    mode: str = "fast"
    template_weight: float = DEFAULT_FAST_FBCCA_TEMPLATE_WEIGHT
    template_win_sec: float = DEFAULT_FAST_FBCCA_TEMPLATE_WIN_SEC
    stimulus_profile_id: str = DEFAULT_REALTIME_STIMULUS_PROFILE_ID
    stim_refresh_rate_hz: float = DEFAULT_STIM_REFRESH_RATE_HZ
    dataset_dir: Path = DEFAULT_FAST_CONTROL_PRETRAIN_DATASET_DIR


def pretrain_trial_count(config: RealtimePretrainConfig) -> int:
    return 4 * int(config.target_repeats) + int(config.idle_repeats)


def pretrain_estimated_seconds(config: RealtimePretrainConfig) -> float:
    trial_sec = float(config.prepare_sec) + float(config.active_sec) + float(config.rest_sec)
    return float(pretrain_trial_count(config)) * trial_sec


def build_realtime_pretrain_dataset_session_id(*, timestamp: float | None = None) -> str:
    return f"fast_control_pretrain_{_now_stamp(timestamp=timestamp)}"


def build_pretrain_profile_path(*, timestamp: float | None = None) -> Path:
    return build_fast_fbcca_history_profile_path(timestamp=timestamp)


def _now_stamp(*, timestamp: float | None = None) -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time() if timestamp is None else float(timestamp)))


def resolve_realtime_model_choice(selected_model: str, profile_model: str) -> tuple[str, bool]:
    selected = normalize_model_name(selected_model)
    profile = normalize_model_name(profile_model)
    return profile, selected != profile


def resolve_realtime_stimulus_mode(*, stimulus_profile_id: str, refresh_rate_hz: float) -> tuple[str, str]:
    mode, reason = select_stimulus_mode_for_profile(
        stimulus_profile_id,
        refresh_rate_hz=float(refresh_rate_hz),
        requested_mode="auto",
    )
    return validate_stimulus_mode(mode), str(reason)


def realtime_pretrain_protocol_config(config: RealtimePretrainConfig, *, saved_trial_count: int) -> dict[str, Any]:
    profile = get_stimulus_profile(config.stimulus_profile_id)
    stim_mode, mode_reason = resolve_realtime_stimulus_mode(
        stimulus_profile_id=config.stimulus_profile_id,
        refresh_rate_hz=float(config.stim_refresh_rate_hz),
    )
    lum_min = max(0.0, float(profile.mean) - abs(float(profile.amp)))
    lum_max = min(1.0, float(profile.mean) + abs(float(profile.amp)))
    denom = lum_min + lum_max
    return {
        "collection_aborted": False,
        "requested_session_id": build_realtime_pretrain_dataset_session_id(),
        "protocol_name": "fast-control-pretrain-v1",
        "planned_total_trials": int(pretrain_trial_count(config)),
        "saved_trial_count": int(saved_trial_count),
        "prepare_sec": float(config.prepare_sec),
        "active_sec": float(config.active_sec),
        "rest_sec": float(config.rest_sec),
        "long_idle_sec": 0.0,
        "target_repeats": int(config.target_repeats),
        "idle_repeats": int(config.idle_repeats),
        "switch_trials": 0,
        "stimulus_profile_id": str(profile.profile_id),
        "stimulus_mode": str(stim_mode),
        "stimulus_mode_selection_reason": str(mode_reason),
        "stimulus_backend": "pyqt_embedded_realtime",
        "stim_refresh_rate_hz": float(config.stim_refresh_rate_hz),
        "stim_mean": float(profile.mean),
        "stim_amp": float(profile.amp),
        "stim_phi": float(profile.phi),
        "stim_luminance_min": float(lum_min),
        "stim_luminance_max": float(lum_max),
        "stim_michelson_contrast": float((lum_max - lum_min) / denom) if denom > 1e-12 else 0.0,
        "ramp_sec": float(profile.ramp_sec),
        "ramp_included_in_saved_window": bool(float(profile.ramp_sec) > 0.0),
        "frame_interval_stats": {},
        "comfort_rating": None,
        "screen_brightness_note": str(profile.screen_brightness_note),
        "sync_stimulus_phase": True,
        "active_saved_window": "last_active_sec_after_first_frame_ack",
    }


def validate_fbcca_demo_profile_path(source: Path | str, *, expected_freqs: Sequence[float] = DEMO_EXPECTED_FREQS) -> dict[str, Any]:
    path = Path(source).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("profile JSON is not an object")
    model_name = normalize_model_name(str(payload.get("model_name", "")))
    if model_name != DEFAULT_MODEL_NAME:
        raise ValueError(f"demo profile must use model_name='fbcca'; got {model_name or '<missing>'}")
    freqs = tuple(float(item) for item in payload.get("freqs", ()))
    expected = tuple(float(item) for item in expected_freqs)
    if len(freqs) != len(expected) or any(abs(left - right) > 1e-6 for left, right in zip(freqs, expected)):
        raise ValueError(f"demo profile freqs must be {expected}; got {freqs}")
    return dict(payload)


def _profile_v2_sibling(source: Path) -> Path:
    return source.with_name(f"{source.stem}_v2.json")


def publish_profile_to_ssvep_realtime(source: Path | str) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    validate_fbcca_demo_profile_path(source_path)
    SSVEP_REALTIME_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != SSVEP_REALTIME_PROFILE_PATH.resolve():
        shutil.copy2(source_path, SSVEP_REALTIME_PROFILE_PATH)
    source_v2 = _profile_v2_sibling(source_path)
    copied_v2 = False
    if source_v2.exists():
        if source_v2.resolve() != SSVEP_REALTIME_PROFILE_V2_PATH.resolve():
            shutil.copy2(source_v2, SSVEP_REALTIME_PROFILE_V2_PATH)
        copied_v2 = True
    return {
        "source": str(source_path),
        "profile_path": str(SSVEP_REALTIME_PROFILE_PATH),
        "profile_v2_path": str(SSVEP_REALTIME_PROFILE_V2_PATH) if copied_v2 else "",
        "copied_v2": bool(copied_v2),
    }


def publish_profile_to_hybrid_controller(source: Path | str, *, timestamp: float | None = None) -> dict[str, Any]:
    if not ENABLE_HYBRID_PROFILE_PUBLISH:
        raise RuntimeError("publishing FBCCA demo profiles to hybrid_controller is disabled in this stage")
    source_path = Path(source).expanduser().resolve()
    validate_fbcca_demo_profile_path(source_path)
    HYBRID_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    history_path = HYBRID_PROFILE_DIR / f"ssvep_fbcca_profile_{_now_stamp(timestamp=timestamp)}.json"
    if source_path.resolve() != history_path.resolve():
        shutil.copy2(source_path, history_path)
    if source_path.resolve() != HYBRID_CURRENT_PROFILE_PATH.resolve():
        shutil.copy2(source_path, HYBRID_CURRENT_PROFILE_PATH)
    return {
        "source": str(source_path),
        "current_profile_path": str(HYBRID_CURRENT_PROFILE_PATH),
        "history_profile_path": str(history_path),
    }


def _weight_vector_summary(values: Optional[Sequence[float]]) -> str:
    if values is None:
        return "none"
    items = [float(value) for value in values]
    if not items:
        return "none"
    mean_value = float(sum(items) / max(len(items), 1))
    return (
        f"len={len(items)} min={min(items):.4f} max={max(items):.4f} "
        f"mean={mean_value:.4f} values={[round(value, 4) for value in items]}"
    )


def profile_runtime_summary(profile: Any, backend_summary: Optional[dict[str, Any]] = None) -> str:
    backend = dict(backend_summary or {})
    return (
        "profile summary | "
        f"model={profile.model_name} | "
        f"channel_mode={profile.channel_weight_mode} | "
        f"channel_weights={_weight_vector_summary(profile.channel_weights)} | "
        f"subband_mode={profile.subband_weight_mode}(global) | "
        f"subband_weights={_weight_vector_summary(profile.subband_weights)} | "
        f"spatial={profile.spatial_filter_mode}/rank={profile.spatial_filter_rank} | "
        f"backend={backend.get('used_backend', backend.get('requested_backend', 'unknown'))}"
    )


def _backend_total_ms(summary: Optional[dict[str, Any]]) -> float:
    payload = dict(summary or {})
    kernel = dict(payload.get("kernel_benchmark", {}))
    if kernel:
        total = float(kernel.get("total_ms", 0.0) or 0.0)
        if np.isfinite(total) and total > 0.0:
            return float(total)
    total = 0.0
    for key in ("host_to_device_ms", "preprocess_ms", "score_ms", "device_to_host_ms", "synchronize_ms"):
        value = float(payload.get(key, 0.0) or 0.0)
        if np.isfinite(value):
            total += float(value)
    return float(total)


def _choose_runtime_backend(
    *,
    profile: Any,
    sampling_rate: int,
    sample_window: np.ndarray,
    requested_backend: str,
    gpu_device: int,
    gpu_precision: str,
    gpu_warmup: bool,
    gpu_cache_policy: str,
) -> tuple[str, dict[str, Any]]:
    requested = parse_compute_backend_name(requested_backend)
    if requested != "auto":
        return requested, {
            "selection_mode": "explicit",
            "requested_backend": requested,
            "used_backend": requested,
        }

    comparison: dict[str, Any] = {
        "selection_mode": "auto-benchmark",
        "requested_backend": requested,
        "candidates": {},
    }
    cpu_decoder = load_decoder_from_profile(
        profile,
        sampling_rate=int(sampling_rate),
        compute_backend="cpu",
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=bool(gpu_warmup),
        gpu_cache_policy=str(gpu_cache_policy),
    )
    cpu_summary = cpu_decoder.run_backend_microbenchmark(sample_window=np.asarray(sample_window, dtype=np.float64), repeats=2)
    comparison["candidates"]["cpu"] = dict(cpu_summary)
    chosen = "cpu"
    chosen_reason = "cpu-baseline"
    try:
        cuda_decoder = load_decoder_from_profile(
            profile,
            sampling_rate=int(sampling_rate),
            compute_backend="cuda",
            gpu_device=int(gpu_device),
            gpu_precision=str(gpu_precision),
            gpu_warmup=bool(gpu_warmup),
            gpu_cache_policy=str(gpu_cache_policy),
        )
        cuda_summary = cuda_decoder.run_backend_microbenchmark(
            sample_window=np.asarray(sample_window, dtype=np.float64),
            repeats=2,
        )
        comparison["candidates"]["cuda"] = dict(cuda_summary)
        cpu_total = _backend_total_ms(cpu_summary)
        cuda_total = _backend_total_ms(cuda_summary)
        if np.isfinite(cuda_total) and cuda_total > 0.0 and cuda_total < cpu_total:
            chosen = "cuda"
            chosen_reason = "cuda-faster"
        else:
            chosen_reason = "cpu-faster-or-equal"
    except Exception as exc:
        comparison["candidates"]["cuda"] = {"error": str(exc)}
        chosen_reason = "cuda-unavailable-or-slower"
    comparison["used_backend"] = chosen
    comparison["reason"] = chosen_reason
    return chosen, comparison


def _validate_loaded_profile(
    profile: Any,
    decoder: Any,
    *,
    eeg_channels: Sequence[int],
) -> dict[str, Any]:
    channel_weights = profile.channel_weights
    if channel_weights is None and hasattr(decoder, "get_channel_weights"):
        channel_weights = decoder.get_channel_weights()
    channel_weight_count = 0 if channel_weights is None else len(channel_weights)
    if channel_weights is not None and int(channel_weight_count) != int(len(eeg_channels)):
        raise RuntimeError(
            f"profile channel_weights mismatch: weights={channel_weight_count} channels={len(eeg_channels)}"
        )
    subband_weights = profile.subband_weights
    if subband_weights is None and hasattr(decoder, "engine") and hasattr(decoder.engine, "get_subband_weights"):
        resolved_subbands = decoder.engine.get_subband_weights()
        if resolved_subbands is not None:
            subband_weights = tuple(float(value) for value in resolved_subbands)
    subband_count = 0 if subband_weights is None else len(subband_weights)
    if subband_weights is not None and hasattr(decoder, "engine") and hasattr(decoder.engine, "subband_sos"):
        expected_subband_count = len(getattr(decoder.engine, "subband_sos", []) or [])
        if expected_subband_count and int(subband_count) != int(expected_subband_count):
            raise RuntimeError(
                f"profile subband_weights mismatch: weights={subband_count} subbands={expected_subband_count}"
            )
    return {
        "loaded_profile_model": str(profile.model_name),
        "channel_weight_count": int(channel_weight_count),
        "subband_weight_count": int(subband_count),
    }


def _suggest_refresh_rate_hz() -> float:
    app = QApplication.instance()
    if app is None:
        return float(DEFAULT_STIM_REFRESH_RATE_HZ)
    screen = app.primaryScreen()
    if screen is None:
        return float(DEFAULT_STIM_REFRESH_RATE_HZ)
    try:
        hz = float(screen.refreshRate())
    except Exception:
        return float(DEFAULT_STIM_REFRESH_RATE_HZ)
    if not np.isfinite(hz) or hz <= 1.0:
        return float(DEFAULT_STIM_REFRESH_RATE_HZ)
    return float(hz)


def _read_probe_window(board: Any, *, sampling_rate: int, profile_win_sec: float) -> tuple[int, int, np.ndarray]:
    fs = int(sampling_rate)
    probe_samples = max(int(round(float(profile_win_sec) * fs)), 1)
    wait_sec = max(float(profile_win_sec), DEFAULT_STREAM_WARMUP_SEC, 0.1)
    timeout_sec = max(3.0, wait_sec + DEFAULT_STREAM_WARMUP_SEC + 1.0)
    ready = int(ensure_stream_ready(board, fs, minimum_sec=wait_sec, timeout_sec=timeout_sec))
    sample_matrix = board.get_current_board_data(max(probe_samples, ready))
    available = int(sample_matrix.shape[1])
    if available < probe_samples:
        raise RuntimeError(
            "buffered probe window is too short after warmup: "
            f"{available}/{probe_samples} samples; "
            f"need {float(profile_win_sec):g}s at {fs}Hz before realtime decoder starts"
        )
    return ready, probe_samples, sample_matrix


class DeviceCheckWorker(QObject):
    connected = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, *, serial_port: str, board_id: int) -> None:
        super().__init__()
        self.serial_port = normalize_serial_port(serial_port)
        self.board_id = int(board_id)

    @pyqtSlot()
    def run(self) -> None:
        board = None
        try:
            board, resolved_port, attempted_ports = prepare_board_session(self.board_id, self.serial_port)
            fs = int(BoardShim.get_sampling_rate(self.board_id))
            board.start_stream(450000)
            ready = int(ensure_stream_ready(board, fs))
            self.connected.emit(
                {
                    "requested_serial_port": self.serial_port,
                    "resolved_serial_port": resolved_port,
                    "attempted_ports": attempted_ports,
                    "sampling_rate": fs,
                    "ready_samples": ready,
                }
            )
        except Exception as exc:
            self.error.emit(f"连接失败：{describe_runtime_error(exc, serial_port=self.serial_port)}")
        finally:
            if board is not None:
                try:
                    board.stop_stream()
                except Exception:
                    pass
                try:
                    board.release_session()
                except Exception:
                    pass
            self.finished.emit()


class RealtimePretrainWorker(QObject):
    phase_changed = pyqtSignal(object)
    log = pyqtSignal(str)
    profile_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: RealtimePretrainConfig) -> None:
        super().__init__()
        self.config = config
        self._stop_event = threading.Event()
        self._stimulus_phase_applied_event = threading.Event()
        self._last_stimulus_phase_payload: dict[str, Any] = {}

    def request_stop(self) -> None:
        self._stop_event.set()
        self._stimulus_phase_applied_event.set()

    def notify_stimulus_phase_presented(self, payload: Optional[dict[str, Any]] = None) -> None:
        if payload is None or str(payload.get("mode", "")) == PHASE_CAL_ACTIVE:
            self._last_stimulus_phase_payload = dict(payload or {})
            self._stimulus_phase_applied_event.set()

    def _emit_stopped_phase(self) -> None:
        self.phase_changed.emit(
            {
                "mode": PHASE_STOPPED,
                "title": "预训练已停止",
                "detail": "用户停止了预训练流程。",
                "flicker": False,
                "cue_freq": None,
            }
        )

    def _wait_phase(self, payload: dict[str, Any], duration_sec: float) -> bool:
        if duration_sec <= 0:
            self.phase_changed.emit({**payload, "remaining_sec": 0})
            return not self._stop_event.is_set()
        deadline = time.perf_counter() + float(duration_sec)
        last_sec: Optional[int] = None
        while not self._stop_event.is_set():
            remaining = max(0.0, deadline - time.perf_counter())
            rem_sec = int(math.ceil(remaining))
            if rem_sec != last_sec:
                self.phase_changed.emit({**payload, "remaining_sec": rem_sec})
                last_sec = rem_sec
            if remaining <= 0:
                break
            self._stop_event.wait(min(0.05, remaining))
        return not self._stop_event.is_set()

    def _wait_for_stimulus_phase_presented(self) -> bool:
        if self._stop_event.is_set():
            return False
        ready = self._stimulus_phase_applied_event.wait(REALTIME_STIMULUS_PHASE_APPLY_TIMEOUT_SEC)
        if not ready:
            self.log.emit("pretrain stimulus first-frame acknowledgement timed out")
        return bool(ready)

    @pyqtSlot()
    def run(self) -> None:
        board = None
        active_serial = self.config.serial_port
        try:
            validate_calibration_plan(
                target_repeats=int(self.config.target_repeats),
                idle_repeats=int(self.config.idle_repeats),
                active_sec=float(self.config.active_sec),
                preferred_win_sec=float(self.config.win_sec),
                step_sec=float(self.config.step_sec),
            )
            board, resolved_port, attempted = prepare_board_session(self.config.board_id, self.config.serial_port)
            active_serial = resolved_port
            self.log.emit(
                f"预训练设备已连接：requested={self.config.serial_port} -> {resolved_port}; attempts={attempted}"
            )
            fs = int(BoardShim.get_sampling_rate(self.config.board_id))
            eeg_channels = tuple(int(ch) for ch in BoardShim.get_eeg_channels(self.config.board_id))
            active_samples = int(round(float(self.config.active_sec) * fs))
            min_samples = int(round(float(self.config.win_sec) * fs))
            if active_samples < min_samples:
                raise ValueError("active_sec must be at least win_sec")

            board.start_stream(450000)
            ready = ensure_stream_ready(board, fs)
            self.log.emit(f"stream ready | fs={fs}Hz | channels={list(eeg_channels)} | buffer={ready}")
            if self._stop_event.wait(max(2.0, DEFAULT_STREAM_WARMUP_SEC)):
                self._emit_stopped_phase()
                return
            board.get_board_data()

            trials = build_calibration_trials(
                self.config.freqs,
                target_repeats=int(self.config.target_repeats),
                idle_repeats=int(self.config.idle_repeats),
                shuffle=True,
                seed=DEFAULT_CALIBRATION_SEED,
            )
            segments: list[tuple[TrialSpec, np.ndarray]] = []
            quality_rows: list[dict[str, Any]] = []
            total = len(trials)
            for index, trial in enumerate(trials, start=1):
                prompt = (
                    f"注视 {direction_label(self.config.freqs, trial.expected_freq)}"
                    if trial.expected_freq is not None
                    else "看中心点，避免注视任一闪烁目标"
                )
                if not self._wait_phase(
                    {
                        "mode": PHASE_CAL_PREPARE,
                        "title": f"预训练 {index}/{total}",
                        "detail": f"{prompt} | 准备",
                        "flicker": False,
                        "cue_freq": trial.expected_freq,
                        "active_sec": float(self.config.active_sec),
                        "stimulus_profile_id": str(self.config.stimulus_profile_id),
                    },
                    float(self.config.prepare_sec),
                ):
                    self._emit_stopped_phase()
                    return

                board.get_board_data()
                active_payload = {
                    "mode": PHASE_CAL_ACTIVE,
                    "title": f"预训练 {index}/{total}",
                    "detail": f"{prompt} | 采集",
                    "flicker": True,
                    "cue_freq": trial.expected_freq,
                    "active_sec": float(self.config.active_sec),
                    "stimulus_profile_id": str(self.config.stimulus_profile_id),
                }
                self._stimulus_phase_applied_event.clear()
                self._last_stimulus_phase_payload = {}
                self.phase_changed.emit(
                    {**active_payload, "remaining_sec": int(math.ceil(float(self.config.active_sec)))}
                )
                stimulus_ready = self._wait_for_stimulus_phase_presented()
                if self._stop_event.is_set():
                    self._emit_stopped_phase()
                    return
                if not stimulus_ready:
                    raise RuntimeError("未收到预训练刺激首帧确认，已停止预训练以避免采集起点不可信")
                board.get_board_data()
                if self._last_stimulus_phase_payload:
                    payload = dict(self._last_stimulus_phase_payload)
                    self.log.emit(
                        "pretrain stimulus first-frame acknowledged | "
                        f"trial={index}/{total} | "
                        f"frame={payload.get('frame_index', 'unknown')} | "
                        f"t={payload.get('presented_t_sec', 'unknown')}"
                    )
                ack_payload = dict(self._last_stimulus_phase_payload)
                if not self._wait_phase(active_payload, float(self.config.active_sec)):
                    self._emit_stopped_phase()
                    return

                segment, _, _ = read_recent_eeg_segment(
                    board,
                    eeg_channels,
                    target_samples=active_samples,
                    minimum_samples=min_samples,
                )
                segments.append((trial, segment))
                quality_rows.append(
                    {
                        "order_index": int(index - 1),
                        "target_samples": int(active_samples),
                        "used_samples": int(np.asarray(segment).shape[0]),
                        "active_sec": float(self.config.active_sec),
                        "sample_ratio": float(np.asarray(segment).shape[0] / max(active_samples, 1)),
                        "shortfall_ratio": float(max(active_samples - int(np.asarray(segment).shape[0]), 0) / max(active_samples, 1)),
                        "retry_count": 0,
                        "stimulus_first_frame_presented_t_sec": ack_payload.get("presented_t_sec"),
                        "stimulus_first_frame_frame_index": ack_payload.get("frame_index"),
                        "stimulus_first_frame_mode": str(ack_payload.get("mode", "")),
                        "stimulus_first_frame_ack_timed_out": False,
                        "stimulus_profile_id": str(self.config.stimulus_profile_id),
                        "stimulus_frame_interval_stats": dict(ack_payload.get("frame_interval_stats", {}) or {}),
                    }
                )
                self.log.emit(f"trial {index}/{total} done: {trial.label}")

                if not self._wait_phase(
                    {
                        "mode": PHASE_CAL_REST,
                        "title": f"预训练 {index}/{total}",
                        "detail": "休息",
                        "flicker": False,
                        "cue_freq": None,
                    },
                    float(self.config.rest_sec),
                ):
                    self._emit_stopped_phase()
                    return
            board.get_board_data()

            dataset_payload: dict[str, Any] = {}
            dataset_save_valid = False
            dataset_save_error = ""
            try:
                protocol_config = realtime_pretrain_protocol_config(self.config, saved_trial_count=len(segments))
                dataset_payload = save_collection_dataset_bundle(
                    dataset_root=Path(self.config.dataset_dir),
                    session_id=str(protocol_config["requested_session_id"]),
                    subject_id="realtime_pretrain",
                    serial_port=active_serial,
                    board_id=int(self.config.board_id),
                    sampling_rate=int(fs),
                    freqs=self.config.freqs,
                    board_eeg_channels=eeg_channels,
                    protocol_config=protocol_config,
                    trial_segments=segments,
                    quality_rows=quality_rows,
                )
                dataset_save_valid = True
                self.log.emit(f"pretrain dataset saved | {dataset_payload.get('dataset_manifest', '')}")
            except Exception as exc:
                dataset_save_error = str(exc)
                self.log.emit(f"warning: pretrain dataset save failed; profile will be marked dataset_save_valid=0: {exc}")

            try:
                if str(self.config.mode or "fast").strip().lower() == "full":
                    profile, metadata = optimize_profile_from_segments(
                        segments,
                        available_board_channels=eeg_channels,
                        sampling_rate=fs,
                        freqs=self.config.freqs,
                        active_sec=float(self.config.active_sec),
                        preferred_win_sec=float(self.config.win_sec),
                        step_sec=float(self.config.step_sec),
                        seed=DEFAULT_CALIBRATION_SEED,
                        compute_backend=self.config.compute_backend,
                        gpu_device=int(self.config.gpu_device),
                        gpu_precision=self.config.gpu_precision,
                        gpu_warmup=bool(self.config.gpu_warmup),
                        gpu_cache_policy=self.config.gpu_cache_policy,
                    )
                else:
                    profile, fast_result = run_fast_fbcca_personalization(
                        FastFBCCAPretrainConfig(
                            base_profile_path=self.config.base_profile_path,
                            fallback_profile_path=self.config.fallback_profile_path,
                            output_profile_path=self.config.output_profile_path,
                            history_profile_path=self.config.history_profile_path,
                            freqs=self.config.freqs,
                            win_sec=float(self.config.win_sec),
                            step_sec=float(self.config.step_sec),
                            template_weight=float(self.config.template_weight),
                            template_win_sec=float(self.config.template_win_sec),
                            seed=DEFAULT_CALIBRATION_SEED,
                            compute_backend=self.config.compute_backend,
                            gpu_device=int(self.config.gpu_device),
                            gpu_precision=self.config.gpu_precision,
                            gpu_warmup=bool(self.config.gpu_warmup),
                            gpu_cache_policy=self.config.gpu_cache_policy,
                        ),
                        trial_segments=segments,
                        sampling_rate=fs,
                        available_board_channels=eeg_channels,
                        collection_duration_sec=pretrain_estimated_seconds(self.config),
                        log_fn=self.log.emit,
                    )
                    metadata = {
                        "validation_summary": dict(fast_result.get("quality_summary") or fast_result.get("quality_metrics") or {}),
                        "selected_eeg_channels": list(fast_result.get("selected_eeg_channels", [])),
                        "fast_pretrain_result": dict(fast_result),
                    }
            except Exception as exc:
                raise RuntimeError(
                    "预训练 profile 拟合失败："
                    f"usable_trials={len(segments)}/{total}. "
                    f"{describe_runtime_error(exc, serial_port=active_serial)}"
                ) from exc
            if self._stop_event.is_set():
                self._emit_stopped_phase()
                return
            quality_summary = dict(metadata.get("validation_summary") or {})
            profile_metadata = dict(profile.metadata or {})
            optimizer_source = profile_metadata.get("source")
            profile_metadata.update(metadata)
            profile_metadata.update(
                {
                    "source": (
                        "realtime_online_ui_full_pretrain"
                        if str(self.config.mode or "fast").strip().lower() == "full"
                        else "realtime_online_ui_fast_pretrain"
                    ),
                    "profile_optimizer_source": optimizer_source,
                    "trial_count": int(total),
                    "dataset_manifest": str(dataset_payload.get("dataset_manifest", "")),
                    "dataset_save_valid": bool(dataset_save_valid),
                    "dataset_save_error": str(dataset_save_error),
                    "pretrain_estimated_sec": pretrain_estimated_seconds(self.config),
                    "compute_backend_requested": str(self.config.compute_backend),
                }
            )
            final_profile = replace(profile, metadata=profile_metadata)
            fast_result = dict(metadata.get("fast_pretrain_result", {}))
            if fast_result:
                save_fast_fbcca_profile_bundle(
                    final_profile,
                    self.config.history_profile_path,
                    dict(fast_result.get("quality_metrics") or {}),
                )
                save_fast_fbcca_profile_bundle(
                    final_profile,
                    self.config.output_profile_path,
                    dict(fast_result.get("quality_metrics") or {}),
                )
            else:
                save_profile(final_profile, self.config.history_profile_path)
                save_profile(final_profile, self.config.output_profile_path)
            summary_text = (
                format_profile_quality_summary(quality_summary)
                if quality_summary
                else "Profile optimized; validation summary unavailable."
            )
            selected_channels_raw = (
                final_profile.eeg_channels
                if final_profile.eeg_channels is not None
                else metadata.get("selected_eeg_channels") or eeg_channels
            )
            selected_channels = [int(channel) for channel in selected_channels_raw]
            if fast_result:
                summary_text = (
                    "fast_pretrain_status={status} | template={template} | gate={gate} | source={source}".format(
                        status=str(fast_result.get("status", "")),
                        template=int(bool(fast_result.get("template_enabled", False))),
                        gate=int(bool(fast_result.get("gate_calibration_enabled", False))),
                        source=str(fast_result.get("source_profile", "")),
                    )
                )
            if not dataset_save_valid:
                summary_text = f"{summary_text} | dataset_save_valid=0"
            self.profile_ready.emit(
                {
                    "profile_path": str(self.config.output_profile_path),
                    "history_profile_path": str(self.config.history_profile_path),
                    "summary": quality_summary,
                    "summary_text": summary_text,
                    "model_name": final_profile.model_name,
                    "selected_eeg_channels": selected_channels,
                    "trial_count": int(total),
                    "fast_pretrain_status": str(fast_result.get("status", "")),
                    "template_enabled": bool(fast_result.get("template_enabled", False)),
                    "gate_calibration_enabled": bool(fast_result.get("gate_calibration_enabled", False)),
                    "source_profile": str(fast_result.get("source_profile", "")),
                    "dataset_manifest": str(dataset_payload.get("dataset_manifest", "")),
                    "dataset_save_valid": bool(dataset_save_valid),
                    "dataset_save_error": str(dataset_save_error),
                }
            )
            self.phase_changed.emit(
                {
                    "mode": PHASE_IDLE,
                    "title": "预训练完成",
                    "detail": f"Profile saved to {self.config.output_profile_path}",
                    "flicker": False,
                    "cue_freq": None,
                }
            )
        except Exception as exc:
            self.error.emit(f"预训练失败：{describe_runtime_error(exc, serial_port=active_serial)}")
            self.phase_changed.emit(
                {
                    "mode": PHASE_ERROR,
                    "title": "预训练错误",
                    "detail": str(exc),
                    "flicker": False,
                    "cue_freq": None,
                }
            )
        finally:
            if board is not None:
                try:
                    board.stop_stream()
                except Exception:
                    pass
                try:
                    board.release_session()
                except Exception:
                    pass
            self.finished.emit()


class RealtimeWorker(QObject):
    phase_changed = pyqtSignal(object)
    log = pyqtSignal(str)
    result = pyqtSignal(object)
    profile_info = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: RealtimeConfig) -> None:
        super().__init__()
        self.config = config
        self._stop_event = threading.Event()
        self._stimulus_phase_applied_event = threading.Event()
        self._last_stimulus_phase_payload: dict[str, Any] = {}

    def request_stop(self) -> None:
        self._stop_event.set()
        self._stimulus_phase_applied_event.set()

    def notify_stimulus_phase_presented(self, payload: Optional[dict[str, Any]] = None) -> None:
        if payload is None or str(payload.get("mode", "")) == PHASE_VALIDATION:
            self._last_stimulus_phase_payload = dict(payload or {})
            self._stimulus_phase_applied_event.set()

    def _wait_for_stimulus_phase_presented(self) -> bool:
        if self._stop_event.is_set():
            return False
        ready = self._stimulus_phase_applied_event.wait(float(REALTIME_STIMULUS_PHASE_APPLY_TIMEOUT_SEC))
        if not ready and not self._stop_event.is_set():
            self.log.emit(
                "warning: realtime stimulus first-frame acknowledgement timed out; "
                "online decoding will not start because the visual onset is not trustworthy"
            )
            return False
        return bool(ready)

    @pyqtSlot()
    def run(self) -> None:
        board = None
        active_serial = self.config.serial_port
        try:
            profile = load_profile(self.config.profile_path, fallback_freqs=self.config.freqs, require_exists=True)
            if profile_is_default_fallback(profile):
                raise RuntimeError("当前 profile 是默认回退值，请先完成训练评测并生成有效 profile")
            selected_model = normalize_model_name(self.config.model_name)
            original_model = normalize_model_name(profile.model_name)
            resolved_model, mismatch = resolve_realtime_model_choice(selected_model, original_model)
            if mismatch:
                self.log.emit(
                    f"模型不一致：UI 选择={selected_model}，profile 模型={original_model}；在线阶段将使用 profile 模型。"
                )
            profile = replace(profile, model_name=resolved_model)

            board, resolved_port, attempted_ports = prepare_board_session(self.config.board_id, self.config.serial_port)
            active_serial = resolved_port
            self.log.emit(
                f"连接成功：请求串口={self.config.serial_port} -> 实际串口={resolved_port}；尝试={attempted_ports}"
            )
            fs = int(BoardShim.get_sampling_rate(self.config.board_id))
            eeg_channels = resolve_selected_eeg_channels(
                BoardShim.get_eeg_channels(self.config.board_id),
                profile.eeg_channels,
            )
            gate = AsyncDecisionGate.from_profile(profile)
            shadow_chain = None
            shadow_summary: dict[str, Any] = {
                "shadow_mode_enabled": bool(self.config.shadow_mode),
                "shadow_mode": "disabled",
            }
            if bool(self.config.shadow_mode):
                try:
                    shadow_chain, shadow_runtime_summary = build_shadow_runtime_chain(
                        profile=profile,
                        profile_path=self.config.profile_path,
                    )
                    shadow_summary.update(dict(shadow_runtime_summary))
                    shadow_summary["shadow_mode_enabled"] = True
                    self.log.emit(
                        "shadow runtime enabled | "
                        f"gate={shadow_summary.get('gate_mode', 'unknown')} | "
                        f"profile_v2={int(bool(shadow_summary.get('profile_v2_loaded', False)))}"
                    )
                except Exception as exc:
                    shadow_chain = None
                    shadow_summary = {
                        "shadow_mode_enabled": False,
                        "shadow_mode": "failed",
                        "error": str(exc),
                    }
                    self.log.emit(f"shadow runtime disabled: {exc}")
            board.start_stream(450000)
            ready, probe_samples, sample_matrix = _read_probe_window(
                board,
                sampling_rate=fs,
                profile_win_sec=float(profile.win_sec),
            )
            probe_window = np.ascontiguousarray(
                sample_matrix[eeg_channels, -probe_samples:].T,
                dtype=np.float64,
            )
            selected_backend, selection_summary = _choose_runtime_backend(
                profile=profile,
                sampling_rate=fs,
                sample_window=probe_window,
                requested_backend=self.config.compute_backend,
                gpu_device=int(self.config.gpu_device),
                gpu_precision=self.config.gpu_precision,
                gpu_warmup=bool(self.config.gpu_warmup),
                gpu_cache_policy=self.config.gpu_cache_policy,
            )
            decoder = load_decoder_from_profile(
                profile,
                sampling_rate=fs,
                compute_backend=selected_backend,
                gpu_device=int(self.config.gpu_device),
                gpu_precision=self.config.gpu_precision,
                gpu_warmup=bool(self.config.gpu_warmup),
                gpu_cache_policy=self.config.gpu_cache_policy,
            )
            decoder.configure_runtime(fs)
            validation_summary = _validate_loaded_profile(profile, decoder, eeg_channels=eeg_channels)
            profile_metadata = dict(profile.metadata or {})
            fast_pretrain_meta = dict(profile_metadata.get("fast_pretrain") or {})
            fast_personalization = dict((profile.model_params or {}).get("fast_personalization") or {})
            backend_summary = (
                decoder.get_compute_backend_summary()
                if hasattr(decoder, "get_compute_backend_summary")
                else {}
            )
            backend_summary["selection_summary"] = dict(selection_summary)
            self.profile_info.emit(
                {
                    "loaded_profile_path": str(self.config.profile_path),
                    **validation_summary,
                    "backend_requested": str(backend_summary.get("requested_backend", self.config.compute_backend)),
                    "backend_used": str(backend_summary.get("used_backend", "cpu")),
                    "selection_summary": dict(selection_summary),
                    "shadow_summary": dict(shadow_summary),
                    "fast_pretrain": fast_pretrain_meta,
                    "fast_personalization": fast_personalization,
                }
            )
            self.log.emit(
                "compute backend summary | "
                f"requested={backend_summary.get('requested_backend', self.config.compute_backend)} | "
                f"used={backend_summary.get('used_backend', 'cpu')} | "
                f"precision={backend_summary.get('precision', self.config.gpu_precision)}"
            )
            self.log.emit(profile_runtime_summary(profile, backend_summary))
            self.log.emit(
                f"实时识别已启动 | 模型={profile.model_name} | fs={fs}Hz | 通道={list(eeg_channels)} | 缓冲={ready}"
            )
            self._stimulus_phase_applied_event.clear()
            self._last_stimulus_phase_payload = {}
            self.phase_changed.emit(
                {
                    "mode": PHASE_VALIDATION,
                    "title": f"实时识别中（{profile.model_name}）",
                    "detail": "注视目标方块会输出结果；看中心点时不输出。",
                    "flicker": True,
                    "cue_freq": None,
                    "active_sec": None,
                    "stimulus_profile_id": str(config.stimulus_profile_id),
                }
            )
            stimulus_ready = self._wait_for_stimulus_phase_presented()
            if self._stop_event.is_set():
                return
            if not stimulus_ready:
                raise RuntimeError("未收到实时刺激首帧确认，已停止识别以避免视觉起点不可信")
            board.get_board_data()
            if self._last_stimulus_phase_payload:
                payload = dict(self._last_stimulus_phase_payload)
                self.log.emit(
                    "realtime stimulus first-frame acknowledged | "
                    f"frame={payload.get('frame_index', 'unknown')} | "
                    f"t={payload.get('presented_t_sec', 'unknown')}"
                )
            consecutive_errors = 0
            while not self._stop_event.is_set():
                try:
                    if board.get_board_data_count() < decoder.win_samples:
                        self._stop_event.wait(0.05)
                        continue
                    data = board.get_current_board_data(decoder.win_samples)
                    if data.shape[1] < decoder.win_samples:
                        self._stop_event.wait(0.05)
                        continue
                    eeg = np.ascontiguousarray(data[eeg_channels, -decoder.win_samples :].T, dtype=np.float64)
                    t0 = time.perf_counter()
                    analysis = dict(decoder.analyze_window(eeg))
                    decision = gate.update(dict(analysis))
                    shadow_decision: dict[str, Any] = {}
                    if shadow_chain is not None:
                        shadow_decision = dict(shadow_chain.update(dict(analysis), timestamp_s=t0))
                    t1 = time.perf_counter()
                    decoder.update_online(decision, eeg)
                    consecutive_errors = 0
                except Exception as exc:
                    consecutive_errors += 1
                    self.log.emit(f"实时读数瞬态错误 {consecutive_errors}: {exc}")
                    if consecutive_errors >= DEFAULT_MAX_TRANSIENT_READ_ERRORS:
                        raise
                    self._stop_event.wait(0.2)
                    continue
                payload = {
                    "state": str(decision["state"]),
                    "pred_freq": None if decision["pred_freq"] is None else float(decision["pred_freq"]),
                    "selected_freq": None if decision["selected_freq"] is None else float(decision["selected_freq"]),
                    "top1_score": float(decision["top1_score"]),
                    "top2_score": float(decision["top2_score"]),
                    "margin": float(decision["margin"]),
                    "ratio": float(decision["ratio"]),
                    "stable_windows": int(decision["stable_windows"]),
                    "control_log_lr": None if decision.get("control_log_lr") is None else float(decision["control_log_lr"]),
                    "acc_log_lr": None if decision.get("acc_log_lr") is None else float(decision["acc_log_lr"]),
                    "decision_latency_ms": float((t1 - t0) * 1000.0),
                    "model_name": str(profile.model_name),
                    "compute_backend_requested": str(backend_summary.get("requested_backend", self.config.compute_backend)),
                    "compute_backend_used": str(backend_summary.get("used_backend", "cpu")),
                    "precision": str(backend_summary.get("precision", self.config.gpu_precision)),
                    "timing_breakdown": dict(backend_summary.get("timing_breakdown", {})),
                    "shadow_mode_enabled": bool(shadow_summary.get("shadow_mode_enabled", False)),
                    "shadow_gate_mode": str(shadow_summary.get("gate_mode", "global_gate")),
                    "shadow_state": None if not shadow_decision else str(shadow_decision.get("state", "")),
                    "shadow_commit": False if not shadow_decision else bool(shadow_decision.get("commit", False)),
                    "shadow_selected_freq": (
                        None
                        if not shadow_decision or shadow_decision.get("selected_freq") is None
                        else float(shadow_decision.get("selected_freq"))
                    ),
                    "shadow_gate_score": (
                        None if not shadow_decision else float(shadow_decision.get("gate_score", 0.0))
                    ),
                    "shadow_p_control": (
                        None if not shadow_decision else float(shadow_decision.get("p_control", 0.0))
                    ),
                }
                self.result.emit(payload)
                self._stop_event.wait(max(0.01, decoder.step_sec))
            self.phase_changed.emit(
                {
                    "mode": PHASE_STOPPED,
                    "title": "实时识别已停止",
                    "detail": "可再次点击开始。",
                    "flicker": False,
                    "cue_freq": None,
                }
            )
        except Exception as exc:
            self.error.emit(f"实时识别失败：{describe_runtime_error(exc, serial_port=active_serial)}")
            self.phase_changed.emit(
                {
                    "mode": PHASE_ERROR,
                    "title": "实时识别错误",
                    "detail": str(exc),
                    "flicker": False,
                    "cue_freq": None,
                }
            )
        finally:
            if board is not None:
                try:
                    board.stop_stream()
                except Exception:
                    pass
                try:
                    board.release_session()
                except Exception:
                    pass
            self.finished.emit()


class RealtimeOnlineWindow(QMainWindow):
    def __init__(self, *, serial_port: str, board_id: int, freqs: Sequence[float], demo_mode: bool = False) -> None:
        super().__init__()
        self.demo_mode = bool(demo_mode)
        self.setWindowTitle("SSVEP FBCCA Demo" if self.demo_mode else "SSVEP 实时识别")
        self.resize(1260, 860)
        self.setMinimumSize(1180, 720)

        self.serial_port_default = normalize_serial_port(serial_port)
        self.board_id_default = int(board_id)
        self.freqs = DEMO_EXPECTED_FREQS if self.demo_mode else tuple(float(freq) for freq in freqs)
        self._stim_refresh_rate_hz = _suggest_refresh_rate_hz()
        self._stimulus_profile_id = DEFAULT_REALTIME_STIMULUS_PROFILE_ID
        self._stimulus_mode, self._stimulus_mode_reason = resolve_realtime_stimulus_mode(
            stimulus_profile_id=self._stimulus_profile_id,
            refresh_rate_hz=self._stim_refresh_rate_hz,
        )

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[RealtimeWorker] = None
        self.pretrain_thread: Optional[QThread] = None
        self.pretrain_worker: Optional[RealtimePretrainWorker] = None
        self.connect_thread: Optional[QThread] = None
        self.connect_worker: Optional[DeviceCheckWorker] = None
        self._last_signature: Optional[tuple[str, Optional[float]]] = None
        self._connecting = False
        self._stimulus_focus_mode = False
        self._start_realtime_after_pretrain = False
        self._pretrain_profile_ready_for_auto_start = False
        self._pending_pretrain_mode = "fast"

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        self._root_layout = layout
        self._root_layout_margins = layout.contentsMargins()

        left = QWidget(root)
        self._control_panel = left
        left.setMinimumWidth(360)
        left.setMaximumWidth(REALTIME_CONTROL_PANEL_WIDTH)
        left.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left)
        form = QFormLayout()

        self.serial_edit = QLineEdit(self.serial_port_default)
        self.board_edit = QLineEdit(str(self.board_id_default))
        self.freqs_edit = QLineEdit(",".join(f"{freq:g}" for freq in self.freqs))
        self.model_combo = QComboBox()
        model_options = (DEFAULT_MODEL_NAME,) if self.demo_mode else MODEL_OPTIONS
        for item in model_options:
            self.model_combo.addItem(item)
        self.model_combo.setCurrentText(DEFAULT_MODEL_NAME)
        self.model_combo.setToolTip(
            "FBCCA demo 固定使用 fbcca。"
            if self.demo_mode
            else "在线识别以 profile 内的 model_name/model_params 为准；下拉框仅用于启动前一致性提示。"
        )
        self.model_combo.setEnabled(not self.demo_mode)
        self.freqs_edit.setReadOnly(self.demo_mode)
        if self.demo_mode:
            self.freqs_edit.setToolTip("FBCCA demo 固定频率：8 / 10 / 12 / 15 Hz")
        self.profile_edit = QLineEdit(str(DEFAULT_REALTIME_PROFILE_PATH))
        self.compute_backend_combo = QComboBox()
        self.compute_backend_combo.addItems(["auto", "cpu", "cuda"])
        self.compute_backend_combo.setCurrentText(str(DEFAULT_COMPUTE_BACKEND_NAME))
        self.gpu_device_edit = QLineEdit(str(DEFAULT_GPU_DEVICE_ID))
        self.gpu_precision_combo = QComboBox()
        self.gpu_precision_combo.addItems(["float32", "float64"])
        self.gpu_precision_combo.setCurrentText(str(DEFAULT_GPU_PRECISION_NAME))
        self.gpu_warmup_edit = QLineEdit("1")
        self.gpu_cache_combo = QComboBox()
        self.gpu_cache_combo.addItems(["windows", "full"])
        self.gpu_cache_combo.setCurrentText(str(DEFAULT_GPU_CACHE_MODE))
        self.shadow_mode_check = QCheckBox("Shadow mode (no robot command)")
        self.shadow_mode_check.setChecked(True)

        form.addRow("串口", self.serial_edit)
        form.addRow("板卡 ID", self.board_edit)
        form.addRow("刺激频率", self.freqs_edit)
        form.addRow("模型", self.model_combo)
        form.addRow("Profile 路径", self.profile_edit)
        form.addRow("计算后端", self.compute_backend_combo)
        form.addRow("GPU 设备", self.gpu_device_edit)
        form.addRow("GPU 精度", self.gpu_precision_combo)
        form.addRow("GPU 预热(1/0)", self.gpu_warmup_edit)
        form.addRow("GPU 缓存", self.gpu_cache_combo)
        form.addRow("Shadow", self.shadow_mode_check)
        left_layout.addLayout(form)

        row = QHBoxLayout()
        self.btn_load_profile = QPushButton("加载Profile")
        self.btn_connect = QPushButton("连接设备")
        self.btn_start = QPushButton("开始实时识别")
        self.btn_no_train_start = QPushButton("无训练直接 FBCCA 识别（测试）")
        self.btn_pretrain_then_start = QPushButton("预训练后开始识别（约5分钟）")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_full_pretrain_then_start = QPushButton("高级：完整训练并开始识别（约5分钟）")
        self.btn_pretrain_then_start.setText("快速预训练并开始正式识别（约60秒）")
        self.btn_load_profile.setText("加载 Profile")
        self.btn_connect.setText("连接设备")
        self.btn_start.setText("开始实时识别")
        self.btn_no_train_start.setText("无训练直接 FBCCA 识别（测试）")
        self.btn_stop.setText("停止")
        row.addWidget(self.btn_load_profile)
        row.addWidget(self.btn_connect)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        left_layout.addLayout(row)
        left_layout.addWidget(self.btn_no_train_start)
        left_layout.addWidget(self.btn_pretrain_then_start)
        left_layout.addWidget(self.btn_full_pretrain_then_start)

        publish_row = QHBoxLayout()
        self.btn_publish_realtime_profile = QPushButton("发布到02实时Profile")
        self.btn_publish_hybrid_profile = QPushButton("发布到主程序Profile")
        publish_row.addWidget(self.btn_publish_realtime_profile)
        publish_row.addWidget(self.btn_publish_hybrid_profile)
        self.btn_publish_hybrid_profile.setVisible(False)
        left_layout.addLayout(publish_row)

        self.phase_label = QLabel("空闲")
        self.phase_label.setStyleSheet("font-size:16px; font-weight:600;")
        self.phase_label.setWordWrap(True)
        left_layout.addWidget(self.phase_label)

        self.result_label = QLabel("输出频率：None")
        self.result_label.setStyleSheet("font-size:18px; font-weight:600;")
        self.result_label.setWordWrap(True)
        left_layout.addWidget(self.result_label)

        self.profile_meta_label = QLabel("Profile：未加载")
        self.profile_meta_label.setWordWrap(True)
        left_layout.addWidget(self.profile_meta_label)

        self.backend_meta_label = QLabel("后端：未选择")
        self.backend_meta_label.setWordWrap(True)
        left_layout.addWidget(self.backend_meta_label)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text, 1)

        right = QWidget(root)
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right)
        stimulus_profile = get_stimulus_profile(self._stimulus_profile_id)
        self.stim = FourArrowStimWidget(
            freqs=self.freqs,
            refresh_rate_hz=self._stim_refresh_rate_hz,
            mean=float(stimulus_profile.mean),
            amp=float(stimulus_profile.amp),
            phi=float(stimulus_profile.phi),
            stimulus_mode=self._stimulus_mode,
            stimulus_profile_id=str(stimulus_profile.profile_id),
            ramp_sec=float(stimulus_profile.ramp_sec),
        )
        self.stim.selected_border_color = REALTIME_SELECTED_BORDER_COLOR
        self.stim.setMinimumSize(REALTIME_STIM_MIN_WIDTH, REALTIME_STIM_MIN_HEIGHT)
        self.stim.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.stim, 1)
        self._right_layout = right_layout
        self._right_layout_margins = right_layout.contentsMargins()

        layout.addWidget(left, 0)
        layout.addWidget(right, 1)
        layout.setStretch(0, 0)
        layout.setStretch(1, 1)

        self.btn_load_profile.clicked.connect(self._pick_profile)
        self.btn_connect.clicked.connect(self._connect_device)
        self.btn_start.clicked.connect(self._start_realtime)
        self.btn_no_train_start.clicked.connect(self._start_no_train_fbcca_realtime)
        self.btn_pretrain_then_start.clicked.connect(self._start_pretrain_then_realtime)
        self.btn_full_pretrain_then_start.clicked.connect(self._start_full_pretrain_then_realtime)
        self.btn_stop.clicked.connect(self._stop_realtime)
        self.btn_publish_realtime_profile.clicked.connect(self._publish_current_profile_to_realtime)
        self.btn_publish_hybrid_profile.clicked.connect(self._publish_current_profile_to_hybrid)
        self.stim.active_phase_frame_presented.connect(self._on_active_phase_frame_presented)
        if self.demo_mode:
            self._log("FBCCA demo mode: model=fbcca, freqs=8/10/12/15Hz. Non-FBCCA profiles will be rejected.")
        self._refresh_task_buttons()

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{stamp}] {text}")

    def _set_running(self, running: bool) -> None:
        del running
        self._refresh_task_buttons()

    def _task_active(self) -> bool:
        return self.worker_thread is not None or self.pretrain_thread is not None

    def _refresh_task_buttons(self) -> None:
        active = self._task_active()
        idle = (not active) and (not self._connecting)
        profile_path = self._current_profile_path()
        self.btn_connect.setEnabled(idle)
        self.btn_start.setEnabled(idle)
        self.btn_no_train_start.setEnabled(idle)
        self.btn_pretrain_then_start.setEnabled(idle)
        self.btn_full_pretrain_then_start.setEnabled(idle)
        self.btn_load_profile.setEnabled(idle)
        self.btn_publish_realtime_profile.setEnabled(idle and profile_path.exists())
        self.btn_publish_hybrid_profile.setEnabled(False)
        self.btn_stop.setEnabled(active)
        self.shadow_mode_check.setEnabled(not active)

    def _set_stimulus_focus_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._stimulus_focus_mode == enabled:
            if enabled and not self.isFullScreen():
                self.showFullScreen()
            return
        self._stimulus_focus_mode = enabled
        self._control_panel.setVisible(not enabled)
        if enabled:
            self._root_layout.setContentsMargins(0, 0, 0, 0)
            self._right_layout.setContentsMargins(0, 0, 0, 0)
            self.showFullScreen()
            self.setFocus(Qt.OtherFocusReason)
            self.stim.update()
            return
        root_margins = self._root_layout_margins
        right_margins = self._right_layout_margins
        self._root_layout.setContentsMargins(
            root_margins.left(),
            root_margins.top(),
            root_margins.right(),
            root_margins.bottom(),
        )
        self._right_layout.setContentsMargins(
            right_margins.left(),
            right_margins.top(),
            right_margins.right(),
            right_margins.bottom(),
        )
        self.stim.update()

    def _set_connecting(self, connecting: bool) -> None:
        self._connecting = bool(connecting)
        if connecting:
            self.phase_label.setText("连接中...")
        else:
            self._refresh_task_buttons()
            return
        self._refresh_task_buttons()

    def _pick_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 Profile", str(Path(self.profile_edit.text()).parent), "JSON (*.json)")
        if path:
            if self.demo_mode:
                try:
                    validate_fbcca_demo_profile_path(path)
                except Exception as exc:
                    self._log(f"Demo profile rejected: {exc}")
                    return
            self.profile_edit.setText(path)
            self._refresh_task_buttons()

    def _current_profile_path(self) -> Path:
        raw_path = self.profile_edit.text().strip()
        if not raw_path:
            return Path.cwd() / "__missing_fbcca_profile__.json"
        return Path(raw_path).expanduser().resolve()

    def _validate_current_demo_profile(self) -> dict[str, Any]:
        return validate_fbcca_demo_profile_path(self._current_profile_path())

    def _publish_current_profile_to_realtime(self) -> None:
        try:
            result = publish_profile_to_ssvep_realtime(self._current_profile_path())
        except Exception as exc:
            self._log(f"Publish to 02_SSVEP realtime rejected: {exc}")
            return
        self.profile_edit.setText(str(result["profile_path"]))
        self._log(f"Published realtime FBCCA profile: {result['profile_path']}")
        if result.get("copied_v2"):
            self._log(f"Published realtime profile_v2: {result['profile_v2_path']}")
        self._refresh_task_buttons()

    def _publish_current_profile_to_hybrid(self) -> None:
        try:
            result = publish_profile_to_hybrid_controller(self._current_profile_path())
        except Exception as exc:
            self._log(f"Publish to hybrid_controller rejected: {exc}")
            return
        self._log(f"Published hybrid current FBCCA profile: {result['current_profile_path']}")
        self._log(f"Published hybrid history FBCCA profile: {result['history_profile_path']}")
        self._refresh_task_buttons()

    def _read_config(self) -> RealtimeConfig:
        serial_port = normalize_serial_port(self.serial_edit.text().strip())
        board_id = int(self.board_edit.text().strip())
        freqs = parse_freqs(self.freqs_edit.text().strip())
        model_name = normalize_model_name(self.model_combo.currentText())
        if self.demo_mode:
            freqs = DEMO_EXPECTED_FREQS
            model_name = DEFAULT_MODEL_NAME
            self.freqs_edit.setText(",".join(f"{freq:g}" for freq in DEMO_EXPECTED_FREQS))
            self.model_combo.setCurrentText(DEFAULT_MODEL_NAME)
        profile_path = self._current_profile_path()
        stim_refresh_rate_hz = float(self._stim_refresh_rate_hz)
        stimulus_mode, _stim_reason = resolve_realtime_stimulus_mode(
            stimulus_profile_id=self._stimulus_profile_id,
            refresh_rate_hz=stim_refresh_rate_hz,
        )
        return RealtimeConfig(
            serial_port=serial_port,
            board_id=board_id,
            freqs=freqs,
            profile_path=profile_path,
            model_name=model_name,
            compute_backend=parse_compute_backend_name(self.compute_backend_combo.currentText().strip()),
            gpu_device=int(self.gpu_device_edit.text().strip() or str(DEFAULT_GPU_DEVICE_ID)),
            gpu_precision=parse_gpu_precision(self.gpu_precision_combo.currentText().strip()),
            gpu_warmup=bool(int(self.gpu_warmup_edit.text().strip() or "1")),
            gpu_cache_policy=parse_gpu_cache_policy(self.gpu_cache_combo.currentText().strip()),
            shadow_mode=bool(self.shadow_mode_check.isChecked()),
            stimulus_profile_id=str(self._stimulus_profile_id),
            stimulus_mode=stimulus_mode,
            stim_refresh_rate_hz=stim_refresh_rate_hz,
        )

    def _read_pretrain_config(self, *, mode: str = "fast") -> RealtimePretrainConfig:
        realtime_cfg = self._read_config()
        history_profile_path = build_pretrain_profile_path()
        normalized_mode = str(mode or "fast").strip().lower()
        is_full = normalized_mode == "full"
        return RealtimePretrainConfig(
            serial_port=realtime_cfg.serial_port,
            board_id=realtime_cfg.board_id,
            freqs=realtime_cfg.freqs,
            base_profile_path=SSVEP_FBCCA_BASE_PROFILE_PATH,
            fallback_profile_path=realtime_cfg.profile_path,
            output_profile_path=realtime_cfg.profile_path if is_full else SSVEP_REALTIME_PROFILE_PATH,
            history_profile_path=history_profile_path,
            compute_backend=realtime_cfg.compute_backend,
            gpu_device=realtime_cfg.gpu_device,
            gpu_precision=realtime_cfg.gpu_precision,
            gpu_warmup=realtime_cfg.gpu_warmup,
            gpu_cache_policy=realtime_cfg.gpu_cache_policy,
            prepare_sec=DEFAULT_FULL_PRETRAIN_PREPARE_SEC if is_full else DEFAULT_PRETRAIN_PREPARE_SEC,
            active_sec=DEFAULT_FULL_PRETRAIN_ACTIVE_SEC if is_full else DEFAULT_PRETRAIN_ACTIVE_SEC,
            rest_sec=DEFAULT_FULL_PRETRAIN_REST_SEC if is_full else DEFAULT_PRETRAIN_REST_SEC,
            target_repeats=DEFAULT_FULL_PRETRAIN_TARGET_REPEATS if is_full else DEFAULT_PRETRAIN_TARGET_REPEATS,
            idle_repeats=DEFAULT_FULL_PRETRAIN_IDLE_REPEATS if is_full else DEFAULT_PRETRAIN_IDLE_REPEATS,
            win_sec=DEFAULT_FULL_PRETRAIN_WIN_SEC if is_full else DEFAULT_PRETRAIN_WIN_SEC,
            step_sec=DEFAULT_FULL_PRETRAIN_STEP_SEC if is_full else DEFAULT_PRETRAIN_STEP_SEC,
            mode="full" if is_full else "fast",
            stimulus_profile_id=realtime_cfg.stimulus_profile_id,
            stim_refresh_rate_hz=float(realtime_cfg.stim_refresh_rate_hz),
        )

    def _connect_device(self) -> None:
        if self.connect_thread is not None or self._connecting:
            return
        if self._task_active():
            self._log("预训练或实时识别运行中，请先停止后再重连设备。")
            return
        try:
            cfg = self._read_config()
        except Exception as exc:
            self._log(f"配置错误：{exc}")
            return
        worker = DeviceCheckWorker(serial_port=cfg.serial_port, board_id=cfg.board_id)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.connected.connect(self._on_connected)
        worker.error.connect(self._on_connect_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_connect_finished)
        thread.started.connect(worker.run)
        self.connect_worker = worker
        self.connect_thread = thread
        self._set_connecting(True)
        thread.start()

    def _on_connected(self, payload: dict[str, Any]) -> None:
        self.phase_label.setText("设备已连接")
        self._log(
            "连接成功：请求串口 {requested_serial_port}，实际串口 {resolved_serial_port}；"
            "采样率 {sampling_rate}Hz，缓存就绪 {ready_samples}".format(**payload)
        )

    def _on_connect_error(self, text: str) -> None:
        self.phase_label.setText("连接失败")
        self._log(text)

    def _on_connect_finished(self) -> None:
        self.connect_worker = None
        self.connect_thread = None
        self._set_connecting(False)

    def _start_pretrain_then_realtime(self) -> None:
        if self._task_active():
            return
        if self._connecting:
            self._log("设备连接中，请稍候。")
            return
        mode = str(getattr(self, "_pending_pretrain_mode", "fast") or "fast").strip().lower()
        self._pending_pretrain_mode = "fast"
        try:
            cfg = self._read_pretrain_config(mode=mode)
        except Exception as exc:
            self._log(f"配置错误：{exc}")
            return

        worker = RealtimePretrainWorker(cfg)
        thread = QThread(self)
        worker.moveToThread(thread)
        stimulus_profile = get_stimulus_profile(cfg.stimulus_profile_id)
        stimulus_mode, _stim_reason = resolve_realtime_stimulus_mode(
            stimulus_profile_id=cfg.stimulus_profile_id,
            refresh_rate_hz=float(self._stim_refresh_rate_hz),
        )
        self.stim.freqs = tuple(float(freq) for freq in cfg.freqs)
        self.stim.refresh_rate_hz = float(self._stim_refresh_rate_hz)
        self.stim.stimulus_mode = str(stimulus_mode)
        self.stim.stimulus_profile_id = str(stimulus_profile.profile_id)
        self.stim.mean = float(stimulus_profile.mean)
        self.stim.amp = float(stimulus_profile.amp)
        self.stim.phi = float(stimulus_profile.phi)
        self.stim.ramp_sec = float(stimulus_profile.ramp_sec)
        worker.log.connect(self._log)
        worker.profile_ready.connect(self._on_pretrain_profile_ready)
        worker.error.connect(self._on_pretrain_error)
        worker.phase_changed.connect(self._on_phase_changed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_pretrain_finished)
        thread.started.connect(worker.run)

        self.pretrain_worker = worker
        self.pretrain_thread = thread
        self._start_realtime_after_pretrain = True
        self._pretrain_profile_ready_for_auto_start = False
        self._set_stimulus_focus_mode(True)
        self._refresh_task_buttons()
        self._last_signature = None
        self.phase_label.setText("正在启动预训练...")
        self._log(
            "预训练计划：{trials} trials，预计 {seconds:.0f} 秒，profile 将写入 {profile}".format(
                trials=pretrain_trial_count(cfg),
                seconds=pretrain_estimated_seconds(cfg),
                profile=cfg.output_profile_path,
            )
        )
        thread.start()

    def _start_full_pretrain_then_realtime(self) -> None:
        self._pending_pretrain_mode = "full"
        self._start_pretrain_then_realtime()

    def _start_no_train_fbcca_realtime(self) -> None:
        if self._task_active():
            return
        if self._connecting:
            self._log("设备连接中，请稍候。")
            return
        try:
            cfg = self._read_config()
            freqs = DEMO_EXPECTED_FREQS if self.demo_mode else cfg.freqs
            profile_path, profile_v2_path = save_no_train_fbcca_profile(freqs=freqs)
            self.profile_edit.setText(str(profile_path))
            self.model_combo.setCurrentText(DEFAULT_MODEL_NAME)
            self._log(f"已生成无训练 FBCCA profile：{profile_path}")
            self._log(f"已生成无训练 FBCCA profile_v2：{profile_v2_path}")
        except Exception as exc:
            self._log(f"无训练 FBCCA profile 生成失败：{exc}")
            return
        self._start_realtime()

    def _start_realtime(self) -> None:
        if self._task_active():
            return
        if self._connecting:
            self._log("设备连接中，请稍候。")
            return
        try:
            cfg = self._read_config()
        except Exception as exc:
            self._log(f"配置错误：{exc}")
            return
        if not cfg.profile_path.exists():
            self._log(f"未找到 Profile：{cfg.profile_path}")
            return
        if self.demo_mode:
            try:
                self._validate_current_demo_profile()
            except Exception as exc:
                self._log(f"Demo profile rejected: {exc}")
                return
        worker = RealtimeWorker(cfg)
        thread = QThread(self)
        worker.moveToThread(thread)
        stimulus_profile = get_stimulus_profile(cfg.stimulus_profile_id)
        self.stim.freqs = tuple(float(freq) for freq in cfg.freqs)
        self.stim.refresh_rate_hz = float(cfg.stim_refresh_rate_hz)
        self.stim.stimulus_mode = str(cfg.stimulus_mode)
        self.stim.stimulus_profile_id = str(stimulus_profile.profile_id)
        self.stim.mean = float(stimulus_profile.mean)
        self.stim.amp = float(stimulus_profile.amp)
        self.stim.phi = float(stimulus_profile.phi)
        self.stim.ramp_sec = float(stimulus_profile.ramp_sec)
        worker.log.connect(self._log)
        worker.result.connect(self._on_result)
        worker.profile_info.connect(self._on_profile_info)
        worker.error.connect(self._on_error)
        worker.phase_changed.connect(self._on_phase_changed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_finished)
        thread.started.connect(worker.run)
        self.worker = worker
        self.worker_thread = thread
        self._set_running(True)
        self._set_stimulus_focus_mode(True)
        self._last_signature = None
        self.phase_label.setText("正在启动实时识别...")
        thread.start()

    def _stop_realtime(self) -> None:
        self._start_realtime_after_pretrain = False
        self._pretrain_profile_ready_for_auto_start = False
        if self.pretrain_worker is not None:
            self.pretrain_worker.request_stop()
        if self.worker is not None:
            self.worker.request_stop()
        self._set_stimulus_focus_mode(False)
        self._refresh_task_buttons()

    def _on_pretrain_profile_ready(self, payload: dict[str, Any]) -> None:
        profile_path_text = str(payload.get("profile_path", "")).strip()
        if profile_path_text:
            self.profile_edit.setText(str(Path(profile_path_text).expanduser()))
        if self.demo_mode:
            try:
                self._validate_current_demo_profile()
            except Exception as exc:
                self._pretrain_profile_ready_for_auto_start = False
                self._log(f"Pretrained demo profile rejected: {exc}")
                return
        model_name = normalize_model_name(str(payload.get("model_name", DEFAULT_MODEL_NAME)))
        self.model_combo.setCurrentText(model_name)
        summary_text = str(payload.get("summary_text", "")).strip()
        channels = payload.get("selected_eeg_channels", [])
        fast_status = str(payload.get("fast_pretrain_status", "")).strip() or "n/a"
        template_enabled = int(bool(payload.get("template_enabled", False)))
        gate_enabled = int(bool(payload.get("gate_calibration_enabled", False)))
        dataset_save_valid = bool(payload.get("dataset_save_valid", True))
        dataset_manifest = str(payload.get("dataset_manifest", "")).strip()
        self.profile_meta_label.setText(
            "Profile：{path}\n模型：{model} | 预训练通道：{channels}".format(
                path=payload.get("profile_path", ""),
                model=model_name,
                channels=list(channels) if isinstance(channels, (list, tuple)) else channels,
            )
        )
        self._log(f"预训练 profile 已生成：{payload.get('profile_path', '')}")
        self.profile_meta_label.setText(
            self.profile_meta_label.text()
            + f"\nfast_pretrain={fast_status} | template={template_enabled} | gate={gate_enabled}"
        )
        self.profile_meta_label.setText(
            self.profile_meta_label.text()
            + f"\ndataset_save_valid={int(dataset_save_valid)} | dataset={dataset_manifest or 'n/a'}"
        )
        history_path = str(payload.get("history_profile_path", "")).strip()
        if history_path:
            self._log(f"预训练历史 profile：{history_path}")
        if not dataset_save_valid:
            self._log(f"warning: pretrain dataset was not saved: {payload.get('dataset_save_error', '')}")
        if summary_text:
            self._log(summary_text)
        self._pretrain_profile_ready_for_auto_start = True

    def _on_pretrain_error(self, text: str) -> None:
        self._start_realtime_after_pretrain = False
        self._pretrain_profile_ready_for_auto_start = False
        self._log(text)

    def _on_pretrain_finished(self) -> None:
        should_start = self._start_realtime_after_pretrain and self._pretrain_profile_ready_for_auto_start
        self.pretrain_worker = None
        self.pretrain_thread = None
        self._start_realtime_after_pretrain = False
        self._pretrain_profile_ready_for_auto_start = False
        self._set_stimulus_focus_mode(False)
        self._refresh_task_buttons()
        if should_start:
            self._log("预训练完成，开始实时识别。")
            self._start_realtime()

    def _on_result(self, payload: dict[str, Any]) -> None:
        self.stim.apply_result(payload)
        control_lr = payload.get("control_log_lr")
        acc_lr = payload.get("acc_log_lr")
        control_lr_text = "n/a" if control_lr is None else f"{float(control_lr):.3f}"
        acc_lr_text = "n/a" if acc_lr is None else f"{float(acc_lr):.3f}"
        self.result_label.setText(
            "pred_freq={pred_freq} | selected_freq={selected_freq} | state={state}\n"
            "margin={margin:.3f} ratio={ratio:.3f} stable_windows={stable_windows} | "
            "control_lr={control_lr} acc_lr={acc_lr}".format(
                pred_freq=payload.get("pred_freq"),
                selected_freq=payload.get("selected_freq"),
                state=payload.get("state"),
                margin=float(payload.get("margin", 0.0) or 0.0),
                ratio=float(payload.get("ratio", 0.0) or 0.0),
                stable_windows=int(payload.get("stable_windows", 0) or 0),
                control_lr=control_lr_text,
                acc_lr=acc_lr_text,
            )
        )
        signature = (str(payload.get("state", "")), payload.get("selected_freq"))
        if signature == self._last_signature:
            return
        self._last_signature = signature
        self._log(
            "状态={state} 预测={pred_freq} 选中={selected_freq} 延迟={decision_latency_ms:.3f}ms".format(
                state=payload.get("state"),
                pred_freq=payload.get("pred_freq"),
                selected_freq=payload.get("selected_freq"),
                decision_latency_ms=float(payload.get("decision_latency_ms", 0.0)),
            )
        )
        if bool(payload.get("shadow_mode_enabled", False)):
            self._log(
                "shadow={state} commit={commit} selected={selected} p={p:.3f}".format(
                    state=payload.get("shadow_state"),
                    commit=bool(payload.get("shadow_commit", False)),
                    selected=payload.get("shadow_selected_freq"),
                    p=float(payload.get("shadow_p_control", 0.0) or 0.0),
                )
            )

    def _on_profile_info(self, payload: dict[str, Any]) -> None:
        self.profile_meta_label.setText(
            "Profile：{path}\n模型：{model} | 通道权重：{cw} | 子带权重：{sw}".format(
                path=payload.get("loaded_profile_path", ""),
                model=payload.get("loaded_profile_model", ""),
                cw=payload.get("channel_weight_count", 0),
                sw=payload.get("subband_weight_count", 0),
            )
        )
        fast_pretrain = dict(payload.get("fast_pretrain", {}))
        fast_personalization = dict(payload.get("fast_personalization", {}))
        self.profile_meta_label.setText(
            self.profile_meta_label.text()
            + "\nfast_pretrain={status} | template={template} | gate={gate}".format(
                status=str(fast_pretrain.get("status", "n/a")),
                template=int(bool(fast_personalization.get("templates"))),
                gate=int(bool(fast_pretrain.get("gate_calibration_enabled", False))),
            )
        )
        selection_summary = dict(payload.get("selection_summary", {}))
        shadow_summary = dict(payload.get("shadow_summary", {}))
        self.backend_meta_label.setText(
            "后端：requested={requested} | used={used}\n选择：{mode} {reason}\nshadow={shadow} gate={gate} v2={v2}".format(
                requested=payload.get("backend_requested", ""),
                used=payload.get("backend_used", ""),
                mode=selection_summary.get("selection_mode", ""),
                reason=selection_summary.get("reason", ""),
                shadow=shadow_summary.get("shadow_mode", "disabled"),
                gate=shadow_summary.get("gate_mode", "global_gate"),
                v2=int(bool(shadow_summary.get("profile_v2_loaded", False))),
            ).strip()
        )

    def _on_error(self, text: str) -> None:
        self._log(text)

    def _on_phase_changed(self, phase: dict[str, Any]) -> None:
        title = str(phase.get("title", ""))
        self.phase_label.setText(title or "实时识别")
        self.stim.apply_phase(phase)

    def _on_active_phase_frame_presented(self, payload: dict[str, Any]) -> None:
        if self.pretrain_worker is not None:
            self.pretrain_worker.notify_stimulus_phase_presented(dict(payload))
        if self.worker is not None:
            self.worker.notify_stimulus_phase_presented(dict(payload))

    def _on_finished(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._set_stimulus_focus_mode(False)
        self._set_running(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.pretrain_worker is not None:
            self.pretrain_worker.request_stop()
        if self.pretrain_thread is not None:
            self.pretrain_thread.quit()
            self.pretrain_thread.wait(3000)
        if self.worker is not None:
            self.worker.request_stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(3000)
        if self.connect_thread is not None:
            self.connect_thread.quit()
            self.connect_thread.wait(3000)
        try:
            self.stim.stop_clock()
        except Exception:
            pass
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape and self._stimulus_focus_mode:
            self._stop_realtime()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
            event.accept()
            return
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
            event.accept()
            return
        super().keyPressEvent(event)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSVEP 实时识别 UI / CLI")
    parser.add_argument("--serial-port", type=str, default="auto")
    parser.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    parser.add_argument("--freqs", type=str, default="8,10,12,15")
    parser.add_argument("--profile", type=Path, default=DEFAULT_REALTIME_PROFILE_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--compute-backend", type=str, default=DEFAULT_COMPUTE_BACKEND_NAME)
    parser.add_argument("--gpu-device", type=int, default=DEFAULT_GPU_DEVICE_ID)
    parser.add_argument("--gpu-precision", type=str, default=DEFAULT_GPU_PRECISION_NAME)
    parser.add_argument("--gpu-warmup", type=int, default=1)
    parser.add_argument("--gpu-cache-policy", type=str, default=DEFAULT_GPU_CACHE_MODE)
    parser.add_argument("--shadow-mode", type=int, default=1)
    parser.add_argument("--demo-mode", type=int, default=0)
    parser.add_argument("--emit-all", action="store_true")
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="仅命令行运行，不启动 UI")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    demo_mode = bool(int(args.demo_mode))
    if bool(args.headless):
        profile_path = Path(args.profile).expanduser().resolve()
        if demo_mode:
            validate_fbcca_demo_profile_path(profile_path)
        runner = OnlineRunner(
            serial_port=args.serial_port,
            board_id=args.board_id,
            freqs=DEMO_EXPECTED_FREQS if demo_mode else parse_freqs(args.freqs),
            profile_path=profile_path,
            emit_all=bool(args.emit_all),
            model_name=DEFAULT_MODEL_NAME if demo_mode else str(args.model),
            compute_backend=parse_compute_backend_name(str(args.compute_backend).strip()),
            gpu_device=int(args.gpu_device),
            gpu_precision=parse_gpu_precision(str(args.gpu_precision).strip()),
            gpu_warmup=bool(int(args.gpu_warmup)),
            gpu_cache_policy=parse_gpu_cache_policy(str(args.gpu_cache_policy).strip()),
        )
        runner.run(max_updates=args.max_updates)
        return 0

    app = QApplication([])
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = RealtimeOnlineWindow(
        serial_port=args.serial_port,
        board_id=args.board_id,
        freqs=parse_freqs(args.freqs),
        demo_mode=demo_mode,
    )
    window.profile_edit.setText(str(Path(args.profile).expanduser().resolve()))
    window.model_combo.setCurrentText(DEFAULT_MODEL_NAME if demo_mode else normalize_model_name(args.model))
    window.compute_backend_combo.setCurrentText(parse_compute_backend_name(str(args.compute_backend).strip()))
    window.gpu_device_edit.setText(str(int(args.gpu_device)))
    window.gpu_precision_combo.setCurrentText(parse_gpu_precision(str(args.gpu_precision).strip()))
    window.gpu_warmup_edit.setText("1" if bool(int(args.gpu_warmup)) else "0")
    window.gpu_cache_combo.setCurrentText(parse_gpu_cache_policy(str(args.gpu_cache_policy).strip()))
    window.shadow_mode_check.setChecked(bool(int(args.shadow_mode)))
    window.showFullScreen()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
