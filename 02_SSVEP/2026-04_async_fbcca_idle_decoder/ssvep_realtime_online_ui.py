from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QCloseEvent, QFont
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
    QVBoxLayout,
    QWidget,
)

from async_fbcca_idle_standalone import (
    AsyncDecisionGate,
    DEFAULT_BENCHMARK_MODELS,
    DEFAULT_BOARD_ID,
    DEFAULT_COMPUTE_BACKEND_NAME,
    DEFAULT_GPU_CACHE_MODE,
    DEFAULT_GPU_DEVICE_ID,
    DEFAULT_GPU_PRECISION_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROFILE_PATH,
    DEFAULT_STREAM_WARMUP_SEC,
    DEFAULT_MAX_TRANSIENT_READ_ERRORS,
    OnlineRunner,
    BoardShim,
    describe_runtime_error,
    ensure_stream_ready,
    load_decoder_from_profile,
    load_profile,
    normalize_model_name,
    normalize_serial_port,
    parse_compute_backend_name,
    parse_freqs,
    parse_gpu_cache_policy,
    parse_gpu_precision,
    prepare_board_session,
    profile_is_default_fallback,
    resolve_selected_eeg_channels,
)
from ssvep_core.runtime_shadow import build_shadow_runtime_chain
from async_fbcca_validation_ui import (
    FourArrowStimWidget,
    PHASE_ERROR,
    PHASE_IDLE,
    PHASE_STOPPED,
    PHASE_VALIDATION,
)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_REALTIME_PROFILE_PATH = DEFAULT_PROFILE_PATH
MODEL_OPTIONS = (DEFAULT_MODEL_NAME,) + tuple(item for item in DEFAULT_BENCHMARK_MODELS if item != DEFAULT_MODEL_NAME)
DEFAULT_STIM_REFRESH_RATE_HZ = 60.0
DEFAULT_STIM_MEAN = 0.5
DEFAULT_STIM_AMP = 0.5
DEFAULT_STIM_PHI = 0.0


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


def resolve_realtime_model_choice(selected_model: str, profile_model: str) -> tuple[str, bool]:
    selected = normalize_model_name(selected_model)
    profile = normalize_model_name(profile_model)
    return profile, selected != profile


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

    def request_stop(self) -> None:
        self._stop_event.set()

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
            ready = ensure_stream_ready(board, fs)
            probe_samples = max(int(round(profile.win_sec * fs)), 1)
            sample_matrix = board.get_current_board_data(max(probe_samples, ready))
            if sample_matrix.shape[1] < probe_samples:
                raise RuntimeError(
                    f"buffered probe window is too short: {sample_matrix.shape[1]}/{probe_samples}"
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
            self.phase_changed.emit(
                {
                    "mode": PHASE_VALIDATION,
                    "title": f"实时识别中（{profile.model_name}）",
                    "detail": "注视目标方块会输出结果；看中心点时不输出。",
                    "flicker": True,
                    "cue_freq": None,
                }
            )
            board.get_board_data()
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
    def __init__(self, *, serial_port: str, board_id: int, freqs: Sequence[float]) -> None:
        super().__init__()
        self.setWindowTitle("SSVEP 实时识别")
        self.resize(1260, 860)

        self.serial_port_default = normalize_serial_port(serial_port)
        self.board_id_default = int(board_id)
        self.freqs = tuple(float(freq) for freq in freqs)
        self._stim_refresh_rate_hz = _suggest_refresh_rate_hz()

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[RealtimeWorker] = None
        self.connect_thread: Optional[QThread] = None
        self.connect_worker: Optional[DeviceCheckWorker] = None
        self._last_signature: Optional[tuple[str, Optional[float]]] = None
        self._connecting = False

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left = QWidget(root)
        left_layout = QVBoxLayout(left)
        form = QFormLayout()

        self.serial_edit = QLineEdit(self.serial_port_default)
        self.board_edit = QLineEdit(str(self.board_id_default))
        self.freqs_edit = QLineEdit(",".join(f"{freq:g}" for freq in self.freqs))
        self.model_combo = QComboBox()
        for item in MODEL_OPTIONS:
            self.model_combo.addItem(item)
        self.model_combo.setCurrentText(DEFAULT_MODEL_NAME)
        self.model_combo.setToolTip("在线识别以 profile 内的 model_name/model_params 为准；下拉框仅用于启动前一致性提示。")
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
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_load_profile.setText("加载 Profile")
        self.btn_connect.setText("连接设备")
        self.btn_start.setText("开始实时识别")
        self.btn_stop.setText("停止")
        row.addWidget(self.btn_load_profile)
        row.addWidget(self.btn_connect)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        left_layout.addLayout(row)

        self.phase_label = QLabel("空闲")
        self.phase_label.setStyleSheet("font-size:16px; font-weight:600;")
        left_layout.addWidget(self.phase_label)

        self.result_label = QLabel("输出频率：None")
        self.result_label.setStyleSheet("font-size:18px; font-weight:600;")
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
        right_layout = QVBoxLayout(right)
        self.stim = FourArrowStimWidget(
            freqs=self.freqs,
            refresh_rate_hz=self._stim_refresh_rate_hz,
            mean=DEFAULT_STIM_MEAN,
            amp=DEFAULT_STIM_AMP,
            phi=DEFAULT_STIM_PHI,
        )
        right_layout.addWidget(self.stim, 1)

        layout.addWidget(left, 0)
        layout.addWidget(right, 1)

        self.btn_load_profile.clicked.connect(self._pick_profile)
        self.btn_connect.clicked.connect(self._connect_device)
        self.btn_start.clicked.connect(self._start_realtime)
        self.btn_stop.clicked.connect(self._stop_realtime)

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{stamp}] {text}")

    def _set_running(self, running: bool) -> None:
        self.btn_connect.setEnabled((not running) and (not self._connecting))
        self.btn_start.setEnabled((not running) and (not self._connecting))
        self.btn_stop.setEnabled(running)
        self.shadow_mode_check.setEnabled(not running)

    def _set_connecting(self, connecting: bool) -> None:
        self._connecting = bool(connecting)
        if connecting:
            self.btn_connect.setEnabled(False)
            self.btn_start.setEnabled(False)
            self.phase_label.setText("连接中...")
        else:
            self.btn_connect.setEnabled(self.worker_thread is None)
            self.btn_start.setEnabled(self.worker_thread is None)

    def _pick_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 Profile", str(Path(self.profile_edit.text()).parent), "JSON (*.json)")
        if path:
            self.profile_edit.setText(path)

    def _read_config(self) -> RealtimeConfig:
        serial_port = normalize_serial_port(self.serial_edit.text().strip())
        board_id = int(self.board_edit.text().strip())
        freqs = parse_freqs(self.freqs_edit.text().strip())
        model_name = normalize_model_name(self.model_combo.currentText())
        profile_path = Path(self.profile_edit.text().strip()).expanduser().resolve()
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
        )

    def _connect_device(self) -> None:
        if self.connect_thread is not None or self._connecting:
            return
        if self.worker_thread is not None:
            self._log("实时识别运行中，请先停止后再重连设备。")
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

    def _start_realtime(self) -> None:
        if self.worker_thread is not None:
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
        worker = RealtimeWorker(cfg)
        thread = QThread(self)
        worker.moveToThread(thread)
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
        self._last_signature = None
        self.phase_label.setText("正在启动实时识别...")
        thread.start()

    def _stop_realtime(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
        self._set_running(False)

    def _on_result(self, payload: dict[str, Any]) -> None:
        signature = (str(payload.get("state", "")), payload.get("selected_freq"))
        if signature == self._last_signature:
            return
        self._last_signature = signature
        self.result_label.setText(
            "输出频率={selected_freq} | 状态={state} | top1={top1:.3f} ratio={ratio:.3f}".format(
                selected_freq=payload.get("selected_freq"),
                state=payload.get("state"),
                top1=float(payload.get("top1_score", 0.0)),
                ratio=float(payload.get("ratio", 0.0)),
            )
        )
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

    def _on_finished(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._set_running(False)

    def closeEvent(self, event: QCloseEvent) -> None:
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
    parser.add_argument("--emit-all", action="store_true")
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="仅命令行运行，不启动 UI")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.headless):
        runner = OnlineRunner(
            serial_port=args.serial_port,
            board_id=args.board_id,
            freqs=parse_freqs(args.freqs),
            profile_path=Path(args.profile).expanduser().resolve(),
            emit_all=bool(args.emit_all),
            model_name=str(args.model),
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
    )
    window.profile_edit.setText(str(Path(args.profile).expanduser().resolve()))
    window.model_combo.setCurrentText(normalize_model_name(args.model))
    window.compute_backend_combo.setCurrentText(parse_compute_backend_name(str(args.compute_backend).strip()))
    window.gpu_device_edit.setText(str(int(args.gpu_device)))
    window.gpu_precision_combo.setCurrentText(parse_gpu_precision(str(args.gpu_precision).strip()))
    window.gpu_warmup_edit.setText("1" if bool(int(args.gpu_warmup)) else "0")
    window.gpu_cache_combo.setCurrentText(parse_gpu_cache_policy(str(args.gpu_cache_policy).strip()))
    window.shadow_mode_check.setChecked(bool(int(args.shadow_mode)))
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())

