from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .async_fbcca_idle import (
    AsyncDecisionGate,
    BoardShim,
    DEFAULT_BENCHMARK_MODELS,
    DEFAULT_BOARD_ID,
    DEFAULT_CALIBRATION_SEED,
    DEFAULT_MODEL_NAME,
    DEFAULT_NH,
    DEFAULT_PROFILE_PATH,
    DEFAULT_STEP_SEC,
    DEFAULT_STREAM_WARMUP_SEC,
    DEFAULT_WIN_SEC,
    TrialSpec,
    build_calibration_trials,
    build_feature_rows_with_decoder,
    create_decoder,
    describe_runtime_error,
    ensure_stream_ready,
    fit_threshold_profile,
    format_profile_quality_summary,
    load_decoder_from_profile,
    load_profile,
    normalize_model_name,
    normalize_serial_port,
    parse_freqs,
    prepare_board_session,
    profile_is_default_fallback,
    read_recent_eeg_segment,
    resolve_selected_eeg_channels,
    save_profile,
    select_auto_eeg_channels,
    summarize_profile_quality,
    validate_calibration_plan,
)
from .validation_ui import (
    FourArrowStimWidget,
    PHASE_CAL_ACTIVE,
    PHASE_CAL_PREPARE,
    PHASE_CAL_REST,
    PHASE_ERROR,
    PHASE_IDLE,
    PHASE_STOPPED,
    PHASE_VALIDATION,
    direction_label,
)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SINGLE_PROFILE_PATH = THIS_DIR / "profiles" / "single_model_profile.json"
MODEL_OPTIONS = (DEFAULT_MODEL_NAME,) + tuple(
    item for item in DEFAULT_BENCHMARK_MODELS if item != DEFAULT_MODEL_NAME
)
DEFAULT_ACTIVE_SEC = 4.0
DEFAULT_PREPARE_SEC = 1.0
DEFAULT_REST_SEC = 1.0
DEFAULT_TARGET_REPEATS = 5
DEFAULT_IDLE_REPEATS = 10
DEFAULT_MAX_TRANSIENT_READ_ERRORS = 5


def default_single_model_profile_path() -> Path:
    return DEFAULT_SINGLE_PROFILE_PATH


def _subset_trial_segments_by_channels(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    available_board_channels: Sequence[int],
    selected_channels: Sequence[int],
) -> list[tuple[TrialSpec, np.ndarray]]:
    available = tuple(int(channel) for channel in available_board_channels)
    selected = tuple(int(channel) for channel in selected_channels)
    positions = [available.index(channel) for channel in selected]
    subset: list[tuple[TrialSpec, np.ndarray]] = []
    for trial, segment in trial_segments:
        subset_segment = np.ascontiguousarray(segment[:, positions], dtype=np.float64)
        subset.append((trial, subset_segment))
    return subset


def fit_single_model_profile_from_segments(
    *,
    model_name: str,
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    available_board_channels: Sequence[int],
    sampling_rate: int,
    freqs: Sequence[float],
    active_sec: float,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float = DEFAULT_STEP_SEC,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> tuple[Any, dict[str, Any]]:
    normalized_model = normalize_model_name(model_name)
    selected_channels, channel_scores = select_auto_eeg_channels(
        trial_segments,
        available_board_channels=available_board_channels,
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=win_sec,
        step_sec=step_sec,
        seed=seed,
    )
    subset_segments = _subset_trial_segments_by_channels(
        trial_segments,
        available_board_channels=available_board_channels,
        selected_channels=selected_channels,
    )
    decoder = create_decoder(
        normalized_model,
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=win_sec,
        step_sec=step_sec,
        model_params={"Nh": DEFAULT_NH},
    )
    if decoder.requires_fit:
        decoder.fit(subset_segments)

    feature_rows = build_feature_rows_with_decoder(decoder, subset_segments)
    profile = fit_threshold_profile(
        feature_rows,
        freqs=freqs,
        win_sec=win_sec,
        step_sec=step_sec,
        min_enter_windows=2,
        min_exit_windows=2,
    )
    quality_summary = summarize_profile_quality(feature_rows, profile)
    model_params = dict(decoder.model_params)
    model_params["state"] = decoder.get_state()
    metadata = {
        "source": "single_model_ui",
        "model_name": normalized_model,
        "selected_eeg_channels": [int(channel) for channel in selected_channels],
        "channel_selection": channel_scores,
        "quality_summary": quality_summary,
        "active_sec": float(active_sec),
    }
    fitted_profile = replace(
        profile,
        model_name=normalized_model,
        model_params=model_params,
        calibration_split_seed=int(seed),
        benchmark_metrics=None,
        eeg_channels=tuple(int(channel) for channel in selected_channels),
        metadata=metadata,
    )
    return fitted_profile, metadata


@dataclass(frozen=True)
class SingleModelConfig:
    serial_port: str
    board_id: int
    freqs: tuple[float, float, float, float]
    model_name: str
    profile_path: Path
    prepare_sec: float
    active_sec: float
    rest_sec: float
    target_repeats: int
    idle_repeats: int
    win_sec: float
    step_sec: float
    allow_default_profile: bool = False


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
            self.error.emit(f"Connect failed: {describe_runtime_error(exc, serial_port=self.serial_port)}")
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


class PretrainWorker(QObject):
    phase_changed = pyqtSignal(object)
    log = pyqtSignal(str)
    profile_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: SingleModelConfig) -> None:
        super().__init__()
        self.config = config
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _emit_stopped_phase(self) -> None:
        self.phase_changed.emit(
            {
                "mode": PHASE_STOPPED,
                "title": "Pretrain stopped",
                "detail": "Stopped by user.",
                "flicker": False,
                "cue_freq": None,
            }
        )

    def _wait_phase(self, payload: dict[str, Any], duration_sec: float) -> bool:
        if duration_sec <= 0:
            self.phase_changed.emit({**payload, "remaining_sec": 0})
            return not self._stop_event.is_set()
        deadline = time.perf_counter() + float(duration_sec)
        last_sec = None
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

    @pyqtSlot()
    def run(self) -> None:
        board = None
        active_serial = self.config.serial_port
        try:
            validate_calibration_plan(
                target_repeats=self.config.target_repeats,
                idle_repeats=self.config.idle_repeats,
                active_sec=self.config.active_sec,
                preferred_win_sec=self.config.win_sec,
                step_sec=self.config.step_sec,
            )
            board, resolved_port, attempted = prepare_board_session(self.config.board_id, self.config.serial_port)
            active_serial = resolved_port
            self.log.emit(
                f"Connected: requested={self.config.serial_port} -> {resolved_port}; attempts={attempted}"
            )
            fs = int(BoardShim.get_sampling_rate(self.config.board_id))
            eeg_channels = tuple(int(ch) for ch in BoardShim.get_eeg_channels(self.config.board_id))
            active_samples = int(round(self.config.active_sec * fs))
            min_samples = int(round(self.config.win_sec * fs))
            if active_samples < min_samples:
                raise ValueError("active_sec must be at least win_sec")

            board.start_stream(450000)
            ready_samples = ensure_stream_ready(board, fs)
            self.log.emit(f"stream ready | fs={fs}Hz | channels={list(eeg_channels)} | buffer={ready_samples}")
            time.sleep(max(2.0, DEFAULT_STREAM_WARMUP_SEC))
            board.get_board_data()

            trials = build_calibration_trials(
                self.config.freqs,
                target_repeats=self.config.target_repeats,
                idle_repeats=self.config.idle_repeats,
                shuffle=True,
                seed=DEFAULT_CALIBRATION_SEED,
            )
            segments: list[tuple[TrialSpec, np.ndarray]] = []
            for index, trial in enumerate(trials, start=1):
                total = len(trials)
                prompt = (
                    f"Focus {direction_label(self.config.freqs, trial.expected_freq)}"
                    if trial.expected_freq is not None
                    else "Look center and avoid all targets"
                )
                if not self._wait_phase(
                    {
                        "mode": PHASE_CAL_PREPARE,
                        "title": f"Pretrain {index}/{total}",
                        "detail": f"{prompt} | Prepare",
                        "flicker": False,
                        "cue_freq": trial.expected_freq,
                    },
                    self.config.prepare_sec,
                ):
                    self._emit_stopped_phase()
                    return

                board.get_board_data()
                if not self._wait_phase(
                    {
                        "mode": PHASE_CAL_ACTIVE,
                        "title": f"Pretrain {index}/{total}",
                        "detail": f"{prompt} | Collect",
                        "flicker": True,
                        "cue_freq": trial.expected_freq,
                    },
                    self.config.active_sec,
                ):
                    self._emit_stopped_phase()
                    return
                segment, _, _ = read_recent_eeg_segment(
                    board,
                    eeg_channels,
                    target_samples=active_samples,
                    minimum_samples=min_samples,
                )
                segments.append((trial, segment))
                self.log.emit(f"Trial {index}/{total} done: {trial.label}")

                if not self._wait_phase(
                    {
                        "mode": PHASE_CAL_REST,
                        "title": f"Pretrain {index}/{total}",
                        "detail": "Rest",
                        "flicker": False,
                        "cue_freq": None,
                    },
                    self.config.rest_sec,
                ):
                    self._emit_stopped_phase()
                    return
                board.get_board_data()

            profile, metadata = fit_single_model_profile_from_segments(
                model_name=self.config.model_name,
                trial_segments=segments,
                available_board_channels=eeg_channels,
                sampling_rate=fs,
                freqs=self.config.freqs,
                active_sec=self.config.active_sec,
                win_sec=self.config.win_sec,
                step_sec=self.config.step_sec,
                seed=DEFAULT_CALIBRATION_SEED,
            )
            save_profile(profile, self.config.profile_path)
            summary = metadata.get("quality_summary", {})
            self.profile_ready.emit(
                {
                    "profile": asdict(profile),
                    "profile_path": str(self.config.profile_path),
                    "summary": summary,
                    "summary_text": format_profile_quality_summary(summary),
                    "model_name": self.config.model_name,
                    "selected_eeg_channels": metadata.get("selected_eeg_channels"),
                }
            )
            self.phase_changed.emit(
                {
                    "mode": PHASE_IDLE,
                    "title": "Pretrain completed",
                    "detail": f"Profile saved to {self.config.profile_path}",
                    "flicker": False,
                    "cue_freq": None,
                }
            )
        except Exception as exc:
            self.error.emit(f"Pretrain failed: {describe_runtime_error(exc, serial_port=active_serial)}")
            self.phase_changed.emit(
                {
                    "mode": PHASE_ERROR,
                    "title": "Pretrain error",
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


class OnlineWorker(QObject):
    phase_changed = pyqtSignal(object)
    log = pyqtSignal(str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: SingleModelConfig) -> None:
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
            if profile_is_default_fallback(profile) and not bool(self.config.allow_default_profile):
                raise RuntimeError("profile is default fallback; run pretrain first")
            board, resolved_port, attempted = prepare_board_session(self.config.board_id, self.config.serial_port)
            active_serial = resolved_port
            self.log.emit(
                f"Connected: requested={self.config.serial_port} -> {resolved_port}; attempts={attempted}"
            )
            if profile_is_default_fallback(profile):
                self.log.emit("Using fallback FBCCA profile without user pretrain; performance may be reduced.")
            fs = int(BoardShim.get_sampling_rate(self.config.board_id))
            eeg_channels = resolve_selected_eeg_channels(
                BoardShim.get_eeg_channels(self.config.board_id),
                profile.eeg_channels,
            )
            decoder = load_decoder_from_profile(profile, sampling_rate=fs)
            decoder.configure_runtime(fs)
            gate = AsyncDecisionGate.from_profile(profile)
            board.start_stream(450000)
            ready = ensure_stream_ready(board, fs)
            self.log.emit(
                f"Online started | model={profile.model_name} | fs={fs}Hz | channels={list(eeg_channels)} | buffer={ready}"
            )
            self.phase_changed.emit(
                {
                    "mode": PHASE_VALIDATION,
                    "title": f"Online running ({profile.model_name})",
                    "detail": "Focus target block to output; look center for no output.",
                    "flicker": True,
                    "cue_freq": None,
                }
            )
            time.sleep(max(2.0, DEFAULT_STREAM_WARMUP_SEC))
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
                    decision = gate.update(decoder.analyze_window(eeg))
                    t1 = time.perf_counter()
                    decoder.update_online(decision, eeg)
                    consecutive_errors = 0
                except Exception as exc:
                    consecutive_errors += 1
                    self.log.emit(f"Online transient read error {consecutive_errors}: {exc}")
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
                    "model_name": str(profile.model_name),
                    "decision_latency_ms": float((t1 - t0) * 1000.0),
                }
                payload["pred_label"] = direction_label(profile.freqs, payload["pred_freq"])
                payload["selected_label"] = direction_label(profile.freqs, payload["selected_freq"])
                self.result.emit(payload)
                self._stop_event.wait(decoder.step_sec)

            self.phase_changed.emit(
                {
                    "mode": PHASE_STOPPED,
                    "title": "Online stopped",
                    "detail": "You can restart online classification.",
                    "flicker": False,
                    "cue_freq": None,
                }
            )
        except Exception as exc:
            self.error.emit(f"Online failed: {describe_runtime_error(exc, serial_port=active_serial)}")
            self.phase_changed.emit(
                {
                    "mode": PHASE_ERROR,
                    "title": "Online error",
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


class SingleModelWindow(QMainWindow):
    def __init__(self, *, serial_port: str, board_id: int, freqs: tuple[float, float, float, float]) -> None:
        super().__init__()
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[QObject] = None
        self.connect_thread: Optional[QThread] = None
        self.connect_worker: Optional[DeviceCheckWorker] = None
        self.current_profile: Optional[Any] = None
        self.last_result_signature: Optional[tuple[str, Optional[float]]] = None

        self.initial_serial = normalize_serial_port(serial_port)
        self.initial_board_id = int(board_id)
        self.initial_freqs = tuple(float(freq) for freq in freqs)
        self.detected_refresh_hz = self._detect_refresh_rate()
        self._build_ui()

    @staticmethod
    def _detect_refresh_rate() -> float:
        app = QApplication.instance()
        if app is None:
            return 60.0
        screen = app.primaryScreen()
        if screen is None:
            return 60.0
        refresh = float(screen.refreshRate() or 0.0)
        if not np.isfinite(refresh) or refresh < 30.0 or refresh > 360.0:
            return 60.0
        return refresh

    def _build_ui(self) -> None:
        self.setWindowTitle("SSVEP Single-model Pretrain + Online UI")
        self.resize(1460, 900)
        root = QWidget(self)
        self.setCentralWidget(root)
        root.setStyleSheet(
            """
            QWidget { background-color: black; color: #dce5ea; }
            QLabel { color: #dce5ea; }
            QLineEdit, QSpinBox, QComboBox, QPlainTextEdit {
                background-color: #121417;
                color: #dce5ea;
                border: 1px solid #2d3943;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton {
                background-color: #1b242b;
                color: #eaf0f4;
                border: 1px solid #33414c;
                border-radius: 5px;
                padding: 6px 12px;
            }
            QPushButton:disabled {
                color: #7b8893;
                background-color: #11161a;
                border: 1px solid #202830;
            }
            """
        )
        layout = QHBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left.setMaximumWidth(420)

        form = QFormLayout()
        self.serial_edit = QLineEdit(self.initial_serial)
        self.board_spin = QSpinBox()
        self.board_spin.setRange(-1, 9999)
        self.board_spin.setValue(self.initial_board_id)
        self.freqs_edit = QLineEdit(",".join(f"{freq:g}" for freq in self.initial_freqs))
        self.model_combo = QComboBox()
        for item in MODEL_OPTIONS:
            self.model_combo.addItem(item)
        self.model_combo.setCurrentText(DEFAULT_MODEL_NAME)
        self.profile_edit = QLineEdit(str(default_single_model_profile_path()))
        self.prepare_spin = QSpinBox()
        self.prepare_spin.setRange(0, 10)
        self.prepare_spin.setValue(int(DEFAULT_PREPARE_SEC))
        self.active_spin = QSpinBox()
        self.active_spin.setRange(1, 20)
        self.active_spin.setValue(int(DEFAULT_ACTIVE_SEC))
        self.rest_spin = QSpinBox()
        self.rest_spin.setRange(0, 10)
        self.rest_spin.setValue(int(DEFAULT_REST_SEC))
        self.target_repeat_spin = QSpinBox()
        self.target_repeat_spin.setRange(1, 20)
        self.target_repeat_spin.setValue(DEFAULT_TARGET_REPEATS)
        self.idle_repeat_spin = QSpinBox()
        self.idle_repeat_spin.setRange(1, 40)
        self.idle_repeat_spin.setValue(DEFAULT_IDLE_REPEATS)
        self._config_widgets = [
            self.serial_edit,
            self.board_spin,
            self.freqs_edit,
            self.model_combo,
            self.profile_edit,
            self.prepare_spin,
            self.active_spin,
            self.rest_spin,
            self.target_repeat_spin,
            self.idle_repeat_spin,
        ]

        form.addRow("Serial", self.serial_edit)
        form.addRow("Board ID", self.board_spin)
        form.addRow("Freqs", self.freqs_edit)
        form.addRow("Model", self.model_combo)
        form.addRow("Profile", self.profile_edit)
        form.addRow("Prepare(s)", self.prepare_spin)
        form.addRow("Active(s)", self.active_spin)
        form.addRow("Rest(s)", self.rest_spin)
        form.addRow("Target repeats", self.target_repeat_spin)
        form.addRow("Idle repeats", self.idle_repeat_spin)
        left_layout.addLayout(form)

        btn1 = QHBoxLayout()
        self.btn_connect = QPushButton("连接设备")
        self.btn_pretrain = QPushButton("开始预训练")
        btn1.addWidget(self.btn_connect)
        btn1.addWidget(self.btn_pretrain)
        left_layout.addLayout(btn1)

        btn2 = QHBoxLayout()
        self.btn_online = QPushButton("开始在线")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        btn2.addWidget(self.btn_online)
        btn2.addWidget(self.btn_stop)
        left_layout.addLayout(btn2)

        btn3 = QHBoxLayout()
        self.btn_load = QPushButton("加载profile")
        self.btn_save = QPushButton("保存profile")
        btn3.addWidget(self.btn_load)
        btn3.addWidget(self.btn_save)
        left_layout.addLayout(btn3)

        self.output_label = QLabel("No output")
        self.output_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #dce5ea;")
        left_layout.addWidget(self.output_label)

        self.metrics_label = QLabel("Ready")
        self.metrics_label.setWordWrap(True)
        left_layout.addWidget(self.metrics_label)

        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(220)
        left_layout.addWidget(self.summary_text, 1)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(240)
        left_layout.addWidget(self.log_text, 1)
        layout.addWidget(left, 0)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)
        self.phase_label = QLabel("Ready")
        self.phase_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #89d7ff;")
        center_layout.addWidget(self.phase_label, 0)
        self.stim_widget = FourArrowStimWidget(
            freqs=self.initial_freqs,
            refresh_rate_hz=self.detected_refresh_hz,
            mean=0.5,
            amp=0.5,
            phi=0.0,
        )
        center_layout.addWidget(self.stim_widget, 1)
        layout.addWidget(center, 1)

        self.btn_connect.clicked.connect(self.connect_device)
        self.btn_pretrain.clicked.connect(self.start_pretrain)
        self.btn_online.clicked.connect(self.start_online)
        self.btn_stop.clicked.connect(self.stop_current_worker)
        self.btn_load.clicked.connect(self.load_profile_from_file)
        self.btn_save.clicked.connect(self.save_profile_to_file)
        self._set_summary(
            f"Ready. 连接设备 -> 开始预训练 -> 开始在线\n"
            f"Stim refresh rate: {self.detected_refresh_hz:.1f} Hz"
        )

    def _set_summary(self, text: str) -> None:
        self.summary_text.setPlainText(str(text))

    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{stamp}] {message}")
        bar = self.log_text.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _collect_config(self) -> SingleModelConfig:
        freqs = parse_freqs(self.freqs_edit.text())
        return SingleModelConfig(
            serial_port=normalize_serial_port(self.serial_edit.text().strip()),
            board_id=int(self.board_spin.value()),
            freqs=freqs,
            model_name=normalize_model_name(self.model_combo.currentText()),
            profile_path=Path(self.profile_edit.text().strip()).expanduser().resolve(),
            prepare_sec=float(self.prepare_spin.value()),
            active_sec=float(self.active_spin.value()),
            rest_sec=float(self.rest_spin.value()),
            target_repeats=int(self.target_repeat_spin.value()),
            idle_repeats=int(self.idle_repeat_spin.value()),
            win_sec=float(DEFAULT_WIN_SEC),
            step_sec=float(DEFAULT_STEP_SEC),
        )

    def _set_running(self, running: bool) -> None:
        self.btn_pretrain.setEnabled(not running)
        self.btn_online.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_connect.setEnabled(not running and self.connect_thread is None)
        self.btn_load.setEnabled(not running)
        self.btn_save.setEnabled(not running)
        for widget in self._config_widgets:
            widget.setEnabled(not running and self.connect_thread is None)

    def _sync_stim_freqs(self, freqs: Sequence[float]) -> None:
        normalized = tuple(float(freq) for freq in freqs)
        if tuple(self.stim_widget.freqs) != normalized:
            self.stim_widget.freqs = normalized
            self.stim_widget.update()

    def connect_device(self) -> None:
        if self.worker_thread is not None:
            self._log("Worker is running; stop first.")
            return
        if self.connect_thread is not None:
            self._log("Device connection in progress.")
            return
        serial_port = normalize_serial_port(self.serial_edit.text().strip())
        board_id = int(self.board_spin.value())
        self.connect_thread = QThread(self)
        self.connect_worker = DeviceCheckWorker(serial_port=serial_port, board_id=board_id)
        self.connect_worker.moveToThread(self.connect_thread)
        self.connect_thread.started.connect(self.connect_worker.run)
        self.connect_worker.connected.connect(self.on_connected)
        self.connect_worker.error.connect(self.on_connect_error)
        self.connect_worker.finished.connect(self.connect_thread.quit)
        self.connect_thread.finished.connect(self.connect_thread.deleteLater)
        self.connect_thread.finished.connect(self._cleanup_connect_worker)
        self.btn_connect.setEnabled(False)
        for widget in self._config_widgets:
            widget.setEnabled(False)
        self._log(f"Connecting device serial={serial_port}, board_id={board_id}...")
        self.connect_thread.start()

    @pyqtSlot(object)
    def on_connected(self, payload: dict[str, Any]) -> None:
        resolved = str(payload.get("resolved_serial_port", ""))
        if resolved:
            self.serial_edit.setText(resolved)
        self._log(
            f"Connected: requested={payload.get('requested_serial_port')} -> {resolved}, "
            f"fs={payload.get('sampling_rate')}Hz"
        )

    @pyqtSlot(str)
    def on_connect_error(self, message: str) -> None:
        self._log(message)

    @pyqtSlot()
    def _cleanup_connect_worker(self) -> None:
        self.connect_worker = None
        self.connect_thread = None
        self.btn_connect.setEnabled(self.worker_thread is None)
        for widget in self._config_widgets:
            widget.setEnabled(self.worker_thread is None)

    def _start_worker(self, worker_obj: QObject) -> None:
        self.worker_thread = QThread(self)
        self.worker = worker_obj
        worker_obj.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(worker_obj.run)
        if hasattr(worker_obj, "phase_changed"):
            worker_obj.phase_changed.connect(self.on_phase_changed)  # type: ignore[attr-defined]
        if hasattr(worker_obj, "log"):
            worker_obj.log.connect(self._log)  # type: ignore[attr-defined]
        if hasattr(worker_obj, "error"):
            worker_obj.error.connect(self.on_worker_error)  # type: ignore[attr-defined]
        worker_obj.finished.connect(self.on_worker_finished)  # type: ignore[attr-defined]
        worker_obj.finished.connect(self.worker_thread.quit)  # type: ignore[attr-defined]
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self._set_running(True)
        self.worker_thread.start()

    def start_pretrain(self) -> None:
        if self.worker_thread is not None or self.connect_thread is not None:
            self._log("Another task is running.")
            return
        try:
            config = self._collect_config()
        except Exception as exc:
            self._log(f"Invalid config: {exc}")
            return
        self._sync_stim_freqs(config.freqs)
        worker = PretrainWorker(config)
        worker.profile_ready.connect(self.on_profile_ready)
        self.phase_label.setText("Pretraining...")
        self._start_worker(worker)

    def start_online(self) -> None:
        if self.worker_thread is not None or self.connect_thread is not None:
            self._log("Another task is running.")
            return
        try:
            config = self._collect_config()
        except Exception as exc:
            self._log(f"Invalid config: {exc}")
            return
        try:
            loaded = load_profile(config.profile_path, fallback_freqs=config.freqs, require_exists=True)
            self._sync_stim_freqs(loaded.freqs)
        except Exception:
            self._sync_stim_freqs(config.freqs)
        worker = OnlineWorker(config)
        worker.result.connect(self.on_result)
        self.phase_label.setText("Online running...")
        self._start_worker(worker)

    def stop_current_worker(self) -> None:
        worker = self.worker
        if worker is None:
            self._log("No active worker.")
            return
        if hasattr(worker, "request_stop"):
            worker.request_stop()  # type: ignore[attr-defined]
        self._log("Stop requested.")

    @pyqtSlot(object)
    def on_phase_changed(self, payload: dict[str, Any]) -> None:
        self.phase_label.setText(str(payload.get("title", "")))
        self.stim_widget.apply_phase(payload)

    @pyqtSlot(object)
    def on_profile_ready(self, payload: dict[str, Any]) -> None:
        profile_data = payload.get("profile", {})
        profile_path = payload.get("profile_path", "")
        loaded = load_profile(Path(profile_path), require_exists=True)
        self.current_profile = loaded
        self.profile_edit.setText(str(profile_path))
        if "model_name" in profile_data:
            self.model_combo.setCurrentText(str(profile_data["model_name"]))
        summary_text = str(payload.get("summary_text", "")).strip()
        selected_channels = payload.get("selected_eeg_channels")
        text = (
            f"Pretrain done.\nProfile: {profile_path}\n"
            f"Model: {profile_data.get('model_name')}\n"
            f"Channels: {selected_channels}\n\n{summary_text}"
        )
        self._set_summary(text)

    @pyqtSlot(object)
    def on_result(self, payload: dict[str, Any]) -> None:
        self.stim_widget.apply_result(payload)
        state = str(payload.get("state", "idle"))
        selected_label = str(payload.get("selected_label", "No output"))
        pred_label = str(payload.get("pred_label", ""))
        if state == "selected":
            self.output_label.setText(selected_label)
            self.output_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #55f0a0;")
        elif state == "candidate":
            self.output_label.setText(f"Candidate: {pred_label}")
            self.output_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #ffd65d;")
        else:
            self.output_label.setText("No output")
            self.output_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #dce5ea;")
        self.metrics_label.setText(
            f"Model={payload.get('model_name')} | state={state}\n"
            f"top1={float(payload.get('top1_score', 0.0)):.5f} | top2={float(payload.get('top2_score', 0.0)):.5f}\n"
            f"margin={float(payload.get('margin', 0.0)):.5f} | ratio={float(payload.get('ratio', 0.0)):.3f}\n"
            f"stable_windows={int(payload.get('stable_windows', 0))} | "
            f"infer={float(payload.get('decision_latency_ms', 0.0)):.2f}ms"
        )
        signature = (state, payload.get("selected_freq"))
        if signature != self.last_result_signature:
            self.last_result_signature = signature

    @pyqtSlot(str)
    def on_worker_error(self, message: str) -> None:
        self._log(message)
        self.phase_label.setText("Worker error")

    @pyqtSlot()
    def on_worker_finished(self) -> None:
        self._log("Worker finished.")

    @pyqtSlot()
    def _cleanup_worker(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._set_running(False)

    def load_profile_from_file(self) -> None:
        start = str(Path(self.profile_edit.text()).expanduser().resolve())
        path, _ = QFileDialog.getOpenFileName(self, "Load profile", start, "JSON (*.json)")
        if not path:
            return
        loaded = load_profile(Path(path), require_exists=True)
        self.current_profile = loaded
        self.profile_edit.setText(str(Path(path).resolve()))
        self.model_combo.setCurrentText(str(loaded.model_name))
        self._set_summary(
            f"Loaded profile: {path}\nModel: {loaded.model_name}\n"
            f"enter_score_th={loaded.enter_score_th:.6f}\n"
            f"enter_ratio_th={loaded.enter_ratio_th:.6f}\n"
            f"enter_margin_th={loaded.enter_margin_th:.6f}"
        )
        self._log(f"Loaded profile: {path}")

    def save_profile_to_file(self) -> None:
        if self.current_profile is None:
            self._log("No in-memory profile. Run pretrain or load profile first.")
            return
        start = str(Path(self.profile_edit.text()).expanduser().resolve())
        path, _ = QFileDialog.getSaveFileName(self, "Save profile", start, "JSON (*.json)")
        if not path:
            return
        target = Path(path).expanduser().resolve()
        save_profile(self.current_profile, target)
        self.profile_edit.setText(str(target))
        self._log(f"Profile saved: {target}")

    def closeEvent(self, event) -> None:
        worker = self.worker
        if worker is not None and hasattr(worker, "request_stop"):
            worker.request_stop()  # type: ignore[attr-defined]
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(3000)
        if self.connect_thread is not None:
            self.connect_thread.quit()
            self.connect_thread.wait(3000)
        self.stim_widget.stop_clock()
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSVEP single-model pretrain + online UI")
    parser.add_argument("--serial-port", type=str, default="auto")
    parser.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    parser.add_argument("--freqs", type=str, default="8,10,12,15")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication([])
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = SingleModelWindow(
        serial_port=args.serial_port,
        board_id=args.board_id,
        freqs=parse_freqs(args.freqs),
    )
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
