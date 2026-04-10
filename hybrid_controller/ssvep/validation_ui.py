from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QObject, QRect, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .async_fbcca_idle import (
    DEFAULT_MAX_CALIBRATION_TRIAL_ERRORS,
    DEFAULT_MAX_TRANSIENT_READ_ERRORS,
    DEFAULT_BOARD_ID,
    DEFAULT_CALIBRATION_SEED,
    DEFAULT_SERIAL_PORT,
    DEFAULT_PROFILE_PATH,
    DEFAULT_STREAM_WARMUP_SEC,
    AsyncDecisionGate,
    BoardShim,
    build_calibration_trials as build_calibration_trials_core,
    describe_runtime_error,
    ensure_stream_ready,
    FBCCAEngine,
    format_profile_quality_summary,
    optimize_profile_from_segments,
    ThresholdProfile,
    TrialSpec,
    fit_threshold_profile,
    load_decoder_from_profile,
    load_profile,
    normalize_serial_port,
    parse_freqs,
    prepare_board_session,
    profile_is_default_fallback,
    profile_log_lr,
    read_recent_eeg_segment,
    resolve_selected_eeg_channels,
    require_brainflow,
    save_profile,
    serial_port_is_auto,
    summarize_profile_quality,
    validate_calibration_plan,
)


DIRECTION_CN = ("UP", "LEFT", "DOWN", "RIGHT")
DIRECTION_EN = ("UP", "LEFT", "DOWN", "RIGHT")
PHASE_IDLE = "idle"
PHASE_CAL_PREPARE = "calibration_prepare"
PHASE_CAL_ACTIVE = "calibration_active"
PHASE_CAL_REST = "calibration_rest"
PHASE_VALIDATION = "validation"
PHASE_ERROR = "error"
PHASE_STOPPED = "stopped"


def freq_index(freqs: Sequence[float], target_freq: Optional[float]) -> Optional[int]:
    if target_freq is None:
        return None
    for index, freq in enumerate(freqs):
        if abs(float(freq) - float(target_freq)) < 1e-8:
            return index
    return None


def direction_label(freqs: Sequence[float], target_freq: Optional[float]) -> str:
    index = freq_index(freqs, target_freq)
    if index is None:
        return "No output"
    return f"{DIRECTION_CN[index]} / {float(freqs[index]):g}Hz"


def build_calibration_trials(
    freqs: Sequence[float],
    *,
    target_repeats: int,
    idle_repeats: int,
) -> list[TrialSpec]:
    return build_calibration_trials_core(
        freqs,
        target_repeats=target_repeats,
        idle_repeats=idle_repeats,
        shuffle=True,
        seed=DEFAULT_CALIBRATION_SEED,
    )


def evaluate_profile_quality(
    feature_rows: Sequence[dict[str, Any]],
    profile: ThresholdProfile,
) -> dict[str, float]:
    return summarize_profile_quality(feature_rows, profile)


def format_quality_summary(summary: dict[str, float]) -> str:
    return format_profile_quality_summary(summary)


def trial_prompt(freqs: Sequence[float], trial: TrialSpec) -> str:
    if trial.expected_freq is None:
        return "Look at center and avoid all flicker targets."
    return f"Focus {direction_label(freqs, trial.expected_freq)}"


@dataclass(frozen=True)
class WorkflowConfig:
    serial_port: str
    board_id: int
    sampling_rate: int
    refresh_rate_hz: float
    freqs: tuple[float, float, float, float]
    profile_path: Path
    prepare_sec: float
    active_sec: float
    rest_sec: float
    target_repeats: int
    idle_repeats: int
    win_sec: float
    step_sec: float
    stim_mean: float
    stim_amp: float
    stim_phi: float


class WindowsTimerResolution:
    def __init__(self) -> None:
        self._winmm = None

    def __enter__(self) -> "WindowsTimerResolution":
        if sys.platform.startswith("win"):
            try:
                self._winmm = ctypes.WinDLL("winmm")
                self._winmm.timeBeginPeriod(1)
            except Exception:
                self._winmm = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._winmm is not None:
            try:
                self._winmm.timeEndPeriod(1)
            except Exception:
                pass
        return False


class StimClockThread(QThread):
    tick = pyqtSignal(float, int)

    def __init__(self, refresh_rate_hz: float) -> None:
        super().__init__()
        self.refresh_rate_hz = float(refresh_rate_hz)
        if self.refresh_rate_hz <= 0:
            raise ValueError("refresh_rate_hz must be positive")
        self.period_ns = int(round(1_000_000_000 / self.refresh_rate_hz))
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self._stop_event.clear()
        frame_index = 0
        start_ns = time.perf_counter_ns()
        next_tick_ns = start_ns
        with WindowsTimerResolution():
            while not self._stop_event.is_set():
                now_ns = time.perf_counter_ns()
                remaining_ns = next_tick_ns - now_ns
                if remaining_ns > 2_000_000:
                    sleep_sec = max((remaining_ns - 500_000) / 1_000_000_000.0, 0.0)
                    if self._stop_event.wait(sleep_sec):
                        break
                    continue
                while not self._stop_event.is_set():
                    if time.perf_counter_ns() >= next_tick_ns:
                        break
                if self._stop_event.is_set():
                    break

                t_sec = (next_tick_ns - start_ns) / 1_000_000_000.0
                self.tick.emit(t_sec, frame_index)
                frame_index += 1
                next_tick_ns = start_ns + frame_index * self.period_ns

                now_ns = time.perf_counter_ns()
                if now_ns - next_tick_ns > self.period_ns:
                    skipped = int((now_ns - next_tick_ns) // self.period_ns)
                    frame_index += skipped
                    next_tick_ns = start_ns + frame_index * self.period_ns


class FourArrowStimWidget(QWidget):
    def __init__(
        self,
        *,
        freqs: Sequence[float],
        refresh_rate_hz: float,
        mean: float,
        amp: float,
        phi: float,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.freqs = tuple(float(freq) for freq in freqs)
        self.refresh_rate_hz = float(refresh_rate_hz)
        self.mean = float(mean)
        self.amp = float(amp)
        self.phi = float(phi)

        self.clock_thread: Optional[StimClockThread] = None
        self.clock_running = False
        self.current_t = 0.0
        self.current_frame = -1

        self.phase_mode = PHASE_IDLE
        self.phase_title = "Ready"
        self.phase_detail = "Click Full Workflow to calibrate, then enter 4-arrow validation."
        self.remaining_sec = 0
        self.flicker_enabled = False
        self.cue_freq: Optional[float] = None
        self.pred_freq: Optional[float] = None
        self.selected_freq: Optional[float] = None
        self.decoder_state = "idle"

        self.setStyleSheet("background-color: black;")
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

    def start_clock(self) -> None:
        if self.clock_running:
            return
        self.clock_running = True
        self.current_t = 0.0
        self.current_frame = -1
        self.clock_thread = StimClockThread(self.refresh_rate_hz)
        self.clock_thread.tick.connect(self.on_tick, type=Qt.QueuedConnection)
        self.clock_thread.start(QThread.TimeCriticalPriority)

    def stop_clock(self) -> None:
        if not self.clock_running:
            self.update()
            return
        self.clock_running = False
        if self.clock_thread is not None:
            self.clock_thread.request_stop()
            self.clock_thread.wait(1500)
            self.clock_thread.deleteLater()
            self.clock_thread = None
        self.current_t = 0.0
        self.current_frame = -1
        self.update()

    @pyqtSlot(float, int)
    def on_tick(self, t_sec: float, frame_index: int) -> None:
        if not self.clock_running or frame_index <= self.current_frame:
            return
        self.current_t = float(t_sec)
        self.current_frame = int(frame_index)
        self.update()

    def apply_phase(self, payload: dict[str, Any]) -> None:
        self.phase_mode = str(payload.get("mode", PHASE_IDLE))
        self.phase_title = str(payload.get("title", ""))
        self.phase_detail = str(payload.get("detail", ""))
        self.remaining_sec = int(payload.get("remaining_sec", 0) or 0)
        self.flicker_enabled = bool(payload.get("flicker", False))
        self.cue_freq = payload.get("cue_freq")

        if self.phase_mode == PHASE_STOPPED:
            self.pred_freq = None
            self.selected_freq = None
            self.decoder_state = "idle"
            self.stop_clock()
        else:
            self.start_clock()
        self.update()

    def apply_result(self, payload: dict[str, Any]) -> None:
        self.pred_freq = payload.get("pred_freq")
        self.selected_freq = payload.get("selected_freq")
        self.decoder_state = str(payload.get("state", "idle"))
        self.update()

    def _box_color(self, freq: float, t_sec: float) -> QColor:
        if self.flicker_enabled and self.clock_running:
            luminance = self.mean + self.amp * np.sin(2.0 * np.pi * freq * t_sec + self.phi)
            luminance = float(np.clip(luminance, 0.0, 1.0))
        else:
            luminance = 0.22
        gray = int(round(255 * luminance))
        return QColor(gray, gray, gray)

    def _border_pen(self, freq: float) -> QPen:
        selected = self.selected_freq is not None and abs(float(freq) - float(self.selected_freq)) < 1e-8
        cue = self.cue_freq is not None and abs(float(freq) - float(self.cue_freq)) < 1e-8
        predicted = self.pred_freq is not None and abs(float(freq) - float(self.pred_freq)) < 1e-8

        if selected:
            return QPen(QColor(64, 220, 140), 9)
        if self.phase_mode in (PHASE_CAL_PREPARE, PHASE_CAL_ACTIVE) and cue:
            return QPen(QColor(80, 200, 255), 8)
        if self.decoder_state == "candidate" and predicted:
            return QPen(QColor(255, 210, 60), 7)
        if self.decoder_state == "selected" and predicted:
            return QPen(QColor(64, 220, 140), 9)
        return QPen(QColor(70, 70, 70), 2)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), Qt.black)

        width = self.width()
        height = self.height()
        layout_size = float(min(width, height))
        side = int(max(120.0, layout_size * 0.24))
        side = min(side, max(120, int(width * 0.28)), max(120, int(height * 0.28)))
        side = min(side, max(80, width // 3), max(80, height // 3))
        padding = int(max(30.0, layout_size * 0.09))
        cx, cy = width // 2, height // 2
        positions = [
            (cx - side // 2, padding),
            (padding, cy - side // 2),
            (cx - side // 2, height - side - padding),
            (width - side - padding, cy - side // 2),
        ]

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 120), 3))
        painter.drawLine(cx - 18, cy, cx + 18, cy)
        painter.drawLine(cx, cy - 18, cx, cy + 18)

        for index, freq in enumerate(self.freqs):
            x, y = positions[index]
            rect = QRect(x, y, side, side)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(self._box_color(freq, self.current_t)))
            painter.drawRect(rect)

            painter.setPen(self._border_pen(freq))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)


class DeviceConnectWorker(QObject):
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
            actual_fs = int(BoardShim.get_sampling_rate(self.board_id))
            eeg_channels = [int(item) for item in BoardShim.get_eeg_channels(self.board_id)]
            board.start_stream(450000)
            ready_samples = ensure_stream_ready(board, actual_fs)
            self.connected.emit(
                {
                    "requested_serial_port": self.serial_port,
                    "resolved_serial_port": resolved_port,
                    "attempted_ports": attempted_ports,
                    "sampling_rate": actual_fs,
                    "eeg_channels": eeg_channels,
                    "ready_samples": int(ready_samples),
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


class WorkflowWorker(QObject):
    phase_changed = pyqtSignal(object)
    log_message = pyqtSignal(str)
    result_updated = pyqtSignal(object)
    profile_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: WorkflowConfig, *, mode: str) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        self._stop_event = threading.Event()
        self._active_serial_port = normalize_serial_port(config.serial_port)

    @pyqtSlot()
    def run(self) -> None:
        board = None
        failed_message: Optional[str] = None
        requested_serial = normalize_serial_port(self.config.serial_port)
        self._active_serial_port = requested_serial
        try:
            if self.mode == "full":
                validate_calibration_plan(
                    target_repeats=self.config.target_repeats,
                    idle_repeats=self.config.idle_repeats,
                    active_sec=self.config.active_sec,
                    preferred_win_sec=self.config.win_sec,
                    step_sec=self.config.step_sec,
                )
            require_brainflow()
            self.phase_changed.emit(
                {
                    "mode": PHASE_IDLE,
                    "title": "Preparing EEG device connection",
                    "detail": f"Serial {requested_serial} | Board ID {self.config.board_id}",
                    "flicker": False,
                    "cue_freq": None,
                }
            )
            self.log_message.emit("Connecting to BrainFlow device...")
            board, resolved_port, attempted_ports = prepare_board_session(self.config.board_id, requested_serial)
            self._active_serial_port = resolved_port
            if serial_port_is_auto(requested_serial):
                attempts = ", ".join(attempted_ports)
                self.log_message.emit(
                    f"Serial auto-select: requested={requested_serial} -> using {resolved_port} "
                    f"(attempted: {attempts})"
                )
            else:
                self.log_message.emit(f"Serial selected: {resolved_port}")

            actual_fs = BoardShim.get_sampling_rate(self.config.board_id)
            eeg_channels = BoardShim.get_eeg_channels(self.config.board_id)
            self.log_message.emit(f"Board connected | fs={actual_fs}Hz | EEG channels={list(eeg_channels)}")
            try:
                board.start_stream(450000)
                ready_samples = ensure_stream_ready(board, actual_fs)
            except Exception as exc:
                raise RuntimeError(describe_runtime_error(exc, serial_port=self._active_serial_port)) from exc
            time.sleep(DEFAULT_STREAM_WARMUP_SEC)
            board.get_board_data()
            self.log_message.emit(f"stream ready | buffered_samples={ready_samples}")

            if self.mode == "full":
                profile = self._run_calibration_robust(board, eeg_channels, actual_fs)
            else:
                profile = load_profile(
                    self.config.profile_path,
                    fallback_freqs=self.config.freqs,
                    require_exists=True,
                )
                if profile_is_default_fallback(profile):
                    raise RuntimeError(
                        f"profile '{self.config.profile_path}' is still a default fallback profile. Run full calibration first."
                    )
                source = "existing_profile"
                self.profile_ready.emit(
                    {
                        "source": source,
                        "profile_path": str(self.config.profile_path),
                        "profile": asdict(profile),
                        "summary_text": "Loaded existing profile",
                    }
                )
                self.log_message.emit(f"Loaded profile: {self.config.profile_path}")

            if self._stop_event.is_set():
                self.log_message.emit("Workflow stopped by user.")
                return

            board.get_board_data()
            self._run_validation_robust(board, eeg_channels, actual_fs, profile)
        except Exception as exc:
            failed_message = f"Workflow failed: {describe_runtime_error(exc, serial_port=self._active_serial_port)}"
            self.error_occurred.emit(failed_message)
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
            if failed_message is not None:
                self.phase_changed.emit(
                    {
                        "mode": PHASE_ERROR,
                        "title": "Workflow error",
                        "detail": failed_message,
                        "flicker": False,
                        "cue_freq": None,
                    }
                )
            else:
                self.phase_changed.emit(
                    {
                        "mode": PHASE_STOPPED,
                        "title": "Workflow stopped",
                        "detail": "You can restart full workflow, or load an existing profile for validation.",
                        "flicker": False,
                        "cue_freq": None,
                    }
                )
            self.finished.emit()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _wait_with_phase(self, payload: dict[str, Any], duration_sec: float) -> bool:
        if duration_sec <= 0:
            self.phase_changed.emit({**payload, "remaining_sec": 0})
            return not self._stop_event.is_set()

        deadline = time.perf_counter() + float(duration_sec)
        last_second: Optional[int] = None
        while not self._stop_event.is_set():
            remaining = max(0.0, deadline - time.perf_counter())
            remaining_sec = int(np.ceil(remaining))
            if remaining_sec != last_second:
                self.phase_changed.emit({**payload, "remaining_sec": remaining_sec})
                last_second = remaining_sec
            if remaining <= 0:
                break
            self._stop_event.wait(min(0.05, remaining))
        return not self._stop_event.is_set()
    def _run_calibration(self, board, eeg_channels: Sequence[int], actual_fs: int) -> ThresholdProfile:
        return self._run_calibration_robust(board, eeg_channels, actual_fs)

    def _run_calibration_robust(self, board, eeg_channels: Sequence[int], actual_fs: int) -> ThresholdProfile:
        engine = FBCCAEngine(
            sampling_rate=actual_fs,
            freqs=self.config.freqs,
            win_sec=self.config.win_sec,
            step_sec=self.config.step_sec,
        )
        active_samples = int(round(self.config.active_sec * actual_fs))
        if active_samples < engine.win_samples:
            raise ValueError("active_sec must be at least as large as win_sec")

        trials = build_calibration_trials(
            self.config.freqs,
            target_repeats=self.config.target_repeats,
            idle_repeats=self.config.idle_repeats,
        )
        trial_segments: list[tuple[TrialSpec, np.ndarray]] = []
        failed_trials = 0

        self.log_message.emit("Entering pre-calibration phase.")
        for index, trial in enumerate(trials, start=1):
            total = len(trials)
            prompt = trial_prompt(self.config.freqs, trial)
            title_base = f"Calibration {index}/{total}"
            cue_freq = trial.expected_freq

            prepare_payload = {
                "mode": PHASE_CAL_PREPARE,
                "title": title_base,
                "detail": f"{prompt} | Prepare",
                "flicker": False,
                "cue_freq": cue_freq,
            }
            if not self._wait_with_phase(prepare_payload, self.config.prepare_sec):
                return load_profile(self.config.profile_path, fallback_freqs=self.config.freqs)

            board.get_board_data()
            active_payload = {
                "mode": PHASE_CAL_ACTIVE,
                "title": title_base,
                "detail": f"{prompt} | Collect",
                "flicker": True,
                "cue_freq": cue_freq,
            }
            if not self._wait_with_phase(active_payload, self.config.active_sec):
                return load_profile(self.config.profile_path, fallback_freqs=self.config.freqs)

            try:
                segment, used_samples, available_samples = read_recent_eeg_segment(
                    board,
                    eeg_channels,
                    target_samples=active_samples,
                    minimum_samples=engine.win_samples,
                )
                if used_samples < active_samples:
                    self.log_message.emit(
                        f"trial {index}/{total} sample shortfall, continue with "
                        f"{used_samples}/{active_samples} points (buffer={available_samples})"
                    )

                trial_segments.append((trial, segment))
                self.log_message.emit(f"Completed trial {index}/{total}: {prompt}")
            except Exception as exc:
                failed_trials += 1
                self.log_message.emit(
                    f"Calibration warning: skipped trial {index}/{total} ({trial.label}): "
                    f"{describe_runtime_error(exc, serial_port=self._active_serial_port)}"
                )
                if failed_trials >= DEFAULT_MAX_CALIBRATION_TRIAL_ERRORS:
                    raise RuntimeError(
                        f"Too many calibration failures, skipped {failed_trials} trials"
                    ) from exc

            rest_payload = {
                "mode": PHASE_CAL_REST,
                "title": title_base,
                "detail": "Rest briefly before the next trial",
                "flicker": False,
                "cue_freq": None,
            }
            if not self._wait_with_phase(rest_payload, self.config.rest_sec):
                return load_profile(self.config.profile_path, fallback_freqs=self.config.freqs)
            board.get_board_data()

        try:
            profile, metadata = optimize_profile_from_segments(
                trial_segments,
                available_board_channels=eeg_channels,
                sampling_rate=actual_fs,
                freqs=self.config.freqs,
                active_sec=self.config.active_sec,
                preferred_win_sec=self.config.win_sec,
                step_sec=self.config.step_sec,
            )
        except Exception as exc:
            raise RuntimeError(
                "Calibration profile fitting failed: "
                f"usable_trials={len(trial_segments)}, "
                f"failed_trials={failed_trials}. "
                f"{describe_runtime_error(exc, serial_port=self._active_serial_port)}"
            ) from exc
        save_profile(profile, self.config.profile_path)
        summary = metadata.get("validation_summary", {})
        summary_text = format_quality_summary(summary)
        self.profile_ready.emit(
            {
                "source": "calibration",
                "profile_path": str(self.config.profile_path),
                "profile": asdict(profile),
                "summary": summary,
                "summary_text": summary_text,
                "metadata": metadata,
            }
        )
        self.log_message.emit(f"Calibration complete, profile saved to {self.config.profile_path}")
        self.log_message.emit(
            f"Calibration selection: channels={metadata.get('selected_eeg_channels')} | "
            f"search={metadata.get('validation_search')}"
        )
        self.log_message.emit(summary_text.replace("\n", " | "))
        return profile

    def _run_validation_robust(
        self,
        board,
        eeg_channels: Sequence[int],
        actual_fs: int,
        profile: ThresholdProfile,
    ) -> None:
        eeg_channels = resolve_selected_eeg_channels(eeg_channels, profile.eeg_channels)
        decoder = load_decoder_from_profile(profile, sampling_rate=actual_fs)
        decoder.configure_runtime(actual_fs)
        gate = AsyncDecisionGate.from_profile(profile)

        self.phase_changed.emit(
            {
                "mode": PHASE_VALIDATION,
                "title": f"4-arrow online validation running ({profile.model_name})",
                "detail": (
                    "Free gaze at arrows or center; check whether output appears only when focusing targets. "
                    f"Decoder={profile.model_name}"
                ),
                "flicker": True,
                "cue_freq": None,
                "remaining_sec": 0,
            }
        )
        self.log_message.emit(f"Entering online validation phase with decoder={profile.model_name}.")
        last_signature: Optional[tuple[str, Optional[float], Optional[float]]] = None
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
                infer_t0 = time.perf_counter()
                decision = gate.update(decoder.analyze_window(eeg))
                infer_t1 = time.perf_counter()
                decoder.update_online(decision, eeg)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                self.log_message.emit(
                    f"Online read warning {consecutive_errors}/{DEFAULT_MAX_TRANSIENT_READ_ERRORS}: {exc}"
                )
                if consecutive_errors >= DEFAULT_MAX_TRANSIENT_READ_ERRORS:
                    raise RuntimeError(
                        "online decode aborted after repeated read failures: "
                        f"{describe_runtime_error(exc, serial_port=self._active_serial_port)}"
                    ) from exc
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
                "normalized_top1": float(decision["normalized_top1"]),
                "score_entropy": float(decision["score_entropy"]),
                "control_log_lr": None
                if decision.get("control_log_lr") is None
                else float(decision["control_log_lr"]),
                "control_confidence": None
                if decision.get("control_confidence") is None
                else float(decision["control_confidence"]),
                "stable_windows": int(decision["stable_windows"]),
                "pred_label": direction_label(profile.freqs, decision["pred_freq"]),
                "selected_label": direction_label(profile.freqs, decision["selected_freq"]),
                "model_name": str(profile.model_name),
                "decision_latency_ms": float((infer_t1 - infer_t0) * 1000.0),
            }
            self.result_updated.emit(payload)

            signature = (payload["state"], payload["selected_freq"], payload["pred_freq"])
            if signature != last_signature:
                if payload["state"] == "selected":
                    self.log_message.emit(f"Output locked: {payload['selected_label']}")
                elif payload["state"] == "candidate":
                    self.log_message.emit(f"Candidate: {payload['pred_label']}")
                else:
                    self.log_message.emit("No output")
                last_signature = signature

            self._stop_event.wait(decoder.step_sec)


class AsyncValidationWindow(QMainWindow):
    def __init__(
        self,
        *,
        serial_port: str,
        board_id: int,
        sampling_rate: int,
        refresh_rate_hz: float,
        freqs: Sequence[float],
        profile_path: Path,
        prepare_sec: float,
        active_sec: float,
        rest_sec: float,
        target_repeats: int,
        idle_repeats: int,
        win_sec: float,
        step_sec: float,
        stim_mean: float,
        stim_amp: float,
        stim_phi: float,
        windowed: bool,
    ) -> None:
        super().__init__()
        self.windowed = bool(windowed)
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[WorkflowWorker] = None
        self.connect_thread: Optional[QThread] = None
        self.connect_worker: Optional[DeviceConnectWorker] = None
        self.last_result_signature: Optional[tuple[str, Optional[float]]] = None
        self.last_worker_error_message: Optional[str] = None

        self.initial_values = {
            "serial_port": normalize_serial_port(serial_port),
            "board_id": int(board_id),
            "sampling_rate": int(sampling_rate),
            "refresh_rate_hz": float(refresh_rate_hz),
            "freqs": ",".join(f"{float(freq):g}" for freq in freqs),
            "profile_path": str(profile_path),
            "prepare_sec": float(prepare_sec),
            "active_sec": float(active_sec),
            "rest_sec": float(rest_sec),
            "target_repeats": int(target_repeats),
            "idle_repeats": int(idle_repeats),
            "win_sec": float(win_sec),
            "step_sec": float(step_sec),
            "stim_mean": float(stim_mean),
            "stim_amp": float(stim_amp),
            "stim_phi": float(stim_phi),
        }
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("SSVEP Async Calibration + 4-Arrow Validation")
        self.setMinimumSize(1400, 860)
        if self.windowed:
            self.resize(1720, 980)

        root = QWidget(self)
        self.setCentralWidget(root)
        root.setStyleSheet("background-color: black;")
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        splitter.setStyleSheet(
            "QSplitter::handle { background-color: rgba(38, 54, 64, 180); border-radius: 4px; }"
        )
        main_layout.addWidget(splitter, 1)

        left_container = QWidget()
        left_container.setMaximumWidth(430)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(18)
        left_layout.addWidget(self._build_settings_panel(), 0)
        left_layout.addWidget(self._build_result_panel(), 0)
        left_layout.addStretch(1)
        splitter.addWidget(left_container)

        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(16)
        center_layout.addWidget(self._build_center_status_panel(), 0)

        self.stim_widget = FourArrowStimWidget(
            freqs=parse_freqs(self.initial_values["freqs"]),
            refresh_rate_hz=self.initial_values["refresh_rate_hz"],
            mean=self.initial_values["stim_mean"],
            amp=self.initial_values["stim_amp"],
            phi=self.initial_values["stim_phi"],
        )
        self.stim_widget.setMinimumSize(920, 760)
        self.stim_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        center_layout.addWidget(self.stim_widget, 1)
        splitter.addWidget(center_container)

        right_container = QWidget()
        right_container.setMaximumWidth(460)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(18)
        right_layout.addWidget(self._build_profile_panel(), 0)
        right_layout.addWidget(self._build_log_panel(), 1)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([360, 1080, 400])

        self._apply_ready_state()
        self._set_center_status(
            title="Ready",
            detail="Click Full Workflow to calibrate, then enter 4-arrow online validation.",
            state_text="READY",
            output_text="Current output: No output",
            accent="rgba(58, 68, 78, 220)",
        )

    def _build_center_status_panel(self) -> QWidget:
        frame = self._panel_frame(0)
        frame.setMinimumWidth(0)
        frame.setMaximumHeight(126)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)

        self.center_phase_title = QLabel("Ready")
        self.center_phase_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #f4f7fa;")
        layout.addWidget(self.center_phase_title)

        self.center_phase_detail = QLabel("Click Full Workflow to calibrate, then enter 4-arrow online validation.")
        self.center_phase_detail.setWordWrap(True)
        self.center_phase_detail.setStyleSheet("font-size: 14px; color: #c8d5dd;")
        layout.addWidget(self.center_phase_detail)

        status_row = QHBoxLayout()
        status_row.setSpacing(16)

        self.center_state_badge = QLabel("IDLE")
        self.center_state_badge.setStyleSheet(
            "background-color: rgba(58, 68, 78, 220);"
            "border-radius: 10px;"
            "padding: 6px 12px;"
            "font-size: 15px;"
            "font-weight: bold;"
            "color: #dce5ea;"
        )
        status_row.addWidget(self.center_state_badge, 0)

        self.center_output_label = QLabel("Current output: No output")
        self.center_output_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #dce5ea;")
        status_row.addWidget(self.center_output_label, 1)
        status_row.addStretch(1)
        layout.addLayout(status_row)
        return frame

    def _set_center_status(self, *, title: str, detail: str, state_text: str, output_text: str, accent: str) -> None:
        self.center_phase_title.setText(title)
        self.center_phase_detail.setText(detail)
        self.center_state_badge.setText(state_text)
        self.center_state_badge.setStyleSheet(
            f"background-color: {accent};"
            "border-radius: 10px;"
            "padding: 6px 12px;"
            "font-size: 15px;"
            "font-weight: bold;"
            "color: white;"
        )
        self.center_output_label.setText(output_text)

    def _panel_frame(self, min_width: int) -> QFrame:
        frame = QFrame()
        frame.setMinimumWidth(min_width)
        frame.setStyleSheet(
            "QFrame {"
            "background-color: rgba(10, 16, 20, 215);"
            "border: 1px solid rgba(96, 152, 170, 200);"
            "border-radius: 16px;"
            "color: white;"
            "}"
            "QLabel { color: white; font-size: 14px; }"
            "QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit {"
            "background-color: rgba(28, 35, 42, 220);"
            "border: 1px solid rgba(86, 112, 128, 220);"
            "border-radius: 8px;"
            "color: white;"
            "font-size: 14px;"
            "padding: 6px 8px;"
            "min-height: 34px;"
            "}"
            "QPushButton {"
            "border-radius: 10px;"
            "padding: 10px 14px;"
            "font-size: 14px;"
            "font-weight: bold;"
            "min-height: 40px;"
            "}"
        )
        return frame

    def _build_settings_panel(self) -> QWidget:
        frame = self._panel_frame(330)
        frame.setMaximumWidth(390)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Runtime Settings")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self.serial_port_edit = QLineEdit(self.initial_values["serial_port"])
        self.board_id_spin = QSpinBox()
        self.board_id_spin.setRange(-1, 9999)
        self.board_id_spin.setValue(self.initial_values["board_id"])

        self.refresh_rate_spin = QDoubleSpinBox()
        self.refresh_rate_spin.setRange(30.0, 500.0)
        self.refresh_rate_spin.setDecimals(1)
        self.refresh_rate_spin.setValue(self.initial_values["refresh_rate_hz"])

        self.freqs_edit = QLineEdit(self.initial_values["freqs"])
        self.profile_path_edit = QLineEdit(self.initial_values["profile_path"])

        self.prepare_sec_spin = QDoubleSpinBox()
        self.prepare_sec_spin.setRange(0.0, 10.0)
        self.prepare_sec_spin.setDecimals(1)
        self.prepare_sec_spin.setValue(self.initial_values["prepare_sec"])

        self.active_sec_spin = QDoubleSpinBox()
        self.active_sec_spin.setRange(1.0, 20.0)
        self.active_sec_spin.setDecimals(1)
        self.active_sec_spin.setValue(self.initial_values["active_sec"])

        self.rest_sec_spin = QDoubleSpinBox()
        self.rest_sec_spin.setRange(0.0, 10.0)
        self.rest_sec_spin.setDecimals(1)
        self.rest_sec_spin.setValue(self.initial_values["rest_sec"])

        self.target_repeats_spin = QSpinBox()
        self.target_repeats_spin.setRange(1, 20)
        self.target_repeats_spin.setValue(self.initial_values["target_repeats"])

        self.idle_repeats_spin = QSpinBox()
        self.idle_repeats_spin.setRange(1, 30)
        self.idle_repeats_spin.setValue(self.initial_values["idle_repeats"])

        form.addRow("Serial Port (auto/COMx)", self.serial_port_edit)
        form.addRow("Board ID", self.board_id_spin)
        form.addRow("Display Refresh (Hz)", self.refresh_rate_spin)
        form.addRow("Frequencies", self.freqs_edit)
        form.addRow("Profile Path", self.profile_path_edit)
        form.addRow("Prepare (s)", self.prepare_sec_spin)
        form.addRow("Active (s)", self.active_sec_spin)
        form.addRow("Rest (s)", self.rest_sec_spin)
        form.addRow("Target Repeats", self.target_repeats_spin)
        form.addRow("idle repeats", self.idle_repeats_spin)
        layout.addLayout(form)

        self.btn_connect_device = QPushButton("连接设备")
        self.btn_connect_device.setStyleSheet("background-color: #2f7f8f; color: white;")
        layout.addWidget(self.btn_connect_device)

        btn_row = QHBoxLayout()
        self.btn_start_full = QPushButton("Full Workflow")
        self.btn_start_full.setStyleSheet("background-color: #0e7a4d; color: white;")
        self.btn_start_online = QPushButton("Validate Existing Profile")
        self.btn_start_online.setStyleSheet("background-color: #285c9b; color: white;")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("background-color: #ab3c3c; color: white;")
        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setStyleSheet("background-color: #59626a; color: white;")
        btn_row.addWidget(self.btn_start_full)
        btn_row.addWidget(self.btn_start_online)
        layout.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        btn_row2.addWidget(self.btn_stop)
        btn_row2.addWidget(self.btn_exit)
        layout.addLayout(btn_row2)

        hint = QLabel("Full Workflow runs calibration first, then automatically enters online 4-arrow validation.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #d7e8ee;")
        layout.addWidget(hint)

        self.btn_connect_device.clicked.connect(self.connect_device)
        self.btn_start_full.clicked.connect(lambda: self.start_workflow("full"))
        self.btn_start_online.clicked.connect(lambda: self.start_workflow("online_only"))
        self.btn_stop.clicked.connect(self.stop_workflow)
        self.btn_exit.clicked.connect(self.close)
        return frame

    def _build_profile_panel(self) -> QWidget:
        frame = self._panel_frame(340)
        frame.setMaximumWidth(420)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Profile / Phase Summary")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.phase_label = QLabel("Ready")
        self.phase_label.setWordWrap(True)
        self.phase_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #89d7ff;")
        layout.addWidget(self.phase_label)

        self.profile_summary_text = QPlainTextEdit()
        self.profile_summary_text.setReadOnly(True)
        self.profile_summary_text.setMaximumHeight(240)
        layout.addWidget(self.profile_summary_text)
        return frame

    def _build_result_panel(self) -> QWidget:
        frame = self._panel_frame(330)
        frame.setMaximumWidth(390)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Realtime Output")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.output_label = QLabel("No output")
        self.output_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #dce5ea;")
        layout.addWidget(self.output_label)

        self.metrics_label = QLabel("")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("font-size: 15px; color: #dce5ea;")
        layout.addWidget(self.metrics_label)
        return frame

    def _build_log_panel(self) -> QWidget:
        frame = self._panel_frame(340)
        frame.setMaximumWidth(420)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Runtime Log")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(400)
        self.log_text.setMinimumHeight(220)
        layout.addWidget(self.log_text)
        return frame

    def _collect_config(self, mode: str) -> WorkflowConfig:
        freqs = parse_freqs(self.freqs_edit.text())
        profile_path = Path(self.profile_path_edit.text().strip() or str(DEFAULT_PROFILE_PATH)).expanduser()
        config = WorkflowConfig(
            serial_port=normalize_serial_port(self.serial_port_edit.text().strip()),
            board_id=int(self.board_id_spin.value()),
            sampling_rate=self.initial_values["sampling_rate"],
            refresh_rate_hz=float(self.refresh_rate_spin.value()),
            freqs=freqs,
            profile_path=profile_path,
            prepare_sec=float(self.prepare_sec_spin.value()),
            active_sec=float(self.active_sec_spin.value()),
            rest_sec=float(self.rest_sec_spin.value()),
            target_repeats=int(self.target_repeats_spin.value()),
            idle_repeats=int(self.idle_repeats_spin.value()),
            win_sec=float(self.initial_values["win_sec"]),
            step_sec=float(self.initial_values["step_sec"]),
            stim_mean=float(self.initial_values["stim_mean"]),
            stim_amp=float(self.initial_values["stim_amp"]),
            stim_phi=float(self.initial_values["stim_phi"]),
        )
        if mode == "full":
            validate_calibration_plan(
                target_repeats=config.target_repeats,
                idle_repeats=config.idle_repeats,
                active_sec=config.active_sec,
                preferred_win_sec=config.win_sec,
                step_sec=config.step_sec,
            )
        return config

    def _apply_ready_state(self) -> None:
        self.btn_start_full.setEnabled(True)
        self.btn_start_online.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_connect_device.setEnabled(True)
        self.output_label.setText("No output")
        self.output_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #dce5ea;")
        self.metrics_label.setText("Waiting to start. Full workflow calibrates first, then enters online validation.")
        self.phase_label.setText("Ready")
        self.profile_summary_text.setPlainText(
            "Default profile path:\n"
            f"{self.profile_path_edit.text()}\n\n"
            "Suggested manual checks:\n"
            "1. While focusing one arrow, output should lock to that target.\n"
            "2. While looking at center, output should return to No output.\n"
            "3. Switching from A to B should not remain stuck on A for too long."
        )

    def _set_running_state(self, running: bool) -> None:
        self.btn_start_full.setEnabled(not running)
        self.btn_start_online.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        if self.connect_thread is None:
            self.btn_connect_device.setEnabled(not running)
        else:
            self.btn_connect_device.setEnabled(False)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{timestamp}] {message}")
        scroll_bar = self.log_text.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def connect_device(self) -> None:
        if self.worker_thread is not None:
            self.log("Workflow is running; stop workflow before connecting device.")
            return
        if self.connect_thread is not None:
            self.log("Device connection is already in progress.")
            return
        try:
            config = self._collect_config("online_only")
        except Exception as exc:
            self.log(f"Invalid configuration: {exc}")
            return
        self.connect_thread = QThread(self)
        self.connect_worker = DeviceConnectWorker(
            serial_port=config.serial_port,
            board_id=config.board_id,
        )
        self.connect_worker.moveToThread(self.connect_thread)
        self.connect_thread.started.connect(self.connect_worker.run)
        self.connect_worker.connected.connect(self.on_device_connected)
        self.connect_worker.error.connect(self.on_device_connect_error)
        self.connect_worker.finished.connect(self.connect_thread.quit)
        self.connect_thread.finished.connect(self.connect_thread.deleteLater)
        self.connect_thread.finished.connect(self._cleanup_connect_refs)
        self.btn_connect_device.setEnabled(False)
        self.btn_connect_device.setText("连接中...")
        self.log(f"Connecting device (serial={config.serial_port}, board_id={config.board_id})...")
        self.connect_thread.start()

    @pyqtSlot(object)
    def on_device_connected(self, payload: dict[str, Any]) -> None:
        resolved_port = str(payload.get("resolved_serial_port", ""))
        requested_port = str(payload.get("requested_serial_port", ""))
        attempts = payload.get("attempted_ports") or []
        sampling_rate = int(payload.get("sampling_rate", 0))
        eeg_channels = payload.get("eeg_channels") or []
        ready_samples = int(payload.get("ready_samples", 0))
        if resolved_port:
            self.serial_port_edit.setText(resolved_port)
        self.log(
            f"Device connected | requested={requested_port} -> {resolved_port} | "
            f"fs={sampling_rate}Hz | eeg_channels={eeg_channels} | ready_samples={ready_samples}"
        )
        if attempts:
            self.log(f"Serial attempts: {attempts}")
        self.phase_label.setText("Device connected")
        self._set_center_status(
            title="Device connected",
            detail=f"Serial {resolved_port} | fs={sampling_rate}Hz | channels={eeg_channels}",
            state_text="CONNECTED",
            output_text="Current output: No output",
            accent="rgba(45, 132, 88, 230)",
        )

    @pyqtSlot(str)
    def on_device_connect_error(self, message: str) -> None:
        self.log(message)
        self._set_center_status(
            title="Device connect failed",
            detail=str(message),
            state_text="ERROR",
            output_text="Current output: No output",
            accent="rgba(176, 64, 64, 230)",
        )

    def start_workflow(self, mode: str) -> None:
        if self.connect_thread is not None:
            self.log("Device connection is in progress, wait until it completes.")
            return
        if self.worker_thread is not None:
            self.log("Workflow is already running.")
            return

        try:
            config = self._collect_config(mode)
        except Exception as exc:
            self.log(f"Invalid configuration: {exc}")
            return

        self.stim_widget.freqs = config.freqs
        self.stim_widget.refresh_rate_hz = config.refresh_rate_hz
        self.stim_widget.mean = config.stim_mean
        self.stim_widget.amp = config.stim_amp
        self.stim_widget.phi = config.stim_phi

        self.worker_thread = QThread(self)
        self.worker = WorkflowWorker(config, mode=mode)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.phase_changed.connect(self.on_phase_changed)
        self.worker.log_message.connect(self.log)
        self.worker.result_updated.connect(self.on_result_updated)
        self.worker.profile_ready.connect(self.on_profile_ready)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._cleanup_worker_refs)

        self.last_result_signature = None
        self.last_worker_error_message = None
        self.log("Workflow started.")
        self._set_running_state(True)
        self.worker_thread.start()

    def stop_workflow(self) -> None:
        if self.worker is None:
            self.log("No active workflow.")
            return
        self.log("Stopping workflow...")
        self.worker.request_stop()

    @pyqtSlot(object)
    def on_phase_changed(self, payload: dict[str, Any]) -> None:
        self.phase_label.setText(str(payload.get("title", "")))
        self.stim_widget.apply_phase(payload)
        detail = str(payload.get("detail", ""))
        phase_mode = str(payload.get("mode", "idle"))
        remaining_sec = int(payload.get("remaining_sec", 0) or 0)
        if remaining_sec > 0:
            detail = f"{detail} | T-{remaining_sec}s"
        accent = "rgba(176, 64, 64, 230)" if phase_mode == PHASE_ERROR else "rgba(58, 68, 78, 220)"
        self._set_center_status(
            title=str(payload.get("title", "")),
            detail=detail,
            state_text=phase_mode.upper(),
            output_text=f"Current output: {direction_label(self.stim_widget.freqs, self.stim_widget.selected_freq)}",
            accent=accent,
        )

    @pyqtSlot(object)
    def on_profile_ready(self, payload: dict[str, Any]) -> None:
        profile = payload.get("profile", {})
        summary_text = str(payload.get("summary_text", "")).strip()
        metadata = payload.get("metadata") or profile.get("metadata") or {}
        selected_channels = profile.get("eeg_channels") or metadata.get("selected_eeg_channels")
        validation_search = metadata.get("validation_search", {})
        model_name = str(profile.get("model_name", "fbcca"))
        benchmark_metrics = profile.get("benchmark_metrics") or {}
        lines = [
            f"Source: {payload.get('source', '')}",
            f"Path: {payload.get('profile_path', '')}",
            f"model_name = {model_name}",
            "",
            f"enter_score_th = {profile.get('enter_score_th', 0.0):.6f}",
            f"enter_ratio_th = {profile.get('enter_ratio_th', 0.0):.6f}",
            f"enter_margin_th = {profile.get('enter_margin_th', 0.0):.6f}",
            f"enter_log_lr_th = {profile.get('enter_log_lr_th', 0.0):.6f}" if profile.get("enter_log_lr_th") is not None else "enter_log_lr_th = None",
            f"exit_score_th = {profile.get('exit_score_th', 0.0):.6f}",
            f"exit_ratio_th = {profile.get('exit_ratio_th', 0.0):.6f}",
            f"exit_log_lr_th = {profile.get('exit_log_lr_th', 0.0):.6f}" if profile.get("exit_log_lr_th") is not None else "exit_log_lr_th = None",
            f"min_enter_windows = {profile.get('min_enter_windows', 0)}",
            f"min_exit_windows = {profile.get('min_exit_windows', 0)}",
            f"eeg_channels = {selected_channels}" if selected_channels is not None else "eeg_channels = all",
            f"search = {validation_search}" if validation_search else "search = {}",
        ]
        if benchmark_metrics:
            lines.extend(
                [
                    "",
                    "benchmark_metrics:",
                    f"  idle_fp_per_min = {benchmark_metrics.get('idle_fp_per_min')}",
                    f"  control_recall = {benchmark_metrics.get('control_recall')}",
                    f"  switch_latency_s = {benchmark_metrics.get('switch_latency_s')}",
                    f"  itr_bpm = {benchmark_metrics.get('itr_bpm')}",
                    f"  inference_ms = {benchmark_metrics.get('inference_ms')}",
                ]
            )
        if summary_text:
            lines.extend(["", summary_text])
        self.profile_summary_text.setPlainText("\n".join(lines))

    @pyqtSlot(object)
    def on_result_updated(self, payload: dict[str, Any]) -> None:
        self.stim_widget.apply_result(payload)
        model_name = str(payload.get("model_name", "fbcca"))
        state = str(payload["state"])
        selected_label = str(payload["selected_label"])
        pred_label = str(payload["pred_label"])
        top1_score = float(payload["top1_score"])
        top2_score = float(payload["top2_score"])
        margin = float(payload["margin"])
        ratio = float(payload["ratio"])
        normalized_top1 = float(payload["normalized_top1"])
        score_entropy = float(payload["score_entropy"])
        control_log_lr = payload.get("control_log_lr")
        control_confidence = payload.get("control_confidence")
        stable_windows = int(payload["stable_windows"])
        decision_latency_ms = float(payload.get("decision_latency_ms", 0.0))
        control_log_lr_text = "None" if control_log_lr is None else f"{float(control_log_lr):.3f}"
        control_conf_text = "None" if control_confidence is None else f"{float(control_confidence):.3f}"

        if state == "selected":
            self.output_label.setText(selected_label)
            self.output_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #55f0a0;")
            badge_text = "SELECTED"
            badge_color = "rgba(45, 132, 88, 230)"
        elif state == "candidate":
            self.output_label.setText(f"Candidate: {pred_label}")
            self.output_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #ffd65d;")
            badge_text = "CANDIDATE"
            badge_color = "rgba(171, 132, 27, 230)"
        else:
            self.output_label.setText("No output")
            self.output_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #dce5ea;")
            badge_text = "IDLE"
            badge_color = "rgba(58, 68, 78, 220)"

        self.metrics_label.setText(
            f"Decoder: {model_name}\n"
            f"State: {state}\n"
            f"Top prediction: {pred_label}\n"
            f"top1={top1_score:.5f} | top2={top2_score:.5f}\n"
            f"margin={margin:.5f} | ratio={ratio:.3f}\n"
            f"norm_top1={normalized_top1:.3f} | entropy={score_entropy:.3f}\n"
            f"control_log_lr={control_log_lr_text} | control_conf={control_conf_text}\n"
            f"Stable windows: {stable_windows} | infer={decision_latency_ms:.2f}ms"
        )

        self._set_center_status(
            title=self.phase_label.text(),
            detail=self.center_phase_detail.text(),
            state_text=badge_text,
            output_text=f"Current output: {selected_label if state == 'selected' else 'No output'}",
            accent=badge_color,
        )
        signature = (state, payload.get("selected_freq"))
        if signature != self.last_result_signature:
            self.last_result_signature = signature

    @pyqtSlot(str)
    def on_worker_error(self, message: str) -> None:
        self.last_worker_error_message = str(message)
        self.log(message)
        self._set_center_status(
            title="Workflow error",
            detail=str(message),
            state_text="ERROR",
            output_text="Current output: No output",
            accent="rgba(176, 64, 64, 230)",
        )

    @pyqtSlot()
    def on_worker_finished(self) -> None:
        self.log("Workflow finished.")
        self._set_running_state(False)
        self.stim_widget.stop_clock()
        if self.last_worker_error_message:
            self._set_center_status(
                title="Workflow error",
                detail=self.last_worker_error_message,
                state_text="ERROR",
                output_text="Current output: No output",
                accent="rgba(176, 64, 64, 230)",
            )
            return
        self._set_center_status(
            title="Workflow stopped",
            detail="You can restart full workflow, or load an existing profile for validation.",
            state_text="STOPPED",
            output_text="Current output: No output",
            accent="rgba(92, 95, 110, 220)",
        )

    @pyqtSlot()
    def _cleanup_worker_refs(self) -> None:
        self.worker = None
        self.worker_thread = None
        if self.connect_thread is None:
            self.btn_connect_device.setEnabled(True)

    @pyqtSlot()
    def _cleanup_connect_refs(self) -> None:
        self.connect_worker = None
        self.connect_thread = None
        self.btn_connect_device.setText("连接设备")
        self.btn_connect_device.setEnabled(self.worker_thread is None)

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.request_stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(3000)
        if self.connect_thread is not None:
            self.connect_thread.quit()
            self.connect_thread.wait(3000)
        self.stim_widget.stop_clock()
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSVEP async calibration + validation UI")
    parser.add_argument("--serial-port", type=str, default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    parser.add_argument("--sampling-rate", type=int, default=250)
    parser.add_argument("--refresh-rate", type=float, default=240.0)
    parser.add_argument("--freqs", type=str, default="8,10,12,15")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--prepare-sec", type=float, default=1.0)
    parser.add_argument("--active-sec", type=float, default=4.0)
    parser.add_argument("--rest-sec", type=float, default=1.0)
    parser.add_argument("--target-repeats", type=int, default=5)
    parser.add_argument("--idle-repeats", type=int, default=10)
    parser.add_argument("--win-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=0.25)
    parser.add_argument("--stim-mean", type=float, default=0.5)
    parser.add_argument("--stim-amp", type=float, default=0.5)
    parser.add_argument("--stim-phi", type=float, default=0.0)
    parser.add_argument("--windowed", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = AsyncValidationWindow(
        serial_port=args.serial_port,
        board_id=args.board_id,
        sampling_rate=args.sampling_rate,
        refresh_rate_hz=args.refresh_rate,
        freqs=parse_freqs(args.freqs),
        profile_path=args.profile,
        prepare_sec=args.prepare_sec,
        active_sec=args.active_sec,
        rest_sec=args.rest_sec,
        target_repeats=args.target_repeats,
        idle_repeats=args.idle_repeats,
        win_sec=args.win_sec,
        step_sec=args.step_sec,
        stim_mean=args.stim_mean,
        stim_amp=args.stim_amp,
        stim_phi=args.stim_phi,
        windowed=args.windowed,
    )
    if args.windowed:
        window.show()
    else:
        window.showFullScreen()
    return int(app.exec_())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)

