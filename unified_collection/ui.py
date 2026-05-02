from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .index_store import append_unified_collection_index, wallclock_iso_timestamp
from .mi_bridge import (
    BoardCaptureWorker,
    BoardIds,
    DEFAULT_CHANNEL_NAMES,
    MIDataCollectorWindow,
    RealtimeEEGPreviewWidget,
    available_board_options,
)
from .mi_bridge import detect_serial_ports, parse_channel_names, parse_channel_positions
from .paths import (
    BRAIN_CODE_ROOT,
    DEFAULT_MI_OUTPUT_ROOT,
    DEFAULT_SSVEP_DATASET_DIR,
    MI_COLLECTION_DIR,
    MI_PROJECT_DIR,
    MI_SHARED_DIR,
    SSVEP_PROJECT_DIR,
    UNIFIED_COLLECTION_INDEX_PATH,
    WORKSPACE_ROOT,
    resolve_ssvep_dataset_dir,
)
from .ssvep_bridge import (
    PHASE_CAL_ACTIVE,
    PHASE_CAL_PREPARE,
    PHASE_CAL_REST,
    PHASE_ERROR,
    PHASE_STOPPED,
    STIMULUS_MODE_ELAPSED_TIME_SINE,
    STIMULUS_MODE_FRAME_LOCKED_SINE,
    validate_stimulus_mode,
)
from .ssvep_bridge import (
    ACTIVE_START_CUE_SEC,
    MAX_TRIAL_RETRIES,
    MIN_TRIAL_QUALITY_RATIO,
    STIM_AMP,
    STIM_FRAME_FORMULA,
    STIM_MEAN,
    STIM_PHI,
    STIMULUS_BACKEND_PYQT_FULLSCREEN,
    STIMULUS_PHASE_APPLY_TIMEOUT_SEC,
    CollectionFullscreenStimWindow,
    build_collection_output_session_id,
    estimate_active_stimulus_arm_sec,
    play_collection_tone_event,
    play_collection_tone_event_sync,
    prompt_text_for_trial,
    resolve_collection_stim_refresh_rate_hz,
    stimulus_backend_metadata,
    stimulus_sample_window_alignment_metadata,
    validate_stimulus_frequency_set,
)
from .ssvep_bridge import CollectionProtocol, build_collection_trials, parse_freqs, save_collection_dataset_bundle

SSVEP_MARKER_SESSION_START = 7000
SSVEP_MARKER_SESSION_END = 7001
SSVEP_MARKER_STIM_FIRST_FRAME = 7010
SSVEP_MARKER_ACTIVE_WINDOW_START = 7020
SSVEP_MARKER_ACTIVE_WINDOW_END = 7021

SSVEP_EVENT_CODE_NAMES = {
    SSVEP_MARKER_SESSION_START: "ssvep_session_start",
    SSVEP_MARKER_SESSION_END: "ssvep_session_end",
    SSVEP_MARKER_STIM_FIRST_FRAME: "ssvep_stim_first_frame",
    SSVEP_MARKER_ACTIVE_WINDOW_START: "ssvep_active_window_start",
    SSVEP_MARKER_ACTIVE_WINDOW_END: "ssvep_active_window_end",
}


def _jsonable_event(
    *,
    marker_code: int,
    trial_order: int | None = None,
    trial_id: int | None = None,
    label: str = "",
    expected_freq: float | None = None,
    details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "event_name": SSVEP_EVENT_CODE_NAMES[int(marker_code)],
        "marker_code": int(marker_code),
        "trial_order": trial_order,
        "trial_id": trial_id,
        "label": str(label),
        "expected_freq": expected_freq,
        "iso_time": wallclock_iso_timestamp(),
        "perf_counter_sec": float(time.perf_counter()),
        **dict(details or {}),
    }


def _marker_occurrences(marker_channel: np.ndarray) -> list[dict[str, int]]:
    values = np.asarray(marker_channel, dtype=np.float64)
    occurrences: list[dict[str, int]] = []
    for sample_index in np.flatnonzero(np.abs(values) > 1e-9):
        occurrences.append(
            {
                "marker_code": int(round(float(values[int(sample_index)]))),
                "sample_index": int(sample_index),
            }
        )
    return occurrences


def match_ssvep_events_to_markers(
    board_data: np.ndarray,
    *,
    marker_row: int,
    event_log: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    matrix = np.asarray(board_data, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("board_data must be a 2-D BrainFlow matrix")
    if int(marker_row) < 0 or int(marker_row) >= int(matrix.shape[0]):
        raise ValueError(f"marker_row out of range: {marker_row}")

    expected_events = [dict(event) for event in event_log]
    if not expected_events:
        return []

    occurrences = _marker_occurrences(matrix[int(marker_row), :])
    expected_codes = [int(event["marker_code"]) for event in expected_events]
    if expected_codes and expected_codes[0] == SSVEP_MARKER_SESSION_START:
        start_candidates = [
            index for index, marker in enumerate(occurrences) if int(marker["marker_code"]) == SSVEP_MARKER_SESSION_START
        ]
        if not start_candidates:
            raise ValueError("SSVEP session_start marker was not recorded")
        occurrences = occurrences[start_candidates[0] :]
    if expected_codes and expected_codes[-1] == SSVEP_MARKER_SESSION_END:
        end_index = next(
            (
                index
                for index, marker in enumerate(occurrences)
                if int(marker["marker_code"]) == SSVEP_MARKER_SESSION_END
            ),
            None,
        )
        if end_index is None:
            raise ValueError("SSVEP session_end marker was not recorded")
        occurrences = occurrences[: end_index + 1]

    recorded_codes = [int(marker["marker_code"]) for marker in occurrences[: len(expected_codes)]]
    if recorded_codes != expected_codes:
        raise ValueError(
            "SSVEP marker sequence mismatch: "
            f"expected={expected_codes}, recorded={recorded_codes}"
        )
    if len(occurrences) < len(expected_codes):
        raise ValueError(
            "SSVEP marker sequence length mismatch: "
            f"expected={len(expected_codes)}, recorded={len(occurrences)}"
        )

    enriched: list[dict[str, Any]] = []
    for event, marker in zip(expected_events, occurrences):
        merged = dict(event)
        merged["sample_index"] = int(marker["sample_index"])
        enriched.append(merged)
    return enriched


def derive_ssvep_trial_segments_from_markers(
    board_data: np.ndarray,
    *,
    eeg_rows: Sequence[int],
    marker_row: int,
    event_log: Sequence[dict[str, Any]],
    trials: Sequence[Any],
    sampling_rate: int,
    default_active_sec: float,
    trial_runtime_rows: Optional[Sequence[dict[str, Any]]] = None,
    min_quality_ratio: float = MIN_TRIAL_QUALITY_RATIO,
) -> tuple[list[tuple[Any, np.ndarray]], list[dict[str, Any]], list[dict[str, Any]]]:
    matrix = np.asarray(board_data, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("board_data must be a 2-D BrainFlow matrix")
    matched_events = match_ssvep_events_to_markers(matrix, marker_row=int(marker_row), event_log=event_log)

    runtime_by_order = {
        int(row.get("trial_order", -1)): dict(row)
        for row in list(trial_runtime_rows or [])
        if isinstance(row, dict)
    }
    starts = {
        int(event.get("trial_order", -1)): event
        for event in matched_events
        if int(event.get("marker_code", 0)) == SSVEP_MARKER_ACTIVE_WINDOW_START
    }
    ends = {
        int(event.get("trial_order", -1)): event
        for event in matched_events
        if int(event.get("marker_code", 0)) == SSVEP_MARKER_ACTIVE_WINDOW_END
    }
    first_frames = {
        int(event.get("trial_order", -1)): event
        for event in matched_events
        if int(event.get("marker_code", 0)) == SSVEP_MARKER_STIM_FIRST_FRAME
    }

    selected_rows = [int(row) for row in eeg_rows]
    saved_segments: list[tuple[Any, np.ndarray]] = []
    quality_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    fs = int(sampling_rate)

    for protocol_order, trial in enumerate(trials, start=1):
        start_event = starts.get(protocol_order)
        if start_event is None:
            continue
        runtime_row = runtime_by_order.get(protocol_order, {})
        target_samples = int(
            runtime_row.get(
                "target_samples",
                round(float(runtime_row.get("active_sec", default_active_sec)) * float(fs)),
            )
        )
        target_samples = max(1, int(target_samples))
        start_sample = int(start_event["sample_index"])
        end_sample = int(ends.get(protocol_order, {}).get("sample_index", start_sample + target_samples))
        stop_sample = min(int(matrix.shape[1]), start_sample + target_samples)
        segment = np.asarray(matrix[selected_rows, start_sample:stop_sample], dtype=np.float64)
        used_samples = int(segment.shape[1])
        available_samples = max(0, min(end_sample, int(matrix.shape[1])) - start_sample)
        sample_ratio = float(used_samples / max(target_samples, 1))
        base_quality = {
            "protocol_order_index": int(protocol_order - 1),
            "trial_order": int(protocol_order),
            "target_samples": int(target_samples),
            "used_samples": int(used_samples),
            "available_samples": int(available_samples),
            "sample_ratio": float(sample_ratio),
            "shortfall_ratio": float(max(target_samples - used_samples, 0) / max(target_samples, 1)),
            "active_window_start_sample": int(start_sample),
            "active_window_end_sample": int(end_sample),
            "stimulus_first_frame_marker_sample": int(first_frames.get(protocol_order, {}).get("sample_index", -1)),
            **runtime_row,
        }
        if sample_ratio >= float(min_quality_ratio):
            base_quality["order_index"] = int(len(saved_segments))
            saved_segments.append((trial, np.ascontiguousarray(segment)))
            quality_rows.append(base_quality)
        else:
            rejected_rows.append(base_quality)

    return saved_segments, quality_rows, rejected_rows


@dataclass(frozen=True)
class SSVEPUnifiedConfig:
    serial_port: str
    board_id: int
    freqs: tuple[float, float, float, float]
    subject_id: str
    session_id: str
    dataset_dir: Path
    prepare_sec: float
    active_sec: float
    rest_sec: float
    target_repeats: int
    idle_repeats: int
    switch_trials: int
    long_idle_sec: float
    seed: int
    stim_refresh_rate_hz: float
    stimulus_mode: str
    simulation_only: bool = False
    ack_timeout_sec: float = STIMULUS_PHASE_APPLY_TIMEOUT_SEC
    max_trial_retries: int = MAX_TRIAL_RETRIES


class SSVEPProtocolOnlyWorker(QObject):
    phase_changed = pyqtSignal(object)
    voice_prompt_event = pyqtSignal(object)
    log = pyqtSignal(str)
    done = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        config: SSVEPUnifiedConfig,
        *,
        marker_writer: Callable[[float], tuple[bool, str]] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.marker_writer = marker_writer
        self._stop_event = threading.Event()
        self._stimulus_ack_event = threading.Event()
        self._stimulus_ack_lock = threading.Lock()
        self._last_stimulus_ack: dict[str, Any] = {}

    def request_stop(self) -> None:
        self._stop_event.set()
        self._stimulus_ack_event.set()

    def notify_stimulus_phase_applied(self, payload: Optional[dict[str, Any]] = None) -> None:
        payload = dict(payload or {})
        if str(payload.get("mode", "")) != PHASE_CAL_ACTIVE:
            return
        payload["ack_wall_time"] = wallclock_iso_timestamp()
        payload["ack_perf_counter_sec"] = float(time.perf_counter())
        with self._stimulus_ack_lock:
            self._last_stimulus_ack = payload
        self._stimulus_ack_event.set()

    def _emit_phase(self, mode: str, title: str, detail: str, *, flicker: bool, cue_freq: float | None) -> None:
        self.phase_changed.emit(
            {
                "mode": str(mode),
                "title": str(title),
                "detail": str(detail),
                "flicker": bool(flicker),
                "cue_freq": cue_freq,
            }
        )

    def _sleep_interruptible(self, seconds: float) -> bool:
        return bool(self._stop_event.wait(max(0.0, float(seconds))))

    def _clear_ack(self) -> None:
        self._stimulus_ack_event.clear()
        with self._stimulus_ack_lock:
            self._last_stimulus_ack = {}

    def _wait_for_ack(self) -> tuple[bool, dict[str, Any]]:
        if self._stop_event.is_set():
            return False, {}
        ready = self._stimulus_ack_event.wait(max(0.0, float(self.config.ack_timeout_sec)))
        with self._stimulus_ack_lock:
            payload = dict(self._last_stimulus_ack)
        return bool(ready) and not self._stop_event.is_set(), payload

    def _write_marker(self, event: dict[str, Any], event_log: list[dict[str, Any]]) -> None:
        if bool(self.config.simulation_only):
            event_log.append(dict(event))
            return
        if self.marker_writer is None:
            raise RuntimeError("SSVEP marker writer is not available")
        ok, message = self.marker_writer(float(event["marker_code"]))
        if not ok:
            raise RuntimeError(message or f"failed to write SSVEP marker {event['marker_code']}")
        event_log.append(dict(event))

    @pyqtSlot()
    def run(self) -> None:
        event_log: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        started_at = wallclock_iso_timestamp()
        collection_aborted = False
        failure_reason = ""
        trials: list[Any] = []
        try:
            protocol = CollectionProtocol(
                name="unified_ssvep",
                prepare_sec=float(self.config.prepare_sec),
                active_sec=float(self.config.active_sec),
                rest_sec=float(self.config.rest_sec),
                target_repeats=int(self.config.target_repeats),
                idle_repeats=int(self.config.idle_repeats),
                switch_trials=int(self.config.switch_trials),
                long_idle_sec=float(self.config.long_idle_sec),
            )
            trials = build_collection_trials(self.config.freqs, protocol, seed=int(self.config.seed))
            total_trials = int(len(trials))
            self._emit_phase(PHASE_CAL_REST, "SSVEP ready", "Prepare for unified SSVEP collection.", flicker=False, cue_freq=None)
            self._write_marker(_jsonable_event(marker_code=SSVEP_MARKER_SESSION_START), event_log)

            for trial_index, trial in enumerate(trials, start=1):
                if self._stop_event.is_set():
                    collection_aborted = True
                    break
                retry_count = 0
                accepted = False
                trial_label = str(getattr(trial, "label", ""))
                cue_freq = getattr(trial, "expected_freq", None)
                cue_freq_float = None if cue_freq is None else float(cue_freq)
                is_long_idle = "long_idle" in trial_label.lower() or "long idle" in trial_label.lower()
                active_sec = (
                    float(self.config.long_idle_sec)
                    if is_long_idle and float(self.config.long_idle_sec) > 0.0
                    else float(self.config.active_sec)
                )
                while not accepted and not self._stop_event.is_set():
                    prompt = prompt_text_for_trial(self.config.freqs, trial)
                    self.voice_prompt_event.emit({"text": str(prompt), "stop": False})
                    self._emit_phase(
                        PHASE_CAL_PREPARE,
                        f"Prepare {trial_index}/{total_trials}",
                        str(prompt),
                        flicker=False,
                        cue_freq=cue_freq_float,
                    )
                    play_collection_tone_event({"event": "prepare_start"})
                    if self._sleep_interruptible(float(self.config.prepare_sec)):
                        collection_aborted = True
                        break

                    self.voice_prompt_event.emit({"text": "", "stop": True})
                    self._clear_ack()
                    apply_perf = time.perf_counter()
                    apply_wall = wallclock_iso_timestamp()
                    self._emit_phase(
                        PHASE_CAL_ACTIVE,
                        "Stimulus armed",
                        str(prompt),
                        flicker=True,
                        cue_freq=cue_freq_float,
                    )
                    ack_ready, ack_payload = self._wait_for_ack()
                    if not ack_ready:
                        retry_count += 1
                        self.log.emit(
                            f"Trial {trial_index} did not receive first-frame ack; retry "
                            f"{retry_count}/{self.config.max_trial_retries}."
                        )
                        if retry_count > int(self.config.max_trial_retries):
                            raise RuntimeError(f"Trial {trial_index} first-frame ack timed out")
                        self._emit_phase(
                            PHASE_CAL_REST,
                            "Retrying",
                            "No rendered first-frame ack was received.",
                            flicker=False,
                            cue_freq=None,
                        )
                        if self._sleep_interruptible(max(0.2, float(self.config.rest_sec) * 0.5)):
                            collection_aborted = True
                            break
                        continue

                    first_frame_event = _jsonable_event(
                        marker_code=SSVEP_MARKER_STIM_FIRST_FRAME,
                        trial_order=trial_index,
                        trial_id=int(getattr(trial, "trial_id", -1)),
                        label=trial_label,
                        expected_freq=cue_freq_float,
                        details={
                            "stimulus_phase_apply_requested_at": apply_wall,
                            "stimulus_first_frame_presented_at": str(ack_payload.get("ack_wall_time", "")),
                            "stimulus_first_frame_presented_t_sec": ack_payload.get("presented_t_sec"),
                            "stimulus_first_frame_frame_index": ack_payload.get("frame_index"),
                            "stimulus_first_frame_cue_freq": ack_payload.get("cue_freq"),
                            "stimulus_first_frame_mode": ack_payload.get("mode"),
                            "stimulus_first_frame_ack_latency_sec": max(
                                0.0,
                                float(ack_payload.get("ack_perf_counter_sec", apply_perf)) - float(apply_perf),
                            ),
                        },
                    )
                    self._write_marker(first_frame_event, event_log)

                    play_collection_tone_event_sync({"event": "active_start"})
                    active_start_event = _jsonable_event(
                        marker_code=SSVEP_MARKER_ACTIVE_WINDOW_START,
                        trial_order=trial_index,
                        trial_id=int(getattr(trial, "trial_id", -1)),
                        label=trial_label,
                        expected_freq=cue_freq_float,
                    )
                    self._write_marker(active_start_event, event_log)
                    self._emit_phase(PHASE_CAL_ACTIVE, "Collecting", str(prompt), flicker=True, cue_freq=cue_freq_float)
                    if self._sleep_interruptible(active_sec):
                        collection_aborted = True
                        break

                    active_end_event = _jsonable_event(
                        marker_code=SSVEP_MARKER_ACTIVE_WINDOW_END,
                        trial_order=trial_index,
                        trial_id=int(getattr(trial, "trial_id", -1)),
                        label=trial_label,
                        expected_freq=cue_freq_float,
                    )
                    self._write_marker(active_end_event, event_log)
                    runtime_rows.append(
                        {
                            "trial_order": int(trial_index),
                            "trial_id": int(getattr(trial, "trial_id", -1)),
                            "label": trial_label,
                            "expected_freq": cue_freq_float,
                            "retry_count": int(retry_count),
                            "active_sec": float(active_sec),
                            "stimulus_phase_apply_requested_at": apply_wall,
                            "stimulus_first_frame_presented_at": str(ack_payload.get("ack_wall_time", "")),
                            "stimulus_first_frame_presented_t_sec": ack_payload.get("presented_t_sec"),
                            "stimulus_first_frame_frame_index": ack_payload.get("frame_index"),
                            "stimulus_first_frame_cue_freq": ack_payload.get("cue_freq"),
                            "stimulus_first_frame_mode": ack_payload.get("mode"),
                            "stimulus_first_frame_ack_timed_out": False,
                        }
                    )
                    accepted = True
                    self._emit_phase(PHASE_CAL_REST, "Rest", "Relax.", flicker=False, cue_freq=None)
                    play_collection_tone_event({"event": "active_end"})
                    if self._sleep_interruptible(float(self.config.rest_sec)):
                        collection_aborted = True
                        break

            if not any(int(event.get("marker_code", 0)) == SSVEP_MARKER_SESSION_END for event in event_log):
                self._write_marker(_jsonable_event(marker_code=SSVEP_MARKER_SESSION_END), event_log)
            self._emit_phase(PHASE_STOPPED, "SSVEP stopped", "Protocol finished.", flicker=False, cue_freq=None)
        except Exception as exc:
            collection_aborted = True
            failure_reason = str(exc)
            self.error.emit(failure_reason)
            try:
                self._emit_phase(PHASE_ERROR, "SSVEP error", failure_reason, flicker=False, cue_freq=None)
                if not bool(self.config.simulation_only):
                    self._write_marker(_jsonable_event(marker_code=SSVEP_MARKER_SESSION_END), event_log)
            except Exception:
                pass
        finally:
            ended_at = wallclock_iso_timestamp()
            self.done.emit(
                {
                    "task_type": "ssvep",
                    "simulation_only": bool(self.config.simulation_only),
                    "collection_aborted": bool(collection_aborted),
                    "failure_reason": str(failure_reason),
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "event_log": event_log,
                    "trial_runtime_rows": runtime_rows,
                    "trials": list(trials),
                    "config": self.config,
                    "completed_trials": int(len(runtime_rows)),
                    "planned_trials": int(len(trials)),
                }
            )
            self.finished.emit()


class UnifiedCollectionWindow(QMainWindow):
    capture_stop_requested = pyqtSignal()
    preview_mode_switch_requested = pyqtSignal(str, int, bool)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Unified MI / SSVEP Collection")
        self.resize(1320, 820)
        self.capture_thread: QThread | None = None
        self.capture_worker: BoardCaptureWorker | None = None
        self.protocol_thread: QThread | None = None
        self.ssvep_worker: SSVEPProtocolOnlyWorker | None = None
        self.device_info: dict[str, Any] | None = None
        self.fullscreen_stimulus: CollectionFullscreenStimWindow | None = None
        self.pending_ssvep_result: dict[str, Any] | None = None
        self.child_windows: list[QWidget] = []
        self._init_ui()
        self.refresh_serial_ports()

    def _init_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left = QWidget(root)
        left_layout = QVBoxLayout(left)
        common_group = QGroupBox("Shared device")
        common_form = QFormLayout(common_group)

        self.board_combo = QComboBox()
        for label, board_id in available_board_options():
            self.board_combo.addItem(f"{label} ({board_id})", int(board_id))
        self.serial_combo = QComboBox()
        self.serial_combo.setEditable(True)
        self.channel_names_edit = QLineEdit(",".join(DEFAULT_CHANNEL_NAMES))
        self.channel_positions_edit = QLineEdit("0,1,2,3,4,5,6,7")
        self.btn_refresh_ports = QPushButton("Refresh ports")
        self.btn_connect = QPushButton("Connect shared device")
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setEnabled(False)
        row = QHBoxLayout()
        row.addWidget(self.btn_refresh_ports)
        row.addWidget(self.btn_connect)
        row.addWidget(self.btn_disconnect)
        common_form.addRow("Board", self.board_combo)
        common_form.addRow("Serial", self.serial_combo)
        common_form.addRow("Channel names", self.channel_names_edit)
        common_form.addRow("Channel positions", self.channel_positions_edit)
        common_form.addRow(row)

        quality_row = QHBoxLayout()
        self.imp_channel_spin = QSpinBox()
        self.imp_channel_spin.setRange(1, 16)
        self.btn_eeg_mode = QPushButton("EEG preview")
        self.btn_imp_mode = QPushButton("Impedance")
        quality_row.addWidget(QLabel("Channel"))
        quality_row.addWidget(self.imp_channel_spin)
        quality_row.addWidget(self.btn_eeg_mode)
        quality_row.addWidget(self.btn_imp_mode)
        common_form.addRow("Quality", quality_row)
        left_layout.addWidget(common_group)

        self.mode_tabs = QTabWidget()
        self.mode_tabs.addTab(self._build_mi_tab(), "MI")
        self.mode_tabs.addTab(self._build_ssvep_tab(), "SSVEP")
        left_layout.addWidget(self.mode_tabs, 1)

        right = QWidget(root)
        right_layout = QVBoxLayout(right)
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("font-weight: 700; font-size: 16px;")
        self.preview_widget = RealtimeEEGPreviewWidget()
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.preview_widget, 2)
        right_layout.addWidget(self.log_text, 1)

        layout.addWidget(left, 0)
        layout.addWidget(right, 1)

        self.btn_refresh_ports.clicked.connect(self.refresh_serial_ports)
        self.btn_connect.clicked.connect(self.connect_shared_device)
        self.btn_disconnect.clicked.connect(self.disconnect_shared_device)
        self.btn_eeg_mode.clicked.connect(lambda: self._request_preview_mode("EEG"))
        self.btn_imp_mode.clicked.connect(lambda: self._request_preview_mode("IMP"))
        self.btn_open_mi.clicked.connect(self.open_legacy_mi_collector)
        self.btn_pick_ssvep_dir.clicked.connect(self.pick_ssvep_dataset_dir)
        self.btn_start_ssvep.clicked.connect(self.start_ssvep_collection)
        self.btn_stop_ssvep.clicked.connect(self.stop_ssvep_collection)

    def _build_mi_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(
            "MI collection keeps the mature MI workflow unchanged. "
            "Use this button to open the full MI collector; the unified shell is used for SSVEP shared capture."
        )
        label.setWordWrap(True)
        self.btn_open_mi = QPushButton("Open MI collector")
        layout.addWidget(label)
        layout.addWidget(self.btn_open_mi)
        layout.addStretch(1)
        return widget

    def _build_ssvep_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()
        self.ssvep_freqs_edit = QLineEdit("8,10,12,15")
        self.ssvep_subject_edit = QLineEdit("subject001")
        self.ssvep_session_edit = QLineEdit(datetime.now().strftime("ssvep_%Y%m%d_%H%M%S"))
        self.ssvep_dataset_dir_edit = QLineEdit(str(DEFAULT_SSVEP_DATASET_DIR))
        self.btn_pick_ssvep_dir = QPushButton("Pick dataset dir")
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.ssvep_dataset_dir_edit)
        dir_row.addWidget(self.btn_pick_ssvep_dir)
        self.ssvep_prepare_spin = QDoubleSpinBox()
        self.ssvep_prepare_spin.setRange(0.0, 30.0)
        self.ssvep_prepare_spin.setValue(1.0)
        self.ssvep_prepare_spin.setSingleStep(0.5)
        self.ssvep_active_spin = QDoubleSpinBox()
        self.ssvep_active_spin.setRange(1.5, 30.0)
        self.ssvep_active_spin.setValue(4.0)
        self.ssvep_active_spin.setSingleStep(0.5)
        self.ssvep_rest_spin = QDoubleSpinBox()
        self.ssvep_rest_spin.setRange(0.0, 30.0)
        self.ssvep_rest_spin.setValue(1.0)
        self.ssvep_rest_spin.setSingleStep(0.5)
        self.ssvep_target_spin = QSpinBox()
        self.ssvep_target_spin.setRange(1, 200)
        self.ssvep_target_spin.setValue(4)
        self.ssvep_idle_spin = QSpinBox()
        self.ssvep_idle_spin.setRange(0, 200)
        self.ssvep_idle_spin.setValue(4)
        self.ssvep_switch_spin = QSpinBox()
        self.ssvep_switch_spin.setRange(0, 200)
        self.ssvep_switch_spin.setValue(0)
        self.ssvep_long_idle_spin = QDoubleSpinBox()
        self.ssvep_long_idle_spin.setRange(0.0, 120.0)
        self.ssvep_long_idle_spin.setValue(0.0)
        self.ssvep_seed_spin = QSpinBox()
        self.ssvep_seed_spin.setRange(1, 999999999)
        self.ssvep_seed_spin.setValue(20260410)
        self.ssvep_refresh_spin = QDoubleSpinBox()
        self.ssvep_refresh_spin.setRange(0.0, 1000.0)
        self.ssvep_refresh_spin.setValue(0.0)
        self.ssvep_refresh_spin.setSuffix(" Hz")
        self.ssvep_stimulus_mode_combo = QComboBox()
        self.ssvep_stimulus_mode_combo.addItem("Elapsed time sine", STIMULUS_MODE_ELAPSED_TIME_SINE)
        self.ssvep_stimulus_mode_combo.addItem("Frame locked sine", STIMULUS_MODE_FRAME_LOCKED_SINE)
        self.ssvep_simulation_check = QCheckBox("Simulation only")
        self.ssvep_simulation_check.setChecked(False)

        form.addRow("Frequencies", self.ssvep_freqs_edit)
        form.addRow("Subject", self.ssvep_subject_edit)
        form.addRow("Session", self.ssvep_session_edit)
        form.addRow("Dataset dir", dir_row)
        form.addRow("Prepare sec", self.ssvep_prepare_spin)
        form.addRow("Active sec", self.ssvep_active_spin)
        form.addRow("Rest sec", self.ssvep_rest_spin)
        form.addRow("Target repeats", self.ssvep_target_spin)
        form.addRow("Idle repeats", self.ssvep_idle_spin)
        form.addRow("Switch trials", self.ssvep_switch_spin)
        form.addRow("Long idle sec", self.ssvep_long_idle_spin)
        form.addRow("Seed", self.ssvep_seed_spin)
        form.addRow("Stim refresh (0=auto)", self.ssvep_refresh_spin)
        form.addRow("Stimulus mode", self.ssvep_stimulus_mode_combo)
        form.addRow("Run mode", self.ssvep_simulation_check)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.btn_start_ssvep = QPushButton("Start SSVEP")
        self.btn_stop_ssvep = QPushButton("Stop SSVEP")
        self.btn_stop_ssvep.setEnabled(False)
        button_row.addWidget(self.btn_start_ssvep)
        button_row.addWidget(self.btn_stop_ssvep)
        layout.addLayout(button_row)
        layout.addStretch(1)
        return widget

    def log(self, text: str) -> None:
        self.log_text.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")

    def current_board_id(self) -> int:
        value = self.board_combo.currentData()
        return int(value if value is not None else 0)

    def refresh_serial_ports(self) -> None:
        current = self.serial_combo.currentText().strip()
        self.serial_combo.clear()
        ports = list(detect_serial_ports())
        self.serial_combo.addItems(ports)
        synthetic = getattr(BoardIds, "SYNTHETIC_BOARD", None)
        if synthetic is not None and int(self.current_board_id()) == int(synthetic.value):
            self.serial_combo.addItem("")
        if current:
            index = self.serial_combo.findText(current)
            if index < 0:
                self.serial_combo.addItem(current)
                index = self.serial_combo.findText(current)
            self.serial_combo.setCurrentIndex(index)
        elif ports:
            self.serial_combo.setCurrentIndex(0)
        self.log(f"Detected serial ports: {ports or 'none'}")

    def _set_busy(self, busy: bool) -> None:
        for widget in (
            self.board_combo,
            self.serial_combo,
            self.channel_names_edit,
            self.channel_positions_edit,
            self.btn_connect,
            self.btn_disconnect,
            self.btn_start_ssvep,
            self.btn_open_mi,
        ):
            widget.setEnabled(not busy)
        self.mode_tabs.setTabEnabled(0, not busy)
        self.mode_tabs.setTabEnabled(1, True)
        self.btn_stop_ssvep.setEnabled(busy)

    def connect_shared_device(self) -> None:
        if self.capture_thread is not None:
            self.log("Shared device is already connected.")
            return
        try:
            channel_names = parse_channel_names(self.channel_names_edit.text(), expected_count=None)
            channel_positions = parse_channel_positions(self.channel_positions_edit.text(), expected_count=len(channel_names))
        except Exception as exc:
            QMessageBox.warning(self, "Invalid channels", str(exc))
            return
        serial_port = self.serial_combo.currentText().strip()
        board_id = self.current_board_id()
        self.capture_thread = QThread(self)
        self.capture_worker = BoardCaptureWorker(
            board_id=board_id,
            serial_port=serial_port,
            channel_positions=channel_positions,
            channel_names=channel_names,
        )
        self.capture_worker.moveToThread(self.capture_thread)
        self.capture_thread.started.connect(self.capture_worker.run)
        self.capture_worker.connection_ready.connect(self._on_capture_ready)
        self.capture_worker.preview_data_ready.connect(self.preview_widget.append_chunk)
        self.capture_worker.status_changed.connect(self.log)
        self.capture_worker.error_occurred.connect(self._on_capture_error)
        self.capture_worker.session_data_ready.connect(self._on_capture_session_data_ready)
        self.capture_worker.quality_mode_switch_finished.connect(lambda payload: self.log(str(payload)))
        self.capture_worker.finished.connect(self.capture_thread.quit)
        self.capture_worker.finished.connect(self.capture_worker.deleteLater)
        self.capture_thread.finished.connect(self._on_capture_thread_finished)
        self.capture_thread.finished.connect(self.capture_thread.deleteLater)
        self.capture_stop_requested.connect(self.capture_worker.request_stop, Qt.DirectConnection)
        self.preview_mode_switch_requested.connect(self.capture_worker.request_quality_mode_switch, Qt.DirectConnection)
        self.status_label.setText("Connecting shared device...")
        self.capture_thread.start()

    def _on_capture_ready(self, payload: object) -> None:
        info = dict(payload or {})
        self.device_info = info
        fs = float(info.get("sampling_rate", 0.0) or 0.0)
        channel_names = [str(item) for item in info.get("channel_names", [])]
        self.preview_widget.configure_stream(sampling_rate=fs, channel_names=channel_names)
        self.status_label.setText(f"Shared device connected | fs={fs:g} Hz")
        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(True)
        self.log(f"Shared device ready: {info}")

    def _on_capture_error(self, text: str) -> None:
        self.status_label.setText("Device error")
        self.log(str(text))

    def _on_capture_thread_finished(self) -> None:
        self.capture_worker = None
        self.capture_thread = None
        try:
            self.capture_stop_requested.disconnect()
        except Exception:
            pass
        try:
            self.preview_mode_switch_requested.disconnect()
        except Exception:
            pass
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        if self.pending_ssvep_result is None and self.status_label.text() not in {"SSVEP saved", "SSVEP save failed"}:
            self.status_label.setText("Shared device disconnected")

    def _on_capture_session_data_ready(self, payload: object) -> None:
        if self.pending_ssvep_result is not None:
            try:
                result = self._save_ssvep_from_capture_payload(dict(payload or {}), self.pending_ssvep_result)
            except Exception as exc:
                self.status_label.setText("SSVEP save failed")
                self.log(f"SSVEP save failed: {exc}")
            else:
                self.status_label.setText("SSVEP saved")
                self.log(f"SSVEP saved: {result.get('dataset_manifest', '')}")
            finally:
                self.pending_ssvep_result = None
            return
        self.log("Shared device returned data without a pending SSVEP save.")

    def disconnect_shared_device(self) -> None:
        if self.capture_worker is None:
            return
        self.status_label.setText("Disconnecting shared device...")
        self.capture_stop_requested.emit()

    def _request_preview_mode(self, mode: str) -> None:
        if self.capture_worker is None:
            self.log("Connect shared device before switching preview mode.")
            return
        self.preview_mode_switch_requested.emit(str(mode), int(self.imp_channel_spin.value()), False)

    def open_legacy_mi_collector(self) -> None:
        window = MIDataCollectorWindow(initial_config={"output_root": str(DEFAULT_MI_OUTPUT_ROOT)})
        self._attach_mi_index_writer(window)
        self.child_windows.append(window)
        window.destroyed.connect(lambda: self.child_windows.remove(window) if window in self.child_windows else None)
        window.show()
        window.raise_()
        window.activateWindow()

    def _attach_mi_index_writer(self, window: MIDataCollectorWindow) -> None:
        original_handler = window.on_session_save_completed

        def _wrapped_on_session_save_completed(result: dict) -> None:
            original_handler(result)
            settings = getattr(window, "current_settings", None)
            if settings is None:
                return
            append_unified_collection_index(
                UNIFIED_COLLECTION_INDEX_PATH,
                {
                    "task_type": "mi",
                    "subject_id": str(getattr(settings, "subject_id", "")),
                    "session_id": str(getattr(settings, "session_id", "")),
                    "board_id": int(getattr(settings, "board_id", 0)),
                    "serial_port": str(getattr(settings, "serial_port", "")),
                    "sampling_rate": str(result.get("sampling_rate_hz", "")),
                    "started_at": "",
                    "ended_at": wallclock_iso_timestamp(),
                    "status": "completed",
                    "native_manifest_path": str(result.get("meta_json_path", "")),
                    "continuous_path": str(result.get("continuous_path", "")),
                },
            )

        window.on_session_save_completed = _wrapped_on_session_save_completed  # type: ignore[method-assign]

    def pick_ssvep_dataset_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pick SSVEP dataset directory", self.ssvep_dataset_dir_edit.text())
        if path:
            self.ssvep_dataset_dir_edit.setText(path)

    def _read_ssvep_config(self) -> SSVEPUnifiedConfig:
        raw_freqs = parse_freqs(self.ssvep_freqs_edit.text().strip())
        freqs = tuple(float(value) for value in raw_freqs)
        if len(freqs) != 4:
            raise ValueError("SSVEP requires exactly four frequencies")
        refresh = float(self.ssvep_refresh_spin.value())
        if refresh <= 1.0:
            refresh = resolve_collection_stim_refresh_rate_hz(QApplication.primaryScreen())
        validate_stimulus_frequency_set(freqs, refresh_rate_hz=refresh)
        stimulus_mode = validate_stimulus_mode(str(self.ssvep_stimulus_mode_combo.currentData()))
        return SSVEPUnifiedConfig(
            serial_port=self.serial_combo.currentText().strip(),
            board_id=self.current_board_id(),
            freqs=(float(freqs[0]), float(freqs[1]), float(freqs[2]), float(freqs[3])),
            subject_id=self.ssvep_subject_edit.text().strip() or "subject001",
            session_id=self.ssvep_session_edit.text().strip() or datetime.now().strftime("ssvep_%Y%m%d_%H%M%S"),
            dataset_dir=resolve_ssvep_dataset_dir(self.ssvep_dataset_dir_edit.text()),
            prepare_sec=float(self.ssvep_prepare_spin.value()),
            active_sec=float(self.ssvep_active_spin.value()),
            rest_sec=float(self.ssvep_rest_spin.value()),
            target_repeats=int(self.ssvep_target_spin.value()),
            idle_repeats=int(self.ssvep_idle_spin.value()),
            switch_trials=int(self.ssvep_switch_spin.value()),
            long_idle_sec=float(self.ssvep_long_idle_spin.value()),
            seed=int(self.ssvep_seed_spin.value()),
            stim_refresh_rate_hz=float(refresh),
            stimulus_mode=stimulus_mode,
            simulation_only=bool(self.ssvep_simulation_check.isChecked()),
        )

    def _show_fullscreen_stimulus(self, config: SSVEPUnifiedConfig) -> None:
        self._close_fullscreen_stimulus()
        self.fullscreen_stimulus = CollectionFullscreenStimWindow(
            config.freqs,
            refresh_rate_hz=float(config.stim_refresh_rate_hz),
            stimulus_mode=str(config.stimulus_mode),
        )
        self.fullscreen_stimulus.active_phase_frame_presented.connect(self._on_ssvep_first_frame_presented)
        self.fullscreen_stimulus.escape_requested.connect(self.stop_ssvep_collection)
        self.fullscreen_stimulus.showFullScreen()

    def _close_fullscreen_stimulus(self) -> None:
        if self.fullscreen_stimulus is None:
            return
        try:
            self.fullscreen_stimulus.active_phase_frame_presented.disconnect(self._on_ssvep_first_frame_presented)
        except Exception:
            pass
        try:
            self.fullscreen_stimulus.escape_requested.disconnect(self.stop_ssvep_collection)
        except Exception:
            pass
        self.fullscreen_stimulus.close_from_owner()
        self.fullscreen_stimulus = None

    def start_ssvep_collection(self) -> None:
        if self.ssvep_worker is not None:
            return
        try:
            config = self._read_ssvep_config()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid SSVEP config", str(exc))
            return
        if not bool(config.simulation_only) and self.capture_worker is None:
            QMessageBox.warning(self, "Device required", "Connect the shared device before formal SSVEP collection.")
            return

        self._show_fullscreen_stimulus(config)
        marker_writer = None if bool(config.simulation_only) else self.capture_worker.insert_marker_sync  # type: ignore[union-attr]
        self.ssvep_worker = SSVEPProtocolOnlyWorker(config, marker_writer=marker_writer)
        self.protocol_thread = QThread(self)
        self.ssvep_worker.moveToThread(self.protocol_thread)
        self.protocol_thread.started.connect(self.ssvep_worker.run)
        self.ssvep_worker.phase_changed.connect(self._on_ssvep_phase_changed)
        self.ssvep_worker.voice_prompt_event.connect(self._on_ssvep_voice_prompt_event)
        self.ssvep_worker.log.connect(self.log)
        self.ssvep_worker.error.connect(lambda text: self.log(f"SSVEP error: {text}"))
        self.ssvep_worker.done.connect(self._on_ssvep_protocol_done)
        self.ssvep_worker.finished.connect(self.protocol_thread.quit)
        self.ssvep_worker.finished.connect(self.ssvep_worker.deleteLater)
        self.protocol_thread.finished.connect(self._on_ssvep_thread_finished)
        self.protocol_thread.finished.connect(self.protocol_thread.deleteLater)
        self._set_busy(True)
        self.status_label.setText("SSVEP running")
        self.protocol_thread.start()

    def stop_ssvep_collection(self) -> None:
        if self.ssvep_worker is not None:
            self.ssvep_worker.request_stop()
        self._close_fullscreen_stimulus()

    def _on_ssvep_phase_changed(self, phase: object) -> None:
        payload = dict(phase or {})
        self.status_label.setText(str(payload.get("title", "SSVEP")))
        if self.fullscreen_stimulus is not None:
            self.fullscreen_stimulus.apply_phase(payload)

    def _on_ssvep_first_frame_presented(self, payload: object) -> None:
        if self.ssvep_worker is not None:
            self.ssvep_worker.notify_stimulus_phase_applied(dict(payload or {}))

    def _on_ssvep_voice_prompt_event(self, payload: object) -> None:
        item = dict(payload or {})
        if bool(item.get("stop", False)):
            return
        text = str(item.get("text", "")).strip()
        if text:
            self.log(f"Voice prompt: {text}")

    def _on_ssvep_protocol_done(self, payload: object) -> None:
        result = dict(payload or {})
        self._close_fullscreen_stimulus()
        if bool(result.get("simulation_only", False)):
            self.log(
                f"SSVEP simulation finished: completed={result.get('completed_trials', 0)}/"
                f"{result.get('planned_trials', 0)}"
            )
            return
        self.pending_ssvep_result = result
        self.status_label.setText("Stopping shared device for SSVEP save...")
        self.capture_stop_requested.emit()

    def _on_ssvep_thread_finished(self) -> None:
        self.ssvep_worker = None
        self.protocol_thread = None
        self._set_busy(False)

    def _save_ssvep_from_capture_payload(self, capture_payload: dict[str, Any], protocol_payload: dict[str, Any]) -> dict[str, Any]:
        config = protocol_payload["config"]
        board_data = np.asarray(capture_payload.get("brainflow_data"), dtype=np.float64)
        fs = int(round(float(capture_payload.get("sampling_rate") or 0)))
        if fs <= 0:
            raise ValueError("missing sampling_rate in shared capture payload")
        selected_rows = [int(value) for value in capture_payload.get("selected_rows", [])]
        marker_row = int(capture_payload.get("marker_row"))
        if not selected_rows:
            raise ValueError("missing selected EEG rows in shared capture payload")

        trials = list(protocol_payload.get("trials", []))
        segments, quality_rows, rejected_rows = derive_ssvep_trial_segments_from_markers(
            board_data,
            eeg_rows=selected_rows,
            marker_row=marker_row,
            event_log=list(protocol_payload.get("event_log", [])),
            trials=trials,
            sampling_rate=fs,
            default_active_sec=float(config.active_sec),
            trial_runtime_rows=list(protocol_payload.get("trial_runtime_rows", [])),
        )
        output_session_id = build_collection_output_session_id(
            str(config.session_id),
            collection_aborted=bool(protocol_payload.get("collection_aborted", False)),
            dataset_dir=Path(config.dataset_dir),
        )
        protocol_config = {
            "collection_aborted": bool(protocol_payload.get("collection_aborted", False)),
            "requested_session_id": str(config.session_id),
            "saved_session_id": str(output_session_id),
            "planned_total_trials": int(len(trials)),
            "saved_trial_count": int(len(segments)),
            "rejected_trial_count": int(len(rejected_rows)),
            "protocol_name": "unified_ssvep",
            "prepare_sec": float(config.prepare_sec),
            "active_sec": float(config.active_sec),
            "rest_sec": float(config.rest_sec),
            "long_idle_sec": float(config.long_idle_sec),
            "target_repeats": int(config.target_repeats),
            "idle_repeats": int(config.idle_repeats),
            "switch_trials": int(config.switch_trials),
            "seed": int(config.seed),
            "stimulus_mode": str(config.stimulus_mode),
            "stim_refresh_rate_hz": float(config.stim_refresh_rate_hz),
            "stim_mean": float(STIM_MEAN),
            "stim_amp": float(STIM_AMP),
            "stim_phi": float(STIM_PHI),
            "stim_frame_formula": str(STIM_FRAME_FORMULA),
            "active_start_cue_sec": float(ACTIVE_START_CUE_SEC),
            "active_saved_window": "marker_aligned_after_active_start_tone",
            "source_alignment_policy": "strict_ssvep_marker_sequence_1_to_1",
            "ssvep_marker_codes": {name: code for code, name in SSVEP_EVENT_CODE_NAMES.items()},
            "ssvep_event_log": list(protocol_payload.get("event_log", [])),
            "ssvep_rejected_trials": rejected_rows,
            "failure_reason": str(protocol_payload.get("failure_reason", "")),
            "active_stimulus_arm_sec_estimate": float(estimate_active_stimulus_arm_sec(config.stim_refresh_rate_hz)),
        }
        protocol_config.update(stimulus_backend_metadata(STIMULUS_BACKEND_PYQT_FULLSCREEN))
        protocol_config.update(
            stimulus_sample_window_alignment_metadata(
                config.freqs,
                refresh_rate_hz=float(config.stim_refresh_rate_hz),
                backend=STIMULUS_BACKEND_PYQT_FULLSCREEN,
                base_phi=STIM_PHI,
            )
        )
        continuous_info = {
            "source": "unified_board_capture_worker_full_matrix",
            "marker_row": int(marker_row),
            "timestamp_row": capture_payload.get("timestamp_row"),
            "package_num_row": capture_payload.get("package_num_row"),
            "selected_rows": selected_rows,
        }
        save_result = save_collection_dataset_bundle(
            dataset_root=Path(config.dataset_dir),
            session_id=output_session_id,
            subject_id=str(config.subject_id),
            serial_port=str(config.serial_port),
            board_id=int(config.board_id),
            sampling_rate=int(fs),
            freqs=tuple(float(value) for value in config.freqs),
            board_eeg_channels=tuple(selected_rows),
            protocol_config=protocol_config,
            trial_segments=segments,
            quality_rows=quality_rows,
            continuous_board_data=board_data,
            continuous_board_info=continuous_info,
        )
        append_unified_collection_index(
            UNIFIED_COLLECTION_INDEX_PATH,
            {
                "task_type": "ssvep",
                "subject_id": str(config.subject_id),
                "session_id": str(output_session_id),
                "board_id": int(config.board_id),
                "serial_port": str(config.serial_port),
                "sampling_rate": int(fs),
                "started_at": str(protocol_payload.get("started_at", "")),
                "ended_at": str(protocol_payload.get("ended_at", "")),
                "status": "aborted" if bool(protocol_payload.get("collection_aborted", False)) else "completed",
                "native_manifest_path": str(save_result.get("dataset_manifest", "")),
                "continuous_path": str(save_result.get("dataset_continuous_board_npz", "")),
            },
        )
        return save_result

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_ssvep_collection()
        if self.capture_worker is not None:
            self.capture_stop_requested.emit()
            if self.capture_thread is not None:
                self.capture_thread.quit()
                self.capture_thread.wait(3000)
        super().closeEvent(event)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ = argv
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = UnifiedCollectionWindow()
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
