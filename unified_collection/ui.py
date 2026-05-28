from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
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
    DATASETS_ROOT,
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


PRETRAIN_FLOW_STEPS: tuple[dict[str, object], ...] = (
    {
        "key": "ssvep_collect",
        "title": "SSVEP 数据采集",
        "detail": "按固定频率呈现视觉刺激并采集响应。",
        "duration": 32,
    },
    {
        "key": "ssvep_package",
        "title": "SSVEP 数据整理",
        "detail": "自动完成标记对齐与有效窗口整理。",
        "duration": 14,
    },
    {
        "key": "mi_collect",
        "title": "MI 数据采集",
        "detail": "按预设类别完成运动想象采集。",
        "duration": 36,
    },
    {
        "key": "mi_package",
        "title": "MI 数据整理",
        "detail": "自动完成试次质检与样本平衡。",
        "duration": 14,
    },
    {
        "key": "feature_build",
        "title": "特征构建",
        "detail": "生成 SSVEP 与 MI 的预训练特征。",
        "duration": 18,
    },
    {
        "key": "training",
        "title": "模型预训练",
        "detail": "展示训练轮次与阶段进度。",
        "duration": 42,
    },
    {
        "key": "export",
        "title": "配置生成",
        "detail": "生成可用于后续实时控制的配置。",
        "duration": 12,
    },
)


class UnifiedCollectionWindow(QMainWindow):
    capture_stop_requested = pyqtSignal()
    preview_mode_switch_requested = pyqtSignal(str, int, bool)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Unified MI / SSVEP Collection")
        self.resize(1600, 900)
        self.setMinimumSize(1200, 760)
        self.capture_thread: QThread | None = None
        self.capture_worker: BoardCaptureWorker | None = None
        self.protocol_thread: QThread | None = None
        self.ssvep_worker: SSVEPProtocolOnlyWorker | None = None
        self.device_info: dict[str, Any] | None = None
        self.fullscreen_stimulus: CollectionFullscreenStimWindow | None = None
        self.pending_ssvep_result: dict[str, Any] | None = None
        self.child_windows: list[QWidget] = []
        self.pretrain_timer = QTimer(self)
        self.pretrain_timer.setInterval(120)
        self.pretrain_step_index = 0
        self.pretrain_step_ticks = 0
        self.pretrain_completed = False
        self.pretrain_step_rows: list[dict[str, Any]] = []
        self._init_ui()
        self.refresh_serial_ports()

    def _init_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("unifiedRoot")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)

        left = QWidget(root)
        left.setObjectName("leftPanel")
        left.setMinimumWidth(820)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)
        common_group = QGroupBox("脑电设备")
        common_form = QFormLayout(common_group)
        common_form.setSpacing(10)

        self.board_combo = QComboBox()
        for label, board_id in available_board_options():
            self.board_combo.addItem(f"{label} ({board_id})", int(board_id))
        self.serial_combo = QComboBox()
        self.serial_combo.setEditable(True)
        self.channel_names_edit = QLineEdit(",".join(DEFAULT_CHANNEL_NAMES))
        self.channel_positions_edit = QLineEdit("0,1,2,3,4,5,6,7")
        self.btn_refresh_ports = QPushButton("刷新端口")
        self.btn_connect = QPushButton("连接脑电设备")
        self.btn_connect.setProperty("controlType", "primary")
        self.btn_disconnect = QPushButton("断开连接")
        self.btn_disconnect.setProperty("controlType", "danger")
        self.btn_disconnect.setEnabled(False)
        row = QHBoxLayout()
        row.addWidget(self.btn_refresh_ports)
        row.addWidget(self.btn_connect)
        row.addWidget(self.btn_disconnect)
        common_form.addRow("采集板", self.board_combo)
        common_form.addRow("串口", self.serial_combo)
        common_form.addRow("通道名称", self.channel_names_edit)
        common_form.addRow("通道位置", self.channel_positions_edit)
        common_form.addRow(row)

        quality_row = QHBoxLayout()
        self.imp_channel_spin = QSpinBox()
        self.imp_channel_spin.setRange(1, 16)
        self.btn_eeg_mode = QPushButton("EEG 预览")
        self.btn_imp_mode = QPushButton("阻抗检查")
        quality_row.addWidget(QLabel("通道"))
        quality_row.addWidget(self.imp_channel_spin)
        quality_row.addWidget(self.btn_eeg_mode)
        quality_row.addWidget(self.btn_imp_mode)
        common_form.addRow("信号检查", quality_row)
        left_layout.addWidget(common_group)

        self.mode_tabs = QTabWidget()
        self.mode_tabs.setObjectName("modeTabs")
        self.mode_tabs.addTab(self._build_pretrain_tab(), "Pretrain")
        self.mode_tabs.addTab(self._build_mi_tab(), "MI")
        self.mode_tabs.addTab(self._build_ssvep_tab(), "SSVEP")
        self.mode_tabs.tabBar().hide()
        self.mode_tabs.setCurrentIndex(0)
        left_layout.addWidget(self.mode_tabs, 1)

        right = QWidget(root)
        right.setObjectName("rightPanel")
        right.setMinimumWidth(360)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("statusLabel")
        self.preview_widget = RealtimeEEGPreviewWidget()
        self.log_text = QPlainTextEdit()
        self.log_text.setObjectName("logPanel")
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.preview_widget, 2)
        right_layout.addWidget(self.log_text, 1)

        layout.addWidget(left, 4)
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
        self.btn_start_pretrain.clicked.connect(self.start_pretrain_flow)
        self.btn_pause_pretrain.clicked.connect(self.pause_pretrain_flow)
        self.btn_reset_pretrain.clicked.connect(self.reset_pretrain_flow)
        self.pretrain_timer.timeout.connect(self._advance_pretrain_flow)
        self.pretrain_ssvep_freqs_edit.textChanged.connect(self._refresh_pretrain_plan_summary)
        self.pretrain_ssvep_rounds_spin.valueChanged.connect(self._refresh_pretrain_plan_summary)
        self.pretrain_mi_classes_edit.textChanged.connect(self._refresh_pretrain_plan_summary)
        self.pretrain_mi_trials_spin.valueChanged.connect(self._refresh_pretrain_plan_summary)
        self._reset_pretrain_state(write_log=False)
        self.setStyleSheet(self._ui_stylesheet())

    def _build_pretrain_tab(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("pretrainTab")
        outer_layout = QVBoxLayout(widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("pretrainScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        outer_layout.addWidget(scroll)

        content = QWidget()
        content.setObjectName("pretrainContent")
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 56, 16)
        layout.setSpacing(14)

        hero = QFrame()
        hero.setObjectName("pretrainHero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(18, 16, 18, 16)
        hero_layout.setSpacing(14)

        hero_copy = QVBoxLayout()
        hero_copy.setSpacing(5)
        title = QLabel("脑机接口预训练")
        title.setObjectName("heroTitle")
        subtitle = QLabel("连接脑电设备后，一键进入预设好的 SSVEP 与 MI 预训练流程。")
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)
        hero_copy.addWidget(title)
        hero_copy.addWidget(subtitle)
        hero_layout.addLayout(hero_copy, 1)

        self.pretrain_status_badge = QLabel("等待连接")
        self.pretrain_status_badge.setObjectName("statusBadge")
        self.pretrain_status_badge.setAlignment(Qt.AlignCenter)
        self.pretrain_status_badge.setMinimumWidth(110)

        hero_side = QVBoxLayout()
        hero_side.setSpacing(8)
        hero_side.addWidget(self.pretrain_status_badge, 0, Qt.AlignRight)
        hero_actions = QHBoxLayout()
        hero_actions.setSpacing(8)
        self.btn_start_pretrain = QPushButton("开始预训练")
        self.btn_start_pretrain.setProperty("controlType", "primary")
        self.btn_pause_pretrain = QPushButton("暂停")
        self.btn_pause_pretrain.setEnabled(False)
        self.btn_reset_pretrain = QPushButton("重置")
        self.btn_reset_pretrain.setProperty("controlType", "neutral")
        hero_actions.addWidget(self.btn_start_pretrain)
        hero_actions.addWidget(self.btn_pause_pretrain)
        hero_actions.addWidget(self.btn_reset_pretrain)
        hero_side.addLayout(hero_actions)
        hero_layout.addLayout(hero_side, 0)
        layout.addWidget(hero)

        body = QGridLayout()
        body.setHorizontalSpacing(14)
        body.setVerticalSpacing(14)
        layout.addLayout(body, 1)

        self._create_pretrain_preset_controls()

        device_card, device_layout = self._make_pretrain_card("设备状态")
        device_card.setObjectName("deviceStatusCard")
        self.pretrain_device_card = device_card
        self.pretrain_device_state_label = QLabel("请先连接脑电设备")
        self.pretrain_device_state_label.setObjectName("deviceStateTitle")
        self.pretrain_device_detail_label = QLabel("连接成功后，系统会自动解锁预训练流程。")
        self.pretrain_device_detail_label.setObjectName("mutedLabel")
        self.pretrain_device_detail_label.setWordWrap(True)
        self.pretrain_device_signal_label = QLabel("实时 EEG 与阻抗预览会显示在右侧。")
        self.pretrain_device_signal_label.setObjectName("mutedLabel")
        self.pretrain_device_signal_label.setWordWrap(True)
        device_layout.addWidget(self.pretrain_device_state_label)
        device_layout.addWidget(self.pretrain_device_detail_label)
        device_layout.addWidget(self.pretrain_device_signal_label)

        preset_grid = QGridLayout()
        preset_grid.setHorizontalSpacing(10)
        preset_grid.setVerticalSpacing(10)
        self._add_pretrain_summary_tile(preset_grid, 0, 0, "SSVEP", "8 / 10 / 12 / 15 Hz")
        self._add_pretrain_summary_tile(preset_grid, 0, 1, "MI", "左手 / 右手 / 双脚 / 舌头")
        self._add_pretrain_summary_tile(preset_grid, 1, 0, "采集量", "SSVEP 24 组 · MI 120 组")
        self._add_pretrain_summary_tile(preset_grid, 1, 1, "训练", "12 epochs dry-run")
        device_layout.addLayout(preset_grid)
        body.addWidget(device_card, 0, 0)
        device_card.setMinimumHeight(330)

        monitor_card, monitor_layout = self._make_pretrain_card("运行进度")
        self.pretrain_active_stage_label = QLabel("等待设备连接")
        self.pretrain_active_stage_label.setObjectName("stageTitle")
        self.pretrain_active_detail_label = QLabel("确认脑电设备连接完成后，点击开始预训练即可。")
        self.pretrain_active_detail_label.setObjectName("mutedLabel")
        self.pretrain_active_detail_label.setWordWrap(True)
        self.pretrain_overall_progress = QProgressBar()
        self.pretrain_overall_progress.setObjectName("pretrainOverallProgress")
        self.pretrain_overall_progress.setRange(0, 100)
        self.pretrain_stage_progress = QProgressBar()
        self.pretrain_stage_progress.setObjectName("pretrainStageProgress")
        self.pretrain_stage_progress.setRange(0, 100)
        self.pretrain_progress_caption = QLabel("总进度 0% | 当前阶段 0%")
        self.pretrain_progress_caption.setObjectName("mutedLabel")
        monitor_layout.addWidget(self.pretrain_active_stage_label)
        monitor_layout.addWidget(self.pretrain_active_detail_label)
        monitor_layout.addWidget(self.pretrain_overall_progress)
        monitor_layout.addWidget(self.pretrain_stage_progress)
        monitor_layout.addWidget(self.pretrain_progress_caption)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(10)
        metrics.setVerticalSpacing(10)
        self.pretrain_metric_labels: dict[str, QLabel] = {}
        self._add_pretrain_metric(metrics, 0, 0, "SSVEP 采集", "ssvep_trials")
        self._add_pretrain_metric(metrics, 0, 1, "MI 采集", "mi_trials")
        self._add_pretrain_metric(metrics, 1, 0, "训练轮次", "training_epoch")
        self._add_pretrain_metric(metrics, 1, 1, "模型准备度", "profile_readiness")
        monitor_layout.addLayout(metrics)
        body.addWidget(monitor_card, 0, 1)
        monitor_card.setMinimumHeight(330)

        flow_card, flow_layout = self._make_pretrain_card("预训练流程")
        for step_number, step in enumerate(PRETRAIN_FLOW_STEPS, start=1):
            row = QFrame()
            row.setObjectName("pretrainStep")
            row.setProperty("stepState", "pending")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 8, 10, 8)
            row_layout.setSpacing(10)

            index_label = QLabel(str(step_number))
            index_label.setObjectName("stepIndex")
            index_label.setAlignment(Qt.AlignCenter)
            index_label.setMinimumSize(24, 24)
            row_layout.addWidget(index_label, 0, Qt.AlignTop)

            text_layout = QVBoxLayout()
            text_layout.setSpacing(2)
            title_label = QLabel(str(step["title"]))
            title_label.setObjectName("stepTitle")
            detail_label = QLabel(str(step["detail"]))
            detail_label.setObjectName("stepDetail")
            detail_label.setWordWrap(True)
            text_layout.addWidget(title_label)
            text_layout.addWidget(detail_label)
            row_layout.addLayout(text_layout, 1)

            state_label = QLabel("等待")
            state_label.setObjectName("stepState")
            state_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            state_label.setMinimumWidth(52)
            row_layout.addWidget(state_label)

            self.pretrain_step_rows.append(
                {
                    "frame": row,
                    "index": index_label,
                    "title": title_label,
                    "detail": detail_label,
                    "state": state_label,
                }
            )
            flow_layout.addWidget(row)
        flow_layout.addStretch(1)
        body.addWidget(flow_card, 0, 2, 2, 1)
        flow_card.setMinimumHeight(560)

        log_card, log_layout = self._make_pretrain_card("流程提示")
        self.pretrain_log_text = QPlainTextEdit()
        self.pretrain_log_text.setObjectName("compactLogPanel")
        self.pretrain_log_text.setReadOnly(True)
        self.pretrain_log_text.setMinimumHeight(150)
        log_layout.addWidget(self.pretrain_log_text)
        body.addWidget(log_card, 1, 0, 1, 2)
        log_card.setMinimumHeight(260)
        body.setColumnStretch(0, 1)
        body.setColumnStretch(1, 1)
        body.setColumnStretch(2, 1)
        body.setRowStretch(1, 1)
        return widget

    def _create_pretrain_preset_controls(self) -> None:
        self.pretrain_subject_edit = QLineEdit("subject001")
        self.pretrain_session_edit = QLineEdit(datetime.now().strftime("pretrain_%Y%m%d_%H%M%S"))
        self.pretrain_dataset_root_edit = QLineEdit(str(DATASETS_ROOT))
        self.pretrain_ssvep_freqs_edit = QLineEdit("8,10,12,15")
        self.pretrain_ssvep_rounds_spin = QSpinBox()
        self.pretrain_ssvep_rounds_spin.setRange(1, 80)
        self.pretrain_ssvep_rounds_spin.setValue(6)
        self.pretrain_mi_classes_edit = QLineEdit("left_hand,right_hand,feet,tongue")
        self.pretrain_mi_trials_spin = QSpinBox()
        self.pretrain_mi_trials_spin.setRange(1, 200)
        self.pretrain_mi_trials_spin.setValue(30)
        self.pretrain_preset_combo = QComboBox()
        self.pretrain_preset_combo.addItems(["Fast UI dry run", "Balanced pretrain", "Thorough pretrain"])

    def _make_pretrain_card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("pretrainCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        layout.addWidget(title_label)
        return card, layout

    def _add_pretrain_metric(self, grid: QGridLayout, row: int, column: int, title: str, key: str) -> None:
        metric = QFrame()
        metric.setObjectName("metricTile")
        metric_layout = QVBoxLayout(metric)
        metric_layout.setContentsMargins(10, 8, 10, 8)
        metric_layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label = QLabel("-")
        value_label.setObjectName("metricValue")
        metric_layout.addWidget(title_label)
        metric_layout.addWidget(value_label)
        self.pretrain_metric_labels[key] = value_label
        grid.addWidget(metric, row, column)

    def _add_pretrain_summary_tile(self, grid: QGridLayout, row: int, column: int, title: str, value: str) -> None:
        tile = QFrame()
        tile.setObjectName("metricTile")
        tile_layout = QVBoxLayout(tile)
        tile_layout.setContentsMargins(10, 8, 10, 8)
        tile_layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("summaryValue")
        value_label.setWordWrap(True)
        tile_layout.addWidget(title_label)
        tile_layout.addWidget(value_label)
        grid.addWidget(tile, row, column)

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

    def _read_pretrain_plan_counts(self) -> dict[str, int]:
        try:
            freqs = tuple(float(value) for value in parse_freqs(self.pretrain_ssvep_freqs_edit.text().strip()))
        except Exception:
            freqs = (8.0, 10.0, 12.0, 15.0)
        class_names = [
            item.strip()
            for item in self.pretrain_mi_classes_edit.text().replace(";", ",").split(",")
            if item.strip()
        ]
        preset = self.pretrain_preset_combo.currentText().strip().lower()
        if "thorough" in preset:
            epochs = 40
        elif "balanced" in preset:
            epochs = 24
        else:
            epochs = 12
        return {
            "ssvep_trials": max(1, len(freqs)) * int(self.pretrain_ssvep_rounds_spin.value()),
            "mi_trials": max(1, len(class_names)) * int(self.pretrain_mi_trials_spin.value()),
            "training_epochs": epochs,
        }

    def _refresh_pretrain_plan_summary(self, *_args: object) -> None:
        if not hasattr(self, "pretrain_metric_labels"):
            return
        self._update_pretrain_progress()

    def _append_pretrain_log(self, text: str) -> None:
        if hasattr(self, "pretrain_log_text"):
            self.pretrain_log_text.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
        self.log(text)

    @staticmethod
    def _refresh_style(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _set_pretrain_step_state(self, index: int, state: str, label: str) -> None:
        if index < 0 or index >= len(self.pretrain_step_rows):
            return
        row = self.pretrain_step_rows[index]
        for key in ("frame", "index", "title", "detail", "state"):
            widget = row[key]
            widget.setProperty("stepState", state)
            self._refresh_style(widget)
        row["state"].setText(label)

    def _is_pretrain_device_ready(self) -> bool:
        return self.device_info is not None and self.capture_worker is not None

    def _refresh_pretrain_device_status(self) -> None:
        if not hasattr(self, "pretrain_device_state_label"):
            return
        ready = self._is_pretrain_device_ready()
        state = "ready" if ready else "waiting"
        self.pretrain_device_card.setProperty("deviceState", state)
        self.pretrain_device_state_label.setProperty("deviceState", state)
        self.pretrain_status_badge.setProperty("deviceState", state)
        if ready:
            info = dict(self.device_info or {})
            fs = float(info.get("sampling_rate", 0.0) or 0.0)
            channels = [str(item) for item in info.get("channel_names", [])]
            self.pretrain_device_state_label.setText("脑电设备已连接")
            self.pretrain_device_detail_label.setText(f"采样率 {fs:g} Hz，已识别 {len(channels)} 个 EEG 通道。")
            self.pretrain_device_signal_label.setText("可以开始预训练；右侧可继续观察 EEG 与阻抗预览。")
            if not self.pretrain_timer.isActive() and not self.pretrain_completed:
                self.pretrain_status_badge.setText("可以开始")
                self.btn_start_pretrain.setEnabled(True)
                self.pretrain_active_stage_label.setText("准备开始")
                self.pretrain_active_detail_label.setText("设备已连接，点击开始预训练即可进入固定流程。")
        else:
            self.pretrain_device_state_label.setText("请先连接脑电设备")
            self.pretrain_device_detail_label.setText("连接成功后，系统会自动解锁预训练流程。")
            self.pretrain_device_signal_label.setText("实时 EEG 与阻抗预览会显示在右侧。")
            if not self.pretrain_timer.isActive() and not self.pretrain_completed:
                self.pretrain_status_badge.setText("等待连接")
                self.btn_start_pretrain.setEnabled(False)
                self.pretrain_active_stage_label.setText("等待设备连接")
                self.pretrain_active_detail_label.setText("确认脑电设备连接完成后，点击开始预训练即可。")
        self._refresh_style(self.pretrain_device_card)
        self._refresh_style(self.pretrain_device_state_label)
        self._refresh_style(self.pretrain_status_badge)

    def _set_pretrain_flow_locked(self, locked: bool) -> None:
        for widget in (
            self.pretrain_subject_edit,
            self.pretrain_session_edit,
            self.pretrain_dataset_root_edit,
            self.pretrain_ssvep_freqs_edit,
            self.pretrain_ssvep_rounds_spin,
            self.pretrain_mi_classes_edit,
            self.pretrain_mi_trials_spin,
            self.pretrain_preset_combo,
        ):
            widget.setEnabled(not locked)
        if self.mode_tabs.count() >= 3:
            self.mode_tabs.setTabEnabled(1, not locked)
            self.mode_tabs.setTabEnabled(2, not locked)
        self.btn_open_mi.setEnabled(not locked)
        self.btn_start_ssvep.setEnabled(not locked)
        if not locked:
            self._refresh_pretrain_device_status()

    def _reset_pretrain_state(self, *, write_log: bool = True) -> None:
        self.pretrain_timer.stop()
        self.pretrain_step_index = 0
        self.pretrain_step_ticks = 0
        self.pretrain_completed = False
        self.pretrain_status_badge.setText("等待连接")
        self.pretrain_active_stage_label.setText("等待设备连接")
        self.pretrain_active_detail_label.setText("确认脑电设备连接完成后，点击开始预训练即可。")
        self.pretrain_progress_caption.setText("总进度 0% | 当前阶段 0%")
        self.pretrain_overall_progress.setValue(0)
        self.pretrain_stage_progress.setValue(0)
        for index in range(len(PRETRAIN_FLOW_STEPS)):
            self._set_pretrain_step_state(index, "pending", "等待")
        self.btn_start_pretrain.setText("开始预训练")
        self.btn_pause_pretrain.setText("暂停")
        self.btn_pause_pretrain.setEnabled(False)
        self.btn_reset_pretrain.setEnabled(True)
        self._set_pretrain_flow_locked(False)
        if hasattr(self, "pretrain_log_text"):
            self.pretrain_log_text.clear()
        self._update_pretrain_progress()
        self._refresh_pretrain_device_status()
        if write_log:
            self._append_pretrain_log("预训练流程已重置。")

    def start_pretrain_flow(self) -> None:
        if self.pretrain_timer.isActive():
            return
        if not self._is_pretrain_device_ready():
            self.status_label.setText("请先连接脑电设备")
            self.pretrain_status_badge.setText("等待连接")
            self._append_pretrain_log("请先连接脑电设备，连接完成后再开始预训练。")
            self._refresh_pretrain_device_status()
            return
        if self.pretrain_completed:
            self._reset_pretrain_state(write_log=False)
        if self.pretrain_step_index == 0 and self.pretrain_step_ticks == 0:
            self.pretrain_log_text.clear()
        self.pretrain_status_badge.setText("运行中")
        self.btn_start_pretrain.setEnabled(False)
        self.btn_pause_pretrain.setText("暂停")
        self.btn_pause_pretrain.setEnabled(True)
        self._set_pretrain_flow_locked(True)
        self.status_label.setText("预训练流程运行中")
        self._append_pretrain_log(
            "已进入预设预训练流程："
            f"{self.pretrain_subject_edit.text().strip() or 'subject001'} / "
            f"{self.pretrain_session_edit.text().strip() or 'pretrain_session'}."
        )
        self._update_pretrain_progress()
        self.pretrain_timer.start()

    def pause_pretrain_flow(self) -> None:
        if self.pretrain_completed:
            return
        if self.pretrain_timer.isActive():
            self.pretrain_timer.stop()
            self.pretrain_status_badge.setText("已暂停")
            self.btn_pause_pretrain.setText("继续")
            self.status_label.setText("预训练流程已暂停")
            self._append_pretrain_log("预训练流程已暂停。")
            self._update_pretrain_progress(paused=True)
            return
        self.pretrain_status_badge.setText("运行中")
        self.btn_pause_pretrain.setText("暂停")
        self.status_label.setText("预训练流程运行中")
        self._append_pretrain_log("预训练流程继续运行。")
        self.pretrain_timer.start()
        self._update_pretrain_progress()

    def reset_pretrain_flow(self) -> None:
        self._reset_pretrain_state(write_log=True)
        self.status_label.setText("Idle")

    def _advance_pretrain_flow(self) -> None:
        if self.pretrain_step_index >= len(PRETRAIN_FLOW_STEPS):
            self._finish_pretrain_flow()
            return
        step = PRETRAIN_FLOW_STEPS[self.pretrain_step_index]
        if self.pretrain_step_ticks == 0:
            self._append_pretrain_log(f"开始阶段：{step['title']}。")
        self.pretrain_step_ticks += 1
        duration = max(1, int(step["duration"]))
        if self.pretrain_step_ticks >= duration:
            self._append_pretrain_log(f"完成阶段：{step['title']}。")
            self.pretrain_step_index += 1
            self.pretrain_step_ticks = 0
            if self.pretrain_step_index >= len(PRETRAIN_FLOW_STEPS):
                self._finish_pretrain_flow()
                return
        self._update_pretrain_progress()

    def _finish_pretrain_flow(self) -> None:
        self.pretrain_timer.stop()
        self.pretrain_completed = True
        self.pretrain_status_badge.setText("已完成")
        self.pretrain_active_stage_label.setText("预训练完成")
        self.pretrain_active_detail_label.setText("当前前端流程已完成，后续可接入真实训练与配置保存。")
        self.pretrain_overall_progress.setValue(100)
        self.pretrain_stage_progress.setValue(100)
        self.pretrain_progress_caption.setText("总进度 100% | 当前阶段 100%")
        for index in range(len(PRETRAIN_FLOW_STEPS)):
            self._set_pretrain_step_state(index, "done", "完成")
        counts = self._read_pretrain_plan_counts()
        self.pretrain_metric_labels["ssvep_trials"].setText(f"{counts['ssvep_trials']}/{counts['ssvep_trials']}")
        self.pretrain_metric_labels["mi_trials"].setText(f"{counts['mi_trials']}/{counts['mi_trials']}")
        self.pretrain_metric_labels["training_epoch"].setText(f"{counts['training_epochs']}/{counts['training_epochs']}")
        self.pretrain_metric_labels["profile_readiness"].setText("100%")
        self.btn_start_pretrain.setText("重新开始")
        self.btn_pause_pretrain.setText("暂停")
        self.btn_pause_pretrain.setEnabled(False)
        self._set_pretrain_flow_locked(False)
        self.btn_start_pretrain.setEnabled(self._is_pretrain_device_ready())
        self.status_label.setText("预训练流程完成")
        self._append_pretrain_log("预训练前端流程已完成。")

    def _update_pretrain_progress(self, *, paused: bool = False) -> None:
        counts = self._read_pretrain_plan_counts()
        total_duration = sum(max(1, int(step["duration"])) for step in PRETRAIN_FLOW_STEPS)
        completed_duration = sum(
            max(1, int(step["duration"]))
            for step in PRETRAIN_FLOW_STEPS[: min(self.pretrain_step_index, len(PRETRAIN_FLOW_STEPS))]
        )
        active_index = min(self.pretrain_step_index, len(PRETRAIN_FLOW_STEPS) - 1)
        active_step = PRETRAIN_FLOW_STEPS[active_index]
        active_duration = max(1, int(active_step["duration"]))
        if self.pretrain_completed:
            overall_progress = 100
            stage_progress = 100
        else:
            overall_progress = min(100, int(round((completed_duration + self.pretrain_step_ticks) * 100 / total_duration)))
            stage_progress = min(100, int(round(self.pretrain_step_ticks * 100 / active_duration)))

        self.pretrain_overall_progress.setValue(overall_progress)
        self.pretrain_stage_progress.setValue(stage_progress)
        self.pretrain_progress_caption.setText(f"总进度 {overall_progress}% | 当前阶段 {stage_progress}%")
        waiting_at_start = (
            not self.pretrain_completed
            and not self.pretrain_timer.isActive()
            and self.pretrain_step_index == 0
            and self.pretrain_step_ticks == 0
        )
        if waiting_at_start:
            if self._is_pretrain_device_ready():
                self.pretrain_active_stage_label.setText("准备开始")
                self.pretrain_active_detail_label.setText("设备已连接，点击开始预训练即可进入固定流程。")
            else:
                self.pretrain_active_stage_label.setText("等待设备连接")
                self.pretrain_active_detail_label.setText("确认脑电设备连接完成后，点击开始预训练即可。")
        elif not self.pretrain_completed:
            self.pretrain_active_stage_label.setText(str(active_step["title"]))
            self.pretrain_active_detail_label.setText(str(active_step["detail"]))

        for index, _step in enumerate(PRETRAIN_FLOW_STEPS):
            if self.pretrain_completed or index < self.pretrain_step_index:
                self._set_pretrain_step_state(index, "done", "完成")
            elif index == self.pretrain_step_index and (self.pretrain_timer.isActive() or self.pretrain_step_ticks > 0):
                self._set_pretrain_step_state(index, "active", "暂停" if paused else "进行")
            else:
                self._set_pretrain_step_state(index, "pending", "等待")

        ssvep_ratio = 1.0 if self.pretrain_step_index > 0 else (stage_progress / 100.0 if self.pretrain_step_index == 0 else 0.0)
        mi_ratio = 1.0 if self.pretrain_step_index > 2 else (stage_progress / 100.0 if self.pretrain_step_index == 2 else 0.0)
        epoch_ratio = 1.0 if self.pretrain_step_index > 5 else (stage_progress / 100.0 if self.pretrain_step_index == 5 else 0.0)
        readiness = 100 if self.pretrain_completed else max(0, overall_progress - 4 if self.pretrain_step_index >= 4 else 0)
        self.pretrain_metric_labels["ssvep_trials"].setText(
            f"{int(round(counts['ssvep_trials'] * ssvep_ratio))}/{counts['ssvep_trials']}"
        )
        self.pretrain_metric_labels["mi_trials"].setText(
            f"{int(round(counts['mi_trials'] * mi_ratio))}/{counts['mi_trials']}"
        )
        self.pretrain_metric_labels["training_epoch"].setText(
            f"{int(round(counts['training_epochs'] * epoch_ratio))}/{counts['training_epochs']}"
        )
        self.pretrain_metric_labels["profile_readiness"].setText(f"{readiness}%")

    @staticmethod
    def _ui_stylesheet() -> str:
        return (
            "QWidget {"
            "  color: #E8EEF6;"
            "  font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;"
            "  background-color: transparent;"
            "}"
            "QMainWindow {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0B0E13, stop:1 #111821);"
            "}"
            "QWidget#unifiedRoot {"
            "  background: #0B0E13;"
            "}"
            "QWidget#leftPanel, QWidget#rightPanel {"
            "  background: #111720;"
            "  border: 1px solid #293241;"
            "  border-radius: 8px;"
            "}"
            "QWidget#leftPanel {"
            "  min-width: 780px;"
            "}"
            "QGroupBox, QFrame#pretrainCard, QFrame#pretrainHero {"
            "  border: 1px solid #2A3444;"
            "  border-radius: 8px;"
            "  background: #151B24;"
            "  margin-top: 8px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 12px;"
            "  padding: 0px 6px;"
            "  color: #A9F5D0;"
            "  background: #151B24;"
            "  font-weight: 700;"
            "}"
            "QFrame#pretrainHero {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #17202C, stop:1 #1B221C);"
            "  border-color: #3A5145;"
            "  margin-top: 0px;"
            "}"
            "QLabel#heroTitle {"
            "  color: #F4F8FB;"
            "  font-size: 18pt;"
            "  font-weight: 800;"
            "}"
            "QLabel#heroSubtitle, QLabel#mutedLabel, QLabel#stepDetail, QLabel#metricTitle {"
            "  color: #AAB7C5;"
            "}"
            "QLabel#statusBadge {"
            "  color: #0D1117;"
            "  background: #A9F5D0;"
            "  border-radius: 8px;"
            "  padding: 8px 12px;"
            "  font-weight: 800;"
            "}"
            "QLabel#statusBadge[deviceState='waiting'] {"
            "  color: #D7DEE8;"
            "  background: #2B3441;"
            "}"
            "QLabel#deviceStateTitle {"
            "  color: #F0F6FC;"
            "  font-size: 13pt;"
            "  font-weight: 800;"
            "}"
            "QLabel#deviceStateTitle[deviceState='ready'] {"
            "  color: #A9F5D0;"
            "}"
            "QLabel#cardTitle, QLabel#stageTitle {"
            "  color: #F0F6FC;"
            "  font-size: 11pt;"
            "  font-weight: 800;"
            "}"
            "QLabel#metricValue {"
            "  color: #F6C667;"
            "  font-size: 14pt;"
            "  font-weight: 800;"
            "}"
            "QFrame#metricTile {"
            "  border: 1px solid #293446;"
            "  border-radius: 8px;"
            "  background: #10161F;"
            "}"
            "QLabel#summaryValue {"
            "  color: #DCE8F4;"
            "  font-size: 10pt;"
            "  font-weight: 700;"
            "}"
            "QFrame#deviceStatusCard[deviceState='ready'] {"
            "  border-color: #57D6A6;"
            "  background: #12201C;"
            "}"
            "QFrame#deviceStatusCard[deviceState='waiting'] {"
            "  border-color: #3A4658;"
            "  background: #151B24;"
            "}"
            "QTabWidget#modeTabs::pane {"
            "  border: 1px solid #2A3444;"
            "  border-radius: 8px;"
            "  background: #111720;"
            "  top: -1px;"
            "}"
            "QTabBar::tab {"
            "  background: #121922;"
            "  color: #9CA9B8;"
            "  border: 1px solid #293241;"
            "  border-bottom: none;"
            "  border-top-left-radius: 8px;"
            "  border-top-right-radius: 8px;"
            "  padding: 8px 16px;"
            "  margin-right: 4px;"
            "  min-width: 84px;"
            "}"
            "QTabBar::tab:selected {"
            "  color: #F0F6FC;"
            "  background: #1A2330;"
            "  border-color: #426056;"
            "}"
            "QTabBar::tab:disabled {"
            "  color: #5F6B76;"
            "  background: #10141B;"
            "}"
            "QScrollArea#pretrainScroll {"
            "  border: none;"
            "  background: #111720;"
            "}"
            "QWidget#pretrainContent {"
            "  background: #111720;"
            "}"
            "QScrollBar:vertical {"
            "  background: #0D1219;"
            "  width: 10px;"
            "  margin: 0px;"
            "}"
            "QScrollBar::handle:vertical {"
            "  background: #526173;"
            "  min-height: 28px;"
            "  border-radius: 5px;"
            "}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
            "  height: 0px;"
            "}"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
            "  background: #0D1219;"
            "}"
            "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {"
            "  background: #0E141C;"
            "  border: 1px solid #2A3545;"
            "  border-radius: 8px;"
            "  color: #E8EEF6;"
            "  padding: 6px 8px;"
            "  selection-background-color: #4EC9A2;"
            "  selection-color: #0B0E13;"
            "}"
            "QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {"
            "  border: 1px solid #6BE7B3;"
            "}"
            "QPlainTextEdit#logPanel, QPlainTextEdit#compactLogPanel {"
            "  background: #080C11;"
            "  color: #C8D3E0;"
            "  font-family: Consolas, 'Microsoft YaHei UI', monospace;"
            "  font-size: 9pt;"
            "}"
            "QLabel#statusLabel {"
            "  color: #F4F8FB;"
            "  background: #151B24;"
            "  border: 1px solid #2A3444;"
            "  border-radius: 8px;"
            "  padding: 9px 12px;"
            "  font-weight: 800;"
            "  font-size: 12pt;"
            "}"
            "QPushButton {"
            "  background: #202A37;"
            "  border: 1px solid #3A4658;"
            "  border-radius: 8px;"
            "  color: #EEF4FA;"
            "  padding: 8px 12px;"
            "  font-weight: 700;"
            "}"
            "QPushButton:hover {"
            "  background: #263444;"
            "  border-color: #6BE7B3;"
            "}"
            "QPushButton:pressed {"
            "  background: #151D27;"
            "}"
            "QPushButton:disabled {"
            "  background: #151A22;"
            "  border-color: #242B35;"
            "  color: #687482;"
            "}"
            "QPushButton[controlType='primary'] {"
            "  background: #176B5A;"
            "  border-color: #42C79D;"
            "  color: #F2FFF9;"
            "}"
            "QPushButton[controlType='primary']:hover {"
            "  background: #1F856F;"
            "}"
            "QPushButton[controlType='primary']:disabled {"
            "  background: #16221F;"
            "  border-color: #2A3A36;"
            "  color: #667872;"
            "}"
            "QPushButton[controlType='danger'] {"
            "  background: #6D2632;"
            "  border-color: #B84A5B;"
            "  color: #FFECEF;"
            "}"
            "QPushButton[controlType='neutral'] {"
            "  background: #2B3441;"
            "}"
            "QProgressBar {"
            "  border: 1px solid #2D394A;"
            "  border-radius: 7px;"
            "  background: #0C1118;"
            "  color: #E8EEF6;"
            "  text-align: center;"
            "  min-height: 16px;"
            "}"
            "QProgressBar::chunk {"
            "  border-radius: 6px;"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #48D6B0, stop:0.55 #9DE2C0, stop:1 #F6C667);"
            "}"
            "QFrame#pretrainStep {"
            "  border: 1px solid #283343;"
            "  border-radius: 8px;"
            "  background: #10161F;"
            "}"
            "QFrame#pretrainStep[stepState='active'] {"
            "  border-color: #67E8B9;"
            "  background: #14251F;"
            "}"
            "QFrame#pretrainStep[stepState='done'] {"
            "  border-color: #3F8F73;"
            "  background: #12201C;"
            "}"
            "QLabel#stepIndex {"
            "  color: #0B0E13;"
            "  background: #7DEBC0;"
            "  border-radius: 8px;"
            "  font-weight: 800;"
            "}"
            "QLabel#stepIndex[stepState='pending'] {"
            "  background: #445161;"
            "  color: #D2DAE5;"
            "}"
            "QLabel#stepIndex[stepState='done'] {"
            "  background: #F6C667;"
            "}"
            "QLabel#stepTitle {"
            "  color: #E8EEF6;"
            "  font-weight: 800;"
            "}"
            "QLabel#stepTitle[stepState='pending'] {"
            "  color: #B6C1CE;"
            "}"
            "QLabel#stepState {"
            "  color: #A9F5D0;"
            "  font-weight: 800;"
            "}"
            "QLabel#stepState[stepState='pending'] {"
            "  color: #7E8A99;"
            "}"
            "QLabel#stepState[stepState='done'] {"
            "  color: #F6C667;"
            "}"
        )

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
            self.btn_start_pretrain,
        ):
            widget.setEnabled(not busy)
        self.mode_tabs.setTabEnabled(0, not busy)
        self.mode_tabs.setTabEnabled(1, not busy)
        self.mode_tabs.setTabEnabled(2, True)
        self.btn_stop_ssvep.setEnabled(busy)
        if busy:
            self.btn_pause_pretrain.setEnabled(False)

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
        self._refresh_pretrain_device_status()

    def _on_capture_error(self, text: str) -> None:
        self.status_label.setText("Device error")
        self.log(str(text))
        self.device_info = None
        self._refresh_pretrain_device_status()

    def _on_capture_thread_finished(self) -> None:
        self.capture_worker = None
        self.capture_thread = None
        self.device_info = None
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
        self._refresh_pretrain_device_status()
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
