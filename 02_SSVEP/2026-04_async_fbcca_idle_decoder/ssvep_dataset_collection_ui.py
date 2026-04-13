from __future__ import annotations

import argparse
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
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

from async_fbcca_idle_standalone import (
    BoardShim,
    DEFAULT_BOARD_ID,
    DEFAULT_STREAM_WARMUP_SEC,
    describe_runtime_error,
    ensure_stream_ready,
    normalize_serial_port,
    parse_freqs,
    prepare_board_session,
    read_recent_eeg_segment,
)
from async_fbcca_validation_ui import (
    FourArrowStimWidget,
    PHASE_CAL_ACTIVE,
    PHASE_CAL_PREPARE,
    PHASE_CAL_REST,
    PHASE_ERROR,
    PHASE_STOPPED,
)
from ssvep_core.dataset import (
    ENHANCED_45M_PROTOCOL,
    CollectionProtocol,
    build_collection_trials,
    save_collection_dataset_bundle,
)

try:
    import winsound
except Exception:
    winsound = None


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = THIS_DIR / "profiles" / "datasets"
MIN_TRIAL_QUALITY_RATIO = 0.90
MAX_TRIAL_RETRIES = 3
MIN_ACTIVE_SEC_FOR_TRAINING = 1.5

ACTIVE_START_TONE_HZ = 1200
ACTIVE_START_TONE_MS = 120
ACTIVE_END_TONE_HZ = 800
ACTIVE_END_TONE_MS = 160

DEFAULT_STABLE_PREPARE_SEC = 1.0
DEFAULT_STABLE_ACTIVE_SEC = 5.0
DEFAULT_STABLE_REST_SEC = 4.0
DEFAULT_STABLE_TARGET_REPEATS = 10
DEFAULT_STABLE_IDLE_REPEATS = 20
DEFAULT_STABLE_SWITCH_TRIALS = 14
DEFAULT_STABLE_LONG_IDLE_SEC = 0.0
DEFAULT_PRESET_NAME = "stable_12m"


@dataclass(frozen=True)
class ProtocolPreset:
    key: str
    display: str
    prepare_sec: float
    active_sec: float
    rest_sec: float
    target_repeats: int
    idle_repeats: int
    switch_trials: int
    long_idle_sec: float


STABLE_12M_PRESET = ProtocolPreset(
    key="stable_12m",
    display="稳态12分钟 (1+5+4, 目标10 空闲20 切换14)",
    prepare_sec=DEFAULT_STABLE_PREPARE_SEC,
    active_sec=DEFAULT_STABLE_ACTIVE_SEC,
    rest_sec=DEFAULT_STABLE_REST_SEC,
    target_repeats=DEFAULT_STABLE_TARGET_REPEATS,
    idle_repeats=DEFAULT_STABLE_IDLE_REPEATS,
    switch_trials=DEFAULT_STABLE_SWITCH_TRIALS,
    long_idle_sec=DEFAULT_STABLE_LONG_IDLE_SEC,
)
ENHANCED_45M_PRESET = ProtocolPreset(
    key="enhanced_45m",
    display="增强45分钟 (1+4+1, 目标24 空闲48 切换32)",
    prepare_sec=float(ENHANCED_45M_PROTOCOL.prepare_sec),
    active_sec=float(ENHANCED_45M_PROTOCOL.active_sec),
    rest_sec=float(ENHANCED_45M_PROTOCOL.rest_sec),
    target_repeats=int(ENHANCED_45M_PROTOCOL.target_repeats),
    idle_repeats=int(ENHANCED_45M_PROTOCOL.idle_repeats),
    switch_trials=int(ENHANCED_45M_PROTOCOL.switch_trials),
    long_idle_sec=float(ENHANCED_45M_PROTOCOL.long_idle_sec),
)
CUSTOM_PRESET = ProtocolPreset(
    key="custom",
    display="自定义 (手动设置)",
    prepare_sec=DEFAULT_STABLE_PREPARE_SEC,
    active_sec=DEFAULT_STABLE_ACTIVE_SEC,
    rest_sec=DEFAULT_STABLE_REST_SEC,
    target_repeats=DEFAULT_STABLE_TARGET_REPEATS,
    idle_repeats=DEFAULT_STABLE_IDLE_REPEATS,
    switch_trials=DEFAULT_STABLE_SWITCH_TRIALS,
    long_idle_sec=DEFAULT_STABLE_LONG_IDLE_SEC,
)
COLLECTION_PRESETS: dict[str, ProtocolPreset] = {
    STABLE_12M_PRESET.key: STABLE_12M_PRESET,
    ENHANCED_45M_PRESET.key: ENHANCED_45M_PRESET,
    CUSTOM_PRESET.key: CUSTOM_PRESET,
}


def normalize_preset_name(raw: Optional[str]) -> str:
    value = str(raw or "").strip().lower()
    if value in COLLECTION_PRESETS:
        return value
    return CUSTOM_PRESET.key


def trial_count_for_protocol(
    target_repeats: int,
    idle_repeats: int,
    switch_trials: int,
    *,
    long_idle_sec: float = 0.0,
) -> int:
    long_idle_trials = 1 if float(long_idle_sec) > 0.0 else 0
    return int(
        max(0, int(target_repeats)) * 4
        + max(0, int(idle_repeats))
        + max(0, int(switch_trials))
        + long_idle_trials
    )


def estimate_round_seconds(
    *,
    prepare_sec: float,
    active_sec: float,
    rest_sec: float,
    target_repeats: int,
    idle_repeats: int,
    switch_trials: int,
    long_idle_sec: float = 0.0,
) -> float:
    base_trial_count = trial_count_for_protocol(
        target_repeats,
        idle_repeats,
        switch_trials,
        long_idle_sec=0.0,
    )
    total_sec = float(base_trial_count) * float(max(0.0, prepare_sec) + max(0.0, active_sec) + max(0.0, rest_sec))
    if float(long_idle_sec) > 0.0:
        total_sec += float(max(0.0, prepare_sec) + max(0.0, long_idle_sec) + max(0.0, rest_sec))
    return total_sec


def format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    mins, secs = divmod(total, 60)
    return f"{mins}m {secs:02d}s"


def _validate_collection_protocol_legacy_unused(*, active_sec: float) -> None:
    if float(active_sec) < float(MIN_ACTIVE_SEC_FOR_TRAINING):
        raise ValueError(
            f"active_sec 必须 >= {MIN_ACTIVE_SEC_FOR_TRAINING:.1f}s，"
            "否则不满足训练质量门槛"
        )


def _validate_collection_protocol(*, active_sec: float, long_idle_sec: float = 0.0) -> None:
    if float(active_sec) < float(MIN_ACTIVE_SEC_FOR_TRAINING):
        raise ValueError(f"active_sec must be >= {MIN_ACTIVE_SEC_FOR_TRAINING:.1f}s")
    if float(long_idle_sec) < 0.0:
        raise ValueError("long_idle_sec must be >= 0")
    if 0.0 < float(long_idle_sec) < float(MIN_ACTIVE_SEC_FOR_TRAINING):
        raise ValueError(f"long_idle_sec must be 0 or >= {MIN_ACTIVE_SEC_FOR_TRAINING:.1f}s")


def _auto_session_base_id(subject_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_subject = (subject_id or "subject").strip().replace(" ", "_")
    return f"{clean_subject}_collection_{stamp}"


def _strip_round_suffix(session_base: str) -> str:
    return re.sub(r"_r\d+$", "", str(session_base).strip(), flags=re.IGNORECASE)


def _build_round_session_id(session_base: str, round_index: int) -> str:
    base = _strip_round_suffix(session_base) or "session"
    return f"{base}_r{int(round_index):02d}"


@dataclass(frozen=True)
class CollectionConfig:
    serial_port: str
    board_id: int
    freqs: tuple[float, float, float, float]
    subject_id: str
    session_id: str
    session_index: int
    dataset_dir: Path
    protocol_name: str = DEFAULT_PRESET_NAME
    prepare_sec: float = DEFAULT_STABLE_PREPARE_SEC
    active_sec: float = DEFAULT_STABLE_ACTIVE_SEC
    rest_sec: float = DEFAULT_STABLE_REST_SEC
    target_repeats: int = DEFAULT_STABLE_TARGET_REPEATS
    idle_repeats: int = DEFAULT_STABLE_IDLE_REPEATS
    switch_trials: int = DEFAULT_STABLE_SWITCH_TRIALS
    long_idle_sec: float = DEFAULT_STABLE_LONG_IDLE_SEC
    seed: int = 20260410
    rounds_planned: int = 1
    round_index: int = 1
    estimated_round_sec: float = 0.0


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


class CollectionWorker(QObject):
    phase_changed = pyqtSignal(object)
    trial_tone_event = pyqtSignal(object)
    log = pyqtSignal(str)
    done = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: CollectionConfig) -> None:
        super().__init__()
        self.config = config
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _emit_phase(self, mode: str, title: str, detail: str, *, flicker: bool, cue_freq: Optional[float]) -> None:
        self.phase_changed.emit(
            {
                "mode": mode,
                "title": title,
                "detail": detail,
                "flicker": flicker,
                "cue_freq": cue_freq,
            }
        )

    def _emit_tone(self, *, event: str, trial_index: int, total_trials: int, retry_index: int) -> None:
        self.trial_tone_event.emit(
            {
                "event": str(event),
                "round_index": int(self.config.round_index),
                "trial_index": int(trial_index),
                "total_trials": int(total_trials),
                "retry_index": int(retry_index),
            }
        )

    @pyqtSlot()
    def run(self) -> None:
        board = None
        active_serial = self.config.serial_port
        try:
            board, resolved_port, attempted_ports = prepare_board_session(self.config.board_id, self.config.serial_port)
            active_serial = resolved_port
            fs = int(BoardShim.get_sampling_rate(self.config.board_id))
            eeg_channels = tuple(int(ch) for ch in BoardShim.get_eeg_channels(self.config.board_id))
            board.start_stream(450000)
            ready = ensure_stream_ready(board, fs)
            self.log.emit(
                f"采集开始：请求串口={self.config.serial_port} -> 实际={resolved_port}，"
                f"尝试={attempted_ports}，fs={fs}Hz，通道={list(eeg_channels)}，缓存就绪={ready}，"
                f"轮次={self.config.round_index}/{self.config.rounds_planned}"
            )
            time.sleep(max(2.0, DEFAULT_STREAM_WARMUP_SEC))
            board.get_board_data()
            protocol = CollectionProtocol(
                name=str(self.config.protocol_name),
                prepare_sec=float(self.config.prepare_sec),
                active_sec=float(self.config.active_sec),
                rest_sec=float(self.config.rest_sec),
                target_repeats=int(self.config.target_repeats),
                idle_repeats=int(self.config.idle_repeats),
                switch_trials=int(self.config.switch_trials),
                long_idle_sec=float(self.config.long_idle_sec),
            )
            trials = build_collection_trials(
                self.config.freqs,
                protocol=protocol,
                seed=self.config.seed,
                session_index=self.config.session_index,
            )
            minimum_samples = max(1, int(round(1.5 * fs)))
            collected: list[tuple[Any, np.ndarray]] = []
            quality_rows: list[dict[str, Any]] = []
            total = len(trials)
            for index, trial in enumerate(trials, start=1):
                if self._stop_event.is_set():
                    break
                cue_freq = None if trial.expected_freq is None else float(trial.expected_freq)
                trial_label_lower = str(trial.label).strip().lower()
                is_long_idle = "long_idle" in trial_label_lower or "long idle" in trial_label_lower
                trial_active_sec = (
                    float(self.config.long_idle_sec)
                    if is_long_idle and float(self.config.long_idle_sec) > 0.0
                    else float(self.config.active_sec)
                )
                active_samples = int(round(trial_active_sec * fs))
                prompt_base = (
                    f"第{self.config.round_index}轮 Trial {index}/{total} 空闲（看中心）"
                    if cue_freq is None
                    else f"第{self.config.round_index}轮 Trial {index}/{total} 注视 {trial.label}"
                )
                if is_long_idle:
                    prompt_base = (
                        f"Round {self.config.round_index} Trial {index}/{total} long-idle "
                        "(keep looking at center, avoid all targets)"
                    )
                accepted_segment: Optional[np.ndarray] = None
                accepted_used_samples = 0
                accepted_shortfall_ratio = 1.0
                retry_count = 0
                available_samples = 0
                while retry_count <= MAX_TRIAL_RETRIES:
                    prompt = (
                        prompt_base
                        if retry_count == 0
                        else f"{prompt_base} | 重采 {retry_count}/{MAX_TRIAL_RETRIES}"
                    )
                    self._emit_phase(PHASE_CAL_PREPARE, "准备", prompt, flicker=False, cue_freq=cue_freq)
                    self.log.emit(prompt)
                    time.sleep(max(0.0, self.config.prepare_sec))
                    if self._stop_event.is_set():
                        break

                    board.get_board_data()
                    self._emit_tone(
                        event="active_start",
                        trial_index=index,
                        total_trials=total,
                        retry_index=retry_count,
                    )
                    self._emit_phase(PHASE_CAL_ACTIVE, "采集中", prompt, flicker=True, cue_freq=cue_freq)
                    time.sleep(max(0.0, trial_active_sec))
                    self._emit_tone(
                        event="active_end",
                        trial_index=index,
                        total_trials=total,
                        retry_index=retry_count,
                    )
                    segment, used_samples, available_samples = read_recent_eeg_segment(
                        board,
                        eeg_channels,
                        target_samples=active_samples,
                        minimum_samples=minimum_samples,
                    )
                    shortfall_ratio = float(max(active_samples - int(used_samples), 0) / max(active_samples, 1))
                    sample_ratio = float(int(used_samples) / max(active_samples, 1))
                    if sample_ratio >= float(MIN_TRIAL_QUALITY_RATIO):
                        accepted_segment = np.ascontiguousarray(segment, dtype=np.float64)
                        accepted_used_samples = int(used_samples)
                        accepted_shortfall_ratio = float(shortfall_ratio)
                        break

                    retry_count += 1
                    self.log.emit(
                        f"Trial {index} 样本不足：{used_samples}/{active_samples} "
                        f"(比例={sample_ratio:.3f}, 缓冲区={available_samples})。"
                    )
                    if retry_count > MAX_TRIAL_RETRIES:
                        raise RuntimeError(
                            f"Trial {index} 连续 {MAX_TRIAL_RETRIES} 次仍未通过质量门槛 "
                            f"(used={used_samples}, target={active_samples})"
                        )
                    self._emit_phase(
                        PHASE_CAL_REST,
                        "重采中",
                        "样本不足，正在重采该 Trial。",
                        flicker=False,
                        cue_freq=None,
                    )
                    time.sleep(max(0.2, self.config.rest_sec * 0.5))

                if self._stop_event.is_set():
                    break
                if accepted_segment is None:
                    raise RuntimeError(f"Trial {index} 未采到有效片段，流程中止")

                collected.append((trial, accepted_segment))
                quality_rows.append(
                    {
                        "order_index": int(index - 1),
                        "target_samples": int(active_samples),
                        "used_samples": int(accepted_used_samples),
                        "shortfall_ratio": float(accepted_shortfall_ratio),
                        "retry_count": int(retry_count),
                        "available_samples": int(available_samples),
                    }
                )
                self._emit_phase(PHASE_CAL_REST, "休息", "请放松并正常眨眼。", flicker=False, cue_freq=None)
                time.sleep(max(0.0, self.config.rest_sec))

            if not collected:
                raise RuntimeError("没有采集到任何 Trial")
            protocol_config = {
                "protocol_name": str(self.config.protocol_name),
                "prepare_sec": float(self.config.prepare_sec),
                "active_sec": float(self.config.active_sec),
                "rest_sec": float(self.config.rest_sec),
                "long_idle_sec": float(self.config.long_idle_sec),
                "target_repeats": int(self.config.target_repeats),
                "idle_repeats": int(self.config.idle_repeats),
                "switch_trials": int(self.config.switch_trials),
                "session_index": int(self.config.session_index),
                "seed": int(self.config.seed),
                "round_index": int(self.config.round_index),
                "rounds_planned": int(self.config.rounds_planned),
                "preset_name": str(self.config.protocol_name),
                "estimated_round_sec": float(self.config.estimated_round_sec),
            }
            metadata = save_collection_dataset_bundle(
                dataset_root=self.config.dataset_dir,
                session_id=self.config.session_id,
                subject_id=self.config.subject_id,
                serial_port=active_serial,
                board_id=self.config.board_id,
                sampling_rate=fs,
                freqs=self.config.freqs,
                board_eeg_channels=eeg_channels,
                protocol_config=protocol_config,
                trial_segments=collected,
                quality_rows=quality_rows,
            )
            self._emit_phase(PHASE_STOPPED, "采集完成", "数据已保存。", flicker=False, cue_freq=None)
            self.done.emit(
                {
                    "collected_trials": len(collected),
                    "total_trials": len(trials),
                    "round_index": int(self.config.round_index),
                    "rounds_planned": int(self.config.rounds_planned),
                    **metadata,
                }
            )
        except Exception as exc:
            self.error.emit(f"采集失败：{describe_runtime_error(exc, serial_port=active_serial)}")
            self._emit_phase(PHASE_ERROR, "采集错误", str(exc), flicker=False, cue_freq=None)
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


def run_collection_cli(config: CollectionConfig) -> dict[str, Any]:
    worker = CollectionWorker(config)
    state: dict[str, Any] = {}

    def _log(text: str) -> None:
        print(text, flush=True)

    worker.log.connect(_log)  # type: ignore[arg-type]
    worker.done.connect(lambda payload: state.update(payload))  # type: ignore[arg-type]
    worker.error.connect(lambda text: (_log(text), state.setdefault("error", text)))  # type: ignore[arg-type]
    worker.run()
    if "error" in state:
        raise RuntimeError(str(state["error"]))
    return state


class DatasetCollectionWindow(QMainWindow):
    def __init__(self, *, serial_port: str, board_id: int, freqs: Sequence[float]) -> None:
        super().__init__()
        self.setWindowTitle("SSVEP 数据采集")
        self.resize(1280, 880)

        self.default_serial = normalize_serial_port(serial_port)
        self.default_board_id = int(board_id)
        self.default_freqs = tuple(float(freq) for freq in freqs)

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[CollectionWorker] = None
        self.connect_thread: Optional[QThread] = None
        self.connect_worker: Optional[DeviceCheckWorker] = None

        self.rounds_completed = 0
        self._session_base_auto_cache: Optional[str] = None
        self._updating_preset = False

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left = QWidget(root)
        left_layout = QVBoxLayout(left)
        form = QFormLayout()

        self.serial_edit = QLineEdit(self.default_serial)
        self.board_edit = QLineEdit(str(self.default_board_id))
        self.freqs_edit = QLineEdit(",".join(f"{freq:g}" for freq in self.default_freqs))
        self.subject_edit = QLineEdit("subject001")
        self.session_index_spin = QSpinBox()
        self.session_index_spin.setRange(1, 999)
        self.session_index_spin.setValue(1)
        self.rounds_planned_spin = QSpinBox()
        self.rounds_planned_spin.setRange(1, 99)
        self.rounds_planned_spin.setValue(1)
        self.current_round_value = QLabel("1")
        self.completed_rounds_value = QLabel("0")
        self.session_base_edit = QLineEdit("")
        self.dataset_dir_edit = QLineEdit(str(DEFAULT_DATASET_DIR))
        self.preset_combo = QComboBox()
        for preset in (STABLE_12M_PRESET, ENHANCED_45M_PRESET, CUSTOM_PRESET):
            self.preset_combo.addItem(preset.display, preset.key)
        self.preset_combo.setCurrentText(STABLE_12M_PRESET.display)
        self.prepare_spin = QDoubleSpinBox()
        self.prepare_spin.setRange(0.0, 20.0)
        self.prepare_spin.setDecimals(1)
        self.prepare_spin.setSingleStep(0.5)
        self.active_spin = QDoubleSpinBox()
        self.active_spin.setRange(MIN_ACTIVE_SEC_FOR_TRAINING, 20.0)
        self.active_spin.setDecimals(1)
        self.active_spin.setSingleStep(0.5)
        self.rest_spin = QDoubleSpinBox()
        self.rest_spin.setRange(0.0, 20.0)
        self.rest_spin.setDecimals(1)
        self.rest_spin.setSingleStep(0.5)
        self.long_idle_spin = QDoubleSpinBox()
        self.long_idle_spin.setRange(0.0, 300.0)
        self.long_idle_spin.setDecimals(1)
        self.long_idle_spin.setSingleStep(5.0)
        self.target_spin = QSpinBox()
        self.target_spin.setRange(1, 60)
        self.idle_spin = QSpinBox()
        self.idle_spin.setRange(1, 120)
        self.switch_spin = QSpinBox()
        self.switch_spin.setRange(0, 120)
        self.estimate_label = QLabel("预计时长：--")

        form.addRow("串口", self.serial_edit)
        form.addRow("板卡 ID", self.board_edit)
        form.addRow("刺激频率", self.freqs_edit)
        form.addRow("被试 ID", self.subject_edit)
        form.addRow("起始轮次", self.session_index_spin)
        form.addRow("计划轮数", self.rounds_planned_spin)
        form.addRow("当前轮次", self.current_round_value)
        form.addRow("已完成轮次", self.completed_rounds_value)
        form.addRow("会话基础 ID（可选）", self.session_base_edit)
        form.addRow("数据集目录", self.dataset_dir_edit)
        form.addRow("预设协议", self.preset_combo)
        form.addRow("准备时长（秒）", self.prepare_spin)
        form.addRow("采集时长（秒）", self.active_spin)
        form.addRow("休息时长（秒）", self.rest_spin)
        form.addRow("目标重复次数", self.target_spin)
        form.addRow("空闲重复次数", self.idle_spin)
        form.addRow("切换 Trial 数", self.switch_spin)
        form.addRow("单轮预计时长", self.estimate_label)
        form.addRow("Long Idle (sec, 0=off)", self.long_idle_spin)
        left_layout.addLayout(form)

        row = QHBoxLayout()
        self.btn_pick_dir = QPushButton("选择目录")
        self.btn_connect = QPushButton("连接设备")
        self.btn_start = QPushButton("开始本轮采集")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_pick_dir)
        row.addWidget(self.btn_connect)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        left_layout.addLayout(row)

        self.phase_label = QLabel("空闲")
        self.phase_label.setStyleSheet("font-size:16px; font-weight:600;")
        left_layout.addWidget(self.phase_label)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text, 1)

        right = QWidget(root)
        right_layout = QVBoxLayout(right)
        self.stim = FourArrowStimWidget(self.default_freqs)
        right_layout.addWidget(self.stim, 1)

        layout.addWidget(left, 0)
        layout.addWidget(right, 1)

        self.config_widgets = [
            self.serial_edit,
            self.board_edit,
            self.freqs_edit,
            self.subject_edit,
            self.session_index_spin,
            self.rounds_planned_spin,
            self.session_base_edit,
            self.dataset_dir_edit,
            self.preset_combo,
            self.prepare_spin,
            self.active_spin,
            self.rest_spin,
            self.long_idle_spin,
            self.target_spin,
            self.idle_spin,
            self.switch_spin,
        ]

        self.btn_pick_dir.clicked.connect(self._pick_dataset_dir)
        self.btn_connect.clicked.connect(self._connect_device)
        self.btn_start.clicked.connect(self._start_collection)
        self.btn_stop.clicked.connect(self._stop_collection)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self.prepare_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.active_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.rest_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.long_idle_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.target_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.idle_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.switch_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.rounds_planned_spin.valueChanged.connect(self._on_round_control_changed)
        self.session_index_spin.valueChanged.connect(self._on_round_control_changed)
        self.subject_edit.textChanged.connect(self._on_session_base_source_changed)
        self.session_base_edit.textChanged.connect(self._on_session_base_source_changed)

        self._apply_preset(DEFAULT_PRESET_NAME)
        self._refresh_estimate_label()
        self._refresh_round_status()

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{stamp}] {text}")

    def _set_running(self, running: bool) -> None:
        self.btn_connect.setEnabled(not running)
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        for widget in self.config_widgets:
            widget.setEnabled(not running)

    def _pick_dataset_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录", self.dataset_dir_edit.text().strip())
        if path:
            self.dataset_dir_edit.setText(path)

    def _on_session_base_source_changed(self, _value: str) -> None:
        if self.worker_thread is not None:
            return
        self._session_base_auto_cache = None

    def _on_round_control_changed(self, _value: int) -> None:
        if self.worker_thread is not None:
            return
        self.rounds_completed = 0
        self._session_base_auto_cache = None
        self._refresh_round_status()

    def _current_preset_name(self) -> str:
        value = self.preset_combo.currentData()
        return normalize_preset_name(str(value) if value is not None else self.preset_combo.currentText())

    def _apply_preset(self, preset_name: str) -> None:
        key = normalize_preset_name(preset_name)
        preset = COLLECTION_PRESETS.get(key, CUSTOM_PRESET)
        self._updating_preset = True
        try:
            self.prepare_spin.setValue(float(preset.prepare_sec))
            self.active_spin.setValue(float(preset.active_sec))
            self.rest_spin.setValue(float(preset.rest_sec))
            self.long_idle_spin.setValue(float(preset.long_idle_sec))
            self.target_spin.setValue(int(preset.target_repeats))
            self.idle_spin.setValue(int(preset.idle_repeats))
            self.switch_spin.setValue(int(preset.switch_trials))
            self.preset_combo.setCurrentText(preset.display)
        finally:
            self._updating_preset = False

    def _on_preset_changed(self, _display: str) -> None:
        key = self._current_preset_name()
        if key in (STABLE_12M_PRESET.key, ENHANCED_45M_PRESET.key):
            self._apply_preset(key)
        self._refresh_estimate_label()

    def _on_protocol_value_changed(self, _value: float) -> None:
        if not self._updating_preset and self._current_preset_name() != CUSTOM_PRESET.key:
            self.preset_combo.setCurrentText(CUSTOM_PRESET.display)
        self._refresh_estimate_label()

    def _round_index_for_next_run(self) -> int:
        return int(self.session_index_spin.value()) + int(self.rounds_completed)

    def _refresh_round_status(self) -> None:
        planned = int(self.rounds_planned_spin.value())
        current_round = self._round_index_for_next_run()
        remaining = max(0, planned - int(self.rounds_completed))
        self.current_round_value.setText(str(current_round))
        self.completed_rounds_value.setText(f"{self.rounds_completed}（剩余 {remaining}）")

    def _refresh_estimate_label(self) -> None:
        trial_count = trial_count_for_protocol(
            target_repeats=int(self.target_spin.value()),
            idle_repeats=int(self.idle_spin.value()),
            switch_trials=int(self.switch_spin.value()),
            long_idle_sec=float(self.long_idle_spin.value()),
        )
        round_sec = estimate_round_seconds(
            prepare_sec=float(self.prepare_spin.value()),
            active_sec=float(self.active_spin.value()),
            rest_sec=float(self.rest_spin.value()),
            target_repeats=int(self.target_spin.value()),
            idle_repeats=int(self.idle_spin.value()),
            switch_trials=int(self.switch_spin.value()),
            long_idle_sec=float(self.long_idle_spin.value()),
        )
        planned = int(self.rounds_planned_spin.value())
        total_sec = round_sec * float(planned)
        self.estimate_label.setText(
            f"每轮 {trial_count} 个 Trial，单轮约 {format_duration(round_sec)}，"
            f"总计约 {format_duration(total_sec)}"
        )

    def _resolve_session_base(self, subject_id: str) -> str:
        raw = _strip_round_suffix(self.session_base_edit.text().strip())
        if raw:
            return raw
        if self._session_base_auto_cache is None:
            self._session_base_auto_cache = _auto_session_base_id(subject_id)
        return self._session_base_auto_cache

    def _read_config(self, *, round_index_override: Optional[int] = None) -> CollectionConfig:
        serial_port = normalize_serial_port(self.serial_edit.text().strip())
        board_id = int(self.board_edit.text().strip())
        freqs = parse_freqs(self.freqs_edit.text().strip())
        subject_id = self.subject_edit.text().strip() or "subject001"
        rounds_planned = int(self.rounds_planned_spin.value())
        round_index = int(round_index_override) if round_index_override is not None else int(self.session_index_spin.value())
        session_base = self._resolve_session_base(subject_id)
        session_id = _build_round_session_id(session_base, round_index)
        dataset_dir = Path(self.dataset_dir_edit.text().strip()).expanduser().resolve()
        protocol_name = self._current_preset_name()
        prepare_sec = float(self.prepare_spin.value())
        active_sec = float(self.active_spin.value())
        long_idle_sec = float(self.long_idle_spin.value())
        _validate_collection_protocol(active_sec=active_sec, long_idle_sec=long_idle_sec)
        rest_sec = float(self.rest_spin.value())
        target_repeats = int(self.target_spin.value())
        idle_repeats = int(self.idle_spin.value())
        switch_trials = int(self.switch_spin.value())
        estimated_round_sec = estimate_round_seconds(
            prepare_sec=prepare_sec,
            active_sec=active_sec,
            rest_sec=rest_sec,
            target_repeats=target_repeats,
            idle_repeats=idle_repeats,
            switch_trials=switch_trials,
            long_idle_sec=long_idle_sec,
        )
        return CollectionConfig(
            serial_port=serial_port,
            board_id=board_id,
            freqs=freqs,
            subject_id=subject_id,
            session_id=session_id,
            session_index=round_index,
            dataset_dir=dataset_dir,
            protocol_name=protocol_name,
            prepare_sec=prepare_sec,
            active_sec=active_sec,
            rest_sec=rest_sec,
            target_repeats=target_repeats,
            idle_repeats=idle_repeats,
            switch_trials=switch_trials,
            long_idle_sec=long_idle_sec,
            rounds_planned=rounds_planned,
            round_index=round_index,
            estimated_round_sec=estimated_round_sec,
        )

    def _connect_device(self) -> None:
        if self.worker_thread is not None:
            self._log("采集中，请先停止再重新连接设备。")
            return
        if self.connect_thread is not None:
            self._log("正在连接设备，请稍候。")
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
        self.phase_label.setText("连接中...")
        thread.start()

    def _on_connected(self, payload: dict[str, Any]) -> None:
        self.phase_label.setText("设备已连接")
        self._log(
            "连接成功：请求串口={requested_serial_port}，实际串口={resolved_serial_port}，"
            "采样率={sampling_rate}Hz，缓存就绪={ready_samples}".format(**payload)
        )

    def _on_connect_error(self, text: str) -> None:
        self.phase_label.setText("连接失败")
        self._log(text)

    def _on_connect_finished(self) -> None:
        self.connect_worker = None
        self.connect_thread = None

    def _start_collection(self) -> None:
        if self.worker_thread is not None:
            return
        if self.connect_thread is not None:
            self._log("设备正在连接，请等待完成。")
            return
        planned = int(self.rounds_planned_spin.value())
        if self.rounds_completed >= planned:
            self.phase_label.setText("计划轮次已完成")
            self._log("计划轮次已全部完成。请调整轮次设置后继续。")
            return
        round_index = self._round_index_for_next_run()
        try:
            cfg = self._read_config(round_index_override=round_index)
        except Exception as exc:
            self._log(f"配置错误：{exc}")
            return
        self._log(
            f"开始第 {cfg.round_index}/{cfg.rounds_planned} 轮：session={cfg.session_id}，"
            f"预计时长={format_duration(cfg.estimated_round_sec)}"
        )
        worker = CollectionWorker(cfg)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.log.connect(self._log)
        worker.error.connect(self._on_error)
        worker.done.connect(self._on_done)
        worker.phase_changed.connect(self._on_phase_changed)
        worker.trial_tone_event.connect(self._on_trial_tone_event)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_finished)
        thread.started.connect(worker.run)
        self.worker = worker
        self.worker_thread = thread
        self._set_running(True)
        self.phase_label.setText("正在启动本轮采集...")
        thread.start()

    def _stop_collection(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
        self._set_running(False)

    def _play_tone_async(self, *, frequency: int, duration_ms: int) -> None:
        if winsound is not None:
            def _beep() -> None:
                try:
                    winsound.Beep(int(frequency), int(duration_ms))
                except Exception:
                    pass

            threading.Thread(target=_beep, daemon=True).start()
            return
        app = QApplication.instance()
        if app is not None:
            app.beep()

    def _on_trial_tone_event(self, payload: dict[str, Any]) -> None:
        event = str(payload.get("event", ""))
        if event == "active_start":
            self._play_tone_async(frequency=ACTIVE_START_TONE_HZ, duration_ms=ACTIVE_START_TONE_MS)
        elif event == "active_end":
            self._play_tone_async(frequency=ACTIVE_END_TONE_HZ, duration_ms=ACTIVE_END_TONE_MS)

    def _on_phase_changed(self, phase: dict[str, Any]) -> None:
        self.phase_label.setText(str(phase.get("title", "采集中")))
        self.stim.apply_phase(phase)

    def _on_error(self, text: str) -> None:
        self._log(text)

    def _on_done(self, payload: dict[str, Any]) -> None:
        self.rounds_completed += 1
        self._refresh_round_status()
        self._log(
            "第 {round_index}/{rounds_planned} 轮完成：采集={collected_trials}/{total_trials}，"
            "manifest={dataset_manifest}，npz={dataset_npz}".format(**payload)
        )
        planned = int(self.rounds_planned_spin.value())
        if self.rounds_completed < planned:
            self.phase_label.setText("本轮完成，请手动开始下一轮")
        else:
            self.phase_label.setText("计划轮次已全部完成")

    def _on_finished(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._set_running(False)

    def closeEvent(self, event) -> None:
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
    parser = argparse.ArgumentParser(description="SSVEP 数据采集 UI / CLI")
    parser.add_argument("--serial-port", type=str, default="auto")
    parser.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    parser.add_argument("--freqs", type=str, default="8,10,12,15")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--subject-id", type=str, default="subject001")
    parser.add_argument("--session-id", type=str, default="")
    parser.add_argument("--session-index", type=int, default=1)
    parser.add_argument("--rounds-planned", type=int, default=1)
    parser.add_argument("--round-index", type=int, default=1)
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET_NAME,
        help="stable_12m|enhanced_45m|custom",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="",
        help="已弃用参数，等价于 --preset",
    )
    parser.add_argument("--prepare-sec", type=float, default=DEFAULT_STABLE_PREPARE_SEC)
    parser.add_argument("--active-sec", type=float, default=DEFAULT_STABLE_ACTIVE_SEC)
    parser.add_argument("--rest-sec", type=float, default=DEFAULT_STABLE_REST_SEC)
    parser.add_argument("--long-idle-sec", type=float, default=DEFAULT_STABLE_LONG_IDLE_SEC)
    parser.add_argument("--target-repeats", type=int, default=DEFAULT_STABLE_TARGET_REPEATS)
    parser.add_argument("--idle-repeats", type=int, default=DEFAULT_STABLE_IDLE_REPEATS)
    parser.add_argument("--switch-trials", type=int, default=DEFAULT_STABLE_SWITCH_TRIALS)
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument("--headless", action="store_true", help="仅命令行采集，不启动 UI")
    return parser


def _resolve_cli_protocol(
    *,
    preset_name: str,
    prepare_sec: float,
    active_sec: float,
    rest_sec: float,
    long_idle_sec: float,
    target_repeats: int,
    idle_repeats: int,
    switch_trials: int,
) -> tuple[str, float, float, float, float, int, int, int]:
    preset_key = normalize_preset_name(preset_name)
    if preset_key in (STABLE_12M_PRESET.key, ENHANCED_45M_PRESET.key):
        preset = COLLECTION_PRESETS[preset_key]
        return (
            preset_key,
            float(preset.prepare_sec),
            float(preset.active_sec),
            float(preset.rest_sec),
            float(preset.long_idle_sec),
            int(preset.target_repeats),
            int(preset.idle_repeats),
            int(preset.switch_trials),
        )
    return (
        CUSTOM_PRESET.key,
        float(prepare_sec),
        float(active_sec),
        float(rest_sec),
        float(long_idle_sec),
        int(target_repeats),
        int(idle_repeats),
        int(switch_trials),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    freqs = parse_freqs(args.freqs)
    requested_preset = str(args.protocol).strip() or str(args.preset).strip()
    (
        protocol_name,
        prepare_sec,
        active_sec,
        rest_sec,
        long_idle_sec,
        target_repeats,
        idle_repeats,
        switch_trials,
    ) = _resolve_cli_protocol(
        preset_name=requested_preset,
        prepare_sec=float(args.prepare_sec),
        active_sec=float(args.active_sec),
        rest_sec=float(args.rest_sec),
        long_idle_sec=float(args.long_idle_sec),
        target_repeats=int(args.target_repeats),
        idle_repeats=int(args.idle_repeats),
        switch_trials=int(args.switch_trials),
    )
    subject_id = str(args.subject_id).strip() or "subject001"
    round_index = int(args.round_index) if int(args.round_index) > 0 else int(args.session_index)
    session_base = _strip_round_suffix(str(args.session_id).strip()) or _auto_session_base_id(subject_id)
    session_id = _build_round_session_id(session_base, round_index)
    estimated_round_sec = estimate_round_seconds(
        prepare_sec=prepare_sec,
        active_sec=active_sec,
        rest_sec=rest_sec,
        target_repeats=target_repeats,
        idle_repeats=idle_repeats,
        switch_trials=switch_trials,
        long_idle_sec=long_idle_sec,
    )
    _validate_collection_protocol(active_sec=active_sec, long_idle_sec=long_idle_sec)
    config = CollectionConfig(
        serial_port=normalize_serial_port(args.serial_port),
        board_id=int(args.board_id),
        freqs=freqs,
        subject_id=subject_id,
        session_id=session_id,
        session_index=round_index,
        dataset_dir=Path(args.dataset_dir).expanduser().resolve(),
        protocol_name=protocol_name,
        prepare_sec=prepare_sec,
        active_sec=active_sec,
        rest_sec=rest_sec,
        target_repeats=target_repeats,
        idle_repeats=idle_repeats,
        switch_trials=switch_trials,
        long_idle_sec=long_idle_sec,
        seed=int(args.seed),
        rounds_planned=max(1, int(args.rounds_planned)),
        round_index=round_index,
        estimated_round_sec=estimated_round_sec,
    )
    if bool(args.headless):
        payload = run_collection_cli(config)
        print(f"Dataset manifest: {payload.get('dataset_manifest', '')}", flush=True)
        return 0

    app = QApplication([])
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = DatasetCollectionWindow(
        serial_port=config.serial_port,
        board_id=config.board_id,
        freqs=config.freqs,
    )
    window.dataset_dir_edit.setText(str(config.dataset_dir))
    window.subject_edit.setText(config.subject_id)
    window.session_index_spin.setValue(config.session_index)
    window.rounds_planned_spin.setValue(config.rounds_planned)
    window.session_base_edit.setText(session_base)
    if config.protocol_name in COLLECTION_PRESETS:
        window.preset_combo.setCurrentText(COLLECTION_PRESETS[config.protocol_name].display)
    else:
        window.preset_combo.setCurrentText(CUSTOM_PRESET.display)
    window.prepare_spin.setValue(config.prepare_sec)
    window.active_spin.setValue(config.active_sec)
    window.rest_spin.setValue(config.rest_sec)
    window.long_idle_spin.setValue(config.long_idle_sec)
    window.target_spin.setValue(config.target_repeats)
    window.idle_spin.setValue(config.idle_repeats)
    window.switch_spin.setValue(config.switch_trials)
    window._refresh_estimate_label()
    window._refresh_round_status()
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
