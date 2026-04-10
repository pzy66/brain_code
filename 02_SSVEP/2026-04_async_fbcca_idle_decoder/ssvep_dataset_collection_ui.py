from __future__ import annotations

import argparse
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
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
    PHASE_IDLE,
    PHASE_STOPPED,
)
from ssvep_core.dataset import (
    ENHANCED_45M_PROTOCOL,
    CollectionProtocol,
    build_collection_trials,
    save_collection_dataset_bundle,
)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = THIS_DIR / "profiles" / "datasets"


@dataclass(frozen=True)
class CollectionConfig:
    serial_port: str
    board_id: int
    freqs: tuple[float, float, float, float]
    subject_id: str
    session_id: str
    session_index: int
    dataset_dir: Path
    protocol_name: str = "enhanced_45m"
    prepare_sec: float = ENHANCED_45M_PROTOCOL.prepare_sec
    active_sec: float = ENHANCED_45M_PROTOCOL.active_sec
    rest_sec: float = ENHANCED_45M_PROTOCOL.rest_sec
    target_repeats: int = ENHANCED_45M_PROTOCOL.target_repeats
    idle_repeats: int = ENHANCED_45M_PROTOCOL.idle_repeats
    switch_trials: int = ENHANCED_45M_PROTOCOL.switch_trials
    seed: int = 20260410


def _auto_session_id(subject_id: str, session_index: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_subject = (subject_id or "subject").strip().replace(" ", "_")
    return f"{clean_subject}_session{int(session_index)}_{stamp}"


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


class CollectionWorker(QObject):
    phase_changed = pyqtSignal(object)
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
                f"Collection started: requested={self.config.serial_port} -> {resolved_port}, "
                f"attempts={attempted_ports}, fs={fs}Hz, channels={list(eeg_channels)}, ready={ready}"
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
            )
            trials = build_collection_trials(
                self.config.freqs,
                protocol=protocol,
                seed=self.config.seed,
                session_index=self.config.session_index,
            )
            active_samples = int(round(self.config.active_sec * fs))
            minimum_samples = max(1, int(round(1.5 * fs)))
            collected: list[tuple[Any, np.ndarray]] = []
            total = len(trials)
            for index, trial in enumerate(trials, start=1):
                if self._stop_event.is_set():
                    break
                cue_freq = None if trial.expected_freq is None else float(trial.expected_freq)
                prompt = (
                    f"Trial {index}/{total} idle (look center)"
                    if cue_freq is None
                    else f"Trial {index}/{total} focus {trial.label}"
                )
                self._emit_phase(PHASE_CAL_PREPARE, "Prepare", prompt, flicker=False, cue_freq=cue_freq)
                self.log.emit(prompt)
                time.sleep(max(0.0, self.config.prepare_sec))
                if self._stop_event.is_set():
                    break

                board.get_board_data()
                self._emit_phase(PHASE_CAL_ACTIVE, "Collecting", prompt, flicker=True, cue_freq=cue_freq)
                time.sleep(max(0.0, self.config.active_sec))
                segment, used_samples, available_samples = read_recent_eeg_segment(
                    board,
                    eeg_channels,
                    target_samples=active_samples,
                    minimum_samples=minimum_samples,
                )
                if used_samples < active_samples:
                    self.log.emit(
                        f"Sample shortfall at trial {index}: using {used_samples}/{active_samples} "
                        f"(buffer={available_samples})"
                    )
                collected.append((trial, segment))
                self._emit_phase(PHASE_CAL_REST, "Rest", "Rest and blink normally.", flicker=False, cue_freq=None)
                time.sleep(max(0.0, self.config.rest_sec))

            if not collected:
                raise RuntimeError("no trial was collected")
            protocol_config = {
                "protocol_name": str(self.config.protocol_name),
                "prepare_sec": float(self.config.prepare_sec),
                "active_sec": float(self.config.active_sec),
                "rest_sec": float(self.config.rest_sec),
                "target_repeats": int(self.config.target_repeats),
                "idle_repeats": int(self.config.idle_repeats),
                "switch_trials": int(self.config.switch_trials),
                "session_index": int(self.config.session_index),
                "seed": int(self.config.seed),
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
            )
            self._emit_phase(PHASE_STOPPED, "Collection finished", "Dataset saved.", flicker=False, cue_freq=None)
            self.done.emit(
                {
                    "collected_trials": len(collected),
                    "total_trials": len(trials),
                    **metadata,
                }
            )
        except Exception as exc:
            self.error.emit(f"Collection failed: {describe_runtime_error(exc, serial_port=active_serial)}")
            self._emit_phase(PHASE_ERROR, "Collection error", str(exc), flicker=False, cue_freq=None)
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
        self.setWindowTitle("SSVEP Dataset Collection UI")
        self.resize(1260, 860)

        self.default_serial = normalize_serial_port(serial_port)
        self.default_board_id = int(board_id)
        self.default_freqs = tuple(float(freq) for freq in freqs)

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[CollectionWorker] = None
        self.connect_thread: Optional[QThread] = None
        self.connect_worker: Optional[DeviceCheckWorker] = None

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
        self.session_index_spin.setRange(1, 99)
        self.session_index_spin.setValue(1)
        self.session_id_edit = QLineEdit("")
        self.dataset_dir_edit = QLineEdit(str(DEFAULT_DATASET_DIR))
        self.protocol_label = QLabel("enhanced_45m (4x24 + idle48 + switch32; 1s+4s+1s)")

        form.addRow("Serial Port", self.serial_edit)
        form.addRow("Board ID", self.board_edit)
        form.addRow("Freqs", self.freqs_edit)
        form.addRow("Subject ID", self.subject_edit)
        form.addRow("Session Index", self.session_index_spin)
        form.addRow("Session ID (optional)", self.session_id_edit)
        form.addRow("Dataset Dir", self.dataset_dir_edit)
        form.addRow("Protocol", self.protocol_label)
        left_layout.addLayout(form)

        row = QHBoxLayout()
        self.btn_pick_dir = QPushButton("选择目录")
        self.btn_connect = QPushButton("连接设备")
        self.btn_start = QPushButton("开始采集")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_pick_dir)
        row.addWidget(self.btn_connect)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        left_layout.addLayout(row)

        self.phase_label = QLabel("Idle")
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

        self.btn_pick_dir.clicked.connect(self._pick_dataset_dir)
        self.btn_connect.clicked.connect(self._connect_device)
        self.btn_start.clicked.connect(self._start_collection)
        self.btn_stop.clicked.connect(self._stop_collection)

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{stamp}] {text}")

    def _set_running(self, running: bool) -> None:
        self.btn_connect.setEnabled(not running)
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def _pick_dataset_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select dataset dir", self.dataset_dir_edit.text().strip())
        if path:
            self.dataset_dir_edit.setText(path)

    def _read_config(self) -> CollectionConfig:
        serial_port = normalize_serial_port(self.serial_edit.text().strip())
        board_id = int(self.board_edit.text().strip())
        freqs = parse_freqs(self.freqs_edit.text().strip())
        subject_id = self.subject_edit.text().strip() or "subject001"
        session_index = int(self.session_index_spin.value())
        session_id_raw = self.session_id_edit.text().strip()
        session_id = session_id_raw if session_id_raw else _auto_session_id(subject_id, session_index)
        dataset_dir = Path(self.dataset_dir_edit.text().strip()).expanduser().resolve()
        return CollectionConfig(
            serial_port=serial_port,
            board_id=board_id,
            freqs=freqs,
            subject_id=subject_id,
            session_id=session_id,
            session_index=session_index,
            dataset_dir=dataset_dir,
        )

    def _connect_device(self) -> None:
        try:
            cfg = self._read_config()
        except Exception as exc:
            self._log(f"Config error: {exc}")
            return
        worker = DeviceCheckWorker(serial_port=cfg.serial_port, board_id=cfg.board_id)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.connected.connect(self._on_connected)
        worker.error.connect(self._on_connect_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)
        self.connect_worker = worker
        self.connect_thread = thread
        self.phase_label.setText("Connecting...")
        thread.start()

    def _on_connected(self, payload: dict[str, Any]) -> None:
        self.phase_label.setText("Device connected")
        self._log(
            "Connected: requested={requested_serial_port}, resolved={resolved_serial_port}, "
            "fs={sampling_rate}Hz, ready={ready_samples}".format(**payload)
        )

    def _on_connect_error(self, text: str) -> None:
        self.phase_label.setText("Connect failed")
        self._log(text)

    def _start_collection(self) -> None:
        if self.worker_thread is not None:
            return
        try:
            cfg = self._read_config()
        except Exception as exc:
            self._log(f"Config error: {exc}")
            return
        self.session_id_edit.setText(str(cfg.session_id))
        worker = CollectionWorker(cfg)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.log.connect(self._log)
        worker.error.connect(self._on_error)
        worker.done.connect(self._on_done)
        worker.phase_changed.connect(self._on_phase_changed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_finished)
        thread.started.connect(worker.run)
        self.worker = worker
        self.worker_thread = thread
        self._set_running(True)
        self.phase_label.setText("Starting collection...")
        thread.start()

    def _stop_collection(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
        self._set_running(False)

    def _on_phase_changed(self, phase: dict[str, Any]) -> None:
        self.phase_label.setText(str(phase.get("title", "Collection")))
        self.stim.apply_phase(phase)

    def _on_error(self, text: str) -> None:
        self._log(text)

    def _on_done(self, payload: dict[str, Any]) -> None:
        self._log(
            "Collection done: collected={collected_trials}/{total_trials}, manifest={dataset_manifest}, npz={dataset_npz}".format(
                **payload
            )
        )

    def _on_finished(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._set_running(False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSVEP dataset collection UI / CLI")
    parser.add_argument("--serial-port", type=str, default="auto")
    parser.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    parser.add_argument("--freqs", type=str, default="8,10,12,15")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--subject-id", type=str, default="subject001")
    parser.add_argument("--session-id", type=str, default="")
    parser.add_argument("--session-index", type=int, default=1)
    parser.add_argument("--protocol", type=str, default="enhanced_45m")
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument("--headless", action="store_true", help="run data collection CLI only, without UI")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    freqs = parse_freqs(args.freqs)
    protocol_name = str(args.protocol).strip().lower()
    if protocol_name != "enhanced_45m":
        raise ValueError(f"unsupported protocol: {protocol_name}")
    subject_id = str(args.subject_id).strip() or "subject001"
    session_id = str(args.session_id).strip() or _auto_session_id(subject_id, int(args.session_index))
    config = CollectionConfig(
        serial_port=normalize_serial_port(args.serial_port),
        board_id=int(args.board_id),
        freqs=freqs,
        subject_id=subject_id,
        session_id=session_id,
        session_index=int(args.session_index),
        dataset_dir=Path(args.dataset_dir).expanduser().resolve(),
        seed=int(args.seed),
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
    window.session_id_edit.setText(config.session_id)
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
