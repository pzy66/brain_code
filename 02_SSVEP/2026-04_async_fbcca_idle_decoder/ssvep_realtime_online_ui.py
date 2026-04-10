from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
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
    QVBoxLayout,
    QWidget,
)

from async_fbcca_idle_standalone import (
    AsyncDecisionGate,
    DEFAULT_BENCHMARK_MODELS,
    DEFAULT_BOARD_ID,
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
    parse_freqs,
    prepare_board_session,
    profile_is_default_fallback,
    read_recent_eeg_segment,
    resolve_selected_eeg_channels,
)
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


@dataclass(frozen=True)
class RealtimeConfig:
    serial_port: str
    board_id: int
    freqs: tuple[float, float, float, float]
    profile_path: Path
    model_name: str


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


class RealtimeWorker(QObject):
    phase_changed = pyqtSignal(object)
    log = pyqtSignal(str)
    result = pyqtSignal(object)
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
                raise RuntimeError("profile is default fallback; run training/evaluation first")
            selected_model = normalize_model_name(self.config.model_name)
            original_model = normalize_model_name(profile.model_name)
            if selected_model != original_model:
                profile = replace(profile, model_name=selected_model)
                self.log.emit(
                    f"Model override: profile={original_model} -> selected={selected_model}"
                )

            board, resolved_port, attempted_ports = prepare_board_session(self.config.board_id, self.config.serial_port)
            active_serial = resolved_port
            self.log.emit(
                f"Connected: requested={self.config.serial_port} -> {resolved_port}; attempts={attempted_ports}"
            )
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
                f"Realtime started | model={profile.model_name} | fs={fs}Hz | channels={list(eeg_channels)} | buffer={ready}"
            )
            self.phase_changed.emit(
                {
                    "mode": PHASE_VALIDATION,
                    "title": f"Realtime running ({profile.model_name})",
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
                    self.log.emit(f"Realtime transient read error {consecutive_errors}: {exc}")
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
                }
                self.result.emit(payload)
                self._stop_event.wait(max(0.01, decoder.step_sec))
            self.phase_changed.emit(
                {
                    "mode": PHASE_STOPPED,
                    "title": "Realtime stopped",
                    "detail": "You can start again.",
                    "flicker": False,
                    "cue_freq": None,
                }
            )
        except Exception as exc:
            self.error.emit(f"Realtime failed: {describe_runtime_error(exc, serial_port=active_serial)}")
            self.phase_changed.emit(
                {
                    "mode": PHASE_ERROR,
                    "title": "Realtime error",
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
        self.setWindowTitle("SSVEP Realtime Online UI")
        self.resize(1260, 860)

        self.serial_port_default = normalize_serial_port(serial_port)
        self.board_id_default = int(board_id)
        self.freqs = tuple(float(freq) for freq in freqs)

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[RealtimeWorker] = None
        self.connect_thread: Optional[QThread] = None
        self.connect_worker: Optional[DeviceCheckWorker] = None
        self._last_signature: Optional[tuple[str, Optional[float]]] = None

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
        self.profile_edit = QLineEdit(str(DEFAULT_REALTIME_PROFILE_PATH))

        form.addRow("Serial Port", self.serial_edit)
        form.addRow("Board ID", self.board_edit)
        form.addRow("Freqs", self.freqs_edit)
        form.addRow("Model", self.model_combo)
        form.addRow("Profile", self.profile_edit)
        left_layout.addLayout(form)

        row = QHBoxLayout()
        self.btn_load_profile = QPushButton("加载Profile")
        self.btn_connect = QPushButton("连接设备")
        self.btn_start = QPushButton("开始实时识别")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_load_profile)
        row.addWidget(self.btn_connect)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        left_layout.addLayout(row)

        self.phase_label = QLabel("Idle")
        self.phase_label.setStyleSheet("font-size:16px; font-weight:600;")
        left_layout.addWidget(self.phase_label)

        self.result_label = QLabel("selected_freq=None")
        self.result_label.setStyleSheet("font-size:18px; font-weight:600;")
        left_layout.addWidget(self.result_label)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text, 1)

        right = QWidget(root)
        right_layout = QVBoxLayout(right)
        self.stim = FourArrowStimWidget(self.freqs)
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
        self.btn_connect.setEnabled(not running)
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def _pick_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select profile", str(Path(self.profile_edit.text()).parent), "JSON (*.json)")
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

    def _start_realtime(self) -> None:
        if self.worker_thread is not None:
            return
        try:
            cfg = self._read_config()
        except Exception as exc:
            self._log(f"Config error: {exc}")
            return
        if not cfg.profile_path.exists():
            self._log(f"Profile not found: {cfg.profile_path}")
            return
        worker = RealtimeWorker(cfg)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.log.connect(self._log)
        worker.result.connect(self._on_result)
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
        self.phase_label.setText("Starting realtime...")
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
            "selected_freq={selected_freq} | state={state} | top1={top1:.3f} ratio={ratio:.3f}".format(
                selected_freq=payload.get("selected_freq"),
                state=payload.get("state"),
                top1=float(payload.get("top1_score", 0.0)),
                ratio=float(payload.get("ratio", 0.0)),
            )
        )
        self._log(
            "state={state} pred={pred_freq} selected={selected_freq} latency={decision_latency_ms:.3f}ms".format(
                state=payload.get("state"),
                pred_freq=payload.get("pred_freq"),
                selected_freq=payload.get("selected_freq"),
                decision_latency_ms=float(payload.get("decision_latency_ms", 0.0)),
            )
        )

    def _on_error(self, text: str) -> None:
        self._log(text)

    def _on_phase_changed(self, phase: dict[str, Any]) -> None:
        title = str(phase.get("title", ""))
        self.phase_label.setText(title or "Realtime")
        self.stim.apply_phase(phase)

    def _on_finished(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._set_running(False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSVEP realtime online UI / CLI")
    parser.add_argument("--serial-port", type=str, default="auto")
    parser.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    parser.add_argument("--freqs", type=str, default="8,10,12,15")
    parser.add_argument("--profile", type=Path, default=DEFAULT_REALTIME_PROFILE_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--emit-all", action="store_true")
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="run realtime CLI only, without UI")
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
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
