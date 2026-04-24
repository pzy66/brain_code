"""Single-file launcher for realtime MI classification in PyCharm.

Before running:
1. Make sure `MI_CLASSIFIER_ROOT` points to your mi_classifier project.
2. Install realtime dependencies:
   pip install brainflow PyQt5
3. Install model dependencies in the same Python environment:
   pip install numpy scipy scikit-learn mne moabb pyriemann joblib

Then open this file in PyCharm and run it directly.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


MI_CLASSIFIER_ROOT = Path(r"C:\Users\P1233\Desktop\brain\mi_classifier")
if not MI_CLASSIFIER_ROOT.exists():
    raise FileNotFoundError(f"MI project not found: {MI_CLASSIFIER_ROOT}")
sys.path.insert(0, str(MI_CLASSIFIER_ROOT))

from src.realtime_mi import (  # noqa: E402
    DEFAULT_REALTIME_CHANNEL_NAMES,
    RealtimeMIPredictor,
    fit_realtime_model,
    load_realtime_model,
)


USER_CONFIG = {
    "serial_port": "COM3",
    "board_id": BoardIds.CYTON_BOARD.value,
    "subject_id": 1,
    "config_path": MI_CLASSIFIER_ROOT / "config.yaml",
    "model_path": Path(__file__).resolve().with_name("mi_realtime_subject1.joblib"),
    "history_len": 5,
    "confidence_threshold": 0.45,
    "step_sec": 0.25,
    "channel_names": list(DEFAULT_REALTIME_CHANNEL_NAMES),
    "board_channel_positions": None,
    "auto_train_if_missing": True,
    "force_retrain": False,
}


class EEGWorker(QObject):
    data_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()
    sampling_rate_ready = pyqtSignal(float)

    def __init__(self, config: dict, predictor: RealtimeMIPredictor) -> None:
        super().__init__()
        self.config = config
        self.predictor = predictor
        self.board = None
        self.is_running = False
        self.stop_event = threading.Event()
        self.selected_rows = None

    def start_collection(self) -> None:
        try:
            self.stop_event.clear()
            BoardShim.enable_dev_board_logger()
            try:
                BoardShim.release_all_sessions()
            except Exception:
                pass

            params = BrainFlowInputParams()
            params.serial_port = str(self.config["serial_port"])

            self.board = BoardShim(int(self.config["board_id"]), params)
            self.board.prepare_session()
            self.board.start_stream(450000)

            sampling_rate = float(BoardShim.get_sampling_rate(int(self.config["board_id"])))
            eeg_rows = BoardShim.get_eeg_channels(int(self.config["board_id"]))
            expected_channels = self.predictor.expected_channel_count

            if self.config.get("board_channel_positions") is None:
                selected_positions = list(range(expected_channels))
            else:
                selected_positions = [int(index) for index in self.config["board_channel_positions"]]

            if len(selected_positions) != expected_channels:
                raise ValueError(
                    "board_channel_positions length must match the model channel count."
                )
            if max(selected_positions) >= len(eeg_rows):
                raise ValueError(
                    f"Board only exposes {len(eeg_rows)} EEG rows, but selected positions are {selected_positions}."
                )

            self.selected_rows = [eeg_rows[index] for index in selected_positions]
            needed_samples = max(8, int(round(self.predictor.artifact["window_sec"] * sampling_rate)))

            self.is_running = True
            self.sampling_rate_ready.emit(sampling_rate)
            self.status_changed.emit(
                "Streaming started | "
                f"fs={sampling_rate:g} Hz | "
                f"board_rows={self.selected_rows} | "
                f"model_channels={self.predictor.artifact['channel_names']}"
            )

            while self.is_running and self.board.get_board_data_count() < needed_samples:
                available = self.board.get_board_data_count()
                self.status_changed.emit(f"Waiting for buffer... {available}/{needed_samples}")
                if self.stop_event.wait(0.2):
                    break

            while self.is_running and not self.stop_event.is_set():
                available = self.board.get_board_data_count()
                if available >= needed_samples:
                    data = self.board.get_current_board_data(needed_samples)
                    eeg_data = np.ascontiguousarray(
                        data[self.selected_rows, -needed_samples:],
                        dtype=np.float32,
                    )
                    self.data_ready.emit(eeg_data)
                if self.stop_event.wait(float(self.config["step_sec"])):
                    break

        except Exception as error:
            self.error_occurred.emit(f"Acquisition error: {error}")
        finally:
            self.cleanup()
            self.finished.emit()

    def request_stop(self) -> None:
        self.is_running = False
        self.stop_event.set()

    def cleanup(self) -> None:
        if self.board is not None:
            try:
                self.board.stop_stream()
            except Exception:
                pass
            try:
                self.board.release_session()
            except Exception:
                pass
            self.board = None
        self.status_changed.emit("Streaming stopped.")


class AnalysisWorker(QObject):
    result_ready = pyqtSignal(object)
    status_ready = pyqtSignal(str)

    def __init__(self, predictor: RealtimeMIPredictor) -> None:
        super().__init__()
        self.predictor = predictor
        self.live_sampling_rate = None

    def set_live_sampling_rate(self, sampling_rate: float) -> None:
        self.live_sampling_rate = float(sampling_rate)

    def process_data(self, eeg_data: np.ndarray) -> None:
        try:
            if self.live_sampling_rate is None:
                self.status_ready.emit("Sampling rate not ready yet.")
                return
            result = self.predictor.analyze_window(eeg_data, self.live_sampling_rate)
            self.result_ready.emit(result)
        except Exception as error:
            self.status_ready.emit(f"Inference error: {error}")


class MIRealtimeWindow(QMainWindow):
    def __init__(self, config: dict, predictor: RealtimeMIPredictor) -> None:
        super().__init__()
        self.config = config
        self.predictor = predictor
        self.eeg_thread = None
        self.analysis_thread = None
        self.worker = None
        self.analysis_worker = None
        self.class_labels = {}
        self._stopping = False
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle("Realtime MI Classifier")
        self.resize(1200, 780)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        header = QLabel(
            "Realtime Motor Imagery Classification | "
            f"Serial={self.config['serial_port']} | "
            f"Subject model={self.predictor.artifact['subject_id']} | "
            f"Pipeline={self.predictor.artifact['selected_pipeline']}"
        )
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 18px; font-weight: bold; padding: 12px;")
        layout.addWidget(header)

        channel_label = QLabel("Model channels: " + ", ".join(self.predictor.artifact["channel_names"]))
        channel_label.setAlignment(Qt.AlignCenter)
        channel_label.setStyleSheet("font-size: 14px; padding-bottom: 8px;")
        layout.addWidget(channel_label)

        button_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Realtime")
        self.btn_stop = QPushButton("Stop")
        self.btn_exit = QPushButton("Exit")
        self.btn_stop.setEnabled(False)
        for button in (self.btn_start, self.btn_stop, self.btn_exit):
            button.setMinimumHeight(42)
            button_row.addWidget(button)
        layout.addLayout(button_row)

        self.result_label = QLabel("WAITING")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 40px; font-weight: bold; color: white; background: #2c3e50; padding: 28px;"
        )
        layout.addWidget(self.result_label)

        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("font-size: 18px; padding: 8px;")
        layout.addWidget(self.confidence_label)

        score_grid = QGridLayout()
        for index, display_name in enumerate(self.predictor.artifact["display_class_names"]):
            card = QLabel(f"{display_name}\n--")
            card.setAlignment(Qt.AlignCenter)
            card.setMinimumHeight(110)
            card.setStyleSheet(
                "font-size: 18px; font-weight: bold; border: 2px solid #bdc3c7; "
                "background: #ecf0f1; border-radius: 10px; padding: 12px;"
            )
            score_grid.addWidget(card, index // 2, index % 2)
            self.class_labels[index] = card
        layout.addLayout(score_grid)

        bottom_row = QHBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("System logs...")
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setPlaceholderText("Realtime scores...")
        bottom_row.addWidget(self.log_text, 1)
        bottom_row.addWidget(self.detail_text, 1)
        layout.addLayout(bottom_row)

        self.btn_start.clicked.connect(self.start_realtime)
        self.btn_stop.clicked.connect(self.stop_realtime)
        self.btn_exit.clicked.connect(self.close)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_result(self, result: dict) -> None:
        stable_name = result["stable_prediction_display_name"]
        confidence = result["confidence"]
        self.result_label.setText(stable_name)
        self.confidence_label.setText(
            f"Current: {result['prediction_display_name']} | Confidence: {confidence:.3f}"
        )

        for index, display_name in enumerate(result["display_class_names"]):
            score = result["probabilities"][index]
            style = (
                "font-size: 18px; font-weight: bold; border: 2px solid #bdc3c7; "
                "background: #ecf0f1; border-radius: 10px; padding: 12px;"
            )
            if result["stable_prediction_index"] == index:
                style = (
                    "font-size: 18px; font-weight: bold; border: 2px solid #1abc9c; "
                    "background: #d1f2eb; border-radius: 10px; padding: 12px;"
                )
            elif result["prediction_index"] == index:
                style = (
                    "font-size: 18px; font-weight: bold; border: 2px solid #3498db; "
                    "background: #d6eaf8; border-radius: 10px; padding: 12px;"
                )
            self.class_labels[index].setStyleSheet(style)
            self.class_labels[index].setText(f"{display_name}\n{score:.3f}")

        detail_lines = [
            f"{display}: {score:.4f}"
            for display, score in zip(result["display_class_names"], result["probabilities"])
        ]
        detail_lines.append(f"Stable output: {stable_name}")
        self.detail_text.setText("\n".join(detail_lines))

    def set_running_state(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def start_realtime(self) -> None:
        if self.eeg_thread is not None:
            self.log("Realtime loop is already running.")
            return

        self._stopping = False
        self.eeg_thread = QThread(self)
        self.analysis_thread = QThread(self)

        self.worker = EEGWorker(self.config, self.predictor)
        self.analysis_worker = AnalysisWorker(self.predictor)

        self.worker.moveToThread(self.eeg_thread)
        self.analysis_worker.moveToThread(self.analysis_thread)

        self.eeg_thread.started.connect(self.worker.start_collection)
        self.worker.finished.connect(self.eeg_thread.quit)
        self.worker.finished.connect(self.analysis_thread.quit)
        self.worker.status_changed.connect(self.log)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.sampling_rate_ready.connect(self.analysis_worker.set_live_sampling_rate)
        self.worker.data_ready.connect(self.analysis_worker.process_data, type=Qt.QueuedConnection)
        self.analysis_worker.result_ready.connect(self.update_result)
        self.analysis_worker.status_ready.connect(self.log)
        self.eeg_thread.finished.connect(self.worker.deleteLater)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)

        self.analysis_thread.start()
        self.eeg_thread.start()
        self.set_running_state(True)
        self.log("Realtime MI classifier started.")

    def on_worker_error(self, message: str) -> None:
        self.log(message)
        if not self._stopping:
            self.stop_realtime()

    def stop_realtime(self) -> None:
        if self._stopping:
            return
        self._stopping = True

        if self.worker is not None:
            self.worker.request_stop()

        if self.eeg_thread is not None:
            self.eeg_thread.quit()
            self.eeg_thread.wait()

        if self.analysis_thread is not None:
            self.analysis_thread.quit()
            self.analysis_thread.wait()

        self.eeg_thread = None
        self.analysis_thread = None
        self.worker = None
        self.analysis_worker = None
        self.set_running_state(False)
        self.log("Realtime MI classifier stopped.")
        self._stopping = False

    def closeEvent(self, event) -> None:
        self.stop_realtime()
        event.accept()


def prepare_artifact(config: dict) -> dict:
    model_path = Path(config["model_path"])
    should_retrain = bool(config.get("force_retrain", False))

    if model_path.exists() and not should_retrain:
        artifact = load_realtime_model(model_path)
        if artifact.get("subject_id") == int(config["subject_id"]) and artifact.get("channel_names") == list(
            config["channel_names"]
        ):
            return artifact

    if not bool(config.get("auto_train_if_missing", True)) and not should_retrain:
        raise FileNotFoundError(
            f"Realtime model not found or mismatched: {model_path}. "
            "Set auto_train_if_missing=True or force_retrain=True."
        )

    artifact = fit_realtime_model(
        config_path=Path(config["config_path"]),
        subject_id=int(config["subject_id"]),
        channel_names=list(config["channel_names"]),
        output_path=model_path,
    )
    return artifact


def main() -> None:
    artifact = prepare_artifact(USER_CONFIG)
    predictor = RealtimeMIPredictor(
        artifact=artifact,
        history_len=int(USER_CONFIG["history_len"]),
        confidence_threshold=float(USER_CONFIG["confidence_threshold"]),
    )

    app = QApplication(sys.argv)
    window = MIRealtimeWindow(USER_CONFIG, predictor)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
