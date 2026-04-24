"""适合在 PyCharm 中直接运行的运动想象实时判别脚本。

这个文件只负责两件事：
1. 加载已经训练好的实时 MI 模型文件；
2. 连接 BrainFlow 设备并启动实时判别界面。

如果当前 Python 环境缺少依赖，脚本会直接给出中文错误提示，
避免出现“点运行后直接闪退”的情况。
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path


_BOOTSTRAP_ERROR: ModuleNotFoundError | None = None
_BOOTSTRAP_TRACEBACK = ""


def pyqtSignal(*_args, **_kwargs):
    return None


class _QtFallback:
    AlignCenter = 0
    QueuedConnection = 0


Qt = _QtFallback()
QObject = object
QThread = object
QApplication = object
QGridLayout = object
QHBoxLayout = object
QLabel = object
QMainWindow = object
QPushButton = object
QTextEdit = object
QVBoxLayout = object
QWidget = object
np = None
BoardIds = None
BoardShim = None
BrainFlowInputParams = None

try:
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
except ModuleNotFoundError as error:
    _BOOTSTRAP_ERROR = error
    _BOOTSTRAP_TRACEBACK = traceback.format_exc()


def show_startup_error(title: str, message: str) -> None:
    """Show a visible startup error even when PyQt5 is unavailable."""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
    except Exception:
        pass
    print(f"{title}\n{message}", file=sys.stderr)


def resolve_mi_classifier_root() -> Path:
    """Locate the mi_classifier project root for both in-project and copied launchers."""
    candidates = []

    env_root = os.environ.get("MI_CLASSIFIER_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    script_root_candidate = Path(__file__).resolve().parents[2]
    candidates.append(script_root_candidate)
    candidates.append(Path(r"C:\Users\P1233\Desktop\brain\mi_classifier"))

    for candidate in candidates:
        if (candidate / "src" / "realtime_mi.py").exists():
            return candidate

    raise FileNotFoundError(
        "找不到 mi_classifier 项目根目录。请检查 MI_CLASSIFIER_ROOT，或者把脚本放回项目目录下运行。"
    )


MI_CLASSIFIER_ROOT = resolve_mi_classifier_root()
if str(MI_CLASSIFIER_ROOT) not in sys.path:
    sys.path.insert(0, str(MI_CLASSIFIER_ROOT))

DEFAULT_BOARD_ID = BoardIds.CYTON_BOARD.value if BoardIds is not None else 0
DEFAULT_MODEL_PATH = MI_CLASSIFIER_ROOT / "models" / "realtime" / "subject_1_mi.joblib"

USER_CONFIG = {
    "serial_port": "COM3",  # 串口号，例如 COM3 / COM5
    "board_id": DEFAULT_BOARD_ID,  # BrainFlow 板卡 ID
    "model_path": DEFAULT_MODEL_PATH,  # 已训练模型路径
    "history_len": 5,  # 平滑历史长度
    "confidence_threshold": 0.45,  # 低于该阈值时输出“不确定”
    "step_sec": 0.25,  # 实时滑窗更新步长
    "board_channel_positions": None,  # 如果板子通道顺序不同，在这里写索引映射
}


def import_realtime_helpers():
    """Import project-level realtime helpers with a friendly error when base deps are missing."""
    try:
        from src.realtime_mi import RealtimeMIPredictor, load_realtime_model

        return RealtimeMIPredictor, load_realtime_model
    except ModuleNotFoundError as error:
        missing_name = getattr(error, "name", None) or str(error)
        raise ModuleNotFoundError(
            f"导入项目内部模块失败，缺少依赖：{missing_name}\n"
            "请先在 MI 环境中安装基础依赖：\n"
            "pip install -r requirements.txt"
        ) from error


class EEGWorker(QObject):
    data_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()
    sampling_rate_ready = pyqtSignal(float)

    def __init__(self, config: dict, predictor) -> None:
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
                raise ValueError("`board_channel_positions` 的长度必须和模型通道数一致。")
            if max(selected_positions) >= len(eeg_rows):
                raise ValueError(
                    f"当前板卡只提供 {len(eeg_rows)} 个 EEG 通道，但你设置的通道索引是 {selected_positions}。"
                )

            self.selected_rows = [eeg_rows[index] for index in selected_positions]
            needed_samples = max(8, int(round(self.predictor.artifact["window_sec"] * sampling_rate)))

            self.is_running = True
            self.sampling_rate_ready.emit(sampling_rate)
            self.status_changed.emit(
                "实时采集已启动 | "
                f"采样率={sampling_rate:g} Hz | "
                f"板卡通道行号={self.selected_rows} | "
                f"模型通道={self.predictor.artifact['channel_names']}"
            )

            while self.is_running and self.board.get_board_data_count() < needed_samples:
                available = self.board.get_board_data_count()
                self.status_changed.emit(f"等待缓冲区数据... {available}/{needed_samples}")
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
            self.error_occurred.emit(f"采集线程出错：{error}")
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
        self.status_changed.emit("实时采集已停止。")


class AnalysisWorker(QObject):
    result_ready = pyqtSignal(object)
    status_ready = pyqtSignal(str)

    def __init__(self, predictor) -> None:
        super().__init__()
        self.predictor = predictor
        self.live_sampling_rate = None

    def set_live_sampling_rate(self, sampling_rate: float) -> None:
        self.live_sampling_rate = float(sampling_rate)

    def process_data(self, eeg_data: np.ndarray) -> None:
        try:
            if self.live_sampling_rate is None:
                self.status_ready.emit("采样率尚未就绪，暂时无法判别。")
                return
            result = self.predictor.analyze_window(eeg_data, self.live_sampling_rate)
            self.result_ready.emit(result)
        except Exception as error:
            self.status_ready.emit(f"判别线程出错：{error}")


class MIRealtimeWindow(QMainWindow):
    def __init__(self, config: dict, predictor) -> None:
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
        self.setWindowTitle("运动想象实时判别")
        self.resize(1200, 780)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        header = QLabel(
            "运动想象实时判别系统 | "
            f"串口={self.config['serial_port']} | "
            f"模型={self.predictor.artifact['selected_pipeline']}"
        )
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 18px; font-weight: bold; padding: 12px;")
        layout.addWidget(header)

        channel_label = QLabel("模型通道顺序： " + ", ".join(self.predictor.artifact["channel_names"]))
        channel_label.setAlignment(Qt.AlignCenter)
        channel_label.setStyleSheet("font-size: 14px; padding-bottom: 8px;")
        layout.addWidget(channel_label)

        button_row = QHBoxLayout()
        self.btn_start = QPushButton("开始实时识别")
        self.btn_stop = QPushButton("停止")
        self.btn_exit = QPushButton("退出")
        self.btn_stop.setEnabled(False)
        for button in (self.btn_start, self.btn_stop, self.btn_exit):
            button.setMinimumHeight(42)
            button_row.addWidget(button)
        layout.addLayout(button_row)

        self.result_label = QLabel("等待中")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 40px; font-weight: bold; color: white; background: #2c3e50; padding: 28px;"
        )
        layout.addWidget(self.result_label)

        self.confidence_label = QLabel("置信度：--")
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
        self.log_text.setPlaceholderText("系统日志...")
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setPlaceholderText("实时分类分数...")
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
            f"当前预测：{result['prediction_display_name']} | 置信度：{confidence:.3f}"
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
        detail_lines.append(f"平滑输出：{stable_name}")
        self.detail_text.setText("\n".join(detail_lines))

    def set_running_state(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def start_realtime(self) -> None:
        if self.eeg_thread is not None:
            self.log("实时判别已经在运行。")
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
        self.log("运动想象实时判别已启动。")

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
        self.log("运动想象实时判别已停止。")
        self._stopping = False

    def closeEvent(self, event) -> None:
        self.stop_realtime()
        event.accept()


def check_startup_dependencies() -> None:
    """Fail early with clear installation instructions."""
    if _BOOTSTRAP_ERROR is None:
        return

    missing_name = getattr(_BOOTSTRAP_ERROR, "name", None) or str(_BOOTSTRAP_ERROR)
    raise ModuleNotFoundError(
        f"当前环境缺少依赖：{missing_name}\n\n"
        f"请在 MI 环境中执行：\n"
        f"cd {MI_CLASSIFIER_ROOT}\n"
        "pip install -r requirements.txt\n"
        "pip install -r requirements-realtime.txt\n\n"
        "原始报错：\n"
        f"{_BOOTSTRAP_TRACEBACK}"
    ) from _BOOTSTRAP_ERROR


def main() -> None:
    try:
        check_startup_dependencies()
        RealtimeMIPredictor, load_realtime_model = import_realtime_helpers()

        model_path = Path(USER_CONFIG["model_path"])
        if not model_path.is_absolute():
            model_path = MI_CLASSIFIER_ROOT / model_path

        if not model_path.exists():
            raise FileNotFoundError(
                "找不到实时模型文件：\n"
                f"{model_path}\n\n"
                "请先确认模型文件存在，或者重新训练实时模型。"
            )

        artifact = load_realtime_model(model_path)
        predictor = RealtimeMIPredictor(
            artifact=artifact,
            history_len=int(USER_CONFIG["history_len"]),
            confidence_threshold=float(USER_CONFIG["confidence_threshold"]),
        )

        app = QApplication(sys.argv)
        window = MIRealtimeWindow(USER_CONFIG, predictor)
        window.show()
        sys.exit(app.exec_())
    except Exception as error:
        show_startup_error("运动想象实时判别启动失败", str(error))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
