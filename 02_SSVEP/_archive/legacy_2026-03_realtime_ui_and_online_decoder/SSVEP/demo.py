import argparse
import ctypes
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from PyQt5.QtCore import QObject, QPoint, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPolygon
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# =============================================================================
# 0. 用户配置区（优先改这里）
# =============================================================================
DEFAULT_CONFIG = {
    "serial_port": "COM3",                      # 串口，例如 COM3 / COM5
    "board_id": BoardIds.CYTON_BOARD.value,      # BrainFlow 板卡 ID
    "sampling_rate": 250,                        # 启动前默认值，连接成功后会被真实设备采样率覆盖
    "refresh_rate_hz": 240.0,                    # 显示器实际刷新率，必须与你的屏幕一致
    "freqs": "8,10,12,15",                     # 四个目标频率：上, 左, 下, 右
    "win_sec": 3.0,
    "step_sec": 0.5,
    "score_th": 0.02,
    "ratio_th": 1.10,
    "history_len": 5,
    "stim_mean": 0.5,
    "stim_amp": 0.5,
    "stim_phi": 0.0,
}


# =============================================================================
# 1. 计时辅助
# =============================================================================
class WindowsTimerResolution:
    """在 Windows 上请求 1ms 系统计时粒度，减小 sleep 抖动。"""

    def __init__(self):
        self._winmm = None

    def __enter__(self):
        if sys.platform.startswith("win"):
            try:
                self._winmm = ctypes.WinDLL("winmm")
                self._winmm.timeBeginPeriod(1)
            except Exception:
                self._winmm = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._winmm is not None:
            try:
                self._winmm.timeEndPeriod(1)
            except Exception:
                pass
        return False


class StimClockThread(QThread):
    """高精度刺激节拍线程：只负责给 GUI 提供当前应显示的时间戳。"""

    tick = pyqtSignal(float, int)  # t_sec, frame_idx

    def __init__(self, refresh_rate_hz: float):
        super().__init__()
        self.refresh_rate_hz = float(refresh_rate_hz)
        if self.refresh_rate_hz <= 0:
            raise ValueError("refresh_rate_hz 必须为正数。")
        self.period_ns = int(round(1_000_000_000 / self.refresh_rate_hz))
        self._stop_event = threading.Event()

    def request_stop(self):
        self._stop_event.set()

    def run(self):
        self._stop_event.clear()
        frame_idx = 0
        start_ns = time.perf_counter_ns()
        next_tick_ns = start_ns

        with WindowsTimerResolution():
            while not self._stop_event.is_set():
                now_ns = time.perf_counter_ns()
                remain_ns = next_tick_ns - now_ns

                if remain_ns > 2_000_000:
                    # 先粗略睡眠，留最后 0.5ms 给忙等，减小抖动
                    sleep_s = max((remain_ns - 500_000) / 1_000_000_000.0, 0.0)
                    if self._stop_event.wait(sleep_s):
                        break
                    continue

                while not self._stop_event.is_set():
                    if time.perf_counter_ns() >= next_tick_ns:
                        break

                if self._stop_event.is_set():
                    break

                t_sec = (next_tick_ns - start_ns) / 1_000_000_000.0
                self.tick.emit(t_sec, frame_idx)

                frame_idx += 1
                next_tick_ns = start_ns + frame_idx * self.period_ns

                # 如果 GUI / 系统调度太慢，直接追到当前帧，避免无限积压
                now_ns = time.perf_counter_ns()
                if now_ns - next_tick_ns > self.period_ns:
                    skipped = int((now_ns - next_tick_ns) // self.period_ns)
                    frame_idx += skipped
                    next_tick_ns = start_ns + frame_idx * self.period_ns


# =============================================================================
# 2. 刺激控件
# =============================================================================
class SimpleStimWidget(QWidget):
    """四目标 SSVEP 刺激控件：上 / 左 / 下 / 右。"""

    def __init__(
        self,
        freqs=None,
        refresh_rate_hz: float = 240.0,
        mean: float = 0.5,
        amp: float = 0.5,
        phi: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self.freqs = list(freqs or [8.0, 10.0, 12.0, 15.0])
        if len(self.freqs) != 4:
            raise ValueError("当前刺激界面固定为四目标，因此 freqs 必须恰好有 4 个频率。")

        self.refresh_rate_hz = float(refresh_rate_hz)
        self.mean = float(mean)
        self.amp = float(amp)
        self.phi = float(phi)

        self.running = False
        self.current_t = 0.0
        self.current_frame_idx = -1
        self.clock_thread = None

        self.directions = ["up", "left", "down", "right"]
        self.setStyleSheet("background-color: black;")
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

    def start_stimulation(self):
        if self.running:
            return

        self.running = True
        self.current_t = 0.0
        self.current_frame_idx = -1

        self.clock_thread = StimClockThread(self.refresh_rate_hz)
        self.clock_thread.tick.connect(self.on_stim_tick, type=Qt.QueuedConnection)
        self.clock_thread.start(QThread.TimeCriticalPriority)
        self.update()

    def stop_stimulation(self):
        if not self.running:
            self.update()
            return

        self.running = False
        if self.clock_thread is not None:
            self.clock_thread.request_stop()
            self.clock_thread.wait(1500)
            self.clock_thread.deleteLater()
            self.clock_thread = None

        self.current_t = 0.0
        self.current_frame_idx = -1
        self.update()

    def on_stim_tick(self, t_sec: float, frame_idx: int):
        if not self.running:
            return
        if frame_idx <= self.current_frame_idx:
            return

        self.current_t = float(t_sec)
        self.current_frame_idx = int(frame_idx)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if not self.running:
            return

        t = self.current_t
        w = self.width()
        h = self.height()
        stim_size = max(40, min(w, h) // 5)
        padding = max(20, min(w, h) // 20)
        cx, cy = w // 2, h // 2

        positions = [
            (cx - stim_size // 2, padding),
            (padding, cy - stim_size // 2),
            (cx - stim_size // 2, h - stim_size - padding),
            (w - stim_size - padding, cy - stim_size // 2),
        ]

        painter.setPen(Qt.NoPen)
        painter.setRenderHint(QPainter.Antialiasing, False)

        for i, f in enumerate(self.freqs):
            luminance = self.mean + self.amp * np.sin(2.0 * np.pi * f * t + self.phi)
            luminance = float(np.clip(luminance, 0.0, 1.0))
            gray_val = int(round(255 * luminance))
            color = QColor(gray_val, gray_val, gray_val)

            x, y = positions[i]
            painter.setBrush(QBrush(color))
            painter.drawRect(x, y, stim_size, stim_size)

            painter.setBrush(QBrush(Qt.black))
            self.draw_arrow(
                painter,
                x + stim_size // 2,
                y + stim_size // 2,
                self.directions[i],
                stim_size // 3,
            )

    @staticmethod
    def draw_arrow(painter, x, y, direction, size):
        half = size // 2
        if direction == "up":
            pts = [(x, y - size), (x - half, y + half), (x + half, y + half)]
        elif direction == "down":
            pts = [(x, y + size), (x - half, y - half), (x + half, y - half)]
        elif direction == "left":
            pts = [(x - size, y), (x + half, y - half), (x + half, y + half)]
        else:
            pts = [(x + size, y), (x - half, y - half), (x - half, y + half)]

        polygon = QPolygon()
        for px, py in pts:
            polygon.append(QPoint(int(px), int(py)))
        painter.drawPolygon(polygon)

    def closeEvent(self, event):
        self.stop_stimulation()
        super().closeEvent(event)


# =============================================================================
# 3. FBCCA 实时分析器
# =============================================================================
class RealTimeFBCCA:
    def __init__(
        self,
        sampling_rate=250,
        freqs=(8, 10, 12, 15),
        win_sec=3.0,
        step_sec=0.5,
        score_threshold=0.02,
        conf_ratio_th=1.10,
        history_len=5,
        notch_freq=50.0,
        notch_q=30.0,
    ):
        self.freqs = [float(f) for f in freqs]
        if len(self.freqs) != 4:
            raise ValueError("当前 GUI 和命令映射固定为四目标，因此 freqs 必须恰好有 4 个频率。")

        self.Nh = 3
        self.base_subbands = [(6, 50), (10, 50), (14, 50), (18, 50), (22, 50)]
        self.subbands = []
        self.subband_filters = []
        self.weights = None

        self.win_sec = float(win_sec)
        self.step_sec = float(step_sec)
        self.score_threshold = float(score_threshold)
        self.conf_ratio_th = float(conf_ratio_th)
        self.notch_freq = float(notch_freq)
        self.notch_q = float(notch_q)
        self.pred_history = deque(maxlen=max(1, int(history_len)))

        self.cmd_map = {
            self.freqs[0]: "上 (UP)",
            self.freqs[1]: "左 (LEFT)",
            self.freqs[2]: "下 (DOWN)",
            self.freqs[3]: "右 (RIGHT)",
        }

        self.fs = None
        self.win_samples = None
        self.Y_refs = None
        self._notch_b = None
        self._notch_a = None
        self._baseline_b = None
        self._baseline_a = None
        self.configure_runtime(sampling_rate)

    def configure_runtime(self, sampling_rate):
        self.fs = int(sampling_rate)
        if self.fs <= 0:
            raise ValueError("采样率必须为正数。")

        self.win_samples = int(round(self.win_sec * self.fs))
        if self.win_samples < 64:
            raise ValueError("窗口太短，至少保证几十个采样点。")

        nyq = self.fs / 2.0

        # 低通估计慢漂移基线
        self._baseline_b, self._baseline_a = butter(1, 3 / nyq, btype="low")

        # 陷波器：只有在 50Hz 明确低于 Nyquist 时才启用
        if self.notch_freq < nyq - 1e-6:
            self._notch_b, self._notch_a = iirnotch(self.notch_freq, self.notch_q, self.fs)
        else:
            self._notch_b, self._notch_a = None, None

        # 根据真实采样率裁剪合法子带，并预计算带通滤波器
        valid_subbands = []
        filters = []
        for fl, fh in self.base_subbands:
            new_fh = min(float(fh), nyq - 1e-3)
            if fl < new_fh:
                valid_subbands.append((float(fl), float(new_fh)))
                b, a = butter(2, [float(fl) / nyq, float(new_fh) / nyq], btype="band")
                filters.append((b, a))
        if not valid_subbands:
            raise ValueError(f"采样率过低，无法构造合法子带滤波器，fs={self.fs}Hz")
        self.subbands = valid_subbands
        self.subband_filters = filters

        a_w = 1.25
        b_w = 0.25
        self.weights = np.array(
            [(k + 1) ** (-a_w) + b_w for k in range(len(self.subbands))],
            dtype=float,
        )
        self.weights = self.weights / self.weights.sum()

        self.Y_refs = {
            f: self.build_ref_matrix(self.fs, self.win_samples, f, self.Nh)
            for f in self.freqs
        }
        self.pred_history.clear()

    @staticmethod
    def build_ref_matrix(fs, T, f, Nh=3):
        t = np.arange(T) / fs
        cols = []
        for h in range(1, Nh + 1):
            cols.append(np.sin(2 * np.pi * h * f * t))
            cols.append(np.cos(2 * np.pi * h * f * t))
        Y = np.stack(cols, axis=1)
        Y = Y - Y.mean(axis=0, keepdims=True)
        return Y

    def detrend_and_notch(self, x_raw_1d):
        base = filtfilt(self._baseline_b, self._baseline_a, x_raw_1d)
        x1 = x_raw_1d - base

        if self._notch_b is not None and self._notch_a is not None:
            x1 = filtfilt(self._notch_b, self._notch_a, x1)
        return x1

    def preprocess_window(self, X_raw):
        X0 = np.zeros_like(X_raw, dtype=float)
        for ci in range(X_raw.shape[1]):
            X0[:, ci] = self.detrend_and_notch(X_raw[:, ci])
        X0 = X0 - X0.mean(axis=0, keepdims=True)
        return X0

    @staticmethod
    def cca_multi_channel_svd(X, Y, reg=1e-8):
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

        denom = max(X.shape[0] - 1, 1)
        Sxx = X.T @ X / denom
        Syy = Y.T @ Y / denom
        Sxy = X.T @ Y / denom

        Sxx += reg * np.eye(Sxx.shape[0])
        Syy += reg * np.eye(Syy.shape[0])

        ex, vx = np.linalg.eigh(Sxx)
        ey, vy = np.linalg.eigh(Syy)
        ex = np.maximum(ex, reg)
        ey = np.maximum(ey, reg)

        Sxx_inv_sqrt = vx @ np.diag(1.0 / np.sqrt(ex)) @ vx.T
        Syy_inv_sqrt = vy @ np.diag(1.0 / np.sqrt(ey)) @ vy.T

        Tmat = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
        s = np.linalg.svd(Tmat, compute_uv=False)
        return float(np.max(s))

    def bandpass_filter_multichannel(self, X_in, coeffs):
        b, a = coeffs
        X_out = np.zeros_like(X_in, dtype=float)
        for ci in range(X_in.shape[1]):
            X_out[:, ci] = filtfilt(b, a, X_in[:, ci])
        X_out = X_out - X_out.mean(axis=0, keepdims=True)
        return X_out

    def classify_window(self, X_window):
        X0 = self.preprocess_window(X_window)

        X_subbands = []
        for coeffs in self.subband_filters:
            Xk = self.bandpass_filter_multichannel(X0, coeffs)
            X_subbands.append(Xk)

        scores = np.zeros(len(self.freqs), dtype=float)
        for fi, f in enumerate(self.freqs):
            Yf = self.Y_refs[f]
            score_f = 0.0
            for k, Xk in enumerate(X_subbands):
                rho_k = self.cca_multi_channel_svd(Xk, Yf)
                score_f += self.weights[k] * (rho_k ** 2)
            scores[fi] = score_f

        best_idx = int(np.argmax(scores))
        max_score = float(scores[best_idx])
        pred_f = self.freqs[best_idx]

        scores_sorted = np.sort(scores)[::-1]
        second_score = float(scores_sorted[1]) if len(scores_sorted) >= 2 else 0.0
        ratio = max_score / (second_score + 1e-12) if second_score > 0 else np.inf

        if max_score < self.score_threshold:
            decision = "none"
            pred_f = None
        elif ratio < self.conf_ratio_th:
            decision = "uncertain"
        else:
            decision = "freq"

        return {
            "decision": decision,
            "pred_f": pred_f,
            "scores": scores,
            "max_score": max_score,
            "second_score": second_score,
            "ratio": float(ratio),
        }

    def smooth_prediction(self, pred_f):
        self.pred_history.append(pred_f)
        vals = [x for x in self.pred_history if x is not None]
        if not vals:
            return None
        return Counter(vals).most_common(1)[0][0]

    def analyze_window(self, eeg_data):
        if eeg_data is None or eeg_data.size == 0:
            return {
                "text": "当前没有可分析的数据。",
                "decision": "none",
                "pred_f": None,
                "smooth_pred": None,
                "command": None,
            }

        _, n_samples = eeg_data.shape
        if n_samples < self.win_samples:
            return {
                "text": f"缓冲数据不足：{n_samples}/{self.win_samples} 点，等待完整 {self.win_sec:.1f}s 窗口...",
                "decision": "none",
                "pred_f": None,
                "smooth_pred": None,
                "command": None,
            }

        X_window = np.asarray(eeg_data[:, -self.win_samples:].T, dtype=float)
        result = self.classify_window(X_window)

        decision = result["decision"]
        pred_f = result["pred_f"]
        scores = result["scores"]
        max_score = result["max_score"]
        ratio = result["ratio"]
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        score_str = "\n".join([f"{f:.1f} Hz: {s:.4f}" for f, s in zip(self.freqs, scores)])

        if decision == "none":
            self.pred_history.append(None)
            text = (
                f"[{ts}] 输出：无\n"
                f"原因：max_score={max_score:.4f} < threshold={self.score_threshold:.4f}\n\n"
                f"{score_str}"
            )
            return {
                "text": text,
                "decision": decision,
                "pred_f": None,
                "smooth_pred": None,
                "command": None,
            }

        if decision == "uncertain":
            self.pred_history.append(None)
            text = (
                f"[{ts}] 输出：不确定\n"
                f"原始预测：{pred_f:.1f} Hz\n"
                f"原因：ratio={ratio:.3f} < {self.conf_ratio_th:.3f}\n\n"
                f"{score_str}"
            )
            return {
                "text": text,
                "decision": decision,
                "pred_f": pred_f,
                "smooth_pred": None,
                "command": None,
            }

        smooth_pred = self.smooth_prediction(pred_f)
        command = self.cmd_map.get(smooth_pred)
        text = (
            f"[{ts}] 输出：{pred_f:.1f} Hz\n"
            f"平滑后：{smooth_pred:.1f} Hz -> {command}\n"
            f"max_score={max_score:.4f}, ratio={ratio:.3f}\n\n"
            f"{score_str}"
        )
        return {
            "text": text,
            "decision": decision,
            "pred_f": pred_f,
            "smooth_pred": smooth_pred,
            "command": command,
        }


# =============================================================================
# 4. 采集线程 Worker
# =============================================================================
class EEGWorker(QObject):
    data_ready = pyqtSignal(np.ndarray)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, board_id, serial_port, analyzer):
        super().__init__()
        self.board_id = int(board_id)
        self.serial_port = serial_port
        self.analyzer = analyzer
        self.board = None
        self.is_running = False
        self.eeg_channels = None
        self.stop_event = threading.Event()

    def start_collection(self):
        try:
            self.stop_event.clear()
            BoardShim.enable_dev_board_logger()
            try:
                BoardShim.release_all_sessions()
            except Exception:
                pass

            params = BrainFlowInputParams()
            params.serial_port = self.serial_port

            self.board = BoardShim(self.board_id, params)
            self.board.prepare_session()

            actual_fs = BoardShim.get_sampling_rate(self.board_id)
            self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            self.analyzer.configure_runtime(actual_fs)

            self.board.start_stream(450000)
            self.is_running = True

            self.status_changed.emit("硬件连接成功，开始流传输...")
            self.status_changed.emit(f"真实采样率: {self.analyzer.fs} Hz")
            self.status_changed.emit(f"EEG 通道: {self.eeg_channels}")
            self.status_changed.emit(
                f"分析窗口: {self.analyzer.win_sec:.1f}s ({self.analyzer.win_samples} 点), 更新步长: {self.analyzer.step_sec:.1f}s"
            )

            last_wait_log = 0.0
            while self.is_running and self.board.get_board_data_count() < self.analyzer.win_samples:
                now = time.perf_counter()
                if now - last_wait_log >= 0.5:
                    cnt = self.board.get_board_data_count()
                    self.status_changed.emit(f"等待缓冲区数据... {cnt}/{self.analyzer.win_samples}")
                    last_wait_log = now
                if self.stop_event.wait(0.02):
                    break

            while self.is_running and not self.stop_event.is_set():
                available = self.board.get_board_data_count()
                if available >= self.analyzer.win_samples:
                    data = self.board.get_current_board_data(self.analyzer.win_samples)
                    if data.shape[1] >= self.analyzer.win_samples:
                        eeg_data = np.ascontiguousarray(
                            data[self.eeg_channels, -self.analyzer.win_samples:],
                            dtype=np.float64,
                        )
                        self.data_ready.emit(eeg_data)

                if self.stop_event.wait(self.analyzer.step_sec):
                    break

        except Exception as e:
            self.error_occurred.emit(f"采集线程错误: {e}")
        finally:
            self.cleanup()
            self.finished.emit()

    def request_stop(self):
        self.is_running = False
        self.stop_event.set()

    def cleanup(self):
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
        self.status_changed.emit("采集停止")


# =============================================================================
# 5. 分析线程 Worker
# =============================================================================
class AnalysisWorker(QObject):
    result_ready = pyqtSignal(str)
    status_ready = pyqtSignal(str)
    command_detected = pyqtSignal(str)

    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.last_command = None

    def process_data(self, eeg_data):
        try:
            result = self.analyzer.analyze_window(eeg_data)
            self.result_ready.emit(result["text"])

            cmd = result.get("command")
            if cmd is not None:
                if cmd != self.last_command:
                    self.status_ready.emit(f"识别结果更新：{cmd}")
                    self.command_detected.emit(cmd)
                self.last_command = cmd
            else:
                self.last_command = None
        except Exception as e:
            self.last_command = None
            self.status_ready.emit(f"分析线程错误: {e}")


# =============================================================================
# 6. 主界面
# =============================================================================
class StatsGUI(QMainWindow):
    def __init__(
        self,
        board_id=BoardIds.CYTON_BOARD.value,
        serial_port="COM3",
        sampling_rate=250,
        refresh_rate_hz=240.0,
        freqs=(8, 10, 12, 15),
        win_sec=3.0,
        step_sec=0.5,
        score_threshold=0.02,
        ratio_threshold=1.10,
        history_len=5,
        stim_mean=0.5,
        stim_amp=0.5,
        stim_phi=0.0,
    ):
        super().__init__()

        self.board_id = int(board_id)
        self.serial_port = serial_port
        self.refresh_rate_hz = float(refresh_rate_hz)
        self._stopping = False

        self.analyzer = RealTimeFBCCA(
            sampling_rate=sampling_rate,
            freqs=freqs,
            win_sec=win_sec,
            step_sec=step_sec,
            score_threshold=score_threshold,
            conf_ratio_th=ratio_threshold,
            history_len=history_len,
        )

        self.stim_mean = float(stim_mean)
        self.stim_amp = float(stim_amp)
        self.stim_phi = float(stim_phi)

        self.eeg_thread = None
        self.analysis_thread = None
        self.worker = None
        self.analysis_worker = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("SSVEP 刺激 + 实时识别系统（优化版）")
        self.showFullScreen()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        btn_layout = QHBoxLayout()
        self.btn_start_all = QPushButton("开启整套系统")
        self.btn_stop_all = QPushButton("停止整套系统")
        self.btn_stim_start = QPushButton("仅开启刺激")
        self.btn_stim_stop = QPushButton("仅停止刺激")
        self.btn_exit = QPushButton("退出")

        self.btn_stop_all.setEnabled(False)
        self.btn_stim_stop.setEnabled(False)

        btns = [
            self.btn_start_all,
            self.btn_stop_all,
            self.btn_stim_start,
            self.btn_stim_stop,
            self.btn_exit,
        ]
        colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800", "#607D8B"]
        for btn, col in zip(btns, colors):
            btn.setStyleSheet(
                f"background-color: {col}; color: white; font-weight: bold; padding: 10px;"
            )
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        self.top_label = QLabel(
            f"频率：上={self.analyzer.freqs[0]:g}Hz，左={self.analyzer.freqs[1]:g}Hz，下={self.analyzer.freqs[2]:g}Hz，右={self.analyzer.freqs[3]:g}Hz    |    刷新率={self.refresh_rate_hz:g}Hz    |    串口={self.serial_port}"
        )
        self.top_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: white; background: #222; padding: 8px;"
        )
        self.top_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.top_label)

        self.stim_widget = SimpleStimWidget(
            freqs=self.analyzer.freqs,
            refresh_rate_hz=self.refresh_rate_hz,
            mean=self.stim_mean,
            amp=self.stim_amp,
            phi=self.stim_phi,
        )
        layout.addWidget(self.stim_widget, 1)

        bottom_layout = QHBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setPlaceholderText("系统日志...")

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(180)
        self.result_text.setStyleSheet("font-size: 14pt; font-weight: bold; color: blue;")
        self.result_text.setPlaceholderText("等待识别结果...")

        bottom_layout.addWidget(self.log_text, 1)
        bottom_layout.addWidget(self.result_text, 1)
        layout.addLayout(bottom_layout)

        self.btn_start_all.clicked.connect(self.start_all)
        self.btn_stop_all.clicked.connect(self.stop_all)
        self.btn_stim_start.clicked.connect(self.start_stim_only)
        self.btn_stim_stop.clicked.connect(self.stop_stim_only)
        self.btn_exit.clicked.connect(self.close)

    def update_btn_state(self, running):
        self.btn_start_all.setEnabled(not running)
        self.btn_stop_all.setEnabled(running)

    def update_stim_btn_state(self, stim_running):
        self.btn_stim_start.setEnabled(not stim_running)
        self.btn_stim_stop.setEnabled(stim_running)

    def log(self, msg):
        t = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{t}] {msg}")
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def show_command(self, cmd):
        self.log(f"当前稳定识别：{cmd}")

    def start_stim_only(self):
        if self.stim_widget.running:
            return
        self.stim_widget.start_stimulation()
        self.update_stim_btn_state(True)
        self.log("刺激已开启")

    def stop_stim_only(self):
        if not self.stim_widget.running:
            self.update_stim_btn_state(False)
            return
        self.stim_widget.stop_stimulation()
        self.update_stim_btn_state(False)
        self.log("刺激已停止")

    def start_all(self):
        if self.eeg_thread is not None:
            self.log("系统已经在运行。")
            return

        self._stopping = False
        self.start_stim_only()
        self.log("正在启动采集线程和分析线程...")

        self.eeg_thread = QThread(self)
        self.analysis_thread = QThread(self)

        self.worker = EEGWorker(self.board_id, self.serial_port, self.analyzer)
        self.analysis_worker = AnalysisWorker(self.analyzer)

        self.worker.moveToThread(self.eeg_thread)
        self.analysis_worker.moveToThread(self.analysis_thread)

        self.eeg_thread.started.connect(self.worker.start_collection)
        self.worker.finished.connect(self.eeg_thread.quit)

        self.worker.status_changed.connect(self.log)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.data_ready.connect(self.analysis_worker.process_data, type=Qt.QueuedConnection)

        self.analysis_worker.result_ready.connect(self.result_text.setText)
        self.analysis_worker.status_ready.connect(self.log)
        self.analysis_worker.command_detected.connect(self.show_command)

        self.eeg_thread.finished.connect(self.worker.deleteLater)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)

        self.analysis_thread.start()
        self.eeg_thread.start()

        self.update_btn_state(True)
        self.log("系统已启动")

    def on_worker_error(self, msg):
        self.log(msg)
        if not self._stopping:
            self.stop_all()

    def stop_all(self):
        if self._stopping:
            return
        self._stopping = True
        self.log("正在停止系统...")

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

        self.stop_stim_only()
        self.update_btn_state(False)
        self.result_text.clear()
        self.log("系统已完全停止")
        self._stopping = False

    def closeEvent(self, event):
        self.stop_all()
        event.accept()


# =============================================================================
# 7. 启动入口
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="SSVEP 刺激 + 实时 FBCCA 识别系统（优化版）")
    parser.add_argument("--serial-port", type=str, default=DEFAULT_CONFIG["serial_port"], help="串口号，例如 COM3")
    parser.add_argument("--board-id", type=int, default=DEFAULT_CONFIG["board_id"])
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=DEFAULT_CONFIG["sampling_rate"],
        help="仅作为启动前默认值，运行时会被真实设备采样率覆盖",
    )
    parser.add_argument("--refresh-rate", type=float, default=DEFAULT_CONFIG["refresh_rate_hz"], help="显示器刷新率，例如 60 / 120 / 144 / 240")
    parser.add_argument("--win-sec", type=float, default=DEFAULT_CONFIG["win_sec"])
    parser.add_argument("--step-sec", type=float, default=DEFAULT_CONFIG["step_sec"])
    parser.add_argument("--score-th", type=float, default=DEFAULT_CONFIG["score_th"])
    parser.add_argument("--ratio-th", type=float, default=DEFAULT_CONFIG["ratio_th"])
    parser.add_argument("--history-len", type=int, default=DEFAULT_CONFIG["history_len"])
    parser.add_argument("--freqs", type=str, default=DEFAULT_CONFIG["freqs"])
    parser.add_argument("--stim-mean", type=float, default=DEFAULT_CONFIG["stim_mean"])
    parser.add_argument("--stim-amp", type=float, default=DEFAULT_CONFIG["stim_amp"])
    parser.add_argument("--stim-phi", type=float, default=DEFAULT_CONFIG["stim_phi"])
    return parser.parse_args()


def main():
    args = parse_args()
    freqs = [float(x) for x in args.freqs.split(",") if x.strip()]

    app = QApplication(sys.argv)
    win = StatsGUI(
        board_id=args.board_id,
        serial_port=args.serial_port,
        sampling_rate=args.sampling_rate,
        refresh_rate_hz=args.refresh_rate,
        freqs=freqs,
        win_sec=args.win_sec,
        step_sec=args.step_sec,
        score_threshold=args.score_th,
        ratio_threshold=args.ratio_th,
        history_len=args.history_len,
        stim_mean=args.stim_mean,
        stim_amp=args.stim_amp,
        stim_phi=args.stim_phi,
    )
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
