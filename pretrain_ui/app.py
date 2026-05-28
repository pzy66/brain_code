from __future__ import annotations

import threading
import os
import time
from datetime import datetime
from typing import Any, Optional, Sequence

from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hybrid_controller.adapters.control_sim_slots import ControlSimSlotCatalog
from hybrid_controller.adapters.robot_client import RobotClient, fetch_robot_status
from hybrid_controller.config import AppConfig
from hybrid_controller.cylindrical import clamp, cylindrical_to_cartesian, normalize_theta_deg
from unified_collection.mi_bridge import (
    BoardCaptureWorker,
    BoardIds,
    DEFAULT_CHANNEL_NAMES,
    RealtimeEEGPreviewWidget,
    available_board_options,
    detect_serial_ports,
)


PRETRAIN_STEPS: tuple[dict[str, object], ...] = (
    {"title": "SSVEP 数据采集", "detail": "按固定频率呈现视觉刺激并采集响应。", "duration": 32},
    {"title": "SSVEP 数据整理", "detail": "自动完成标记对齐与有效窗口整理。", "duration": 14},
    {"title": "MI 数据采集", "detail": "按预设类别完成运动想象采集。", "duration": 36},
    {"title": "MI 数据整理", "detail": "自动完成试次质检与样本平衡。", "duration": 14},
    {"title": "特征构建", "detail": "生成 SSVEP 与 MI 的预训练特征。", "duration": 18},
    {"title": "模型预训练", "detail": "展示训练轮次与阶段进度。", "duration": 42},
    {"title": "配置生成", "detail": "生成可用于后续实时控制的配置。", "duration": 12},
)

CONTROL_SIM_STEPS: tuple[dict[str, object], ...] = (
    {
        "state_index": 1,
        "top_status": "SSVEP 选择流程：视觉刺激中",
        "target": "A",
        "camera_status": "SSVEP 注视目标 A",
        "ssvep_index": 0,
        "mi_index": -1,
        "pose": ("X 128 · Y -42 · Z 86 mm", "Roll 0.0 · Pitch 12.4 · Yaw -4.8", "待命 · 张开", "限位正常 · 选择中"),
    },
    {
        "state_index": 1,
        "top_status": "SSVEP 选择流程：目标 A 已锁定",
        "target": "A",
        "camera_status": "目标 A 已锁定",
        "ssvep_index": 1,
        "mi_index": -1,
        "pose": ("X 128 · Y -42 · Z 86 mm", "Roll 0.0 · Pitch 12.4 · Yaw -4.8", "待命 · 张开", "限位正常 · 已锁定"),
    },
    {
        "state_index": 2,
        "top_status": "MI 移动流程：向右微调",
        "target": "A",
        "camera_status": "MI 右移微调",
        "ssvep_index": 2,
        "mi_index": 0,
        "pose": ("X 136 · Y -38 · Z 86 mm", "Roll 0.0 · Pitch 12.1 · Yaw -3.6", "待命 · 张开", "限位正常 · 移动中"),
    },
    {
        "state_index": 2,
        "top_status": "MI 移动流程：连续靠近目标",
        "target": "A",
        "camera_status": "末端靠近目标 A",
        "ssvep_index": 2,
        "mi_index": 1,
        "pose": ("X 151 · Y -31 · Z 80 mm", "Roll 0.0 · Pitch 11.8 · Yaw -2.4", "待命 · 张开", "限位正常 · 移动中"),
    },
    {
        "state_index": 3,
        "top_status": "机械臂执行：准备抓取",
        "target": "A",
        "camera_status": "末端对准目标 A",
        "ssvep_index": 2,
        "mi_index": 2,
        "pose": ("X 164 · Y -25 · Z 62 mm", "Roll 0.0 · Pitch 10.9 · Yaw -1.2", "准备抓取 · 张开", "限位正常 · 执行中"),
    },
    {
        "state_index": 3,
        "top_status": "机械臂执行：夹爪闭合",
        "target": "A",
        "camera_status": "抓取动作执行",
        "ssvep_index": 2,
        "mi_index": 2,
        "pose": ("X 164 · Y -25 · Z 58 mm", "Roll 0.0 · Pitch 10.9 · Yaw -1.2", "抓取完成 · 闭合", "限位正常 · 已执行"),
    },
    {
        "state_index": 4,
        "top_status": "等待下一轮：动作完成",
        "target": "",
        "camera_status": "等待下一轮选择",
        "ssvep_index": -1,
        "mi_index": -1,
        "pose": ("X 142 · Y -36 · Z 92 mm", "Roll 0.0 · Pitch 12.0 · Yaw -3.0", "待命 · 张开", "限位正常 · 待命"),
    },
)

CONTROL_STATE_NAMES = ("预训练完成", "摄像头读取", "WASD 移动", "数字选块", "机械臂执行")
BLOCK_SLOT_LABELS = ("小木块 1", "小木块 2", "小木块 3", "小木块 4")
MOVE_FLOW_STEPS = (
    ("等待按键", "W/A/S/D 控制末端前后左右微调"),
    ("发送移动", "按住按键持续发送机械臂移动指令"),
    ("姿态反馈", "刷新当前半径、角度和执行状态"),
)
BLOCK_FLOW_STEPS = (
    ("数字选择", "按 1 / 2 / 3 / 4 选择目标木块"),
    ("目标锁定", "高亮摄像头画面中的候选木块"),
    ("执行确认", "可继续移动机械臂或确认抓取"),
)


class _NullPreviewWidget(QWidget):
    def configure_stream(self, *, sampling_rate: float, channel_names: Sequence[str]) -> None:
        _ = sampling_rate, channel_names

    def append_chunk(self, payload: object) -> None:
        _ = payload


class CameraCaptureThread(QThread):
    frame_received = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, source: str, *, parent: QObject | None = None, target_fps: float = 24.0) -> None:
        super().__init__(parent)
        self.source = str(source)
        self.target_fps = max(1.0, float(target_fps))
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self._stop_event.clear()
        try:
            import cv2
        except Exception as error:
            self.status_changed.emit(f"摄像头不可用：未能加载 OpenCV ({error})")
            return

        source: object = self.source
        if self.source.strip().isdigit():
            source = int(self.source.strip())
        capture = None
        try:
            self.status_changed.emit(f"正在打开摄像头流：{self.source}")
            capture = cv2.VideoCapture()
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1500)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 800)
            capture.open(source)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if capture is None or not capture.isOpened():
                self.status_changed.emit("摄像头未连接，当前使用离线画面")
                return

            self.status_changed.emit("摄像头已连接")
            frame_interval = 1.0 / self.target_fps
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if ok and frame is not None:
                    self.frame_received.emit(frame)
                else:
                    self.status_changed.emit("摄像头读取中断，正在等待下一帧")
                    if self._stop_event.wait(0.25):
                        break
                if self._stop_event.wait(frame_interval):
                    break
        except Exception as error:
            self.status_changed.emit(f"摄像头读取异常：{error}")
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            self.status_changed.emit("摄像头已停止")


class RobotCameraPreviewWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(720, 460)
        self.setObjectName("robotCamera")
        self.active_target = ""
        self.selected_slot_id: int | None = None
        self.status_text = "等待接入真实相机流"
        self.motion_phase = 0
        self._frame_pixmap: QPixmap | None = None
        self._frame_size = (0, 0)

    def set_simulation_state(self, *, target: str, status: str, phase: int) -> None:
        self.active_target = str(target)
        self.status_text = str(status)
        self.motion_phase = int(phase)
        self.update()

    def set_camera_status(self, status: str) -> None:
        self.status_text = str(status)
        self.update()

    def set_selected_slot(self, slot_id: int | None) -> None:
        self.selected_slot_id = slot_id
        self.active_target = "" if slot_id is None else str(slot_id)
        self.update()

    def set_frame_bgr(self, frame_bgr: object) -> None:
        try:
            frame = frame_bgr
            height, width = frame.shape[:2]
            if len(frame.shape) == 2:
                image = QImage(frame.data, int(width), int(height), int(width), QImage.Format_Grayscale8).copy()
            else:
                rgb = frame[:, :, ::-1].copy()
                bytes_per_line = int(width) * 3
                image = QImage(rgb.data, int(width), int(height), bytes_per_line, QImage.Format_RGB888).copy()
        except Exception:
            return
        self._frame_pixmap = QPixmap.fromImage(image)
        self._frame_size = (int(width), int(height))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(8, 8, -8, -8)
        frame_rect = self._draw_frame_or_placeholder(painter, rect)
        self._draw_block_overlay(painter, frame_rect)

        painter.setPen(QColor("#DCE8F4"))
        painter.setFont(QFont("Microsoft YaHei UI", 12, QFont.Bold))
        painter.drawText(rect.adjusted(18, 16, -18, -16), Qt.AlignTop | Qt.AlignLeft, "机械臂摄像头画面")
        painter.setFont(QFont("Microsoft YaHei UI", 9))
        painter.setPen(QColor("#AAB7C5"))
        painter.drawText(rect.adjusted(18, 42, -18, -16), Qt.AlignTop | Qt.AlignLeft, self.status_text)

    def _draw_frame_or_placeholder(self, painter: QPainter, rect) -> object:
        if self._frame_pixmap is not None and self._frame_size[0] > 0 and self._frame_size[1] > 0:
            frame_width, frame_height = self._frame_size
            scale = min(rect.width() / float(frame_width), rect.height() / float(frame_height))
            draw_width = max(1, int(frame_width * scale))
            draw_height = max(1, int(frame_height * scale))
            left = rect.left() + (rect.width() - draw_width) // 2
            top = rect.top() + (rect.height() - draw_height) // 2
            frame_rect = rect.__class__(left, top, draw_width, draw_height)
            painter.setPen(QPen(QColor("#2F4058"), 1))
            painter.setBrush(QColor("#0B1724"))
            painter.drawRoundedRect(rect, 12, 12)
            painter.drawPixmap(frame_rect, self._frame_pixmap)
            return frame_rect

        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QColor("#0B1724"))
        gradient.setColorAt(1.0, QColor("#182235"))
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor("#2F4058"), 1))
        painter.drawRoundedRect(rect, 12, 12)

        painter.setPen(QPen(QColor(120, 165, 210, 55), 1))
        for index in range(1, 6):
            x = rect.left() + int(rect.width() * index / 6)
            painter.drawLine(x, rect.top() + 18, x, rect.bottom() - 18)
        for index in range(1, 4):
            y = rect.top() + int(rect.height() * index / 4)
            painter.drawLine(rect.left() + 18, y, rect.right() - 18, y)

        painter.setPen(QPen(QColor("#82E6C4"), 2))
        center_x = rect.center().x()
        center_y = rect.center().y()
        painter.drawLine(center_x - 24, center_y, center_x + 24, center_y)
        painter.drawLine(center_x, center_y - 24, center_x, center_y + 24)
        painter.drawEllipse(center_x - 36, center_y - 36, 72, 72)
        return rect

    def _draw_block_overlay(self, painter: QPainter, rect) -> None:
        center_x = rect.center().x()
        center_y = rect.center().y()
        block_specs = [
            (0.23, 0.36, "#F6C667", "1"),
            (0.50, 0.34, "#56D6A6", "2"),
            (0.32, 0.68, "#6CA8FF", "3"),
            (0.68, 0.62, "#E879A6", "4"),
        ]
        for x_ratio, y_ratio, color, label in block_specs:
            x = rect.left() + int(rect.width() * x_ratio)
            y = rect.top() + int(rect.height() * y_ratio)
            is_active = label == self.active_target or str(self.selected_slot_id or "") == label
            painter.setBrush(QColor(color))
            painter.setPen(QPen(QColor("#FFFFFF" if is_active else "#F8FBFF"), 4 if is_active else 2))
            painter.drawRoundedRect(x - 34, y - 22, 68, 44, 8, 8)
            if is_active:
                painter.setPen(QPen(QColor("#82E6C4"), 2))
                pulse = 10 + (self.motion_phase % 4) * 4
                painter.drawEllipse(x - 34 - pulse, y - 22 - pulse, 68 + pulse * 2, 44 + pulse * 2)
            painter.setPen(QColor("#08111B"))
            painter.setFont(QFont("Microsoft YaHei UI", 12, QFont.Bold))
            painter.drawText(x - 34, y - 22, 68, 44, Qt.AlignCenter, label)

        if self.active_target:
            target_lookup = {label: (x_ratio, y_ratio) for x_ratio, y_ratio, _color, label in block_specs}
            x_ratio, y_ratio = target_lookup.get(self.active_target, (0.5, 0.5))
            target_x = rect.left() + int(rect.width() * x_ratio)
            target_y = rect.top() + int(rect.height() * y_ratio)
            painter.setPen(QPen(QColor("#82E6C4"), 2))
            painter.drawLine(center_x, center_y, target_x, target_y)


class PretrainWindow(QMainWindow):
    capture_stop_requested = pyqtSignal()
    camera_frame_received = pyqtSignal(object)
    camera_status_received = pyqtSignal(str)
    robot_connection_result = pyqtSignal(bool, object)
    robot_status_snapshot_received = pyqtSignal(object)
    robot_log_received = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("脑机接口预训练")
        self.resize(1520, 900)
        self.setMinimumSize(1200, 760)
        self.setFocusPolicy(Qt.StrongFocus)

        self.capture_thread: QThread | None = None
        self.capture_worker: BoardCaptureWorker | None = None
        self.device_info: dict[str, Any] | None = None
        self.robot_config = AppConfig(input_profile="operator_keyboard", move_source="sim", decision_source="sim").resolved()
        self.slot_catalog = ControlSimSlotCatalog(self.robot_config)
        self.robot_client: RobotClient | None = None
        self.robot_connected = False
        self.robot_connecting = False
        self.robot_io_lock = threading.Lock()
        self.robot_status_poll_in_flight = False
        self.last_robot_status_poll_ts = 0.0
        self.robot_pose_cyl = list(self.robot_config.robot_start_cyl)
        self.pressed_move_keys: set[str] = set()
        self.last_move_command_ts = 0.0
        self.selected_block_id: int | None = None
        self.camera_thread: CameraCaptureThread | None = None
        self.camera_source = self.robot_config.resolve_vision_stream_url()
        self.pretrain_timer = QTimer(self)
        self.pretrain_timer.setInterval(120)
        self.control_timer = QTimer(self)
        self.control_timer.setInterval(120)
        self.step_index = 0
        self.step_ticks = 0
        self.control_tick = 0
        self.completed = False
        self.control_screen_ready = False
        self.step_rows: list[dict[str, Any]] = []
        self.control_state_nodes: list[QFrame] = []
        self.control_state_labels: list[QLabel] = []
        self.ssvep_flow_rows: list[QFrame] = []
        self.mi_flow_rows: list[QFrame] = []
        self.pose_value_labels: dict[str, QLabel] = {}
        self.camera_widget: RobotCameraPreviewWidget | None = None
        self.control_status_label: QLabel | None = None
        self.robot_status_label: QLabel | None = None
        self.camera_status_label: QLabel | None = None
        self.command_status_label: QLabel | None = None
        self.selected_block_label: QLabel | None = None
        self.block_buttons: dict[int, QPushButton] = {}
        self.btn_robot_connect: QPushButton | None = None
        self.btn_robot_reset: QPushButton | None = None
        self.btn_robot_abort: QPushButton | None = None
        self.btn_pick_selected: QPushButton | None = None
        self.btn_camera_restart: QPushButton | None = None

        self._init_ui()
        self.camera_frame_received.connect(self._on_camera_frame_received)
        self.camera_status_received.connect(self._on_camera_status_received)
        self.robot_connection_result.connect(self._on_robot_connection_result)
        self.robot_status_snapshot_received.connect(self._on_robot_status_snapshot_received)
        self.robot_log_received.connect(self._on_robot_log_received)
        self.refresh_serial_ports()

    def _init_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("root")
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(18)

        left = QWidget(root)
        left.setObjectName("leftPanel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)

        device_group = QGroupBox("脑电设备")
        device_form = QGridLayout(device_group)
        device_form.setContentsMargins(14, 18, 14, 10)
        device_form.setHorizontalSpacing(12)
        device_form.setVerticalSpacing(8)
        self.board_combo = QComboBox()
        for label, board_id in available_board_options():
            self.board_combo.addItem(f"{label} ({board_id})", int(board_id))
        self.serial_combo = QComboBox()
        self.serial_combo.setEditable(True)
        self.btn_refresh_ports = QPushButton("刷新端口")
        self.btn_connect = QPushButton("连接脑电设备")
        self.btn_connect.setProperty("controlType", "primary")
        self.btn_disconnect = QPushButton("断开连接")
        self.btn_disconnect.setProperty("controlType", "danger")
        self.btn_disconnect.setEnabled(False)
        self.imp_channel_spin = QSpinBox()
        self.imp_channel_spin.setRange(1, 16)
        self.btn_eeg_mode = QPushButton("EEG 预览")
        self.btn_imp_mode = QPushButton("阻抗检查")

        device_form.addWidget(QLabel("采集板"), 0, 0)
        device_form.addWidget(self.board_combo, 0, 1, 1, 5)
        device_form.addWidget(QLabel("串口"), 1, 0)
        device_form.addWidget(self.serial_combo, 1, 1, 1, 5)
        device_form.addWidget(self.btn_refresh_ports, 2, 0, 1, 2)
        device_form.addWidget(self.btn_connect, 2, 2, 1, 2)
        device_form.addWidget(self.btn_disconnect, 2, 4, 1, 2)
        device_form.addWidget(QLabel("信号检查"), 3, 0)
        device_form.addWidget(QLabel("通道"), 3, 1)
        device_form.addWidget(self.imp_channel_spin, 3, 2)
        device_form.addWidget(self.btn_eeg_mode, 3, 3)
        device_form.addWidget(self.btn_imp_mode, 3, 4, 1, 2)
        left_layout.addWidget(device_group)

        self.hero = QFrame()
        self.hero.setObjectName("hero")
        hero_layout = QHBoxLayout(self.hero)
        hero_layout.setContentsMargins(18, 14, 18, 14)
        hero_copy = QVBoxLayout()
        hero_title = QLabel("脑机接口预训练")
        hero_title.setObjectName("heroTitle")
        hero_subtitle = QLabel("连接脑电设备后，一键进入预设好的 SSVEP 与 MI 预训练流程。")
        hero_subtitle.setObjectName("mutedLabel")
        hero_subtitle.setWordWrap(True)
        hero_copy.addWidget(hero_title)
        hero_copy.addWidget(hero_subtitle)
        hero_layout.addLayout(hero_copy, 1)

        hero_controls = QVBoxLayout()
        self.status_badge = QLabel("等待连接")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setAlignment(Qt.AlignCenter)
        hero_controls.addWidget(self.status_badge, 0, Qt.AlignRight)
        button_row = QHBoxLayout()
        self.btn_start = QPushButton("开始预训练")
        self.btn_start.setProperty("controlType", "primary")
        self.btn_pause = QPushButton("暂停")
        self.btn_reset = QPushButton("重置")
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(False)
        button_row.addWidget(self.btn_start)
        button_row.addWidget(self.btn_pause)
        button_row.addWidget(self.btn_reset)
        hero_controls.addLayout(button_row)
        hero_layout.addLayout(hero_controls)
        left_layout.addWidget(self.hero)

        content = QGridLayout()
        content.setHorizontalSpacing(14)
        content.setVerticalSpacing(14)
        left_layout.addLayout(content, 1)

        device_card, device_layout = self._card("设备状态")
        self.device_card = device_card
        self.device_state_label = QLabel("请先连接脑电设备")
        self.device_state_label.setObjectName("deviceStateTitle")
        self.device_detail_label = QLabel("连接成功后，系统会自动解锁预训练流程。")
        self.device_detail_label.setObjectName("mutedLabel")
        self.device_detail_label.setWordWrap(True)
        self.device_signal_label = QLabel("实时 EEG 与阻抗预览会显示在右侧。")
        self.device_signal_label.setObjectName("mutedLabel")
        self.device_signal_label.setWordWrap(True)
        device_layout.addWidget(self.device_state_label)
        device_layout.addWidget(self.device_detail_label)
        device_layout.addWidget(self.device_signal_label)
        preset_label = QLabel("预设方案：SSVEP 8/10/12/15 Hz · MI 左手/右手/双脚/舌头 · 12 epochs")
        preset_label.setObjectName("summaryValue")
        preset_label.setWordWrap(True)
        device_layout.addWidget(preset_label)
        content.addWidget(device_card, 0, 0)

        progress_card, progress_layout = self._card("运行进度")
        self.active_stage_label = QLabel("等待设备连接")
        self.active_stage_label.setObjectName("stageTitle")
        self.active_detail_label = QLabel("确认脑电设备连接完成后，点击开始预训练即可。")
        self.active_detail_label.setObjectName("mutedLabel")
        self.active_detail_label.setWordWrap(True)
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.stage_progress = QProgressBar()
        self.stage_progress.setRange(0, 100)
        self.progress_caption = QLabel("总进度 0% | 当前阶段 0%")
        self.progress_caption.setObjectName("mutedLabel")
        progress_layout.addWidget(self.active_stage_label)
        progress_layout.addWidget(self.active_detail_label)
        progress_layout.addWidget(self.overall_progress)
        progress_layout.addWidget(self.stage_progress)
        progress_layout.addWidget(self.progress_caption)
        self.metric_labels = {
            "ssvep": QLabel("0/24"),
            "mi": QLabel("0/120"),
            "epoch": QLabel("0/12"),
            "ready": QLabel("0%"),
        }
        self.metric_summary_label = QLabel("SSVEP 0/24 · MI 0/120 · 训练 0/12 · 准备度 0%")
        self.metric_summary_label.setObjectName("summaryValue")
        self.metric_summary_label.setWordWrap(True)
        self.metric_summary_label.hide()
        content.addWidget(progress_card, 0, 1)

        steps_card, steps_layout = self._card("预训练流程")
        for step_number, step in enumerate(PRETRAIN_STEPS, start=1):
            row = QFrame()
            row.setObjectName("stepRow")
            row.setProperty("stepState", "pending")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 8, 10, 8)
            row_layout.setSpacing(10)
            index_label = QLabel(str(step_number))
            index_label.setObjectName("stepIndex")
            index_label.setAlignment(Qt.AlignCenter)
            index_label.setMinimumSize(24, 24)
            title_label = QLabel(str(step["title"]))
            title_label.setObjectName("stepTitle")
            detail_label = QLabel(str(step["detail"]))
            detail_label.setObjectName("stepDetail")
            detail_label.setWordWrap(True)
            text_layout = QVBoxLayout()
            text_layout.addWidget(title_label)
            detail_label.setVisible(False)
            state_label = QLabel("等待")
            state_label.setObjectName("stepState")
            state_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row_layout.addWidget(index_label, 0, Qt.AlignTop)
            row_layout.addLayout(text_layout, 1)
            row_layout.addWidget(state_label)
            self.step_rows.append({"frame": row, "index": index_label, "title": title_label, "detail": detail_label, "state": state_label})
            steps_layout.addWidget(row)
        steps_layout.addStretch(1)
        content.addWidget(steps_card, 0, 2, 2, 1)

        log_card, log_layout = self._card("流程提示")
        self.pretrain_log = QPlainTextEdit()
        self.pretrain_log.setObjectName("logPanel")
        self.pretrain_log.setReadOnly(True)
        log_layout.addWidget(self.pretrain_log)
        content.addWidget(log_card, 1, 0, 1, 2)
        content.setColumnStretch(0, 1)
        content.setColumnStretch(1, 1)
        content.setColumnStretch(2, 1)

        right = QWidget(root)
        right.setObjectName("rightPanel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        self.top_status_label = QLabel("等待连接")
        self.top_status_label.setObjectName("topStatus")
        self.preview_widget: QObject | QWidget
        try:
            self.preview_widget = RealtimeEEGPreviewWidget()
        except Exception:
            self.preview_widget = _NullPreviewWidget()
        self.log_text = QPlainTextEdit()
        self.log_text.setObjectName("logPanel")
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.top_status_label)
        right_layout.addWidget(self.preview_widget, 2)  # type: ignore[arg-type]
        right_layout.addWidget(self.log_text, 1)

        root_layout.addWidget(left, 4)
        root_layout.addWidget(right, 1)

        self.btn_refresh_ports.clicked.connect(self.refresh_serial_ports)
        self.btn_connect.clicked.connect(self.connect_device)
        self.btn_disconnect.clicked.connect(self.disconnect_device)
        self.btn_start.clicked.connect(self.start_pretrain)
        self.btn_pause.clicked.connect(self.pause_pretrain)
        self.btn_reset.clicked.connect(self.reset_pretrain)
        self.pretrain_timer.timeout.connect(self._advance)
        self.control_timer.timeout.connect(self._advance_control_simulation)
        self.capture_stop_requested.connect(self._noop)
        self.setStyleSheet(self._stylesheet())
        self._reset_state(write_log=False)

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        layout.addWidget(title_label)
        return card, layout

    def _summary_tile(self, grid: QGridLayout, row: int, column: int, title: str, value: str) -> None:
        tile = QFrame()
        tile.setObjectName("tile")
        layout = QVBoxLayout(tile)
        layout.setContentsMargins(10, 8, 10, 8)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("summaryValue")
        value_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        grid.addWidget(tile, row, column)

    def _metric(self, grid: QGridLayout, row: int, column: int, title: str, key: str) -> None:
        tile = QFrame()
        tile.setObjectName("tile")
        layout = QVBoxLayout(tile)
        layout.setContentsMargins(10, 8, 10, 8)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label = QLabel("-")
        value_label.setObjectName("metricValue")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        self.metric_labels[key] = value_label
        grid.addWidget(tile, row, column)

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
        self._log(f"Detected serial ports: {ports or 'none'}")

    def connect_device(self) -> None:
        if self.capture_thread is not None:
            return
        self.capture_thread = QThread(self)
        self.capture_worker = BoardCaptureWorker(
            board_id=self.current_board_id(),
            serial_port=self.serial_combo.currentText().strip(),
            channel_positions=tuple(range(len(DEFAULT_CHANNEL_NAMES))),
            channel_names=DEFAULT_CHANNEL_NAMES,
        )
        self.capture_worker.moveToThread(self.capture_thread)
        self.capture_thread.started.connect(self.capture_worker.run)
        self.capture_worker.connection_ready.connect(self._on_device_ready)
        self.capture_worker.preview_data_ready.connect(self.preview_widget.append_chunk)  # type: ignore[attr-defined]
        self.capture_worker.status_changed.connect(self._log)
        self.capture_worker.error_occurred.connect(self._on_device_error)
        self.capture_worker.finished.connect(self.capture_thread.quit)
        self.capture_worker.finished.connect(self.capture_worker.deleteLater)
        self.capture_thread.finished.connect(self._on_device_thread_finished)
        self.capture_thread.finished.connect(self.capture_thread.deleteLater)
        self.capture_stop_requested.connect(self.capture_worker.request_stop, Qt.DirectConnection)
        self.top_status_label.setText("正在连接脑电设备...")
        self.btn_connect.setEnabled(False)
        self.capture_thread.start()

    def _on_device_ready(self, payload: object) -> None:
        self.device_info = dict(payload or {})
        self._apply_device_ready()

    def apply_demo_device_ready(self, sampling_rate: float = 250.0, channels: Sequence[str] = DEFAULT_CHANNEL_NAMES) -> None:
        self.device_info = {"sampling_rate": float(sampling_rate), "channel_names": list(channels)}
        self.capture_worker = object()  # type: ignore[assignment]
        self._apply_device_ready()

    def _apply_device_ready(self) -> None:
        info = dict(self.device_info or {})
        fs = float(info.get("sampling_rate", 0.0) or 0.0)
        channels = [str(item) for item in info.get("channel_names", [])]
        self.preview_widget.configure_stream(sampling_rate=fs, channel_names=channels)  # type: ignore[attr-defined]
        self.top_status_label.setText(f"脑电设备已连接 | {fs:g} Hz")
        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(True)
        self.device_card.setProperty("deviceState", "ready")
        self.device_state_label.setProperty("deviceState", "ready")
        self.status_badge.setProperty("deviceState", "ready")
        self.device_state_label.setText("脑电设备已连接")
        self.device_detail_label.setText(f"采样率 {fs:g} Hz，已识别 {len(channels)} 个 EEG 通道。")
        self.device_signal_label.setText("可以开始预训练；右侧可继续观察 EEG 与阻抗预览。")
        if not self.pretrain_timer.isActive() and not self.completed:
            self.status_badge.setText("可以开始")
            self.active_stage_label.setText("准备开始")
            self.active_detail_label.setText("设备已连接，点击开始预训练即可进入固定流程。")
            self.btn_start.setEnabled(True)
        self._refresh_polish(self.device_card, self.device_state_label, self.status_badge)

    def _on_device_error(self, text: str) -> None:
        self._log(str(text))
        self.top_status_label.setText("设备连接异常")
        self.device_info = None
        self._refresh_device_waiting()

    def _on_device_thread_finished(self) -> None:
        self.capture_worker = None
        self.capture_thread = None
        self.device_info = None
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self._refresh_device_waiting()

    def disconnect_device(self) -> None:
        if self.capture_worker is not None and hasattr(self.capture_worker, "request_stop"):
            self.top_status_label.setText("正在断开脑电设备...")
            self.capture_stop_requested.emit()
        else:
            self._on_device_thread_finished()

    def _refresh_device_waiting(self) -> None:
        self.device_card.setProperty("deviceState", "waiting")
        self.device_state_label.setProperty("deviceState", "waiting")
        self.status_badge.setProperty("deviceState", "waiting")
        self.device_state_label.setText("请先连接脑电设备")
        self.device_detail_label.setText("连接成功后，系统会自动解锁预训练流程。")
        self.device_signal_label.setText("实时 EEG 与阻抗预览会显示在右侧。")
        if not self.pretrain_timer.isActive() and not self.completed:
            self.status_badge.setText("等待连接")
            self.active_stage_label.setText("等待设备连接")
            self.active_detail_label.setText("确认脑电设备连接完成后，点击开始预训练即可。")
            self.btn_start.setEnabled(False)
        self._refresh_polish(self.device_card, self.device_state_label, self.status_badge)

    def _device_ready(self) -> bool:
        return self.device_info is not None and self.capture_worker is not None

    def start_pretrain(self) -> None:
        if self.pretrain_timer.isActive():
            return
        if not self._device_ready():
            self._log("请先连接脑电设备，连接完成后再开始预训练。")
            self.top_status_label.setText("请先连接脑电设备")
            self._refresh_device_waiting()
            return
        if self.completed:
            self._reset_state(write_log=False)
            self._apply_device_ready()
        self.status_badge.setText("运行中")
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("暂停")
        self.top_status_label.setText("预训练流程运行中")
        self._log("已进入预设预训练流程。")
        self._update_progress()
        self.pretrain_timer.start()

    def pause_pretrain(self) -> None:
        if self.completed:
            return
        if self.pretrain_timer.isActive():
            self.pretrain_timer.stop()
            self.status_badge.setText("已暂停")
            self.btn_pause.setText("继续")
            self.top_status_label.setText("预训练流程已暂停")
            self._log("预训练流程已暂停。")
            self._update_progress(paused=True)
            return
        self.status_badge.setText("运行中")
        self.btn_pause.setText("暂停")
        self.top_status_label.setText("预训练流程运行中")
        self._log("预训练流程继续运行。")
        self.pretrain_timer.start()
        self._update_progress()

    def reset_pretrain(self) -> None:
        self._reset_state(write_log=True)
        if self._device_ready():
            self._apply_device_ready()

    def _reset_state(self, *, write_log: bool) -> None:
        self.pretrain_timer.stop()
        self.control_timer.stop()
        self.step_index = 0
        self.step_ticks = 0
        self.control_tick = 0
        self.completed = False
        self.control_screen_ready = False
        self.overall_progress.setValue(0)
        self.stage_progress.setValue(0)
        self.progress_caption.setText("总进度 0% | 当前阶段 0%")
        self.btn_start.setText("开始预训练")
        self.btn_pause.setText("暂停")
        self.btn_pause.setEnabled(False)
        for index in range(len(PRETRAIN_STEPS)):
            self._set_step_state(index, "pending", "等待")
        self.metric_labels["ssvep"].setText("0/24")
        self.metric_labels["mi"].setText("0/120")
        self.metric_labels["epoch"].setText("0/12")
        self.metric_labels["ready"].setText("0%")
        self._refresh_metric_summary()
        self.pretrain_log.clear()
        self._refresh_device_waiting()
        if write_log:
            self._log("预训练流程已重置。")

    def _advance(self) -> None:
        if self.step_index >= len(PRETRAIN_STEPS):
            self._finish()
            return
        step = PRETRAIN_STEPS[self.step_index]
        if self.step_ticks == 0:
            self._log(f"开始阶段：{step['title']}。")
        self.step_ticks += 1
        if self.step_ticks >= int(step["duration"]):
            self._log(f"完成阶段：{step['title']}。")
            self.step_index += 1
            self.step_ticks = 0
            if self.step_index >= len(PRETRAIN_STEPS):
                self._finish()
                return
        self._update_progress()

    def _finish(self) -> None:
        self.pretrain_timer.stop()
        self.completed = True
        self.status_badge.setText("已完成")
        self.active_stage_label.setText("预训练完成")
        self.active_detail_label.setText("当前前端流程已完成，后续可接入真实训练与配置保存。")
        self.overall_progress.setValue(100)
        self.stage_progress.setValue(100)
        self.progress_caption.setText("总进度 100% | 当前阶段 100%")
        for index in range(len(PRETRAIN_STEPS)):
            self._set_step_state(index, "done", "完成")
        self.metric_labels["ssvep"].setText("24/24")
        self.metric_labels["mi"].setText("120/120")
        self.metric_labels["epoch"].setText("12/12")
        self.metric_labels["ready"].setText("100%")
        self._refresh_metric_summary()
        self.btn_start.setText("重新开始")
        self.btn_start.setEnabled(self._device_ready())
        self.btn_pause.setEnabled(False)
        self.top_status_label.setText("预训练流程完成")
        self._log("预训练前端流程已完成。")

        self.show_control_screen()

    def _update_progress(self, *, paused: bool = False) -> None:
        total_duration = sum(int(step["duration"]) for step in PRETRAIN_STEPS)
        completed_duration = sum(int(step["duration"]) for step in PRETRAIN_STEPS[: self.step_index])
        active_index = min(self.step_index, len(PRETRAIN_STEPS) - 1)
        active_step = PRETRAIN_STEPS[active_index]
        active_duration = int(active_step["duration"])
        overall = min(100, round((completed_duration + self.step_ticks) * 100 / total_duration))
        stage = min(100, round(self.step_ticks * 100 / active_duration))
        self.overall_progress.setValue(int(overall))
        self.stage_progress.setValue(int(stage))
        self.progress_caption.setText(f"总进度 {overall}% | 当前阶段 {stage}%")
        self.active_stage_label.setText(str(active_step["title"]))
        self.active_detail_label.setText(str(active_step["detail"]))
        for index in range(len(PRETRAIN_STEPS)):
            if index < self.step_index:
                self._set_step_state(index, "done", "完成")
            elif index == self.step_index:
                self._set_step_state(index, "active", "暂停" if paused else "进行")
            else:
                self._set_step_state(index, "pending", "等待")
        ssvep_ratio = 1.0 if self.step_index > 0 else (stage / 100.0 if self.step_index == 0 else 0.0)
        mi_ratio = 1.0 if self.step_index > 2 else (stage / 100.0 if self.step_index == 2 else 0.0)
        epoch_ratio = 1.0 if self.step_index > 5 else (stage / 100.0 if self.step_index == 5 else 0.0)
        readiness = max(0, overall - 4 if self.step_index >= 4 else 0)
        self.metric_labels["ssvep"].setText(f"{round(24 * ssvep_ratio)}/24")
        self.metric_labels["mi"].setText(f"{round(120 * mi_ratio)}/120")
        self.metric_labels["epoch"].setText(f"{round(12 * epoch_ratio)}/12")
        self.metric_labels["ready"].setText(f"{readiness}%")
        self._refresh_metric_summary()

    def _refresh_metric_summary(self) -> None:
        self.metric_summary_label.setText(
            "SSVEP "
            f"{self.metric_labels['ssvep'].text()} · MI {self.metric_labels['mi'].text()} · "
            f"训练 {self.metric_labels['epoch'].text()} · 准备度 {self.metric_labels['ready'].text()}"
        )

    def _set_step_state(self, index: int, state: str, label: str) -> None:
        row = self.step_rows[index]
        for key in ("frame", "index", "title", "detail", "state"):
            widget = row[key]
            widget.setProperty("stepState", state)
            self._refresh_polish(widget)
        row["state"].setText(label)

    def _log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        for attr in ("pretrain_log", "log_text"):
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            try:
                widget.appendPlainText(f"[{stamp}] {text}")
            except RuntimeError:
                continue

    @staticmethod
    def _refresh_polish(*widgets: QWidget) -> None:
        for widget in widgets:
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            widget.update()

    def show_control_screen(self) -> None:
        self.control_screen_ready = True
        self.control_timer.stop()
        self.control_tick = 0
        self.control_state_nodes = []
        self.control_state_labels = []
        self.ssvep_flow_rows = []
        self.mi_flow_rows = []
        self.pose_value_labels = {}
        self.camera_widget = None
        self.control_status_label = None
        self.robot_status_label = None
        self.camera_status_label = None
        self.command_status_label = None
        self.selected_block_label = None
        self.block_buttons = {}
        self.selected_block_id = None
        self.pressed_move_keys.clear()
        root = QWidget(self)
        root.setObjectName("controlRoot")
        root.setFocusPolicy(Qt.StrongFocus)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        state_bar = QFrame()
        state_bar.setObjectName("stateBar")
        state_layout = QHBoxLayout(state_bar)
        state_layout.setContentsMargins(14, 12, 14, 12)
        state_layout.setSpacing(10)

        state_header = QWidget()
        state_header_layout = QVBoxLayout(state_header)
        state_header_layout.setContentsMargins(0, 0, 6, 0)
        state_header_layout.setSpacing(4)
        title = QLabel("状态机")
        title.setObjectName("stateBarTitle")
        self.control_status_label = QLabel("摄像头读取中，WASD 与数字键已就绪")
        self.control_status_label.setObjectName("controlStatusLabel")
        state_header_layout.addWidget(title)
        state_header_layout.addWidget(self.control_status_label)
        state_layout.addWidget(state_header)

        for index, name in enumerate(CONTROL_STATE_NAMES):
            node = QFrame()
            node.setObjectName("stateNode")
            node.setProperty("stateState", "active" if index == 1 else "pending")
            node_layout = QVBoxLayout(node)
            node_layout.setContentsMargins(10, 6, 10, 6)
            node_label = QLabel(name)
            node_label.setObjectName("stateNodeLabel")
            node_label.setAlignment(Qt.AlignCenter)
            node_layout.addWidget(node_label)
            state_layout.addWidget(node, 1)
            self.control_state_nodes.append(node)
            self.control_state_labels.append(node_label)
        layout.addWidget(state_bar)

        main = QHBoxLayout()
        main.setSpacing(14)
        layout.addLayout(main, 1)

        camera_card = QFrame()
        camera_card.setObjectName("cameraCard")
        camera_layout = QVBoxLayout(camera_card)
        camera_layout.setContentsMargins(14, 14, 14, 14)
        camera_layout.setSpacing(10)
        camera_header = QHBoxLayout()
        camera_title = QLabel("机械臂摄像头")
        camera_title.setObjectName("controlTitle")
        camera_header.addWidget(camera_title, 1)
        self.camera_status_label = QLabel("等待摄像头")
        self.camera_status_label.setObjectName("controlPill")
        camera_header.addWidget(self.camera_status_label)
        self.btn_camera_restart = QPushButton("重连摄像头")
        self.btn_camera_restart.setFocusPolicy(Qt.NoFocus)
        self.btn_camera_restart.clicked.connect(self.restart_camera_capture)
        camera_header.addWidget(self.btn_camera_restart)
        camera_layout.addLayout(camera_header)
        self.camera_widget = RobotCameraPreviewWidget()
        camera_layout.addWidget(self.camera_widget, 1)
        main.addWidget(camera_card, 3)

        side = QWidget()
        side.setObjectName("controlSide")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(12)

        connection_card = QFrame()
        connection_card.setObjectName("poseCard")
        connection_layout = QVBoxLayout(connection_card)
        connection_layout.setContentsMargins(14, 14, 14, 14)
        connection_layout.setSpacing(8)
        connection_title = QLabel("机械臂连接")
        connection_title.setObjectName("controlTitle")
        connection_layout.addWidget(connection_title)
        self.robot_status_label = QLabel(f"默认地址 {self.robot_config.robot_host}:{self.robot_config.robot_port}")
        self.robot_status_label.setObjectName("mutedLabel")
        self.robot_status_label.setWordWrap(True)
        connection_layout.addWidget(self.robot_status_label)
        self.command_status_label = QLabel("WASD 待命")
        self.command_status_label.setObjectName("controlStatusLabel")
        self.command_status_label.setWordWrap(True)
        connection_layout.addWidget(self.command_status_label)
        connection_buttons = QHBoxLayout()
        self.btn_robot_connect = QPushButton("连接")
        self.btn_robot_connect.setProperty("controlType", "primary")
        self.btn_robot_reset = QPushButton("复位")
        self.btn_robot_reset.setProperty("controlType", "warning")
        self.btn_robot_abort = QPushButton("中止")
        self.btn_robot_abort.setProperty("controlType", "danger")
        for button in (self.btn_robot_connect, self.btn_robot_reset, self.btn_robot_abort):
            button.setFocusPolicy(Qt.NoFocus)
        connection_buttons.addWidget(self.btn_robot_connect)
        connection_buttons.addWidget(self.btn_robot_reset)
        connection_buttons.addWidget(self.btn_robot_abort)
        connection_layout.addLayout(connection_buttons)
        self.btn_robot_connect.clicked.connect(self.connect_robot)
        self.btn_robot_reset.clicked.connect(lambda: self._send_robot_command_async("RESET", "复位指令已发送"))
        self.btn_robot_abort.clicked.connect(lambda: self._send_robot_command_async("ABORT", "中止指令已发送"))
        side_layout.addWidget(connection_card)

        pose_card = QFrame()
        pose_card.setObjectName("poseCard")
        pose_layout = QVBoxLayout(pose_card)
        pose_layout.setContentsMargins(14, 14, 14, 14)
        pose_layout.setSpacing(8)
        pose_title = QLabel("机械臂姿态")
        pose_title.setObjectName("controlTitle")
        pose_layout.addWidget(pose_title)
        for key, label, value in (
            ("position", "末端位置", "X 128 · Y -42 · Z 86 mm"),
            ("attitude", "姿态角", "Roll 0.0 · Pitch 12.4 · Yaw -4.8"),
            ("gripper", "夹爪", "待命 · 张开"),
            ("safety", "安全状态", "离线模拟 · 未执行"),
        ):
            row = self._pose_row(label, value)
            value_label = row.findChild(QLabel, "poseValue")
            if value_label is not None:
                self.pose_value_labels[key] = value_label
            pose_layout.addWidget(row)
        side_layout.addWidget(pose_card)

        block_card = QFrame()
        block_card.setObjectName("flowCard")
        block_layout = QVBoxLayout(block_card)
        block_layout.setContentsMargins(14, 14, 14, 14)
        block_layout.setSpacing(8)
        block_title = QLabel("小木块选择")
        block_title.setObjectName("controlTitle")
        block_layout.addWidget(block_title)
        self.selected_block_label = QLabel("按 1 / 2 / 3 / 4 选择目标")
        self.selected_block_label.setObjectName("flowStepDetail")
        self.selected_block_label.setWordWrap(True)
        block_layout.addWidget(self.selected_block_label)
        block_grid = QGridLayout()
        block_grid.setSpacing(8)
        for index, label in enumerate(BLOCK_SLOT_LABELS, start=1):
            button = QPushButton(label)
            button.setProperty("blockState", "pending")
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(lambda _checked=False, slot_id=index: self.select_block(slot_id))
            block_grid.addWidget(button, (index - 1) // 2, (index - 1) % 2)
            self.block_buttons[index] = button
        block_layout.addLayout(block_grid)
        self.btn_pick_selected = QPushButton("确认抓取")
        self.btn_pick_selected.setProperty("controlType", "primary")
        self.btn_pick_selected.setFocusPolicy(Qt.NoFocus)
        self.btn_pick_selected.clicked.connect(self.confirm_selected_block_pick)
        block_layout.addWidget(self.btn_pick_selected)
        side_layout.addWidget(block_card)

        side_layout.addWidget(self._flow_card("WASD 移动控制", MOVE_FLOW_STEPS, active_index=0, row_store=self.mi_flow_rows))
        side_layout.addWidget(self._flow_card("数字选块流程", BLOCK_FLOW_STEPS, active_index=0, row_store=self.ssvep_flow_rows))
        side_layout.addStretch(1)
        main.addWidget(side, 1)

        footer = QLabel("当前控制页已用 WASD 代替 MI，用数字键 1~4 代替 SSVEP 目标选择；无硬件时会保持离线模拟，连接后发送真实机械臂指令。")
        footer.setObjectName("controlFooter")
        layout.addWidget(footer)
        self.setStyleSheet(self._stylesheet())
        self._refresh_manual_control_ui()
        self._start_camera_capture()
        self.connect_robot(force=False)
        self.control_timer.start()
        root.setFocus()

    def _pose_row(self, label: str, value: str) -> QFrame:
        row = QFrame()
        row.setObjectName("poseRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(10, 8, 10, 8)
        title = QLabel(label)
        title.setObjectName("poseLabel")
        detail = QLabel(value)
        detail.setObjectName("poseValue")
        detail.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        detail.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(detail, 1)
        return row

    def _flow_card(
        self,
        title: str,
        steps: Sequence[tuple[str, str]],
        *,
        active_index: int,
        row_store: list[QFrame] | None = None,
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("flowCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        title_label = QLabel(title)
        title_label.setObjectName("controlTitle")
        layout.addWidget(title_label)
        for index, (step_title, detail) in enumerate(steps):
            row = QFrame()
            row.setObjectName("flowRow")
            row.setProperty("flowState", "active" if index == active_index else "pending")
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(10, 7, 10, 7)
            name_label = QLabel(step_title)
            name_label.setObjectName("flowStepTitle")
            detail_label = QLabel(detail)
            detail_label.setObjectName("flowStepDetail")
            detail_label.setWordWrap(True)
            row_layout.addWidget(name_label)
            row_layout.addWidget(detail_label)
            layout.addWidget(row)
            if row_store is not None:
                row_store.append(row)
        return card

    def _hardware_autostart_enabled(self) -> bool:
        if os.environ.get("BRAIN_PRETRAIN_UI_DISABLE_HARDWARE", "").strip() in {"1", "true", "yes"}:
            return False
        app = QApplication.instance()
        platform = "" if app is None else str(app.platformName()).lower()
        return platform not in {"offscreen", "minimal"}

    def _start_camera_capture(self, checked: bool = False, *, force: bool = False) -> None:
        _ = checked
        if self.camera_thread is not None and self.camera_thread.isRunning():
            return
        if not force and not self._hardware_autostart_enabled():
            self._on_camera_status_received("离线测试模式：未自动打开摄像头")
            return
        self.camera_thread = CameraCaptureThread(self.camera_source, parent=self)
        self.camera_thread.frame_received.connect(self.camera_frame_received.emit)
        self.camera_thread.status_changed.connect(self.camera_status_received.emit)
        self.camera_thread.finished.connect(lambda: setattr(self, "camera_thread", None))
        self.camera_thread.start()

    def _stop_camera_capture(self) -> None:
        thread = self.camera_thread
        self.camera_thread = None
        if thread is None:
            return
        thread.request_stop()
        if not thread.wait(1500):
            thread.terminate()
            thread.wait(500)

    def restart_camera_capture(self) -> None:
        self._stop_camera_capture()
        self._start_camera_capture(force=True)

    def _on_camera_frame_received(self, frame: object) -> None:
        if self.camera_widget is not None:
            self.camera_widget.set_frame_bgr(frame)

    def _on_camera_status_received(self, status: str) -> None:
        text = str(status)
        if self.camera_status_label is not None:
            self.camera_status_label.setText(text)
        if self.camera_widget is not None:
            self.camera_widget.set_camera_status(text)

    def connect_robot(self, checked: bool = False, *, force: bool = True) -> None:
        _ = checked
        if self.robot_connecting:
            return
        if not force and not self._hardware_autostart_enabled():
            self._on_robot_connection_result(False, "离线测试模式：未自动连接机械臂")
            return
        self.robot_connecting = True
        if self.btn_robot_connect is not None:
            self.btn_robot_connect.setEnabled(False)
        if self.robot_status_label is not None:
            self.robot_status_label.setText("正在连接机械臂...")

        def _worker() -> None:
            client: RobotClient | None = None
            try:
                self._close_robot_client()
                client = RobotClient(
                    self.robot_config.robot_host,
                    int(self.robot_config.robot_port),
                    event_callback=lambda event: self.robot_log_received.emit(
                        f"机械臂反馈：{event.type} {event.value}"
                    ),
                    timeout_sec=max(float(self.robot_config.robot_timeout_sec), 0.2),
                    reconnect_delay_sec=float(self.robot_config.robot_reconnect_delay_sec),
                )
                client.connect()
                self.robot_client = client
                snapshot: object = {}
                try:
                    snapshot = fetch_robot_status(
                        self.robot_config.robot_host,
                        int(self.robot_config.robot_port),
                        timeout_sec=max(float(self.robot_config.robot_timeout_sec), 0.2),
                    )
                except Exception as status_error:
                    snapshot = {"status_warning": str(status_error)}
                self.robot_connection_result.emit(True, snapshot)
            except Exception as error:
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass
                self.robot_client = None
                self.robot_connection_result.emit(False, str(error))

        threading.Thread(target=_worker, name="pretrain-ui-robot-connect", daemon=True).start()

    def _close_robot_client(self) -> None:
        client = self.robot_client
        self.robot_client = None
        if client is None:
            return
        try:
            client.close()
        except Exception:
            pass

    def _on_robot_connection_result(self, ok: bool, payload: object) -> None:
        self.robot_connecting = False
        self.robot_connected = bool(ok)
        if self.btn_robot_connect is not None:
            self.btn_robot_connect.setEnabled(True)
            self.btn_robot_connect.setText("已连接" if ok else "重连")
        if ok:
            message = "机械臂已连接"
            if isinstance(payload, dict) and payload.get("status_warning"):
                message += f" | 状态读取待重试：{payload.get('status_warning')}"
            if self.robot_status_label is not None:
                self.robot_status_label.setText(message)
            if isinstance(payload, dict):
                self._sync_robot_pose_from_snapshot(payload)
        else:
            if self.robot_status_label is not None:
                self.robot_status_label.setText(f"未连接机械臂，当前为离线模拟：{payload}")
        self._refresh_manual_control_ui()

    def _on_robot_status_snapshot_received(self, payload: object) -> None:
        self.robot_status_poll_in_flight = False
        if isinstance(payload, dict):
            self._sync_robot_pose_from_snapshot(payload)
            if self.robot_status_label is not None:
                state = payload.get("state") or payload.get("health") or "online"
                self.robot_status_label.setText(f"机械臂在线 | {state}")
        elif self.robot_status_label is not None:
            self.robot_status_label.setText(f"状态读取失败：{payload}")
        self._refresh_manual_control_ui()

    def _on_robot_log_received(self, text: str) -> None:
        self._log(str(text))
        if self.command_status_label is not None:
            self.command_status_label.setText(str(text))

    def _poll_robot_status_async(self) -> None:
        if not self.robot_connected or self.robot_status_poll_in_flight:
            return
        self.robot_status_poll_in_flight = True

        def _worker() -> None:
            try:
                snapshot = fetch_robot_status(
                    self.robot_config.robot_host,
                    int(self.robot_config.robot_port),
                    timeout_sec=max(float(self.robot_config.robot_timeout_sec), 0.2),
                )
                self.robot_status_snapshot_received.emit(snapshot)
            except Exception as error:
                self.robot_status_snapshot_received.emit(str(error))

        threading.Thread(target=_worker, name="pretrain-ui-robot-status", daemon=True).start()

    def _sync_robot_pose_from_snapshot(self, snapshot: dict[str, object]) -> None:
        robot_cyl = snapshot.get("robot_cyl")
        if isinstance(robot_cyl, dict):
            try:
                self.robot_pose_cyl = [
                    float(robot_cyl.get("theta_deg", self.robot_pose_cyl[0])),
                    float(robot_cyl.get("radius_mm", self.robot_pose_cyl[1])),
                    float(robot_cyl.get("z_mm", self.robot_pose_cyl[2])),
                ]
                return
            except (TypeError, ValueError):
                pass
        robot_xy = snapshot.get("robot_xy")
        if isinstance(robot_xy, (list, tuple)) and len(robot_xy) >= 2:
            try:
                from hybrid_controller.cylindrical import cartesian_to_cylindrical

                self.robot_pose_cyl = list(
                    cartesian_to_cylindrical(
                        float(robot_xy[0]),
                        float(robot_xy[1]),
                        float(snapshot.get("robot_z", self.robot_pose_cyl[2])),
                    )
                )
            except (TypeError, ValueError):
                pass

    def _refresh_manual_control_ui(self) -> None:
        theta_deg, radius_mm, z_mm = (float(value) for value in self.robot_pose_cyl)
        x_mm, y_mm, _z = cylindrical_to_cartesian(theta_deg, radius_mm, z_mm)
        if "position" in self.pose_value_labels:
            self.pose_value_labels["position"].setText(f"X {x_mm:.1f} · Y {y_mm:.1f} · Z {z_mm:.1f} mm")
        if "attitude" in self.pose_value_labels:
            self.pose_value_labels["attitude"].setText(f"Theta {theta_deg:.1f}° · R {radius_mm:.1f} mm")
        if "gripper" in self.pose_value_labels:
            target = "--" if self.selected_block_id is None else f"小木块 {self.selected_block_id}"
            self.pose_value_labels["gripper"].setText(f"目标 {target} · 待命")
        if "safety" in self.pose_value_labels:
            mode = "真实连接" if self.robot_connected else "离线模拟"
            self.pose_value_labels["safety"].setText(f"{mode} · WASD 可用")

        selected_text = "未选择目标" if self.selected_block_id is None else f"已选择小木块 {self.selected_block_id}"
        if self.selected_block_label is not None:
            self.selected_block_label.setText(selected_text)
        if self.camera_widget is not None:
            self.camera_widget.set_selected_slot(self.selected_block_id)

        for slot_id, button in self.block_buttons.items():
            button.setProperty("blockState", "active" if slot_id == self.selected_block_id else "pending")
            self._refresh_polish(button)

        if self.pressed_move_keys:
            self._set_control_state(2)
            self._set_flow_state(self.mi_flow_rows, 1)
            if self.control_status_label is not None:
                self.control_status_label.setText("WASD 移动流程：正在微调机械臂")
        elif self.selected_block_id is not None:
            self._set_control_state(3)
            self._set_flow_state(self.mi_flow_rows, 2)
            if self.control_status_label is not None:
                self.control_status_label.setText(f"数字选块流程：小木块 {self.selected_block_id} 已锁定")
        else:
            self._set_control_state(1)
            self._set_flow_state(self.mi_flow_rows, 0)
            if self.control_status_label is not None:
                self.control_status_label.setText("摄像头读取中，WASD 与数字键已就绪")
        self._set_flow_state(self.ssvep_flow_rows, 1 if self.selected_block_id is not None else 0)

    def select_block(self, slot_id: int) -> None:
        if int(slot_id) not in {1, 2, 3, 4}:
            return
        self.selected_block_id = int(slot_id)
        self._log(f"选择小木块 {slot_id}")
        if self.command_status_label is not None:
            self.command_status_label.setText(f"小木块 {slot_id} 已选中")
        self._refresh_manual_control_ui()
        self.setFocus()

    def confirm_selected_block_pick(self) -> None:
        if self.selected_block_id is None:
            if self.command_status_label is not None:
                self.command_status_label.setText("请先用 1 / 2 / 3 / 4 选择小木块")
            return
        slots = {slot.slot_id: slot for slot in self.slot_catalog.list_pick_slots(source="hardware")}
        slot = slots.get(int(self.selected_block_id))
        if slot is None:
            self._on_robot_log_received(f"小木块 {self.selected_block_id} 未配置抓取坐标")
            return
        theta_deg, radius_mm, _z_mm = slot.cylindrical_trz
        self._set_control_state(4)
        self._set_flow_state(self.ssvep_flow_rows, 2)
        self._send_robot_command_async(
            f"PICK_CYL {theta_deg:.2f} {radius_mm:.2f}",
            f"小木块 {self.selected_block_id} 抓取指令已发送",
        )

    def _send_robot_command_async(self, command: str, status_text: str | None = None) -> None:
        command_text = str(command).strip()
        if not command_text:
            return
        if status_text and self.command_status_label is not None:
            self.command_status_label.setText(status_text)
        if not self.robot_connected or self.robot_client is None:
            self.robot_log_received.emit(f"离线模拟指令：{command_text}")
            return

        def _worker() -> None:
            try:
                with self.robot_io_lock:
                    if self.robot_client is None:
                        raise RuntimeError("robot client disconnected")
                    self.robot_client.send_command(command_text)
                self.robot_log_received.emit(f"机械臂 <= {command_text}")
            except Exception as error:
                self.robot_connected = False
                self.robot_log_received.emit(f"机械臂发送失败：{error}")

        threading.Thread(target=_worker, name="pretrain-ui-robot-send", daemon=True).start()

    def _handle_control_key_press(self, token: str) -> None:
        normalized = str(token).strip().lower()
        if normalized in {"w", "a", "s", "d"}:
            self.pressed_move_keys.add(normalized)
            self._send_move_step(force=True)
            self._refresh_manual_control_ui()
            return
        if normalized in {"1", "2", "3", "4"}:
            self.select_block(int(normalized))
            return
        if normalized in {"enter", "return"}:
            self.confirm_selected_block_pick()
            return
        if normalized in {"escape", "x"}:
            self._send_robot_command_async("ABORT", "中止指令已发送")

    def _handle_control_key_release(self, token: str) -> None:
        normalized = str(token).strip().lower()
        if normalized not in {"w", "a", "s", "d"}:
            return
        self.pressed_move_keys.discard(normalized)
        if not self.pressed_move_keys and self.command_status_label is not None:
            self.command_status_label.setText("WASD 待命")
        self._refresh_manual_control_ui()

    @staticmethod
    def _key_to_token(key: int) -> str:
        mapping = {
            Qt.Key_W: "w",
            Qt.Key_A: "a",
            Qt.Key_S: "s",
            Qt.Key_D: "d",
            Qt.Key_1: "1",
            Qt.Key_2: "2",
            Qt.Key_3: "3",
            Qt.Key_4: "4",
            Qt.Key_Return: "enter",
            Qt.Key_Enter: "enter",
            Qt.Key_Escape: "escape",
            Qt.Key_X: "x",
        }
        return mapping.get(key, "")

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self.control_screen_ready:
            token = self._key_to_token(int(event.key()))
            if token:
                self._handle_control_key_press(token)
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if self.control_screen_ready and not event.isAutoRepeat():
            token = self._key_to_token(int(event.key()))
            if token:
                self._handle_control_key_release(token)
                event.accept()
                return
        super().keyReleaseEvent(event)

    def _compute_move_delta(self) -> tuple[float, float]:
        theta_delta = 0.0
        radius_delta = 0.0
        if "a" in self.pressed_move_keys:
            theta_delta -= float(self.robot_config.teleop_theta_step_deg)
        if "d" in self.pressed_move_keys:
            theta_delta += float(self.robot_config.teleop_theta_step_deg)
        if "w" in self.pressed_move_keys:
            radius_delta += float(self.robot_config.teleop_radius_step_mm)
        if "s" in self.pressed_move_keys:
            radius_delta -= float(self.robot_config.teleop_radius_step_mm)
        return theta_delta, radius_delta

    def _send_move_step(self, *, force: bool = False) -> None:
        if not self.pressed_move_keys:
            return
        now = time.monotonic()
        repeat_sec = max(0.05, float(self.robot_config.teleop_repeat_interval_ms) / 1000.0)
        if not force and (now - self.last_move_command_ts) < repeat_sec:
            return
        theta_delta, radius_delta = self._compute_move_delta()
        if theta_delta == 0.0 and radius_delta == 0.0:
            return
        theta_deg = normalize_theta_deg(float(self.robot_pose_cyl[0]) + theta_delta)
        radius_mm = clamp(float(self.robot_pose_cyl[1]) + radius_delta, self.robot_config.robot_auto_radius_limits_mm)
        self.robot_pose_cyl[0] = theta_deg
        self.robot_pose_cyl[1] = radius_mm
        self.last_move_command_ts = now
        command = f"MOVE_CYL_AUTO {theta_deg:.2f} {radius_mm:.2f}"
        if self.command_status_label is not None:
            keys = "+".join(sorted(self.pressed_move_keys)).upper()
            self.command_status_label.setText(f"{keys} -> θ {theta_deg:.1f}° · R {radius_mm:.1f} mm")
        self._send_robot_command_async(command)

    def _pump_manual_control(self) -> None:
        if not self.control_screen_ready:
            return
        self.control_tick += 1
        self._send_move_step()
        now = time.monotonic()
        if self.robot_connected and (now - self.last_robot_status_poll_ts) >= 1.0:
            self.last_robot_status_poll_ts = now
            self._poll_robot_status_async()

    def _advance_control_simulation(self) -> None:
        if not self.control_screen_ready:
            return
        self._pump_manual_control()

    def _apply_control_step(self, index: int) -> None:
        if not CONTROL_SIM_STEPS:
            return
        step = CONTROL_SIM_STEPS[int(index) % len(CONTROL_SIM_STEPS)]
        state_index = int(step.get("state_index", 0))
        self._set_control_state(state_index)
        if self.control_status_label is not None:
            self.control_status_label.setText(str(step.get("top_status", "")))
        if self.camera_widget is not None:
            self.camera_widget.set_simulation_state(
                target=str(step.get("target", "")),
                status=str(step.get("camera_status", "")),
                phase=int(index),
            )
        pose_values = tuple(str(item) for item in step.get("pose", ()))
        for key, value in zip(("position", "attitude", "gripper", "safety"), pose_values):
            label = self.pose_value_labels.get(key)
            if label is not None:
                label.setText(value)
        self._set_flow_state(self.ssvep_flow_rows, int(step.get("ssvep_index", -1)))
        self._set_flow_state(self.mi_flow_rows, int(step.get("mi_index", -1)))

    def _set_control_state(self, active_index: int) -> None:
        for index, node in enumerate(self.control_state_nodes):
            if index < active_index:
                state = "done"
            elif index == active_index:
                state = "active"
            else:
                state = "pending"
            node.setProperty("stateState", state)
            node.setStyleSheet(self._state_node_style(state))

    def _set_flow_state(self, rows: Sequence[QFrame], active_index: int) -> None:
        for index, row in enumerate(rows):
            if active_index < 0:
                state = "pending"
            elif index < active_index:
                state = "done"
            elif index == active_index:
                state = "active"
            else:
                state = "pending"
            row.setProperty("flowState", state)
            row.setStyleSheet(self._flow_row_style(state))

    @staticmethod
    def _state_node_style(state: str) -> str:
        if state == "active":
            return "background: #122A22; border: 1px solid #57D6A6; border-radius: 8px;"
        if state == "done":
            return "background: #172233; border: 1px solid #41617F; border-radius: 8px;"
        return "background: #10161F; border: 1px solid #293446; border-radius: 8px;"

    @staticmethod
    def _flow_row_style(state: str) -> str:
        if state == "active":
            return "background: #122A22; border: 1px solid #57D6A6; border-radius: 8px;"
        if state == "done":
            return "background: #172233; border: 1px solid #41617F; border-radius: 8px;"
        return "background: #10161F; border: 1px solid #293446; border-radius: 8px;"

    def _noop(self) -> None:
        pass

    @staticmethod
    def _stylesheet() -> str:
        return (
            "QWidget { color: #E8EEF6; font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif; background: transparent; }"
            "QMainWindow, QWidget#root { background: #0B0E13; }"
            "QWidget#controlRoot { background: #0B0E13; }"
            "QWidget#leftPanel, QWidget#rightPanel, QGroupBox, QFrame#card, QFrame#hero {"
            "  background: #151B24; border: 1px solid #2A3444; border-radius: 8px;"
            "}"
            "QFrame#stateBar, QFrame#cameraCard, QFrame#poseCard, QFrame#flowCard {"
            "  background: #151B24; border: 1px solid #2A3444; border-radius: 8px;"
            "}"
            "QFrame#stateNode { background: #10161F; border: 1px solid #293446; border-radius: 8px; }"
            "QFrame#stateNode[stateState='active'] { background: #122A22; border-color: #57D6A6; }"
            "QFrame#stateNode[stateState='done'] { background: #172233; border-color: #41617F; }"
            "QLabel#stateBarTitle, QLabel#controlTitle { color: #F0F6FC; font-size: 12pt; font-weight: 900; }"
            "QLabel#controlStatusLabel { color: #A9F5D0; font-size: 9pt; font-weight: 800; }"
            "QLabel#stateNodeLabel { color: #DCE8F4; font-weight: 800; }"
            "QWidget#controlSide { background: transparent; }"
            "QFrame#poseRow, QFrame#flowRow { background: #10161F; border: 1px solid #293446; border-radius: 8px; }"
            "QFrame#flowRow[flowState='active'] { background: #122A22; border-color: #57D6A6; }"
            "QFrame#flowRow[flowState='done'] { background: #172233; border-color: #41617F; }"
            "QLabel#poseLabel, QLabel#flowStepDetail, QLabel#controlFooter { color: #AAB7C5; }"
            "QLabel#poseValue, QLabel#flowStepTitle { color: #F0F6FC; font-weight: 800; }"
            "QLabel#controlPill { color: #0D1117; background: #A9F5D0; border-radius: 8px; padding: 6px 10px; font-weight: 900; }"
            "QWidget#robotCamera { background: #0B1724; border: 1px solid #2F4058; border-radius: 12px; }"
            "QGroupBox { margin-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #A9F5D0; font-weight: 800; }"
            "QFrame#hero { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #17202C,stop:1 #1B221C); border-color: #3A5145; }"
            "QLabel#heroTitle { color: #F4F8FB; font-size: 20pt; font-weight: 900; }"
            "QLabel#mutedLabel, QLabel#stepDetail, QLabel#metricTitle { color: #AAB7C5; }"
            "QLabel#topStatus { background: #151B24; border: 1px solid #2A3444; border-radius: 8px; padding: 9px 12px; font-size: 12pt; font-weight: 800; }"
            "QLabel#statusBadge { color: #0D1117; background: #A9F5D0; border-radius: 8px; padding: 8px 12px; font-weight: 900; }"
            "QLabel#statusBadge[deviceState='waiting'] { color: #D7DEE8; background: #2B3441; }"
            "QLabel#deviceStateTitle { color: #F0F6FC; font-size: 13pt; font-weight: 900; }"
            "QLabel#deviceStateTitle[deviceState='ready'] { color: #A9F5D0; }"
            "QLabel#cardTitle, QLabel#stageTitle, QLabel#stepTitle { color: #F0F6FC; font-size: 11pt; font-weight: 900; }"
            "QLabel#summaryValue { color: #DCE8F4; font-weight: 700; }"
            "QLabel#metricValue { color: #F6C667; font-size: 14pt; font-weight: 900; }"
            "QFrame#tile { border: 1px solid #293446; border-radius: 8px; background: #10161F; }"
            "QFrame#card[deviceState='ready'] { border-color: #57D6A6; background: #12201C; }"
            "QComboBox, QSpinBox, QPlainTextEdit { background: #0E141C; border: 1px solid #2A3545; border-radius: 8px; color: #E8EEF6; padding: 6px 8px; }"
            "QPlainTextEdit#logPanel { background: #080C11; color: #C8D3E0; font-family: Consolas, 'Microsoft YaHei UI', monospace; }"
            "QPushButton { background: #202A37; border: 1px solid #3A4658; border-radius: 8px; color: #EEF4FA; padding: 8px 12px; font-weight: 800; }"
            "QPushButton:hover { background: #263444; border-color: #6BE7B3; }"
            "QPushButton:disabled { background: #151A22; border-color: #242B35; color: #687482; }"
            "QPushButton[controlType='primary'] { background: #176B5A; border-color: #42C79D; color: #F2FFF9; }"
            "QPushButton[controlType='primary']:disabled { background: #16221F; border-color: #2A3A36; color: #667872; }"
            "QPushButton[controlType='danger'] { background: #6D2632; border-color: #B84A5B; color: #FFECEF; }"
            "QPushButton[controlType='warning'] { background: #6E5425; border-color: #D5A64B; color: #FFF4D6; }"
            "QPushButton[blockState='active'] { background: #1C5D4D; border-color: #70E7B8; color: #F2FFF9; }"
            "QProgressBar { border: 1px solid #2D394A; border-radius: 7px; background: #0C1118; color: #E8EEF6; text-align: center; min-height: 16px; }"
            "QProgressBar::chunk { border-radius: 6px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #48D6B0,stop:0.55 #9DE2C0,stop:1 #F6C667); }"
            "QFrame#stepRow { border: 1px solid #283343; border-radius: 8px; background: #10161F; min-height: 34px; }"
            "QFrame#stepRow[stepState='active'] { border-color: #67E8B9; background: #14251F; }"
            "QFrame#stepRow[stepState='done'] { border-color: #3F8F73; background: #12201C; }"
            "QLabel#stepIndex { color: #0B0E13; background: #7DEBC0; border-radius: 8px; font-weight: 900; }"
            "QLabel#stepIndex[stepState='pending'] { background: #445161; color: #D2DAE5; }"
            "QLabel#stepIndex[stepState='done'] { background: #F6C667; }"
            "QLabel#stepState { color: #A9F5D0; font-weight: 900; }"
            "QLabel#stepState[stepState='pending'] { color: #8B97A5; }"
            "QLabel#stepState[stepState='done'] { color: #F6C667; }"
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self.control_timer.stop()
        self._stop_camera_capture()
        self._close_robot_client()
        if self.capture_worker is not None and hasattr(self.capture_worker, "request_stop"):
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
    window = PretrainWindow()
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
