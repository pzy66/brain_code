from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Sequence

try:
    from PyQt5.QtCore import Qt, QTimer, QRectF
    from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QPointF
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QProgressBar,
        QSizePolicy,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as error:  # pragma: no cover - import guard for machines without GUI deps
    raise RuntimeError("PyQt5 is required to run the workbench UI") from error


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brain_workspace.paths import (  # noqa: E402
    HYBRID_CONTROLLER_DIR,
    MI_DATASET_DIR,
    SSVEP_DATASET_DIR,
)
from .state import ArtifactRow, StatusCard, collect_workbench_state, short_path  # noqa: E402


APP_TITLE = "软件著作权流程工作台 V1.0"
APP_SUBTITLE = "先做端到端流程UI：MI/SSVEP采集 -> 预训练 -> 机器人控制（先用键盘模拟）"

WORKFLOW_STEPS = (
    ("collect_mi", "MI 数据采集", 20),
    ("collect_ssvep", "SSVEP 数据采集", 18),
    ("pretrain", "模型预训练", 28),
    ("ready", "流程就绪", 8),
)


KEYBOARD_CONTROL_MAP = {
    Qt.Key_W: ("MOVE", "FORWARD"),
    Qt.Key_S: ("MOVE", "BACKWARD"),
    Qt.Key_A: ("MOVE", "LEFT"),
    Qt.Key_D: ("MOVE", "RIGHT"),
    Qt.Key_Q: ("ROTATE", "CCW"),
    Qt.Key_E: ("ROTATE", "CW"),
    Qt.Key_Z: ("Z_AXIS", "UP"),
    Qt.Key_X: ("Z_AXIS", "DOWN"),
    Qt.Key_C: ("GRIP", "CLOSE"),
    Qt.Key_V: ("GRIP", "OPEN"),
}


def _format_percent(value: float) -> str:
    return f"{int(value * 100)}%"


@dataclass
class WorkflowState:
    is_running: bool = False
    is_paused: bool = False
    step_index: int = 0
    step_ticks: list[int] = field(default_factory=lambda: [0] * len(WORKFLOW_STEPS))
    input_source: str = "Keyboard"
    robot_connected: bool = False
    ready_for_control: bool = False
    current_command: str = "IDLE"
    command_history: Deque[str] = field(default_factory=lambda: deque(maxlen=30))
    epoch: int = 0
    accuracy: float = 0.0
    loss: float = 1.0
    last_key_seconds: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.is_running = False
        self.is_paused = False
        self.step_index = 0
        self.step_ticks = [0] * len(WORKFLOW_STEPS)
        self.ready_for_control = False
        self.current_command = "IDLE"
        self.epoch = 0
        self.accuracy = 0.0
        self.loss = 1.0
        self.last_key_seconds.clear()
        self.command_history.clear()

    def has_completed(self) -> bool:
        return self.step_index >= len(WORKFLOW_STEPS) and not self.is_running


class StatusPill(QLabel):
    COLORS = {
        "good": ("#1F7A4D", "#E9F7EF"),
        "warn": ("#A35A00", "#FFF3DF"),
        "bad": ("#A33838", "#FCEAEA"),
        "neutral": ("#4C5B6B", "#EEF2F6"),
    }

    def __init__(self, text: str, level: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        fg, bg = self.COLORS.get(level, self.COLORS["neutral"])
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(24)
        self.setStyleSheet(
            "QLabel {"
            f"color: {fg}; background: {bg}; border: 1px solid {fg};"
            "border-radius: 4px; padding: 2px 8px; font: 9pt 'Microsoft YaHei UI';"
            "}"
        )


class MetricTile(QFrame):
    def __init__(self, title: str, value: str, detail: str, level: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("metricTile")
        self.setMinimumHeight(92)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        row = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("tileTitle")
        row.addWidget(title_label, stretch=1)
        row.addWidget(StatusPill(value, level))
        layout.addLayout(row)
        detail_label = QLabel(detail)
        detail_label.setObjectName("tileDetail")
        detail_label.setWordWrap(True)
        layout.addWidget(detail_label)


class PipelineWidget(QWidget):
    STEPS = [
        ("MI 采集", "#4F7DB8"),
        ("SSVEP 采集", "#7A5CA8"),
        ("预训练", "#8C6D31"),
        ("流程就绪", "#3F7F87"),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(176)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.step_progress = [0.0] * len(self.STEPS)
        self.active_step = -1
        self.is_paused = False
        self.overall_progress = 0.0
        self.current_label = "IDLE"

    def set_state(self, active_step: int, step_progress: Sequence[float], overall_progress: float, label: str, *, is_paused: bool = False) -> None:
        self.active_step = max(-1, min(active_step, len(self.STEPS) - 1))
        self.step_progress = list(step_progress)
        self.overall_progress = max(0.0, min(overall_progress, 1.0))
        self.current_label = label
        self.is_paused = is_paused
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#F7F9FB"))
        painter.setFont(QFont("Microsoft YaHei UI", 10))

        margin_x = 34
        margin_top = 26
        line_y = self.height() // 2 + 8
        usable_w = max(1, self.width() - margin_x * 2)
        gap = usable_w / max(1, len(self.STEPS) - 1)

        painter.setPen(QPen(QColor("#CBD3DC"), 3))
        painter.drawLine(margin_x, line_y, self.width() - margin_x, line_y)

        for index, (label, color) in enumerate(self.STEPS):
            x = margin_x + index * gap
            radius = 21
            progress = 0.0
            if index < len(self.step_progress):
                progress = max(0.0, min(1.0, self.step_progress[index]))
            color_value = QColor(color)
            if index < self.active_step:
                fill = QColor("#4E8A62")
            elif index == self.active_step:
                fill = color_value
            else:
                fill = QColor("#B9C4D0")

            painter.setBrush(fill)
            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            painter.drawEllipse(QPointF(float(x), float(line_y)), radius, radius)

            if index == self.active_step and self.active_step >= 0 and not self.is_paused:
                arc_rect = QRectF(x - radius, line_y - radius, radius * 2, radius * 2)
                painter.setPen(QPen(QColor("#FFFFFF"), 3))
                painter.drawArc(arc_rect, 90 * 16, int(-360 * progress) * 16)

            painter.setPen(QColor("#25313D"))
            painter.drawText(QRectF(x - 36, line_y + 28, 72, 24), Qt.AlignCenter, label)
            painter.setPen(QColor("#25313D"))
            painter.setFont(QFont("Microsoft YaHei UI", 9))
            painter.drawText(QRectF(x - 18, line_y - 38, 36, 18), Qt.AlignCenter, f"{int(progress*100)}%")

        painter.setPen(QColor("#25313D"))
        painter.setFont(QFont("Microsoft YaHei UI", 11))
        painter.drawText(20, margin_top, "流程执行总览")
        painter.setFont(QFont("Microsoft YaHei UI", 9))
        state_color = QColor("#1F7A4D") if not self.is_paused else QColor("#A35A00")
        painter.setPen(state_color)
        painter.drawText(130, margin_top, f"状态: {self.current_label}")
        painter.drawText(130, margin_top + 16, f"整体进度: {int(self.overall_progress * 100)}%")


class RobotPreview(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.last_command = "IDLE"

    def set_command(self, text: str) -> None:
        self.last_command = text
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#20252C"))
        frame = self.rect().adjusted(16, 16, -16, -16)
        painter.setPen(QPen(QColor("#6E7B8A"), 2))
        painter.drawRect(frame)
        painter.setPen(QColor("#D7DEE8"))
        painter.setFont(QFont("Consolas", 10))
        painter.drawText(frame.adjusted(10, 10, -10, -10), Qt.AlignLeft | Qt.AlignTop, "robot camera / simulation overlay")

        targets = [
            (0.27, 0.38, 0.16, 0.20, "A", "#69A86E"),
            (0.57, 0.34, 0.18, 0.24, "B", "#D19A45"),
            (0.42, 0.64, 0.14, 0.17, "C", "#5E95C8"),
        ]
        for cx, cy, w, h, label, color in targets:
            rect = QRectF(
                frame.left() + frame.width() * (cx - w / 2),
                frame.top() + frame.height() * (cy - h / 2),
                frame.width() * w,
                frame.height() * h,
            )
            painter.setPen(QPen(QColor(color), 3))
            painter.setBrush(QColor(255, 255, 255, 18))
            painter.drawRoundedRect(rect, 4, 4)
            center = rect.center()
            painter.drawLine(QPointF(center.x() - 8, center.y()), QPointF(center.x() + 8, center.y()))
            painter.drawLine(QPointF(center.x(), center.y() - 8), QPointF(center.x(), center.y() + 8))
            painter.drawText(rect.adjusted(5, 3, -5, -3), Qt.AlignLeft | Qt.AlignTop, f"target {label}")

        path = QPainterPath()
        base_x = frame.left() + frame.width() * 0.83
        base_y = frame.top() + frame.height() * 0.80
        painter.setPen(QPen(QColor("#F0F3F7"), 4))
        path.moveTo(base_x, base_y)
        path.cubicTo(base_x - 34, base_y - 60, base_x - 79, base_y - 90, base_x - 135, base_y - 132)
        painter.drawPath(path)
        painter.setBrush(QColor("#F0F3F7"))
        painter.drawEllipse(QPointF(base_x, base_y), 7, 7)

        painter.setFont(QFont("Microsoft YaHei UI", 9))
        painter.drawText(frame.adjusted(10, frame.height() - 30, -10, -8), Qt.AlignLeft, f"last command: {self.last_command}")


class Section(QGroupBox):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.setObjectName("section")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)


class WorkbenchWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.state = collect_workbench_state()
        self.flow = WorkflowState()
        self.metric_layout: QGridLayout | None = None
        self.material_table: QTableWidget | None = None
        self.pipeline_widget: PipelineWidget | None = None
        self.log_view: QTextEdit | None = None
        self.command_view: QTextEdit | None = None
        self.robot_preview: RobotPreview | None = None
        self.current_cmd_label: QLabel | None = None
        self.control_mode_combo: QComboBox | None = None
        self.mi_collect_bar: QProgressBar | None = None
        self.ssvep_collect_bar: QProgressBar | None = None
        self.train_bar: QProgressBar | None = None
        self.ready_bar: QProgressBar | None = None
        self.epoch_label: QLabel | None = None
        self.loss_label: QLabel | None = None
        self.acc_label: QLabel | None = None
        self.overall_label: QLabel | None = None
        self.robot_state_label: QLabel | None = None
        self.connect_button: QPushButton | None = None
        self.mi_dataset_label: QLabel | None = None
        self.mi_entry_label: QLabel | None = None
        self.ssvep_dataset_label: QLabel | None = None
        self.ssvep_entry_label: QLabel | None = None

        self.setWindowTitle(APP_TITLE)
        self.resize(1440, 900)
        self.setMinimumSize(1180, 760)
        self.setFocusPolicy(Qt.StrongFocus)

        self.page_builders = (
            self._build_overview_page,
            self._build_acquisition_page,
            self._build_training_page,
            self._build_control_page,
            self._build_robot_page,
            self._build_copyright_page,
        )

        root = QWidget(self)
        shell = QVBoxLayout(root)
        shell.setContentsMargins(16, 14, 16, 14)
        shell.setSpacing(12)
        self.setCentralWidget(root)

        shell.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(12)
        shell.addLayout(body, stretch=1)

        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(208)
        for title in ("总览", "采集", "预训练", "控制台", "机械臂", "版权材料"):
            self.nav.addItem(QListWidgetItem(title))
        body.addWidget(self.nav)

        self.pages = QStackedWidget()
        body.addWidget(self.pages, stretch=1)
        self._rebuild_pages()

        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)

        self.log_view = QTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(96)
        shell.addWidget(self.log_view)
        self._log("Workbench loaded. Keyboard-only control mode is enabled for simulation.")
        self._log(f"Repository root: {REPO_ROOT}")

        self._flow_timer = QTimer(self)
        self._flow_timer.setInterval(220)
        self._flow_timer.timeout.connect(self._tick_workflow)
        self._flow_timer.start()

        self.flow_update_last_tick = time.time()
        self._update_labels_and_progress()

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)

        title_box = QVBoxLayout()
        title = QLabel(APP_TITLE)
        title.setObjectName("title")
        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box, stretch=1)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["模拟控制（推荐）", "只看状态", "自动化接入模式（预留 MI/SSVEP）"])
        self.mode_combo.setToolTip("当前为演示架构；键盘可直接驱动机械臂指令队列。")
        layout.addWidget(self.mode_combo)

        self.header_start_button = QPushButton("启动自动流程")
        self.header_start_button.clicked.connect(self.start_workflow)
        layout.addWidget(self.header_start_button)

        refresh = QPushButton("刷新状态")
        refresh.clicked.connect(self.refresh_state)
        layout.addWidget(refresh)

        reset = QPushButton("重置流程")
        reset.clicked.connect(self.reset_workflow)
        layout.addWidget(reset)

        return header

    def _build_overview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(10)
        metrics.setVerticalSpacing(10)
        self.metric_layout = metrics
        self._populate_metrics()
        layout.addLayout(metrics)

        section = Section("流程进度总览")
        section_layout = QVBoxLayout(section)
        self.pipeline_widget = PipelineWidget()
        section_layout.addWidget(self.pipeline_widget)

        self.overall_label = QLabel("系统正在等待启动。")
        section_layout.addWidget(self.overall_label)
        layout.addWidget(section)

        checklist = Section("当前执行清单")
        checklist_layout = QVBoxLayout(checklist)
        for text in (
            "1. MI + SSVEP 采集阶段先以路径和脚本可达性为准。",
            "2. 采集完成后进入统一预训练阶段，自动输出可视化训练指标。",
            "3. 模型可发布后手动进入控制环节，当前先使用键盘作为输入源。",
            "4. 预计未来可无缝切换到 MI / SSVEP 推理适配器。",
        ):
            checklist_layout.addWidget(QLabel("• " + text))
        layout.addWidget(checklist)
        layout.addStretch(1)
        return page

    def _build_acquisition_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)

        mi = Section("MI 采集")
        mi_layout = QVBoxLayout(mi)
        self.mi_dataset_label = QLabel(_short_path(MI_DATASET_DIR))
        self.mi_entry_label = QLabel(_short_path(self.state.mi_contract.collection_entry or Path("未检测到 MI 采集入口")))
        mi_layout.addWidget(self._kv("默认数据目录", _short_path(MI_DATASET_DIR)))
        mi_layout.addWidget(self._kv("采集触发脚本", self._path_or_text(self.state.mi_contract.collection_entry)))
        mi_layout.addWidget(self._kv("MI 状态文件", _short_path(self.state.mi_contract.status_path)))
        mi_layout.addWidget(self._kv("Profile 目录", _short_path(self.state.mi_contract.profile_path.parent)))
        self.mi_collect_bar = QProgressBar()
        self.mi_collect_bar.setMaximum(100)
        mi_layout.addWidget(QLabel("MI 采集进度"))
        mi_layout.addWidget(self.mi_collect_bar)
        mi_layout.addWidget(self._button("打开 MI 采集脚本", self._open_path_action(self.state.mi_contract.collection_entry or REPO_ROOT)))
        mi_layout.addWidget(self._button("打开 MI 状态文件", self._open_path_action(self.state.mi_contract.status_path)))
        layout.addWidget(mi, 0, 0)

        ssvep = Section("SSVEP 采集")
        ssvep_layout = QVBoxLayout(ssvep)
        self.ssvep_dataset_label = QLabel(_short_path(SSVEP_DATASET_DIR))
        self.ssvep_entry_label = QLabel(_short_path(self.state.ssvep_entry))
        ssvep_layout.addWidget(self._kv("默认数据目录", _short_path(SSVEP_DATASET_DIR)))
        ssvep_layout.addWidget(self._kv("采集/启动脚本", self._path_or_text(self.state.ssvep_entry)))
        ssvep_layout.addWidget(self._kv("当前 Profile", _short_path(self.state.ssvep_profile)))
        self.ssvep_collect_bar = QProgressBar()
        self.ssvep_collect_bar.setMaximum(100)
        ssvep_layout.addWidget(QLabel("SSVEP 采集进度"))
        ssvep_layout.addWidget(self.ssvep_collect_bar)
        ssvep_layout.addWidget(self._button("打开 SSVEP 目录", self._open_path_action(SSVEP_DATASET_DIR)))
        ssvep_layout.addWidget(self._button("打开 SSVEP Profile", self._open_path_action(self.state.ssvep_profile.parent)))
        layout.addWidget(ssvep, 0, 1)

        control = Section("流程控制（模拟）")
        control_layout = QHBoxLayout(control)
        start_btn = QPushButton("启动 MI+SSVEP+预训练")
        start_btn.clicked.connect(self.start_workflow)
        pause_btn = QPushButton("暂停/恢复")
        pause_btn.clicked.connect(self.toggle_pause)
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset_workflow)
        self.pipeline_control_btn = start_btn
        control_layout.addWidget(start_btn)
        control_layout.addWidget(pause_btn)
        control_layout.addWidget(reset_btn)
        self._pipeline_control_btn = pause_btn
        layout.addWidget(control, 1, 0, 1, 2)
        return page

    def _build_training_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        section = Section("预训练进度")
        section_layout = QVBoxLayout(section)
        self.train_bar = QProgressBar()
        self.train_bar.setMaximum(100)
        self.train_bar.setValue(0)
        section_layout.addWidget(QLabel("训练阶段"))
        section_layout.addWidget(self.train_bar)
        self.epoch_label = QLabel("epoch: 0 / 0")
        self.loss_label = QLabel("loss: 1.000")
        self.acc_label = QLabel("acc: 0.00")
        metric_grid = QVBoxLayout()
        metric_grid.setSpacing(6)
        metric_grid.addWidget(self.epoch_label)
        metric_grid.addWidget(self.loss_label)
        metric_grid.addWidget(self.acc_label)
        section_layout.addLayout(metric_grid)
        layout.addWidget(section)

        section2 = Section("训练参数与输出")
        section2_layout = QVBoxLayout(section2)
        section2_layout.addWidget(
            self._simple_table(
                ("参数", "当前值"),
                (
                    ("训练轮数", "40"),
                    ("批大小", "128"),
                    ("学习率", "1e-3"),
                    ("可见通道", "C3, Cz, C4, Pz, O1, O2"),
                    ("SSVEP 刺激频点", "9.8 / 12.0 / 14.8 / 15.8 Hz"),
                ),
            )
        )
        layout.addWidget(section2)

        section3 = Section("阶段进度")
        section3_layout = QVBoxLayout(section3)
        self.ready_bar = QProgressBar()
        self.ready_bar.setMaximum(100)
        section3_layout.addWidget(QLabel("流程就绪进度"))
        section3_layout.addWidget(self.ready_bar)
        layout.addWidget(section3)
        layout.addStretch(1)
        return page

    def _build_control_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)

        sources = Section("输入源")
        sources_layout = QVBoxLayout(sources)
        self.control_mode_combo = QComboBox()
        self.control_mode_combo.addItems(["Keyboard", "MI", "SSVEP"])
        self.control_mode_combo.setCurrentText("Keyboard")
        self.control_mode_combo.currentTextChanged.connect(self._set_control_source)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(QLabel("当前控制输入:"))
        row_layout.addWidget(self.control_mode_combo)
        row_layout.addStretch(1)
        sources_layout.addWidget(row)

        sources_layout.addWidget(QLabel("提示：键盘映射"))
        for label in ("W/S/A/D: 平移", "Q/E: 旋转", "Z/X: Z轴", "C/V: 夹爪", "Space: 急停"):
            sources_layout.addWidget(QLabel("• " + label))
        sources_layout.addWidget(self._kv("接管状态", "Keyboard 模拟模式可直接发指令（仅离线仿真）"))
        layout.addWidget(sources, 0, 0)

        state = Section("控制面板")
        state_layout = QVBoxLayout(state)
        self.current_cmd_label = QLabel("最近指令: IDLE")
        state_layout.addWidget(self.current_cmd_label)
        self.control_state_label = QLabel("流程：空闲")
        self.control_state_label.setWordWrap(True)
        state_layout.addWidget(self.control_state_label)
        state_layout.addWidget(QLabel("最近指令日志:"))
        self.command_view = QTextEdit()
        self.command_view.setReadOnly(True)
        self.command_view.setObjectName("commandView")
        self.command_view.setMinimumHeight(180)
        state_layout.addWidget(self.command_view, stretch=1)
        quick = QHBoxLayout()
        quick.addWidget(self._button("快捷：向前", lambda: self._emit_command("MANUAL", "MOVE_FORWARD")))
        quick.addWidget(self._button("快捷：停止", lambda: self._emit_command("MANUAL", "STOP")))
        quick.addWidget(self._button("快捷：抓取", lambda: self._emit_command("MANUAL", "GRIP_CLOSE")))
        quick.addWidget(self._button("快捷：放开", lambda: self._emit_command("MANUAL", "GRIP_OPEN")))
        state_layout.addLayout(quick)
        layout.addWidget(state, 0, 1)

        return page

    def _build_robot_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setSpacing(12)

        left = QVBoxLayout()
        preview = Section("模拟视觉与机械臂展示")
        preview_layout = QVBoxLayout(preview)
        self.robot_preview = RobotPreview()
        preview_layout.addWidget(self.robot_preview, stretch=1)
        left.addWidget(preview, stretch=1)
        layout.addLayout(left, stretch=2)

        right = QVBoxLayout()
        safety = Section("安全与连接")
        safety_layout = QVBoxLayout(safety)
        self.robot_state_label = QLabel("机械臂状态：离线")
        safety_layout.addWidget(self.robot_state_label)
        safety_layout.addWidget(StatusPill("dry-run", "good"))
        safety_layout.addWidget(StatusPill("camera-only", "good"))
        safety_layout.addWidget(StatusPill("manual-control", "warn"))
        safety_layout.addWidget(StatusPill("auto-reach", "warn"))
        safety_layout.addWidget(StatusPill("emergency-stop", "good"))
        btn_row = QHBoxLayout()
        self.connect_button = QPushButton("连接机械臂")
        self.connect_button.clicked.connect(self._toggle_robot_connection)
        btn_row.addWidget(self.connect_button)
        btn_row.addWidget(self._button("急停", lambda: self._emit_command("MANUAL", "EMERGENCY_STOP")))
        safety_layout.addLayout(btn_row)
        right.addWidget(safety)

        info = Section("控制联动说明")
        info_layout = QVBoxLayout(info)
        info_layout.addWidget(QLabel("当前版本默认只开放 Keyboard 控制。"))
        info_layout.addWidget(QLabel("MI/SSVEP 输出接口预留在‘输入源’下拉框，后续替换 source->decoder 即可。"))
        info_layout.addWidget(QLabel("机械臂状态仅影响可视化，不会驱动真实硬件。"))
        right.addWidget(info)
        right.addStretch(1)
        layout.addLayout(right, stretch=1)
        return page

    def _build_copyright_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        section = Section("版权材料清单")
        section_layout = QVBoxLayout(section)
        table = QTableWidget()
        rows = [ArtifactRow(row.item, row.status, row.path, row.owner) for row in self.state.artifact_rows()]
        self.material_table = table
        table.setRowCount(len(rows))
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["物料", "状态", "路径", "归属"])
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate((row.item, row.status, row.path, row.owner)):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                table.setItem(row_index, col_index, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        section_layout.addWidget(table)
        section_layout.addWidget(self._button("打开版权目录", self._open_path_action(REPO_ROOT / "docs" / "softcopyright")))
        section_layout.addWidget(self._button("打开 UI 预览目录", self._open_path_action(REPO_ROOT / "08_SoftCopyright_UI" / "artifacts")))
        section_layout.addWidget(self._button("手动查看里程碑清单", lambda: self._log(
            "建议顺序：先冻结材料清单 -> 形成 source-manifest -> 撰写 version_notes -> 归档到 docs/softcopyright"
        )))
        layout.addWidget(section)

        source = Section("说明")
        source_layout = QVBoxLayout(source)
        source_layout.addWidget(QLabel("本页面承接 V1.0 的软件著作权展示与回顾。"))
        source_layout.addWidget(QLabel("后续可补充：输入源适配器状态、模型元信息、实验日志自动归档。"))
        layout.addWidget(source)
        layout.addStretch(1)
        return page

    def _kv(self, key: str, value: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(key)
        label.setObjectName("kvKey")
        text = QLabel(value)
        text.setObjectName("kvValue")
        text.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(text, stretch=1)
        return row

    def _path_or_text(self, path: Path | None, fallback: str = "未配置") -> str:
        if path is None:
            return fallback
        return _short_path(path)

    def _button(self, label: str, log_text: str | Callable[[], None]) -> QPushButton:
        button = QPushButton(label)
        if callable(log_text):
            button.clicked.connect(lambda _checked=False, action=log_text: action())
        else:
            button.clicked.connect(lambda _checked=False, message=log_text: self._log(message))
        return button

    def _log_action(self, message: str) -> Callable[[], None]:
        return lambda: self._log(message)

    def _show_command(self, entry: Path | None, extra_args: str = "") -> Callable[[], None]:
        def action() -> None:
            if entry is None:
                self._log("未检测到入口路径。")
                return
            suffix = f" {extra_args.strip()}" if extra_args.strip() else ""
            self._log(f"建议执行：python {short_path(entry)}{suffix}")

        return action

    def _locate_or_log(self, path: Path | None, missing_message: str) -> Callable[[], None]:
        def action() -> None:
            if path is None or not Path(path).exists():
                self._log(missing_message)
                return
            self._open_path(path)

        return action

    def _open_path_action(self, path: Path) -> Callable[[], None]:
        return lambda: self._open_path(path)

    def _open_path(self, path: Path) -> None:
        target = Path(path)
        if not target.exists():
            self._log(f"路径不存在：{short_path(target)}")
            return
        try:
            if sys.platform.startswith("win"):
                if target.is_file():
                    subprocess.Popen(["explorer", "/select,", str(target)])
                else:
                    subprocess.Popen(["explorer", str(target)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
            self._log(f"打开路径：{short_path(target)}")
        except OSError as error:
            self._log(f"打开失败：{target} ({error})")

    def _rebuild_pages(self) -> None:
        while self.pages.count():
            widget = self.pages.widget(0)
            self.pages.removeWidget(widget)
            widget.deleteLater()
        for builder in self.page_builders:
            self.pages.addWidget(builder())

    def _rebuild_metric_cards(self) -> list[StatusCard]:
        mi_pct = self._step_percentage(0)
        ssvep_pct = self._step_percentage(1)
        train_pct = self._step_percentage(2)
        ready_pct = self._step_percentage(3)
        overall = self._workflow_overall()

        if self.flow.is_running:
            if self.flow.is_paused:
                flow_state = "PAUSED"
                level = "warn"
            else:
                flow_state = "RUNNING"
                level = "good"
        elif self.flow.ready_for_control:
            flow_state = "READY"
            level = "good"
        else:
            flow_state = "IDLE"
            level = "neutral"

        cards = [
            StatusCard("流程状态", flow_state, f"阶段 {self.flow.step_index}/{len(WORKFLOW_STEPS)}  整体 {overall:.1f}%", level),
            StatusCard("MI 采集", _format_percent(mi_pct), f"阶段标签: MI collection", "good" if mi_pct >= 1.0 else ("warn" if self.flow.step_index >= 1 else "neutral")),
            StatusCard("SSVEP 采集", _format_percent(ssvep_pct), f"阶段标签: SSVEP collection", "good" if ssvep_pct >= 1.0 else ("warn" if self.flow.step_index >= 2 else "neutral")),
            StatusCard("预训练", _format_percent(train_pct), f"acc={self.flow.accuracy:.2f}, loss={self.flow.loss:.2f}", "good" if train_pct >= 1.0 else ("warn" if self.flow.step_index >= 3 else "neutral")),
            StatusCard("控制输入", self.flow.input_source, f"键盘优先，MI/SSVEP 后续接入", "good" if self.flow.input_source == "Keyboard" else "warn"),
            StatusCard("控制就绪", "已连接" if self.flow.robot_connected else "未连接", "ready to run" if self.flow.robot_connected else "请在机械臂页连接", "good" if self.flow.robot_connected else "warn"),
        ]
        cards.extend(self.state.status_cards()[3:])
        return cards[:6]

    def _populate_metrics(self) -> None:
        if self.metric_layout is None:
            return
        while self.metric_layout.count():
            item = self.metric_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, card in enumerate(self._rebuild_metric_cards()):
            self.metric_layout.addWidget(MetricTile(card.name, card.state, card.detail, card.level), index // 3, index % 3)

    def _populate_material_table(self) -> None:
        if self.material_table is None:
            return
        rows = [ArtifactRow(row.item, row.status, row.path, row.owner) for row in self.state.artifact_rows()]
        self.material_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate((row.item, row.status, row.path, row.owner)):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.material_table.setItem(row_index, col_index, item)
        self.material_table.resizeColumnsToContents()
        self.material_table.horizontalHeader().setStretchLastSection(True)

    def _populate_paths(self) -> None:
        if self.mi_dataset_label is not None:
            self.mi_dataset_label.setText(_short_path(MI_DATASET_DIR))
        if self.mi_entry_label is not None:
            self.mi_entry_label.setText(self._path_or_text(self.state.mi_contract.collection_entry))
        if self.ssvep_dataset_label is not None:
            self.ssvep_dataset_label.setText(_short_path(SSVEP_DATASET_DIR))
        if self.ssvep_entry_label is not None:
            self.ssvep_entry_label.setText(self._path_or_text(self.state.ssvep_entry))

    def refresh_state(self) -> None:
        self.state = collect_workbench_state()
        self._populate_material_table()
        self._populate_paths()
        self._update_labels_and_progress()
        self._log("系统状态刷新完成。")

    def _set_control_source(self, source: str) -> None:
        self.flow.input_source = source
        if source != "Keyboard":
            self._log(f"输入源已切为：{source}。当前该源为占位模式，指令暂不输出。")
        else:
            self._log("输入源已切回：Keyboard。")
        self._update_labels_and_progress()

    def _toggle_robot_connection(self) -> None:
        if not self.flow.ready_for_control:
            self._log("流程尚未就绪，请先完成预训练流程。")
            return
        self.flow.robot_connected = not self.flow.robot_connected
        self._log(f"模拟机械臂连接：{'已连接' if self.flow.robot_connected else '已断开'}。")
        self._update_labels_and_progress()

    def start_workflow(self) -> None:
        if self.flow.is_running:
            self._log("流程已在运行。")
            return
        self.flow = WorkflowState()
        self.flow.is_running = True
        self._log("启动流程：MI 采集 -> SSVEP 采集 -> 预训练 -> 流程就绪。")
        self._update_labels_and_progress()

    def toggle_pause(self) -> None:
        if not self.flow.is_running:
            self._log("当前流程未运行。")
            return
        self.flow.is_paused = not self.flow.is_paused
        state = "暂停" if self.flow.is_paused else "恢复"
        self._log(f"流程 {state}。")
        self._update_labels_and_progress()

    def reset_workflow(self) -> None:
        if self.flow.is_running and not self.flow.is_paused:
            self.flow.is_running = False
        self.flow.reset()
        self._log("流程重置。")
        self._update_labels_and_progress()

    def _workflow_overall(self) -> float:
        if not WORKFLOW_STEPS:
            return 0.0
        ticks_done = 0
        total = 0
        for idx, (_, _, max_tick) in enumerate(WORKFLOW_STEPS):
            total += max_tick
            if idx < len(self.flow.step_ticks):
                ticks_done += min(self.flow.step_ticks[idx], max_tick)
        if total == 0:
            return 0.0
        return min(1.0, ticks_done / total)

    def _step_percentage(self, index: int) -> float:
        if index < 0 or index >= len(WORKFLOW_STEPS):
            return 0.0
        _, _, max_tick = WORKFLOW_STEPS[index]
        if max_tick <= 0:
            return 0.0
        return max(0.0, min(1.0, self.flow.step_ticks[index] / max_tick))

    def _current_phase_label(self) -> str:
        if self.flow.is_running and not self.flow.is_paused:
            if self.flow.step_index >= len(WORKFLOW_STEPS):
                return "IDLE"
            return WORKFLOW_STEPS[self.flow.step_index][1]
        if self.flow.is_running and self.flow.is_paused:
            return "PAUSED"
        if self.flow.ready_for_control:
            return "READY"
        return "IDLE"

    def _update_labels_and_progress(self) -> None:
        self._populate_metrics()
        for i, (_, _, total) in enumerate(WORKFLOW_STEPS):
            if i == 0 and self.mi_collect_bar is not None:
                self.mi_collect_bar.setValue(int(self._step_percentage(i) * 100))
            elif i == 1 and self.ssvep_collect_bar is not None:
                self.ssvep_collect_bar.setValue(int(self._step_percentage(i) * 100))
            elif i == 2 and self.train_bar is not None:
                self.train_bar.setValue(int(self._step_percentage(i) * 100))
            elif i == 3 and self.ready_bar is not None:
                self.ready_bar.setValue(int(self._step_percentage(i) * 100))

        if self.epoch_label is not None:
            self.epoch_label.setText(f"epoch: {self.flow.epoch} / {20 + 20}")
        if self.loss_label is not None:
            self.loss_label.setText(f"loss: {self.flow.loss:.3f}")
        if self.acc_label is not None:
            self.acc_label.setText(f"acc: {self.flow.accuracy:.2%}")

        if self.overall_label is not None:
            self.overall_label.setText(f"当前阶段：{self._current_phase_label()}，进度 {self._workflow_overall()*100:.1f}%")

        if self.current_cmd_label is not None:
            self.current_cmd_label.setText(f"最近指令: {self.flow.current_command}")
        if self.control_state_label is not None:
            self.control_state_label.setText(f"流程：{self._current_phase_label()}，输入源：{self.flow.input_source}。")
        if self.robot_state_label is not None:
            status = "在线（模拟）" if self.flow.robot_connected else "离线"
            self.robot_state_label.setText(f"机械臂状态：{status}")
        if self.connect_button is not None:
            self.connect_button.setText("断开机械臂" if self.flow.robot_connected else "连接机械臂")

        if self.pipeline_widget is not None:
            self.pipeline_widget.set_state(
                self.flow.step_index if self.flow.step_index < len(WORKFLOW_STEPS) else len(WORKFLOW_STEPS) - 1,
                [
                    self._step_percentage(0),
                    self._step_percentage(1),
                    self._step_percentage(2),
                    self._step_percentage(3),
                ],
                self._workflow_overall(),
                self._current_phase_label(),
                is_paused=self.flow.is_paused,
            )

        if self.robot_preview is not None:
            self.robot_preview.set_command(self.flow.current_command)

        if self.command_view is not None:
            self.command_view.setText("\n".join(self.flow.command_history))

        if self.control_mode_combo is not None:
            self.control_mode_combo.setCurrentText(self.flow.input_source)

    def _tick_workflow(self) -> None:
        now = time.time()
        if now - self.flow_update_last_tick < 0.2:
            return
        self.flow_update_last_tick = now

        if not self.flow.is_running or self.flow.is_paused:
            return

        if self.flow.step_index >= len(WORKFLOW_STEPS):
            self.flow.is_running = False
            self.flow.ready_for_control = True
            self._log("流程完成：可进入控制环节。")
            self._update_labels_and_progress()
            return

        step_key, step_label, max_ticks = WORKFLOW_STEPS[self.flow.step_index]
        self.flow.step_ticks[self.flow.step_index] += 1
        if self.flow.step_ticks[self.flow.step_index] >= max_ticks:
            self.flow.step_ticks[self.flow.step_index] = max_ticks
            self._log(f"阶段完成：{step_label}。")
            self.flow.step_index += 1
            if self.flow.step_index >= len(WORKFLOW_STEPS):
                self.flow.is_running = False
                self.flow.ready_for_control = True
                self.flow.current_command = "READY"
                self._emit_command("WORKFLOW", "READY")
                self._log("流程已就绪，可连接机械臂并进入控制。")
                self.flow_update_last_tick = now
                self._update_labels_and_progress()
                return

        if step_key == "pretrain":
            ratio = self._step_percentage(2)
            self.flow.epoch = int(40 * ratio)
            self.flow.loss = max(0.06, 1.0 - 0.88 * ratio)
            self.flow.accuracy = 0.52 + 0.43 * ratio
            if ratio == 1.0 and self.flow.is_running:
                self.flow.current_command = "BEST_MODEL_READY"
        elif step_key == "ready":
            self.flow.current_command = "READY"

        self._update_labels_and_progress()

    def _emit_command(self, source: str, command: str) -> None:
        if source == "Keyboard" and self.flow.input_source != "Keyboard":
            return
        now = datetime.now().strftime("%H:%M:%S")
        entry = f"{now} | {source:9} | {command}"
        self.flow.current_command = command
        self.flow.command_history.appendleft(entry)
        self._update_labels_and_progress()
        self._log(f"命令：{command}")

    def _handle_keyboard_command(self, key: int) -> None:
        if self.flow.input_source != "Keyboard":
            self._log("当前输入源非 Keyboard，键盘按键暂不映射。")
            return
        if self.flow.is_running and not self.flow.ready_for_control:
            self._log("流程未完成，先完成 MI/SSVEP 采集与预训练后再下发控制。")
            return
        if not self.flow.robot_connected:
            self._log("机械臂未连接，当前仅记录命令。")

        action = KEYBOARD_CONTROL_MAP.get(key)
        if action is None:
            if key == Qt.Key_Space:
                self._emit_command("Keyboard", "EMERGENCY_STOP")
            return
        verb, direction = action
        self._emit_command("Keyboard", f"{verb}_{direction}")

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if not event.isAutoRepeat():
            key = event.key()
            if key in KEYBOARD_CONTROL_MAP or key == Qt.Key_Space:
                self._handle_keyboard_command(key)
                return
        super().keyPressEvent(event)

    def _log(self, message: str) -> None:
        if self.log_view is not None:
            self.log_view.append(str(message))


def _short_path(path: Path | None) -> str:
    if path is None:
        return "未配置"
    return str(path)


def apply_style(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(
        """
        QMainWindow, QWidget {
            background: #F1F4F7;
            color: #25313D;
        }
        QFrame#header {
            background: #FFFFFF;
            border: 1px solid #CBD3DC;
            border-radius: 6px;
        }
        QLabel#title {
            font: 16pt 'Microsoft YaHei UI';
            color: #17202A;
        }
        QLabel#subtitle {
            color: #52616F;
        }
        QListWidget#nav {
            background: #FFFFFF;
            border: 1px solid #CBD3DC;
            border-radius: 6px;
            padding: 6px;
        }
        QListWidget#nav::item {
            min-height: 38px;
            padding: 7px 10px;
            border-radius: 4px;
            color: #25313D;
        }
        QListWidget#nav::item:selected {
            background: #DCEAF7;
            color: #1E5B8E;
        }
        QFrame#metricTile {
            background: #FFFFFF;
            border: 1px solid #CBD3DC;
            border-radius: 6px;
        }
        QLabel#tileTitle {
            font: 11pt 'Microsoft YaHei UI';
            color: #17202A;
        }
        QLabel#tileDetail {
            color: #52616F;
        }
        QGroupBox#section {
            background: #FFFFFF;
            border: 1px solid #CBD3DC;
            border-radius: 6px;
            margin-top: 18px;
            padding: 10px;
        }
        QGroupBox#section::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #17202A;
            font: 11pt 'Microsoft YaHei UI';
        }
        QLabel#kvKey {
            min-width: 118px;
            color: #52616F;
        }
        QLabel#kvValue {
            color: #25313D;
        }
        QPushButton {
            background: #FFFFFF;
            border: 1px solid #8EA4B8;
            border-radius: 4px;
            padding: 6px 10px;
            min-height: 24px;
        }
        QPushButton:hover {
            background: #E8F1FA;
        }
        QComboBox {
            background: #FFFFFF;
            border: 1px solid #8EA4B8;
            border-radius: 4px;
            padding: 5px 8px;
            min-width: 132px;
        }
        QTableWidget {
            background: #FFFFFF;
            alternate-background-color: #F7F9FB;
            gridline-color: #D8E0E8;
            border: 1px solid #CBD3DC;
        }
        QHeaderView::section {
            background: #E9EEF3;
            color: #25313D;
            border: 1px solid #D8E0E8;
            padding: 5px;
        }
        QTextEdit#logView {
            background: #18202A;
            color: #E7EDF5;
            border: 1px solid #2F3B48;
            border-radius: 6px;
            font: 9pt 'Consolas';
        }
        QTextEdit#commandView {
            background: #20262D;
            color: #EAF0F7;
            border: 1px solid #3A4759;
            border-radius: 6px;
            font: 9pt 'Consolas';
        }
        QProgressBar {
            border: 1px solid #B8C6D4;
            border-radius: 5px;
            text-align: center;
            color: #24303C;
            background: #F4F7FA;
            height: 16px;
        }
        QProgressBar::chunk {
            background-color: #4F7DB8;
            border-radius: 5px;
        }
        """
    )


def save_window_screenshot(window: WorkbenchWindow, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    QApplication.processEvents()
    pixmap = QPixmap(window.size())
    window.render(pixmap)
    if not pixmap.save(str(output_path)):
        raise RuntimeError(f"failed to save screenshot: {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the soft-copyright workbench workflow UI.")
    parser.add_argument("--screenshot", type=Path, help="save a screenshot and exit")
    args = parser.parse_args(list(argv or sys.argv[1:]))

    if args.screenshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication.instance() or QApplication([])
    apply_style(app)
    window = WorkbenchWindow()
    window.show()

    if args.screenshot:
        def _capture() -> None:
            save_window_screenshot(window, args.screenshot)
            app.quit()

        QTimer.singleShot(800, _capture)
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
