from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

try:
    from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
    from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
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
        QSizePolicy,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as error:  # pragma: no cover - import guard for machines without GUI deps
    raise RuntimeError("PyQt5 is required to run the soft-copyright workbench UI") from error


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brain_workspace.paths import (  # noqa: E402
    HYBRID_CONTROLLER_DIR,
    MI_DATASET_DIR,
    SSVEP_DATASET_DIR,
)
from .state import ArtifactRow, collect_workbench_state, short_path  # noqa: E402


APP_TITLE = "基于混合脑机接口的智能机械臂协同控制软件 V1.0"


def _short(path: Path) -> str:
    return short_path(Path(path))


def _maybe_short(path: Path | None) -> str:
    return short_path(path) if path is not None else "待接入"


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
        ("采集", "#4F7DB8"),
        ("训练", "#4E8A62"),
        ("发布", "#8C6D31"),
        ("识别", "#7A5CA8"),
        ("视觉", "#3F7F87"),
        ("执行", "#9A4F4F"),
        ("报告", "#59636F"),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#F7F9FB"))
        margin_x = 28
        center_y = self.height() // 2
        usable_w = max(1, self.width() - margin_x * 2)
        gap = usable_w / max(1, len(self.STEPS) - 1)
        painter.setPen(QPen(QColor("#CBD3DC"), 3))
        painter.drawLine(margin_x, center_y, self.width() - margin_x, center_y)
        painter.setFont(QFont("Microsoft YaHei UI", 10))
        for index, (label, color) in enumerate(self.STEPS):
            x = margin_x + index * gap
            radius = 21
            painter.setBrush(QColor(color))
            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            painter.drawEllipse(QPointF(float(x), float(center_y)), radius, radius)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(QRectF(x - radius, center_y - radius, radius * 2, radius * 2), Qt.AlignCenter, str(index + 1))
            painter.setPen(QColor("#25313D"))
            painter.drawText(QRectF(x - 36, center_y + 28, 72, 24), Qt.AlignCenter, label)


class VisionRobotPreview(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#20252C"))
        frame = self.rect().adjusted(18, 18, -18, -18)
        painter.setPen(QPen(QColor("#6E7B8A"), 2))
        painter.drawRect(frame)
        painter.setPen(QColor("#D7DEE8"))
        painter.setFont(QFont("Consolas", 10))
        painter.drawText(frame.adjusted(12, 10, -12, -10), Qt.AlignLeft | Qt.AlignTop, "official MJPEG view | demo frame")

        targets = [
            (0.25, 0.38, 0.16, 0.20, "1", "#69A86E"),
            (0.56, 0.34, 0.18, 0.24, "2", "#D19A45"),
            (0.42, 0.62, 0.14, 0.17, "3", "#5E95C8"),
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
            painter.drawText(rect.adjusted(5, 3, -5, -3), Qt.AlignLeft | Qt.AlignTop, f"slot {label}")

        path = QPainterPath()
        base_x = frame.left() + frame.width() * 0.82
        base_y = frame.top() + frame.height() * 0.80
        painter.setPen(QPen(QColor("#F0F3F7"), 4))
        path.moveTo(base_x, base_y)
        path.cubicTo(base_x - 30, base_y - 60, base_x - 72, base_y - 85, base_x - 120, base_y - 122)
        painter.drawPath(path)
        painter.setBrush(QColor("#F0F3F7"))
        painter.drawEllipse(QPointF(base_x, base_y), 7, 7)


class Section(QGroupBox):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.setObjectName("section")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)


class WorkbenchWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.state = collect_workbench_state()
        self.metric_layout: QGridLayout | None = None
        self.material_table: QTableWidget | None = None
        self.setWindowTitle(APP_TITLE)
        self.resize(1440, 900)
        self.setMinimumSize(1180, 760)

        self.page_builders = (
            self._build_overview_page,
            self._build_acquisition_page,
            self._build_training_page,
            self._build_online_page,
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
        for title in ("总览", "数据采集", "训练评估", "在线控制", "视觉机械臂", "软著材料"):
            item = QListWidgetItem(title)
            item.setSizeHint(item.sizeHint())
            self.nav.addItem(item)
        body.addWidget(self.nav)

        self.pages = QStackedWidget()
        body.addWidget(self.pages, stretch=1)
        self._rebuild_pages()

        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)

        self.log_view = QTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(92)
        shell.addWidget(self.log_view)
        self._log("Workbench loaded in software-copyright demo mode.")
        self._log(f"Repository root: {REPO_ROOT}")

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)

        title_box = QVBoxLayout()
        title = QLabel(APP_TITLE)
        title.setObjectName("title")
        subtitle = QLabel("软著 V1.0 UI 原型 | 当前目标：把 MI、SSVEP、视觉抓取和机械臂控制整理成可演示闭环")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box, stretch=1)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["软著演示模式", "研发调试模式", "真机联调模式"])
        layout.addWidget(self.mode_combo)

        refresh = QPushButton("刷新状态")
        refresh.clicked.connect(self.refresh_state)
        layout.addWidget(refresh)

        export = QPushButton("导出截图")
        export.clicked.connect(lambda: self._log("Use --screenshot to export a reproducible UI screenshot."))
        layout.addWidget(export)
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

        section = Section("V1.0 闭环流程")
        section_layout = QVBoxLayout(section)
        section_layout.addWidget(PipelineWidget())
        layout.addWidget(section)

        checklist = Section("当前最重要的工程任务")
        checklist_layout = QVBoxLayout(checklist)
        for text in (
            "把新版 MI 分类器并入 01_MI 或独立包，提供训练、评估、实时推理和 profile 输出。",
            "把 MI/SSVEP/键盘三种输入源在主状态机里统一成 command event。",
            "保持硬件无关演示模式，软著截图和用户手册不依赖真实脑电板或 JetMax 在线。",
            "冻结 softcopyright-v1.0 源码边界，排除数据、日志、缓存、论文 PDF 和临时输出。",
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
        mi_layout.addWidget(self._kv("默认数据目录", _short(MI_DATASET_DIR)))
        mi_layout.addWidget(self._kv("固定通道", "C3, Cz, C4, PO3, PO4, O1, Oz, O2"))
        mi_layout.addWidget(self._kv("V1.0 目标", "采集 -> 训练 -> 实时推理 -> profile 发布"))
        mi_layout.addWidget(self._kv("训练入口", _maybe_short(self.state.mi_contract.train_entry)))
        mi_layout.addWidget(self._kv("实时入口", _maybe_short(self.state.mi_contract.realtime_entry)))
        mi_layout.addWidget(self._kv("默认 profile", _short(self.state.mi_contract.profile_path)))
        mi_layout.addWidget(self._button("定位 MI 采集入口", self._locate_or_log(self.state.mi_contract.collection_entry, "MI 采集入口尚未接入。")))
        mi_layout.addWidget(self._button("显示 MI 训练命令", self._show_command(self.state.mi_contract.train_entry, "--help")))
        mi_layout.addWidget(self._button("打开 MI profile 目录", self._open_path_action(self.state.mi_contract.profile_path.parent)))
        layout.addWidget(mi, 0, 0)

        ssvep = Section("SSVEP 采集")
        ssvep_layout = QVBoxLayout(ssvep)
        ssvep_layout.addWidget(self._kv("默认数据目录", _short(SSVEP_DATASET_DIR)))
        ssvep_layout.addWidget(self._kv("目标频率", "9.8 / 12.0 / 14.8 / 15.8 Hz"))
        ssvep_layout.addWidget(self._kv("协议重点", "短预训练、idle/no-control、manifest 可追溯"))
        ssvep_layout.addWidget(self._kv("Launcher", _short(self.state.ssvep_entry)))
        ssvep_layout.addWidget(self._kv("Profile", _short(self.state.ssvep_profile)))
        ssvep_layout.addWidget(self._button("定位 SSVEP launcher", self._locate_or_log(self.state.ssvep_entry, "SSVEP launcher 缺失。")))
        ssvep_layout.addWidget(self._button("打开 SSVEP profile 目录", self._open_path_action(self.state.ssvep_profile.parent)))
        layout.addWidget(ssvep, 0, 1)

        shared = Section("采集保存与质量控制")
        shared_layout = QVBoxLayout(shared)
        rows = [
            ("真源数据", "board_data / raw_trials / events"),
            ("语义数据", "trials / segments / labels"),
            ("派生数据", "MI epochs / gate epochs / SSVEP features"),
            ("质量门槛", "串口有效、通道一致、采样率一致、保存完整"),
        ]
        shared_layout.addWidget(self._simple_table(("层级", "内容"), rows))
        layout.addWidget(shared, 1, 0, 1, 2)
        return page

    def _build_training_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        section = Section("训练与评估主线")
        section_layout = QVBoxLayout(section)
        rows = [
            ("MI classifier", self.state.mi_contract.state, "accuracy / kappa / gate FP / realtime latency", _short(self.state.mi_contract.profile_path)),
            ("SSVEP FBCCA/TDCA", "profile 可用" if self.state.ssvep_profile_exists else "待发布", "idle_fp_per_min / control_recall / release latency", _short(self.state.ssvep_profile)),
            ("Vision YOLO/servo", "profile 可用" if self.state.vision_profile_exists else "待补齐", "target stability / grasp quality / reject reasons", _short(self.state.vision_profile)),
        ]
        section_layout.addWidget(self._simple_table(("模型", "状态", "验收指标", "发布产物"), rows))
        action_row = QHBoxLayout()
        action_row.addWidget(self._button("显示 MI 训练命令", self._show_command(self.state.mi_contract.train_entry)))
        action_row.addWidget(self._button("显示 MI 实时命令", self._show_command(self.state.mi_contract.realtime_entry)))
        action_row.addWidget(self._button("打开 profile 根目录", self._open_path_action(self.state.profile_root)))
        section_layout.addLayout(action_row)
        layout.addWidget(section)

        gates = Section("发布 gate")
        gates_layout = QGridLayout(gates)
        gate_rows = [
            ("训练数据可追溯", "必须"),
            ("测试报告生成", "必须"),
            ("硬件无关 smoke", "必须"),
            ("profile schema 校验", "必须"),
            ("真实硬件验证", "推荐"),
            ("软著材料同步", "必须"),
        ]
        for index, (name, state) in enumerate(gate_rows):
            gates_layout.addWidget(QLabel(name), index, 0)
            gates_layout.addWidget(StatusPill(state, "good" if state == "必须" else "warn"), index, 1)
        layout.addWidget(gates)
        layout.addStretch(1)
        return page

    def _build_online_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)

        sources = Section("输入源仲裁")
        sources_layout = QVBoxLayout(sources)
        rows = [
            ("Keyboard", "调试/兜底", "方向、选择、确认、取消"),
            ("MI", "连续意图", "移动方向、运动/静息 gate"),
            ("SSVEP", "离散命令", "目标选择、确认、释放"),
        ]
        sources_layout.addWidget(self._simple_table(("输入源", "角色", "输出"), rows))
        layout.addWidget(sources, 0, 0)

        indicators = Section("实时指标")
        ind_layout = QGridLayout(indicators)
        for index, (label, value, level) in enumerate(
            (
                ("MI confidence", "--", "neutral"),
                ("SSVEP margin", "--", "neutral"),
                ("selected command", "idle", "warn"),
                ("release timer", "--", "neutral"),
                ("switch latency", "--", "neutral"),
                ("false-positive guard", "armed", "good"),
            )
        ):
            ind_layout.addWidget(QLabel(label), index, 0)
            ind_layout.addWidget(StatusPill(value, level), index, 1)
        layout.addWidget(indicators, 0, 1)

        decision = Section("状态机输出")
        decision_layout = QVBoxLayout(decision)
        decision_layout.addWidget(QLabel("IDLE -> S1_MI_MOVE -> S1_DECISION -> S2_TARGET_SELECT -> S2_PICKING -> S3_PLACING"))
        decision_layout.addWidget(QLabel("V1.0 规则：键盘为调试兜底，MI 负责连续移动意图，SSVEP 负责离散选择/确认/释放。"))
        decision_layout.addWidget(QLabel("低置信度统一进入 idle/no-control；任何抓取动作必须通过视觉、机器人、profile 三层安全 gate。"))
        layout.addWidget(decision, 1, 0, 1, 2)
        return page

    def _build_robot_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setSpacing(12)

        left = QVBoxLayout()
        preview = Section("视觉目标与抓取预览")
        preview_layout = QVBoxLayout(preview)
        preview_layout.addWidget(VisionRobotPreview())
        left.addWidget(preview, stretch=1)
        layout.addLayout(left, stretch=2)

        right = QVBoxLayout()
        safety = Section("安全阶梯")
        safety_layout = QVBoxLayout(safety)
        for label, level in (
            ("dry-run", "good"),
            ("camera-only", "good"),
            ("resolve-only", "good"),
            ("move-only", "warn"),
            ("execute-move", "warn"),
            ("allow-pick", "bad"),
        ):
            safety_layout.addWidget(StatusPill(label, level))
        right.addWidget(safety)

        robot = Section("机器人状态")
        robot_layout = QVBoxLayout(robot)
        robot_layout.addWidget(self._kv("入口", _short(self.state.hybrid_entry)))
        robot_layout.addWidget(self._kv("配置", _short(self.state.hybrid_config)))
        robot_layout.addWidget(self._kv("视觉模型", _short(self.state.vision_model)))
        robot_layout.addWidget(self._kv("抓取 profile", _short(self.state.vision_profile)))
        robot_layout.addWidget(self._button("定位 hybrid_controller", self._open_path_action(HYBRID_CONTROLLER_DIR)))
        robot_layout.addWidget(self._button("显示只读启动命令", self._show_command(self.state.hybrid_entry, "--smoke-test-ms 3000")))
        right.addWidget(robot)
        right.addStretch(1)
        layout.addLayout(right, stretch=1)
        return page

    def _build_copyright_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        section = Section("软著材料清单")
        section_layout = QVBoxLayout(section)
        table = QTableWidget()
        rows = [ArtifactRow(row.item, row.status, row.path, row.owner) for row in self.state.artifact_rows()]
        self.material_table = table
        table.setRowCount(len(rows))
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["材料", "状态", "路径/说明", "负责人"])
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate((row.item, row.status, row.path, row.owner)):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                table.setItem(row_index, col_index, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        section_layout.addWidget(table)
        action_row = QHBoxLayout()
        action_row.addWidget(self._button("打开材料目录", self._open_path_action(REPO_ROOT / "docs" / "softcopyright")))
        action_row.addWidget(self._button("打开 UI 截图目录", self._open_path_action(REPO_ROOT / "08_SoftCopyright_UI" / "artifacts")))
        action_row.addWidget(self._button("显示冻结命令", self._log_action("冻结前建议命令：git tag softcopyright-v1.0 <commit>；先确认测试报告和源码交存清单已更新。")))
        section_layout.addLayout(action_row)
        layout.addWidget(section)

        source = Section("源码交存边界")
        source_layout = QVBoxLayout(source)
        source_layout.addWidget(QLabel("纳入：自有源码、配置、UI、profile schema、测试和必要说明文档。"))
        source_layout.addWidget(QLabel("排除：真实 EEG 数据、外部数据集、论文 PDF、训练输出、日志、缓存、临时图片。"))
        source_layout.addWidget(QLabel("冻结：完成 MI 接入和 UI 演示后创建 softcopyright-v1.0 分支或 tag。"))
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
                self._log("入口尚未接入，无法生成命令。")
                return
            suffix = f" {extra_args.strip()}" if extra_args.strip() else ""
            self._log(f"只读命令预览：python {short_path(entry)}{suffix}")

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
            self._log(f"路径不存在，未创建任何文件或目录：{short_path(target)}")
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
            self._log(f"已定位：{short_path(target)}")
        except OSError as error:
            self._log(f"定位失败：{target} ({error})")

    def _populate_metrics(self) -> None:
        if self.metric_layout is None:
            return
        while self.metric_layout.count():
            item = self.metric_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, card in enumerate(self.state.status_cards()):
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

    def _rebuild_pages(self) -> None:
        self.metric_layout = None
        self.material_table = None
        while self.pages.count():
            widget = self.pages.widget(0)
            self.pages.removeWidget(widget)
            widget.deleteLater()
        for builder in self.page_builders:
            self.pages.addWidget(builder())

    def refresh_state(self) -> None:
        current_index = max(0, self.pages.currentIndex())
        self.state = collect_workbench_state()
        self._rebuild_pages()
        self.pages.setCurrentIndex(min(current_index, self.pages.count() - 1))
        self._log("状态已刷新：已重建全部页面；UI 仅重新读取路径、profile、schema 和材料草稿，不启动硬件流程。")

    def _simple_table(self, headers: tuple[str, ...], rows: Sequence[Sequence[str]]) -> QTableWidget:
        table = QTableWidget()
        table.setRowCount(len(rows))
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(list(headers))
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                table.setItem(row_index, col_index, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        table.setMinimumHeight(120)
        return table

    def _log(self, message: str) -> None:
        self.log_view.append(str(message))


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
            font: 16pt "Microsoft YaHei UI";
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
            font: 11pt "Microsoft YaHei UI";
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
            font: 11pt "Microsoft YaHei UI";
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
            font: 9pt "Consolas";
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
    parser = argparse.ArgumentParser(description="Run the soft-copyright workbench UI prototype.")
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
