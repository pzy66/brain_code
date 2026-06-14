# -*- coding: utf-8 -*-
"""
脑机接口混合控制系统 - 全流程总集成版（完美恢复 MI 原厂手、脚、舌头矢量图渲染）
"""
from __future__ import annotations
import sys
import time
import random
import math
from collections import deque
import numpy as np

from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal, pyqtSlot, QPointF
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush, QLinearGradient
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QLabel, QFrame, QProgressBar,
    QTextEdit, QGridLayout, QSpinBox, QFormLayout
)

# ==========================================
# 统一科技暗黑风格全局 QSS 样式表
# ==========================================
CYBERPUNK_STYLE = """
QMainWindow {
    background-color: #0B0E13;
}
QWidget {
    color: #D2DAE5;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}
QFrame#navBar {
    background-color: #11161D;
    border-right: 1px solid #1F2631;
}
QLabel#navTitle {
    color: #7DEBC0;
    font-size: 16px;
    font-weight: bold;
    padding: 15px 10px;
}
QPushButton#navBtn {
    background-color: transparent;
    border: none;
    color: #52616F;
    text-align: left;
    padding: 12px 20px;
    font-size: 14px;
    border-left: 3px solid transparent;
}
QPushButton#navBtn:hover:enabled {
    background-color: #1A222D;
    color: #FFFFFF;
}
QPushButton#navBtn[active="true"] {
    background-color: #151B24;
    color: #A9F5D0;
    border-left: 3px solid #67E8B9;
    font-weight: bold;
}
QPushButton#navBtn:disabled { color: #2F3B4C; }
QFrame#cardPanel, QFrame#stateBar, QFrame#cameraCard, QFrame#poseCard, QFrame#flowCard {
    background-color: #151B24;
    border: 1px solid #252E3C;
    border-radius: 6px;
}
QLabel#panelTitle, QLabel#stateBarTitle, QLabel#controlTitle {
    color: #A9F5D0;
    font-size: 14px;
    font-weight: bold;
    border-bottom: 1px solid #252E3C;
    padding-bottom: 6px;
}
QPushButton#actionBtn {
    background-color: #202B38;
    border: 1px solid #3F4F64;
    color: #D2DAE5;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}
QPushButton#actionBtn:hover:enabled {
    background-color: #2A394A;
    border-color: #569AFF;
    color: #FFFFFF;
}
QPushButton#accentBtn {
    background-color: #14251F;
    border: 1px solid #2C5A47;
    color: #67E8B9;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}
QPushButton#accentBtn:hover:enabled {
    background-color: #1B352B;
    border-color: #4EBA93;
    color: #A9F5D0;
}
QTextEdit#logger {
    background-color: #0D1117;
    border: 1px solid #1F2631;
    border-radius: 4px;
    color: #A4B1CD;
    font-family: "Consolas", monospace;
    font-size: 12px;
}
QProgressBar {
    border: 1px solid #252E3C;
    background-color: #0D1117;
    text-align: center;
    color: #FFFFFF;
    font-weight: bold;
    border-radius: 4px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2C5A47, stop:1 #67E8B9);
    border-radius: 3px;
}
QFrame#stateNode, QFrame#poseRow {
    background: #10161F;
    border: 1px solid #293446;
    border-radius: 6px;
}
QLabel#controlPill {
    color: #0D1117;
    background: #A9F5D0;
    border-radius: 4px;
    padding: 4px 8px;
    font-weight: bold;
    font-size: 11px;
}
"""

class MIVisualCueWidget(QWidget):
    """完美恢复原厂 MI 采集画布：包含实心抗锯齿手掌、双脚趾、贝塞尔面肌舌头，以及双行全中文文字提示"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 320)
        self.state = "IDLE"  # "IDLE", "READY", "CUE", "REST"
        self.direction = "LEFT"  # "LEFT", "RIGHT", "FEET", "TONGUE"

    def update_cue(self, state: str, direction: str = "LEFT"):
        self.state = state
        self.direction = direction
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        # 核心：必须开启极致抗锯齿和高精度像素平滑，否则手掌和舌头边缘会有毛刺
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing, True)

        # 1. 铺设科技暗黑深邃底色
        painter.fillRect(self.rect(), QColor("#090D14"))

        cx, cy = self.width() / 2.0, self.height() / 2.0
        # 留出四周安全间距，给底部文字留出专门的空间
        rect = QRectF(self.rect()).adjusted(40, 30, -40, -70)

        # 配色规范：CUE阶段激活荧光绿，READY阶段使用科技蓝
        accent_color = QColor("#67E8B9") if self.state == "CUE" else QColor("#569AFF")

        # 2. 绘制图形层
        if self.state == "READY":
            # 原厂经典大十字注视点
            painter.setPen(QPen(accent_color, 6, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(cx - 30, cy - 20), QPointF(cx + 30, cy - 20))
            painter.drawLine(QPointF(cx, cy - 50), QPointF(cx, cy + 10))

        elif self.state == "CUE":
            if self.direction == "LEFT":
                self._draw_factory_hand(painter, rect, accent_color, mirrored=False)
            elif self.direction == "RIGHT":
                self._draw_factory_hand(painter, rect, accent_color, mirrored=True)
            elif self.direction == "FEET":
                self._draw_factory_feet(painter, rect, accent_color)
            elif self.direction == "TONGUE":
                self._draw_factory_tongue(painter, rect, accent_color)

        elif self.state == "REST":
            # 绘制原厂大柔和注视圆
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor("#475569")))
            painter.drawEllipse(QPointF(cx, cy - 20), 16, 16)

        # 3. 完美保留并常驻原厂全中文双行文字提示
        self._draw_factory_text_hints(painter)

    # ----- 核心改动：采用实心填充（Brush）与更优雅的几何解算，提升渲染精细度 -----
    def _draw_factory_hand(self, painter, rect, color, mirrored: bool):
        w, h = rect.width(), rect.height()
        # 精确确立手掌丰满度比例
        palm_w, palm_h = w * 0.24, h * 0.32
        palm_x = rect.left() + w * 0.38
        palm_y = rect.top() + h * 0.35

        if mirrored:
            palm_x = rect.right() - palm_w - (palm_x - rect.left())

        palm_rect = QRectF(palm_x, palm_y, palm_w, palm_h)

        # 使用厚实饱满的半透明色或实色填充手掌
        painter.setPen(QPen(color, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))  # 内部富有科技感微光填充
        painter.drawRoundedRect(palm_rect, 14, 14)

        # 渲染五根饱满的手指
        spacing = palm_w / 5.0
        painter.setBrush(QBrush(color))  # 手指使用全实心高亮填充，视觉效果极佳
        for f_idx in range(5):
            fx = palm_rect.left() + spacing * (f_idx + 0.5)
            fy_top = palm_rect.top() - h * 0.16 if f_idx != 0 and f_idx != 4 else palm_rect.top() - h * 0.11  # 中指长，两边短
            painter.drawRoundedRect(QRectF(fx - 4, fy_top, 8, palm_rect.top() - fy_top + 4), 4, 4)

        # 运用三次贝塞尔曲线让大拇指的外展肌肉群变饱满
        thumb_path = QPainterPath()
        if mirrored:
            thumb_path.moveTo(palm_rect.right() - 4, palm_rect.top() + palm_h * 0.4)
            thumb_path.cubicTo(palm_rect.right() + w * 0.12, palm_rect.top() + palm_h * 0.5,
                               palm_rect.right() + w * 0.08, palm_rect.top() + palm_h * 0.8,
                               palm_rect.right() - 2, palm_rect.top() + palm_h * 0.85)
        else:
            thumb_path.moveTo(palm_rect.left() + 4, palm_rect.top() + palm_h * 0.4)
            thumb_path.cubicTo(palm_rect.left() - w * 0.12, palm_rect.top() + palm_h * 0.5,
                               palm_rect.left() - w * 0.08, palm_rect.top() + palm_h * 0.8,
                               palm_rect.left() + 2, palm_rect.top() + palm_h * 0.85)
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
        painter.drawPath(thumb_path)

    def _draw_factory_feet(self, painter, rect, color):
        w, h = rect.width(), rect.height()
        painter.setPen(QPen(color, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))

        # 左右两边厚实足廓矩阵配置
        foot_w, foot_h = w * 0.20, h * 0.45
        left_box = QRectF(rect.left() + w * 0.24, rect.top() + h * 0.25, foot_w, foot_h)
        right_box = QRectF(rect.left() + w * 0.56, rect.top() + h * 0.25, foot_w, foot_h)

        for foot in (left_box, right_box):
            painter.drawRoundedRect(foot, 18, 18)
            # 五个实心饱满小脚趾排布
            painter.setBrush(QBrush(color))
            r_toe = foot.width() * 0.09
            for t_idx in range(5):
                tx = foot.left() + foot.width() * (0.16 + t_idx * 0.17)
                ty = foot.top() - r_toe * 0.6
                current_r = r_toe * (1.2 if t_idx == 0 else 0.9)  # 大脚趾画大一点
                painter.drawEllipse(QPointF(tx, ty), current_r, current_r)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))

    def _draw_factory_tongue(self, painter, rect, color):
        w, h = rect.width(), rect.height()
        painter.setPen(QPen(color, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 25)))

        # 下颌轮廓大曲线填充
        face = QPainterPath()
        face.moveTo(rect.left() + w * 0.32, rect.top() + h * 0.25)
        face.cubicTo(rect.left() + w * 0.55, rect.top() + h * 0.12,
                     rect.left() + w * 0.72, rect.top() + h * 0.28,
                     rect.left() + w * 0.66, rect.top() + h * 0.52)
        face.cubicTo(rect.left() + w * 0.60, rect.top() + h * 0.72,
                     rect.left() + w * 0.40, rect.top() + h * 0.75,
                     rect.left() + w * 0.32, rect.top() + h * 0.52)
        painter.drawPath(face)

        # 中线舌骨结构
        painter.setPen(QPen(color, 3, Qt.SolidLine))
        painter.drawLine(QPointF(rect.left() + w * 0.48, rect.top() + h * 0.46),
                         QPointF(rect.left() + w * 0.62, rect.top() + h * 0.48))

        # 吐出的核心深红色/高亮舌苔肉质块
        tongue = QPainterPath(QPointF(rect.left() + w * 0.62, rect.top() + h * 0.48))
        tongue.cubicTo(rect.left() + w * 0.76, rect.top() + h * 0.50,
                       rect.left() + w * 0.76, rect.top() + h * 0.64,
                       rect.left() + w * 0.64, rect.top() + h * 0.64)
        painter.setPen(QPen(color, 4))
        painter.setBrush(QBrush(QColor(239, 68, 68, 180)))  # 还原原厂明艳红暗示色
        painter.drawPath(tongue)

    def _draw_factory_text_hints(self, painter):
        """完美找回并固化原厂全中文文字解析提示堆栈"""
        bottom_y = self.height() - 55

        # 数据字典映射
        title_map = {"LEFT": "左手握拳", "RIGHT": "右手握拳", "FEET": "双脚运动", "TONGUE": "舌头伸缩"}
        desc_map = {
            "LEFT": "持续想象左手握拳与放松动作，不要真实移动手臂。",
            "RIGHT": "持续想象右手握拳与放松动作，不要真实移动手臂。",
            "FEET": "持续想象双脚交替踩踏动作，不要真实移动腿部。",
            "TONGUE": "持续想象舌头前伸/上抬动作，不要真实张口吐舌。"
        }

        if self.state == "READY":
            main_title, sub_desc = "准备阶段", "请注视中央十字，保持全身放松并集中注意力"
            color_title = QColor("#569AFF")
        elif self.state == "CUE":
            main_title = f"想象任务：{title_map.get(self.direction, '')}"
            sub_desc = desc_map.get(self.direction, "")
            color_title = QColor("#67E8B9")
        elif self.state == "REST":
            main_title, sub_desc = "休息恢复", "放空当前想象，尽量减少眨眼与吞咽次数"
            color_title = QColor("#94A3B8")
        else:
            main_title, sub_desc = "操作系统就绪", "等待主控流水线状态机触发下发实验范式"
            color_title = QColor("#445161")

        # 第一行：大标题标签
        painter.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        painter.setPen(QPen(color_title))
        painter.drawText(QRectF(10, bottom_y, self.width() - 20, 26), Qt.AlignCenter, main_title)

        # 第二行：详细行为行为引导说明小字
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.setPen(QPen(QColor("#8B97A5")))
        painter.drawText(QRectF(10, bottom_y + 26, self.width() - 20, 20), Qt.AlignCenter, sub_desc)


class ParticipantDisplayWindow(QWidget):
    """提取自原厂源码：全屏置顶的受试者专用视觉提示窗（包含原厂按键互锁与全状态视觉投影）"""
    pause_requested = pyqtSignal()
    mark_bad_requested = pyqtSignal()
    advance_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self) -> None:
        # 设置无边框、窗口置顶属性，隔绝操作员界面的光线与视觉干扰
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("受试者提示屏")
        self.setStyleSheet("background: #04111C;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 36, 40, 36)
        layout.setSpacing(18)

        self.stage_label = QLabel("等待开始")
        self.stage_label.setAlignment(Qt.AlignCenter)
        self.stage_label.setStyleSheet(
            "color: white; font-size: 28px; font-weight: bold; background: #475569; border-radius: 18px; padding: 12px 20px;")
        layout.addWidget(self.stage_label)

        # 挂载精修的手脚舌头实心抗锯齿矢量画布
        self.cue_widget = MIVisualCueWidget()
        layout.addWidget(self.cue_widget, stretch=1)

        self.countdown_label = QLabel("--")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet(
            "color: #F8FAFC; font-size: 56px; font-weight: bold; background: rgba(15, 23, 42, 170); border: 2px solid #334155; border-radius: 22px; padding: 10px 20px;")
        layout.addWidget(self.countdown_label)

        self.hint_label = QLabel("快捷键：[Space] 暂停/继续  |  [B] 标记坏试次  |  [Esc] 停止并保存")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setStyleSheet("color: #64748B; font-size: 14px;")
        layout.addWidget(self.hint_label)

    def set_prompt(self, phase: str, direction: str, stage_text: str, countdown_text: str):
        """同步操作员状态机下发的时序动作参数"""
        self.stage_label.setText(stage_text)
        self.cue_widget.update_cue(phase, direction)
        self.countdown_label.setText(countdown_text)

        # 根据实验范式实时动态变换受试者屏的任务标头颜色
        if phase == "READY":
            self.stage_label.setStyleSheet(
                "color: white; font-size: 28px; font-weight: bold; background: #569AFF; border-radius: 18px; padding: 12px 20px;")
        elif phase == "CUE":
            self.stage_label.setStyleSheet(
                "color: white; font-size: 28px; font-weight: bold; background: #67E8B9; border-radius: 18px; padding: 12px 20px;")
        elif phase == "REST":
            self.stage_label.setStyleSheet(
                "color: white; font-size: 28px; font-weight: bold; background: #475569; border-radius: 18px; padding: 12px 20px;")

    def keyPressEvent(self, event) -> None:
        """接管受试者键盘响应，就地映射并投递至主框架网络"""
        if event.key() == Qt.Key_Space:
            self.pause_requested.emit()
            event.accept()
            return
        elif event.key() == Qt.Key_B:
            self.mark_bad_requested.emit()
            event.accept()
            return
        elif event.key() == Qt.Key_Escape:
            self.stop_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

# 保留组件二【源自MI_UI源码】：原厂高清实时波形示波器
# ==========================================
class RealtimeEEGPreviewWidget(QWidget):
    MODE_EEG = "EEG"
    MODE_IMPEDANCE = "IMP"
    CHANNEL_COLORS = ["#38BDF8", "#22C55E", "#F59E0B", "#A855F7", "#EF4444", "#14B8A6", "#EAB308", "#F97316"]

    def __init__(self, parent=None, window_seconds: float = 5.0):
        super().__init__(parent)
        self.window_seconds = window_seconds
        self.sampling_rate = 250.0
        self.channel_names = ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"]
        self.max_points = int(self.window_seconds * self.sampling_rate)
        self.buffers = [deque(maxlen=self.max_points) for _ in self.channel_names]
        self.mode = self.MODE_EEG
        self.last_impedance_ohms = [random.uniform(4000, 12000) for _ in self.channel_names]

        # 基础仿真波动
        for i in range(self.max_points):
            for ch in range(8):
                v = math.sin(i * 0.04 + ch) * 20.0 + random.normalvariate(0, 3)
                self.buffers[ch].append(v)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(40)
        self.refresh_timer.timeout.connect(self.update)
        self.refresh_timer.start()

    def set_mode(self, mode_str: str):
        self.mode = mode_str
        self.update()

    def paintEvent(self, event):
        # 模拟推数
        for ch in range(8):
            for _ in range(8):
                v = math.sin(time.time() * 8 + ch) * 12.0 + random.normalvariate(0,
                                                                                 5) if self.mode == self.MODE_EEG else random.normalvariate(
                    0, 2) + (80.0 if int(time.time() * 6) % 2 == 0 else -80.0)
                self.buffers[ch].append(v)
            if self.mode == self.MODE_IMPEDANCE:
                self.last_impedance_ohms[ch] = max(100, self.last_impedance_ohms[ch] + random.randint(-150, 150))

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#08111F"))
        w, h = self.width(), self.height()
        row_height = h / 8.0

        for ch in range(8):
            row_top = ch * row_height
            plot_rect = QRectF(55, row_top + 4, w - 145, row_height - 8)
            painter.setPen(QPen(QColor("#1E293B"), 1, Qt.DashLine))
            painter.drawLine(int(plot_rect.left()), int(plot_rect.center().y()), int(plot_rect.right()),
                             int(plot_rect.center().y()))

            painter.setPen(QPen(QColor("#94A3B8"), 1))
            painter.setFont(QFont("Consolas", 10, QFont.Bold))
            painter.drawText(QRectF(10, row_top, 40, row_height), Qt.AlignLeft | Qt.AlignVCenter,
                             self.channel_names[ch])

            if self.mode == self.MODE_EEG:
                ptp = np.ptp(list(self.buffers[ch])) if len(self.buffers[ch]) > 0 else 0.0
                info_text = f"{ptp:.0f} uV"
                painter.setPen(QPen(QColor("#64748B")))
            else:
                z_k = self.last_impedance_ohms[ch] / 1000.0
                info_text = f"{z_k:.1f} kΩ"
                painter.setPen(QPen(QColor("#67E8B9") if z_k < 10.0 else QColor("#F59E0B")))
            painter.drawText(QRectF(w - 85, row_top, 75, row_height), Qt.AlignRight | Qt.AlignVCenter, info_text)

            y_arr = list(self.buffers[ch])
            if len(y_arr) < 2: continue
            waveform = QPainterPath()
            x_scale = plot_rect.width() / (len(y_arr) - 1)
            v_max = 70.0 if self.mode == self.MODE_EEG else 200.0

            for pt_idx, val in enumerate(y_arr):
                x_p = plot_rect.left() + pt_idx * x_scale
                y_p = plot_rect.center().y() - (val / v_max) * (plot_rect.height() / 2.0)
                y_p = max(plot_rect.top(), min(plot_rect.bottom(), y_p))
                if pt_idx == 0:
                    waveform.moveTo(x_p, y_p)
                else:
                    waveform.lineTo(x_p, y_p)

            painter.setPen(QPen(QColor(self.CHANNEL_COLORS[ch % len(self.CHANNEL_COLORS)]), 1.2))
            painter.drawPath(waveform)


# ==========================================
# 拓扑环形状态图
# ==========================================
class PipelineProgressWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(140, 140)
        self.current_stage = 1
        self.stage_percent = 0.0

    def set_stage_progress(self, stage: int, percent: float):
        self.current_stage = stage
        self.stage_percent = max(0.0, min(100.0, percent))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self);
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()
        side = min(width, height)
        rect = QRectF((width - side) / 2 + 10, (height - side) / 2 + 10, side - 20, side - 20)
        painter.setPen(QPen(QColor("#1F2631"), 8))
        painter.drawArc(rect, 0 * 16, 360 * 16)
        total_angle = (self.current_stage - 1) * 120 + (self.stage_percent / 100.0) * 120
        painter.setPen(QPen(QColor("#67E8B9"), 8, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, 90 * 16, -int(total_angle * 16))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold));
        painter.setPen(QPen(QColor("#A9F5D0")))
        stage_names = {1: "SSVEP 阶段", 2: "MI 阶段", 3: "实时控制中"}
        painter.drawText(rect, Qt.AlignCenter, f"{stage_names.get(self.current_stage, '')}\n{int(self.stage_percent)}%")


# ==========================================
# 阶段一：SSVEP 工作台
# ==========================================
class SSVEPStimulusGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.is_flashing = False
        self.targets = [
            {"freq": 9, "name": "目标 1\n9 Hz", "state": True, "rect": QRectF(), "color": "#4CAF50"},
            {"freq": 11, "name": "目标 2\n11 Hz", "state": True, "rect": QRectF(), "color": "#2196F3"},
            {"freq": 13, "name": "目标 3\n13 Hz", "state": True, "rect": QRectF(), "color": "#FF9800"},
            {"freq": 15, "name": "目标 4\n15 Hz", "state": True, "rect": QRectF(), "color": "#E91E63"}
        ]
        self.flash_timer = QTimer(self)
        self.frame_interval_ms = 16  # 约60fps
        self.flash_timer.setInterval(self.frame_interval_ms)
        self.flash_timer.timeout.connect(self._on_tick)
        self.start_time = 0.0
        self.frame_count = 0
        self.hover_target = None

    def start_flashing(self):
        self.is_flashing = True
        self.start_time = time.time()
        self.frame_count = 0
        self.flash_timer.start()

    def stop_flashing(self):
        self.is_flashing = False
        self.flash_timer.stop()
        for t in self.targets:
            t["state"] = True
        self.update()

    def _on_tick(self):
        if not self.is_flashing:
            return
        self.frame_count += 1
        elapsed = self.frame_count * self.frame_interval_ms / 1000.0

        for t in self.targets:
            freq = t["freq"]
            half_period = 0.5 / freq
            t["state"] = (int(elapsed / half_period) % 2 == 0)
        self.update()

# ==========================================
# 阶段一：SSVEP 工作台（集成真实脑电采集）
# ==========================================
class SSVEPStageWidget(QWidget):
    pipeline_updated = pyqtSignal(int, float)
    stage_completed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_collecting = False
        self.is_training = False
        self.collection_thread = None
        self.collection_worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        self.setFixedWidth(380)
        layout.setContentsMargins(10, 10, 10, 10)

        left = QVBoxLayout()
        card1 = QFrame()
        card1.setObjectName("cardPanel")
        c1_l = QVBoxLayout(card1)
        lbl1 = QLabel("SSVEP 脑电数据采集")
        lbl1.setObjectName("panelTitle")
        c1_l.addWidget(lbl1)

        # 采集参数配置区域
        param_layout = QFormLayout()
        self.serial_port_edit = QLineEdit("auto")
        self.serial_port_edit.setStyleSheet("background:#0D1117;color:#67E8B9;")
        self.board_id_spin = QSpinBox()
        self.board_id_spin.setRange(-1, 9999)
        self.board_id_spin.setValue(0)
        self.board_id_spin.setStyleSheet("background:#0D1117;color:#67E8B9;")
        self.freqs_edit = QLineEdit("8,10,12,15")
        self.freqs_edit.setStyleSheet("background:#0D1117;color:#67E8B9;")
        self.prepare_spin = QDoubleSpinBox()
        self.prepare_spin.setRange(0.0, 10.0)
        self.prepare_spin.setDecimals(1)
        self.prepare_spin.setValue(1.0)
        self.prepare_spin.setStyleSheet("background:#0D1117;color:#67E8B9;")
        self.active_spin = QDoubleSpinBox()
        self.active_spin.setRange(1.0, 20.0)
        self.active_spin.setDecimals(1)
        self.active_spin.setValue(4.0)
        self.active_spin.setStyleSheet("background:#0D1117;color:#67E8B9;")
        self.rest_spin = QDoubleSpinBox()
        self.rest_spin.setRange(0.0, 10.0)
        self.rest_spin.setDecimals(1)
        self.rest_spin.setValue(1.0)
        self.rest_spin.setStyleSheet("background:#0D1117;color:#67E8B9;")
        self.target_repeats_spin = QSpinBox()
        self.target_repeats_spin.setRange(1, 20)
        self.target_repeats_spin.setValue(5)
        self.target_repeats_spin.setStyleSheet("background:#0D1117;color:#67E8B9;")
        self.idle_repeats_spin = QSpinBox()
        self.idle_repeats_spin.setRange(1, 30)
        self.idle_repeats_spin.setValue(10)
        self.idle_repeats_spin.setStyleSheet("background:#0D1117;color:#67E8B9;")

        param_layout.addRow("串口", self.serial_port_edit)
        param_layout.addRow("板卡ID", self.board_id_spin)
        param_layout.addRow("频率", self.freqs_edit)
        param_layout.addRow("准备(s)", self.prepare_spin)
        param_layout.addRow("采集(s)", self.active_spin)
        param_layout.addRow("休息(s)", self.rest_spin)
        param_layout.addRow("目标重复", self.target_repeats_spin)
        param_layout.addRow("空闲重复", self.idle_repeats_spin)
        c1_l.addLayout(param_layout)

        self.p_bar = QProgressBar()
        c1_l.addWidget(self.p_bar)
        b_l = QHBoxLayout()
        self.btn_start = QPushButton("开始采集")
        self.btn_start.setObjectName("accentBtn")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("actionBtn")
        b_l.addWidget(self.btn_start)
        b_l.addWidget(self.btn_stop)
        c1_l.addLayout(b_l)
        left.addWidget(card1)

        card2 = QFrame()
        card2.setObjectName("cardPanel")
        c2_l = QVBoxLayout(card2)
        lbl2 = QLabel("FBCCA 分类器训练与建模")
        lbl2.setObjectName("panelTitle")
        c2_l.addWidget(lbl2)
        self.btn_train = QPushButton("一键训练分类器")
        self.btn_train.setObjectName("actionBtn")
        self.lbl_status = QLabel("模型状态: 未就绪")
        self.lbl_status.setStyleSheet("color:#FF8F8F;")
        c2_l.addWidget(self.btn_train)
        c2_l.addWidget(self.lbl_status)
        left.addWidget(card2)
        left.addStretch()
        layout.addLayout(left, 2)

        right = QVBoxLayout()
        self.grid = SSVEPStimulusGrid()
        right.addWidget(QLabel("高频外周刺激源视口 (SSVEP 刺激面板)"), 0, Qt.AlignTop)
        right.addWidget(self.grid, 3)
        self.logger = QTextEdit()
        self.logger.setObjectName("logger")
        self.logger.setReadOnly(True)
        right.addWidget(self.logger, 2)
        layout.addLayout(right, 3)

        self.btn_start.clicked.connect(self._start_collection)
        self.btn_stop.clicked.connect(self._stop_collection)
        self.btn_train.clicked.connect(self._train)
        self.btn_stop.setEnabled(False)

        self.collected_segments = []
        self.total_trials = 0
        self.completed_trials = 0

    def _start_collection(self):
        """启动真实SSVEP脑电采集"""
        if self.is_collecting:
            return

        try:
            freqs = self._parse_freqs(self.freqs_edit.text())
        except Exception as e:
            self.logger.append(f"[ERROR] 频率解析失败: {e}")
            return

        self.is_collecting = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.collected_segments = []
        self.completed_trials = 0

        # 计算总试次
        target_repeats = self.target_repeats_spin.value()
        idle_repeats = self.idle_repeats_spin.value()
        self.total_trials = 4 * target_repeats + idle_repeats

        self.p_bar.setMaximum(self.total_trials)
        self.p_bar.setValue(0)

        self.logger.append(f"[INFO] 开始SSVEP采集，总试次: {self.total_trials}")
        self.logger.append(f"[INFO] 频率: {freqs}, 采集时长: {self.active_spin.value()}s")

        # 启动采集线程
        self.collection_thread = QThread()
        self.collection_worker = SSVEPCollectionWorker(
            serial_port=self.serial_port_edit.text().strip(),
            board_id=self.board_id_spin.value(),
            freqs=freqs,
            prepare_sec=self.prepare_spin.value(),
            active_sec=self.active_spin.value(),
            rest_sec=self.rest_spin.value(),
            target_repeats=self.target_repeats_spin.value(),
            idle_repeats=self.idle_repeats_spin.value()
        )
        self.collection_worker.moveToThread(self.collection_thread)

        self.collection_worker.trial_progress.connect(self._on_trial_progress)
        self.collection_worker.trial_log.connect(self._on_trial_log)
        self.collection_worker.collection_finished.connect(self._on_collection_finished)
        self.collection_worker.collection_error.connect(self._on_collection_error)

        self.collection_thread.started.connect(self.collection_worker.run)
        self.collection_thread.finished.connect(self.collection_thread.deleteLater)

        self.collection_thread.start()

        # 启动刺激闪烁
        self.grid.start_flashing()

    def _stop_collection(self):
        """停止采集"""
        if self.collection_worker:
            self.collection_worker.request_stop()
        self._cleanup_collection()

    def _cleanup_collection(self):
        """清理采集资源"""
        self.is_collecting = False
        self.grid.stop_flashing()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        if self.collection_thread and self.collection_thread.isRunning():
            self.collection_thread.quit()
            self.collection_thread.wait(2000)

        self.collection_worker = None
        self.collection_thread = None

    def _on_trial_progress(self, current: int, total: int):
        """试次进度更新"""
        self.completed_trials = current
        self.p_bar.setValue(current)
        percent = (current / total) * 100
        self.pipeline_updated.emit(1, percent)

    def _on_trial_log(self, message: str):
        """采集日志"""
        self.logger.append(message)

    def _on_collection_finished(self, segments: list, quality_rows: list):
        """采集完成"""
        self.logger.append(f"[SUCCESS] SSVEP采集完成！共采集 {len(segments)} 个有效试次")
        self.collected_segments = segments
        self._cleanup_collection()

        if segments:
            self.btn_train.setEnabled(True)
            self.logger.append("[INFO] 已采集到有效数据，可以开始训练分类器")

    def _on_collection_error(self, error_msg: str):
        """采集错误"""
        self.logger.append(f"[ERROR] 采集失败: {error_msg}")
        self._cleanup_collection()

    def _parse_freqs(self, freqs_str: str) -> tuple:
        """解析频率字符串"""
        freqs = []
        for part in freqs_str.replace("，", ",").split(","):
            part = part.strip()
            if part:
                freqs.append(float(part))
        if len(freqs) != 4:
            raise ValueError(f"需要4个频率，当前: {len(freqs)}")
        return tuple(freqs)

    def _train(self):
        """训练FBCCA分类器"""
        if not self.collected_segments:
            self.logger.append("[WARN] 没有采集数据，无法训练")
            return

        if self.is_training:
            return

        self.is_training = True
        self.btn_train.setEnabled(False)
        self.lbl_status.setText("模型状态: 求解中...")
        self.lbl_status.setStyleSheet("color:#F6C667;")

        self.logger.append("[INFO] 开始训练FBCCA分类器...")

        # 启动训练线程
        self.train_thread = QThread()
        self.train_worker = SSVEPTrainWorker(
            segments=self.collected_segments,
            freqs=self._parse_freqs(self.freqs_edit.text()),
            sampling_rate=250,
            win_sec=3.0,
            step_sec=0.25
        )
        self.train_worker.moveToThread(self.train_thread)

        self.train_worker.train_log.connect(self.logger.append)
        self.train_worker.train_finished.connect(self._on_train_finished)
        self.train_worker.train_error.connect(self._on_train_error)

        self.train_thread.started.connect(self.train_worker.run)
        self.train_thread.finished.connect(self.train_thread.deleteLater)

        self.train_thread.start()

    def _on_train_finished(self, profile_path: str, accuracy: float):
        """训练完成"""
        self.is_training = False
        self.btn_train.setEnabled(True)
        self.lbl_status.setText(f"模型状态: 就绪 (Acc: {accuracy:.1f}%)")
        self.lbl_status.setStyleSheet("color:#67E8B9;")
        self.logger.append(f"[SUCCESS] 训练完成！准确率: {accuracy:.2f}%, Profile: {profile_path}")
        self.pipeline_updated.emit(1, 100.0)
        self.stage_completed.emit()

    def _on_train_error(self, error_msg: str):
        """训练错误"""
        self.is_training = False
        self.btn_train.setEnabled(True)
        self.lbl_status.setText("模型状态: 训练失败")
        self.lbl_status.setStyleSheet("color:#FF8F8F;")
        self.logger.append(f"[ERROR] 训练失败: {error_msg}")


class SSVEPCollectionWorker(QObject):
    """SSVEP脑电采集工作线程"""
    trial_progress = pyqtSignal(int, int)
    trial_log = pyqtSignal(str)
    collection_finished = pyqtSignal(list, list)
    collection_error = pyqtSignal(str)

    def __init__(self, serial_port: str, board_id: int, freqs: tuple,
                 prepare_sec: float, active_sec: float, rest_sec: float,
                 target_repeats: int, idle_repeats: int):
        super().__init__()
        self.serial_port = serial_port
        self.board_id = board_id
        self.freqs = freqs
        self.prepare_sec = prepare_sec
        self.active_sec = active_sec
        self.rest_sec = rest_sec
        self.target_repeats = target_repeats
        self.idle_repeats = idle_repeats
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def _sleep_interruptible(self, seconds: float) -> bool:
        """可中断的等待"""
        deadline = time.time() + seconds
        while not self._stop_requested and time.time() < deadline:
            QThread.msleep(50)
        return self._stop_requested

    def _direction_label(self, freq: float) -> str:
        """获取方向标签"""
        dir_map = {self.freqs[0]: "上", self.freqs[1]: "左",
                   self.freqs[2]: "下", self.freqs[3]: "右"}
        return dir_map.get(freq, f"{freq}Hz")

    def run(self):
        """执行采集"""
        board = None
        try:
            # 动态导入SSVEP核心模块
            from ssvep_core.async_fbcca_idle_standalone import (
                BoardShim, prepare_board_session, ensure_stream_ready,
                build_calibration_trials, read_recent_eeg_segment,
                normalize_serial_port, DEFAULT_STREAM_WARMUP_SEC
            )

            # 构建试次计划
            trials = build_calibration_trials(
                self.freqs,
                target_repeats=self.target_repeats,
                idle_repeats=self.idle_repeats,
                shuffle=True,
                seed=42
            )

            total_trials = len(trials)
            collected_segments = []
            quality_rows = []

            # 连接设备
            serial = normalize_serial_port(self.serial_port)
            self.trial_log.emit(f"[INFO] 连接设备: serial={serial}, board_id={self.board_id}")

            board, resolved_port, attempted = prepare_board_session(self.board_id, serial)
            fs = BoardShim.get_sampling_rate(self.board_id)
            eeg_channels = BoardShim.get_eeg_channels(self.board_id)

            board.start_stream(450000)
            ensure_stream_ready(board, fs)

            self.trial_log.emit(f"[INFO] 设备已连接, 采样率={fs}Hz, EEG通道={eeg_channels}")

            # 预热
            if self._sleep_interruptible(DEFAULT_STREAM_WARMUP_SEC):
                return
            board.get_board_data()

            # 执行各试次
            for idx, trial in enumerate(trials, 1):
                if self._stop_requested:
                    break

                cue_freq = trial.expected_freq
                prompt = (f"注视{self._direction_label(cue_freq)}" if cue_freq is not None
                         else "注视中心点")

                # 准备阶段
                self.trial_log.emit(f"[TRIAL {idx}/{total_trials}] {prompt} - 准备")
                if self._sleep_interruptible(self.prepare_sec):
                    break

                # 清空缓冲区
                board.get_board_data()

                # 激活阶段
                self.trial_log.emit(f"[TRIAL {idx}/{total_trials}] {prompt} - 采集中")
                if self._sleep_interruptible(self.active_sec):
                    break

                # 读取EEG片段
                active_samples = int(round(self.active_sec * fs))
                min_samples = int(round(3.0 * fs))

                try:
                    segment, used, available = read_recent_eeg_segment(
                        board, eeg_channels,
                        target_samples=active_samples,
                        minimum_samples=min_samples
                    )
                except Exception as e:
                    self.trial_log.emit(f"[WARN] 试次{idx}读取失败: {e}")
                    continue

                if used < active_samples:
                    self.trial_log.emit(f"[WARN] 试次{idx}样本不足: {used}/{active_samples}")
                    continue

                collected_segments.append((trial, segment))
                quality_rows.append({
                    "order_index": idx - 1,
                    "target_samples": active_samples,
                    "used_samples": used,
                    "active_sec": self.active_sec,
                    "sample_ratio": used / active_samples
                })

                self.trial_log.emit(f"[TRIAL {idx}/{total_trials}] {prompt} - 完成 ✓")
                self.trial_progress.emit(idx, total_trials)

                # 休息阶段
                if idx < total_trials and not self._stop_requested:
                    self.trial_log.emit(f"[INFO] 休息 {self.rest_sec}s")
                    if self._sleep_interruptible(self.rest_sec):
                        break
                    board.get_board_data()

            # 释放设备
            if board:
                try:
                    board.stop_stream()
                except:
                    pass
                try:
                    board.release_session()
                except:
                    pass

            if self._stop_requested:
                self.trial_log.emit("[INFO] 采集已停止")
            else:
                self.trial_log.emit(f"[INFO] 采集完成，有效试次: {len(collected_segments)}/{total_trials}")

            self.collection_finished.emit(collected_segments, quality_rows)

        except Exception as e:
            self.collection_error.emit(str(e))
        finally:
            if board:
                try:
                    board.release_session()
                except:
                    pass


class SSVEPTrainWorker(QObject):
    """SSVEP训练工作线程"""
    train_log = pyqtSignal(str)
    train_finished = pyqtSignal(str, float)
    train_error = pyqtSignal(str)

    def __init__(self, segments: list, freqs: tuple, sampling_rate: int,
                 win_sec: float, step_sec: float):
        super().__init__()
        self.segments = segments
        self.freqs = freqs
        self.sampling_rate = sampling_rate
        self.win_sec = win_sec
        self.step_sec = step_sec

    def run(self):
        """执行训练"""
        try:
            from ssvep_core.async_fbcca_idle_standalone import (
                optimize_profile_from_segments, save_profile,
                format_profile_quality_summary, DEFAULT_PROFILE_PATH
            )
            from pathlib import Path

            self.train_log.emit("[INFO] 开始优化FBCCA模型参数...")

            # 确定EEG通道（使用标准8通道）
            available_channels = list(range(8))

            # 优化Profile
            profile, metadata = optimize_profile_from_segments(
                self.segments,
                available_board_channels=available_channels,
                sampling_rate=self.sampling_rate,
                freqs=self.freqs,
                active_sec=4.0,
                preferred_win_sec=self.win_sec,
                step_sec=self.step_sec,
                seed=42
            )

            # 保存Profile
            profile_path = Path(DEFAULT_PROFILE_PATH)
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            save_profile(profile, profile_path)

            # 获取质量摘要
            summary = metadata.get("validation_summary", {})
            accuracy = summary.get("acc_4class", 0.0) * 100

            summary_text = format_profile_quality_summary(summary)
            self.train_log.emit(f"[INFO] {summary_text}")

            self.train_finished.emit(str(profile_path), accuracy)

        except Exception as e:
            self.train_error.emit(str(e))
# ==========================================
# 阶段二：MI 工作台（融入原厂精细手、脚、舌头图）
# ==========================================
class MIStageWidget(QWidget):
    """阶段二：MI 工作台（深度集成受试者双屏联动状态机与动态倒计时倒推）"""
    pipeline_updated = pyqtSignal(int, float)
    stage_completed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = None  # 运行期由主窗口注入引用
        self.is_paused = False
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        left = QVBoxLayout()
        card = QFrame();
        card.setObjectName("cardPanel")
        f_l = QFormLayout(card)
        self.s_trials = QSpinBox();
        self.s_trials.setValue(10);
        self.s_trials.setStyleSheet("background:#0D1117;color:#67E8B9;")
        f_l.addRow("采集总试次:", self.s_trials)
        left.addWidget(card)

        card2 = QFrame();
        card2.setObjectName("cardPanel")
        c2_l = QVBoxLayout(card2)
        self.lbl_t = QLabel("当前进度: Trial 0 / 10");
        c2_l.addWidget(self.lbl_t)
        self.p_bar = QProgressBar();
        c2_l.addWidget(self.p_bar)
        self.btn_run = QPushButton("启动 MI 实验范式");
        self.btn_run.setObjectName("accentBtn")
        c2_l.addWidget(self.btn_run)
        left.addWidget(card2)

        card3 = QFrame();
        card3.setObjectName("cardPanel")
        c3_l = QVBoxLayout(card3)
        self.btn_train = QPushButton("空间滤波与特征机建模");
        self.btn_train.setObjectName("actionBtn")
        self.lbl_status = QLabel("MI分类器状态: 未就绪");
        self.lbl_status.setStyleSheet("color:#FF8F8F;")
        c3_l.addWidget(self.btn_train);
        c3_l.addWidget(self.lbl_status)
        left.addWidget(card3);
        left.addStretch()
        layout.addLayout(left, 2)

        right = QVBoxLayout()
        self.cue = MIVisualCueWidget()
        right.addWidget(QLabel("MI 动态视觉运动诱导视口"), 0, Qt.AlignTop)
        right.addWidget(self.cue, 3)
        self.logger = QTextEdit();
        self.logger.setObjectName("logger");
        self.logger.setReadOnly(True)
        right.addWidget(self.logger, 2)
        layout.addLayout(right, 3)

        self.btn_run.clicked.connect(self._run_paradigm)
        self.btn_train.clicked.connect(self._train)

        # 主时序状态机驱动器
        self.tmr = QTimer(self)
        self.tmr.timeout.connect(self._step)

        # 辅倒计时像素级平滑定时器
        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(100)
        self.countdown_timer.timeout.connect(self._update_countdown_label)

        self.cur_trial = 0
        self.sub_step = 0
        self.cur_direction = "LEFT"
        self.step_remaining_ms = 0

    def _run_paradigm(self):
        """开启范式：调出物理隔离受试者屏并进行全屏覆写"""
        self.cur_trial = 1
        self.sub_step = 0
        self.is_paused = False
        self.btn_run.setText("范式运行中...")
        self.btn_run.setEnabled(False)

        # 激活受试者屏幕资产
        if self.main_window and self.main_window.participant_window:
            self.main_window.participant_window.set_prompt("READY", "LEFT", "实验即将启动", "准备")
            self.main_window.participant_window.showFullScreen()
            self.main_window.participant_window.raise_()
            self.main_window.participant_window.activateWindow()

        self._step()

    def _step(self):
        """核心双屏时序状态轮转"""
        if self.cur_trial > self.s_trials.value():
            self.tmr.stop()
            self.countdown_timer.stop()
            self.cue.update_cue("IDLE")
            self.btn_run.setText("启动 MI 实验范式")
            self.btn_run.setEnabled(True)
            self.logger.append("[SUCCESS] MI 四分类物理空间拓扑样本入库。")

            # 收回受试者窗口，让控制台重回视野
            if self.main_window and self.main_window.participant_window:
                self.main_window.participant_window.hide()
                self.main_window.showNormal()
                self.main_window.raise_()
            return

        title_ch = {"LEFT": "左手想象", "RIGHT": "右手想象", "FEET": "双脚想象", "TONGUE": "舌头想象"}

        if self.sub_step == 0:
            # 状态 A：注视十字准备
            self.lbl_t.setText(f"当前进度: Trial {self.cur_trial} / {self.s_trials.value()}")
            self.cue.update_cue("READY")
            self.sub_step = 1

            self.step_remaining_ms = 1500
            self.tmr.start(1500)
            self.countdown_timer.start()

            if self.main_window and self.main_window.participant_window:
                self.main_window.participant_window.set_prompt("READY", "LEFT", "请注视中央十字", "1.5s")

        elif self.sub_step == 1:
            # 状态 B：实心运动提示呈递
            self.cur_direction = random.choice(["LEFT", "RIGHT", "FEET", "TONGUE"])
            self.cue.update_cue("CUE", self.cur_direction)
            self.logger.append(f"[TRIAL {self.cur_trial}] 提示：向动作 [{self.cur_direction}] 产生空间想象特征...")
            self.sub_step = 2

            self.step_remaining_ms = 2500
            self.tmr.start(2500)

            if self.main_window and self.main_window.participant_window:
                self.main_window.participant_window.set_prompt("CUE", self.cur_direction,
                                                               f"请执行：{title_ch[self.cur_direction]}", "2.5s")

        elif self.sub_step == 2:
            # 状态 C：休息空闲
            self.cue.update_cue("REST")
            percent = int((self.cur_trial / self.s_trials.value()) * 100)
            self.p_bar.setValue(percent)
            self.pipeline_updated.emit(2, percent * 0.5)
            self.sub_step = 0
            self.cur_trial += 1

            self.step_remaining_ms = 1500
            self.tmr.start(1500)

            if self.main_window and self.main_window.participant_window:
                self.main_window.participant_window.set_prompt("REST", "LEFT", "休息 恢复放空", "1.5s")

    def _update_countdown_label(self):
        """高精度计算剩余时间，实时平滑刷新受试者视口标签"""
        if self.is_paused: return
        self.step_remaining_ms = max(0, self.step_remaining_ms - 100)
        sec_text = f"{self.step_remaining_ms / 1000.0:.1f}s"
        if self.main_window and self.main_window.participant_window:
            self.main_window.participant_window.countdown_label.setText(sec_text)

    def set_paused_state(self, paused: bool):
        """互锁挂起时序计时器"""
        self.is_paused = paused
        if self.is_paused:
            self.tmr.stop()
            self.countdown_timer.stop()
            self.logger.append("[PAUSE] 物理采集范式已被挂起暂停。")
        else:
            # 恢复时重新校正并恢复非均匀定时脉冲
            self.tmr.start(self.step_remaining_ms)
            self.countdown_timer.start()
            self.logger.append("[RESUME] 范式唤醒，继续当前试次。")

    def _train(self):
        if self.p_bar.value() < 100: return
        self.lbl_status.setText("MI分类器状态: 解耦中...");
        self.lbl_status.setStyleSheet("color:#F6C667;")
        QTimer.singleShot(1000, self._done)

    def _done(self):
        self.lbl_status.setText("MI分类器状态: 已就绪");
        self.lbl_status.setStyleSheet("color:#67E8B9;")
        self.pipeline_updated.emit(2, 100.0);
        self.stage_completed.emit()
# ==========================================
# 阶段三：机械臂实时控制面板
# ==========================================
class RobotCameraWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(450, 300)
        self.active_id = "";
        self.phase = 0
        self.specs = [(0.25, 0.35, "#F6C667", "1"), (0.55, 0.30, "#56D6A6", "2"),
                      (0.35, 0.70, "#6CA8FF", "3"), (0.70, 0.65, "#E879A6", "4")]
        self.tmr = QTimer(self);
        self.tmr.setInterval(150)
        self.tmr.timeout.connect(self._tick);
        self.tmr.start()

    def _tick(self):
        self.phase += 1
        if self.active_id: self.update()

    def paintEvent(self, event):
        painter = QPainter(self);
        painter.setRenderHint(QPainter.Antialiasing)
        grad = QLinearGradient(self.rect().topLeft(), self.rect().bottomRight())
        grad.setColorAt(0.0, QColor("#0B1724"));
        grad.setColorAt(1.0, QColor("#182235"))
        painter.setBrush(grad);
        painter.setPen(QPen(QColor("#2F4058"), 1))
        painter.drawRoundedRect(self.rect().adjusted(5, 5, -5, -5), 8, 8)

        cx, cy = self.width() / 2, self.height() / 2
        painter.setPen(QPen(QColor(120, 165, 210, 40), 1))
        painter.drawLine(20, int(cy), self.width() - 20, int(cy))
        painter.drawLine(int(cx), 20, int(cx), self.height() - 20)

        for xr, yr, c_s, lbl in self.specs:
            bx, by = self.width() * xr, self.height() * yr
            is_a = (lbl == self.active_id)
            painter.setBrush(QColor(c_s))
            painter.setPen(QPen(QColor("#FFFFFF") if is_a else QColor("#F8FBFF"), 2 if is_a else 1))
            painter.drawRoundedRect(QRectF(bx - 25, by - 15, 50, 30), 4, 4)
            if is_a:
                painter.setBrush(Qt.NoBrush);
                painter.setPen(QPen(QColor("#82E6C4"), 2))
                pulse = 5 + (self.phase % 4) * 4
                painter.drawEllipse(QRectF(bx - 25 - pulse, by - 15 - pulse, 50 + pulse * 2, 30 + pulse * 2))
                painter.setPen(QPen(QColor("#82E6C4"), 1, Qt.DashLine));
                painter.drawLine(int(cx), int(cy), int(bx), int(by))
            painter.setPen(QColor("#08111B"));
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(QRectF(bx - 25, by - 15, 50, 30), Qt.AlignCenter, lbl)
        painter.setPen(QPen(QColor("#67E8B9"), 2));
        painter.drawEllipse(int(cx - 12), int(cy - 12), 24, 24)

class RobotControlStageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        state_bar = QFrame();
        state_bar.setObjectName("stateBar")
        sb_l = QHBoxLayout(state_bar)
        w_text = QWidget();
        wt_l = QVBoxLayout(w_text);
        wt_l.setContentsMargins(0, 0, 0, 0)
        lbl_t = QLabel("混合脑控状态拓扑总线");
        lbl_t.setObjectName("stateBarTitle")
        self.lbl_run_status = QLabel("系统就绪：等待高维脑控指令执行下发");
        self.lbl_run_status.setObjectName("controlStatusLabel")
        wt_l.addWidget(lbl_t);
        wt_l.addWidget(self.lbl_run_status);
        sb_l.addWidget(w_text, 2)

        self.nodes = []
        node_names = ["预训练完成", "摄像头读取", "WASD 移动", "数字选块", "机械臂执行"]
        for i, name in enumerate(node_names):
            nd = QFrame();
            nd.setObjectName("stateNode")
            nd.setStyleSheet(
                "background:#122A22;border:1px solid #57D6A6;" if i == 1 else "background:#10161F;border:1px solid #293446;")
            nl = QVBoxLayout(nd);
            nl.setContentsMargins(6, 4, 6, 4)
            lbl = QLabel(name);
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#67E8B9;font-weight:bold;" if i == 1 else "color:#8B97A5;")
            nl.addWidget(lbl);
            sb_l.addWidget(nd, 1)
            self.nodes.append((nd, lbl))
        layout.addWidget(state_bar)

        grid = QHBoxLayout()
        cam_card = QFrame();
        cam_card.setObjectName("cameraCard")
        cc_l = QVBoxLayout(cam_card)
        cc_l.addWidget(QLabel("数字孪生机械臂末端寻靶相机流"), 0, Qt.AlignTop)
        self.cam = RobotCameraWidget();
        cc_l.addWidget(self.cam, 1)
        grid.addWidget(cam_card, 3)

        side = QVBoxLayout()
        pc = QFrame();
        pc.setObjectName("poseCard");
        pc_l = QVBoxLayout(pc)
        lbl_p = QLabel("绝对空间几何矩阵 (Kinematics Pose)");
        lbl_p.setObjectName("stateBarTitle");
        pc_l.addWidget(lbl_p)
        self.pos_lbl = QLabel("末端空间三维坐标: X: 142.50  Y: -36.20  Z: 92.10 mm");
        row1 = QFrame();
        row1.setObjectName("poseRow");
        r1l = QVBoxLayout(row1);
        r1l.addWidget(self.pos_lbl);
        pc_l.addWidget(row1)
        self.grip_lbl = QLabel("末端夹爪运动姿态: STANDBY · OPEN (安全释放)");
        row2 = QFrame();
        row2.setObjectName("poseRow");
        r2l = QVBoxLayout(row2);
        r2l.addWidget(self.grip_lbl);
        pc_l.addWidget(row2)
        side.addWidget(pc)

        bc = QFrame();
        bc.setObjectName("flowCard");
        bc_l = QVBoxLayout(bc)
        lbl_b = QLabel("靶区目标物理快选 (模拟融合驱动器)");
        lbl_b.setObjectName("stateBarTitle");
        bc_l.addWidget(lbl_b)
        bg = QGridLayout()
        self.b_btns = {}
        for idx in range(1, 5):
            btn = QPushButton(f"目标木块 {idx}");
            btn.setProperty("blockState", "pending")
            btn.clicked.connect(lambda checked, b_id=str(idx): self._select(b_id))
            bg.addWidget(btn, (idx - 1) // 2, (idx - 1) % 2);
            self.b_btns[str(idx)] = btn
        bc_l.addLayout(bg)
        self.btn_exec = QPushButton("下发真实网络机械臂抓取命令序列 (TX)");
        self.btn_exec.setProperty("controlType", "primary")
        bc_l.addWidget(self.btn_exec);
        side.addWidget(bc);
        side.addStretch()
        grid.addLayout(side, 2)
        layout.addLayout(grid, 4)

        self.logger = QTextEdit();
        self.logger.setObjectName("logger");
        self.logger.setReadOnly(True)
        layout.addWidget(self.logger, 1)

        self.btn_exec.clicked.connect(self._exec_grab)
        self.cur_target = ""

    def _select(self, b_id: str):
        self.cur_target = b_id;
        self.cam.active_id = b_id;
        self._update_nodes(3)
        self.lbl_run_status.setText(f"SSVEP 特征解算成功：已高亮锁存候选木块目标 [{b_id}]")
        for k, btn in self.b_btns.items():
            btn.setProperty("blockState", "active" if k == b_id else "pending")
            btn.style().unpolish(btn);
            btn.style().polish(btn)

    def _update_nodes(self, active_idx: int):
        for idx, (node, lbl) in enumerate(self.nodes):
            if idx == active_idx:
                node.setStyleSheet("background:#122A22;border:1px solid #57D6A6;");
                lbl.setStyleSheet("color:#67E8B9;font-weight:bold;")
            else:
                node.setStyleSheet("background:#10161F;border:1px solid #293446;");
                lbl.setStyleSheet("color:#8B97A5;")

    def _exec_grab(self):
        if not self.cur_target: return
        self._update_nodes(4)
        self.pos_lbl.setText("末端空间三维坐标: X: 164.20  Y: -25.00  Z: 58.00 mm")
        self.grip_lbl.setText("末端夹爪运动姿态: BUSY · GRIP_CLOSE (抓取紧固中)")
        self.grip_lbl.setStyleSheet("color:#F6C667; font-weight:bold;")
        QTimer.singleShot(1500, self._grab_done)

    def _grab_done(self):
        self._update_nodes(1);
        self.lbl_run_status.setText("系统就绪：闭环指令队列完毕，重置视觉瞄准基准")
        self.pos_lbl.setText("末端空间三维坐标: X: 142.50  Y: -36.20  Z: 92.10 mm")
        self.grip_lbl.setText("末端夹爪运动姿态: STANDBY · OPEN (安全释放)")
        self.grip_lbl.setStyleSheet("color:#E8EEF6; font-weight:normal;")


# ==========================================
# 系统总集成多页面中央控制工作台
# ==========================================
class BCIIntegratedWorkbenchWindow(QMainWindow):
    """系统总集成多页面中央控制工作台（完美适配双屏状态同步、按键信号互锁与closeEvent安全清场）"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("混合脑机接口一体化控制工作台 (保留原厂手脚舌头精细矢量图版)")
        self.resize(1380, 780)
        self.setStyleSheet(CYBERPUNK_STYLE)

        # 1. 实例化核心全屏独立屏
        self.participant_window = ParticipantDisplayWindow()

        # 2. 接管受试者全屏屏发出的底层通讯透传信号
        self.participant_window.pause_requested.connect(self.toggle_pause)
        self.participant_window.mark_bad_requested.connect(self.mark_bad_trial)
        self.participant_window.stop_requested.connect(self._on_manual_stop)

        self._init_ui()

        # 3. 注入引用，使 MI 时序控制台能自由变换受试者屏
        self.mi_page.main_window = self

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QHBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # 左侧常驻导航
        nav_bar = QFrame();
        nav_bar.setObjectName("navBar");
        nav_bar.setFixedWidth(210)
        nav_layout = QVBoxLayout(nav_bar);
        nav_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("BCI 主控中枢");
        title.setObjectName("navTitle");
        nav_layout.addWidget(title)

        self.monitor = PipelineProgressWidget()
        nav_layout.addWidget(self.monitor, 0, Qt.AlignCenter)
        nav_layout.addSpacing(20)

        self.nav_btns = []
        stages_config = [("1. SSVEP 采集与训练", True), ("2. MI 运动想象采集", False), ("3. 机械臂脑控控制台", False)]
        for name, enabled in stages_config:
            btn = QPushButton(name);
            btn.setObjectName("navBtn")
            btn.setEnabled(enabled);
            btn.setProperty("active", "false")
            nav_layout.addWidget(btn);
            self.nav_btns.append(btn)

        nav_layout.addStretch()
        copyright_lbl = QLabel("© 2026 混合智能控制实验室\n保留所有软著资产权利\nVersion 1.0.5")
        copyright_lbl.setStyleSheet("color:#394555; font-size:11px; padding:15px; text-align:center;")
        nav_layout.addWidget(copyright_lbl);
        outer_layout.addWidget(nav_bar)

        # 中间多页面堆栈切换区
        self.container = QStackedWidget()
        self.ssvep_page = SSVEPStageWidget();
        self.container.addWidget(self.ssvep_page)
        self.mi_page = MIStageWidget();
        self.container.addWidget(self.mi_page)
        self.robot_page = RobotControlStageWidget();
        self.container.addWidget(self.robot_page)
        outer_layout.addWidget(self.container, 5)

        # 右侧常驻脑电检查监看面板
        eeg_panel = QFrame();
        eeg_panel.setStyleSheet("background: #11161D; border-left: 1px solid #1F2631;");
        eeg_panel.setFixedWidth(350)
        ep_l = QVBoxLayout(eeg_panel);
        ep_l.setContentsMargins(10, 10, 10, 10)
        ep_l.addWidget(QLabel("物理层实时信号监看 (8-Ch Cyton)"))

        self.realtime_wave_canvas = RealtimeEEGPreviewWidget()
        ep_l.addWidget(self.realtime_wave_canvas, 1)

        toggle_lay = QHBoxLayout()
        self.btn_eeg_mode = QPushButton("时域波形");
        self.btn_eeg_mode.setObjectName("accentBtn")
        self.btn_imp_mode = QPushButton("引脚阻抗");
        self.btn_imp_mode.setObjectName("actionBtn")
        toggle_lay.addWidget(self.btn_eeg_mode);
        toggle_lay.addWidget(self.btn_imp_mode);
        ep_l.addLayout(toggle_lay)
        outer_layout.addWidget(eeg_panel)

        # 信号槽状态机互锁绑定
        self.ssvep_page.pipeline_updated.connect(self.monitor.set_stage_progress)
        self.ssvep_page.stage_completed.connect(self._unlock_mi)
        self.mi_page.pipeline_updated.connect(self.monitor.set_stage_progress)
        self.mi_page.stage_completed.connect(self._unlock_robot)

        self.btn_eeg_mode.clicked.connect(lambda: self._switch_canvas_mode(RealtimeEEGPreviewWidget.MODE_EEG))
        self.btn_imp_mode.clicked.connect(lambda: self._switch_canvas_mode(RealtimeEEGPreviewWidget.MODE_IMPEDANCE))

        self._update_highlight(0)
        self.nav_btns[0].clicked.connect(lambda: (self.container.setCurrentIndex(0), self._update_highlight(0)))
        self.nav_btns[1].clicked.connect(lambda: (self.container.setCurrentIndex(1), self._update_highlight(1)))
        self.nav_btns[2].clicked.connect(lambda: (self.container.setCurrentIndex(2), self._update_highlight(2)))

    def _switch_canvas_mode(self, mode_type: str):
        self.realtime_wave_canvas.set_mode(mode_type)
        self.btn_eeg_mode.setObjectName("accentBtn" if mode_type == RealtimeEEGPreviewWidget.MODE_EEG else "actionBtn")
        self.btn_imp_mode.setObjectName(
            "accentBtn" if mode_type == RealtimeEEGPreviewWidget.MODE_IMPEDANCE else "actionBtn")
        self.btn_eeg_mode.style().unpolish(self.btn_eeg_mode);
        self.btn_eeg_mode.style().polish(self.btn_eeg_mode)
        self.btn_imp_mode.style().unpolish(self.btn_imp_mode);
        self.btn_imp_mode.style().polish(self.btn_imp_mode)

    def _update_highlight(self, index: int):
        for i, btn in enumerate(self.nav_btns):
            btn.setProperty("active", "true" if i == index else "false")
            btn.style().unpolish(btn);
            btn.style().polish(btn)

    @pyqtSlot()
    def _unlock_mi(self):
        self.nav_btns[1].setEnabled(True)
        QTimer.singleShot(400, lambda: (self.container.setCurrentIndex(1), self._update_highlight(1)))

    @pyqtSlot()
    def _unlock_robot(self):
        self.nav_btns[2].setEnabled(True)
        QTimer.singleShot(400, lambda: (self.container.setCurrentIndex(2), self._update_highlight(2)))

    # ----- 双屏网络互锁控制槽接口实现 -----
    def toggle_pause(self):
        """受试者击打[Space]或操作员触发暂停"""
        if self.container.currentIndex() == 1:  # 仅在MI采集页激活
            next_state = not self.mi_page.is_paused
            self.mi_page.set_paused_state(next_state)
            if next_state:
                self.participant_window.stage_label.setText("实验暂停中")
                self.participant_window.stage_label.setStyleSheet(
                    "color:white; background:#EF4444; font-size:28px; font-weight:bold; border-radius:18px; padding:12px;")
            else:
                self.participant_window.stage_label.setText("实验继续")

    def mark_bad_trial(self):
        """受试者或实验员键入 B 键强行标记当前试次损毁"""
        if self.container.currentIndex() == 1:
            self.mi_page.logger.append(f"[WARNING] Operator/Subject marked Trial {self.mi_page.cur_trial} as BAD样本.")
            self.participant_window.stage_label.setText("× 坏试次标记成功")

    def _on_manual_stop(self):
        """受试者键入 Esc 中止范式"""
        if self.container.currentIndex() == 1:
            self.mi_page.cur_trial = 9999  # 强行推向状态机终点
            self.mi_page._step()

    def closeEvent(self, event):
        """优雅关闭：强行销毁全屏屏，阻断后台死锁隐患"""
        if hasattr(self, 'participant_window') and self.participant_window is not None:
            self.participant_window.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BCIIntegratedWorkbenchWindow()
    window.show()
    sys.exit(app.exec_())
