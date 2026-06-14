# -*- coding: utf-8 -*-
"""
ui_widgets.py - 核心画布组件与受试者提示大窗总库（SSVEP 频闪块适度放大版）
修改点：
1. 优化全屏采集模式下交叉 4 个方块的 box_size 边界，使其比原版物理尺寸放大一点点。
2. 保持全局高清晰度浅蓝色科技调色盘适配。
"""
import math
import time
import random
from collections import deque
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush, QLinearGradient
from PyQt5.QtWidgets import QWidget, QStackedWidget, QLabel, QVBoxLayout, QHBoxLayout, QProgressBar

# 全局高清晰度亮色调色盘
THEME_BG_DARK = QColor("#080B0F")
THEME_CARD_BG = QColor("#151B24")
THEME_BORDER_COLOR = QColor("#252E3C")
THEME_ACCENT_COLOR = QColor("#0284C7")
THEME_TEXT_MUTED = QColor("#64748B")


class SSVEPStimulusGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.is_flashing = False
        self.targets = [
            {"freq": 9.0, "name": "目标 1 (上)", "state": True, "rect": QRectF(), "highlight": False},
            {"freq": 11.0, "name": "目标 2 (左)", "state": True, "rect": QRectF(), "highlight": False},
            {"freq": 13.0, "name": "目标 3 (下)", "state": True, "rect": QRectF(), "highlight": False},
            {"freq": 15.0, "name": "目标 4 (右)", "state": True, "rect": QRectF(), "highlight": False}
        ]
        self.flash_timer = QTimer(self)
        self.flash_timer.setInterval(16)
        self.flash_timer.timeout.connect(self._on_render_tick)
        self.start_time = 0.0

    def start_flashing(self):
        self.is_flashing = True
        self.start_time = time.time()
        self.flash_timer.start()
        self.update()

    def stop_flashing(self):
        self.is_flashing = False
        self.flash_timer.stop()
        for t in self.targets:
            t["state"] = True
            t["highlight"] = False
        self.update()

    def _on_render_tick(self):
        if not self.is_flashing: return
        elapsed = max(0.0, time.time() - self.start_time)
        for t in self.targets:
            half_period = 0.5 / t["freq"]
            t["state"] = (int(math.floor(elapsed / half_period)) % 2 == 0)
        self.update()

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0 - 10
        # 🔥【尺寸微调】：将方块的缩放上调至 max 120 / min 170 之间，屏幕系数放大至 0.27，比原本大一点点，诱发强度更高
        box_size = max(120.0, min(170.0, min(w * 0.27, h * 0.28)))
        self.targets[0]["rect"] = QRectF(cx - box_size / 2.0, 25, box_size, box_size)
        self.targets[1]["rect"] = QRectF(25, cy - box_size / 2.0, box_size, box_size)
        self.targets[2]["rect"] = QRectF(cx - box_size / 2.0, h - box_size - 65, box_size, box_size)
        self.targets[3]["rect"] = QRectF(w - box_size - 25, cy - box_size / 2.0, box_size, box_size)
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing, True)
        painter.fillRect(self.rect(), THEME_BG_DARK)
        for t in self.targets:
            r = t["rect"]
            if r.width() <= 0: continue
            bg_color = (QColor("#FFFFFF") if t["state"] else QColor("#000000")) if self.is_flashing else QColor(
                "#151B24")
            text_color = (QColor("#000000") if t["state"] else QColor("#445161")) if self.is_flashing else QColor(
                "#8B97A5")
            painter.setBrush(QBrush(bg_color))
            painter.setPen(QPen(THEME_ACCENT_COLOR, 4) if t["highlight"] else QPen(THEME_BORDER_COLOR, 1))
            painter.drawRoundedRect(r, 8, 8)  # 圆角微调
            painter.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))  # 字体同步稍微放大
            painter.setPen(text_color)
            painter.drawText(r, Qt.AlignCenter, t["name"])


class MIVisualCueWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 320)
        self.state, self.direction = "IDLE", "LEFT"

    def update_cue(self, state: str, direction: str = "LEFT"):
        self.state, self.direction = state, direction
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing, True)
        painter.fillRect(self.rect(), QColor("#04111C"))
        cx, cy = self.width() / 2.0, self.height() / 2.0

        if self.state == "READY":
            painter.setPen(QPen(QColor("#569AFF"), 6, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(int(cx - 40), int(cy), int(cx + 40), int(cy))
            painter.drawLine(int(cx), int(cy - 40), int(cx), int(cy + 40))
        elif self.state == "CUE":
            painter.setFont(QFont("Microsoft YaHei", 28, QFont.Bold))
            painter.setPen(QPen(QColor("#34D399")))
            t_map = {
                "LEFT": "⬅️ 【左 手 动 作】 意 图 激 发 中",
                "RIGHT": "➡️ 【右 手 动 作】 意 图 激 发 中",
                "FEET": "🦶 【双 脚 动 作】 意 图 激 发 中",
                "TONGUE": "👅 【舌 部 动 作】 意 图 激 发 中"
            }
            painter.drawText(self.rect(), Qt.AlignCenter, t_map.get(self.direction, "运动想象中..."))
        elif self.state == "REST":
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor("#64748B")))
            painter.drawEllipse(QPointF(cx, cy), 20, 20)


class RealtimeEEGPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.buffers = [deque(maxlen=400) for _ in range(8)]
        self.mode = "EEG"
        for i in range(400):
            for ch in range(8): self.buffers[ch].append(math.sin(i * 0.05 + ch) * 15.0 + random.normalvariate(0, 2))
        self.t = QTimer(self)
        self.t.setInterval(50)
        self.t.timeout.connect(self.update)
        self.t.start()

    def set_mode(self, m):
        self.mode = m

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))
        w, h = self.width(), self.height()
        rh = h / 8.0
        for ch in range(8):
            painter.setPen(QPen(QColor("#E2E8F0"), 1))
            painter.drawLine(0, int(ch * rh + rh / 2), w, int(ch * rh + rh / 2))
            painter.setPen(QPen(QColor("#0369A1")))
            painter.setFont(QFont("Consolas", 11, QFont.Bold))
            painter.drawText(10, int(ch * rh + 22), f"Ch {ch + 1}")
            y_arr = list(self.buffers[ch])
            path = QPainterPath()
            xs = w / max(1, len(y_arr) - 1)
            for i, val in enumerate(y_arr):
                xp = i * xs
                yp = ch * rh + rh / 2 - val
                if i == 0:
                    path.moveTo(xp, yp)
                else:
                    path.lineTo(xp, yp)
            painter.setPen(QPen(QColor("#0284C7"), 1.5))
            painter.drawPath(path)


class RobotCameraWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_id, self.phase = "", 0
        self.tmr = QTimer(self)
        self.tmr.setInterval(100)
        self.tmr.timeout.connect(self._tick)
        self.tmr.start()

    def _tick(self):
        self.phase += 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0F172A"))
        cx, cy = self.width() / 2.0, self.height() / 2.0
        specs = [(0.25, 0.4, "#F6C667", "1"), (0.45, 0.35, "#38BDF8", "2"), (0.35, 0.7, "#A855F7", "3"),
                 (0.65, 0.6, "#F43F5E", "4")]
        for xr, yr, col, lbl in specs:
            bx, by = self.width() * xr, self.height() * yr
            is_a = (lbl == self.active_id)
            painter.setBrush(QBrush(QColor(col)))
            painter.setPen(QPen(QColor("#FFFFFF") if is_a else QColor(0, 0, 0, 0), 2 if is_a else 0))
            painter.drawRoundedRect(QRectF(bx - 30, by - 20, 60, 40), 6, 6)
            if is_a:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor("#38BDF8"), 2))
                p = 4 + (self.phase % 4) * 3
                painter.drawEllipse(QPointF(bx, by), 40 + p, 30 + p)
                painter.setPen(QPen(QColor("#38BDF8"), 1, Qt.DashLine))
                painter.drawLine(int(cx), int(cy), int(bx), int(by))
            painter.setPen(QColor("#000000"))
            painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
            painter.drawText(QRectF(bx - 30, by - 20, 60, 40), Qt.AlignCenter, lbl)
        painter.setPen(QPen(QColor("#38BDF8"), 2))
        painter.drawEllipse(QPointF(cx, cy), 12, 12)


class ParticipantDisplayWindow(QWidget):
    stop_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background: #04111C;")
        self.current_calib_color = QColor("#569AFF")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 40, 50, 40)
        layout.setSpacing(20)

        self.stage_label = QLabel("多模态脑机接口实验工作台")
        self.stage_label.setAlignment(Qt.AlignCenter)
        self.stage_label.setStyleSheet("color: white; font-size: 32px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.stage_label)

        self.screen_pbar = QProgressBar()
        self.screen_pbar.setFixedHeight(24)
        self.screen_pbar.setStyleSheet("""
            QProgressBar { border: 2px solid #1E293B; background: #0F172A; text-align: center; color: white; font-weight: bold; border-radius: 6px; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284C7, stop:1 #38BDF8); }
        """)
        layout.addWidget(self.screen_pbar)

        self.visual_stack = QStackedWidget()
        self.ssvep_grid_widget = SSVEPStimulusGrid()
        self.mi_cue_widget = MIVisualCueWidget()

        self.visual_stack.addWidget(self.ssvep_grid_widget)
        self.visual_stack.addWidget(self.mi_cue_widget)
        layout.addWidget(self.visual_stack, stretch=1)

        self.countdown_label = QLabel("--")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("color: #38BDF8; font-size: 64px; font-weight: bold;")
        layout.addWidget(self.countdown_label)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        lbl_geom = self.stage_label.geometry()
        r = QRectF(lbl_geom.x(), lbl_geom.y(), lbl_geom.width(), lbl_geom.height())
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.current_calib_color))
        painter.drawRoundedRect(r, 12, 12)

    def set_prompt_ssvep(self, is_flashing, stage_text, countdown_text, cur_trial, total_trials, active_freq=None):
        self.visual_stack.setCurrentIndex(0)
        self.stage_label.setText(stage_text)
        self.countdown_label.setText(countdown_text)
        self.screen_pbar.setMaximum(total_trials)
        self.screen_pbar.setValue(cur_trial)
        self.screen_pbar.setFormat(f"SSVEP 目标数据采集进度: 第 {cur_trial} 试次 / 共 {total_trials} 试次")
        if is_flashing:
            self.ssvep_grid_widget.start_flashing()
        else:
            self.ssvep_grid_widget.stop_flashing()

    def set_prompt_mi(self, phase, direction, stage_text, countdown_text, cur_trial=None, total_trials=None):
        self.visual_stack.setCurrentIndex(1)
        self.stage_label.setText(stage_text)
        self.countdown_label.setText(countdown_text)
        self.mi_cue_widget.update_cue(phase, direction)
        if cur_trial and total_trials:
            self.screen_pbar.setMaximum(total_trials)
            self.screen_pbar.setValue(cur_trial)
            self.screen_pbar.setFormat(f"MI 感觉动作想象采集进度: 第 {cur_trial} 试次 / 共 {total_trials} 试次")

    def set_prompt_calibration_safe(self, stage_text, countdown_text, color_hex, current_ms=None, total_ms=None):
        self.visual_stack.setCurrentIndex(1)
        self.stage_label.setText(stage_text)
        self.countdown_label.setText(countdown_text)
        self.current_calib_color = QColor(color_hex)
        self.mi_cue_widget.update_cue("READY")
        if current_ms is not None and total_ms is not None:
            self.screen_pbar.setMaximum(total_ms)
            self.screen_pbar.setValue(total_ms - current_ms)
            self.screen_pbar.setFormat(f"生理噪声基准校准进度: 倒计时 {(current_ms / 1000.0):.1f} 秒")

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.stop_requested.emit()
            event.accept()
        else:
            super().keyPressEvent(event)
