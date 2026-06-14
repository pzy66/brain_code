# -*- coding: utf-8 -*-
"""
脑机接口混合控制系统 - 阶段二：MI 运动想象采集与训练（独立运行测试版）
"""
from __future__ import annotations
import sys
import random
from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QProgressBar, QTextEdit, QSpinBox, QFormLayout
)

# ==========================================
# 统一科技暗黑风格样式表（独立测试版）
# ==========================================
MI_TEST_QSS = """
QMainWindow {
    background-color: #0B0E13;
}
QWidget {
    color: #D2DAE5;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}
QFrame#cardPanel {
    background-color: #151B24;
    border: 1px solid #252E3C;
    border-radius: 6px;
}
QLabel#panelTitle {
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
QPushButton#actionBtn:hover {
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
QPushButton#accentBtn:hover {
    background-color: #1B352B;
    border-color: #4EBA93;
    color: #A9F5D0;
}
QTextEdit#logger {
    background-color: #0D1117;
    border: 1px solid #1F2631;
    border-radius: 4px;
    color: #A4B1CD;
    font-family: "Consolas", "Courier New", monospace;
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
"""


# ==========================================
# 核心刺激组件：MI 视觉引导提示画布
# ==========================================
class MIVisualCueWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 350)
        self.current_state = "IDLE"
        self.cue_direction = "LEFT"

    def update_cue(self, state: str, direction: str = "LEFT"):
        self.current_state = state
        self.cue_direction = direction
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#090D14"))

        cx, cy = self.width() / 2, self.height() / 2

        if self.current_state == "READY":
            pen = QPen(QColor("#569AFF"), 5)
            painter.setPen(pen)
            painter.drawLine(int(cx - 30), int(cy), int(cx + 30), int(cy))
            painter.drawLine(int(cx), int(cy - 30), int(cx), int(cy + 30))

        elif self.current_state == "CUE":
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor("#67E8B9")))

            path = QPainterPath()
            if self.cue_direction == "LEFT":
                path.moveTo(cx + 40, cy - 20)
                path.lineTo(cx - 10, cy - 20)
                path.lineTo(cx - 10, cy - 40)
                path.lineTo(cx - 50, cy)
                path.lineTo(cx - 10, cy + 40)
                path.lineTo(cx - 10, cy + 20)
                path.lineTo(cx + 40, cy + 20)
            elif self.cue_direction == "RIGHT":
                path.moveTo(cx - 40, cy - 20)
                path.lineTo(cx + 10, cy - 20)
                path.lineTo(cx + 10, cy - 40)
                path.lineTo(cx + 50, cy)
                path.lineTo(cx + 10, cy + 40)
                path.lineTo(cx + 10, cy + 20)
                path.lineTo(cx - 40, cy + 20)
            elif self.cue_direction == "FEET":
                path.moveTo(cx - 20, cy - 40)
                path.lineTo(cx - 20, cy + 10)
                path.lineTo(cx - 40, cy + 10)
                path.lineTo(cx, cy + 50)
                path.lineTo(cx + 40, cy + 10)
                path.lineTo(cx + 20, cy + 10)
                path.lineTo(cx + 20, cy - 40)
            elif self.cue_direction == "TONGUE":
                path.moveTo(cx - 20, cy + 40)
                path.lineTo(cx - 20, cy - 10)
                path.lineTo(cx - 40, cy - 10)
                path.lineTo(cx, cy - 50)
                path.lineTo(cx + 40, cy - 10)
                path.lineTo(cx + 20, cy - 10)
                path.lineTo(cx + 20, cy + 40)

            painter.drawPath(path)

            painter.setFont(QFont("Segoe UI", 14, QFont.Bold))
            painter.setPen(QPen(QColor("#FFFFFF")))
            text_map = {"LEFT": "【想象：左手握拳】", "RIGHT": "【想象：右手握拳】", "FEET": "【想象：双脚运动】",
                        "TONGUE": "【想象：舌头伸缩】"}
            painter.drawText(self.rect(), Qt.AlignBottom | Qt.HCenter, text_map.get(self.cue_direction, ""))

        elif self.current_state == "REST":
            painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
            painter.setPen(QPen(QColor("#8B97A5")))
            painter.drawText(self.rect(), Qt.AlignCenter, "REST (请休息...)")

        else:
            painter.setFont(QFont("Segoe UI", 13))
            painter.setPen(QPen(QColor("#445161")))
            painter.drawText(self.rect(), Qt.AlignCenter, "操作系统就绪，等待开启实验")


# ==========================================
# 核心主面板
# ==========================================
class MIStageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)

        config_card = QFrame()
        config_card.setObjectName("cardPanel")
        config_layout = QVBoxLayout(config_card)
        config_title = QLabel("采集时序范式配置")
        config_title.setObjectName("panelTitle")
        config_layout.addWidget(config_title)

        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        form_layout.setContentsMargins(0, 5, 0, 5)
        form_layout.setSpacing(8)

        self.spin_ready = QSpinBox()
        self.spin_ready.setRange(1, 5)
        self.spin_ready.setValue(2)
        self.spin_ready.setStyleSheet("background: #0D1117; color: #67E8B9; border: 1px solid #252E3C;")
        form_layout.addRow("准备时间 (秒):", self.spin_ready)

        self.spin_cue = QSpinBox()
        self.spin_cue.setRange(2, 8)
        self.spin_cue.setValue(4)
        self.spin_cue.setStyleSheet("background: #0D1117; color: #67E8B9; border: 1px solid #252E3C;")
        form_layout.addRow("提示想象时间 (秒):", self.spin_cue)

        self.spin_rest = QSpinBox()
        self.spin_rest.setRange(1, 5)
        self.spin_rest.setValue(2)
        self.spin_rest.setStyleSheet("background: #0D1117; color: #67E8B9; border: 1px solid #252E3C;")
        form_layout.addRow("休息时间 (秒):", self.spin_rest)

        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(5, 100)
        self.spin_trials.setValue(20)
        self.spin_trials.setStyleSheet("background: #0D1117; color: #67E8B9; border: 1px solid #252E3C;")
        form_layout.addRow("总试次 (Trials):", self.spin_trials)

        config_layout.addWidget(form_widget)
        left_panel.addWidget(config_card)

        run_card = QFrame()
        run_card.setObjectName("cardPanel")
        run_layout = QVBoxLayout(run_card)
        run_title = QLabel("MI 运动想象实时采集")
        run_title.setObjectName("panelTitle")
        run_layout.addWidget(run_title)

        self.trial_lbl = QLabel("当前进度: Trial 0 / 20")
        self.trial_lbl.setStyleSheet("color: #A4B1CD;")
        run_layout.addWidget(self.trial_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        run_layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开启 MI 范式")
        self.start_btn.setObjectName("accentBtn")
        self.stop_btn = QPushButton("终止实验")
        self.stop_btn.setObjectName("actionBtn")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        run_layout.addLayout(btn_layout)
        left_panel.addWidget(run_card)

        train_card = QFrame()
        train_card.setObjectName("cardPanel")
        train_layout = QVBoxLayout(train_card)
        train_title = QLabel("CSP + SVM 模式识别训练")
        train_title.setObjectName("panelTitle")
        train_layout.addWidget(train_title)

        self.train_btn = QPushButton("并行矩阵解耦与特征训练")
        self.train_btn.setObjectName("actionBtn")
        train_layout.addWidget(self.train_btn)

        self.model_status = QLabel("MI分类器状态: 未就绪")
        self.model_status.setStyleSheet("color: #FF8F8F; font-weight: bold;")
        train_layout.addWidget(self.model_status)
        left_panel.addWidget(train_card)

        main_layout.addLayout(left_panel, 2)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)

        cue_title = QLabel("被试端正向实时视觉引导系统")
        cue_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #569AFF;")
        right_panel.addWidget(cue_title)

        self.visual_cue = MIVisualCueWidget()
        right_panel.addWidget(self.visual_cue, 3)

        log_title = QLabel("MI 阶段系统流水日志")
        log_title.setStyleSheet("font-size: 12px; color: #8B97A5;")
        right_panel.addWidget(log_title)

        self.log_output = QTextEdit()
        self.log_output.setObjectName("logger")
        self.log_output.setReadOnly(True)
        self.log_output.append("[INFO] MI 运动想象独立测绘视口加载成功。")
        right_panel.addWidget(self.log_output, 1)

        main_layout.addLayout(right_panel, 3)

        self.start_btn.clicked.connect(self._start_mi_paradigm)
        self.stop_btn.clicked.connect(self._stop_mi_paradigm)
        self.train_btn.clicked.connect(self._train_mi_classifier)

        self.paradigm_timer = QTimer(self)
        self.paradigm_timer.timeout.connect(self._paradigm_step)
        self.current_trial = 0
        self.sub_step = 0
        self.directions_pool = ["LEFT", "RIGHT", "FEET", "TONGUE"]

    def _start_mi_paradigm(self):
        self.current_trial = 1
        self.sub_step = 0
        self.progress_bar.setValue(0)
        self.log_output.append(f"[START] >>> 启动独立测试实验范式，总计: {self.spin_trials.value()} 试次。")
        self._enter_ready_step()

    def _stop_mi_paradigm(self):
        self.paradigm_timer.stop()
        self.visual_cue.update_cue("IDLE")
        self.log_output.append("[HALT] 实验已被操作员终止。")

    def _enter_ready_step(self):
        if self.current_trial > self.spin_trials.value():
            self._end_mi_paradigm()
            return
        self.sub_step = 0
        self.trial_lbl.setText(f"当前进度: Trial {self.current_trial} / {self.spin_trials.value()}")
        self.visual_cue.update_cue("READY")
        self.log_output.append(f"[TRIAL {self.current_trial}] 十字准备闪烁中...")
        self.paradigm_timer.start(self.spin_ready.value() * 1000)

    def _enter_cue_step(self):
        self.sub_step = 1
        target_dir = random.choice(self.directions_pool)
        self.visual_cue.update_cue("CUE", target_dir)
        self.log_output.append(f"[TRIAL {self.current_trial}] 动作诱导: 方向 [{target_dir}] 想象开始。")
        self.paradigm_timer.start(self.spin_cue.value() * 1000)

    def _enter_rest_step(self):
        self.sub_step = 2
        self.visual_cue.update_cue("REST")
        self.log_output.append(f"[TRIAL {self.current_trial}] 进入空闲休息区。")
        total_percent = int((self.current_trial / self.spin_trials.value()) * 100)
        self.progress_bar.setValue(total_percent)
        self.paradigm_timer.start(self.spin_rest.value() * 1000)

    def _paradigm_step(self):
        if self.sub_step == 0:
            self._enter_cue_step()
        elif self.sub_step == 1:
            self._enter_rest_step()
        elif self.sub_step == 2:
            self.current_trial += 1
            self._enter_ready_step()

    def _end_mi_paradigm(self):
        self.paradigm_timer.stop()
        self.visual_cue.update_cue("IDLE")
        self.log_output.append("[SUCCESS] 数据采集测试完毕，已生成离线缓存矩阵。")

    def _train_mi_classifier(self):
        if self.progress_bar.value() < 100:
            self.log_output.append("[WARNING] 请先开启实验并跑满全部试次。")
            return
        self.log_output.append("[MATH] 正在建立带通滤波与特征映射，优化分类超平面...")
        self.model_status.setText("MI分类器状态: 求解中...")
        self.model_status.setStyleSheet("color: #F6C667;")
        QTimer.singleShot(1500, self._on_train_completed)

    def _on_train_completed(self):
        self.model_status.setText("MI分类器状态: 已就绪")
        self.model_status.setStyleSheet("color: #67E8B9;")
        self.log_output.append("[SUCCESS] 测试 SVM 分类模型固化成功！")


# ==========================================
# 独立调测主窗口外壳
# ==========================================
class MITestMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("运动想象 (MI) 采集控制台 - 独立调测外壳")
        self.resize(950, 600)
        self.setStyleSheet(MI_TEST_QSS)

        # 将MI组件作为核心展示中央窗口
        self.mi_panel = MIStageWidget()
        self.setCentralWidget(self.mi_panel)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MITestMainWindow()
    window.show()
    sys.exit(app.exec_())
