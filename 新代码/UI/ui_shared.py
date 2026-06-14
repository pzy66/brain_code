# -*- coding: utf-8 -*-
"""
ui_shared.py - 全局共享样式表与进度组件（高清晰浅蓝色美化版）
优化点：
1. 遵照用户指示，全面采用浅蓝色高感官科技感背景色。
2. 字体字号整体拉粗放大，全面复写前台控件，保障在亮色背景下字字锐利。
"""
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QPainter, QFont, QPen, QLinearGradient, QBrush
from PyQt5.QtWidgets import QWidget

# 🔥 核心皮肤重构：主背景换为高质感浅蓝色（#E0F2FE），字体颜色深度强化为深海军蓝（#0F172A）以确保清晰度
CYBERPUNK_STYLE = """
QMainWindow { background-color: #E0F2FE; } /* 窗体最底层更换为浅蓝色背景 */
QWidget { color: #0F172A; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; font-size: 15px; }

/* 顶部书签选项卡 TabBar */
QPushButton#navBtn {
    background-color: transparent; border: none; color: #334155;
    padding: 0px 20px; font-size: 15px; font-weight: bold; height: 50px;
}
QPushButton#navBtn:hover { background-color: #BAE6FD; color: #0F172A; }
QPushButton#navBtn[active="true"] {
    background-color: #F0F9FF; color: #0369A1;
    border-bottom: 4px solid #0284C7; /* 蓝天深蓝高亮底条 */
}

/* 分组框与面板（浅色卡片化设计） */
QFrame#cardPanel, QFrame#stateBar, QFrame#cameraCard, QFrame#poseCard, QFrame#flowCard {
    background-color: #F0F9FF; border: 1px solid #7DD3FC; border-radius: 6px;
}
QGroupBox {
    background-color: #F0F9FF; border: 1px solid #7DD3FC; border-radius: 6px;
    margin-top: 15px; padding-top: 15px; color: #0369A1; font-weight: bold; font-size: 16px;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 5px; }

QLabel { color: #0F172A; font-size: 15px; }

/* 强化亮色系表单：底色适度微暗，文字强制使用墨黑色，彻底杜绝看不清的问题 */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #FFFFFF;
    border: 1px solid #0284C7;
    color: #0F172A; /* 强制纯深色字 */
    padding: 5px 8px;
    border-radius: 4px;
    font-size: 15px;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background-color: #E2E8F0;
    color: #64748B;
}

/* 下拉项展开弹窗样式对齐 */
QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border: 1px solid #0284C7;
    color: #0F172A;
    selection-background-color: #7DD3FC;
    selection-color: #0369A1;
}

/* 按钮样式微调 */
QPushButton#actionBtn { background-color: #E0F2FE; border: 1px solid #0284C7; color: #0369A1; padding: 8px 16px; border-radius: 4px; font-weight: bold; font-size: 15px; }
QPushButton#actionBtn:hover:enabled { background-color: #BAE6FD; }
QPushButton#accentBtn { background-color: #0284C7; border: 1px solid #0369A1; color: #FFFFFF; padding: 8px 16px; border-radius: 4px; font-weight: bold; font-size: 15px; }
QPushButton#accentBtn:hover:enabled { background-color: #0369A1; }

/* 进度条 */
QProgressBar { border: 1px solid #7DD3FC; background-color: #FFFFFF; text-align: center; color: #0F172A; font-weight: bold; border-radius: 4px; font-size: 14px; }
QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7DD3FC, stop:1 #0284C7); }

/* 机械臂选块快捷方块浅色适配 */
QPushButton[blockState="pending"] { background-color: #F1F5F9; border: 1px solid #CBD5E1; color: #334155; border-radius: 4px; padding: 6px; font-size: 15px; }
QPushButton[blockState="active"] { background-color: #E0F2FE; border: 2px solid #0284C7; color: #0369A1; font-weight: bold; border-radius: 4px; padding: 5px; font-size: 15px; }
"""


class PipelineProgressWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(55)
        self.current_stage = 1
        self.stage_percent = 0.0

    def set_stage_progress(self, stage: int, percent: float):
        self.current_stage = stage
        self.stage_percent = max(0.0, min(100.0, percent))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 底部横条的底色换为温和的浅灰蓝，与主屏相呼应
        painter.fillRect(self.rect(), QColor("#F1F5F9"))

        margin, spacing = 15, 10
        total_w = self.width() - (margin * 2) - (spacing * 2)
        block_w = total_w / 3.0
        block_h = 14.0

        stages_info = [(1, "阶段一：SSVEP 采集与验证"), (2, "阶段二：MI 感觉运动想象"), (3, "阶段三：混合脑控机械臂")]

        for i, (stage_id, label_text) in enumerate(stages_info):
            rx = margin + i * (block_w + spacing)
            ry_bar = 28.0
            bar_rect = QRectF(rx, ry_bar, block_w, block_h)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor("#E2E8F0")))
            painter.drawRoundedRect(bar_rect, 4, 4)

            if self.current_stage > stage_id:
                # 已完成段：饱满的亮深蓝填充
                grad = QLinearGradient(bar_rect.topLeft(), bar_rect.topRight())
                grad.setColorAt(0.0, QColor("#0284C7"));
                grad.setColorAt(1.0, QColor("#0369A1"))
                painter.setBrush(QBrush(grad));
                painter.drawRoundedRect(bar_rect, 4, 4)
            elif self.current_stage == stage_id:
                # 进行段：高亮的水蓝色进度推进
                current_w = block_w * (self.stage_percent / 100.0)
                if current_w > 4:
                    prog_rect = QRectF(rx, ry_bar, current_w, block_h)
                    grad = QLinearGradient(prog_rect.topLeft(), prog_rect.topRight())
                    grad.setColorAt(0.0, QColor("#38BDF8"));
                    grad.setColorAt(1.0, QColor("#0284C7"))
                    painter.setBrush(QBrush(grad));
                    painter.drawRoundedRect(prog_rect, 4, 4)

            painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
            if self.current_stage == stage_id:
                painter.setPen(QPen(QColor("#0369A1")))  # 当前段强制使用深色高亮
                disp_text = f"{label_text} ({int(self.stage_percent)}%)"
            elif self.current_stage > stage_id:
                painter.setPen(QPen(QColor("#64748B")))
                disp_text = f"{label_text} (100%)"
            else:
                painter.setPen(QPen(QColor("#94A3B8")))
                disp_text = f"{label_text} (等待)"

            painter.drawText(QRectF(rx, 4.0, block_w, 22.0), Qt.AlignCenter, disp_text)
