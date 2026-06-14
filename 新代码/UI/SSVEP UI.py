# -*- coding: utf-8 -*-
"""
脑机接口混合控制系统 - 阶段一：SSVEP 采集与训练工作台
"""
from __future__ import annotations
import sys
import time
from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QLabel, QFrame, QProgressBar,
    QTableWidget, QTableWidgetItem, QTextEdit, QGridLayout, QGroupBox
)

# ==========================================
# 核心 QSS 科技暗黑风格样式表
# ==========================================
COMPACT_CYBERPUNK_QSS = """
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
    color: #8B97A5;
    text-align: left;
    padding: 12px 20px;
    font-size: 14px;
    border-left: 3px solid transparent;
}
QPushButton#navBtn:hover {
    background-color: #1A222D;
    color: #FFFFFF;
}
QPushButton#navBtn[active="true"] {
    background-color: #151B24;
    color: #A9F5D0;
    border-left: 3px solid #67E8B9;
    font-weight: bold;
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
QPushButton#actionBtn:pressed {
    background-color: #1A2430;
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
QTableWidget {
    background-color: #0D1117;
    border: 1px solid #1F2631;
    gridline-color: #1F2631;
    color: #D2DAE5;
}
QHeaderView::section {
    background-color: #1A222D;
    color: #8B97A5;
    padding: 6px;
    border: 1px solid #1F2631;
    font-weight: bold;
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
# 自定义组件：Pipeline 流程环形进度条
# ==========================================
class PipelineProgressWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(140, 140)
        self.current_stage = 1  # 1: SSVEP, 2: MI, 3: Control
        self.stage_percent = 0.0

    def set_stage_progress(self, stage: int, percent: float):
        self.current_stage = stage
        self.stage_percent = max(0.0, min(100.0, percent))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        side = min(width, height)
        rect = QRectF((width - side) / 2 + 10, (height - side) / 2 + 10, side - 20, side - 20)

        # 1. 绘制暗色底环
        painter.setPen(QPen(QColor("#1F2631"), 8, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, 0 * 16, 360 * 16)

        # 2. 计算并绘制当前激活阶段的业务进度
        # 将3个步骤映射到360度空间中
        total_angle = (self.current_stage - 1) * 120 + (self.stage_percent / 100.0) * 120
        painter.setPen(QPen(QColor("#67E8B9"), 8, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, 90 * 16, -int(total_angle * 16))

        # 3. 绘制中心中心文本标签
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.setPen(QPen(QColor("#A9F5D0")))
        stage_names = {1: "SSVEP 阶段", 2: "MI 阶段", 3: "实时控制中"}
        painter.drawText(rect, Qt.AlignCenter, f"{stage_names.get(self.current_stage, '')}\n{int(self.stage_percent)}%")


# ==========================================
# 核心刺激组件：SSVEP 高频多靶区闪烁色块矩阵
# ==========================================
class SSVEPStimulusGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.is_flashing = False

        # 配置4个靶区的标称频率
        self.targets = [
            {"freq": 9, "name": "目标 1 (9Hz)", "state": True, "rect": QRectF()},
            {"freq": 11, "name": "目标 2 (11Hz)", "state": True, "rect": QRectF()},
            {"freq": 13, "name": "目标 3 (13Hz)", "state": True, "rect": QRectF()},
            {"freq": 15, "name": "目标 4 (15Hz)", "state": True, "rect": QRectF()}
        ]

        # 使用高精度定时器驱动刷新 (5ms 步长模拟高频采样基准)
        self.flash_timer = QTimer(self)
        self.flash_timer.setInterval(5)
        self.flash_timer.timeout.connect(self._on_tick)
        self.start_time = 0.0

    def start_flashing(self):
        self.is_flashing = True
        self.start_time = time.time()
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
        elapsed = time.time() - self.start_time
        # 方波闪烁公式：根据各自频率和流逝时间决定亮灭
        for t in self.targets:
            cycle = 1.0 / t["freq"]
            t["state"] = (int(elapsed / (cycle / 2.0)) % 2 == 0)
        self.update()

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        margin_x, margin_y = 20, 20
        box_w = (w - margin_x * 3) / 2
        box_h = (h - margin_y * 3) / 2

        # 划分2x2方阵空间
        self.targets[0]["rect"] = QRectF(margin_x, margin_y, box_w, box_h)
        self.targets[1]["rect"] = QRectF(margin_x * 2 + box_w, margin_y, box_w, box_h)
        self.targets[2]["rect"] = QRectF(margin_x, margin_y * 2 + box_h, box_w, box_h)
        self.targets[3]["rect"] = QRectF(margin_x * 2 + box_w, margin_y * 2 + box_h, box_w, box_h)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 填充极黑背景以强化视觉刺激对比度
        painter.fillRect(self.rect(), QColor("#080B0F"))

        for t in self.targets:
            r = t["rect"]
            if t["state"]:
                # 激活状态：耀眼的高亮高对比色
                painter.fillRect(r, QColor("#FFFFFF"))
                painter.setPen(QPen(QColor("#67E8B9"), 2))
                painter.drawRect(r)
                painter.setPen(QPen(QColor("#0B0E13")))
            else:
                # 熄灭状态：暗色调背景
                painter.fillRect(r, QColor("#151B24"))
                painter.setPen(QPen(QColor("#252E3C"), 1))
                painter.drawRect(r)
                painter.setPen(QPen(QColor("#8B97A5")))

            # 渲染频率说明文字
            painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
            painter.drawText(r, Qt.AlignCenter, t["name"])


# ==========================================
# 第一阶段主视图：SSVEP 采集与训练控制台
# ==========================================
class SSVEPStageWidget(QWidget):
    pipeline_updated = pyqtSignal(int, float)  # 向上级主窗口投递业务链状态的信号
    stage_completed = pyqtSignal()  # SSVEP阶段全部完成信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ---------- 左侧面板：控制区与指标展示 ----------
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)

        # 1. 硬件连接配置卡片
        conn_card = QFrame()
        conn_card.setObjectName("cardPanel")
        conn_layout = QVBoxLayout(conn_card)
        conn_title = QLabel("脑电设备配置")
        conn_title.setObjectName("panelTitle")
        conn_layout.addWidget(conn_title)

        conn_info = QLabel("板卡型号: Cyton (BoardId 2)\n端口状态: 串口已就绪 (COM3)\n采样频率: 250 Hz")
        conn_info.setStyleSheet("color: #A4B1CD; line-height: 20px;")
        conn_layout.addWidget(conn_info)
        left_panel.addWidget(conn_card)

        # 2. 数据采集控制面板
        collect_card = QFrame()
        collect_card.setObjectName("cardPanel")
        collect_layout = QVBoxLayout(collect_card)
        collect_title = QLabel("SSVEP 脑电数据采集")
        collect_title.setObjectName("panelTitle")
        collect_layout.addWidget(collect_title)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        collect_layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        self.start_collect_btn = QPushButton("开始采集")
        self.start_collect_btn.setObjectName("accentBtn")
        self.stop_collect_btn = QPushButton("停止")
        self.stop_collect_btn.setObjectName("actionBtn")
        btn_layout.addWidget(self.start_collect_btn)
        btn_layout.addWidget(self.stop_collect_btn)
        collect_layout.addLayout(btn_layout)
        left_panel.addWidget(collect_card)

        # 3. 算法离线预训练面板
        train_card = QFrame()
        train_card.setObjectName("cardPanel")
        train_layout = QVBoxLayout(train_card)
        train_title = QLabel("FBCCA 分类器离线训练")
        train_title.setObjectName("panelTitle")
        train_layout.addWidget(train_title)

        self.train_btn = QPushButton("一键提取特征与训练")
        self.train_btn.setObjectName("actionBtn")
        train_layout.addWidget(self.train_btn)

        self.model_status = QLabel("模型状态: 未训练")
        self.model_status.setStyleSheet("color: #FF8F8F; font-weight: bold;")
        train_layout.addWidget(self.model_status)
        left_panel.addWidget(train_card)

        main_layout.addLayout(left_panel, 2)

        # ---------- 右侧面板：刺激源呈现与日志数据监控 ----------
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)

        stim_title = QLabel("视觉诱发高频刺激阵列 (SSVEP 靶区)")
        stim_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #569AFF;")
        right_panel.addWidget(stim_title)

        # 实例化精细闪烁色块组件
        self.stimulus_grid = SSVEPStimulusGrid()
        right_panel.addWidget(self.stimulus_grid, 3)

        log_title = QLabel("系统实时流水日志")
        log_title.setStyleSheet("font-size: 12px; color: #8B97A5;")
        right_panel.addWidget(log_title)

        self.log_output = QTextEdit()
        self.log_output.setObjectName("logger")
        self.log_output.setReadOnly(True)
        self.log_output.append("[INFO] 混合脑机接口系统工作台启动成功。")
        self.log_output.append("[SYSTEM] 当前控制链定位: 阶段一 (SSVEP数据采集训练).")
        right_panel.addWidget(self.log_output, 1)

        main_layout.addLayout(right_panel, 3)

        # 绑定模拟交互信号链（后续可由底层硬件线程触发替换）
        self.start_collect_btn.clicked.connect(self._simulate_collection)
        self.train_btn.clicked.connect(self._simulate_training)

        # 初始化定时器用于模拟采集过程中的进度条
        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self._on_collect_tick)
        self.sim_counter = 0

    def _simulate_collection(self):
        self.log_output.append("[BUSY] 正在开启脑电数据物理采集板卡，启动高频视觉刺激源...")
        self.stimulus_grid.start_flashing()
        self.sim_counter = 0
        self.progress_bar.setValue(0)
        self.sim_timer.start(100)  # 100ms步进

    def _on_collect_tick(self):
        self.sim_counter += 2
        self.progress_bar.setValue(self.sim_counter)
        self.pipeline_updated.emit(1, self.sim_counter * 0.5)  # 采集占SSVEP阶段的50%进度

        if self.sim_counter >= 100:
            self.sim_timer.stop()
            self.stimulus_grid.stop_flashing()
            self.log_output.append("[SUCCESS] SSVEP 4频点靶区脑电数据采集序列完成，样本集已落盘。")

    def _simulate_training(self):
        if self.progress_bar.value() < 100:
            self.log_output.append("[WARNING] 请先执行完整的数据采集序列。")
            return
        self.log_output.append("[TRAIN] 正在并行提取多频点脑电特征空间，构建 CCA 空间滤波器...")
        self.model_status.setText("模型状态: 正在训练...")
        self.model_status.setStyleSheet("color: #F6C667;")

        # 快速模拟训练延时
        QTimer.singleShot(1500, self._on_train_done)

    def _on_train_done(self):
        self.model_status.setText("模型状态: 训练就绪 (Acc: 94.2%)")
        self.model_status.setStyleSheet("color: #67E8B9;")
        self.log_output.append("[SUCCESS] SSVEP 识别模型训练完成。前序通道已就绪！")
        self.pipeline_updated.emit(1, 100.0)
        # 激活阶段完成信号，解除下一阶段的封锁
        self.stage_completed.emit()


# ==========================================
# 系统总外壳主窗口
# ==========================================
class BCIWorkbenchWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("混合脑机接口全流程一体化控制工作台 (软著集成规范版)")
        self.resize(1100, 700)
        self.setStyleSheet(COMPACT_CYBERPUNK_QSS)
        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        outer_layout = QHBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # ---------- 1. 左侧常驻主控导航栏 ----------
        nav_bar = QFrame()
        nav_bar.setObjectName("navBar")
        nav_bar.setFixedWidth(200)
        nav_layout = QVBoxLayout(nav_bar)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(5)

        title = QLabel("BCI 系统工作台")
        title.setObjectName("navTitle")
        nav_layout.addWidget(title)

        # 嵌入状态环形拓扑图
        self.pipeline_monitor = PipelineProgressWidget()
        nav_layout.addWidget(self.pipeline_monitor, 0, Qt.AlignCenter)
        nav_layout.addSpacing(20)

        # 创建三大串联步骤的侧边栏按钮
        self.nav_btns = []
        stages_config = [
            ("1. SSVEP 采集与训练", True),
            ("2. MI 运动想象采集", False),  # 初始置灰，由状态机解锁
            ("3. 机械臂实时控制", False)  # 初始置灰
        ]

        for name, enabled in stages_config:
            btn = QPushButton(name)
            btn.setObjectName("navBtn")
            btn.setEnabled(enabled)
            btn.setProperty("active", "false")
            nav_layout.addWidget(btn)
            self.nav_btns.append(btn)

        nav_layout.addStretch()

        # 版权声明标签（软著申报规范要求）
        copyright_lbl = QLabel("© 2026 脑机接口实验室\n保留所有软著资产权利\nV1.0.0")
        copyright_lbl.setStyleSheet("color: #445161; font-size: 11px; padding: 15px; text-align: center;")
        nav_layout.addWidget(copyright_lbl)

        outer_layout.addWidget(nav_bar)

        # ---------- 2. 右侧核心多页面堆栈切换容器 ----------
        self.stage_container = QStackedWidget()

        # 页面一：SSVEP 工作面板
        self.ssvep_page = SSVEPStageWidget()
        self.stage_container.addWidget(self.ssvep_page)

        # 页面二：MI 占位面板（待后续下发合并）
        self.mi_placeholder = QLabel("【阶段二：MI 运动想象数据采集与训练】\n（等待SSVEP训练就绪后解锁导入）")
        self.mi_placeholder.setAlignment(Qt.AlignCenter)
        self.mi_placeholder.setStyleSheet("font-size: 16px; color: #8B97A5; background-color: #0E131A;")
        self.stage_container.addWidget(self.mi_placeholder)

        # 页面三：机械臂占位面板（待后续下发合并）
        self.robot_placeholder = QLabel("【阶段三：机械臂混合脑控多目标控制台】\n（请依次完成SSVEP与MI的预训练）")
        self.robot_placeholder.setAlignment(Qt.AlignCenter)
        self.robot_placeholder.setStyleSheet("font-size: 16px; color: #8B97A5; background-color: #0E131A;")
        self.stage_container.addWidget(self.robot_placeholder)

        outer_layout.addWidget(self.stage_container, 1)

        # ---------- 3. 业务状态信号槽联动 ----------
        self.ssvep_page.pipeline_updated.connect(self.pipeline_monitor.set_stage_progress)
        self.ssvep_page.stage_completed.connect(self._unlock_mi_stage)

        # 初始化高亮第一个侧边栏
        self._update_nav_highlight(0)
        self.nav_btns[0].clicked.connect(lambda: self._switch_page(0))
        self.nav_btns[1].clicked.connect(lambda: self._switch_page(1))
        self.nav_btns[2].clicked.connect(lambda: self._switch_page(2))

    def _update_nav_highlight(self, index: int):
        for i, btn in enumerate(self.nav_btns):
            btn.setProperty("active", "true" if i == index else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _switch_page(self, index: int):
        self.stage_container.setCurrentIndex(index)
        self._update_nav_highlight(index)

    @pyqtSlot()
    def _unlock_mi_stage(self):
        """当SSVEP部分训练彻底完毕，状态机下发解锁MI采集权限"""
        self.nav_btns[1].setEnabled(True)
        self.ssvep_page.log_output.append("[SYSTEM-PIPELINE] >>> 状态机跳转：阶段二 [MI 运动想象采集] 已解锁。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BCIWorkbenchWindow()
    window.show()
    sys.exit(app.exec_insitu() if hasattr(app, "exec_insitu") else app.exec_())
