# -*- coding: utf-8 -*-
"""
ui_main.py - 综合脑机接口工作台（内嵌决策窗频闪块放大版）
修改点：
1. 🔥【核心设计】：将机械臂内嵌视频的左右频闪块尺寸适度放大（150x160 -> 180x200），比原本大一点点。
2. 同步微调边缘贴边逻辑，确保放大后块依然固定在左右边缘最死角，绝不阻挡机械臂工作视野。
3. 全面保持浅蓝高对比度科技换装。
"""
import sys
import random
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QRectF, QPointF
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFrame, QLabel, QStackedWidget, QPushButton, QComboBox, QGroupBox, QProgressBar, QFormLayout
)
from PyQt5.QtGui import QColor, QKeyEvent

from ui_shared import CYBERPUNK_STYLE, PipelineProgressWidget
from ui_ssvep import SSVEPStageWidget
from ui_mi import MIStageWidget


class RobotControlStageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = None
        self.control_phase = "IDLE"
        self.phase_remaining_ms = 0
        self._init_ui()

        self.bci_clock = QTimer(self)
        self.bci_clock.setInterval(100)
        self.bci_clock.timeout.connect(self._bci_core_pulse_tick)

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # 左侧精简控制面板
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)

        group_control = QGroupBox("混合脑控流水线总控")
        v_ctrl = QVBoxLayout(group_control)
        v_ctrl.setContentsMargins(15, 25, 15, 25)
        v_ctrl.setSpacing(15)

        self.lbl_run_status = QLabel("系统就绪：等待触发")
        self.lbl_run_status.setAlignment(Qt.AlignCenter)
        self.lbl_run_status.setStyleSheet("color: #0369A1; font-weight: bold; font-size: 16px;")
        v_ctrl.addWidget(self.lbl_run_status)

        self.pbar_countdown = QProgressBar()
        self.pbar_countdown.setValue(0)
        self.pbar_countdown.setFixedHeight(26)
        v_ctrl.addWidget(self.pbar_countdown)

        self.btn_master_start = QPushButton("启动混合脑控机器人任务")
        self.btn_master_start.setObjectName("accentBtn")
        self.btn_master_start.setMinimumHeight(45)
        self.btn_master_start.clicked.connect(self._start_integrated_flow)
        v_ctrl.addWidget(self.btn_master_start)

        left_panel.addWidget(group_control)
        left_panel.addStretch()
        layout.addLayout(left_panel, 2)

        # 右侧面板：内嵌视频及两侧大间距、大字频闪块
        cam_card = QFrame()
        cam_card.setObjectName("cameraCard")
        cc_l = QVBoxLayout(cam_card)
        cc_l.setContentsMargins(0, 0, 0, 0)

        self.video_container = QWidget()
        self.video_container.setStyleSheet("background: #0F172A;")

        from ui_widgets import RobotCameraWidget
        self.cam = RobotCameraWidget(self.video_container)

        self.popup_dialog = QFrame(self.video_container)
        self.popup_dialog.setStyleSheet("""
            QFrame { background-color: rgba(15, 23, 42, 230); border: 2px solid #0284C7; border-radius: 10px; }
            QLabel { color: #FFFFFF; font-weight: bold; font-size: 15px; }
        """)
        pd_lay = QVBoxLayout(self.popup_dialog)
        self.lbl_popup_title = QLabel("BCI 状态决策激活")
        self.lbl_popup_title.setAlignment(Qt.AlignCenter)
        self.lbl_popup_desc = QLabel("请凝视两侧闪烁目标执行脑控决策")
        self.lbl_popup_desc.setAlignment(Qt.AlignCenter)
        self.lbl_popup_desc.setStyleSheet("color: #94A3B8; font-size: 13px;")
        pd_lay.addWidget(self.lbl_popup_title)
        pd_lay.addWidget(self.lbl_popup_desc)
        self.popup_dialog.hide()

        self.flash_box_8hz = QLabel("", self.video_container)
        self.flash_box_15hz = QLabel("", self.video_container)

        for lbl in [self.flash_box_8hz, self.flash_box_15hz]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.hide()

        cc_l.addWidget(self.video_container, 1)
        layout.addWidget(cam_card, 4)

        self.tmr_rd_flash = QTimer(self)
        self.tmr_rd_flash.setInterval(40)
        self.tmr_rd_flash.timeout.connect(self._on_rd_flash_tick)
        self.rd_state = True

    def resizeEvent(self, event):
        """自适应算法调整：方块适度拉长、拉高，且位置进一步死死压在屏幕左右两侧死角"""
        super().resizeEvent(event)
        vw, vh = self.video_container.width(), self.video_container.height()
        if vw <= 0 or vh <= 0: return
        self.cam.setGeometry(0, 0, vw, vh)
        self.popup_dialog.setGeometry(int((vw - 340) / 2), int((vh - 120) / 2), 340, 120)

        # 🔥【关键重构点】：长宽由 150x160 调大至 180x200 像素，右侧贴边系数由 165 微调至 195，大间距完美避开中央
        b_w, b_h = 180, 200
        self.flash_box_8hz.setGeometry(15, int((vh - b_h) / 2), b_w, b_h)
        self.flash_box_15hz.setGeometry(vw - (b_w + 15), int((vh - b_h) / 2), b_w, b_h)

    def _on_rd_flash_tick(self):
        self.rd_state = not self.rd_state
        # 保持极高对比度交替消隐频闪
        self.flash_box_8hz.setStyleSheet(
            f"background: {'#FFFFFF' if self.rd_state else '#000000'}; color: {'#000000' if self.rd_state else '#0284C7'}; font-weight: 900; font-size: 21px; border: 3px solid #0284C7; border-radius: 8px;")
        if random.choice([True, False]):
            self.flash_box_15hz.setStyleSheet(
                "background: #FFFFFF; color: #000000; font-weight: 900; font-size: 21px; border: 3px solid #F43F5E; border-radius: 8px;")
        else:
            self.flash_box_15hz.setStyleSheet(
                "background: #000000; color: #F43F5E; font-weight: 900; font-size: 21px; border: 3px solid #F43F5E; border-radius: 8px;")

    def _start_integrated_flow(self):
        self.btn_master_start.setEnabled(False)
        self._enter_mi_move_stage_1()
        self.bci_clock.start()

    def _enter_mi_move_stage_1(self):
        self.control_phase = "MI_MOVE_1"
        self.phase_remaining_ms = 10000
        self.pbar_countdown.setMaximum(10000)
        self.popup_dialog.hide()
        self.flash_box_8hz.hide()
        self.flash_box_15hz.hide()
        self.tmr_rd_flash.stop()
        self.lbl_run_status.setText("阶段1：MI 引导平移移动中...")

    def _bci_core_pulse_tick(self):
        if self.control_phase in ["MI_MOVE_1", "MI_MOVE_2"]:
            self.phase_remaining_ms -= 100
            self.pbar_countdown.setValue(self.phase_remaining_ms)
            self.pbar_countdown.setFormat(f"运动意图捕获中: {self.phase_remaining_ms / 1000.0:.1f}s")
            if self.phase_remaining_ms <= 0:
                if self.control_phase == "MI_MOVE_1":
                    self._enter_decision_stage_1()
                else:
                    self._enter_decision_stage_2()

    def _enter_decision_stage_1(self):
        self.control_phase = "DECIDE_1"
        self.lbl_run_status.setText("决策点1：等待状态指令下发")
        self.lbl_popup_title.setText("【阶段1 移动引导结束】")
        self.lbl_popup_desc.setText("请凝视两侧目标块下发决策命令")
        self.flash_box_8hz.setText("确认抓取\n(进入阶段2)")
        self.flash_box_15hz.setText("继续移动\n(额外加10秒)")
        self.popup_dialog.show()
        self.flash_box_8hz.show()
        self.flash_box_15hz.show()
        self.tmr_rd_flash.start()

    def mousePressEvent(self, event):
        if self.control_phase == "IDLE": return
        if event.button() == Qt.LeftButton:
            self._inject_cca_trigger_8hz()
        elif event.button() == Qt.RightButton:
            self._inject_cca_trigger_15hz()

    def _inject_cca_trigger_8hz(self):
        if self.control_phase == "DECIDE_1":
            self._enter_ssvep_grab_stage()
        elif self.control_phase == "SSVEP_GRAB":
            self._execute_physical_grab()
        elif self.control_phase == "DECIDE_2":
            self._execute_physical_release()

    def _inject_cca_trigger_15hz(self):
        if self.control_phase == "DECIDE_1":
            self._enter_mi_move_stage_1()
        elif self.control_phase == "SSVEP_GRAB":
            self._enter_mi_move_stage_1()
        elif self.control_phase == "DECIDE_2":
            self._enter_mi_move_stage_2()

    def _enter_ssvep_grab_stage(self):
        self.control_phase = "SSVEP_GRAB"
        self.cam.active_id = "2"
        self.lbl_run_status.setText("阶段2：SSVEP 抓取状态判别中...")
        self.lbl_popup_title.setText("【阶段2：CCA 抓取确认】")
        self.flash_box_8hz.setText("确认抓取\n当前目标物")
        self.flash_box_15hz.setText("放弃抓取\n返回阶段1")

    def _execute_physical_grab(self):
        self.popup_dialog.hide()
        self.flash_box_8hz.hide()
        self.flash_box_15hz.hide()
        self.tmr_rd_flash.stop()
        self.lbl_run_status.setText("⚙️ 真空吸嘴闭合连贯执行中...")
        QTimer.singleShot(1200, self._enter_mi_move_stage_2)

    def _enter_mi_move_stage_2(self):
        self.control_phase = "MI_MOVE_2"
        self.phase_remaining_ms = 10000
        self.pbar_countdown.setMaximum(10000)
        self.popup_dialog.hide()
        self.flash_box_8hz.hide()
        self.flash_box_15hz.hide()
        self.tmr_rd_flash.stop()
        self.lbl_run_status.setText("阶段3：MI 带载引导搬运中...")

    def _enter_decision_stage_2(self):
        self.control_phase = "DECIDE_2"
        self.lbl_run_status.setText("最终决策点：等待物料落地指令")
        self.lbl_popup_title.setText("【阶段3 带载移动结束】")
        self.lbl_popup_desc.setText("请注视两侧频闪块进行最终卸料判断")
        self.flash_box_8hz.setText("确认放下\n小木块")
        self.flash_box_15hz.setText("继续带载\n额外加10秒")
        self.popup_dialog.show()
        self.flash_box_8hz.show()
        self.flash_box_15hz.show()
        self.tmr_rd_flash.start()

    def _execute_physical_release(self):
        self.bci_clock.stop()
        self.tmr_rd_flash.stop()
        self.control_phase = "TASK_DONE"
        self.lbl_run_status.setText("✅ 全控制路径圆满通关闭环！")
        self.lbl_popup_title.setText("任务顺利达成")
        self.lbl_popup_desc.setText("小木块已安全释放，重置控制总线。")
        self.flash_box_8hz.hide()
        self.flash_box_15hz.hide()
        self.cam.active_id = ""
        QTimer.singleShot(2500, lambda: (self.popup_dialog.hide(), self.btn_master_start.setEnabled(True),
                                         self.lbl_run_status.setText("中枢就绪：等待点击启动按钮")))

    def _stop_current_task_safely(self):
        self.bci_clock.stop()
        self.tmr_rd_flash.stop()
        self.control_phase = "IDLE"
        self.popup_dialog.hide()
        self.flash_box_8hz.hide()
        self.flash_box_15hz.hide()
        self.cam.active_id = ""
        self.btn_master_start.setEnabled(True)
        self.pbar_countdown.setValue(0)
        self.lbl_run_status.setText("中枢已安全复位重置")


class BCIIntegratedWorkbenchWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多模态混合脑机接口一体化控制工作台 (浅蓝极简按键修复版)")
        self.resize(1380, 780)
        self.setStyleSheet(CYBERPUNK_STYLE)

        from ui_widgets import ParticipantDisplayWindow
        self.participant_window = ParticipantDisplayWindow()
        self.participant_window.stop_requested.connect(self._on_manual_stop)

        self._init_ui()
        self.ssvep_page.main_window = self
        self.mi_page.main_window = self
        self.robot_page.main_window = self

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        header_bar = QFrame()
        header_bar.setStyleSheet("background: #F0F9FF; border-bottom: 1px solid #7DD3FC;")
        header_bar.setFixedHeight(50)
        hl = QHBoxLayout(header_bar)
        hl.setContentsMargins(20, 0, 20, 0)
        hl.setSpacing(15)

        logo = QLabel("🧬 SYSTEM INTERFACE")
        logo.setStyleSheet("color: #0369A1; font-weight: 900; font-size: 15px; margin-right: 20px;")
        hl.addWidget(logo)

        self.nav_btns = []
        nav_configs = [("SSVEP 采集与验证", 0), ("MI 运动想象控制", 1), ("混合脑控机械臂", 2)]
        for name, page_index in nav_configs:
            btn = QPushButton(name)
            btn.setObjectName("navBtn")
            btn.clicked.connect(lambda checked, idx=page_index: self._on_tab_changed(idx))
            hl.addWidget(btn)
            self.nav_btns.append(btn)

        hl.addStretch()
        outer_layout.addWidget(header_bar)

        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(10, 10, 10, 10)
        body_layout.setSpacing(15)

        self.container = QStackedWidget()
        self.ssvep_page = SSVEPStageWidget()
        self.container.addWidget(self.ssvep_page)
        self.mi_page = MIStageWidget()
        self.container.addWidget(self.mi_page)
        self.robot_page = RobotControlStageWidget()
        self.container.addWidget(self.robot_page)
        body_layout.addWidget(self.container, 1)

        self.eeg_panel = QFrame()
        self.eeg_panel.setStyleSheet("background: #F0F9FF; border: 1px solid #7DD3FC; border-radius: 6px;")
        ep_l = QVBoxLayout(self.eeg_panel)
        ep_l.setContentsMargins(15, 15, 15, 15)

        lbl_eeg_title = QLabel("物理层实时电生理信号同步示波器")
        lbl_eeg_title.setStyleSheet("color: #0369A1; font-weight: bold; font-size: 15px;")
        ep_l.addWidget(lbl_eeg_title)

        from ui_widgets import RealtimeEEGPreviewWidget
        self.realtime_wave_canvas = RealtimeEEGPreviewWidget()
        ep_l.addWidget(self.realtime_wave_canvas, 1)

        toggle_lay = QHBoxLayout()
        self.btn_eeg_mode = QPushButton("时域波形图 (EEG)")
        self.btn_eeg_mode.setObjectName("accentBtn")
        self.btn_imp_mode = QPushButton("引脚接触阻抗 (IMP)")
        self.btn_imp_mode.setObjectName("actionBtn")
        toggle_lay.addWidget(self.btn_eeg_mode)
        toggle_lay.addWidget(self.btn_imp_mode)
        ep_l.addLayout(toggle_lay)
        body_layout.addWidget(self.eeg_panel, 1)

        outer_layout.addLayout(body_layout, 1)

        self.monitor = PipelineProgressWidget()
        outer_layout.addWidget(self.monitor)

        self.ssvep_page.pipeline_updated.connect(self.monitor.set_stage_progress)
        self.mi_page.pipeline_updated.connect(self.monitor.set_stage_progress)
        self.btn_eeg_mode.clicked.connect(lambda: self._switch_canvas_mode("EEG"))
        self.btn_imp_mode.clicked.connect(lambda: self._switch_canvas_mode("IMP"))

        self._on_tab_changed(0)

    def _on_tab_changed(self, index):
        self.container.setCurrentIndex(index)
        if index == 2:
            self.eeg_panel.hide()
            self.monitor.set_stage_progress(3, 50.0)
        else:
            self.eeg_panel.show()
        for i, btn in enumerate(self.nav_btns):
            btn.setProperty("active", "true" if i == index else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _switch_canvas_mode(self, mode_type):
        self.realtime_wave_canvas.set_mode(mode_type)
        self.btn_eeg_mode.setObjectName("accentBtn" if mode_type == "EEG" else "actionBtn")
        self.btn_imp_mode.setObjectName("accentBtn" if mode_type == "IMP" else "actionBtn")
        for b in [self.btn_eeg_mode, self.btn_imp_mode]:
            b.style().unpolish(b)
            b.style().polish(b)

    def _on_manual_stop(self):
        current_view_idx = self.container.currentIndex()
        if current_view_idx == 0:
            self.ssvep_page._stop_ssvep_paradigm()
        elif current_view_idx == 1:
            self.mi_page._stop_machinery("⚠️ 运动想象采集范式已被手动停止存盘。")
        self.participant_window.hide()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            current_view_idx = self.container.currentIndex()
            if current_view_idx == 2:
                self.robot_page._stop_current_task_safely()
            else:
                self._on_manual_stop()
            event.accept()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BCIIntegratedWorkbenchWindow()
    window.show()
    sys.exit(app.exec_())
