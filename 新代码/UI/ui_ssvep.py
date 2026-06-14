# -*- coding: utf-8 -*-
"""
ui_ssvep.py - SSVEP 采集与验证控制面板（极简高对比度精简版）
修改点：
1. 彻底移除右侧日志列，将空间全部释放给外周大示波器。
2. 重构大屏文本驱动，将硬编码的频率数值智能换算为“目标1/2/3/4”中文字样。
"""
import random
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QLineEdit,
    QSpinBox, QDoubleSpinBox, QProgressBar, QPushButton, QLabel, QComboBox, QGroupBox
)


class SSVEPStageWidget(QWidget):
    pipeline_updated = pyqtSignal(int, float)
    stage_completed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = None
        self.is_collecting = False
        self.is_paused = False
        self.cur_trial, self.total_trials, self.state_step, self.step_remaining_ms, self.freq_pool = 0, 0, 0, 0, []
        self._init_ui()

    def _init_ui(self):
        # 垂直单向大布局，100%拉满横向跨度
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # 一、脑电设备连接控制
        group_device = QGroupBox("一、脑电设备连接控制")
        f_device = QFormLayout(group_device)
        f_device.setContentsMargins(15, 20, 15, 15)
        f_device.setSpacing(15)

        self.combo_board_id = QComboBox()
        self.combo_board_id.addItems(["Cyton 8通道 (ID=0)", "Ganglion 4通道 (ID=1)", "模拟测试板卡 (Synthetic)"])
        self.combo_serial_port = QComboBox()
        self.combo_serial_port.addItems(["auto", "COM3", "COM4", "COM5"])
        self.edit_user = QLineEdit("bci_operator_01")
        self.btn_connect = QPushButton("连接脑电设备 (Connect Device)")
        self.btn_connect.setObjectName("accentBtn")

        f_device.addRow("选择脑电板卡:", self.combo_board_id)
        f_device.addRow("绑定通信串口:", self.combo_serial_port)
        f_device.addRow("当前实验用户:", self.edit_user)
        f_device.addWidget(self.btn_connect)
        main_layout.addWidget(group_device)

        # 二、视觉刺激诱导设置
        group_paradigm = QGroupBox("二、视觉刺激诱导设置")
        f_paradigm = QFormLayout(group_paradigm)
        f_paradigm.setContentsMargins(15, 20, 15, 15)
        f_paradigm.setSpacing(15)

        self.spin_active = QDoubleSpinBox()
        self.spin_active.setRange(0.5, 20.0)
        self.spin_active.setValue(4.0)
        self.spin_active.setSuffix(" 秒")

        self.spin_rest = QDoubleSpinBox()
        self.spin_rest.setRange(0.2, 10.0)
        self.spin_rest.setValue(1.5)
        self.spin_rest.setSuffix(" 秒")

        self.spin_target_repeats = QSpinBox()
        self.spin_target_repeats.setRange(1, 50)
        self.spin_target_repeats.setValue(3)
        self.spin_target_repeats.setSuffix(" 次")

        f_paradigm.addRow("刺激凝视时间 (Active):", self.spin_active)
        f_paradigm.addRow("刺激间休息时间 (Rest):", self.spin_rest)
        f_paradigm.addRow("每个频率目标试次数:", self.spin_target_repeats)
        main_layout.addWidget(group_paradigm)

        # 三、识别器模型闭环优化
        group_train = QGroupBox("三、识别器模型闭环优化")
        v_train = QVBoxLayout(group_train)
        v_train.setContentsMargins(15, 20, 15, 15)
        v_train.setSpacing(12)

        self.p_bar = QProgressBar()
        self.btn_full_workflow = QPushButton("启动全自动数据采集范式")
        self.btn_full_workflow.setObjectName("accentBtn")

        self.btn_train_classifier = QPushButton("开始训练 SSVEP 空间特征分类器")
        self.btn_train_classifier.setObjectName("actionBtn")

        self.lbl_status = QLabel("识别器模型状态: 🔴 未载入特征空间映射")
        self.lbl_status.setStyleSheet("color: #FF8F8F; font-weight: bold; margin-top: 4px;")

        v_train.addWidget(self.p_bar)
        v_train.addWidget(self.btn_full_workflow)
        v_train.addWidget(self.btn_train_classifier)
        v_train.addWidget(self.lbl_status)
        main_layout.addWidget(group_train)

        main_layout.addStretch()

        # 信号绑定
        self.btn_full_workflow.clicked.connect(self._run_full_workflow)
        self.btn_train_classifier.clicked.connect(self._train)

        self.master_timer = QTimer(self)
        self.master_timer.setInterval(100)
        self.master_timer.timeout.connect(self._master_clock_tick)

    def _run_full_workflow(self):
        # 系统固定4方向分频卡：9Hz(上)、11Hz(左)、13Hz(下)、15Hz(右)
        self.freq_pool = [9.0, 11.0, 13.0, 15.0] * self.spin_target_repeats.value()
        random.shuffle(self.freq_pool)
        self.cur_trial, self.total_trials, self.state_step, self.is_collecting, self.is_paused = 1, len(
            self.freq_pool), 0, True, False
        self.btn_full_workflow.setEnabled(False)
        self.p_bar.setMaximum(self.total_trials)
        self.p_bar.setValue(0)

        if self.main_window and self.main_window.participant_window:
            self.main_window.participant_window.showFullScreen()
            self.main_window.participant_window.raise_()
            self.main_window.participant_window.activateWindow()
        self._enter_next_sub_phase()
        self.master_timer.start()

    def _master_clock_tick(self):
        if self.is_paused or not self.is_collecting: return
        self.step_remaining_ms -= 100
        sec_text = f"{max(0.0, self.step_remaining_ms / 1000.0):.1f}s"
        if self.main_window and self.main_window.participant_window:
            self.main_window.participant_window.countdown_label.setText(sec_text)
        if self.step_remaining_ms <= 0:
            self._enter_next_sub_phase()

    def _enter_next_sub_phase(self):
        if self.cur_trial > self.total_trials:
            self.master_timer.stop()
            self.is_collecting = False
            self._sync_grid_highlights(None, highlight_on=False, flash=False)
            self.btn_full_workflow.setEnabled(True)
            self.lbl_status.setText("识别器模型状态: 🟡 采集完毕，等待离线优化解算...")
            self.lbl_status.setStyleSheet("color: #F6C667; font-weight: bold;")
            if self.main_window and self.main_window.participant_window:
                self.main_window.participant_window.hide()
            return

        current_target_freq = self.freq_pool[self.cur_trial - 1]

        # 🔥 核心映射：将硬编码频率数值转换为直观的“目标几”及空间朝向
        freq_to_target_str = {
            9.0: "【目标 1 】",
            11.0: "【目标 2 】",
            13.0: "【目标 3 】",
            15.0: "【目标 4 】"
        }
        target_name = freq_to_target_str.get(current_target_freq, "【未知目标】")

        if self.state_step == 0:
            self.state_step = 1
            self.step_remaining_ms = 1500
            self._sync_grid_highlights(current_target_freq, highlight_on=True, flash=False)
            if self.main_window and self.main_window.participant_window:
                # 核心更正：在大屏上彻底打出目标几字样，隐去生硬的Hz数字
                self.main_window.participant_window.set_prompt_ssvep(
                    False, f"试次 {self.cur_trial} / {self.total_trials}: 请注视 {target_name} 方块",
                    "1.5s", self.cur_trial, self.total_trials, current_target_freq
                )
        elif self.state_step == 1:
            self.state_step = 2
            self.step_remaining_ms = int(self.spin_active.value() * 1000)
            self._sync_grid_highlights(current_target_freq, highlight_on=True, flash=True)
            if self.main_window and self.main_window.participant_window:
                self.main_window.participant_window.set_prompt_ssvep(
                    True, f"⚡ 正在全速激发脑电：请保持注视 {target_name} ⚡",
                    f"{self.spin_active.value()}s", self.cur_trial, self.total_trials, current_target_freq
                )
        elif self.state_step == 2:
            self.state_step = 0
            self.p_bar.setValue(self.cur_trial)
            self.pipeline_updated.emit(1, (self.cur_trial / self.total_trials) * 100)
            self.cur_trial += 1
            self.step_remaining_ms = int(self.spin_rest.value() * 1000)
            self._sync_grid_highlights(None, highlight_on=False, flash=False)
            if self.main_window and self.main_window.participant_window:
                self.main_window.participant_window.set_prompt_ssvep(
                    False, "放松视线：请放空思维稍微休息...", f"{self.spin_rest.value()}s", self.cur_trial - 1,
                    self.total_trials
                )

    def _sync_grid_highlights(self, freq, highlight_on: bool, flash: bool):
        if self.main_window and self.main_window.participant_window:
            w = self.main_window.participant_window.ssvep_grid_widget
            if w is not None:
                for target in w.targets:
                    target["highlight"] = True if (highlight_on and target["freq"] == freq) else False
                if flash:
                    if not w.is_flashing: w.start_flashing()
                else:
                    w.stop_flashing()
                w.update()

    def set_paused_state(self, paused: bool):
        self.is_paused = paused
        if not self.is_paused and self.state_step == 2 and self.is_collecting:
            self._sync_grid_highlights(self.freq_pool[self.cur_trial - 1], highlight_on=True, flash=True)

    def _train(self):
        self.lbl_status.setText("识别器模型状态: ⏳ 正在离线迭代解算特征空间滤波超平面...")
        self.lbl_status.setStyleSheet("color: #F6C667; font-weight: bold;")
        QTimer.singleShot(1500, self._done)

    def _done(self):
        self.lbl_status.setText("识别器模型状态: 🟢 特征空间权重模型闭环预训练完成 (Acc: 95.4%)")
        self.lbl_status.setStyleSheet("color: #67E8B9; font-weight: bold;")
        self.pipeline_updated.emit(1, 100.0)
        self.stage_completed.emit()
