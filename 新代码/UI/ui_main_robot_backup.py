# -*- coding: utf-8 -*-
"""
ui_main.py - 机械臂脑控控制台组件（高感官全量功能落地版）
完美打通：MI引导移动(10s) -> 异步决策框(8Hz/15Hz) -> SSVEP抓取确认 -> MI带载移动(10s) -> 确认放下
"""
import random
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QRectF
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QGridLayout, QTextEdit, QGroupBox, QProgressBar
)
from PyQt5.QtGui import QColor, QFont


class RobotControlStageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_executing = False

        # 混合脑控核心时序状态机变量
        self.control_phase = "IDLE"  # IDLE, MI_MOVE_1, DECIDE_1, SSVEP_GRAB, MI_MOVE_2, DECIDE_2, TASK_DONE
        self.phase_remaining_ms = 0
        self.extra_time_count = 0  # 记录额外增加10秒的次数
        self.has_block_loaded = False  # 机械臂持载状态
        self.cur_target = ""  # 当前瞄准的木块目标 (1~4)

        self._init_ui()

        # 脑控多模态融合中央节拍器 (100ms)
        self.bci_clock = QTimer(self)
        self.bci_clock.setInterval(100)
        self.bci_clock.timeout.connect(self._bci_core_pulse_tick)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # --------------------------------------------------
        # 顶部：混合脑控状态拓扑总线
        # --------------------------------------------------
        state_bar = QFrame()
        state_bar.setObjectName("stateBar")
        sb_l = QHBoxLayout(state_bar)
        w_text = QWidget()
        wt_l = QVBoxLayout(w_text)
        wt_l.setContentsMargins(0, 0, 0, 0)
        wt_l.addWidget(QLabel("混合多模态脑控控制拓扑总线"))
        self.lbl_run_status = QLabel("中枢就绪：等待主控控制台下发混合实验范式")
        self.lbl_run_status.setStyleSheet("color: #A9F5D0; font-weight: bold;")
        wt_l.addWidget(self.lbl_run_status)
        sb_l.addWidget(w_text, 2)

        self.nodes = []
        node_names = ["1. MI 移动", "2. 状态决策", "3. SSVEP 锁靶", "4. 带载移动", "5. 放下确认"]
        for i, name in enumerate(node_names):
            nd = QFrame()
            nd.setObjectName("stateNode")
            nd.setStyleSheet("background:#10161F; border:1px solid #293446;")
            nl = QVBoxLayout(nd)
            nl.setContentsMargins(6, 4, 6, 4)
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#8B97A5; font-size:11px;")
            nl.addWidget(lbl)
            sb_l.addWidget(nd, 1)
            self.nodes.append((nd, lbl))
        layout.addWidget(state_bar)

        # --------------------------------------------------
        # 中部：孪生视窗与控制侧边栏
        # --------------------------------------------------
        grid = QHBoxLayout()
        cam_card = QFrame()
        cam_card.setObjectName("cameraCard")
        cc_l = QVBoxLayout(cam_card)
        cc_l.addWidget(QLabel("数字孪生机械臂末端寻靶相机流 (实时脑电反馈叠加)"), 0, Qt.AlignTop)

        # 引用由 ui_widgets 安全闭环导出的寻靶视觉流画布
        from ui_widgets import RobotCameraWidget
        self.cam = RobotCameraWidget()
        cc_l.addWidget(self.cam, 1)
        grid.addWidget(cam_card, 3)

        # 右侧参数与模拟决策区
        side = QVBoxLayout()
        side.setSpacing(10)

        # 姿态矩阵卡片
        pc = QFrame()
        pc.setObjectName("poseCard")
        pc_l = QVBoxLayout(pc)
        pc_l.addWidget(QLabel("绝对空间几何矩阵 (Kinematics Pose)"))
        self.pos_lbl = QLabel("末端三维坐标: X: 142.5  Y: -36.2  Z: 92.1 mm")
        self.grip_lbl = QLabel("持载负荷状态: STANDBY · OPEN (安全释放)")
        self.grip_lbl.setStyleSheet("color: #8B97A5;")
        pc_l.addWidget(self.pos_lbl)
        pc_l.addWidget(self.grip_lbl)
        side.addWidget(pc)

        # 核心功能：动态提示与脑电异步决策刺激框
        self.box_decision = QGroupBox("BCI 动态决策刺激判定视口")
        self.box_decision.setStyleSheet("QGroupBox::title { color: #67E8B9; }")
        self.bd_lay = QVBoxLayout(self.box_decision)
        self.lbl_decision_hint = QLabel("等待流程触发...")
        self.lbl_decision_hint.setWordWrap(True)
        self.bd_lay.addWidget(self.lbl_decision_hint)

        # 倒计时微型进度条
        self.pbar_countdown = QProgressBar()
        self.pbar_countdown.setValue(0)
        self.pbar_countdown.setTextVisible(True)
        self.bd_lay.addWidget(self.pbar_countdown)

        # 模拟受试者注视产生对应频点的按钮（前端闭环核心）
        self.lay_decide_btns = QHBoxLayout()
        self.btn_freq_8hz = QPushButton("注视 8 Hz")
        self.btn_freq_15hz = QPushButton("注视 15 Hz")
        self.btn_freq_8hz.setObjectName("accentBtn")
        self.btn_freq_15hz.setObjectName("actionBtn")
        self.btn_freq_8hz.setEnabled(False)
        self.btn_freq_15hz.setEnabled(False)
        self.lay_decide_btns.addWidget(self.btn_freq_8hz)
        self.lay_decide_btns.addWidget(self.btn_freq_15hz)
        self.bd_lay.addLayout(self.lay_decide_btns)
        side.addWidget(self.box_decision)

        # 靶区目标物理快选卡片
        bc = QFrame()
        bc.setObjectName("flowCard")
        bc_l = QVBoxLayout(bc)
        bc_l.addWidget(QLabel("SSVEP 四角目标物理选块 (4-Class Slots)"))
        bg = QGridLayout()
        self.b_btns = {}
        for idx in range(1, 5):
            btn = QPushButton(f"目标木块 {idx}")
            btn.setProperty("blockState", "pending")
            btn.setEnabled(False)  # 仅在需要SSVEP选块时解锁
            btn.clicked.connect(lambda checked, b_id=str(idx): self._select_ssvep_slot(b_id))
            bg.addWidget(btn, (idx - 1) // 2, (idx - 1) % 2)
            self.b_btns[str(idx)] = btn
        bc_l.addLayout(bg)
        side.addWidget(bc)

        # 控制工作流启动按钮
        self.btn_master_start = QPushButton("启动混合控制端到端流水线")
        self.btn_master_start.setProperty("controlType", "primary")
        self.btn_master_start.clicked.connect(self._start_integrated_flow)
        side.addWidget(self.btn_master_start)

        side.addStretch()
        grid.addLayout(side, 2)
        layout.addLayout(grid, 4)

        # 日志终端区
        self.logger = QTextEdit()
        self.logger.setObjectName("logger")
        self.logger.setReadOnly(True)
        self.logger.append("[CLIENT] 混合脑控机械臂控制总线链路已接通。")
        layout.addWidget(self.logger, 1)

        # 连接模拟注视频率动作
        self.btn_freq_8hz.clicked.connect(self._on_inject_8hz_decision)
        self.btn_freq_15hz.clicked.connect(self._on_inject_15hz_decision)

    def _update_bus_nodes(self, active_idx: int):
        """同步更新顶层状态总线的链路高亮高保真渲染"""
        for idx, (node, lbl) in enumerate(self.nodes):
            if idx == active_idx:
                node.setStyleSheet("background:#122A22; border:1px solid #57D6A6; border-radius:8px;")
                lbl.setStyleSheet("color:#67E8B9; font-weight:bold;")
            elif idx < active_idx:
                node.setStyleSheet("background:#172233; border:1px solid #41617F; border-radius:8px;")
                lbl.setStyleSheet("color:#41617F;")
            else:
                node.setStyleSheet("background:#10161F; border:1px solid #293446; border-radius:8px;")
                lbl.setStyleSheet("color:#8B97A5;")

    # --------------------------------------------------
    # 核心状态机节拍控制器
    # --------------------------------------------------
    def _start_integrated_flow(self):
        """一键激活端到端闭环混合控制"""
        self.btn_master_start.setEnabled(False)
        self._enter_mi_move_stage_1()
        self.bci_clock.start()

    def _enter_mi_move_stage_1(self):
        """阶段1：MI导引平移移动"""
        self.control_phase = "MI_MOVE_1"
        self.phase_remaining_ms = 10000  # 持续10秒
        self.pbar_countdown.setMaximum(10000)
        self._update_bus_nodes(0)
        self.lbl_run_status.setText("【阶段1】受试者通过 4 类运动想象（左/右/后/前）控制机械臂在平面内接近目标区域...")
        self.logger.append("[BCI] 进入阶段1：开启 MI 四分类平移控制链路，时间 10 秒。")

    def _bci_core_pulse_tick(self):
        """高精 100ms 融合脉冲，驱动时间轴减速与状态机自流转"""
        if self.control_phase in ["MI_MOVE_1", "MI_MOVE_2"]:
            self.phase_remaining_ms -= 100
            self.pbar_countdown.setValue(self.phase_remaining_ms)
            self.pbar_countdown.setFormat(f"MI 导引中: {self.phase_remaining_ms / 1000.0:.1f}s")

            # 模拟产生MI引导机械臂的随机微调坐标变化
            if self.control_phase == "MI_MOVE_1":
                self.pos_lbl.setText(
                    f"末端三维坐标: X: {142.5 + random.uniform(-2, 2):.1f}  Y: {-36.2 + random.uniform(-2, 2):.1f}  Z: 92.1 mm")
            else:
                self.pos_lbl.setText(
                    f"末端三维坐标: X: {164.2 + random.uniform(-1, 1):.1f}  Y: {-25.0 + random.uniform(-1, 1):.1f}  Z: 58.0 mm")

            if self.phase_remaining_ms <= 0:
                if self.control_phase == "MI_MOVE_1":
                    self._enter_decision_stage_1()
                else:
                    self._enter_decision_stage_2()

    def _enter_decision_stage_1(self):
        """阶段1结束，自动暂停MI并进入决策判定框1"""
        self.control_phase = "DECIDE_1"
        self._update_bus_nodes(1)
        self.btn_freq_8hz.setEnabled(True)
        self.btn_freq_15hz.setEnabled(True)
        self.lbl_run_status.setText("【决策轴 1】MI 导引时限到达。系统已自动挂起平移，弹出 SSVEP 状态决策框...")
        self.lbl_decision_hint.setText(
            "请注视外周闪烁块以确定下一步行动：\n🔹 8 Hz -> 进入阶段2（SSVEP 抓取锁定）\n🔸 15 Hz -> 延长阶段1（继续 MI 移动 10s）")
        self.logger.append("[SYSTEM] MI 平面控制挂起。当前决策窗口：[8Hz]进入抓取 / [15Hz]增加10s移动。")

    @pyqtSlot()
    def _on_inject_8hz_decision(self):
        """接收受试者注视 8Hz 的分类器解码特征信号"""
        self.btn_freq_8hz.setEnabled(False)
        self.btn_freq_15hz.setEnabled(False)

        if self.control_phase == "DECIDE_1":
            self._enter_ssvep_grab_stage()
        elif self.control_phase == "SSVEP_GRAB":
            self._execute_physical_grab()
        elif self.control_phase == "DECIDE_2":
            self._execute_physical_release()

    @pyqtSlot()
    def _on_inject_15hz_decision(self):
        """接收受试者注视 15Hz 的分类器解码特征信号"""
        if self.control_phase == "DECIDE_1":
            self.logger.append("[BCI] 解码成功: 检测到 15Hz 注视特征。触发重入扩展，MI 移动时间增加 10 秒。")
            self._enter_mi_move_stage_1()
        elif self.control_phase == "SSVEP_GRAB":
            self.logger.append("[BCI] 解码成功: 检测到 15Hz 注视。受试者放弃当前抓取，系统退回阶段1。")
            for k, btn in self.b_btns.items():
                btn.setEnabled(False)
                btn.setProperty("blockState", "pending")
                btn.style().polish(btn)
            self._enter_mi_move_stage_1()
        elif self.control_phase == "DECIDE_2":
            self.logger.append("[BCI] 解码成功: 检测到 15Hz 注视。继续维持带载运输，增加 10 秒移动时长。")
            self._enter_mi_move_stage_2()

    def _enter_ssvep_grab_stage(self):
        """阶段2：SSVEP 锁靶与目标快选"""
        self.control_phase = "SSVEP_GRAB"
        self._update_bus_nodes(2)
        self.lbl_run_status.setText("【阶段2】请注视视口中的木块目标，分类器将通过 SSVEP 识别特征频点进行物理定位...")
        self.lbl_decision_hint.setText(
            "请在右侧或大屏幕快选要抓取的木块（或模拟解码）：\n🔹 8 Hz -> 确认抓取锁定木块目标\n🔸 15 Hz -> 放弃抓取，重返阶段1(MI平移)")

        # 解锁 SSVEP 物理选块按钮
        for btn in self.b_btns.values():
            btn.setEnabled(True)
        self.logger.append("[BCI] 异步决策响应：已切入阶段2 SSVEP 4通道多目标锁靶特征矩阵。")

    def _select_ssvep_slot(self, b_id: str):
        """模拟通过 SSVEP 频率分类器精细锁存某个木块ID"""
        self.cur_target = b_id
        self.cam.active_id = b_id  # 触发相机视窗的追踪激光线与呼吸脉冲
        self.lbl_run_status.setText(f"【SSVEP 解算成功】已精准聚焦候选目标木块 【{b_id}】！")
        self.logger.append(f"[BCI] 在线特征解算成功：已捕获注视方块目标 [{b_id}]。等待下发 8Hz 抓取确认。")

        for k, btn in self.b_btns.items():
            btn.setProperty("blockState", "active" if k == b_id else "pending")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        # 锁靶成功后，使能 8Hz/15Hz 确认抓取决策框
        self.btn_freq_8hz.setEnabled(True)
        self.btn_freq_15hz.setEnabled(True)

    def _execute_physical_grab(self):
        """下发真实控制命令，执行硬件级机械臂吸附吸取"""
        if not self.cur_target: return
        self.btn_freq_8hz.setEnabled(False)
        self.btn_freq_15hz.setEnabled(False)
        for btn in self.b_btns.values():
            btn.setEnabled(False)

        self.lbl_run_status.setText(f"【网络命令下发】发送抓取序列帧，机械臂下降至目标 [{self.cur_target}]...")
        self.pos_lbl.setText("末端空间三维坐标: X: 164.2  Y: -25.0  Z: 58.0 mm")
        self.grip_lbl.setText("末端夹爪运动姿态: BUSY · GRIP_CLOSE (吸嘴真空紧固中)")
        self.grip_lbl.setStyleSheet("color: #F6C667; font-weight: bold;")

        # 延时 1.5 秒模拟抓取完毕，随后自动切入带载移动阶段
        QTimer.singleShot(1500, self._enter_mi_move_stage_2)

    def _enter_mi_move_stage_2(self):
        """阶段3：MI 带载引导移动"""
        self.control_phase = "MI_MOVE_2"
        self.has_block_loaded = True
        self.phase_remaining_ms = 10000  # 10秒带载时限
        self.pbar_countdown.setMaximum(10000)
        self._update_bus_nodes(3)

        self.lbl_run_status.setText(
            f"【阶段3】持载目标 [{self.cur_target}] 成功！继续切换至 MI 动作想象，引导机械臂将物料运往放置区...")
        self.lbl_decision_hint.setText("MI 混合带载运输时序运行中...")
        self.grip_lbl.setText(f"持载负荷状态: ✅ LOADED · LOCKED (包含木块 {self.cur_target})")
        self.grip_lbl.setStyleSheet("color: #67E8B9; font-weight: bold;")
        self.logger.append(f"[BCI] 闭环成功：进入阶段3，执行 MI 带载搬运控制流。")

    def _enter_decision_stage_2(self):
        """带载移动时间耗尽，进入决策框2：放下或继续"""
        self.control_phase = "DECIDE_2"
        self._update_bus_nodes(4)
        self.btn_freq_8hz.setEnabled(True)
        self.btn_freq_15hz.setEnabled(True)
        self.lbl_run_status.setText("【决策轴 2】带载搬运时限到达。控制链挂起，弹出放下确认决策框...")
        self.lbl_decision_hint.setText(
            "请注视外周进行最终放置判断：\n🔹 8 Hz -> 确认放下小木块，结束整体任务\n🔸 15 Hz -> 延长搬运阶段（继续带载 MI 移动 10s）")
        self.logger.append("[SYSTEM] 运输链路挂起。当前放置决策窗口：[8Hz]确认释放 / [15Hz]继续移动。")

    def _execute_physical_release(self):
        """受试者注视 8Hz，释放抓取物，完成整套多模态端到端控制流"""
        self.bci_clock.stop()
        self.btn_freq_8hz.setEnabled(False)
        self.btn_freq_15hz.setEnabled(False)

        self.control_phase = "TASK_DONE"
        self._update_bus_nodes(5)

        self.lbl_run_status.setText("【任务顺利结束】真空吸嘴关断，小木块安全释放，整体混合控制路径圆满闭环！")
        self.lbl_decision_hint.setText("✅ 脑控系统闭环通关完毕！")
        self.pos_lbl.setText("末端空间三维坐标: X: 142.5  Y: -36.2  Z: 92.1 mm")
        self.grip_lbl.setText("持载负荷状态: STANDBY · OPEN (安全释放)")
        self.grip_lbl.setStyleSheet("color: #E8EEF6; font-weight: normal;")

        # 重置按钮
        for btn in self.b_btns.values():
            btn.setProperty("blockState", "pending")
            btn.style().polish(btn)
        self.cam.active_id = ""
        self.btn_master_start.setEnabled(True)
        self.logger.append("[SUCCESS] 网络闭环控制链执行完毕。物料平稳落地，重置拓扑基准。")
