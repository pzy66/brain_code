# -*- coding: utf-8 -*-
"""
脑机接口混合控制系统 - 阶段三：机械臂脑控实时控制台（独立运行测试版）
"""
from __future__ import annotations
import sys
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush, QLinearGradient
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QGridLayout, QTextEdit
)

# ==========================================
# 核心 QSS 科技暗黑风格样式表
# ==========================================
ROBOT_CONTROL_QSS = """
QMainWindow {
    background-color: #0B0E13;
}
QWidget {
    color: #E8EEF6;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}
QFrame#stateBar, QFrame#cameraCard, QFrame#poseCard, QFrame#flowCard {
    background-color: #151B24;
    border: 1px solid #2A3444;
    border-radius: 8px;
}
QFrame#stateNode {
    background: #10161F;
    border: 1px solid #293446;
    border-radius: 6px;
}
QLabel#stateBarTitle, QLabel#controlTitle {
    color: #F0F6FC;
    font-size: 13px;
    font-weight: bold;
}
QLabel#controlStatusLabel {
    color: #A9F5D0;
    font-size: 12px;
    font-weight: bold;
}
QFrame#poseRow, QFrame#flowRow {
    background: #10161F;
    border: 1px solid #293446;
    border-radius: 6px;
}
QLabel#poseLabel, QLabel#flowStepDetail, QLabel#controlFooter {
    color: #AAB7C5;
}
QLabel#poseValue, QLabel#flowStepTitle {
    color: #F0F6FC;
    font-weight: bold;
}
QLabel#controlPill {
    color: #0D1117;
    background: #A9F5D0;
    border-radius: 4px;
    padding: 4px 8px;
    font-weight: bold;
    font-size: 11px;
}
QTextEdit#logger {
    background-color: #080C11;
    border: 1px solid #2A3444;
    border-radius: 6px;
    color: #C8D3E0;
    font-family: "Consolas", monospace;
    font-size: 12px;
}
QPushButton {
    background: #202A37;
    border: 1px solid #3A4658;
    border-radius: 6px;
    color: #EEF4FA;
    padding: 8px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background: #263444;
    border-color: #6BE7B3;
}
QPushButton[controlType='primary'] {
    background: #176B5A;
    border-color: #42C79D;
    color: #F2FFF9;
}
QPushButton[controlType='primary']:hover {
    background: #1D8570;
}
QPushButton[controlType='danger'] {
    background: #6D2632;
    border-color: #B84A5B;
    color: #FFECEF;
}
QPushButton[controlType='danger']:hover {
    background: #852F3D;
}
QPushButton[blockState='active'] {
    background: #1C5D4D;
    border-color: #70E7B8;
    color: #F2FFF9;
}
"""


# ==========================================
# 核心组件：机械臂多目标虚拟相机流仿真画布
# ==========================================
class RobotCameraPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 380)
        self.active_target = ""  # 当前锁定的目标标号: "1", "2", "3", "4"
        self.motion_phase = 0  # 用于脉冲动画的阶跃计数

        # 4个靶向小木块在平面视口中的固定相对配给坐标
        self.block_specs = [
            (0.25, 0.35, "#F6C667", "1"),
            (0.55, 0.30, "#56D6A6", "2"),
            (0.35, 0.70, "#6CA8FF", "3"),
            (0.70, 0.65, "#E879A6", "4"),
        ]

        # 内部动画计时器
        self.anime_timer = QTimer(self)
        self.anime_timer.setInterval(150)
        self.anime_timer.timeout.connect(self._on_anime_tick)
        self.anime_timer.start()

    def set_active_target(self, target_id: str):
        self.active_target = target_id
        self.update()

    def _on_anime_tick(self):
        self.motion_phase += 1
        if self.active_target:
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(5, 5, -5, -5)

        # 1. 渲染雷达/相机背景流科技质感
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QColor("#0B1724"))
        gradient.setColorAt(1.0, QColor("#182235"))
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor("#2F4058"), 1))
        painter.drawRoundedRect(rect, 8, 8)

        # 2. 渲染中心十字十字对齐线
        painter.setPen(QPen(QColor(120, 165, 210, 40), 1))
        cx, cy = rect.center().x(), rect.center().y()
        painter.drawLine(rect.left() + 20, int(cy), rect.right() - 20, int(cy))
        painter.drawLine(int(cx), rect.top() + 20, int(cx), rect.bottom() - 20)

        # 3. 遍历渲染4个小木块靶区
        for x_ratio, y_ratio, color_str, label in self.block_specs:
            bx = rect.left() + int(rect.width() * x_ratio)
            by = rect.top() + int(rect.height() * y_ratio)
            is_active = (label == self.active_target)

            painter.setBrush(QColor(color_str))
            painter.setPen(QPen(QColor("#FFFFFF" if is_active else "#F8FBFF"), 3 if is_active else 1))
            painter.drawRoundedRect(bx - 30, by - 20, 60, 40, 6, 6)

            # 如果被脑控锁定，圈定绘制绿色扩散脉冲波
            if is_active:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor("#82E6C4"), 2))
                pulse = 6 + (self.motion_phase % 4) * 4
                painter.drawEllipse(bx - 30 - pulse, by - 20 - pulse, 60 + pulse * 2, 40 + pulse * 2)

                # 绘制末端当前追踪对齐引导线
                painter.setPen(QPen(QColor("#82E6C4"), 1, Qt.DashLine))
                painter.drawLine(int(cx), int(cy), bx, by)

            # 文字数字
            painter.setPen(QColor("#08111B"))
            painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
            painter.drawText(bx - 30, by - 20, 60, 40, Qt.AlignCenter, label)

        # 绘制中心瞄准准星
        painter.setPen(QPen(QColor("#67E8B9"), 2))
        painter.drawEllipse(int(cx - 15), int(cy - 15), 30, 30)


# ==========================================
# 第三阶段主视口面板
# ==========================================
class RobotControlStageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ---------- 1. 顶部状态机时序总线 ----------
        state_bar = QFrame()
        state_bar.setObjectName("stateBar")
        state_layout = QHBoxLayout(state_bar)
        state_layout.setContentsMargins(12, 10, 12, 10)

        header_text_w = QWidget()
        header_text_v = QVBoxLayout(header_text_w)
        header_text_v.setContentsMargins(0, 0, 0, 0)
        lbl_title = QLabel("混合脑控状态机")
        lbl_title.setObjectName("stateBarTitle")
        self.lbl_status = QLabel("系统就绪：等待下发脑控指令流")
        self.lbl_status.setObjectName("controlStatusLabel")
        header_text_v.addWidget(lbl_title)
        header_text_v.addWidget(self.lbl_status)
        state_layout.addWidget(header_text_w, 2)

        # 复制 JXB_UI 的 5 大控制状态节点
        self.state_nodes = []
        state_names = ["预训练完成", "摄像头读取", "WASD 移动", "数字选块", "机械臂执行"]
        for i, name in enumerate(state_names):
            node = QFrame()
            node.setObjectName("stateNode")
            node.setStyleSheet(
                "background: #122A22; border: 1px solid #57D6A6;" if i == 1 else "background: #10161F; border: 1px solid #293446;")
            nl = QVBoxLayout(node)
            nl.setContentsMargins(8, 4, 8, 4)
            text_lbl = QLabel(name)
            text_lbl.setAlignment(Qt.AlignCenter)
            text_lbl.setStyleSheet("color: #67E8B9; font-weight: bold;" if i == 1 else "color: #8B97A5;")
            nl.addWidget(text_lbl)
            state_layout.addWidget(node, 1)
            self.state_nodes.append((node, text_lbl))

        main_layout.addWidget(state_bar)

        # ---------- 2. 中部核心工作格栅布局 ----------
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(15)

        # Left Grid: 虚拟摄像机画面
        cam_card = QFrame()
        cam_card.setObjectName("cameraCard")
        cam_layout = QVBoxLayout(cam_card)
        cam_layout.setContentsMargins(12, 12, 12, 12)

        cam_header = QHBoxLayout()
        cam_lbl = QLabel("机械臂末端 Camera 视觉对齐流")
        cam_lbl.setObjectName("stateBarTitle")
        self.pill_lbl = QLabel("CAMERA ONLINE")
        self.pill_lbl.setObjectName("controlPill")
        cam_header.addWidget(cam_lbl)
        cam_header.addStretch()
        cam_header.addWidget(self.pill_lbl)
        cam_layout.addLayout(cam_header)

        self.camera_view = RobotCameraPreviewWidget()
        cam_layout.addWidget(self.camera_view, 1)
        grid_layout.addWidget(cam_card, 3)

        # Right Grid: 控制台侧边栏卡片组
        side_panel = QVBoxLayout()
        side_panel.setSpacing(12)

        # 卡片 A: 机械臂网络套接字连接
        conn_card = QFrame()
        conn_card.setObjectName("poseCard")
        cc_l = QVBoxLayout(conn_card)
        cc_l.setContentsMargins(12, 12, 12, 12)
        lbl_c_t = QLabel("物理机器人网络套接字")
        lbl_c_t.setObjectName("stateBarTitle")
        cc_l.addWidget(lbl_c_t)

        self.lbl_net_state = QLabel("TCP 链路状态: 离线仿真模拟模式")
        self.lbl_net_state.setStyleSheet("color: #AAB7C5;")
        cc_l.addWidget(self.lbl_net_state)

        btn_net_l = QHBoxLayout()
        self.btn_connect = QPushButton("建立网络握手")
        self.btn_connect.setProperty("controlType", "primary")
        self.btn_stop = QPushButton("安全紧急断开")
        self.btn_stop.setProperty("controlType", "danger")
        btn_net_l.addWidget(self.btn_connect)
        btn_net_l.addWidget(self.btn_stop)
        cc_l.addLayout(btn_net_l)
        side_panel.addWidget(conn_card)

        # 卡片 B: 运动位姿解析矩阵
        pose_card = QFrame()
        pose_card.setObjectName("poseCard")
        pc_l = QVBoxLayout(pose_card)
        pc_l.setContentsMargins(12, 12, 12, 12)
        lbl_p_t = QLabel("空间绝对解算坐标系 (Pose Matrix)")
        lbl_p_t.setObjectName("stateBarTitle")
        pc_l.addWidget(lbl_p_t)

        self.pose_lbls = []
        pose_data = [("末端空间三维坐标", "X: 142.50  Y: -36.20  Z: 92.10 mm"),
                     ("机械轴关节角姿态", "Theta: 12.0°  Radius: -3.0 mm"),
                     ("执行夹爪运动状态", "STANDBY · OPEN (释放)")]
        for p_t, p_v in pose_data:
            row = QFrame()
            row.setObjectName("poseRow")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(8, 6, 8, 6)
            lt = QLabel(p_t)
            lt.setObjectName("poseLabel")
            lv = QLabel(p_v)
            lv.setObjectName("poseValue")
            rl.addWidget(lt)
            rl.addStretch()
            rl.addWidget(lv)
            pc_l.addWidget(row)
            self.pose_lbls.append(lv)
        side_panel.addWidget(pose_card)

        # 卡片 C: 目标快选阵列 (脑控命令下发端)
        block_card = QFrame()
        block_card.setObjectName("flowCard")
        bc_l = QVBoxLayout(block_card)
        bc_l.setContentsMargins(12, 12, 12, 12)
        lbl_b_t = QLabel("目标分块快速选择 (SSVEP 模拟锁存)")
        lbl_b_t.setObjectName("stateBarTitle")
        bc_l.addWidget(lbl_b_t)

        block_grid = QGridLayout()
        block_grid.setSpacing(8)
        self.block_btns = {}
        for idx in range(1, 5):
            btn = QPushButton(f"小木块 {idx}")
            btn.setProperty("blockState", "pending")
            # 通过lambda闭包安全传递选中标号
            btn.clicked.connect(lambda checked, b_id=str(idx): self._select_block_target(b_id))
            block_grid.addWidget(btn, (idx - 1) // 2, (idx - 1) % 2)
            self.block_btns[str(idx)] = btn
        bc_l.addLayout(block_grid)

        self.btn_execute = QPushButton("脑控确认下发抓取 (EXEC)")
        self.btn_execute.setProperty("controlType", "primary")
        bc_l.addWidget(self.btn_execute)
        side_panel.addWidget(block_card)

        grid_layout.addLayout(side_panel, 2)
        main_layout.addLayout(grid_layout, 4)

        # ---------- 3. 底部终端日志流 ----------
        lbl_log_title = QLabel("Socket 指令总线实时通讯流")
        lbl_log_title.setStyleSheet("font-size: 12px; color: #8B97A5;")
        main_layout.addWidget(lbl_log_title)

        self.log_output = QTextEdit()
        self.log_output.setObjectName("logger")
        self.log_output.setReadOnly(True)
        self.log_output.append("[CLIENT] 初始化异步 Socket 环形指令队列监听...")
        self.log_output.append("[CLIENT] 脑控适配决策源定位: 实时混合仿真推演模式开启.")
        main_layout.addWidget(self.log_output, 1)

        # 联动控制绑定
        self.btn_connect.clicked.connect(self._simulate_net_connect)
        self.btn_stop.clicked.connect(self._simulate_emergency_stop)
        self.btn_execute.clicked.connect(self._execute_grab)

        self.current_selected_block = ""

    def _select_block_target(self, target_id: str):
        self.current_selected_block = target_id
        # 1. 刷新左侧画布焦点锁定
        self.camera_view.set_active_target(target_id)

        # 2. 刷新状态机大本营高亮节点（跳转至第3阶段状态高亮）
        self._update_top_states(3)
        self.lbl_status.setText(f"SSVEP 识别锁定：已捕获目标块 [{target_id}]，等待下发执行指令")

        # 3. 刷新小木块按钮阵列状态样式
        for k, btn in self.block_btns.items():
            btn.setProperty("blockState", "active" if k == target_id else "pending")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        self.log_output.append(f"[DECODER] >>> SSVEP 频点频率匹配成功：锁定小木块候选靶区 {target_id}号.")

    def _update_top_states(self, active_idx: int):
        for idx, (node, txt_lbl) in enumerate(self.state_nodes):
            if idx == active_idx:
                node.setStyleSheet("background: #122A22; border: 1px solid #57D6A6;")
                txt_lbl.setStyleSheet("color: #67E8B9; font-weight: bold;")
            else:
                node.setStyleSheet("background: #10161F; border: 1px solid #293446;")
                txt_lbl.setStyleSheet("color: #8B97A5;")

    def _execute_grab(self):
        if not self.current_selected_block:
            self.log_output.append("[WARNING] 命令拒绝：请先选择或通过脑电锁存一个木块目标。")
            return

        self._update_top_states(4)  # 跳转状态至“机械臂执行”
        self.lbl_status.setText(
            f"MI 指令融合成功：多普勒追踪机械臂向目标 [{self.current_selected_block}] 下发运动抓取命令序列")
        self.log_output.append(f"[SOCKET] TX => MOVE_CYL_AUTO_PICK_SLOT_{self.current_selected_block}")

        # 实时位姿状态框反馈转换
        self.pose_lbls[0].setText("X: 164.20  Y: -25.00  Z: 58.00 mm (对准靶心)")
        self.pose_lbls[2].setText("BUSY · GRIP_CLOSE (夹爪闭合执行中)")
        self.pose_lbls[2].setStyleSheet("color: #F6C667; font-weight: bold;")

        # 延时恢复待命状态
        QTimer.singleShot(2000, self._on_grab_finished)

    def _on_grab_finished(self):
        self._update_top_states(1)  # 回归摄像头读取
        self.lbl_status.setText("系统就绪：动作执行序列完毕，重回视觉读取寻靶区")
        self.log_output.append("[SOCKET] RX <= CMD_EXEC_SUCCESS | 末端已回归安全待命空间。")
        self.pose_lbls[0].setText("X: 142.50  Y: -36.20  Z: 92.10 mm")
        self.pose_lbls[2].setText("STANDBY · OPEN (抓取完成已释放)")
        self.pose_lbls[2].setStyleSheet("color: #F0F6FC; font-weight: normal;")

    def _simulate_net_connect(self):
        self.lbl_net_state.setText("TCP 链路状态: 已连接 (网口主控端 192.168.1.100:23)")
        self.lbl_net_state.setStyleSheet("color: #67E8B9; font-weight: bold;")
        self.pill_lbl.setText("ROBOT ONLINE")
        self.pill_lbl.setStyleSheet("color: #0D1117; background: #67E8B9; font-weight: bold;")
        self.log_output.append("[NET] TCP Socket 通讯管道无缝握手连接成功。")

    def _simulate_emergency_stop(self):
        self.log_output.append("[HALT] !!! 紧急中止下发 !!! 关节制动器已强行抱闸制动！")
        self._simulate_net_connect()
        self.lbl_status.setText("系统挂起：由于强行中止，系统指令队列已被清空重置")


# ==========================================
# 独立调测主窗口外壳
# ==========================================
class RobotTestMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("机械臂脑控控制台 - 阶段三独立调测外壳")
        self.resize(1000, 680)
        self.setStyleSheet(ROBOT_CONTROL_QSS)

        self.robot_panel = RobotControlStageWidget()
        self.setCentralWidget(self.robot_panel)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotTestMainWindow()
    window.show()
    sys.exit(app.exec_())
