#!/usr/bin/env python3
import sys
import socket
import time
import math
import cv2
import numpy as np
from ultralytics import YOLO

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QBrush, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint

# === 1. 配置区域 ===
ROBOT_IP = "192.168.149.1"
ROBOT_PORT = 8888
STREAM_URL = f"http://{ROBOT_IP}:8080/stream?topic=/usb_cam/image_rect_color"

# YOLO 模型路径 (请修改为你电脑上的实际路径)
WEIGHTS = r"C:\Users\Administrator\PycharmProjects\Robot\runs\segment\train_640_x_model2\weights\best.pt"
CONF = 0.60

# 机械臂物理范围 (和 Server 端保持一致)
REAL_X_MIN, REAL_X_MAX = -120, 120
REAL_Y_MIN, REAL_Y_MAX = -200, -40
START_X, START_Y = 0, -120

# 窗口大小
WIN_W, WIN_H = 960, 720

# SSVEP 刺激参数
ROI_RADIUS = 200
ROI_CENTER = (WIN_W // 2, WIN_H // 2)
STIM_FREQS = [8.0, 10.0, 12.0, 15.0]  # 对应键盘 1, 2, 3, 4

# 状态定义
STATE_SEARCH = 0  # 搜索模式 (眼动导航)
STATE_PICKING = 1  # 抓取中 (锁定控制)
STATE_CARRY = 2  # 搬运模式 (眼动导航)
STATE_PLACING = 3  # 放置中 (锁定控制)


class StimClockThread(QThread):
    """ 高精度闪烁时钟 """
    tick = pyqtSignal(float)

    def __init__(self, fps=60):
        super().__init__()
        self.fps = fps
        self._stop = False

    def run(self):
        t0 = time.perf_counter()
        while not self._stop:
            t = time.perf_counter() - t0
            self.tick.emit(t)
            time.sleep(1.0 / self.fps)

    def stop(self):
        self._stop = True


class YoloVideoThread(QThread):
    """ 视频流 + YOLO CPU 推理 """
    frame_and_boxes = pyqtSignal(object, object)

    def run(self):
        model = YOLO(WEIGHTS)   # 加载模型
        cap = cv2.VideoCapture(STREAM_URL)

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap.open(STREAM_URL)
                continue

            # CPU 推理
            results = model.predict(
                source=frame,
                conf=CONF,
                device="cpu",
                verbose=False
            )

            # plot() 返回 BGR，这里转成 RGB 供 Qt 显示
            frame_plot = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

            boxes = []
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

            self.frame_and_boxes.emit(frame_plot, boxes)

class MainController(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(WIN_W, WIN_H)
        self.setWindowTitle("Brain-Controlled Robot System Simulation")
        self.setMouseTracking(True)

        # 状态机
        self.state = STATE_SEARCH
        self.robot_pos = [START_X, START_Y]
        self.mouse_pos = QPoint(WIN_W // 2, WIN_H // 2)

        # 数据存储
        self.video_frame = None
        self.boxes_disp = []  # 转换到窗口坐标的框
        self.stim_targets = []  # [(x1,y1,x2,y2, freq, center_x, center_y), ...]
        self.stim_t = 0.0

        # 网络连接
        self.sock = None
        self.connect_robot()

        # 线程启动
        self.vt = YoloVideoThread()
        self.vt.frame_and_boxes.connect(self.on_video_update)
        self.vt.start()

        self.clock = StimClockThread(60)
        self.clock.tick.connect(self.on_stim_tick)
        self.clock.start()

        # 控制循环 (30Hz)
        self.timer = QTimer()
        self.timer.timeout.connect(self.control_loop)
        self.timer.start(33)

    def connect_robot(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((ROBOT_IP, ROBOT_PORT))
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("✅ 连接机械臂成功")
        except:
            print("❌ 连接失败")

    def send_cmd(self, cmd):
        if self.sock:
            try:
                self.sock.sendall((cmd + "\n").encode())
            except:
                pass

    def on_video_update(self, frame, boxes):
        self.video_frame = frame

        # 坐标映射: 视频 -> 窗口
        if frame is not None:
            fh, fw = frame.shape[:2]
            scale = min(WIN_W / fw, WIN_H / fh)
            off_x = (WIN_W - fw * scale) // 2
            off_y = (WIN_H - fh * scale) // 2

            self.boxes_disp = []
            for b in boxes:
                x1, y1, x2, y2 = b
                # 转换到窗口坐标用于显示
                dx1, dy1 = x1 * scale + off_x, y1 * scale + off_y
                dx2, dy2 = x2 * scale + off_x, y2 * scale + off_y

                # 计算中心点
                cx, cy = (dx1 + dx2) / 2, (dy1 + dy2) / 2

                # 原始视频坐标中心 (用于发给Robot)
                raw_cx, raw_cy = (x1 + x2) / 2, (y1 + y2) / 2

                self.boxes_disp.append({
                    "disp_box": (dx1, dy1, dx2, dy2),
                    "center": (cx, cy),
                    "raw_center": (raw_cx, raw_cy)
                })

            self.update_stim_targets()
        self.update()

    def update_stim_targets(self):
        """ 筛选在圆圈内的目标，并分配频率 """
        candidates = []
        rcx, rcy = ROI_CENTER

        for item in self.boxes_disp:
            cx, cy = item['center']
            dist = math.sqrt((cx - rcx) ** 2 + (cy - rcy) ** 2)
            if dist < ROI_RADIUS:
                candidates.append((dist, item))

        # 按距离排序，最近的优先
        candidates.sort(key=lambda x: x[0])

        self.stim_targets = []
        for i, (dist, item) in enumerate(candidates[:4]):
            x1, y1, x2, y2 = item['disp_box']
            freq = STIM_FREQS[i]
            # 存储目标信息，包含原始图像坐标用于抓取
            self.stim_targets.append({
                "box": (x1, y1, x2, y2),
                "freq": freq,
                "raw_center": item['raw_center'],
                "key": str(i + 1)  # 对应键盘 '1', '2', '3', '4'
            })

    def on_stim_tick(self, t):
        self.stim_t = t
        if self.state == STATE_SEARCH and self.stim_targets:
            self.update()

    def control_loop(self):
        # 只有在 搜索 或 搬运 状态下，才允许鼠标控制移动
        if self.state not in [STATE_SEARCH, STATE_CARRY]:
            return

        # 简单的比例控制 (鼠标偏离中心 -> 速度)
        cx, cy = WIN_W / 2, WIN_H / 2
        dx = self.mouse_pos.x() - cx
        dy = self.mouse_pos.y() - cy

        dead_zone = 40
        gain = 0.05

        if math.sqrt(dx ** 2 + dy ** 2) > dead_zone:
            # 更新虚拟坐标
            self.robot_pos[0] -= dx * gain  # X反向
            self.robot_pos[1] += dy * gain

            # 限制范围
            self.robot_pos[0] = max(REAL_X_MIN, min(self.robot_pos[0], REAL_X_MAX))
            self.robot_pos[1] = max(REAL_Y_MIN, min(self.robot_pos[1], REAL_Y_MAX))

            # 发送指令
            self.send_cmd(f"MOVE {self.robot_pos[0]:.2f} {self.robot_pos[1]:.2f}")

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()

    def keyPressEvent(self, event):
        key = event.key()

        # === 搜索模式下的操作 (选择物体) ===
        if self.state == STATE_SEARCH:
            # 按 1, 2, 3, 4 模拟 SSVEP 识别
            target_idx = -1
            if key == Qt.Key_1:
                target_idx = 0
            elif key == Qt.Key_2:
                target_idx = 1
            elif key == Qt.Key_3:
                target_idx = 2
            elif key == Qt.Key_4:
                target_idx = 3

            if target_idx >= 0 and target_idx < len(self.stim_targets):
                target = self.stim_targets[target_idx]
                raw_x, raw_y = target['raw_center']
                print(f"🧠 模拟SSVEP识别: {target['freq']}Hz -> 发送抓取指令")

                # 发送 PICK 指令
                self.send_cmd(f"PICK {raw_x:.1f} {raw_y:.1f}")
                self.state = STATE_PICKING

                # 3秒后自动切到 Carry 模式 (假设抓取耗时3秒)
                QTimer.singleShot(4000, lambda: self.set_state(STATE_CARRY))

        # === 搬运模式下的操作 (放置) ===
        elif self.state == STATE_CARRY:
            if key == Qt.Key_Space:
                print("🧠 放置指令触发")
                self.send_cmd("PLACE")
                self.state = STATE_PLACING

                # 2秒后切回 Search
                QTimer.singleShot(2500, lambda: self.set_state(STATE_SEARCH))

    def set_state(self, new_state):
        self.state = new_state
        print(f"State Changed -> {new_state}")

    def paintEvent(self, event):
        qp = QPainter(self)

        # 1. 绘制视频背景
        if self.video_frame is not None:
            h, w, ch = self.video_frame.shape
            bytes_per_line = ch * w
            img = QImage(self.video_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # 简单的居中缩放绘制
            scale = min(WIN_W / w, WIN_H / h)
            dw, dh = int(w * scale), int(h * scale)
            dx, dy = (WIN_W - dw) // 2, (WIN_H - dh) // 2
            qp.drawImage(dx, dy, img.scaled(dw, dh))
        else:
            qp.fillRect(0, 0, WIN_W, WIN_H, Qt.black)
            qp.setPen(Qt.white)
            qp.drawText(WIN_W // 2 - 50, WIN_H // 2, "Waiting for Video...")

        # 2. 绘制 ROI 圆圈
        cx, cy = ROI_CENTER
        qp.setPen(QPen(QColor(0, 255, 0, 100), 2, Qt.DashLine))
        qp.setBrush(Qt.NoBrush)
        qp.drawEllipse(QPoint(cx, cy), ROI_RADIUS, ROI_RADIUS)

        # 3. 绘制 SSVEP 闪烁 (仅在 Search 模式)
        if self.state == STATE_SEARCH:
            for t in self.stim_targets:
                x1, y1, x2, y2 = t['box']
                freq = t['freq']

                # SSVEP 亮度公式: 0.5 * (1 + sin(2*pi*f*t))
                lum = 0.5 * (1 + math.sin(2 * math.pi * freq * self.stim_t))
                alpha = int(100 + 155 * lum)  # 透明度闪烁

                # 绘制闪烁遮罩
                qp.setPen(Qt.NoPen)
                color = QColor(255, 255, 255, alpha)
                # 不同频率不同色调微调
                if freq == 8:
                    color = QColor(255, 0, 0, alpha)
                elif freq == 10:
                    color = QColor(0, 255, 0, alpha)
                elif freq == 12:
                    color = QColor(0, 0, 255, alpha)

                qp.setBrush(QBrush(color))
                qp.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

                # 绘制标签 (Key: 1, 2...)
                qp.setPen(Qt.yellow)
                qp.setFont(QFont("Arial", 16, QFont.Bold))
                qp.drawText(int(x1), int(y1) - 5, f"[{t['key']}] {freq}Hz")

        # 4. 绘制 HUD 状态
        qp.setPen(Qt.cyan)
        qp.setFont(QFont("Arial", 14))
        state_str = ["SEARCH (Use Mouse)", "PICKING...", "CARRY (Use Mouse)", "PLACING..."][self.state]
        qp.drawText(20, 40, f"MODE: {state_str}")

        if self.state == STATE_SEARCH:
            qp.drawText(20, 70, "Action: Mouse->Move | Keys 1-4->Select Block")
        elif self.state == STATE_CARRY:
            qp.drawText(20, 70, "Action: Mouse->Move | Space->Place Block")

        # 鼠标准星
        mx, my = self.mouse_pos.x(), self.mouse_pos.y()
        qp.setPen(QPen(Qt.red, 2))
        qp.drawLine(mx - 10, my, mx + 10, my)
        qp.drawLine(mx, my - 10, mx, my + 10)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainController()
    win.show()
    sys.exit(app.exec_())