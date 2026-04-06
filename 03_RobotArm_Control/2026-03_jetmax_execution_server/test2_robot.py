#!/usr/bin/env python3
import socket
import threading
import time
import math
import numpy as np
import cv2
import hiwonder
import rospy

# ================= 配置区 =================
HOST = "0.0.0.0"
PORT = 8888

# 机械臂物理限位 (根据实际场地调整)
LIMIT_X_MIN, LIMIT_X_MAX = -140, 140
LIMIT_Y_MIN, LIMIT_Y_MAX = -200, -40
SAFE_Z_TRAVEL = 130  # 移动时的安全高度
SAFE_Z_PICK = 85     # 抓取高度

# ================= 硬件初始化 =================
jetmax = hiwonder.JetMax()
sucker = hiwonder.Sucker()
rospy.init_node("combined_server", anonymous=True, disable_signals=True)

# 忙碌锁，防止动作冲突
busy_lock = threading.Lock()
is_busy = False

# ================= 相机标定与坐标转换 =================
class CamCal:
    def __init__(self):
        self.K = None
        self.R = None
        self.T = None
        # 尝试从 ROS 参数服务器加载，如果失败请手动填入标定数据
        try:
            params = rospy.get_param('/camera_cal/block_params', None)
            if params:
                self.K = np.array(params['K'], dtype=np.float64).reshape(3, 3)
                self.R = np.array(params['R'], dtype=np.float64).reshape(3, 1)
                self.T = np.array(params['T'], dtype=np.float64).reshape(3, 1)
        except:
            pass
        
        # 默认兜底参数 (如果读不到ROS参数)
        if self.K is None:
            print("⚠️ 未找到标定参数，使用默认值(可能不准)")
            self.K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]])
            self.R = np.eye(3)
            self.T = np.zeros((3,1))

cal = CamCal()

def camera_to_world(img_points):
    """ 将像素坐标转换为相对于当前相机中心的物理偏移量 (mm) """
    if cal.K is None: return [[0,0,0]]
    inv_k = np.asmatrix(cal.K).I
    r_mat = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(cal.R, r_mat)
    inv_r = np.asmatrix(r_mat).I
    transPlaneToCam = np.dot(inv_r, np.asmatrix(cal.T))
    
    world_pt = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_pt in img_points:
        coords[0][0] = img_pt[0]
        coords[1][0] = img_pt[1]
        coords[2][0] = 1.0
        worldPtCam = np.dot(inv_k, coords)
        worldPtPlane = np.dot(inv_r, worldPtCam)
        scale = transPlaneToCam[2][0] / worldPtPlane[2][0]
        scale_worldPtPlane = np.multiply(scale, worldPtPlane)
        worldPtPlaneReproject = np.asmatrix(scale_worldPtPlane) - np.asmatrix(transPlaneToCam)
        world_pt.append(worldPtPlaneReproject.T.tolist()[0]) # [x, y, 0]
    return world_pt

# ================= 核心控制逻辑 =================

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def cmd_move(x, y):
    """ 纯移动指令 (模拟眼动导航) """
    global is_busy
    if is_busy: return # 正在抓取时忽略移动指令

    # 软限位
    tx = clamp(x, LIMIT_X_MIN, LIMIT_X_MAX)
    ty = clamp(y, LIMIT_Y_MIN, LIMIT_Y_MAX)
    
    cx, cy, cz = jetmax.position
    dist = math.sqrt((tx-cx)**2 + (ty-cy)**2)
    t = max(0.04, dist / 150.0) # 动态时间

    try:
        # 保持当前 Z 高度，只移动 XY
        jetmax.set_position((tx, ty, cz), t)
    except Exception as e:
        print(f"Move Error: {e}")

def cmd_pick(pixel_x, pixel_y):
    """ 抓取指令 (传入像素坐标) """
    global is_busy
    with busy_lock:
        if is_busy: return
        is_busy = True
    
    try:
        print(f"🎯 PICK 请求: Pixel({pixel_x}, {pixel_y})")
        # 1. 计算物理偏移
        offsets = camera_to_world([(pixel_x, pixel_y)])
        off_x, off_y, _ = offsets[0]
        
        cur_x, cur_y, _ = jetmax.position
        # 2. 计算目标物理坐标
        target_x = cur_x + off_x
        target_y = cur_y + off_y
        
        # 3. 执行动作序列
        # 移动到目标上方
        jetmax.set_position((target_x, target_y, SAFE_Z_TRAVEL), 1.0)
        time.sleep(1.0)
        
        # 开启吸泵
        sucker.set_state(True)
        
        # 下探
        jetmax.set_position((target_x, target_y, SAFE_Z_PICK), 0.8)
        time.sleep(0.9)
        
        # 抬起 (进入 Carry 模式的高度)
        jetmax.set_position((target_x, target_y, SAFE_Z_TRAVEL), 0.8)
        time.sleep(0.8)
        
        print("✅ 抓取完成，进入搬运模式")

    except Exception as e:
        print(f"❌ 抓取失败: {e}")
        sucker.set_state(False)
        jetmax.go_home(1.5)
    finally:
        with busy_lock:
            is_busy = False

def cmd_place():
    """ 放置指令 """
    global is_busy
    with busy_lock:
        if is_busy: return
        is_busy = True
        
    try:
        print("⬇️ 执行放置")
        cur_x, cur_y, _ = jetmax.position
        
        # 下放
        jetmax.set_position((cur_x, cur_y, SAFE_Z_PICK), 0.8)
        time.sleep(0.8)
        
        # 关泵
        sucker.set_state(False)
        time.sleep(0.2)
        
        # 抬起
        jetmax.set_position((cur_x, cur_y, SAFE_Z_TRAVEL), 0.6)
        time.sleep(0.6)
        
    finally:
        with busy_lock:
            is_busy = False

# ================= 网络服务 =================
def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"🚀 Robot Server listening on {PORT}...")

    jetmax.go_home(1.5)
    time.sleep(1.5)

    conn, addr = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"Connected: {addr}")

    buffer = ""
    while True:
        try:
            data = conn.recv(1024)
            if not data: break
            buffer += data.decode('utf-8', errors='ignore')
            
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if not line: continue
                
                parts = line.split()
                cmd = parts[0]
                
                if cmd == "MOVE" and len(parts) == 3:
                    cmd_move(float(parts[1]), float(parts[2]))
                elif cmd == "PICK" and len(parts) == 3:
                    threading.Thread(target=cmd_pick, args=(float(parts[1]), float(parts[2]))).start()
                elif cmd == "PLACE":
                    threading.Thread(target=cmd_place).start()

        except Exception as e:
            print(f"Socket Error: {e}")
            break

if __name__ == "__main__":
    start_server()