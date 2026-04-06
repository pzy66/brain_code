import cv2
import numpy as np
import time

# ==============================
# 视频流地址（你的 Jetson 摄像头）
# ==============================
STREAM_URL = "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color"

# ==============================
# 采样与阈值存储
# ==============================
samples = {
    "red": [],
    "green": [],
    "blue": []
}

active_color = "blue"
last_lab_img = None
generated_ranges = {}

# ==============================
# 计算鲁棒阈值（分位数 + margin）
# ==============================
def compute_range(vals, lo_q=5, hi_q=95, margin=(10, 10, 10)):
    a = np.array(vals, dtype=np.float32)
    lo = np.percentile(a, lo_q, axis=0)
    hi = np.percentile(a, hi_q, axis=0)
    lo = np.maximum(lo - np.array(margin), 0)
    hi = np.minimum(hi + np.array(margin), 255)
    return lo.astype(np.uint8), hi.astype(np.uint8)

# ==============================
# 鼠标回调：点选取样
# ==============================
def on_mouse(event, x, y, flags, param):
    global last_lab_img
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if last_lab_img is None:
        return

    L, A, B = last_lab_img[y, x]
    samples[active_color].append([int(L), int(A), int(B)])
    print(f"[{active_color}] sample #{len(samples[active_color])}: LAB=({L},{A},{B})")

# ==============================
# 主程序
# ==============================
def main():
    global last_lab_img, active_color

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("❌ 无法打开视频流")
        return

    win = "LAB Threshold Calibrator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    print("====== LAB 阈值标定工具 ======")
    print("1: red   2: green   3: blue")
    print("鼠标左键：在木块上取样（≥15次）")
    print("G: 根据采样生成阈值")
    print("C: 清空当前颜色采样")
    print("P: 打印所有阈值（复制用）")
    print("ESC: 退出")
    print("==============================")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # BGR -> RGB -> LAB
        lab = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                           cv2.COLOR_RGB2LAB)
        last_lab_img = lab

        display = frame.copy()

        # 如果已经生成阈值，显示 mask
        if active_color in generated_ranges:
            mn, mx = generated_ranges[active_color]
            mask = cv2.inRange(lab, mn, mx)
            display = cv2.bitwise_and(display, display, mask=mask)

        # 显示提示信息
        cv2.putText(display,
                    f"Active: {active_color} | Samples: {len(samples[active_color])}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('1'):
            active_color = "red"
            print(">> Active color = red")
        elif key == ord('2'):
            active_color = "green"
            print(">> Active color = green")
        elif key == ord('3'):
            active_color = "blue"
            print(">> Active color = blue")
        elif key in (ord('c'), ord('C')):
            samples[active_color].clear()
            generated_ranges.pop(active_color, None)
            print(f"🧹 清空 {active_color} 采样")
        elif key in (ord('g'), ord('G')):
            if len(samples[active_color]) < 10:
                print("⚠️ 采样太少，至少点 10 下")
            else:
                mn, mx = compute_range(samples[active_color])
                generated_ranges[active_color] = (mn, mx)
                print(f"✅ {active_color} 阈值生成：")
                print(f"   min = {mn.tolist()}")
                print(f"   max = {mx.tolist()}")
        elif key in (ord('p'), ord('P')):
            print("\n====== 可复制阈值 ======")
            for k, (mn, mx) in generated_ranges.items():
                print(f'"{k}": {{')
                print(f'    "min": np.array({mn.tolist()}, dtype=np.uint8),')
                print(f'    "max": np.array({mx.tolist()}, dtype=np.uint8),')
                print("},")
            print("========================\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
