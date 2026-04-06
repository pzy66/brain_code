import cv2
import numpy as np
import socket
import time

# =========================
# 1. 配置区
# =========================
STREAM_URL = "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color"

JETSON_IP = "192.168.149.1"
JETSON_PORT = 8888

# LAB 颜色阈值（用你刚才调好的！下面只是示例）
COLOR_RANGES = {
    "red": {
        "min": np.array([20, 150, 140], dtype=np.uint8),
        "max": np.array([255, 210, 210], dtype=np.uint8),
        "draw": (0, 255, 255),   # BGR
    },
    "green": {
        "min": np.array([114, 77, 110], dtype=np.uint8),
        "max": np.array([205, 123, 144], dtype=np.uint8),
        "draw": (0, 255, 0),
    },
    "blue": {
        "min": np.array([124, 119, 59], dtype=np.uint8),
        "max": np.array([181, 158, 112], dtype=np.uint8),
        "draw": (255, 0, 0),
    },
}

AREA_MIN = 1200   # 最小面积，防噪声
AREA_MAX = 20000  # 最大面积，防桌面

FONT = cv2.FONT_HERSHEY_SIMPLEX


# =========================
# 2. 发送 PICK 命令
# =========================
def send_pick(cx, cy, angle):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((JETSON_IP, JETSON_PORT))
        msg = f"PICK {cx:.2f} {cy:.2f} {angle:.2f}\n"
        s.sendall(msg.encode("utf-8"))
        s.close()
        print(f"📤 SEND → {msg.strip()}")
    except Exception as e:
        print("❌ 发送失败：", e)


# =========================
# 3. 木块检测
# =========================
def detect_blocks(frame):
    """
    return:
      blocks = [
        {
          "id": 1,
          "color": "green",
          "rect": ((cx,cy),(w,h),angle),
          "area": area
        },
        ...
      ]
    """
    blocks = []
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    for color_name, cr in COLOR_RANGES.items():
        mask = cv2.inRange(lab, cr["min"], cr["max"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3)))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < AREA_MIN or area > AREA_MAX:
                continue

            rect = cv2.minAreaRect(cnt)
            blocks.append({
                "color": color_name,
                "rect": rect,
                "area": area
            })

    # 按面积排序（大的优先编号）
    blocks.sort(key=lambda b: b["area"], reverse=True)

    # 编号
    for i, b in enumerate(blocks):
        b["id"] = i + 1

    return blocks


# =========================
# 4. 主循环
# =========================
def main():
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("❌ 无法打开视频流")
        return

    cv2.namedWindow("Block Selector", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Block Selector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("====== 操作说明 ======")
    print("1 / 2 / 3 / 4 : 选择对应编号的木块并抓取")
    print("ESC           : 退出")
    print("=====================")

    last_blocks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        blocks = detect_blocks(frame)
        last_blocks = blocks

        # === 绘制 ===
        for b in blocks:
            rect = b["rect"]
            (cx, cy), (w, h), angle = rect
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)

            color_draw = {
                "red": (0, 0, 255),
                "green": (0, 255, 0),
                "blue": (255, 0, 0)
            }[b["color"]]

            cv2.drawContours(frame, [box], 0, color_draw, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (255, 255, 255), -1)

            label = f"{b['id']}:{b['color']}"
            cv2.putText(frame, label,
                        (int(cx - 20), int(cy - 10)),
                        FONT, 0.7, (255, 255, 255), 2)

        cv2.imshow("Block Selector", frame)

        key = cv2.waitKey(1) & 0xFF

        # ESC
        if key == 27:
            break

        # 数字键选择
        if ord('1') <= key <= ord('9'):
            idx = key - ord('0')
            chosen = next((b for b in last_blocks if b["id"] == idx), None)
            if chosen:
                (cx, cy), (_, _), angle = chosen["rect"]
                print(f"🎯 选择木块 {idx} ({chosen['color']})")
                send_pick(cx, cy, angle)
            else:
                print(f"⚠️ 没有编号为 {idx} 的木块")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
