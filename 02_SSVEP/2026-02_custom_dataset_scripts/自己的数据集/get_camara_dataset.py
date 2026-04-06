import cv2
import time
from pathlib import Path

# =========================
# 1) 配置区：改这里就行
# =========================
STREAM_URL = "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color"

# 数据集根目录（会自动创建 images/）
DATASET_ROOT = r"C:\Users\P1233\Desktop\brain\dataset\camara\row"

# 保存格式：jpg 更省空间；png 更无损
SAVE_EXT = ".jpg"
JPG_QUALITY = 95  # 1~100

# =========================
# 2) 建立目录
# =========================
images_dir = Path(DATASET_ROOT) / "cylinder"
images_dir.mkdir(parents=True, exist_ok=True)

def now_stamp():
    # 20260206_153012_123（到毫秒）
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"_{ms:03d}"

def next_index():
    # 根据已有文件数生成序号（避免覆盖）
    existing = list(images_dir.glob(f"*{SAVE_EXT}"))
    return len(existing) + 1

def save_frame(frame, idx):
    name = f"img_{idx:05d}_{now_stamp()}{SAVE_EXT}"
    path = str(images_dir / name)

    if SAVE_EXT.lower() in [".jpg", ".jpeg"]:
        ok = cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
    else:
        ok = cv2.imwrite(path, frame)

    return ok, path

# =========================
# 3) 鼠标点击保存：回调
# =========================
WIN_NAME = "Dataset Capture"

# 用全局/闭包变量存状态
idx = None
last_frame = None
last_saved_path = ""
click_request = False  # 点击触发一次保存请求

def on_mouse(event, x, y, flags, param):
    global click_request
    # 左键按下：触发保存
    if event == cv2.EVENT_LBUTTONDOWN:
        click_request = True

def main():
    global idx, last_frame, last_saved_path, click_request

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("❌ 无法打开视频流：", STREAM_URL)
        return

    idx = next_index()

    # 显式创建窗口，确保能接收鼠标事件
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)
    cv2.setMouseCallback(WIN_NAME, on_mouse)

    print("====== 抓图工具（点击保存） ======")
    print("鼠标左键点击画面：保存一张到 images/")
    print("ESC / q：退出")
    print("保存目录：", images_dir)
    print("=================================")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        # 始终保留最新帧供点击保存
        last_frame = frame

        show = frame.copy()
        cv2.putText(show, "CLICK: save | ESC/q: quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(show, f"saved: {idx-1}",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if last_saved_path:
            # 路径太长就显示后半段
            short_path = last_saved_path
            if len(short_path) > 80:
                short_path = "..." + short_path[-77:]
            cv2.putText(show, f"last: {short_path}",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(WIN_NAME, show)

        # 如果检测到点击请求，就保存一次
        if click_request:
            click_request = False
            if last_frame is not None:
                ok, path = save_frame(last_frame.copy(), idx)
                if ok:
                    last_saved_path = path
                    print(f"💾 Saved [{idx:05d}] -> {path}")
                    idx += 1
                else:
                    print("❌ 保存失败：", path)

        key = cv2.waitKeyEx(10)
        if key != -1:
            key8 = key & 0xFF
            if key8 == 27 or key8 in (ord('q'), ord('Q')):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
