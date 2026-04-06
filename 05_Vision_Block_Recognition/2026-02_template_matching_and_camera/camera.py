import time
import numpy as np
import cv2

# ==============================
# 里程碑1：确定的视频流（MJPEG）
# ==============================
STREAM_URL = "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color"

# ==============================
# 里程碑2：多颜色检测（LAB阈值）
# 说明：下面阈值是“能跑起来的初值”，不同光照/相机一定需要微调
# 调参技巧：
#   - 按 H 切到 mask 看分割效果
#   - 按 C 切换当前查看的颜色（red/green/blue）
#   - 先把某一种颜色调准，再调下一种
# ==============================
COLOR_RANGES_LAB = {
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

AREA_MIN_DEFAULT = 1200  # ROI面积阈值
RECONNECT_AFTER_SEC = 0.8  # 连续读帧失败超过该秒数就重连


def open_capture(url: str, warmup_sec: float = 0.3):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        cap.release()
        return None
    # 热身：读几帧让缓存稳定
    t0 = time.time()
    while time.time() - t0 < warmup_sec:
        cap.read()
    return cap


def preprocess_to_lab(frame_bgr):
    """BGR -> RGB -> LAB + blur（与课程/厂家常用一致）"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    lab = cv2.GaussianBlur(lab, (5, 5), 5)
    return lab


def build_mask(lab_img, lab_min, lab_max):
    mask = cv2.inRange(lab_img, lab_min, lab_max)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def detect_and_draw_multicolor(frame_bgr, color_ranges_lab, area_min=1200):
    """
    返回：
      draw：绘制后的 BGR 图
      masks：dict[color_name] = mask
      rois：[(color_name, rect, area), ...]  rect是minAreaRect输出
    """
    draw = frame_bgr.copy()
    lab = preprocess_to_lab(frame_bgr)

    masks = {}
    rois = []

    for color_name, cfg in color_ranges_lab.items():
        lab_min = cfg["min"]
        lab_max = cfg["max"]
        draw_color = cfg.get("draw", (0, 255, 255))

        mask = build_mask(lab, lab_min, lab_max)
        masks[color_name] = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < area_min:
                continue

            rect = cv2.minAreaRect(c)     # ((cx,cy),(w,h),angle)
            box = cv2.boxPoints(rect)     # float
            box = box.astype(np.int32)    # ✅ 兼容新 numpy（替代 np.int0）

            (cx, cy), (w, h), ang = rect

            # 画ROI框
            cv2.drawContours(draw, [box], -1, draw_color, 2)
            cv2.circle(draw, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(draw, f"{color_name} area={int(area)}",
                        (int(cx) + 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

            rois.append((color_name, rect, area))

    # 按面积从大到小编号（跨颜色统一编号）
    rois.sort(key=lambda x: x[2], reverse=True)
    for idx, (color_name, rect, area) in enumerate(rois, start=1):
        (cx, cy), _, _ = rect
        cv2.putText(draw, f"ROI#{idx}",
                    (int(cx) - 25, int(cy) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return draw, masks, rois


def main():
    print("✅ 使用视频流：", STREAM_URL)
    print("操作：")
    print("  ESC 退出")
    print("  H   显示/隐藏 mask")
    print("  C   切换当前 mask 颜色（red/green/blue）")
    print("  +/- 调整面积阈值 area_min")
    print("  P   暂停/继续")
    print("  S   截图保存当前帧")

    cap = open_capture(STREAM_URL)
    if cap is None:
        print("❌ 打不开视频流。请确认：电脑与机械臂同一WiFi，浏览器能打开该URL并看到画面。")
        return

    win = "JETMAX_CAM"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    show_mask = False
    paused = False

    area_min = AREA_MIN_DEFAULT
    fps = 0.0
    last_t = time.time()
    last_ok_frame_t = time.time()

    color_names = list(COLOR_RANGES_LAB.keys())
    mask_color_idx = 0  # 当前显示哪个颜色的mask

    last_frame = None
    last_draw = None
    last_masks = None
    last_rois = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                # 断流重连
                if time.time() - last_ok_frame_t > RECONNECT_AFTER_SEC:
                    print("⚠️ 读帧失败，尝试重连视频流...")
                    cap.release()
                    time.sleep(0.3)
                    cap = open_capture(STREAM_URL)
                    if cap is None:
                        time.sleep(0.5)
                        continue
                else:
                    time.sleep(0.02)
                    continue

            last_ok_frame_t = time.time()

            draw, masks, rois = detect_and_draw_multicolor(frame, COLOR_RANGES_LAB, area_min=area_min)

            # FPS
            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 1e-6:
                fps = fps * 0.9 + (1.0 / dt) * 0.1

            # 叠加状态信息
            cur_color = color_names[mask_color_idx]
            info1 = f"FPS:{fps:.1f}  ROIs:{len(rois)}  area_min:{area_min}"
            info2 = f"Mask:{cur_color}  (H:mask  C:switch)"
            cv2.putText(draw, info1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(draw, info2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 缓存最后一帧，供暂停/截图使用
            last_frame = frame
            last_draw = draw
            last_masks = masks
            last_rois = rois

            if show_mask:
                cv2.imshow(win, masks[cur_color])
            else:
                cv2.imshow(win, draw)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key in (ord('h'), ord('H')):
            show_mask = not show_mask
        elif key in (ord('c'), ord('C')):
            mask_color_idx = (mask_color_idx + 1) % len(color_names)
        elif key in (ord('p'), ord('P')):
            paused = not paused
            print("⏸️ 暂停" if paused else "▶️ 继续")
        elif key in (ord('+'), ord('=')):
            area_min += 200
        elif key in (ord('-'), ord('_')):
            area_min = max(200, area_min - 200)
        elif key in (ord('s'), ord('S')):
            if last_draw is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = f"frame_{ts}.png"
                cv2.imwrite(fname, last_draw)
                print("📸 已保存：", fname)
            else:
                print("⚠️ 还没有可保存的画面")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
