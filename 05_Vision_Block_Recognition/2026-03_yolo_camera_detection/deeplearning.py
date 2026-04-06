import cv2
import numpy as np
import time
from ultralytics import YOLO

# --- 1. 配置参数 (请确认路径正确) ---
STREAM_URL = "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color"
WEIGHTS = r"C:\Users\P1233\Desktop\brain\dataset\best.pt"

CONF = 0.35  # 置信度阈值
IOU = 0.5  # NMS 阈值
DEVICE = 0  # 显卡ID，如果没有N卡请改为 "cpu"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_color_palette(n):
    """生成固定颜色盘"""
    palette = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255)
    ]
    return palette[n % len(palette)]


def main():
    # --- 2. 初始化模型 ---
    print(f"正在加载模型: {WEIGHTS} ...")
    model = YOLO(WEIGHTS)

    # --- 3. 打开视频流 ---
    print(f"正在连接视频流: {STREAM_URL} ...")
    cap = cv2.VideoCapture(STREAM_URL)

    # 尝试设置缓冲区为1（减少网络流延迟，但取决于后端是否支持）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: 无法连接视频流，请检查网络或 IP 地址！")
        return

    # --- FPS 统计变量初始化 ---
    prev_loop_time = time.time()
    fps = 0.0

    print("开始推理，按 'q' 键退出...")

    while True:
        # A. 读取图像 (网络接收耗时)
        ok, frame = cap.read()
        if not ok:
            print("警告: 视频流中断或丢帧")
            # 如果流断了，尝试重连或者短暂等待
            time.sleep(0.1)
            continue

        # B. YOLO 推理 (GPU/CPU 计算耗时)
        results = model.predict(
            source=frame,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            half=(DEVICE != "cpu"),  # 如果是 GPU 则开启半精度加速
            verbose=False
        )

        r = results[0]
        vis = frame.copy()

        # C. 数据处理与绘图 (CPU 处理耗时)
        if r.masks is not None:
            # 提取数据
            masks_data = r.masks.data.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            # 按面积排序
            areas = np.sum(masks_data, axis=(1, 2))
            order = np.argsort(-areas)

            # 创建一个空层用于批量绘制 Mask
            combined_mask_layer = np.zeros_like(frame)

            # 遍历检测到的目标
            for rank, idx in enumerate(order, start=1):
                mask01 = (masks_data[idx] > 0.5).astype(np.uint8)

                # 计算质心
                M = cv2.moments(mask01, binaryImage=True)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                # 获取颜色
                color = get_color_palette(rank)

                # 填充 Mask 到空层
                combined_mask_layer[mask01 == 1] = color

                # 画框
                x1, y1, x2, y2 = boxes[idx]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                # 画中心点
                cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
                cv2.circle(vis, (cx, cy), 3, color, -1)

                # 画文字标签
                label = f"#{rank} {confs[idx]:.2f}"
                cv2.putText(vis, label, (x1, max(y1 - 10, 20)), FONT, 0.6, color, 2)

                # 可选：如果你要发数据给机械臂，就在这里 print 或串口发送
                # print(f"Target #{rank}: ({cx}, {cy})")

            # 一次性叠加所有 Mask (性能优化关键)
            vis = cv2.addWeighted(vis, 1.0, combined_mask_layer, 0.4, 0)

            # 显示总目标数
            cv2.putText(vis, f"Objects: {len(order)}", (20, 80), FONT, 1, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No Object", (20, 80), FONT, 1, (0, 0, 255), 2)

        # --- D. 正确的 FPS 计算 ---
        # 逻辑：计算当前时刻与上一次循环结束时刻的时间差
        curr_loop_time = time.time()
        delta_time = curr_loop_time - prev_loop_time

        # 防止时间差为0（极少见）
        if delta_time > 0:
            current_fps = 1.0 / delta_time
        else:
            current_fps = 0.0

        # 平滑 FPS 显示 (90% 旧值 + 10% 新值)
        fps = fps * 0.9 + current_fps * 0.1

        # 更新上一帧时间
        prev_loop_time = curr_loop_time

        # --- E. 绘制 UI 并显示 ---
        # 黑色背景块，确保文字清晰
        cv2.rectangle(vis, (15, 10), (200, 50), (0, 0, 0), -1)
        cv2.putText(vis, f"FPS: {fps:.1f}", (25, 40), FONT, 1.0, (0, 255, 255), 2)

        cv2.imshow("YOLOv8 Robotic Arm Vision", vis)

        # 按 'q' 或 ESC 退出
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()