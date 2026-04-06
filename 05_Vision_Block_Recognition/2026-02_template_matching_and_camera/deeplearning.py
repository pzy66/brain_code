import cv2
import numpy as np
from ultralytics import YOLO

STREAM_URL = "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color"
WEIGHTS = r"C:\Users\P1233\Desktop\brain\dataset\best.pt"

# 只保留置信度较高的结果
CONF = 0.35
IOU = 0.5
DEVICE = 0  # 有GPU就 0；没GPU就 "cpu"

def mask_centroid(mask01: np.ndarray):
    """mask01: HxW, 0/1"""
    m = mask01.astype(np.uint8)
    M = cv2.moments(m, binaryImage=True)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def main():
    model = YOLO(WEIGHTS)

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        raise RuntimeError("打不开视频流：请确认浏览器能打开该地址且同 WiFi")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # 推理（Ultralytics 接受 BGR ndarray）
        results = model.predict(
            source=frame,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            verbose=False
        )

        vis = frame.copy()

        r = results[0]
        if r.masks is not None and len(r.masks.data) > 0:
            # masks.data: [N, H, W] (torch)
            masks = r.masks.data.cpu().numpy()  # float 0~1
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else None

            # 选一个目标：这里选“掩膜面积最大”的那个（更稳）
            areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
            idx = int(np.argmax(areas))

            mask = masks[idx]
            mask01 = (mask > 0.5).astype(np.uint8)

            # 形态学去噪（可选，能让中心更稳）
            kernel = np.ones((5, 5), np.uint8)
            mask01 = cv2.morphologyEx(mask01, cv2.MORPH_OPEN, kernel, iterations=1)
            mask01 = cv2.morphologyEx(mask01, cv2.MORPH_CLOSE, kernel, iterations=1)

            # 质心
            c = mask_centroid(mask01)
            if c is not None:
                cx, cy = c
                cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(vis, f"center=({cx},{cy})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # 你要发给机械臂的坐标就是 cx, cy
                # send_to_robot(cx, cy)

            # 画掩膜（半透明叠加）
            color = np.zeros_like(vis)
            color[:, :, 2] = (mask01 * 180).astype(np.uint8)  # 红色通道
            vis = cv2.addWeighted(vis, 1.0, color, 0.5, 0)

            # 画框（可选）
            if boxes is not None:
                x1, y1, x2, y2 = boxes[idx].astype(int)
                score = float(confs[idx]) if confs is not None else 0.0
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"conf={score:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            cv2.putText(vis, "no object", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("YOLOv8-seg inference", vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
