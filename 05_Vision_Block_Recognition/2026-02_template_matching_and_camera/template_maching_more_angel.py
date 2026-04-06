import cv2
import numpy as np
import time

# --- 配置 ---
VIDEO_URL = "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color"
TEMPLATE_PATH = 'template.jpg'
SCALE_FACTOR = 0.5
ANGLE_STEP = 15
MATCH_THRESHOLD = 0.50  # 识别多个目标时，阈值建议设高一点，防止误报
NMS_IOU_THRESHOLD = 0.3  # 重叠比例阈值，超过此值则认为是同一个物体


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(0, 0, 0))


def py_nms(boxes, scores, iou_threshold):
    """ 简易非极大值抑制 (NMS) """
    if len(boxes) == 0: return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def main():
    template_orig = cv2.imread(TEMPLATE_PATH)
    if template_orig is None:
        print("错误：找不到 template.jpg")
        return

    print("预生成旋转模板库...")
    templates_db = []
    for angle in range(0, 90, ANGLE_STEP):  # 建议 360 度全范围
        rotated_img = rotate_image(template_orig, angle)
        small_rotated = cv2.resize(rotated_img, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA)
        templates_db.append({
            "angle": angle,
            "img_small": small_rotated,
            "orig_size": rotated_img.shape[:2],
            "small_size": small_rotated.shape[:2]
        })

    cap = cv2.VideoCapture(VIDEO_URL)
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break·

        small_frame = cv2.resize(frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA)

        all_candidates = []  # 存储所有角度发现的候选框
        all_scores = []
        all_angles = []

        # 4. 遍历模板
        for t_data in templates_db:
            res = cv2.matchTemplate(small_frame, t_data["img_small"], cv2.TM_CCOEFF_NORMED)

            # 找到所有大于阈值的位置
            loc = np.where(res >= MATCH_THRESHOLD)

            h, w = t_data["small_size"]
            orig_h, orig_w = t_data["orig_size"]

            for pt in zip(*loc[::-1]):  # pt 为 (x, y)
                # 记录在原图上的坐标范围 [x1, y1, x2, y2]
                x1 = int(pt[0] / SCALE_FACTOR)
                y1 = int(pt[1] / SCALE_FACTOR)
                x2 = x1 + orig_w
                y2 = y1 + orig_h

                all_candidates.append([x1, y1, x2, y2])
                all_scores.append(float(res[pt[1], pt[0]]))
                all_angles.append(t_data["angle"])

        # 5. NMS 过滤重叠框
        keep_indices = py_nms(all_candidates, all_scores, NMS_IOU_THRESHOLD)

        # 6. 绘制最终结果
        for idx in keep_indices:
            box = all_candidates[idx]
            score = all_scores[idx]
            angle = all_angles[idx]

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label = f"{angle}deg {score:.2f}"
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # FPS 显示
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f} Objects: {len(keep_indices)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Multi-Target Matching', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()