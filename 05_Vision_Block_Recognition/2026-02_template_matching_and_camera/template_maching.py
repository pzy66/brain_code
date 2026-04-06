import cv2
import numpy as np
import time


def run_multi_object_matching():
    # --- 配置 ---
    video_url = "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color"
    template_path = 'template.jpg'

    # 匹配阈值：大于此分数的会被选出
    threshold = 0.65

    # NMS 阈值：控制重叠去重 (0.3 表示重叠超过 30% 就合并)
    nms_threshold = 0.4
    # -----------

    # 1. 加载模板
    template = cv2.imread(template_path)
    if template is None:
        print(f"错误：未找到 '{template_path}'，请检查路径。")
        return

    h, w = template.shape[:2]

    print(f"正在尝试连接视频流: {video_url} ...")
    cap = cv2.VideoCapture(video_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("错误：无法打开视频流。")
        return

    prev_time = 0
    fps = 0

    print("开始多目标匹配。按 'q' 键退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧 (流可能已断开)")
            break

        # --- A. 模板匹配 ---
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

        # 筛选出所有大于阈值的点
        locs = np.where(result >= threshold)
        points = list(zip(*locs[::-1]))

        # --- B. 准备 NMS 数据 ---
        rects = []
        scores = []

        if len(points) > 0:
            for pt in points:
                x, y = pt
                score = result[y, x]
                rects.append([int(x), int(y), int(w), int(h)])
                scores.append(float(score))

            # --- C. 非极大值抑制 (NMS) ---
            indices = cv2.dnn.NMSBoxes(rects, scores, score_threshold=threshold, nms_threshold=nms_threshold)

            # --- D. 绘制结果 (修复部分) ---
            if len(indices) > 0:
                # 【关键修改】：将 indices 转换为 numpy 数组并拍平
                # 这样无论 cv2 返回的是 [[1], [2]] 还是 [1, 2]，都会变成 [1, 2]
                indices = np.array(indices).flatten()

                for i in indices:
                    # 现在 i 肯定是一个整数索引
                    box = rects[i]
                    curr_score = scores[i]

                    x, y, bw, bh = box

                    # 画矩形
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

                    # 写分数
                    label = f"{curr_score:.2f}"
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 显示检测数量
            cv2.putText(frame, f"Count: {len(indices)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            cv2.putText(frame, "Searching...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- E. FPS 计算 ---
        curr_time = time.time()
        time_diff = curr_time - prev_time
        if time_diff > 0:
            curr_fps = 1.0 / time_diff
            fps = fps * 0.9 + curr_fps * 0.1
        prev_time = curr_time

        # 显示 FPS
        cv2.rectangle(frame, (5, 5), (150, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('Multi-Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_multi_object_matching()