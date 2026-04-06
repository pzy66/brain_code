import cv2
import numpy as np


def run_geometric_matching():
    # --- 配置区域 (需要根据实际情况微调) ---
    video_url = "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color"

    # 1. Canny 阈值：决定了什么样的线条会被提取出来
    # 如果阴影较重，把 lower 调低 (比如 30)
    canny_lower = 30
    canny_upper = 150

    # 2. 【核心筛选器】木块的物理特征
    # 请拿尺子量一下你的木块，算出 长边 / 短边 的比例
    # 例如：木块长 10cm，宽 5cm，比例就是 2.0
    target_aspect_ratio = 1.0
    ratio_tolerance = 0.3  # 允许的误差范围 (比如 1.7 ~ 2.3 都在范围内)

    # 3. 面积筛选：防止把噪点或者背景里的大物体当成木块
    min_area = 1000  # 太小的噪点不要
    max_area = 50000  # 太大的背景不要
    # ------------------------------------

    print(f"正在连接: {video_url} ...")
    cap = cv2.VideoCapture(video_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("错误：无法打开视频流。")
        return

    # 定义膨胀核 (5x5 的矩形结构元素)
    # 这个核越大，修补断裂边缘的能力越强，但形状会越“臃肿”
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    print("开始轮廓几何匹配。按 'q' 退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- A. 预处理 (Pre-processing) ---
        # 1. 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. 高斯模糊：去除噪点，特别是反光带来的高频噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Canny 边缘检测
        edges = cv2.Canny(blurred, canny_lower, canny_upper)

        # 4. 【关键步骤】膨胀 (Dilation)
        # 作用：连接因反光或阴影而断裂的边缘，让轮廓闭合
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # --- B. 查找轮廓 (Find Contours) ---
        # cv2.RETR_EXTERNAL：只找最外面的轮廓（忽略木块内部的反光纹理）
        # cv2.CHAIN_APPROX_SIMPLE：压缩轮廓点，节省内存
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ... (前文省略) ...

        # --- C. 几何筛选 (Geometric Filtering) ---
        matched_boxes = []

        for cnt in contours:
            # 1. 面积初筛
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            # 2. 计算最小外接矩形
            rect = cv2.minAreaRect(cnt)
            (center_x, center_y), (w, h), angle = rect

            # 3. 计算长宽比
            long_side = max(w, h)
            short_side = min(w, h)

            if short_side == 0: continue

            current_ratio = long_side / short_side

            # 4. 【核心判断】形状是否匹配？
            if abs(current_ratio - target_aspect_ratio) < ratio_tolerance:
                # 找到了！获取矩形的 4 个顶点用于画图
                box = cv2.boxPoints(rect)

                # 【这里是修复的关键点】
                box = np.int32(box)

                matched_boxes.append((box, current_ratio, angle))

        # ... (后文省略) ...

        # --- D. 绘制结果 ---
        # 画出预处理图 (调试用)
        cv2.imshow('Debug: Edges + Dilate', dilated)

        # 画出最终结果
        for box, ratio, ang in matched_boxes:
            # 画绿色的框
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            # 在框旁边写上信息
            # 取框的第一个点作为文字坐标
            text_x, text_y = box[0]
            label = f"Ratio: {ratio:.1f}"
            cv2.putText(frame, label, (text_x, text_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Detected: {len(matched_boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Geometric Matching', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_geometric_matching()