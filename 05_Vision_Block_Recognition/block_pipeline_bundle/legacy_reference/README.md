# 旧代码定位

这个目录不重复拷贝旧实验脚本，避免把主流程弄乱。旧代码仍保留在原位置：

- `..\2026-03_yolo_camera_detection\deeplearning.py`: 早期 OpenCV 窗口 + YOLO mask 可视化。
- `..\2026-03_yolo_camera_detection\computer\test2.py`: 早期 PyQt + YOLO GPU 联调脚本，路径和依赖较旧。
- `..\2026-03_yolo_camera_detection\computer\test2_CPU.py`: CPU 版旧联调脚本。
- `..\2026-02_template_matching_and_camera\`: 模板匹配、颜色阈值、轮廓匹配等传统视觉尝试。

当前推荐继续维护本 bundle 的 `01_realtime_recognize.py`、`02_train_yolo_segmentation.py`、`03_collect_images.py`。
