# 小木块识别流程整理

这个文件夹把当前小木块视觉识别相关程序集中到一起，按实际使用顺序分成三类：

1. `01_realtime_recognize.py`
   实时读取 JetMax 机械臂摄像头，加载训练好的 YOLO profile/权重，识别小木块顶面并输出中心点。
2. `02_train_yolo_segmentation.py`
   训练或微调 YOLO segmentation profile，输出新的 `best.pt`。
3. `03_collect_images.py`
   从 JetMax 机械臂摄像头采集训练照片，按 session 保存图片和 manifest。

## 当前识别流程

当前可用的主流程不是传统“分类照片”模型，而是 YOLO 分割模型：

1. 摄像头输入
   默认从 JetMax 的 MJPEG 流读取：
   `http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80`
2. 模型加载
   默认权重是：
   `datasets/vision/models/best.pt`
   也可以用 `BRAIN_VISION_WEIGHTS` 指向其它本地 `.pt` 文件。
3. 模型类型
   已检查当前 `best.pt`：Ultralytics YOLO `segment`，类别名是 `upside of cube`。
4. 后处理
   程序读取 YOLO mask，提取最大连通区域，算 mask 质心、轮廓、多边形、bbox、面积和置信度。
5. 目标槽位
   ROI 内最多保留 4 个目标，按槽位追踪；槽位对应 SSVEP 闪烁频率 `8,10,12,15 Hz`。
6. 输出
   实时窗口显示检测结果；标准输出持续输出 JSON Lines，包括 `slots`、`pixel_center`、`bbox`、`confidence`、`selected_slot`、`selected_center`。

## 直接运行

在这个文件夹里双击或命令行运行：

```bat
START_01_REALTIME_RECOGNITION.cmd
START_03_COLLECT_IMAGES.cmd
```

训练脚本需要先准备 YOLO segmentation 格式的数据集：

```bat
START_02_TRAIN_YOLO_SEGMENTATION.cmd --init-only
START_02_TRAIN_YOLO_SEGMENTATION.cmd --check-only
START_02_TRAIN_YOLO_SEGMENTATION.cmd --epochs 100 --imgsz 640 --device auto
```

训练完成后，如果要把新 profile 部署给实时识别使用：

```bat
START_02_TRAIN_YOLO_SEGMENTATION.cmd --epochs 100 --imgsz 640 --device auto --deploy-to C:\Users\P1233\Desktop\brain\brain_code\datasets\vision\models\best.pt
```

## 训练数据要求

`02_train_yolo_segmentation.py` 默认使用仓库内这个训练目录：

```text
C:\Users\P1233\Desktop\brain\brain_code\datasets\vision\yolo_seg\
  data.yaml
  images\
    train\
    val\
    test\
  labels\
    train\
    val\
    test\
```

标签必须是 YOLO segmentation 格式，不是普通分类文件夹：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

坐标是 0 到 1 的归一化多边形点。空场景或负样本可以放空的 `.txt` 标签文件。仅有 `images/train`、`images/val` 这种照片文件夹还不能直接训练当前分割模型，必须先用 Labelme、CVAT、Roboflow 等工具给木块顶面画多边形并导出 YOLO segmentation 标签。

## 采集程序保存内容

`03_collect_images.py` 默认保存到：

```text
C:\Users\P1233\Desktop\brain\brain_code\datasets\vision\captures
```

每次新建 session 会生成：

```text
block_collect_YYYYMMDD_HHMMSS\
  session_meta.json
  manifest.jsonl
  images\
    train\
    val\
    test\
    raw\
```

界面支持单张保存、连拍、定时自动采集、scene tag、split、note、negative sample。快捷键：

- `Space`: 保存当前帧
- `B`: 连拍
- `A`: 自动采集
- `N`: 标记负样本
- `S`: 新建 session
- `Esc`: 退出

## 我在仓库里找到的相关程序

当前独立实时识别主程序：

- `C:\Users\P1233\Desktop\brain\brain_code\05_Vision_Block_Recognition\2026-03_yolo_camera_detection\block_center_ssvep_single.py`

当前集成到主控制器里的视觉链路：

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\vision\runtime.py`
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\vision\processing.py`
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\vision\target_resolver.py`
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\run_real.py`

采集程序原始位置：

- `C:\Users\P1233\Desktop\brain\brain_code\06_Data_Collection\2026-04_jetmax_block_dataset_collection\block_dataset_collector.py`

旧 YOLO / OpenCV 参考程序：

- `C:\Users\P1233\Desktop\brain\brain_code\05_Vision_Block_Recognition\2026-03_yolo_camera_detection\deeplearning.py`
- `C:\Users\P1233\Desktop\brain\brain_code\05_Vision_Block_Recognition\2026-03_yolo_camera_detection\computer\test2.py`
- `C:\Users\P1233\Desktop\brain\brain_code\05_Vision_Block_Recognition\2026-03_yolo_camera_detection\computer\test2_CPU.py`

更早的传统视觉参考：

- `C:\Users\P1233\Desktop\brain\brain_code\05_Vision_Block_Recognition\2026-02_template_matching_and_camera\camera.py`
- `C:\Users\P1233\Desktop\brain\brain_code\05_Vision_Block_Recognition\2026-02_template_matching_and_camera\template_maching.py`
- `C:\Users\P1233\Desktop\brain\brain_code\05_Vision_Block_Recognition\2026-02_template_matching_and_camera\template_maching_more_angel.py`

## 当前数据现状

历史机器上曾经使用过这些位置，现已迁移到 `datasets/vision/legacy_camara/workspace_dataset_camara/`，后续只作为迁移参考，不作为默认路径：

- `datasets/vision/legacy_camara/workspace_dataset_camara/row/images` 约 101 张原始图片
- `datasets/vision/legacy_camara/workspace_dataset_camara/row/cylinder` 约 100 张原始图片
- `datasets/vision/legacy_camara/workspace_dataset_camara/captures` 下有若干 session，但目前基本只有 `session_meta.json`

这些 `row` 图片更像早期原始/分类照片，不是完整 YOLO segmentation 训练集；若要继续训练当前模型，需要补齐 `labels/train`、`labels/val` 的多边形标注。

旧 `05_Vision_Block_Recognition/dataset` Roboflow 导出已迁移到：

- `datasets/vision/legacy_migrated/05_Vision_Block_Recognition_dataset`
