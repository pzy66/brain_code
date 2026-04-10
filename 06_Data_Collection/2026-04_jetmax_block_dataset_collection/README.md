# Block Dataset Collector

这是一个用于采集 JetMax 机械臂摄像头图像的数据采集程序，主要给木块识别模型训练使用。

程序入口文件：

- `C:\Users\P1233\Desktop\brain\brain_code\06_Data_Collection\2026-04_jetmax_block_dataset_collection\block_dataset_collector.py`

默认相机源：

- `http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80`

默认保存根目录：

- `C:\Users\P1233\Desktop\brain\dataset\camara\captures`

## 这个程序能做什么

- 实时预览 JetMax 摄像头画面
- 单张保存
- 连拍保存
- 定时自动采集
- 每次采集自动创建一个 session 目录
- 自动写入 `manifest.jsonl`
- 支持记录 `scene tag`、`split`、`note`、`negative sample`
- 异步写盘，尽量减少保存图片时的卡顿

## 如何启动

最方便的方式：

- 双击运行：
  - `C:\Users\P1233\Desktop\brain\brain_code\06_Data_Collection\2026-04_jetmax_block_dataset_collection\START_BLOCK_DATASET_COLLECTOR.cmd`
- 或者在 PyCharm 里直接选择运行配置：
  - `Block_Dataset_Collector`

### 方式 1：PowerShell 启动

在 PowerShell 中运行：

```powershell
& "C:\Users\P1233\miniconda3\envs\brain-vision\python.exe" `
  "C:\Users\P1233\Desktop\brain\brain_code\06_Data_Collection\2026-04_jetmax_block_dataset_collection\block_dataset_collector.py"
```

### 方式 2：PyCharm 启动

已经提供现成运行配置：

- `Block_Dataset_Collector`

如果需要手动创建，配置如下：

- `Script path`
  - `C:\Users\P1233\Desktop\brain\brain_code\06_Data_Collection\2026-04_jetmax_block_dataset_collection\block_dataset_collector.py`
- `Python interpreter`
  - `C:\Users\P1233\miniconda3\envs\brain-vision\python.exe`
- `Working directory`
  - `C:\Users\P1233\Desktop\brain`

默认情况下不需要额外参数，直接运行即可连接 JetMax 相机流。

## 数据保存在哪里

默认所有采集数据都会保存到：

`C:\Users\P1233\Desktop\brain\dataset\camara\captures`

每次创建新 session 后，会生成一个独立目录，例如：

```text
C:\Users\P1233\Desktop\brain\dataset\camara\captures\
  block_collect_20260407_163955\
    session_meta.json
    manifest.jsonl
    images\
      train\
      val\
      test\
      raw\
```

其中：

- `block_collect_20260407_163955`
  - 一次采集会话目录
- `session_meta.json`
  - 保存本次 session 的基础信息，例如相机源、图像格式、创建时间
- `manifest.jsonl`
  - 每保存一张图片，就追加一行 JSON 记录
- `images/train`
  - 保存当前 `split=train` 的图片
- `images/val`
  - 保存当前 `split=val` 的图片
- `images/test`
  - 保存当前 `split=test` 的图片
- `images/raw`
  - 保存当前 `split=raw` 的图片

## manifest.jsonl 里记录什么

每保存一张图，都会写入一行 JSON，主要字段包括：

- `timestamp`
- `session_name`
- `frame_id`
- `mode`
  - `manual` / `burst` / `auto`
- `split`
- `scene_tag`
- `negative_sample`
- `note`
- `image_path`
- `capture_age_ms`
- `frame_size`
- `sharpness`
- `delta_from_last_saved`

这份文件后续可以直接用于：

- 标注前筛图
- 按场景筛样本
- 检查 train/val/test 是否混乱

## 界面里要填什么

- `Session Prefix`
  - 会话名前缀，默认是 `block_collect`
- `Scene Tag`
  - 当前场景标签，例如：
  - `single_block`
  - `double_block`
  - `stacked`
  - `edge_case`
  - `negative`
- `Split`
  - 当前图片要保存到哪个目录：`train` / `val` / `test` / `raw`
- `Note`
  - 可选短备注
- `Negative Sample`
  - 勾选后表示当前采集的是负样本

## 快捷键

- `Space`
  - 保存当前帧
- `B`
  - 开始连拍
- `A`
  - 开关自动采集
- `N`
  - 开关负样本标记
- `S`
  - 新建 session
- `Esc`
  - 退出程序

## 常用采集建议

- 不要对连续几乎不变的画面每帧都保存。
- 尽量在姿态、光照、位置、遮挡发生变化后再采。
- `Scene Tag` 尽量认真填写，后面整理训练集会很有用。
- 最好在采集阶段就把 `Split` 分好，避免后面重新整理时混乱。
- 要专门采一些难例：
  - 反光
  - 阴影
  - 视野边缘
  - 堆叠
  - 空场景
- 同一摆放序列不要同时放到 `train` 和 `val/test`

## 可选参数

示例：

```powershell
& "C:\Users\P1233\miniconda3\envs\brain-vision\python.exe" `
  "C:\Users\P1233\Desktop\brain\brain_code\06_Data_Collection\2026-04_jetmax_block_dataset_collection\block_dataset_collector.py" `
  --source "0" `
  --output-root "C:\Users\P1233\Desktop\brain\dataset\camara\captures" `
  --session-prefix "block_collect" `
  --image-ext "jpg" `
  --jpeg-quality 95 `
  --fullscreen
```

主要参数：

- `--source`
  - 摄像头编号或视频流 URL
- `--output-root`
  - 输出根目录
- `--session-prefix`
  - session 名前缀
- `--image-ext`
  - `jpg` 或 `png`
- `--jpeg-quality`
  - JPEG 质量，默认 `95`
- `--exit-after-sec`
  - 运行若干秒后自动退出
- `--fullscreen`
  - 全屏显示

## 一个最简单的使用流程

1. 启动程序
2. 新建一个 session
3. 设置 `Scene Tag`
4. 选择 `Split`
5. 用 `Space`、`B` 或 `A` 开始采集
6. 到当前 session 目录查看图片和 `manifest.jsonl`

## 已验证

这个采集程序已经做过：

- Python 语法检查
- 短时间启动烟雾测试
- 自动创建 session 目录测试

示例 session 目录：

- `C:\Users\P1233\Desktop\brain\dataset\camara\captures\block_collect_20260407_163955`
