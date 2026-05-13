# Block Dataset Collector

这是一个用于采集 JetMax 机械臂摄像头图像的数据采集程序，主要给木块识别模型训练使用。

程序入口文件：

- `<repo>\06_Data_Collection\2026-04_jetmax_block_dataset_collection\block_dataset_collector.py`

默认相机源：

- `http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80`

默认保存根目录：

- `<repo>\datasets\vision\captures`

## 这个程序能做什么

- 实时预览 JetMax 摄像头画面
- 单张保存
- 连拍保存
- 定时自动采集
- 每次采集自动创建一个 session 目录
- 自动写入 `manifest.jsonl`
- 支持记录 `scene tag`、`split`、`note`、`negative sample`
- 支持在采集界面控制机械臂 `theta/r/z` 位置，便于采不同视角和高度的数据集
- 异步写盘，尽量减少保存图片时的卡顿

## 如何启动

最方便的方式：

- 双击或在 PyCharm 里运行：
  - `<repo>\06_Data_Collection\2026-04_jetmax_block_dataset_collection\START_BLOCK_DATASET_COLLECTOR.py`
- 或者在 PyCharm 里直接选择运行配置：
  - `Block_Dataset_Collector`

### 方式 1：Python 启动脚本

在 PowerShell 中运行：

```powershell
python "<repo>\06_Data_Collection\2026-04_jetmax_block_dataset_collection\START_BLOCK_DATASET_COLLECTOR.py"
```

这个入口会自动优先使用 `BRAIN_PYTHON_EXE`、项目 `.venv` 或
`brain-vision` conda 环境来启动真正的采集器。

### 方式 2：PyCharm 启动

已经提供现成运行配置：

- `Block_Dataset_Collector`

如果需要手动创建，配置如下：

- `Script path`
  - `<repo>\06_Data_Collection\2026-04_jetmax_block_dataset_collection\block_dataset_collector.py`
- `Python interpreter`
  - `%BRAIN_PYTHON_EXE%`
- `Working directory`
  - `<workspace>`

默认情况下不需要额外参数，直接运行即可连接 JetMax 相机流。

如果要使用界面里的机械臂位置控制功能，需要 JetMax 端已经启动
`hybrid_controller` 的控制 runtime。采集器默认使用 `auto` 链路：

- 优先连接 ROS bridge：`192.168.149.1:9091`
- 如果 ROS 不通，再回退到 TCP legacy runtime：`192.168.149.1:8888`

位置控制按钮会调用 `move_cyl` / `MOVE_CYL`，按当前界面里的 `theta/r/z` 目标移动。
如果 runtime 没启动，采集器仍然可以正常预览和保存图片，只是高度调整会提示失败。

## 数据保存在哪里

默认所有采集数据都会保存到：

`<repo>\datasets\vision\captures`

每次创建新 session 后，会生成一个独立目录，例如：

```text
<repo>\datasets\vision\captures\
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
- `robot_pose_cyl`
- `home_height_z_mm`
- `last_robot_status`
- `last_robot_height_command`

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
- `theta`
  - 机械臂圆柱坐标角度，单位度
- `r`
  - 机械臂圆柱坐标半径，单位 mm
- `z`
  - 机械臂拍摄高度，单位 mm
- `步长`
  - 小步移动按钮使用的步长；`theta` 按度调整，`r/z` 按 mm 调整
- `读取位置`
  - 从 JetMax runtime 读取当前机械臂位置，并填回 `theta/r/z`
- `移动到该位置`
  - 把机械臂移动到当前 `theta/r/z`
- `Home 水平位置`
  - 把 `theta/r` 设回默认 Home 水平位置，并使用当前 `z` 移动
- `theta -` / `theta +` / `r -` / `r +` / `z -` / `z +`
  - 按当前步长做小步移动，便于快速调整拍摄视角

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
- `H`
  - 移动机械臂到当前 `theta/r/z`
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
- 采不同机械臂位置的数据集时，先设置 `theta/r/z` 或使用小步按钮，等状态显示完成后再采集。

## 可选参数

示例：

```powershell
& "%BRAIN_PYTHON_EXE%" `
  "<repo>\06_Data_Collection\2026-04_jetmax_block_dataset_collection\block_dataset_collector.py" `
  --source "0" `
  --output-root "<repo>\datasets\vision\captures" `
  --session-prefix "block_collect" `
  --image-ext "jpg" `
  --jpeg-quality 95 `
  --home-z-mm 160 `
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
- `--robot-host`
  - JetMax runtime 主机地址，默认 `192.168.149.1`
- `--robot-transport`
  - 位置控制链路，`auto` / `ros` / `tcp`，默认 `auto`
- `--rosbridge-port`
  - JetMax ROS bridge 端口，默认 `9091`
- `--robot-port`
  - JetMax TCP legacy runtime 端口，默认 `8888`
- `--robot-timeout-sec`
  - 机械臂命令等待超时时间
- `--home-theta-deg`
  - 无法从 runtime 读取 `home_pose` 时使用的 Home 角度，默认 `0`
- `--home-radius-mm`
  - 无法从 runtime 读取 `home_pose` 时使用的 Home 半径，默认 `120`
- `--home-z-mm`
  - 界面默认 Home 高度，默认 `160`

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

- `<repo>\datasets\vision\captures\block_collect_20260407_163955`
