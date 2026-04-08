# 05 小木块识别与 SSVEP 视觉主程序

这个目录现在已经按“当前主入口 + 辅助工具 + 旧参考脚本”的方式整理过了。

当前推荐你真正运行的主程序只有一个：

- `C:\Users\P1233\Desktop\brain\brain_code\05_Vision_Block_Recognition\2026-03_yolo_camera_detection\block_center_ssvep_single.py`

它负责：

- 读取 JetMax 机械臂摄像头视频流
- 识别小木块顶面
- 对最多 4 个目标做不同频率的 SSVEP 掩膜闪烁
- 持续通过 `stdout JSON Lines` 输出当前槽位和像素中心坐标

## 便捷入口

最方便的启动方式就是直接使用当前目录里的双击入口。

### 方式 1：双击启动

直接双击下面这个脚本：

- `C:\Users\P1233\Desktop\brain\brain_code\05_Vision_Block_Recognition\2026-03_yolo_camera_detection\START_BLOCK_SSVEP_VIEWER.cmd`

这个入口默认会使用：

- 权重：`C:\Users\P1233\Desktop\brain\dataset\camara\best.pt`
- 相机流：`http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80`
- 设备：`auto`
- `imgsz=512`
- `max_det=6`
- `warmup_runs=1`

如果你想临时追加参数，也可以在命令行后面继续加，例如：

```powershell
cmd /c "C:\Users\P1233\Desktop\brain\brain_code\05_Vision_Block_Recognition\2026-03_yolo_camera_detection\START_BLOCK_SSVEP_VIEWER.cmd --fullscreen"
```

### 方式 2：PyCharm 运行

直接在 PyCharm 里选择运行配置：

- `Vision_Block_SSVEP_Robot_Camera`

## 程序输出到哪里

这个主程序默认**不保存图片文件**，而是：

- 在窗口中显示实时识别和 SSVEP 掩膜闪烁
- 把识别结果持续写到标准输出 `stdout`
- 把状态信息、报错、选中槽位等写到标准错误 `stderr`

`stdout` 输出的是一行一条的 JSON，主要字段包括：

- `timestamp`
- `frame_id`
- `image_size`
- `selected_slot`
- `selected_center`
- `capture_fps`
- `packet_fps`
- `queue_age_ms`
- `infer_ms`
- `post_ms`
- `detected_count`
- `slots`

## 快捷键

- `1-4`：选择槽位 1 到 4
- `0`：清除当前选择
- `Esc`：退出程序

## 当前目录里各文件的作用

### 当前主程序

- `block_center_ssvep_single.py`
  - 当前推荐版本
  - 单文件实现实时识别、精确 SSVEP 掩膜闪烁和中心坐标输出

### 辅助工具

- `jetmax_stream_probe.py`
  - 用来测试 JetMax `web_video_server` 不同 URL 参数下的实际取流帧率
- `test_FPS.py`
  - 早期 FPS 测试脚本，保留作参考

### 旧参考脚本

- `deeplearning.py`
  - 旧版实验代码，保留作参考
- `computer/test2.py`
  - 原始联调脚本，依赖更重，不再作为当前主入口
- `computer/test2_CPU.py`
  - CPU 版本旧脚本，保留作参考

## 建议的使用顺序

如果你只是要直接看视觉效果并联调机械臂摄像头，按这个顺序就够了：

1. 先双击 `START_BLOCK_SSVEP_VIEWER.cmd`
2. 看窗口里木块识别和 SSVEP 闪烁是否正常
3. 看控制台里的 JSON 输出是否有正确的中心坐标
4. 如果怀疑视频流本身有问题，再单独运行 `jetmax_stream_probe.py`

## 常见说明

- 如果窗口能开起来，但没有识别结果，先检查权重文件 `dataset/camara/best.pt` 是否存在。
- 如果画面打不开，先检查 JetMax 是否在线，以及 `192.168.149.1:8080` 这路流是否正常。
- 如果你之后要继续增强识别准确率，建议只在这个主程序上继续迭代，不要再回到 `computer/test2.py` 那条旧主线。
