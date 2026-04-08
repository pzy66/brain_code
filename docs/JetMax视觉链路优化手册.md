# JetMax 视觉链路优化手册

## 当前结论

- 识别主瓶颈不在 RTX 4080 上的 YOLO 推理，而在 JetMax 相机与 ROS 取流链路。
- 视觉主程序继续保留在 PC 端执行，JetMax 负责发布相机图像和执行机械臂动作。
- 当前机械臂抓取坐标契约依赖 `/usb_cam/image_rect_color` 的像素空间，不能随便改成别的图像源后直接复用原来的 `PICK x y`。

## 已落地的工具

- PC 侧视觉主程序：
  - [block_center_ssvep_single.py](C:/Users/P1233/Desktop/brain/brain_code/05_Vision_Block_Recognition/2026-03_yolo_camera_detection/block_center_ssvep_single.py)
- PC 侧 web_video_server 探测器：
  - [jetmax_stream_probe.py](C:/Users/P1233/Desktop/brain/brain_code/05_Vision_Block_Recognition/2026-03_yolo_camera_detection/jetmax_stream_probe.py)
- JetMax 侧诊断脚本：
  - [jetmax_camera_probe.sh](C:/Users/P1233/Desktop/brain/brain_code/03_RobotArm_Control/2026-03_jetmax_execution_server/jetmax_camera_probe.sh)
- PC 侧自动基准脚本：
  - [benchmark_jetmax_viewer.ps1](C:/Users/P1233/Desktop/brain/scripts/benchmark_jetmax_viewer.ps1)
  - [benchmark_jetmax_viewer.cmd](C:/Users/P1233/Desktop/brain/scripts/benchmark_jetmax_viewer.cmd)

## 当前实测结果

- JetMax `web_video_server` URL 探测：
  - `baseline` 约 `23.8 FPS`
  - `type=mjpeg&width=640&height=480&quality=80` 约 `22.9 FPS`
  - 带 `default_transport=compressed` 的几个配置在当前 JetMax 上都无法正常打开
- 视觉主程序优化版平均值：
  - `capture_fps` 约 `19.9`
  - `packet_fps` 约 `10.0`
  - `queue_age_ms` 约 `5.1`
  - `infer_ms` 约 `34.0`
  - 这说明当前版本已经不再积压旧帧，主瓶颈也不在推理阶段

## 默认运行方式

- PyCharm 默认运行配置已经改成显式 MJPEG，不再带当前无效的 `default_transport=compressed`
- 直接运行视觉主程序：

```powershell
& "C:\Users\P1233\miniconda3\envs\brain-vision\python.exe" `
  "C:\Users\P1233\Desktop\brain\brain_code\05_Vision_Block_Recognition\2026-03_yolo_camera_detection\block_center_ssvep_single.py" `
  --weights "C:\Users\P1233\Desktop\brain\dataset\camara\best.pt" `
  --source "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80" `
  --device 0 `
  --imgsz 512 `
  --max-det 6 `
  --warmup-runs 1 `
  --fullscreen
```

## 阶段 0：先定位 JetMax 端真实瓶颈

### JetMax 端

```bash
bash /home/jetmax/jetmax_camera_probe.sh 10
```

如果脚本不在 JetMax 上，就把 [jetmax_camera_probe.sh](C:/Users/P1233/Desktop/brain/brain_code/03_RobotArm_Control/2026-03_jetmax_execution_server/jetmax_camera_probe.sh) 传上去再执行。

重点看这几个结果：

- `v4l2-ctl --list-formats-ext`
- `rostopic hz /usb_cam/image_raw`
- `rostopic hz /usb_cam/image_rect_color`
- `rostopic bw /usb_cam/image_rect_color`
- `tegrastats`

### PC 端

```powershell
& "C:\Users\P1233\miniconda3\envs\brain-vision\python.exe" `
  "C:\Users\P1233\Desktop\brain\brain_code\05_Vision_Block_Recognition\2026-03_yolo_camera_detection\jetmax_stream_probe.py" `
  --base-url "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color" `
  --duration 4
```

## 判定逻辑

- 如果 `/usb_cam/image_raw` 也只有 `6-8 FPS`
  - 先改 `usb_cam`
  - 重点收紧到 `640x480 @ 30 FPS`
  - 优先试 `io_method=mmap`
  - 相机支持时优先试 MJPEG
- 如果 `/usb_cam/image_raw` 高，但 `/usb_cam/image_rect_color` 明显低
  - 说明 `image_proc/rectify` 是主要瓶颈
  - 下一步改成 JetMax 只发 `image_raw + camera_info`，PC 端做去畸变
  - 但切换前必须验证像素空间与现有抓取标定一致
- 如果 ROS topic 很快，但 Windows 侧 HTTP 拉流还是慢
  - 优先改 `web_video_server` URL 参数
  - 当前 JetMax 上不要启用 `default_transport=compressed`
  - 再不够就进入 GStreamer/RTSP 路线

## 阶段 1：优先保留现有像素语义

- 现阶段默认仍使用 `/usb_cam/image_rect_color`
- 原因：
  - [test2.py](C:/Users/P1233/Desktop/brain/brain_code/05_Vision_Block_Recognition/2026-03_yolo_camera_detection/computer/test2.py) 直接把图像坐标发给机械臂
  - [test2_robot.py](C:/Users/P1233/Desktop/brain/brain_code/03_RobotArm_Control/2026-03_jetmax_execution_server/test2_robot.py) 用当前标定参数做像素到世界坐标转换
- 只要还没重新验证标定一致性，就不要把正式抓取链路切到别的图像空间

## 阶段 2：需要时再把校正搬到 PC

- 触发条件：
  - `/usb_cam/image_raw` 可以稳定到 `20+ FPS`
  - `/usb_cam/image_rect_color` 明显更低
- 做法：
  - JetMax 发布 `image_raw + camera_info`
  - PC 端去畸变后再做检测和 SSVEP
- 验收门槛：
  - 同一物点在新旧链路里的中心偏差均值不超过 `3 px`

## 阶段 3：只有 HTTP 拉流慢时才替换传输协议

- 如果 ROS topic 已经够快，但 `:8080/stream` 还是明显偏慢
- 再考虑 Jetson 侧 GStreamer 硬件编码和 RTSP
- 这一步仍然要保留校正后图像语义，或者补一层明确的像素映射验证

## 自动基准

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\P1233\Desktop\brain\scripts\benchmark_jetmax_viewer.ps1" -DurationSec 15
```

或者直接运行：

```powershell
& "C:\Users\P1233\Desktop\brain\scripts\benchmark_jetmax_viewer.cmd" -DurationSec 15
```

脚本会自动：

- 启动视觉主程序
- 运行固定时长
- 自动退出
- 将 stdout JSON 和 stderr 日志写到 `logs/`
- 输出平均 `capture_fps`、`packet_fps`、`queue_age_ms`、`infer_ms`、`post_ms`、`detected_count`

## 当前默认建议

- 识别继续放在 PC 4080 上
- JetMax 端先优化 `usb_cam/image_proc/web_video_server`
- 当前正式运行先用：
  - `/usb_cam/image_rect_color`
  - `imgsz=512`
  - `max_det=6`
  - `warmup_runs=1`
  - `type=mjpeg&width=640&height=480&quality=80`
- 当前 JetMax 不要把 `default_transport=compressed` 作为默认路径
