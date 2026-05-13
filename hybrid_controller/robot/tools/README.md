# robot/tools（部署与联调工具）

这些工具用于桌面端远程启动和探针，不是机械臂端常驻服务。

## 主线工具（建议使用）

- `jetmax_start_ros_runtime.py`
  - 通过 SSH 一键启动 JetMax ROS runtime
  - 会清理残留进程并停用系统自启旧 rosbridge
  - 默认不重写、不重启 JetMax 官方 `usb_cam.service`
  - 默认不订阅 `/usb_cam/image_rect_color`，不请求 `8080` MJPEG/H264 视频流
  - 只有显式传 `--check-camera-stream` 时才做相机读取验证
  - 只有显式传 `--allow-camera-sender-mutation` 加修复参数时才允许修改相机发送
- `ros_service_probe.py`
  - 通过 rosbridge 调用 ROS 服务，做联通与动作探针
- `diagnose_jetmax_wifi_windows.py`（在 `hybrid_controller/tools/`）
  - Windows 本机只读 Wi-Fi 诊断
  - 不 ping、不 SSH、不扫端口、不拉相机视频，优先用于排查 JetMax Wi-Fi 自己断开
- `jetmax_move_probe.py`
  - TCP 兼容链路的移动探针（可选）
- `jetmax_env_probe.py`
  - JetMax 端运行环境自检（可选）

## legacy 工具（仅兼容）

- `jetmax_start_runtime.py`（默认拒绝执行；必须显式加 `--allow-legacy-tcp-start`）
- `deploy_jetmax_runtime.py`

说明：这两项是 TCP-only 老路径，不是当前 GUI 的主链。需要临时回退到 TCP-only 时优先用 `deploy_jetmax_runtime.py`，因为它会同步 `robot_runtime_py36.py` 及其运行依赖；`jetmax_start_runtime.py` 只作为历史兼容入口保留，避免误用旧单文件启动路径。

## 一键启动（ROS 主链）

```powershell
.\.venv\Scripts\python.exe .\hybrid_controller\robot\tools\jetmax_start_ros_runtime.py --host 192.168.149.1 --user hiwonder --password $env:JETMAX_PASSWORD --remote-root /home/hiwonder/brain_code
```

默认检查：

- `9091`（rosbridge）

GUI 点击“连接机器人”时默认直接建立 rosbridge WebSocket，不再额外预探测 `9091`。如确实需要端口对照诊断，显式使用 `--ros-probe-before-connect`。

若 Windows 连接 JetMax Wi-Fi 后自己断开，先在电脑端运行：

```powershell
python -m hybrid_controller.tools.diagnose_jetmax_wifi_windows
```

重点确认 WLAN 是否被 Windows 配成默认网关/DNS 到 `192.168.149.1`，以及是否出现 WLAN AutoConfig `4003`/`8003` 事件。该问题属于本机网络层，不要通过重启 `usb_cam.service`、`web_video_server` 或拉取视频流来验证。

默认不检查：

- `8080`（官方 `usb_cam.service` 提供的 web_video_server）
- `/usb_cam/image_rect_color`（官方 rectified color topic）
- 任何 MJPEG/H264 视频帧读取

## 摄像头链路保护

默认启动工具不验证、不修复、不接管摄像头：

```text
usb_cam.service -> /usb_cam/image_rect_color -> web_video_server:8080
```

默认 PC 取流地址固定为：

```text
http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80
```

改动工具时需要保持：

- 不要默认调用 `repair_official_camera_sender()`。
- 不要默认停止或重启 `usb_cam.service`。
- 不要默认 `pkill web_video_server`。
- 不要默认连接 `8080` 视频流或订阅 `/usb_cam/image_rect_color`。
- 不要恢复 `web_video_server` 接管逻辑；`--manage-web-video` / `--restart-web-video` 是拒绝执行的废弃参数。
- 不要恢复多个摄像头候选 URL 轮询；默认只读 `/usb_cam/image_rect_color`。

可选参数：

- `--require-tcp-check`：额外强制检查 `8888`
- `--skip-tcp-check`：显式跳过 TCP 检查
- `--check-camera-stream`：显式检查 `8080` 和 `/usb_cam/image_rect_color`；会拉取视频流
- `--skip-camera-check`：兼容参数；默认已经跳过相机检查
- `--repair-camera-sender`：显式改写/重启官方 `usb_cam.service` 相机发送配置
- `--skip-camera-repair`：兼容旧参数；默认已经不修复摄像头发送
- `--repair-camera-driver`：显式写入/重载 `uvcvideo` 兼容参数，仅用于诊断；必须同时带 `--allow-camera-sender-mutation`。默认不强制 `quirks`，并保持 `nodrop=0`，让驱动丢弃不完整帧，避免条带/拼接坏帧进入视觉识别。
- `--allow-camera-sender-mutation`：允许本工具改动相机发送链路；修复类参数必须同时带这个开关
- `--remove-camera-driver-override`：显式移除本工具写过的 `uvcvideo` 覆盖文件；默认不碰
- `--keep-camera-driver-override`：兼容旧参数；默认已经保留已有覆盖文件
- `--camera-only`：只修复官方相机发送，不同步 runtime、不处理 rosbridge
- `--camera-stream-type mjpeg|h264`：修复 `usb_cam.launch` 时写入的 `web_video_server` 类型；当前锁定默认是 `mjpeg`
- `--camera-framerate 20`：修复 `usb_cam.launch` 时写入的 `usb_cam_node` 帧率；当前锁定默认是 `20`
- `--manage-web-video`：废弃参数；当前会拒绝执行
- `--restart-web-video`：废弃参数；当前会拒绝执行

相机单独修复：

```powershell
.\.venv\Scripts\python.exe .\hybrid_controller\robot\tools\jetmax_start_ros_runtime.py --camera-only --host 192.168.149.1 --user hiwonder --password $env:JETMAX_PASSWORD
```

修复相机发送链路必须显式确认：

```powershell
.\.venv\Scripts\python.exe .\hybrid_controller\robot\tools\jetmax_start_ros_runtime.py --camera-only --allow-camera-sender-mutation --host 192.168.149.1 --user hiwonder --password $env:JETMAX_PASSWORD
```

当前已验证的机械臂端摄像头参数：

```text
/home/hiwonder/ros/autostart/usb_cam.launch:
  video_device=/dev/usb_cam0
  image_width=640
  image_height=480
  pixel_format=yuyv
  framerate=20
  io_method=mmap
  web_video_server type=mjpeg
  web_video_server quality=80

uvcvideo:
  不保留 hybrid 工具写入的 /etc/modprobe.d/hiwonder-uvcvideo.conf
  当前修复目标是默认不强制 quirks，nodrop=0，让驱动丢弃不完整帧
```

## ROS 服务探针

```powershell
.\.venv\Scripts\python.exe .\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action status
.\.venv\Scripts\python.exe .\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action pick_world --x 0 --y -162.94
.\.venv\Scripts\python.exe .\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action place
.\.venv\Scripts\python.exe .\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action reset
```

支持 `--action`：

- `status`
- `move_cyl`
- `move_cyl_auto`
- `pick_world`
- `pick_cyl`
- `place`
- `abort`
- `reset`

## 机械臂端补丁备份

修改 `robot/` 下的实机相关文件前，先在上位机创建备份目录：

```text
hybrid_controller/robot/backups/robot_runtime_<purpose>_<timestamp>/
```

备份必须包含 `MANIFEST.json`，记录每个文件的相对路径和 SHA256。建议覆盖 ROS runtime node、`runtime/` 核心文件、启动/部署工具和本说明文档。备份只保存在上位机，不随启动工具自动同步到 JetMax。
