# Hybrid Controller 启动与部署说明

本文只覆盖当前真机主线，不覆盖历史实验目录。

## 主线目录

- 仓库内相对路径：`.\hybrid_controller`

## 解释器

- 默认优先使用：`.\.venv\Scripts\python.exe`
- 也可使用你自己准备的 `brain_code` 环境，并通过 `BRAIN_PYTHON_EXE` 覆盖

## 电脑端启动

在 `hybrid_controller` 目录执行：

```bash
python run_real.py
python run_real_ssvep.py
```

## JetMax 端部署与启动

将整个 `hybrid_controller` 目录复制到 JetMax：

- `/home/hiwonder/brain_code/hybrid_controller`

JetMax 上执行：

```bash
cd ~/brain_code/hybrid_controller/robot
python3 -m pip install -r requirements-jetmax-robot-python.txt
bash run_hybrid_controller_ros_runtime.sh
```

## 端口约定（当前主线）

- `9091`：ROS bridge（主链必须）
- `8080`：web_video_server（视觉识别阶段使用，启动 runtime 时不拉流）
- `8888`：TCP 兼容/诊断（可选）

`run_real.py` 默认用于“机械臂重启后，电脑连接 JetMax Wi-Fi，然后启动主程序”的流程：GUI 启动后会自动连接 `rosbridge`；如果首连失败，允许通过 SSH 自动启动一次 JetMax 控制 runtime。自动启动只恢复控制链路，固定传入 `--skip-camera-check --skip-tcp-check`，不会检查 `8080`，不会拉取视频，不会启动、重启或修复 `usb_cam.service` / `web_video_server`。一次自动启动失败后只进入温和重连，不反复 SSH。

视觉自动启动是电脑端读取行为，不是相机发送端管理行为。GUI 只有在 `rosbridge` 已连接、runtime 不在启动中、机器人状态新鲜后，才创建 `VisionRuntime` 并读取唯一官方 URL：`http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80`。等待控制稳定时显示 `waiting_control:*`，开始读官方流时显示 `starting_8080:*`，运行后显示 `camera_fps=...`。

点击“连接机器人”或启动自动连接时默认也不做额外 `9091` TCP 预探测，而是直接建立 rosbridge WebSocket。只有显式加 `--ros-probe-before-connect` 时才恢复预探测。

如果要关闭启动即连接，可以显式加：

```powershell
python .\hybrid_controller\run_real.py --no-robot-connect-on-start --robot-auto-start-disabled
```

如果连接 JetMax Wi-Fi 后 Windows 自己断开，先做本机只读诊断：

```powershell
python -m hybrid_controller.tools.diagnose_jetmax_wifi_windows
```

该工具只读 Windows WLAN 事件、路由、DNS 和 Intel 网卡高级属性，不 ping 机械臂、不 SSH、不扫端口、不拉相机视频。重点检查：

- 是否有 `0.0.0.0/0 -> 192.168.149.1` 默认路由落在 WLAN 上。
- WLAN DNS 是否指向 `192.168.149.1`。
- WLAN AutoConfig 是否出现 `4003 limited connectivity recovery`。
- WLAN AutoConfig 是否出现 `8003 网络被驱动程序断开连接`。
- Intel 网卡是否仍为 Roaming Aggressiveness=中间、MIMO Power Save=自动 SMPS、Packet Coalescing=启用。

推荐网络形态：WLAN 只负责 `192.168.149.0/24` 机械臂本地网段；公网默认路由和公共 DNS 留给以太网或其他联网网卡。若仍断开，再按顺序尝试降低 Intel 漫游主动性、关闭 MIMO 省电、关闭包合并，并保留回滚记录。

`robot/tools/jetmax_start_ros_runtime.py` 默认只校验 `9091`，不连接 `8080`，不订阅 `/usb_cam/image_rect_color`，不拉取 MJPEG/H264 视频流。
脚本会自动 `stop/disable` JetMax 自带的 `rosbridge.service`，避免旧消息定义冲突。
如需强制校验 `8888`，加 `--require-tcp-check`。
如需单独验证官方视频流，必须显式加 `--check-camera-stream`。

摄像头输出链路定死为 JetMax/Hiwonder 官方默认方式：

```text
usb_cam.service -> usb_cam_node -> /usb_cam/image_rect_color -> web_video_server:8080 -> PC
```

默认启动、部署、ROS runtime 自恢复都不得改写 `/home/hiwonder/ros/autostart/usb_cam.launch`，不得停止/重启 `usb_cam.service`，不得重载 `uvcvideo`，不得让 hybrid runtime 接管 `web_video_server`。上位机视觉识别只消费官方端口输出。

## 路由契约（ROS 模式）

当 `robot_transport=ros` 时，以下命令必须走 ROS，不静默回退 TCP：

- `MOVE_CYL`
- `MOVE_CYL_AUTO`
- `PICK_WORLD`
- `PICK_CYL`
- `PLACE`
- `ABORT`
- `RESET`

## 抓取命令约定（当前实现）

- 在 `vision_mode=robot_camera_detection` 下，默认启用连续视觉伺服，但自动视觉只识别和显示目标，不会自行启动运动或抓取。
- 只有手动 `Pick 1-4` 或 SSVEP/任务状态机确认目标后，才创建 continuous pending；没有 pending 时显示 `continuous_idle awaiting_pick`。
- 后续视觉包只续跑已锁定 slot，并通过 `/hybrid_controller/teleop_cyl_cmd` 连续发布 `theta_rate_deg_s / radius_rate_mm_s / z_rate_mm_s`，且 `use_auto_z=false`。
- 到确认高度并稳定对中后发最终 `PICK_CYL`，当前 profile 固化为 `vision_pick_confirm_z_mm=130.0`、`vision_eye_in_hand_pick_radius_bias_mm=40.0`。
- 离散 `MOVE_CYL -> wait -> re-detect` 保留为 fallback，可用 `--no-vision-continuous-servo` 关闭连续模式。
- 发真实 `PICK_CYL` 前仍检查 profile ready、ROS 状态新鲜、机器人不 busy、slot 可执行、目标未过期、中心稳定和最终半径不越界。
- 如果机器人端处于 `sucker_frozen`，最终 PICK 按 dry-run 语义执行，GUI/日志会显示 `release_mode_effective=sucker_frozen`。

## 当前控制主语义

- 真机 GUI 主 transport：ROS
- TCP：兼容/诊断 fallback
- teleop 主语义：圆柱坐标
- 默认范围：
  - `theta ∈ [-120°, 120°]`
  - `MOVE_CYL_AUTO radius ∈ [80, 260]`
  - `MOVE_CYL radius ∈ [50, 280]`
