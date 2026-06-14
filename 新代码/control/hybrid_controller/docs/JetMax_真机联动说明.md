# JetMax 真机联动说明

## 联动目标

当前主线是：

- JetMax 端运行 ROS runtime
- 电脑端运行 `hybrid_controller` 主界面
- 通过 rosbridge + ROS service 完成控制

## JetMax 端放置目录

- 复制源：`<repo>\hybrid_controller`
- JetMax 目标：`/home/hiwonder/brain_code/hybrid_controller`

## JetMax 端启动

```bash
cd ~/brain_code/hybrid_controller/robot
python3 -m pip install -r requirements-jetmax-robot-python.txt
bash run_hybrid_controller_ros_runtime.sh
```

## 电脑端启动

```bash
cd <repo>\hybrid_controller
python run_real.py
# 或
python run_real_ssvep.py
```

GUI 默认适配“机械臂重启后，电脑连接 JetMax Wi-Fi，然后启动主程序”的流程：启动后自动连接 `rosbridge`；如果首连失败，允许自动 SSH 启动一次 JetMax 控制 runtime。自动启动固定使用相机安全参数，只恢复 `9091` 控制链路，不检查 `8080`，不拉取视频，不启动、重启或修复 `usb_cam.service` / `web_video_server`，不会改官方摄像头发送链路。一次自动启动失败后只进入温和重连，不反复 SSH。

视觉默认自动启动，但必须等控制链路稳定：`rosbridge` 已连接、runtime 不在启动中、机器人状态新鲜。满足后电脑端只创建 `VisionRuntime` 并读取官方 8080 MJPEG URL：

```text
http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80
```

状态栏含义：`waiting_control:*` 表示控制链路还没到可读视觉的条件，`starting_8080:*` 表示开始读官方视频流，`camera_fps=...` 表示已经收到图像，`unavailable` 表示电脑端读取失败。

启动自动连接和点击“连接机器人”时都默认直接建立 rosbridge WebSocket，不再先额外探测 `9091` TCP 端口。只有显式加 `--ros-probe-before-connect` 时才启用预探测。调试时不要用 ping、SSH、端口扫描或拉相机流来确认 Wi-Fi 是否稳定；先看 Windows 本机事件：

```powershell
python -m hybrid_controller.tools.diagnose_jetmax_wifi_windows
```

如果诊断看到 WLAN 默认路由 `0.0.0.0/0 -> 192.168.149.1`、WLAN DNS `192.168.149.1`、WLAN AutoConfig `4003 limited connectivity recovery` 或 `8003 网络被驱动程序断开连接`，优先按本机网络问题处理。目标形态是 WLAN 只保留 `192.168.149.0/24` 机械臂本地路由，公网默认路由和 DNS 不走 JetMax AP。

## 主链端口

- `9091`：ROS bridge（必须，默认主链端口）
- `8080`：视频流（必须）
- `8888`：TCP 兼容链路（可选）

## 抓取命令决策（关键）

在 `vision_mode=robot_camera_detection` 下：

- 默认启用连续视觉伺服，但自动视觉只识别和显示目标，不会自行启动运动或抓取。
- 只有手动 `Pick 1-4` 或 SSVEP/任务状态机确认目标后，才创建 continuous pending；没有 pending 时显示 `continuous_idle awaiting_pick`。
- 后续视觉包只续跑已锁定 slot，连续命令发布 `theta_rate_deg_s / radius_rate_mm_s / z_rate_mm_s`，并固定 `use_auto_z=false`。
- 到 `vision_pick_confirm_z_mm=130.0` 且中心稳定后，最终发 `PICK_CYL`；最终半径为确认半径 `+ vision_eye_in_hand_pick_radius_bias_mm=40.0`。
- 关闭连续模式时才回到离散 `MOVE_CYL -> wait -> re-detect` fallback。
- `actionable=false` 或 ROS 状态不新鲜、机器人 busy、半径越界时拒绝抓取，不发 PICK。
- 如果机器人端 `release_mode_effective=sucker_frozen`，PICK 按 dry-run 语义执行，不能误判为真实吸取。

## 联调顺序（建议）

1. 重启 JetMax，电脑连接 JetMax Wi-Fi。
2. 启动电脑端 GUI，观察最多一次自动 runtime start。
3. 等 `robot_connected=True` 且 robot state fresh。
4. 确认视觉状态从 `waiting_control:*` 进入 `starting_8080:*`，再到 `camera_fps=...`。
5. 确认 GUI 显示 `servo_mode=continuous`、`confirm_z=130.0`、`radius_bias=40.0`。
6. 先在 `sucker_frozen` 下跑一次，确认 `PICK_DRY_RUN_DONE`。
7. 再解除 freeze 做真实抓取，确认 `PICK_DONE`、`carrying=true`、无 `last_error_code`。
