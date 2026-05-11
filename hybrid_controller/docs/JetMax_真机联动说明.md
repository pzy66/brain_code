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

GUI 默认适配“机械臂重启后，电脑连接 JetMax Wi-Fi，然后启动主程序”的流程：启动后自动连接 `rosbridge`，但不会在 `9091` 不稳定时自动 SSH 启动 JetMax ROS runtime。需要启动 runtime 时使用显式按钮或显式命令；默认流程不会碰 `8080` 视频流，不会订阅 `/usb_cam/image_rect_color`，不会改官方摄像头发送链路。

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

- `command_mode=world` -> 下发 `PICK_WORLD x y`（默认视觉主路径）
- `command_mode=cyl` -> 下发 `PICK_CYL theta r`（手动/特定路径）
- `actionable=false` -> 拒绝抓取，不发命令

## 联调顺序（建议）

1. 启动 JetMax 端 runtime
2. 启动电脑端 GUI
3. 点击“连接机器人”，确认 `robot_connected=True`
4. 先测移动（`a/d/w/s`）
5. 再测视觉槽位
6. 最后测抓放（`PICK_WORLD -> PLACE`）
