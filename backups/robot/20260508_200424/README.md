# robot（JetMax 机械臂端）

这个目录是机械臂端运行入口。当前主线是 ROS，不再使用 9092。

## 当前端口约定

- `9091`：`rosbridge`，桌面端 GUI 通过它调用 ROS 服务和订阅状态
- `8080`：`web_video_server`，桌面端相机画面来源
- `8888`：TCP 兼容链路（可选，仅诊断/回退）

说明：
- 现在没有业务代码依赖 `9092`。
- `8888` 不影响 ROS 主链启动；只有你显式要求时才会检查 TCP 端口。

## 摄像头链路契约

JetMax 摄像头必须保持 Hiwonder 官方链路：

```text
usb_cam.service -> usb_cam_node -> /usb_cam/image_rect_color -> web_video_server:8080 -> PC
```

桌面端只读取：

```text
http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80
```

维护规则：

- 默认启动流程不得改写 `/home/hiwonder/ros/autostart/usb_cam.launch`。
- 默认启动流程不得停止/重启 `usb_cam.service`。
- 默认启动流程不得 `pkill web_video_server`，也不得由 hybrid runtime 接管 `web_video_server`。
- 电脑端不得探测非官方摄像头路径，也不得尝试打开 JetMax 的 `/dev/video*`。
- 只有明确诊断摄像头发送故障时，才使用 `--repair-camera-sender` 或 `--camera-only`。
- 只有明确诊断 UVC 驱动故障时，才使用 `--repair-camera-driver`。

## 目录说明

- `run_hybrid_controller_ros_runtime.sh`
  - 机械臂端主启动脚本（推荐）
- `run_jetmax_robot_runtime.sh`
  - 仅 TCP 兼容入口（legacy）
- `runtime/`
  - 机械臂执行核心（状态机、安全逻辑、TCP 兼容）
- `ros_pkg/hybrid_controller_ros/`
  - ROS 消息、服务、runtime node
- `tools/`
  - 桌面端远程部署/探针脚本（通过 SSH 调机械臂）
- `requirements-jetmax-robot-python.txt`
  - 机械臂端 Python 依赖

## 机械臂端最小必需代码

真机 ROS 主链只需要下面这些：

1. `robot/run_hybrid_controller_ros_runtime.sh`
2. `robot/runtime/`
3. `robot/ros_pkg/hybrid_controller_ros/`
4. `robot/requirements-jetmax-robot-python.txt`

`tools/` 是桌面端辅助工具，不需要在机械臂上手工运行。

## 机械臂端启动（手工）

在 JetMax 上执行：

```bash
cd ~/brain_code/hybrid_controller/robot
python3 -m pip install -r requirements-jetmax-robot-python.txt
bash run_hybrid_controller_ros_runtime.sh
```

脚本会做这几件事：

1. 拷贝并编译 `hybrid_controller_ros` 到 `~/catkin_ws`
2. 启动 `rosbridge`（默认 `9091`）
3. 复用 JetMax 官方 `usb_cam.service` 的 `web_video_server`（默认 `8080`）
4. 启动 `hybrid_controller_runtime_node.py`

## 桌面端一键远程启动（推荐）

在 Windows（repo `.venv` 或你自己的 `brain_code` 环境）执行：

```powershell
.\.venv\Scripts\python.exe .\hybrid_controller\robot\tools\jetmax_start_ros_runtime.py --host 192.168.149.1 --user hiwonder --password $env:JETMAX_PASSWORD --remote-root /home/hiwonder/brain_code
```

这个工具会：

1. 保持已有 `rosbridge` 连接；只有显式加 `--disable-autostart-rosbridge` 才会停止系统 `rosbridge.service`
2. 保持 JetMax 官方 `usb_cam.service` 摄像头发送链路，不默认重写或重启摄像头
3. 杀掉残留 hybrid runtime 进程，不默认杀掉 `rosbridge_websocket`
4. 启动当前仓库版本的 ROS runtime
5. 只等待 `9091` rosbridge 就绪；默认不连接 `8080`，不订阅 `/usb_cam/image_rect_color`

补充：

- `jetmax_start_ros_runtime.py` 默认不验证摄像头发送，不拉取任何视频帧。需要单独验证官方流时显式加 `--check-camera-stream`。
- 只有显式加 `--repair-camera-sender --allow-camera-sender-mutation` 或 `--camera-only --allow-camera-sender-mutation` 时，工具才会改写 `/home/hiwonder/ros/autostart/usb_cam.launch` 并重启 `usb_cam.service`。
- 只有显式加 `--repair-camera-driver --allow-camera-sender-mutation` 时，工具才会停止 `usb_cam.service`、重载 `uvcvideo`，并恢复官方服务。
- `run_hybrid_controller_ros_runtime.sh` 默认不再接管 `web_video_server`，避免和官方 `usb_cam.service` 抢占摄像头。
- `--manage-web-video` / `--restart-web-video` 是废弃参数，当前会拒绝执行；上位机和 hybrid runtime 不接管视频发送。

## 常用健康检查

```powershell
.\.venv\Scripts\python.exe .\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action status
```

如果要额外验证 TCP 兼容端口：

```powershell
.\.venv\Scripts\python.exe .\hybrid_controller\robot\tools\jetmax_start_ros_runtime.py --host 192.168.149.1 --user hiwonder --password $env:JETMAX_PASSWORD --require-tcp-check
```

## legacy（保留但非主线）

- `run_jetmax_robot_runtime.sh`
- `tools/jetmax_start_runtime.py`
- `tools/deploy_jetmax_runtime.py`

这些用于老的 TCP-only 路径。主流程调试不要优先使用它们。
