# robot（JetMax 机械臂端）

这个目录是机械臂端运行入口。当前主线是 ROS，不再使用 9092。

## 当前端口约定

- `9091`：`rosbridge`，桌面端 GUI 通过它调用 ROS 服务和订阅状态
- `8080`：`web_video_server`，桌面端相机画面来源
- `8888`：TCP 兼容链路（可选，仅诊断/回退）

说明：
- 现在没有业务代码依赖 `9092`。
- `8888` 不影响 ROS 主链启动；只有你显式要求时才会检查 TCP 端口。

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
3. 启动 `web_video_server`（默认 `8080`）
4. 启动 `hybrid_controller_runtime_node.py`

## 桌面端一键远程启动（推荐）

在 Windows（`brain-vision` 解释器）执行：

```powershell
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\jetmax_start_ros_runtime.py --host 192.168.149.1 --user hiwonder --password hiwonder --remote-root /home/hiwonder/brain_code
```

这个工具会：

1. 关闭 JetMax 上系统自启的旧 `rosbridge.service`（避免消息 md5 冲突）
2. 杀掉残留 `rosbridge/web_video/runtime` 进程
3. 启动当前仓库版本的 ROS runtime
4. 等待端口就绪（默认仅检查 `9091` + `8080`）

补充：

- `run_hybrid_controller_ros_runtime.sh` 现在默认会强制重启 `rosbridge_websocket` 和 `web_video_server`，避免“旧进程占用端口但消息定义不一致”导致的连接异常。

## 常用健康检查

```powershell
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action status
```

如果要额外验证 TCP 兼容端口：

```powershell
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\jetmax_start_ros_runtime.py --host 192.168.149.1 --user hiwonder --password hiwonder --require-tcp-check
```

## legacy（保留但非主线）

- `run_jetmax_robot_runtime.sh`
- `tools/jetmax_start_runtime.py`
- `tools/deploy_jetmax_runtime.py`

这些用于老的 TCP-only 路径。主流程调试不要优先使用它们。
