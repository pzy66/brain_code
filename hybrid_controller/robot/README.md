# robot

这个目录是 JetMax 真机部署入口。  
如果你只关心“机械臂上要放什么、怎么启动”，只看这里。

## 目录说明

- [runtime](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\runtime)
  - JetMax 端运行时
  - 包含 TCP 兼容入口、圆柱控制内核、teleop kernel
- [ros_pkg](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\ros_pkg)
  - 内置 ROS package：`hybrid_controller_ros`
- [tools](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools)
  - 部署、探测、探针脚本
- [run_hybrid_controller_ros_runtime.sh](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\run_hybrid_controller_ros_runtime.sh)
  - 真机推荐主入口
- [run_jetmax_robot_runtime.sh](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\run_jetmax_robot_runtime.sh)
  - TCP 兼容入口
- [requirements-jetmax-robot-python.txt](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\requirements-jetmax-robot-python.txt)
  - JetMax 端 Python 依赖

## 推荐部署方式

推荐直接把整个：

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller`

复制到 JetMax，例如：

- `/home/hiwonder/brain_code/hybrid_controller`

这样 JetMax 端需要的一切都在同一个目录里：
- 主程序包
- 机器人运行时
- ROS package
- 视觉权重
- SSVEP profile 数据目录

## JetMax 端启动

进入 JetMax 后执行：

```bash
cd ~/brain_code/hybrid_controller/robot
python3 -m pip install -r requirements-jetmax-robot-python.txt
bash run_hybrid_controller_ros_runtime.sh
```

这个脚本会完成：
1. 加载 ROS 环境
2. 将内部 ROS 包复制到 `~/catkin_ws/src/`
3. 编译 `hybrid_controller_ros`
4. 确保 `rosbridge` 使用主线端口
5. 启动 `hybrid_controller_runtime_node.py`

默认只要求 ROS 主链端口就绪（`9091` + `8080`），不再因为 `8888` 未开启而判启动失败。  
如需强制同时检查 TCP 兼容端口，再使用 `--require-tcp-check`。

## 远程一键启动与探针

桌面端可直接通过以下工具启动/检查 JetMax：

- [jetmax_start_ros_runtime.py](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\jetmax_start_ros_runtime.py)
  - SSH 启动 JetMax ROS runtime
- [ros_service_probe.py](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\ros_service_probe.py)
  - 调用 ROS 服务探针（`status / pick_world / place / reset / abort`）

示例（桌面端执行）：

```powershell
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\jetmax_start_ros_runtime.py --host 192.168.149.1 --user hiwonder --password hiwonder --remote-root /home/hiwonder/brain_code

C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action status
```

## 兼容 TCP 入口

如果只想起旧的 TCP 兼容 runtime：

```bash
cd ~/brain_code/hybrid_controller/robot
bash run_jetmax_robot_runtime.sh
```

它会直接启动：

- [robot_runtime_py36.py](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\runtime\robot_runtime_py36.py)

## 当前真机主线

- GUI 真机主链：ROS
- TCP：兼容/诊断 fallback
- teleop 主语义：圆柱坐标
- 当前默认范围：
  - `theta ∈ [-120°, 120°]`
  - `MOVE_CYL_AUTO radius ∈ [80, 260]`
  - `MOVE_CYL radius ∈ [50, 280]`
