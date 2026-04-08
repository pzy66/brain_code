# Hybrid Controller 启动与部署说明

如果你现在的重点是：

- JetMax 机械臂端先运行什么
- 代码该放哪里
- 电脑端再启动什么

优先看：

- `JetMax_真机联动说明.md`

这份文档负责讲完整主线，包括：

- 电脑端环境
- PyCharm 启动方式
- JetMax 端部署
- 真机和仿真的入口
- 协议和测试入口

## 1. 当前唯一主线

统一总控现在只看这里：

- 本地仓库根：`C:\Users\P1233\Desktop\brain\brain_code`
- 统一解释器：`C:\Users\P1233\miniconda3\envs\brain-vision\python.exe`
- 主程序：`brain_code/hybrid_controller/`

## 2. 电脑端环境

### 2.1 统一解释器

当前统一解释器固定为：

- `C:\Users\P1233\miniconda3\envs\brain-vision\python.exe`

### 2.2 安装依赖

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
.\scripts\setup_desktop_env.ps1
```

或者双击：

- `scripts/SETUP_HYBRID_ENV.cmd`

这个脚本会在 `brain-vision` 里安装：

- `requirements-hybrid-controller.txt`
- `requirements-hybrid-runtime-optional.txt`
- `requests`
- `paramiko`

## 3. 电脑端怎么启动

### 3.1 PyCharm 直接运行

根项目已经预置了这几个运行配置：

- `Hybrid_Controller_Real_GUI`
- `Hybrid_Controller_Sim_GUI`
- `Hybrid_Controller_Fake_Robot_Server`

如果直接打开的是 `brain_code` 子目录，这 3 个配置也已经同步存在。

如果你不想切运行配置，也可以直接右键运行：

- `run_hybrid_real.py`
- `run_hybrid_sim_gui.py`
- `run_fake_robot_server.py`

### 3.2 真机 GUI

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
.\scripts\run_desktop_fixed_world_real.ps1 -RobotHost 192.168.149.1 -VisionMode fixed_cyl_slots
```

等价双击入口：

- `hybrid_controller/START_HYBRID_REAL.cmd`

### 3.3 仿真 fake robot + GUI

先启动 fake robot：

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
.\scripts\run_fake_robot_server_fixed_world.ps1 -VisionMode fixed_cyl_slots
```

再启动仿真 GUI：

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
.\scripts\run_desktop_fixed_world_sim.ps1 -VisionMode fixed_cyl_slots
```

等价双击入口：

- `hybrid_controller/START_HYBRID_SIM.cmd`
- `hybrid_controller/START_HYBRID_SIM_GUI.cmd`

## 4. JetMax 端需要放哪些文件

现在 JetMax 端最小运行集已经单独整理到：

- `brain_code/jetmax_robot/`

如果你只想把真机跑起来，最简单的做法是直接复制整个：

- `brain_code/jetmax_robot`

到 JetMax：

- `~/brain_code/jetmax_robot`

JetMax 实际运行的是：

- `jetmax_robot/robot_runtime_py36.py`

## 5. JetMax 端启动

```bash
cd ~/brain_code/jetmax_robot
python3 -m pip install -r requirements-jetmax-robot-python.txt
bash run_jetmax_robot_runtime.sh
```

这个脚本会优先尝试：

- `/opt/ros/melodic/setup.bash`
- `~/ros/devel/setup.bash`
- `~/catkin_ws/devel/setup.bash`

然后启动：

- `jetmax_robot/robot_runtime_py36.py`

## 6. 从电脑端部署 JetMax 运行时

如果你已经能通过 Wi‑Fi 连到 JetMax，可以直接从本机推送最新 py36 运行时：

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
& C:\Users\P1233\miniconda3\envs\brain-vision\python.exe .\scripts\deploy_jetmax_runtime.py
```

默认参数：

- host: `192.168.149.1`
- user: `hiwonder`
- password: `hiwonder`
- remote root: `/home/hiwonder/brain_code`

可覆盖：

```powershell
& C:\Users\P1233\miniconda3\envs\brain-vision\python.exe .\scripts\deploy_jetmax_runtime.py --host 192.168.149.1 --user hiwonder --password hiwonder --remote-root /home/hiwonder/brain_code
```

## 7. 当前协议

下行命令：

- `PING`
- `STATUS`
- `MOVE x y`
- `MOVE_CYL theta r z`
- `MOVE_CYL_AUTO theta r`
- `PICK u v`
- `PICK_WORLD x y`
- `PICK_CYL theta r`
- `PLACE`
- `ABORT`
- `RESET`

上行命令：

- `ACK PONG`
- `ACK STATUS <json>`
- `ACK MOVE`
- `ACK PICK_STARTED`
- `ACK PICK_DONE`
- `ACK PLACE_STARTED`
- `ACK PLACE_DONE`
- `ACK ABORT`
- `ACK RESET`
- `BUSY`
- `ERR <code>: <message>`

## 8. 当前键位

- `n`
  开始任务
- `r`
  重置任务
- 长按 `a / d`
  左转 / 右转
- 长按 `w / s`
  收回 / 前伸
- `1 / 2 / 3 / 4`
  选择固定槽位
- `Enter` 或 `c`
  确认
- `Esc` 或 `x`
  取消
- `Abort`
  立即中止动作
- `Reset`
  从错误态恢复

## 9. 当前推荐实机顺序

1. JetMax 端先启动 `robot_runtime_py36.py`
2. 电脑端启动真机 GUI
3. 先做 `MOVE-only`
4. 再做 `ABORT / RESET`
5. 再做圆柱移动 `MOVE_CYL / MOVE_CYL_AUTO`
6. 最后再做 `PICK_CYL` 或 `PICK_WORLD`

## 10. 测试命令

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
& C:\Users\P1233\miniconda3\envs\brain-vision\python.exe -m pytest tests -q
```

如果只测 hybrid controller 当前主线：

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
& C:\Users\P1233\miniconda3\envs\brain-vision\python.exe -m pytest tests\test_cylindrical_control.py tests\test_robot_runtime.py tests\test_simulation_world.py tests\test_state_machine.py tests\test_robot_protocol.py tests\test_event_routing.py -q
```
