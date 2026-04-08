# JetMax 机械臂端最小运行集

这个目录是专门给 JetMax 真机部署准备的最小运行集。

如果你只想知道：

- 机械臂上到底放什么代码
- 机械臂上先运行什么
- 电脑端再运行什么
- 两边怎么联动

就先看这份。

## 1. 这个目录里有什么

- `robot_runtime_py36.py`
  JetMax 机械臂端实际运行的 TCP 执行服务。
- `run_jetmax_robot_runtime.sh`
  JetMax 端启动脚本。会先 `source` ROS 环境，再启动 `robot_runtime_py36.py`。
- `requirements-jetmax-robot-python.txt`
  JetMax 端额外 Python 依赖。

说明：

- JetMax 官方镜像里的 `rospy`、`hiwonder` 不在这个 requirements 里安装
- 这套目录只负责机械臂端执行服务
- 统一总控、GUI、测试仍然在 `brain_code/` 主线里维护

## 2. 正式源码在哪里

正式维护的运行时代码仍然是：

- `../hybrid_controller/integrations/robot_runtime_py36.py`

如果你修改了这份正式源码，先在电脑端运行：

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
& C:\Users\P1233\miniconda3\envs\brain-vision\python.exe .\scripts\sync_jetmax_robot_bundle.py
```

再把这个 `jetmax_robot/` 目录重新复制到 JetMax。

## 3. 机械臂端放哪里

推荐直接复制到：

- `/home/hiwonder/brain_code/jetmax_robot`

## 4. 机械臂端怎么运行

在 JetMax 终端执行：

```bash
cd ~/brain_code/jetmax_robot
python3 -m pip install -r requirements-jetmax-robot-python.txt
bash run_jetmax_robot_runtime.sh
```

启动成功后应看到：

```text
JetMax runtime listening on 0.0.0.0:8888
```

## 5. 当前支持的控制协议

兼容保留：

- `MOVE x y`
- `PICK u v`
- `PICK_WORLD x y`
- `PLACE`
- `STATUS`
- `ABORT`
- `RESET`

推荐主路径：

- `MOVE_CYL theta r z`
- `MOVE_CYL_AUTO theta r`
- `PICK_CYL theta r`

当前执行端是双线路：

- `legacy_cartesian`
- `cylindrical_kernel`

其中：

- legacy 指令继续可用
- 新功能和主路径优先走圆柱线路

## 6. 电脑端怎么配合

电脑端应从 `brain_code/` 根目录启动：

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
& C:\Users\P1233\miniconda3\envs\brain-vision\python.exe .\run_hybrid_real.py
```

或者在 PyCharm 运行：

- `Hybrid_Controller_Real_GUI`

默认连接：

- `192.168.149.1:8888`

## 7. 正确顺序

1. 先在 JetMax 上启动这个目录里的执行服务
2. 再在电脑端启动 GUI
3. GUI 连上后先看 `STATUS`
4. 再测 `MOVE-only`
5. 再测 `MOVE_CYL / MOVE_CYL_AUTO`
6. 最后才测 `PICK_WORLD / PICK_CYL / PLACE`
