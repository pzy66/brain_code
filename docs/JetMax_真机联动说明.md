# JetMax 真机联动说明

这份文档只讲一件事：

- JetMax 机械臂端先运行什么代码
- 代码应该放在哪里
- 电脑端再启动什么
- 两边怎样联动

如果你只是要把真机控制链跑起来，优先看这份。

## 1. 目录分工

### JetMax 端最小运行集

JetMax 端推荐只放这一套最小目录：

- `brain_code/jetmax_robot/`

其中最关键的文件是：

- `robot_runtime_py36.py`
- `run_jetmax_robot_runtime.sh`
- `requirements-jetmax-robot-python.txt`

说明：

- 这是 JetMax 真机执行端最小运行集
- 电脑端 GUI、状态机、仿真和测试都不在这里运行
- 正式维护的源码仍以 `brain_code/hybrid_controller/integrations/robot_runtime_py36.py` 为准
- 如果你更新了正式运行时代码，先运行：

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
& C:\Users\P1233\miniconda3\envs\brain-vision\python.exe .\scripts\sync_jetmax_robot_bundle.py
```

然后再把 `jetmax_robot/` 重新复制到 JetMax。

### 电脑端主控

电脑端统一使用：

- `brain_code/hybrid_controller/`
- `brain_code/run_hybrid_real.py`

## 2. 机械臂端代码放哪里

推荐直接复制到 JetMax 的：

- `/home/hiwonder/brain_code/jetmax_robot`

后面文档默认都按这个路径写。

## 3. JetMax 端怎么启动

在 JetMax 终端执行：

```bash
cd ~/brain_code/jetmax_robot
python3 -m pip install -r requirements-jetmax-robot-python.txt
bash run_jetmax_robot_runtime.sh
```

启动成功后，终端应看到类似输出：

```text
JetMax runtime listening on 0.0.0.0:8888
```

这表示 JetMax 执行服务已经启动，并开始监听：

- `192.168.149.1:8888`

## 4. 机械臂端现在支持什么控制线路

JetMax 执行端当前是双线路：

- `legacy_cartesian`
  - `MOVE x y`
  - `PICK u v`
  - `PICK_WORLD x y`
  - `PLACE`
- `cylindrical_kernel`
  - `MOVE_CYL theta r z`
  - `MOVE_CYL_AUTO theta r`
  - `PICK_CYL theta r`

注意：

- 旧笛卡尔指令继续保留，主要用于兼容
- 推荐的新主路径是圆柱坐标线路
- 即使走圆柱线路，最底层仍然会调用 JetMax 官方 `set_position(x, y, z, t)`

## 5. 电脑端怎么启动

电脑端工作目录统一为：

- `C:\Users\P1233\Desktop\brain\brain_code`

解释器统一为：

- `C:\Users\P1233\miniconda3\envs\brain-vision\python.exe`

最简单的方式是在 PyCharm 里运行：

- `Hybrid_Controller_Real_GUI`

等价命令是：

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
& C:\Users\P1233\miniconda3\envs\brain-vision\python.exe .\run_hybrid_real.py
```

默认连接：

- `192.168.149.1:8888`

## 6. 正确联动顺序

必须按这个顺序：

1. JetMax 先连上 Wi-Fi
2. 在 JetMax 上启动 `jetmax_robot/run_jetmax_robot_runtime.sh`
3. 确认 `8888` 已监听
4. 再在电脑端启动 `run_hybrid_real.py` 或 `Hybrid_Controller_Real_GUI`
5. GUI 连上后确认：
   - `robot_connected=True`
   - `preflight ok=True`
6. 再开始做移动、急停、抓取等测试

不要反过来先开电脑端 GUI，再去启动 JetMax 服务。

## 7. 推荐真机测试顺序

先测安全和连接，再测动作：

1. `STATUS`
2. `MOVE-only`
3. `ABORT / RESET`
4. `MOVE_CYL / MOVE_CYL_AUTO`
5. `PICK_WORLD / PICK_CYL / PLACE`

## 8. STATUS 里重点看什么

真机联调时，重点看这些字段：

- `state`
- `busy`
- `robot_xy`
- `robot_cyl`
- `control_kernel`
- `limits_cyl`
- `auto_z_current`
- `calibration_ready`
- `last_error_code`
- `last_error`

推荐把 `robot_cyl` 当主状态看，`robot_xy` 只作为兼容和调试。
