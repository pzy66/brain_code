# JetMax 真机联动说明

## 目标

这份文档说明当前真机主链如何联动：

- JetMax 端运行 ROS runtime
- 电脑端运行主界面
- 二者通过 `rosbridge` 和 `hybrid_controller_ros` 通信

## 机械臂端要放什么

推荐直接复制整个主线目录：

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller`

到 JetMax，例如：

- `/home/hiwonder/brain_code/hybrid_controller`

这样就不需要再单独维护旧的 `jetmax_robot` 或 `hybrid_controller_ros` 副本。

## JetMax 端启动

进入：

```bash
cd ~/brain_code/hybrid_controller/robot
python3 -m pip install -r requirements-jetmax-robot-python.txt
bash run_hybrid_controller_ros_runtime.sh
```

这个脚本会：
1. 加载 ROS 环境
2. 把内部 ROS package 复制到 `~/catkin_ws/src/`
3. 编译 `hybrid_controller_ros`
4. 确保 `rosbridge` 使用主线端口
5. 启动 `hybrid_controller_runtime_node.py`

## 电脑端启动

推荐解释器：

- `C:\Users\P1233\miniconda3\envs\brain-vision\python.exe`

进入：

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller`

运行：

```bash
python run_real.py
```

如果要启用 SSVEP 联调：

```bash
python run_real_ssvep.py
```

## 当前主链端口

- ROS bridge：`9091`
- TCP fallback：`8888`

## 当前控制主线

- teleop：ROS 本地圆柱控制 loop
- 兼容诊断：TCP 文本协议
- 默认视觉：`robot_camera_detection`
- 默认抓取：`PICK_CYL`

## 实机联调建议顺序

1. 先启动 JetMax 端 `run_hybrid_controller_ros_runtime.sh`
2. 再启动电脑端 `run_real.py`
3. 确认 GUI 中：
   - `robot_connected=True`
   - `preflight ok=True`
4. 先测移动：
   - `a/d` 旋转
   - `w/s` 前后
5. 再测视觉目标和 `PICK_CYL`
6. 最后测 `PLACE`
