# Hybrid Controller 启动与部署说明

这份文档只描述当前仍在使用的主线启动方式。

## 目录根

当前唯一主线目录：

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller`

电脑端运行、JetMax 部署、测试、文档都以这里为根。

## 电脑端启动

推荐解释器：

- `C:\Users\P1233\miniconda3\envs\brain-vision\python.exe`

推荐工作目录：

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller`

### 真机主界面

```bash
python run_real.py
```

### SSVEP 联调界面

```bash
python run_real_ssvep.py
```

## JetMax 端部署

推荐直接把整个 `hybrid_controller` 目录复制到 JetMax，例如：

- `/home/hiwonder/brain_code/hybrid_controller`

然后进入：

- `/home/hiwonder/brain_code/hybrid_controller/robot`

再执行：

```bash
python3 -m pip install -r requirements-jetmax-robot-python.txt
bash run_hybrid_controller_ros_runtime.sh
```

如果只想起 TCP 兼容入口：

```bash
bash run_jetmax_robot_runtime.sh
```

## 当前主线约定

- 真机 GUI 主 transport：ROS
- TCP：兼容/诊断 fallback
- 视觉主模式：`robot_camera_detection`
- 控制主语义：圆柱坐标
- 默认 teleop：`MOVE_CYL_AUTO`
- 默认抓取：`PICK_CYL`

当前圆柱范围：
- `theta ∈ [-120°, 120°]`
- `MOVE_CYL_AUTO radius ∈ [80, 260]`
- `MOVE_CYL radius ∈ [50, 280]`

## 不再作为主线入口的内容

以下内容保留为历史/实验参考，但主程序运行时不再依赖：

- `brain_code/01_MI/`
- `brain_code/02_SSVEP/`
- `brain_code/03_RobotArm_Control/`
- `brain_code/04_Communication_And_Integration/`
- `brain_code/05_Vision_Block_Recognition/`
- `brain_code/06_Data_Collection/`

仿真实验目录单独放在：

- `brain_code/07_Simulation_Lab/hybrid_controller_sim`
