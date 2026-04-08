# Start Here

如果你现在要继续开发或联调这个项目，按这个顺序看就够了。

## 1. 先确认唯一主线

统一总控现在只看这里：

- `brain_code/hybrid_controller/`

不要再把外层根目录里旧的 `hybrid_controller/` 当成主线。

## 2. 先确认解释器

统一解释器：

- `C:\Users\P1233\miniconda3\envs\brain-vision\python.exe`

推荐工作目录：

- `C:\Users\P1233\Desktop\brain\brain_code`

## 3. 第一次使用先装环境

双击：

- `scripts/SETUP_HYBRID_ENV.cmd`

或者：

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
.\scripts\setup_desktop_env.ps1
```

## 4. 想直接启动当前主线

### 在 PyCharm 里最简单

直接选运行配置：

- `Hybrid_Controller_Real_GUI`
- `Hybrid_Controller_Sim_GUI`
- `Hybrid_Controller_Fake_Robot_Server`

或者更直接一点，右键运行：

- `run_hybrid_real.py`
- `run_hybrid_sim_gui.py`
- `run_fake_robot_server.py`

如果你在做本地联调，先跑：

1. `Hybrid_Controller_Fake_Robot_Server`
2. `Hybrid_Controller_Sim_GUI`

### 真机

- `hybrid_controller/START_HYBRID_REAL.cmd`

默认会连接：

- `192.168.149.1:8888`

### 仿真

先开 fake robot：

- `hybrid_controller/START_HYBRID_SIM.cmd`

再开 GUI：

- `hybrid_controller/START_HYBRID_SIM_GUI.cmd`

## 5. 想看部署和使用细节

看这份：

- `docs/JetMax_真机联动说明.md`
- `docs/Hybrid_Controller_启动与部署说明.md`

## 5.1 想单独运行 05 视觉识别

如果你现在只是想直接打开小木块识别和 SSVEP 闪烁主程序，不想走总控：

- 双击 `05_Vision_Block_Recognition/2026-03_yolo_camera_detection/START_BLOCK_SSVEP_VIEWER.cmd`
- 或看 `05_Vision_Block_Recognition/2026-03_yolo_camera_detection/README.md`
- 在 PyCharm 里也可以直接运行 `Vision_Block_SSVEP_Robot_Camera`

## 5.2 想单独运行 02 的 async FBCCA 校验界面

- 双击 `02_SSVEP/2026-04_async_fbcca_idle_decoder/START_ASYNC_SSVEP_VALIDATION_UI.cmd`

## 5.3 想单独运行 06 数据采集器

- 双击 `06_Data_Collection/2026-04_jetmax_block_dataset_collection/START_BLOCK_DATASET_COLLECTOR.cmd`
- 或在 PyCharm 里运行 `Block_Dataset_Collector`

## 6. 想看当前代码结构

看这几个位置：

- `hybrid_controller/app.py`
- `hybrid_controller/controller/task_controller.py`
- `hybrid_controller/integrations/robot_runtime.py`
- `hybrid_controller/integrations/robot_runtime_py36.py`
- `jetmax_robot/`
- `tests/`

## 7. 想跑测试

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
& C:\Users\P1233\miniconda3\envs\brain-vision\python.exe -m pytest tests -q
```
