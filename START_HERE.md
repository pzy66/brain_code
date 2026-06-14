# Start Here

Main-program default: use `python -m brain` or `run_integrated_workbench.py` for the integrated robot workbench. It is backed by `hybrid_controller/run_real.py` and runs in keyboard/operator mode (`operator_keyboard`): robot, ROS, camera vision, MOVE/PICK/PLACE, tuning, logs, and safety gates remain active, while MI/SSVEP realtime recognition is not started. Use `01_MI` and `02_SSVEP` as standalone modules for algorithm testing.

`brain_code` 是正式代码仓库。请在这个目录里开发、测试、提交和推送。

## 第一次运行

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
$py = & .\tools\resolve_brain_python.cmd
& $py -m pip install -e ".[dev,gui,ssvep,mi,hybrid]"
& $py -m brain diagnose
& $py -m brain launch --simulate
```

需要打开一体化机械臂界面时运行：

```powershell
& $py -m brain
```

也可以双击或运行：

```powershell
.\START_INTEGRATED_WORKBENCH.cmd
```

需要生成可双击分发的软件目录时运行：

```powershell
.\tools\build_integrated_workbench.ps1
```

生成文件在 `dist\BrainRobotWorkbench\BrainRobotWorkbench.exe`。

## 常用入口

- 一体化机械臂控制工作台：`python -m brain` 或 `run_integrated_workbench.py`
- Windows 软件构建：`tools\build_integrated_workbench.ps1`
- 环境诊断：`python -m brain diagnose`
- MI/SSVEP 旧统一采集：`python -m brain launch --target unified` 或 `run_unified_collection.py`
- SSVEP 工具启动器：`02_SSVEP/START_SSVEP.py`
- Hybrid Controller：`hybrid_controller/run_real.py`
- Hybrid Controller SSVEP 模式：`hybrid_controller/run_real_ssvep.py`

## 数据位置

默认使用仓库内 `datasets/`：

- `datasets/MI/`
- `datasets/SSVEP/`
- `datasets/vision/`
- `datasets/profiles/`

如果数据放在外部磁盘，设置：

```powershell
$env:BRAIN_DATA_ROOT = "D:\brain_data"
```

真实数据、额外模型权重、运行报告和日志不会进入 Git；默认视觉模型权重位于 `datasets/vision/models/best.pt` 并随仓库提供。
