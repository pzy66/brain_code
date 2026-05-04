# Setup

## Python

推荐 Windows + Python 3.11。可以用项目 `.venv`，也可以用 Conda。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,gui,ssvep,mi,hybrid]"
```

已有解释器时：

```powershell
$env:BRAIN_PYTHON_EXE = "C:\path\to\python.exe"
```

## Verify

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py -m brain diagnose
& $py -m brain launch --simulate
& $py -m pytest --collect-only -q -o addopts=
```

`diagnose` 会列出 Python 版本、缺失依赖、本地数据根目录和可选模型位置。缺少硬件、摄像头、ROS 或 BrainFlow 板卡时，不应影响无硬件启动检查；默认视觉模型权重随仓库提供。

## Local Assets

本仓库不跟踪真实数据和额外模型。默认视觉模型权重会跟踪在 `datasets/vision/models/best.pt`。默认结构：

```text
datasets/
  MI/
  SSVEP/
  vision/
    models/
    calibration/
  profiles/
    SSVEP/
    hybrid_controller/ssvep_profiles/
```

外部数据盘：

```powershell
$env:BRAIN_DATA_ROOT = "D:\brain_data"
```

设置后，代码会使用：

- `D:\brain_data\MI`
- `D:\brain_data\SSVEP`
- `D:\brain_data\vision`
- `D:\brain_data\profiles`

## Hardware Notes

- BrainFlow、串口、JetMax、ROS、摄像头和 CUDA 都是可选运行时集成。
- 无硬件环境应能完成 `brain diagnose`、`brain launch --simulate` 和测试收集。
- JetMax ROS 端的 `rospy` 来自 JetMax 系统镜像，不通过 pip 安装。
