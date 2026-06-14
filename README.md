# brain_code

Current main-program policy: `python -m brain` starts the integrated robot workbench, backed by `hybrid_controller/run_real.py`, in `operator_keyboard` mode by default. MI and SSVEP realtime recognition are not started in this main path; the UI keeps the brain-computer workflow visible and injects keyboard events instead. Legacy MI/SSVEP collection remains available with `python -m brain launch --target unified`.

一体化机械臂界面启动入口：

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py -m brain
# or
& $py .\run_integrated_workbench.py
```

构建成 Windows 桌面软件：

```powershell
.\tools\build_integrated_workbench.ps1
```

生成路径：`dist\BrainRobotWorkbench\BrainRobotWorkbench.exe`。

这是团队协作使用的正式 Git 仓库。上一级 `brain` 目录只作为本地工作区，用来放个人环境文件、未入库数据、备份和交付物。

## Windows 快速开始

```powershell
git clone https://github.com/pzy66/brain_code.git
cd brain_code

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,gui,ssvep,mi,hybrid]"

.\.venv\Scripts\python.exe -m brain diagnose
.\.venv\Scripts\python.exe -m brain launch --simulate
.\.venv\Scripts\python.exe -m brain
```

如果已经有 Conda 环境，也可以设置：

```powershell
$env:BRAIN_PYTHON_EXE = "C:\path\to\python.exe"
```

## 主入口

- `python -m brain`：启动一体化机械臂控制工作台，默认键盘替代脑电识别。
- `brain`：安装后等价入口。
- `brain diagnose`：检查 Python、依赖、仓库路径和本地资产位置。
- `brain launch --simulate`：无硬件、无 GUI 启动的快速检查。
- `brain launch --target unified`：旧 MI/SSVEP 统一采集界面。
- `brain-integrated-workbench`：安装后的一体化机械臂软件入口。
- `run_unified_collection.py`、`02_SSVEP/START_SSVEP.py`、`hybrid_controller/run_real.py`：保留为兼容入口。

## 本地数据和模型

GitHub 仓库保存源码、配置、小型占位文件、文档和默认视觉模型权重。真实数据集、额外模型权重、运行报告和硬件日志由团队成员本地拷贝，不进入 Git。

默认本地数据根目录：

```text
datasets/
  MI/
  SSVEP/
  vision/
  profiles/
```

可以用 `BRAIN_DATA_ROOT` 指向外部磁盘：

```powershell
$env:BRAIN_DATA_ROOT = "D:\brain_data"
python -m brain diagnose
```

示例位置：

- MI 数据：`datasets/MI/`
- MI 外部参考数据迁移存档：`datasets/MI/external/bci_iv_2a/`
- SSVEP 数据：`datasets/SSVEP/`
- 旧外层 SSVEP 数据迁移存档：`datasets/SSVEP/legacy_migrated/`
- 旧外层视觉数据迁移存档：`datasets/vision/legacy_camara/`
- 旧视觉训练集迁移存档：`datasets/vision/legacy_migrated/`
- 默认视觉模型：`datasets/vision/models/best.pt`，随仓库提供，可用 `BRAIN_VISION_WEIGHTS` 指向其它本地权重。
- SSVEP profile：`datasets/profiles/SSVEP/current_fbcca_profile.json`
- Hybrid Controller 使用的 SSVEP profile：`datasets/profiles/hybrid_controller/ssvep_profiles/current_fbcca_profile.json`

## 常用检查

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py -m brain diagnose
& $py -m pytest --collect-only -q -o addopts=
& $py -m pytest tests -q -o addopts=
& $py -m pytest 02_SSVEP\tests hybrid_controller\tests -q -o addopts=
powershell -ExecutionPolicy Bypass -File .\tools\diagnose_workspace.ps1
```

## 仓库边界

- 不从上一级 `brain\dataset` 或个人绝对路径读取默认数据。
- 不从 `_archive`、历史运行目录或另一个仓库直接 import 活跃代码。
- `artifacts/`、`logs/`、`runtime/`、`datasets/` 中除默认视觉模型和占位文件外的真实内容默认被 `.gitignore` 忽略。
- 需要共享的其它大文件请通过本地拷贝或团队约定的外部存储分发，不提交到 Git。
