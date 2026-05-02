# 环境配置

这个仓库优先服务完整个人备份，其次才是尽力方便他人复用。因此，大型已跟踪产物、训练模型、运行报告、数据集和归档目录会有意识地保留在 Git 中。

## 推荐目录结构

克隆仓库后，请从仓库根目录工作：

```powershell
git clone https://github.com/pzy66/brain_code.git
cd brain_code
```

可以使用本地 `.venv`、名为 `brain-vision` 的 conda 环境，或设置 `BRAIN_PYTHON_EXE`：

```powershell
$env:BRAIN_PYTHON_EXE = (& .\tools\resolve_brain_python.cmd)
& $env:BRAIN_PYTHON_EXE -m pip install -e ".[dev,gui,ssvep]"
```

混合控制器和视觉相关工作：

```powershell
& $env:BRAIN_PYTHON_EXE -m pip install -e ".[hybrid]"
```

MI 训练和实时相关工作：

```powershell
& $env:BRAIN_PYTHON_EXE -m pip install -e ".[mi]"
```

GPU 加速是可选项，只在 CUDA/CuPy 匹配的本地机器上安装：

```powershell
& $env:BRAIN_PYTHON_EXE -m pip install -e ".[gpu]"
```

## 已有依赖文件

`pyproject.toml` 中的 optional dependencies 是一层便利用法。各子系统的详细依赖仍以旧文件为准：

- `01_MI/mi_classifier_latest/requirements*.txt`
- `02_SSVEP/environment.ssvep.yml`
- `hybrid_controller/requirements-hybrid-*.txt`
- `hybrid_controller/robot/requirements-jetmax-robot-python.txt`

JetMax ROS 和 `rospy` 应来自官方 JetMax 系统镜像，不通过 pip 安装。

## 采集数据位置

采集入口默认使用仓库内存储，这样一个 `brain_code` 文件夹就能整体备份和恢复：

- MI 采集：`01_MI/mi_classifier_latest/datasets/custom_mi`
- SSVEP 采集：`02_SSVEP/artifacts/datasets`
- 统一 MI/SSVEP 索引：`artifacts/unified_collection_index.csv`

相对路径形式的 `--output-root` 和 `--dataset-dir` 会解析到对应 MI 或 SSVEP 项目目录下。GUI 会拒绝把采集输出意外写到 `brain_code` 之外。

## 冒烟检查

在仓库根目录运行：

```powershell
tools\resolve_brain_python.cmd
$env:BRAIN_PYTHON_EXE = (& .\tools\resolve_brain_python.cmd)
& $env:BRAIN_PYTHON_EXE -m brain_workspace.environment
& $env:BRAIN_PYTHON_EXE -c "import brain_workspace.paths; import unified_collection.app"
& $env:BRAIN_PYTHON_EXE -m pytest --collect-only -q -o addopts=
```

无头 GUI 测试会通过测试引导代码设置 `QT_QPA_PLATFORM=offscreen`。真实 GUI 使用仍需要桌面会话。

## 硬件说明

- BrainFlow 板卡、串口、JetMax 机器人网络、CUDA、摄像头和 ROS 都是可选运行时集成。
- 缺少这些可选集成不应该影响 import 冒烟测试或测试收集。
- 视觉历史脚本会优先使用 `BRAIN_VISION_WEIGHTS`；未设置时回退到 `hybrid_controller/models/vision/best.pt`。
