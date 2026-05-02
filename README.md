# brain_code

`brain_code` 是这个工作区的正式 Git 仓库。日常开发、提交、分支、测试和推送都应该在这个目录里完成；上一级 `brain` 目录只作为本地工作区，用来放环境文件、本地数据集、PyCharm 配置、交付物和备份。

## 主要入口

- 新手入口：[START_HERE.md](./START_HERE.md)
- MI/SSVEP 统一采集界面：[run_unified_collection.py](./run_unified_collection.py)
- 统一采集实现包：[unified_collection](./unified_collection)
- 工作区路径工具：[brain_workspace](./brain_workspace)
- 混合控制器：[hybrid_controller/README.md](./hybrid_controller/README.md)
- SSVEP 工具链：[02_SSVEP/README.md](./02_SSVEP/README.md)
- MI 采集与训练：[01_MI/README.md](./01_MI/README.md)
- 环境配置：[docs/SETUP.md](./docs/SETUP.md)
- 代码状态：[docs/CODE_STATUS.md](./docs/CODE_STATUS.md)
- 产物说明：[docs/ARTIFACTS.md](./docs/ARTIFACTS.md)

## 仓库边界

- Git 命令请在当前 `brain_code` 目录里运行，不要在上一级本地工作区运行。
- 真实数据集、部署用 profile、正式运行结果、模型和归档实验输出会保留在仓库内，方便完整备份与复现。
- 新的 MI 和 SSVEP 采集数据默认保存在仓库内：
  `01_MI/mi_classifier_latest/datasets/custom_mi` 和
  `02_SSVEP/artifacts/datasets`。
- 生成缓存、临时 pytest 目录、冒烟测试截图和 GPU 编译缓存不应该进入 Git。

## Python 项目基线

仓库根目录现在提供 `pyproject.toml`，用于 pytest 发现和轻量内部包安装：

- `brain_workspace`：统一路径、运行时 import 引导、环境诊断。
- `unified_collection`：统一 MI/SSVEP 采集界面的实际实现。

历史入口仍然保留：

```powershell
tools\resolve_brain_python.cmd
$env:BRAIN_PYTHON_EXE = (& .\tools\resolve_brain_python.cmd)
& $env:BRAIN_PYTHON_EXE run_unified_collection.py
```

## 清理与诊断

清理脚本只处理可重新生成的文件：

```powershell
powershell -ExecutionPolicy Bypass -File tools\clean_workspace_temp.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File tools\clean_workspace_temp.ps1
```

做较大的仓库维护前，建议先运行诊断脚本：

```powershell
powershell -ExecutionPolicy Bypass -File tools\diagnose_workspace.ps1
```

## 常用检查

在仓库根目录运行：

```powershell
$env:BRAIN_PYTHON_EXE = (& .\tools\resolve_brain_python.cmd)
& $env:BRAIN_PYTHON_EXE -m brain_workspace.environment
& $env:BRAIN_PYTHON_EXE -m pytest --collect-only -q -o addopts=
& $env:BRAIN_PYTHON_EXE -m py_compile unified_collection_ui.py run_unified_collection.py
& $env:BRAIN_PYTHON_EXE -m pytest tests -q -o addopts=
& $env:BRAIN_PYTHON_EXE -m pytest 02_SSVEP\tests\test_server_train_client_gpu_and_paths.py 02_SSVEP\tests\test_server_train_client_cuda_policy.py -q -o addopts=
```
