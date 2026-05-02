# START HERE

这个目录是正式代码仓库。日常开发、提交、分支、测试和推送都应该在 `brain_code` 里完成。

上一级目录是本地工作区，用来放环境文件、本地数据集、交付物和 IDE 配置。

## 常用入口

- 统一采集界面：[run_unified_collection.py](./run_unified_collection.py)
- 统一采集历史兼容入口：[unified_collection_ui.py](./unified_collection_ui.py)
- 混合控制器电脑端 GUI：[hybrid_controller/run_real.py](./hybrid_controller/run_real.py)
- 混合控制器 SSVEP GUI：[hybrid_controller/run_real_ssvep.py](./hybrid_controller/run_real_ssvep.py)
- JetMax 端运行脚本：[hybrid_controller/robot/run_hybrid_controller_ros_runtime.sh](./hybrid_controller/robot/run_hybrid_controller_ros_runtime.sh)

## 目录说明

- `brain_workspace`：共享路径、启动引导和环境诊断工具。
- `unified_collection`：统一 MI/SSVEP 采集界面的实际实现。
- `01_MI`：MI 采集、训练、实时推理和共享工具。
- `02_SSVEP`：SSVEP 采集、训练、回放、验证和产物。
- `hybrid_controller`：集成 MI、SSVEP、视觉和机械臂控制的主程序。
- `docs`：环境配置、产物策略、代码状态和路线图。

## 维护命令

```powershell
cd <repo>
tools\resolve_brain_python.cmd
$env:BRAIN_PYTHON_EXE = (& .\tools\resolve_brain_python.cmd)
& $env:BRAIN_PYTHON_EXE -m brain_workspace.environment
& $env:BRAIN_PYTHON_EXE -m pytest --collect-only -q -o addopts=
powershell -ExecutionPolicy Bypass -File .\tools\clean_workspace_temp.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File .\tools\diagnose_workspace.ps1
git status --short --ignored
```

清理脚本只处理缓存和临时产物，不会删除正式数据集、部署 profile、模型或运行结果。
