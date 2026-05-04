# Artifacts

仓库策略已经改为轻量协作模式：Git 保存源码、配置、文档和少量占位文件；真实数据、模型权重、运行报告和硬件日志留在本地。

## Ignored By Default

- `datasets/**` 中除 README 和 `.gitkeep` 外的真实数据。
- `artifacts/**` 运行报告、训练输出和临时产物。
- `logs/**` 硬件和 GUI 日志。
- `runtime/**` 本地运行状态。
- `02_SSVEP/artifacts/**`
- `05_Vision_Block_Recognition/dataset/**`
- `hybrid_controller/models/**`
- `hybrid_controller/dataset/**`

## Local Copy Targets

- MI 数据：`datasets/MI/`
- SSVEP 数据：`datasets/SSVEP/`
- 视觉模型：`datasets/vision/models/best.pt`
- 视觉标定：`datasets/vision/calibration/current_profile.json`
- SSVEP profile：`datasets/profiles/SSVEP/`
- Hybrid Controller SSVEP profile：`datasets/profiles/hybrid_controller/ssvep_profiles/`

## Diagnostics

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\diagnose_workspace.ps1
python -m brain diagnose
```

`diagnose_workspace.ps1` 用于统计 Git 当前追踪体积；`brain diagnose` 用于检查运行环境和本地资产位置。
