# Artifacts

仓库策略已经改为轻量协作模式：Git 保存源码、配置、文档、少量占位文件和默认视觉模型权重；真实数据、额外模型权重、运行报告和硬件日志留在本地。

## Ignored By Default

- `datasets/**` 中除 README、`.gitkeep`、`datasets/vision/models/best.pt` 外的真实数据。
- `artifacts/**` 运行报告、训练输出和临时产物。
- `logs/**` 硬件和 GUI 日志。
- `runtime/**` 本地运行状态。
- `02_SSVEP/artifacts/**`
- `hybrid_controller/models/**`

## Local Copy Targets

- MI 数据：`datasets/MI/`
- SSVEP 数据：`datasets/SSVEP/`
- 默认视觉模型：`datasets/vision/models/best.pt`，随仓库提供。
- 视觉标定：`datasets/vision/calibration/current_profile.json`
- SSVEP profile：`datasets/profiles/SSVEP/`
- Hybrid Controller SSVEP profile：`datasets/profiles/hybrid_controller/ssvep_profiles/`
- Hybrid Controller 抓取调参：`datasets/profiles/hybrid_controller/robot_pick_tuning/`
- 旧模块内数据迁移存档：`datasets/MI/external/`、`datasets/vision/legacy_migrated/`

## Diagnostics

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\diagnose_workspace.ps1
python -m brain diagnose
```

`diagnose_workspace.ps1` 用于统计 Git 当前追踪体积；`brain diagnose` 用于检查运行环境和本地资产位置。
