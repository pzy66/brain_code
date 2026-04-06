# Brain Code Hub

`brain_code` 是当前脑控机械臂项目的统一代码目录。这里不再按旧项目名组织，而是按功能模块归档，便于后续继续开发、联调和上传 GitHub。

## 第一入口

如果你是第一次重新接手这个仓库，建议按下面顺序看：

1. `START_HERE.md`
2. `docs/主线入口与代码地图.md`
3. `docs/系统架构与主线开发图.md`
4. `docs/系统现状与开发建议报告_2026-04-06.md`

## 目录结构

- `01_MI/`
  MI 数据采集、训练、实时推理主线代码
- `02_SSVEP/`
  SSVEP 刺激、实时识别、算法脚本、数据处理脚本
- `03_RobotArm_Control/`
  机械臂执行端控制代码
- `04_Communication_And_Integration/`
  socket 通信、联调脚本、调试工具
- `05_Vision_Block_Recognition/`
  木块识别、相机处理、YOLO、模板匹配相关代码
- `docs/`
  项目报告、系统分析、后续开发说明

## 当前推荐入口

- MI 主线：
  `01_MI/mi_classifier_latest/`
- SSVEP 最新主线：
  `02_SSVEP/2026-03_realtime_ui_and_online_decoder/`
- 机械臂控制主线：
  `03_RobotArm_Control/2026-03_jetmax_execution_server/`
- 视觉与总控原型主线：
  `05_Vision_Block_Recognition/2026-03_yolo_camera_detection/`

## 仓库说明

本仓库以“代码和文档”为主，不把本地运行环境、缓存、临时测试产物和大体量数据一起上传。

默认忽略的本地内容包括：

- `.idea/`
- `.venv/`
- `.pytest_cache/`
- `01_MI/mi_classifier_latest/datasets/`
- `01_MI/mi_classifier_latest/runtime/`

## 版本索引

详细的“哪个目录是当前主线、哪些目录是历史补充版本”见：

- `VERSION_MAP.md`

## 报告文档

主线代码地图见：

- `docs/主线入口与代码地图.md`

系统架构与状态机建议见：

- `docs/系统架构与主线开发图.md`

系统现状和后续开发建议见：

- `docs/系统现状与开发建议报告_2026-04-06.md`

英文版分析报告见：

- `docs/SYSTEM_STATUS_REPORT_2026-04-06.md`
