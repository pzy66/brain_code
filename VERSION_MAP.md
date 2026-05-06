# Version Map

本文件用于标注 `brain_code` 中各类别当前应优先阅读和继续维护的目录。

## 1. MI

- 当前主线：
  `01_MI/mi_classifier_latest`
- 状态：
  当前 MI 主线代码
- 备注：
  这是目前最完整的一套 MI 采集、训练、实时推理代码

## 2. SSVEP

- 当前主线：
  `02_SSVEP`
- 主要结构：
  `02_SSVEP/apps`
  `02_SSVEP/entrypoints`
  `02_SSVEP/ssvep_core`
  `02_SSVEP/tools`
  `02_SSVEP/tests`
- 历史补充：
  `02_SSVEP/_archive`
- 备注：
  当前 SSVEP 主线已经收敛为 `apps / entrypoints / ssvep_core` 结构。
  新开发应围绕数据采集、实时在线解码、训练评测、local-opt、profile/profile_v2
  和 async-first 指标推进，不应继续从旧月份目录 import 活跃代码。

## 3. 机械臂控制

- 当前主线：
  `03_RobotArm_Control/2026-03_jetmax_execution_server`
- 核心文件：
  `03_RobotArm_Control/2026-03_jetmax_execution_server/test2_robot.py`

## 4. 通信与联调

- 主要目录：
  `04_Communication_And_Integration/2026-02_socket_and_pick_command`
  `04_Communication_And_Integration/2026-03_signal_monitoring_and_debug`
  `04_Communication_And_Integration/2026-05_vision_grasp_flow_debug`
- 备注：
  这里主要保留 socket 通信、抓取指令联调和实时调试脚本，不是单独的一套完整主线项目。
  识别并抓取全过程调试应放在这里，验证稳定后再复制最小必要代码进 `hybrid_controller`。

## 5. 视觉与木块识别

- 当前主线：
  `05_Vision_Block_Recognition/2026-03_yolo_camera_detection`
- 历史补充：
  `05_Vision_Block_Recognition/2026-02_template_matching_and_camera`
- 备注：
  `2026-03` 更偏 YOLO 与摄像头实时检测；
  `2026-02` 更偏模板匹配、颜色处理和相机辅助脚本。

## 6. 文档

- 数据采集辅助：
  `06_Data_Collection/2026-04_jetmax_block_dataset_collection`
- 仿真实验：
  `07_Simulation_Lab/hybrid_controller_sim`

- 系统分析报告：
  `docs/系统现状与开发建议报告_2026-04-06.md`
- 英文版系统报告：
  `docs/SYSTEM_STATUS_REPORT_2026-04-06.md`
