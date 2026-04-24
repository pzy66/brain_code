# brain_code

`brain_code` 当前主线开发目录为 `hybrid_controller`。后续开发、部署、联调、文档维护都建议从这里进入。

## 快速导航

- 主程序说明：[hybrid_controller/README.md](./hybrid_controller/README.md)
- 真机部署与机械臂侧说明：[hybrid_controller/robot/README.md](./hybrid_controller/robot/README.md)
- 文档总览：[hybrid_controller/docs/README.md](./hybrid_controller/docs/README.md)
- SSVEP 子模块说明：[hybrid_controller/ssvep/README.md](./hybrid_controller/ssvep/README.md)
- 仿真实验目录：[07_Simulation_Lab/hybrid_controller_sim](./07_Simulation_Lab/hybrid_controller_sim)

## 仓库结构

- `hybrid_controller`：当前主线程序、GUI、联调入口与混合控制逻辑。
- `01_MI`：MI 采集、训练、实时推理相关历史模块。
- `02_SSVEP`：SSVEP 采集、训练、验证与工具链模块。
- `03_RobotArm_Control`：机械臂控制相关历史目录。
- `04_Communication_And_Integration`：通信与联调实验目录。
- `05_Vision_Block_Recognition`：视觉识别实验目录。
- `06_Data_Collection`：历史采集目录。
- `07_Simulation_Lab`：仿真与沙盒实验目录。

## 使用建议

- 首先阅读 [hybrid_controller/README.md](./hybrid_controller/README.md)，再根据需要进入机械臂或 SSVEP 子文档。
- 新机器优先使用仓库内 `.venv`，或按各模块 README 中约定的 `brain_code` 环境运行。
- 历史目录会继续保留用于复现与参考，但主程序运行时不再直接依赖这些目录。
