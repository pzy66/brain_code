# 代码状态

这个仓库会保留完整项目历史。下面的状态表用于帮助选择正确入口。

| 区域 | 状态 | 推荐入口 | 说明 |
| --- | --- | --- | --- |
| `brain_workspace` | 稳定 | `python -m brain_workspace.environment` | 共享路径和环境诊断。 |
| `unified_collection` | 可用 | `run_unified_collection.py` | 当前统一 MI/SSVEP 采集界面包。 |
| `01_MI` | 可用 | `01_MI/mi_classifier_latest/code/README.md` | MI 采集、训练和实时代码，仍保留部分历史兼容路径。 |
| `02_SSVEP` | 可用 | `02_SSVEP/START_SSVEP.py` | 活跃的 SSVEP 采集、训练、回放和验证工具链。 |
| `hybrid_controller` | 可用 | `hybrid_controller/run_real.py` | 当前主线集成控制器；硬件功能需要本地设备。 |
| `03_RobotArm_Control` | 实验性 | 目录 README/源码 | 历史机械臂实验，保留用于参考。 |
| `04_Communication_And_Integration` | 实验性 | 目录 README/源码 | 通信实验和集成说明。 |
| `05_Vision_Block_Recognition` | 实验性 | 历史脚本 | 早期视觉实验；当前模型兜底使用 `hybrid_controller/models/vision/best.pt`。 |
| `06_Data_Collection` | 实验性 | 历史脚本 | 数据采集工具和说明。 |
| `07_Simulation_Lab` | 实验性 | 目录 README/源码 | 仿真和沙盒实验。 |
| `02_SSVEP/_archive` | 归档 | 默认不直接运行 | 保留历史代码和输出，不作为默认 import 目标。 |
| `artifacts`、`02_SSVEP/artifacts` | 本地数据 | `docs/ARTIFACTS.md` | 为备份和复现保留的输出，不是最小化发布包。 |

默认用户入口见 `START_HERE.md`。直接运行实验性或归档脚本时，可能需要本地路径、真实设备或旧版假设。
