# brain 工作区与 brain_code 仓库进展报告

生成日期：2026-05-11
核对范围：`C:\Users\P1233\Desktop\brain` 与正式 Git 仓库 `C:\Users\P1233\Desktop\brain\brain_code`
核对方式：只读盘点目录、README、状态文档、profile、manifest、run/report 产物、Git 状态，并运行当前可用的轻量验证命令。

## 1. 总结

当前 `C:\Users\P1233\Desktop\brain` 不是正式代码仓库，而是本地工作区。它下面只有一个明确的 Git 仓库：

```text
C:\Users\P1233\Desktop\brain\brain_code
```

`brain_code` 是正式协作代码仓库，远端为 `https://github.com/pzy66/brain_code.git`，当前本地 `main` 与 `origin/main` 都指向：

```text
a688c42 Gate vision control on fresh ROS state
```

但工作区不是干净状态，当前有 17 个已修改文件，主要集中在两条正在推进的主线：

- `02_SSVEP`：外部短预训练 benchmark 的可部署预算、通道兼容性、缓存和报告字段增强。
- `hybrid_controller`：JetMax 视觉抓取连续伺服、官方相机读取、启动状态门控、调试文档和对应测试。

当前基础健康状态较好：

- `python -m brain diagnose` 通过，必需路径缺失数为 0，缺失模块数为 0。
- `python -m brain launch --simulate` 通过。
- `pytest --collect-only` 当前能收集 707 个测试。
- 根目录轻量测试 `tests`：16 passed。
- SSVEP 短预训练与 Hybrid 视觉/启动相关目标测试：152 passed。
- `git diff --check` 没有空白错误，只有 Windows 换行符 `LF will be replaced by CRLF` 提示。

## 2. brain 顶层工作区

`C:\Users\P1233\Desktop\brain` 的职责是本地工作区，不建议在这里直接做 Git 提交。顶层主要内容如下：

| 路径 | 作用 | 当前判断 |
| --- | --- | --- |
| `brain_code/` | 正式 Git 仓库，所有源码、测试、主要文档入口都在这里 | 当前唯一正式仓库 |
| `references/` | 父工作区参考资料包，含 SSVEP 参考库 zip 和当前使用参考资料副本 | 辅助资料，不是源码仓库 |
| `deliverables/` | 交付物目录，当前含 `jetmax_runtime_bundle` | 本地交付/打包输出 |
| `汇报/` | PPT、讲稿、PDF 预览等汇报材料 | 展示材料，不是源码 |
| `artifacts/`、`logs/` | 父工作区运行产物和日志 | 本地保留 |
| `_local_backups/`、`_workspace_archive/` | 备份和历史归档 | 本地保留 |
| `environment.brain-vision*.yml` | 当前 Python/Conda 环境描述 | 环境复现参考 |

父工作区 README 已明确：运行 Git、测试、代码编辑都应进入 `brain_code`。

## 3. brain_code 总体结构

`brain_code` 当前是一个多模块脑电控制机械臂项目。仓库维护策略已经从“历史脚本堆叠”转向“可移植协作仓库”：

- `brain`：统一 CLI 入口，提供 `brain diagnose`、`brain launch --simulate` 等硬件无关检查。
- `brain_workspace`：统一路径、环境诊断和数据根解析。
- `unified_collection`：统一 MI/SSVEP 采集入口。
- `01_MI`：运动想象 MI 采集、训练、实时推理主线。
- `02_SSVEP`：SSVEP 采集、训练评测、异步回放、短预训练优化主线。
- `hybrid_controller`：当前 JetMax 真机主程序和最终集成运行时。
- `03` 到 `07`：机械臂、通信、视觉、数据采集、仿真实验等历史/辅助工作区。
- `08_SoftCopyright_UI`：软件著作权 V1.0 演示与材料准备 UI。
- `datasets`：本地数据根，真实数据大多不进 Git，默认视觉模型 `datasets/vision/models/best.pt` 随仓库提供。
- `references`：随仓库保留的小型可共享参考材料。
- `docs`：仓库状态、边界、路线图、软著材料等文档。

当前 Git 跟踪规模：

- tracked files：589
- tracked size：约 39.3 MiB
- 最大跟踪文件：`datasets/vision/models/best.pt`，约 22.75 MiB
- 主要跟踪目录数量：`hybrid_controller` 164、`02_SSVEP` 146、`01_MI` 67、`references` 51

本地未跟踪或忽略的大数据实际很大，`datasets/` 当前约 38 GB，这符合“数据本地保留、源码轻量协作”的仓库策略。

## 4. 各模块用途与进展

### 4.1 `01_MI`

用途：

- 运动想象 MI 数据采集、训练、实时推理。
- 当前主线在 `01_MI/mi_classifier_latest/`。
- 包含 `collection`、`training`、`realtime` 三条链路。

当前进展：

- 采集和训练代码仍可用，保留了历史兼容路径。
- 规范数据根已经收敛到 `datasets/MI`。
- `datasets/MI/selection_policy.json` 明确当前筛选规则：保留四类全覆盖，并且每类至少 3 个 accepted epoch；不再按时长单独过滤。
- `datasets/MI/organize_report.csv` 当前记录：
  - 保留 `include_trainable_coverage`：11 个 session
  - 排除 `exclude_not_trainable_coverage`：13 个 session
- 当前保留 session 覆盖 subject：001、003、004、009。
- 四类类别名：`left_hand`、`right_hand`、`feet`、`tongue`。

当前数据状态：

- 文件系统中能看到 15 个 `*_mi_epochs.npz`。
- 其中 `selection_policy.json` 明确保留 11 个 session。
- 后续又出现了新 session，例如 `sub-003/ses-20260508_224122`、`sub-004/ses-20260507_*`、`sub-001/ses-20260510_185202`。这些不一定已经纳入最近一次整理策略。
- `sub-001/ses-20260510_185202` 文件名显示 `n-000_ok-000`，应视为未成功采集或不可训练候选，不能直接当成有效数据。

缺口：

- `datasets/profiles/MI/current_mi_profile.json` 当前不存在。
- 训练报告主要是 2026-03 的历史结果，新整理后的 11 个 canonical session 还需要跑一次当前训练/验证并产出新的 profile。
- 软著 UI 当前只依赖薄 MI contract，真正 MI 新分类器和 profile 还未形成稳定发布边界。

判断：

`01_MI` 已经完成“数据根规范化”和“可训练 session 保留规则”，但还没有完成“新 canonical 数据集 -> 当前 MI profile -> 主程序集成”的闭环。

### 4.2 `02_SSVEP`

用途：

- 当前 SSVEP 主线工程。
- 目标不是单一离线 top-1 分类器，而是完整异步控制链：采集 -> decoder -> gate/confidence -> decision -> report/profile -> profile 发布。
- 核心指标以 async-first 为主，包括 `idle_fp_per_min`、`control_recall`、`control_recall_at_2.5s/3s`、`switch_latency_s`、`release_latency_s`。

当前结构：

- `START_SSVEP.py`：统一启动入口。
- `apps/`：PyQt UI。
- `entrypoints/`：训练、采集、实时、local-opt 薄入口。
- `ssvep_core/`：核心算法、数据、gate、decision 和 run 组织。
- `tools/`：CLI、服务器辅助、外部 benchmark。
- `artifacts/`：本地 run、deployed profile、外部 replay 结果。

当前已部署 profile：

路径：

```text
02_SSVEP/artifacts/deployed_profiles/default_profile.json
02_SSVEP/artifacts/deployed_profiles/default_profile_v2.json
02_SSVEP/artifacts/deployed_profiles/profile_index.json
```

当前 index 显示：

- 更新时间：2026-04-30 19:03:58
- 来源任务：`fbcca-threshold-pretrain`
- 来源 run：`run_20260430_190356_fbcca-threshold-pretrain`
- profile 类型：`fbcca_threshold_only`

默认 profile 关键参数：

- 频率：`8 / 10 / 12 / 15 Hz`
- `win_sec = 3.0`
- `step_sec = 0.25`
- `min_enter_windows = 1`
- `min_exit_windows = 2`
- backend：CPU
- `recommended_for_realtime = true`

该 profile 的 benchmark 指标：

- `idle_fp_per_min = 0.0`
- `control_recall = 1.0`
- `control_recall_at_3s = 0.2059`
- `switch_latency_s = 3.5`
- `release_latency_s = 3.25`

这说明当前可部署 profile 已能抑制 idle false positive，但响应速度仍偏慢，尤其 3 秒内控制召回不高。

外部短预训练主线：

- 当前研究频率集：`9.8 / 12.0 / 14.8 / 15.8 Hz`
- 当前本地最新摘要：`02_SSVEP/artifacts/runs/local/external_short_pretrain_20260506_215442/reports/summary.json`
- 数据集：BETA
- 被试：S1
- 方法：`zero_shot_default`、`fast_fbcca`、`threshold_pretrain`
- 当前 best recipe：`threshold_pretrain search_w2_gpbalanced_me1_mx1_csunified`
- 指标：
  - `mean_idle_fp_per_min = 0.0`
  - `mean_control_recall = 0.3333`
  - `mean_control_recall_at_3s = 0.3333`
  - `mean_switch_latency_s = 4.0`
  - `mean_release_latency_s = 2.0`

判断：

- 这条短预训练结果还只是研究候选，且覆盖面只有 BETA S1，不能当成正式部署 profile。
- 当前 README 也明确指出，BETA 证据显示主要瓶颈是异步 hard no-control 拒识，而不是固定窗口四命令分频能力。
- 自适应 evidence gate 已有提升但未达可部署预算；下一步重点是 `lrt_multiwindow_reject_gate` 这种多窗口命令/非命令证据累积，而不是直接扩大 BETA 70 人 full run。

当前未提交进展：

- `run_external_short_pretrain_benchmark.py` 新增或增强：
  - deployable budget payload
  - deployable shared recipe summary
  - 通道兼容性 payload 与 summary
  - YSU-an no-control subtype metrics 保留
  - decoder/scored trial cache 复用
  - markdown 中显示 deployable 与 coverage
- 相关测试已覆盖并通过。

判断：

`02_SSVEP` 已经有完整工程化链路和可部署旧 profile，但“短预训练五分类可部署方案”仍处于研究验证阶段。下一步应继续围绕 idle/no-target 拒识、coverage、deployable budget 和服务器完整数据集验证推进。

### 4.3 `03_RobotArm_Control`

用途：

- 机械臂执行端历史控制代码。
- 当前核心路径为 `2026-03_jetmax_execution_server/test2_robot.py`。
- 包含 JetMax、ROS、socket 服务端和抓取动作逻辑。

当前进展：

- 已不作为当前主程序集成入口。
- 主要用于参考和对照。
- 真机主线已经转移到 `hybrid_controller` 的 ROS runtime 和 PC GUI。

判断：

这是历史/执行端参考模块，保留价值在于对照旧动作逻辑，不应作为新开发默认入口。

### 4.4 `04_Communication_And_Integration`

用途：

- 通信与跨模块联调脚本。
- 包含 socket、抓取命令发送、信号监控、视觉到抓取流程 debug。

当前进展：

- `2026-05_vision_grasp_flow_debug/` 保存摄像头识别到机械臂抓取命令的全过程联调工具。
- 安全流程明确为：

```text
dry-run -> camera-only -> resolve-only -> move-only -> execute-move -> allow-pick
```

判断：

这是 debug-to-main 的中间层。稳定行为应被提升进 `hybrid_controller`，而不是让主程序依赖这里的私有脚本。

### 4.5 `05_Vision_Block_Recognition`

用途：

- 木块识别、摄像头处理、视觉算法和早期训练脚本。
- 包含 YOLO 相机检测、模板匹配、颜色/轮廓处理等历史路径。

当前进展：

- 默认视觉权重已统一为 `datasets/vision/models/best.pt`，该模型作为基线资产随 Git 跟踪。
- 稳定识别输出应对齐 `vision_detection_schema.py` 的 schema。
- 真机视觉运行和抓取决策主线已经在 `hybrid_controller/vision` 中落地。

判断：

这是视觉算法来源和历史实验区。当前主程序不应直接从这里 runtime import，稳定代码应以复制/提升方式进入 `hybrid_controller`。

### 4.6 `06_Data_Collection`

用途：

- 非 MI/SSVEP 专属的独立数据采集工具。
- 当前主要内容是 JetMax block image/data collection 支持。

当前进展：

- 目录轻量，定位清楚。
- MI 采集仍归 `01_MI`，SSVEP 采集仍归 `02_SSVEP`。

判断：

这是视觉/机械臂数据采集辅助区，当前不是主运行入口。

### 4.7 `07_Simulation_Lab`

用途：

- 无硬件仿真和主程序接口验证。
- 当前包含 `hybrid_controller_sim/`。

当前进展：

- 有少量测试和仿真适配代码。
- 约束是仿真可以依赖稳定 `hybrid_controller` API，但 `hybrid_controller` 不能反向依赖仿真代码。

判断：

仿真层适合做硬件无关 smoke 和接口稳定性验证，后续可继续补充更小的演示数据和自动化流程。

### 4.8 `08_SoftCopyright_UI`

用途：

- 软件著作权 V1.0 演示和材料准备工作台。
- 目标是把系统作为一个完整软件产品展示：

```text
data acquisition -> model training/evaluation -> profile publishing
-> online MI/SSVEP recognition -> visual target resolution
-> JetMax move/pick/place -> logs, reports, replay, copyright materials
```

当前进展：

- 已建立独立 PyQt workbench。
- UI 第一阶段是硬件无关、只读、演示安全的操作：打开目录、定位文件、展示命令，不直接控制 BrainFlow、JetMax、ROS 或摄像头。
- 六个页面已经规划：Overview、Acquisition、Training、Online Control、Vision + Robot、Copyright Kit。
- 状态读取来自 `softcopyright_workbench.state` 和 `softcopyright_workbench.mi_contract`。
- `docs/softcopyright/` 已有材料草稿：
  - `softcopyright_planning.md`
  - `software_manual.md`
  - `user_manual.md`
  - `test_report.md`
  - `source_deposit_scope.md`
  - `source_manifest.draft.json`
  - `version_notes.md`
- `08_SoftCopyright_UI/artifacts/` 当前已有：
  - `workbench.png`
  - `workbench_static_preview.png`

当前验证：

- `py_compile` 已通过相关 UI 文件。
- `brain diagnose` 与 `brain launch --simulate` 已通过。

缺口：

- `source_manifest.draft.json` 仍是 draft，不是正式 freeze。
- MI 新 profile 和最终提交范围还未冻结。
- 如果用于正式软著提交，还需要在最终 commit/tag、测试结果、材料版本一致后再把 manifest 从 draft 推进到 frozen。

判断：

软著 UI 和材料包已经形成 V1.0 雏形，适合做硬件无关演示和材料整理，但还未进入最终冻结状态。

### 4.9 `hybrid_controller`

用途：

- 当前 JetMax 真机主程序和最终集成运行时。
- 负责 PC 端 GUI、ROS runtime、官方摄像头读取、视觉检测、连续视觉伺服、吸盘抓取、放置流程、键盘/SSVEP 决策接入和日志。

当前默认运行策略：

- 推荐入口：`hybrid_controller/run_real.py`
- 默认 `operator_keyboard` 模式，键盘/操作者输入替代 MI 和 SSVEP 识别。
- `run_real_ssvep.py` 保留为实验性/manual BCI 路径。
- 真机主流程默认走 ROS：
  - rosbridge：`192.168.149.1:9091`
  - 官方相机 MJPEG：`192.168.149.1:8080`
  - TCP `8888` 保留兼容诊断。

官方相机契约：

```text
usb_cam.service -> usb_cam_node -> /usb_cam/image_rect_color -> web_video_server:8080 -> PC
```

当前明确禁止：

- PC 主程序默认启动、重启、修复 `usb_cam.service`。
- 默认扫描多个非官方视频源。
- runtime 接管 JetMax 官方 camera sender。

当前视觉抓取主线：

- 默认使用 `delta_servo`。
- 默认使用 `command_bias` 偏置策略。
- 低处闭环对中目标使用当前帧 ROI/帧中心。
- 最终 `PICK_CYL` 只在当前半径上前伸一次 `vision_eye_in_hand_pick_radius_bias_mm = 40.0`。
- `pick_cyl_radius_bias_mm = 0.0`，避免二次前伸。

当前抓取 profile：

路径：

```text
datasets/profiles/hybrid_controller/vision_grasp/current_grasp_profile.json
```

关键字段：

- `profile_id = jetmax-wood-block-vision-grasp-default-v1`
- `real_pick_enabled = true`
- `vision_pick_confirm_z_mm = 130.0`
- `vision_eye_in_hand_pick_radius_bias_mm = 40.0`
- `pick_cyl_radius_bias_mm = 0.0`
- `vision_servo_low_action_tolerance_px = 8.0`

当前视觉标定：

路径：

```text
datasets/vision/calibration/current_profile.json
```

关键指标：

- `profile_id = eye-in-hand-live-20260503`
- image size：640 x 480
- residual median：2.36 mm
- residual p95：2.56 mm
- residual max：2.61 mm
- max allowed error：6.0 mm

这些标定指标在当前文件里看是合格的。

当前运行产物：

- `output/jetmax_no_suction_test_20260510_01` 到 `..._04` 存在连续视觉伺服/无吸盘调试输出。
- 这说明最近有 no-suction 视觉/伺服调试产物，但不等于正式抓取成功率验收已经完成。

当前未提交进展：

- 主程序连续视觉伺服只在明确 pick 意图后启动：
  - 手动 `Pick 1-4`
  - 或任务状态机确认目标
- 连续伺服新增最小置信度和最小面积配置：`vision_continuous_servo_min_confidence = 0.55`、`vision_continuous_servo_min_area_px = 1500`。
- 无 pending 时状态为 `continuous_idle awaiting_pick`，视觉自动启动不会自动运动或吸取。
- slot 不存在、机器人状态不新鲜、目标丢失、机器人失败事件会阻止或清空 pending，避免回退成直接 PICK。
- `VisionRuntime` 新增水平撕裂帧检测，用于拒绝拼接/破碎 MJPEG 帧。
- debug 工具的 continuous auto selection 更早锁定 pending slot。
- README 和 `references/Vision_Grasp/README.md` 已更新连续伺服、官方相机契约、MoveIt Servo/ros2_control/视觉伺服等参考说明。

当前验证：

- Hybrid 视觉、启动、连续伺服相关目标测试在本次核对中通过。

判断：

`hybrid_controller` 是当前最接近实机可用的主程序。它已经从“单次识别后开环抓取”推进到“明确目标意图后的连续视觉伺服闭环”，并且安全门控比之前更严格。下一步不应扩大功能，而应做 ROS 包/消息部署一致性、无吸盘定位复核和小范围真实抓取验收。

## 5. references 参考资料状态

仓库内 `references/` 已经按模块维护参考资料：

- `references/SSVEP/`：SSVEP 当前方法和数据集参考，包括 CCA、FBCCA、TRCA、TDCA、idle-state、dynamic stopping、pseudo-online evaluation、BETA、Wang2016、YSU-an 等。
- `references/MI/`：CSP、FBCSP、EEGNet、ATCNet family 和 BCI Competition IV 数据集说明。
- `references/Vision_Grasp/`：YOLO segmentation、Ultralytics predict、Hiwonder JetMax object tracking、visual servo control、MoveIt Servo、ros2_control、GG-CNN、Dex-Net 3.0、SAHI 等。
- `references/JetMax_WiFi/`：JetMax Wi-Fi/Windows 网络诊断相关材料。

符合当前 AGENTS.md 规则：如果后续实现继续使用某个论文/官方方法，应把它加入对应 reference。当前视觉抓取未提交改动已经把新增参考补到了 `references/Vision_Grasp/README.md`。

## 6. 当前未提交改动清单

当前 `git status --short` 显示以下文件已修改：

```text
02_SSVEP/tests/test_external_short_pretrain_benchmark.py
02_SSVEP/tools/run_external_short_pretrain_benchmark.py
hybrid_controller/README.md
hybrid_controller/app.py
hybrid_controller/config.py
hybrid_controller/docs/Hybrid_Controller_启动与部署说明.md
hybrid_controller/docs/JetMax_机械臂改动与落盘清单.md
hybrid_controller/docs/JetMax_真机联动说明.md
hybrid_controller/tests/test_continuous_vision_servo_controller.py
hybrid_controller/tests/test_debug_vision_grasp_flow.py
hybrid_controller/tests/test_robot_bootstrap.py
hybrid_controller/tests/test_vision_pick_resolution.py
hybrid_controller/tests/test_vision_runtime.py
hybrid_controller/tools/debug_vision_grasp_flow.py
hybrid_controller/vision/continuous_servo_controller.py
hybrid_controller/vision/runtime.py
references/Vision_Grasp/README.md
```

改动性质：

- `02_SSVEP` 改动偏研究评估和报告可靠性，不是直接替换在线 profile。
- `hybrid_controller` 改动偏真机视觉抓取安全门控和连续伺服行为。
- 参考文献改动已跟随视觉抓取方法更新。

建议：

在提交这些改动前，至少再运行一次：

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py -m pytest 02_SSVEP\tests hybrid_controller\tests -q -o addopts=
& $py -m brain diagnose
& $py -m brain launch --simulate
git diff --check
```

本次只运行了根测试和相关目标测试，没有跑完整 707 个测试。

## 7. 当前验证证据

本次核对已运行：

```text
python -m brain diagnose
python -m brain launch --simulate
python -m pytest --collect-only -q -o addopts=
python -m pytest tests -q -o addopts=
python -m pytest 02_SSVEP\tests\test_external_short_pretrain_benchmark.py hybrid_controller\tests\test_vision_pick_resolution.py hybrid_controller\tests\test_vision_runtime.py hybrid_controller\tests\test_robot_bootstrap.py hybrid_controller\tests\test_continuous_vision_servo_controller.py -q -o addopts=
python -m py_compile 08_SoftCopyright_UI 相关入口和状态文件
git diff --check
```

结果：

- `brain diagnose`：通过。
- `brain launch --simulate`：通过。
- pytest collect-only：707 tests collected。
- 根测试：16 passed。
- 目标测试：152 passed。
- 软著 UI py_compile：通过。
- `git diff --check`：仅换行符提示，无空白错误。

`tools/diagnose_workspace.ps1` 的额外信息：

- tracked files：589
- tracked size：39.3 MiB
- ignored entries：93
- permission warnings：112

permission warnings 主要来自 `.tmp`、pytest cache、历史临时目录和 GPU runtime tmp，不是当前源码测试失败，但会干扰仓库扫描和统计，建议后续清理或统一忽略/权限处理。

## 8. 主要风险与缺口

1. 工作区有未提交改动
   当前 `main` 与 `origin/main` 同步，但本地 dirty。报告中的“当前进展”包含未提交内容，不能直接等同 GitHub 最新主线。

2. 部分中文文档在默认 PowerShell 输出中显示 mojibake
   用 UTF-8 读取能正常显示，但默认 shell 输出会乱码。建议统一文档编码和编辑器配置，避免团队成员误判 README 内容损坏。

3. MI 数据整理报告与文件系统新增 session 不完全一致
   `selection_policy.json` 和 `organize_report.csv` 保留 11 个 session，但当前 `datasets/MI` 里有 15 个 `*_mi_epochs.npz`。需要重新跑一次数据整理或明确哪些是新采未审核数据。

4. MI 还没有当前 deployed profile
   `datasets/profiles/MI/current_mi_profile.json` 不存在。软著 UI 和未来主程序只能通过薄 contract 表示 MI 入口，不能宣称新 MI classifier 已完整发布。

5. SSVEP 短预训练仍未达到部署判断
   当前可部署 profile 是 8/10/12/15 的旧 FBCCA threshold profile。9.8/12/14.8/15.8 短预训练研究还没通过 hard no-control 和覆盖率要求。

6. Hybrid 连续视觉伺服需要实机再验收
   单元测试通过，no-suction 输出存在，但真实抓取成功率、失败原因分布、ROS 消息版本和 JetMax 端部署一致性仍需实机确认。

7. 软著材料仍是 draft
   `source_manifest.draft.json` 未冻结。正式提交前需要固定 commit/tag、测试结果、材料版本和源代码边界。

8. 本地数据体量大且多为 ignored
   `datasets/` 约 38 GB，适合本机开发，但团队协作时需要外部数据同步规则，不能靠 GitHub 自动获得完整复现实验条件。

## 9. 推荐下一步

1. 先决定当前 dirty worktree 的提交边界
   建议把 `02_SSVEP` benchmark 增强和 `hybrid_controller` 视觉伺服增强分成两个提交，便于回滚和审查。

2. 补跑完整模块测试
   如果时间允许，跑：

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py -m pytest 02_SSVEP\tests hybrid_controller\tests -q -o addopts=
```

3. MI 重新整理并训练当前 profile
   重新核对新增 session，更新 `organize_report.csv`，再跑当前训练入口，目标是产出 `datasets/profiles/MI/current_mi_profile.json` 和最小 smoke test。

4. SSVEP 继续围绕 no-control 拒识推进
   不建议直接启动大规模 full run。先用弱被试 smoke 验证 `lrt_multiwindow_reject_gate`，并要求报告中 `best_deployable_shared_recipe` 满足 coverage 与预算。

5. Hybrid 先做部署一致性检查，再做真机抓取
   确认 JetMax ROS 包和 `CylindricalTeleop` 消息已经与 PC 端代码一致；先 no-suction 定位，再小范围真实抓取。

6. 软著包等待源代码冻结
   等当前源码提交、测试结果稳定、MI contract 状态明确后，再把 `source_manifest.draft.json` 推进为 frozen，并同步更新 test report。

7. 清理临时目录和编码问题
   使用已有清理脚本处理 pytest/gpu tmp 权限噪声；对 README/状态文档统一 UTF-8 显示和编辑规则。

## 10. 一句话状态

`brain_code` 当前已经从散乱实验目录整理成一个可协作的多模块仓库：MI 数据根已规范化，SSVEP 有完整异步评估与旧可部署 profile，Hybrid Controller 是当前真机主程序且正在强化连续视觉伺服和安全门控，软著 UI/材料包已成型但未冻结。当前最关键的未完成项是：提交并验证本地 dirty 改动、产出 MI 当前 profile、让 SSVEP 短预训练通过 no-control 可部署门槛，以及完成 Hybrid 实机抓取验收。
