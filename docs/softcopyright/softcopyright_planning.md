# 软著 V1.0 规划文档

## 1. 申报目标

软件名称暂定为：**基于混合脑机接口的智能机械臂协同控制软件 V1.0**。

V1.0 的申报对象不是单独的 MI 分类器、SSVEP 解码器或机械臂控制脚本，而是一个完整的软件闭环：脑电数据采集、模型训练评估、profile 发布、在线意图识别、视觉目标解析、机械臂安全执行、运行验证和软著材料管理。

软著申报强调软件表达和工程实现。V1.0 因此要把仓库整理成“可运行、可截图、可解释、可测试、可冻结”的状态，而不是只堆算法实验。

## 2. 官方依据与材料方向

规划参考以下官方公开资料：

- 中国版权保护中心与中国版权登记平台的软件著作权登记入口。
- 《计算机软件保护条例》。
- 《计算机软件著作权登记办法》。

材料准备方向：

- 软件说明书：说明系统目标、架构、模块、流程和运行环境。
- 用户手册：说明如何启动 UI、查看状态、生成截图、执行无硬件验证。
- 测试报告：记录语法检查、UI 截图、静态预览、`brain diagnose`、`brain launch --simulate` 和后续 MI smoke test。
- 源码交存清单：明确纳入与排除范围，排除真实 EEG 数据、外部数据集、论文 PDF、训练输出、日志和缓存。
- 版本说明：说明 V1.0 功能边界、暂不纳入内容和后续版本计划。

## 3. 当前仓库定位

当前仓库根目录为 `brain_code`。V1.0 规划采用以下模块边界：

- `08_SoftCopyright_UI/`：软著演示工作台，负责六页 UI、只读状态读取、路径定位、命令预览和截图。
- `01_MI/`：MI 采集、训练和实时推理。新 MI 分类器并入后，应通过薄契约向 UI 暴露状态。
- `02_SSVEP/`：SSVEP 采集、短预训练、异步控制评估和 profile 管理。
- `hybrid_controller/`：视觉识别、机械臂协同控制、机器人通信和安全门控。
- `datasets/profiles/`：可公开的 profile/schema/default config 存放位置。
- `docs/softcopyright/`：软著材料草稿、测试记录和冻结清单。
- `references/`：方法引用、官方资料索引和可公开参考说明。

UI 当前采用 PyQt 独立工作台，不并入 `brain` CLI。等 V1.0 演示稳定后，再考虑增加 `brain launch --target softcopyright`。

## 4. V1.0 功能边界

### 4.1 纳入 V1.0

- 六页主导航：总览、数据采集、训练评估、在线控制、视觉机械臂、软著材料。
- 演示模式状态源：读取仓库路径、profile、schema、默认视觉模型、材料草稿和入口文件状态。
- MI 接入契约：训练入口、实时推理入口、profile 输出、状态文件、smoke test 和 schema。
- SSVEP 复用：只读取既有入口、profile 和 report，不复制 SSVEP 训练逻辑。
- 视觉机械臂复用：只读取 `hybrid_controller` 入口、配置、视觉模型和抓取 profile。
- 在线控制规则表达：键盘负责调试兜底，MI 负责连续移动意图，SSVEP 负责离散选择/确认/释放，低置信度进入 idle/no-control。
- 无硬件演示：不要求 BrainFlow、摄像头、ROS、JetMax 在线。

### 4.2 暂不纳入 V1.0

- 软著 UI 直接启动真实抓取。
- 软著 UI 绕过 `hybrid_controller` 的 MOVE/PICK/PLACE 安全门控。
- 软著 UI 写入真实 profile。
- 真实 EEG 数据、外部数据集原始文件、论文 PDF、训练输出、日志和缓存。
- 尚未入库的新 MI 分类器最终权重和最终精度承诺。

## 5. MI 分类器接入规划

新 MI 分类器并入仓库后，必须满足以下接口契约：

1. 训练入口：可从命令行启动训练或 smoke training。
2. 实时推理入口：可在无真实硬件或模拟输入下做 smoke inference。
3. profile 输出：默认发布到 `datasets/profiles/MI/current_mi_profile.json`。
4. 状态文件：建议输出 `datasets/profiles/MI/mi_status.json`，字段包含 `phase`、`updated_at`、`last_error`、`metrics`。
5. schema：遵循 `08_SoftCopyright_UI/schemas/mi_profile.schema.json`。
6. 测试：至少提供训练 smoke test、实时推理 smoke test 和 profile schema 校验。
7. 引用：若采用 CSP、FBCSP、EEGNet、ATCNet 或其他方法，必须在 `references/MI/README.md` 补充来源和工程边界。

UI 只依赖薄适配层，不直接 import 深层实验脚本。MI 状态建议区分：

- `missing`：没有可识别入口。
- `legacy_detected`：检测到旧入口，但未满足 V1.0 契约。
- `ready`：训练、实时、schema、smoke test 已就绪，等待发布 profile。
- `training`：检测到 MI 状态文件，训练或发布流程正在运行或由 MI 模块管理。
- `published`：当前 profile 已发布，UI 可读取。
- `error`：状态文件或 profile 声明失败。

## 6. UI 规划

V1.0 UI 的核心不是“炫酷”，而是申报友好和审查友好：

- 第一屏要能说明系统闭环，不依赖硬件在线。
- 每页都要能回答“这个功能对应哪段源码、哪份材料、哪条验证命令”。
- 所有按钮必须是安全只读动作：打开目录、定位文件、显示命令、刷新状态。
- 不在 UI 中直接触发真实机器人动作。
- 截图必须稳定，Qt offscreen 不稳定时使用静态预览工具。

页面优化方向：

- 总览：展示 MI、SSVEP、视觉、机械臂、材料的状态卡。
- 数据采集：展示 MI/SSVEP 入口、数据目录、profile 路径和采集质量门槛。
- 训练评估：展示模型状态、验收指标、发布 gate 和 smoke test。
- 在线控制：展示输入源仲裁、idle/no-control、低置信度保护。
- 视觉机械臂：展示视觉模型、抓取 profile、hybrid_controller 入口、安全阶梯。
- 软著材料：展示说明书、用户手册、测试报告、源码交存清单、版本说明、截图目录和冻结命令。

## 7. 源码交存边界

源码冻结以 `docs/softcopyright/source_manifest.draft.json` 为草案。

纳入：

- 自有源码：`brain/`、`brain_workspace/`、`01_MI/`、`02_SSVEP/`、`08_SoftCopyright_UI/`、`hybrid_controller/`。
- 软著材料：`docs/softcopyright/`。
- 测试：`tests/`、`02_SSVEP/tests/`、`hybrid_controller/tests/` 和后续 MI tests。
- 可公开 profile/schema/config。
- 默认视觉基线资产 `datasets/vision/models/best.pt`。
- 可公开引用说明 `references/`。

排除：

- `datasets/MI/**`、`datasets/SSVEP/**` 中的真实 EEG 数据。
- 外部数据集原始文件。
- 论文 PDF。
- `logs/`、`artifacts/`、`output/`、缓存和临时图片。
- 大体积训练权重和中间产物，除非明确列为可公开基线资产。

冻结前应生成最终只读清单，记录 commit、tag、测试命令、测试结果、截图路径和源码范围。

## 8. 测试与验收计划

第一阶段无硬件验收：

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py -m py_compile `
  .\08_SoftCopyright_UI\run_softcopyright_workbench.py `
  .\08_SoftCopyright_UI\softcopyright_workbench\app.py `
  .\08_SoftCopyright_UI\softcopyright_workbench\state.py `
  .\08_SoftCopyright_UI\softcopyright_workbench\mi_contract.py `
  .\08_SoftCopyright_UI\tools\render_static_preview.py

& $py -m json.tool .\08_SoftCopyright_UI\schemas\mi_profile.schema.json > $null
& $py -m json.tool .\docs\softcopyright\source_manifest.draft.json > $null
& $py .\08_SoftCopyright_UI\run_softcopyright_workbench.py --screenshot .\08_SoftCopyright_UI\artifacts\workbench.png
& $py .\08_SoftCopyright_UI\tools\render_static_preview.py --output .\08_SoftCopyright_UI\artifacts\workbench_static_preview.png
& $py -m brain diagnose
& $py -m brain launch --simulate
```

MI 并入后新增：

- MI profile schema 校验。
- MI 训练 smoke test。
- MI 实时推理 smoke test。
- MI 状态文件读取测试。

冻结前目标测试：

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py -m pytest tests -q -o addopts=
& $py -m pytest .\02_SSVEP\tests -q -o addopts=
& $py -m pytest .\hybrid_controller\tests -q -o addopts=
```

失败项必须分类为：UI 无关、硬件依赖、环境缺失、V1.0 必须修复。

## 9. 里程碑

### M1：UI 演示工作台稳定

- 六页 UI 可启动。
- 状态读取可刷新。
- 静态预览与真实 UI 复用同一状态源。
- 软著材料页能读取材料草稿状态。

### M2：MI 分类器并入

- 新 MI 分类器入库。
- 输出训练入口、实时入口、profile、status、schema 和 smoke test。
- `references/MI/README.md` 补齐实际采用方法的来源。

### M3：系统闭环材料补齐

- 软件说明书从草稿扩展为正式版。
- 用户手册补截图和运行步骤。
- 测试报告记录实测结果。
- 源码交存清单从 draft 转为 freeze 版本。

### M4：冻结与申报

- 运行冻结前测试。
- 生成最终截图。
- 生成最终源码清单。
- 创建 `softcopyright-v1.0` tag。
- 固化申报材料包。

## 10. 风险与控制

- MI 分类器未入库：用契约先锁定边界，避免 UI 直接依赖实验脚本。
- UI 与真实硬件耦合：V1.0 UI 保持只读演示，真实动作留在 `hybrid_controller`。
- 源码交存范围过大：用 `source_manifest.draft.json` 逐步收紧文件级规则。
- 引用材料不完整：采用的方法必须进入 `references/`，论文 PDF 不放入交存包。
- 截图不稳定：保留 PyQt screenshot 和 PIL static preview 两条路径。
- 测试失败混杂：冻结前必须把失败项分类，避免把硬件依赖失败误判为软件不可运行。

## 11. 下一步执行清单

1. 把新 MI 分类器按契约并入仓库。
2. 增加 MI profile/status 生成与 schema 校验。
3. 增加 MI smoke tests。
4. 更新 `test_report.md`，补 MI 并入后的实测结果。
5. 更新 `source_manifest.draft.json`，确认最终 include/exclude。
6. 补正式版截图到用户手册和软件说明书。
7. 创建 `softcopyright-v1.0` tag 前，运行冻结前目标测试。
