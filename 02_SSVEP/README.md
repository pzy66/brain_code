# 02_SSVEP

本目录是当前 SSVEP 主线工程。它已经从早期按月份堆叠的脚本目录，收敛成一套以 `run` 为中心的可追溯结构，覆盖：

- 数据采集
- 实时在线解码
- 离线训练与评测
- 本地异步优化
- 外部数据回放仿真
- 结果归档与 profile 管理

如果你只想知道“从哪里开始”，直接看：

- 统一入口：`START_SSVEP.py`
- 训练评测主入口：`entrypoints/start_training_eval.py`
- 本地 TDCA 异步优化：`entrypoints/start_tdca_local_opt.py`
- 本地 FBCCA 异步优化：`entrypoints/start_fbcca_local_opt.py`
- 外部 8 通道 replay 优化：`entrypoints/start_fbcca_external_replay.py`

---

## 1. 当前主线在做什么

这一版代码的目标，不是只做一个“离线 top-1 分类器”，而是把 SSVEP 整条使用链路跑通：

1. 采集真实 session 数据。
2. 用统一的 dataset schema 落盘。
3. 用 decoder 产出每个窗口的频率分数。
4. 用 gate / confidence 过滤低可信窗口。
5. 用 decision state machine 形成 enter / commit / release。
6. 产出 report、selection snapshot、profile、profile_v2。
7. 只有满足 acceptance 条件时，才允许覆盖部署 profile。

也就是说，仓库里的“结果”默认不是单个准确率，而是一整组 async-first 指标，例如：

- `control_recall`
- `control_recall_at_3s`
- `idle_fp_per_min`
- `release_latency_s`
- `switch_latency_s`
- `inference_ms`

---

## 2. 目录结构

```text
02_SSVEP/
  START_SSVEP.py
  README.md
  apps/                PyQt UI 程序
  entrypoints/         薄入口，适合日常启动
  ssvep_core/          核心算法、数据、gate、decision、run 主线
  tools/               CLI、服务器辅助、迁移脚本
  docs/                额外文档与方法说明
  tests/               pytest 测试
  artifacts/           数据集、run 产物、deployed profiles
  _archive/            历史归档，不应被新代码 import
```

各子目录详细说明：

- [apps/README.md](./apps/README.md)
- [entrypoints/README.md](./entrypoints/README.md)
- [tools/README.md](./tools/README.md)
- [ssvep_core/README.md](./ssvep_core/README.md)
- [docs/README.md](./docs/README.md)

---

## 3. 推荐使用路径

### 路径 A：日常使用

适合你手动点 UI、看日志、看 run 目录。

1. 运行 `python START_SSVEP.py`
2. 从 launcher 进入：
   - 数据采集
   - 实时在线解码
   - 训练评测
   - TDCA 本地异步优化

### 路径 B：直接进入某条工作流

- 采集：`python entrypoints/start_collection.py`
- 实时：`python entrypoints/start_realtime.py`
- 训练评测：`python entrypoints/start_training_eval.py`
- TDCA local opt：`python entrypoints/start_tdca_local_opt.py`
- FBCCA local opt：`python entrypoints/start_fbcca_local_opt.py`
- external replay：`python entrypoints/start_fbcca_external_replay.py`

### 路径 C：纯命令行

适合服务器、批跑、CI、或你想把参数写死在命令里。

- 入口：`python tools/training_evaluation_cli.py ...`

---

## 4. 三条主要业务链路

### 4.1 数据采集链路

入口：

- `apps/data_collection_ui.py`
- `entrypoints/start_collection.py`

作用：

- 控制采集轮次、刺激阶段和 trial 协议。
- 从 BrainFlow/BoardShim 读数据。
- 把每个 trial 的原始片段存到 `raw_trials.npz`。
- 生成 `session_manifest.json`，作为后续训练和评测的标准输入。

当前采集窗口与提示音语义：

- `prepare` 阶段会先发语音提示，等待语音播报完成后再发准备提示音；如果语音特别短，仍保留一个最小 guard，避免入耳重叠。
- 当前默认会最多等待 `5s` 的语音完成确认，并在语音结束后额外保留 `0.8s` 保护间隔；如果确认超时，会先强制停语音，再进入后续提示音。
- `active_start` 提示音发生在 `board.get_board_data()` 清空缓冲之后，但保存窗口取的是“提示音之后最近 `active_sec` 的样本”，因此开始提示音本身不会落进最终保存片段。
- `active_end` 提示音在片段抓取完成之后触发，因此也不会污染保存窗口。
- `session_manifest.json` 里会记录 `voice_prompt_guard_sec`、`active_start_buffer_clear_timing`、`active_saved_window`、`active_end_cue_timing` 等字段，后续做训练、回放或跨机复核时应以这些字段为准。
- 每个 trial 还会记录 `active_start_tone_started_at`、`active_window_started_at`、`active_window_ended_at`、`segment_captured_at`、`active_end_tone_started_at` 这组带时区的 ISO 8601 时间戳，便于人工复核“提示音位置”和“最终保存窗口”是否一致。
- UI 里新增了“流程测试模式（不连接板卡，不保存数据）”：勾选后，点击“开始本轮采集”会正常运行语音、提示音、闪烁刺激、trial/轮次流程，但不会连接设备、不会读取 EEG、不会生成 manifest/npz，也不会计入正式完成轮次。
- 同样也可以从命令行预设：`python entrypoints/start_collection.py --simulation-only`。如果再叠加 `--headless`，则只跑时序流程，不会渲染视觉刺激。

当前落盘格式特性：

- `raw_trials.npz` 采用 `np.savez_compressed`，每个 trial 单独一个 key，因此不同 trial 可以拥有不同的样本长度。
- `session_manifest.json` 保存 trial 顺序、`npz_key`、`used_samples`、`target_samples`、`active_sec`、`retry_count` 等元数据，读取时优先依赖 manifest，而不是假设所有 trial 长度一致。
- `generated_at` 使用带时区的 ISO 8601 时间戳；如果把数据包从采集机拷到训练机，时间语义仍然明确。
- 如果用户手动停止，或者运行时在中途异常但前面已有有效 trial，程序也会把已采到的片段落盘；manifest 的 `protocol_config.collection_aborted`、`protocol_config.aborted_reason`、`protocol_config.failure_reason` 可用于区分是正常完成、用户停止还是异常中断。
- UI 先做设备检查时，成功后会把解析出的实际串口回写到串口输入框；这样下一次正式采集会继续使用同一个真实设备，而不是再次依赖 `auto`。
- `entrypoints/start_collection.py --headless` 只适合联调保存链路，不会渲染视觉刺激，不应用于正式 SSVEP 数据采集。

当前主线协议重点：

- 默认目标频率：`8 / 10 / 12 / 15 Hz`
- 固定 4 目标本地主线
- 支持 `stable_12m` 与 `enhanced_45m`
- trial 会被标注为：
  - `control`
  - `clean_idle`
  - `hard_idle`

适用场景：

- 采新数据
- 做 session 级追加
- 为 local-opt 或 model-compare 准备输入

### 4.2 实时在线链路

入口：

- `apps/realtime_online_ui.py`
- `entrypoints/start_realtime.py`

作用：

- 加载已发布的 deployed profile。
- 根据 profile 恢复 decoder、gate、decision 参数。
- 接真实在线 EEG 流，做窗口打分与状态更新。
- 支持 shadow runtime，便于在线观察但不改主输出。

当前主线原则：

- 在线优先读取 `artifacts/deployed_profiles/`
- profile 与 profile_v2 并行保留
- 若 profile 不满足约束，不应直接手工替换 deployed profile

### 4.3 训练评测与优化链路

入口：

- `apps/training_evaluation_ui.py`
- `entrypoints/start_training_eval.py`
- `tools/training_evaluation_cli.py`

作用：

- 统一发起 benchmark、compare、local-opt、external replay
- 写标准 run 目录
- 产出完整报告与快照
- 控制 profile 是否允许发布

这是当前最核心的“研究与迭代”主界面。

---

## 5. 训练评测任务总览

当前工程里至少存在两层任务体系：

1. **UI 主公开任务**：训练评测 UI 里直接暴露的任务
2. **CLI 研究任务**：只在 `tools/training_evaluation_cli.py` 中暴露，偏脚本化

### 5.1 UI 主公开任务

#### `fbcca-weights`

用途：

- 快速跑一轮 FBCCA 工程权重筛选
- 适合日常 smoke 和 quick screen

特点：

- 偏快速
- 常用于初筛

#### `model-compare`

用途：

- 比较核心模型族在同一数据上的离线与异步表现

常见对象：

- `tdca`
- `trca_r`
- `etrca_r`
- `fbcca`

#### `fbcca-weighted-compare`

用途：

- 聚焦 FBCCA 家族工程变体对比
- 包括 legacy FBCCA 与权重化 FBCCA 变体

#### `tdca-local-opt`

用途：

- 在本地 4 目标数据上，专门跑 TDCA 的 async-first 搜索

特点：

- 搜 decoder variant、窗长、delay、components、confidence、gate、decision
- 输出的是 run 内 profile，不一定自动发布
- 当前 preset：
  - `smoke4`
  - `reduced13`
  - `full96`

#### `fbcca-local-opt`

用途：

- 在本地 4 目标数据上，专门跑 FBCCA 家族 async-first 搜索

当前主线：

- 5 个 FBCCA 工程变体
- 2 条 confidence 路线
- 固定窗长搜索

当前 preset：

- `smoke20`
- `reduced40`

#### `fbcca-external-replay-opt`

用途：

- 在外部公开 8 通道数据上做“预训练 -> held-out session 连续回放仿真”

固定科学口径：

- 真实 3 目标：`13 / 17 / 21 Hz`
- 加 `rest`
- 不伪造第 4 目标
- 不把外部 profile 回流到真实在线主 profile

当前主线：

- minimal CCA family：
  - `fbcca_fixed_all8`
  - `fbcca_sw_all8`
  - `itcca_all8`
  - `ecca_all8`
- 2 条 confidence：
  - `global_correctness_logistic`
  - `bayesian_gap_gmm`

当前 preset：

- `smoke8`
- `reduced24`

### 5.2 CLI 额外研究任务

这些任务不一定在 UI 中长期公开，但在 CLI 里可直接调用：

- `focused-compare`
- `classifier-compare`
- `profile-eval`

它们更适合：

- 脚本化对照
- 扩展模型盘点
- 评估某个既有 profile 的冻结表现

---

## 6. 外部频率选择与短预训练五分类

这条主线的目标是：用很短的个人预训练，在公开数据集上得到可迁移、可比较的 SSVEP 五分类配置。这里的“五分类”按当前代码语义是 `idle + 4 个 SSVEP 命令频率`，不是 5 个命令频率。

### 6.1 四个频率怎么判别最合适

当前正式工作频率集是：

```text
9.8 / 12.0 / 14.8 / 15.8 Hz
```

频率选择不要只看离线 top-1 accuracy。当前更合理的排序口径是 async-first：

1. 目标频率必须能在目标公开数据集中取到，且所有被试可比。
2. 频率间距要足够，当前 sweep 通常使用 `min_spacing=1.0`、`min_freq=9.5` 过滤候选。
3. 若不伤害指标，优先选择 240 Hz 下更接近 frame-lock 的组合。
4. 先压低 `idle_fp_per_min` 和 `idle_selected_windows_per_min`，再看控制态召回。
5. 控制态重点看 `control_recall_at_2.5s`、`control_recall_at_3s` 和总体 `control_recall`。
6. 再看每个频率的最小召回，避免某一个频率明显拖后腿。
7. 最后比较 macro-F1、accuracy 和 latency。

对应脚本是 `tools/run_external_frequency_server_sweep.py`。它会把这些候选放进正式评估：

- 当前频率集：`9.8,12,14.8,15.8`
- Wang-like baseline：`8,10,12,15`
- 若干 240 Hz exact/frame-lock 候选
- Wang2016/Beta sweep 里排在前面的组合

脚本里的 `_rank_key()` 明确把 idle 假阳性、idle selected windows、控制态召回、每频率召回下限、latency 和 240 Hz frame-lock 放进排序，因此看报告时应优先看这些字段，而不是只看分类准确率。

### 6.2 当前短预训练分类器优化

当前主脚本是 `tools/run_external_short_pretrain_benchmark.py`。最近优化重点有两个：

1. 新增 coverage-aware 汇总，避免“只覆盖单个被试的 recipe”被误读成共享最优。
2. 增加被试级 scored-trial cache，减少同一 split/idle 组合下重复打分，服务器全量 Wang2016 扫描会更可控。

这条线现在已经从“只盯 FBCCA 网格”切到“文献导向短预训练方法族”：

- baseline 仍保留 `fbcca_lda5,fbcca_ridge5`
- 第一批新增并正式比较 `itcca5,ecca5,trca5,trca_r5,tdca5`
- 门控统一走 score-to-idle gate，不给每个 decoder 单独写一套门控逻辑
- `tdca5` 只在 raw window 足够长时参与候选，按有效 raw length 过滤

当前这一轮的实际顺序是：

- Beta sanity：`S1,S16,S35,S70`
- Beta full：`S1-S70`
- Wang confirm：只在 Beta 候选固定后再跑，不重新大搜 Wang

sanity 配置：

- dataset：`beta`
- subjects：`S1,S16,S35,S70`
- methods：`itcca5,ecca5,trca5,trca_r5,tdca5`
- freqs：`9.8,12,14.8,15.8`
- calibration blocks：`2`
- idle multipliers：`2.0,3.0`
- classifier windows：`1.0,1.25,1.5,1.75,2.0`
- min enter windows：`1,2`
- max splits per subject：`1`
- compute backend：`cuda`

full Beta 配置：

- dataset：`beta`
- subjects：`S1-S70`
- methods：从 sanity 中选出的 top 2-3 个
- windows：围绕 sanity 最佳窗口 ±`0.25s`
- min enter windows：`1,2,3`
- idle multipliers：`2.0,3.0,4.0`
- max splits per subject：`2`
- compute backend：`cuda`

远端输出只允许写在 `/data1/zkx/brain/ssvep/` 下：

```text
/data1/zkx/brain/ssvep/reports/external_short_pretrain/<run_id>
/data1/zkx/brain/ssvep/data/external_short_pretrain_datasets/<run_id>
/data1/zkx/brain/ssvep/logs/<run_id>.log
```

数据集状态：

- 本地 `wang2016` 不完整，目前只确认有 `S1.mat`。
- 服务器 `/data1/zkx/brain/ssvep/data/external_sources/wang2016/raw` 完整，确认有 `S1.mat` 到 `S35.mat`，共 35 人，并有 `64-channels.loc`。
- 下一步使用服务器完整 Wang2016，不把全量数据同步回本地。

### 6.3 共享 recipe 判定规则

`summary.json` 里现在同时保留两套结论：

- `best_recipe`：兼容旧报告，可能来自 partial coverage。
- `best_shared_recipe`：当前应该采用的主结论，必须覆盖预期被试集合。

每个 recipe summary 会记录：

- `expected_subject_count`
- `coverage_subject_count`
- `shared_eligible`

正式结论只看满足下面条件的 recipe：

```text
shared_eligible == true
coverage_subject_count == expected_subject_count
```

这样可以避免 `S16-only win3` 这类 partial coverage 结果误导决策。

已确认的 Beta 小规模结果是：最新 `beta:S1,S16` run 中，旧 `best_recipe = fbcca_lda5 win3_me1` 实际只覆盖 `S16`，不能当共享最优。按 `S1+S16` 都覆盖的共享条件重算，当前最佳是：

```text
method: fbcca_ridge5
window/min-enter: win2_me1
calibration_blocks: [2,3]
idle_multiplier: 2.0
mean async 5-class acc: 0.9625
mean async 5-class macro-F1: 0.8996
mean idle FP/min: 1.0417
mean control recall: 1.0
mean latency: 2.0s
```

### 6.4 当前 Wang2016 运行状态与下一步

当前远端 Wang2016 全量 run 记录为：

```text
external_short_pretrain_wang2016_shared_cache_20260507_180858
```

最近一次记录显示它仍在运行，已产出 `partial_summary.json`，但最终 `summary.json` 和 `best_shared_recipe` 还没有落盘。因此 README 里不要把任何 Wang2016 partial coverage 结果写成最终共享结论。

第一轮完成后的验收条件：

1. `summary.json` 存在。
2. `best_shared_recipe.coverage_subject_count == 35`。
3. `shared_recipe_summaries` 非空。
4. Markdown 报告中的共享榜单必须显示 coverage，不能把 partial coverage recipe 当共享最佳。

如果第一轮 best shared recipe 覆盖 35 人并达到预算：

- `control_recall >= 0.80`
- `idle_fp_per_min <= 1.0`
- 优先最大化 `async_macro_f1_5class`
- latency 优先 `<= 2.0s`

下一步固定该 recipe 做稳健性验证：窗口在最优附近微调，例如 `1.25,1.5,1.75,2.0,2.25`，并把 `max_splits_per_subject` 提高到 `2` 或 `3`。

如果第一轮未达预算，先处理 idle FP/min 超标问题，而不是盲目追求 4-class 准确率。优先把 `idle_multiplier` 扩到 `3.0`，并继续保留 `fbcca_ridge5`，因为它在当前 Beta 小规模验证中更稳。

这些 benchmark artifact 目前仍是研究候选结果。进入在线部署前，还需要补 runtime loader/profile 路径、replay 路径和真实在线路径的一致性测试。

---

## 7. 运行产物与 run 目录规则

当前仓库已经统一采用按任务、按日期、按 run id 的归档方式。

### 7.1 标准路径

```text
artifacts/
  datasets/
  runs/
    local/<task>/<YYYYMMDD>/<run_id>/
    remote/<task>/<YYYYMMDD>/<run_id>/
    _legacy_imported/
  deployed_profiles/
    default_profile.json
    default_profile_v2.json
    profile_index.json
```

### 7.2 一个标准 run 目录通常包含

- `report.json`
- `report.md`
- `selection_snapshot.json`
- `run_config.json`
- `progress_snapshot.json`
- `run.log`
- `profile.json`
- `profile_v2.json`
- `figures/`

并不是所有任务都会发布 profile，但标准路径会统一保留。

### 7.3 `progress_snapshot.json` 的意义

这是训练评测 UI 和其他监控逻辑的主要状态来源，通常会记录：

- 当前阶段
- 当前候选
- 当前进度
- 当前 run 目录
- 最终状态

如果你要做“长任务可观察”，优先盯这个文件。

---

## 8. profile、profile_v2 与 deployed profile

### 8.1 `profile.json`

偏旧格式，但仍是许多运行时链路的直接输入。

### 8.2 `profile_v2.json`

结构化程度更高，拆成：

- decoder
- gate
- evidence
- runtime
- metrics
- metadata

### 8.3 `artifacts/deployed_profiles/`

这里是在线系统真正默认读取的位置，通常包含：

- `default_profile.json`
- `default_profile_v2.json`
- `profile_index.json`

`profile_index.json` 记录最近一次发布来源。

### 8.4 发布原则

不是每次 run 结束都应该发布 profile。

一般原则是：

1. run 必须有效。
2. 排名与 acceptance 达标。
3. 任务语义允许发布。

尤其要注意：

- `fbcca-external-replay-opt` 产物默认属于 `simulation_only_profile`
- 它们允许复制到 external replay 专用路径
- 但不应覆盖真实在线 4 目标系统的默认 deployed profile

---

## 9. 外部 replay 主线需要特别知道的事

当前 external replay 是一条**独立研究主线**，不是本地 4 目标在线系统的替代品。

### 固定数据协议

- 数据格式：GDF
- 目标频率：`13 / 17 / 21 Hz`
- 休息段：`rest`
- 通道固定：`Oz, O1, O2, PO3, POz, PO7, PO8, PO4`

### 固定评估视角

- 主口径：`loso4`
- 部署故事线：`chronological_last_session`

### 固定仿真性质

- `simulation_protocol = "continuous_session_replay"`
- `simulation_only_profile = true`

### 当前最重要的结论读取方式

不要只看 raw top-1。

应至少一起看：

- `raw_correct_rate`
- `gate_pass_rate`
- `commit_rate`
- `release_seen_rate`
- `median_max_p_correct`
- `median_max_decision_evidence`
- `error_attribution_board`
- `decision_bottleneck_summary`

---

## 10. 常见工作流

### 工作流 1：采一轮新数据

1. 打开 `START_SSVEP.py`
2. 进入“数据采集”
3. 选择协议
4. 保存到 `artifacts/datasets/<session_id>/`
5. 确认生成：
   - `session_manifest.json`
   - `raw_trials.npz`

### 工作流 2：比较模型

1. 打开 `entrypoints/start_training_eval.py`
2. 选择 `model-compare`
3. 载入一个或多个 manifest
4. 运行
5. 看 `report.json` 和 `report.md`

### 工作流 3：本地 TDCA 调优

1. 打开 `entrypoints/start_tdca_local_opt.py`
2. 先跑 `smoke4`
3. 再跑 `reduced13`
4. 只有通过后才考虑 `full96`

### 工作流 4：本地 FBCCA 调优

1. 打开 `entrypoints/start_fbcca_local_opt.py`
2. 默认从 `reduced40` 开始
3. 看 holdout 选型、confidence 诊断和 async 指标

### 工作流 5：外部 8 通道 replay

1. 打开 `entrypoints/start_fbcca_external_replay.py`
2. 选择 external dataset root
3. 选择 subject
4. 先跑 `smoke8`
5. 再跑 `reduced24`
6. 用 `apps/external_replay_viewer.py` 看 timeline

---

## 11. 环境与依赖建议

本目录至少依赖下列类型组件：

- Python 3
- `PyQt5`
- `numpy`
- `scipy`
- `mne`
- `brainflow`
- `pytest`

其中：

- 数据采集与实时在线依赖 BrainFlow/BoardShim 和设备串口
- external replay 依赖 `mne.io.read_raw_gdf`
- GPU 相关运行取决于当前 backend 与本机 CUDA 环境

如果你本机已经有统一环境，直接复用即可。若要单独整理环境，优先围绕本目录已有依赖清单和当前可运行环境做，不建议再新开一套互不兼容的环境名。

---

## 12. 测试建议

最基本的回归入口：

```bash
pytest <repo>\02_SSVEP\tests -q
```

如果你刚改的是某条主线，至少再补一轮对应任务的 smoke：

- TDCA：`smoke4`
- FBCCA local：`smoke20` 或最小本地任务
- external replay：`smoke8`

---

## 13. 当前约束与注意事项

1. 新代码不要再 import `_archive/`。
2. 外部 replay 结果默认只用于研究，不直接回流真实在线系统。
3. `all8` 仍是当前主线通道模式，很多 local-opt 任务会显式拒绝其它通道模式。
4. 很多优化链路已经从“只看分类准确率”改成“async-first 排名”，阅读报告时不要再只看单个准确率。
5. UI 是方便入口，不是真理来源；最终请以 run 目录内的 `report.json`、`selection_snapshot.json` 和 profile 文件为准。
6. 服务器写操作必须限制在 `/data1/zkx/brain/ssvep/`；其它服务器路径只能只读检查。
7. 外部短预训练五分类结论以 `best_shared_recipe` 为准，不能把 partial coverage 的 `best_recipe` 当共享最优。

---

## 14. 阅读顺序建议

如果你第一次接手这套代码，建议按下面顺序看：

1. 本文件：整体框架和规则
2. [apps/README.md](./apps/README.md)：先知道每个 UI 干什么
3. [entrypoints/README.md](./entrypoints/README.md)：再知道应该从哪里启动
4. [tools/README.md](./tools/README.md)：需要批跑或服务器时再看
5. [ssvep_core/README.md](./ssvep_core/README.md)：最后进入核心实现

---

## 15. 快速索引

- 顶层启动：`START_SSVEP.py`
- UI 说明：[apps/README.md](./apps/README.md)
- 薄入口说明：[entrypoints/README.md](./entrypoints/README.md)
- CLI / 工具说明：[tools/README.md](./tools/README.md)
- 核心模块说明：[ssvep_core/README.md](./ssvep_core/README.md)
- 文档导航：[docs/README.md](./docs/README.md)
- 外部频率 sweep：`tools/run_external_frequency_server_sweep.py`
- 外部短预训练 benchmark：`tools/run_external_short_pretrain_benchmark.py`
