# ssvep_core 说明

`ssvep_core/` 是 02_SSVEP 工程的核心实现层。这里负责四类事情：

1. 把原始 EEG 或外部数据切成可评分的 trial / window。
2. 用 decoder 产出每个窗的频率证据。
3. 用 gate / confidence / decision 把单窗分数变成异步事件。
4. 把训练、调参、回放、报告和 profile 组织成可复现的 run。

这层不只是“模型代码”。当前主线里真正决定 async 表现的，是 `decoder -> gate -> correctness -> decision -> replay/report` 这一整条链路。

---

## 1. 目录和职责

### 数据入口
- `dataset.py`
  - 本地 4 目标采集数据的主 schema。
  - 负责 manifest、session bundle、trial segment 等本地数据协议。
- `external_replay_dataset.py`
  - 外部 GDF 数据入口。
  - 当前 external replay 固定使用 `13/17/21 Hz + rest` 和 8 通道 montage：
    `Oz, O1, O2, PO3, POz, PO7, PO8, PO4`
- `trial_roles.py`
  - 统一解析 `control / clean_idle / hard_idle`。
  - gate、confidence、error attribution、报告语义都依赖它，不能各自做一套 fallback。

### 解码器
- `async_fbcca_idle_standalone.py`
  - 共享 decoder 工厂、profile 工具、benchmark 工具。
  - 很多 local-opt / replay 都通过这里构造 decoder。
- `decoders/`
  - 解码器实现层。
  - 当前主线实际会用到 FBCCA、TDCA、TRCA 相关实现。

### gate / confidence / decision
- `gating/`
  - `PerFrequencyLogRegGate`：一阶 gate scorer。
  - `CorrectnessCalibrator`：把 gate 分数和 gap/entropy 等特征映射成 `p_correct / correctness_logit`。
- `decision/`
  - `DecisionEngine` 负责证据累积和状态机。
  - 当前 external replay 与 local-opt 都用它做 fixed-window async replay。

### 训练 / 调参 / 回放主线
- `tdca_local_opt.py`
  - 本地主线 TDCA 搜索与报告。
- `fbcca_local_opt.py`
  - 本地主线 FBCCA 搜索与报告。
- `fbcca_external_replay_opt.py`
  - 外部 8 通道 session-based replay 主线。
  - 当前 external 数据方法开发的主入口。

### 产物与 profile
- `profile_v2.py`
  - `profile_v2` schema 和导出逻辑。
- `run_artifacts.py`
  - run 目录、artifact alias、canonical 路径规则。

---

## 2. 当前 external replay 主线到底在做什么

当前 external 主线任务是：

- task 名：`fbcca-external-replay-opt`
- 输入：外部 GDF 数据集
- 评估口径：within-subject、session-based
- 主要视图：
  - `loso4`：4 折 leave-one-session-out 聚合
  - `chronological_last_session`：前 3 个 session 训练，第 4 个 session 回放
- 输出：run 目录、report、selection snapshot、simulation-only profile

这条链路的目标不是直接替代真实在线系统，而是模拟：

`历史 session 训练/调参 -> 新 session 固定窗异步回放`

因此它的 profile 永远带：

- `simulation_only_profile = true`
- 不覆盖真实在线 deployed profile

---

## 3. External replay 的关键语义

### 3.1 outer-train / holdout 的含义

对每个 outer fold：

- `train sessions`
  - 用来训练 decoder
  - 用来训练 gate
  - 用来做 OOF correctness calibration
  - 用来做 tune 搜索
- `holdout session`
  - 只用于最终 continuous replay
  - 不允许参与 frontend、gate、confidence、threshold、decision 参数训练

### 3.2 为什么 tune rows 必须是 OOF-only

当前 external replay 没有独立 gate session，因此 `decision_search_target="tune_split"` 的准确含义是：

- `outer-train sessions` 内部的 OOF rows

这里**不能**混入 `train_full` in-sample rows。原因很直接：

1. `train_full` 会高估 gate / calibrator 的可分性。
2. 它会把 per-frequency reference 推向过于乐观的方向。
3. tune Brier/AUC、tune frequency breakdown、threshold search 会失真。

当前实现已经固定：

- `scored_tune_rows` 只允许 `tune_origin=train_oof`
- 任何 tune-derived board 出现 `train_full` 都视为实现错误

### 3.3 strict selection 和 diagnostic row 的区别

external replay 现在同时保留两层结果：

#### strict selection
用于真正“可不可以作为当前主线候选”的判断。

依赖：
- `selection_eligible == true`
- 并且该候选不是 `diagnostic_only`

当前 strict validity 包含：
- `frequency_balance_valid`
- `confidence_dominance_valid`

#### diagnostic best row
用于回答：

- 当前最接近可用的是谁
- 它为什么还没过 strict gate
- 是频点失衡、confidence 过严，还是 decision 卡住

即使本轮 `status=invalid`，也必须有可读的 `diagnostic_best_row` 和配套诊断板。

---

## 4. per-frequency reference 现在怎么做

external replay 当前保留两条全局 confidence 路线：

- `global_correctness_logistic`
- `bayesian_gap_gmm`

它们负责产出全局可比较的：

- `p_correct`
- `correctness_logit`

在此之上，external replay 允许做一层**低自由度**的 per-frequency enter-reference 修正，目的不是训练频点独立 classifier，而是避免不同频点共用同一证据零点。

当前规则：

1. 只在 OOF tune rows 上估计。
2. 只修 `enter_reference`。
3. `exit_reference` 继续固定为全局 `exit_p_th`。
4. 频点 reference 必须受 trial-level 分布约束：
   - 先按 `trial_id` 聚合 `max p_correct`
   - 再用正样本 trial 的 `p50 / p75` 和负样本 trial 的 `p90` 做诊断
5. `enter_reference` 不允许高于该频点正样本 trial 的合理上沿。

这一步的目标是修“raw 基本正确，但 gate / decision 尺度把某个频点压死”的问题，尤其是当前 external 数据里出现过的 `13 Hz raw 对但 gate 过弱` 情况。

相关输出：
- `per_frequency_enter_reference`
- `reference_diagnostics_board`
- `reference_headroom_p50`

如果 `reference_headroom_p50` 长期为负，通常说明这个频点的 enter reference 已经高于正确 trial 的典型峰值，需要先修 calibration，而不是先扩搜索网格。

---

## 5. 报告里几个容易混淆的板子

### `fbcca_search_board`
- 候选级摘要板。
- 看每个 decoder/confidence/window 组合的 tune-side async 表现和 gate 有效性。

### `decision_search_board`
- 只表示 `tune_split` 上 decision 参数搜索结果。
- 不是最终选型板。

### `holdout_selection_board`
- 最终 holdout 聚合结果。
- strict selection 和 diagnostic best row 都从这里派生。

### `replay_frequency_breakdown`
- 按 `13/17/21 Hz` 拆开看：
  - `raw_correct_rate`
  - `gate_pass_rate`
  - `commit_rate`
  - `release_seen_rate`
  - `median_max_p_correct`
  - `median_max_decision_evidence`

如果总体 `control_recall` 还能看，但某个频点 `gate_pass_rate` 接近 0，这里会第一时间暴露问题。

### `decision_bottleneck_summary`
- 用于判断主瓶颈在哪一层：
  - `decoder_miss`
  - `confidence_reject_miss`
  - `decision_miss`

### `reference_diagnostics_board`
- 用来判断 enter reference 是否已经压过该频点正确 trial 的典型 `p_correct`。
- 当前 external replay 的核心诊断板之一。

---

## 6. decoder family 的当前策略

external replay 任务当前会同时跑：

- `fbcca_fixed_all8`
- `fbcca_sw_all8`
- `itcca_all8`
- `ecca_all8`

但主线含义不同：

### 主线候选
- `fbcca_fixed_all8`
- `fbcca_sw_all8`

它们允许参与 strict selection 和 profile promotion。

### 诊断候选
- `itcca_all8`
- `ecca_all8`

当前仓库中这两条仍属于 `engineering-approx`，模板构建口径也还是简化版 last-window 近似，因此：

- 会保留在 board 中做对照
- 但标记为 `diagnostic_only=true`
- 不进入最终 strict 选型

---

## 7. profile 规则

external replay 产出的 profile 只用于模拟和研究：

- 会生成 `profile.json` 和 `profile_v2.json`
- metadata 中固定带 `simulation_only_profile=true`
- 不覆盖真实在线主 profile
- 不修改 realtime UI 当前正在读取的 deployed profile 逻辑

如果你在 run 目录中看到 profile 被写出，不代表它已经是可上线 profile，只代表这次 run 的最佳 strict 候选被序列化了。

---

## 8. 阅读和维护建议

如果你现在要继续调 external replay，推荐按这个顺序看：

1. `fbcca_external_replay_opt.py`
2. `gating/correctness_calibrator.py`
3. `decision/engine.py`
4. 当前 run 的 `report.json`
5. 本目录 README 的 external replay 语义说明

判断优先级时，先看：

1. `replay_frequency_breakdown`
2. `decision_bottleneck_summary`
3. `reference_diagnostics_board`

而不是先看总体 `control_recall` 一行数字。

如果 decoder 已经接近全对，但 `confidence_reject_miss` 仍占大头，下一轮应该继续修 calibration correctness，不应直接扩大模型网格或引入 dynamic stopping。
