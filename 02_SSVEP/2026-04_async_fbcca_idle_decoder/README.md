# SSVEP 三程序主线说明（2026-04）

本目录是一套面向异步 SSVEP 的完整工程链路，核心目标是：

1. 看目标时输出目标频率。
2. 不看目标时输出 `None`（不误触发）。
3. 训练、评测、在线运行三件事完全解耦。

当前主线分成三个程序：

- 实时识别：`ssvep_realtime_online_ui.py`
- 数据采集：`ssvep_dataset_collection_ui.py`
- 训练评测：`ssvep_training_evaluation_ui.py`

---

## 1. 对脑电信号到底做了什么（总览）

从工程上，信号路径是固定的：

`设备原始EEG -> 采样窗口 -> 预处理 -> 模型打分 -> 特征提取 -> 异步门控状态机 -> 输出 selected_freq / None`

从数学上，主线是：

- 分类器：FBCCA（可叠加通道权重和 TRCA 空间前端）
- 控制态判决：阈值 + 迟滞 +（可选）动态累计证据
- 评测：论文口径（四分类）+ 异步口径（误触发和时延）双口径并行

---

## 2. 采集阶段（程序 B）对信号做了什么

### 2.1 设备连接与采样稳定

- 使用 BrainFlow 连接设备（支持 `serial_port=auto` 自动选串口）。
- 自动串口策略会优先考虑常见 USB 串口设备描述，并对 `COM4` 这类端口给更高优先级。
- 连接后先 `ensure_stream_ready`，确认缓存样本达到最小稳定量，再正式采集。

### 2.2 Trial 时序（你现在默认）

默认预设是 `stable_12m`：

- `prepare_sec = 1.0`
- `active_sec = 5.0`
- `rest_sec = 4.0`
- `target_repeats = 10`
- `idle_repeats = 20`
- `switch_trials = 14`

每个 trial 在 `active` 开始和结束都会发提示音（采集 UI 已实现）。

### 2.3 只保存 active 段 EEG，并做质量门槛

每个 trial 实际保存的是 active 段 EEG 矩阵 `X ∈ R^{samples × channels}`，并执行质量门槛：

- `MIN_TRIAL_QUALITY_RATIO = 0.90`
- `MAX_TRIAL_RETRIES = 3`
- `MIN_ACTIVE_SEC_FOR_TRAINING = 1.5`

如果 `used_samples / target_samples < 0.9`，该 trial 自动重采，最多 3 次。

### 2.4 采集数据如何落盘

每轮独立 session 目录，固定输出：

- `session_manifest.json`
- `raw_trials.npz`

其中 `NPZ` 保存每个 trial 的原始矩阵，`manifest` 保存索引和协议元数据（label、expected_freq、trial_id、短缺比例、重试次数等）。

新增关键字段：

- `protocol_signature`：对协议核心参数做哈希签名（采样率、prepare/active/rest、target/idle/switch、频率、通道）。
- `quality_summary`：整轮采集质量汇总。

这一步的意义是：训练评测可以严格保证“只用同协议数据”，避免混协议导致结论失真。

---

## 3. 解码阶段对信号做了什么（核心）

核心代码在 `async_fbcca_idle_standalone.py`。

### 3.1 窗口化

在线和离线都把 trial 切成滑动窗：

- 窗长 `win_sec`（常见 1.0~3.0s，配置可搜索）
- 步长 `step_sec=0.25s`

每个窗口独立做一次“预处理 + 打分 + 门控更新”。

### 3.2 预处理（每个窗口）

对每个通道做：

1. 低频基线估计并去除  
`baseline = LPF(x, 3Hz)`，`x' = x - baseline`

2. 工频陷波（若采样率允许）  
默认 50Hz notch：`iirnotch(50Hz, Q=30)`

3. 子带带通滤波  
默认 5 个子带：`(6-50),(10-50),(14-50),(18-50),(22-50) Hz`

### 3.3 FBCCA 打分

对每个候选频率 `f`：

1. 生成参考信号矩阵  
`Y_f = [sin(2πhft), cos(2πhft)]_{h=1..Nh}`，默认 `Nh=3`

2. 在每个子带上做 CCA，取最大典型相关 `ρ_m(f)`（SVD/eigh 白化实现）

3. 子带加权融合  
`score(f) = Σ_m w_m * ρ_m(f)^2`  
其中权重默认 `w_m ∝ (m+1)^(-1.25)+0.25`，再归一化

最终得到四频分数向量 `scores[4]`。

### 3.4 从分数提取门控特征

每窗生成：

- `top1_score`
- `top2_score`
- `margin = top1 - top2`
- `ratio = top1 / top2`
- `normalized_top1`
- `score_entropy`

这些特征进入异步状态机，不直接把 `argmax` 当最终输出。

---

## 4. 异步门控状态机（为什么会“有输出/无输出”）

状态固定三态：

- `idle`
- `candidate`
- `selected`

### 4.1 进入 selected（迟滞进入）

需要同时满足：

- `top1_score >= enter_score_th`
- `ratio >= enter_ratio_th`
- `margin >= enter_margin_th`
- 连续 `min_enter_windows` 个窗口通过

### 4.2 退出 selected（迟滞退出）

当当前目标不再满足保持条件，连续 `min_exit_windows` 个窗口失败后释放到 `idle`，外部输出回到 `None`。

### 4.3 speed 策略的直接切换（A->B）

当 `gate_policy=speed` 时，在 `selected(A)` 可直接切到 `selected(B)`，不必强制 `A->idle->B`：

- `pred_freq != current_selected_freq`
- 达到 `switch_enter_*` 阈值
- 连续 `min_switch_windows` 满足

### 4.4 动态停止（累计证据）

可选 `dynamic_stop`：

- 先估计 control-vs-idle 的对数似然比 `log_lr_t`
- 再做累计证据  
`S_t = alpha * S_{t-1} + log_lr_t`（默认 `alpha=0.7`）
- 进入/退出还可叠加 `S_t` 阈值

这个机制用于减少“单窗抖动”引发的误判。

---

## 5. 可学习权重：通道权重 + 空间滤波权重

### 5.1 通道权重（FBCCA 对角权重）

模式：`channel_weight_mode=fbcca_diag`

做法：

1. 单通道可分性初始化：用 control vs idle 的 `top1_score` 估计 `d'`
2. `softmax(d')` 得初始通道权重
3. 归一化并约束（均值归一，数值裁剪）
4. 在 gate 集上按异步目标函数做坐标搜索微调

### 5.2 空间滤波前端（TRCA shared）

模式：`spatial_filter_mode=trca_shared`（FBCCA 主线）

做法：

1. 每个频率训练 TRCA 空间滤波向量
2. 拼接后做正交化得到共享基 `W`
3. 选 rank `K`（候选 `{1,2,3}`）
4. 前端变换：先通道加权，再空间投影  
`X1 = X * diag(w)`，`X2 = X1 @ W[:, :K]`
5. `X2` 再进入 FBCCA

### 5.3 联合优化（工程近似）

支持交替优化：

- 固定 `W` 优化 `w`
- 固定 `w` 重估 `W`

迭代日志会写入 profile 的 `joint_weight_training` 字段。

---

## 6. 训练评测（程序 C）做了什么

### 6.1 数据读取与筛选

输入是一组 `session_manifest.json`（可在 UI 选择哪些 session 参与训练）。

先做质量过滤：

- `quality_min_sample_ratio`（默认 0.90）
- `quality_max_retry_count`（默认 3）

再做一致性过滤：

- `strict_protocol_consistency`
- `strict_subject_consistency`

### 6.2 新协议优先策略（默认）

默认 `data_policy=new-only`：

- 要求 manifest 必须有 `protocol_signature`
- 且所有训练 session 的 `protocol_signature` 必须一致
- 不一致 session 会被排除并记录到 `excluded_sessions`

### 6.3 trial 级无泄漏切分

评测切分不是窗口随机打散，而是 trial 级分层切分：

- `train : gate : holdout = 60% : 20% : 20%`

并按类别分组后再抽样，保证每类在各 split 都有样本，减少泄漏。

### 6.4 评测双口径并行（重点）

#### Paper lens（对齐论文 6.2.1）

输出：

- `Acc_SSVEP = N_correct / N_total`
- `Macro-F1`
- 混淆矩阵
- 平均决策时间
- `ITR = [log2(N)+Plog2P+(1-P)log2((1-P)/(N-1))] * 60/T`

默认 `decision_time_mode=fixed-window`（论文口径更稳）。

#### Async lens（在线可用性）

输出：

- `idle_fp_per_min`
- `control_recall`
- `control_recall_at_2s / 3s`
- `switch_detect_rate`
- `switch_latency_s`
- `release_latency_s`
- `inference_ms`

`idle_fp_per_min` 的分母是**真实 idle 时长（秒）/60**，不是窗口数近似。

### 6.5 排名与报告

默认 `ranking_policy=async-first`，先看可用性再看分类精度。

报告统一存放到：

`profiles/reports/train_eval/<run_id>/`

固定产物：

- `offline_train_eval.json`
- `offline_train_eval.md`（中文）
- `run.log`
- `selection_snapshot.json`
- `figures/*.png`（若开启）

### 6.6 训练产物如何安全保存

训练评测结束后，会先在本次报告目录内保存候选 profile，再决定是否覆盖默认实时 profile。

核心规则：

- `profile_best_candidate.json`：本轮最佳候选 profile
- `profile_best_fbcca_weighted.json`：本轮最佳加权 FBCCA profile
- `default_profile.json`：只有达到验收阈值时才覆盖

保存采用原子写入：先写临时文件，再 `replace()` 到正式文件，减少半写入导致的损坏风险。

报告 JSON 中还会额外记录：

- `profile_for_realtime_path`
- `profile_for_realtime_type`
- `roundtrip_ready`
- `atomic_write_completed`

---

## 7. 实时在线（程序 A）做了什么

实时程序只做三件事：

1. 读取 profile（含模型状态和门控阈值）
2. 在线滑窗解码
3. 输出状态变化

输出结构包含：

- `state`
- `pred_freq`
- `selected_freq`（最终控制输出）
- `top1_score/top2_score/margin/ratio`
- `control_log_lr/acc_log_lr`
- `stable_windows`
- `decision_latency_ms`

默认只在 `(state, selected_freq)` 变化时发输出，避免刷屏；`--emit-all` 可改为每窗输出。

### 7.1 实时端如何直接调用训练结果

实时端不直接读取训练数据集，只读取训练评测阶段生成的 profile。

标准链路是：

`数据采集 -> 训练评测 -> 生成 profile -> 实时端加载 profile`

实时加载时会做三类校验：

- `model_name/model_params.state` 是否完整
- `channel_weights` 数量是否等于当前 EEG 通道数
- `subband_weights` 数量是否等于当前 filter-bank 子带数

若 `compute_backend=auto`，实时端会先做 CPU/CUDA 小基准，再选择实际后端；若 CUDA 单窗不占优，会自动回退 CPU。

---

## 8. 三程序职责边界（务必遵守）

- 程序 A（实时）：不训练，不写训练报告。
- 程序 B（采集）：只采集并落盘，不做模型排名。
- 程序 C（训练评测）：只吃数据集，不直接连设备采集。

这样能保证结果可复现、问题可定位、工程不会互相污染。

---

## 9. 推荐使用流程（你当前版本）

1. 用程序 B 采 2~4 轮（每轮独立 session）。
2. 在程序 C 勾选要参与训练的 session，跑 train/eval，生成 profile 和报告。
3. 用程序 A 加载 profile 做在线识别。
4. 如果在线速度慢：优先调 gate（`balanced/speed`、`min_enter/min_exit`、`win_sec`），再考虑模型替换。

---

## 10. 常见问题与定位

### 10.1 识别慢（进入/退出都慢）

优先检查：

- `win_sec` 是否过大
- `min_enter_windows` / `min_exit_windows` 是否过大
- gate policy 是否过于保守
- 动态停止阈值是否过高

### 10.2 训练结果很好，在线不好

优先检查：

- 是否混入旧协议数据（看 `protocol_signature`）
- 采集质量是否短段过多（看 `shortfall_ratio`、`retry_count`）
- session2 跨会话结果是否掉得明显（报告里有对照）

### 10.3 一直误触发

优先检查：

- idle 样本是否不足
- `enter_ratio_th` / `enter_margin_th` 是否过低
- speed 策略下 `switch_enter_*` 是否过宽

---

## 11. 主要参数默认值（当前代码）

- `DEFAULT_GATE_POLICY = balanced`
- `DEFAULT_CHANNEL_WEIGHT_MODE = fbcca_diag`
- `DEFAULT_SPATIAL_FILTER_MODE = trca_shared`
- `DEFAULT_DYNAMIC_STOP_ENABLED = True`
- `DEFAULT_DYNAMIC_STOP_ALPHA = 0.7`
- `DEFAULT_NH = 3`
- `DEFAULT_SUBBANDS = (6-50, 10-50, 14-50, 18-50, 22-50)`
- `DEFAULT_STEP_SEC = 0.25`
- 采集质量门槛：`min_sample_ratio=0.90`, `max_retry=3`, `active_sec>=1.5`

---

## 12. 关键文献对齐（实现依据）

- CCA 基线：Bin et al., J Neural Eng 2009, DOI: `10.1088/1741-2560/6/4/046002`
- FBCCA：Chen et al., PLOS ONE 2015, DOI: `10.1371/journal.pone.0140703`
- TRCA：Nakanishi et al., TBME 2018, DOI: `10.1109/TBME.2017.2694818`
- 异步 control-state：DOI `10.1142/S0129065715500306`
- 动态停止：PMID `26736447`, PMID `32731432`
- 伪在线评测框架：DOI `10.1088/1741-2552/ad171a`

当前 `FBCCA + 通道权重 + TRCA共享前端` 属于工程组合方案，报告中按 `engineering-approx` 标注，并通过 A/B 对照证明增益。
