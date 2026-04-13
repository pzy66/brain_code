# SSVEP 专项优化工作区（不接主程序）

> 更新时间：2026-04-13  
> 本目录定位：只在 `02_SSVEP/2026-04_async_fbcca_idle_decoder` 内做方法升级与验证，暂不改 `hybrid_controller` 主链行为。

## 1. 目标与边界

本阶段目标：
- 把 SSVEP 管线拆成 `decoder -> gate -> decision -> replay/shadow` 四层。
- 保留旧模型长期可对比（训练/评测链路中始终可见）。
- 在本目录先把异步能力验证透，再考虑回灌主程序。

明确不做：
- 不改主程序默认控制链路。
- 不引入服务器训练。
- 不要求实时链路支持任意模型热切换。

## 2. 当前实现状态（基于代码现状）

### 2.1 已完成

- Decoder 统一接口：
  - `ssvep_core/decoders/base.py` 定义 `BaseDecoder` 与 `DecoderOutput`。
  - 已接入 `FBCCADecoder`、`TRCARDecoder`、`TDCADecoder`（通过 legacy adapter 保持原算法行为）。
- Gate 双轨：
  - `GlobalThresholdGate`（全局阈值基线）。
  - `PerFrequencyLogRegGate`（每频率 logistic，含 `fit/to_payload/from_payload`）。
  - `RollingFeatureHistory`（一致性与 K 窗统计）。
- Decision 层：
  - `EvidenceAccumulator`：`S_t = λ*S_{t-1} + gate_score + β*consistency + prior`。
  - `FiveStateMachine`：`Idle/Candidate/Armed/Commit/Refractory`。
  - `DecisionEngine.step(...)` 统一输出状态、commit 与选中频率。
- Replay 评估：
  - `ReplayEvaluator` 按时间顺序回放，输出轨迹与主指标。
- Shadow Runtime：
  - `ssvep_core/runtime_shadow.py` + `ssvep_realtime_online_ui.py`。
  - 在线可开启 shadow（默认开），完整跑新链路但不改变现有实际输出链。
- 训练评测链路增强：
  - `ModelRegistry` 固定 benchmark 模型组。
  - `BenchmarkSuite` 主指标统一化。
  - `ArtifactStore` 统一产物目录。
  - staged 训练流程增加 `profile_v2`、artifact、replay report、主指标汇总。
- 数据层支持 trial role：
  - `control/clean_idle/hard_idle` 推断与汇总。
  - manifest `quality_summary` 增加 `trial_role_counts`。

### 2.2 当前保留的旧模型对比能力

`ModelRegistry.list_models(task="benchmark")` 固定输出：
- `cca`
- `itcca`
- `ecca`
- `msetcca`
- `fbcca`
- `trca`
- `trca_r`
- `sscor`
- `tdca`
- `etrca_r`

含义：主线升级后，旧模型仍会在训练评测中保留并可复评。

### 2.3 仍在实验演进中的点

- per-frequency logistic gate 训练 API 已实现并有单测；staged 管线当前优先兼容现有 profile 阈值参数并生成 `profile_v2`，你可以在后续将显式 `fit(...)` 接入训练主流程。
- 实时 UI 的 shadow 是“并行观察链路”，默认真实输出仍来自现有兼容链。

## 3. 目录结构（关键）

```text
02_SSVEP/2026-04_async_fbcca_idle_decoder/
  ssvep_core/
    decoders/
      base.py
      fbcca_decoder.py
      trca_r_decoder.py
      tdca_decoder.py
      _legacy_adapter.py
    gating/
      base.py
      global_gate.py
      per_freq_logreg_gate.py
      feature_history.py
    decision/
      accumulator.py
      state_machine.py
      engine.py
    evaluation/
      replay_eval.py
    runtime_shadow.py
    registry.py
    benchmark_suite.py
    artifact_store.py
    profile_v2.py
    _train_eval_staged.py
    train_eval.py
    dataset.py
  ssvep_dataset_collection_ui.py
  ssvep_training_evaluation_ui.py
  ssvep_training_evaluation_cli.py
  ssvep_realtime_online_ui.py
  async_fbcca_idle_standalone.py
  README.md
```

## 4. 分层接口约定

### 4.1 Decoder

- `BaseDecoder.fit(train_trials, fs, freqs, channels)`
- `BaseDecoder.predict_window(window) -> DecoderOutput`

`DecoderOutput` 统一字段：
- `pred_freq`
- `scores`
- `top1_score`
- `top2_score`
- `margin`
- `ratio`
- `entropy`
- `normalized_top1`

### 4.2 Gate

- `BaseGate.predict(feature_row, pred_freq) -> GateOutput`

`GateOutput` 统一字段：
- `p_control`
- `gate_score`
- `pred_freq`
- `gate_name`

### 4.3 Decision

- `DecisionEngine.step(pred_freq, gate_score, consistency, prior=0.0, timestamp_s=None)`

返回：
- `state`
- `commit`
- `selected_freq`
- `stable_windows`
- `refractory_remaining_sec`
- `evidence_score`

### 4.4 Replay

- `ReplayEvaluator.run(stream, labels, config=None) -> replay_report`

主指标字段：
- `idle_false_trigger_per_min`
- `wrong_action_rate`
- `median_commit_latency`

## 5. 数据采集与标签

采集入口：`ssvep_dataset_collection_ui.py`

关键点：
- 默认四目标频率（例如 `8,10,12,15`）。
- 支持 `Long Idle (sec, 0=off)`。
- 保存到 `profiles/datasets/<session_id>/`。
- 产物：
  - `session_manifest.json`
  - `raw_trials.npz`

`trial_role` 规则（`ssvep_core/dataset.py`）：
- `control`：`expected_freq` 非空且非 `switch_to_*`
- `hard_idle`：`switch/transition/scan/long_idle/hard_idle` 等标签
- `clean_idle`：其余 idle/no-control

建议：训练 gate 时显式保留 hard-idle 负类，不要只用 clean-idle。

## 6. 训练评测

### 6.1 UI

入口：`ssvep_training_evaluation_ui.py`

已支持：
- 基线模型组保留开关（`保留基线模型组`）。
- 全模型对比与权重训练任务。
- 统一模型列表来源于 `ModelRegistry`。

### 6.2 CLI（推荐可复现实验）

入口：`ssvep_training_evaluation_cli.py`

常用任务：
- `fbcca-weights`
- `model-compare`
- `fbcca-weighted-compare`
- `focused-compare`
- `classifier-compare`
- `profile-eval`

示例（Windows）：

```powershell
# 1) 全模型对比（含旧模型）
python ssvep_training_evaluation_cli.py \
  --task model-compare \
  --dataset-manifest .\profiles\datasets\<session1>\session_manifest.json \
  --dataset-manifest-session2 .\profiles\datasets\<session2>\session_manifest.json \
  --win-candidates 2.5,3.0,3.5,4.0 \
  --models cca,itcca,ecca,msetcca,fbcca,trca,trca_r,sscor,tdca,etrca_r

# 2) 训练+对比（推荐）
python ssvep_training_evaluation_cli.py \
  --task fbcca-weighted-compare \
  --dataset-manifest .\profiles\datasets\<session1>\session_manifest.json \
  --dataset-manifest-session2 .\profiles\datasets\<session2>\session_manifest.json
```

### 6.3 报告与产物

默认报告目录：`profiles/reports/train_eval/...`

关键输出：
- `offline_train_eval.json`
- `offline_train_eval.md`
- `profile`（v1）
- `<output_profile_stem>_v2.json`（Profile V2）
- `artifacts/`（按 subject/session/date/model/version 组织）

`offline_train_eval.json` 新增重点字段：
- `requested_models`
- `resolved_models`
- `primary_metrics_table`
- `profile_v2_saved`
- `profile_v2_path`
- `artifact_root_path`
- `artifact_write_summary`
- `replay_report`
- `trial_role_counts_session1`

## 7. Profile 双轨

### 7.1 v1

- 兼容旧 runtime 与旧 UI。
- 仍是当前稳定基线。

### 7.2 v2

强制结构（见 `ssvep_core/profile_v2.py`）：
- `decoder`
- `gate.per_freq`
- `evidence`
- `runtime`
- `metrics`

`profile_v2` 用于 SSVEP 专项实验闭环与 shadow/replay；不影响主程序集成。

## 8. 实时在线与 Shadow

入口：`ssvep_realtime_online_ui.py`

新增：
- `--shadow-mode` 参数（UI 默认开启）。
- UI 面板可显示 shadow 状态、commit、p_control。

说明：
- shadow 模式会并行执行新链路（gate+decision），用于线上风险评估。
- 当前默认真实输出链仍保持兼容逻辑，避免影响现有使用。
- `--headless` 分支当前主要走兼容 `OnlineRunner`，shadow 主要在 UI 路径使用。

## 9. 评估口径（建议作为验收门槛）

主指标优先：
- `idle_false_trigger_per_min`
- `wrong_action_rate`
- `median_commit_latency`

建议门槛：
- 相比当前 FBCCA baseline，误触发显著下降。
- 延迟增幅可控（常用目标：+200~500ms 可接受）。
- 状态机稳定，无明显连发与抖动。

## 10. 测试与质量说明

本轮已完成的定向测试：
- `tests/test_gate_decision_replay.py`
- `tests/test_dataset_trial_roles.py`
- `tests/test_realtime_shadow_mode.py`

当前记录：
- 上述定向测试通过（8 passed）。
- 按你的要求，本轮未执行全量回归测试（避免耗时）。

建议你需要发布前再跑一次：

```powershell
python -m pytest -q tests
```

## 11. 与主程序集成关系

当前仓库策略：
- 本目录先完成专项优化与验证。
- 主程序 (`hybrid_controller`) 不在本阶段改行为。

当以下条件满足再考虑回灌：
- 主指标达到门槛。
- shadow 在线日志稳定。
- 对比报告可复现。

## 12. 已知事项

- 部分历史 UI 文件存在中文编码遗留（旧代码问题），不影响本次核心链路功能。
- 如果你要推进到“实时链路默认切到新主线”，建议下一步优先把 per-frequency gate 的显式训练结果稳定落到 `profile_v2.gate.per_freq`，再打开真实 commit 链。

## 13. 服务器训练/评测（GPU 优先）

入口：`ssvep_server_train_client.py`

当前策略：
- 训练与评测任务可全部在服务器发起（包括 `model-compare/focused-compare/classifier-compare/profile-eval`）。
- 默认优先 GPU：`--compute-backend cuda --gpu-device 0 --gpu-precision float32`。
- 支持同时上传 `session1 + session2`，并把 `session2` 作为冻结外测输入给远端 CLI。
- 远端路径强约束：仅允许 `/data1/zkx/brain/ssvep` 子树，且总前缀必须在 `/data1/zkx` 下。

常用命令：

```powershell
# 1) 仅上传 session1/session2 到服务器
python ssvep_server_train_client.py `
  --action upload `
  --dataset-manifest .\profiles\datasets\<session1>\session_manifest.json `
  --dataset-manifest-session2 .\profiles\datasets\<session2>\session_manifest.json

# 2) 服务器上做全模型端到端对比（GPU 优先）
python ssvep_server_train_client.py `
  --action model-compare `
  --dataset-manifest .\profiles\datasets\<session1>\session_manifest.json `
  --dataset-manifest-session2 .\profiles\datasets\<session2>\session_manifest.json `
  --compute-backend cuda `
  --gpu-device 0 `
  --gpu-precision float32 `
  --gpu-warmup 1 `
  --gpu-cache-policy windows `
  --win-candidates 2.5,3.0,3.5,4.0 `
  --multi-seed-count 5

# 3) 查看服务器任务状态与下载结果
python ssvep_server_train_client.py --action status
python ssvep_server_train_client.py --action download
```
