# 02_SSVEP 专项优化说明（服务器优先训练评测）
更新时间：2026-04-13

本目录只负责 `02_SSVEP/2026-04_async_fbcca_idle_decoder` 内部优化，不涉及主程序集成。

## 1. 目标与边界
- 本地职责：数据采集、在线推理、shadow 验证。
- 服务器职责：训练、评测、模型对比、报告与 profile 产出。
- 安全边界：服务器仅允许操作 `/data1/zkx/brain/ssvep` 子树。

## 2. 当前主策略
- 主线方案：`TDCA + per-frequency gate + evidence accumulator + 5-state machine`。
- 对照方案：`TRCA-R / eTRCA-R`。
- `FBCCA` 仅作为 baseline / fallback，不作为默认主线。
- 旧模型长期保留在训练评测链路中，随时可复评对比。

## 3. 模型选择与排序规则（已收敛）
### 3.1 主对比模型集（默认）
- `tdca,trca_r,etrca_r,fbcca`

### 3.2 统一搜索设置（默认）
- `win_candidates = 2.5,3.0,3.5,4.0`
- `step_sec = 0.25`
- `multi_seed_count = 5`

### 3.3 TDCA 专项网格（已启用）
- `delay_steps = 2,3,4`
- `n_components = 2,3,4`
- 在 quick-screen（decoder-only）和 end-to-end（decoder+gate+decision）均参与搜索。

### 3.4 排序硬约束（已启用）
- 排序前置硬约束：`acc_4class >= 0.80`（不足会被 `low_acc_penalty` 强惩罚）。
- 异步验收阈值保持：
  - `idle_fp_per_min <= 1.5`
  - `control_recall >= 0.75`
  - `switch_latency_s <= 2.8`
  - `release_latency_s <= 1.5`
  - `inference_ms < 40`

## 4. 报告字段（已标准化）
每个模型行（`model_compare_table`）强制包含：
- `acc_4class`
- `macro_f1_4class`
- `idle_fp_per_min`
- `control_recall`
- `switch_latency_s`
- `release_latency_s`
- `inference_ms`
- `compute_backend_used`
- `metrics_source`
- `meets_acceptance`

`metrics_source` 取值固定：
- `cross_session`
- `session1_holdout`
- `no_session2`

并继续输出两套榜单：
- `ranking_boards.end_to_end`（decoder+gate+decision）
- `ranking_boards.classifier_only`（decoder-only）

## 5. 服务器训练策略
### 5.1 GPU 策略
- 默认 `compute_backend=cuda`。
- 服务器预检 GPU 失败时，远端提交会直接失败（严格策略）。
- 兼容模式下，单模型 CUDA 路径不稳定可回退 CPU，并在日志显式记录 `Backend fallback`。

### 5.2 远端目录约束
仅允许以下路径：
- `/data1/zkx/brain/ssvep/code`
- `/data1/zkx/brain/ssvep/data`
- `/data1/zkx/brain/ssvep/reports`
- `/data1/zkx/brain/ssvep/profiles`
- `/data1/zkx/brain/ssvep/logs`
- `/data1/zkx/brain/ssvep/tmp`

### 5.3 本地下载落地
- 运行产物：`profiles/server_runs/<run_id>/`
- 可部署 profile：`profiles/server_profiles/`

## 6. 训练评测入口
- UI：`ssvep_training_evaluation_ui.py`
  - 默认远端模式（remote-first）
  - 本地训练默认禁用，仅高级兜底可启用
- 远端编排：`ssvep_server_train_client.py`
- 训练内核：`ssvep_training_evaluation_cli.py`

## 7. 常用命令
```powershell
# 1) 远端模型对比（建议带 session2）
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

# 2) 查询任务状态
python ssvep_server_train_client.py --action status

# 3) 下载产物
python ssvep_server_train_client.py --action download
```

## 8. 结果解读建议
- 先看 `model_compare_table` 的全量行，不只看第一名。
- 主判据优先级：`idle_fp_per_min`、`wrong_action_rate`、`median_commit_latency`、`control_recall`。
- 若缺失 `session2`，报告会标记 `metrics_source=no_session2`，仅可做内部筛选，不建议上线。

## 9. 测试说明
- 本轮按要求不跑全量长耗时测试。
- 建议最小自检：
  - 关键脚本语法检查
  - 单次小样本远端任务提交流程（submit -> status -> download）
