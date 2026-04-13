# 02_SSVEP 异步专项（服务器优先训练评测）

更新时间：2026-04-13

本目录目标：只优化 `02_SSVEP/2026-04_async_fbcca_idle_decoder`，不改主程序集成逻辑。

## 1. 当前策略

- 本地：只做数据采集 + 在线推理/影子验证。
- 服务器：默认执行训练与评测（GPU 优先）。
- 模型路线：`TDCA + per-frequency gate + evidence + 5-state` 为主线，同时长期保留旧模型可对比。

## 2. 强约束（必须满足）

### 2.1 路径安全

远端读写必须同时满足：

- 在 `/data1/zkx` 前缀内。
- 在 `/data1/zkx/brain/ssvep` 子树内。

已在 `ssvep_server_train_client.py` 做双重校验。

### 2.2 GPU 严格策略

- 默认 `compute_backend=cuda`。
- 当 backend 为 `cuda` 时，提交前必须通过 `nvidia-smi` 预检。
- 预检失败直接报错并终止，不允许静默降级到 CPU。

## 3. 主要入口

- 训练评测 UI：`ssvep_training_evaluation_ui.py`
  - 默认远端模式（remote-first）。
  - 本地模式默认关闭，仅作为高级兜底。
- 服务器编排：`ssvep_server_train_client.py`
- 训练内核 CLI（远端实际执行）：`ssvep_training_evaluation_cli.py`
- 数据采集 UI：`ssvep_dataset_collection_ui.py`
- 在线 UI（含 shadow）：`ssvep_realtime_online_ui.py`

## 4. 训练评测流程（推荐）

1. 本地采集 `session1`（必填）和 `session2`（推荐）。
2. 在训练评测 UI 发起远端任务（默认）。
3. 服务器执行训练/比较并输出报告。
4. 本地轮询任务状态并下载结果。
5. 将下载的 profile 用于本地在线验证。

`session2` 缺失时会弹窗确认；若继续，结果会标记为 `no_session2`（无冻结外测）。

## 5. 服务器目录规范

远端固定目录：

- `/data1/zkx/brain/ssvep/code`
- `/data1/zkx/brain/ssvep/data`
- `/data1/zkx/brain/ssvep/reports`
- `/data1/zkx/brain/ssvep/profiles`
- `/data1/zkx/brain/ssvep/logs`
- `/data1/zkx/brain/ssvep/tmp`

本地下载落地：

- `profiles/server_runs/<run_id>/`
- 可部署 profile 复制到 `profiles/server_profiles/`

## 6. 关键接口（已统一）

### 6.1 server helper CLI

`ssvep_server_train_client.py` 公开参数（重点）：

- `--dataset-manifest`（必填）
- `--dataset-manifest-session2`（可选）
- `--compute-backend {cuda,auto,cpu}`（默认 `cuda`）
- `--gpu-device`（默认 `0`）
- `--gpu-precision {float32,float64}`（默认 `float32`）
- `--gpu-warmup {0,1}`（默认 `1`）
- `--gpu-cache-policy {windows,full}`（默认 `windows`）
- `--win-candidates`（默认 `2.5,3.0,3.5,4.0`）
- `--multi-seed-count`（默认 `5`）

### 6.2 状态响应字段

最小状态字段：

- `run_id`
- `pid`
- `process`
- `progress`
- `artifacts`
- `gpu_device_fuser`
- `log_path`
- `report_dir`
- `tail`

### 6.3 下载响应字段

- `local_run_dir`
- `local_profile`

## 7. 结果元信息（可复现）

任务记录包含：

- `run_id`
- `task`
- `session1`
- `session2`
- `remote_manifest_paths`
- `gpu_params`
- `started_at`

下载后的报告会追加：

- `metrics_source`（`session1_holdout | cross_session | no_session2`）
- `server_gpu_params`
- `remote_path_snapshot`

并生成 `server_run_metadata.json`。

## 8. 模型保留与对比

训练评测链路长期保留旧模型对比，不会因主线升级而移除。基准模型组：

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

## 9. 常用命令

```powershell
# 1) 服务器全模型对比（推荐）
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

# 2) 查看状态
python ssvep_server_train_client.py --action status

# 3) 下载产物
python ssvep_server_train_client.py --action download
```

## 10. 测试说明

本轮只做了定向测试与编译检查，未跑全量测试（按你的要求避免耗时）：

- `tests/test_server_train_client_gpu_and_paths.py`
- `tests/test_server_train_client_cuda_policy.py`
- `tests/test_train_eval_ui_remote_defaults.py`
- `py_compile`：`ssvep_server_train_client.py`、`ssvep_training_evaluation_ui.py`

如果后续需要发布，再执行全量回归：

```powershell
python -m pytest -q tests
```
