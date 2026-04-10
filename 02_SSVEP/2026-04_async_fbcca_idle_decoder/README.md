# SSVEP 三程序说明（实时识别 / 数据采集 / 训练评测）

本目录已经按职责拆成三套程序，目标是避免“一个大脚本做所有事”带来的混乱。

三套程序的边界固定如下：

- 程序 A：`ssvep_realtime_online_ui.py`
  - 只做在线识别（读 profile，连设备，实时输出）。
- 程序 B：`ssvep_dataset_collection_ui.py`
  - 只做数据采集（按协议落盘数据集）。
- 程序 C：`ssvep_training_evaluation_ui.py`
  - 只做离线训练评测（读已采数据集，输出报告和 profile）。

---

## 1. 推荐使用顺序

1. 先跑程序 B 采集数据（至少 session1，建议 session1+session2）。
2. 再跑程序 C 训练评测，得到推荐模型和 profile。
3. 最后跑程序 A 用 profile 做实时识别验证。

---

## 2. 快速启动（推荐入口）

在目录 `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP` 下运行：

```powershell
# A. 实时识别
python .\START_FBCCA_REALTIME_UI.py --serial-port COM4 --model fbcca

# B. 数据采集（建议采两次：session1/session2）
python .\START_SSVEP_DATA_COLLECTION_UI.py --serial-port COM4 --subject-id subject001 --session-index 1
python .\START_SSVEP_DATA_COLLECTION_UI.py --serial-port COM4 --subject-id subject001 --session-index 2

# C. 训练评测（读取已采数据）
python .\START_MODEL_EVALUATION_UI.py `
  --dataset-manifest .\2026-04_async_fbcca_idle_decoder\profiles\datasets\<session1>\session_manifest.json `
  --dataset-manifest-session2 .\2026-04_async_fbcca_idle_decoder\profiles\datasets\<session2>\session_manifest.json
```

如果你在 PyCharm 里运行，直接运行这三个 `START_*.py` 文件即可。

---

## 3. 程序 A：实时识别

脚本：`ssvep_realtime_online_ui.py`

### 3.1 功能

- 连接 BrainFlow 设备（支持 `auto` 自动串口）。
- 加载训练得到的 profile。
- 允许手动切换模型名（可覆盖 profile 内模型名）。
- 实时输出状态机结果：`idle / candidate / selected`。

### 3.2 输入

- `serial_port`、`board_id`、`freqs`
- `profile`（必须是有效训练产物，不允许 fallback profile）
- `model`（可选覆盖）

### 3.3 输出（在线事件）

- `state`
- `pred_freq`
- `selected_freq`（无输出时为 `None`）
- `top1_score`, `top2_score`, `margin`, `ratio`
- `stable_windows`
- `control_log_lr`, `acc_log_lr`
- `decision_latency_ms`
- `model_name`

### 3.4 典型命令（无 UI）

```powershell
python .\ssvep_realtime_online_ui.py `
  --headless `
  --serial-port COM4 --board-id 0 --freqs 8,10,12,15 `
  --profile .\profiles\default_profile.json `
  --model fbcca
```

---

## 4. 程序 B：数据采集

脚本：`ssvep_dataset_collection_ui.py`

### 4.1 功能

- 按固定协议引导刺激并采样。
- 采集中只做“采样与标注”，不做模型训练。
- 会话结束后保存可复现数据集。

### 4.2 默认协议（`enhanced_45m`）

- 每会话：`4频 × 24` + `idle × 48` + `switch × 32`
- 每 trial 时序：`1s prepare + 4s active + 1s rest`

### 4.3 输出文件

- `session_manifest.json`
- `raw_trials.npz`

默认目录：`profiles/datasets/<session_id>/`

### 4.4 数据字段（Manifest 关键字段）

- `data_schema_version`
- `session_id`, `subject_id`
- `sampling_rate`, `board_eeg_channels`
- `freqs`, `protocol_config`
- `trials[]`（每个 trial 的标签、频率、样本数、npz_key）
- `files.raw_trials_npz`

### 4.5 典型命令（无 UI）

```powershell
python .\ssvep_dataset_collection_ui.py `
  --headless `
  --serial-port COM4 --board-id 0 --freqs 8,10,12,15 `
  --dataset-dir .\profiles\datasets `
  --subject-id subject001 --session-index 1 --protocol enhanced_45m
```

---

## 5. 程序 C：训练评测

脚本：`ssvep_training_evaluation_ui.py`

### 5.1 功能

- 读取程序 B 采集的数据集（不连设备）。
- 跑多模型离线训练、门控拟合、伪在线评测。
- 生成报告（JSON/MD/图表）。
- 如果有模型达标，自动保存 profile。

### 5.2 默认模型池

`cca,itcca,ecca,msetcca,fbcca,trca,trca_r,sscor,tdca,oacca`

### 5.3 默认评测口径

- `metric_scope=dual`：并行输出四分类与 control-vs-idle 二分类指标。
- `decision_time_mode=first-correct`：决策时间按“首个正确输出”定义。
- `ranking_policy=async-first`：排序优先异步可用性。

`async-first` 默认排序键：

`idle_fp_per_min -> control_recall -> switch_latency_s -> release_latency_s -> Acc_4class -> MacroF1_4class -> ITR_4class -> inference_ms`

### 5.4 输出文件

- `offline_train_eval_*.json`
- `offline_train_eval_*.md`
- `figures/confusion_4class.png`
- `figures/confusion_2class.png`
- `figures/decision_time_hist.png`
- `figures/model_radar_async_vs_cls.png`
- `profiles/default_profile.json`（仅在有达标模型时保存）

### 5.5 典型命令（无 UI）

```powershell
python .\ssvep_training_evaluation_ui.py `
  --headless `
  --dataset-manifest .\profiles\datasets\<session1>\session_manifest.json `
  --dataset-manifest-session2 .\profiles\datasets\<session2>\session_manifest.json `
  --metric-scope dual `
  --decision-time-mode first-correct `
  --ranking-policy async-first `
  --export-figures 1 `
  --output-profile .\profiles\default_profile.json `
  --report-path .\profiles\offline_train_eval_manual.json
```

可选参数：

- `--metric-scope`: `dual | 4class | 5class`
- `--decision-time-mode`: `first-correct | first-any | fixed-window`
- `--ranking-policy`: `async-first | paper-first | dual-board`
- `--export-figures`: `1 | 0`

---

## 6. 目录结构（核心）

- `ssvep_realtime_online_ui.py`：程序 A
- `ssvep_dataset_collection_ui.py`：程序 B
- `ssvep_training_evaluation_ui.py`：程序 C
- `ssvep_core/`
  - `dataset.py`：数据集协议和读写
  - `train_eval.py`：离线训练评测主流程
  - `reporting.py`：报告图表导出
- `async_fbcca_idle_standalone.py`：兼容入口（Facade）

---

## 7. 兼容入口（可选）

`async_fbcca_idle_standalone.py` 仍保留旧入口与别名：

- 旧：`calibrate`, `online`, `benchmark`
- 新：`realtime`, `collect`, `train-eval`

示例：

```powershell
python .\async_fbcca_idle_standalone.py collect --serial-port COM4 --subject-id subject001
python .\async_fbcca_idle_standalone.py train-eval --dataset-manifest .\profiles\datasets\<session1>\session_manifest.json
```

---

## 8. 常见问题

### Q1：实时程序报 profile fallback / 无法启动

你传入的 profile 不是训练评测产物。先跑程序 C 生成有效 profile，再启动程序 A。

### Q2：为什么程序 C 不直接连设备采集？

这是刻意的职责分离：采集和训练评测解耦，保证数据可复现、评测可重跑。

### Q3：两个 session 有什么意义？

session1 用于训练+验证+holdout，session2 用于跨会话外测，能更真实评估泛化稳定性。

---

## 9. 最小闭环（建议）

1. 跑一次 `session1` 采集。
2. 直接训练评测拿到报告和 profile。
3. 在线验证“注视输出目标、移开输出 None”。
4. 再补 `session2` 做跨会话复评，确认模型排名是否稳定。
