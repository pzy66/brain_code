# 异步 SSVEP 解码系统（4 目标 + Idle）

本目录是一套独立的异步 SSVEP 链路，目标是实现：

- 注视某个闪烁目标时，稳定输出该目标频率；
- 不注视任何目标（Idle）时，输出 `None`（不触发控制）。

在不破坏旧代码的前提下，新增了两个独立 UI 程序：

- `ssvep_model_evaluation_ui.py`：多模型评测 UI
- `ssvep_single_model_ui.py`：单模型预训练 + 在线识别 UI

底层算法与流程统一复用：

- `async_fbcca_idle_standalone.py`

## 1. 文件说明

- `async_fbcca_idle_standalone.py`
  - 核心算法与 CLI 入口；
  - 提供 `calibrate` / `benchmark` / `online` 三个子命令；
  - 包含模型工厂、异步门控、阈值拟合、评测指标计算。
- `ssvep_model_evaluation_ui.py`
  - 固定协议自动完成：连接设备 -> 校准采集 -> 评测采集 -> 多模型排序 -> 报告输出。
  - 评测时同步保存原始 trial 数据集（`NPZ + JSON 索引`）。
- `ssvep_single_model_ui.py`
  - 先选择一个模型做预训练，再进入在线识别；
  - 适合日常调参与在线验证。
- `async_fbcca_validation_ui.py`
  - 旧验证 UI（保留可用）。
- `profiles/default_profile.json`
  - Benchmark 推荐模型输出的默认 profile。
- `profiles/single_model_profile.json`
  - 单模型 UI 默认 profile。

## 2. 运行环境

- Python 3.10+（当前环境已验证 Python 3.11）
- BrainFlow 兼容设备与驱动
- PyQt5
- NumPy / SciPy / scikit-learn

推荐解释器：

- `C:\Users\P1233\miniconda3\envs\brain-vision\python.exe`

## 3. 快速启动（两个入口）

在目录 `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP` 下执行：

```powershell
# 单模型预训练 + 在线识别 UI（默认 FBCCA）
python .\START_FBCCA_REALTIME_UI.py --serial-port auto --board-id 0 --freqs 8,10,12,15

# 多模型评测 UI
python .\START_MODEL_EVALUATION_UI.py --serial-port auto --board-id 0 --freqs 8,10,12,15
```

如果设备固定在 COM4，可改为 `--serial-port COM4`。

## 4. 程序 A：多模型评测 UI（`eval-ui`）

脚本：

- `ssvep_model_evaluation_ui.py`

### 4.1 按钮功能

- `连接设备`：检测串口与采样流是否正常；
- `开始评测`：按固定协议自动跑完整评测；
- `停止`：中断当前评测任务；
- `导出报告`：导出 Markdown 报告副本。

评测 UI 关键参数：

- `Channel Modes`：`auto`（自动选通道）/ `all8`（固定 8 通道）；
- `Multi-seed Count`：同一批采集数据做多随机种子 trial 级重分割，输出稳健性排名。

### 4.2 固定评测协议

- 校准段：
  - 4 目标 × 24 trials
  - idle × 48 trials
- 评测段：
  - 4 目标 × 24 trials
  - idle × 48 trials
  - A<->B 切换任务 × 32 trials
- 窗口参数搜索：
  - `win_sec in {1.5, 2.0, 2.5, 3.0}`
  - `step_sec = 0.25`
  - `min_enter_windows in {1,2}`
  - `min_exit_windows in {1,2,3}`
- 门控策略：
  - 默认 `gate_policy=balanced`
  - 默认启用动态停止（累计证据）与 `alpha=0.7`
- 通道权重：
  - 默认 `channel_weight_mode=fbcca_diag`（仅 FBCCA 学习 8 通道对角权重）
- 伪在线连续性增强：
  - 评测阶段额外采集并回放 `Rest + Next Prepare` 形成的 `transition_idle` 段；
  - 这些段按 `idle` 类参与门控评估，用于更真实地评估释放与误触发。

### 4.3 默认模型池

- `cca,itcca,ecca,msetcca,fbcca,trca,trca_r,sscor,tdca,oacca`

### 4.4 排序优先级（严格）

为避免“0 误触发但 0 召回”被错误排第一，当前采用“约束 + 字典序”：

1. `control_recall >= 0.10` 的模型优先（低于该下限会先被降级）
2. `idle_fp_per_min`（越低越好）
3. `control_recall`（越高越好）
4. `switch_detect_rate`（越高越好）
5. `switch_latency_s`（越低越好）
6. `release_latency_s`（越低越好）
7. `itr_bpm`（越高越好）
8. `inference_ms`（越低越好）

### 4.5 输出产物

- `benchmark_report_YYYYMMDD_HHMMSS.json`
- `benchmark_report_YYYYMMDD_HHMMSS.md`
- `profiles/datasets/benchmark_session_YYYYMMDD_HHMMSS/session_manifest.json`
- `profiles/datasets/benchmark_session_YYYYMMDD_HHMMSS/raw_trials.npz`
  - `trials[].label` 可能包含 `transition_idle` / `transition_idle_tail`

JSON 关键字段：

- `model_results`
- `chosen_model`
- `chosen_metrics`
- `chosen_rank`
- `chosen_meets_acceptance`
- `dataset_dir`
- `dataset_manifest`
- `dataset_npz`
- `data_schema_version`
- `metric_definition`（`switch_latency_s` 使用惩罚版定义）
- `metric_definition`（`switch_latency_s` 与 `release_latency_s` 都是惩罚版定义）
- `robustness`（按 `channel_mode` + `seed` 的多次复评结果与聚合统计）
- `robust_recommendation`（稳健性推荐模型）

## 5. 程序 B：单模型预训练 + 在线识别 UI（`single-ui`）

脚本：

- `ssvep_single_model_ui.py`

### 5.1 按钮功能

- `连接设备`
- `开始预训练`
- `开始在线`
- `停止`
- `加载profile`
- `保存profile`

### 5.2 使用流程

1. 点击 `连接设备`；
2. 选择模型（默认 `fbcca`）；
3. 点击 `开始预训练`；
4. 预训练完成后点击 `开始在线`；
5. 观察输出：
   - 注视目标 -> 输出目标频率；
   - Idle -> 输出 `None`。

### 5.3 关键行为

- 预训练不再硬编码 FBCCA，而是按当前所选模型拟合；
- 预训练内部采用 trial 级 `fit/gate/holdout` 分离：
  - `fit` 用于模型状态拟合；
  - `gate` 用于门控阈值拟合；
  - `holdout` 用于质量摘要（若样本不足则回退到 gate）；
- 通道选择仅在 `outer fit` 子集上执行，避免把 holdout 信息泄漏到预训练调参。
- profile 会写入：
  - `model_name`
  - `model_params.state`
  - 门控阈值与窗口参数
- 在线阶段严格按已加载 profile 解码。
- 若 UI 下拉模型与 profile 内模型不一致，系统会自动以 profile 模型为准并在日志提示。
- 单模型 UI 支持直接设置门控延迟参数：
  - `Window(s)`、`Step(s)`、`Enter windows`、`Exit windows`；
  - 默认低延迟配置为 `2.0s / 0.10s / 1 / 1`（比旧的 `3.0s / 0.25s / 2 / 2` 更快）。

### 5.4 识别慢时的建议调参

- 先重新预训练，不要沿用旧 profile（旧 profile 常见是 `win_sec=3.0`）。
- 优先尝试：
  - `Window(s)=2.0`
  - `Step(s)=0.10`
  - `Enter windows=1`
  - `Exit windows=1`
- 若误触发增加，再逐步调回保守值：
  - 先把 `Enter windows` 提到 `2`；
  - 再把 `Window(s)` 提到 `2.5`。

## 6. 核心 CLI（直接使用）

### 6.1 多模型评测

```powershell
python .\async_fbcca_idle_standalone.py benchmark `
  --serial-port COM4 --board-id 0 `
  --freqs 8,10,12,15 `
  --models cca,itcca,ecca,msetcca,fbcca,trca,trca_r,sscor,tdca,oacca `
  --output-profile .\profiles\default_profile.json `
  --report-path .\profiles\benchmark_report_manual.json `
  --dataset-dir .\profiles\datasets
```

### 6.2 单模型校准

```powershell
python .\async_fbcca_idle_standalone.py calibrate `
  --serial-port COM4 --board-id 0 `
  --freqs 8,10,12,15 `
  --output .\profiles\single_model_profile.json
```

### 6.3 在线识别

```powershell
python .\async_fbcca_idle_standalone.py online `
  --serial-port COM4 --board-id 0 `
  --profile .\profiles\single_model_profile.json
```

## 7. 在线输出结构

在线循环输出标准结构：

- `state`: `idle | candidate | selected`
- `pred_freq`
- `selected_freq`（Idle 时为 `null`）
- `top1_score`
- `top2_score`
- `margin`
- `ratio`
- `stable_windows`
- `model_name`
- `decision_latency_ms`

## 8. Profile 关键字段

- `freqs`
- `win_sec`
- `step_sec`
- `enter_score_th`
- `enter_ratio_th`
- `enter_margin_th`
- `exit_score_th`
- `exit_ratio_th`
- `min_enter_windows`
- `min_exit_windows`
- `model_name`
- `model_params`（含需持久化的模型状态）
- `gate_policy`
- `dynamic_stop`
- `channel_weight_mode`
- `channel_weights`
- `benchmark_metrics`（可选）

## 9. 全流程验收建议

每次改动后建议按以下顺序验收：

1. 设备连通性
   - 打开 UI，点击 `连接设备`；
   - 日志中确认采样率与缓冲样本正常。
2. 评测流程
   - 在 `eval-ui` 完整跑 1 轮；
   - 确认生成 JSON + Markdown 报告 + 数据集目录（manifest + npz）。
3. 单模型在线流程
   - 在 `single-ui` 完成预训练并进入在线；
   - 验证三点：
     - 持续注视时能稳定锁频；
     - 长 idle 能回到 `selected_freq=None`；
     - A->B 切换不出现长时间粘连旧目标。

## 10. 常见问题排查

### 10.1 串口打不开 / 刚启动就断

- 关闭所有占用 COM 的程序（旧脚本、串口助手、IDE 残留调试）；
- 先试 `--serial-port auto`，再试手动固定 `COM4`；
- 重新插拔设备后重试。

### 10.2 提示缓冲样本不足

- 刚启动流后等待 1-2 秒；
- 降低后台负载；
- 避免不稳定 USB 集线器。

### 10.3 Idle 误触发偏多

- 重新跑 benchmark（包含充足 idle 与切换任务）；
- idle 段严格注视中心，不扫视刺激块；
- 优先选择 `idle_fp_per_min` 更低的模型与参数。

### 10.4 加载到 fallback profile

- 说明当前未使用真实校准文件；
- 先执行 `calibrate` 或 `benchmark` 生成 profile；
- 检查 profile 路径是否指向最新文件。

## 11. 后续优化顺序（建议）

若在线效果仍不达标，按以下顺序调优：

1. 窗口候选集合（`win_sec`）
2. 进入/退出阈值（`enter_*` / `exit_*`）
3. 进入/退出最小稳定窗数
4. 仅保留 benchmark 前几名模型继续细调

## 12. 方法对照文档

- `METHOD_ALIGNMENT_AND_OPTIMIZATION_MATRIX.md`
  - 给出每个模型的实现级别（`paper-faithful` / `engineering-approx`）；
  - 汇总本轮问题优化矩阵与论文依据；
  - 明确 `switch_latency_s` 的惩罚版定义。
