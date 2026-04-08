# Async SSVEP 多模型异步识别（4 目标 + idle）

本目录是独立的 SSVEP 异步识别链路，目标是解决：

- 看目标时稳定输出 `selected_freq`
- 不看任何目标时输出 `None`
- 统一比较 FBCCA 之外的核心分类器

主程序：

- `async_fbcca_idle_standalone.py`

配套 UI：

- `async_fbcca_validation_ui.py`

UI 关键行为：

- `Full Workflow`：先做校准（当前为 FBCCA 校准链路）再进入在线验证
- `Validate Existing Profile`：按 profile 内 `model_name` 加载对应解码器（可使用 benchmark 选出的非 FBCCA 模型）

## 1. 功能概览

`async_fbcca_idle_standalone.py` 提供 3 个子命令：

- `calibrate`：仅做校准并拟合门控阈值（单模型，默认 FBCCA）
- `benchmark`：采集校准 + 评测数据，批量比较多模型，保存最佳 profile
- `online`：按 profile 做实时异步识别

统一策略：

- 所有模型统一输出 `scores[4]`
- 统一提特征：`top1/top2/margin/ratio/entropy`
- 统一异步门控状态机：`idle -> candidate -> selected`
- 统一对外输出：`selected_freq` 或 `None`

## 2. 当前支持模型

- CCA 家族：`cca`, `itcca`, `ecca`, `msetcca`, `fbcca`
- 空间滤波家族：`trca`, `trca_r`, `sscor`, `tdca`
- 在线自适应：`oacca`

说明：这里是工程可运行实现，用于你当前设备在线对比与筛选，不是逐字复刻某单篇论文代码。

## 3. 快速运行（推荐）

在 `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP` 下：

1) 多模型 benchmark（COM4）：

- `START_ASYNC_SSVEP_BENCHMARK_COM4.cmd`

2) 在线识别（COM4，使用默认 profile 路径）：

- `START_ASYNC_SSVEP_ONLINE_COM4.cmd`

3) 视觉校验 UI（COM4）：

- `START_ASYNC_SSVEP_COM4.cmd`

## 4. 完整工作流（建议顺序）

1. 启动你的闪烁刺激界面（四箭头 8/10/12/15Hz）。
2. 运行 `benchmark`，程序会自动做两段采集：
- 校准段：`4目标 x 5 + idle x 10`
- 评测段：`4目标 x 8 + idle x 16 + A<->B 切换任务`
3. 程序按 trial 级切分 `60/20/20`（train/gate/holdout），搜索窗口参数和门控阈值。
4. 对候选模型统一评分，按以下优先级排序：
- `idle_fp_per_min`（越小越好）
- `control_recall`（越大越好）
- `switch_latency_s`（越小越好）
- `itr_bpm`（越大越好）
- `inference_ms`（越小越好）
5. 输出：
- 最优 profile（默认 `profiles/default_profile.json`）
- benchmark 报告（`benchmark_report_*.json`）
6. 运行 `online`，用该 profile 进行实时识别。

## 5. CLI 用法

### 5.1 benchmark

```powershell
python async_fbcca_idle_standalone.py benchmark `
  --serial-port COM4 --board-id 0 `
  --output-profile .\profiles\default_profile.json `
  --models cca,itcca,ecca,msetcca,fbcca,trca,trca_r,sscor,tdca `
  --win-candidates 2.0,2.5,3.0
```

### 5.2 online

```powershell
python async_fbcca_idle_standalone.py online `
  --serial-port COM4 --board-id 0 `
  --profile .\profiles\default_profile.json
```

可选：

- `--model <name>`：临时覆盖 profile 内模型
- `--emit-all`：每个窗口都输出，不仅状态变化时输出

### 5.3 calibrate

```powershell
python async_fbcca_idle_standalone.py calibrate `
  --serial-port COM4 --board-id 0 `
  --output .\profiles\default_profile.json
```

## 6. 在线输出字段

每次状态变化会输出一条 JSON，关键字段：

- `state`: `idle|candidate|selected`
- `pred_freq`
- `selected_freq`（空闲时为 `null`）
- `top1_score`, `top2_score`, `margin`, `ratio`
- `stable_windows`
- `model_name`
- `decision_latency_ms`

## 7. Profile 关键字段

- `freqs`, `win_sec`, `step_sec`
- `enter_*`, `exit_*`, `min_enter_windows`, `min_exit_windows`
- `model_name`
- `model_params`（包含已拟合模型状态）
- `calibration_split_seed`
- `benchmark_metrics`

注意：`online` 对某些模型（如 `trca/tdca/oacca`）要求 profile 里存在已拟合 `model_params.state`。

## 8. 常见问题

1) 串口打不开 / 一开始就断：

- 检查 COM 口是否被其他进程占用（串口助手、旧脚本、IDE 调试残留）
- 先断开并重连设备，再重启脚本

2) 提示 buffered samples 不足：

- 设备刚启动时常见，等待 1-2 秒
- 降低 USB 丢包风险，避免同时跑高负载程序

3) 误触发多：

- 先重新做 benchmark（包含 idle 和切换段）
- 提高 `idle` 采集质量（看中心，避免扫视目标）

4) profile 是 fallback：

- 说明你在没有校准文件的情况下启动了 online
- 先跑 `benchmark` 或 `calibrate` 生成 profile

## 9. 迭代建议（继续优化）

每轮按固定闭环执行：

1. 跑一次 benchmark，保存报告。
2. 用新 profile 跑 8 分钟在线人工验证。
3. 记录 3 类失败片段：误触发、粘连、切换慢。
4. 针对失败片段再调：
- 窗口候选（2.0/2.5/3.0）
- 门控最小窗口数（enter/exit）
- 模型候选集合（先保留前 3 名）

这样能把“空闲不误触”优先落地，而不是只看离线准确率。
