# 已采数据离线评测中文结论报告

## 1. 报告信息
- 结果文件：`offline_eval_after_upgrade_20260410.json`
- 评测时间：2026-04-10
- 数据会话：`benchmark_session_20260409_220059`
- 采样率：250 Hz
- 通道：8 通道（`[1,2,3,4,5,6,7,8]`）
- 目标频率：`8/10/12/15 Hz`
- 鲁棒性评测：`channel_mode = auto + all8`，`5` 个 seed（20260407~20260411）

## 2. 数据概况
- 总 trial 数：157
- 校准段：30
- 评测段：127
- 评测包含：目标注视、idle、目标切换（A->B）与切换后 idle 片段

## 3. 排序结果（核心指标）

### 3.1 `auto` 通道模式（综合推荐来源）
前 3 名（按既定排序规则）：

1. `cca`  
   - `idle_fp_per_min = 3.40`  
   - `control_recall = 0.58`  
   - `switch_latency_s = 4.325`  
   - `inference_ms = 1.589`
2. `fbcca`  
   - `idle_fp_per_min = 28.00`  
   - `control_recall = 0.60`  
   - `switch_latency_s = 4.225`  
   - `inference_ms = 3.676`
3. `tdca`  
   - `idle_fp_per_min = 0.00`  
   - `control_recall = 0.04`  
   - `switch_latency_s = 6.000`  
   - `inference_ms = 5.746`

### 3.2 `all8` 通道模式
前 3 名（按既定排序规则）：

1. `msetcca`  
   - `idle_fp_per_min = 0.00`  
   - `control_recall = 0.10`  
   - `switch_latency_s = 6.000`  
   - `inference_ms = 1.937`
2. `ecca`  
   - `idle_fp_per_min = 2.40`  
   - `control_recall = 0.18`  
   - `switch_latency_s = 6.000`  
   - `inference_ms = 2.338`
3. `trca_r`  
   - `idle_fp_per_min = 4.00`  
   - `control_recall = 0.10`  
   - `switch_latency_s = 6.000`  
   - `inference_ms = 7.059`

## 4. 中文结论（直接可执行）

1. 这次离线评测可以用来做“相对比较”，但**不能直接作为最终上线依据**。  
   原因：100 个鲁棒性运行里，`meets_acceptance = 0/100`，没有模型达到当前验收门槛。

2. 按你现在定义的排序优先级（`idle_fp_per_min` 优先），“鲁棒推荐”是：  
   - `auto` 模式下的 `cca`（当前系统给出的第一推荐）

3. 从可用性角度看，当前评测呈现明显权衡：  
   - `cca/fbcca`：召回相对高一些，但 idle 误触发偏高（尤其 `fbcca`）  
   - `trca/tdca/msetcca` 等：idle 很低，但召回过低，容易“几乎不输出”

4. `all8` 相比 `auto` 没有显示稳定优势。  
   在本批数据上，`all8` 的高排名模型普遍召回偏低，说明“通道全开”并未自动带来更好控制性能。

## 5. 对你当前场景的建议

1. 单模型在线实用优先：继续保留你可稳定使用的 `FBCCA` 作为在线主线。  
2. 评测阶段单独优化门控阈值与时延目标，避免“低误触发但几乎不输出”的模型被排到前面。  
3. 下一轮数据采集增加“短窗口快速切换”任务比例，用于专门压低进入/退出时延。

---

本报告基于以下文件生成：  
- `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\offline_eval_after_upgrade_20260410.json`  
- `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\datasets\benchmark_session_20260409_220059\session_manifest.json`  
- `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\datasets\benchmark_session_20260409_220059\raw_trials.npz`
