# SSVEP训练评测报告

## 总览

- 生成时间：`2026-04-13T02:10:14`
- 任务模式：`focused-compare`
- 推荐模型族：`legacy_fbcca_202603`
- 部署候选模型：`legacy_fbcca_202603`
- 是否达到验收阈值：`否`
- 默认实时profile是否覆盖：`否`
- 最佳候选profile：`C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\reports\smoke11\20260413\run_20260413_020917_subject001\profile_best_candidate.json`
- 最佳加权FBCCA profile：`C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\reports\smoke11\20260413\run_20260413_020917_subject001\profile_best_fbcca_weighted.json`

## 权重定义

- 通道权重：Eight-channel diagonal EEG weights for all8 FBCCA. The realtime window is transformed as X_weighted[:, c] = X[:, c] * channel_weights[c] before FBCCA scoring; len(w) must match the realtime EEG channel count.
- 子带权重：Five global filter-bank fusion weights shared by all EEG channels. FBCCA combines scores as score[f] = sum_b subband_weights[b] * rho[b, f]^2. These weights are not per-channel weights and they are not learned cutoff frequencies.
- 当前主线：The default trainable FBCCA frontend is separable: channel_weights[8] and subband_weights[5] are learned separately and applied as channel scaling plus global subband-score fusion. A full 8x5 channel-by-subband matrix is intentionally not trained by default to reduce overfitting.
- 空间滤波：Optional TRCA/shared spatial frontend state. It is a separate spatial projection and is not mixed into the pure FBCCA channel/subband-weight mainline unless the chosen profile enables it.

## 数据质量与协议一致性

- Session1：`C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\datasets\subject001_collection_20260411_220406_r01\session_manifest.json`
- Session2：`None`
- 保留trial：`74` / `74`
- 最小样本比例：`0.900`
- 数据策略：`new-only`
- 协议签名：`sha1:304bf885cdd01e18cff32e90563e23b613196a37`

## 6.2.1 SSVEP分类准确率

| 口径 | Acc_SSVEP | Macro-F1 | 平均决策时间(s) | ITR(bits/min) |
|---|---:|---:|---:|---:|
| 四分类 8/10/12/15Hz | 0.9091 | 0.9143 | 1.5000 | 56.6566 |
| 二分类 control vs idle | 0.4667 | 0.4643 | 1.5000 | 0.1283 |

## 6.2.2 异步可用性

| 指标 | 数值 |
|---|---:|
| idle_fp_per_min | 0.0000 |
| control_recall | 0.2727 |
| switch_latency_s | 6.5000 |
| release_latency_s | 1.5000 |
| inference_ms | 26.7726 |

## FBCCA权重增益表

| Rank | Model | 权重策略 | Acc4 | MacroF1_4 | ITR4 | idleFP/min | Recall | 通道权重数 | 子带权重 | Accepted |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 2 | fbcca_fixed_all8 | plain_fbcca_all8 | 0.9091 | 0.9143 | 56.6566 | 0.0000 | 0.2727 | 0 | `[0.386488, 0.207296, 0.155609, 0.131955, 0.118651]` | N |

## 加权提升归因

| 对比 | 可用 | idleFP变化 | Recall变化 | Switch时延变化(s) | 是否可接受提升 |
|---|---|---:|---:|---:|---|
| legacy_fbcca_202603 -> fbcca_plain_all8 | Y | 0.0000 | 0.0000 | 0.0000 | N |
| fbcca_plain_all8 -> fbcca_channel_weighted | N | 0.0000 | 0.0000 | 0.0000 | N |
| fbcca_plain_all8 -> fbcca_subband_weighted | N | 0.0000 | 0.0000 | 0.0000 | N |
| fbcca_plain_all8 -> fbcca_channel_subband_weighted | N | 0.0000 | 0.0000 | 0.0000 | N |
| fbcca_channel_weighted -> fbcca_channel_subband_weighted | N | 0.0000 | 0.0000 | 0.0000 | N |
| fbcca_channel_subband_weighted -> fbcca_channel_subband_trca_shared | N | 0.0000 | 0.0000 | 0.0000 | N |
| balanced -> speed | N | 0.0000 | 0.0000 | 0.0000 | N |

## 全模型识别效果对比

| Rank | Model | Impl | Acc4 | MacroF1_4 | ITR4 | idleFP/min | Recall | Switch(s) | Release(s) | Inference(ms) | Accepted |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | legacy_fbcca_202603 | paper-faithful | 0.9091 | 0.9143 | 56.6566 | 0.0000 | 0.2727 | 6.5000 | 1.5000 | 26.7726 | N |
| 2 | fbcca_fixed_all8 | paper-faithful | 0.9091 | 0.9143 | 56.6566 | 0.0000 | 0.2727 | 6.5000 | 1.5000 | 27.2886 | N |

## 图表文件

- 四分类混淆矩阵：``
- 二分类混淆矩阵：``
- 决策时间直方图：``
- 模型雷达图：``
