# SSVEP训练评测报告

## 总览

- 生成时间：`2026-04-13T22:42:32`
- 任务模式：`model-compare`
- 推荐模型族：`msetcca`
- 部署候选模型：`msetcca`
- 是否达到验收阈值：`否`
- 默认实时profile是否覆盖：`否`
- 最佳候选profile：`/data1/zkx/brain/ssvep/reports/20260413/model_compare_20260413_223538/profile_best_candidate.json`
- 最佳加权FBCCA profile：`/data1/zkx/brain/ssvep/reports/20260413/model_compare_20260413_223538/profile_best_fbcca_weighted.json`

## 权重定义

- 通道权重：Eight-channel diagonal EEG weights for all8 FBCCA. The realtime window is transformed as X_weighted[:, c] = X[:, c] * channel_weights[c] before FBCCA scoring; len(w) must match the realtime EEG channel count.
- 子带权重：Five global filter-bank fusion weights shared by all EEG channels. FBCCA combines scores as score[f] = sum_b subband_weights[b] * rho[b, f]^2. These weights are not per-channel weights and they are not learned cutoff frequencies.
- 当前主线：The default trainable FBCCA frontend is separable: channel_weights[8] and subband_weights[5] are learned separately and applied as channel scaling plus global subband-score fusion. A full 8x5 channel-by-subband matrix is intentionally not trained by default to reduce overfitting.
- 空间滤波：Optional TRCA/shared spatial frontend state. It is a separate spatial projection and is not mixed into the pure FBCCA channel/subband-weight mainline unless the chosen profile enables it.

## 数据质量与协议一致性

- Session1：`/data1/zkx/brain/ssvep/data/subject001_collection_20260411_220406_r01/session_manifest.json`
- Session2：`None`
- 保留trial：`74` / `74`
- 最小样本比例：`0.900`
- 数据策略：`new-only`
- 协议签名：`sha1:304bf885cdd01e18cff32e90563e23b613196a37`

## 6.2.1 SSVEP分类准确率

| 口径 | Acc_SSVEP | Macro-F1 | 平均决策时间(s) | ITR(bits/min) |
|---|---:|---:|---:|---:|
| 四分类 8/10/12/15Hz | 0.4545 | 0.4464 | 2.0000 | 4.2433 |
| 二分类 control vs idle | 1.0000 | 1.0000 | 2.0000 | 29.9994 |

## 6.2.2 异步可用性

| 指标 | 数值 |
|---|---:|
| idle_fp_per_min | 0.0000 |
| control_recall | 1.0000 |
| switch_latency_s | 2.5000 |
| release_latency_s | 2.0000 |
| inference_ms | 2.4101 |

## FBCCA权重增益表

| Rank | Model | 权重策略 | Acc4 | MacroF1_4 | ITR4 | idleFP/min | Recall | 通道权重数 | 子带权重 | Accepted |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 5 | fbcca | profile_or_config_weighted | 0.8182 | 0.8143 | 30.8336 | 6.0000 | 0.6364 | 0 | `[0.386488, 0.207296, 0.155609, 0.131955, 0.118651]` | N |

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
| 1 | msetcca | engineering-approx | 0.4545 | 0.4464 | 4.2433 | 0.0000 | 1.0000 | 2.5000 | 2.0000 | 2.4101 | N |
| 2 | ecca | engineering-approx | 0.4545 | 0.4464 | 4.2433 | 0.0000 | 0.8182 | 2.2500 | 2.0000 | 3.1853 | N |
| 3 | oacca | engineering-approx | 0.9091 | 0.8810 | 56.6566 | 0.0000 | 0.7273 | 3.3750 | 1.5000 | 2.3704 | N |
| 4 | cca | paper-faithful | 1.0000 | 1.0000 | 79.9991 | 3.0000 | 0.9091 | 3.1250 | 1.5000 | 7.7676 | N |
| 5 | fbcca | paper-faithful | 0.8182 | 0.8143 | 30.8336 | 6.0000 | 0.6364 | 5.1250 | 2.0000 | 35.6345 | N |
| 6 | legacy_fbcca_202603 | paper-faithful | 0.8182 | 0.8143 | 30.8336 | 6.0000 | 0.6364 | 5.1250 | 2.0000 | 49.9670 | N |
| 7 | tdca | engineering-approx | 0.4545 | 0.3667 | 5.6578 | 0.0000 | 0.5455 | 2.7500 | 1.5000 | 10.3917 | N |
| 8 | trca_r | engineering-approx | 0.2727 | 0.2111 | 0.0780 | 3.0000 | 0.3636 | 5.3750 | 1.5000 | 12.3450 | N |
| 9 | trca | paper-faithful | 0.3636 | 0.3083 | 1.8291 | 0.0000 | 0.0909 | 6.5000 | 1.5000 | 1.7225 | N |
| 10 | sscor | engineering-approx | 0.4545 | 0.3857 | 5.6578 | 0.0000 | 0.0000 | 6.5000 | 1.5000 | 10.1732 | N |
| 11 | itcca | engineering-approx | 0.2727 | 0.3083 | 0.0780 | 0.0000 | 0.0000 | 6.5000 | 1.5000 | 2.2569 | N |

## 图表文件

- 四分类混淆矩阵：`/data1/zkx/brain/ssvep/reports/20260413/model_compare_20260413_223538/figures/confusion_4class.png`
- 二分类混淆矩阵：`/data1/zkx/brain/ssvep/reports/20260413/model_compare_20260413_223538/figures/confusion_2class.png`
- 决策时间直方图：`/data1/zkx/brain/ssvep/reports/20260413/model_compare_20260413_223538/figures/decision_time_hist.png`
- 模型雷达图：`/data1/zkx/brain/ssvep/reports/20260413/model_compare_20260413_223538/figures/model_radar_async_vs_cls.png`
