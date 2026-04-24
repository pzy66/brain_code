# SSVEP训练评测报告

## 总览

- 生成时间：`2026-04-13T02:14:47`
- 任务模式：`profile-eval`
- 推荐模型族：``
- 部署候选模型：`msetcca`
- 是否达到验收阈值：`否`
- 默认实时profile是否覆盖：`否`
- 最佳候选profile：`C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\reports\smoke12\20260413\run_20260413_021035_subject001\summary__rec-msetcca__dep-msetcca__m-all8__s-20260410__profile.json`
- 最佳加权FBCCA profile：`C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\reports\smoke12\20260413\run_20260413_021035_subject001\summary__rec-msetcca__dep-msetcca__m-all8__s-20260410__profile.json`

## 权重定义

- 通道权重：Eight-channel diagonal EEG weights for all8 FBCCA. The realtime window is transformed as X_weighted[:, c] = X[:, c] * channel_weights[c] before FBCCA scoring; len(w) must match the realtime EEG channel count.
- 子带权重：Five global filter-bank fusion weights shared by all EEG channels. FBCCA combines scores as score[f] = sum_b subband_weights[b] * rho[b, f]^2. These weights are not per-channel weights and they are not learned cutoff frequencies.
- 当前主线：The default trainable FBCCA frontend is separable: channel_weights[8] and subband_weights[5] are learned separately and applied as channel scaling plus global subband-score fusion. A full 8x5 channel-by-subband matrix is intentionally not trained by default to reduce overfitting.
- 空间滤波：Optional TRCA/shared spatial frontend state. It is a separate spatial projection and is not mixed into the pure FBCCA channel/subband-weight mainline unless the chosen profile enables it.

## 数据质量与协议一致性

- Session1：`C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\datasets\subject001_collection_20260411_220406_r01\session_manifest.json`
- Session2：`None`
- 保留trial：`0` / `0`
- 最小样本比例：`0.900`
- 数据策略：`new-only`
- 协议签名：``

## 6.2.1 SSVEP分类准确率

| 口径 | Acc_SSVEP | Macro-F1 | 平均决策时间(s) | ITR(bits/min) |
|---|---:|---:|---:|---:|
| 四分类 8/10/12/15Hz | 0.7273 | 0.6952 | 1.5000 | 28.8955 |
| 二分类 control vs idle | 0.8000 | 0.7847 | 1.5000 | 11.1229 |

## 6.2.2 异步可用性

| 指标 | 数值 |
|---|---:|
| idle_fp_per_min | 0.0000 |
| control_recall | 0.7273 |
| switch_latency_s | 3.3750 |
| release_latency_s | 1.5000 |
| inference_ms | 6.7420 |

## FBCCA权重增益表

| Rank | Model | 权重策略 | Acc4 | MacroF1_4 | ITR4 | idleFP/min | Recall | 通道权重数 | 子带权重 | Accepted |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 6 | fbcca_plain_all8 | profile_or_config_weighted | 0.9091 | 0.9143 | 56.6566 | 0.0000 | 0.2727 | 0 | `[0.386488, 0.207296, 0.155609, 0.131955, 0.118651]` | N |
| 8 | fbcca_profile_weighted | profile_or_config_weighted | 0.9091 | 0.9143 | 56.6566 | 0.0000 | 0.2727 | 0 | `[]` | N |

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
| 1 | msetcca | engineering-approx | 0.7273 | 0.6952 | 28.8955 | 0.0000 | 0.7273 | 3.3750 | 1.5000 | 6.7420 | N |
| 2 | cca | paper-faithful | 1.0000 | 1.0000 | 79.9991 | 3.0000 | 0.9091 | 3.1250 | 1.5000 | 6.9125 | N |
| 3 | ecca | engineering-approx | 0.7273 | 0.6952 | 28.8955 | 6.0000 | 0.9091 | 1.7500 | 1.5000 | 7.0556 | N |
| 4 | tdca | engineering-approx | 0.4545 | 0.3667 | 5.6578 | 0.0000 | 0.5455 | 2.7500 | 1.5000 | 10.6931 | N |
| 5 | trca_r | engineering-approx | 0.2727 | 0.2111 | 0.0780 | 3.0000 | 0.3636 | 5.3750 | 1.5000 | 11.2903 | N |
| 6 | fbcca_plain_all8 | paper-faithful | 0.9091 | 0.9143 | 56.6566 | 0.0000 | 0.2727 | 6.5000 | 1.5000 | 27.8482 | N |
| 7 | legacy_fbcca_202603 | paper-faithful | 0.9091 | 0.9143 | 56.6566 | 0.0000 | 0.2727 | 6.5000 | 1.5000 | 29.7168 | N |
| 8 | fbcca_profile_weighted | engineering-approx | 0.9091 | 0.9143 | 56.6566 | 0.0000 | 0.2727 | 6.5000 | 1.5000 | 37.1569 | N |
| 9 | trca | paper-faithful | 0.3636 | 0.3083 | 1.8291 | 0.0000 | 0.0909 | 6.5000 | 1.5000 | 6.5094 | N |
| 10 | oacca | engineering-approx | 0.8182 | 0.7976 | 41.1115 | 0.0000 | 0.0000 | 6.5000 | 1.5000 | 6.7501 | N |
| 11 | sscor | engineering-approx | 0.4545 | 0.3857 | 5.6578 | 0.0000 | 0.0000 | 6.5000 | 1.5000 | 10.4761 | N |
| 12 | itcca | engineering-approx | 0.2727 | 0.3083 | 0.0780 | 0.0000 | 0.0000 | 6.5000 | 1.5000 | 6.7023 | N |

## 图表文件

- 四分类混淆矩阵：``
- 二分类混淆矩阵：``
- 决策时间直方图：``
- 模型雷达图：``
