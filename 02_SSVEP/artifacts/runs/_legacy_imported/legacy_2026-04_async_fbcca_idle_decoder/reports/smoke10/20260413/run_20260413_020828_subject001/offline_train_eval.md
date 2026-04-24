# SSVEP训练评测报告

## 总览

- 生成时间：`2026-04-13T02:08:33`
- 任务模式：`classifier-compare`
- 推荐模型族：``
- 部署候选模型：`oacca`
- 是否达到验收阈值：`否`
- 默认实时profile是否覆盖：`否`
- 最佳候选profile：``
- 最佳加权FBCCA profile：``

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
- 数据策略：``
- 协议签名：``

## 6.2.1 SSVEP分类准确率

| 口径 | Acc_SSVEP | Macro-F1 | 平均决策时间(s) | ITR(bits/min) |
|---|---:|---:|---:|---:|
| 四分类 8/10/12/15Hz | 1.0000 | 1.0000 | 1.5000 | 79.9991 |
| 二分类 control vs idle | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 6.2.2 异步可用性

| 指标 | 数值 |
|---|---:|
| idle_fp_per_min | 0.0000 |
| control_recall | 0.0000 |
| switch_latency_s | 0.0000 |
| release_latency_s | 0.0000 |
| inference_ms | 0.0000 |

## FBCCA权重增益表

| Rank | Model | 权重策略 | Acc4 | MacroF1_4 | ITR4 | idleFP/min | Recall | 通道权重数 | 子带权重 | Accepted |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
|  | 无FBCCA权重候选 |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | `[]` | N |

## 加权提升归因

| 对比 | 可用 | idleFP变化 | Recall变化 | Switch时延变化(s) | 是否可接受提升 |
|---|---|---:|---:|---:|---|
| 无归因对比 | N | 0.0000 | 0.0000 | 0.0000 | N |

## 全模型识别效果对比

| Rank | Model | Impl | Acc4 | MacroF1_4 | ITR4 | idleFP/min | Recall | Switch(s) | Release(s) | Inference(ms) | Accepted |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | oacca | engineering-approx | 1.0000 | 1.0000 | 79.9991 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.9211 | N |
| 2 | cca | paper-faithful | 1.0000 | 1.0000 | 79.9991 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.9884 | N |
| 3 | ecca | engineering-approx | 0.8182 | 0.7893 | 41.1115 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.0608 | N |
| 4 | fbcca_plain_all8 | paper-faithful | 0.7273 | 0.7060 | 28.8955 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 29.0230 | N |
| 5 | legacy_fbcca_202603 | paper-faithful | 0.7273 | 0.7060 | 28.8955 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 214.2990 | N |
| 6 | msetcca | engineering-approx | 0.7273 | 0.6310 | 28.8955 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.5386 | N |
| 7 | trca_r | engineering-approx | 0.6364 | 0.5375 | 19.1196 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.3983 | N |
| 8 | tdca | engineering-approx | 0.5455 | 0.4571 | 11.4213 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.7131 | N |
| 9 | trca | paper-faithful | 0.3636 | 0.3409 | 1.8291 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.3588 | N |
| 10 | itcca | engineering-approx | 0.3636 | 0.3095 | 1.8291 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.4506 | N |
| 11 | sscor | engineering-approx | 0.3636 | 0.2364 | 1.8291 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.8297 | N |

## 图表文件

- 四分类混淆矩阵：``
- 二分类混淆矩阵：``
- 决策时间直方图：``
- 模型雷达图：``
