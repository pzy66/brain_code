# SSVEP训练评测报告

## 总览

- 生成时间：`2026-04-13T10:12:49`
- 任务模式：`profile-eval`
- 推荐模型族：``
- 部署候选模型：`fbcca_plain_all8`
- 是否达到验收阈值：`否`
- 默认实时profile是否覆盖：`否`
- 最佳候选profile：`None`
- 最佳加权FBCCA profile：`C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\reports\smoke13\20260413\run_20260413_101236_subject001\summary__rec-fbcca_plain_all8__dep-fbcca_plain_all8__m-all8__s-20260410__profile.json`

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
| 四分类 8/10/12/15Hz | 1.0000 | 1.0000 | 1.5000 | 79.9991 |
| 二分类 control vs idle | 0.9333 | 0.9206 | 1.5000 | 25.8656 |

## 6.2.2 异步可用性

| 指标 | 数值 |
|---|---:|
| idle_fp_per_min | 0.0000 |
| control_recall | 0.9091 |
| switch_latency_s | 2.6250 |
| release_latency_s | 1.5000 |
| inference_ms | 3.0771 |

## FBCCA权重增益表

| Rank | Model | 权重策略 | Acc4 | MacroF1_4 | ITR4 | idleFP/min | Recall | 通道权重数 | 子带权重 | Accepted |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | fbcca_plain_all8 | profile_or_config_weighted | 1.0000 | 1.0000 | 79.9991 | 0.0000 | 0.9091 | 0 | `[0.386488, 0.207296, 0.155609, 0.131955, 0.118651]` | Y |
| 2 | fbcca_profile_weighted | profile_or_config_weighted | 1.0000 | 1.0000 | 79.9991 | 0.0000 | 0.0000 | 0 | `[0.386488, 0.207296, 0.155609, 0.131955, 0.118651]` | N |

## 加权提升归因

| 对比 | 可用 | idleFP变化 | Recall变化 | Switch时延变化(s) | 是否可接受提升 |
|---|---|---:|---:|---:|---|
| legacy_fbcca_202603 -> fbcca_plain_all8 | N | 0.0000 | 0.0000 | 0.0000 | N |
| fbcca_plain_all8 -> fbcca_channel_weighted | N | 0.0000 | 0.0000 | 0.0000 | N |
| fbcca_plain_all8 -> fbcca_subband_weighted | N | 0.0000 | 0.0000 | 0.0000 | N |
| fbcca_plain_all8 -> fbcca_channel_subband_weighted | N | 0.0000 | 0.0000 | 0.0000 | N |
| fbcca_channel_weighted -> fbcca_channel_subband_weighted | N | 0.0000 | 0.0000 | 0.0000 | N |
| fbcca_channel_subband_weighted -> fbcca_channel_subband_trca_shared | N | 0.0000 | 0.0000 | 0.0000 | N |
| balanced -> speed | N | 0.0000 | 0.0000 | 0.0000 | N |

## 全模型识别效果对比

| Rank | Model | Impl | Acc4 | MacroF1_4 | ITR4 | idleFP/min | Recall | Switch(s) | Release(s) | Inference(ms) | Accepted |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | fbcca_plain_all8 | paper-faithful | 1.0000 | 1.0000 | 79.9991 | 0.0000 | 0.9091 | 2.6250 | 1.5000 | 3.0771 | Y |
| 2 | fbcca_profile_weighted | engineering-approx | 1.0000 | 1.0000 | 79.9991 | 0.0000 | 0.0000 | 6.5000 | 1.5000 | 3.3453 | N |

## 图表文件

- 四分类混淆矩阵：`C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\reports\smoke13\20260413\run_20260413_101236_subject001\figures\confusion_4class.png`
- 二分类混淆矩阵：`C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\reports\smoke13\20260413\run_20260413_101236_subject001\figures\confusion_2class.png`
- 决策时间直方图：`C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\reports\smoke13\20260413\run_20260413_101236_subject001\figures\decision_time_hist.png`
- 模型雷达图：`C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\reports\smoke13\20260413\run_20260413_101236_subject001\figures\model_radar_async_vs_cls.png`
