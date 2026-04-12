# SSVEP Offline Training-Evaluation Report

- Generated at: `2026-04-12T13:55:50`
- Session1 manifest: `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\datasets\subject001_collection_20260411_220406_r01\session_manifest.json`
- Session2 manifest: `None`
- Chosen model: `legacy_fbcca_202603`
- Chosen rank: `1`
- Chosen meets acceptance: `False`
- Profile saved: `False`
- Profile path: `None`

## 数据质量过滤

- min_sample_ratio: `0.900`
- max_retry_count: `3`
- Session1 kept/total: `74`/`74`

| Session | Kept/Total | Drop Ratio | Drop by Shortfall | Drop by Retry |
|---|---:|---:|---:|---:|
| subject001_collection_20260411_220406_r01 | 74/74 | 0.000 | 0 | 0 |

## 6.2.1 SSVEP 分类准确率

| 口径 | Acc_SSVEP | Macro-F1 | Mean Decision Time(s) | ITR(bits/min) |
|---|---:|---:|---:|---:|
| 四分类(8/10/12/15Hz) | 1.0000 | 1.0000 | 1.5000 | 79.9991 |
| 二分类(control vs idle) | 0.9333 | 0.9068 | 1.5000 | 25.8656 |

## 6.2.2 异步可用性评测

| 指标 | 数值 |
|---|---:|
| idle_fp_per_min | 3.0000 |
| control_recall | 1.0000 |
| switch_latency_s | 1.7500 |
| release_latency_s | 1.5000 |
| inference_ms | 2.9480 |

## FBCCA 前端与权重摘要

- channel_weight_mode: `None`
- spatial_filter_mode: `None`
- spatial_filter_rank: `None`

## 图表文件

- confusion_4class: `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\reports\train_eval\20260412\run_20260412_135529_subject001\figures\confusion_4class.png`
- confusion_2class: `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\reports\train_eval\20260412\run_20260412_135529_subject001\figures\confusion_2class.png`
- decision_time_hist: `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\reports\train_eval\20260412\run_20260412_135529_subject001\figures\decision_time_hist.png`
- model_radar_async_vs_cls: `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\2026-04_async_fbcca_idle_decoder\profiles\reports\train_eval\20260412\run_20260412_135529_subject001\figures\model_radar_async_vs_cls.png`
