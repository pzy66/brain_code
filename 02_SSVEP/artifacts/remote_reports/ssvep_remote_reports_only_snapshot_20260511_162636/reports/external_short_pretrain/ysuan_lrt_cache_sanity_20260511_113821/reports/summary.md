# External Short-Pretrain 5-Class Benchmark

- run_id: `ysuan_lrt_cache_sanity_20260511_113821`
- generated_at: `2026-05-11T12:04:38`
- freqs: `8,10.5,12,15`
- 240hz_all_integer_frames_per_cycle: `False`
- subject_count: `4`
- row_count: `432`
- score_bank_mode: `full_reference_bank`
- frequency_selection_mode: `none`
- idle_eval_mode: `hard_noncommand`
- pretrain_budget_sec: `120.0`
- estimated_pretrain_duration_sec: `102.0`
- pretrain_budget_pass: `True`
- channel_contract: `strict_required_8_posterior`
- project_channel_names: `Oz,O1,O2,PO3,POz,PO7,PO8,PO4`

> Idle/no-control is proxied with non-command target stimulus trials for Wang/BETA; YSU-an uses explicit NS1/NS2/NS3 no-control trials.
> External datasets are evaluated after selecting only this posterior 8-channel subset; deployed numeric board channels must be wired to the same electrode order.

## Top Shared Recipes

| Rank | Deployable | Method | Recipe | Freqs | Coverage | Cal Blocks | Idle Mult | Mean Fixed 5c Acc | Mean Fixed 5c Macro-F1 | Mean Async 5c Acc | Mean Async 5c Macro-F1 | Mean Idle FP/min | Mean Control Recall | Recall <=2.5s | Recall <=3s | Mean Detection Latency s |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | no | fbcca_ridge5 | `win2p5_me2_sm3_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 3.00 | 0.9156 | 0.9136 | 0.9219 | 0.9072 | 0.4688 | 0.8750 | 0.0000 | 0.8250 | 2.7500 |
| 2 | no | fbcca_ridge5 | `win2p5_me2_sm3_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 4.00 | 0.9156 | 0.9136 | 0.9219 | 0.9072 | 0.4688 | 0.8750 | 0.0000 | 0.8250 | 2.7500 |
| 3 | no | fbcca_ridge5 | `win2p5_me2_sm2_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 3.00 | 0.9078 | 0.9059 | 0.9250 | 0.9097 | 0.4219 | 0.8781 | 0.0000 | 0.8250 | 2.7500 |
| 4 | no | fbcca_ridge5 | `win2p5_me2_sm2_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 4.00 | 0.9078 | 0.9059 | 0.9250 | 0.9097 | 0.4219 | 0.8781 | 0.0000 | 0.8250 | 2.7500 |
| 5 | yes | fbcca_ridge5 | `win2_me2_sm3_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 3.00 | 0.8942 | 0.8843 | 0.9486 | 0.9128 | 0.4536 | 0.8875 | 0.8219 | 0.8469 | 2.2500 |
| 6 | yes | fbcca_ridge5 | `win2_me2_sm3_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 4.00 | 0.8942 | 0.8843 | 0.9486 | 0.9128 | 0.4536 | 0.8875 | 0.8219 | 0.8469 | 2.2500 |
| 7 | yes | fbcca_ridge5 | `win2_me3_sm2_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 3.00 | 0.8871 | 0.8772 | 0.9516 | 0.9130 | 0.2722 | 0.8781 | 0.7844 | 0.8344 | 2.5000 |
| 8 | yes | fbcca_ridge5 | `win2_me3_sm2_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 4.00 | 0.8871 | 0.8772 | 0.9516 | 0.9130 | 0.2722 | 0.8781 | 0.7844 | 0.8344 | 2.5000 |
| 9 | yes | fbcca_ridge5 | `win2_me3_sm3_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 3.00 | 0.8942 | 0.8843 | 0.9496 | 0.9093 | 0.2722 | 0.8719 | 0.7875 | 0.8156 | 2.5000 |
| 10 | yes | fbcca_ridge5 | `win2_me3_sm3_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 4.00 | 0.8942 | 0.8843 | 0.9496 | 0.9093 | 0.2722 | 0.8719 | 0.7875 | 0.8156 | 2.5000 |


## YSU-an No-Control Subtype FP

| Rank | Method | Recipe | NS1 FP/min | NS2 FP/min | NS3 FP/min | NS All FP/min | CS Control Recall |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | fbcca_ridge5 | `win2p5_me2_sm3_lrtmw` | 0.1875 | 0.7500 | 0.0000 | 0.4688 | 0.8750 |
| 2 | fbcca_ridge5 | `win2p5_me2_sm3_lrtmw` | 0.1875 | 0.7500 | 0.0000 | 0.4688 | 0.8750 |
| 3 | fbcca_ridge5 | `win2p5_me2_sm2_lrtmw` | 0.1875 | 0.6562 | 0.0000 | 0.4219 | 0.8781 |
| 4 | fbcca_ridge5 | `win2p5_me2_sm2_lrtmw` | 0.1875 | 0.6562 | 0.0000 | 0.4219 | 0.8781 |
| 5 | fbcca_ridge5 | `win2_me2_sm3_lrtmw` | 0.0000 | 1.4062 | 0.0000 | 0.4536 | 0.8875 |
| 6 | fbcca_ridge5 | `win2_me2_sm3_lrtmw` | 0.0000 | 1.4062 | 0.0000 | 0.4536 | 0.8875 |
| 7 | fbcca_ridge5 | `win2_me3_sm2_lrtmw` | 0.0000 | 0.8438 | 0.0000 | 0.2722 | 0.8781 |
| 8 | fbcca_ridge5 | `win2_me3_sm2_lrtmw` | 0.0000 | 0.8438 | 0.0000 | 0.2722 | 0.8781 |
| 9 | fbcca_ridge5 | `win2_me3_sm3_lrtmw` | 0.0000 | 0.8438 | 0.0000 | 0.2722 | 0.8719 |
| 10 | fbcca_ridge5 | `win2_me3_sm3_lrtmw` | 0.0000 | 0.8438 | 0.0000 | 0.2722 | 0.8719 |

## Top Recipes

| Rank | Deployable | Method | Recipe | Freqs | Coverage | Cal Blocks | Idle Mult | Mean Fixed 5c Acc | Mean Fixed 5c Macro-F1 | Mean Async 5c Acc | Mean Async 5c Macro-F1 | Mean Idle FP/min | Mean Control Recall | Recall <=2.5s | Recall <=3s | Mean Detection Latency s |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | no | fbcca_ridge5 | `win2p5_me2_sm3_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 3.00 | 0.9156 | 0.9136 | 0.9219 | 0.9072 | 0.4688 | 0.8750 | 0.0000 | 0.8250 | 2.7500 |
| 2 | no | fbcca_ridge5 | `win2p5_me2_sm3_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 4.00 | 0.9156 | 0.9136 | 0.9219 | 0.9072 | 0.4688 | 0.8750 | 0.0000 | 0.8250 | 2.7500 |
| 3 | no | fbcca_ridge5 | `win2p5_me2_sm2_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 3.00 | 0.9078 | 0.9059 | 0.9250 | 0.9097 | 0.4219 | 0.8781 | 0.0000 | 0.8250 | 2.7500 |
| 4 | no | fbcca_ridge5 | `win2p5_me2_sm2_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 4.00 | 0.9078 | 0.9059 | 0.9250 | 0.9097 | 0.4219 | 0.8781 | 0.0000 | 0.8250 | 2.7500 |
| 5 | yes | fbcca_ridge5 | `win2_me2_sm3_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 3.00 | 0.8942 | 0.8843 | 0.9486 | 0.9128 | 0.4536 | 0.8875 | 0.8219 | 0.8469 | 2.2500 |
| 6 | yes | fbcca_ridge5 | `win2_me2_sm3_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 4.00 | 0.8942 | 0.8843 | 0.9486 | 0.9128 | 0.4536 | 0.8875 | 0.8219 | 0.8469 | 2.2500 |
| 7 | yes | fbcca_ridge5 | `win2_me3_sm2_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 3.00 | 0.8871 | 0.8772 | 0.9516 | 0.9130 | 0.2722 | 0.8781 | 0.7844 | 0.8344 | 2.5000 |
| 8 | yes | fbcca_ridge5 | `win2_me3_sm2_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 4.00 | 0.8871 | 0.8772 | 0.9516 | 0.9130 | 0.2722 | 0.8781 | 0.7844 | 0.8344 | 2.5000 |
| 9 | yes | fbcca_ridge5 | `win2_me3_sm3_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 3.00 | 0.8942 | 0.8843 | 0.9496 | 0.9093 | 0.2722 | 0.8719 | 0.7875 | 0.8156 | 2.5000 |
| 10 | yes | fbcca_ridge5 | `win2_me3_sm3_lrtmw` | `8,10.5,12,15` | 4/4 | 2 | 4.00 | 0.8942 | 0.8843 | 0.9496 | 0.9093 | 0.2722 | 0.8719 | 0.7875 | 0.8156 | 2.5000 |


## Weak Subject Audit


## Subjects

| Dataset | Subject | |
|---|---|---|
| ysu_an | S01 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S01` |
| ysu_an | S02 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S02` |
| ysu_an | S03 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S03` |
| ysu_an | S04 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S04` |
