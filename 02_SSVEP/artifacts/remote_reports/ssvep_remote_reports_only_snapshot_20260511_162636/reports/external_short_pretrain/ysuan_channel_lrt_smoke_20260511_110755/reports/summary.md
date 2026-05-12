# External Short-Pretrain 5-Class Benchmark

- run_id: `ysuan_channel_lrt_smoke_20260511_110755`
- generated_at: `2026-05-11T11:10:33`
- freqs: `8,10.5,12,15`
- 240hz_all_integer_frames_per_cycle: `False`
- subject_count: `1`
- row_count: `1`
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
| 1 | no | fbcca_ridge5 | `win1p5_me1_lrtmw` | `8,10.5,12,15` | 1/1 | 2 | 3.00 | 0.5161 | 0.5446 | 0.7016 | 0.5841 | 4.8387 | 0.5750 | 0.5250 | 0.5750 | 1.5000 |


## YSU-an No-Control Subtype FP

| Rank | Method | Recipe | NS1 FP/min | NS2 FP/min | NS3 FP/min | NS All FP/min | CS Control Recall |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | fbcca_ridge5 | `win1p5_me1_lrtmw` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5750 |

## Top Recipes

| Rank | Deployable | Method | Recipe | Freqs | Coverage | Cal Blocks | Idle Mult | Mean Fixed 5c Acc | Mean Fixed 5c Macro-F1 | Mean Async 5c Acc | Mean Async 5c Macro-F1 | Mean Idle FP/min | Mean Control Recall | Recall <=2.5s | Recall <=3s | Mean Detection Latency s |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | no | fbcca_ridge5 | `win1p5_me1_lrtmw` | `8,10.5,12,15` | 1/1 | 2 | 3.00 | 0.5161 | 0.5446 | 0.7016 | 0.5841 | 4.8387 | 0.5750 | 0.5250 | 0.5750 | 1.5000 |


## Weak Subject Audit


## Subjects

| Dataset | Subject | |
|---|---|---|
| ysu_an | S01 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S01` |
