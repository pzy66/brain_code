# External Short-Pretrain 5-Class Benchmark

- run_id: `ysuan_full24_fbcca_ridge5_lrt_s2_20260511_120724`
- generated_at: `2026-05-11T14:55:10`
- freqs: `8,10.5,12,15`
- 240hz_all_integer_frames_per_cycle: `False`
- subject_count: `24`
- row_count: `2592`
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
| 1 | no | fbcca_ridge5 | `win2p5_me3_sm3_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 3.00 | 0.9086 | 0.9039 | 0.9180 | 0.9001 | 0.6328 | 0.8781 | 0.0000 | 0.7969 | 3.0000 |
| 2 | no | fbcca_ridge5 | `win2p5_me3_sm3_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 4.00 | 0.9086 | 0.9039 | 0.9180 | 0.9001 | 0.6328 | 0.8781 | 0.0000 | 0.7969 | 3.0000 |
| 3 | no | fbcca_ridge5 | `win2p5_me3_sm2_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 3.00 | 0.9036 | 0.9005 | 0.9185 | 0.9007 | 0.6406 | 0.8797 | 0.0000 | 0.7958 | 3.0026 |
| 4 | no | fbcca_ridge5 | `win2p5_me3_sm2_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 4.00 | 0.9036 | 0.9005 | 0.9185 | 0.9007 | 0.6406 | 0.8797 | 0.0000 | 0.7958 | 3.0026 |
| 5 | no | fbcca_ridge5 | `win2p5_me2_sm3_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 3.00 | 0.9086 | 0.9039 | 0.9120 | 0.8955 | 0.8828 | 0.8828 | 0.0000 | 0.8219 | 2.7500 |
| 6 | no | fbcca_ridge5 | `win2p5_me2_sm3_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 4.00 | 0.9086 | 0.9039 | 0.9120 | 0.8955 | 0.8828 | 0.8828 | 0.0000 | 0.8219 | 2.7500 |
| 7 | no | fbcca_ridge5 | `win2p5_me2_sm2_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 3.00 | 0.9036 | 0.9005 | 0.9096 | 0.8930 | 0.9531 | 0.8828 | 0.0000 | 0.8229 | 2.7500 |
| 8 | no | fbcca_ridge5 | `win2p5_me2_sm2_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 4.00 | 0.9036 | 0.9005 | 0.9096 | 0.8930 | 0.9531 | 0.8828 | 0.0000 | 0.8229 | 2.7500 |
| 9 | no | fbcca_ridge5 | `win2_me3_sm3_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 3.00 | 0.8861 | 0.8669 | 0.9424 | 0.8973 | 0.4536 | 0.8682 | 0.7062 | 0.8042 | 2.5391 |
| 10 | no | fbcca_ridge5 | `win2_me3_sm3_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 4.00 | 0.8861 | 0.8669 | 0.9424 | 0.8973 | 0.4536 | 0.8682 | 0.7062 | 0.8042 | 2.5391 |


## YSU-an No-Control Subtype FP

| Rank | Method | Recipe | NS1 FP/min | NS2 FP/min | NS3 FP/min | NS All FP/min | CS Control Recall |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | fbcca_ridge5 | `win2p5_me3_sm3_lrtmw` | 0.2656 | 1.0000 | 0.0000 | 0.6328 | 0.8781 |
| 2 | fbcca_ridge5 | `win2p5_me3_sm3_lrtmw` | 0.2656 | 1.0000 | 0.0000 | 0.6328 | 0.8781 |
| 3 | fbcca_ridge5 | `win2p5_me3_sm2_lrtmw` | 0.3281 | 0.9531 | 0.0000 | 0.6406 | 0.8797 |
| 4 | fbcca_ridge5 | `win2p5_me3_sm2_lrtmw` | 0.3281 | 0.9531 | 0.0000 | 0.6406 | 0.8797 |
| 5 | fbcca_ridge5 | `win2p5_me2_sm3_lrtmw` | 0.5000 | 1.2656 | 0.0000 | 0.8828 | 0.8828 |
| 6 | fbcca_ridge5 | `win2p5_me2_sm3_lrtmw` | 0.5000 | 1.2656 | 0.0000 | 0.8828 | 0.8828 |
| 7 | fbcca_ridge5 | `win2p5_me2_sm2_lrtmw` | 0.5469 | 1.3594 | 0.0000 | 0.9531 | 0.8828 |
| 8 | fbcca_ridge5 | `win2p5_me2_sm2_lrtmw` | 0.5469 | 1.3594 | 0.0000 | 0.9531 | 0.8828 |
| 9 | fbcca_ridge5 | `win2_me3_sm3_lrtmw` | 0.3594 | 1.0469 | 0.0000 | 0.4536 | 0.8682 |
| 10 | fbcca_ridge5 | `win2_me3_sm3_lrtmw` | 0.3594 | 1.0469 | 0.0000 | 0.4536 | 0.8682 |

## Top Recipes

| Rank | Deployable | Method | Recipe | Freqs | Coverage | Cal Blocks | Idle Mult | Mean Fixed 5c Acc | Mean Fixed 5c Macro-F1 | Mean Async 5c Acc | Mean Async 5c Macro-F1 | Mean Idle FP/min | Mean Control Recall | Recall <=2.5s | Recall <=3s | Mean Detection Latency s |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | no | fbcca_ridge5 | `win2p5_me3_sm3_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 3.00 | 0.9086 | 0.9039 | 0.9180 | 0.9001 | 0.6328 | 0.8781 | 0.0000 | 0.7969 | 3.0000 |
| 2 | no | fbcca_ridge5 | `win2p5_me3_sm3_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 4.00 | 0.9086 | 0.9039 | 0.9180 | 0.9001 | 0.6328 | 0.8781 | 0.0000 | 0.7969 | 3.0000 |
| 3 | no | fbcca_ridge5 | `win2p5_me3_sm2_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 3.00 | 0.9036 | 0.9005 | 0.9185 | 0.9007 | 0.6406 | 0.8797 | 0.0000 | 0.7958 | 3.0026 |
| 4 | no | fbcca_ridge5 | `win2p5_me3_sm2_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 4.00 | 0.9036 | 0.9005 | 0.9185 | 0.9007 | 0.6406 | 0.8797 | 0.0000 | 0.7958 | 3.0026 |
| 5 | no | fbcca_ridge5 | `win2p5_me2_sm3_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 3.00 | 0.9086 | 0.9039 | 0.9120 | 0.8955 | 0.8828 | 0.8828 | 0.0000 | 0.8219 | 2.7500 |
| 6 | no | fbcca_ridge5 | `win2p5_me2_sm3_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 4.00 | 0.9086 | 0.9039 | 0.9120 | 0.8955 | 0.8828 | 0.8828 | 0.0000 | 0.8219 | 2.7500 |
| 7 | no | fbcca_ridge5 | `win2p5_me2_sm2_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 3.00 | 0.9036 | 0.9005 | 0.9096 | 0.8930 | 0.9531 | 0.8828 | 0.0000 | 0.8229 | 2.7500 |
| 8 | no | fbcca_ridge5 | `win2p5_me2_sm2_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 4.00 | 0.9036 | 0.9005 | 0.9096 | 0.8930 | 0.9531 | 0.8828 | 0.0000 | 0.8229 | 2.7500 |
| 9 | no | fbcca_ridge5 | `win2_me3_sm3_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 3.00 | 0.8861 | 0.8669 | 0.9424 | 0.8973 | 0.4536 | 0.8682 | 0.7062 | 0.8042 | 2.5391 |
| 10 | no | fbcca_ridge5 | `win2_me3_sm3_lrtmw` | `8,10.5,12,15` | 24/24 | 2 | 4.00 | 0.8861 | 0.8669 | 0.9424 | 0.8973 | 0.4536 | 0.8682 | 0.7062 | 0.8042 | 2.5391 |


## Weak Subject Audit

| Subject | Control Recall | Idle FP/min | Async 5c Macro-F1 | Detection Latency s |
|---|---:|---:|---:|---:|
| S11 | 0.2375 | 1.8750 | 0.3548 | 3.0000 |

## Subjects

| Dataset | Subject | |
|---|---|---|
| ysu_an | S01 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S01` |
| ysu_an | S02 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S02` |
| ysu_an | S03 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S03` |
| ysu_an | S04 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S04` |
| ysu_an | S05 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S05` |
| ysu_an | S06 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S06` |
| ysu_an | S07 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S07` |
| ysu_an | S08 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S08` |
| ysu_an | S09 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S09` |
| ysu_an | S10 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S10` |
| ysu_an | S11 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S11` |
| ysu_an | S12 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S12` |
| ysu_an | S13 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S13` |
| ysu_an | S14 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S14` |
| ysu_an | S15 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S15` |
| ysu_an | S16 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S16` |
| ysu_an | S17 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S17` |
| ysu_an | S18 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S18` |
| ysu_an | S19 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S19` |
| ysu_an | S20 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S20` |
| ysu_an | S21 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S21` |
| ysu_an | S22 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S22` |
| ysu_an | S23 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S23` |
| ysu_an | S24 | `/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw/S24` |
