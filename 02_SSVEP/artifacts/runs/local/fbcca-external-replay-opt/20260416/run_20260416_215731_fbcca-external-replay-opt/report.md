# FBCCA External Replay Opt

- Generated at: `2026-04-16T22:26:12`
- Task: `fbcca-external-replay-opt`
- Subject: `Subject2`
- Outer eval mode: `loso4`
- Deployment view: `chronological_last_session`
- Replay speed: `max`
- Search preset: `smoke8`
- Chosen model: `fbcca`
- FBCCA variant: `fbcca_fixed_all8`
- Confidence variant: `global_correctness_logistic`
- Status: `ok`
- Status reasons: `none`
- Chosen rationale: `fbcca_not_clearly_improved`
- Simulation only profile: `True`

## Holdout Selection

- idle_fp_per_min: `8.216343128237401`
- control_recall: `0.4166666666666667`
- control_recall_at_3s: `0.18333333333333335`
- release_latency_s: `2.375`
- inference_ms: `140.8817249975982`

## Deployment View Replay

- idle_fp_per_min: `6.272352132049519`
- control_recall: `0.3333333333333333`
- control_recall_at_3s: `0.13333333333333333`
- release_latency_s: `2.6171875`
- first_detection_latency_s: `3.2421875`

## Replay Frequency Breakdown

- freq=`13.0`Hz trials=`10` raw=`1.0` gate=`0.0` commit=`0.0` release=`0.1`
- freq=`17.0`Hz trials=`10` raw=`1.0` gate=`0.4` commit=`0.4` release=`0.6`
- freq=`21.0`Hz trials=`10` raw=`1.0` gate=`1.0` commit=`0.6` release=`0.5`

## Diagnostics

- fbcca_search_board rows: `4`
- decision_search_board rows: `576`
- holdout_selection_board rows: `4`
- replay_timeline rows: `1327`
- confidence_diagnostics rows: `20`
- error_attribution rows: `3`
- replay_frequency_breakdown rows: `3`
- sanity_compare rows: `2`
