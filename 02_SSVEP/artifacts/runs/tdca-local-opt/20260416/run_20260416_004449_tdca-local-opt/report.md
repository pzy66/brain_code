# TDCA Local Opt

- Generated at: `2026-04-16T00:51:17`
- Task: `tdca-local-opt`
- Search preset: `reduced13`
- Chosen model: `tdca`
- Decoder variant: `tdca_like_legacy`
- Confidence variant: `global_correctness_logistic`
- Confidence training scheme: `oof_gate_logreg_on_train_split`
- OOF group key: `block_index`
- OOF group count: `17`
- Sample weight mode: `per_trial_equal`
- Training window policy: `last_window_only`
- Training latency sec: `0.0`
- Profile saved: `True`
- Profile path: `D:\brain\brain_code\02_SSVEP\artifacts\runs\tdca-local-opt\20260416\run_20260416_004449_tdca-local-opt\default_profile.json`
- Report status: `ok`
- Status reasons: `none`
- Chosen model rationale: `tdca_not_clearly_superior`
- Gate calibration: `valid`
- Decision search: `effective`
- Decision search target: `gate_split`
- Final selection target: `holdout_split`
- Run valid for deployment: `False`

## Async Metrics

- idle_fp_per_min: `0.0`
- control_recall: `0.25`
- control_recall_at_3s: `0.16666666666666666`
- switch_latency_s: `7.0`
- release_latency_s: `7.0`
- inference_ms: `2.6865999989240663`

## 4-Class

- acc: `0.4166666666666667`
- macro_f1: `0.3611111111111111`
- itr_bpm: `0.9725829669015452`

## Gate Calibration

- positive_windows: `301`
- negative_windows: `661`
- positive_trials: `40`
- negative_trials: `74`
- idle_trial_count: `34`
- brier_score: `0.1577206313896724`
- auc_roc: `0.8510662893732943`

## Data Sufficiency

- session_count: `1`
- trial_count: `74`
- unique_split_fingerprints: `5`
- current_sessions_sufficient_for_deployment: `False`
