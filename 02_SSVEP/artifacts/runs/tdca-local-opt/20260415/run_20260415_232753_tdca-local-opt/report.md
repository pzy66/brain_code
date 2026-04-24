# TDCA Local Opt

- Generated at: `2026-04-15T23:29:26`
- Task: `tdca-local-opt`
- Search preset: `smoke4`
- Chosen model: `tdca`
- Decoder variant: `tdca_like_legacy`
- Confidence variant: `global_correctness_logistic`
- Training window policy: `last_window_only`
- Profile saved: `True`
- Profile path: `D:\brain\brain_code\02_SSVEP\artifacts\runs\tdca-local-opt\20260415\run_20260415_232753_tdca-local-opt\default_profile.json`
- Report status: `ok`
- Status reasons: `none`
- Chosen model rationale: `tdca_superior_on_primary_ranking`
- Gate calibration: `valid`
- Decision search: `effective`
- Run valid for deployment: `False`

## Async Metrics

- idle_fp_per_min: `0.0`
- control_recall: `0.9166666666666666`
- control_recall_at_3s: `0.6666666666666666`
- switch_latency_s: `2.75`
- release_latency_s: `2.0`
- inference_ms: `7.3841999992509955`

## 4-Class

- acc: `1.0`
- macro_f1: `1.0`
- itr_bpm: `36.22599922912501`

## Gate Calibration

- positive_windows: `269`
- negative_windows: `693`
- idle_trial_count: `34`
- brier_score: `0.146605473881552`
- auc_roc: `0.8647494595449986`

## Data Sufficiency

- session_count: `1`
- trial_count: `74`
- unique_split_fingerprints: `1`
- current_sessions_sufficient_for_deployment: `False`
