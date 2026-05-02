# TDCA Local Opt

- Generated at: `2026-04-15T23:42:51`
- Task: `tdca-local-opt`
- Search preset: `reduced13`
- Chosen model: `tdca`
- Decoder variant: `tdca_like_legacy`
- Confidence variant: `global_correctness_logistic`
- Training window policy: `last_window_only`
- Profile saved: `True`
- Profile path: `<repo>\02_SSVEP\artifacts\runs\tdca-local-opt\20260415\run_20260415_233022_tdca-local-opt\default_profile.json`
- Report status: `ok`
- Status reasons: `none`
- Chosen model rationale: `tdca_not_clearly_superior`
- Gate calibration: `valid`
- Decision search: `effective`
- Run valid for deployment: `False`

## Async Metrics

- idle_fp_per_min: `0.0`
- control_recall: `0.16666666666666666`
- control_recall_at_3s: `0.16666666666666666`
- switch_latency_s: `7.0`
- release_latency_s: `7.0`
- inference_ms: `7.325100000343809`

## 4-Class

- acc: `0.4166666666666667`
- macro_f1: `0.33636363636363636`
- itr_bpm: `0.8417155340462914`

## Gate Calibration

- positive_windows: `319`
- negative_windows: `643`
- idle_trial_count: `34`
- brier_score: `0.13952501806873507`
- auc_roc: `0.8755149500041439`

## Data Sufficiency

- session_count: `1`
- trial_count: `74`
- unique_split_fingerprints: `5`
- current_sessions_sufficient_for_deployment: `False`
