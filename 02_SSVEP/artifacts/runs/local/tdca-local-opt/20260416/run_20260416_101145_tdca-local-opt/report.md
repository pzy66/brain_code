# TDCA Local Opt

- Generated at: `2026-04-16T11:07:45`
- Task: `tdca-local-opt`
- Search preset: `reduced13`
- Chosen model: `tdca`
- Decoder variant: `tdca_like_legacy`
- Confidence variant: `global_correctness_logistic`
- Confidence training scheme: `oof_gate_logreg_on_train_split`
- Decision evidence variant: `centered_logit_over_enter_threshold`
- OOF group key: `block_index`
- OOF group count: `17`
- Sample weight mode: `per_trial_equal`
- Training window policy: `last_window_only`
- Training latency sec: `0.0`
- Analysis latency sec: `0.0`
- Effective raw window sec: `2.0`
- Paper alignment level: `partial`
- Profile saved: `True`
- Profile path: `D:\brain\brain_code\02_SSVEP\artifacts\runs\local\tdca-local-opt\20260416\run_20260416_101145_tdca-local-opt\default_profile.json`
- Report status: `ok`
- Status reasons: `none`
- Chosen model rationale: `tdca_not_clearly_superior`
- Gate calibration: `valid`
- Decision search: `effective`
- Decision search target: `tune_split`
- Final selection target: `holdout_split`
- Run valid for deployment: `False`

## Async Metrics

- idle_fp_per_min: `0.0`
- control_recall: `0.25`
- control_recall_at_3s: `0.16666666666666666`
- switch_latency_s: `7.0`
- release_latency_s: `7.0`
- inference_ms: `8.19999999657739`

## 4-Class

- acc: `0.4166666666666667`
- macro_f1: `0.3511904761904762`
- itr_bpm: `1.0270185807206615`

## Gate Calibration

- positive_windows: `638`
- negative_windows: `1286`
- positive_trials: `40`
- negative_trials: `73`
- idle_trial_count: `34`
- brier_score: `0.15109165394951293`
- auc_roc: `0.8569353100913137`
- diagnostics_rows: `20`

## Tune Summary

- rows_total: `754`
- min_control_trials_by_freq: `8`
- idle_trial_count: `26`
- tune_rows_valid: `True`

## Data Sufficiency

- session_count: `1`
- trial_count: `74`
- unique_split_fingerprints: `5`
- current_sessions_sufficient_for_deployment: `False`

## Decision Bottleneck

- control_trials: `60`
- switch_trials: `50`
- release_trials: `15`
- raw_correct_seen_count: `51`
- gate_pass_correct_seen_count: `30`
- commit_seen_count: `12`
- median_first_gate_pass_latency_s: `2.5`
- median_max_p_correct: `0.7699979836044142`
- median_max_decision_evidence: `0.839286669613222`
- failure_breakdown: `{"decoder_miss": 9, "confidence_reject_miss": 21, "decision_miss": 18}`
