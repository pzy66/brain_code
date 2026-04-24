# FBCCA External Replay Opt

- Generated at: `2026-04-17T09:57:11`
- Task: `fbcca-external-replay-opt`
- Subject: `Subject2`
- Outer eval mode: `loso4`
- Deployment view: `chronological_last_session`
- Replay speed: `max`
- Search preset: `smoke8`
- Chosen model: ``
- Decoder variant: ``
- Decoder family: ``
- FBCCA variant: ``
- Template usage: ``
- Confidence variant: ``
- Status: `invalid`
- Status reasons: `no_selection_eligible_candidates`
- Chosen rationale: `invalid_run_not_comparable`
- frequency_balance_valid: `False`
- confidence_dominance_valid: `False`
- Simulation only profile: `True`
- strict_eligible_candidate_count: `0`
- gate_valid_candidate_count: `8`

## Holdout Selection

- idle_fp_per_min: ``
- control_recall: ``
- control_recall_at_3s: ``
- release_latency_s: ``
- inference_ms: ``

## Diagnostic Best Row

- candidate_key: `variant=ecca_all8|win=2.5|confidence=global_correctness_logistic`
- decoder_variant: `ecca_all8`
- confidence_variant: `global_correctness_logistic`
- diagnostic_only: `True`
- invalid_reasons: `frequency_balance_invalid`
- idle_fp_per_min: `0.3298970630469943`
- control_recall: `0.0`
- control_recall_at_3s: `0.0`
- release_latency_s: `1.625`

## Deployment View Replay

- idle_fp_per_min: `0.33012379642365886`
- control_recall: `0.0`
- control_recall_at_3s: `0.0`
- release_latency_s: `7.8671875`
- first_detection_latency_s: `inf`

## Replay Frequency Breakdown

- freq=`13` trials=`10` raw=`1.0` gate=`0.3` commit=`0.0` release=`0.0`
- freq=`17` trials=`10` raw=`1.0` gate=`0.3` commit=`0.0` release=`0.1`
- freq=`21` trials=`10` raw=`1.0` gate=`0.4` commit=`0.0` release=`0.0`

## Tune Frequency Breakdown

- freq=`13.0`Hz rows=`1787` positive_windows=`514` negative_windows=`1273` median_p=`0.4571965345323674` median_logit=`-0.17163396023959104`
- freq=`17.0`Hz rows=`1867` positive_windows=`589` negative_windows=`1278` median_p=`0.4919377868447744` median_logit=`-0.032251647926621184`
- freq=`21.0`Hz rows=`1626` positive_windows=`542` negative_windows=`1084` median_p=`0.5103860051663203` median_logit=`0.04154999731120036`

## Reference Diagnostics

- freq=`13` enter=`0.5898776120886177` p50=`0.5783436585825301` p75=`0.6284652807389128` neg_p90=`0.6030214708255235` headroom_p50=`-0.011533953506087546` bound=`True`
- freq=`17` enter=`0.5997038982804959` p50=`0.6173055752817413` p75=`0.6866662861528382` neg_p90=`0.6478568848007575` headroom_p50=`0.02024430605057037` bound=`True`
- freq=`21` enter=`0.6` p50=`0.6420547209647958` p75=`0.7313675455953443` neg_p90=`0.6605874096432683` headroom_p50=`0.04205472096479579` bound=`True`

## Diagnostics

- fbcca_search_board rows: `8`
- decision_search_board rows: `1152`
- holdout_selection_board rows: `8`
- replay_timeline rows: `1327`
- confidence_diagnostics rows: `16`
- error_attribution rows: `3`
- replay_frequency_breakdown rows: `3`
- reference_diagnostics rows: `3`
- sanity_compare rows: `2`
- invalid_reason_histogram: `{'confidence_dominance_invalid': 7, 'frequency_balance_invalid': 5}`
