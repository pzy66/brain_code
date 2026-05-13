# 02_SSVEP Cleanup Inventory

This inventory records the intended ownership of `02_SSVEP` after the practical
cleanup pass. It is a maintenance guide, not a benchmark report.

## Active Mainline

Keep these paths in the normal daily workflow:

- `START_SSVEP.py`
- `apps/data_collection_ui.py`
- `apps/realtime_online_ui.py`
- `apps/training_evaluation_ui.py`
- `apps/async_fbcca_validation_ui.py`
- `entrypoints/start_collection.py`
- `entrypoints/start_realtime.py`
- `entrypoints/start_training_eval.py`
- `ssvep_core/dataset.py`
- `ssvep_core/custom_ssvep_protocol.py`
- `ssvep_core/session_no_control_classifier.py`
- `ssvep_core/stimulus_profiles.py`
- `ssvep_core/score_classifier_runtime.py`
- `ssvep_core/async_fbcca_idle_standalone.py`

The active realtime-safe classifier gate variant is `baseline_lrtmw`. Session
no-control profiles use that gate variant with session-specific score statistics.

## Research-Only

Keep these paths for reproducibility and offline investigation, but do not present
them as the daily realtime path:

- `tools/run_external_short_pretrain_benchmark.py`
- `tools/run_external_frequency_server_sweep.py`
- `tools/compare_external_ssvep_fbcca_tdca.py`
- `ssvep_core/external_*_dataset.py`
- `ssvep_core/tdca_local_opt.py`
- `ssvep_core/decoders/tdca_decoder.py`
- `ssvep_core/decoders/trca_r_decoder.py`
- NS2 gate variants in benchmark/profile artifacts

Research-only classifier gates include:

- `lrtmw_margin_gate`
- `lrtmw_entropy_gate`
- `subject_threshold_floor`
- `ns2_aware_gate`
- `subject_floor_ns2_aware_gate`
- `weak_subject_guard`
- `frequency_specific_threshold_gate`
- `frequency_specific_logistic_gate`
- `conditional_frequency_specific_logistic_gate`
- `tenp5_ns2_hard_negative_veto`

These variants can be parsed for old report/profile inspection. Realtime startup
must reject them unless a future validation pass explicitly promotes one.

## Archive

Archive paths are historical snapshots only:

- `_archive/`
- `2026-03_realtime_ui_and_online_decoder/`
- `2026-04_async_fbcca_idle_decoder/`

Active code must not import from archive paths.

## Generated Or Local-Only

These paths are generated or local-only and should stay out of source review:

- `__pycache__/`
- `.tmp_role_test/`
- `.tmp_test_artifacts/`
- `.pytest_cache/`
- bulk remote report snapshots under `artifacts/remote_reports/`
- run outputs under `artifacts/runs/`
- runtime logs and temporary board captures

Small curated profile/report summaries may remain tracked when they document a
specific decision. Do not delete tracked evidence blindly.
