# 02_SSVEP Research Archive Index

This index separates research and abandoned recognition plans from the realtime
mainline. Keeping these records helps reproduce reports without suggesting that
the old variants are ready for online use.

## Current Realtime Mainline

The online mainline is:

```text
fbcca_ridge5
+ full_reference_bank
+ lrt_multiwindow_reject_gate
+ baseline_lrtmw
+ session no-control calibration
```

Default self-collection uses `8,10,12,15 Hz`. Runtime profiles must be selected
explicitly and must not overwrite `default_profile.json`.

## Research-Only Recognition Plans

### Stronger Global/Floor Gates

Variants:

- `subject_threshold_floor`
- `ns2_aware_gate`
- `subject_floor_ns2_aware_gate`
- `weak_subject_guard`
- global `min_enter=3`

Status: stopped for mainline use.

Reason: these can reduce NS2 false positives, but they make all command windows
harder to accept and tend to damage recall or 2.5 s response.

### Frequency-Specific Logistic / Conditional Gates

Variants:

- `frequency_specific_logistic_gate`
- `conditional_frequency_specific_logistic_gate`

Status: research-only.

Reason: high-risk smoke showed NS2 reduction came with excessive command TP loss.
The parser and benchmark runner still support these variants for trace analysis.

### 10.5 Hz NS2 Hard-Negative Veto

Variant:

- `tenp5_ns2_hard_negative_veto`

Status: stopped for full24 and runtime export.

Reason: short calibration had insufficient 10.5 Hz NS2 hard-negative support and
the trained veto mostly removed command TP rather than fixing NS2 FP.

### TDCA/TRCA/ECCA-Style Comparisons

Paths:

- `ssvep_core/tdca_local_opt.py`
- `ssvep_core/decoders/tdca_decoder.py`
- `ssvep_core/decoders/trca_r_decoder.py`
- `tools/compare_external_ssvep_fbcca_tdca.py`

Status: research comparison only.

Reason: current deployment work is fixed to FBCCA-ridge plus session no-control
calibration. TDCA/TRCA code remains useful for method comparisons.

## Runtime Guardrail

Research-only gate variants remain parseable for old benchmark artifacts, but the
realtime startup path rejects them by default. Promote a variant only after a new
validation pass updates this document, adds tests, and marks the profile as
runtime-safe.
