# 02_SSVEP

`02_SSVEP` is the active SSVEP workspace for local data collection, session-level
calibration, and realtime online decoding. The daily path is intentionally narrow:

1. Collect a custom session with command and no-control blocks.
2. Train a session no-control FBCCA-ridge profile from calibration data only.
3. Start realtime decoding with an explicitly selected runtime-safe profile.

Historical external-dataset sweeps, TDCA/TRCA comparisons, and failed NS2 gate
experiments are kept for research traceability, but they are not the current
deployment path.

## Daily Workflow

Use the launcher when operating locally:

```powershell
py -3.11 02_SSVEP\START_SSVEP.py
```

The daily buttons are:

- `Data Collection`: collect `custom_ssvep_command_nc_pseudoonline_v1` or the short variant.
- `Realtime Decode`: load a runtime-safe profile and run online decisions.
- `Training Eval`: inspect or run training/evaluation jobs.
- `FBCCA Pretrain`: build FBCCA threshold profiles from local collection data.

The default self-collection and realtime stimulus frequencies are:

```text
8, 10, 12, 15 Hz
```

## Current Mainline

The current practical classifier path is:

```text
custom session collection
-> session_manifest.json + raw_trials.npz
-> fbcca_ridge5 score classifier
-> full_reference_bank features
-> lrt_multiwindow_reject_gate
-> session no-control calibration
-> realtime profile selected by the operator
```

The mainline keeps these constraints:

- Do not overwrite `default_profile.json` automatically.
- Do not use pseudo-online test trials for training.
- Do not use abandoned research gates for realtime startup.
- Keep channels, frequency profile, and profile provenance visible in the UI.

## Research And Archive

Research-only code and reports are documented in:

- `docs/SSVEP_CLEANUP_INVENTORY.md`
- `docs/RESEARCH_ARCHIVE_INDEX.md`
- `docs/ARCHIVE_INDEX.md`

Examples of research-only material:

- YSU-an / Wang2016 / BETA external benchmark runners.
- TDCA/TRCA comparison and local optimization paths.
- NS2 gate variants that reduced false positives by rejecting too many command TPs.
- 10.5 Hz hard-negative veto experiments that were not supported by short calibration.

These paths may still be useful for reproducing old reports, but they should not be
presented as the day-to-day realtime classifier.

## Directory Map

```text
02_SSVEP/
  START_SSVEP.py      Unified local launcher.
  apps/               PyQt collection, realtime, and training/evaluation UIs.
  entrypoints/        Thin startup wrappers.
  ssvep_core/         Dataset, decoder, gate, profile, and runtime code.
  tools/              CLI/server/research tooling.
  docs/               Workflow, cleanup, and archive documentation.
  tests/              Regression tests for active and research paths.
  artifacts/          Local profiles/reports; mostly ignored generated output.
  _archive/           Historical snapshots only; active code must not import it.
```

## Verification

Before treating a cleanup or classifier change as safe, run:

```powershell
py -3.11 -m py_compile 02_SSVEP\apps\data_collection_ui.py 02_SSVEP\apps\realtime_online_ui.py 02_SSVEP\ssvep_core\dataset.py 02_SSVEP\ssvep_core\session_no_control_classifier.py
py -3.11 -m pytest 02_SSVEP\tests -q
git diff --check -- 02_SSVEP
```
