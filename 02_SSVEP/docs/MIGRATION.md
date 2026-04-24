# 02_SSVEP Migration

## What Changed

- The active SSVEP code now lives under functional folders:
  - `apps/`
  - `ssvep_core/`
  - `tools/`
  - `entrypoints/`
  - `artifacts/`
  - `docs/`
- Legacy time-stamped directories move to `_archive/`.
- New code must not import `_archive`.
- Local and remote outputs are stored by run:
  - `artifacts/runs/local/<task>/<YYYYMMDD>/<run_id>/`
  - `artifacts/runs/remote/<task>/<YYYYMMDD>/<run_id>/`
- Active deployed profiles live in `artifacts/deployed_profiles/`.

## Main Entry

- Top-level launcher: `START_SSVEP.py`
- Thin entrypoints:
  - `entrypoints/start_collection.py`
  - `entrypoints/start_realtime.py`
  - `entrypoints/start_training_eval.py`
  - `entrypoints/start_tdca_local_opt.py`

## Artifact Rules

Each run directory keeps:

- `report.json`
- `report.md`
- `profile.json` when a deployable profile is produced
- `profile_v2.json` when available
- `run_config.json`
- `selection_snapshot.json`
- `progress_snapshot.json`
- `run.log`
- `figures/`

`artifacts/deployed_profiles/profile_index.json` records which run produced the currently deployed profile copy.

## Legacy Migration

Use `tools/migrate_legacy_artifacts.py` to copy legacy datasets, default profiles, and historical report trees into the new artifact layout.
