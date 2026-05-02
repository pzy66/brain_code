# Artifacts

This repository intentionally keeps large artifacts in Git for complete backup
and reproducibility. It is not optimized for a small clone.

## Tracked Artifact Categories

- `02_SSVEP/artifacts/runs`: formal local and imported SSVEP run results,
  reports, profile snapshots, and selection snapshots.
- `02_SSVEP/artifacts/datasets`: SSVEP collection bundles and external replay
  datasets used by the current experiments.
- `02_SSVEP/artifacts/deployed_profiles`: deployed or candidate SSVEP profiles.
- `02_SSVEP/_archive`: preserved legacy code and associated historical outputs.
- `hybrid_controller/models`: trained models needed by integrated vision/robot
  workflows.
- `artifacts`: repository-level generated indices and cross-workflow outputs.

## What Is Not Meant To Be Tracked

The following are cache or local temporary products and should remain ignored:

- `__pycache__`, `.pytest_cache`, `.pytest_tmp*`, `.tmp*`
- `pytest-cache-files-*`, `pytest_tmp*`, `pytest_temp*`, `tmp_pytest*`
- `02_SSVEP/artifacts/gpu_runtime/cupy_cache`
- `02_SSVEP/artifacts/gpu_runtime/tmp`
- smoke-test screenshots and local UI probe outputs

## Diagnostics

Use:

```powershell
powershell -ExecutionPolicy Bypass -File tools\diagnose_workspace.ps1
```

The diagnostic script reports tracked size, largest files, artifact category
counts, ignored entries, and permission-denied temporary directories.
