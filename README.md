# brain_code

`brain_code` is the formal Git repository for this workspace. The parent
directory is a local workspace for environment files, local datasets, PyCharm
settings, deliverables, and backups.

## Main Entrypoints

- Start here: [START_HERE.md](./START_HERE.md)
- Unified MI/SSVEP collection GUI: [run_unified_collection.py](./run_unified_collection.py)
- Unified collection package: [unified_collection](./unified_collection)
- Workspace path helpers: [brain_workspace](./brain_workspace)
- Hybrid Controller: [hybrid_controller/README.md](./hybrid_controller/README.md)
- SSVEP toolchain: [02_SSVEP/README.md](./02_SSVEP/README.md)
- MI collection and training: [01_MI/README.md](./01_MI/README.md)
- Setup: [docs/SETUP.md](./docs/SETUP.md)
- Code status: [docs/CODE_STATUS.md](./docs/CODE_STATUS.md)
- Artifacts policy: [docs/ARTIFACTS.md](./docs/ARTIFACTS.md)

## Repository Boundaries

- Run Git commands from this directory, not from the parent workspace.
- Keep real datasets, deployed profiles, formal runs, models, and archived
  experiment outputs in place.
- New MI and SSVEP collection data defaults stay inside `brain_code`:
  `01_MI/mi_classifier_latest/datasets/custom_mi` and
  `02_SSVEP/artifacts/datasets`.
- Keep generated caches, temporary pytest folders, smoke screenshots, and GPU
  compiler caches out of Git.

## Python Project Baseline

The repository now has a root `pyproject.toml` for pytest discovery and the
lightweight internal packages:

- `brain_workspace`: canonical paths, runtime import bootstrap, environment diagnostics.
- `unified_collection`: real implementation of the unified collection GUI.

Legacy scripts remain supported:

```powershell
tools\resolve_brain_python.cmd
$env:BRAIN_PYTHON_EXE = (& .\tools\resolve_brain_python.cmd)
& $env:BRAIN_PYTHON_EXE run_unified_collection.py
```

## Cleanup And Diagnostics

Use the cleanup script for regenerable files only:

```powershell
powershell -ExecutionPolicy Bypass -File tools\clean_workspace_temp.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File tools\clean_workspace_temp.ps1
```

Use the diagnostic script before larger repository maintenance:

```powershell
powershell -ExecutionPolicy Bypass -File tools\diagnose_workspace.ps1
```

## Common Checks

Run these from the repo root:

```powershell
$env:BRAIN_PYTHON_EXE = (& .\tools\resolve_brain_python.cmd)
& $env:BRAIN_PYTHON_EXE -m brain_workspace.environment
& $env:BRAIN_PYTHON_EXE -m pytest --collect-only -q -o addopts=
& $env:BRAIN_PYTHON_EXE -m py_compile unified_collection_ui.py run_unified_collection.py
& $env:BRAIN_PYTHON_EXE -m pytest tests -q -o addopts=
& $env:BRAIN_PYTHON_EXE -m pytest 02_SSVEP\tests\test_server_train_client_gpu_and_paths.py 02_SSVEP\tests\test_server_train_client_cuda_policy.py -q -o addopts=
```
