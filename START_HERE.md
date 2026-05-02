# START HERE

This directory is the formal code repository. Daily development, commits,
branches, tests, and pushes should happen inside `brain_code`.

The parent directory is a local workspace for environment files, local datasets,
deliverables, and IDE settings.

## Common Entrypoints

- Unified collection GUI: [run_unified_collection.py](./run_unified_collection.py)
- Legacy unified collection wrapper: [unified_collection_ui.py](./unified_collection_ui.py)
- Hybrid Controller PC GUI: [hybrid_controller/run_real.py](./hybrid_controller/run_real.py)
- Hybrid Controller SSVEP GUI: [hybrid_controller/run_real_ssvep.py](./hybrid_controller/run_real_ssvep.py)
- JetMax runtime script: [hybrid_controller/robot/run_hybrid_controller_ros_runtime.sh](./hybrid_controller/robot/run_hybrid_controller_ros_runtime.sh)

## Structure To Know

- `brain_workspace`: shared path/bootstrap/environment helpers.
- `unified_collection`: real unified MI/SSVEP collection implementation.
- `01_MI`: MI collection, training, realtime inference, and shared helpers.
- `02_SSVEP`: SSVEP collection, training, replay, validation, and artifacts.
- `hybrid_controller`: integrated controller for MI, SSVEP, vision, and robot control.
- `docs`: setup, artifact policy, code status, and roadmap notes.

## Maintenance Commands

```powershell
cd <repo>
tools\resolve_brain_python.cmd
$env:BRAIN_PYTHON_EXE = (& .\tools\resolve_brain_python.cmd)
& $env:BRAIN_PYTHON_EXE -m brain_workspace.environment
& $env:BRAIN_PYTHON_EXE -m pytest --collect-only -q -o addopts=
powershell -ExecutionPolicy Bypass -File .\tools\clean_workspace_temp.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File .\tools\diagnose_workspace.ps1
git status --short --ignored
```

The cleanup script is intentionally scoped to caches and temporary products. It
does not remove formal datasets, deployed profiles, models, or run results.
