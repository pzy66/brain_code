# Setup

This repository is optimized for complete personal backup first and best-effort
reuse by other users second. Large tracked artifacts, trained models, run
reports, datasets, and archives are intentionally kept in Git.

## Recommended Layout

Clone the repository and work from the repo root:

```powershell
git clone https://github.com/pzy66/brain_code.git
cd brain_code
```

Use either a local `.venv`, a conda environment named `brain-vision`, or set
`BRAIN_PYTHON_EXE`:

```powershell
$env:BRAIN_PYTHON_EXE = (& .\tools\resolve_brain_python.cmd)
& $env:BRAIN_PYTHON_EXE -m pip install -e ".[dev,gui,ssvep]"
```

For Hybrid Controller and vision work:

```powershell
& $env:BRAIN_PYTHON_EXE -m pip install -e ".[hybrid]"
```

For MI training and realtime work:

```powershell
& $env:BRAIN_PYTHON_EXE -m pip install -e ".[mi]"
```

GPU acceleration is optional. Install CUDA/CuPy only on a matching local
machine:

```powershell
& $env:BRAIN_PYTHON_EXE -m pip install -e ".[gpu]"
```

## Existing Dependency Files

The optional dependencies in `pyproject.toml` are a convenience layer. The
older subsystem dependency files remain the source of detail for specialized
work:

- `01_MI/mi_classifier_latest/requirements*.txt`
- `02_SSVEP/environment.ssvep.yml`
- `hybrid_controller/requirements-hybrid-*.txt`
- `hybrid_controller/robot/requirements-jetmax-robot-python.txt`

JetMax ROS and `rospy` are expected to come from the official JetMax system
image, not from pip.

## Collection Data Locations

Collection entrypoints default to repo-local storage so a checkout can be
backed up and restored as one `brain_code` folder:

- MI collection: `01_MI/mi_classifier_latest/datasets/custom_mi`
- SSVEP collection: `02_SSVEP/artifacts/datasets`
- Unified MI/SSVEP index: `artifacts/unified_collection_index.csv`

Relative `--output-root` and `--dataset-dir` values are resolved under the
corresponding MI or SSVEP project directory. The GUI rejects accidental
collection output paths outside `brain_code`.

## Smoke Checks

From the repo root:

```powershell
tools\resolve_brain_python.cmd
$env:BRAIN_PYTHON_EXE = (& .\tools\resolve_brain_python.cmd)
& $env:BRAIN_PYTHON_EXE -m brain_workspace.environment
& $env:BRAIN_PYTHON_EXE -c "import brain_workspace.paths; import unified_collection.app"
& $env:BRAIN_PYTHON_EXE -m pytest --collect-only -q -o addopts=
```

Headless GUI tests use `QT_QPA_PLATFORM=offscreen` through test bootstrap code.
Real GUI usage still requires a desktop session.

## Hardware Notes

- BrainFlow boards, serial ports, JetMax robot networking, CUDA, cameras, and
  ROS are optional runtime integrations.
- Missing optional integrations should not prevent import smoke tests or test
  collection.
- Vision history scripts use `BRAIN_VISION_WEIGHTS` when set and otherwise
  fall back to `hybrid_controller/models/vision/best.pt`.
