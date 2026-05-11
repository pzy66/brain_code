# Soft Copyright Workbench UI

This folder contains a standalone PyQt workbench for the planned software
copyright V1.0 demonstration and material-preparation workflow.

The goal is to present the whole system as one coherent application:

```text
data acquisition -> model training/evaluation -> profile publishing
-> online MI/SSVEP recognition -> visual target resolution
-> JetMax move/pick/place -> logs, reports, replay, copyright materials
```

The workbench is intentionally separate from the existing `01_MI`, `02_SSVEP`,
and `hybrid_controller` runtime code. It gives us a stable software-copyright
UI shell without disturbing the current research and robot debugging worktree.
All first-stage actions are safe demo actions: open a directory, locate a file,
or show a command that the developer can run manually.

## Run

From the repository root:

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py .\08_SoftCopyright_UI\run_softcopyright_workbench.py
```

For headless screenshot verification:

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py .\08_SoftCopyright_UI\run_softcopyright_workbench.py --screenshot .\08_SoftCopyright_UI\artifacts\workbench.png
```

Some Windows/Qt offscreen environments render text poorly. If that happens,
use the deterministic static preview renderer:

```powershell
C:\Users\P1233\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  .\08_SoftCopyright_UI\tools\render_static_preview.py `
  --output .\08_SoftCopyright_UI\artifacts\workbench_static_preview.png
```

## UI Scope

The first version contains six work surfaces:

- Overview: V1.0 pipeline, subsystem readiness, next implementation work.
- Acquisition: MI and SSVEP session setup, dataset paths, protocol summary.
- Training: MI, SSVEP, and vision model readiness plus publish gates.
- Online Control: MI/SSVEP/keyboard arbitration and realtime indicators.
- Vision + Robot: camera, target slots, safety ladder, robot action state.
- Copyright Kit: source/document/test/reference checklist for submission.

## Demo State Source

The UI reads repository status through `softcopyright_workbench.state` and the
MI adapter contract in `softcopyright_workbench.mi_contract`. It checks:

- repository root and dataset/profile roots
- MI collection, training, realtime, schema, and profile paths
- SSVEP launcher and current profile paths
- default vision model and vision-grasp profile
- `hybrid_controller` entry/config paths
- material drafts under `docs/softcopyright/`

The UI does not import BrainFlow, JetMax, ROS, camera drivers, or deep training
scripts on startup.

## MI Adapter Contract

When the new MI classifier is merged, keep the UI dependency thin. The required
stable outputs are:

- training entry
- realtime inference entry
- profile/model output entry
- `datasets/profiles/MI/current_mi_profile.json`
- schema-compatible profile metadata

The current schema draft lives at
`08_SoftCopyright_UI/schemas/mi_profile.schema.json`. Real subject weights and
training artifacts should normally stay out of Git.

## Soft-Copyright Materials

The material draft directory is `docs/softcopyright/`:

- `software_manual.md`
- `user_manual.md`
- `test_report.md`
- `source_deposit_scope.md`
- `version_notes.md`

The "Copyright Kit" page reads these files directly and shows whether they are
present before V1.0 freeze.

## Integration Plan

This folder is the UI shell. Later integration should happen in small steps:

1. Add the new MI classifier as a normal module with training, realtime
   inference, profile output, tests, and references.
2. Wire the MI classifier status into this dashboard.
3. Reuse existing SSVEP profile/report/status files instead of duplicating
   SSVEP logic.
4. Reuse `hybrid_controller` robot and vision runtime APIs for live operation.
5. Keep a hardware-free demo mode for software-copyright screenshots and
   review.

## Validation

Recommended first-stage checks:

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py -m py_compile `
  .\08_SoftCopyright_UI\run_softcopyright_workbench.py `
  .\08_SoftCopyright_UI\softcopyright_workbench\app.py `
  .\08_SoftCopyright_UI\softcopyright_workbench\state.py `
  .\08_SoftCopyright_UI\softcopyright_workbench\mi_contract.py `
  .\08_SoftCopyright_UI\tools\render_static_preview.py

& $py .\08_SoftCopyright_UI\run_softcopyright_workbench.py --screenshot .\08_SoftCopyright_UI\artifacts\workbench.png
& $py .\08_SoftCopyright_UI\tools\render_static_preview.py --output .\08_SoftCopyright_UI\artifacts\workbench_static_preview.png
& $py -m brain diagnose
& $py -m brain launch --simulate
```

## Reference Notes

The materials used to shape this UI are recorded in:

- `docs/REFERENCES.md`
- `..\references\SoftCopyright_UI\README.md`
