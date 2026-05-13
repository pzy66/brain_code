# 02_SSVEP Apps

PyQt applications for the active SSVEP workflow.

## Daily Apps

- `launcher_ui.py`: unified local launcher.
- `data_collection_ui.py`: custom SSVEP collection with command, no-control, and pseudo-online stages.
- `realtime_online_ui.py`: realtime decoder UI. It rejects research-only classifier gate variants by default.
- `training_evaluation_ui.py`: training/evaluation job UI.
- `async_fbcca_validation_ui.py`: shared fullscreen stimulus widget and validation helpers.

## Current Daily Path

```text
Data Collection
-> session no-control profile training
-> Realtime Decode
```

The default collection/realtime frequencies are `8,10,12,15 Hz`.

## Research-Only UI Paths

Some buttons or presets still expose older optimization tasks, including TDCA local
optimization and external replay. Treat these as method-comparison tools, not as
the current online classifier path. See `../docs/RESEARCH_ARCHIVE_INDEX.md`.
