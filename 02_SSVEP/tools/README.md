# 02_SSVEP Tools

Command-line and server-oriented tools. These are not the primary local operator
interface; use `START_SSVEP.py` for daily collection and realtime work.

## Current Support Tools

- `training_evaluation_cli.py`: CLI entrypoint for local/server training and evaluation.
- `server_train_client.py`: remote job sync/submit/download helper.
- `prepare_server_layout.py`: prepares the expected remote directory layout.
- `export_classifier_candidate_profile.py`: converts validated candidate artifacts into profile JSON.
- `migrate_legacy_artifacts.py`: moves older artifacts into the current artifact layout.

## Research-Only Tools

- `run_external_short_pretrain_benchmark.py`: external YSU-an/Wang/BETA benchmark runner and NS2 gate experiments.
- `run_external_frequency_server_sweep.py`: external frequency sweep runner.
- `compare_external_ssvep_fbcca_tdca.py`: lightweight FBCCA/TDCA external comparison.

Research-only tools may still parse abandoned gate variants for reproducibility.
They must not publish or overwrite `default_profile.json` without an explicit
validated export step.

## Remote Boundary

Remote SSVEP jobs must write only under:

```text
/data1/zkx/brain/ssvep/
```

Do not store passwords in commands, scripts, logs, or reports.
