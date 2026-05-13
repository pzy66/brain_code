# 02_SSVEP Research Area

This directory is the marker for research-only SSVEP work.

The large legacy runners currently remain in `tools/` and `ssvep_core/` because
they are still covered by tests and used to reproduce old reports. Their status is
documented in `../docs/RESEARCH_ARCHIVE_INDEX.md`.

Research-only means:

- Useful for offline comparisons, report reproduction, and method audits.
- Not part of the daily collection -> session no-control -> realtime workflow.
- Not automatically runtime-loadable.
- Not allowed to overwrite `default_profile.json`.

When a future cleanup physically moves research scripts here, update imports,
entrypoints, tests, and the archive index in the same change.
