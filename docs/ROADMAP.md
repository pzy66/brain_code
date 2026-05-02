# Roadmap

This roadmap separates completed work from usable-but-messy areas and open
cleanup. It is not a promise of public API stability.

## Completed Or Stable

- Formal Git boundary is `brain_code`.
- `brain_workspace` centralizes repository paths and environment diagnostics.
- `unified_collection` contains the real unified MI/SSVEP collection GUI, with
  legacy wrappers retained.
- CuPy compiler cache files are removed from Git tracking and ignored.
- Cleanup and diagnostic scripts exist for workspace maintenance.

## Usable But Still Needs Cleanup

- SSVEP training/replay code is functional but still has large modules and
  large tracked run reports.
- Hybrid Controller is usable but `hybrid_controller/app.py` remains a large
  integration surface.
- MI collection and training retain legacy manifest compatibility paths.
- Vision and data collection history is visible but not polished as a public
  package.

## Experimental Or Local-Only

- Robot/JetMax deployment paths are deployment defaults for a specific hardware
  environment.
- Remote SSVEP server paths under `/data1/zkx` are deployment defaults, not
  generic local defaults.
- Historical experiment directories are preserved for backup and reference.

## Next Cleanup Candidates

- Split SSVEP CLI/report/profile utilities out of the largest modules.
- Split Hybrid Controller UI shell from runtime wiring and hardware adapters.
- Add smaller public sample datasets for quick demos while keeping full tracked
  artifacts for backup.
- Reduce duplicated legacy reports after a separate backup/export decision.
