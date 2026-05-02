# Code Status

The repository intentionally keeps complete project history visible. Use this
status map to choose the right entrypoint.

| Area | Status | Recommended Entry | Notes |
| --- | --- | --- | --- |
| `brain_workspace` | stable | `python -m brain_workspace.environment` | Shared path and environment diagnostics. |
| `unified_collection` | usable | `run_unified_collection.py` | Current unified MI/SSVEP collection GUI package. |
| `01_MI` | usable | `01_MI/mi_classifier_latest/code/README.md` | MI collection, training, and realtime code; some legacy compatibility remains. |
| `02_SSVEP` | usable | `02_SSVEP/START_SSVEP.py` | Active SSVEP collection, training, replay, and validation toolchain. |
| `hybrid_controller` | usable | `hybrid_controller/run_real.py` | Main integrated PC controller; hardware features require local devices. |
| `03_RobotArm_Control` | experimental | directory README/source files | Historical robot-arm experiments kept visible. |
| `04_Communication_And_Integration` | experimental | directory README/source files | Communication experiments and integration notes. |
| `05_Vision_Block_Recognition` | experimental | historical scripts | Older vision experiments; current model fallback uses `hybrid_controller/models/vision/best.pt`. |
| `06_Data_Collection` | experimental | historical scripts | Dataset collection tools and notes. |
| `07_Simulation_Lab` | experimental | directory README/source files | Simulation and sandbox work. |
| `02_SSVEP/_archive` | archive | none by default | Preserved historical code and outputs; not a default import target. |
| `artifacts`, `02_SSVEP/artifacts` | local-only data | `docs/ARTIFACTS.md` | Tracked outputs retained for backup/reproducibility, not a minimal package. |

Default user-facing entrypoints are documented in `START_HERE.md`. Directly
running experimental or archive scripts may require local paths, devices, or
older assumptions.
