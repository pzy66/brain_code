# 06 Data Collection

This folder contains standalone dataset collection helpers that do not belong
inside the MI or SSVEP mainline folders.

## Current content

- `2026-04_jetmax_block_dataset_collection/`: JetMax block image/data collection
  support for the vision and grasp workflow.

## Boundary

- MI-specific collection stays in `01_MI`.
- SSVEP-specific collection stays in `02_SSVEP`.
- Vision-only image collection may live here or in `05_Vision_Block_Recognition`
  when it is tightly coupled to the vision training scripts.
- Runtime integration and robot command debugging belong in
  `04_Communication_And_Integration` until promoted into `hybrid_controller`.
