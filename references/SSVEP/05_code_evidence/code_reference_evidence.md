# Code Evidence For Current SSVEP References

This note links the curated reference set to the current SSVEP codebase.

## Direct method/report references

- `brain_code/02_SSVEP/ssvep_core/async_fbcca_idle_standalone.py:7422-7430` exports method references into reports:
  - CCA baseline, PMID 17549911
  - FBCCA, PMID 26035476
  - TRCA, PMID 28436836
  - TRCA-R framework, PMID 32091986
  - TDCA, PMID 34543200
  - Async control-state detection, PMID 26246229
  - Dynamic stopping, PMID 26736447
  - Pseudo-online evaluation, PMID 38113535

## SSVEP README evidence

- `brain_code/02_SSVEP/README.md:16-17` lists TDCA and FBCCA local optimization entrypoints.
- `brain_code/02_SSVEP/README.md:280-283` documents the minimal CCA family and IT-CCA style methods.
- `brain_code/02_SSVEP/README.md:348-359` describes the literature-oriented short-pretrain method family and BETA/Wang runs.
- `brain_code/02_SSVEP/README.md:435-443` records current Wang2016 run status and warns against treating partial results as final.

## Dataset source evidence

- `brain_code/02_SSVEP/ssvep_core/external_beta_dataset.py:16-24` defines the BETA Figshare DOI/API, required channels, frequencies, sample rate, and local source root.
- `brain_code/02_SSVEP/ssvep_core/external_wang2016_dataset.py:16-59` defines the Wang2016 Zenodo record, required channels, target frequencies, sample rate, and selected 4-command frequency mapping.
- `brain_code/02_SSVEP/ssvep_core/external_ysuan_dataset.py` integrates the Zhao et al. 2024 YSU-an asynchronous SSVEP dataset adapter. It loads CS plus NS1/NS2/NS3 variables, applies baseline removal / 50 Hz notch / 250 Hz resampling, maps selected CS frequencies to command labels, and keeps NS subtypes available for no-control false-positive reporting.
- `brain_code/02_SSVEP/tools/run_external_short_pretrain_benchmark.py` supports `ysu_an` as an external dataset, `ysu_an_all8` as a candidate frequency source, YSU-an calibration/holdout split semantics, and NS subtype metrics (`ns1_fp_per_min`, `ns2_fp_per_min`, `ns3_fp_per_min`, `ns_all_fp_per_min`).

## Implementation evidence from explorer pass

- FBCCA is directly implemented and described as filter-bank CCA with harmonic fusion.
- CCA, IT-CCA, eCCA, MsetCCA, TRCA, TRCA-R, and TDCA are registered or benchmarked in the current SSVEP method family.
- Chen-style fixed/subband FBCCA weights are used in the training/evaluation configuration.
- `fbcca_lda5` and `fbcca_ridge5` are internal engineering comparison recipes, not separate literature references.
