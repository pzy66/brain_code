# SSVEP Current Used References

This folder is the curated reference set for the current `02_SSVEP` mainline. It is included in the GitHub repository for team onboarding and report writing.

- Code stays in the normal source folders such as `02_SSVEP/`.
- Real datasets stay under `datasets/` or an external data disk according to `.gitignore`.
- Papers, source pages, DOI metadata, and dataset reference documents stay here.

## What Is Included

### 01_core_method_papers

Openly available PDFs that passed a `%PDF` check and are suitable to share in the repository.

- `2015_nakanishi_cca_comparison_plosone.pdf`: CCA-method comparison, template-assisted CCA family context.
- `2021_liu_tdca_tnsre_author_copy.pdf`: TDCA method paper from the author/open access source.
- `2023_carrara_pseudo_online_arxiv.pdf`: arXiv version of the pseudo-online evaluation framework paper.

Some important papers are not stored as PDFs because their publisher PDFs are access-controlled, the public endpoint did not return a valid PDF, or the shareability is unclear for a public GitHub repository. Their PubMed pages and DOI/Crossref metadata are still stored under `03_metadata` and `04_source_pages`.

### 02_dataset_references

Dataset source records and local documentation for the external SSVEP datasets currently used by the code.

- `BETA/`: Figshare record metadata plus local description files.
- `Wang2016/`: Zenodo record metadata plus local readme/channel-location files.

### 03_metadata

PubMed and Crossref JSON metadata for the current method references.

### 04_source_pages

Saved PubMed source pages for method papers whose PDF is not kept locally or where the PubMed page is useful for citation checking.

### 05_code_evidence

Evidence notes linking each reference to current code/docs.

## Core Bibliography

| Area | Reference | Local material |
|---|---|---|
| CCA baseline | Lin et al., frequency recognition based on CCA for SSVEP BCI; Bin et al., online multi-channel SSVEP BCI using CCA | PubMed/Crossref metadata |
| FBCCA | Chen et al., filter bank CCA for high-speed SSVEP BCI | PubMed/Crossref metadata |
| CCA variants / IT-CCA / eCCA | Nakanishi et al., comparison of CCA-based methods | PDF + metadata |
| TRCA | Nakanishi et al., TRCA for high-speed SSVEP speller | PubMed/Crossref metadata |
| TRCA-R / spatial filtering | Wong et al., spatial filtering in SSVEP-based BCIs | PubMed/Crossref metadata |
| TDCA | Liu et al., task-discriminant component analysis | PDF + PubMed/Crossref metadata |
| Async idle/control state | Zhang et al., idle-state detection for SSVEP BCIs | PubMed/Crossref metadata |
| Dynamic stopping | Nakanishi et al., dynamic stopping method for SSVEP BCIs | PubMed/Crossref metadata |
| Pseudo-online evaluation | Carrara and Papadopoulo, pseudo-online framework for BCI evaluation | PDF + PubMed/Crossref metadata |
| External dataset | BETA SSVEP database | Figshare metadata + description files |
| External dataset | Wang2016 SSVEP benchmark | Zenodo metadata + readme/channel docs |

## Maintenance Rule

For future work, add new SSVEP references here first, then update `bibliography.json` and `05_code_evidence/code_reference_evidence.md`. Avoid storing papers inside `brain_code` unless they are part of a tracked dataset release.


