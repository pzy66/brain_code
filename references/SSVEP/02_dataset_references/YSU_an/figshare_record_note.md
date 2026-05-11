# YSU-an Asynchronous SSVEP-BCI Dataset Reference

Source record:

- Figshare: https://figshare.com/articles/dataset/YSU-an_EEG_dataset_for_studying_asynchronous_SSVEP-BCI/24906300
- Dataset DOI: https://doi.org/10.6084/m9.figshare.24906300
- Article DOI: https://doi.org/10.1080/27706710.2024.2418650
- Article: Zhao, Zhang, Wang, Liu, Li, Fan, Liang, and Li, 2024, "An EEG dataset for studying asynchronous steady-state visual evoked potential (SSVEP) based brain computer interfaces"
- License: CC BY 4.0 according to the Figshare record and open-access article page.

Why this dataset matters for this project:

- It is explicitly designed for asynchronous SSVEP-BCI research.
- It contains control-state SSVEP data and three non-control-state tasks rather than only synchronous target trials.
- It is a strong candidate for evaluating the current `02_SSVEP` no-control rejection bottleneck, especially for `fbcca_ridge5 + full_reference_bank + lrt_multiwindow_reject_gate`.

Key protocol details from the article/source record:

- 24 subjects.
- 63 EEG channels.
- Control-state targets: 8, 9, 10.5, 11, 12, 13, 14, and 15 Hz.
- Control-state data variable: `data_CS`, shape `8 x 63 x 22500 x 12`.
- Non-control variables:
  - `data_NS1`, shape `63 x 20000 x 24`.
  - `data_NS2`, shape `63 x 20000 x 24`.
  - `data_NS3`, shape `63 x 10000 x 48`.
- Dataset file organization is described as one folder per subject, `S01` through `S24`, with MATLAB `.mat` files for CS, NS1, NS2, and NS3.

Integration note:

- Do not treat YSU-an as equivalent to BETA hard-noncommand idle. YSU-an has explicit NS tasks and is more suitable for true asynchronous no-control validation.
- A first adapter should map selected CS frequencies to command labels and map NS1/NS2/NS3 to separate idle/no-control evaluation modes.
