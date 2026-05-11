# MI References

This folder records open reference material for the motor-imagery classifier
part of the software-copyright V1.0 plan. Do not place licensed paper PDFs,
raw external datasets, or private EEG data in this directory.

## Dataset And Protocol References

- BCI Competition IV Dataset 2a / 2b are common public motor-imagery benchmark
  datasets used to evaluate multi-class MI decoding and session-transfer
  behavior.
- For this repository, external datasets are references for method alignment
  and benchmarking. They are not part of the soft-copyright source deposit.

## Method References

- CSP: Blankertz, Tomioka, Lemm, Kawanabe, and Mueller, "Optimizing Spatial
  Filters for Robust EEG Single-Trial Analysis," IEEE Signal Processing
  Magazine, 2008. DOI: https://doi.org/10.1109/MSP.2008.4408441
- FBCSP: Ang, Chin, Wang, Guan, and Zhang, "Filter Bank Common Spatial Pattern
  Algorithm on BCI Competition IV Datasets 2a and 2b," Frontiers in
  Neuroscience, 2012. DOI: https://doi.org/10.3389/fnins.2012.00039
- EEGNet: Lawhern, Solon, Waytowich, Gordon, Hung, and Lance, "EEGNet: a
  Compact Convolutional Neural Network for EEG-Based Brain-Computer
  Interfaces," Journal of Neural Engineering, 2018.
  DOI: https://doi.org/10.1088/1741-2552/aace8c
- ATCNet family: Altaheri, Muhammad, and Alsulaiman, "Physics-Informed
  Attention Temporal Convolutional Network for EEG-Based Motor Imagery
  Classification," IEEE Transactions on Industrial Informatics, 2023.
  DOI: https://doi.org/10.1109/TII.2022.3197419

## Engineering Boundary For This Repo

- V1.0 UI depends on a thin MI contract only. It does not import deep
  experimental scripts directly.
- The required MI integration outputs are:
  - training entry
  - realtime inference entry
  - `datasets/profiles/MI/current_mi_profile.json`
  - model/profile schema and smoke tests
- Real model weights, subject EEG data, and external benchmark data remain
  outside the software-copyright source deposit unless a later release
  explicitly designates a small baseline asset.
