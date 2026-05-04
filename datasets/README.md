# Local Datasets

This directory is the default local data root for a cloned checkout.

Real datasets, trained models, profiles, and run outputs are not tracked in Git.
Copy them here when needed, or set `BRAIN_DATA_ROOT` to another directory.

Expected layout:

- `MI/`: motor-imagery collection and training datasets.
- `SSVEP/`: SSVEP collection bundles and external replay datasets.
- `vision/`: camera calibration files, YOLO datasets, and local vision models.
- `profiles/`: deployed MI, SSVEP, and hybrid-controller runtime profiles.

For example, a vision model can live at:

```text
datasets/vision/models/best.pt
```

An SSVEP current profile can live at:

```text
datasets/profiles/SSVEP/current_fbcca_profile.json
```
