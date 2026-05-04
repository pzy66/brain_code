# Local Datasets

This directory is the default local data root for a cloned checkout.

Real datasets, extra trained models, profiles, and run outputs are not tracked in Git.
The default vision model at `vision/models/best.pt` is tracked so a fresh clone can start the vision path with the same baseline weight. Copy additional assets here when needed, or set `BRAIN_DATA_ROOT` to another directory.

Expected layout:

- `MI/`: motor-imagery collection and training datasets.
- `SSVEP/`: SSVEP collection bundles and external replay datasets.
- `vision/`: camera calibration files, YOLO datasets, and local vision models.
- `profiles/`: deployed MI, SSVEP, and hybrid-controller runtime profiles.

The default vision model lives at:

```text
datasets/vision/models/best.pt
```

An SSVEP current profile can live at:

```text
datasets/profiles/SSVEP/current_fbcca_profile.json
```
