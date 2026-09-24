# Instrument Segmentation

Binary segmentation of robotic surgical instruments (RBE544).

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -r requirements-dev.txt  # optional, for formatting
```

## Dataset

Ground-truth masks from the source dataset are split per instrument class
(e.g. `Left_Prograsp_Forceps_labels`), not pre-merged into a single binary
mask. Run `scripts/make_binary_masks.py` after downloading the dataset into
`dataset/` to generate `binary_masks/` (foreground = any instrument-class
mask has a nonzero pixel) alongside each `instrument_dataset_N/`:

```bash
./.venv/bin/python scripts/make_binary_masks.py
```

## Baseline & model training

Both scripts below use the shared `segmentation/` package (train/test
split, IoU/Dice metrics) and must be run as modules from the repo root
so that package is importable:

```bash
# Classical Otsu-thresholding baseline, evaluated on the held-out test
# split (instrument_dataset_1..8, last 75 frames/sequence). Saves
# results/baseline_threshold.json.
./.venv/bin/python -m scripts.baseline_threshold

# Fine-tunes a U-Net (ResNet34 encoder) on the train split
# (instrument_dataset_1..8, first 225 frames/sequence). Saves
# checkpoints/unet_resnet34.pt, results/train_log.csv, and qualitative
# overlays in results/sanity_overlays/.
./.venv/bin/python -m scripts.train_model
```

Both accept `--dataset-root` and other flags; run with `--help` to see
them. The train/test split covers only `instrument_dataset_1..8`;
`instrument_dataset_9` and `_10` are reserved as full 300-frame
held-out sequences for a later stretch-goal evaluation.

## Code style

- **Python**: [PEP 8](https://peps.python.org/pep-0008/). Formatted with
  `black`, configured in `pyproject.toml` (installed via
  `requirements-dev.txt`). Run before committing:
  ```bash
  ./.venv/bin/python -m black .
  ```

- **C++**: [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
  Formatted with `clang-format` (`.clang-format` at the repo root sets
  `BasedOnStyle: Google`). Run before committing:
  ```bash
  clang-format -i path/to/file.cpp path/to/file.h
  ```
