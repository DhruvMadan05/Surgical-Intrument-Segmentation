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
