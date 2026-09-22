"""Collapses per-instrument-class ground truth into single binary masks.

Each per-instrument-class mask encodes instrument parts as {0, 10, 20, 30,
...}. This script collapses them into one binary mask per frame:
  foreground (255) = any instrument-class mask has a nonzero pixel there
  background (0)   = otherwise

Input layout (per instrument_dataset_N):
  ground_truth/<Instrument_Class>_labels/frameNNN.png
    (one or more class folders)

Output:
  binary_masks/frameNNN.png (uint8, values in {0, 255})
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def merge_dataset(
    dataset_dir: Path,
    out_subdir: str = "binary_masks",
    overwrite: bool = False,
) -> int:
    """Merges one instrument_dataset's per-class masks into binary masks.

    Args:
        dataset_dir: Path to an instrument_dataset_N directory containing a
            ground_truth/ subdirectory with one folder per instrument class.
        out_subdir: Name of the subdirectory (under dataset_dir) to write
            merged binary masks into.
        overwrite: If True, recompute masks that already exist in
            out_subdir. If False, skip frames whose output already exists.

    Returns:
        The number of binary mask files written.
    """
    gt_dir = dataset_dir / "ground_truth"
    if not gt_dir.is_dir():
        return 0

    class_dirs = [d for d in gt_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        return 0

    # Collect every frame filename that appears in at least one class folder.
    frame_names = sorted(
        {p.name for d in class_dirs for p in d.glob("frame*.png")}
    )

    out_dir = dataset_dir / out_subdir
    out_dir.mkdir(exist_ok=True)

    written = 0
    for frame_name in frame_names:
        out_path = out_dir / frame_name
        if out_path.exists() and not overwrite:
            continue

        combined = None
        for class_dir in class_dirs:
            frame_path = class_dir / frame_name
            if not frame_path.exists():
                continue
            arr = np.array(Image.open(frame_path))
            # Some frames are stored as RGB even though they're
            # single-channel label maps; collapse to 2D before thresholding.
            mask = arr.any(axis=-1) if arr.ndim == 3 else arr > 0
            combined = mask if combined is None else (combined | mask)

        if combined is None:
            continue

        Image.fromarray((combined * 255).astype(np.uint8)).save(out_path)
        written += 1

    return written


def main() -> None:
    """Generates binary masks for every instrument_dataset_N in the root."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = (
        Path(__file__).resolve().parent.parent / "dataset" / "training"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_root,
        help=(
            "Directory containing instrument_dataset_N folders "
            "(default: dataset/training)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute masks that already exist",
    )
    args = parser.parse_args()

    dataset_dirs = sorted(
        p
        for p in args.dataset_root.iterdir()
        if p.is_dir() and p.name.startswith("instrument_dataset_")
    )
    if not dataset_dirs:
        print(
            f"No instrument_dataset_* folders found under {args.dataset_root}"
        )
        return

    total = 0
    for dataset_dir in dataset_dirs:
        n = merge_dataset(dataset_dir, overwrite=args.overwrite)
        print(
            f"{dataset_dir.name}: wrote {n} binary masks -> "
            f"{dataset_dir.name}/binary_masks/"
        )
        total += n

    print(f"Done. {total} binary masks written total.")


if __name__ == "__main__":
    main()
