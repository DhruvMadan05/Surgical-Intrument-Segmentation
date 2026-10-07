"""Replaces mislabeled official test masks for sequences 1 and 2.

Spot-checks (see segmentation/dataset.py's module docstring) found that
the official BinarySegmentation test masks for instrument_dataset_1 and
instrument_dataset_2 omit a real, visible instrument on some frames --
our self-computed binary_masks/ (scripts/data_prep/make_binary_masks.py) always
agreed with or exceeded the official masks, never fell short of them,
confirming it's an omission in the official release rather than an
error in our merge logic.

This script replaces exactly the 75 official test mask files for
sequences 1 and 2 (the ones matching a frame name actually in that
sequence's test set, e.g. frame225.png..frame299.png) with our
self-computed equivalents. Originals are backed up first to a sibling
*_official_backup/ directory so the change is reversible and the
discrepancy stays inspectable.

Re-running scripts/data_prep/download_test_masks.py later would re-fetch the
original (mislabeled) files from HuggingFace and silently undo this,
so re-run this script afterward if that happens.
"""

import argparse
import shutil
from pathlib import Path

from segmentation import paths
from segmentation.dataset import official_mask_dir, test_frame_names

FIXED_SEQUENCES = (1, 2)


def fix_sequence(dataset_root: Path, test_root: Path, sequence: int) -> int:
    """Backs up and replaces one sequence's official test masks.

    Args:
        dataset_root: Directory containing instrument_dataset_N folders
            (e.g. dataset/training), source of our self-computed masks.
        test_root: Directory containing the downloaded official test
            masks (e.g. dataset/test).
        sequence: Sequence number to fix.

    Returns:
        The number of files replaced.
    """
    mask_dir = official_mask_dir(test_root, sequence)
    backup_dir = mask_dir.parent / f"{mask_dir.name}_official_backup"
    backup_dir.mkdir(exist_ok=True)

    source_dir = (
        dataset_root / f"instrument_dataset_{sequence}" / "binary_masks"
    )

    replaced = 0
    for name in test_frame_names(test_root, sequence):
        source = source_dir / name
        if not source.exists():
            raise FileNotFoundError(
                f"Missing self-computed mask {source}; run "
                "scripts/data_prep/make_binary_masks.py first."
            )

        official = mask_dir / name
        backup = backup_dir / name
        if not backup.exists():
            shutil.copy2(official, backup)

        shutil.copy2(source, official)
        replaced += 1

    return replaced


def main() -> None:
    """Parses CLI arguments and fixes sequences 1 and 2's test masks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=paths.DATASET_ROOT
    )
    parser.add_argument("--test-root", type=Path, default=paths.TEST_ROOT)
    args = parser.parse_args()

    for n in FIXED_SEQUENCES:
        replaced = fix_sequence(args.dataset_root, args.test_root, n)
        print(f"instrument_dataset_{n}: replaced {replaced} test masks")


if __name__ == "__main__":
    main()
