"""Checks how much the two sources of ground truth agree.

The model is trained against self-computed binary masks (merged from the
per-instrument-class labels by scripts/data_prep/make_binary_masks.py)
but tested against the official EndoVis 2017 BinarySegmentation masks.
If those two sources disagree systematically, part of any train/test
gap is label inconsistency and not generalization.

For every test frame this computes the IoU between the self-computed mask
and the official mask. Sequences 1 and 2 had their official masks
replaced by the self-computed ones (scripts/data_prep/fix_test_masks.py),
so for those the comparison uses the backed-up original official masks.

Outputs results/metrics/mask_source_agreement.csv (per frame) and
results/tables/mask_source_agreement.{md,csv} (per sequence).
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from segmentation import paths
from segmentation.crop import crop_camera_view
from segmentation.dataset import (
    ALL_SEQUENCES,
    original_official_dir,
    test_frame_names,
)
from segmentation.metrics import iou
from segmentation.reporting import write_tables


def read_mask(path: Path) -> np.ndarray:
    """Loads a mask, cropped to the camera view, as a boolean array."""
    return np.array(crop_camera_view(Image.open(path).convert("L"))) > 127


def compare_sequence(
    dataset_root: Path, test_root: Path, sequence: int
) -> List[Dict]:
    """Per-frame IoU between self-computed and original official masks."""
    mine_dir = dataset_root / f"instrument_dataset_{sequence}" / "binary_masks"
    official_dir = original_official_dir(test_root, sequence)
    rows = []
    for name in test_frame_names(test_root, sequence):
        mine = read_mask(mine_dir / name)
        official = read_mask(official_dir / name)
        rows.append(
            {
                "sequence": sequence,
                "frame": name,
                "iou": iou(mine, official),
                "mine_frac": float(mine.mean()),
                "official_frac": float(official.mean()),
            }
        )
    return rows


def main() -> None:
    """Parses CLI arguments, compares all sequences, writes outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=paths.DATASET_ROOT
    )
    parser.add_argument("--test-root", type=Path, default=paths.TEST_ROOT)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=paths.METRICS_DIR / "mask_source_agreement.csv",
    )
    parser.add_argument("--table-dir", type=Path, default=paths.TABLES_DIR)
    args = parser.parse_args()

    rows: List[Dict] = []
    for n in ALL_SEQUENCES:
        rows.extend(compare_sequence(args.dataset_root, args.test_root, n))

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = []
    for n in ALL_SEQUENCES:
        seq = [r for r in rows if r["sequence"] == n]
        ious = np.array([r["iou"] for r in seq])
        mine = np.mean([r["mine_frac"] for r in seq])
        official = np.mean([r["official_frac"] for r in seq])
        table.append(
            [
                n,
                len(seq),
                f"{ious.mean():.3f}",
                f"{ious.min():.3f}",
                int((ious < 0.95).sum()),
                f"{mine:.1%}",
                f"{official:.1%}",
            ]
        )
    all_ious = np.array([r["iou"] for r in rows])
    table.append(
        [
            "All",
            len(rows),
            f"{all_ious.mean():.3f}",
            f"{all_ious.min():.3f}",
            int((all_ious < 0.95).sum()),
            f"{np.mean([r['mine_frac'] for r in rows]):.1%}",
            f"{np.mean([r['official_frac'] for r in rows]):.1%}",
        ]
    )
    write_tables(
        [
            "Sequence",
            "Frames",
            "Mean IoU (ours vs official)",
            "Min IoU",
            "Frames < 0.95",
            "Instrument area, ours",
            "Instrument area, official",
        ],
        table,
        args.table_dir / "mask_source_agreement.md",
        args.table_dir / "mask_source_agreement.csv",
    )
    print((args.table_dir / "mask_source_agreement.md").read_text())


if __name__ == "__main__":
    main()
