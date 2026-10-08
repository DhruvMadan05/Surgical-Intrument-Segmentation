"""Computes per-frame error statistics for the trained model.

Runs the model on every official test frame (all 10 sequences) and, for
each frame, records overlap scores and a breakdown of *what kind* of
errors were made. Outputs:
  - results/metrics/per_frame_errors.csv: one row per frame (see
    COLUMNS).
  - results/metrics/error_maps.npz: per-sequence spatial counts of
    ground-truth, false-positive and false-negative pixels (block-summed
    to a coarse grid) for plotting where errors occur.

Error pixels are bucketed by how far they sit from the other mask:
  - false positive (predicted instrument, truly background): distance to
    the nearest ground-truth instrument pixel;
  - false negative (missed instrument): distance to the nearest
    predicted instrument pixel (infinite if nothing was predicted).
Buckets are `boundary` (<= BOUNDARY_PX), `near`, and `gross`
(> GROSS_PX). Boundary errors are slightly misplaced edges, which is
partly annotation ambiguity (blur, polygon-labelled edges); gross errors
are whole regions that were hallucinated or missed.

Read the results with scripts/analysis/error_analysis.py.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from segmentation import paths
from segmentation.crop import crop_camera_view
from segmentation.dataset import InstrumentSegDataset, all_sequences_test_pairs
from segmentation.metrics import dice, iou
from segmentation.model import get_device, load_model, make_predict_fn

# Error pixels within this many pixels (at 1280x1024) of the other mask
# are "boundary" errors; beyond GROSS_PX they are "gross" errors.
BOUNDARY_PX = 5
GROSS_PX = 20

# Coarse grid for spatial error maps: 1024x1280 -> 256x320.
MAP_BLOCK = 4

COLUMNS = [
    "sequence",
    "frame",
    "iou",
    "dice",
    "gt_frac",
    "pred_frac",
    "fp_px",
    "fn_px",
    "fp_boundary",
    "fp_near",
    "fp_gross",
    "fn_boundary",
    "fn_near",
    "fn_gross",
]


def distance_buckets(
    error: np.ndarray, other_mask: np.ndarray
) -> Dict[str, int]:
    """Counts error pixels by distance to `other_mask`.

    Args:
        error: Boolean map of error pixels (false positives or negatives).
        other_mask: Boolean mask the error pixels are measured against
            (ground truth for false positives, prediction for false
            negatives).

    Returns:
        Dict with counts for "boundary", "near" and "gross".
    """
    if not error.any():
        return {"boundary": 0, "near": 0, "gross": 0}
    if other_mask.any():
        # Distance of every pixel to the nearest pixel inside other_mask.
        dist = distance_transform_edt(~other_mask)[error]
    else:
        dist = np.full(int(error.sum()), np.inf)
    return {
        "boundary": int((dist <= BOUNDARY_PX).sum()),
        "near": int(((dist > BOUNDARY_PX) & (dist <= GROSS_PX)).sum()),
        "gross": int((dist > GROSS_PX).sum()),
    }


def block_sum(mask: np.ndarray, block: int) -> np.ndarray:
    """Sums a boolean map over block x block cells (coarse spatial grid)."""
    h, w = mask.shape
    return (
        mask.reshape(h // block, block, w // block, block)
        .sum(axis=(1, 3))
        .astype(np.int32)
    )


def main() -> None:
    """Parses CLI arguments, scores every test frame, writes outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=paths.DATASET_ROOT
    )
    parser.add_argument("--test-root", type=Path, default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=paths.DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only the first N frames per sequence (quick sanity check)",
    )
    parser.add_argument("--out", type=Path, default=paths.PER_FRAME_ERRORS)
    parser.add_argument("--maps-out", type=Path, default=paths.ERROR_MAPS)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    model = load_model(args.checkpoint, device)
    predict_fn = make_predict_fn(
        model, device, InstrumentSegDataset.DEFAULT_SIZE
    )

    pairs_by_sequence = all_sequences_test_pairs(
        args.dataset_root, args.test_root
    )
    sequences = sorted(pairs_by_sequence)
    grid = (1024 // MAP_BLOCK, 1280 // MAP_BLOCK)
    maps = {
        name: np.zeros((len(sequences),) + grid, dtype=np.int32)
        for name in ("gt", "fp", "fn")
    }

    rows: List[list] = []
    for si, n in enumerate(sequences):
        pairs = pairs_by_sequence[n]
        if args.limit:
            pairs = pairs[: args.limit]
        for frame_path, mask_path in pairs:
            gt_img = crop_camera_view(Image.open(mask_path).convert("L"))
            gt = np.array(gt_img) > 127
            pred = predict_fn(frame_path)

            fp = pred & ~gt
            fn = gt & ~pred
            fp_b = distance_buckets(fp, gt)
            fn_b = distance_buckets(fn, pred)
            rows.append(
                [
                    n,
                    frame_path.name,
                    iou(pred, gt),
                    dice(pred, gt),
                    gt.mean(),
                    pred.mean(),
                    int(fp.sum()),
                    int(fn.sum()),
                    fp_b["boundary"],
                    fp_b["near"],
                    fp_b["gross"],
                    fn_b["boundary"],
                    fn_b["near"],
                    fn_b["gross"],
                ]
            )
            maps["gt"][si] += block_sum(gt, MAP_BLOCK)
            maps["fp"][si] += block_sum(fp, MAP_BLOCK)
            maps["fn"][si] += block_sum(fn, MAP_BLOCK)
        seq_ious = [r[2] for r in rows if r[0] == n]
        print(
            f"instrument_dataset_{n}: {len(seq_ious)} frames, "
            f"IoU={np.mean(seq_ious):.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    np.savez_compressed(
        args.maps_out,
        sequences=np.array(sequences),
        block=MAP_BLOCK,
        **maps,
    )
    print(f"Saved {len(rows)} rows to {args.out}")
    print(f"Saved error maps to {args.maps_out}")


if __name__ == "__main__":
    main()
