"""Classical baseline: Otsu thresholding for instrument segmentation.

Instruments in this dataset tend to be desaturated metal against
saturated tissue, so thresholding the HSV saturation channel separates
them better than plain grayscale intensity. Otsu's method picks the
threshold automatically per frame; a small morphological open/close
cleans up salt-and-pepper noise from specular highlights.

Evaluates this baseline separately on each of the 10 sequences' official
test frames (instrument_dataset_1..8: last 75 frames per sequence;
9..10: all 300 frames each, since they have no training portion at
all), scored against the official BinarySegmentation masks -- see
scripts/download_test_masks.py -- using the same segmentation.metrics
code as the fine-tuned model's evaluate.py, so the two are directly
comparable. Reports a per-sequence breakdown plus an overall mean
across all 10 sequences, and saves results to
results/baseline_threshold.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk

from segmentation.crop import crop_camera_view
from segmentation.dataset import all_sequences_test_pairs
from segmentation.metrics import evaluate_predictions
from segmentation.visualize import save_prediction_overlays


def predict_mask(frame_path: Path) -> np.ndarray:
    """Segments an instrument mask via Otsu thresholding on saturation.

    Operates on the cropped 1280x1024 camera view only.
    """
    image = np.array(crop_camera_view(Image.open(frame_path).convert("HSV")))
    saturation = image[:, :, 1]
    thresh = threshold_otsu(saturation)
    mask = saturation < thresh  # low saturation -> instrument
    mask = binary_opening(mask, disk(3))
    mask = binary_closing(mask, disk(3))
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "dataset" / "training",
        help="Directory containing instrument_dataset_N folders",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=None,
        help=(
            "Directory with official test masks (default: a 'test' "
            "sibling of --dataset-root)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N frames per sequence "
            "(quick sanity check)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "results" / "baseline_threshold.json",
        help="Where to save the results JSON",
    )
    parser.add_argument(
        "--overlays-dir",
        type=Path,
        default=repo_root / "results" / "baseline_overlays",
        help="Where to save qualitative input/gt/prediction overlays",
    )
    args = parser.parse_args()

    pairs_by_sequence = all_sequences_test_pairs(
        args.dataset_root, args.test_root
    )
    if args.limit:
        pairs_by_sequence = {
            n: pairs[: args.limit] for n, pairs in pairs_by_sequence.items()
        }

    per_sequence = {}
    all_pairs = []
    for n in sorted(pairs_by_sequence):
        pairs = pairs_by_sequence[n]
        all_pairs.extend(pairs)
        results = evaluate_predictions(pairs, predict_mask)
        per_sequence[n] = results
        print(
            f"instrument_dataset_{n}: {results['num_frames']} frames, "
            f"IoU={results['mean_iou']:.4f}, "
            f"Dice={results['mean_dice']:.4f}"
        )

    # Two ways to summarize "overall": averaging the 10 per-sequence
    # means weights every sequence equally regardless of frame count;
    # pooling scores every one of the 1200 frames once and averages
    # those directly. Both are reported since they can diverge when
    # frame counts differ as much as they do here (75 vs. 300).
    mean_of_sequences_iou = float(
        np.mean([r["mean_iou"] for r in per_sequence.values()])
    )
    mean_of_sequences_dice = float(
        np.mean([r["mean_dice"] for r in per_sequence.values()])
    )
    pooled = evaluate_predictions(all_pairs, predict_mask)
    print(
        f"\nOverall (mean of 10 sequence means): "
        f"IoU={mean_of_sequences_iou:.4f}, "
        f"Dice={mean_of_sequences_dice:.4f}"
    )
    print(
        f"Overall (pooled across all {pooled['num_frames']} frames): "
        f"IoU={pooled['mean_iou']:.4f}, Dice={pooled['mean_dice']:.4f}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "method": "otsu_saturation_threshold",
                "per_sequence": {
                    str(n): {
                        "mean_iou": r["mean_iou"],
                        "mean_dice": r["mean_dice"],
                        "num_frames": r["num_frames"],
                    }
                    for n, r in per_sequence.items()
                },
                "overall_mean_of_sequences": {
                    "mean_iou": mean_of_sequences_iou,
                    "mean_dice": mean_of_sequences_dice,
                },
                "overall_pooled": {
                    "mean_iou": pooled["mean_iou"],
                    "mean_dice": pooled["mean_dice"],
                    "num_frames": pooled["num_frames"],
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {args.out}")

    save_prediction_overlays(all_pairs, predict_mask, args.overlays_dir)
    print(f"Saved overlays to {args.overlays_dir}")


if __name__ == "__main__":
    main()
