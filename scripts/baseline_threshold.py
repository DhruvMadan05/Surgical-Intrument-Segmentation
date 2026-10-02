"""Classical baseline: Otsu thresholding for instrument segmentation.

Instruments in this dataset tend to be desaturated metal against
saturated tissue, so thresholding the HSV saturation channel separates
them better than plain grayscale intensity. Otsu's method picks the
threshold automatically per frame; a small morphological open/close
cleans up salt-and-pepper noise from specular highlights.

Evaluates this baseline on the official EndoVis 2017 test split
(instrument_dataset_1..8, last 75 frames per sequence, scored against
the downloaded official BinarySegmentation masks -- see
scripts/download_test_masks.py) using IoU and Dice, and saves results
to results/baseline_threshold.json so the fine-tuned model can be
benchmarked against it later on the same masks.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk

from segmentation.crop import crop_camera_view
from segmentation.dataset import train_test_split
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
        help="Evaluate only the first N test frames (quick sanity check)",
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

    _, test_frame_pairs = train_test_split(args.dataset_root, args.test_root)
    if args.limit:
        test_frame_pairs = test_frame_pairs[: args.limit]

    results = evaluate_predictions(test_frame_pairs, predict_mask)
    print(
        f"Evaluated {results['num_frames']} test frames: "
        f"mean IoU={results['mean_iou']:.4f}, "
        f"mean Dice={results['mean_dice']:.4f}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "method": "otsu_saturation_threshold",
                "mean_iou": results["mean_iou"],
                "mean_dice": results["mean_dice"],
                "num_frames": results["num_frames"],
            },
            f,
            indent=2,
        )
    print(f"Saved results to {args.out}")

    save_prediction_overlays(test_frame_pairs, predict_mask, args.overlays_dir)
    print(f"Saved sanity overlays to {args.overlays_dir}")


if __name__ == "__main__":
    main()
