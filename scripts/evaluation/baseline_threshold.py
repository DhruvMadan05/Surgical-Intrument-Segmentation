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
scripts/data_prep/download_test_masks.py -- using the same
segmentation.metrics code as the fine-tuned model's evaluate_model.py, so
the two are directly comparable. Reports a per-sequence breakdown plus
overall means across all 10 sequences, and saves results to
results/metrics/baseline_threshold.json.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk

from segmentation import paths
from segmentation.crop import crop_camera_view
from segmentation.dataset import all_sequences_test_pairs
from segmentation.metrics import evaluate_by_sequence, save_results
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
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=paths.DATASET_ROOT,
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
        default=paths.BASELINE_METRICS,
        help="Where to save the results JSON",
    )
    parser.add_argument(
        "--overlays-dir",
        type=Path,
        default=paths.OVERLAYS_DIR / "baseline",
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

    summary = evaluate_by_sequence(
        pairs_by_sequence, predict_mask, method="otsu_saturation_threshold"
    )
    save_results(summary, args.out)

    all_pairs = [
        p for n in sorted(pairs_by_sequence) for p in pairs_by_sequence[n]
    ]
    save_prediction_overlays(all_pairs, predict_mask, args.overlays_dir)
    print(f"Saved overlays to {args.overlays_dir}")


if __name__ == "__main__":
    main()
