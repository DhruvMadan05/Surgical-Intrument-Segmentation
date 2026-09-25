"""Classical baseline: Otsu thresholding for instrument segmentation.

Instruments in this dataset tend to be desaturated metal against
saturated tissue, so thresholding the HSV saturation channel separates
them better than plain grayscale intensity. Otsu's method picks the
threshold automatically per frame; a small morphological open/close
cleans up salt-and-pepper noise from specular highlights.

Evaluates this baseline on the standard EndoVis 2017 test split
(instrument_dataset_1..8, last 75 frames per sequence) using IoU and
Dice, and saves results to results/baseline_threshold.json so the
fine-tuned model can be benchmarked against it later.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk

from segmentation.crop import crop_camera_view
from segmentation.dataset import train_test_split
from segmentation.metrics import evaluate_predictions


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


def save_sanity_overlays(
    pairs: List[Tuple[Path, Path]], out_dir: Path, num_examples: int = 6
) -> None:
    """Saves input/ground-truth/prediction overlays for a few frames.

    Samples evenly across `pairs` (rather than taking the first N) so
    the examples aren't all drawn from a single sequence.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(pairs) // num_examples)
    sampled = pairs[::step][:num_examples]

    for i, (frame_path, mask_path) in enumerate(sampled):
        image = np.array(crop_camera_view(Image.open(frame_path)))
        gt = np.array(crop_camera_view(Image.open(mask_path))) > 127
        pred = predict_mask(frame_path)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image)
        axes[0].set_title("input")
        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("ground truth")
        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("prediction")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"test_frame_{i}.png")
        plt.close(fig)


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

    _, test_pairs = train_test_split(args.dataset_root)
    if args.limit:
        test_pairs = test_pairs[: args.limit]

    results = evaluate_predictions(test_pairs, predict_mask)
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

    save_sanity_overlays(test_pairs, args.overlays_dir)
    print(f"Saved sanity overlays to {args.overlays_dir}")


if __name__ == "__main__":
    main()
