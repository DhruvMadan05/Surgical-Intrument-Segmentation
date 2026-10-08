"""Makes a qualitative comparison grid on randomly chosen test frames.

Each row is one randomly sampled test frame; the columns are the input
image, the ground truth, the Otsu baseline's output and the U-Net's
output. Frames are sampled uniformly from all 1,200 test frames with a
fixed seed (change --seed for a different draw), so the figure is
reproducible and not hand-picked.

Ground truth is the original official mask (see
segmentation.dataset.original_official_dir), so a sampled frame from
sequence 1 or 2 does not show the ultrasound probe as an instrument.

Saves results/figures/qualitative_comparison.png.
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from segmentation import paths
from segmentation.baseline import predict_mask
from segmentation.crop import crop_camera_view
from segmentation.dataset import (
    ALL_SEQUENCES,
    InstrumentSegDataset,
    original_official_dir,
    test_frame_names,
)
from segmentation.metrics import iou
from segmentation.model import get_device, load_model, make_predict_fn
from segmentation.plotting import apply_style

COLUMN_TITLES = ["Input", "Ground truth", "Otsu baseline", "U-Net"]


def sample_frames(dataset_root: Path, test_root: Path, count: int, seed: int):
    """Randomly samples `count` test frames as (sequence, frame_path, mask).

    Args:
        dataset_root: Directory containing instrument_dataset_N folders.
        test_root: Directory containing the official test masks.
        count: Number of frames to sample.
        seed: Seed for the random draw.

    Returns:
        List of (sequence number, frame path, ground-truth mask path).
    """
    candidates = []
    for n in ALL_SEQUENCES:
        frames_dir = dataset_root / f"instrument_dataset_{n}" / "left_frames"
        mask_dir = original_official_dir(test_root, n)
        for name in test_frame_names(test_root, n):
            candidates.append((n, frames_dir / name, mask_dir / name))
    return random.Random(seed).sample(candidates, count)


def main() -> None:
    """Parses CLI arguments, draws the figure, saves it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=paths.DATASET_ROOT
    )
    parser.add_argument("--test-root", type=Path, default=paths.TEST_ROOT)
    parser.add_argument(
        "--checkpoint", type=Path, default=paths.DEFAULT_CHECKPOINT
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.FIGURES_DIR / "qualitative_comparison.png",
    )
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.checkpoint, device)
    model_predict = make_predict_fn(
        model, device, InstrumentSegDataset.DEFAULT_SIZE
    )

    apply_style()
    samples = sample_frames(
        args.dataset_root, args.test_root, args.rows, args.seed
    )
    fig, axes = plt.subplots(
        args.rows, 4, figsize=(13, 2.9 * args.rows), squeeze=False
    )
    for row, (sequence, frame_path, mask_path) in enumerate(samples):
        image = np.array(
            crop_camera_view(Image.open(frame_path).convert("RGB"))
        )
        gt_image = crop_camera_view(Image.open(mask_path).convert("L"))
        gt = np.array(gt_image) > 127
        baseline = predict_mask(frame_path)
        pred = model_predict(frame_path)

        panels = [
            (image, None),
            (gt, None),
            (baseline, iou(baseline, gt)),
            (pred, iou(pred, gt)),
        ]
        for col, (panel, score) in enumerate(panels):
            ax = axes[row][col]
            ax.imshow(panel, cmap=None if panel.ndim == 3 else "gray")
            ax.axis("off")
            ax.grid(False)
            if row == 0:
                ax.set_title(COLUMN_TITLES[col], loc="center", fontsize=12)
            if score is not None:
                ax.text(
                    0.5,
                    -0.02,
                    f"IoU {score:.2f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=10,
                )
        axes[row][0].text(
            0.0,
            1.0,
            f"seq {sequence}, {frame_path.stem}",
            transform=axes[row][0].transAxes,
            va="bottom",
            fontsize=8,
            color="#52514e",
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    for sequence, frame_path, _ in samples:
        print(f"seq {sequence}, {frame_path.name}")
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
