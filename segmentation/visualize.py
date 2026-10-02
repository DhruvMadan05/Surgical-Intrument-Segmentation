"""Shared qualitative overlay-saving for predicted segmentation masks.

Used by both the classical baseline and the trained-model evaluation so
the two produce directly comparable qualitative output.
"""

from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from segmentation.crop import crop_camera_view


def save_prediction_overlays(
    pairs: List[Tuple[Path, Path]],
    predict_fn: Callable[[Path], np.ndarray],
    out_dir: Path,
    num_examples: int = 6,
) -> None:
    """Saves input/ground-truth/prediction overlays for a sample of frames.

    Samples evenly across `pairs` (rather than taking the first N) so
    the examples aren't all drawn from a single sequence.

    Args:
        pairs: List of (frame_path, mask_path) tuples.
        predict_fn: Callable that takes a frame_path and returns a
            predicted binary mask, same (height, width) as the cropped
            ground-truth mask.
        out_dir: Directory to save overlay PNGs into.
        num_examples: Number of frames to sample and visualize.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(pairs) // num_examples)
    sampled = pairs[::step][:num_examples]

    for i, (frame_path, mask_path) in enumerate(sampled):
        image = np.array(crop_camera_view(Image.open(frame_path)))
        gt = np.array(crop_camera_view(Image.open(mask_path))) > 127
        pred = predict_fn(frame_path)

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
