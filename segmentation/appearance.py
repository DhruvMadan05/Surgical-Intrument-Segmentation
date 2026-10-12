"""Simple appearance statistics of the labelled instrument pixels.

Used by the error analysis to ask whether the model fails on instruments
that look different from what it was trained on, e.g. unusually bright,
near-white instrument bodies when most training instruments are dark
metal.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from segmentation.crop import crop_camera_view

# HSV channels are 0-255 in PIL. "Bright and desaturated" = near white.
BRIGHT_V_MIN = 200
DESATURATED_S_MAX = 60


def bright_instrument_fraction(
    frame_path: Path,
    mask_path: Path,
    size: Tuple[int, int] = (320, 256),
) -> float:
    """Share of labelled instrument pixels that are very bright/near-white.

    Computed at reduced resolution (`size`) for speed.

    Args:
        frame_path: Path to a 1920x1080 frame PNG.
        mask_path: Path to the matching binary ground-truth mask.
        size: (width, height) the camera view is resized to.

    Returns:
        A value in [0, 1], or NaN if the mask contains no instrument.
    """
    hsv = np.array(
        crop_camera_view(Image.open(frame_path).convert("HSV")).resize(size)
    )
    gt_image = crop_camera_view(Image.open(mask_path).convert("L"))
    gt = np.array(gt_image.resize(size, Image.NEAREST)) > 127
    if not gt.any():
        return float("nan")
    bright = (hsv[..., 2] > BRIGHT_V_MIN) & (hsv[..., 1] < DESATURATED_S_MAX)
    return float((bright & gt).sum() / gt.sum())
