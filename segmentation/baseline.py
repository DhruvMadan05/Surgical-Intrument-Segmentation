"""Classical baseline: Otsu thresholding for instrument segmentation.

Instruments in this dataset tend to be desaturated metal against
saturated tissue, so thresholding the HSV saturation channel separates
them better than plain grayscale intensity. Otsu's method picks the
threshold automatically per frame; a small morphological open/close
cleans up salt-and-pepper noise from specular highlights.

Lives in the library (rather than the evaluation script) so that both
scripts/evaluation/baseline_threshold.py and the speed benchmark can
import the exact same predictor.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk

from segmentation.crop import crop_camera_view


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
