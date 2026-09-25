"""IoU and Dice metrics for binary segmentation masks.

Shared by the classical baseline and the fine-tuned model so both are
scored with an identical implementation on an identical split.
"""

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from PIL import Image

from segmentation.crop import crop_camera_view


def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection-over-union between two binary masks.

    Returns 1.0 if both masks are empty (no foreground in either), since
    that's a correct prediction, not an undefined one.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice coefficient between two binary masks.

    Returns 1.0 if both masks are empty (no foreground in either).
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * intersection) / float(denom)


def evaluate_predictions(
    pairs: List[Tuple[Path, Path]],
    predict_fn: Callable[[Path], np.ndarray],
) -> Dict:
    """Evaluates a mask predictor over a list of (frame, mask) pairs.

    Args:
        pairs: List of (frame_path, mask_path) tuples.
        predict_fn: Callable that takes a frame_path and returns a
            predicted binary mask, same (height, width) as the
            cropped ground-truth mask (1024x1280), values 0/1 or bool.

    Returns:
        A dict with mean_iou, mean_dice, per-frame IoU/Dice lists, and
        num_frames.
    """
    ious = []
    dices = []
    for frame_path, mask_path in pairs:
        gt_image = crop_camera_view(Image.open(mask_path).convert("L"))
        gt_mask = np.array(gt_image) > 127
        pred_mask = predict_fn(frame_path)
        ious.append(iou(pred_mask, gt_mask))
        dices.append(dice(pred_mask, gt_mask))

    return {
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "per_frame_iou": ious,
        "per_frame_dice": dices,
        "num_frames": len(pairs),
    }
