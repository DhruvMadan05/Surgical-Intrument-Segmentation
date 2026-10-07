"""IoU and Dice metrics for binary segmentation masks.

Shared by the classical baseline and the fine-tuned model so both are
scored with an identical implementation on an identical split.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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


def evaluate_by_sequence(
    pairs_by_sequence: Dict[int, List[Tuple[Path, Path]]],
    predict_fn: Callable[[Path], np.ndarray],
    method: str,
    extra: Optional[Dict] = None,
) -> Dict:
    """Evaluates a predictor on every sequence and summarizes the results.

    Two "overall" numbers are reported because they can diverge when
    sequence sizes differ as much as they do here (75 vs. 300 frames):
      - overall_mean_of_sequences: unweighted mean of the per-sequence
        means, so every sequence counts equally.
      - overall_pooled: mean over every frame scored once, i.e.
        weighted by sequence size. This is the same weighting the
        EndoVis 2017 challenge used for its overall scores.

    Each frame is predicted exactly once; the pooled numbers are
    computed from the per-frame scores collected along the way.

    Args:
        pairs_by_sequence: Sequence number -> (frame, mask) pairs.
        predict_fn: See evaluate_predictions().
        method: Short identifier stored in the output (e.g.
            "unet_resnet34").
        extra: Additional top-level fields to store (e.g. checkpoint).

    Returns:
        A JSON-serializable dict with per_sequence, overall_mean_of_
        sequences and overall_pooled entries.
    """
    per_sequence = {}
    pooled_ious: List[float] = []
    pooled_dices: List[float] = []
    for n in sorted(pairs_by_sequence):
        results = evaluate_predictions(pairs_by_sequence[n], predict_fn)
        pooled_ious.extend(results["per_frame_iou"])
        pooled_dices.extend(results["per_frame_dice"])
        per_sequence[str(n)] = {
            "mean_iou": results["mean_iou"],
            "mean_dice": results["mean_dice"],
            "num_frames": results["num_frames"],
        }
        print(
            f"instrument_dataset_{n}: {results['num_frames']} frames, "
            f"IoU={results['mean_iou']:.4f}, "
            f"Dice={results['mean_dice']:.4f}"
        )

    summary = {
        "method": method,
        **(extra or {}),
        "per_sequence": per_sequence,
        "overall_mean_of_sequences": {
            "mean_iou": float(
                np.mean([r["mean_iou"] for r in per_sequence.values()])
            ),
            "mean_dice": float(
                np.mean([r["mean_dice"] for r in per_sequence.values()])
            ),
        },
        "overall_pooled": {
            "mean_iou": float(np.mean(pooled_ious)),
            "mean_dice": float(np.mean(pooled_dices)),
            "num_frames": len(pooled_ious),
        },
    }
    print(
        "\nOverall (mean of sequence means): "
        f"IoU={summary['overall_mean_of_sequences']['mean_iou']:.4f}, "
        f"Dice={summary['overall_mean_of_sequences']['mean_dice']:.4f}"
    )
    print(
        f"Overall (pooled across {len(pooled_ious)} frames): "
        f"IoU={summary['overall_pooled']['mean_iou']:.4f}, "
        f"Dice={summary['overall_pooled']['mean_dice']:.4f}"
    )
    return summary


def save_results(summary: Dict, out_path: Path) -> None:
    """Writes an evaluate_by_sequence() summary to a JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {out_path}")
