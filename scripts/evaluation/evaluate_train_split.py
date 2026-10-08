"""Evaluates the trained model on its own training frames.

Scores the checkpoint on the 1,800 training frames (instrument_dataset_
1..8, frames 0-224) against the self-computed binary masks they were
trained on, using the same code as scripts/evaluation/evaluate_model.py
does for the test frames. Comparing the two shows how large the
train/test gap is, and unusually low-scoring *training* frames point at
label problems (a model that has fit its training set should score
highly on it, so low training IoU is a hint that the label is wrong).

Outputs:
  - results/metrics/train_eval.json: per-sequence and overall means, same
    schema as model_eval.json.
  - results/metrics/train_per_frame.csv: one row per training frame.

Compare with the test results using
scripts/analysis/compare_train_vs_test.py.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from segmentation import paths
from segmentation.dataset import InstrumentSegDataset, train_pairs
from segmentation.metrics import evaluate_by_sequence, save_results
from segmentation.model import get_device, load_model, make_predict_fn


def main() -> None:
    """Parses CLI arguments, scores every training frame, saves results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=paths.DATASET_ROOT
    )
    parser.add_argument("--test-root", type=Path, default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=paths.DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only the first N frames per sequence (quick sanity check)",
    )
    parser.add_argument("--out", type=Path, default=paths.TRAIN_METRICS)
    parser.add_argument(
        "--per-frame-out", type=Path, default=paths.TRAIN_PER_FRAME
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    model = load_model(args.checkpoint, device)
    predict_fn = make_predict_fn(
        model, device, InstrumentSegDataset.DEFAULT_SIZE
    )

    # train_pairs() returns one flat list; regroup by sequence number
    # (taken from .../instrument_dataset_N/left_frames/frame.png).
    pairs_by_sequence = defaultdict(list)
    for frame, mask in train_pairs(args.dataset_root, args.test_root):
        sequence = int(frame.parent.parent.name.split("_")[-1])
        pairs_by_sequence[sequence].append((frame, mask))
    if args.limit:
        pairs_by_sequence = {
            n: pairs[: args.limit] for n, pairs in pairs_by_sequence.items()
        }

    summary = evaluate_by_sequence(
        pairs_by_sequence,
        predict_fn,
        method="unet_resnet34_on_training_frames",
        extra={"checkpoint": str(args.checkpoint)},
        keep_per_frame=True,
    )

    # Per-frame rows go to CSV; the JSON keeps only the summary numbers.
    args.per_frame_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.per_frame_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "frame", "iou", "dice"])
        for n in sorted(pairs_by_sequence):
            entry = summary["per_sequence"][str(n)]
            for (frame, _), iou_value, dice_value in zip(
                pairs_by_sequence[n],
                entry["per_frame_iou"],
                entry["per_frame_dice"],
            ):
                writer.writerow([n, frame.name, iou_value, dice_value])
            del entry["per_frame_iou"], entry["per_frame_dice"]
    print(f"Saved per-frame results to {args.per_frame_out}")
    save_results(summary, args.out)


if __name__ == "__main__":
    main()
