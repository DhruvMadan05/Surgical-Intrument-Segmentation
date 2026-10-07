"""Evaluates a trained checkpoint on the official EndoVis 2017 test split.

Loads a U-Net checkpoint (see scripts/training/train_model.py) and runs it
separately on each of the 10 sequences' official test frames
(instrument_dataset_1..8: last 75 frames per sequence; 9..10: all 300
frames each, since they have no training portion at all -- a stronger
generalization check), scored against the official BinarySegmentation
masks (see scripts/data_prep/download_test_masks.py) using the same
segmentation.metrics code as the classical baseline, so the two are
directly comparable. Reports a per-sequence breakdown plus an overall
mean across all 10 sequences.
"""

import argparse
from pathlib import Path

from segmentation import paths
from segmentation.dataset import InstrumentSegDataset, all_sequences_test_pairs
from segmentation.metrics import evaluate_by_sequence, save_results
from segmentation.model import get_device, load_model, make_predict_fn
from segmentation.visualize import save_prediction_overlays


def main() -> None:
    """Parses CLI arguments, evaluates the checkpoint, saves results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=paths.DATASET_ROOT
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
        "--checkpoint", type=Path, default=paths.DEFAULT_CHECKPOINT
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
    parser.add_argument("--out", type=Path, default=paths.MODEL_METRICS)
    parser.add_argument(
        "--overlays-dir", type=Path, default=paths.OVERLAYS_DIR / "model"
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    pairs_by_sequence = all_sequences_test_pairs(
        args.dataset_root, args.test_root
    )
    if args.limit:
        pairs_by_sequence = {
            n: pairs[: args.limit] for n, pairs in pairs_by_sequence.items()
        }

    model = load_model(args.checkpoint, device)
    predict_fn = make_predict_fn(
        model, device, InstrumentSegDataset.DEFAULT_SIZE
    )

    summary = evaluate_by_sequence(
        pairs_by_sequence,
        predict_fn,
        method="unet_resnet34",
        extra={"checkpoint": str(args.checkpoint)},
    )
    save_results(summary, args.out)

    all_pairs = [
        p for n in sorted(pairs_by_sequence) for p in pairs_by_sequence[n]
    ]
    save_prediction_overlays(all_pairs, predict_fn, args.overlays_dir)
    print(f"Saved overlays to {args.overlays_dir}")


if __name__ == "__main__":
    main()
