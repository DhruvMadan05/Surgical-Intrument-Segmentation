"""Evaluates the model against the *original* official test masks.

scripts/data_prep/fix_test_masks.py overwrote the official masks for
sequences 1 and 2 with self-computed ones (backing the originals up to
*_official_backup/). scripts/analysis/compare_mask_sources.py shows that
replacement was a mistake: the self-computed masks include the
ultrasound probe (the "Other_labels" class), which the challenge defines
as background. This script re-scores the trained model against the
original official masks (the backups where they exist) so the effect on
the reported numbers is measured.

Saves results/metrics/model_eval_original_masks.json, in the same
format as model_eval.json.
"""

import argparse
from pathlib import Path

from segmentation import paths
from segmentation.dataset import (
    ALL_SEQUENCES,
    InstrumentSegDataset,
    original_official_dir,
    test_frame_names,
)
from segmentation.metrics import evaluate_by_sequence, save_results
from segmentation.model import get_device, load_model, make_predict_fn


def main() -> None:
    """Parses CLI arguments, evaluates against original masks, saves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=paths.DATASET_ROOT
    )
    parser.add_argument("--test-root", type=Path, default=paths.TEST_ROOT)
    parser.add_argument(
        "--checkpoint", type=Path, default=paths.DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.METRICS_DIR / "model_eval_original_masks.json",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    model = load_model(args.checkpoint, device)
    predict_fn = make_predict_fn(
        model, device, InstrumentSegDataset.DEFAULT_SIZE
    )

    pairs_by_sequence = {}
    for n in ALL_SEQUENCES:
        frames_dir = (
            args.dataset_root / f"instrument_dataset_{n}" / "left_frames"
        )
        mask_dir = original_official_dir(args.test_root, n)
        pairs_by_sequence[n] = [
            (frames_dir / name, mask_dir / name)
            for name in test_frame_names(args.test_root, n)
        ]

    summary = evaluate_by_sequence(
        pairs_by_sequence,
        predict_fn,
        method="unet_resnet34_vs_original_official_masks",
        extra={"checkpoint": str(args.checkpoint)},
    )
    save_results(summary, args.out)


if __name__ == "__main__":
    main()
