"""Compares the model's IoU on training frames vs test frames.

Reads results/metrics/train_eval.json (model on its training frames, scored
against the self-computed masks it trained on), model_eval.json (test
frames, scored against the test masks currently on disk) and
model_eval_original_masks.json (test frames scored against the original
official masks; differs only for sequences 1 and 2). Produces:
  - results/tables/train_vs_test.{md,csv}: per-sequence IoU and the
    train-to-test gap.
  - results/tables/empty_ground_truth_frames.{md,csv}: how the model does
    on frames that contain no instrument at all.
  - results/figures/train_vs_test.png.

A large train/test gap means overfitting; a train score well below the
level the model could reach points at label problems. Sequences 9 and 10
have no training frames.
"""

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from segmentation import paths
from segmentation.dataset import train_pairs
from segmentation.plotting import (
    COLOR_MODEL,
    SURFACE,
    TEXT_SECONDARY,
    apply_style,
)
from segmentation.reporting import (
    frame_weighted_mean,
    load_metrics,
    sequence_scores,
    write_tables,
)

TRAIN_SEQUENCES = range(1, 9)


def comparison_rows(
    train: dict, test: dict, test_original: dict
) -> List[list]:
    """Per-sequence rows plus overall rows for the comparison table."""
    train_iou = sequence_scores(train)
    test_iou = sequence_scores(test)
    original_iou = sequence_scores(test_original)
    rows = []
    for n in sorted(test_iou):
        in_train = n in train_iou
        rows.append(
            [
                n,
                f"{train_iou[n]:.3f}" if in_train else "-",
                f"{test_iou[n]:.3f}",
                f"{train_iou[n] - test_iou[n]:+.3f}" if in_train else "-",
                f"{original_iou[n]:.3f}",
                (
                    f"{train_iou[n] - original_iou[n]:+.3f}"
                    if in_train
                    else "-"
                ),
            ]
        )
    seqs_1_8 = list(TRAIN_SEQUENCES)
    all_seqs = sorted(test_iou)
    train_mean = frame_weighted_mean(train, seqs_1_8)
    for label, group in (("Mean, sequences 1-8", seqs_1_8),):
        t = frame_weighted_mean(test, group)
        o = frame_weighted_mean(test_original, group)
        rows.append(
            [
                label,
                f"{train_mean:.3f}",
                f"{t:.3f}",
                f"{train_mean - t:+.3f}",
                f"{o:.3f}",
                f"{train_mean - o:+.3f}",
            ]
        )
    rows.append(
        [
            "Mean, all 10 (test only)",
            "-",
            f"{frame_weighted_mean(test, all_seqs):.3f}",
            "-",
            f"{frame_weighted_mean(test_original, all_seqs):.3f}",
            "-",
        ]
    )
    return rows


def empty_ground_truth_rows(
    per_frame_csv: Path, dataset_root: Path, test_root: Path
) -> List[list]:
    """How the model behaves on training frames with no instrument.

    Both-empty counts as IoU 1.0 and any prediction on an empty mask as
    0.0, so these frames are all-or-nothing in the IoU metric.
    """
    iou = {}
    with open(per_frame_csv, newline="") as f:
        for r in csv.DictReader(f):
            iou[(int(r["sequence"]), r["frame"])] = float(r["iou"])

    empty = []
    for frame, mask in train_pairs(dataset_root, test_root):
        gt = np.array(Image.open(mask).convert("L")) > 127
        if not gt.any():
            n = int(frame.parent.parent.name.split("_")[-1])
            empty.append((n, frame.name))

    rows = []
    for n in sorted({n for n, _ in empty}) + ["All"]:
        group = [k for k in empty if n == "All" or k[0] == n]
        correct = sum(iou[k] == 1.0 for k in group)
        rows.append(
            [
                n,
                len(group),
                correct,
                len(group) - correct,
                f"{(len(group) - correct) / len(group):.0%}",
            ]
        )
    return rows


def plot_train_vs_test(
    train: dict, test: dict, test_original: dict, out_path: Path
) -> None:
    """Grouped bars: train IoU, test IoU, test IoU vs original masks."""
    apply_style()
    train_iou = sequence_scores(train)
    test_iou = sequence_scores(test)
    original_iou = sequence_scores(test_original)
    sequences = sorted(test_iou)
    x = np.arange(len(sequences))
    width = 0.27

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.bar(
        x - width,
        [train_iou.get(n, 0) for n in sequences],
        width,
        color=TEXT_SECONDARY,
        edgecolor=SURFACE,
        label="Training frames (vs self-computed masks)",
    )
    ax.bar(
        x,
        [test_iou[n] for n in sequences],
        width,
        color=COLOR_MODEL,
        edgecolor=SURFACE,
        label="Test frames (vs masks currently on disk)",
    )
    ax.bar(
        x + width,
        [original_iou[n] for n in sequences],
        width,
        color=SURFACE,
        edgecolor=COLOR_MODEL,
        hatch="///",
        linewidth=1.2,
        label="Test frames (vs original official masks)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in sequences])
    ax.set_xlabel("Sequence (9-10 have no training frames)")
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("Train vs test IoU, and the effect of the mask mismatch")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Parses CLI arguments, writes the tables and figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=paths.TRAIN_METRICS)
    parser.add_argument("--test", type=Path, default=paths.MODEL_METRICS)
    parser.add_argument(
        "--test-original",
        type=Path,
        default=paths.METRICS_DIR / "model_eval_original_masks.json",
    )
    parser.add_argument(
        "--train-per-frame", type=Path, default=paths.TRAIN_PER_FRAME
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=paths.DATASET_ROOT
    )
    parser.add_argument("--test-root", type=Path, default=None)
    parser.add_argument("--table-dir", type=Path, default=paths.TABLES_DIR)
    parser.add_argument(
        "--figure",
        type=Path,
        default=paths.FIGURES_DIR / "train_vs_test.png",
    )
    args = parser.parse_args()

    train = load_metrics(args.train)
    test = load_metrics(args.test)
    test_original = load_metrics(args.test_original)

    write_tables(
        [
            "Sequence",
            "Train IoU",
            "Test IoU (masks on disk)",
            "Train - test",
            "Test IoU (original official masks)",
            "Train - test (original)",
        ],
        comparison_rows(train, test, test_original),
        args.table_dir / "train_vs_test.md",
        args.table_dir / "train_vs_test.csv",
    )
    write_tables(
        [
            "Sequence",
            "Training frames with no instrument",
            "Model predicted nothing (correct)",
            "Model hallucinated an instrument",
            "False-alarm rate",
        ],
        empty_ground_truth_rows(
            args.train_per_frame, args.dataset_root, args.test_root
        ),
        args.table_dir / "empty_ground_truth_frames.md",
        args.table_dir / "empty_ground_truth_frames.csv",
    )
    plot_train_vs_test(train, test, test_original, args.figure)

    for name in ("train_vs_test", "empty_ground_truth_frames"):
        print((args.table_dir / f"{name}.md").read_text())
    print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
