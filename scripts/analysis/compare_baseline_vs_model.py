"""Compares the Otsu baseline against the fine-tuned U-Net.

Reads results/metrics/baseline_threshold.json and model_eval.json (both
written by the scripts in scripts/evaluation/) and produces:
  - results/tables/baseline_vs_model.{md,csv}: per-sequence IoU and Dice
    for both methods with the absolute improvement, plus overall rows.
  - results/figures/baseline_vs_model.png: grouped bars of per-sequence
    IoU and Dice.

Sequences 9 and 10 are the cleanest comparison: the model never saw any
frame from them. Sequences 1-8 are scored on their last 75 frames, but
the model trained on the first 225 frames of those same videos, so
those scores partly reflect seeing the same scene earlier in time.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from segmentation import paths
from segmentation.plotting import COLOR_BASELINE, COLOR_MODEL, apply_style
from segmentation.reporting import (
    HELD_OUT,
    frame_weighted_mean,
    load_metrics,
    sequence_scores,
    write_tables,
)

TABLE_HEADERS = [
    "Sequence",
    "Baseline IoU",
    "Model IoU",
    "IoU gain",
    "Baseline Dice",
    "Model Dice",
    "Dice gain",
]


def build_rows(baseline: dict, model: dict) -> list:
    """Builds table rows: one per sequence, then three overall rows."""
    sequences = sorted(sequence_scores(baseline))

    def row(label, b_iou, m_iou, b_dice, m_dice):
        """Formats one table row; gains are model minus baseline."""
        return [
            label,
            f"{b_iou:.3f}",
            f"{m_iou:.3f}",
            f"{m_iou - b_iou:+.3f}",
            f"{b_dice:.3f}",
            f"{m_dice:.3f}",
            f"{m_dice - b_dice:+.3f}",
        ]

    b_iou, m_iou = sequence_scores(baseline), sequence_scores(model)
    b_dice = sequence_scores(baseline, "mean_dice")
    m_dice = sequence_scores(model, "mean_dice")
    rows = []
    for n in sequences:
        label = f"{n} (held out)" if n in HELD_OUT else str(n)
        rows.append(row(label, b_iou[n], m_iou[n], b_dice[n], m_dice[n]))

    groups = [
        ("Mean, all 10 (pooled)", sequences),
        ("Mean, sequences 1-8", [n for n in sequences if n not in HELD_OUT]),
        ("Mean, held-out 9-10", list(HELD_OUT)),
    ]
    for label, group in groups:
        rows.append(
            row(
                label,
                frame_weighted_mean(baseline, group, "mean_iou"),
                frame_weighted_mean(model, group, "mean_iou"),
                frame_weighted_mean(baseline, group, "mean_dice"),
                frame_weighted_mean(model, group, "mean_dice"),
            )
        )
    return rows


def plot_comparison(baseline: dict, model: dict, out_path: Path) -> None:
    """Grouped bars (baseline vs model) of per-sequence IoU and Dice."""
    apply_style()
    sequences = sorted(sequence_scores(baseline))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)

    for ax, key, title in (
        (axes[0], "mean_iou", "Mean IoU"),
        (axes[1], "mean_dice", "Mean Dice"),
    ):
        base = sequence_scores(baseline, key)
        mod = sequence_scores(model, key)
        # Last group is the pooled overall score across all 10 sequences.
        base_vals = [base[n] for n in sequences] + [
            frame_weighted_mean(baseline, sequences, key)
        ]
        mod_vals = [mod[n] for n in sequences] + [
            frame_weighted_mean(model, sequences, key)
        ]
        x = np.arange(len(base_vals))
        width = 0.38
        # 2 px-ish surface-colored edge keeps adjacent bars distinct.
        ax.bar(
            x - width / 2,
            base_vals,
            width,
            color=COLOR_BASELINE,
            label="Otsu baseline",
            edgecolor="#fcfcfb",
            linewidth=1,
        )
        ax.bar(
            x + width / 2,
            mod_vals,
            width,
            color=COLOR_MODEL,
            label="U-Net (ResNet34)",
            edgecolor="#fcfcfb",
            linewidth=1,
        )
        # Label only the overall bars; the table has every value.
        for xi, val in (
            (x[-1] - width / 2, base_vals[-1]),
            (x[-1] + width / 2, mod_vals[-1]),
        ):
            ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in sequences] + ["All"])
        ax.set_xlabel(
            "Sequence (9-10 never seen in training)", color="#52514e"
        )
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        # Shade held-out sequences so the fairer comparison stands out.
        ax.axvspan(
            sequences.index(HELD_OUT[0]) - 0.5,
            sequences.index(HELD_OUT[-1]) + 0.5,
            color="#e4e3df",
            alpha=0.5,
            zorder=0,
        )

    axes[0].legend(loc="upper left", ncol=2, bbox_to_anchor=(0, -0.2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Parses CLI arguments, writes the comparison table and figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=paths.BASELINE_METRICS
    )
    parser.add_argument("--model", type=Path, default=paths.MODEL_METRICS)
    parser.add_argument(
        "--figure",
        type=Path,
        default=paths.FIGURES_DIR / "baseline_vs_model.png",
    )
    parser.add_argument("--table-dir", type=Path, default=paths.TABLES_DIR)
    args = parser.parse_args()

    baseline = load_metrics(args.baseline)
    model = load_metrics(args.model)

    rows = build_rows(baseline, model)
    write_tables(
        TABLE_HEADERS,
        rows,
        args.table_dir / "baseline_vs_model.md",
        args.table_dir / "baseline_vs_model.csv",
    )
    plot_comparison(baseline, model, args.figure)

    print((args.table_dir / "baseline_vs_model.md").read_text())
    print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
