"""Plots the per-epoch training loss recorded by train_model.py.

Reads results/logs/train_log.csv (columns: epoch, mean_loss) and saves a
line chart to results/figures/training_loss.png. The logged value is the
mean soft-Dice loss over all training batches in the epoch (0 = perfect
overlap). Only training loss is logged -- no validation loss -- so the
curve shows optimization progress, not generalization; see the test-set
IoU in results/metrics/ for that.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

from segmentation import paths
from segmentation.plotting import COLOR_MODEL, apply_style


def read_log(log_path: Path) -> Tuple[List[int], List[float]]:
    """Reads (epochs, mean_loss values) from a train_model.py CSV log."""
    epochs, losses = [], []
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["mean_loss"]))
    return epochs, losses


def plot_training_loss(
    epochs: List[int], losses: List[float], out_path: Path
) -> None:
    """Draws the loss curve (log y-axis) and saves it to out_path."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(epochs, losses, color=COLOR_MODEL, linewidth=2, marker="o")
    # Log scale: the loss drops ~10x over training, and on a linear axis
    # the late-epoch improvement would be squashed into a flat line.
    ax.set_yscale("log")
    # Explicit plain-number ticks; matplotlib's default log ticks would
    # label only the 10^-1 decade here.
    ticks = [0.03, 0.05, 0.1, 0.2, 0.4]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Dice loss (log scale)")
    ax.set_title("U-Net (ResNet34) training loss")
    ax.set_xlim(min(epochs) - 0.5, max(epochs) + 0.5)

    # Label only the endpoints rather than every point.
    for epoch, loss, offset in (
        (epochs[0], losses[0], (8, 0)),
        (epochs[-1], losses[-1], (-8, 10)),
    ):
        ax.annotate(
            f"{loss:.3f}",
            (epoch, loss),
            textcoords="offset points",
            xytext=offset,
            ha="left" if offset[0] > 0 else "right",
            color="#52514e",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Parses CLI arguments and plots the training loss log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=paths.TRAIN_LOG)
    parser.add_argument(
        "--out", type=Path, default=paths.FIGURES_DIR / "training_loss.png"
    )
    args = parser.parse_args()

    epochs, losses = read_log(args.log)
    plot_training_loss(epochs, losses, args.out)
    print(f"Saved training loss plot to {args.out}")


if __name__ == "__main__":
    main()
