"""Shared matplotlib styling so every figure in results/figures/ matches.

Colors come from a validated categorical palette (distinct under
color-vision deficiency and >= 3:1 against the surface). Each method
keeps the same color in every figure, regardless of how many series a
particular figure shows.
"""

import matplotlib.pyplot as plt

# Per-method colors, fixed across all figures.
COLOR_MODEL = "#2a78d6"  # blue: fine-tuned U-Net
COLOR_BASELINE = "#eb6834"  # orange: Otsu-threshold baseline
COLOR_CHALLENGE = "#8a8984"  # neutral gray: other challenge teams

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def apply_style() -> None:
    """Applies the project-wide matplotlib rcParams (light surface)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT_PRIMARY,
            "legend.frameon": False,
            "font.size": 10,
            "figure.dpi": 100,
            "savefig.dpi": 200,
        }
    )
