"""Figures and tables for the model's error analysis.

Reads results/metrics/per_frame_errors.csv and error_maps.npz (written
by scripts/evaluation/per_frame_errors.py) and produces, under
results/figures/error_analysis/:
  - iou_distribution.png: per-frame IoU histogram and per-sequence spread.
  - error_types.png: per-sequence composition of error pixels (false
    positive vs false negative, boundary vs near vs gross).
  - error_heatmaps.png: where in the image false positives / negatives
    occur.
  - worst_frames.png: input, ground truth, prediction and error overlay
    for the worst frame of the worst sequences, plus a typical frame.
and, under results/tables/, error_by_sequence.{md,csv} and
bright_instrument_vs_iou.{md,csv} (does the model fail on near-white
instruments, which are rare in training?) and error_near_border.{md,csv}
(do errors cluster at the frame edge?).

The written analysis lives in ERROR_ANALYSIS.md.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from PIL import Image

from segmentation import paths
from segmentation.appearance import bright_instrument_fraction
from segmentation.crop import crop_camera_view
from segmentation.dataset import (
    InstrumentSegDataset,
    all_sequences_test_pairs,
    train_pairs,
)
from segmentation.model import get_device, load_model, make_predict_fn
from segmentation.plotting import (
    COLOR_BASELINE,
    COLOR_MODEL,
    GRID,
    SURFACE,
    TEXT_SECONDARY,
    apply_style,
)
from segmentation.reporting import write_tables

# False positives (model said instrument, truly background) are orange;
# false negatives (missed instrument) are blue. Darker = farther from
# the true boundary (boundary -> near -> gross).
FP_COLORS = ["#f6b49a", "#eb6834", "#a93f13"]
FN_COLORS = ["#a9cbf0", "#2a78d6", "#144a8c"]
ERROR_CATEGORIES = [
    ("fp_boundary", "False positive, <=5 px from edge", FP_COLORS[0]),
    ("fp_near", "False positive, 5-20 px", FP_COLORS[1]),
    ("fp_gross", "False positive, >20 px (spurious region)", FP_COLORS[2]),
    ("fn_boundary", "False negative, <=5 px from edge", FN_COLORS[0]),
    ("fn_near", "False negative, 5-20 px", FN_COLORS[1]),
    ("fn_gross", "False negative, >20 px (missed region)", FN_COLORS[2]),
]

NUM_GALLERY_SEQUENCES = 6


def load_rows(path: Path) -> List[Dict]:
    """Reads per_frame_errors.csv into a list of typed dicts."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {k: float(v) for k, v in r.items() if k != "frame"}
            row["sequence"] = int(r["sequence"])
            row["frame"] = r["frame"]
            rows.append(row)
    return rows


def column(rows: List[Dict], key: str, sequence: int = None) -> np.ndarray:
    """One column as an array, optionally restricted to a sequence."""
    return np.array(
        [r[key] for r in rows if sequence is None or r["sequence"] == sequence]
    )


def sequence_table(rows: List[Dict]) -> List[list]:
    """Per-sequence summary rows (plus an overall row) for the table."""
    sequences = sorted({r["sequence"] for r in rows})
    out = []
    for label, seq in [(str(n), n) for n in sequences] + [("All", None)]:
        iou = column(rows, "iou", seq)
        fp, fn = column(rows, "fp_px", seq), column(rows, "fn_px", seq)
        gross = (
            column(rows, "fp_gross", seq).sum()
            + column(rows, "fn_gross", seq).sum()
        )
        boundary = (
            column(rows, "fp_boundary", seq).sum()
            + column(rows, "fn_boundary", seq).sum()
        )
        total = fp.sum() + fn.sum()
        out.append(
            [
                label,
                f"{iou.mean():.3f}",
                f"{np.median(iou):.3f}",
                f"{iou.min():.3f}",
                int((iou < 0.7).sum()),
                f"{fp.sum() / total:.0%}",
                f"{boundary / total:.0%}",
                f"{gross / total:.0%}",
            ]
        )
    return out


# Bins of "share of instrument pixels that are near-white".
BRIGHT_BINS = [
    ("< 0.1", 0.0, 0.1),
    ("0.1-0.2", 0.1, 0.2),
    ("0.2-0.3", 0.2, 0.3),
    ("0.3-0.4", 0.3, 0.4),
    ("0.4-0.5", 0.4, 0.5),
    (">= 0.5", 0.5, 1.01),
]

# Every Nth training frame is measured for the training-set reference
# (frames within a video are near-duplicates, so this loses little).
TRAIN_SAMPLE_STEP = 4


def brightness_table(
    rows: List[Dict], dataset_root: Path, test_root: Path
) -> List[list]:
    """Test IoU vs instrument brightness, next to how common it is in training.

    Args:
        rows: Per-frame test results (with a bright_frac column).
        dataset_root: Dataset root, used to find the training frames.
        test_root: Official test-mask root (decides which frames are
            training frames).

    Returns:
        One table row per brightness bin: test frames, share of test
        frames, mean IoU, and share of sampled training frames.
    """
    bright = column(rows, "bright_frac")
    iou = column(rows, "iou")
    train = np.array(
        [
            bright_instrument_fraction(frame, mask)
            for frame, mask in train_pairs(dataset_root, test_root)[
                ::TRAIN_SAMPLE_STEP
            ]
        ]
    )
    train = train[~np.isnan(train)]
    out = []
    for label, lo, hi in BRIGHT_BINS:
        in_bin = (bright >= lo) & (bright < hi)
        out.append(
            [
                label,
                int(in_bin.sum()),
                f"{in_bin.mean():.1%}",
                f"{iou[in_bin].mean():.3f}" if in_bin.any() else "-",
                f"{((train >= lo) & (train < hi)).mean():.1%}",
            ]
        )
    return out


# Border bands (px at 1280x1024) used for the "do errors cluster at the
# frame edge?" table.
BORDER_BANDS_PX = (32, 64, 128)


def border_table(maps_path: Path) -> List[list]:
    """Share of instrument / error pixels near the image border.

    Instruments enter the view from the frame edge, so some of any error
    concentration there is expected. The table puts each band's share of
    ground-truth instrument pixels next to its share of false positives
    and false negatives so the two can be compared.

    Args:
        maps_path: error_maps.npz from per_frame_errors.py.

    Returns:
        One row per band: share of image area, of instrument pixels, of
        false-positive pixels and of false-negative pixels.
    """
    maps = np.load(maps_path)
    gt, fp, fn = (
        maps[k].sum(axis=0).astype(float) for k in ("gt", "fp", "fn")
    )
    height, width = gt.shape
    block = int(maps["block"])
    yy, xx = np.mgrid[0:height, 0:width]
    # Distance (full-resolution px) from each cell to the nearest border.
    dist = np.minimum.reduce([xx, width - 1 - xx, yy, height - 1 - yy]) * block
    rows = []
    for band in BORDER_BANDS_PX:
        near = dist < band
        rows.append(
            [
                f"< {band} px",
                f"{near.mean():.1%}",
                f"{gt[near].sum() / gt.sum():.1%}",
                f"{fp[near].sum() / fp.sum():.1%}",
                f"{fn[near].sum() / fn.sum():.1%}",
            ]
        )
    return rows


def plot_iou_distribution(rows: List[Dict], out: Path) -> None:
    """Histogram of per-frame IoU and per-sequence box plots."""
    iou = column(rows, "iou")
    sequences = sorted({r["sequence"] for r in rows})
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.3]}
    )

    ax = axes[0]
    ax.hist(
        iou,
        bins=np.linspace(0, 1, 41),
        color=COLOR_MODEL,
        edgecolor=SURFACE,
    )
    ax.axvline(
        np.median(iou),
        color="#0b0b0b",
        linestyle="--",
        linewidth=1,
        label=f"median {np.median(iou):.3f}",
    )
    ax.axvline(
        iou.mean(),
        color="#0b0b0b",
        linestyle=":",
        linewidth=1.5,
        label=f"mean {iou.mean():.3f}",
    )
    ax.legend(loc="upper left")
    ax.set_xlabel("Per-frame IoU")
    ax.set_ylabel("Number of test frames")
    ax.set_title("Most frames are good; a long tail fails badly")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    data = [column(rows, "iou", n) for n in sequences]
    ax.boxplot(
        data,
        positions=range(len(sequences)),
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": COLOR_MODEL, "edgecolor": SURFACE},
        medianprops={"color": "#0b0b0b"},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": TEXT_SECONDARY,
            "markeredgecolor": SURFACE,
        },
        whiskerprops={"color": TEXT_SECONDARY},
        capprops={"color": TEXT_SECONDARY},
    )
    ax.set_xticks(range(len(sequences)))
    ax.set_xticklabels([str(n) for n in sequences])
    ax.set_xlabel("Sequence")
    ax.set_ylabel("Per-frame IoU")
    ax.set_ylim(0, 1.02)
    ax.set_title("Spread within each sequence")
    ax.grid(axis="x", visible=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_error_types(rows: List[Dict], out: Path) -> None:
    """Stacked bars: composition of error pixels per sequence."""
    sequences = sorted({r["sequence"] for r in rows})
    groups = [(str(n), n) for n in sequences] + [("All", None)]
    fig, ax = plt.subplots(figsize=(10, 4.5))

    bottoms = np.zeros(len(groups))
    totals = np.array(
        [
            column(rows, "fp_px", s).sum() + column(rows, "fn_px", s).sum()
            for _, s in groups
        ]
    )
    for key, label, color in ERROR_CATEGORIES:
        share = (
            np.array([column(rows, key, s).sum() for _, s in groups]) / totals
        )
        ax.bar(
            range(len(groups)),
            share,
            bottom=bottoms,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            label=label,
            width=0.75,
        )
        bottoms += share

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g for g, _ in groups])
    ax.set_xlabel("Sequence")
    ax.set_ylabel("Share of error pixels")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("What kind of errors: edge slips vs whole regions")
    ax.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_error_heatmaps(maps_path: Path, out: Path) -> None:
    """Where false positives / false negatives occur in the image."""
    maps = np.load(maps_path)
    block = int(maps["block"])
    # Fraction of test frames in which each coarse cell is wrong,
    # normalized by the number of pixels per cell.
    num_frames = 1200
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, color, title in (
        (axes[0], "fp", COLOR_BASELINE, "False positives (spurious)"),
        (axes[1], "fn", COLOR_MODEL, "False negatives (missed)"),
    ):
        freq = maps[key].sum(axis=0) / (block * block * num_frames)
        cmap = LinearSegmentedColormap.from_list("e", [SURFACE, color])
        im = ax.imshow(
            freq,
            cmap=cmap,
            norm=PowerNorm(0.5, vmin=0, vmax=np.percentile(freq, 99.9)),
        )
        ax.set_title(title)
        ax.axis("off")
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Fraction of frames wrong at this location")
    fig.suptitle(
        "Error frequency over all 1,200 test frames "
        "(dominated by a few failure episodes)",
        x=0.01,
        y=1.04,
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def error_overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray):
    """Dims the image and paints FP orange and FN blue on top of it."""
    out = (image.astype(np.float32) * 0.45).astype(np.uint8)
    hexes = (COLOR_BASELINE, COLOR_MODEL)
    for mask, hx in zip((pred & ~gt, gt & ~pred), hexes):
        rgb = np.array([int(hx[i : i + 2], 16) for i in (1, 3, 5)])
        out[mask] = rgb
    return out


def plot_worst_frames(
    rows: List[Dict],
    checkpoint: Path,
    dataset_root: Path,
    test_root: Path,
    out: Path,
) -> None:
    """Gallery: worst frame of the worst sequences plus a typical frame."""
    sequences = sorted({r["sequence"] for r in rows})
    worst = {
        n: min((r for r in rows if r["sequence"] == n), key=lambda r: r["iou"])
        for n in sequences
    }
    picks = sorted(worst.values(), key=lambda r: r["iou"])
    picks = picks[:NUM_GALLERY_SEQUENCES]
    median = float(np.median(column(rows, "iou")))
    typical = min(rows, key=lambda r: abs(r["iou"] - median))
    picks.append(typical)

    paths_by_name = {
        (n, p[0].name): p
        for n, pairs in all_sequences_test_pairs(
            dataset_root, test_root
        ).items()
        for p in pairs
    }
    device = get_device()
    model = load_model(checkpoint, device)
    predict_fn = make_predict_fn(
        model, device, InstrumentSegDataset.DEFAULT_SIZE
    )

    titles = ["Input", "Ground truth", "Prediction", "Errors"]
    fig, axes = plt.subplots(
        len(picks), 4, figsize=(13, 2.75 * len(picks)), squeeze=False
    )
    for ri, r in enumerate(picks):
        frame_path, mask_path = paths_by_name[(r["sequence"], r["frame"])]
        image = np.array(
            crop_camera_view(Image.open(frame_path).convert("RGB"))
        )
        gt = (
            np.array(crop_camera_view(Image.open(mask_path).convert("L")))
            > 127
        )
        pred = predict_fn(frame_path)
        panels = [image, gt, pred, error_overlay(image, pred, gt)]
        for ci, panel in enumerate(panels):
            ax = axes[ri][ci]
            ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            ax.axis("off")
            ax.grid(False)
            if ri == 0:
                ax.set_title(titles[ci], fontsize=10, loc="center")
        tag = "typical" if r is typical else "worst in sequence"
        axes[ri][0].text(
            0.0,
            1.0,
            f"seq {r['sequence']} {r['frame']}  IoU {r['iou']:.2f} ({tag})",
            transform=axes[ri][0].transAxes,
            va="bottom",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    fig.text(
        0.01,
        0.005,
        "Errors: orange = false positive (model said instrument, truth is "
        "background); blue = false negative (missed instrument).",
        fontsize=9,
        color=TEXT_SECONDARY,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Parses CLI arguments, writes all error-analysis outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, default=paths.PER_FRAME_ERRORS)
    parser.add_argument("--maps", type=Path, default=paths.ERROR_MAPS)
    parser.add_argument(
        "--dataset-root", type=Path, default=paths.DATASET_ROOT
    )
    parser.add_argument("--test-root", type=Path, default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=paths.DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=paths.FIGURES_DIR / "error_analysis"
    )
    parser.add_argument("--table-dir", type=Path, default=paths.TABLES_DIR)
    args = parser.parse_args()

    apply_style()
    rows = load_rows(args.errors)

    write_tables(
        [
            "Sequence",
            "Mean IoU",
            "Median IoU",
            "Min IoU",
            "Frames < 0.7",
            "False-positive share",
            "Boundary share",
            "Gross share",
        ],
        sequence_table(rows),
        args.table_dir / "error_by_sequence.md",
        args.table_dir / "error_by_sequence.csv",
    )
    write_tables(
        [
            "Near-white share of instrument pixels",
            "Test frames",
            "% of test frames",
            "Mean test IoU",
            "% of training frames",
        ],
        brightness_table(rows, args.dataset_root, args.test_root),
        args.table_dir / "bright_instrument_vs_iou.md",
        args.table_dir / "bright_instrument_vs_iou.csv",
    )
    write_tables(
        [
            "Distance from image border",
            "% of image area",
            "% of instrument pixels",
            "% of false-positive pixels",
            "% of false-negative pixels",
        ],
        border_table(args.maps),
        args.table_dir / "error_near_border.md",
        args.table_dir / "error_near_border.csv",
    )
    plot_iou_distribution(rows, args.figure_dir / "iou_distribution.png")
    plot_error_types(rows, args.figure_dir / "error_types.png")
    plot_error_heatmaps(args.maps, args.figure_dir / "error_heatmaps.png")
    plot_worst_frames(
        rows,
        args.checkpoint,
        args.dataset_root,
        args.test_root,
        args.figure_dir / "worst_frames.png",
    )
    for name in (
        "error_by_sequence",
        "bright_instrument_vs_iou",
        "error_near_border",
    ):
        print((args.table_dir / f"{name}.md").read_text())
    print(f"Saved figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
