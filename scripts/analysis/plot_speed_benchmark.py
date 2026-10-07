"""Turns the speed benchmark JSON into a table and a figure.

Reads results/metrics/speed_benchmark.json (written by
scripts/evaluation/benchmark_speed.py) and produces:
  - results/tables/speed_benchmark.{md,csv}: mean/std/median/p95 latency
    per frame and the processing rate in Hz for every benchmark.
  - results/figures/speed_benchmark.png: latency and rate side by side.
    Solid bars are end-to-end (PNG decode to full-resolution mask);
    hatched bars are the U-Net forward pass alone.

Rate (Hz) is 1 / mean latency, i.e. frames per second when processing
frames back to back.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from segmentation import paths
from segmentation.plotting import (
    COLOR_BASELINE,
    COLOR_MODEL,
    SURFACE,
    TEXT_SECONDARY,
    apply_style,
)
from segmentation.reporting import load_metrics, write_tables

# (benchmark key, display label, color, hatched?) in top-to-bottom order.
BARS = [
    ("otsu_baseline_cpu", "Otsu baseline (CPU)", COLOR_BASELINE, False),
    ("unet_cpu", "U-Net (CPU)", COLOR_MODEL, False),
    ("unet_mps", "U-Net (Apple GPU)", COLOR_MODEL, False),
    ("unet_cpu_forward", "U-Net forward only (CPU)", COLOR_MODEL, True),
    ("unet_mps_forward", "U-Net forward only (Apple GPU)", COLOR_MODEL, True),
]

# Common real-time reference: a 30 Hz video stream.
REALTIME_HZ = 30


def build_rows(benchmarks: dict) -> list:
    """Table rows for every benchmark present in the results."""
    rows = []
    for key, label, _, _ in BARS:
        if key not in benchmarks:
            continue
        b = benchmarks[key]
        scope = (
            "forward pass only" if key.endswith("_forward") else "end to end"
        )
        rows.append(
            [
                label,
                scope,
                f"{b['mean_ms']:.1f}",
                f"{b['std_ms']:.1f}",
                f"{b['median_ms']:.1f}",
                f"{b['p95_ms']:.1f}",
                f"{b['hz']:.1f}",
            ]
        )
    return rows


def plot_speed(benchmarks: dict, out_path: Path) -> None:
    """Draws latency and rate bars for each benchmark present."""
    apply_style()
    bars = [bar for bar in BARS if bar[0] in benchmarks]
    y = np.arange(len(bars))[::-1]  # first bar at the top

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    panels = (
        (
            axes[0],
            "mean_ms",
            "Mean latency per frame (ms), lower is faster",
            "{:.0f}",
        ),
        (axes[1], "hz", "Processing rate (Hz), higher is faster", "{:.1f}"),
    )
    for ax, key, title, fmt in panels:
        values = [benchmarks[k][key] for k, *_ in bars]
        for yi, value, (_, _, color, hatched) in zip(y, values, bars):
            ax.barh(
                yi,
                value,
                color=SURFACE if hatched else color,
                edgecolor=color,
                hatch="///" if hatched else None,
                linewidth=1.2,
                height=0.7,
            )
            ax.text(
                value + max(values) * 0.015,
                yi,
                fmt.format(value),
                va="center",
                fontsize=9,
                color=TEXT_SECONDARY,
            )
        ax.set_xlim(0, max(values) * 1.15)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)

    # Real-time reference on the rate panel, if it is within view.
    rate_ax = axes[1]
    if rate_ax.get_xlim()[1] > REALTIME_HZ:
        rate_ax.axvline(
            REALTIME_HZ, color="#0b0b0b", linestyle="--", linewidth=1
        )
        rate_ax.text(
            REALTIME_HZ + 0.5,
            y.max() + 0.6,
            f"{REALTIME_HZ} Hz video rate",
            fontsize=8,
            color="#0b0b0b",
            va="bottom",
        )

    # Headroom above the top bar so the reference-line label sits inside
    # the axes instead of colliding with the panel title.
    axes[0].set_ylim(-0.6, y.max() + 1.1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label for _, label, _, _ in bars])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Parses CLI arguments, writes the speed table and figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=paths.SPEED_METRICS)
    parser.add_argument(
        "--figure",
        type=Path,
        default=paths.FIGURES_DIR / "speed_benchmark.png",
    )
    parser.add_argument("--table-dir", type=Path, default=paths.TABLES_DIR)
    args = parser.parse_args()

    results = load_metrics(args.metrics)
    benchmarks = results["benchmarks"]

    write_tables(
        [
            "Method",
            "Scope",
            "Mean (ms)",
            "Std (ms)",
            "Median (ms)",
            "p95 (ms)",
            "Rate (Hz)",
        ],
        build_rows(benchmarks),
        args.table_dir / "speed_benchmark.md",
        args.table_dir / "speed_benchmark.csv",
        left_aligned=(0, 1),
    )
    plot_speed(benchmarks, args.figure)

    print((args.table_dir / "speed_benchmark.md").read_text())
    settings = results["settings"]
    print(
        f"{settings['num_frames']} frames, {settings['warmup']} warmup; "
        f"{results['system']['platform']}"
    )
    print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
