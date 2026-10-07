"""Helpers for turning saved evaluation JSON into tables.

Used by the scripts in scripts/analysis/ to read the per-method files
written by segmentation.metrics.save_results() and to emit Markdown /
CSV tables.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from segmentation.dataset import HELD_OUT_SEQUENCES

# Sequences whose frames never appear in training at all.
HELD_OUT = tuple(HELD_OUT_SEQUENCES)


def load_metrics(path: Path) -> Dict:
    """Loads an evaluation JSON written by save_results()."""
    with open(path) as f:
        return json.load(f)


def sequence_scores(metrics: Dict, key: str = "mean_iou") -> Dict[int, float]:
    """Returns {sequence number: score} for `key` (mean_iou/mean_dice)."""
    return {int(n): r[key] for n, r in metrics["per_sequence"].items()}


def frame_weighted_mean(
    metrics: Dict, sequences: Iterable[int], key: str = "mean_iou"
) -> float:
    """Mean of a score over `sequences`, weighted by frame count.

    With every sequence included this equals the "pooled" overall score
    (every frame counted once), which is also how the EndoVis 2017
    challenge computed its overall numbers.
    """
    entries = [metrics["per_sequence"][str(n)] for n in sequences]
    total_frames = sum(e["num_frames"] for e in entries)
    return sum(e[key] * e["num_frames"] for e in entries) / total_frames


def write_tables(
    headers: Sequence[str],
    rows: List[Sequence[str]],
    markdown_path: Path,
    csv_path: Path,
) -> None:
    """Writes the same table as a GitHub-flavored Markdown file and a CSV."""
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    markdown_path.write_text("\n".join(lines) + "\n")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
