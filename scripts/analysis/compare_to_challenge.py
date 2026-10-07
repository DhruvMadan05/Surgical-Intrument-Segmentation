"""Compares this project's results to the EndoVis 2017 challenge teams.

Reads results/metrics/model_eval.json (U-Net), baseline_threshold.json
(Otsu) and reference/endovis2017_binary_results.json (Table I of the
challenge paper, arXiv:1902.06426) and produces:
  - results/tables/challenge_per_dataset.{md,csv}: per dataset, our IoU
    vs the best team, the mean over all 10 teams, and our rank among the
    10 teams + us.
  - results/tables/challenge_leaderboard.{md,csv}: every team, the
    baseline and our model ranked on all-10 and held-out-9-10 IoU.
  - results/figures/challenge_per_dataset.png and challenge_overall.png.

How comparable are the numbers?
  - Datasets 9 and 10: closely comparable. No method saw any frame of
    these videos, and our model trained on the same 8 x 225 training
    frames the challenge provided.
  - Datasets 1-8: NOT like-for-like. Challenge rules forbade a team from
    training on the first 225 frames of the sequence it was tested on
    (9 models per team). Our single model trained on all 8 sequences'
    first 225 frames and is tested on the last 75 of the same videos, so
    it benefits from having seen the same scene earlier. In addition, the
    official masks for sequences 1 and 2 were replaced with our own (see
    scripts/data_prep/fix_test_masks.py), so those two are scored
    against different ground truth than the paper's teams.

The metric is mean per-frame foreground IoU, matching the paper's
evaluation (frames with no instrument in either mask score 1.0 here).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from segmentation import paths
from segmentation.plotting import (
    COLOR_BASELINE,
    COLOR_CHALLENGE,
    COLOR_MODEL,
    SURFACE,
    TEXT_SECONDARY,
    apply_style,
)
from segmentation.reporting import (
    HELD_OUT,
    frame_weighted_mean,
    load_metrics,
    sequence_scores,
    write_tables,
)

OURS = "Ours (U-Net)"
BASELINE = "Otsu baseline"


def rank_among(score: float, others: List[float]) -> int:
    """1-based rank of `score` among `others` (ties share the better rank)."""
    return 1 + sum(1 for o in others if o > score)


def per_dataset_rows(challenge: Dict, ours: Dict[int, float]) -> List[list]:
    """Builds the per-dataset comparison table rows."""
    rows = []
    for n in sorted(ours):
        teams = challenge["per_dataset"][str(n)]
        best_team = max(teams, key=teams.get)
        best = teams[best_team]
        mean_teams = float(np.mean(list(teams.values())))
        rank = rank_among(ours[n], list(teams.values()))
        comparable = (
            "yes"
            if n in HELD_OUT
            else "no (trained on same video)"
            + ("; masks replaced" if n in (1, 2) else "")
        )
        rows.append(
            [
                n,
                f"{ours[n]:.3f}",
                f"{best_team} {best:.3f}",
                f"{ours[n] - best:+.3f}",
                f"{mean_teams:.3f}",
                f"{ours[n] - mean_teams:+.3f}",
                f"{rank} of 11",
                comparable,
            ]
        )
    return rows


def leaderboard(challenge: Dict, model: Dict, baseline: Dict) -> List[dict]:
    """Scores every method on all-10 and held-out-9-10 IoU, ranked."""
    entries = []
    for team in challenge["teams"]:
        per_dataset = challenge["per_dataset"]
        # Datasets 9 and 10 both have 300 frames, so a plain mean is the
        # frame-weighted mean.
        held = float(np.mean([per_dataset[str(n)][team] for n in HELD_OUT]))
        entries.append(
            {
                "name": team,
                "all": challenge["overall"][team],
                "held": held,
            }
        )
    all_sequences = sorted(sequence_scores(model))
    for name, metrics in ((OURS, model), (BASELINE, baseline)):
        entries.append(
            {
                "name": name,
                "all": frame_weighted_mean(metrics, all_sequences),
                "held": frame_weighted_mean(metrics, HELD_OUT),
            }
        )
    entries.sort(key=lambda e: e["all"], reverse=True)
    return entries


def leaderboard_rows(entries: List[dict]) -> List[list]:
    """Table rows with ranks on both metrics."""
    all_scores = [e["all"] for e in entries]
    held_scores = [e["held"] for e in entries]
    return [
        [
            rank_among(e["all"], all_scores),
            e["name"],
            f"{e['all']:.3f}",
            rank_among(e["held"], held_scores),
            f"{e['held']:.3f}",
        ]
        for e in entries
    ]


def plot_per_dataset(
    challenge: Dict, ours: Dict[int, float], baseline: Dict[int, float], out
) -> None:
    """Strip plot: every team's IoU per dataset vs ours and the baseline."""
    apply_style()
    datasets = sorted(ours)
    fig, ax = plt.subplots(figsize=(10, 4.8))

    ax.axvspan(
        datasets.index(HELD_OUT[0]) - 0.5,
        datasets.index(HELD_OUT[-1]) + 0.5,
        color="#e4e3df",
        alpha=0.5,
        zorder=0,
    )
    ax.text(
        datasets.index(HELD_OUT[0]) + 0.5,
        0.02,
        "like-for-like\n(never seen)",
        ha="center",
        va="bottom",
        fontsize=8,
        color=TEXT_SECONDARY,
    )

    for i, n in enumerate(datasets):
        teams = list(challenge["per_dataset"][str(n)].values())
        ax.scatter(
            [i] * len(teams),
            teams,
            s=28,
            color=COLOR_CHALLENGE,
            alpha=0.75,
            edgecolor=SURFACE,
            linewidth=0.8,
            label="Challenge teams (one dot each)" if i == 0 else None,
            zorder=2,
        )
        ax.hlines(
            np.mean(teams),
            i - 0.3,
            i + 0.3,
            color="#0b0b0b",
            linewidth=1.5,
            label="Mean of 10 teams" if i == 0 else None,
            zorder=3,
        )
    x = np.arange(len(datasets))
    ax.scatter(
        x,
        [baseline[n] for n in datasets],
        s=40,
        marker="s",
        color=COLOR_BASELINE,
        edgecolor=SURFACE,
        linewidth=1,
        label=BASELINE,
        zorder=4,
    )
    ax.scatter(
        x,
        [ours[n] for n in datasets],
        s=90,
        marker="D",
        color=COLOR_MODEL,
        edgecolor=SURFACE,
        linewidth=1.5,
        label=OURS,
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in datasets])
    ax.set_xlabel("Dataset (test sequence)")
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 1.02)
    ax.set_title("Binary segmentation: per-dataset IoU vs challenge teams")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_overall(entries: List[dict], out) -> None:
    """Two ranked horizontal bar charts: all-10 and held-out-only IoU."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = (
        ("all", "All 10 datasets (1-8 not like-for-like)"),
        ("held", "Held-out datasets 9-10 only"),
    )
    for ax, (key, title) in zip(axes, panels):
        ordered = sorted(entries, key=lambda e: e[key])  # best at top
        colors = [
            (
                COLOR_MODEL
                if e["name"] == OURS
                else (
                    COLOR_BASELINE
                    if e["name"] == BASELINE
                    else COLOR_CHALLENGE
                )
            )
            for e in ordered
        ]
        y = np.arange(len(ordered))
        ax.barh(
            y,
            [e[key] for e in ordered],
            color=colors,
            edgecolor=SURFACE,
            linewidth=1,
            height=0.8,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([e["name"] for e in ordered])
        for tick, e in zip(ax.get_yticklabels(), ordered):
            if e["name"] in (OURS, BASELINE):
                tick.set_fontweight("bold")
        for yi, e in zip(y, ordered):
            ax.text(
                e[key] + 0.01,
                yi,
                f"{e[key]:.3f}",
                va="center",
                fontsize=8,
                color=TEXT_SECONDARY,
            )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Mean IoU")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=paths.MODEL_METRICS)
    parser.add_argument(
        "--baseline", type=Path, default=paths.BASELINE_METRICS
    )
    parser.add_argument(
        "--challenge", type=Path, default=paths.CHALLENGE_RESULTS
    )
    parser.add_argument("--table-dir", type=Path, default=paths.TABLES_DIR)
    parser.add_argument("--figure-dir", type=Path, default=paths.FIGURES_DIR)
    args = parser.parse_args()

    model = load_metrics(args.model)
    baseline = load_metrics(args.baseline)
    with open(args.challenge) as f:
        challenge = json.load(f)

    ours = sequence_scores(model)
    rows = per_dataset_rows(challenge, ours)
    write_tables(
        [
            "Dataset",
            "Our IoU",
            "Best team",
            "vs best",
            "Mean of 10 teams",
            "vs mean",
            "Our rank",
            "Like-for-like?",
        ],
        rows,
        args.table_dir / "challenge_per_dataset.md",
        args.table_dir / "challenge_per_dataset.csv",
        left_aligned=(0, 2, 7),
    )

    entries = leaderboard(challenge, model, baseline)
    write_tables(
        [
            "Rank (all 10)",
            "Method",
            "IoU, all 10",
            "Rank (9-10)",
            "IoU, held-out 9-10",
        ],
        leaderboard_rows(entries),
        args.table_dir / "challenge_leaderboard.md",
        args.table_dir / "challenge_leaderboard.csv",
        left_aligned=(1,),
    )

    plot_per_dataset(
        challenge,
        ours,
        sequence_scores(baseline),
        args.figure_dir / "challenge_per_dataset.png",
    )
    plot_overall(entries, args.figure_dir / "challenge_overall.png")

    for name in ("challenge_per_dataset", "challenge_leaderboard"):
        print((args.table_dir / f"{name}.md").read_text())
    print(f"Saved figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
