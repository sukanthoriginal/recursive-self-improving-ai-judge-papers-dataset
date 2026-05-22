"""Plot the baseline sample distribution by normalized score and acceptance."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt


EVALS_DIR = Path(__file__).resolve().parent
PAPERS_CSV = EVALS_DIR / "data" / "papers.csv"
DEFAULT_OUT = EVALS_DIR / "baseline_sample_distribution.png"

RAW_SCORE_CONFIG = {
    "iclr2025": {
        "edges": list(range(1, 10)) + [10.01],
        "xticks": list(range(1, 11)),
        "xlim": (1, 10),
        "xlabel": "weighted_mean_overall (ICLR 2025 scale: 1-10)",
        "out": EVALS_DIR / "baseline_sample_iclr_raw_distribution.png",
    },
    "neurips2025": {
        "edges": [round(1.0 + 0.5 * i, 1) for i in range(11)],
        "xticks": [round(1.0 + 0.5 * i, 1) for i in range(11)],
        "xlim": (1, 6),
        "xlabel": "weighted_mean_overall (NeurIPS 2025 scale: 1-6)",
        "out": EVALS_DIR / "baseline_sample_neurips_raw_distribution.png",
    },
}


def load_papers() -> list[dict]:
    with PAPERS_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_sample(papers: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size > len(papers):
        raise ValueError(f"sample_size={sample_size} exceeds available papers={len(papers)}")
    return random.Random(seed).sample(papers, sample_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the baseline sample distribution.")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--conference",
        choices=["all", "iclr2025", "neurips2025"],
        default="all",
        help="Subset to plot. Use with --raw-score for conference-scale plots.",
    )
    parser.add_argument(
        "--raw-score",
        action="store_true",
        help="Plot conference raw weighted_mean_overall instead of normalized scores.",
    )
    parser.add_argument(
        "--score-column",
        default="score_percentile_1_10",
        choices=["score_percentile_1_10", "score_z_1_10_clipped"],
        help="Normalized 1-10 score to plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = select_sample(load_papers(), args.sample_size, args.seed)
    if args.conference != "all":
        sample = [paper for paper in sample if paper["conference"] == args.conference]

    if args.raw_score and args.conference == "all":
        raise ValueError("--raw-score requires --conference iclr2025 or --conference neurips2025")

    if args.raw_score:
        cfg = RAW_SCORE_CONFIG[args.conference]
        score_column = "weighted_mean_overall"
        edges = cfg["edges"]
        xticks = cfg["xticks"]
        xlim = cfg["xlim"]
        xlabel = cfg["xlabel"]
        out = args.out if args.out != DEFAULT_OUT else cfg["out"]
    else:
        score_column = args.score_column
        edges = list(range(1, 11)) + [10.01]
        xticks = list(range(1, 11))
        xlim = (1, 10)
        xlabel = f"{args.score_column} (normalized 1-10 within venue)"
        out = args.out

    rows = [
        {
            "score": float(paper[score_column]),
            "accept": paper["accept_reject"] == "accept",
            "conference": paper["conference"],
        }
        for paper in sample
    ]

    accept_vals = [row["score"] for row in rows if row["accept"]]
    reject_vals = [row["score"] for row in rows if not row["accept"]]
    bin_totals = [sum(1 for row in rows if lo <= row["score"] < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    y_top = max(bin_totals) + 1 if bin_totals else 1

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hist(
        [accept_vals, reject_vals],
        bins=edges,
        stacked=True,
        color=["#2ca02c", "#d62728"],
        label=["Accept", "Reject"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    title_prefix = "Baseline sample distribution"
    if args.conference != "all":
        title_prefix += f" - {args.conference}"
    ax.set_title(f"{title_prefix} - n={len(rows)}, seed={args.seed}")
    ax.set_xticks(xticks)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, y_top)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    counts = {}
    for row in rows:
        item = counts.setdefault(row["conference"], {"accept": 0, "reject": 0})
        item["accept" if row["accept"] else "reject"] += 1
    subtitle = " | ".join(
        f"{conf}: A={counts[conf]['accept']} R={counts[conf]['reject']}" for conf in sorted(counts)
    )
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(out, dpi=140)

    print(f"saved {out}")
    print(f"n={len(rows)}, accepts={len(accept_vals)}, rejects={len(reject_vals)}")
    print(counts)


if __name__ == "__main__":
    main()
