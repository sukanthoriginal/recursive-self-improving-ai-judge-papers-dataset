"""Build the eval paper manifest from the current dataset folders."""

from __future__ import annotations

import csv
import json
import statistics
from bisect import bisect_left, bisect_right
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "data" / "papers.csv"
NORMALIZATION_POOLS = Path(__file__).resolve().parent / "cache" / "venue_score_pools.json"

DATASETS = {
    "iclr2025": REPO_ROOT / "iclr-2025-dataset",
    "neurips2025": REPO_ROOT / "neurips-2025-dataset-v2",
}

FIELDS = [
    "paper_id",
    "title",
    "conference",
    "path",
    "accept_reject",
    "accept_type",
    "weighted_mean_overall",
    "score_z_within_venue",
    "score_z_1_10_clipped",
    "score_percentile_within_venue",
    "score_percentile_1_10",
    "normalization_pool_n",
    "decision",
    "venue",
    "dataset",
    "slug",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_for(conference: str, dataset_dir: Path, gt_path: Path) -> dict:
    slug = gt_path.stem
    meta = load_json(dataset_dir / "papers" / f"{slug}.meta.json")
    gt = load_json(gt_path)["ground_truth"]
    pdf_path = dataset_dir / "papers-anonymized" / f"{slug}_anon.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing anonymized PDF for {slug}: {pdf_path}")

    accept = bool(gt["accept"])
    decision = meta["decision"]
    return {
        "paper_id": meta["paper_id"],
        "title": meta["title"],
        "conference": conference,
        "path": pdf_path.relative_to(REPO_ROOT).as_posix(),
        "accept_reject": "accept" if accept else "reject",
        "accept_type": decision if accept else "Reject",
        "weighted_mean_overall": gt["weighted_mean_overall"],
        "score_z_within_venue": "",
        "score_z_1_10_clipped": "",
        "score_percentile_within_venue": "",
        "score_percentile_1_10": "",
        "normalization_pool_n": "",
        "decision": decision,
        "venue": meta.get("venue", ""),
        "dataset": dataset_dir.name,
        "slug": slug,
    }


def percentile_from_pool(score: float, sorted_scores: list[float]) -> float:
    n = len(sorted_scores)
    if n <= 1:
        return 0.5
    less = bisect_left(sorted_scores, score)
    greater = bisect_right(sorted_scores, score)
    midrank_zero_based = less + ((greater - less) - 1) / 2
    return midrank_zero_based / (n - 1)


def load_normalization_pools() -> dict[str, list[float]]:
    if not NORMALIZATION_POOLS.exists():
        raise FileNotFoundError(
            f"Missing full-venue normalization pool: {NORMALIZATION_POOLS}. "
            "Run `uv run python evals/fetch_venue_score_pools.py` first."
        )

    data = load_json(NORMALIZATION_POOLS)
    pools = {}
    for conference in DATASETS:
        scores = data.get(conference, {}).get("scores", [])
        if not scores:
            raise ValueError(f"No full-venue scores found for {conference} in {NORMALIZATION_POOLS}")
        pools[conference] = [float(score) for score in scores]
    return pools


def add_normalized_scores(rows: list[dict]) -> None:
    pools = load_normalization_pools()

    for conference in DATASETS:
        conf_rows = [row for row in rows if row["conference"] == conference]
        scores = pools[conference]
        sorted_scores = sorted(scores)
        mean = statistics.mean(scores)
        stdev = statistics.pstdev(scores)

        for row in conf_rows:
            score = float(row["weighted_mean_overall"])
            z = (score - mean) / stdev if stdev else 0.0
            # Bounded display scale: -2 sigma -> 1, mean -> 5.5, +2 sigma -> 10.
            z_1_10 = 1.0 + 9.0 * min(max((z + 2.0) / 4.0, 0.0), 1.0)
            row["score_z_within_venue"] = round(z, 4)
            row["score_z_1_10_clipped"] = round(z_1_10, 4)
            percentile = percentile_from_pool(score, sorted_scores)
            percentile_1_10 = 1.0 + 9.0 * percentile
            row["score_percentile_within_venue"] = round(percentile, 4)
            row["score_percentile_1_10"] = round(percentile_1_10, 4)
            row["normalization_pool_n"] = len(scores)


def main() -> None:
    rows = []
    for conference, dataset_dir in DATASETS.items():
        gt_dir = dataset_dir / "open-reviews-ground_truth"
        rows.extend(row_for(conference, dataset_dir, path) for path in sorted(gt_dir.glob("*.json")))

    add_normalized_scores(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_conf = {
        conf: {
            "accept": sum(1 for row in rows if row["conference"] == conf and row["accept_reject"] == "accept"),
            "reject": sum(1 for row in rows if row["conference"] == conf and row["accept_reject"] == "reject"),
        }
        for conf in DATASETS
    }
    print(f"Wrote {OUT.relative_to(REPO_ROOT)} with {len(rows)} papers")
    print(json.dumps(by_conf, indent=2))


if __name__ == "__main__":
    main()
