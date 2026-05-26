"""
Normalize NeurIPS (1-6), ICLR ({1,3,5,6,8,10}), and AI judge (1-10) scores
onto a common CONTINUOUS 1-10 scale, then compute accuracy metrics.

Normalization strategy: linear stretch (Option B from brainstorm.md).

  NeurIPS (1-6 continuous weighted mean) -> 1 + (x - 1) * 9/5
  ICLR    (1-10 continuous weighted mean, raw reviewers pick from {1,3,5,6,8,10})
          -> identity (already 1-10)
  AI Judge (1-10 integer)                -> identity

No binning. MAE/Spearman/Pearson computed on continuous 1-10 values.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Optional

import numpy as np
from scipy import stats

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"
NEURIPS_GT_DIR = REPO_ROOT / "neurips-2025-dataset-v2" / "open-reviews-ground_truth"
ICLR_GT_DIR = REPO_ROOT / "iclr-2025-dataset" / "open-reviews-ground_truth"
BASELINE_RESULTS = EVALS_DIR / "baseline_results.jsonl"

OUT_DIR = pathlib.Path(__file__).resolve().parent
GT_NEURIPS_FILE = OUT_DIR / "ground_truth_neurips.json"
GT_ICLR_FILE = OUT_DIR / "ground_truth_iclr.json"
AI_NEURIPS_FILE = OUT_DIR / "ai_judge_neurips.json"
AI_ICLR_FILE = OUT_DIR / "ai_judge_iclr.json"
RESULTS_FILE = OUT_DIR / "results_normalized.json"


def neurips_to_10(score: float) -> float:
    """Linear stretch NeurIPS 1-6 onto 1-10."""
    return round(1 + (score - 1) * 9 / 5, 4)


def iclr_to_10(score: float) -> float:
    """ICLR weighted_mean_overall is already on 1-10 (reviewers pick from {1,3,5,6,8,10})."""
    return round(score, 4)


def ai_to_10(score: float) -> float:
    """AI judge outputs integer 1-10 directly."""
    return float(score)


def parse_ai_judge_score(raw_response: str) -> Optional[int]:
    m = re.search(r"<overall_score>\s*(\d+)\s*</overall_score>", raw_response)
    return int(m.group(1)) if m else None


def parse_ai_judge_confidence(raw_response: str) -> Optional[int]:
    m = re.search(r"<score_confidence>\s*(\d+)\s*</score_confidence>", raw_response)
    return int(m.group(1)) if m else None


def load_ground_truths() -> dict[str, dict]:
    gt = {}
    for path, venue, normalize_fn, raw_scale in [
        (NEURIPS_GT_DIR, "neurips2025", neurips_to_10, "1-6"),
        (ICLR_GT_DIR, "iclr2025", iclr_to_10, "{1,3,5,6,8,10}"),
    ]:
        for f in path.glob("*.json"):
            d = json.loads(f.read_text())
            pid = d["paper_id"]
            raw = d["ground_truth"]["weighted_mean_overall"]
            gt[pid] = {
                "paper_id": pid,
                "title": d.get("title", ""),
                "venue": venue,
                "raw_scale": raw_scale,
                "score_raw": round(raw, 4),
                "score_normalized": normalize_fn(raw),
                "accept": d["ground_truth"]["accept"],
            }
    return gt


def load_ai_predictions() -> dict[str, dict]:
    preds = {}
    with open(BASELINE_RESULTS) as f:
        for line in f:
            d = json.loads(line)
            if d.get("event") != "baseline_prediction":
                continue
            raw = d.get("raw_response", "")
            ai_raw = parse_ai_judge_score(raw)
            pid = d["paper_id"]
            preds[pid] = {
                "paper_id": pid,
                "title": d.get("title", ""),
                "venue": d["conference"] if d["conference"] != "neurips" else "neurips2025",
                "score_raw": ai_raw,
                "score_normalized": ai_to_10(ai_raw) if ai_raw is not None else None,
                "prediction": d.get("prediction"),
                "confidence": parse_ai_judge_confidence(raw),
                "correct": d.get("correct"),
            }
            if preds[pid]["venue"] == "iclr":
                preds[pid]["venue"] = "iclr2025"
    return preds


def write_organized_files(gt: dict, ai: dict):
    """Filter GT to papers that have an AI prediction so both sides align (n=48)."""
    tested_ids = set(ai.keys())
    gt_tested = {pid: g for pid, g in gt.items() if pid in tested_ids}

    def split_by_venue(records: dict, venue: str) -> list:
        return sorted(
            [r for r in records.values() if r["venue"] == venue],
            key=lambda x: x["paper_id"],
        )

    gt_meta = {
        "normalization": "linear stretch to continuous 1-10",
        "neurips_formula": "1 + (raw - 1) * 9/5",
        "iclr_formula": "identity (raw weighted mean already on 1-10)",
        "accept_boundary_note": "no fixed numeric threshold; use venue's `accept` boolean",
    }

    GT_NEURIPS_FILE.write_text(json.dumps({
        "venue": "neurips2025",
        "raw_scale": "1-6 continuous (confidence-weighted mean)",
        **gt_meta,
        "papers": split_by_venue(gt_tested, "neurips2025"),
    }, indent=2))

    GT_ICLR_FILE.write_text(json.dumps({
        "venue": "iclr2025",
        "raw_scale": "1-10 continuous (weighted mean of discrete {1,3,5,6,8,10})",
        **gt_meta,
        "papers": split_by_venue(gt_tested, "iclr2025"),
    }, indent=2))

    ai_meta = {
        "normalization": "identity (AI judge already on 1-10 integer)",
    }

    AI_NEURIPS_FILE.write_text(json.dumps({
        "venue": "neurips2025",
        "raw_scale": "1-10 integer (from <overall_score>)",
        **ai_meta,
        "papers": split_by_venue(ai, "neurips2025"),
    }, indent=2))

    AI_ICLR_FILE.write_text(json.dumps({
        "venue": "iclr2025",
        "raw_scale": "1-10 integer (from <overall_score>)",
        **ai_meta,
        "papers": split_by_venue(ai, "iclr2025"),
    }, indent=2))

    for p, label in [
        (GT_NEURIPS_FILE, "GT NeurIPS"),
        (GT_ICLR_FILE, "GT ICLR"),
        (AI_NEURIPS_FILE, "AI NeurIPS"),
        (AI_ICLR_FILE, "AI ICLR"),
    ]:
        n = len(json.loads(p.read_text())["papers"])
        print(f"  -> {p.name}  ({n} papers, {label})")


def compute_metrics(records: list[dict]) -> dict:
    valid = [r for r in records if r["ai_score_norm"] is not None and r["gt_score_norm"] is not None]
    ai = np.array([r["ai_score_norm"] for r in valid], dtype=float)
    gt = np.array([r["gt_score_norm"] for r in valid], dtype=float)

    def stats_for(ai_v, gt_v, records_v):
        mae = float(np.mean(np.abs(ai_v - gt_v)))
        rmse = float(np.sqrt(np.mean((ai_v - gt_v) ** 2)))
        bias = float(np.mean(ai_v - gt_v))
        sr, sp = stats.spearmanr(ai_v, gt_v)
        pr, pp = stats.pearsonr(ai_v, gt_v)
        binary_acc = float(np.mean([r["correct"] for r in records_v if r["correct"] is not None]))
        return {
            "n": len(records_v),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mean_error_ai_minus_gt": round(bias, 4),
            "spearman_r": round(float(sr), 4),
            "spearman_p": round(float(sp), 4),
            "pearson_r": round(float(pr), 4),
            "pearson_p": round(float(pp), 4),
            "binary_accuracy": round(binary_acc, 4),
        }

    overall = stats_for(ai, gt, valid)
    by_venue = {}
    for venue in ("neurips2025", "iclr2025"):
        v = [r for r in valid if r["venue"] == venue]
        ai_v = np.array([r["ai_score_norm"] for r in v], dtype=float)
        gt_v = np.array([r["gt_score_norm"] for r in v], dtype=float)
        if len(v):
            by_venue[venue] = stats_for(ai_v, gt_v, v)

    return {"overall": overall, "by_venue": by_venue}


def main():
    print("Loading ground truths...")
    gt = load_ground_truths()

    print("Loading AI judge predictions...")
    ai = load_ai_predictions()

    print("Writing organized per-venue files...")
    write_organized_files(gt, ai)

    print("Merging for metrics...")
    records = []
    for pid, g in gt.items():
        a = ai.get(pid)
        if a is None:
            continue
        records.append({
            "paper_id": pid,
            "title": g["title"],
            "venue": g["venue"],
            "gt_score_raw": g["score_raw"],
            "gt_score_norm": g["score_normalized"],
            "gt_accept": g["accept"],
            "ai_score_raw": a["score_raw"],
            "ai_score_norm": a["score_normalized"],
            "ai_predicted_accept": a["prediction"] == "ACCEPT" if a["prediction"] else None,
            "ai_confidence": a["confidence"],
            "correct": a["correct"],
        })

    print("Computing metrics on continuous 1-10...")
    metrics = compute_metrics(records)

    RESULTS_FILE.write_text(json.dumps({
        "normalization": "linear stretch to continuous 1-10",
        "neurips_formula": "1 + (raw - 1) * 9/5",
        "iclr_formula": "identity",
        "ai_formula": "identity",
        "metrics": metrics,
        "papers": records,
    }, indent=2))
    print(f"  -> {RESULTS_FILE.name}")

    o = metrics["overall"]
    print("\n=== RESULTS (continuous 1-10) ===")
    print(f"  n papers:        {o['n']}")
    print(f"  MAE:             {o['mae']:.3f}  (1-10 points)")
    print(f"  RMSE:            {o['rmse']:.3f}")
    print(f"  Mean error:      {o['mean_error_ai_minus_gt']:+.3f}  (AI - GT)")
    print(f"  Spearman r:      {o['spearman_r']:.3f}  (p={o['spearman_p']:.4f})")
    print(f"  Pearson r:       {o['pearson_r']:.3f}  (p={o['pearson_p']:.4f})")
    print(f"  Binary accuracy: {o['binary_accuracy']:.1%}")
    print()
    for venue, m in metrics["by_venue"].items():
        print(f"  {venue} (n={m['n']}):  MAE={m['mae']:.2f}  Spearman={m['spearman_r']:.3f}  Binary={m['binary_accuracy']:.1%}")


if __name__ == "__main__":
    main()
