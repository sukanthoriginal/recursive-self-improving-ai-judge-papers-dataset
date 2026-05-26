"""
Analysis of RSI-AI-Judge vs Ground Truth — CONTINUOUS 1-10 scale.

Two independent tasks:
  Task A: Score prediction (regression on continuous 1-10)
  Task B: Binary accept/reject prediction (independent of score)

Stratification by rounded GT score (1-10 integer bin), only non-empty bins shown.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import stats

ND  = pathlib.Path(__file__).resolve().parent.parent / "normalization-dataset"
OUT = pathlib.Path(__file__).resolve().parent


# ─── load ─────────────────────────────────────────────────────────────────────
def load_records() -> list[dict]:
    records = []
    for venue_key in ("neurips", "iclr"):
        gt = json.loads((ND / f"ground_truth_{venue_key}.json").read_text())["papers"]
        ai = json.loads((ND / f"ai_judge_{venue_key}.json").read_text())["papers"]
        gt_map = {p["paper_id"]: p for p in gt}
        for p in ai:
            g = gt_map[p["paper_id"]]
            records.append({
                "paper_id":            p["paper_id"],
                "title":               p["title"],
                "venue":               venue_key,
                "gt_score_raw":        g["score_raw"],
                "gt_score_norm":       g["score_normalized"],
                "gt_accept":           g["accept"],
                "ai_score_raw":        p["score_raw"],
                "ai_score_norm":       p["score_normalized"],
                "ai_predicted_accept": p["prediction"] == "ACCEPT",
                "ai_confidence":       p["confidence"],
            })
    return records


def gt_bin(score: float) -> int:
    """Round continuous GT score to nearest integer in [1, 10] for stratification."""
    return int(np.clip(round(score), 1, 10))


# ─── Task A: Score regression ─────────────────────────────────────────────────
def score_regression(records: list[dict]) -> dict:
    if not records:
        return {}
    gt = np.array([r["gt_score_norm"] for r in records], dtype=float)
    ai = np.array([r["ai_score_norm"] for r in records], dtype=float)
    err = ai - gt

    sr = stats.spearmanr(gt, ai)
    pr = stats.pearsonr(gt, ai)
    r2 = 1.0 - np.sum(err ** 2) / np.sum((gt - gt.mean()) ** 2)

    overall = {
        "n":               int(len(records)),
        "mae":             round(float(np.mean(np.abs(err))), 3),
        "rmse":            round(float(np.sqrt(np.mean(err ** 2))), 3),
        "mean_error":      round(float(np.mean(err)), 3),
        "std_error":       round(float(np.std(err)), 3),
        "within_1_point":  round(float(np.mean(np.abs(err) <= 1.0)), 3),
        "within_2_points": round(float(np.mean(np.abs(err) <= 2.0)), 3),
        "spearman_r":      round(float(sr[0]), 3),
        "pearson_r":       round(float(pr[0]), 3),
        "r_squared":       round(float(r2), 3),
    }

    # Per-bin stratification: round GT to nearest int in [1,10]
    per_bin = {}
    for b in range(1, 11):
        bucket = [r for r in records if gt_bin(r["gt_score_norm"]) == b]
        if not bucket:
            continue
        ai_v = np.array([r["ai_score_norm"] for r in bucket], dtype=float)
        gt_v = np.array([r["gt_score_norm"] for r in bucket], dtype=float)
        errors = ai_v - gt_v
        per_bin[str(b)] = {
            "n":                int(len(bucket)),
            "gt_score_mean":    round(float(np.mean(gt_v)), 2),
            "ai_score_mean":    round(float(np.mean(ai_v)), 2),
            "ai_score_min":     round(float(np.min(ai_v)), 2),
            "ai_score_max":     round(float(np.max(ai_v)), 2),
            "mae":              round(float(np.mean(np.abs(errors))), 2),
            "mean_error":       round(float(np.mean(errors)), 2),
        }
    return {"overall": overall, "per_gt_bin": per_bin}


# ─── Task B: Binary accept/reject classification ─────────────────────────────
def binary_classification(records: list[dict]) -> dict:
    if not records:
        return {}
    n = len(records)
    tp = sum(1 for r in records if r["gt_accept"] and r["ai_predicted_accept"])
    tn = sum(1 for r in records if not r["gt_accept"] and not r["ai_predicted_accept"])
    fp = sum(1 for r in records if not r["gt_accept"] and r["ai_predicted_accept"])
    fn = sum(1 for r in records if r["gt_accept"] and not r["ai_predicted_accept"])

    def safe_div(a, b):
        return round(a / b, 3) if b else None

    overall = {
        "n":              n,
        "confusion":      {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "accuracy":       safe_div(tp + tn, n),
        "precision":      safe_div(tp, tp + fp),
        "recall":         safe_div(tp, tp + fn),
        "specificity":    safe_div(tn, tn + fp),
        "f1":             safe_div(2 * tp, 2 * tp + fp + fn),
        "gt_accept_rate": safe_div(tp + fn, n),
        "ai_accept_rate": safe_div(tp + fp, n),
    }

    per_bin = {}
    for b in range(1, 11):
        bucket = [r for r in records if gt_bin(r["gt_score_norm"]) == b]
        if not bucket:
            continue
        bn = len(bucket)
        gt_acc = sum(1 for r in bucket if r["gt_accept"])
        ai_acc = sum(1 for r in bucket if r["ai_predicted_accept"])
        correct = sum(1 for r in bucket if r["gt_accept"] == r["ai_predicted_accept"])
        per_bin[str(b)] = {
            "n":               bn,
            "gt_accept_count": gt_acc,
            "gt_reject_count": bn - gt_acc,
            "gt_accept_rate":  round(gt_acc / bn, 3),
            "ai_accept_count": ai_acc,
            "ai_reject_count": bn - ai_acc,
            "ai_accept_rate":  round(ai_acc / bn, 3),
            "binary_correct":  correct,
            "binary_accuracy": round(correct / bn, 3),
        }
    return {"overall": overall, "per_gt_bin": per_bin}


# ─── divergence ──────────────────────────────────────────────────────────────
def divergence_analysis(records: list[dict]) -> dict:
    """Where score-implied decision differs from actual accept label.

    Implied threshold: score >= 5.5 (between Borderline Reject and Borderline
    Accept on the continuous 1-10 scale, matches ICLR's 5-vs-6 boundary)."""
    THRESHOLD = 5.5

    def implied(score): return score >= THRESHOLD

    gt_mismatches  = [r for r in records if implied(r["gt_score_norm"]) != r["gt_accept"]]
    ai_inconsistent = [r for r in records if implied(r["ai_score_norm"]) != r["ai_predicted_accept"]]

    return {
        "implied_accept_threshold": THRESHOLD,
        "gt_score_vs_accept_mismatches": {
            "count": len(gt_mismatches),
            "rate":  round(len(gt_mismatches) / len(records), 3) if records else 0,
            "ai_got_binary_right_on_these": sum(
                1 for r in gt_mismatches if r["gt_accept"] == r["ai_predicted_accept"]
            ),
            "papers": [
                {
                    "paper_id":            r["paper_id"],
                    "venue":               r["venue"],
                    "title":               r["title"][:80],
                    "gt_score":            r["gt_score_norm"],
                    "gt_accept":           r["gt_accept"],
                    "ai_score":            r["ai_score_norm"],
                    "ai_predicted_accept": r["ai_predicted_accept"],
                    "ai_binary_correct":   r["gt_accept"] == r["ai_predicted_accept"],
                }
                for r in gt_mismatches
            ],
        },
        "ai_internal_inconsistency": {
            "count": len(ai_inconsistent),
            "rate":  round(len(ai_inconsistent) / len(records), 3) if records else 0,
            "note": "AI papers where its own score crosses the 5.5 threshold differently from its own ACCEPT/REJECT call",
        },
    }


# ─── run ──────────────────────────────────────────────────────────────────────
records = load_records()
neurips = [r for r in records if r["venue"] == "neurips"]
iclr    = [r for r in records if r["venue"] == "iclr"]

result = {
    "scale": "continuous 1-10",
    "normalization": {
        "neurips": "1 + (raw - 1) * 9/5  (linear stretch from 1-6)",
        "iclr":    "identity (weighted mean already on 1-10)",
        "ai":      "identity (raw 1-10 integer)",
    },
    "task_A_score_regression": {
        "description": "How close is AI's continuous 1-10 score to GT?",
        "overall":     score_regression(records),
        "neurips2025": score_regression(neurips),
        "iclr2025":    score_regression(iclr),
    },
    "task_B_binary_classification": {
        "description": "How well does AI predict the GT binary accept/reject decision?",
        "overall":     binary_classification(records),
        "neurips2025": binary_classification(neurips),
        "iclr2025":    binary_classification(iclr),
    },
    "task_divergence": {
        "description": "Where GT score-implied decision (>=5.5) and actual accept disagree.",
        "overall":     divergence_analysis(records),
        "neurips2025": divergence_analysis(neurips),
        "iclr2025":    divergence_analysis(iclr),
    },
    "papers": sorted(records, key=lambda r: (r["venue"], r["paper_id"])),
}

(OUT / "summary.json").write_text(json.dumps(result, indent=2))


# ─── pretty print ────────────────────────────────────────────────────────────
def print_score_section(title, data):
    o = data["overall"]
    print(f"\n  {title} (n={o['n']})")
    print(f"    MAE:              {o['mae']:.2f}  (1-10 points)")
    print(f"    RMSE:             {o['rmse']:.2f}")
    print(f"    Mean error:       {o['mean_error']:+.2f}")
    print(f"    Within 1 point:   {o['within_1_point']:.1%}")
    print(f"    Within 2 points:  {o['within_2_points']:.1%}")
    print(f"    Spearman r:       {o['spearman_r']:.3f}")
    print(f"    Pearson r:        {o['pearson_r']:.3f}")
    print(f"    R²:               {o['r_squared']:.3f}")


def print_binary_section(title, data):
    o = data["overall"]
    c = o["confusion"]
    print(f"\n  {title} (n={o['n']})")
    print(f"    TP/TN/FP/FN:      {c['TP']}/{c['TN']}/{c['FP']}/{c['FN']}")
    print(f"    Accuracy:         {o['accuracy']:.1%}")
    print(f"    Precision:        {o['precision']:.1%}")
    print(f"    Recall:           {o['recall']:.1%}")
    print(f"    F1:               {o['f1']:.3f}")
    print(f"    GT accept rate:   {o['gt_accept_rate']:.1%}")
    print(f"    AI accept rate:   {o['ai_accept_rate']:.1%}")


print("=" * 75)
print("TASK A — SCORE REGRESSION  (continuous 1-10)")
print("=" * 75)
for v, d in result["task_A_score_regression"].items():
    if v == "description": continue
    print_score_section(v, d)

print("\n  Per GT bin (OVERALL):")
print("  ┌──────────┬─────┬──────────┬──────────┬───────┬────────────┐")
print("  │ GT bin   │  n  │  GT mean │  AI mean │  MAE  │ mean error │")
print("  ├──────────┼─────┼──────────┼──────────┼───────┼────────────┤")
for b, pb in result["task_A_score_regression"]["overall"]["per_gt_bin"].items():
    print(f"  │   ~{int(b):2d}    │ {pb['n']:3d} │   {pb['gt_score_mean']:5.2f}  │   {pb['ai_score_mean']:5.2f}  │ {pb['mae']:5.2f} │   {pb['mean_error']:+6.2f}   │")
print("  └──────────┴─────┴──────────┴──────────┴───────┴────────────┘")

print("\n" + "=" * 75)
print("TASK B — BINARY ACCEPT/REJECT  (independent of score)")
print("=" * 75)
for v, d in result["task_B_binary_classification"].items():
    if v == "description": continue
    print_binary_section(v, d)

print("\n  Per GT bin (OVERALL):")
print("  ┌──────────┬─────┬────────────┬────────────┬──────────┐")
print("  │ GT bin   │  n  │ GT accept% │ AI accept% │ binary % │")
print("  ├──────────┼─────┼────────────┼────────────┼──────────┤")
for b, pb in result["task_B_binary_classification"]["overall"]["per_gt_bin"].items():
    print(f"  │   ~{int(b):2d}    │ {pb['n']:3d} │    {pb['gt_accept_rate']:>5.1%}  │    {pb['ai_accept_rate']:>5.1%}  │   {pb['binary_accuracy']:>5.1%}  │")
print("  └──────────┴─────┴────────────┴────────────┴──────────┘")

print("\n" + "=" * 75)
print("DIVERGENCE — where GT score (>=5.5) and GT accept disagree")
print("=" * 75)
d = result["task_divergence"]["overall"]
m = d["gt_score_vs_accept_mismatches"]
print(f"\n  GT mismatch papers:")
print(f"    Count:                       {m['count']} / {len(records)} ({m['rate']:.1%})")
print(f"    AI got binary right on:      {m['ai_got_binary_right_on_these']} / {m['count']}")
print(f"\n  AI self-inconsistency (score vs accept call):")
print(f"    Count:                       {d['ai_internal_inconsistency']['count']} / {len(records)} ({d['ai_internal_inconsistency']['rate']:.1%})")

print(f"\nOutput: {OUT / 'summary.json'}")
