# Score Normalization Brainstorm

## The Problem

Three score spaces need to be compared:

| Source | Scale | Type |
|---|---|---|
| NeurIPS 2025 ground truth | 1–6 (continuous, conf-weighted mean) | Continuous |
| ICLR 2025 ground truth | {1,3,5,6,8,10} (discrete) | Ordinal (6 buckets) |
| AI Judge output | 1–10 (as prompted) | Continuous |

Note on thresholds: there is **no clean numerical accept threshold** for either venue. The actual accept/reject decision is holistic (AC judgment, rebuttal, quota), and is only ~70–80% correlated with the weighted mean score. Mismatch papers (low-score accepts, high-score rejects) exist in both datasets deliberately.

---

## Core Insight

Both venues use **6 semantic levels** that map 1:1:

| Semantic Level | NeurIPS | ICLR |
|---|---|---|
| Strong Reject | 1 | 1 |
| Reject | 2 | 3 |
| Borderline Reject | 3 | 5 |
| Borderline Accept | 4 | 6 |
| Accept | 5 | 8 |
| Strong Accept | 6 | 10 |

ICLR's gaps (1→3→5→6→8→10) are intentional. The 5→6 gap is the accept/reject boundary.
NeurIPS 1–6 integer labels map cleanly onto these same 6 semantic buckets.

This means **ICLR's {1,3,5,6,8,10} scale is already the natural normalized target.**

---

## Option A: Semantic Label Mapping (Recommended)

Map NeurIPS continuous scores to the ICLR discrete scale via the 6 semantic buckets:

```
NeurIPS score → Normalized (ICLR-equivalent)
[1.0, 1.5)   → 1   (Strong Reject)
[1.5, 2.5)   → 3   (Reject)
[2.5, 3.5)   → 5   (Borderline Reject)
[3.5, 4.5)   → 6   (Borderline Accept)
[4.5, 5.5)   → 8   (Accept)
[5.5, 6.0]   → 10  (Strong Accept)
```

AI Judge score (1–10 continuous) → round/bin to nearest ICLR bucket:
```
[1, 2)   → 1
[2, 4)   → 3
[4, 5.5) → 5
[5.5, 7) → 6
[7, 9)   → 8
[9, 10]  → 10
```

**Pros:** Semantically grounded. Respects that 5→6 is the meaningful boundary, not 5→6 numerically equal steps.
**Cons:** Loses continuous signal within a bucket (a NeurIPS 3.2 and 3.8 both → 5).

---

## Option B: Linear Stretch (Simplest)

Map NeurIPS 1–6 linearly onto 1–10:

```
score_normalized = 1 + (score - 1) * (9 / 5)
```

| NeurIPS | Normalized |
|---|---|
| 1 | 1.0 |
| 2 | 2.8 |
| 3 | 4.6 |
| 4 | 6.4 |
| 5 | 8.2 |
| 6 | 10.0 |

AI Judge already on 1–10, no change needed.

**Pros:** Trivial. Preserves continuous signal.
**Cons:** Ignores that the accept/reject boundary is at 3.5→4 (NeurIPS) and 5→6 (ICLR). A NeurIPS 4 maps to 6.4 but ICLR "borderline accept" is exactly 6 — close but semantically off.

---

## Option C: Z-score (Most Statistically Rigorous)

Normalize each score by the empirical mean and std of its venue's ground truth distribution in our dataset:

```
score_z = (score - venue_mean) / venue_std
```

Then compare AI judge z-score (using same venue pool) to ground truth z-score.

**Pros:** Accounts for actual reviewer behavior (NeurIPS reviewers cluster around 3–4, ICLR around 5–6). Fair comparison across venues.
**Cons:** z-scores are not interpretable as "quality." Requires computing pool stats from our dataset. AI judge z-score would need the venue's pool stats at inference time.

---

## Option D: Percentile Rank (Most Robust to Distribution Shape)

Map each score to its percentile within the venue's empirical distribution:

```
percentile = rank(score) / N
```

**Pros:** Completely venue-agnostic. A paper at the 80th percentile means the same regardless of venue.
**Cons:** Requires the full venue score pool at normalization time. Loses absolute magnitude.

---

## Recommendation

**For evaluating AI judge continuous score accuracy:** Use **Option A (semantic mapping)** as primary, **Option B (linear)** as secondary sanity check. Report both MAE and Spearman correlation.

**For cross-venue comparison plots:** Use **Option C (z-score)** so the AI judge's systematic biases (over-scoring, under-scoring) are visible relative to venue norms.

**Do not use a fixed accept threshold** to compute accuracy — use the ground truth `predicted_accept` boolean directly, since that captures AC judgment, rebuttals, and quota effects that a score threshold cannot.

---

## What to compute

For each paper in baseline_results.jsonl:

1. `ai_judge_score` — raw `overall_score` from XML (1–10)
2. `gt_score_raw` — raw `weighted_mean_overall` from ground truth
3. `gt_score_normalized` — Option A mapping of gt_score to {1,3,5,6,8,10}
4. `ai_judge_score_normalized` — Option A binning of ai_judge_score to {1,3,5,6,8,10}
5. `score_mae` — |ai_judge_score_normalized - gt_score_normalized|
6. `score_spearman` — Spearman correlation across all papers, per venue

Report separately for NeurIPS and ICLR since the raw-to-normalized mapping differs.
