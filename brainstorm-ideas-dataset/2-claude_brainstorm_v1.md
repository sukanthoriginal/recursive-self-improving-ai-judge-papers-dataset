# AI Judge — Prediction Target Specification

## Task

An AI judge reads a blind paper (no authors, no affiliations) and predicts what human reviewers would conclude.

**Venue:** NeurIPS 2025. Overall scale: **1–6**. Confidence scale: **1–5**. (Note: NeurIPS 2024 used 1–10; do not reuse this spec across years without re-pinning the scale.)

## Final Schema

```json
{
  "paper_id": "LPUr2CexmX",
  "predicted_weighted_mean_overall": 3.7,
  "predicted_accept": true
}
```

## Adopted Metrics

**`predicted_weighted_mean_overall`** — Range: 1.0–6.0, continuous. Confidence-weighted mean of reviewer overall scores. Ground truth: `Σ(overall_i × confidence_i) / Σ(confidence_i)`.

Rationale: approximates the Bayes-optimal aggregator under the assumption that self-reported confidence tracks reliability (CWMV; Hutchinson et al. 2020, arXiv:2005.00039). Peer-review aggregation pipelines (e.g. arXiv:2410.04202) use the same target. *Caveat:* NeurIPS AC guidelines warn confidence ≈ personality, and low-confidence reviewers show author-fame bias (arXiv:2211.15849), so weighting amplifies any bias in confident reviewers. During pilot, compute unweighted mean alongside as a sanity check.

**`predicted_accept`** — Boolean. Final accept/reject decision. Imperfectly correlated (~70–80%) with weighted mean; the gap encodes rebuttal outcomes, AC judgment, and quota effects. *Known ceiling:* rebuttals are decisive (arXiv:2511.15462) and structurally invisible to a paper-only judge — this caps achievable accuracy.

Tension between the two fields implicitly encodes contestedness: agreement = clean call; disagreement = borderline.

## Dropped Metrics

**`predicted_std_overall`** — Decision-relevant variance signal is already captured by tension between weighted_mean and accept. Empirical analysis of ICLR 2017–2025 (arXiv:2509.25701) shows variance's effect on acceptance is non-monotonic and largely mediated by review sentiment, which a paper-only judge cannot observe. Retained on the ground-truth side as an evaluation stratifier (report accuracy separately for low-std vs high-std papers).

**`predicted_significance`** — Folded into overall score. Independent effect surfaces as weighted_mean vs accept divergence.

**`quality`, `clarity`, `originality`** — Sub-scores on 1–4. High correlation (~0.7–0.85) with overall makes them redundant as prediction targets.

**`confidence`** — Property of the reviewer, not the paper. The AI reads the full paper every time; the domain-mismatch problem confidence was designed to signal does not apply. Still used in ground-truth weighting.

**`ethics_flagged`** — Too rare to serve as a general prediction signal.

**`rebuttal_delta`** — Not visible from paper alone.

**`ac_meta_review`** — Downstream of reviews, not the paper.

**`min/max/range/borderline_count/num_reviewers`** — Derivable from mean and std.
