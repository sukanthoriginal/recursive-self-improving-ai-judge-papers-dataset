# NeurIPS 2025 Paper Dataset for AI Judge Evaluation

A curated, blinded dataset of NeurIPS 2025 papers for evaluating AI scientific reviewers.
Each paper ships with the original PDF, a sanitized (author-blind) PDF, full review records,
and ground-truth labels for the prediction task.

## Prediction task

Given a blinded paper, predict what human reviewers concluded:

```json
{
  "paper_id": "...",
  "predicted_weighted_mean_overall": 4.13,
  "predicted_accept": true
}
```

- `weighted_mean_overall` — continuous, 1.0–6.0 (NeurIPS 2025 scale), confidence-weighted across reviewers.
- `accept` — binary, from the final decision note.

See `brainstorm-ideas-dataset/2-claude_brainstorm_v1.md` for the full target spec.

## Layout

```
neurips-2025-dataset-v2/
├── papers/                       # original PDFs + per-paper metadata
├── sanitized-papers/             # blinded PDFs (author identity removed)
├── open-reviews-raw/             # full review records from OpenReview
├── open-reviews-quantized/       # normalized review JSONs (scores only)
├── open-reviews-ground_truth/    # minimal ground-truth blob per paper
└── _cache/                       # selection cache + selection.json
```

## Dataset composition (v2)

44 papers, stratified across the score range and explicitly oversampling
score↔decision mismatches:

- 40 papers: 4 per 0.5-wide bin × 10 bins across the 1–6 scale
- 4 mismatch papers: 1 low-score accept (wmean<3.0) + 1 high-score reject per bin in [4.0–5.5)

![v2 ground-truth distribution](neurips-2025-dataset-v2/v2_ground_truth.png)

Mismatch papers are flagged with `mismatch: true` in their meta.json.

## ICLR 2025 dataset

49 papers from ICLR 2025 (`iclr-2025-dataset/`) — different review scale (1–10) but same prediction-target shape:

- 45 papers: 5 per bin × 9 bins (`[1,2)` through `[9,10)`)
- 1 paper from `[10,10]` (the only one)
- 3 mismatch papers: 1 low-score accept + 1 high-score reject per high bin

Confidence-weighting still applies (ICLR confidence field exists, 1–5 scale).

![ICLR 2025 ground-truth distribution](iclr-2025-dataset/iclr_ground_truth.png)

## Sanitization

The blinded PDFs (`sanitized-papers/`) have:
- Title-page author block redacted
- All author names (and variants ≥4 chars) redacted across body
- Emails, ORCIDs, GitHub/lab URLs, arXiv-ID footnotes redacted
- Acknowledgments / Author Contributions / Funding sections redacted
- All link annotations stripped (clickable URI metadata removed)
- PDF metadata blanked

An aggressive audit pass (`scripts/sanitization/sanity_check.py`) runs against
each PDF and flags any residual leakage. Current status: 0 FAILs across 44 papers.

## Scripts

- `scripts/build.py` — end-to-end builder
- `scripts/plot.py` — ground-truth distribution histogram
- `scripts/add_mismatches.py` — incremental mismatch coverage
- `scripts/trim_mismatches.py` — reduce mismatch coverage
- `scripts/verify_dataset.py` — integrity check across all artifact dirs
- `scripts/sanitization/sanity_check.py` — blind-review audit
- `scripts/sanitization/fix_sanitization.py` — idempotent post-process

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install openreview-py pymupdf requests matplotlib
```

## Source

All papers and reviews from [OpenReview](https://openreview.net/group?id=NeurIPS.cc/2025/Conference).
