# Dataset Curation Plan — NeurIPS 2025 Pilot

## Goal

Build a 5-paper pilot dataset for the AI judge task (see `2-claude_brainstorm_v1.md`). Validate the pipeline end-to-end before scaling.

## Scope

- **Venue:** NeurIPS 2025 main track only (no Datasets & Benchmarks track in pilot — different review rubric).
- **Inclusion:** Papers with ≥3 completed reviews and a final accept/reject decision.
- **Exclusion:** Withdrawn, desk-rejected, ethics-flagged, and papers where any reviewer has `confidence` missing.

## Directory Layout

```
NeurIPS/2025/
├── papers/                              # raw PDF + OpenReview metadata JSON
│   └── {paper_id}.pdf
│   └── {paper_id}.meta.json
├── sanitized-papers/                    # blinded PDF (authors/acks/IDs redacted)
│   └── {paper_id}.pdf
├── open-reviews-raw/                    # raw OpenReview API dump per paper
│   └── {paper_id}.json
└── open-reviews-quantized/              # normalized review JSON (schema below)
    └── {paper_id}.json
```

Rename `open-reviews-review-groundtruths` → `open-reviews-raw` and `open-reviews-review-quantized-jsons` → `open-reviews-quantized` for clarity.

## Pilot Paper Selection (n=5)

Random sampling will be ~85% rejects given NeurIPS base rates. Instead, **stratify**:

- 2 clean accepts (weighted_mean ≥ 5.0)
- 1 borderline accept (weighted_mean 4.0–4.5)
- 1 borderline reject (weighted_mean 3.0–3.5)
- 1 clean reject (weighted_mean ≤ 2.5)

Pick one paper per stratum at random from the OpenReview pool. This exercises the full prediction range on day one.

## Source Field Mapping (OpenReview → our schema)

NeurIPS 2025 OpenReview review form fields:

| OpenReview field | Our usage |
|---|---|
| `rating` (1–6) | `overall` — the score we predict |
| `confidence` (1–5) | weighting for ground-truth mean |
| `soundness`, `presentation`, `contribution` | stored, not used as targets |
| `summary`, `strengths`, `weaknesses`, `questions` | stored as text |
| Decision note (`Accept` / `Reject` / `Spotlight` / `Oral`) | `accept` = any acceptance tier |

**Verify on paper 1 before processing the rest.** Field names have changed across years; confirm against a live OpenReview record.

## Quantized Review Schema

`open-reviews-quantized/{paper_id}.json`:

```json
{
  "paper_id": "LPUr2CexmX",
  "decision": "Accept",
  "accept": true,
  "reviews": [
    {"reviewer_id": "r1", "overall": 5, "confidence": 4,
     "soundness": 3, "presentation": 3, "contribution": 3}
  ],
  "ground_truth": {
    "weighted_mean_overall": 4.43,
    "unweighted_mean_overall": 4.33,
    "std_overall": 0.94,
    "n_reviewers": 3
  }
}
```

Weighted mean formula: `Σ(overall_i × confidence_i) / Σ(confidence_i)`. If any reviewer has missing confidence → drop the paper (per exclusion rule).

## Sanitization Protocol

Goal: a reviewer reading the sanitized PDF cannot identify authors. Preserve figures, tables, equations, and layout — multimodal LLMs ingest PDF natively.

**Strip:**
1. Title page authors, affiliations, emails, ORCIDs.
2. Acknowledgments section (entire).
3. Author contributions section (entire).
4. arXiv IDs, GitHub URLs, project websites, dataset/codebase names that uniquely identify a lab (e.g. "DeepMind's Gemini").
5. First-person self-citations: rewrite "as we showed in [Smith 2023]" → "as shown in prior work [REF]" (keep the citation in bibliography; remove the "we" framing in prose).
6. Funding statements.

**Keep:**
- All technical content, figures, tables, equations.
- Citations and bibliography (peer reviewers see these too).

**How:** PyMuPDF (`fitz`) for region-level redaction that actually removes the underlying text (not just visual cover — covered-but-extractable text leaks under copy-paste or model OCR). Title-page authors and acknowledgment pages get redacted/dropped; in-body self-citations and lab-identifying URLs get redacted in place. **Manually spot-check all 5 pilot papers** by opening the sanitized PDF and trying to identify the authors — sanitization quality is the single biggest risk to the project.

## Pipeline Steps

1. **Select** 5 papers per stratification rule above.
2. **Download** PDF + OpenReview JSON via OpenReview API v2.
3. **Quantize** reviews → `open-reviews-quantized/{paper_id}.json`. Compute both weighted and unweighted means.
4. **Sanitize** PDF → redacted PDF → `sanitized-papers/{paper_id}.pdf`. Spot-check.
5. **Sanity check:** weighted vs unweighted mean agree on accept/reject for all 5? If not, log which papers diverge — these are the cases where the weighting choice matters.

## Risks

- **OpenReview API rate limits / auth** — may need a registered account for some 2025 records.
- **Sanitization leakage** — author names embedded in figure captions, dataset names, or writing style. Pilot is small enough to catch by eye; need automated detection before scaling.
- **Decision granularity** — NeurIPS 2025 has Oral / Spotlight / Poster / Reject. Binary `accept` collapses the top three; revisit if downstream task needs tier prediction.

## Done When

- All 5 papers have `papers/`, `sanitized-papers/`, and `open-reviews-quantized/` entries.
- Manual review confirms no author-identifying leakage in any sanitized paper.
- Weighted vs unweighted divergence is logged.
- A one-page pilot retro identifies which steps to automate vs harden before scaling to n=50.
