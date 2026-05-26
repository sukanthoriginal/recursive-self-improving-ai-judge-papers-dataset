"""
RMSE-based comparison of AI judge scores vs ground truth scores.

Same visual layout as the GT and AI distribution figures (2x2 panels), but
the y-axis quantity is now the score ERROR (AI - GT) and its summary RMSE.

  RMSE = sqrt(mean((ai_score - gt_score)^2))  on the continuous 1-10 scale

Top row    — per-venue histogram of signed errors (one bar per integer error)
Bottom-left — RMSE per venue (and overall)
Bottom-right — RMSE by GT score bin (where does AI err most?)
"""

from __future__ import annotations
import json, pathlib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "normalization-dataset" / "results_normalized.json"
OUT = pathlib.Path(__file__).resolve().parent

COL_UNDER = "#4C72B0"   # AI scored LOWER than GT (negative error)
COL_OVER  = "#DD8452"   # AI scored HIGHER than GT (positive error)
COL_RMSE  = "#7F3F98"   # purple for RMSE summaries

data = json.loads(RESULTS.read_text())
papers = [p for p in data["papers"]
          if p.get("ai_score_norm") is not None and p.get("gt_score_norm") is not None]

for p in papers:
    p["err"] = p["ai_score_norm"] - p["gt_score_norm"]   # AI - GT

neurips = [p for p in papers if p["venue"] == "neurips2025"]
iclr    = [p for p in papers if p["venue"] == "iclr2025"]


def rmse(records) -> float:
    e = np.array([r["err"] for r in records])
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")


def mean_err(records) -> float:
    e = np.array([r["err"] for r in records])
    return float(np.mean(e)) if len(e) else float("nan")


rmse_n, rmse_i, rmse_all = rmse(neurips), rmse(iclr), rmse(papers)
bias_n, bias_i, bias_all = mean_err(neurips), mean_err(iclr), mean_err(papers)

print(f"n={len(papers)}  RMSE overall={rmse_all:.3f}  "
      f"NeurIPS={rmse_n:.3f}  ICLR={rmse_i:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        f"NeurIPS 2025 — error histogram (RMSE={rmse_n:.2f})",
        f"ICLR 2025 — error histogram (RMSE={rmse_i:.2f})",
        "RMSE per venue + overall",
        "RMSE by GT score bin (where AI errs most)",
    ],
    vertical_spacing=0.16,
)

# Bins for signed error: integer bins from -8 to +8
err_bins = list(range(-8, 9))
err_labels = [str(b) for b in err_bins]


def err_bin_counts(recs) -> tuple[list[int], list[int]]:
    """Return (under_counts, over_counts) — under has positive height when err<0."""
    under = [0] * len(err_bins)   # err <= 0
    over  = [0] * len(err_bins)   # err >  0
    for r in recs:
        idx = int(np.clip(round(r["err"]), err_bins[0], err_bins[-1])) - err_bins[0]
        if r["err"] <= 0:
            under[idx] += 1
        else:
            over[idx] += 1
    return under, over


# ─── Top row: per-venue signed-error histograms ─────────────────────────────
for col, recs, vname, rval, bval in [
    (1, neurips, "NeurIPS", rmse_n, bias_n),
    (2, iclr,    "ICLR",    rmse_i, bias_i),
]:
    under, over = err_bin_counts(recs)
    fig.add_trace(go.Bar(
        x=err_labels, y=under,
        name="AI < GT (under-scored)", marker_color=COL_UNDER, opacity=0.85,
        text=[c if c else "" for c in under], textposition="outside",
        showlegend=(col == 1), legendgroup="under",
    ), row=1, col=col)
    fig.add_trace(go.Bar(
        x=err_labels, y=over,
        name="AI > GT (over-scored)",  marker_color=COL_OVER, opacity=0.85,
        text=[c if c else "" for c in over], textposition="outside",
        showlegend=(col == 1), legendgroup="over",
    ), row=1, col=col)

    fig.update_xaxes(title_text="Score error (AI − GT, rounded)", row=1, col=col)
    fig.update_yaxes(title_text="Paper count", row=1, col=col)
    # vertical line at zero error
    fig.add_vline(x=0, line_width=2, line_dash="dot", line_color="grey",
                  row=1, col=col)
    # annotate mean bias
    xref = "x domain" if col == 1 else f"x{col} domain"
    yref = "y domain" if col == 1 else f"y{col} domain"
    fig.add_annotation(
        text=f"mean err = {bval:+.2f}",
        xref=xref, yref=yref,
        x=0.02, y=0.95, showarrow=False, font=dict(size=11, color="grey"),
    )


# ─── Bottom-left: RMSE per venue + overall ──────────────────────────────────
labels = ["NeurIPS", "ICLR", "Overall"]
rmse_vals = [rmse_n, rmse_i, rmse_all]
fig.add_trace(go.Bar(
    x=labels, y=rmse_vals,
    marker_color=[COL_UNDER, COL_OVER, COL_RMSE], opacity=0.85,
    text=[f"{v:.2f}" for v in rmse_vals], textposition="outside",
    showlegend=False,
), row=2, col=1)
fig.update_yaxes(title_text="RMSE (1-10 points)", range=[0, max(rmse_vals) * 1.25],
                 row=2, col=1)


# ─── Bottom-right: RMSE by GT score bin ─────────────────────────────────────
bin_labels = [str(i) for i in range(1, 11)]
per_bin_rmse = [None] * 10
per_bin_n    = [0] * 10
for p in papers:
    idx = int(np.clip(round(p["gt_score_norm"]), 1, 10)) - 1
    per_bin_n[idx] += 1

for idx in range(10):
    bin_papers = [p for p in papers
                  if int(np.clip(round(p["gt_score_norm"]), 1, 10)) - 1 == idx]
    if bin_papers:
        per_bin_rmse[idx] = rmse(bin_papers)

ymax = max([v for v in per_bin_rmse if v is not None] + [rmse_all]) * 1.3
fig.add_trace(go.Bar(
    x=bin_labels,
    y=[v if v is not None else 0 for v in per_bin_rmse],
    marker_color=COL_RMSE, opacity=0.85,
    text=[f"{v:.2f}<br>n={n}" if v is not None else f"n={n}"
          for v, n in zip(per_bin_rmse, per_bin_n)],
    textposition="outside",
    showlegend=False,
), row=2, col=2)
fig.update_yaxes(title_text="RMSE in this bin (1-10 points)",
                 range=[0, ymax], row=2, col=2)
fig.update_xaxes(title_text="GT score bin (1-10)", row=2, col=2)

# horizontal line at overall RMSE for reference
fig.add_hline(y=rmse_all, line_width=2, line_dash="dot", line_color="grey",
              annotation_text=f"overall RMSE = {rmse_all:.2f}",
              annotation_position="top right",
              row=2, col=2)


fig.update_layout(
    title=(f"<b>RMSE comparison — AI judge vs Ground Truth (n={len(papers)})</b><br>"
           f"<sup>RMSE = average score error in 1-10 points.   "
           f"Overall RMSE = {rmse_all:.2f}.   "
           f"Blue = AI under-scored, Orange = AI over-scored.</sup>"),
    barmode="stack",
    height=820, width=1200,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=12),
    legend=dict(orientation="h", y=1.07, x=0.30),
)

out_path = OUT / "0_rmse_comparison.png"
fig.write_image(str(out_path), scale=2)
print(f"  saved: {out_path.name}")


# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n=== RMSE comparison summary ===")
print(f"  Overall: RMSE={rmse_all:.3f}  mean_err={bias_all:+.3f} (n={len(papers)})")
print(f"  NeurIPS: RMSE={rmse_n:.3f}  mean_err={bias_n:+.3f} (n={len(neurips)})")
print(f"  ICLR:    RMSE={rmse_i:.3f}  mean_err={bias_i:+.3f} (n={len(iclr)})")
print("\n  RMSE by GT bin:")
for i, (v, n) in enumerate(zip(per_bin_rmse, per_bin_n), start=1):
    if v is not None:
        print(f"    bin {i}: RMSE={v:.2f}  n={n}")
