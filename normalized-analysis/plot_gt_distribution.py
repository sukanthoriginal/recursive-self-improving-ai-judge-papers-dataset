"""
Ground-truth-only distribution plot: shows score distribution (1-10) and
accept/reject breakdown for ALL papers in both venues' ground truth.

Loads directly from the venue dataset dirs (not the filtered tested subset).
"""

from __future__ import annotations
import json, pathlib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = pathlib.Path(__file__).resolve().parent.parent
NEURIPS_GT_DIR = REPO / "neurips-2025-dataset-v2" / "open-reviews-ground_truth"
ICLR_GT_DIR    = REPO / "iclr-2025-dataset" / "open-reviews-ground_truth"
OUT = pathlib.Path(__file__).resolve().parent

COL_ACCEPT = "#27ae60"
COL_REJECT = "#e74c3c"
COL_NEURIPS = "#4C72B0"
COL_ICLR    = "#DD8452"


def neurips_to_10(x: float) -> float:
    return 1 + (x - 1) * 9 / 5


def load_all_gt() -> list[dict]:
    papers = []
    for f in NEURIPS_GT_DIR.glob("*.json"):
        d = json.loads(f.read_text())
        raw = d["ground_truth"]["weighted_mean_overall"]
        papers.append({
            "paper_id":  d["paper_id"],
            "title":     d.get("title", ""),
            "venue":     "neurips",
            "score_raw": raw,
            "score_10":  neurips_to_10(raw),
            "accept":    d["ground_truth"]["accept"],
        })
    for f in ICLR_GT_DIR.glob("*.json"):
        d = json.loads(f.read_text())
        raw = d["ground_truth"]["weighted_mean_overall"]
        papers.append({
            "paper_id":  d["paper_id"],
            "title":     d.get("title", ""),
            "venue":     "iclr",
            "score_raw": raw,
            "score_10":  raw,
            "accept":    d["ground_truth"]["accept"],
        })
    return papers


papers = load_all_gt()
neurips = [p for p in papers if p["venue"] == "neurips"]
iclr    = [p for p in papers if p["venue"] == "iclr"]

print(f"Loaded {len(papers)} papers total ({len(neurips)} NeurIPS, {len(iclr)} ICLR)")


# ══════════════════════════════════════════════════════════════════════════════
# Figure: 2x2 layout
#   Top row    — per-venue stacked histograms of normalized score (1-10)
#   Bottom row — overall accept/reject counts and accept rate by score bin
# ══════════════════════════════════════════════════════════════════════════════
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        f"NeurIPS 2025 — score distribution (n={len(neurips)})",
        f"ICLR 2025 — score distribution (n={len(iclr)})",
        "Accept / Reject counts per venue",
        "GT accept RATE by score bin (across both venues)",
    ],
    vertical_spacing=0.16,
)

# Bin edges (10 integer bins, width = 1)
edges = np.arange(1, 11)  # bin starts at 1, 2, ..., 9; last bin includes 10
labels = [str(i) for i in range(1, 11)]

def bin_counts(recs, accept: bool) -> list[int]:
    scores = [r["score_10"] for r in recs if r["accept"] == accept]
    counts = [0] * 10
    for s in scores:
        idx = int(np.clip(np.floor(s), 1, 10)) - 1
        counts[idx] += 1
    return counts


# ─── Top row: per-venue stacked histograms ──────────────────────────────────
for col, recs, vname in [(1, neurips, "NeurIPS"), (2, iclr, "ICLR")]:
    acc_counts = bin_counts(recs, accept=True)
    rej_counts = bin_counts(recs, accept=False)

    fig.add_trace(go.Bar(
        x=labels, y=rej_counts, name="REJECT",
        marker_color=COL_REJECT, opacity=0.85,
        text=[c if c else "" for c in rej_counts], textposition="inside",
        showlegend=(col == 1),
        legendgroup="reject",
    ), row=1, col=col)
    fig.add_trace(go.Bar(
        x=labels, y=acc_counts, name="ACCEPT",
        marker_color=COL_ACCEPT, opacity=0.85,
        text=[c if c else "" for c in acc_counts], textposition="inside",
        showlegend=(col == 1),
        legendgroup="accept",
    ), row=1, col=col)

    fig.update_xaxes(title_text="Score (1-10)", row=1, col=col)
    fig.update_yaxes(title_text="Paper count", row=1, col=col)


# ─── Bottom-left: per-venue accept/reject counts ────────────────────────────
venues   = ["NeurIPS", "ICLR"]
accepts  = [sum(1 for p in neurips if p["accept"]),
            sum(1 for p in iclr    if p["accept"])]
rejects  = [sum(1 for p in neurips if not p["accept"]),
            sum(1 for p in iclr    if not p["accept"])]

fig.add_trace(go.Bar(
    x=venues, y=rejects, name="REJECT",
    marker_color=COL_REJECT, opacity=0.85,
    text=rejects, textposition="outside",
    showlegend=False, legendgroup="reject",
), row=2, col=1)
fig.add_trace(go.Bar(
    x=venues, y=accepts, name="ACCEPT",
    marker_color=COL_ACCEPT, opacity=0.85,
    text=accepts, textposition="outside",
    showlegend=False, legendgroup="accept",
), row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)


# ─── Bottom-right: accept rate by score bin (overall) ───────────────────────
all_acc = bin_counts(papers, accept=True)
all_rej = bin_counts(papers, accept=False)
totals  = [a + r for a, r in zip(all_acc, all_rej)]
rates   = [round(a / t * 100, 1) if t else None for a, t in zip(all_acc, totals)]

fig.add_trace(go.Bar(
    x=labels, y=rates,
    marker_color=COL_ACCEPT, opacity=0.85,
    text=[f"{r}%<br>n={t}" if r is not None else "" for r, t in zip(rates, totals)],
    textposition="outside",
    showlegend=False,
), row=2, col=2)
fig.update_yaxes(title_text="Accept rate (%)", range=[0, 115], row=2, col=2)
fig.update_xaxes(title_text="Score bin (1-10)", row=2, col=2)


# Threshold line on bottom-right at 5.5 (rough accept boundary)
fig.add_vline(x=4.5, line_width=2, line_dash="dot", line_color="grey", row=2, col=2)


fig.update_layout(
    title=(f"<b>Ground Truth distribution — all papers (n={len(papers)})</b><br>"
           "<sup>Score normalized to 1-10 (NeurIPS linearly stretched from 1-6). "
           "Accept/reject is the actual venue decision.</sup>"),
    barmode="stack",
    height=820, width=1200,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=12),
    legend=dict(orientation="h", y=1.07, x=0.35),
)

# bottom-left should not be stacked since x is venue not score
# We'll set it to stack visually too since accept+reject = total per venue (acceptable)

out_path = OUT / "0_gt_distribution.png"
fig.write_image(str(out_path), scale=2)
print(f"  saved: {out_path.name}")


# ─── Summary stats ───────────────────────────────────────────────────────────
print("\n=== GT distribution summary ===")
for vname, recs in [("NeurIPS 2025", neurips), ("ICLR 2025", iclr)]:
    n = len(recs)
    a = sum(1 for r in recs if r["accept"])
    scores = np.array([r["score_10"] for r in recs])
    print(f"\n  {vname} (n={n})")
    print(f"    Accept: {a} ({a/n:.1%})  |  Reject: {n-a} ({(n-a)/n:.1%})")
    print(f"    Score 1-10: mean={scores.mean():.2f}  median={np.median(scores):.2f}  "
          f"min={scores.min():.2f}  max={scores.max():.2f}")
