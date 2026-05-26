"""
Ground-truth distribution plot RESTRICTED to the 48 tested papers (same set
as the AI judge baseline figure).

Mirrors plot_ai_distribution.py one-for-one so the two figures can be compared
side-by-side. Loads from the filtered ground_truth_*.json files written by
normalize_scores.py (tested subset only).
"""

from __future__ import annotations
import json, pathlib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = pathlib.Path(__file__).resolve().parent.parent
GT_NEURIPS_FILE = REPO / "normalization-dataset" / "ground_truth_neurips.json"
GT_ICLR_FILE    = REPO / "normalization-dataset" / "ground_truth_iclr.json"
OUT = pathlib.Path(__file__).resolve().parent

COL_ACCEPT = "#27ae60"
COL_REJECT = "#e74c3c"


def load_gt(file: pathlib.Path) -> list[dict]:
    d = json.loads(file.read_text())
    out = []
    for p in d["papers"]:
        out.append({
            "paper_id": p["paper_id"],
            "title":    p.get("title", ""),
            "venue":    p["venue"],
            "score_10": p["score_normalized"],
            "accept":   p["accept"],
        })
    return out


neurips = load_gt(GT_NEURIPS_FILE)
iclr    = load_gt(GT_ICLR_FILE)
papers  = neurips + iclr

print(f"Loaded {len(papers)} papers total ({len(neurips)} NeurIPS, {len(iclr)} ICLR)")


# ══════════════════════════════════════════════════════════════════════════════
# Figure: 2x2 layout (same as AI plot)
# ══════════════════════════════════════════════════════════════════════════════
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        f"NeurIPS 2025 — GT score distribution (n={len(neurips)})",
        f"ICLR 2025 — GT score distribution (n={len(iclr)})",
        "GT accept / reject counts per venue",
        "GT accept RATE by score bin (across both venues)",
    ],
    vertical_spacing=0.16,
)

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
        x=labels, y=rej_counts, name="REJECT (GT)",
        marker_color=COL_REJECT, opacity=0.85,
        text=[c if c else "" for c in rej_counts], textposition="inside",
        showlegend=(col == 1),
        legendgroup="reject",
    ), row=1, col=col)
    fig.add_trace(go.Bar(
        x=labels, y=acc_counts, name="ACCEPT (GT)",
        marker_color=COL_ACCEPT, opacity=0.85,
        text=[c if c else "" for c in acc_counts], textposition="inside",
        showlegend=(col == 1),
        legendgroup="accept",
    ), row=1, col=col)

    fig.update_xaxes(title_text="GT score (1-10)", row=1, col=col)
    fig.update_yaxes(title_text="Paper count", row=1, col=col)


# ─── Bottom-left: per-venue GT accept/reject counts ─────────────────────────
venues   = ["NeurIPS", "ICLR"]
accepts  = [sum(1 for p in neurips if p["accept"]),
            sum(1 for p in iclr    if p["accept"])]
rejects  = [sum(1 for p in neurips if not p["accept"]),
            sum(1 for p in iclr    if not p["accept"])]

fig.add_trace(go.Bar(
    x=venues, y=rejects, name="REJECT (GT)",
    marker_color=COL_REJECT, opacity=0.85,
    text=rejects, textposition="outside",
    showlegend=False, legendgroup="reject",
), row=2, col=1)
fig.add_trace(go.Bar(
    x=venues, y=accepts, name="ACCEPT (GT)",
    marker_color=COL_ACCEPT, opacity=0.85,
    text=accepts, textposition="outside",
    showlegend=False, legendgroup="accept",
), row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)


# ─── Bottom-right: GT accept rate by score bin (overall) ────────────────────
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
fig.update_yaxes(title_text="GT accept rate (%)", range=[0, 115], row=2, col=2)
fig.update_xaxes(title_text="GT score bin (1-10)", row=2, col=2)


fig.add_vline(x=4.5, line_width=2, line_dash="dot", line_color="grey", row=2, col=2)


fig.update_layout(
    title=(f"<b>Ground Truth distribution — tested papers only (n={len(papers)})</b><br>"
           "<sup>Same 48 papers as the AI judge figure. "
           "Score normalized to 1-10 (NeurIPS linearly stretched from 1-6). "
           "Accept/reject is the actual venue decision.</sup>"),
    barmode="stack",
    height=820, width=1200,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=12),
    legend=dict(orientation="h", y=1.07, x=0.35),
)

out_path = OUT / "0_gt_tested_distribution.png"
fig.write_image(str(out_path), scale=2)
print(f"  saved: {out_path.name}")


# ─── Summary stats ───────────────────────────────────────────────────────────
print("\n=== GT distribution summary (tested subset) ===")
for vname, recs in [("NeurIPS 2025", neurips), ("ICLR 2025", iclr)]:
    n = len(recs)
    a = sum(1 for r in recs if r["accept"])
    scores = np.array([r["score_10"] for r in recs])
    print(f"\n  {vname} (n={n})")
    print(f"    Accept: {a} ({a/n:.1%})  |  Reject: {n-a} ({(n-a)/n:.1%})")
    print(f"    Score 1-10: mean={scores.mean():.2f}  median={np.median(scores):.2f}  "
          f"min={scores.min():.2f}  max={scores.max():.2f}")
