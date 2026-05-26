"""
Plots: RSI-AI-Judge vs Ground Truth on a CONTINUOUS 1-10 scale.

Two independent tasks:
  A) Score regression (continuous 1-10)
  B) Binary accept/reject classification

Stratification by rounded GT score (integer 1-10), only non-empty bins shown.
"""

from __future__ import annotations
import json, pathlib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT = pathlib.Path(__file__).resolve().parent
summary = json.loads((OUT / "summary.json").read_text())

COL_GT       = "#27ae60"
COL_AI       = "#e74c3c"
COL_NEURIPS  = "#4C72B0"
COL_ICLR     = "#DD8452"
COL_CORRECT  = "#27ae60"
COL_WRONG    = "#e74c3c"

papers  = summary["papers"]
neurips = [p for p in papers if p["venue"] == "neurips"]
iclr    = [p for p in papers if p["venue"] == "iclr"]


def save(fig, name):
    fig.write_image(str(OUT / name), scale=2)
    print(f"  saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — TASK A: Continuous scatter, GT vs AI
# ══════════════════════════════════════════════════════════════════════════════
fig = go.Figure()

# Perfect prediction line y=x
fig.add_trace(go.Scatter(
    x=[1, 10], y=[1, 10],
    mode="lines", name="Perfect prediction (AI = GT)",
    line=dict(color="black", dash="dash", width=2),
))

# Per-bin AI mean line
per_bin = summary["task_A_score_regression"]["overall"]["per_gt_bin"]
mean_x, mean_y = [], []
for b in sorted(per_bin.keys(), key=int):
    pb = per_bin[b]
    mean_x.append(pb["gt_score_mean"])
    mean_y.append(pb["ai_score_mean"])

fig.add_trace(go.Scatter(
    x=mean_x, y=mean_y, mode="lines+markers",
    name="AI mean prediction (per GT bin)",
    line=dict(color=COL_AI, width=3),
    marker=dict(size=12, color=COL_AI),
))

# Individual papers (continuous, slight jitter for overlap)
rng = np.random.default_rng(42)
for venue_recs, venue_name, color in [(neurips, "NeurIPS 2025", COL_NEURIPS),
                                       (iclr,    "ICLR 2025",    COL_ICLR)]:
    xs, ys, hovers = [], [], []
    for r in venue_recs:
        jx = rng.uniform(-0.10, 0.10)
        jy = rng.uniform(-0.10, 0.10)
        xs.append(r["gt_score_norm"] + jx)
        ys.append(r["ai_score_norm"] + jy)
        hovers.append(
            f"<b>{r['title'][:60]}</b><br>"
            f"GT score: {r['gt_score_norm']:.2f}<br>"
            f"AI score: {r['ai_score_norm']:.2f}<br>"
            f"Error: {r['ai_score_norm'] - r['gt_score_norm']:+.2f}"
        )
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", name=venue_name,
        marker=dict(color=color, size=10, opacity=0.55, line=dict(width=1, color="white")),
        hovertext=hovers, hoverinfo="text",
    ))

o = summary["task_A_score_regression"]["overall"]["overall"]
fig.update_layout(
    title=(f"<b>Task A — Score Prediction (continuous 1-10):</b> "
           f"MAE={o['mae']}, Spearman={o['spearman_r']}, Pearson={o['pearson_r']}<br>"
           "<sup>AI compresses scores toward the middle: over-scores rejects, under-scores accepts.</sup>"),
    xaxis=dict(title="Ground Truth score (1-10)", range=[0.5, 10.5],
               dtick=1, gridcolor="#eee"),
    yaxis=dict(title="RSI-AI-Judge score (1-10)", range=[0.5, 10.5],
               dtick=1, gridcolor="#eee"),
    height=600, width=950,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=12),
    legend=dict(orientation="h", y=-0.18, x=0.0),
)
save(fig, "1_score_regression.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — TASK A: MAE & mean error per GT bin
# ══════════════════════════════════════════════════════════════════════════════
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Mean Absolute Error per GT score bin",
                    "Mean Error (AI − GT) — Bias direction"],
)

for col, key in enumerate(["mae", "mean_error"], start=1):
    for venue_key, venue_name, color in [("neurips2025", "NeurIPS 2025", COL_NEURIPS),
                                          ("iclr2025",    "ICLR 2025",    COL_ICLR)]:
        pb = summary["task_A_score_regression"][venue_key]["per_gt_bin"]
        xs, ys, ns = [], [], []
        for b in sorted(pb.keys(), key=int):
            xs.append(int(b))
            ys.append(pb[b][key])
            ns.append(pb[b]["n"])
        fig.add_trace(go.Bar(
            x=xs, y=ys, name=venue_name,
            marker_color=color, opacity=0.85,
            text=[f"n={n}" for n in ns], textposition="outside",
            showlegend=(col == 1),
        ), row=1, col=col)

fig.add_hline(y=0, line_width=1.5, line_color="black", row=1, col=2)
fig.update_yaxes(title_text="MAE (points on 1-10 scale)", row=1, col=1)
fig.update_yaxes(title_text="Mean Error (AI − GT)", row=1, col=2)
fig.update_xaxes(title_text="GT score bin (rounded)", row=1, col=1,
                 dtick=1, range=[0.5, 10.5])
fig.update_xaxes(title_text="GT score bin (rounded)", row=1, col=2,
                 dtick=1, range=[0.5, 10.5])

fig.update_layout(
    title="<b>Task A — Score Error Breakdown:</b> systematic regression-to-the-mean<br>"
          "<sup>Positive bias on low GT scores (AI over-scores), negative bias on high GT scores (AI under-scores).</sup>",
    barmode="group", height=480, width=1050,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=12),
    legend=dict(orientation="h", y=-0.22, x=0.3),
)
save(fig, "2_score_bias_by_bucket.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — TASK B: Binary accuracy stratified by GT score bin
# ══════════════════════════════════════════════════════════════════════════════
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["GT vs AI accept rate per GT bin",
                    "Binary accuracy per GT bin"],
)

pb = summary["task_B_binary_classification"]["overall"]["per_gt_bin"]
bins = sorted(pb.keys(), key=int)

gt_rates = [pb[b]["gt_accept_rate"] * 100 for b in bins]
ai_rates = [pb[b]["ai_accept_rate"] * 100 for b in bins]
ns       = [pb[b]["n"] for b in bins]
correct  = [pb[b]["binary_correct"] for b in bins]
wrong    = [pb[b]["n"] - pb[b]["binary_correct"] for b in bins]
xs       = [int(b) for b in bins]

fig.add_trace(go.Bar(
    x=xs, y=gt_rates, name="GT accept %",
    marker_color=COL_GT, opacity=0.85,
    text=[f"n={n}" for n in ns], textposition="outside",
), row=1, col=1)
fig.add_trace(go.Bar(
    x=xs, y=ai_rates, name="AI accept %",
    marker_color=COL_AI, opacity=0.85,
), row=1, col=1)

fig.add_trace(go.Bar(
    x=xs, y=correct, name="Correct",
    marker_color=COL_CORRECT, opacity=0.85,
    text=correct, textposition="inside",
), row=1, col=2)
fig.add_trace(go.Bar(
    x=xs, y=wrong, name="Wrong",
    marker_color=COL_WRONG, opacity=0.85,
    text=[w if w else "" for w in wrong], textposition="inside",
), row=1, col=2)

# Mark the accept-implied threshold (5.5) on left panel
fig.add_vline(x=5.5, line_width=2, line_dash="dot", line_color="grey", row=1, col=1)
fig.add_annotation(x=5.5, y=110, text="accept boundary (5.5)", showarrow=False,
                   font=dict(size=10, color="grey"), row=1, col=1)

fig.update_layout(
    title="<b>Task B — Binary Accept/Reject Prediction, stratified by GT score bin</b><br>"
          "<sup>AI is near-perfect at score extremes; struggles at borderline GT bins around 5.5.</sup>",
    barmode="group", height=500, width=1150,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=12),
    legend=dict(orientation="h", y=-0.22, x=0.20),
)
fig.update_yaxes(title_text="Accept rate (%)", row=1, col=1, range=[0, 120])
fig.update_yaxes(title_text="Count", row=1, col=2)
fig.update_xaxes(title_text="GT score bin (rounded)", row=1, col=1,
                 dtick=1, range=[0.5, 10.5])
fig.update_xaxes(title_text="GT score bin (rounded)", row=1, col=2,
                 dtick=1, range=[0.5, 10.5])
save(fig, "3_binary_by_bucket.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — TASK B: Confusion matrices + headline metrics
# ══════════════════════════════════════════════════════════════════════════════
fig = make_subplots(
    rows=1, cols=3,
    column_widths=[0.30, 0.30, 0.40],
    subplot_titles=["NeurIPS 2025", "ICLR 2025", "Headline metrics"],
    specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "bar"}]],
)

for col, venue_key in enumerate(["neurips2025", "iclr2025"], start=1):
    c = summary["task_B_binary_classification"][venue_key]["overall"]["confusion"]
    matrix = [[c["TP"], c["FN"]], [c["FP"], c["TN"]]]
    fig.add_trace(go.Heatmap(
        z=matrix,
        x=["GT: Accept", "GT: Reject"],
        y=["AI: Accept", "AI: Reject"],
        text=[[str(v) for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=20),
        colorscale="Greens", showscale=False,
        xgap=4, ygap=4,
    ), row=1, col=col)

metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
neurips_vals = [summary["task_B_binary_classification"]["neurips2025"]["overall"][k.lower()] for k in metrics_names]
iclr_vals    = [summary["task_B_binary_classification"]["iclr2025"]["overall"][k.lower()] for k in metrics_names]

fig.add_trace(go.Bar(
    x=metrics_names, y=neurips_vals,
    name="NeurIPS 2025", marker_color=COL_NEURIPS, opacity=0.85,
    text=[f"{v:.1%}" if isinstance(v, float) else f"{v}" for v in neurips_vals],
    textposition="outside",
), row=1, col=3)
fig.add_trace(go.Bar(
    x=metrics_names, y=iclr_vals,
    name="ICLR 2025", marker_color=COL_ICLR, opacity=0.85,
    text=[f"{v:.1%}" if isinstance(v, float) else f"{v}" for v in iclr_vals],
    textposition="outside",
), row=1, col=3)

fig.update_layout(
    title="<b>Task B — Binary Confusion Matrices & Headline Metrics</b><br>"
          "<sup>Rows = AI prediction, Cols = GT label. Right: per-venue accuracy/precision/recall/F1.</sup>",
    barmode="group", height=480, width=1200,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=12),
    legend=dict(orientation="h", y=-0.15, x=0.6),
)
fig.update_yaxes(range=[0, 1.15], row=1, col=3)
save(fig, "4_binary_confusion.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Divergence: GT mismatch papers (score >= 5.5 vs accept label)
# ══════════════════════════════════════════════════════════════════════════════
THRESHOLD = 5.5
mm_info = summary["task_divergence"]["overall"]["gt_score_vs_accept_mismatches"]
mm_count = mm_info["count"]
mm_right = mm_info["ai_got_binary_right_on_these"]

fig = go.Figure()

# Zones
fig.add_shape(type="rect", x0=THRESHOLD, x1=10.5, y0=0.5, y1=1.5,
              fillcolor="rgba(46, 204, 113, 0.10)", line_width=0, layer="below")
fig.add_shape(type="rect", x0=0.5, x1=THRESHOLD, y0=-0.5, y1=0.5,
              fillcolor="rgba(46, 204, 113, 0.10)", line_width=0, layer="below")
fig.add_shape(type="rect", x0=0.5, x1=THRESHOLD, y0=0.5, y1=1.5,
              fillcolor="rgba(231, 76, 60, 0.08)", line_width=0, layer="below")
fig.add_shape(type="rect", x0=THRESHOLD, x1=10.5, y0=-0.5, y1=0.5,
              fillcolor="rgba(231, 76, 60, 0.08)", line_width=0, layer="below")

rng = np.random.default_rng(7)
for r in papers:
    is_mismatch = (r["gt_score_norm"] >= THRESHOLD) != r["gt_accept"]
    ai_correct  = r["gt_accept"] == r["ai_predicted_accept"]
    fig.add_trace(go.Scatter(
        x=[r["gt_score_norm"] + rng.uniform(-0.15, 0.15)],
        y=[int(r["gt_accept"]) + rng.uniform(-0.06, 0.06)],
        mode="markers",
        marker=dict(
            color=COL_CORRECT if ai_correct else COL_WRONG,
            size=15 if is_mismatch else 9,
            opacity=0.85,
            symbol="star" if is_mismatch else "circle",
            line=dict(width=1.5 if is_mismatch else 0.5, color="white"),
        ),
        hovertext=(
            f"<b>{r['title'][:60]}</b><br>"
            f"Venue: {r['venue']}<br>"
            f"GT score: {r['gt_score_norm']:.2f}<br>"
            f"GT accept: {r['gt_accept']}<br>"
            f"AI predicted: {'ACCEPT' if r['ai_predicted_accept'] else 'REJECT'}<br>"
            f"AI got it: {'YES' if ai_correct else 'NO'}<br>"
            f"Mismatch paper: {'YES' if is_mismatch else 'NO'}"
        ),
        hoverinfo="text",
        showlegend=False,
    ))

# Legend
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
    marker=dict(color=COL_CORRECT, size=9), name="AI correct"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
    marker=dict(color=COL_WRONG, size=9), name="AI wrong"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
    marker=dict(color="grey", size=15, symbol="star"), name="GT mismatch paper"))

# Threshold line
fig.add_vline(x=THRESHOLD, line_width=2, line_dash="dot", line_color="grey")

fig.update_layout(
    title="<b>Score vs Accept divergence:</b> where score and decision disagree<br>"
          f"<sup>Threshold = 5.5. Stars = mismatch papers. "
          f"AI got {mm_right}/{mm_count} of these right.</sup>",
    xaxis=dict(title="GT score (continuous 1-10)", range=[0.5, 10.5],
               dtick=1, gridcolor="#eee"),
    yaxis=dict(title="GT decision", tickvals=[0, 1], ticktext=["REJECT", "ACCEPT"],
               range=[-0.5, 1.5], gridcolor="#eee"),
    height=480, width=1050,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=12),
    legend=dict(orientation="h", y=-0.20, x=0.25),
)

fig.add_annotation(x=3.0, y=1.35, text="<b>Low-score ACCEPT</b><br>(mismatch zone)",
                   showarrow=False, font=dict(size=10, color="#c0392b"), opacity=0.7)
fig.add_annotation(x=8.0, y=-0.35, text="<b>High-score REJECT</b><br>(mismatch zone)",
                   showarrow=False, font=dict(size=10, color="#c0392b"), opacity=0.7)
fig.add_annotation(x=3.0, y=-0.35, text="Clean reject zone",
                   showarrow=False, font=dict(size=10, color="#27ae60"), opacity=0.6)
fig.add_annotation(x=8.0, y=1.35, text="Clean accept zone",
                   showarrow=False, font=dict(size=10, color="#27ae60"), opacity=0.6)

save(fig, "5_divergence.png")

print("\nAll plots saved to:", OUT)
