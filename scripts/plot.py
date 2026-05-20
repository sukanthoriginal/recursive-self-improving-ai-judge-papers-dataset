"""Plot v2 ground-truth distribution: 4 papers per 0.5-wide bin × 10 bins."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

QUANT = Path("neurips-2025-dataset-v2/open-reviews-quantized")
OUT = Path("neurips-2025-dataset-v2/v2_ground_truth.png")

rows = []
for f in sorted(QUANT.glob("*.json")):
    d = json.loads(f.read_text())
    rows.append({"wmean": d["ground_truth"]["weighted_mean_overall"],
                 "accept": d["ground_truth"]["accept"]})

edges = [round(1.0 + 0.5*i, 1) for i in range(11)]
accept_vals = [r["wmean"] for r in rows if r["accept"]]
reject_vals = [r["wmean"] for r in rows if not r["accept"]]

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist([accept_vals, reject_vals], bins=edges, stacked=True,
        color=["#2ca02c", "#d62728"], label=["Accept", "Reject"],
        edgecolor="black", linewidth=0.5)

ax.set_xlabel("weighted_mean_overall (NeurIPS 2025 scale: 1–6)")
ax.set_ylabel("count")
ax.set_title(f"v2 ground-truth distribution — n={len(rows)}")
ax.set_xticks(edges)
ax.set_xlim(1, 6)
bin_totals = [sum(1 for r in rows if lo <= r["wmean"] < hi) for lo, hi in zip(edges[:-1], edges[1:])]
y_top = max(bin_totals) + 1
ax.set_ylim(0, y_top)
ax.axvline(3.5, color="gray", linestyle="--", lw=0.7, alpha=0.6)
ax.text(3.5, y_top - 0.3, " borderline", color="gray", fontsize=8, va="top")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=140)
print(f"saved {OUT}")
print(f"  n={len(rows)}, accepts={len(accept_vals)}, rejects={len(reject_vals)}")
