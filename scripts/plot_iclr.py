"""Plot ICLR 2025 ground-truth distribution. Scale 1-10, 1.0-wide bins."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

GT = Path("iclr-2025-dataset/open-reviews-ground_truth")
OUT = Path("iclr-2025-dataset/iclr_ground_truth.png")

rows = []
for f in sorted(GT.glob("*.json")):
    d = json.loads(f.read_text())
    rows.append({"wmean": d["ground_truth"]["weighted_mean_overall"],
                 "accept": d["ground_truth"]["accept"]})

edges = list(range(1, 12))  # 1..11 for [1,2)..[10,11)
accept_vals = [r["wmean"] for r in rows if r["accept"]]
reject_vals = [r["wmean"] for r in rows if not r["accept"]]

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.hist([accept_vals, reject_vals], bins=edges, stacked=True,
        color=["#2ca02c", "#d62728"], label=["Accept", "Reject"],
        edgecolor="black", linewidth=0.5)
ax.set_xlabel("weighted_mean_overall (ICLR 2025 scale: 1–10)")
ax.set_ylabel("count")
ax.set_title(f"ICLR 2025 ground-truth distribution — n={len(rows)}")
ax.set_xticks(edges)
ax.set_xlim(1, 11)
bin_totals = [sum(1 for r in rows if lo <= r["wmean"] < hi) for lo, hi in zip(edges[:-1], edges[1:])]
y_top = max(bin_totals) + 1
ax.set_ylim(0, y_top)
ax.axvline(5.5, color="gray", linestyle="--", lw=0.7, alpha=0.6)
ax.text(5.5, y_top - 0.3, " accept threshold (~5.5)", color="gray", fontsize=8, va="top")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=140)
print(f"saved {OUT}  n={len(rows)} accepts={len(accept_vals)} rejects={len(reject_vals)}")
