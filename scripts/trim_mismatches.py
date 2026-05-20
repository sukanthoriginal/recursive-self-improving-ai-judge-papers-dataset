"""Trim mismatch papers in v2 down to 1 per high-score bin (+ 1 low-score accept).

Keeps the first mismatch paper (by slug, sorted) in each bin and deletes the rest
across all artifact directories.
"""
import json
from pathlib import Path

ROOT = Path("neurips-2025-dataset-v2")
GT = ROOT / "open-reviews-ground_truth"
META = ROOT / "papers"
DIRS = ["papers", "open-reviews-raw", "open-reviews-quantized",
        "open-reviews-ground_truth", "sanitized-papers"]

# Group mismatch papers by 0.5-bin
mismatch_by_bin = {}
for f in GT.glob("*.json"):
    slug = f.stem
    meta = json.loads((META / f"{slug}.meta.json").read_text())
    if not meta.get("mismatch"):
        continue
    wmean = json.loads(f.read_text())["ground_truth"]["weighted_mean_overall"]
    lo = int(wmean * 2) / 2
    bin_key = (lo, lo + 0.5)
    mismatch_by_bin.setdefault(bin_key, []).append((slug, wmean, meta["accept"]))

print("Current mismatch coverage:")
keep, drop = [], []
for bin_key in sorted(mismatch_by_bin):
    papers = sorted(mismatch_by_bin[bin_key])  # deterministic by slug
    kept = papers[0]
    dropped = papers[1:]
    keep.append(kept)
    drop.extend(dropped)
    print(f"  [{bin_key[0]:.1f},{bin_key[1]:.1f})  total={len(papers)}  keep={kept[0][:50]}")
    for d in dropped:
        print(f"    drop: {d[0][:60]}")

print(f"\nDropping {len(drop)} files × {len(DIRS)} dirs...")
for slug, _, _ in drop:
    for sub in DIRS:
        d = ROOT / sub
        for f in d.glob(f"{slug}.*"):
            f.unlink()
            print(f"  rm {f.relative_to(ROOT)}")

n_total = len(list((ROOT / "open-reviews-ground_truth").glob("*.json")))
print(f"\nv2 now has {n_total} papers.")
