"""Verify v2 dataset integrity. Checks:
  1. Each artifact dir has the same set of slugs (no orphans, no missing).
  2. paper_id is consistent across all files for a given slug.
  3. ground_truth in quantized JSON matches ground_truth in ground_truth JSON.
  4. accept field in meta/quantized/ground_truth all agree.
  5. Sanitized PDF exists for every paper and is a valid PDF.
  6. mismatch flag is consistent with the score↔decision rule.
  7. No duplicate paper_ids across slugs.
"""
import json
from pathlib import Path

ROOT = Path("neurips-2025-dataset-v2")
DIRS = {
    "papers_pdf":   (ROOT / "papers", "*.pdf"),
    "papers_meta":  (ROOT / "papers", "*.meta.json"),
    "raw":          (ROOT / "open-reviews-raw", "*.json"),
    "quantized":    (ROOT / "open-reviews-quantized", "*.json"),
    "ground_truth": (ROOT / "open-reviews-ground_truth", "*.json"),
    "sanitized":    (ROOT / "sanitized-papers", "*.pdf"),
}

issues = []
def fail(msg): issues.append(("FAIL", msg)); print(f"  FAIL: {msg}")
def warn(msg): issues.append(("WARN", msg)); print(f"  WARN: {msg}")

# --- 1. Slug sets per dir
print("[1] Slug counts per artifact dir:")
slug_sets = {}
for name, (d, pat) in DIRS.items():
    slugs = set()
    for f in d.glob(pat):
        s = f.name
        if s.endswith(".meta.json"): s = s[:-len(".meta.json")]
        elif s.endswith(".json"): s = s[:-len(".json")]
        elif s.endswith(".pdf"): s = s[:-len(".pdf")]
        elif s.endswith(".sanitize.log.json"): s = s[:-len(".sanitize.log.json")]
        slugs.add(s)
    slug_sets[name] = slugs
    print(f"  {name:<14} {len(slugs)} slugs")

reference = slug_sets["ground_truth"]
for name, slugs in slug_sets.items():
    missing = reference - slugs
    extra = slugs - reference
    if missing: fail(f"{name} is MISSING {len(missing)} slugs: {sorted(missing)[:3]}")
    if extra:   fail(f"{name} has EXTRA {len(extra)} slugs: {sorted(extra)[:3]}")

# --- 2/3/4. Cross-file consistency
print("\n[2-4] Cross-file consistency:")
paper_id_seen = {}  # pid -> slug
for slug in sorted(reference):
    meta = json.loads((ROOT / "papers" / f"{slug}.meta.json").read_text())
    raw  = json.loads((ROOT / "open-reviews-raw" / f"{slug}.json").read_text())
    quant = json.loads((ROOT / "open-reviews-quantized" / f"{slug}.json").read_text())
    gt    = json.loads((ROOT / "open-reviews-ground_truth" / f"{slug}.json").read_text())

    # paper_id consistency
    ids = {meta["paper_id"], raw["paper_id"], quant["paper_id"], gt["paper_id"]}
    if len(ids) != 1:
        fail(f"{slug}: paper_id mismatch across files: {ids}")
    pid = meta["paper_id"]
    if pid in paper_id_seen:
        fail(f"duplicate paper_id {pid}: in both {paper_id_seen[pid]} and {slug}")
    paper_id_seen[pid] = slug

    # ground_truth consistency
    if quant["ground_truth"] != gt["ground_truth"]:
        fail(f"{slug}: ground_truth mismatch quantized vs ground_truth dir")

    # accept agreement
    accepts = {meta["accept"], quant["accept"], gt["ground_truth"]["accept"]}
    if len(accepts) != 1:
        fail(f"{slug}: accept disagrees across files: meta={meta['accept']} quant={quant['accept']} gt={gt['ground_truth']['accept']}")

    # decision matches accept
    dec = meta["decision"]
    if dec.lower().startswith("accept") != meta["accept"]:
        fail(f"{slug}: decision={dec!r} disagrees with accept={meta['accept']}")

# --- 5. Sanitized PDFs valid
print("\n[5] Sanitized PDF validity:")
import fitz
for slug in sorted(reference):
    p = ROOT / "sanitized-papers" / f"{slug}.pdf"
    if not p.exists():
        fail(f"{slug}: sanitized PDF missing")
        continue
    if p.stat().st_size < 1024:
        fail(f"{slug}: sanitized PDF suspiciously small ({p.stat().st_size} bytes)")
    try:
        doc = fitz.open(p)
        if doc.page_count == 0:
            fail(f"{slug}: sanitized PDF has 0 pages")
        doc.close()
    except Exception as e:
        fail(f"{slug}: cannot open sanitized PDF: {e}")
print(f"  checked {len(reference)} PDFs")

# --- 6. mismatch flag consistency
print("\n[6] mismatch flag consistency:")
mm_total = 0
for slug in sorted(reference):
    meta = json.loads((ROOT / "papers" / f"{slug}.meta.json").read_text())
    gt = json.loads((ROOT / "open-reviews-ground_truth" / f"{slug}.json").read_text())
    w = gt["ground_truth"]["weighted_mean_overall"]
    a = gt["ground_truth"]["accept"]
    expected = (w < 3.0 and a) or (w >= 4.0 and not a)
    actual = meta.get("mismatch", False)
    if expected != actual:
        warn(f"{slug}: mismatch flag = {actual} but expected {expected} (wmean={w:.2f}, accept={a})")
    if actual:
        mm_total += 1
print(f"  {mm_total} papers flagged mismatch=True")

# --- 7. Summary
print(f"\n{'='*60}")
print(f"Dataset size: {len(reference)} papers")
fails = sum(1 for s, _ in issues if s == "FAIL")
warns = sum(1 for s, _ in issues if s == "WARN")
print(f"FAILs: {fails}  WARNs: {warns}")
print("DATASET SOUND ✓" if fails == 0 else "DATASET HAS ISSUES ✗")
