"""Integrity verification for iclr-2025-dataset/ (mirrors verify_dataset.py for v2)."""
import json
from pathlib import Path
import fitz

ROOT = Path("iclr-2025-dataset")
DIRS = {
    "papers_pdf":   (ROOT / "papers", "*.pdf"),
    "papers_meta":  (ROOT / "papers", "*.meta.json"),
    "raw":          (ROOT / "open-reviews-raw", "*.json"),
    "quantized":    (ROOT / "open-reviews-quantized", "*.json"),
    "ground_truth": (ROOT / "open-reviews-ground_truth", "*.json"),
    "sanitized":    (ROOT / "sanitized-papers", "*.pdf"),
}

issues = []
def fail(m): issues.append(("FAIL", m)); print(f"  FAIL: {m}")

print("[1] Slug counts per dir:")
slug_sets = {}
for name, (d, pat) in DIRS.items():
    s = set()
    for f in d.glob(pat):
        n = f.name
        for suf in (".meta.json", ".sanitize.log.json", ".json", ".pdf"):
            if n.endswith(suf): n = n[:-len(suf)]; break
        s.add(n)
    slug_sets[name] = s
    print(f"  {name:<14} {len(s)}")

ref = slug_sets["ground_truth"]
for name, s in slug_sets.items():
    if s - ref: fail(f"{name} has {len(s-ref)} extra slugs")
    if ref - s: fail(f"{name} missing {len(ref-s)} slugs")

print("\n[2-4] Cross-file consistency:")
seen_pid = {}
for slug in sorted(ref):
    meta = json.loads((ROOT / "papers" / f"{slug}.meta.json").read_text())
    raw  = json.loads((ROOT / "open-reviews-raw" / f"{slug}.json").read_text())
    quant = json.loads((ROOT / "open-reviews-quantized" / f"{slug}.json").read_text())
    gt    = json.loads((ROOT / "open-reviews-ground_truth" / f"{slug}.json").read_text())
    ids = {meta["paper_id"], raw["paper_id"], quant["paper_id"], gt["paper_id"]}
    if len(ids) != 1: fail(f"{slug}: paper_id mismatch {ids}")
    pid = meta["paper_id"]
    if pid in seen_pid: fail(f"duplicate {pid} in {seen_pid[pid]} and {slug}")
    seen_pid[pid] = slug
    if quant["ground_truth"] != gt["ground_truth"]:
        fail(f"{slug}: ground_truth differs quant vs gt")
    if not (meta["accept"] == quant["accept"] == gt["ground_truth"]["accept"]):
        fail(f"{slug}: accept inconsistent")
    if meta["decision"].lower().startswith("accept") != meta["accept"]:
        fail(f"{slug}: decision/accept disagree")

print("\n[5] Sanitized PDFs valid:")
for slug in sorted(ref):
    p = ROOT / "sanitized-papers" / f"{slug}.pdf"
    if not p.exists() or p.stat().st_size < 1024:
        fail(f"{slug}: bad sanitized PDF")
        continue
    try:
        d = fitz.open(p)
        if d.page_count == 0: fail(f"{slug}: 0 pages")
        d.close()
    except Exception as e:
        fail(f"{slug}: cannot open: {e}")
print(f"  checked {len(ref)}")

print(f"\nDataset size: {len(ref)}")
fails = sum(1 for s, _ in issues if s == "FAIL")
print(f"FAILs: {fails}")
print("DATASET SOUND" if fails == 0 else "DATASET HAS ISSUES")
