"""Add score-vs-decision mismatch papers to v2.

Adds 11 papers covering cases where weighted_mean_overall and accept disagree —
these are the hard cases for a paper-only AI judge (rebuttal-blindness ceiling).

Composition:
  1  low-score accept  (wmean < 3.0, accept=True)
  4  high-score reject  [4.0, 4.5)
  4  high-score reject  [4.5, 5.0)
  2  high-score reject  [5.0, 5.5)  (entire pool — only 2 exist)

Skips any paper_id already present in v2. Sanitizes + post-fixes inline so no
manual re-run of the sanitization pipeline is needed.
"""
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import fitz
import requests
from openreview.api import OpenReviewClient

ROOT = Path("neurips-2025-dataset-v2")
PAPERS = ROOT / "papers"
RAW = ROOT / "open-reviews-raw"
QUANT = ROOT / "open-reviews-quantized"
GT = ROOT / "open-reviews-ground_truth"
SANI = ROOT / "sanitized-papers"
CACHE = ROOT / "_cache"

V1_CACHE = Path("neurips-2025-dataset/_cache/all_papers.json")
SEED = 42

client = OpenReviewClient(baseurl="https://api2.openreview.net")

def slugify(t, n=60):
    s = re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
    return s[:n].rstrip("-")

papers = json.loads(V1_CACHE.read_text())
papers = [p for p in papers if not p["ethics_flagged"]]

existing_ids = {json.loads(f.read_text())["paper_id"] for f in QUANT.glob("*.json")}
print(f"v2 currently has {len(existing_ids)} papers")

# Pools (excluding ones already in v2)
def pool(lo, hi, accept):
    return [p for p in papers
            if lo <= p["weighted_mean_overall"] < hi
            and p["accept"] == accept
            and p["paper_id"] not in existing_ids]

rng = random.Random(SEED)
adds = []
# Low-score accept: take the (only) one
low_acc = [p for p in papers if p["weighted_mean_overall"] < 3.0 and p["accept"] and p["paper_id"] not in existing_ids]
adds.extend(low_acc)
print(f"  low-score accept: +{len(low_acc)}")

# High-score rejects per bin
for lo, hi, k in [(4.0, 4.5, 4), (4.5, 5.0, 4), (5.0, 5.5, 2)]:
    candidates = pool(lo, hi, accept=False)
    k_eff = min(k, len(candidates))
    picks = rng.sample(candidates, k_eff)
    adds.extend(picks)
    print(f"  high-score reject [{lo},{hi}): pool={len(candidates)} took {k_eff}")

print(f"\nAdding {len(adds)} papers")

# ---- Sanitization helpers (same as build.py) ----
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
URL_RE = re.compile(
    r"\b(?:https?://)?(?:github\.com|gitlab\.com|huggingface\.co|sites\.google\.com|"
    r"[a-z0-9\-]+\.github\.io|[a-z0-9\-]+\.(?:edu|ac\.[a-z]{2}|ai|io|org))/[A-Za-z0-9_./\-?=&%#~+]*",
    re.IGNORECASE,
)
ARXIV_RE = re.compile(r"arXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?", re.IGNORECASE)
SECTION_PREFIXES = ["acknowledg", "author contribution", "funding", "financial disclosure",
                    "conflict of interest", "conflicts of interest",
                    "declaration of interest", "declarations of interest",
                    "competing interest", "competing interests"]
STOP_PREFIXES = ["reference", "bibliography", "appendix", "appendices",
                 "supplementary", "supplemental", "broader impact", "ethics statement"]
COMMON_NAME_WORDS = {"or","an","in","on","is","be","of","to","we","do","go","no","so","as",
                     "the","and","for","but","our","all","any","new","use","see","let",
                     "wei","li","wu","xu","ma","he","su","yu","lu"}

def name_variants(n):
    parts = [p for p in re.split(r"\s+", n.strip()) if p and not p.endswith(".")]
    out = set()
    if len(n.strip()) >= 4:
        out.add(n.strip())
    if len(parts) >= 2:
        if len(parts[-1]) >= 4 and parts[-1].lower() not in COMMON_NAME_WORDS:
            out.add(parts[-1])
        if len(parts[0]) >= 4 and parts[0].lower() not in COMMON_NAME_WORDS:
            out.add(parts[0])
    return out

def title_abstract_y(page):
    spans = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0: continue
        for line in b.get("lines", []):
            for sp in line.get("spans", []):
                spans.append({"text": sp["text"], "size": sp["size"], "y0": sp["bbox"][1], "y1": sp["bbox"][3]})
    if not spans: return 0.0, page.rect.height
    mx = max(s["size"] for s in spans)
    title_spans = [s for s in spans if abs(s["size"]-mx) < 0.5 and s["y0"] < page.rect.height*0.5]
    y_t = max((s["y1"] for s in title_spans), default=0.0)
    y_a = page.rect.height
    for s in spans:
        if s["text"].strip().lower().startswith("abstract") and s["y0"] > y_t:
            y_a = s["y0"]; break
    return y_t, y_a

def section_ranges(doc):
    items = []
    for pno, page in enumerate(doc):
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0: continue
            for line in b.get("lines", []):
                txt = "".join(sp["text"] for sp in line["spans"]).strip()
                if not txt or len(txt) > 80: continue
                items.append({"page": pno, "text": txt, "bbox": line["bbox"]})
    def classify(t):
        n = re.sub(r"^\d+(\.\d+)*\.?\s*", "", t).strip().lower().rstrip(":").rstrip(".")
        for p in SECTION_PREFIXES:
            if n.startswith(p): return "sensitive"
        for p in STOP_PREFIXES:
            if n.startswith(p): return "stop"
        return None
    heads = [(i, x, classify(x["text"])) for i, x in enumerate(items)]
    heads = [(i, x, k) for i, x, k in heads if k]
    out = []
    for idx, (i, x, k) in enumerate(heads):
        if k != "sensitive": continue
        end = heads[idx+1][1] if idx+1 < len(heads) else None
        out.append((x["page"], fitz.Rect(x["bbox"]),
                    end["page"] if end else (items[-1]["page"] if items else x["page"]),
                    fitz.Rect(end["bbox"]) if end else None))
    return out

def sanitize_pdf(in_p, out_p, authors):
    doc = fitz.open(in_p)
    doc.set_metadata({})
    if len(doc) > 0:
        page = doc[0]
        y_t, y_a = title_abstract_y(page)
        if y_a > y_t:
            page.add_redact_annot(fitz.Rect(0, y_t+1, page.rect.width, y_a-1), fill=(0,0,0))
    for page in doc:
        text = page.get_text()
        for rx in (EMAIL_RE, ORCID_RE, URL_RE, ARXIV_RE):
            for m in rx.finditer(text):
                for q in page.search_for(m.group(0), quads=True):
                    page.add_redact_annot(q.rect, fill=(0,0,0))
    for (sp, sr, ep, er) in section_ranges(doc):
        if sp == ep:
            page = doc[sp]
            page.add_redact_annot(fitz.Rect(0, sr.y0, page.rect.width, er.y0 if er else page.rect.height), fill=(0,0,0))
        else:
            doc[sp].add_redact_annot(fitz.Rect(0, sr.y0, doc[sp].rect.width, doc[sp].rect.height), fill=(0,0,0))
            for mid in range(sp+1, ep):
                doc[mid].add_redact_annot(doc[mid].rect, fill=(0,0,0))
            doc[ep].add_redact_annot(fitz.Rect(0, 0, doc[ep].rect.width, er.y0 if er else doc[ep].rect.height), fill=(0,0,0))
    for a in authors:
        for v in name_variants(a):
            for page in doc:
                for q in page.search_for(v, quads=True):
                    page.add_redact_annot(q.rect, fill=(0,0,0))
    # Strip link annotations (the fix from sanitize-fix pass)
    for page in doc:
        for ln in list(page.get_links()):
            try: page.delete_link(ln)
            except Exception: pass
    for page in doc:
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    doc.save(out_p, garbage=4, deflate=True, clean=True)
    doc.close()

def bin_for(w):
    lo = int(w * 2) / 2  # floor to 0.5
    return f"[{lo:.1f},{lo+0.5:.1f})"

def download_one(p):
    slug = slugify(p["title"])
    pid = p["paper_id"]
    pdf_path = PAPERS / f"{slug}.pdf"
    url = "https://openreview.net" + p["pdf_url_path"]
    if not pdf_path.exists():
        r = requests.get(url, timeout=180); r.raise_for_status()
        assert r.content[:5] == b"%PDF-"
        pdf_path.write_bytes(r.content)

    # mismatch label
    mismatch = ((p["weighted_mean_overall"] < 3.0 and p["accept"]) or
                (p["weighted_mean_overall"] >= 4.0 and not p["accept"]))

    (PAPERS / f"{slug}.meta.json").write_text(json.dumps({
        "paper_id": pid, "title": p["title"], "venue": p["venue"],
        "decision": p["decision"], "accept": p["accept"],
        "bin": bin_for(p["weighted_mean_overall"]),
        "mismatch": mismatch,
        "openreview_url": f"https://openreview.net/forum?id={pid}",
    }, indent=2))
    (RAW / f"{slug}.json").write_text(json.dumps({
        "paper_id": pid, "decision": p["decision"], "reviews": p["reviews"],
    }, indent=2))
    (QUANT / f"{slug}.json").write_text(json.dumps({
        "paper_id": pid, "decision": p["decision"], "accept": p["accept"],
        "reviews": [{"reviewer_id": r["reviewer_id"], "overall": r["overall"],
                     "confidence": r["confidence"], "quality": r["quality"],
                     "clarity": r["clarity"], "significance": r["significance"],
                     "originality": r["originality"]} for r in p["reviews"]],
        "ground_truth": {"weighted_mean_overall": p["weighted_mean_overall"],
                         "accept": p["accept"]},
    }, indent=2))
    (GT / f"{slug}.json").write_text(json.dumps({
        "paper_id": pid,
        "ground_truth": {"weighted_mean_overall": p["weighted_mean_overall"],
                         "accept": p["accept"]},
    }, indent=2))

    # sanitize
    authors = client.get_note(pid).content.get("authors", {}).get("value", []) or []
    sanitize_pdf(pdf_path, SANI / f"{slug}.pdf", authors)
    return slug

print("\nDownloading + sanitizing...")
with ThreadPoolExecutor(max_workers=6) as ex:
    futs = [ex.submit(download_one, p) for p in adds]
    for i, fut in enumerate(as_completed(futs), 1):
        slug = fut.result()
        print(f"  [{i:>2}/{len(adds)}] {slug[:60]}")

# Backfill mismatch=False on the existing 40 meta.jsons so the field is consistent
print("\nBackfilling mismatch=False on existing v2 meta files...")
for f in PAPERS.glob("*.meta.json"):
    d = json.loads(f.read_text())
    if "mismatch" not in d:
        d["mismatch"] = False
        f.write_text(json.dumps(d, indent=2))

n_total = len(list(QUANT.glob("*.json")))
print(f"\nDone. v2 now has {n_total} papers.")
