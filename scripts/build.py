"""v2 builder: 40 papers, 4 per 0.5-wide bin across the NeurIPS 2025 1-6 scale.
Reuses v1 cache (skips 5540-submission refetch). Parallel PDF download.
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
SANI = ROOT / "sanitized-papers"
CACHE = ROOT / "_cache"
for d in (PAPERS, RAW, QUANT, SANI, CACHE):
    d.mkdir(parents=True, exist_ok=True)

V1_CACHE = Path("neurips-2025-dataset/_cache/all_papers.json")
SEED = 42
N_PER_BIN = 4
EDGES = [round(1.0 + 0.5*i, 1) for i in range(11)]  # 1.0 .. 6.0

client = OpenReviewClient(baseurl="https://api2.openreview.net")

def slugify(t, n=60):
    s = re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
    return s[:n].rstrip("-")

# ---- load v1 cache ----
print(f"Loading {V1_CACHE}...")
papers = json.loads(V1_CACHE.read_text())
papers = [p for p in papers if not p["ethics_flagged"]]
print(f"  {len(papers)} non-ethics-flagged papers")

# ---- stratified select ----
rng = random.Random(SEED)
selection = []
print("\nSampling 4 per bin:")
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    pool = [p for p in papers if lo <= p["weighted_mean_overall"] < hi]
    k = min(N_PER_BIN, len(pool))
    picks = rng.sample(pool, k)
    for p in picks:
        selection.append({"bin": f"[{lo:.1f},{hi:.1f})", "slug": slugify(p["title"]), **p})
    print(f"  [{lo:.1f},{hi:.1f}) pool={len(pool):>5} -> took {k}")
print(f"  total selected: {len(selection)}")

(CACHE / "selection.json").write_text(json.dumps(
    [{k: v for k, v in s.items() if k != "reviews"} for s in selection], indent=2))

# ---- parallel PDF download + JSON writes ----
def download_one(s):
    slug, pid = s["slug"], s["paper_id"]
    url = "https://openreview.net" + s["pdf_url_path"]
    pdf_path = PAPERS / f"{slug}.pdf"
    if not pdf_path.exists():
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        assert r.content[:5] == b"%PDF-"
        pdf_path.write_bytes(r.content)

    (PAPERS / f"{slug}.meta.json").write_text(json.dumps({
        "paper_id": pid, "title": s["title"], "venue": s["venue"],
        "decision": s["decision"], "accept": s["accept"], "bin": s["bin"],
        "openreview_url": f"https://openreview.net/forum?id={pid}",
    }, indent=2))
    (RAW / f"{slug}.json").write_text(json.dumps({
        "paper_id": pid, "decision": s["decision"], "reviews": s["reviews"],
    }, indent=2))
    (QUANT / f"{slug}.json").write_text(json.dumps({
        "paper_id": pid, "decision": s["decision"], "accept": s["accept"],
        "reviews": [{"reviewer_id": r["reviewer_id"], "overall": r["overall"],
                     "confidence": r["confidence"], "quality": r["quality"],
                     "clarity": r["clarity"], "significance": r["significance"],
                     "originality": r["originality"]} for r in s["reviews"]],
        "ground_truth": {"weighted_mean_overall": s["weighted_mean_overall"],
                         "accept": s["accept"]},
    }, indent=2))
    return slug

print("\nDownloading PDFs (8 workers)...")
with ThreadPoolExecutor(max_workers=8) as ex:
    for i, fut in enumerate(as_completed([ex.submit(download_one, s) for s in selection]), 1):
        slug = fut.result()
        print(f"  [{i:>2}/{len(selection)}] {slug[:60]}")

# ---- sanitize (same pipeline as v1) ----
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

def name_variants(n):
    parts = [p for p in re.split(r"\s+", n.strip()) if p and not p.endswith(".")]
    out = {n.strip()}
    if len(parts) >= 2:
        out.add(parts[-1]); out.add(f"{parts[0]} {parts[-1]}"); out.add(parts[0])
    return [v for v in out if len(v) >= 3]

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

def sanitize_one(s):
    slug = s["slug"]
    in_p = PAPERS / f"{slug}.pdf"
    out_p = SANI / f"{slug}.pdf"
    log_p = SANI / f"{slug}.sanitize.log.json"

    authors = client.get_note(s["paper_id"]).content.get("authors", {}).get("value", []) or []
    doc = fitz.open(in_p)
    doc.set_metadata({})
    log = {"redactions": [], "pattern": {}, "section_redactions": 0, "author_name_redactions": {}}

    if len(doc) > 0:
        page = doc[0]
        y_t, y_a = title_abstract_y(page)
        if y_a > y_t:
            page.add_redact_annot(fitz.Rect(0, y_t+1, page.rect.width, y_a-1), fill=(0,0,0))
            log["redactions"].append("title_page_author_block")

    counts = {"email": 0, "orcid": 0, "url": 0, "arxiv": 0}
    for page in doc:
        text = page.get_text()
        for name, rx in [("email", EMAIL_RE), ("orcid", ORCID_RE), ("url", URL_RE), ("arxiv", ARXIV_RE)]:
            for m in rx.finditer(text):
                for q in page.search_for(m.group(0), quads=True):
                    page.add_redact_annot(q.rect, fill=(0,0,0))
                    counts[name] += 1
    log["pattern"] = counts

    for (sp, sr, ep, er) in section_ranges(doc):
        if sp == ep:
            page = doc[sp]
            page.add_redact_annot(fitz.Rect(0, sr.y0, page.rect.width, er.y0 if er else page.rect.height), fill=(0,0,0))
        else:
            doc[sp].add_redact_annot(fitz.Rect(0, sr.y0, doc[sp].rect.width, doc[sp].rect.height), fill=(0,0,0))
            for mid in range(sp+1, ep):
                doc[mid].add_redact_annot(doc[mid].rect, fill=(0,0,0))
            doc[ep].add_redact_annot(fitz.Rect(0, 0, doc[ep].rect.width, er.y0 if er else doc[ep].rect.height), fill=(0,0,0))
        log["section_redactions"] += 1

    name_hits = {}
    for a in authors:
        for v in name_variants(a):
            for page in doc:
                for q in page.search_for(v, quads=True):
                    page.add_redact_annot(q.rect, fill=(0,0,0))
                    name_hits[v] = name_hits.get(v, 0) + 1
    log["author_name_redactions"] = name_hits

    for page in doc:
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    doc.save(out_p, garbage=4, deflate=True, clean=True)
    doc.close()

    # verify
    doc = fitz.open(out_p)
    text = "\n".join(p.get_text() for p in doc)
    doc.close()
    issues = []
    for a in authors:
        if re.search(r"\b" + re.escape(a) + r"\b", text, re.IGNORECASE):
            issues.append({"type": "author_full_name_leak", "name": a})
        parts = a.split()
        if len(parts) >= 2 and len(parts[-1]) >= 5:
            if re.search(r"\b" + re.escape(parts[-1]) + r"\b", text):
                issues.append({"type": "author_last_name_leak", "name": parts[-1]})
    for nm, rx in [("email", EMAIL_RE), ("orcid", ORCID_RE), ("url", URL_RE)]:
        h = rx.findall(text)
        if h:
            issues.append({"type": f"residual_{nm}", "count": len(h), "samples": list(set(h))[:3]})
    log["authors_groundtruth"] = authors
    log["verify"] = {"ok": not issues, "issues": issues}
    log_p.write_text(json.dumps(log, indent=2))
    return slug, log["verify"]["ok"], issues

print("\nSanitizing...")
fail = []
for i, s in enumerate(selection, 1):
    slug, ok, issues = sanitize_one(s)
    print(f"  [{i:>2}/{len(selection)}] {slug[:55]:<55} VERIFY={'OK' if ok else 'FAIL'}")
    if not ok:
        fail.append((slug, issues))

print(f"\nDone. {len(selection)} papers. Sanitization failures: {len(fail)}")
for slug, issues in fail:
    print(f"  ! {slug}: {issues}")
