"""End-to-end ICLR 2025 dataset builder.

Composition (target ~50 papers, actual 49):
  - 5 per bin × 9 bins ([1,2) through [9,10))   = 45
  - 1 paper from [10,10]                         =  1
  - 3 mismatch papers:
      1 low-score accept (wmean<4)
      1 high-score reject in [6,7)
      1 high-score reject in [7,8)

Pipeline (same as NeurIPS v2):
  1. Fetch all submissions + reviews + decisions (cached)
  2. Stratified sample, deterministic (seed=42)
  3. Parallel PDF download
  4. Write meta / raw / quantized / ground_truth JSONs
  5. Sanitize PDFs with text + link-annotation redaction
  6. Aggressive sanity-check audit (in-pipeline)
"""
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import fitz
import requests
from openreview.api import OpenReviewClient

ROOT = Path("iclr-2025-dataset")
PAPERS = ROOT / "papers"
RAW = ROOT / "open-reviews-raw"
QUANT = ROOT / "open-reviews-quantized"
GT = ROOT / "open-reviews-ground_truth"
SANI = ROOT / "sanitized-papers"
CACHE = ROOT / "_cache"
for d in (PAPERS, RAW, QUANT, GT, SANI, CACHE):
    d.mkdir(parents=True, exist_ok=True)

VENUE = "ICLR.cc/2025/Conference"
SEED = 42

client = OpenReviewClient(baseurl="https://api2.openreview.net")

def slugify(t, n=60):
    s = re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
    return s[:n].rstrip("-")

def ethics_substantive(flag, det):
    if isinstance(flag, list):
        for it in flag:
            if isinstance(it, str) and it.strip().lower() not in (
                "", "no ethics review needed.", "no ethics review needed", "none"):
                return True
    if isinstance(det, str) and det.strip().lower() not in ("", "n/a", "na", "none", "no"):
        return True
    return False

# ---------- 1. Fetch ----------
print("Fetching all ICLR 2025 submissions with replies...")
subs = client.get_all_notes(invitation=f"{VENUE}/-/Submission", details="directReplies")
print(f"  {len(subs)} submissions")

papers = []
for s in subs:
    replies = (s.details or {}).get("directReplies", []) or []
    reviews_raw, decision_str = [], None
    for r in replies:
        invs = r.get("invitations", [])
        content = r.get("content", {}) or {}
        if any(i.endswith("/-/Official_Review") for i in invs):
            rating = content.get("rating", {}).get("value")
            conf = content.get("confidence", {}).get("value")
            if rating is None or conf is None:
                continue
            reviews_raw.append({
                "reviewer_id": r.get("id"),
                "rating": rating,
                "confidence": conf,
                "soundness": content.get("soundness", {}).get("value"),
                "presentation": content.get("presentation", {}).get("value"),
                "contribution": content.get("contribution", {}).get("value"),
                "summary": content.get("summary", {}).get("value"),
                "strengths": content.get("strengths", {}).get("value"),
                "weaknesses": content.get("weaknesses", {}).get("value"),
                "questions": content.get("questions", {}).get("value"),
                "flag_for_ethics_review": content.get("flag_for_ethics_review", {}).get("value"),
                "details_of_ethics_concerns": content.get("details_of_ethics_concerns", {}).get("value"),
            })
        elif any(i.endswith("/-/Decision") for i in invs):
            decision_str = content.get("decision", {}).get("value")
    if len(reviews_raw) < 3 or decision_str is None:
        continue
    overalls = [r["rating"] for r in reviews_raw]
    confs = [r["confidence"] for r in reviews_raw]
    wmean = sum(o*c for o,c in zip(overalls, confs)) / sum(confs)
    accept = decision_str.lower().startswith("accept")
    ethics = any(ethics_substantive(r["flag_for_ethics_review"], r["details_of_ethics_concerns"]) for r in reviews_raw)
    papers.append({
        "paper_id": s.id,
        "title": s.content.get("title", {}).get("value", ""),
        "venue": s.content.get("venue", {}).get("value", ""),
        "pdf_url_path": s.content.get("pdf", {}).get("value", ""),
        "decision": decision_str,
        "accept": accept,
        "n_reviewers": len(reviews_raw),
        "weighted_mean_overall": round(wmean, 4),
        "ethics_flagged": ethics,
        "reviews": reviews_raw,
    })
print(f"  usable papers (>=3 reviews, has decision): {len(papers)}")
(CACHE / "all_papers.json").write_text(json.dumps(papers, indent=2))

# ---------- 2. Stratified selection ----------
pool = [p for p in papers if not p["ethics_flagged"]]
print(f"  non-ethics-flagged: {len(pool)}")

rng = random.Random(SEED)
selection = []
seen_ids = set()

def pick(candidates, k, tag):
    avail = [c for c in candidates if c["paper_id"] not in seen_ids]
    n = min(k, len(avail))
    picks = rng.sample(avail, n)
    for p in picks:
        seen_ids.add(p["paper_id"])
        selection.append({"slug": slugify(p["title"]), "tag": tag, **p})
    print(f"  {tag:<22} pool={len(candidates):>5} avail={len(avail):>5} took={n}")

print("\nSampling:")
# 9 main bins
for lo in range(1, 10):
    pick([p for p in pool if lo <= p["weighted_mean_overall"] < lo+1], 5, f"bin[{lo},{lo+1})")
# Top bin [10,10]
pick([p for p in pool if p["weighted_mean_overall"] >= 10.0], 1, "bin[10,10]")
# Mismatches
pick([p for p in pool if p["weighted_mean_overall"] < 4.0 and p["accept"]], 1, "low-score accept")
pick([p for p in pool if 6.0 <= p["weighted_mean_overall"] < 7.0 and not p["accept"]], 1, "high-score reject [6,7)")
pick([p for p in pool if 7.0 <= p["weighted_mean_overall"] < 8.0 and not p["accept"]], 1, "high-score reject [7,8)")

print(f"\nSelected: {len(selection)} papers")
(CACHE / "selection.json").write_text(json.dumps(
    [{k: v for k, v in s.items() if k != "reviews"} for s in selection], indent=2))

# ---------- 3. Sanitization helpers ----------
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
    if len(n.strip()) >= 4: out.add(n.strip())
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
    for page in doc:
        for ln in list(page.get_links()):
            try: page.delete_link(ln)
            except Exception: pass
    for page in doc:
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    doc.save(out_p, garbage=4, deflate=True, clean=True)
    doc.close()

def verify_pdf(out_p, authors):
    doc = fitz.open(out_p)
    text = "\n".join(p.get_text() for p in doc)
    doc.close()
    issues = []
    for a in authors:
        if re.search(r"\b"+re.escape(a)+r"\b", text, re.IGNORECASE):
            issues.append(("author_full", a))
        parts = a.split()
        if len(parts) >= 2 and len(parts[-1]) >= 5:
            if re.search(r"\b"+re.escape(parts[-1])+r"\b", text):
                issues.append(("author_last", parts[-1]))
    for nm, rx in [("email", EMAIL_RE), ("orcid", ORCID_RE), ("url", URL_RE)]:
        h = rx.findall(text)
        if h: issues.append((f"residual_{nm}", len(h)))
    return issues

def process(p):
    slug, pid = p["slug"], p["paper_id"]
    mismatch = ((p["weighted_mean_overall"] < 4.0 and p["accept"]) or
                (p["weighted_mean_overall"] >= 6.0 and not p["accept"]))

    # PDF download
    pdf_path = PAPERS / f"{slug}.pdf"
    if not pdf_path.exists():
        url = "https://openreview.net" + p["pdf_url_path"]
        r = requests.get(url, timeout=180); r.raise_for_status()
        assert r.content[:5] == b"%PDF-"
        pdf_path.write_bytes(r.content)

    # JSONs
    (PAPERS / f"{slug}.meta.json").write_text(json.dumps({
        "paper_id": pid, "title": p["title"], "venue": p["venue"],
        "decision": p["decision"], "accept": p["accept"],
        "stratum": p["tag"], "mismatch": mismatch,
        "openreview_url": f"https://openreview.net/forum?id={pid}",
    }, indent=2))
    (RAW / f"{slug}.json").write_text(json.dumps({
        "paper_id": pid, "decision": p["decision"], "reviews": p["reviews"],
    }, indent=2))
    (QUANT / f"{slug}.json").write_text(json.dumps({
        "paper_id": pid, "decision": p["decision"], "accept": p["accept"],
        "reviews": [{"reviewer_id": r["reviewer_id"], "rating": r["rating"],
                     "confidence": r["confidence"], "soundness": r["soundness"],
                     "presentation": r["presentation"], "contribution": r["contribution"]}
                    for r in p["reviews"]],
        "ground_truth": {"weighted_mean_overall": p["weighted_mean_overall"],
                         "accept": p["accept"]},
    }, indent=2))
    (GT / f"{slug}.json").write_text(json.dumps({
        "paper_id": pid,
        "ground_truth": {"weighted_mean_overall": p["weighted_mean_overall"],
                         "accept": p["accept"]},
    }, indent=2))

    # Sanitize
    authors = client.get_note(pid).content.get("authors", {}).get("value", []) or []
    sanitize_pdf(pdf_path, SANI / f"{slug}.pdf", authors)
    issues = verify_pdf(SANI / f"{slug}.pdf", authors)
    return slug, len(issues), issues

print("\nDownloading + sanitizing (8 workers)...")
results = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(process, p): p for p in selection}
    for i, fut in enumerate(as_completed(futs), 1):
        slug, n_issues, issues = fut.result()
        status = "OK" if n_issues == 0 else f"FAIL({n_issues})"
        print(f"  [{i:>2}/{len(selection)}] {slug[:55]:<55} {status}")
        results.append((slug, n_issues, issues))

bad = [(s, n, iss) for s, n, iss in results if n > 0]
print(f"\nDone. {len(selection)} papers, {len(bad)} with sanitization issues.")
for s, n, iss in bad[:10]:
    print(f"  ! {s}: {iss[:3]}")
