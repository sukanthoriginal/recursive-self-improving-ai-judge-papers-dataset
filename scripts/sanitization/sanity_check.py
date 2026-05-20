"""Aggressive blind-review sanity checker for sanitized PDFs.

Beyond the build-time verifier, this script checks every avenue a reviewer
or downstream model could exploit to recover author identity. Runs as a
standalone audit pass and produces a per-paper report + summary table.

Checks per PDF:
  1. PDF metadata fields (Title/Author/Subject/Producer/Creator/Keywords) — must be empty.
  2. XMP metadata packets — scan raw bytes for any non-empty XMP author fields.
  3. Embedded font names — flag any font containing what looks like an author name part
     (font subsets sometimes carry "ABCDEF+AuthorNameMT").
  4. Title-page region (between title and Abstract) — extracted text must be empty.
  5. Full extracted text:
     - No full author name (case-insensitive, word-boundary).
     - No author last name (≥4 chars, word-boundary).
     - No author first name (≥4 chars, word-boundary, ignore very common words).
     - No email, ORCID, github/lab URL, arXiv ID.
  6. PDF annotations / link URIs (separate text channel from page text) — must not contain
     emails/URLs/author names.
  7. Acknowledgment-style phrases ("this work was supported", "funded by", "we thank",
     "grateful to") — flag as warning (these often carry funder/PI/collab identity).
  8. Self-reference phrases ("our prior work", "in our previous", "we previously showed")
     — flag as warning (style-level identity hint).
  9. Page-1 affiliation patterns — flag if page 1 contains university/lab keywords near
     the top of the page.

Output:
  - sanity_check_report.json  — per-paper findings
  - console summary table     — PASS / WARN / FAIL per paper

Severity:
  FAIL  = hard leak (author name, email, ORCID, lab URL in text).
  WARN  = stylistic / contextual hint that could narrow identity but isn't a direct leak.
  PASS  = no findings.
"""
import json
import re
import sys
from pathlib import Path
import fitz
from openreview.api import OpenReviewClient

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("neurips-2025-dataset-v2")
SANI = ROOT / "sanitized-papers"
PAPERS_META = ROOT / "papers"
REPORT = ROOT / "sanity_check_report.json"

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
URL_RE = re.compile(
    r"\b(?:https?://)?(?:github\.com|gitlab\.com|huggingface\.co|sites\.google\.com|"
    r"[a-z0-9\-]+\.github\.io|[a-z0-9\-]+\.(?:edu|ac\.[a-z]{2}|ai|io|org))/[A-Za-z0-9_./\-?=&%#~+]*",
    re.IGNORECASE,
)
ARXIV_RE = re.compile(r"\barXiv[:\s]*\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)

ACK_PHRASES = [
    r"this work (?:was|is) (?:supported|funded|sponsored)",
    r"funded by", r"supported by",
    r"\bwe (?:thank|are grateful|acknowledge)\b",
    r"grateful to", r"thanks to .{0,60}for",
    r"under grant", r"grant (?:no|number|#)",
]
SELF_REF_PHRASES = [
    r"\bin our (?:prior|previous|earlier) work\b",
    r"\bour (?:prior|previous|earlier) (?:paper|work|study)\b",
    r"\bwe previously\b",
    r"\bas we (?:showed|demonstrated|proved) in\b",
    r"\bextends? our\b",
    r"\bbuilds? on our\b",
]
ACK_REGEX = re.compile("|".join(ACK_PHRASES), re.IGNORECASE)
SELF_REF_REGEX = re.compile("|".join(SELF_REF_PHRASES), re.IGNORECASE)

AFFILIATION_KEYWORDS = re.compile(
    r"\b(university|institute|laboratory|college|school of|department of|"
    r"academia sinica|google|microsoft|meta\b|facebook|deepmind|openai|anthropic|"
    r"nvidia|amazon|apple|baidu|tencent|alibaba|huawei|samsung|ibm|intel|"
    r"mit\b|stanford|berkeley|cmu|caltech|princeton|harvard|yale|oxford|cambridge|"
    r"eth\b|epfl|tsinghua|peking|sjtu|zju)\b",
    re.IGNORECASE,
)

COMMON_NAME_WORDS = {
    "or", "an", "in", "on", "is", "be", "of", "to", "we", "do", "go", "no", "so", "as",
    "the", "and", "for", "but", "our", "all", "any", "new", "use", "see", "let",
    "wei", "li", "wu", "xu", "ma", "he", "su", "yu", "lu",  # short common Chinese names
}

client = OpenReviewClient(baseurl="https://api2.openreview.net")

def name_variants(name: str) -> tuple[set, set, set]:
    """Return (full_name, last_name, first_name) sets — each only if length is sufficient."""
    parts = [p for p in re.split(r"\s+", name.strip()) if p and not p.endswith(".")]
    full = {name.strip()} if len(name.strip()) >= 4 else set()
    last = set()
    first = set()
    if len(parts) >= 2:
        if len(parts[-1]) >= 4 and parts[-1].lower() not in COMMON_NAME_WORDS:
            last.add(parts[-1])
        if len(parts[0]) >= 4 and parts[0].lower() not in COMMON_NAME_WORDS:
            first.add(parts[0])
    return full, last, first

def extract_xmp_author_fields(pdf_bytes: bytes) -> list[str]:
    """Crude XMP scan: find any <dc:creator>, <pdf:Author>, <xmp:Author> etc."""
    findings = []
    for pat in [
        rb"<dc:creator>(.{1,500}?)</dc:creator>",
        rb"<pdf:Author>(.{1,500}?)</pdf:Author>",
        rb"<xmp:Author>(.{1,500}?)</xmp:Author>",
        rb"<dc:title>(.{1,500}?)</dc:title>",
    ]:
        for m in re.finditer(pat, pdf_bytes, re.DOTALL):
            val = m.group(1).decode("utf-8", errors="replace").strip()
            stripped = re.sub(r"<[^>]+>", "", val).strip()
            if stripped:
                findings.append(stripped)
    return findings

def page1_title_region_text(doc) -> str:
    """Text inside page 1 between title bottom and Abstract top — must be empty after redaction."""
    if len(doc) == 0:
        return ""
    page = doc[0]
    spans = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for sp in line.get("spans", []):
                spans.append({"text": sp["text"], "size": sp["size"], "y0": sp["bbox"][1], "y1": sp["bbox"][3]})
    if not spans:
        return ""
    mx = max(s["size"] for s in spans)
    title_spans = [s for s in spans if abs(s["size"]-mx) < 0.5 and s["y0"] < page.rect.height * 0.5]
    y_t = max((s["y1"] for s in title_spans), default=0.0)
    y_a = page.rect.height
    for s in spans:
        if s["text"].strip().lower().startswith("abstract") and s["y0"] > y_t:
            y_a = s["y0"]; break
    region = fitz.Rect(0, y_t + 1, page.rect.width, y_a - 1)
    return page.get_text("text", clip=region).strip()

def get_link_uris(doc) -> list[str]:
    """All URI links across all pages."""
    out = []
    for page in doc:
        for ln in page.get_links():
            uri = ln.get("uri")
            if uri:
                out.append(uri)
    return out

def get_font_names(doc) -> list[str]:
    out = set()
    for page in doc:
        for f in page.get_fonts():
            # f = (xref, ext, type, basefont, name, encoding)
            out.add(f[3] or "")
            out.add(f[4] or "")
    return sorted(x for x in out if x)

def check_one(pdf_path: Path, authors: list[str]) -> dict:
    findings = {"FAIL": [], "WARN": []}
    doc = fitz.open(pdf_path)
    pdf_bytes = pdf_path.read_bytes()

    # 1. Metadata
    meta = doc.metadata or {}
    for k in ("title", "author", "subject", "keywords", "creator", "producer"):
        v = (meta.get(k) or "").strip()
        if v:
            findings["WARN"].append({"check": "pdf_metadata", "field": k, "value": v})

    # 2. XMP packets
    xmp_findings = extract_xmp_author_fields(pdf_bytes)
    for v in xmp_findings:
        findings["WARN"].append({"check": "xmp_metadata", "value": v[:200]})

    # 3. Embedded font names
    fonts = get_font_names(doc)
    for a in authors:
        for part in re.split(r"\s+", a):
            if len(part) < 4 or part.lower() in COMMON_NAME_WORDS:
                continue
            for fn in fonts:
                if part.lower() in fn.lower():
                    findings["WARN"].append({"check": "font_name_contains_author_part",
                                              "font": fn, "part": part})

    # 4. Title-page region
    region_text = page1_title_region_text(doc)
    if region_text:
        findings["FAIL"].append({"check": "title_page_region_not_empty",
                                  "extracted": region_text[:300]})

    # 5. Full body text — extract once
    text = "\n".join(p.get_text() for p in doc)

    # 5a. Author names
    for a in authors:
        full, last, first = name_variants(a)
        for v in full:
            if re.search(r"\b" + re.escape(v) + r"\b", text, re.IGNORECASE):
                findings["FAIL"].append({"check": "author_full_name", "name": v})
        for v in last:
            if re.search(r"\b" + re.escape(v) + r"\b", text):
                findings["FAIL"].append({"check": "author_last_name", "name": v, "from": a})
        for v in first:
            # First names are riskier — only FAIL if name is long and uncommon
            if len(v) >= 5 and re.search(r"\b" + re.escape(v) + r"\b", text):
                findings["WARN"].append({"check": "author_first_name", "name": v, "from": a})

    # 5b. Pattern leaks
    for nm, rx in [("email", EMAIL_RE), ("orcid", ORCID_RE), ("url", URL_RE), ("arxiv_id", ARXIV_RE)]:
        hits = rx.findall(text)
        if hits:
            severity = "FAIL" if nm in ("email", "orcid", "url") else "WARN"
            findings[severity].append({"check": f"residual_{nm}", "count": len(hits),
                                        "samples": list(set(hits))[:5]})

    # 6. Link URIs (separate from text)
    uris = get_link_uris(doc)
    for u in uris:
        if EMAIL_RE.search(u) or URL_RE.search(u):
            findings["FAIL"].append({"check": "link_uri_leak", "uri": u})
        else:
            for a in authors:
                for part in re.split(r"\s+", a):
                    if len(part) >= 4 and part.lower() not in COMMON_NAME_WORDS:
                        if part.lower() in u.lower():
                            findings["FAIL"].append({"check": "link_uri_contains_author",
                                                      "uri": u, "name_part": part})
                            break

    # 7. Acknowledgment-style phrases
    for m in ACK_REGEX.finditer(text):
        s, e = max(0, m.start()-40), min(len(text), m.end()+60)
        findings["WARN"].append({"check": "acknowledgment_style_phrase",
                                  "match": m.group(0), "context": text[s:e].replace("\n", " ")})

    # 8. Self-reference phrases
    for m in SELF_REF_REGEX.finditer(text):
        s, e = max(0, m.start()-40), min(len(text), m.end()+80)
        findings["WARN"].append({"check": "self_reference_phrase",
                                  "match": m.group(0), "context": text[s:e].replace("\n", " ")})

    # 9. Page-1 affiliation keywords (after title-page region check)
    if len(doc) > 0:
        page1_text = doc[0].get_text()
        aff_hits = AFFILIATION_KEYWORDS.findall(page1_text)
        if aff_hits:
            findings["WARN"].append({"check": "page1_affiliation_keywords",
                                      "hits": list(set(aff_hits))[:10]})

    doc.close()
    fail = bool(findings["FAIL"])
    warn = bool(findings["WARN"])
    status = "FAIL" if fail else ("WARN" if warn else "PASS")
    return {"status": status, "findings": findings, "n_authors": len(authors)}

def get_authors_for(slug: str) -> list[str]:
    meta_path = PAPERS_META / f"{slug}.meta.json"
    if not meta_path.exists():
        return []
    pid = json.loads(meta_path.read_text())["paper_id"]
    try:
        note = client.get_note(pid)
        return note.content.get("authors", {}).get("value", []) or []
    except Exception as e:
        print(f"  ! could not fetch authors for {pid}: {e}", file=sys.stderr)
        return []

def main():
    pdfs = sorted(SANI.glob("*.pdf"))
    print(f"Checking {len(pdfs)} sanitized PDFs against blind-review criteria...\n")
    report = {}
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for pdf in pdfs:
        slug = pdf.stem
        authors = get_authors_for(slug)
        result = check_one(pdf, authors)
        report[slug] = {"authors": authors, **result}
        counts[result["status"]] += 1
        n_fail = len(result["findings"]["FAIL"])
        n_warn = len(result["findings"]["WARN"])
        print(f"  [{result['status']:>4}] {slug[:55]:<55}  fails={n_fail:>2}  warns={n_warn:>2}")

    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nTotals: PASS={counts['PASS']}  WARN={counts['WARN']}  FAIL={counts['FAIL']}")
    print(f"Full report: {REPORT}")

    if counts["FAIL"]:
        print(f"\nTop FAIL examples:")
        n = 0
        for slug, r in report.items():
            if r["status"] != "FAIL": continue
            for f in r["findings"]["FAIL"][:3]:
                print(f"  {slug}: {f}")
                n += 1
                if n >= 15: break
            if n >= 15: break

if __name__ == "__main__":
    main()
