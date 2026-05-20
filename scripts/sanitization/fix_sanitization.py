"""Post-process pass over v2 sanitized PDFs to fix what the sanity checker found.

Fixes applied per PDF:
  1. Strip all link annotations (the clickable URI metadata under hyperlinks survived
     the original text-only redaction). Removes the embedded URI even if the rendered
     text was already redacted to black.
  2. Re-redact any residual emails / ORCIDs / lab-URL strings that the original pass
     missed (formatting edge cases where the regex match crossed PDF span boundaries).
  3. Strip any author-name substring (≥4 chars, not in COMMON_NAME_WORDS) that
     survived in the text — second sweep with the same name-variant logic.

Idempotent: safe to re-run. Writes back to neurips-2025-dataset-v2/sanitized-papers/.
"""
import json
import re
from pathlib import Path
import fitz
from openreview.api import OpenReviewClient

ROOT = Path("neurips-2025-dataset-v2")
SANI = ROOT / "sanitized-papers"
PAPERS_META = ROOT / "papers"

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
URL_RE = re.compile(
    r"\b(?:https?://)?(?:github\.com|gitlab\.com|huggingface\.co|sites\.google\.com|"
    r"[a-z0-9\-]+\.github\.io|[a-z0-9\-]+\.(?:edu|ac\.[a-z]{2}|ai|io|org))/[A-Za-z0-9_./\-?=&%#~+]*",
    re.IGNORECASE,
)
COMMON_NAME_WORDS = {
    "or", "an", "in", "on", "is", "be", "of", "to", "we", "do", "go", "no", "so", "as",
    "the", "and", "for", "but", "our", "all", "any", "new", "use", "see", "let",
    "wei", "li", "wu", "xu", "ma", "he", "su", "yu", "lu",
}

client = OpenReviewClient(baseurl="https://api2.openreview.net")

def name_variants(name: str):
    parts = [p for p in re.split(r"\s+", name.strip()) if p and not p.endswith(".")]
    out = set()
    if len(name.strip()) >= 4:
        out.add(name.strip())
    if len(parts) >= 2:
        if len(parts[-1]) >= 4 and parts[-1].lower() not in COMMON_NAME_WORDS:
            out.add(parts[-1])
        if len(parts[0]) >= 4 and parts[0].lower() not in COMMON_NAME_WORDS:
            out.add(parts[0])
    return out

def fix_one(pdf_path: Path, authors: list[str]) -> dict:
    stats = {"links_stripped": 0, "residual_email": 0, "residual_url": 0,
             "residual_orcid": 0, "residual_name": 0}
    doc = fitz.open(pdf_path)

    # 1. Strip all link annotations on every page
    for page in doc:
        links = list(page.get_links())
        for ln in links:
            try:
                page.delete_link(ln)
                stats["links_stripped"] += 1
            except Exception:
                pass

    # 2/3. Re-sweep: redact any residual emails/ORCIDs/URLs and author-name strings
    for page in doc:
        text = page.get_text()
        for nm, rx, key in [("email", EMAIL_RE, "residual_email"),
                            ("orcid", ORCID_RE, "residual_orcid"),
                            ("url", URL_RE, "residual_url")]:
            for m in rx.finditer(text):
                for q in page.search_for(m.group(0), quads=True):
                    page.add_redact_annot(q.rect, fill=(0, 0, 0))
                    stats[key] += 1
        for a in authors:
            for v in name_variants(a):
                for q in page.search_for(v, quads=True):
                    page.add_redact_annot(q.rect, fill=(0, 0, 0))
                    stats["residual_name"] += 1

    for page in doc:
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    # Strip metadata again (in case any survived)
    doc.set_metadata({})

    # Atomic write
    tmp = pdf_path.with_suffix(".pdf.tmp")
    doc.save(tmp, garbage=4, deflate=True, clean=True)
    doc.close()
    tmp.replace(pdf_path)
    return stats

def get_authors_for(slug: str) -> list[str]:
    meta = json.loads((PAPERS_META / f"{slug}.meta.json").read_text())
    note = client.get_note(meta["paper_id"])
    return note.content.get("authors", {}).get("value", []) or []

def main():
    pdfs = sorted(SANI.glob("*.pdf"))
    print(f"Fixing {len(pdfs)} sanitized PDFs...\n")
    summary = {}
    for pdf in pdfs:
        slug = pdf.stem
        authors = get_authors_for(slug)
        stats = fix_one(pdf, authors)
        summary[slug] = stats
        print(f"  {slug[:55]:<55}  "
              f"links={stats['links_stripped']:>3}  "
              f"email={stats['residual_email']:>2}  "
              f"url={stats['residual_url']:>2}  "
              f"name={stats['residual_name']:>3}")
    (ROOT / "fix_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. summary: {ROOT / 'fix_summary.json'}")

if __name__ == "__main__":
    main()
