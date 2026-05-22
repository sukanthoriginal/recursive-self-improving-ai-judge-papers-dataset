"""
Anonymize paper PDFs already present in local dataset folders.

Expected layout:
  <repo>/
    iclr-2025-dataset/papers/*.pdf
    neurips-2025-dataset-v2/papers/*.pdf

Default output:
  <dataset>/papers-anonymized/<original_stem>_anon.pdf

This script does not fetch anything from OpenReview. It only reads local PDFs,
redacts likely author/conference/decision-identifying content, and writes
anonymized copies into a separate folder inside each dataset.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


DEFAULT_DATASETS = ("iclr-2025-dataset", "neurips-2025-dataset-v2")

CONF_KEYWORDS = [
    "neurips",
    "icml",
    "iclr",
    "pmlr",
    "proceedings of the",
    "conference on",
    "published as a conference",
    "submitted to",
    "under review",
    "paper under double-blind review",
    "copyright 20",
    "do not distribute",
    "preliminary work",
    "international conference on machine learning",
    "international conference on learning representations",
    "neural information processing",
]

FOOTNOTE_AUTHOR_KEYWORDS = [
    "equal contribution",
    "work done",
    "project lead",
    "correspondence",
    "internship",
    "corresponding author",
    "author contribution",
    "@",
]

ANON_AUTHOR_PHRASES = [
    "anonymous author",
    "anonymous authors",
    "anonymous institution",
    "anonymous city",
    "anonymous region",
    "anonymous country",
    "affiliation\naddress",
    "affiliation address",
    "address\nemail",
    "address email",
]

LINE_NUMBER_RE = re.compile(r"^(\d{3}\s+){3,}", re.DOTALL)

REDACT_SECTION_HEADING_RE = re.compile(
    r"(acknowledg(?:e)?ments?|author\s+contributions?|"
    r"reproducibility\s+statement|broader\s+impact\s+statement|"
    r"impact\s+statement|ethics\s+statement)",
    re.IGNORECASE,
)

FUNDING_INLINE_RE = re.compile(
    r"^(this\s+work\s+(was|is)\s+(supported|funded|partially)|"
    r"we\s+(gratefully\s+)?acknowledge\s+(funding|support|the\s+support)|"
    r"the\s+authors?\s+(gratefully\s+)?(acknowledge|thank)|"
    r"this\s+(research|project)\s+(was|is)\s+(supported|funded|partially))",
    re.IGNORECASE,
)

HARD_STOP_HEADING_RE = re.compile(
    r"(^|\s)(references|appendix|supplementary\s+material|bibliography)(\s|$)",
    re.IGNORECASE,
)

GITHUB_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com)/\S+",
    re.IGNORECASE,
)

OPENREVIEW_FORUM_RE = re.compile(r"https?://(?:www\.)?openreview\.net/forum\?id=([^&#\s)]+)", re.IGNORECASE)
CODE_HOST_RE = re.compile(r"https?://(?:www\.)?(?:github\.com|gitlab\.com)/", re.IGNORECASE)
CODE_CONTEXT_RE = re.compile(
    r"\b(code|source|repository|repo|implementation|available|release|artifact|supplementary)\b",
    re.IGNORECASE,
)

BLANK_METADATA = {
    "title": "",
    "author": "",
    "subject": "",
    "keywords": "",
    "creator": "",
    "producer": "",
    "creationDate": "",
    "modDate": "",
    "trapped": "",
}


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    input_dir: Path
    output_dir: Path


def find_title_bottom(blocks: list[tuple]) -> float:
    text_blocks = [b for b in blocks if b[6] == 0]
    text_blocks.sort(key=lambda b: b[1])
    for b in text_blocks:
        text = b[4].strip().lower()
        if len(text) <= 10:
            continue
        if any(keyword in text for keyword in CONF_KEYWORDS):
            continue
        return b[3]
    return text_blocks[0][3] if text_blocks else 0


def find_abstract_top(blocks: list[tuple], page_height: float) -> float:
    for b in blocks:
        if b[6] != 0:
            continue
        if re.match(r"^Abstract\b", b[4].strip(), re.IGNORECASE):
            return b[1]
    return page_height * 0.45


def blocks_to_redact_page1(page: fitz.Page) -> list[fitz.Rect]:
    page_h = page.rect.height
    page_w = page.rect.width
    blocks = [(b[0], b[1], b[2], b[3], b[4], b[5], b[6]) for b in page.get_text("blocks")]

    title_bottom = find_title_bottom(blocks)
    abstract_top = find_abstract_top(blocks, page_h)

    rects: list[fitz.Rect] = []
    if abstract_top > title_bottom:
        rects.append(fitz.Rect(0, title_bottom, page_w, abstract_top))

    footnote_threshold = page_h * 0.75
    line_num_x1 = 0.0

    for x0, y0, x1, y1, text, _bno, btype in blocks:
        if btype != 0:
            continue

        lowered = text.lower()

        if any(keyword in lowered for keyword in CONF_KEYWORDS):
            rects.append(fitz.Rect(0, y0 - 2, page_w, y1 + 2))

        if y0 > footnote_threshold:
            is_keyword_footnote = any(keyword in lowered for keyword in FOOTNOTE_AUTHOR_KEYWORDS)
            is_numbered_affil = bool(re.match(r"^\d+[A-Z\u00C0-\u017F]", text.strip()))
            is_symbol_footnote = bool(re.match(r"^[*\u2020\u2021\u00A7\u00B6]", text.strip()))
            if is_keyword_footnote or is_numbered_affil or is_symbol_footnote:
                rects.append(fitz.Rect(0, y0 - 2, page_w, y1 + 2))

        if any(keyword in lowered for keyword in ANON_AUTHOR_PHRASES):
            rects.append(fitz.Rect(0, y0 - 2, page_w, y1 + 2))

        if LINE_NUMBER_RE.match(text):
            line_num_x1 = max(line_num_x1, x1)

    if line_num_x1 > 0:
        rects.append(fitz.Rect(0, 0, line_num_x1 + 4, page_h))

    return rects


def is_identifying_url(url: str) -> bool:
    return "anonymous" not in url.lower()


def load_paper_id(src_path: Path) -> str | None:
    meta_path = src_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")).get("paper_id")
    except Exception:
        return None


def link_context(page: fitz.Page, rect: fitz.Rect) -> str:
    clip = fitz.Rect(rect.x0 - 120, rect.y0 - 20, rect.x1 + 120, rect.y1 + 20) & page.rect
    return page.get_text("text", clip=clip)


def should_delete_link(url: str, context: str, paper_id: str | None) -> bool:
    """Remove links that would not belong in a double-blind submission."""
    if paper_id:
        match = OPENREVIEW_FORUM_RE.search(url)
        if match and match.group(1) == paper_id:
            return True

    # Anonymous artifact links are normal in double-blind submissions. Non-anonymous
    # code-host links are only removed when the local text suggests the link is this
    # paper's own code/artifact, not a citation to someone else's dependency or prior work.
    if CODE_HOST_RE.search(url) and is_identifying_url(url) and CODE_CONTEXT_RE.search(context):
        return True

    return False


def redact_sections_headers_and_urls(doc: fitz.Document) -> None:
    in_redact_section = False

    for pnum in range(len(doc)):
        page = doc[pnum]
        page_w = page.rect.width
        blocks = sorted(page.get_text("blocks"), key=lambda b: (round(b[1] / 5) * 5, b[0]))
        added_redaction = False

        for block in blocks:
            if block[6] != 0:
                continue

            x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
            stripped = text.strip()
            if not stripped:
                continue

            first_line = stripped.split("\n")[0]
            heading_search_text = stripped if stripped.count("\n") <= 3 else first_line

            if REDACT_SECTION_HEADING_RE.search(heading_search_text):
                in_redact_section = True
            elif not in_redact_section and FUNDING_INLINE_RE.match(stripped):
                in_redact_section = True
            elif in_redact_section and HARD_STOP_HEADING_RE.search(first_line):
                in_redact_section = False

            if in_redact_section:
                rect = fitz.Rect(0, y0 - 2, page_w, y1 + 2) & page.rect
                if not rect.is_empty:
                    page.add_redact_annot(rect, fill=(1, 1, 1))
                    added_redaction = True
                continue

            lowered = stripped.lower()
            if any(keyword in lowered for keyword in CONF_KEYWORDS):
                rect = fitz.Rect(0, y0 - 2, page_w, y1 + 2) & page.rect
                if not rect.is_empty:
                    page.add_redact_annot(rect, fill=(1, 1, 1))
                    added_redaction = True
                continue

            if GITHUB_URL_RE.search(text):
                precise_rects = []
                for match in GITHUB_URL_RE.finditer(text):
                    url = match.group(0)
                    if not is_identifying_url(url):
                        continue
                    precise_rects.extend(page.search_for(url, quads=False))

                if precise_rects:
                    for rect in precise_rects:
                        padded = fitz.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 1, rect.y1 + 1) & page.rect
                        if not padded.is_empty:
                            page.add_redact_annot(padded, fill=(1, 1, 1))
                            added_redaction = True
                elif any(is_identifying_url(match.group(0)) for match in GITHUB_URL_RE.finditer(text)):
                    rect = fitz.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1) & page.rect
                    if not rect.is_empty:
                        page.add_redact_annot(rect, fill=(1, 1, 1))
                        added_redaction = True

        if added_redaction:
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)


def anonymize_pdf(src_path: Path, out_path: Path) -> None:
    doc = fitz.open(str(src_path))
    try:
        if len(doc) == 0:
            raise ValueError("PDF has no pages")

        paper_id = load_paper_id(src_path)
        doc.set_metadata(BLANK_METADATA)
        if hasattr(doc, "del_xml_metadata"):
            doc.del_xml_metadata()

        page = doc[0]
        for rect in blocks_to_redact_page1(page):
            clipped = rect & page.rect
            if not clipped.is_empty:
                page.add_redact_annot(clipped, fill=(1, 1, 1))
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        redact_sections_headers_and_urls(doc)

        for page in doc:
            for link in list(page.get_links()):
                url = link.get("uri")
                rect = link.get("from")
                context = link_context(page, rect) if url and rect else ""
                if url and should_delete_link(url, context, paper_id):
                    page.delete_link(link)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_path), garbage=4, deflate=True, clean=True)
    finally:
        doc.close()


def discover_datasets(root: Path, dataset_names: list[str], input_dir: str, output_dir: str) -> list[DatasetPaths]:
    datasets = []
    for name in dataset_names:
        dataset_dir = root / name
        paper_dir = dataset_dir / input_dir
        if not paper_dir.is_dir():
            raise FileNotFoundError(f"Missing input folder: {paper_dir}")
        datasets.append(
            DatasetPaths(
                name=name,
                input_dir=paper_dir,
                output_dir=dataset_dir / output_dir,
            )
        )
    return datasets


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    datasets = discover_datasets(root, args.datasets, args.input_dir, args.output_dir)

    total = 0
    ok = 0
    failed = 0
    skipped = 0

    for dataset in datasets:
        pdfs = sorted(pdf for pdf in dataset.input_dir.glob("*.pdf") if not pdf.stem.endswith("_anon"))
        print(f"\n{dataset.name}: {len(pdfs)} PDFs")
        print(f"  input:  {dataset.input_dir}")
        print(f"  output: {dataset.output_dir}")

        for src in pdfs:
            total += 1
            out = dataset.output_dir / f"{src.stem}_anon.pdf"
            if out.exists() and not args.overwrite:
                skipped += 1
                print(f"  [skip] {out.name}")
                continue

            try:
                anonymize_pdf(src, out)
                ok += 1
                print(f"  [ok]   {src.name} -> {out.name}")
            except Exception as exc:
                failed += 1
                print(f"  [fail] {src.name}: {exc}")

    print(f"\nDone: {ok} anonymized, {skipped} skipped, {failed} failed, {total} total.")
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anonymize local dataset PDFs into per-dataset output folders.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository/dataset root containing the dataset folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset folder names to process.",
    )
    parser.add_argument("--input-dir", default="papers", help="Input PDF folder name inside each dataset.")
    parser.add_argument(
        "--output-dir",
        default="papers-anonymized",
        help="Output folder name inside each dataset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing anonymized PDFs.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
