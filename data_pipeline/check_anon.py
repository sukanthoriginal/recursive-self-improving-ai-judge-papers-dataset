"""
Check anonymized dataset PDFs for likely remaining identifying information.

By default this scans:
  iclr-2025-dataset/papers-anonymized/*.pdf
  neurips-2025-dataset-v2/papers-anonymized/*.pdf

The checks are intentionally conservative. Some findings can be false positives,
especially institution words or grant-like phrases in citations or ordinary text.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import fitz


DEFAULT_DATASETS = ("iclr-2025-dataset", "neurips-2025-dataset-v2")

CHECKS = [
    ("GitHub/GitLab URL", re.compile(r"https?://(www\.)?(github|gitlab)\.com/\S+", re.I)),
    (
        "Grant number",
        re.compile(
            r"\b(NSF|NIH|ERC|ANR|MOE|NRF|AISG|GENCI|EPSRC|DFG|ARC|NSERC|SNF|FNR|SERI|CIFAR|ONR|DARPA|AFOSR|ARO)\b.*?\b\d{3,}",
            re.I,
        ),
    ),
    ("Grant award pattern", re.compile(r"\b(grant|award|contract|fellowship|programme|allocation)\s+(no\.?|number|#)?\s*[A-Z0-9-]{4,}", re.I)),
    ("Author initial supported", re.compile(r"\b[A-Z]\.[A-Z]\.?\s+(is|was|were)\s+(supported|funded|partially)", re.I)),
    ("We thank <Name>", re.compile(r"\bwe\s+thank\s+[A-Z][a-z]+\s+[A-Z][a-z]+", re.I)),
    ("Acknowledgement heading", re.compile(r"acknowledg(?:e)?ments?", re.I)),
    ("Author contributions", re.compile(r"author\s+contributions?", re.I)),
    ("Reproducibility stmt", re.compile(r"reproducibility\s+statement", re.I)),
    (
        "University name",
        re.compile(
            r"\b(university|universit\u00E9|universit\u00E4t|instituto|institute\s+of\s+technology|ETH|MIT|CMU|Stanford|Oxford|Cambridge|NTU|NUS|IISc|INRIA|EPFL|Mila|Vector\s+Institute)\b",
            re.I,
        ),
    ),
    ("Personal email", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.I)),
    ("Lab name leak", re.compile(r"\b(lab|laboratory|group|team)\b.{0,40}\b(at|@)\b.{0,40}\b[A-Z]{2,}", re.I)),
]

HEADER_CHECKS = [
    ("Conference header", re.compile(r"\b(iclr|icml|neurips|pmlr|under review|published as a conference|proceedings of|conference on|copyright 20|do not distribute)\b", re.I)),
]

REFERENCES_RE = re.compile(r"^references\s*$", re.I)
LOW_SIGNAL_IN_REFERENCES = {
    "University name",
    "Personal email",
    "Lab name leak",
    "We thank <Name>",
}


def scan_pdf(path: Path) -> list[dict]:
    doc = fitz.open(str(path))
    findings = []
    in_references = False

    try:
        for pnum in range(len(doc)):
            page = doc[pnum]
            blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))

            for block in blocks:
                if block[6] != 0:
                    continue

                x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4].strip()
                if not text:
                    continue

                first_line = text.split("\n")[0].strip()
                if REFERENCES_RE.match(first_line):
                    in_references = True

                checks = list(CHECKS)
                if y0 <= 60:
                    checks.extend(HEADER_CHECKS)

                for check_name, pattern in checks:
                    for match in pattern.finditer(text):
                        if in_references and check_name in LOW_SIGNAL_IN_REFERENCES:
                            continue
                        snippet = text[max(0, match.start() - 40) : match.end() + 40].replace("\n", " ").strip()
                        findings.append(
                            {
                                "page": pnum + 1,
                                "check": check_name,
                                "match": match.group(0)[:120],
                                "context": snippet[:220],
                                "in_references": in_references,
                            }
                        )
    finally:
        doc.close()

    return findings


def scan_dataset(dataset_dir: Path, output_dir: str) -> tuple[int, int, list[str]]:
    pdf_dir = dataset_dir / output_dir
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"Missing anonymized folder: {pdf_dir}")

    pdfs = sorted(pdf_dir.glob("*_anon.pdf"))
    lines = [f"\n{dataset_dir.name}: {len(pdfs)} anonymized PDFs"]
    flagged_pdfs = 0
    total_findings = 0

    for pdf in pdfs:
        findings = scan_pdf(pdf)
        if not findings:
            lines.append(f"  [OK] {pdf.name}")
            continue

        flagged_pdfs += 1
        total_findings += len(findings)
        lines.append(f"\n  [!!] {pdf.name}")
        for finding in findings:
            ref_tag = " [in-refs]" if finding["in_references"] else ""
            lines.append(f"       p{finding['page']:>3}  {finding['check']:<28}{ref_tag}")
            lines.append(f"            match:   {finding['match']}")
            lines.append(f"            context: {finding['context']}")

    lines.append(f"\n  Summary: {flagged_pdfs}/{len(pdfs)} PDFs flagged, {total_findings} findings.")
    return flagged_pdfs, total_findings, lines


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    all_lines = []
    total_flagged = 0
    total_findings = 0

    for dataset_name in args.datasets:
        flagged, findings, lines = scan_dataset(root / dataset_name, args.output_dir)
        total_flagged += flagged
        total_findings += findings
        all_lines.extend(lines)

    all_lines.append(f"\nDone: {total_flagged} PDFs flagged, {total_findings} total findings.")
    report = "\n".join(all_lines)
    print(report)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n", encoding="utf-8")
        print(f"\nReport written to {args.report}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan anonymized dataset PDFs for likely identity leaks.")
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
        help="Dataset folder names to scan.",
    )
    parser.add_argument(
        "--output-dir",
        default="papers-anonymized",
        help="Anonymized PDF folder name inside each dataset.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "check_anon_report.txt",
        help="Path to write the text report. Use --report '' to skip.",
    )
    args = parser.parse_args()
    if isinstance(args.report, Path) and str(args.report) == ".":
        args.report = None
    return args


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(run(parse_args()))
