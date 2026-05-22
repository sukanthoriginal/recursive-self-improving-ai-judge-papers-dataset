"""Run a baseline accept/reject evaluation on a random 50-paper sample.

This is the evaluation-only part of run_experiment.py: no evolution, no meta-review
updates, and one model call per selected paper. Results are logged immediately to
JSONL and can be resumed.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import json
import random
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import boto3
import fitz
from botocore.config import Config as BotocoreConfig
from dotenv import dotenv_values
from tqdm import tqdm


EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
PAPERS_CSV = EVALS_DIR / "data" / "papers.csv"
EVAL_PROMPT_FILE = EVALS_DIR / "EVALUATION_PROMPT.md"
DEFAULT_RESULTS_FILE = EVALS_DIR / "baseline_results.jsonl"
ENV_FILES = (EVALS_DIR / ".env", REPO_ROOT / ".env")

MAX_TOKENS = 4096
MAX_PDF_PAGES = 100
BEDROCK_PROFILE = "Shourya_Jain"
BEDROCK_REGION = "us-east-1"
RETRY_DELAY = 5.0
POLL_INTERVAL = 0.3

_stop = threading.Event()
_log_lock = threading.Lock()
_client_local = threading.local()


def load_env() -> tuple[dict, Path | None]:
    env = {}
    env_file = None
    for candidate in ENV_FILES:
        if candidate.exists():
            env.update(dotenv_values(candidate))
            env_file = candidate
    return env, env_file


_env, _env_file = load_env()
MODEL_ARN = _env.get("ANTHROPIC_MODEL", "")
if not MODEL_ARN:
    searched = ", ".join(str(path) for path in ENV_FILES)
    sys.exit(f"ANTHROPIC_MODEL not found. Searched: {searched}")


class StopRequested(Exception):
    """Raised when _stop is set while waiting for an API call."""


def get_client():
    if not hasattr(_client_local, "client"):
        session = boto3.Session(profile_name=BEDROCK_PROFILE, region_name=BEDROCK_REGION)
        _client_local.client = session.client(
            "bedrock-runtime",
            config=BotocoreConfig(
                read_timeout=300,
                connect_timeout=10,
                retries={"max_attempts": 0},
            ),
        )
    return _client_local.client


def get_pdf_b64(pdf_path: Path) -> str:
    """Return base64 PDF, truncated to the Bedrock page limit if needed."""
    doc = fitz.open(str(pdf_path))
    try:
        if len(doc) > MAX_PDF_PAGES:
            trunc = fitz.open()
            trunc.insert_pdf(doc, from_page=0, to_page=MAX_PDF_PAGES - 1)
            data = trunc.tobytes(garbage=1, deflate=True)
            trunc.close()
        else:
            data = pdf_path.read_bytes()
    finally:
        doc.close()
    return base64.standard_b64encode(data).decode()


def invoke_blocking(body: str) -> dict:
    resp = get_client().invoke_model(
        modelId=MODEL_ARN,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(resp["body"].read())
    usage = result.get("usage", {}) or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = None
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "text": result["content"][0]["text"].strip(),
        "usage": usage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def call_model(system: str, pdf_path: Path, user_text: str, label: str = "") -> dict:
    if _stop.is_set():
        raise StopRequested()

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": get_pdf_b64(pdf_path),
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        }
    )

    for api_attempt in range(1, 4):
        if _stop.is_set():
            raise StopRequested()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(invoke_blocking, body)
            while True:
                try:
                    return fut.result(timeout=POLL_INTERVAL)
                except concurrent.futures.TimeoutError:
                    if _stop.is_set():
                        fut.cancel()
                        raise StopRequested()
                except Exception as exc:
                    if api_attempt == 3:
                        raise
                    wait = RETRY_DELAY * api_attempt
                    tqdm.write(f"  [API retry {api_attempt}/3] {label}: {type(exc).__name__}; waiting {wait}s")
                    deadline = time.monotonic() + wait
                    while time.monotonic() < deadline:
                        if _stop.is_set():
                            raise StopRequested()
                        time.sleep(min(POLL_INTERVAL, deadline - time.monotonic()))
                    break
    return {"text": "", "usage": {}, "input_tokens": None, "output_tokens": None, "total_tokens": None}


def xml_text(root, tag: str, default: str = "") -> str:
    el = root.find(tag)
    return (el.text or "").strip() if el is not None else default


def xml_list(root, tag: str, child: str) -> list[str]:
    el = root.find(tag)
    if el is None:
        return []
    return [(item.text or "").strip() for item in el.findall(child)]


def parse_eval_response(xml_str: str) -> dict:
    try:
        xml_str = re.sub(r"```xml\s*", "", xml_str)
        xml_str = re.sub(r"```\s*", "", xml_str)
        match = re.search(r"<review>.*?</review>", xml_str, re.DOTALL)
        if not match:
            pred_match = re.search(
                r"<locked_prediction>\s*(ACCEPT|REJECT)\s*</locked_prediction>",
                xml_str,
                re.IGNORECASE,
            )
            score_match = re.search(r"<overall_score>\s*([^<]+?)\s*</overall_score>", xml_str, re.IGNORECASE)
            percentile_match = re.search(r"<paper_percentile>\s*([^<]+?)\s*</paper_percentile>", xml_str, re.IGNORECASE)
            return {
                "overall_score": score_match.group(1).strip() if score_match else None,
                "locked_prediction": pred_match.group(1).upper() if pred_match else None,
                "paper_percentile": percentile_match.group(1).strip() if percentile_match else None,
                "parse_error": "no <review> block",
            }
        root = ET.fromstring(match.group(0))
        return {
            "overall_score": xml_text(root, "overall_score"),
            "score_confidence": xml_text(root, "score_confidence"),
            "locked_prediction": xml_text(root, "locked_prediction").upper(),
            "confidence": xml_text(root, "confidence"),
            "paper_percentile": xml_text(root, "paper_percentile"),
            "paper_type": xml_text(root, "paper_type"),
            "central_claim": xml_text(root, "central_claim"),
            "claim_evidence_match": xml_text(root, "claim_evidence_match"),
            "decisive_reasons": xml_list(root, "decisive_reasons", "reason"),
            "main_concerns": xml_list(root, "main_concerns", "concern"),
            "meta_lessons_applied": xml_list(root, "meta_lessons_applied", "lesson"),
            "would_flip_if": xml_list(root, "would_flip_if", "condition"),
            "parse_error": None,
        }
    except Exception as exc:
        pred_match = re.search(
            r"<locked_prediction>\s*(ACCEPT|REJECT)\s*</locked_prediction>",
            xml_str,
            re.IGNORECASE,
        )
        score_match = re.search(r"<overall_score>\s*([^<]+?)\s*</overall_score>", xml_str, re.IGNORECASE)
        percentile_match = re.search(r"<paper_percentile>\s*([^<]+?)\s*</paper_percentile>", xml_str, re.IGNORECASE)
        return {
            "overall_score": score_match.group(1).strip() if score_match else None,
            "locked_prediction": pred_match.group(1).upper() if pred_match else None,
            "paper_percentile": percentile_match.group(1).strip() if percentile_match else None,
            "parse_error": str(exc),
        }


def load_papers() -> list[dict]:
    with PAPERS_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_sample(papers: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size > len(papers):
        raise ValueError(f"sample_size={sample_size} exceeds available papers={len(papers)}")
    rng = random.Random(seed)
    return rng.sample(papers, sample_size)


def eval_system_prompt() -> str:
    return EVAL_PROMPT_FILE.read_text(encoding="utf-8").strip()


def eval_user_text() -> str:
    return "Please review this paper and provide your prediction in the XML format specified."


def log(results_file: Path, record: dict) -> None:
    record["timestamp"] = datetime.utcnow().isoformat()
    with _log_lock:
        with results_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_predictions(results_file: Path) -> dict[str, dict]:
    if not results_file.exists():
        return {}
    completed = {}
    with results_file.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "baseline_prediction":
                completed[record.get("paper_id", "")] = record
    return completed


def log_error(results_file: Path, paper: dict, exc: BaseException) -> dict:
    record = {
        "event": "baseline_error",
        "paper_id": paper["paper_id"],
        "title": paper["title"],
        "conference": paper["conference"],
        "path": paper["path"],
        "error_type": type(exc).__name__,
        "error": str(exc),
        "model": MODEL_ARN,
    }
    log(results_file, record)
    return record


def predict_paper(paper: dict, system_prompt: str, results_file: Path) -> dict:
    pdf_path = REPO_ROOT / paper["path"]
    ground_truth = paper["accept_reject"].upper()
    label = f"[baseline][{paper['title'][:40]}]"

    model_result = call_model(system_prompt, pdf_path, eval_user_text(), label)
    raw_response = model_result["text"]
    parsed = parse_eval_response(raw_response)
    prediction = parsed.get("locked_prediction")
    correct = (prediction == ground_truth) if prediction else None

    record = {
        "event": "baseline_prediction",
        "paper_id": paper["paper_id"],
        "title": paper["title"],
        "conference": paper["conference"],
        "path": paper["path"],
        "accept_reject": paper["accept_reject"],
        "accept_type": paper.get("accept_type", ""),
        "ground_truth": ground_truth,
        "prediction": prediction,
        "correct": correct,
        "model": MODEL_ARN,
        "raw_response": raw_response,
        "input_tokens": model_result.get("input_tokens"),
        "output_tokens": model_result.get("output_tokens"),
        "total_tokens": model_result.get("total_tokens"),
        "token_usage": model_result.get("usage", {}),
        "weighted_mean_overall": paper.get("weighted_mean_overall"),
        "score_z_within_venue": paper.get("score_z_within_venue"),
        "score_z_1_10_clipped": paper.get("score_z_1_10_clipped"),
        "score_percentile_within_venue": paper.get("score_percentile_within_venue"),
        "score_percentile_1_10": paper.get("score_percentile_1_10"),
        "normalization_pool_n": paper.get("normalization_pool_n"),
        **parsed,
    }
    log(results_file, record)
    return record


def summarize(records: list[dict]) -> dict:
    total = len(records)
    correct = sum(1 for record in records if record.get("correct") is True)
    parsed = sum(1 for record in records if record.get("prediction") in {"ACCEPT", "REJECT"})
    input_tokens = sum(int(record.get("input_tokens") or 0) for record in records)
    output_tokens = sum(int(record.get("output_tokens") or 0) for record in records)
    total_tokens = sum(int(record.get("total_tokens") or 0) for record in records)
    by_conference = {}
    for record in records:
        conf = record.get("conference", "")
        item = by_conference.setdefault(conf, {"total": 0, "correct": 0})
        item["total"] += 1
        item["correct"] += int(record.get("correct") is True)
    for item in by_conference.values():
        item["accuracy"] = item["correct"] / item["total"] if item["total"] else 0.0
    return {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "by_conference": by_conference,
    }


def print_sample(sample: list[dict]) -> None:
    accepts = sum(1 for paper in sample if paper["accept_reject"] == "accept")
    rejects = len(sample) - accepts
    by_conf = {}
    for paper in sample:
        item = by_conf.setdefault(paper["conference"], {"total": 0, "accept": 0, "reject": 0})
        item["total"] += 1
        item[paper["accept_reject"]] += 1
    print(f"Sample: {len(sample)} papers ({accepts} accept, {rejects} reject)")
    print(json.dumps(by_conf, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline Bedrock evaluation on a random paper sample.")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of papers to sample from papers.csv.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent Bedrock API calls.")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE, help="JSONL output path.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected sample without calling Bedrock.")
    return parser.parse_args()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    results_file = args.results_file if args.results_file.is_absolute() else REPO_ROOT / args.results_file

    papers = load_papers()
    sample = select_sample(papers, args.sample_size, args.seed)
    print(f"Model : {MODEL_ARN}")
    print(f"Env   : {_env_file}")
    print(f"Output: {results_file}")
    print_sample(sample)

    if args.dry_run:
        for idx, paper in enumerate(sample, 1):
            print(f"  [{idx:02d}] [{paper['conference']}] [{paper['accept_reject']}] {paper['title'][:80]}")
        return 0

    results_file.parent.mkdir(parents=True, exist_ok=True)
    log(
        results_file,
        {
            "event": "baseline_start",
            "sample_size": args.sample_size,
            "seed": args.seed,
            "workers": args.workers,
            "model": MODEL_ARN,
            "sample_paper_ids": [paper["paper_id"] for paper in sample],
        },
    )

    done = completed_predictions(results_file)
    sample_to_run = [paper for paper in sample if paper["paper_id"] not in done]
    if done:
        print(f"Resume: {len(sample) - len(sample_to_run)} selected papers already logged.")

    system_prompt = eval_system_prompt()
    new_records = []
    interrupted = False

    pbar = tqdm(total=len(sample), initial=len(sample) - len(sample_to_run), desc="Baseline", unit="paper")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="baseline") as ex:
            futures = {ex.submit(predict_paper, paper, system_prompt, results_file): paper for paper in sample_to_run}
            for fut in concurrent.futures.as_completed(futures):
                paper = futures[fut]
                try:
                    record = fut.result()
                except Exception as exc:
                    log_error(results_file, paper, exc)
                    tqdm.write(
                        f"  ERROR [{paper['conference']}] {paper['title'][:55]} "
                        f"| {type(exc).__name__}: {exc}"
                    )
                    pbar.update(1)
                    continue
                new_records.append(record)
                status = "OK" if record.get("correct") else "MISS"
                tqdm.write(
                    f"  {status} [{paper['conference']}] {paper['title'][:55]} "
                    f"| pred={record.get('prediction')} gt={record.get('ground_truth')}"
                )
                pbar.update(1)
    except KeyboardInterrupt:
        _stop.set()
        interrupted = True
        for fut in futures:
            fut.cancel()
        tqdm.write("\nCtrl+C received. Progress already logged; re-run to resume.")
    except StopRequested:
        interrupted = True
        tqdm.write("\nStopped cleanly. Progress already logged; re-run to resume.")
    finally:
        pbar.close()

    final_records_by_id = completed_predictions(results_file)
    selected_records = [final_records_by_id[paper["paper_id"]] for paper in sample if paper["paper_id"] in final_records_by_id]
    summary = summarize(selected_records)
    log(
        results_file,
        {
            "event": "baseline_summary",
            "sample_size": args.sample_size,
            "seed": args.seed,
            "interrupted": interrupted,
            **summary,
        },
    )

    print(f"\nCompleted: {len(selected_records)}/{len(sample)}")
    print(f"Accuracy : {summary['correct']}/{summary['total']} = {summary['accuracy']:.1%}" if summary["total"] else "Accuracy : n/a")
    print(
        f"Tokens   : input={summary['input_tokens']} "
        f"output={summary['output_tokens']} total={summary['total_tokens']}"
    )
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
