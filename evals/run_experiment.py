"""
Meta-reviewer self-evolution experiment.

Split (fixed seed=42):
  - Evolution set : 20 papers (iclr2025 + neurips2025, 5 accept + 5 reject each conf)
  - Test set      : 20 papers (iclr2025 + neurips2025, 5 accept + 5 reject each conf)

Evolution loop per paper:
  - Predict → if correct, move on
  - If wrong: update meta review → retry (max 3 attempts total)
  - After 3 failed attempts: defer paper to second pass
  - 3 consecutive deferred papers in a row → end evolution early

Test checkpoints: after every 5 evolution papers (after_evo_5, 10, 15, 20)
Baseline test runs once at the start (before any evolution).
No separate "final" test — after_evo_20 IS the final.

Resume: on restart, reads results.jsonl to skip already-completed work.
Ctrl+C: safe at any point — all results are logged immediately to results.jsonl.

Model: Claude via AWS Bedrock (ARN from evals/.env or repo .env ANTHROPIC_MODEL)
PDF delivery: base64 document block (figures/tables/equations fully visible)
"""

from __future__ import annotations

import base64
import concurrent.futures
import csv
import json
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import boto3
import fitz  # PyMuPDF — used for PDF page-count truncation
from botocore.config import Config as BotocoreConfig
from dotenv import dotenv_values
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

EVALS_DIR        = Path(__file__).resolve().parent
REPO_ROOT        = EVALS_DIR.parent
PAPERS_CSV       = EVALS_DIR / "data" / "papers.csv"
META_REVIEW_FILE = EVALS_DIR / "meta_review.txt"
EVAL_PROMPT_FILE = EVALS_DIR / "EVALUATION_PROMPT.md"
META_PROMPT_FILE = EVALS_DIR / "META_REVIEW_PROMPT.md"
RESULTS_FILE     = EVALS_DIR / "results.jsonl"
ENV_FILES        = (EVALS_DIR / ".env", REPO_ROOT / ".env")

# ── Config ────────────────────────────────────────────────────────────────────

EVOLUTION_SIZE    = 20
TEST_SIZE         = 20
TEST_EVERY_N      = 5
MAX_RETRIES       = 3
CONSEC_FAIL_LIMIT = 3
MAX_TOKENS        = 4096
BEDROCK_PROFILE   = "Shourya_Jain"
BEDROCK_REGION    = "us-east-1"
RETRY_DELAY       = 5.0
TEST_WORKERS      = 5    # parallel API calls during test phases
POLL_INTERVAL     = 0.3  # seconds between polls when waiting for a blocking API call
CONFERENCES       = ("iclr2025", "neurips2025")

# Global stop flag — set on Ctrl+C so all threads can exit cleanly.
_stop = threading.Event()

# ── Bedrock client ────────────────────────────────────────────────────────────

_env = {}
_env_file = None
for candidate in ENV_FILES:
    if candidate.exists():
        _env.update(dotenv_values(candidate))
        _env_file = candidate

MODEL_ARN = _env.get("ANTHROPIC_MODEL", "")
if not MODEL_ARN:
    searched = ", ".join(str(path) for path in ENV_FILES)
    sys.exit(f"ANTHROPIC_MODEL not found. Searched: {searched}")

# One client per thread (boto3 clients are not thread-safe).
_client_local = threading.local()

def get_client():
    if not hasattr(_client_local, "client"):
        session = boto3.Session(profile_name=BEDROCK_PROFILE, region_name=BEDROCK_REGION)
        _client_local.client = session.client(
            "bedrock-runtime",
            config=BotocoreConfig(read_timeout=300, connect_timeout=10,
                                  retries={"max_attempts": 0}),
        )
    return _client_local.client

# ── API call ──────────────────────────────────────────────────────────────────

MAX_PDF_PAGES = 100  # Bedrock hard limit for PDF document blocks


def _get_pdf_b64(pdf_path: Path) -> str:
    """Return base64-encoded PDF, truncated to MAX_PDF_PAGES pages if necessary."""
    doc = fitz.open(str(pdf_path))
    if len(doc) > MAX_PDF_PAGES:
        trunc = fitz.open()
        trunc.insert_pdf(doc, from_page=0, to_page=MAX_PDF_PAGES - 1)
        data = trunc.tobytes(garbage=1, deflate=True)
        trunc.close()
    else:
        data = pdf_path.read_bytes()
    doc.close()
    return base64.standard_b64encode(data).decode()


class StopRequested(Exception):
    """Raised when _stop is set while waiting for an API call."""

def _invoke_blocking(body: str) -> str:
    """Run in a worker thread. Returns raw response text."""
    resp   = get_client().invoke_model(
        modelId=MODEL_ARN,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"].strip()


def call_model(system: str, pdf_path: Path, user_text: str, label: str = "") -> str:
    """
    Call Bedrock with a PDF document block.
    Runs the blocking boto3 call in a thread and polls every POLL_INTERVAL
    seconds so Ctrl+C (KeyboardInterrupt) is handled promptly by the main thread.
    Raises StopRequested if _stop is set while waiting.
    """
    if _stop.is_set():
        raise StopRequested()

    pdf_b64 = _get_pdf_b64(pdf_path)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "document",
                 "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64}},
                {"type": "text", "text": user_text},
            ],
        }],
    })

    for api_attempt in range(1, 4):
        if _stop.is_set():
            raise StopRequested()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_invoke_blocking, body)
            while True:
                try:
                    return fut.result(timeout=POLL_INTERVAL)
                except concurrent.futures.TimeoutError:
                    if _stop.is_set():
                        fut.cancel()
                        raise StopRequested()
                except Exception as e:
                    if api_attempt == 3:
                        raise
                    wait = RETRY_DELAY * api_attempt
                    tqdm.write(f"  [API retry {api_attempt}/3] {label}: {type(e).__name__} — waiting {wait}s")
                    # Interruptible sleep
                    deadline = time.monotonic() + wait
                    while time.monotonic() < deadline:
                        if _stop.is_set():
                            raise StopRequested()
                        time.sleep(min(POLL_INTERVAL, deadline - time.monotonic()))
                    break  # retry outer loop
    return ""

# ── Meta review I/O ───────────────────────────────────────────────────────────

def read_meta_review() -> str:
    if not META_REVIEW_FILE.exists():
        return ""
    return META_REVIEW_FILE.read_text(encoding="utf-8").strip()

def write_meta_review(content: str) -> None:
    META_REVIEW_FILE.write_text(content.strip() + "\n", encoding="utf-8")

# ── XML parsing ───────────────────────────────────────────────────────────────

def _xml_text(root, tag: str, default: str = "") -> str:
    el = root.find(tag)
    return (el.text or "").strip() if el is not None else default

def _xml_list(root, tag: str, child: str) -> list[str]:
    el = root.find(tag)
    if el is None:
        return []
    return [(c.text or "").strip() for c in el.findall(child)]

def parse_eval_response(xml_str: str) -> dict:
    try:
        xml_str = re.sub(r"```xml\s*", "", xml_str)
        xml_str = re.sub(r"```\s*", "", xml_str)
        m = re.search(r"<review>.*?</review>", xml_str, re.DOTALL)
        if not m:
            pred_m = re.search(r"<locked_prediction>\s*(ACCEPT|REJECT)\s*</locked_prediction>",
                               xml_str, re.IGNORECASE)
            return {"locked_prediction": pred_m.group(1).upper() if pred_m else None,
                    "parse_error": "no <review> block"}
        root = ET.fromstring(m.group(0))
        return {
            "locked_prediction":    _xml_text(root, "locked_prediction").upper(),
            "confidence":           _xml_text(root, "confidence"),
            "paper_type":           _xml_text(root, "paper_type"),
            "central_claim":        _xml_text(root, "central_claim"),
            "claim_evidence_match": _xml_text(root, "claim_evidence_match"),
            "decisive_reasons":     _xml_list(root, "decisive_reasons", "reason"),
            "main_concerns":        _xml_list(root, "main_concerns", "concern"),
            "meta_lessons_applied": _xml_list(root, "meta_lessons_applied", "lesson"),
            "would_flip_if":        _xml_list(root, "would_flip_if", "condition"),
            "parse_error":          None,
        }
    except Exception as e:
        pred_m = re.search(r"<locked_prediction>\s*(ACCEPT|REJECT)\s*</locked_prediction>",
                           xml_str, re.IGNORECASE)
        return {"locked_prediction": pred_m.group(1).upper() if pred_m else None,
                "parse_error": str(e)}

def parse_meta_review_response(xml_str: str) -> dict:
    try:
        xml_str = re.sub(r"```xml\s*", "", xml_str)
        xml_str = re.sub(r"```\s*", "", xml_str)
        m = re.search(r"<meta_review_update>.*?</meta_review_update>", xml_str, re.DOTALL)
        if not m:
            m2 = re.search(r"<updated_meta_review>(.*?)</updated_meta_review>", xml_str, re.DOTALL)
            return {"updated_meta_review": m2.group(1).strip() if m2 else None,
                    "parse_error": "no <meta_review_update> block"}
        root = ET.fromstring(m.group(0))
        diag = root.find("diagnosis")
        updated_el = root.find("updated_meta_review")
        return {
            "failure_mode":        _xml_text(diag, "failure_mode") if diag is not None else "",
            "is_generalizable":    _xml_text(diag, "is_generalizable") if diag is not None else "",
            "action":              _xml_text(diag, "action") if diag is not None else "",
            "updated_meta_review": (updated_el.text or "").strip() if updated_el is not None else "",
            "parse_error":         None,
        }
    except Exception as e:
        m2 = re.search(r"<updated_meta_review>(.*?)</updated_meta_review>", xml_str, re.DOTALL)
        return {"updated_meta_review": m2.group(1).strip() if m2 else None,
                "parse_error": str(e)}

# ── Prompt assembly ───────────────────────────────────────────────────────────

def eval_system_prompt(meta_review: str) -> str:
    base = EVAL_PROMPT_FILE.read_text(encoding="utf-8").strip()
    if meta_review:
        base += f"\n\n---\n\n## Your Meta Review\n\n{meta_review}"
    return base

def eval_user_text() -> str:
    return "Please review this paper and provide your prediction in the XML format specified."

def meta_system_prompt() -> str:
    return META_PROMPT_FILE.read_text(encoding="utf-8").strip()

def meta_user_text(prediction: str, reasoning: str, correct_label: str, meta_review: str) -> str:
    parts = [
        f"**Your original prediction:** {prediction}",
        f"**Your original reasoning:**\n{reasoning}",
        f"**Correct label:** {correct_label}",
        (f"**Your current Meta Review:**\n\n{meta_review}" if meta_review
         else "**Your current Meta Review:** (empty — this is your first error)"),
    ]
    return "\n\n".join(parts)

# ── Logging ───────────────────────────────────────────────────────────────────

def log(record: dict) -> None:
    record["timestamp"] = datetime.utcnow().isoformat()
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ── Resume state ──────────────────────────────────────────────────────────────

def load_resume_state() -> dict:
    """
    Read results.jsonl and return:
      completed_evo_papers  : set of paper_ids where evolution is done
                              (either got correct OR exhausted all retries)
      completed_test_phases : set of phase_labels fully summarised (test_phase_summary logged)
      completed_test_papers : set of (phase_label, paper_id) pairs already predicted in a test
      baseline_done         : bool

    A paper's evolution is "done" if:
      - any attempt was correct=True, OR
      - attempt 3 was logged (regardless of outcome)
    """
    if not RESULTS_FILE.exists():
        return {"completed_evo_papers": set(), "completed_test_phases": set(),
                "completed_test_papers": set(), "baseline_done": False}

    attempts: dict[str, list[dict]] = {}     # paper_id → list of evo attempt records
    completed_test_phases: set[str] = set()
    completed_test_papers: set[tuple] = set()  # (phase_label, paper_id)
    baseline_done = False

    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = rec.get("event", "")
            phase = rec.get("phase", "")

            if event == "prediction" and phase == "evolution":
                pid = rec.get("paper_id", "")
                attempts.setdefault(pid, []).append(rec)

            elif event == "prediction" and phase.startswith("test_"):
                phase_label = phase[len("test_"):]
                pid = rec.get("paper_id", "")
                completed_test_papers.add((phase_label, pid))

            elif event == "test_phase_summary":
                label = rec.get("phase_label", "")
                completed_test_phases.add(label)
                if label == "baseline":
                    baseline_done = True

    completed_evo_papers: set[str] = set()
    for pid, recs in attempts.items():
        if any(r.get("correct") for r in recs):
            completed_evo_papers.add(pid)
        elif any(r.get("attempt", 0) >= MAX_RETRIES for r in recs):
            completed_evo_papers.add(pid)

    return {
        "completed_evo_papers": completed_evo_papers,
        "completed_test_phases": completed_test_phases,
        "completed_test_papers": completed_test_papers,
        "baseline_done": baseline_done,
    }

def last_attempt_number(paper_id: str) -> int:
    """Return the highest attempt number already logged for this evolution paper."""
    if not RESULTS_FILE.exists():
        return 0
    highest = 0
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                if (rec.get("event") == "prediction"
                        and rec.get("phase") == "evolution"
                        and rec.get("paper_id") == paper_id):
                    highest = max(highest, rec.get("attempt", 0))
            except Exception:
                continue
    return highest

# ── Paper loading & splitting ─────────────────────────────────────────────────

def load_papers() -> list[dict]:
    with open(PAPERS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def make_splits(papers: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    import random
    rng = random.Random(seed)
    evolution, test = [], []
    for conf in CONFERENCES:
        accepts = [p for p in papers if p["conference"] == conf and p["accept_reject"] == "accept"]
        rejects = [p for p in papers if p["conference"] == conf and p["accept_reject"] == "reject"]
        if len(accepts) < 10 or len(rejects) < 10:
            raise ValueError(
                f"Need at least 10 accepts and 10 rejects for {conf}; "
                f"found {len(accepts)} accepts and {len(rejects)} rejects"
            )
        rng.shuffle(accepts)
        rng.shuffle(rejects)
        evolution += accepts[:5] + rejects[:5]
        test      += accepts[5:10] + rejects[5:10]
    rng.shuffle(evolution)
    rng.shuffle(test)
    return evolution, test

# ── Single paper prediction ───────────────────────────────────────────────────

def predict_paper(paper: dict, meta_review: str, phase: str, attempt: int,
                  pbar_retry: tqdm | None = None) -> dict:
    pdf_path     = REPO_ROOT / paper["path"]
    ground_truth = paper["accept_reject"].upper()
    label        = f"[{phase}][{paper['title'][:40]}][a{attempt}]"

    if pbar_retry:
        pbar_retry.set_description(f"Model call… attempt {attempt}/{MAX_RETRIES}")

    raw_resp = call_model(eval_system_prompt(meta_review), pdf_path, eval_user_text(), label)
    parsed   = parse_eval_response(raw_resp)
    pred     = parsed.get("locked_prediction")
    correct  = (pred == ground_truth) if pred else None

    record = {
        "event":                "prediction",
        "phase":                phase,
        "paper_id":             paper["path"],
        "title":                paper["title"],
        "conference":           paper["conference"],
        "accept_reject":        paper["accept_reject"],
        "accept_type":          paper.get("accept_type", "NA"),
        "ground_truth":         ground_truth,
        "prediction":           pred,
        "correct":              correct,
        "attempt":              attempt,
        "model":                MODEL_ARN,
        "meta_review_snapshot": meta_review,
        "raw_response":         raw_resp,
        **{k: v for k, v in parsed.items()},
    }
    log(record)
    return record

# ── Meta review update ────────────────────────────────────────────────────────

def update_meta_review(paper: dict, prediction: str, raw_reasoning: str,
                       correct_label: str, current_meta: str,
                       pbar_retry: tqdm | None = None) -> str:
    pdf_path = REPO_ROOT / paper["path"]
    if pbar_retry:
        pbar_retry.set_description("Updating meta review…")

    raw_resp = call_model(
        meta_system_prompt(), pdf_path,
        meta_user_text(prediction, raw_reasoning, correct_label, current_meta),
        label=f"[meta][{paper['title'][:40]}]",
    )
    parsed   = parse_meta_review_response(raw_resp)
    new_meta = parsed.get("updated_meta_review") or current_meta

    log({
        "event":              "meta_review_update",
        "paper_id":           paper["path"],
        "title":              paper["title"],
        "prediction_was":     prediction,
        "correct_label":      correct_label,
        "failure_mode":       parsed.get("failure_mode", ""),
        "is_generalizable":   parsed.get("is_generalizable", ""),
        "action":             parsed.get("action", ""),
        "raw_response":       raw_resp,
        "meta_review_before": current_meta,
        "meta_review_after":  new_meta,
        "parse_error":        parsed.get("parse_error"),
    })
    if new_meta and new_meta != current_meta:
        write_meta_review(new_meta)
    return new_meta

# ── Test phase (parallelised) ─────────────────────────────────────────────────

def run_test_phase(test_papers: list[dict], phase_label: str,
                   pbar_test: tqdm, pbar_evo: tqdm,
                   resume_state: dict | None = None) -> float:
    """
    Run test papers in parallel (TEST_WORKERS concurrent API calls).
    Skips papers whose (phase_label, paper_id) is already in completed_test_papers.
    Writes test_phase_summary once all papers (new + previously logged) are accounted for.
    Raises StopRequested if interrupted mid-phase.
    """
    meta_review = read_meta_review()
    done_pairs: set[tuple] = (resume_state or {}).get("completed_test_papers", set())

    papers_to_run = [p for p in test_papers
                     if (phase_label, p["path"]) not in done_pairs]
    already_done  = len(test_papers) - len(papers_to_run)

    if already_done:
        pbar_evo.write(f"  TEST [{phase_label}]: {already_done} papers already logged, "
                       f"running {len(papers_to_run)} remaining")

    log({"event": "test_phase_start", "phase_label": phase_label,
         "n_papers": len(test_papers), "n_remaining": len(papers_to_run),
         "meta_review_snapshot": meta_review})

    pbar_test.reset(total=len(test_papers))
    pbar_test.n = already_done
    pbar_test.set_description(f"TEST [{phase_label}]")
    pbar_test.refresh()

    new_results: list[dict] = []

    def _run_one(paper: dict) -> dict:
        return predict_paper(paper, meta_review, phase=f"test_{phase_label}", attempt=1)

    if papers_to_run:
        with concurrent.futures.ThreadPoolExecutor(max_workers=TEST_WORKERS,
                                                   thread_name_prefix="test") as ex:
            futures = {ex.submit(_run_one, p): p for p in papers_to_run}
            try:
                for fut in concurrent.futures.as_completed(futures):
                    if _stop.is_set():
                        for f in futures:
                            f.cancel()
                        raise StopRequested()
                    paper = futures[fut]
                    record = fut.result()   # re-raises StopRequested if set inside thread
                    new_results.append(record)
                    status = "✓" if record["correct"] else "✗"
                    pbar_evo.write(
                        f"  TEST {status} [{paper['conference']}] {paper['title'][:55]} "
                        f"| pred={record['prediction']} gt={record['ground_truth']}"
                    )
                    pbar_test.update(1)
                    pbar_test.set_postfix(done=already_done + len(new_results),
                                          total=len(test_papers))
            except StopRequested:
                for f in futures:
                    f.cancel()
                raise

    # Count correct predictions for this phase from the log, deduplicating by paper_id
    # (a crash+resume can produce duplicate entries for the same paper).
    seen: dict[str, bool] = {}  # paper_id → correct
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                if (rec.get("event") == "prediction"
                        and rec.get("phase") == f"test_{phase_label}"):
                    seen[rec["paper_id"]] = bool(rec.get("correct"))
            except Exception:
                continue
    correct_count = sum(1 for v in seen.values() if v)

    total = len(test_papers)
    accuracy = correct_count / total if total else 0.0
    log({"event": "test_phase_summary", "phase_label": phase_label,
         "correct": correct_count, "total": total, "accuracy": accuracy})
    pbar_evo.write(
        f"\n  ── TEST [{phase_label}]: {correct_count}/{total} = {accuracy:.1%} ──\n"
    )
    pbar_test.set_description(f"TEST [{phase_label}] done")
    return accuracy

# ── Evolution: single paper ───────────────────────────────────────────────────

def run_evolution_paper(paper: dict, resume_state: dict,
                        pbar_evo: tqdm, pbar_retry: tqdm) -> bool:
    """
    Evolve on one paper. Returns True if correct within MAX_RETRIES.
    Resumes from wherever a previous run left off within this paper.
    """
    ground_truth = paper["accept_reject"].upper()
    title_short  = paper["title"][:55]
    paper_id     = paper["path"]

    # Find which attempts already happened for this paper
    already_done = last_attempt_number(paper_id)
    start_attempt = already_done + 1

    pbar_retry.reset(total=MAX_RETRIES)
    pbar_retry.n = already_done
    pbar_retry.set_description(f"Retries [{title_short}]")
    pbar_retry.refresh()

    if already_done >= MAX_RETRIES:
        pbar_evo.write(f"  EVO skip [{paper['conference']}] {title_short} (already exhausted {MAX_RETRIES} attempts)")
        return False

    for attempt in range(start_attempt, MAX_RETRIES + 1):
        meta_review = read_meta_review()
        pbar_retry.set_postfix(attempt=f"{attempt}/{MAX_RETRIES}")

        record  = predict_paper(paper, meta_review, phase="evolution", attempt=attempt,
                                pbar_retry=pbar_retry)
        pred    = record["prediction"]
        correct = record["correct"]
        status  = "✓" if correct else "✗"

        pbar_evo.write(
            f"  EVO {status} [{paper['conference']}] {title_short} "
            f"| attempt {attempt}/{MAX_RETRIES}  pred={pred}  gt={ground_truth}"
        )
        pbar_retry.update(1)

        if correct:
            pbar_retry.set_description(f"✓ correct on attempt {attempt}")
            return True

        pbar_evo.write(f"    → Wrong. Updating meta review (attempt {attempt})…")
        new_meta = update_meta_review(
            paper=paper,
            prediction=pred or "UNKNOWN",
            raw_reasoning=record.get("raw_response", ""),
            correct_label=ground_truth,
            current_meta=meta_review,
            pbar_retry=pbar_retry,
        )
        pbar_evo.write(
            f"    → Meta review {'updated' if new_meta != meta_review else 'unchanged'}."
        )

    pbar_retry.set_description(f"✗ failed all {MAX_RETRIES} attempts")
    return False

# ── Evolution phase ───────────────────────────────────────────────────────────

def run_evolution_phase(evolution_papers: list[dict], test_papers: list[dict],
                        resume_state: dict,
                        pbar_evo: tqdm, pbar_retry: tqdm, pbar_test: tqdm) -> None:
    completed_evo  = resume_state["completed_evo_papers"]
    completed_test = resume_state["completed_test_phases"]

    deferred          = []
    consecutive_fails = 0
    papers_done       = len(completed_evo)  # count already-done papers toward checkpoints

    log({"event": "evolution_phase_start", "n_papers": len(evolution_papers),
         "resuming_from": papers_done})

    # Advance evo bar to reflect already-completed papers
    pbar_evo.n = papers_done
    pbar_evo.refresh()

    for i, paper in enumerate(evolution_papers):
        paper_id = paper["path"]

        if paper_id in completed_evo:
            pbar_evo.write(
                f"  EVO skip [{paper['conference']}] {paper['title'][:55]} (already completed)"
            )
            continue

        pbar_evo.set_description(f"EVO paper {i+1}/{len(evolution_papers)}")
        pbar_evo.set_postfix(title=paper["title"][:30], deferred=len(deferred))

        success = run_evolution_paper(paper, resume_state, pbar_evo, pbar_retry)
        papers_done += 1
        pbar_evo.update(1)

        if success:
            consecutive_fails = 0
        else:
            consecutive_fails += 1
            deferred.append(paper)
            pbar_evo.write(f"    → Deferred. Consecutive failures: {consecutive_fails}")
            if consecutive_fails >= CONSEC_FAIL_LIMIT:
                pbar_evo.write(
                    f"\n  ── {CONSEC_FAIL_LIMIT} consecutive failures — ending main evolution pass early ──\n"
                )
                break

        # Test checkpoint (skip if already done in a previous run)
        if papers_done % TEST_EVERY_N == 0:
            label = f"after_evo_{papers_done}"
            if label not in completed_test:
                pbar_evo.write(f"\n  ── Running test checkpoint: {label} ──")
                run_test_phase(test_papers, label, pbar_test, pbar_evo, resume_state)
            else:
                pbar_evo.write(f"\n  ── Test checkpoint {label} already done, skipping ──\n")

    # Second pass on deferred papers
    if deferred:
        pbar_evo.write(f"\n  ── Second pass on {len(deferred)} deferred papers ──\n")
        log({"event": "deferred_pass_start", "n_deferred": len(deferred)})
        consecutive_fails = 0
        deferred_done     = 0

        for paper in deferred:
            paper_id = paper["path"]
            if paper_id in completed_evo:
                continue

            pbar_evo.set_description(f"EVO deferred: {paper['title'][:30]}")
            success = run_evolution_paper(paper, resume_state, pbar_evo, pbar_retry)
            pbar_evo.update(1)
            deferred_done += 1

            if success:
                consecutive_fails = 0
            else:
                consecutive_fails += 1
                if consecutive_fails >= CONSEC_FAIL_LIMIT:
                    pbar_evo.write(
                        f"\n  ── {CONSEC_FAIL_LIMIT} consecutive failures on deferred pass — stopping ──\n"
                    )
                    break

    log({"event": "evolution_phase_end"})

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Model : {MODEL_ARN}")
    print(f"Env   : {_env_file}")
    print(f"Output: {RESULTS_FILE}\n")

    papers = load_papers()
    evolution_papers, test_papers = make_splits(papers)
    resume_state = load_resume_state()

    n_evo_done = len(resume_state["completed_evo_papers"])
    n_test_done = len(resume_state["completed_test_phases"])
    if n_evo_done or n_test_done:
        print(f"Resuming: {n_evo_done} evolution papers done, "
              f"{n_test_done} test phases done.")
    else:
        print(f"Evolution set : {len(evolution_papers)} papers")
        for p in evolution_papers:
            print(f"  [{p['accept_reject']:6}] [{p['conference']}] {p['title'][:65]}")
        print(f"\nTest set : {len(test_papers)} papers")
        for p in test_papers:
            print(f"  [{p['accept_reject']:6}] [{p['conference']}] {p['title'][:65]}")
    print()

    log({"event": "experiment_start", "model": MODEL_ARN,
         "resuming": n_evo_done > 0,
         "evolution_papers": [p["path"] for p in evolution_papers],
         "test_papers":      [p["path"] for p in test_papers]})

    pbar_evo   = tqdm(total=len(evolution_papers), desc="Evolution", unit="paper",
                      position=0, leave=True, colour="cyan")
    pbar_retry = tqdm(total=MAX_RETRIES, desc="Retries", unit="attempt",
                      position=1, leave=False, colour="yellow")
    pbar_test  = tqdm(total=len(test_papers), desc="Test", unit="paper",
                      position=2, leave=True, colour="green")

    interrupted = False
    try:
        # Baseline test (skip if already done)
        if not resume_state["baseline_done"]:
            pbar_evo.write("\n── Baseline test (before evolution) ──")
            run_test_phase(test_papers, "baseline", pbar_test, pbar_evo, resume_state)
        else:
            pbar_evo.write("── Baseline test already done, skipping ──")

        run_evolution_phase(evolution_papers, test_papers, resume_state,
                            pbar_evo, pbar_retry, pbar_test)

    except KeyboardInterrupt:
        # Signal all threads to stop, then wait briefly for any in-flight call to notice.
        _stop.set()
        interrupted = True
        pbar_evo.write("\n\nCtrl+C received — stopping after current API call finishes…")
        time.sleep(POLL_INTERVAL * 2)
        pbar_evo.write("Progress saved. Re-run to continue from where you left off.\n")
    except StopRequested:
        interrupted = True
        pbar_evo.write("\nStopped cleanly. Re-run to continue.\n")
    finally:
        pbar_retry.close()
        pbar_evo.close()
        pbar_test.close()

    log({"event": "experiment_end" if not interrupted else "experiment_interrupted"})
    print(f"\n{'Done' if not interrupted else 'Interrupted'}. Results in {RESULTS_FILE}")
    print(f"Meta review : {META_REVIEW_FILE}")


if __name__ == "__main__":
    main()
