"""Fetch full-venue score pools from OpenReview for normalization.

This script does not download PDFs. It only reads submissions, official reviews,
and decisions, then writes the confidence-weighted score distribution used by
evals/build_papers_csv.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openreview.api import OpenReviewClient
from tqdm import tqdm


OUT = Path(__file__).resolve().parent / "cache" / "venue_score_pools.json"

VENUES = {
    "iclr2025": {
        "venue_id": "ICLR.cc/2025/Conference",
        "score_field": "rating",
    },
    "neurips2025": {
        "venue_id": "NeurIPS.cc/2025/Conference",
        "score_field": "rating",
    },
}


def value(content: dict[str, Any], key: str) -> Any:
    item = content.get(key)
    if isinstance(item, dict) and "value" in item:
        return item["value"]
    return item


def is_official_review(reply: dict[str, Any]) -> bool:
    invitations = reply.get("invitations", []) or []
    return any(inv.endswith("/-/Official_Review") for inv in invitations)


def is_decision(reply: dict[str, Any]) -> bool:
    invitations = reply.get("invitations", []) or []
    return any(inv.endswith("/-/Decision") for inv in invitations)


def ethics_substantive(flag: Any, details: Any) -> bool:
    if isinstance(flag, list):
        return any(
            isinstance(item, str)
            and item.strip().lower() not in ("", "no ethics review needed.", "no ethics review needed", "none")
            for item in flag
        )
    if isinstance(flag, str) and flag.strip().lower() not in ("", "n/a", "na", "none", "no"):
        return True
    if isinstance(details, str) and details.strip().lower() not in ("", "n/a", "na", "none", "no"):
        return True
    return False


def parse_paper(submission: Any, score_field: str) -> dict[str, Any] | None:
    replies = (submission.details or {}).get("directReplies", []) or []
    reviews = []
    decision = None

    for reply in replies:
        content = reply.get("content", {}) or {}
        if is_official_review(reply):
            score = value(content, score_field)
            confidence = value(content, "confidence")
            if score is None or confidence is None:
                continue
            reviews.append(
                {
                    "score": score,
                    "confidence": confidence,
                    "flag_for_ethics_review": value(content, "flag_for_ethics_review"),
                    "details_of_ethics_concerns": value(content, "details_of_ethics_concerns"),
                }
            )
        elif is_decision(reply):
            decision = value(content, "decision")

    if len(reviews) < 3 or not decision:
        return None

    weighted_sum = sum(review["score"] * review["confidence"] for review in reviews)
    confidence_sum = sum(review["confidence"] for review in reviews)
    if confidence_sum == 0:
        return None

    ethics_flagged = any(
        ethics_substantive(review["flag_for_ethics_review"], review["details_of_ethics_concerns"])
        for review in reviews
    )

    return {
        "paper_id": submission.id,
        "title": value(submission.content or {}, "title") or "",
        "decision": decision,
        "accept": str(decision).lower().startswith("accept"),
        "n_reviewers": len(reviews),
        "weighted_mean_overall": round(weighted_sum / confidence_sum, 4),
        "ethics_flagged": ethics_flagged,
    }


def fetch_pool(client: OpenReviewClient, conference: str, venue_id: str, score_field: str) -> dict[str, Any]:
    print(f"\nFetching {conference}: {venue_id}")
    submissions = client.get_all_notes(invitation=f"{venue_id}/-/Submission", details="directReplies")
    print(f"  submissions: {len(submissions)}")

    papers = []
    for submission in tqdm(submissions, desc=conference, unit="paper"):
        paper = parse_paper(submission, score_field)
        if paper is not None:
            papers.append(paper)

    scores = [paper["weighted_mean_overall"] for paper in papers]
    non_ethics_scores = [paper["weighted_mean_overall"] for paper in papers if not paper["ethics_flagged"]]
    return {
        "venue_id": venue_id,
        "score_field": score_field,
        "n_submissions": len(submissions),
        "n_usable": len(papers),
        "n_ethics_flagged": sum(1 for paper in papers if paper["ethics_flagged"]),
        "scores": scores,
        "scores_non_ethics": non_ethics_scores,
        "papers": papers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch full venue score pools from OpenReview.")
    parser.add_argument("--out", type=Path, default=OUT, help="Output JSON path.")
    args = parser.parse_args()

    client = OpenReviewClient(baseurl="https://api2.openreview.net")
    pools = {
        conference: fetch_pool(client, conference, cfg["venue_id"], cfg["score_field"])
        for conference, cfg in VENUES.items()
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pools, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
