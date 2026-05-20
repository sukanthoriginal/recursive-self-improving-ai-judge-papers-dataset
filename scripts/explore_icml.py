"""Probe OpenReview for ICML 2025 to understand schema before building.

Outputs:
  - venue ID + group config
  - submission count
  - a sample review's full field schema (rating scale, confidence scale, sub-scores)
  - a sample decision's structure (tier names)
"""
import json
from openreview.api import OpenReviewClient

client = OpenReviewClient(baseurl="https://api2.openreview.net")

# Try common venue ID patterns
candidates = ["ICML.cc/2025/Conference", "ICML.cc/2025"]
venue_id = None
for v in candidates:
    try:
        g = client.get_group(v)
        venue_id = v
        print(f"Found venue: {v}")
        break
    except Exception as e:
        print(f"  {v} -> {type(e).__name__}")

if not venue_id:
    raise SystemExit("Could not find ICML 2025 venue")

venue_group = client.get_group(venue_id)
content = venue_group.content or {}
print("\nVenue group content keys:")
for k, v in content.items():
    val = v.get("value") if isinstance(v, dict) else v
    if isinstance(val, (str, int, bool, type(None))):
        print(f"  {k}: {val}")

# Count submissions
print("\nCounting submissions (fetching all)...")
subs = client.get_all_notes(invitation=f"{venue_id}/-/Submission")
print(f"Total submissions: {len(subs)}")

# Sample submission with replies
print("\nFetching one submission with replies for schema inspection...")
sample = client.get_notes(invitation=f"{venue_id}/-/Submission", details="directReplies", limit=1)[0]
print(f"  sample: {sample.id} — {sample.content.get('title', {}).get('value', '')[:80]}")
print(f"  venueid: {sample.content.get('venueid', {}).get('value')}")
print(f"  venue label: {sample.content.get('venue', {}).get('value')}")

replies = (sample.details or {}).get("directReplies", []) or []
print(f"  {len(replies)} replies")

review_sample = None
decision_sample = None
for r in replies:
    invs = r.get("invitations", [])
    if any(i.endswith("/-/Official_Review") for i in invs) and review_sample is None:
        review_sample = r
    elif any(i.endswith("/-/Decision") for i in invs) and decision_sample is None:
        decision_sample = r

# REVIEW
print("\n=== SAMPLE REVIEW SCHEMA ===")
if review_sample:
    content = review_sample.get("content", {}) or {}
    print(f"Fields: {list(content.keys())}")
    for k, v in content.items():
        val = v.get("value") if isinstance(v, dict) else v
        if isinstance(val, (int, float, bool)) or (isinstance(val, str) and len(val) < 80):
            print(f"  {k}: {val!r}")
        else:
            t = type(val).__name__
            n = len(str(val)) if val is not None else 0
            print(f"  {k}: <{t} len={n}>")
else:
    print("  NO REVIEW FOUND on sample submission — may need to try a different one")

# DECISION
print("\n=== SAMPLE DECISION ===")
if decision_sample:
    content = decision_sample.get("content", {}) or {}
    for k, v in content.items():
        val = v.get("value") if isinstance(v, dict) else v
        if isinstance(val, str) and len(val) < 200:
            print(f"  {k}: {val!r}")
        else:
            print(f"  {k}: <{type(val).__name__} len={len(str(val))}>")
else:
    print("  NO DECISION FOUND on sample submission")
