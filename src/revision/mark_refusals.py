"""CompLLM revision: mark refused or truncated judgments as missing.

    python src/revision/mark_refusals.py --panel frontier --condition shuffle_lineup

A request that ends with finish_reason "content_filter" returns no text. The
runner parses that as an answer naming no model, which the analysis would
score as a wrong attribution. This reads each item's archived raw response
and rewrites refused items with ok=false and refused=true, so the analysis
(which keeps ok items only) treats them as missing. On 5 Oct 2026 Claude
Opus 4.7 refused all 38 word-shuffled lineups, on both Anthropic's and
AWS's endpoints, while answering the unshuffled lineup normally. Requests
that stop at the output-token limit ("length") without a parseable answer
are marked missing the same way (Gemini, 11 shuffled lineups).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from panels import PANELS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=sorted(PANELS))
    ap.add_argument("--condition", required=True)
    args = ap.parse_args()
    panel = PANELS[args.panel]
    path = panel.results_dir / f"{args.condition}.jsonl"
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = 0
    for r in recs:
        raw = panel.results_dir / "raw" / r["judge"] / f"{args.condition}__{r['item_id']}.json"
        if not raw.exists():
            continue
        body = json.loads(raw.read_text(encoding="utf-8"))["response"]
        reason = body["choices"][0].get("finish_reason")
        if reason == "content_filter":
            r["ok"], r["refused"] = False, True
            n += 1
        elif reason == "length" and r.get("needs_review"):
            # Ran out of output tokens before giving an answer that parses.
            r["ok"], r["truncated"] = False, True
            n += 1
        elif r.get("needs_review") and not r.get("parsed"):
            # A normal stop with no answer at all (blank, or the provider's
            # "An error occurred" text).
            r["ok"], r["empty"] = False, True
            n += 1
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    by_judge: dict[str, int] = {}
    for r in recs:
        for flag in ("refused", "truncated", "empty"):
            if r.get(flag):
                by_judge[f"{r['judge']} {flag}"] = by_judge.get(f"{r['judge']} {flag}", 0) + 1
    print(f"{path.name}: {n} of {len(recs)} items marked missing {by_judge}")
    left = [r["item_id"] for r in recs if r.get("ok") and r.get("needs_review")]
    if left:
        print(f"  still answered but not fully parsed: {left}")


if __name__ == "__main__":
    main()
