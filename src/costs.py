"""CompLLM: the spend ledger.

    python src/costs.py                  # rebuild results/costs.csv + print summary
    python src/costs.py --topup 20.00    # also reconcile against money added
    python src/costs.py --by date        # group the summary differently

Every archived response in `raw/` carries OpenRouter's own `usage.cost` for that
call, so this reconstructs the complete spend history from the archives rather
than keeping a running log alongside them. That is deliberate: a separate
accumulating log drifts the moment a run crashes, is resumed, or is re-parsed,
whereas a derived ledger is correct by construction and reproducible by anyone
who clones the repo -- no API key required.

Failed calls (401 auth, 402 insufficient credit) are never archived because
they never produced a response, and they cost nothing, so their absence is
correct rather than a gap.

Outputs
    results/costs.csv       one row per API call -- the durable record
    results/costs_summary.json  totals by condition and judge, for the paper
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
RESULTS = ROOT / "results"
LEDGER = RESULTS / "costs.csv"
SUMMARY = RESULTS / "costs_summary.json"

FIELDS = [
    "ts_utc", "condition", "prompt_id", "shown_author", "judge",
    "model_slug", "provider", "prompt_tokens", "completion_tokens",
    "reasoning_tokens", "total_tokens", "cost_usd", "source_file",
]


def parse_name(stem: str) -> tuple[str, str, str, str]:
    """raw/ filename -> (condition, prompt_id, shown_author, judge).

    lineup:  <prompt_id>__<judge>
    single:  single__<prompt_id>__<shown_author>__<judge>
    """
    parts = stem.split("__")
    if parts and parts[0] == "single":
        if len(parts) == 4:
            return "single", parts[1], parts[2], parts[3]
        return "single", parts[1] if len(parts) > 1 else "", "", parts[-1]
    if len(parts) == 2:
        return "lineup", parts[0], "", parts[1]
    return "unknown", stem, "", parts[-1] if parts else ""


def collect() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(RAW / "*.json"))):
        stem = os.path.basename(path)[:-5]
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        resp = d.get("response") or {}
        usage = resp.get("usage") or {}
        if not usage:
            continue  # error record or unarchived failure -- no cost incurred
        condition, prompt_id, shown, judge = parse_name(stem)
        created = resp.get("created")
        ts = (dt.datetime.fromtimestamp(created, dt.timezone.utc).isoformat()
              if isinstance(created, (int, float)) else "")
        rows.append({
            "ts_utc": ts,
            "condition": condition,
            "prompt_id": prompt_id,
            "shown_author": shown,
            "judge": judge,
            "model_slug": resp.get("model") or d.get("request_model") or "",
            "provider": resp.get("provider") or "",
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {})
                                .get("reasoning_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cost_usd": round(float(usage.get("cost", 0) or 0), 6),
            "source_file": f"raw/{stem}.json",
        })
    rows.sort(key=lambda r: (r["ts_utc"], r["source_file"]))
    return rows


def write_ledger(rows: list[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    with LEDGER.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def summarise(rows: list[dict], group: str, topup: float | None) -> dict:
    total = sum(r["cost_usd"] for r in rows)
    by_cond: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    by_judge: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    by_key: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for r in rows:
        for bucket, key in ((by_cond, r["condition"]),
                            (by_judge, (r["condition"], r["judge"])),
                            (by_key, r["ts_utc"][:10] if group == "date"
                                     else r["model_slug"])):
            bucket[key][0] += r["cost_usd"]
            bucket[key][1] += 1

    print(f"\n{'=' * 68}\nCompLLM SPEND LEDGER\n{'=' * 68}")
    print(f"calls archived : {len(rows)}")
    print(f"total spend    : ${total:.4f}")
    if rows:
        print(f"date range     : {rows[0]['ts_utc'][:10]} .. {rows[-1]['ts_utc'][:10]}")

    print(f"\n-- by condition --\n  {'condition':10s}{'calls':>7}{'$':>10}{'$/call':>10}")
    for k in sorted(by_cond):
        c, n = by_cond[k]
        print(f"  {k:10s}{n:7d}{c:10.4f}{c / n if n else 0:10.4f}")

    print(f"\n-- by condition x judge --\n  {'condition':10s}{'judge':10s}"
          f"{'calls':>7}{'$':>10}{'$/call':>10}")
    for k in sorted(by_judge):
        c, n = by_judge[k]
        print(f"  {k[0]:10s}{k[1]:10s}{n:7d}{c:10.4f}{c / n if n else 0:10.4f}")

    label = "date" if group == "date" else "model"
    print(f"\n-- by {label} --\n  {label:34s}{'calls':>7}{'$':>10}")
    for k in sorted(by_key):
        c, n = by_key[k]
        print(f"  {str(k):34s}{n:7d}{c:10.4f}")

    out = {
        "total_usd": round(total, 4),
        "calls": len(rows),
        "by_condition": {k: {"usd": round(v[0], 4), "calls": v[1]}
                         for k, v in by_cond.items()},
        "by_condition_judge": {f"{k[0]}/{k[1]}": {"usd": round(v[0], 4),
                                                  "calls": v[1]}
                               for k, v in by_judge.items()},
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }

    if topup is not None:
        remaining = topup - total
        out["topped_up_usd"] = round(topup, 2)
        out["remaining_usd"] = round(remaining, 4)
        print(f"\n-- reconciliation --\n  topped up   ${topup:10.2f}"
              f"\n  spent       ${total:10.4f}\n  remaining   ${remaining:10.4f}")
        if remaining < 2:
            print("  WARNING: under $2 left. Top up before the next full run.")
    print("=" * 68)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topup", type=float, default=None,
                    help="total USD added to the account, to reconcile against")
    ap.add_argument("--by", choices=("model", "date"), default="model",
                    help="third grouping in the summary (default: model)")
    args = ap.parse_args()

    rows = collect()
    if not rows:
        sys.exit(f"no archived responses with usage found under {RAW}")
    write_ledger(rows)
    summary = summarise(rows, args.by, args.topup)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {LEDGER.relative_to(ROOT)}  ({len(rows)} rows)")
    print(f"wrote {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
