"""CompLLM: turn raw sessions into the analysis table.

    python src/parse_judgments.py

Reads results/sessions.jsonl, writes results/judgments.csv (one row per
slot judgment), and prints the descriptive summary that Engine B's numbers
come from.

Columns
-------
session_id, prompt_id, judge, slot, true_author, guessed_model, correct,
confidence, reason, is_self_slot, self_recognised, needs_review
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    JUDGMENTS_CSV,
    JUDGMENTS_SINGLE_CSV,
    MODELS,
    SESSIONS_JSONL,
    SESSIONS_SINGLE_JSONL,
    SLOTS,
)

CONDITION = "lineup"


def load_sessions(path=None) -> list[dict]:
    path = path or SESSIONS_JSONL
    if not path.exists():
        raise SystemExit(f"{path} not found - run the collector first.")
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    # keep the LAST successful record per session (re-runs append)
    latest: dict[str, dict] = {}
    for rec in out:
        if rec.get("ok"):
            latest[rec["session_id"]] = rec
    return list(latest.values())


def to_frame_single(sessions: list[dict]) -> pd.DataFrame:
    """One row per (prompt, response, judge). No slots - judgments are
    independent, which is the whole point of this condition."""
    rows = []
    for s in sessions:
        entry = s.get("parsed") or {}
        guessed = entry.get("model")
        true_author = s.get("true_author")
        rows.append(
            {
                "session_id": s["session_id"],
                "prompt_id": s["prompt_id"],
                "judge": s["judge"],
                "slot": None,
                "true_author": true_author,
                "guessed_model": guessed,
                "correct": None if guessed is None else int(guessed == true_author),
                "confidence": entry.get("confidence"),
                "reason": entry.get("reason"),
                "is_self_slot": int(true_author == s["judge"]),
                "self_recognised": (
                    None if guessed is None
                    else int(true_author == s["judge"] and guessed == s["judge"])
                ),
                "needs_review": int(bool(s.get("needs_review"))),
            }
        )
    return pd.DataFrame(rows)


def to_frame(sessions: list[dict]) -> pd.DataFrame:
    rows = []
    for s in sessions:
        parsed = s.get("parsed", {}) or {}
        slot_to_author = s.get("slot_to_author", {}) or {}
        for slot in SLOTS:
            true_author = slot_to_author.get(slot)
            entry = parsed.get(slot) or {}
            guessed = entry.get("model")
            rows.append(
                {
                    "session_id": s["session_id"],
                    "prompt_id": s["prompt_id"],
                    "judge": s["judge"],
                    "slot": slot,
                    "true_author": true_author,
                    "guessed_model": guessed,
                    "correct": None if guessed is None else int(guessed == true_author),
                    "confidence": entry.get("confidence"),
                    "reason": entry.get("reason"),
                    "is_self_slot": int(true_author == s["judge"]),
                    "self_recognised": (
                        None if guessed is None
                        else int(true_author == s["judge"] and guessed == s["judge"])
                    ),
                    "needs_review": int(bool(s.get("needs_review"))),
                }
            )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> None:
    n = len(df)
    parsed = df["guessed_model"].notna().sum()
    print("=" * 72)
    print("ENGINE B - DESCRIPTIVE SUMMARY")
    print("=" * 72)
    print(f"sessions              : {df['session_id'].nunique()}")
    print(f"judgments             : {n}   (parsed {parsed}, unparsed {n - parsed})")
    print(f"prompts               : {df['prompt_id'].nunique()}")
    print(f"judges                : {df['judge'].nunique()}")

    d = df.dropna(subset=["guessed_model"])
    if d.empty:
        print("\nNo parsed judgments yet.")
        return

    print(f"\noverall accuracy      : {d['correct'].mean():.1%}   (chance 20%)")

    if CONDITION == "lineup":
        print("\n-- INSTRUMENT CHECK: are judges assigning permutations? --")
        perm = nonperm = 0
        for _, grp in d.groupby("session_id"):
            if len(grp) == len(SLOTS):
                if grp["guessed_model"].nunique() == len(SLOTS):
                    perm += 1
                else:
                    nonperm += 1
        tot_sess = perm + nonperm
        if tot_sess:
            frac = perm / tot_sess
            print(f"  complete sessions that named 5 distinct models: {perm}/{tot_sess} = {frac:.0%}")
            if frac > 0.8:
                print("  WARNING: judges are treating this as a matching task.")
                print("  Under a permutation every model gets exactly one guess per session,")
                print("  so the marginal blame distribution is uniform BY CONSTRUCTION and")
                print("  concentration is unmeasurable here. Use the single-text condition")
                print("  (run_single.py) to measure blame concentration.")

    print("\n-- BLAME DISTRIBUTION (all judgments) --")
    vc = d["guessed_model"].value_counts()
    for m in MODELS:
        c = int(vc.get(m, 0))
        print(f"  {m:<10} {c:>5}  {c/len(d):>6.1%}   (expected 20.0%)")

    obs = np.array([vc.get(m, 0) for m in MODELS], dtype=float)
    if obs.sum() > 0:
        chi2, p = stats.chisquare(obs)
        print(f"\n  chi2 (pooled judgments) = {chi2:.1f}, p = {p:.3g}, df = {len(MODELS)-1}")
        print("  NOTE: pooled judgments are not independent - they cluster in")
        print(f"  {d['judge'].nunique()} judges and {d['prompt_id'].nunique()} prompts. Report the test below instead.")

        # Judge-averaged: each judge contributes one proportion vector, so the
        # unit of analysis is the judge (n=5), not the judgment.
        shares = (pd.crosstab(d["judge"], d["guessed_model"], normalize="index")
                  .reindex(columns=MODELS).fillna(0))
        mean_share = shares.mean(axis=0)
        print("\n  judge-averaged blame share (unit of analysis = judge, n="
              f"{len(shares)}):")
        for m in MODELS:
            vals = shares[m]
            print(f"    {m:<10} {mean_share[m]:>6.1%}  (per-judge: "
                  f"{', '.join(f'{v:.0%}' for v in vals)})")
        maxes = shares.max(axis=1)
        if len(shares) > 1 and maxes.std() > 1e-12:
            t, pt = stats.ttest_1samp(maxes, 1.0 / len(MODELS))
            print(f"    max-share vs 20% across judges: t = {t:.2f}, p = {pt:.3g}")
        elif len(shares) > 1:
            print("    max-share vs 20%: not testable - every judge gave an "
                  "identical share (permutation behaviour).")

    err = d[d["correct"] == 0]
    print(f"\n-- BLAME DISTRIBUTION ON ERRORS ONLY (n={len(err)}) --")
    if len(err):
        vce = err["guessed_model"].value_counts()
        for m in MODELS:
            c = int(vce.get(m, 0))
            print(f"  {m:<10} {c:>5}  {c/len(err):>6.1%}")

    print("\n-- SELF-RECOGNITION (diagonal, chance 20%) --")
    for judge in MODELS:
        sub = d[(d["judge"] == judge) & (d["is_self_slot"] == 1)]
        if len(sub):
            print(f"  {judge:<10} {int(sub['correct'].sum()):>3}/{len(sub):<3} = {sub['correct'].mean():>6.1%}")

    print("\n-- PER-JUDGE BLAME SHARE (row = judge, cell = % of its guesses) --")
    ct = pd.crosstab(d["judge"], d["guessed_model"], normalize="index")
    ct = ct.reindex(index=[m for m in MODELS if m in ct.index],
                    columns=[m for m in MODELS if m in ct.columns]).fillna(0)
    print((ct * 100).round(1).to_string())

    if CONDITION == "lineup":
        print("\n-- SLOT DISTRIBUTION OF GUESSES (position-bias check) --")
        print("   Latin square makes slot orthogonal to model identity, so any")
        print("   deviation here is a judge-side position preference, not a confound.")
        sc = d["slot"].value_counts().reindex(SLOTS).fillna(0)
        for slot in SLOTS:
            c = int(sc.get(slot, 0))
            print(f"  slot {slot}: {c:>5}  {c/len(d):>6.1%}")

        print("\n-- BALANCE CHECK: (true_author, slot) occurrences --")
        bal = pd.crosstab(df["true_author"], df["slot"])
        print(bal.to_string())
        flat = bal.values.flatten()
        print(f"  min={flat.min()}  max={flat.max()}  "
              f"{'BALANCED' if flat.min() == flat.max() else 'UNBALANCED - investigate'}")

    nr = df[df["needs_review"] == 1]["session_id"].nunique()
    if nr:
        print(f"\nWARNING: {nr} session(s) flagged needs_review (incomplete parse).")
        print("  Inspect raw/<session_id>.json for those before analysis.")
    print("=" * 72)


def main() -> None:
    global CONDITION
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["lineup", "single"], default="lineup")
    args = ap.parse_args()
    CONDITION = args.condition

    if CONDITION == "single":
        sessions = load_sessions(SESSIONS_SINGLE_JSONL)
        df = to_frame_single(sessions)
        out_path = JUDGMENTS_SINGLE_CSV
    else:
        sessions = load_sessions(SESSIONS_JSONL)
        df = to_frame(sessions)
        out_path = JUDGMENTS_CSV

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"condition: {CONDITION.upper()}")
    print(f"wrote {out_path}  ({len(df)} rows)\n")
    summarise(df)


if __name__ == "__main__":
    main()
