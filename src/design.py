"""CompLLM: lineup design.

Builds the position-counterbalanced blind lineup: every (model, slot) pair
occurs exactly the same number of times across the design, so slot position
is orthogonal to model identity BY CONSTRUCTION rather than by post-hoc
adjustment.

The square
----------
For prompt index p and judge index j, the rotation offset is

    offset = (j + p) mod 5

and model m is placed in slot

    slot = (index(m) + offset) mod 5

Two properties follow, and both are asserted in verify_balance():

1.  GLOBAL BALANCE. For a fixed model m and slot s the offset is pinned to a
    single value, (s - index(m)) mod 5. For each prompt there is exactly one
    judge satisfying it, so every (model, slot) pair occurs exactly n_prompts
    times across the whole design. With 39 prompts: 39 occurrences each.

2.  WITHIN-JUDGE VARIATION. Because the offset depends on p as well as j, a
    given judge does not see the same model in the same slot for every
    prompt. A naive square using offset = j alone is globally balanced too,
    but pins each model to one fixed slot for all 39 prompts of a given
    judge, which confounds that judge's position preference with one model.
"""
from __future__ import annotations

import csv
from collections import Counter

from config import MODELS, SLOTS, DESIGN_CSV, EXCLUDE_PROMPTS


def build_design(prompt_ids: list[str], judges: list[str]) -> list[dict]:
    """Return one row per (prompt, judge, slot) = the full presentation plan."""
    prompts = sorted(prompt_ids)
    rows: list[dict] = []
    for p_idx, prompt_id in enumerate(prompts):
        for j_idx, judge in enumerate(judges):
            offset = (j_idx + p_idx) % len(MODELS)
            for m_idx, model in enumerate(MODELS):
                slot_idx = (m_idx + offset) % len(SLOTS)
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "judge": judge,
                        "slot": SLOTS[slot_idx],
                        "true_author": model,
                        "offset": offset,
                        "session_id": f"{prompt_id}__{judge}",
                    }
                )
    return rows


def verify_balance(rows: list[dict], n_prompts: int) -> None:
    """Hard-fail if the square is not balanced. Called before every run."""
    pair_counts = Counter((r["true_author"], r["slot"]) for r in rows)

    expected = n_prompts
    bad = {k: v for k, v in pair_counts.items() if v != expected}
    if bad:
        raise AssertionError(
            f"Latin square is not balanced. Expected {expected} occurrences of "
            f"every (model, slot) pair; got deviations: {bad}"
        )
    if len(pair_counts) != len(MODELS) * len(SLOTS):
        raise AssertionError(
            f"Expected {len(MODELS) * len(SLOTS)} distinct (model, slot) pairs, "
            f"got {len(pair_counts)}."
        )

    # Within a session, the five slots must be a permutation of the five models.
    by_session: dict[str, list[str]] = {}
    for r in rows:
        by_session.setdefault(r["session_id"], []).append(r["slot"])
    for sid, slots in by_session.items():
        if sorted(slots) != sorted(SLOTS):
            raise AssertionError(f"Session {sid} does not fill all five slots: {slots}")


def write_design(rows: list[dict]) -> None:
    DESIGN_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DESIGN_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["session_id", "prompt_id", "judge", "slot", "true_author", "offset"],
        )
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["prompt_id"], r["judge"], r["slot"])):
            w.writerow(
                {
                    "session_id": r["session_id"],
                    "prompt_id": r["prompt_id"],
                    "judge": r["judge"],
                    "slot": r["slot"],
                    "true_author": r["true_author"],
                    "offset": r["offset"],
                }
            )


if __name__ == "__main__":
    import pandas as pd
    from config import RESPONSES_CSV

    df = pd.read_csv(RESPONSES_CSV)
    all_prompts = sorted(df["prompt_id"].astype(str).unique())
    prompts = [p for p in all_prompts if p not in EXCLUDE_PROMPTS]
    if len(prompts) != len(all_prompts):
        print(f"excluded : {sorted(set(all_prompts) - set(prompts))}  (config.EXCLUDE_PROMPTS)")
    rows = build_design(prompts, MODELS)
    verify_balance(rows, n_prompts=len(prompts))
    write_design(rows)

    print(f"prompts        : {len(prompts)}")
    print(f"judges         : {len(MODELS)}")
    print(f"sessions       : {len(prompts) * len(MODELS)}")
    print(f"judgments      : {len(rows)}")
    print(f"balance check  : PASS  (every (model, slot) pair occurs {len(prompts)}x)")
    print(f"written        : {DESIGN_CSV}")
