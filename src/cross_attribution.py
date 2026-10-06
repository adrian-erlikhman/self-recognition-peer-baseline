"""Cross-attribution: the shape of the errors, not just their count.

Engine B reports how often a judge is right. This reports what happens when it
is wrong: whose text gets mistaken for whom, and whether that pattern depends
on who is judging. It backs the appendix on attribution of others' text: the
cross-attribution matrix, the largest error flows, non-self accuracy, and
where each judge sends its guesses.

    python src/cross_attribution.py                    # lineup
    python src/cross_attribution.py --condition single
    python src/cross_attribution.py --json             # both, to results/

A pure re-read of the tables parse_judgments.py already wrote. No API key.

A flow's share is quoted against the author's row, all 190 judgments passed on
that author's text, not against the error total. "Grok to DeepSeek, 32.6%"
means 32.6% of everything said about Grok-authored text named DeepSeek. The
share of errors is reported alongside, labelled as such.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    EXCLUDE_PROMPTS, JUDGMENTS_CSV, JUDGMENTS_SINGLE_CSV, MODELS, RESULTS_DIR,
)

SOURCES = {"lineup": JUDGMENTS_CSV, "single": JUDGMENTS_SINGLE_CSV}


def load(condition: str) -> pd.DataFrame:
    path = SOURCES[condition]
    if not path.exists():
        sys.exit(f"cross_attribution: {path} not found; run parse_judgments.py first.")
    df = pd.read_csv(path)
    df = df[~df["prompt_id"].isin(EXCLUDE_PROMPTS)].copy()
    df = df[df["guessed_model"].notna()].copy()
    df["correct"] = df["correct"].astype(int)
    return df


def analyse(df: pd.DataFrame) -> dict:
    """Every cross-attribution quantity the paper cites, in one dict."""
    models = [m for m in MODELS
              if m in set(df["true_author"]) | set(df["guessed_model"])]

    matrix = (pd.crosstab(df["true_author"], df["guessed_model"])
              .reindex(index=models, columns=models, fill_value=0))
    row_totals = matrix.sum(axis=1)

    wrong = df[df["correct"] == 0]
    n_wrong = len(wrong)

    flows = []
    for author in models:
        for named in models:
            if author == named:
                continue
            k = int(matrix.at[author, named])
            if not k:
                continue
            flows.append({
                "true_author": author, "guessed_model": named, "n": k,
                "share_of_author_row": k / int(row_totals[author]),
                "share_of_all_errors": k / n_wrong if n_wrong else 0.0,
            })
    flows.sort(key=lambda r: -r["n"])

    non_self = df[df["judge"] != df["true_author"]]
    per_judge = {}
    for j in models:
        seen = df[df["judge"] == j]
        ns = seen[seen["true_author"] != j]
        guesses = seen["guessed_model"].value_counts(normalize=True)
        per_judge[j] = {
            "overall_accuracy": float(seen["correct"].mean()),
            "non_self_accuracy": float(ns["correct"].mean()),
            "n_non_self": len(ns),
            "guess_share": {m: float(guesses.get(m, 0.0)) for m in models},
            "top_target": str(guesses.idxmax()),
            "top_target_share": float(guesses.max()),
        }

    return {
        "n": len(df),
        "models": models,
        "matrix": {a: {b: int(matrix.at[a, b]) for b in models} for a in models},
        "row_rates": {a: {b: float(matrix.at[a, b] / row_totals[a])
                          for b in models} for a in models},
        "per_author_recall": {a: float(matrix.at[a, a] / row_totals[a])
                              for a in models},
        "n_wrong": n_wrong,
        "blame_share_on_errors": {
            m: float((wrong["guessed_model"] == m).mean()) if n_wrong else 0.0
            for m in models},
        "top_flows": flows[:6],
        "pooled_non_self_accuracy": float(non_self["correct"].mean()),
        "n_pooled_non_self": len(non_self),
        "per_judge": per_judge,
    }


def render(condition: str, res: dict) -> str:
    """The same numbers as `analyse`, laid out for a terminal."""
    models = res["models"]
    w = max(len(m) for m in models) + 2
    out = [f"CROSS-ATTRIBUTION -- {condition}  (n = {res['n']} judgments)", ""]

    out.append("[1] matrix: rows = true author, cols = guessed model (row %)")
    out.append("    " + " " * w + "".join(f"{m:>10}" for m in models) + f"{'recall':>10}")
    for a in models:
        cells = "".join(f"{res['row_rates'][a][b]:>9.1%} " for b in models)
        out.append(f"    {a:<{w}}{cells}{res['per_author_recall'][a]:>9.1%}")

    out.append("")
    out.append("[2] where each judge sends its guesses")
    out.append("    " + " " * w + "".join(f"{m:>10}" for m in models))
    for j in models:
        share = res["per_judge"][j]["guess_share"]
        out.append(f"    {j:<{w}}" + "".join(f"{share[m]:>9.1%} " for m in models))

    out.append("")
    out.append(f"[3] largest misattribution flows  (errors n = {res['n_wrong']})")
    for f in res["top_flows"]:
        out.append(f"    {f['true_author']:>9} -> {f['guessed_model']:<9}"
                   f"{f['n']:>5}   {f['share_of_author_row']:>6.1%} of that "
                   f"author's row   {f['share_of_all_errors']:>6.1%} of errors")

    out.append("")
    out.append("[4] accuracy on text the judge did not write")
    out.append(f"    {'judge':<{w}}{'non-self':>11}{'overall':>11}{'n':>7}")
    for j in models:
        p = res["per_judge"][j]
        out.append(f"    {j:<{w}}{p['non_self_accuracy']:>11.1%}"
                   f"{p['overall_accuracy']:>11.1%}{p['n_non_self']:>7}")
    out.append(f"    {'POOLED':<{w}}{res['pooled_non_self_accuracy']:>11.1%}"
               f"{'':>11}{res['n_pooled_non_self']:>7}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", choices=("lineup", "single", "both"),
                    default="lineup")
    ap.add_argument("--json", action="store_true",
                    help="write results/cross_attribution.json (implies --condition both)")
    args = ap.parse_args()

    conditions = (["lineup", "single"]
                  if args.json or args.condition == "both" else [args.condition])

    res = {}
    for c in conditions:
        res[c] = analyse(load(c))
        print(render(c, res[c]))
        print()

    if args.json:
        path = RESULTS_DIR / "cross_attribution.json"
        path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
