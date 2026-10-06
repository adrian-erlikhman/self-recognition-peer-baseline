"""CompLLM: what do judges SAY they are using?

    python src/analyze_reasons.py

Every judgment carries a one-sentence rationale (950 per condition, 1,900
total). Nothing else in the pipeline touches them. This script asks the
question they are actually good for:

    Do judges cite the cues that actually discriminate?

Engine A establishes that length proxies carry ~63% of the random forest's
feature importance. If judges' stated reasoning is dominated by tone, hedging
and voice instead, then their introspective account of the task is
systematically different from the signal that identifies authors, which is
the text-side version of the mechanism question Bai et al. raise and do not
test.

Method and its limits.

This is keyword matching against a hand-built lexicon, not semantic parsing.
It is a coarse instrument and the categories are ours, not the judges'. A
reason can match several categories or none. It cannot detect sarcasm,
negation ("not the long one"), or a cue described without any of our keywords.
Treat the output as a descriptive summary of stated rationales, not as ground
truth about judge cognition.

The lexicon is deliberately inspectable and lives in CUES below. Anyone can
disagree with a keyword and re-run.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    EXCLUDE_PROMPTS, JUDGMENTS_CSV, JUDGMENTS_SINGLE_CSV, MODELS, RESULTS_DIR,
)

OUT_JSON = RESULTS_DIR / "reasons_analysis.json"

# Cue categories. `family` maps each to how Engine A treats that kind of
# signal, so stated rationale can be compared against measured importance.
#   length  -> the feature family that dominates Engine A (63% of importance)
#   surface -> non-length stylistic features
#   content -> not a stylometric feature at all; about what was said
#   meta    -> naming a model, or expressing (un)certainty
CUES: dict[str, tuple[str, list[str]]] = {
    "length": ("length", [
        r"\blong(er|est)?\b", r"\bshort(er|est)?\b", r"\bbrief\b", r"\blength\b",
        r"\bverbose\b", r"\bwordy\b", r"\bconcise\b", r"\bterse\b",
        r"\bextended\b", r"\bexpansive\b", r"\bcompact\b",
    ]),
    "structure": ("length", [
        r"\bparagraph", r"\bstructur", r"\bformat", r"\bbullet", r"\blist\b",
        r"\bheading", r"\borganiz", r"\bsection", r"\blayout\b",
    ]),
    "sentence": ("length", [
        r"\bsentence", r"\bclause", r"\bsyntax", r"\bsyntactic",
        r"\bflowing\b", r"\bchoppy\b", r"\brun-on\b", r"\bcompound\b",
    ]),
    "tone": ("surface", [
        r"\btone\b", r"\bvoice\b", r"\bwarm", r"\bformal", r"\bcasual",
        r"\bdirect\b", r"\bmeasured\b", r"\bconfident", r"\bauthoritative",
        r"\breflective\b", r"\bconversational\b", r"\bpersonable\b",
    ]),
    "hedging": ("surface", [
        r"\bhedg", r"\bqualif", r"\bnuance", r"\bcaveat", r"\btentative",
        r"\bcautious", r"\bequivocat", r"\bbalanced\b",
    ]),
    "vocabulary": ("surface", [
        r"\bvocabular", r"\bword choice\b", r"\blexical", r"\bdiction\b",
        r"\bjargon\b", r"\bterminolog", r"\bphrasing\b", r"\bwording\b",
        r"\bidiom", r"\bregister\b",
    ]),
    "punctuation": ("surface", [
        r"\bdash", r"\bem-dash\b", r"\bcomma", r"\bsemicolon", r"\bcolon\b",
        r"\bpunctuation\b", r"\bparenthe",
    ]),
    "readability": ("surface", [
        r"\bsimple\b", r"\bcomplex", r"\baccessible\b", r"\bdense\b",
        r"\breadab", r"\btechnical\b", r"\bplain\b", r"\bstraightforward\b",
    ]),
    "content": ("content", [
        r"\bexample", r"\bconcrete\b", r"\bdetail", r"\bspecific",
        r"\bargument", r"\bevidence\b", r"\bpoint\b", r"\bsubstance\b",
        r"\bdepth\b", r"\bthorough",
    ]),
    "names_a_model": ("meta", [
        r"\bgpt\b", r"\bchatgpt\b", r"\bopenai\b", r"\bclaude\b",
        r"\banthropic\b", r"\bgemini\b", r"\bgoogle\b", r"\bgrok\b",
        r"\bdeepseek\b",
    ]),
    "uncertainty": ("meta", [
        r"\bguess", r"\bnot sure\b", r"\bunsure\b", r"\bhard to\b",
        r"\bdifficult to\b", r"\buncertain", r"\bmight\b", r"\bpossibly\b",
    ]),
}

COMPILED = {k: (fam, [re.compile(p, re.I) for p in pats])
            for k, (fam, pats) in CUES.items()}


def tag(text: str) -> set[str]:
    return {k for k, (_, pats) in COMPILED.items()
            if any(p.search(text) for p in pats)}


def load(condition: str) -> pd.DataFrame:
    path = JUDGMENTS_CSV if condition == "lineup" else JUDGMENTS_SINGLE_CSV
    df = pd.read_csv(path)
    df = df[~df["prompt_id"].isin(EXCLUDE_PROMPTS)]
    df = df[df["reason"].notna()].copy()
    df["reason"] = df["reason"].astype(str)
    df = df[df["reason"].str.strip() != ""].copy()
    df["cues"] = df["reason"].map(tag)
    for k in CUES:
        df[f"cue_{k}"] = df["cues"].map(lambda s, k=k: k in s)
    return df


def analyse(condition: str, df: pd.DataFrame) -> dict:
    n = len(df)
    bar = "=" * 74
    print(f"\n{bar}\nSTATED RATIONALES - {condition.upper()}\n{bar}")
    print(f"{n} reasons, mean {df['reason'].str.split().str.len().mean():.1f} words")

    # --- which cues are cited, and do they differ when the judge is right? ---
    rows = []
    for k, (fam, _) in CUES.items():
        col = f"cue_{k}"
        share = float(df[col].mean())
        r = df[df["correct"] == 1][col]
        w = df[df["correct"] != 1][col]
        if len(r) and len(w) and (r.sum() + w.sum()) > 0:
            table = [[int(r.sum()), int(len(r) - r.sum())],
                     [int(w.sum()), int(len(w) - w.sum())]]
            _, p = stats.fisher_exact(table)
        else:
            p = float("nan")
        rows.append({"cue": k, "family": fam, "share": share,
                     "n": int(df[col].sum()),
                     "share_when_correct": float(r.mean()) if len(r) else None,
                     "share_when_wrong": float(w.mean()) if len(w) else None,
                     "fisher_p": float(p)})
    rows.sort(key=lambda d: -d["share"])

    print(f"\n-- cited cues --\n  {'cue':14s}{'family':9s}{'share':>8}"
          f"{'correct':>9}{'wrong':>8}{'p':>10}")
    for d in rows:
        c = "n/a" if d["share_when_correct"] is None else f"{d['share_when_correct']:.1%}"
        w = "n/a" if d["share_when_wrong"] is None else f"{d['share_when_wrong']:.1%}"
        print(f"  {d['cue']:14s}{d['family']:9s}{d['share']:8.1%}{c:>9}{w:>8}"
              f"{d['fisher_p']:10.3f}")

    # --- the comparison that matters: stated family vs Engine A importance ---
    fam_share = {}
    for fam in ("length", "surface", "content", "meta"):
        cols = [f"cue_{k}" for k, (f, _) in CUES.items() if f == fam]
        fam_share[fam] = float(df[cols].any(axis=1).mean())
    print(f"\n-- rationale by feature family (a reason can hit several) --")
    for fam, v in fam_share.items():
        print(f"  {fam:9s}{v:8.1%}")

    n_none = int((~df[[f"cue_{k}" for k in CUES]].any(axis=1)).sum())
    print(f"  {'uncategorised':9s}{n_none / n:8.1%}  ({n_none} reasons matched "
          f"no keyword - the lexicon's blind spot)")

    # --- does naming a model in the rationale track naming it as the guess? ---
    named = df[df["cue_names_a_model"]]
    print(f"\n-- {len(named)} reasons name a model explicitly "
          f"({len(named) / n:.1%}) --")

    return {
        "condition": condition,
        "n_reasons": n,
        "mean_words": float(df["reason"].str.split().str.len().mean()),
        "cues": rows,
        "family_share": fam_share,
        "uncategorised_share": n_none / n,
        "names_a_model_share": float(len(named) / n),
    }


def main() -> None:
    out = {}
    for cond in ("lineup", "single"):
        out[cond] = analyse(cond, load(cond))

    # Contrast against what actually discriminates, if Engine A has run.
    ea = RESULTS_DIR / "engine_a_results.json"
    if ea.exists():
        share = json.loads(ea.read_text(encoding="utf-8")).get(
            "length_proxy_importance_share")
        if share is not None:
            print(f"\n{'=' * 74}\nSTATED vs MEASURED\n{'=' * 74}")
            print(f"  Engine A: length proxies are {share:.1%} of RF feature "
                  f"importance.")
            for cond in out:
                print(f"  {cond:8s} rationales citing a length-family cue: "
                      f"{out[cond]['family_share']['length']:.1%}   "
                      f"surface-family: {out[cond]['family_share']['surface']:.1%}")
            out["engine_a_length_proxy_importance"] = share
            print("\n  Read this as a descriptive contrast between what judges "
                  "say they use\n  and what the classifier finds informative - "
                  "not as evidence about\n  what judges internally compute. The "
                  "lexicon is a coarse instrument.")

    RESULTS_DIR.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
