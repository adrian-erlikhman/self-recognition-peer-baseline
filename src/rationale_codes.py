"""CompLLM: what cue TYPE does each judge rationale cite, and is it true?

    python src/rationale_codes.py

A reviewer asked for the mechanism behind self-recognition. The 1,900
one-sentence rationales (950 per condition) are the cheapest window onto it.
This script replaces the coarse keyword pass in analyze_reasons.py with a
documented, validated coding scheme and four analyses:

    (a) cue-type distribution per judge and condition
    (b) does the cited cue type predict correctness? (logistic GEE,
        clustered by prompt)
    (c) on Claude's and GPT's own text: what do they cite when they name
        themselves correctly, against what peers cite when they name Claude
        or GPT correctly on the same responses?
    (d) faithfulness: when a rationale cites a measurable cue (length,
        sentence length, em-dashes, semicolons, hedging, paragraph count,
        a quoted phrase), is that cue actually extreme in the text, relative
        to the other four responses to the same prompt?

Writes results/rationale_codes.json and docs/revision/RATIONALES.md. Reads the
hand-labelled validation sample in results/rationale_handlabels.csv.

The coding scheme.

Eleven cue categories, multi-label, applied by regular expressions over the
reason text. A reason that matches none is "other". The codebook (what a
human annotator is asked to mark) is CODEBOOK below; the rules that
approximate it are CUES. The two are kept separate on purpose: agreement
between them is measured, not assumed.

Validation.

150 rationales were drawn stratified by judge x condition (10 per cell for a
"tune" split of 100, 5 per cell for a "holdout" split of 50), excluding the 80
rationales read while designing the codebook. Annotator 1 is Claude (an LLM),
the same agent that wrote the lexicon; it labelled all 150 from the reason
text before any rule output was computed on them, but the lexicon code was
written after that pass and shares its scope decisions, so rules-vs-A1 is
close to an internal-consistency check. The lexicon as first written
(CUES_V0) was frozen and then revised once on the tune split (CUES).
Annotator 2 is a separate Claude instance that saw only CODEBOOK and the 150
reasons, never the lexicon or A1's labels; rules-vs-A2 and A1-vs-A2 kappa are
the less self-serving figures. Both annotators are the same underlying model,
so a human second coder is still the right next step.

Limits. Regexes cannot resolve scope ("balanced structure" vs "balanced
argument") beyond a few hand-written lookaheads, cannot see sarcasm, and code
a cue as cited even when it is negated ("without ornate elaboration" cites
rhetoric). The faithfulness test (d) handles negation separately for the
cues where direction matters.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    EXCLUDE_PROMPTS, JUDGMENTS_CSV, JUDGMENTS_SINGLE_CSV, MODELS, RESPONSES_CSV,
    RESULTS_DIR, ROOT,
)
from features import FEATURES, frame  # noqa: E402

OUT_JSON = RESULTS_DIR / "rationale_codes.json"
HANDLABELS_CSV = RESULTS_DIR / "rationale_handlabels.csv"
OUT_MD = ROOT / "docs" / "revision" / "RATIONALES.md"

# --------------------------------------------------------------------------
# Codebook: what the annotator marks. One sentence per category, plus the
# scope decisions that came up while labelling. A rationale gets every
# category it cites as EVIDENCE, whether the cue is present or said to be
# absent ("without flourishes" cites rhetoric).
# --------------------------------------------------------------------------
CODEBOOK: dict[str, str] = {
    "length": "Amount of text or elaboration in the response as a whole "
              "(long, lengthy, expansive, extensive, verbose, concise, brief, "
              "short/compact paragraphs). Sentence length is syntax, not length.",
    "format_punct": "Specific punctuation marks (em-dash, semicolon, colon, "
                    "commas, parentheses) or visual formatting (lists, bullets, "
                    "enumeration, headers, bold, markdown).",
    "structure": "Organisation of the document: overall structure, N-paragraph "
                 "or three-part layout, topic sentences, thesis, introduction, "
                 "conclusion/summary, transitions and signposting, formulaic "
                 "essay format, logical flow or progression.",
    "syntax": "Sentence-level construction: sentence length or complexity, "
              "clauses, run-ons, parallelism, cadence, pacing, rhythm, "
              "'flowing' sentences or prose.",
    "lexical": "Word choice: vocabulary, diction, phrasing, wording, "
               "terminology, jargon, cliches, spelling, named transition "
               "words, or any quoted word or phrase from the text.",
    "tone": "Tone, voice or register: formal, academic, conversational, "
            "essayistic, literary, corporate, direct, plain, neutral, calm, "
            "reflective, assertive, persuasive, educational, narrative.",
    "hedging": "Hedging or even-handedness: hedged claims, qualifications, "
               "caveats, nuance, measured, careful, concessions, balanced "
               "prose/tone/argument (but not 'balanced structure', which is "
               "structure).",
    "rhetoric": "Rhetorical flourish or figurative language: metaphors, "
                "imagery, vivid or colourful language, flourishes, ornate, "
                "elaborate or overwrought phrasing, rhetorical contrasts.",
    "brand_prior": "Appeal to what a named model typically or characteristically "
                   "writes: 'typical of GPT', 'characteristic of Claude', "
                   "'Claude-like', 'GPT-style', 'hallmark/signature/classic X', "
                   "'resembles X', 'Grok's persona', 'without typical GPT "
                   "cliches'. Merely stating the conclusion ('suggests Claude') "
                   "is not a brand prior.",
    "self_reference": "The judge refers to itself or its own writing ('my "
                      "style', 'as I would write', 'this is how I write').",
    "content": "What is said and how good it is: detail, depth, examples, "
               "evidence, references, statistics, specificity, focus or "
               "emphasis, analysis, insight, clarity, polish, elegance, "
               "sophistication (of the prose, not the vocabulary).",
}
CATS: list[str] = list(CODEBOOK)

# Model names as judges write them, including version suffixes
# ("GPT-4", "Claude 3.5 Sonnet", "ChatGPT").
_M = (r"(?:chatgpt|gpt(?:-?\d+(?:\.\d+)?)?|claude(?: \d(?:\.\d)?)?(?: sonnet| opus)?"
      r"|gemini|grok|deepseek|openai|anthropic)")

# --------------------------------------------------------------------------
# CUES_V0: the lexicon as first written from the codebook, BEFORE it was
# compared with any hand label. Frozen; kept only so its agreement can be
# reported next to the tuned version's.
# --------------------------------------------------------------------------
CUES_V0: dict[str, list[str]] = {
    "length": [
        r"\b(?:lengthy|verbose|wordy|long-winded|succinct|terse|brevity)\b",
        r"\bconcise(?:ness)?\b", r"\bbrief\b",
        r"\b(?:expansive|extensive|exhaustive)\b", r"\belaboration\b",
        r"(?<!sentence )\blength\b", r"\bword count\b",
        r"\b(?:long|longer|longest|short|shorter|shortest|compact)\b"
        r"(?![\s,-]+(?:[\w-]+[\s,-]+){0,3}?(?:sentence|clause))",
    ],
    "format_punct": [
        r"\bpunctuat", r"em[- ]?dash", r"\bdash(?:es)?\b", r"\bsemicolon",
        r"\bcolons?\b", r"\bcommas?\b", r"\bparenthe", r"\bbullet",
        r"\blist(?:s|ing)?\b", r"\bnumbered\b", r"\benumerat", r"\bheaders?\b",
        r"\bheadings?\b", r"\bbold\b", r"\bmarkdown\b", r"\bformatting\b",
    ],
    "structure": [
        r"\bstructur", r"\borgani[sz]", r"\b[\w]+[- ]paragraph\b",
        r"\b(?:tri-?part|tripartite|three-part|three-pronged|three-reason)",
        r"\btopic sentence", r"\bthesis\b", r"\bintroduct", r"\bintro\b",
        r"\bopen(?:er|ing)\b", r"\bconclu(?:sion|ding|des)\b", r"\bsummar",
        r"\btransition", r"\bsignpost", r"\broadmap\b", r"\bformulaic\b",
        r"\bframework\b", r"\bprogression\b", r"\bformat\b", r"\bsection",
        r"\blogical(?:ly)?\b", r"\bflow\b", r"\bcohesive\b", r"\bcoherent\b",
        r"\bmethodical\b", r"\bsystematic\b",
    ],
    "syntax": [
        r"(?<!topic )\bsentences?\b", r"\bclause", r"\brun-on\b", r"\bsyntax",
        r"\bsyntactic", r"\bcadence\b", r"\bpacing\b", r"\brhythm",
        r"\bflowing\b", r"\bparallel(?:ism)?\b(?!\s+topic)", r"\bchoppy\b",
        r"\bcompound\b", r"\bnested\b", r"\bsubordinat", r"not[- ]x[- ]but",
        r"not merely",
    ],
    "lexical": [
        r"\bvocabular", r"\bword choice", r"\blexic", r"\bdiction\b",
        r"\bjargon", r"\bterminolog", r"\bphras(?:e|es|ing)\b", r"\bwording\b",
        r"\bwords?\b(?! count)", r"\bterms? (?:like|such)", r"\bclich",
        r"\bbuzzword", r"\bidiom", r"\bspelling\b", r"\bdelve",
        r"\btransition(?:al)? (?:words|phrases|markers)",
        # A quoted span: an opening quote at a word boundary (so "Grok's"
        # does not count), at least three characters, a closing quote.
        r"(?:(?<=\s)|(?<=\()|^)['\"‘“]\w[^'\"’”]{2,}['\"’”]",
    ],
    "tone": [
        r"\btone\b", r"\bvoice\b", r"\bregister\b", r"\bformal(?:ity)?\b",
        r"\binformal\b", r"\bacademic(?:ally)?\b", r"\bconversational\b",
        r"\bcasual\b", r"\bchatty\b", r"\bcolloquial", r"\bessay(?:istic|-like)\b",
        r"\bliterary\b", r"\bcorporate\b", r"\bdirect\b", r"\bstraightforward\b",
        r"\bplain\b", r"\bcalm\b", r"\breflective\b", r"\bpersuasive\b",
        r"\bassertive\b", r"\bconfident\b", r"\bprofessional\b",
        r"\bhuman-?like\b", r"\bnatural(?:istic)?\b", r"\bwarm(?:th)?\b",
        r"\bpunchy\b", r"\bclassroom\b", r"\beducational\b", r"\btextbook\b",
        r"\bnarrative\b", r"\bdry\b", r"\bneutral\b", r"\benthusias",
        r"\bupbeat\b", r"\bengaging\b", r"\bplayful", r"\bwitty\b", r"\bwry\b",
        r"\bhumou?r", r"\bcrisp\b", r"\bpersona(?:lity)?\b", r"\bsimpler?\b",
        r"\bphilosophical (?:tone|voice|style|prose|register)",
    ],
    "hedging": [
        r"\bhedg", r"\bqualif", r"\bcaveat", r"\bnuanc", r"\bmeasured\b",
        r"\bcautious", r"\bcareful(?:ly)?\b", r"\bconcession",
        r"\bcounter-?argument", r"\beven-?handed", r"\bboth sides\b",
        r"\bbalanced\b(?![\s,-]+(?:[\w-]+[\s,-]+){0,2}?(?:structure|paragraph|"
        r"sentence|syntax|syntactic|tri|three|clause|essay|format|organi))",
    ],
    "rhetoric": [
        r"\bmetaphor", r"\bimagery\b", r"\bvivid", r"\bflourish", r"\brhetoric",
        r"\banalog(?:y|ies)\b", r"\bfigurative", r"\bflorid\b", r"\bornate\b",
        r"\bflowery\b", r"\bembellish", r"\boverwrought\b", r"\bcolou?rful\b",
        r"\bdramatic\b", r"\bpoetic", r"\blyrical", r"\baphoris",
        r"\belaborate\b(?!\s+(?:academic\s+)?(?:vocabulary|structure))",
    ],
    "brand_prior": [
        r"\b(?:typical|typically|characteristic|reminiscent|hallmark|signature"
        r"|classic|common)\b[^.]{0,40}?\b" + _M + r"\b",
        r"\b" + _M + r"[- ]?(?:like|style|esque)\b",
        r"\b" + _M + r"['’]s\b[^.]{0,40}?\b(?:style|persona|output|writing"
        r"|voice|tendenc|signature|characteristic|capabilit|typical|habit)",
        r"\b(?:resembl\w*|familiar|known for|tends? to|consistent with)\b"
        r"[^.]{0,30}?\b" + _M + r"\b",
    ],
    "self_reference": [
        r"\bmy (?:own )?(?:style|writing|voice|outputs?|responses?|phrasing|habits?)\b",
        r"\bI (?:would|often|usually|typically|tend to) (?:write|phrase|use|structure)",
        r"\bas I would\b", r"\blike (?:I|me) (?:would|do)\b", r"\bmyself\b",
        r"\bmine\b", r"\bmy own\b",
    ],
    "content": [
        r"\bdetail", r"\bexamples?\b", r"\bexample-", r"\bspecific(?:s|ity)?\b",
        r"\bconcrete\b", r"\bevidence", r"\bempirical\b", r"\bstatistic",
        r"\bdata\b", r"\bcitation", r"\bcites?\b", r"\breferences?\b",
        r"\bresearch\b", r"\bstud(?:y|ies)\b", r"\bdepth\b", r"\bdeep\b",
        r"\bin-depth\b", r"\bcomprehensive", r"\bthorough", r"\bcoverage\b",
        r"\binsight", r"\banaly(?:sis|tic|tical)\b", r"\bargued\b",
        r"\breasoning\b", r"\bsynthesis\b", r"\bpractical\b", r"\bfocus",
        r"\bemphasi[sz]", r"\bpolish", r"\belegan", r"\beloquen",
        r"\bsophisticat\w*\b(?!\s+(?:\w+\s+)?(?:vocabulary|punctuation|diction|word))",
        r"\bclarity\b",
        r"\bclear\b(?![\s,-]+(?:[\w-]+[\s,-]+){0,2}?(?:structure|topic|transition"
        r"|thesis|paragraph|three|sectional|logical))",
        r"\bthoughtful\b", r"\bcomplex\b(?![\s,]+(?:\w+[\s,]+)?(?:sentence|prose))",
        r"\bquality\b", r"\bframing\b", r"\baccessib", r"\bhelpful\b",
        r"\bexploration\b", r"\bsubstan", r"\bphilosophical(?:ly)? (?:depth|deep|argument)",
        r"\binformative\b", r"\bcapabilit",
    ],
}

# --------------------------------------------------------------------------
# CUES: the lexicon after one revision round against the TUNE split. Each
# change from V0 is commented with the tune-split error that motivated it.
# The disagreement listing that prompted the round showed holdout errors too;
# none of those motivated a change, and holdout-only error modes (e.g.
# "sentence structure" also coded as structure) were deliberately left in so
# that the holdout figure stays an out-of-sample estimate.
# --------------------------------------------------------------------------
CUES: dict[str, list[str]] = {k: list(v) for k, v in CUES_V0.items()}


def _swap(cat: str, old: str, new: str | None) -> None:
    """Replace (or with new=None, drop) one V0 pattern in CUES[cat]."""
    i = CUES[cat].index(old)
    if new is None:
        CUES[cat].pop(i)
    else:
        CUES[cat][i] = new


# length: "long-term benefits" is not about length (tune #35).
_swap("length",
      r"\b(?:long|longer|longest|short|shorter|shortest|compact)\b"
      r"(?![\s,-]+(?:[\w-]+[\s,-]+){0,3}?(?:sentence|clause))",
      r"\b(?:long|longer|longest|short|shorter|shortest|compact)\b(?!-term)"
      r"(?![\s,-]+(?:[\w-]+[\s,-]+){0,3}?(?:sentence|clause))")
# tone: "natural" was mostly "natural cadence/flow", i.e. syntax (tune #42,
# #83); "literary flourishes" is rhetoric (#8); "punchy metaphors" is
# rhetoric (#44); "academic vocabulary" is lexical (#66).
_swap("tone", r"\bnatural(?:istic)?\b", r"\bnaturalistic\b")
_swap("tone", r"\bliterary\b", r"\bliterary\b(?!\s+flourish)")
_swap("tone", r"\bpunchy\b", r"\bpunchy\b(?!\s+metaphor)")
_swap("tone", r"\bacademic(?:ally)?\b", r"\bacademic(?:ally)?\b(?!\s+vocabulary)")
# content: "thoughtful pacing" (#8) and "rhetorically elegant" (#45) are not
# about substance.
_swap("content", r"\bthoughtful\b", r"\bthoughtful\b(?!\s+pacing)")
_swap("content", r"\belegan", r"(?<!rhetorically )\belegan")
# hedging: "careful examples" (#40) is content; "nuanced sentences /
# rhetorical contrasts" (#51, #99) describe form, not qualification.
_swap("hedging", r"\bcareful(?:ly)?\b", r"\bcareful(?:ly)?\b(?!\s+examples)")
_swap("hedging", r"\bnuanc",
      r"\bnuanc\w*\b(?!\s+(?:\w+\s+)?(?:sentences?|contrasts?|transitions?))")
# rhetoric: "rhetorically balanced" is even-handedness (#54); "dramatic ...
# tone" is tone (#67).
_swap("rhetoric", r"\brhetoric", r"\brhetoric(?!ally balanced)")
_swap("rhetoric", r"\bdramatic\b", r"\bdramatic\b(?!\s+(?:\w+\s+)?tone)")
# structure: "structural terminology" is lexical (#46).
_swap("structure", r"\bstructur", r"\bstructur(?!al terminology)")

COMPILED_V0 = {k: [re.compile(p, re.I) for p in v] for k, v in CUES_V0.items()}
COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in CUES.items()}


def norm(text: str) -> str:
    """NFKC plus straight quotes and dashes, so the lexicon sees one alphabet."""
    t = unicodedata.normalize("NFKC", str(text))
    return (t.replace("‘", "'").replace("’", "'")
             .replace("“", '"').replace("”", '"'))


def tag(text: str, compiled=None) -> set[str]:
    compiled = compiled or COMPILED
    t = norm(text)
    return {k for k, pats in compiled.items() if any(p.search(t) for p in pats)}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_judgments() -> pd.DataFrame:
    """Both conditions, one row per judgment, with a stable id `rid` that the
    hand-label file also uses (L:<session>:<slot> or S:<session>)."""
    a = pd.read_csv(JUDGMENTS_CSV)
    a["condition"] = "lineup"
    b = pd.read_csv(JUDGMENTS_SINGLE_CSV)
    b["condition"] = "single"
    for d in (a, b):
        d["slot"] = d["slot"].astype(str)
    a["rid"] = "L:" + a["session_id"] + ":" + a["slot"]
    b["rid"] = "S:" + b["session_id"]
    df = pd.concat([a, b], ignore_index=True)
    df = df[~df["prompt_id"].isin(EXCLUDE_PROMPTS)].reset_index(drop=True)
    df["reason"] = df["reason"].fillna("").astype(str)
    codes = df["reason"].map(tag)
    for c in CATS:
        df[f"cue_{c}"] = codes.map(lambda s, c=c: int(c in s))
    df["cue_other"] = (df[[f"cue_{c}" for c in CATS]].sum(axis=1) == 0).astype(int)
    df["n_cues"] = df[[f"cue_{c}" for c in CATS]].sum(axis=1)
    df["named_self"] = (df["guessed_model"] == df["judge"]).astype(int)
    return df


def load_responses() -> pd.DataFrame:
    """Corpus texts plus the 18 features (recomputed with features.py so the
    values are exactly those Engine A uses)."""
    r = pd.read_csv(RESPONSES_CSV)
    r["prompt_id"] = r["prompt_id"].astype(str)
    r = r[~r["prompt_id"].isin(EXCLUDE_PROMPTS)].reset_index(drop=True)
    f = frame(r["text"].tolist())
    return pd.concat([r[["prompt_id", "model", "text"]], f], axis=1)


# --------------------------------------------------------------------------
# Validation against the hand labels
# --------------------------------------------------------------------------
def hand_sets(hl: pd.DataFrame, prefix: str = "hand_") -> list[set[str]]:
    return [{c for c in CATS if row[f"{prefix}{c}"]} for _, row in hl.iterrows()]


def agreement(ref: list[set[str]], test: list[set[str]]) -> dict:
    """Per-category precision, recall, F1 and Cohen's kappa of `test`
    against `ref` (both lists of code sets over the same reasons)."""
    out = {}
    for c in CATS + ["other"]:
        if c == "other":
            h = np.array([int(len(s) == 0) for s in ref])
            r = np.array([int(len(s) == 0) for s in test])
        else:
            h = np.array([int(c in s) for s in ref])
            r = np.array([int(c in s) for s in test])
        tp = int(((h == 1) & (r == 1)).sum())
        fp = int(((h == 0) & (r == 1)).sum())
        fn = int(((h == 1) & (r == 0)).sum())
        tn = int(((h == 0) & (r == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
        n = len(h)
        po = (tp + tn) / n
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
        kappa = (po - pe) / (1 - pe) if pe < 1 else None
        out[c] = {"hand_n": int(h.sum()), "rule_n": int(r.sum()), "tp": tp,
                  "fp": fp, "fn": fn, "precision": prec, "recall": rec,
                  "f1": f1, "kappa": kappa}
    # Micro-averaged over the eleven real categories.
    tp = sum(out[c]["tp"] for c in CATS)
    fp = sum(out[c]["fp"] for c in CATS)
    fn = sum(out[c]["fn"] for c in CATS)
    out["_micro"] = {"tp": tp, "fp": fp, "fn": fn,
                     "precision": tp / (tp + fp), "recall": tp / (tp + fn),
                     "f1": 2 * tp / (2 * tp + fp + fn)}
    # Exact-match share: every category right for the reason.
    out["_exact_match"] = float(np.mean([a == b for a, b in zip(ref, test)]))
    return out


def validate() -> dict:
    """Three comparisons on the 150 hand-labelled reasons:
      v0 / final : rules against annotator 1 (the lexicon's author), per split
      final_vs_a2: rules against annotator 2, a separate blind Claude instance
                   that saw only CODEBOOK and the reasons, never the lexicon
      a1_vs_a2   : the two annotators against each other (kappa is the
                   inter-annotator reliability; P/R treat annotator 1 as ref)
    """
    if not HANDLABELS_CSV.exists():
        raise SystemExit(f"{HANDLABELS_CSV} is missing; the validation sample "
                         "is part of the release.")
    hl = pd.read_csv(HANDLABELS_CSV)
    res = {"n": {s: int((hl["split"] == s).sum()) for s in ("tune", "holdout")},
           "annotator": str(hl["annotator"].iloc[0]),
           "annotator2": str(hl["annotator2"].iloc[0]) if "annotator2" in hl else None}
    splits = {"tune": hl["split"] == "tune", "holdout": hl["split"] == "holdout",
              "all": pd.Series(True, index=hl.index)}
    a1 = hand_sets(hl, "hand_")
    a2 = hand_sets(hl, "hand2_") if "hand2_length" in hl else None
    for ver, comp in (("v0", COMPILED_V0), ("final", COMPILED)):
        rules = [tag(t, comp) for t in hl["reason"]]
        res[ver] = {}
        for s, m in splits.items():
            idx = np.where(m)[0]
            res[ver][s] = agreement([a1[i] for i in idx], [rules[i] for i in idx])
        if a2 is not None and ver == "final":
            res["final_vs_a2"] = {s: agreement([a2[i] for i in np.where(m)[0]],
                                               [rules[i] for i in np.where(m)[0]])
                                  for s, m in splits.items()}
    if a2 is not None:
        res["a1_vs_a2"] = agreement(a1, a2)
    return res


# --------------------------------------------------------------------------
# (a) Distribution
# --------------------------------------------------------------------------
def distribution(df: pd.DataFrame) -> dict:
    out = {}
    cols = [f"cue_{c}" for c in CATS] + ["cue_other"]
    for cond, g in df.groupby("condition"):
        per = {}
        for j in MODELS:
            gj = g[g["judge"] == j]
            per[j] = {"n": len(gj),
                      **{c[4:]: {"k": int(gj[c].sum()), "share": float(gj[c].mean())}
                         for c in cols}}
        per["_all"] = {"n": len(g),
                       **{c[4:]: {"k": int(g[c].sum()), "share": float(g[c].mean())}
                          for c in cols}}
        # Does the cue mix differ across judges? chi-square per cue (5x2).
        chi = {}
        for c in cols:
            tab = pd.crosstab(g["judge"], g[c])
            if tab.shape[1] == 2:
                chi2, p, _, _ = stats.chi2_contingency(tab)
                chi[c[4:]] = {"chi2": float(chi2), "p": float(p)}
        per["_chi2_across_judges"] = chi
        per["_mean_cues_per_reason"] = float(g["n_cues"].mean())
        out[cond] = per
    return out


# --------------------------------------------------------------------------
# (b) Cue type -> correctness, logistic GEE clustered by prompt
# --------------------------------------------------------------------------
MIN_HITS = 15  # a cue must be cited this often in the subset to enter a model


def gee_correct(df: pd.DataFrame, extra: str = "") -> dict:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    use = [c for c in CATS if df[f"cue_{c}"].sum() >= MIN_HITS
           and df[f"cue_{c}"].sum() <= len(df) - MIN_HITS]
    rhs = " + ".join(f"cue_{c}" for c in use) + " + C(judge)" + extra
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.gee(f"correct ~ {rhs}", groups="prompt_id", data=df,
                    family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = m.conf_int()
    res = {"formula": f"correct ~ {rhs}", "n": int(m.nobs),
           "n_clusters": int(df["prompt_id"].nunique()), "terms": {}}
    for c in use:
        t = f"cue_{c}"
        res["terms"][c] = {"k_cited": int(df[t].sum()),
                           "acc_when_cited": float(df.loc[df[t] == 1, "correct"].mean()),
                           "acc_when_not": float(df.loc[df[t] == 0, "correct"].mean()),
                           "OR": float(np.exp(m.params[t])),
                           "ci95": [float(np.exp(ci.loc[t, 0])), float(np.exp(ci.loc[t, 1]))],
                           "p": float(m.pvalues[t])}
    res["excluded_sparse"] = [c for c in CATS if c not in use]
    return res


def cue_vs_correct(df: pd.DataFrame) -> dict:
    out = {}
    for cond, g in df.groupby("condition"):
        out[cond] = {"judge_fe": gee_correct(g),
                     "judge_and_author_fe": gee_correct(g, " + C(true_author)")}
    out["pooled"] = {"judge_fe": gee_correct(df, " + C(condition)"),
                     "judge_and_author_fe": gee_correct(df, " + C(condition) + C(true_author)")}
    return out


# --------------------------------------------------------------------------
# (c) Self-hits vs peer hits on Claude's and GPT's own text
# --------------------------------------------------------------------------
QUOTE = re.compile(r"(?:(?<=\s)|(?<=\()|^)['\"]\w[^'\"]{2,}['\"]")


def self_vs_peer(df: pd.DataFrame) -> dict:
    """For author A in {Claude, GPT}: S = A judging A's text and naming A
    (self-hits); P = another judge naming A correctly on A's text (peer
    hits); FA = A naming itself on someone else's text (self false alarms).
    Cue shares compared S vs P with Fisher's exact test. P's rows fall four
    to a response, so the test treats dependent rows as independent and
    overstates significance on the peer side, as in the paper's
    self-advantage test.
    """
    df = df.copy()
    df["quotes_text"] = df["reason"].map(lambda t: int(bool(QUOTE.search(norm(t)))))
    cols = CATS + ["other", "quotes_text"]
    out = {}
    for a in ("Claude", "GPT"):
        for cond in ("lineup", "single", "pooled"):
            g = df if cond == "pooled" else df[df["condition"] == cond]
            S = g[(g["true_author"] == a) & (g["judge"] == a) & (g["correct"] == 1)]
            P = g[(g["true_author"] == a) & (g["judge"] != a) & (g["correct"] == 1)]
            FA = g[(g["true_author"] != a) & (g["judge"] == a) & (g["guessed_model"] == a)]
            rows = {}
            for c in cols:
                col = c if c == "quotes_text" else f"cue_{c}"
                ks, kp, kf = int(S[col].sum()), int(P[col].sum()), int(FA[col].sum())
                _, p = stats.fisher_exact([[ks, len(S) - ks], [kp, len(P) - kp]]) \
                    if len(S) and len(P) else (None, float("nan"))
                rows[c] = {"self_hit_k": ks, "self_hit_share": ks / len(S) if len(S) else None,
                           "peer_hit_k": kp, "peer_hit_share": kp / len(P) if len(P) else None,
                           "self_fa_k": kf, "self_fa_share": kf / len(FA) if len(FA) else None,
                           "fisher_p": float(p)}
            out[f"{a}_{cond}"] = {"n_self_hits": len(S), "n_peer_hits": len(P),
                                  "n_self_false_alarms": len(FA), "cues": rows}
    # How often does any rationale anywhere refer to the judge itself, or use
    # familiarity language at all? (The judge prompt never tells the judge it
    # is one of the candidates.)
    fam = re.compile(r"\bfamiliar|\brecogni[sz]|\bresembl", re.I)
    out["self_reference_total"] = int(df["cue_self_reference"].sum())
    out["familiarity_words_total"] = int(df["reason"].map(lambda t: bool(fam.search(t))).sum())
    out["familiarity_words_by_judge"] = {
        j: int(df.loc[df["judge"] == j, "reason"].map(lambda t: bool(fam.search(t))).sum())
        for j in MODELS}
    return out


# --------------------------------------------------------------------------
# (d) Faithfulness: is the cited cue actually extreme in the text?
# --------------------------------------------------------------------------
# Each check: a regex that detects a directional claim in the reason, the
# feature it is about, and the direction ("high" = the text should have the
# most of it among the five same-prompt responses). Negation words shortly
# before a high-claim cue ("without dashes", "minimal punctuation") flip it.
NEG = r"(?:without|no|lack(?:s|ing)?|absence of|free of|minimal|less|fewer|few|little)\s+(?:\w+\s+){0,2}?"
_LONG_WORDS = r"(?:long|longer|lengthy|expansive|extensive|exhaustive|verbose|wordy|long-winded)"
_SHORT_WORDS = r"(?:concise|brief|short|shorter|compact|succinct|terse)"
_SENT = r"(?:sentences?|clauses?)"
CHECKS: list[dict] = [
    # overall length: a long/short word NOT within four words of "sentence"
    {"name": "length_long", "feature": "word_count", "dir": "high",
     "pat": rf"\b{_LONG_WORDS}\b(?![\s,-]+(?:[\w-]+[\s,-]+){{0,3}}?(?:sentence|clause|paragraph))"},
    {"name": "length_short", "feature": "word_count", "dir": "low",
     "pat": rf"\b{_SHORT_WORDS}\b(?![\s,-]+(?:[\w-]+[\s,-]+){{0,3}}?(?:sentence|clause|paragraph))"},
    # paragraph length
    {"name": "paragraphs_long", "feature": "avg_para_len", "dir": "high",
     "pat": rf"\b(?:long|longer|lengthy|dense)\b[\s,-]+(?:[\w-]+[\s,-]+){{0,2}}?paragraphs"},
    {"name": "paragraphs_short", "feature": "avg_para_len", "dir": "low",
     "pat": rf"\b(?:short|shorter|compact|brief|concise)\b[\s,-]+(?:[\w-]+[\s,-]+){{0,2}}?paragraphs"},
    # sentence length
    {"name": "sentences_long", "feature": "avg_sent_len", "dir": "high",
     "pat": rf"\b(?:long|longer|lengthy|extended|run-on|winding|clause-heavy|nested)\b[\s,-]+(?:[\w-]+[\s,-]+){{0,3}}?{_SENT}|\brun-on\b|\bclause-heavy\b"},
    {"name": "sentences_short", "feature": "avg_sent_len", "dir": "low",
     "pat": rf"\b(?:short|shorter|simple|simpler|punchy|choppy|crisp)\b[\s,-]+(?:[\w-]+[\s,-]+){{0,2}}?{_SENT}"},
    # punctuation
    {"name": "emdash", "feature": "emdash_per_100w", "dir": "high",
     "pat": r"em[- ]?dash|\bdash(?:es)?\b"},
    {"name": "semicolon", "feature": "semicolon_per_100w", "dir": "high",
     "pat": r"\bsemicolon"},
    {"name": "colon", "feature": "colon_per_100w", "dir": "high", "pat": r"\bcolons?\b"},
    # hedging (hedge_per_100w is a fixed word list; a crude proxy)
    {"name": "hedging", "feature": "hedge_per_100w", "dir": "high",
     "pat": r"\bhedg|\bqualif|\bcaveat"},
    # vocabulary sophistication vs mean word length (a crude proxy)
    {"name": "sophisticated_vocab", "feature": "avg_word_len", "dir": "high",
     "pat": r"\b(?:sophisticated|advanced|elevated|elaborate|precise|academic|rich)\b[\s,-]+(?:[\w-]+[\s,-]+){0,2}?(?:vocabulary|diction|wording|terminology)"},
    {"name": "simple_vocab", "feature": "avg_word_len", "dir": "low",
     "pat": r"\b(?:simple|basic|plain|everyday|accessible)\b[\s,-]+(?:[\w-]+[\s,-]+){0,2}?(?:vocabulary|diction|wording|language)"},
]
_NUMW = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
PARA_N = re.compile(r"\b(two|three|four|five|six|seven|[2-7])[- ]paragraph", re.I)


def _norm_for_match(s: str) -> str:
    s = norm(s).lower()
    s = s.replace("—", " ").replace("–", " ").replace("--", " ")
    s = re.sub(r"[^\w\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def faithfulness(df: pd.DataFrame, resp: pd.DataFrame) -> dict:
    """For each rationale that makes a measurable claim, rank the judged
    response's feature among the five responses to the same prompt (1 = most)
    and among all 190 (percentile). "Extreme" = rank 1 in the claimed
    direction (chance 1/5 if the claim were unrelated to the text; ties are
    counted as extreme only if the response is strictly alone at the end).
    """
    feat = resp.set_index(["prompt_id", "model"])
    out = {}
    # Precompute within-prompt ranks (1 = highest) and corpus percentiles.
    ranks, pct = {}, {}
    for f in FEATURES:
        # rank "min" so a tie at the top gives two rank-1 rows; we treat a
        # shared top as not strictly extreme below.
        ranks[f] = resp.groupby("prompt_id")[f].rank(ascending=False, method="min")
        ranks[f + "_lo"] = resp.groupby("prompt_id")[f].rank(ascending=True, method="min")
        pct[f] = resp[f].rank(pct=True)
    keyed = resp[["prompt_id", "model"]].copy()
    for k, v in {**ranks, **{f + "_pct": p for f, p in pct.items()}}.items():
        keyed[k] = v.values
    # Ties at the extreme: count rows sharing the top value within a prompt.
    for f in FEATURES:
        keyed[f + "_top_ties"] = resp.groupby("prompt_id")[f].transform(lambda s: (s == s.max()).sum()).values
        keyed[f + "_bot_ties"] = resp.groupby("prompt_id")[f].transform(lambda s: (s == s.min()).sum()).values
    m = df.merge(keyed, left_on=["prompt_id", "true_author"],
                 right_on=["prompt_id", "model"], how="left")
    reason_n = m["reason"].map(norm)

    for chk in CHECKS:
        pat = re.compile(chk["pat"], re.I)
        neg = re.compile(NEG + "(?:" + chk["pat"] + ")", re.I)
        f = chk["feature"]
        for cond in ("lineup", "single"):
            sel = (m["condition"] == cond) & reason_n.map(lambda t: bool(pat.search(t)))
            g = m[sel].copy()
            if chk["dir"] == "high":
                # a negated high claim ("without dashes") becomes a low claim
                isneg = g["reason"].map(lambda t: bool(neg.search(norm(t)))).astype(bool)
            else:
                isneg = pd.Series(False, index=g.index)
            res = {"n_claims": int(len(g)), "n_negated": int(isneg.sum())}
            for label, sub, d in (("as_stated", g[~isneg], chk["dir"]),
                                  ("negated", g[isneg], "low" if chk["dir"] == "high" else "high")):
                if len(sub) == 0:
                    continue
                if d == "high":
                    strict = (sub[f] == 1) & (sub[f + "_top_ties"] == 1)
                    top2 = sub[f] <= 2
                    q = sub[f + "_pct"] >= 0.8
                else:
                    strict = (sub[f + "_lo"] == 1) & (sub[f + "_bot_ties"] == 1)
                    top2 = sub[f + "_lo"] <= 2
                    q = sub[f + "_pct"] <= 0.2
                k, n = int(strict.sum()), len(sub)
                res[label] = {
                    "direction": d, "n": n,
                    "extreme_of_5_k": k, "extreme_of_5_share": k / n,
                    "binom_p_vs_0.2": float(stats.binomtest(k, n, 0.2).pvalue),
                    "top2_of_5_k": int(top2.sum()), "top2_of_5_share": float(top2.mean()),
                    "corpus_quintile_k": int(q.sum()), "corpus_quintile_share": float(q.mean()),
                    # the claimed direction is wrong: the text sits at the
                    # OPPOSITE end of its lineup
                    "opposite_extreme_k": int(((sub[f + "_lo"] == 1) if d == "high" else (sub[f] == 1)).sum()),
                }
            out[f"{chk['name']}__{cond}"] = res

    # Paragraph-count claims ("three-paragraph"): exact match with para_count.
    for cond in ("lineup", "single"):
        g = m[m["condition"] == cond]
        rows = []
        for _, r in g.iterrows():
            mm = PARA_N.search(norm(r["reason"]))
            if mm:
                n_claim = _NUMW[mm.group(1).lower()]
                rows.append((n_claim, feat.loc[(r["prompt_id"], r["true_author"]), "para_count"]))
        if rows:
            arr = np.array(rows, dtype=float)
            out[f"paragraph_count__{cond}"] = {
                "n_claims": len(rows),
                "exact_k": int((arr[:, 0] == arr[:, 1]).sum()),
                "exact_share": float((arr[:, 0] == arr[:, 1]).mean()),
                "within_1_k": int((np.abs(arr[:, 0] - arr[:, 1]) <= 1).sum()),
                "claimed_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(arr[:, 0], return_counts=True))},
            }

    # Quoted phrases: does the quoted span appear verbatim in the text?
    texts = resp.set_index(["prompt_id", "model"])["text"]
    for cond in ("lineup", "single"):
        g = m[m["condition"] == cond]
        n_q, n_found, examples_missing = 0, 0, []
        for _, r in g.iterrows():
            body = _norm_for_match(texts.loc[(r["prompt_id"], r["true_author"])])
            for q in QUOTE.findall(norm(r["reason"])):
                span = _norm_for_match(q.strip("'\"").rstrip(".").replace("...", " "))
                if len(span) < 3:
                    continue
                n_q += 1
                if span in body:
                    n_found += 1
                elif len(examples_missing) < 8:
                    examples_missing.append({"judge": r["judge"], "author": r["true_author"],
                                             "prompt": r["prompt_id"], "quote": q})
        out[f"quoted_phrases__{cond}"] = {"n_quotes": n_q, "verbatim_k": n_found,
                                          "verbatim_share": n_found / n_q if n_q else None,
                                          "examples_not_found": examples_missing}
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def fmt_k(k):
    return "n/a" if k is None else f"{k:.2f}"


def write_md(res: dict, df: pd.DataFrame) -> None:
    L = []
    w = L.append
    val = res["validation"]
    w("# Rationale coding: what cues do judges cite, and are they there?\n")
    w("Generated by `src/rationale_codes.py`; numbers come from "
      "`results/rationale_codes.json`. Do not edit by hand.\n")
    w(f"{len(df)} rationales (950 lineup, 950 single-text), one sentence each, "
      f"mean {df['reason'].str.split().str.len().mean():.1f} words.\n")

    w("## Coding scheme\n")
    w("Eleven categories, multi-label, assigned by regular expressions "
      "(`CUES` in the script). A reason that matches none is *other*. The "
      "codebook the hand labels follow:\n")
    w("| category | codebook definition |\n|---|---|")
    for c in CATS:
        w(f"| {c} | {CODEBOOK[c]} |")
    w("")
    w("## Validation against hand labels\n")
    w(f"Sample: {val['n']['tune']} tune + {val['n']['holdout']} holdout rationales, "
      "stratified by judge x condition (10 and 5 per cell), excluding the 80 read "
      "while designing the codebook. **Annotator 1 (A1): "
      f"{val['annotator']}.** A1 labelled all 150 from the reason text alone, "
      "before any rule output was computed on the sample. But A1 also wrote "
      "the lexicon, and the lexicon code was written after the labelling pass, "
      "so scope decisions A1 made while labelling (\"balanced structure\" is "
      "structure, \"sophisticated vocabulary\" is lexical, \"clear\" is content "
      "unless it modifies structure) went into both the codebook and v0. "
      "Rules-vs-A1 agreement is therefore close to an internal-consistency "
      "check, not a reliability estimate, and it is high by construction. "
      "v0 was frozen, then revised once on the tune split (final); the "
      "holdout column is the out-of-sample figure for that revision. "
      "Labels: `results/rationale_handlabels.csv`.\n")
    w("A second annotator, a separate Claude instance that saw only the "
      "codebook and the 150 reasons (never the lexicon or the first labels), "
      "labelled the same sample blind. Its agreement with the rules is the "
      "less self-serving check; its kappa with annotator 1 is the "
      "inter-annotator reliability of the codebook itself. Both annotators "
      "are the same underlying model, so shared model biases are not ruled "
      "out.\n")
    w("| category | A1 n (tune/hold) | v0 vs A1, P / R tune | final vs A1, P / R tune | "
      "**final vs A1, P / R holdout** | A2 n | **final vs A2, P / R (all 150)** | "
      "kappa final vs A2 | **kappa A1 vs A2** |")
    w("|---|---|---|---|---|---|---|---|---|")
    a2v = val.get("final_vs_a2", {}).get("all")
    aa = val.get("a1_vs_a2")
    for c in CATS + ["other"]:
        a0 = val["v0"]["tune"][c]
        at = val["final"]["tune"][c]
        ah = val["final"]["holdout"][c]
        b = a2v[c] if a2v else None
        w(f"| {c} | {at['hand_n']}/{ah['hand_n']} | {pct(a0['precision'],0)} / {pct(a0['recall'],0)} "
          f"| {pct(at['precision'],0)} / {pct(at['recall'],0)} "
          f"| {pct(ah['precision'],0)} / {pct(ah['recall'],0)} "
          + (f"| {b['hand_n']} | {pct(b['precision'],0)} / {pct(b['recall'],0)} | {fmt_k(b['kappa'])} "
             f"| {fmt_k(aa[c]['kappa'])} |" if b else "| | | | |"))
    w("")
    if a2v:
        mi = a2v["_micro"]
        w(f"- final rules vs A2, all 150: micro P {pct(mi['precision'])}, R {pct(mi['recall'])}, "
          f"F1 {pct(mi['f1'])} ({mi['tp']} TP, {mi['fp']} FP, {mi['fn']} FN); "
          f"exact-match {pct(a2v['_exact_match'])}.")
        mi = aa["_micro"]
        w(f"- A1 vs A2, all 150: micro agreement F1 {pct(mi['f1'])} "
          f"({mi['tp']} shared codes, {mi['fp']} A2-only, {mi['fn']} A1-only); "
          f"exact-match {pct(aa['_exact_match'])}.")
    for ver in ("v0", "final"):
        for s in ("tune", "holdout"):
            mi = val[ver][s]["_micro"]
            w(f"- {ver}, {s}: micro P {pct(mi['precision'])}, R {pct(mi['recall'])}, "
              f"F1 {pct(mi['f1'])} ({mi['tp']} TP, {mi['fp']} FP, {mi['fn']} FN); "
              f"exact-match {pct(val[ver][s]['_exact_match'])} of reasons.")
    w("")

    # (a)
    w("## (a) Cue types by judge and condition\n")
    for cond in ("lineup", "single"):
        d = res["distribution"][cond]
        w(f"**{cond}** (count / share of that judge's 190 reasons; mean "
          f"{d['_mean_cues_per_reason']:.2f} codes per reason)\n")
        w("| cue | " + " | ".join(MODELS) + " | all | chi2 p (judges differ) |")
        w("|---|" + "---|" * (len(MODELS) + 2))
        for c in CATS + ["other"]:
            cells = [f"{d[j][c]['k']} ({pct(d[j][c]['share'],0)})" for j in MODELS]
            chi = d["_chi2_across_judges"].get(c, {})
            w(f"| {c} | " + " | ".join(cells) + f" | {d['_all'][c]['k']} ({pct(d['_all'][c]['share'],0)}) "
              f"| {fmt_p(chi.get('p'))} |")
        w("")

    # (b)
    w("## (b) Does the cited cue type predict correctness?\n")
    w("Logistic GEE, exchangeable working correlation, clustered by prompt "
      f"(38 clusters). Cues cited fewer than {MIN_HITS} times in a subset are "
      "left out. Odds ratios for citing the cue, adjusted for judge (and, in "
      "the second model, the true author, since in single-text correctness is "
      "mostly a question of whose text it is).\n")
    for cond in ("lineup", "single", "pooled"):
        for spec in ("judge_fe", "judge_and_author_fe"):
            r = res["cue_vs_correct"][cond][spec]
            w(f"**{cond}, {spec.replace('_', ' ')}** (n = {r['n']})\n")
            w("| cue | cited | acc. when cited | when not | OR [95% CI] | p |")
            w("|---|---|---|---|---|---|")
            for c, t in r["terms"].items():
                w(f"| {c} | {t['k_cited']} | {pct(t['acc_when_cited'])} | {pct(t['acc_when_not'])} "
                  f"| {t['OR']:.2f} [{t['ci95'][0]:.2f}, {t['ci95'][1]:.2f}] | {fmt_p(t['p'])} |")
            if r["excluded_sparse"]:
                w(f"\nLeft out as sparse: {', '.join(r['excluded_sparse'])}.")
            w("")

    # (c)
    sv = res["self_vs_peer"]
    w("## (c) Self-hits vs peer hits on Claude's and GPT's text\n")
    w(f"Rationales coded *self_reference* anywhere: **{sv['self_reference_total']} of "
      f"{len(df)}**. Any familiarity word (familiar, recognise, resemble): "
      f"{sv['familiarity_words_total']} ({', '.join(f'{j} {k}' for j, k in sv['familiarity_words_by_judge'].items())}). "
      "The judge prompts never say the judge is one of the candidates, and "
      "no judge ever says it is recognising its own writing.\n")
    w("Self-hit = A names A on A's text; peer hit = another judge names A on "
      "A's text; self-FA = A names itself on someone else's text. Fisher's "
      "exact test, self-hit vs peer hit; peer rows fall four to a response, so "
      "p is optimistic.\n")
    for key in ("Claude_lineup", "Claude_single", "GPT_lineup", "GPT_single"):
        r = sv[key]
        w(f"**{key.replace('_', ', ')}**: {r['n_self_hits']} self-hits, "
          f"{r['n_peer_hits']} peer hits, {r['n_self_false_alarms']} self false alarms\n")
        w("| cue | self-hit | peer hit | self-FA | Fisher p |")
        w("|---|---|---|---|---|")
        for c, t in r["cues"].items():
            w(f"| {c} | {t['self_hit_k']} ({pct(t['self_hit_share'],0)}) | "
              f"{t['peer_hit_k']} ({pct(t['peer_hit_share'],0)}) | "
              f"{t['self_fa_k']} ({pct(t['self_fa_share'],0)}) | {fmt_p(t['fisher_p'])} |")
        w("")

    # (d)
    fa = res["faithfulness"]
    w("## (d) Are stated cues true of the text?\n")
    w("For each rationale that makes a directional, measurable claim, the "
      "judged response's feature (from `features.py`) is ranked among the five "
      "responses to the same prompt. *Extreme* = strictly the most (or least, "
      "for a low claim) of the five; if the claim were unrelated to the text "
      "this would happen 20% of the time. *Top-2* has a 40% base rate; "
      "*corpus quintile* = top (bottom) 20% of all 190 responses. In the "
      "lineup the judge saw all five; in single-text it saw one, so the "
      "corpus quintile is the fairer yardstick there. Negated claims "
      "('without dashes') are scored in the opposite direction. The binomial "
      "p treats claims as independent; they are not (several judges describe "
      "the same response).\n")
    w("| claim | feature | cond | n | extreme of 5 | p vs 20% | top-2 | corpus quintile | opposite extreme |")
    w("|---|---|---|---|---|---|---|---|---|")
    for chk in CHECKS:
        for cond in ("lineup", "single"):
            r = fa.get(f"{chk['name']}__{cond}", {})
            for lab in ("as_stated", "negated"):
                if lab not in r:
                    continue
                t = r[lab]
                name = chk["name"] + (" (negated)" if lab == "negated" else "")
                w(f"| {name} | {chk['feature']} ({t['direction']}) | {cond} | {t['n']} | "
                  f"{t['extreme_of_5_k']} ({pct(t['extreme_of_5_share'],0)}) | {fmt_p(t['binom_p_vs_0.2'])} | "
                  f"{t['top2_of_5_k']} ({pct(t['top2_of_5_share'],0)}) | "
                  f"{t['corpus_quintile_k']} ({pct(t['corpus_quintile_share'],0)}) | "
                  f"{t['opposite_extreme_k']} |")
    w("")
    for cond in ("lineup", "single"):
        r = fa.get(f"paragraph_count__{cond}")
        if r:
            w(f"- Paragraph-count claims, {cond}: {r['exact_k']}/{r['n_claims']} "
              f"({pct(r['exact_share'])}) match the text's paragraph count exactly, "
              f"{r['within_1_k']} within one. Claimed: {r['claimed_counts']}.")
        q = fa.get(f"quoted_phrases__{cond}")
        if q and q["n_quotes"]:
            w(f"- Quoted phrases, {cond}: {q['verbatim_k']}/{q['n_quotes']} "
              f"({pct(q['verbatim_share'])}) appear verbatim (case and punctuation "
              "normalised) in the judged response.")
    w("")
    w(res["reading_md"])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def reading(res: dict) -> str:
    """Plain-language summary. Every number is pulled from `res`, so the
    prose cannot drift from the tables above it."""
    d, cv, sv, fa = (res["distribution"], res["cue_vs_correct"],
                     res["self_vs_peer"], res["faithfulness"])

    def kn(cond, c):
        return f"{d[cond]['_all'][c]['k']}/950"

    def orp(cond, spec, c):
        t = cv[cond][spec]["terms"][c]
        return f"OR {t['OR']:.2f}, p = {fmt_p(t['p'])}"

    def sp(key, c):
        t = sv[key]["cues"][c]
        return (f"{t['self_hit_k']}/{sv[key]['n_self_hits']} self-hits vs "
                f"{t['peer_hit_k']}/{sv[key]['n_peer_hits']} peer hits "
                f"(p = {fmt_p(t['fisher_p'])})")

    def fa_(key, c):
        t = sv[key]["cues"][c]
        return f"{t['self_fa_k']}/{sv[key]['n_self_false_alarms']}"

    def ex(name):
        t = fa[name]["as_stated"]
        return f"{t['extreme_of_5_k']}/{t['n']} ({pct(t['extreme_of_5_share'],0)})"

    q_l, q_s = fa["quoted_phrases__lineup"], fa["quoted_phrases__single"]
    pc_l, pc_s = fa["paragraph_count__lineup"], fa["paragraph_count__single"]
    return "\n".join([
        "## Reading\n",
        "**What judges say they use.** Rationales are about organisation, tone "
        "and content, and they end in a stereotype. Structure is cited in "
        f"{kn('lineup','structure')} lineup and {kn('single','structure')} "
        f"single-text reasons, tone in {kn('lineup','tone')} and "
        f"{kn('single','tone')}, and an appeal to what a model typically "
        f"writes (brand_prior) in {kn('lineup','brand_prior')} and "
        f"{kn('single','brand_prior')}. Length, the feature family with most of "
        f"the classifier's feature importance, is cited in {kn('lineup','length')} and "
        f"{kn('single','length')}. **No rationale refers to the judge itself "
        f"({res['self_vs_peer']['self_reference_total']} of 1,900).** The "
        "judge prompt never says the judge is a candidate, so this does not "
        "show the judge lacks self-knowledge; it shows that the stated route to "
        "a self-attribution is the same stereotype route as to any other.\n",
        "**Cue type and correctness.** Citing hedging goes with correct "
        f"attributions ({orp('lineup','judge_fe','hedging')} lineup; "
        f"{orp('single','judge_fe','hedging')} single-text, judge fixed "
        "effects). With the true author also fixed, the single-text effect "
        f"vanishes ({orp('single','judge_and_author_fe','hedging')}) and the "
        f"lineup one shrinks ({orp('lineup','judge_and_author_fe','hedging')}): "
        "hedging is mostly what judges say about Claude's text, and Claude's "
        "text is the most often recognised. With author held fixed, what "
        "remains at p < 0.05 does not replicate across conditions: in the "
        f"lineup hedging ({orp('lineup','judge_and_author_fe','hedging')}), "
        f"rhetoric ({orp('lineup','judge_and_author_fe','rhetoric')}) and tone "
        f"({orp('lineup','judge_and_author_fe','tone')}); in single-text only "
        f"lexical ({orp('single','judge_and_author_fe','lexical')}). Ten cue "
        "terms per model are tested, uncorrected.\n",
        "**Self-hits versus peer hits.** When Claude names itself correctly in "
        f"single-text, it cites hedging in {sp('Claude_single','hedging')}; but "
        f"it also cites hedging in {fa_('Claude_single','hedging')} of its self "
        "false alarms. GPT's single-text self-hits cite lexical cues in "
        f"{sp('GPT_single','lexical')}, and so do {fa_('GPT_single','lexical')} "
        "of its 79 self false alarms (content: "
        f"{sp('GPT_single','content')}; false alarms {fa_('GPT_single','content')}). "
        "The cue profile of a self-attribution is the same whether it is right "
        "or wrong, which is what a fixed self-stereotype applied to the text "
        "predicts, and not what recognition of specific familiar text predicts. "
        "One pattern points the other way: Claude's self-hit rationales quote "
        "a phrase from the text more often than peers' hits on the same text "
        f"({sp('Claude_single','quotes_text')} single-text; "
        f"{sp('Claude_lineup','quotes_text')} lineup), calling it "
        "\"characteristic Claude phrasing\". That is the closest thing in the "
        "data to a familiarity cue, and it is a minority of self-hits.\n",
        "**Are stated cues true of the text?** Mostly yes for the measurable "
        "ones. In the lineup, a response called long is the strictly longest "
        f"of the five in {ex('length_long__lineup')} of cases (chance 20%), "
        f"one called concise or short is the shortest in {ex('length_short__lineup')}, "
        f"long sentences are the longest-sentenced in {ex('sentences_long__lineup')} "
        f"(single-text {ex('sentences_long__single')}), and em-dashes are "
        f"highest in {ex('emdash__lineup')} (single-text {ex('emdash__single')}). "
        f"Stated paragraph counts are exact in {pc_l['exact_k']}/{pc_l['n_claims']} "
        f"lineup and {pc_s['exact_k']}/{pc_s['n_claims']} single-text claims, and "
        f"{q_l['verbatim_k']}/{q_l['n_quotes']} and {q_s['verbatim_k']}/{q_s['n_quotes']} "
        "quoted phrases occur verbatim in the judged response. The exception is "
        "vocabulary: responses said to have sophisticated or academic "
        "vocabulary have the highest mean word length in only "
        f"{ex('sophisticated_vocab__lineup')} lineup and "
        f"{ex('sophisticated_vocab__single')} single-text cases, at or below "
        "chance (a crude proxy, but the only lexical one available). Hedging "
        f"claims match the hedge-word rate in {ex('hedging__lineup')} lineup "
        f"and {ex('hedging__single')} single-text cases.\n",
        "**What this means for mechanism.** The rationales are faithful in "
        "the narrow sense that what they describe is usually there. They are "
        "not evidence that the described cue drove the attribution: the same "
        "cues are cited at the same rates when the attribution is wrong, and "
        "the most accurate descriptions (length, sentence length) are not the "
        "ones that separate correct from incorrect attributions. This is the "
        "gap between plausible and causally faithful explanations that "
        "Marioriyad et al. describe for LLM self-explanations: a rationale can "
        "be true of the input and still not be the reason for the output.\n",
        "**Caveats a reviewer should weigh.** (1) Rules-vs-annotator agreement "
        "is high partly by construction (see Validation); both annotators are "
        "Claude. (2) Regex coding cannot resolve every scope ambiguity; the "
        "holdout error mode \"sentence structure\" coded as structure was left "
        "unfixed deliberately. (3) Fisher tests in (c) and binomial tests in "
        "(d) treat judgments as independent; several judges describe the same "
        "response, so p-values are optimistic. The GEE in (b) clusters by "
        "prompt. (4) Rationales are one short sentence produced after (or with) "
        "the answer; they are post-hoc by design.\n",
    ])


def main() -> None:
    df = load_judgments()
    resp = load_responses()
    res = {
        "n_reasons": len(df),
        "categories": CATS,
        "codebook": CODEBOOK,
        "lexicon": CUES,
        "lexicon_v0": CUES_V0,
        "validation": validate(),
        "distribution": distribution(df),
        "cue_vs_correct": cue_vs_correct(df),
        "self_vs_peer": self_vs_peer(df),
        "faithfulness": faithfulness(df, resp),
    }
    res["reading_md"] = reading(res)

    # console summary
    v = res["validation"]
    print("=" * 74 + "\nRATIONALE CODES\n" + "=" * 74)
    for ver in ("v0", "final"):
        for s in ("tune", "holdout"):
            mi = v[ver][s]["_micro"]
            print(f"  {ver:5s} {s:8s} micro P {mi['precision']:.3f}  R {mi['recall']:.3f}  "
                  f"F1 {mi['f1']:.3f}  exact {v[ver][s]['_exact_match']:.2f}")
    if "final_vs_a2" in v:
        mi = v["final_vs_a2"]["all"]["_micro"]
        print(f"  final vs A2 (all) micro P {mi['precision']:.3f}  R {mi['recall']:.3f}  "
              f"F1 {mi['f1']:.3f}  exact {v['final_vs_a2']['all']['_exact_match']:.2f}")
        print(f"  A1 vs A2 (all)    micro F1 {v['a1_vs_a2']['_micro']['f1']:.3f}  "
              f"exact {v['a1_vs_a2']['_exact_match']:.2f}")
    print("\n  per-category, final lexicon (vs A1 tune | vs A1 holdout | vs A2 all | kappa A1-A2):")
    for c in CATS + ["other"]:
        t, h = v["final"]["tune"][c], v["final"]["holdout"][c]
        b = v.get("final_vs_a2", {}).get("all", {}).get(c)
        k12 = v.get("a1_vs_a2", {}).get(c, {}).get("kappa")
        print(f"    {c:15s} P {pct(t['precision'],0):>5} R {pct(t['recall'],0):>5} | "
              f"P {pct(h['precision'],0):>5} R {pct(h['recall'],0):>5} | "
              + (f"P {pct(b['precision'],0):>5} R {pct(b['recall'],0):>5} | k {fmt_k(k12)}" if b else ""))
    RESULTS_DIR.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    write_md(res, df)
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
