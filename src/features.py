"""CompLLM: the 18 interpretable surface features.

These are the SAME features used for both engines. That is the point of the
paper: Engine A shows they carry author identity, Engine B shows they do not
carry the blame target. If the feature set diverged between the two, the
positive control would be worthless.

Conventions: hapax_rate is the hapax-legomena rate (distinct from TTR);
punctuation and hedge counts are per 100 words, since raw counts are length
proxies.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

# --------------------------------------------------------------------------
# Readability, implemented here rather than via textstat.
#
# textstat >=0.7.5 routes syllable counting through NLTK's cmudict, which
# requires a corpus download. When that download is unavailable the calls
# raise, and any except-and-default wrapper turns three features into silent
# constant zeros - dead features that still occupy slots in the "18 features"
# claim. Self-contained implementations are deterministic, need no network,
# and let a reviewer run this package with nothing but pip install.
# --------------------------------------------------------------------------
_VOWEL_RUN = re.compile(r"[aeiouy]+")


def count_syllables(word: str) -> int:
    """Vowel-group heuristic with silent-e correction. Deterministic."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    n = len(_VOWEL_RUN.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    if w.endswith(("es", "ed")) and n > 1 and not w.endswith(("ies", "ted", "ded")):
        n -= 1
    return max(n, 1)


def flesch_reading_ease(words: list[str], n_sents: int) -> float:
    n_w = max(len(words), 1)
    syl = sum(count_syllables(w) for w in words)
    return 206.835 - 1.015 * (n_w / max(n_sents, 1)) - 84.6 * (syl / n_w)


def flesch_kincaid_grade(words: list[str], n_sents: int) -> float:
    n_w = max(len(words), 1)
    syl = sum(count_syllables(w) for w in words)
    return 0.39 * (n_w / max(n_sents, 1)) + 11.8 * (syl / n_w) - 15.59


def gunning_fog(words: list[str], n_sents: int) -> float:
    """Complex words = 3+ syllables. The classic formula also excludes proper
    nouns and familiar suffixes; we use the plain 3+ rule and say so."""
    n_w = max(len(words), 1)
    complex_w = sum(1 for w in words if count_syllables(w) >= 3)
    return 0.4 * (n_w / max(n_sents, 1) + 100.0 * complex_w / n_w)

FEATURES: list[str] = [
    "word_count", "ttr", "hapax_rate", "avg_word_len", "avg_sent_len",
    "sent_len_sd", "sent_count", "passive_rate", "flesch_reading_ease",
    "flesch_kincaid_grade", "gunning_fog", "hedge_per_100w", "para_count",
    "avg_para_len", "comma_per_100w", "emdash_per_100w", "colon_per_100w",
    "semicolon_per_100w",
]

# Features that are length, or are mechanically driven by length.
# TTR and hapax_rate are included because both fall monotonically with text
# length by construction - they are length proxies wearing a lexical costume.
LENGTH_PROXIES: set[str] = {
    "word_count", "para_count", "avg_para_len", "avg_sent_len", "sent_count",
    "ttr", "hapax_rate", "emdash_per_100w", "semicolon_per_100w",
}

LENGTH_FREE: list[str] = [f for f in FEATURES if f not in LENGTH_PROXIES]

# The minimal "is this just verbosity?" baseline: two gross size measures.
#
# Note sent_count is deliberately NOT here. After length is equalised it still
# separates models at ~49%, because sentence count at fixed word count is
# average sentence length - a style feature wearing a size costume. Including
# it would make the length-only baseline look stronger than length really is.
LENGTH_ONLY: list[str] = ["word_count", "para_count"]

# Grouping for the feature table in the appendix. This is a reading aid, not an
# analysis variable -- nothing is ever fit on a family. It is kept here rather
# than in the reporting code so there is one definition of what each feature is,
# and asserted to be a partition below so a new feature cannot be added without
# being placed.
FAMILIES: dict[str, str] = {
    "word_count": "length", "sent_count": "length", "para_count": "length",
    "avg_sent_len": "length", "avg_para_len": "length", "sent_len_sd": "length",

    "ttr": "lexical", "hapax_rate": "lexical", "avg_word_len": "lexical",
    "hedge_per_100w": "lexical", "passive_rate": "lexical",

    "comma_per_100w": "punctuation", "semicolon_per_100w": "punctuation",
    "colon_per_100w": "punctuation", "emdash_per_100w": "punctuation",

    "flesch_reading_ease": "readability",
    "flesch_kincaid_grade": "readability", "gunning_fog": "readability",
}
assert set(FAMILIES) == set(FEATURES), (
    "FAMILIES must cover exactly FEATURES; unplaced: "
    f"{sorted(set(FEATURES) ^ set(FAMILIES))}")

HEDGES = {
    "may", "might", "could", "perhaps", "possibly", "arguably", "seemingly",
    "apparently", "presumably", "likely", "unlikely", "often", "sometimes",
    "generally", "typically", "usually", "somewhat", "relatively", "fairly",
    "rather", "quite", "suggest", "suggests", "indicate", "indicates",
    "appear", "appears", "tend", "tends", "roughly", "approximately",
    "potentially", "conceivably", "probably", "assume", "assumes",
}

BE_FORMS = r"(?:is|are|was|were|be|been|being|am|get|gets|got|becomes|became)"
_PASSIVE = re.compile(rf"\b{BE_FORMS}\b\s+(?:\w+ly\s+)?\b\w+(?:ed|en|wn|ne|de)\b", re.I)
_WORD = re.compile(r"\b[\w'-]+\b")
_SENT = re.compile(r"[.!?]+(?:\s|$)")


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT.split(text) if s and s.strip()]
    return parts or [text.strip()]


def truncate_words(text: str, n: int) -> str:
    """Word-level truncation to a fixed cap."""
    words = text.split()
    return " ".join(words[:n]) if len(words) > n else text


def truncate_to_group_min(texts, groups):
    """Per-prompt truncation: every response in a prompt group is cut to the
    length of the SHORTEST response in that group.

    A fixed cap (e.g. 250 words) leaves every already-short response
    untouched, so length stays correlated with model in the short tail and the
    length-only baseline never falls to chance. Truncating to the group minimum
    equalises length exactly where it matters: within the lineup a judge sees.
    """
    texts, groups = list(texts), list(groups)
    lengths = [len(t.split()) for t in texts]
    floor: dict = {}
    for g, n in zip(groups, lengths):
        floor[g] = min(floor.get(g, n), n)
    return [truncate_words(t, floor[g]) for t, g in zip(texts, groups)]


def extract(text: str) -> dict[str, float]:
    text = (text or "").strip()
    words = _WORD.findall(text)
    lower = [w.lower() for w in words]
    n_w = max(len(words), 1)

    sents = _sentences(text)
    sent_lens = [len(_WORD.findall(s)) for s in sents] or [0]

    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()] or [text]

    counts = Counter(lower)
    hapax = sum(1 for c in counts.values() if c == 1)

    per100 = lambda k: 100.0 * k / n_w  # noqa: E731
    n_s = len(sents)

    return {
        "word_count": float(len(words)),
        "ttr": len(set(lower)) / n_w,
        "hapax_rate": hapax / n_w,
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "avg_sent_len": float(np.mean(sent_lens)),
        "sent_len_sd": float(np.std(sent_lens)),
        "sent_count": float(len(sents)),
        "passive_rate": per100(len(_PASSIVE.findall(text))),
        "flesch_reading_ease": flesch_reading_ease(words, n_s),
        "flesch_kincaid_grade": flesch_kincaid_grade(words, n_s),
        "gunning_fog": gunning_fog(words, n_s),
        "hedge_per_100w": per100(sum(1 for w in lower if w in HEDGES)),
        "para_count": float(len(paras)),
        "avg_para_len": float(np.mean([len(_WORD.findall(p)) for p in paras])),
        "comma_per_100w": per100(text.count(",")),
        "emdash_per_100w": per100(text.count("\u2014") + text.count("--")),
        "colon_per_100w": per100(text.count(":")),
        "semicolon_per_100w": per100(text.count(";")),
    }


def frame(texts, guard: bool = True) -> "object":
    """Build the feature matrix.

    guard=True hard-fails on any feature that is constant or non-finite
    across the corpus. A dead feature that quietly returns 0.0 for every row
    still occupies a slot in the "18 features" claim while contributing
    nothing, which is how a broken readability dependency goes unnoticed.
    """
    import pandas as pd

    df = pd.DataFrame([extract(t) for t in texts], columns=FEATURES)
    if guard:
        bad_const = [c for c in FEATURES if df[c].nunique() <= 1]
        bad_nan = [c for c in FEATURES if not np.isfinite(df[c]).all()]
        if bad_const or bad_nan:
            raise AssertionError(
                "Feature extraction produced dead columns - do not analyse this.\n"
                f"  constant across all rows : {bad_const}\n"
                f"  non-finite values        : {bad_nan}"
            )
    return df
