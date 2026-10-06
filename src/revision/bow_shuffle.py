"""CompLLM revision: is authorship still recoverable from shuffled words?

    python src/revision/bow_shuffle.py

The shuffled-lineup condition (conditions.shuffle_words) scrambles the order
of every text's whitespace-delimited words, punctuation attached. A
bag-of-words classifier over those same tokens cannot see order, so its
accuracy on the shuffled texts is its accuracy on the originals; we fit it on
the shuffled texts anyway, with the seeds the judges saw, as a check.
TF-IDF over lowercased tokens (unigrams), logistic regression, five folds
grouped by prompt, on the 190 chat-app texts. Chance is 20%.
Writes results/revision/frontier/bow_shuffle.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import conditions as C  # noqa: E402
from config import EXCLUDE_PROMPTS, ROOT  # noqa: E402
from engine_a import wilson  # noqa: E402


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "responses_v1.csv", dtype=str, keep_default_na=False)
    df = df[~df["prompt_id"].isin(EXCLUDE_PROMPTS)].reset_index(drop=True)
    out = {}
    for name, texts in (
        ("original", df["text"].tolist()),
        ("shuffled", [C.shuffle_words(t, C.stable_seed(p, a))
                      for t, p, a in zip(df["text"], df["prompt_id"], df["model"])]),
    ):
        pipe = make_pipeline(
            TfidfVectorizer(token_pattern=r"\S+", lowercase=True, min_df=2, sublinear_tf=True),
            LogisticRegression(C=10.0, max_iter=5000))
        pred = cross_val_predict(pipe, texts, df["model"].to_numpy(),
                                 groups=df["prompt_id"].to_numpy(), cv=GroupKFold(n_splits=5))
        k = int((pred == df["model"].to_numpy()).sum())
        lo, hi = wilson(k, len(df))
        out[name] = {"correct": k, "n": len(df), "accuracy": k / len(df), "ci95": [lo, hi]}
        print(f"{name}: {k}/{len(df)} = {k / len(df):.1%} [{lo:.1%}, {hi:.1%}]")
    path = ROOT / "results" / "revision" / "frontier" / "bow_shuffle.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
