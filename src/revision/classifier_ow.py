"""CompLLM revision: the stylometric classifier on the open-weight corpus.

The same 18 surface features and random forest as Engine A (the positive
control of Study 1), trained and tested on disjoint prompts (GroupKFold by
prompt), on the four open-weight models' 480 texts: pooled, and within each
genre. Chance is 25%.

    python src/revision/classifier_ow.py
Writes results/revision/openweight/classifier.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import features  # noqa: E402
from engine_a import N_FOLDS, rf, wilson  # noqa: E402
from panels import OPENWEIGHT  # noqa: E402

OUT = OPENWEIGHT.results_dir / "classifier.json"


def score(df: pd.DataFrame) -> dict:
    X = features.frame(df["text"].tolist(), guard=False).to_numpy(float)
    y = df["model"].to_numpy()
    g = df["prompt_id"].to_numpy()
    pred = cross_val_predict(rf(), X, y, groups=g, cv=GroupKFold(n_splits=N_FOLDS))
    k = int((pred == y).sum())
    lo, hi = wilson(k, len(y))
    return {"n": int(len(y)), "correct": k, "accuracy": k / len(y), "ci95": [lo, hi]}


def cross_sample(s1: pd.DataFrame, s2: pd.DataFrame) -> dict:
    """Within-model consistency: train on the first sample of the training
    prompts, test on the second, independent sample of the held-out prompts.
    Accuracy near the within-sample figure means each model's style is a
    stable property of the model, not of one draw."""
    X1 = features.frame(s1["text"].tolist(), guard=False).to_numpy(float)
    X2 = features.frame(s2["text"].tolist(), guard=False).to_numpy(float)
    y1, y2 = s1["model"].to_numpy(), s2["model"].to_numpy()
    g1, g2 = s1["prompt_id"].to_numpy(), s2["prompt_id"].to_numpy()
    correct = 0
    for tr, _ in GroupKFold(n_splits=N_FOLDS).split(X1, y1, g1):
        train_prompts = set(g1[tr])
        te = np.array([p not in train_prompts for p in g2])
        m = rf().fit(X1[tr], y1[tr])
        correct += int((m.predict(X2[te]) == y2[te]).sum())
    lo, hi = wilson(correct, len(y2))
    return {"n": int(len(y2)), "correct": correct, "accuracy": correct / len(y2),
            "ci95": [lo, hi]}


def main() -> None:
    df = pd.read_csv(OPENWEIGHT.corpus_csv, dtype=str, keep_default_na=False)
    df = df[df["model"].isin(OPENWEIGHT.names)]
    out = {"pooled": score(df)}
    for genre, d in df.groupby("genre"):
        out[genre] = score(d)
    s2_path = OPENWEIGHT.corpus_csv.with_name("responses_ow_s2.csv")
    if s2_path.exists():
        s2 = pd.read_csv(s2_path, dtype=str, keep_default_na=False)
        s2 = s2[s2["model"].isin(OPENWEIGHT.names)]
        out["crosssample"] = cross_sample(df, s2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(f"{k:8s} {v['accuracy']:.1%} [{v['ci95'][0]:.1%}, {v['ci95'][1]:.1%}] n={v['n']}")


if __name__ == "__main__":
    main()
