"""Strongest-single-peer control for the 28 lineup and one-text cases.

    python strongest_peer.py [REPO] [OUT]

For judge J on one set of texts in one format, the strongest peer is the other
judge that names J most often on J's own texts. The test compares, text by
text, whether J names itself and whether that peer names J: an exact one-sided
McNemar test (sign flip of the pair), Holm-corrected across the judges of the
setting as elsewhere in the paper. Picking the strongest peer after looking at
the data makes the test conservative. Intervals: 4,000 bootstrap resamples of
prompts, with the strongest peer fixed. Writes OUT/strongest_peer.csv and
prints the cases that pass.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent
R = REPO / "results"
FRONT = ["GPT", "Claude", "Gemini", "Grok", "DeepSeek"]
OW = ["Qwen", "Llama", "Mistral", "Phi"]
rng = np.random.default_rng(20261010)

sys.argv = sys.argv[:1]  # keep peer_discrimination's argv parsing at its defaults
import importlib.util  # noqa: E402

src = Path(__file__).with_name("peer_discrimination.py").read_text(encoding="utf-8")
loaders = {"__file__": str(Path(__file__).with_name("peer_discrimination.py"))}
exec(src.split("SET=[]")[0], loaders)  # functions only, not the analysis at the bottom


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    m = len(ps)
    out = np.empty(m)
    run = 0
    for i, k in enumerate(o):
        run = max(run, min(1, (m - i) * ps[k]))
        out[k] = run
    return out


SETTINGS = [
    ("frontier chat-app, lineup", FRONT, lambda: loaders["attr_from_csv"](R / "judgments.csv")),
    ("frontier chat-app, single text", FRONT, lambda: loaders["attr_from_csv"](R / "judgments_single.csv")),
    ("frontier API, lineup", FRONT, lambda: loaders["attr_from_jsonl"](R / "revision/frontier/lineup_api120.jsonl", False)),
    ("frontier API, single text", FRONT, lambda: loaders["attr_from_jsonl"](R / "revision/frontier/single_api120.jsonl", True)),
    ("open-weight, lineup", OW, lambda: loaders["attr_from_jsonl"](R / "revision/openweight/lineup.jsonl", False)),
    ("open-weight, single text", OW, lambda: loaders["attr_from_jsonl"](R / "revision/openweight/single.jsonl", True)),
]

rows = []
for setting, names, load in SETTINGS:
    a = load()[0]
    block = []
    for J in names:
        N = loaders["cube"](a, names, J)
        j = names.index(J)
        own = N[:, j, j]
        peers = [p for p in range(len(names)) if p != j]
        rates = {names[p]: N[:, j, p].mean() for p in peers}
        best = max(rates, key=rates.get)
        b_ = N[:, j, names.index(best)]
        b = int(((own == 1) & (b_ == 0)).sum())
        c = int(((own == 0) & (b_ == 1)).sum())
        p = float(binom.sf(b - 1, b + c, 0.5)) if b + c else 1.0
        G = len(own)
        idx = rng.integers(0, G, size=(4000, G))
        boots = own[idx].mean(axis=1) - b_[idx].mean(axis=1)
        block.append(dict(setting=setting, judge=J, self_rate=own.mean(), best_peer=best,
                          best_rate=rates[best], diff=own.mean() - rates[best],
                          lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                          b=b, c=c, p=p))
    for r, ph in zip(block, holm([r["p"] for r in block])):
        r["p_holm"] = ph
    rows += block

T = pd.DataFrame(rows)
T.to_csv(OUT / "strongest_peer.csv", index=False)
pd.set_option("display.width", 200)
print(T.assign(**{k: (T[k] * 100).round(1) for k in ["self_rate", "best_rate", "diff", "lo", "hi"]})
      .to_string(index=False))
print("\npass (diff > 0, Holm p < 0.05):")
print(T[(T["diff"] > 0) & (T.p_holm < 0.05)][["setting", "judge", "best_peer"]].to_string(index=False))
