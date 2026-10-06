"""Score the rule-based rationale coder against a human annotator.

The 150-reason validation sample (results/rationale_handlabels.csv) was
first labelled by two Claude instances. A human then labelled the same 150
reasons blind to judge, author and both LLM labels, from the codebook alone
(annotate/label_rationales.html, which exports
results/rationale_handlabels_human.csv). This script reports:

  rules_vs_human  the coder against the human, all 150 and the 50 held out
  a1_vs_human     LLM annotator 1 (who wrote the lexicon) against the human
  a2_vs_human     LLM annotator 2 (blind) against the human

    python src/rationale_human.py [path/to/rationale_handlabels_human.csv]

Writes results/rationale_human.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rationale_codes import (CATS, COMPILED, HANDLABELS_CSV, RESULTS_DIR,  # noqa: E402
                             agreement, hand_sets, tag)

HUMAN_CSV = RESULTS_DIR / "rationale_handlabels_human.csv"
OUT = RESULTS_DIR / "rationale_human.json"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HUMAN_CSV
    hu = pd.read_csv(path)
    if not hu["human_labelled"].all():
        raise SystemExit(f"{(~hu['human_labelled'].astype(bool)).sum()} reasons unlabelled")
    hl = pd.read_csv(HANDLABELS_CSV).merge(hu, on="rid", validate="one_to_one")
    assert len(hl) == 150, len(hl)
    human = hand_sets(hl, "human_")
    rules = [tag(t, COMPILED) for t in hl["reason"]]
    a1, a2 = hand_sets(hl, "hand_"), hand_sets(hl, "hand2_")
    hold = np.where(hl["split"] == "holdout")[0]
    res = {
        "n": len(hl), "n_holdout": int(len(hold)),
        "rules_vs_human": agreement(human, rules),
        "rules_vs_human_holdout": agreement([human[i] for i in hold], [rules[i] for i in hold]),
        "a1_vs_human": agreement(human, a1),
        "a2_vs_human": agreement(human, a2),
    }
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")

    def line(name, r):
        m = r["_micro"]
        ks = [r[c]["kappa"] for c in CATS if r[c]["kappa"] is not None and r[c]["hand_n"] >= 3]
        print(f"{name:24s} P {100 * m['precision']:.1f}  R {100 * m['recall']:.1f}  "
              f"F1 {100 * m['f1']:.1f}  exact {100 * r['_exact_match']:.0f}%  "
              f"kappa {min(ks):.2f}-{max(ks):.2f} (median {np.median(ks):.2f})")
    for k in ("rules_vs_human", "rules_vs_human_holdout", "a1_vs_human", "a2_vs_human"):
        line(k, res[k])
    print("\nper category, rules vs human:")
    for c in CATS:
        r = res["rules_vs_human"][c]
        print(f"  {c:15s} human {r['hand_n']:3d} rules {r['rule_n']:3d}  "
              f"P {r['precision'] if r['precision'] is None else round(100 * r['precision'])}  "
              f"R {r['recall'] if r['recall'] is None else round(100 * r['recall'])}  "
              f"kappa {r['kappa'] if r['kappa'] is None else round(r['kappa'], 2)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
