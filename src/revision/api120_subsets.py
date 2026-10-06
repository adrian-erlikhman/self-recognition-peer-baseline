"""CompLLM revision: why does Study 1 at scale differ from Study 1?

The all-API frontier corpus (data/responses_api120.csv) contains API answers
to the same 38 prompts the chat-app corpus of Study 1 used. Scoring the
lineup on that subset separates the two differences between the studies:
the collection channel (chat app vs API) and the scale and genre mix
(38 prose prompts vs 120 prompts in four genres).

    python src/revision/api120_subsets.py
Writes results/revision/frontier/api120_subsets.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze_panel as A  # noqa: E402
from config import EXCLUDE_PROMPTS, ROOT  # noqa: E402
from panels import FRONTIER  # noqa: E402

OUT = FRONTIER.results_dir / "api120_subsets.json"


def main() -> None:
    A.TAG = "api120"
    v1 = pd.read_csv(ROOT / "data" / "responses_v1.csv", dtype=str)
    v1_prompts = set(v1["prompt_id"]) - set(EXCLUDE_PROMPTS)
    out = {}
    for cond in ("lineup", "single"):
        df = A.load(FRONTIER, cond)
        if df is None:
            continue
        a = A.attribution_rows(df, cond)
        rng = np.random.default_rng(A.RNG_SEED)
        sub = a[a["prompt_id"].isin(v1_prompts)]
        out[cond] = {
            "study1_prompts_via_api": A.attribution_metrics(sub, FRONTIER.names, rng),
            "new_prompts_only": A.attribution_metrics(a[~a["prompt_id"].isin(v1_prompts)],
                                                      FRONTIER.names, rng),
        }
        for k, S in out[cond].items():
            ps = {j: g["perm_p"] for j, g in S["judges"].items() if "perm_p" in g}
            for j, v in A.holm(ps).items():
                S["judges"][j]["perm_p_holm"] = v
            ps = {j: g["disc_perm_p"] for j, g in S["judges"].items() if "disc_perm_p" in g}
            for j, v in A.holm(ps).items():
                S["judges"][j]["disc_p_holm"] = v
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for cond, D in out.items():
        for k, S in D.items():
            print(f"{cond} / {k}: n={S['n_judgments']} acc={S['accuracy']:.3f}")
            for j, g in S["judges"].items():
                if "self_rate" in g:
                    print(f"   {j:8s} self {g['self_rate']:.2f} peer {g['peer_baseline']:.2f} "
                          f"FA {g['false_alarm']:.2f} adv {g['self_advantage']:+.3f} "
                          f"(Holm {g['perm_p_holm']:.3f})  disc {g['hit_minus_fa']:+.3f} "
                          f"(Holm {g['disc_p_holm']:.3f})")


if __name__ == "__main__":
    main()
