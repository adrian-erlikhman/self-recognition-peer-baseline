"""CompLLM revision: how often does the standard test reach the wrong verdict?

The standard test of self-recognition asks whether a judge names itself on
its own text more often than chance. The test this paper argues for asks
whether the judge (a) beats its peers on its own text (self-advantage) and
(b) names itself more on its own text than on others' (discrimination). We
score every judge in every setting we ran with both and count disagreements.

Settings: frontier chat-app corpus (lineup, single text), frontier API corpus
of 120 prompts (lineup, single text), open-weight panel (lineup, single
text, yes/no). For yes/no the standard test is a "yes" rate on own text above
50%; the discrimination test is the AUC permutation test.

    python src/revision/standard_vs_proper.py
    python src/revision/standard_vs_proper.py --arxiv
Writes results/revision/standard_vs_proper.json (the 32 judge-settings of the
TACL submission), or with --arxiv standard_vs_proper_arxiv.json, which adds
the frontier chat-app yes/no setting run for the arXiv version.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT  # noqa: E402

R = ROOT / "results"
ALPHA = 0.05


def holm(ps: dict) -> dict:
    keys = sorted(ps, key=lambda k: ps[k])
    out, run, m = {}, 0.0, len(keys)
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def rows_attr(S: dict, setting: str, chance: float) -> list[dict]:
    out = []
    raw = {}
    for j, g in S["judges"].items():
        if "self_rate" not in g:
            continue
        raw[j] = binomtest(g["self_hits"], g["n_self"], chance, alternative="greater").pvalue
    std = holm(raw)
    for j, g in S["judges"].items():
        if "self_rate" not in g:
            continue
        standard = std[j] < ALPHA
        proper = (g["perm_p_holm"] < ALPHA and g["self_advantage"] > 0
                  and g["disc_p_holm"] < ALPHA and g["hit_minus_fa"] > 0)
        out.append({"setting": setting, "judge": j, "self_rate": g["self_rate"],
                    "standard": bool(standard), "proper": bool(proper)})
    return out


def rows_yesno(S: dict, setting: str) -> list[dict]:
    out = []
    raw = {j: binomtest(g["hits"], g["n_own"], 0.5, alternative="greater").pvalue
           for j, g in S["judges"].items()}
    std = holm(raw)
    for j, g in S["judges"].items():
        out.append({"setting": setting, "judge": j, "self_rate": g["hit_rate"],
                    "standard": bool(std[j] < ALPHA),
                    "proper": bool(g["auc_p_holm"] < ALPHA and g["auc"] > 0.5)})
    return out


def frontier_v1(cond: str) -> dict:
    """Study 1 in the same shape as analyze_panel's summaries."""
    S = json.loads((R / "stats_revision.json").read_text())[cond]["per_judge"]
    D = json.loads((R / "revision" / "frontier_discrimination.json").read_text())[cond]
    judges = {}
    for j, g in S.items():
        d = D.get(j) or {}
        judges[j] = {"self_rate": g["self"]["k"] / 38, "self_hits": g["self"]["k"], "n_self": 38,
                     "self_advantage": g["self_advantage"]["adv"],
                     "perm_p_holm": g["self_advantage"]["p_holm_randomization"],
                     "hit_minus_fa": d.get("hit_minus_fa", 0.0),
                     "disc_p_holm": d.get("disc_p_holm", 1.0)}
    return {"judges": judges}


def main() -> None:
    rows = []
    for cond, tag in (("lineup", "lineup"), ("single", "single text")):
        rows += rows_attr(frontier_v1(cond), f"frontier chat-app, {tag}", 0.2)
    FX = json.loads((R / "revision" / "frontier" / "summary_api120.json").read_text())
    for cond, tag in (("lineup", "lineup"), ("single", "single text")):
        rows += rows_attr(FX[cond], f"frontier API, {tag}", 0.2)
    OW = json.loads((R / "revision" / "openweight" / "summary.json").read_text())
    for cond, tag in (("lineup", "lineup"), ("single", "single text")):
        rows += rows_attr(OW[cond], f"open-weight, {tag}", 0.25)
    rows += rows_yesno(OW["binary"], "open-weight, yes/no")
    arxiv = "--arxiv" in sys.argv
    if arxiv:
        FB = json.loads((R / "revision" / "frontier" / "summary.json").read_text())
        rows += rows_yesno(FB["binary"], "frontier chat-app, yes/no")
    n = len(rows)
    std_pos = [r for r in rows if r["standard"]]
    false_pos = [r for r in std_pos if not r["proper"]]
    missed = [r for r in rows if r["proper"] and not r["standard"]]
    out = {"cells": n, "standard_positive": len(std_pos), "proper_positive":
           sum(r["proper"] for r in rows), "standard_false_positive": len(false_pos),
           "standard_missed": len(missed), "rows": rows}
    name = "standard_vs_proper_arxiv.json" if arxiv else "standard_vs_proper.json"
    (R / "revision" / name).write_text(json.dumps(out, indent=1, default=lambda o: bool(o) if hasattr(o, "dtype") or isinstance(o, bool) else float(o)))
    print(f"{n} judge-settings; standard test says 'recognises itself' in {len(std_pos)}; "
          f"proper test in {out['proper_positive']}; standard false positives {len(false_pos)}; "
          f"missed {len(missed)}")
    for r in rows:
        print(f"  {r['setting']:28s} {r['judge']:9s} self {r['self_rate']:.2f}  "
              f"standard {'YES' if r['standard'] else ' no'}  proper {'YES' if r['proper'] else ' no'}")


if __name__ == "__main__":
    main()
