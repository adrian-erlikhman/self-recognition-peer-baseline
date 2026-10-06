"""CompLLM revision: is frontier self-naming a quality effect? (A1)

The open-weight panel rated every frontier response for quality (1-9,
expected value over the rating digits; run_condition.py --condition quality
--corpus data/responses_v1.csv --tag frontiercorpus). Four raters, none of
them a frontier judge, give each text a quality score independent of who is
judging it. Two questions:

  1. Do frontier judges name themselves on the texts that are best, rather
     than on their own? Among texts a judge did NOT write, compare the quality
     of those it claimed with those it did not (within-prompt permutation).
  2. Does self-advantage survive holding quality fixed? Logistic GEE of
     correctness on judge-is-author, text quality (z within prompt) and author
     fixed effects, clustered by prompt, on each judge's own texts vs peers'
     judgments of the same texts.

    python src/revision/quality_frontier.py
Writes results/revision/quality_frontier.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EXCLUDE_PROMPTS, MODELS, ROOT  # noqa: E402

RATINGS = ROOT / "results" / "revision" / "openweight" / "quality_frontiercorpus.jsonl"
OUT = ROOT / "results" / "revision" / "quality_frontier.json"
N_PERM = 10000
SEED = 20260929


def quality_table() -> pd.DataFrame:
    recs = [json.loads(l) for l in RATINGS.read_text(encoding="utf-8").splitlines() if l.strip()]
    q = pd.DataFrame(recs)
    q = q[~q["prompt_id"].isin(EXCLUDE_PROMPTS)]
    per = q.groupby(["prompt_id", "true_author"])["expected"].mean().rename("quality").reset_index()
    # Standardise within prompt: quality relative to the other four answers
    # to the same task, which is what a judge comparing them could see.
    per["qz"] = per.groupby("prompt_id")["quality"].transform(lambda v: (v - v.mean()) / (v.std() or 1))
    # Inter-rater agreement: mean pairwise Spearman correlation across raters.
    wide = q.pivot_table(index=["prompt_id", "true_author"], columns="judge", values="expected")
    rho = wide.corr(method="spearman").to_numpy()
    iu = np.triu_indices_from(rho, 1)
    per.attrs["rater_rho"] = float(np.nanmean(rho[iu]))
    per.attrs["by_author"] = per.groupby("true_author")["quality"].mean().round(3).to_dict()
    return per


def claimed_quality(j: pd.DataFrame, per: pd.DataFrame, judge: str, rng) -> dict:
    d = j[(j.judge == judge) & (j.true_author != judge)].merge(
        per, on=["prompt_id", "true_author"])
    y = (d["guessed_model"] == judge).to_numpy()
    z = d["qz"].to_numpy()
    if y.sum() < 3 or y.sum() > len(y) - 3:
        return {"n": int(len(d)), "claims": int(y.sum()), "note": "too few claims"}
    obs = z[y].mean() - z[~y].mean()
    groups = [g.index.to_numpy() for _, g in d.reset_index(drop=True).groupby("prompt_id")]
    sims = np.empty(N_PERM)
    for k in range(N_PERM):
        yp = y.copy()
        for idx in groups:
            yp[idx] = rng.permutation(y[idx])
        sims[k] = z[yp].mean() - z[~yp].mean()
    return {"n": int(len(d)), "claims": int(y.sum()), "qz_claimed_minus_not": float(obs),
            "perm_p": float((np.abs(sims) >= abs(obs) - 1e-12).mean())}


def adv_given_quality(j: pd.DataFrame, per: pd.DataFrame, judge: str) -> dict:
    import statsmodels.api as sm
    d = j[j.true_author == judge].merge(per, on=["prompt_id", "true_author"])
    d["is_self"] = (d.judge == judge).astype(float)
    X = sm.add_constant(d[["is_self", "qz"]])
    try:
        m = sm.GEE(d["correct"], X, groups=d["prompt_id"], family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable()).fit()
        return {"n": int(len(d)), "or_self": float(np.exp(m.params["is_self"])),
                "p_self": float(m.pvalues["is_self"]),
                "or_quality": float(np.exp(m.params["qz"])), "p_quality": float(m.pvalues["qz"])}
    except Exception as exc:  # noqa: BLE001
        return {"n": int(len(d)), "error": str(exc)}


def main() -> None:
    if not RATINGS.exists():
        raise SystemExit(f"{RATINGS} missing: run the quality_frontier step of run_openweight.py")
    per = quality_table()
    rng = np.random.default_rng(SEED)
    out = {"rater_mean_spearman": per.attrs["rater_rho"],
           "mean_quality_by_author": per.attrs["by_author"]}
    for cond, f in (("lineup", "judgments.csv"), ("single", "judgments_single.csv")):
        j = pd.read_csv(ROOT / "results" / f, dtype={"prompt_id": str})
        j = j[~j["prompt_id"].isin(EXCLUDE_PROMPTS)]
        out[cond] = {m: {"claims_vs_quality": claimed_quality(j, per, m, rng),
                         "advantage_given_quality": adv_given_quality(j, per, m)}
                     for m in MODELS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
