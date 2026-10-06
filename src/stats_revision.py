"""CompLLM: clustered inference for the TACL revision (reviewer items C2-C8).

    python src/stats_revision.py

Writes results/stats_revision.json and docs/revision/STATS.md. No API key, no
network; everything runs off results/judgments*.csv. Deterministic: every
random draw comes from a generator seeded with (SEED, condition, task).

Why this exists. The submitted paper tested self-advantage with Fisher's exact
test on 38 self vs 152 peer judgments, as if the 152 were independent. They
are not: four peer judgments fall on each response, and in the lineup all of a
judge's judgments on one prompt come from a single session. Every test below
respects that structure by treating the PROMPT as the unit of resampling.

What this computes, per condition (lineup, single-text):

  C6  Self-advantage (self rate - peer baseline) with
        (a) a within-response randomization test of which judge is "self"
            -- exact (Poisson-binomial) and 20,000 Monte Carlo draws;
        (b) a prompt-level cluster bootstrap CI (10,000 resamples);
        (c) a GEE logistic, correct ~ is_self on the 190 judgments of each
            judge's own-authored text, exchangeable within prompt;
        (d) a crossed random-effects logistic (prompt, judge), variational Bayes.
  C7  Intervals on every rate (Clopper-Pearson and Wilson for self rates,
      cluster bootstrap for the rest) and Holm-Bonferroni within families.
  C2  Non-self accuracy against the 20% floor, per judge-condition cell.
  C3  Pairwise differences in non-self accuracy, prompt-clustered.
  C8  Self-advantage net of judge skill and author identifiability, by a
      GEE with judge and author effects and by a direct difference-in-
      differences reweighting of the peer baseline.
  ICC Intraclass correlation of correctness within prompt and within response,
      and the design effect it implies for the peer baseline.

Notation. C[p, a, j] = 1 if judge j named the true author of the response to
prompt p written by author a. Every (a, j) cell holds exactly one judgment per
prompt, so a prompt-level bootstrap is just a reweighting of prompts and every
statistic is a function of the 5 x 5 table of cell rates.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    EXCLUDE_PROMPTS, JUDGMENTS_CSV, JUDGMENTS_SINGLE_CSV, MODELS, RESULTS_DIR,
    ROOT,
)

CHANCE = 0.20
ALPHA = 0.05
SEED = 2026
N_PERM = 20_000
N_BOOT = 10_000

CONDITIONS = {"lineup": JUDGMENTS_CSV, "single": JUDGMENTS_SINGLE_CSV}
OLD_JSON = RESULTS_DIR / "engine_b_results.json"
OUT_JSON = RESULTS_DIR / "stats_revision.json"
OUT_MD = ROOT / "docs" / "revision" / "STATS.md"

K = len(MODELS)
# The paper's qualitative reading of lineup self-advantage (abstract), which C8
# checks survives the skill adjustment.
PAPER_PATTERN = {"lineup": {"GPT": "+", "Claude": "+", "Gemini": "0",
                            "Grok": "-", "DeepSeek": "0"},
                 "single": {"GPT": "+", "Claude": "0", "Gemini": "0",
                            "Grok": "0", "DeepSeek": "0"}}
# Judges the abstract calls "no self-advantage" (C3's group contrast).
NO_ADV = ["Gemini", "Grok", "DeepSeek"]
ADV = ["GPT", "Claude"]


# --------------------------------------------------------------------- utils
def rng_for(cond: str, task: int) -> np.random.Generator:
    """One independent, reproducible stream per (condition, task)."""
    return np.random.default_rng([SEED, list(CONDITIONS).index(cond), task])


def load(condition: str) -> pd.DataFrame:
    """Same filters as engine_b.load, plus the derived columns used here."""
    df = pd.read_csv(CONDITIONS[condition])
    df = df[~df["prompt_id"].isin(EXCLUDE_PROMPTS)].copy()
    if "needs_review" in df:
        df = df[df["needs_review"] != 1].copy()
    df = df[df["guessed_model"].notna() & (df["guessed_model"] != "")].copy()
    df["correct"] = (df["guessed_model"] == df["true_author"]).astype(int)
    df["is_self"] = (df["judge"] == df["true_author"]).astype(int)
    df["named_self"] = (df["guessed_model"] == df["judge"]).astype(int)
    return df.reset_index(drop=True)


def tensors(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                        list[str]]:
    """C[p, a, j] (correct), F[p, a, j] (judge named itself) and G[p, a, j]
    (index of the model named), each 38 x 5 x 5.

    Refuses anything but a complete, balanced design: the bootstrap and the
    randomization test below both rely on one judgment per (prompt, a, j).
    """
    prompts = sorted(df["prompt_id"].unique())
    pi = {p: i for i, p in enumerate(prompts)}
    mi = {m: i for i, m in enumerate(MODELS)}
    C = np.full((len(prompts), K, K), -1, dtype=int)
    F = np.full_like(C, -1)
    G = np.full_like(C, -1)
    for r in df.itertuples():
        idx = (pi[r.prompt_id], mi[r.true_author], mi[r.judge])
        if C[idx] != -1:
            raise SystemExit(f"stats_revision: duplicate judgment at {idx}")
        C[idx], F[idx], G[idx] = r.correct, r.named_self, mi[r.guessed_model]
    if (C < 0).any():
        raise SystemExit("stats_revision: design is not complete and balanced")
    return C.astype(float), F.astype(float), G, prompts


def cell_rates(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(B, P) prompt weights x (P, 5, 5) outcomes -> (B, 5, 5) cell rates."""
    return np.einsum("bp,paj->baj", w, X) / w.sum(axis=1)[:, None, None]


def summaries(R: np.ndarray, Fr: np.ndarray) -> dict[str, np.ndarray]:
    """Every per-judge rate the paper uses, from (B, 5, 5) cell rates.

    Row a = author, column j = judge. For judge J:
      self      R[J, J]                      (38 judgments)
      peer      mean of row J off-diagonal   (152: peers on J's text)
      nonself   mean of column J off-diag    (152: J on others' text)
      fa        same for Fr                  (152: J naming itself wrongly)
    """
    diag = np.diagonal(R, axis1=1, axis2=2)
    fdiag = np.diagonal(Fr, axis1=1, axis2=2)
    self_ = diag
    peer = (R.sum(axis=2) - diag) / (K - 1)
    nonself = (R.sum(axis=1) - diag) / (K - 1)
    fa = (Fr.sum(axis=1) - fdiag) / (K - 1)
    return {"self": self_, "peer": peer, "adv": self_ - peer,
            "nonself": nonself, "fa": fa}


def third_party(R: np.ndarray, j: int, k: int) -> np.ndarray:
    """Judge j's accuracy on text by the three authors other than j and k."""
    others = [a for a in range(K) if a not in (j, k)]
    return R[:, others, j].mean(axis=1)


def weighted_adv(R: np.ndarray) -> dict[str, np.ndarray]:
    """C8(ii): peer baseline adjusted for each peer's attribution skill.

    For judge J and peer K, both are scored on the SAME 114 responses by the
    three third-party authors. The additive version shifts K's hit rate on J's
    text by (J's third-party accuracy - K's third-party accuracy), so the
    adjusted self-advantage is a difference in differences:

        mean_K [ (self_J - s_J^{JK}) - (h_KJ - s_K^{JK}) ]

    i.e. how much better J does on its own text than on third-party text,
    against how much better K does on J's text than on the same third-party
    text. The ratio version rescales h_KJ by s_J / s_K instead.
    """
    B = R.shape[0]
    add = np.zeros((B, K))
    ratio = np.zeros((B, K))
    for j in range(K):
        a_terms, r_terms = [], []
        for k in range(K):
            if k == j:
                continue
            h = R[:, j, k]
            sj, sk = third_party(R, j, k), third_party(R, k, j)
            a_terms.append(h + (sj - sk))
            with np.errstate(divide="ignore", invalid="ignore"):
                r_terms.append(np.where(sk > 0, h * sj / sk, np.nan))
        add[:, j] = R[:, j, j] - np.mean(a_terms, axis=0)
        ratio[:, j] = R[:, j, j] - np.mean(r_terms, axis=0)
    return {"wadv_add": add, "wadv_ratio": ratio}


def pct_ci(x: np.ndarray) -> list[float]:
    """Percentile 95% interval, ignoring NaN resamples."""
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] \
        if len(x) else [float("nan")] * 2


def boot_p(x: np.ndarray, null: float = 0.0) -> float:
    """Two-sided bootstrap p: twice the smaller tail mass beyond the null.

    Inverts the percentile interval, so p < .05 exactly when the 95%
    interval excludes the null.
    """
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan")
    lo = (np.sum(x <= null) + 1) / (len(x) + 1)
    hi = (np.sum(x >= null) + 1) / (len(x) + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(p: list[float]) -> list[float]:
    """Holm-Bonferroni over the finite entries; NaN stays NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out.tolist()


def binom_ci(k: int, n: int, method: str) -> list[float]:
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method=method)
    return [float(ci.low), float(ci.high)]


def gee(formula: str, data: pd.DataFrame, offset: np.ndarray | None = None):
    """Binomial GEE, exchangeable within prompt; robust and bias-reduced SEs.

    38 clusters is few enough that the plain sandwich can run small, so the
    Mancl-DeRouen bias-reduced covariance is reported beside it.
    """
    model = sm.GEE.from_formula(formula, groups="prompt_id", data=data,
                                offset=offset, family=sm.families.Binomial(),
                                cov_struct=sm.cov_struct.Exchangeable())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        robust = model.fit()
        reduced = model.fit(cov_type="bias_reduced")
    return robust, reduced


def gee_term(robust, reduced, term: str) -> dict:
    b = float(robust.params[term])
    return {"log_or": b, "or": float(np.exp(b)),
            "se_robust": float(robust.bse[term]),
            "p_robust": float(robust.pvalues[term]),
            "se_bias_reduced": float(reduced.bse[term]),
            "p_bias_reduced": float(reduced.pvalues[term]),
            "working_corr": float(robust.model.cov_struct.dep_params)}


def degenerate(k: int, n: int) -> bool:
    return k == 0 or k == n


# ------------------------------------------------ C6(a). randomization test
def randomization_test(C: np.ndarray, j: int, rng: np.random.Generator) -> dict:
    """Which of the five judgments on each of J's responses is labelled "self"?

    Under H0 (no self-advantage, in the sharp form: the five judgments on a
    response are exchangeable with respect to judge identity), relabelling any
    one of the five as "self" leaves the joint distribution unchanged. Each of
    J's 38 responses sits in a different prompt, so the relabelling is done
    independently per prompt and the response's total number of correct
    judgments T_p is held fixed -- which is exactly what conditions away how
    identifiable that response is and keeps the four peers on it together.

    The statistic is a monotone function of S = number of "self" hits:
        adv = S/38 - (T - S)/152,
    and S is a sum of independent Bernoulli(T_p / 5), so the permutation
    distribution is Poisson-binomial and the p-value is exact. The Monte Carlo
    version is reported as a check.
    """
    Y = C[:, j, :]                        # (38 responses, 5 judges)
    P = Y.shape[0]
    n_self, n_peer = P, P * (K - 1)
    T = Y.sum(axis=1)
    s_obs = Y[:, j].sum()

    def adv(s):
        return s / n_self - (T.sum() - s) / n_peer

    obs = adv(s_obs)
    # Exact: Poisson-binomial pmf by convolution.
    pmf = np.array([1.0])
    for q in T / K:
        pmf = np.convolve(pmf, [1 - q, q])
    support = adv(np.arange(len(pmf)))
    tol = 1e-12
    p_exact = float(pmf[np.abs(support) >= abs(obs) - tol].sum())
    p_lower = float(pmf[support <= obs + tol].sum())
    p_upper = float(pmf[support >= obs - tol].sum())

    # Monte Carlo: draw the "self" column per response, vectorised.
    pick = rng.integers(0, K, size=(N_PERM, P))
    s_perm = Y[np.arange(P)[None, :], pick].sum(axis=1)
    null = adv(s_perm)
    p_mc = float((np.sum(np.abs(null) >= abs(obs) - tol) + 1) / (N_PERM + 1))
    return {"adv": float(obs), "p_exact": min(1.0, p_exact), "p_mc": p_mc,
            "p_exact_lower": p_lower, "p_exact_upper": p_upper,
            "null_sd": float(null.std())}


# ------------------------------------------------ C2. guess-shuffle floor
def guess_shuffle(G: np.ndarray, j: int) -> dict:
    """Non-self accuracy against the judge's OWN guessing habits, exactly.

    The 20% floor assumes guesses spread over all five labels. A judge that
    never names itself on others' text has a 25% floor by always naming the
    same one of the other four, and a judge whose guesses pile onto one model
    can sit anywhere near that. This null keeps J's guesses and shuffles them:
    within each prompt, J's four labels on the four responses it did not write
    are permuted among those responses (in the lineup, that keeps each
    session's multiset of labels and so the matching-puzzle structure). Under
    H0 (the label J gives is unrelated to which of the four texts it is on),
    every one of the 4! assignments is equally likely, prompts are
    independent, and the null distribution of the correct count is an exact
    convolution of 38 per-prompt distributions.
    """
    from itertools import permutations
    others = [a for a in range(K) if a != j]
    perms = np.array(list(permutations(range(K - 1))))          # 24 x 4
    g = G[:, others, j]                                          # 38 x 4
    target = np.array(others)
    # hits[p, r]: correct count on prompt p under relabelling r.
    hits = (g[:, perms] == target[None, None, :]).sum(axis=2)   # 38 x 24
    pmf = np.array([1.0])
    for row in hits:
        pmf = np.convolve(pmf, np.bincount(row, minlength=K) / len(perms))
    obs = int((g == target[None, :]).sum())
    x = np.arange(len(pmf))
    mean = float((x * pmf).sum())
    tol = 1e-9
    n = g.size
    return {"floor": mean / n, "k": obs, "n": n,
            "p_one_sided": float(pmf[x >= obs - tol].sum()),
            "p_two_sided": float(min(1.0, pmf[np.abs(x - mean)
                                              >= abs(obs - mean) - tol].sum()))}


# ------------------------------------------------ C6(d). crossed random effects
def mixed_model(df: pd.DataFrame, hits: dict[str, int]) -> dict:
    """correct ~ author + per-judge self terms, random intercepts prompt, judge.

    The judge random intercept cannot sit in a per-judge model on the 190
    judgments of J's text (is_self is then a judge-level dummy), so this is
    fitted on all 950. The self term is therefore net of the judge's general
    skill, which makes it a random-effects cousin of C8(i), not of C6(c).
    Variational Bayes with statsmodels' default N(0, 2^2) prior on fixed
    effects; the MAP fit is run beside it and the model is flagged unstable if
    the two disagree by more than half a logit on any estimable term.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    d = df.copy()
    for m in MODELS:
        d[f"self_{m}"] = d["is_self"] * (d["true_author"] == m)
    formula = "correct ~ 0 + C(true_author) + " + " + ".join(
        f"self_{m}" for m in MODELS)
    vc = {"prompt": "0 + C(prompt_id)", "judge": "0 + C(judge)"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = BinomialBayesMixedGLM.from_formula(formula, vc, d)
            vb = model.fit_vb()
            mp = model.fit_map()
    except Exception as exc:  # noqa: BLE001 -- optional analysis; report it
        return {"error": f"{type(exc).__name__}: {exc}"}
    # The optimiser's last few digits wobble between runs (BLAS threading), so
    # everything from this fit is rounded to 4 places to keep the JSON
    # byte-identical run to run.
    def r4(x: float) -> float:
        return round(float(x), 4)

    names = list(model.exog_names)
    out, unstable = {}, False
    for m in MODELS:
        i = names.index(f"self_{m}")
        mean, sd = float(vb.fe_mean[i]), float(vb.fe_sd[i])
        est = not degenerate(hits[m], 38)
        gap = abs(mean - float(mp.fe_mean[i]))
        unstable |= est and gap > 0.5
        out[m] = {"log_or": r4(mean), "or": r4(np.exp(mean)), "post_sd": r4(sd),
                  "p_normal_approx": r4(2 * stats.norm.sf(abs(mean / sd))),
                  "map_log_or": r4(mp.fe_mean[i]),
                  "estimable": est,
                  "note": None if est else "0 or 38 self hits: estimate is the "
                                           "prior, not the data"}
    vcp = list(model.vcp_names)
    return {"per_judge": out,
            "sd_prompt": r4(np.exp(vb.vcp_mean[vcp.index("prompt")])),
            "sd_judge": r4(np.exp(vb.vcp_mean[vcp.index("judge")])),
            "stable": not unstable}


# ---------------------------------------------------------------- per condition
def analyse(cond: str, old: dict) -> dict:
    df = load(cond)
    C, F, G, prompts = tensors(df)
    P = len(prompts)
    obs_R = cell_rates(C, np.ones((1, P)))
    obs_F = cell_rates(F, np.ones((1, P)))
    obs = {k: v[0] for k, v in summaries(obs_R, obs_F).items()}
    obs.update({k: v[0] for k, v in weighted_adv(obs_R).items()})

    # One set of prompt-bootstrap weights shared by every statistic, so all
    # intervals in this condition come from the same resamples.
    w = rng_for(cond, 1).multinomial(P, np.full(P, 1 / P), size=N_BOOT)
    bR, bF = cell_rates(C, w), cell_rates(F, w)
    boot = summaries(bR, bF)
    boot.update(weighted_adv(bR))

    old_fisher = {r["model"]: r["fisher_p"] for r in old[cond]["self_advantage"]}
    old_binom = {r["judge"]: r["p_vs_chance"]
                 for r in old[cond]["self_recognition"]}

    per_judge = {}
    for j, J in enumerate(MODELS):
        k_self = int(C[:, j, j].sum())
        k_peer = int(C[:, j, :].sum() - k_self)
        k_fa = int(F[:, :, j].sum() - F[:, j, j].sum())
        k_ns = int(C[:, :, j].sum() - k_self)
        n_self, n_other = P, P * (K - 1)

        # C6(a) randomization and (b) bootstrap.
        rt = randomization_test(C, j, rng_for(cond, 10 + j))
        # C6(c) GEE on the 190 judgments of J's text.
        sub = df[df["true_author"] == J]
        if degenerate(k_self, n_self) or degenerate(k_peer, n_other):
            g6 = {"estimable": False,
                  "note": f"{k_self}/{n_self} self, {k_peer}/{n_other} peer: "
                          "separation, the log-odds ratio is infinite"}
        else:
            g6 = {"estimable": True, **gee_term(*gee("correct ~ is_self", sub),
                                                "is_self")}

        # C2: non-self accuracy against the floor, exact and prompt-clustered.
        ns_prompt = (C[:, :, j].sum(axis=1) - C[:, j, j]) / (K - 1)
        t2 = stats.ttest_1samp(ns_prompt, CHANCE)
        t1 = stats.ttest_1samp(ns_prompt, CHANCE, alternative="greater")
        nsd = df[(df["judge"] == J) & (df["is_self"] == 0)]
        gns = gee_term(*gee("correct ~ 1", nsd,
                            offset=np.full(len(nsd), np.log(CHANCE / (1 - CHANCE)))),
                       "Intercept")
        gs = guess_shuffle(G, j)

        per_judge[J] = {
            "self": {
                "k": k_self, "n": n_self, "rate": obs["self"][j],
                "ci_clopper_pearson": binom_ci(k_self, n_self, "exact"),
                "ci_wilson": binom_ci(k_self, n_self, "wilson"),
                "p_vs_chance": float(stats.binomtest(k_self, n_self, CHANCE).pvalue),
                "p_vs_chance_paper": old_binom[J],
            },
            "peer": {"k": k_peer, "n": n_other, "rate": obs["peer"][j],
                     "ci_cluster": pct_ci(boot["peer"][:, j]),
                     "ci_clopper_pearson_naive": binom_ci(k_peer, n_other, "exact")},
            "false_alarm": {"k": k_fa, "n": n_other, "rate": obs["fa"][j],
                            "ci_cluster": pct_ci(boot["fa"][:, j]),
                            "ci_clopper_pearson_naive": binom_ci(k_fa, n_other, "exact")},
            "nonself": {
                "k": k_ns, "n": n_other, "rate": obs["nonself"][j],
                "ci_cluster": pct_ci(boot["nonself"][:, j]),
                "ci_clopper_pearson_naive": binom_ci(k_ns, n_other, "exact"),
                "p_exact_two_sided": float(stats.binomtest(k_ns, n_other, CHANCE).pvalue),
                "p_exact_one_sided": float(stats.binomtest(
                    k_ns, n_other, CHANCE, alternative="greater").pvalue),
                "p_cluster_t_two_sided": float(t2.pvalue),
                "p_cluster_t_one_sided": float(t1.pvalue),
                "p_cluster_gee_two_sided": gns["p_robust"],
                "p_cluster_gee_bias_reduced": gns["p_bias_reduced"],
                # Against J's own guessing habits rather than 20%.
                "shuffle_floor": gs["floor"],
                "p_shuffle_one_sided": gs["p_one_sided"],
                "p_shuffle_two_sided": gs["p_two_sided"],
            },
            "self_advantage": {
                "adv": obs["adv"][j],
                "ci_cluster": pct_ci(boot["adv"][:, j]),
                "p_bootstrap": boot_p(boot["adv"][:, j]),
                "p_fisher_paper": old_fisher[J],
                "randomization": rt,
                "gee": g6,
            },
            "weighted": {
                "additive": {"adv": obs["wadv_add"][j],
                             "adjusted_peer": obs["self"][j] - obs["wadv_add"][j],
                             "ci_cluster": pct_ci(boot["wadv_add"][:, j]),
                             "p_bootstrap": boot_p(boot["wadv_add"][:, j])},
                "ratio": {"adv": obs["wadv_ratio"][j],
                          "adjusted_peer": obs["self"][j] - obs["wadv_ratio"][j],
                          "ci_cluster": pct_ci(boot["wadv_ratio"][:, j]),
                          "p_bootstrap": boot_p(boot["wadv_ratio"][:, j]),
                          "undefined_resamples": int(np.sum(
                              ~np.isfinite(boot["wadv_ratio"][:, j])))},
            },
        }
        # Empirical design effect of the peer baseline: cluster-bootstrap
        # variance against the binomial variance that treats the 152 as iid.
        p_hat = obs["peer"][j]
        binvar = p_hat * (1 - p_hat) / n_other
        per_judge[J]["peer"]["deff_bootstrap"] = (
            float(boot["peer"][:, j].var() / binvar) if binvar > 0 else float("nan"))

    # ---- Holm families within this condition (family 3 spans conditions and
    # is applied in main()).
    perm_p = [per_judge[J]["self_advantage"]["randomization"]["p_exact"] for J in MODELS]
    gee_p = [per_judge[J]["self_advantage"]["gee"].get("p_robust", np.nan) for J in MODELS]
    gee_br = [per_judge[J]["self_advantage"]["gee"].get("p_bias_reduced", np.nan)
              for J in MODELS]
    chance_p = [per_judge[J]["self"]["p_vs_chance"] for J in MODELS]
    for J, a, b, c, d in zip(MODELS, holm(perm_p), holm(gee_p), holm(gee_br),
                             holm(chance_p)):
        per_judge[J]["self_advantage"]["p_holm_randomization"] = a
        per_judge[J]["self_advantage"]["p_holm_gee"] = b
        per_judge[J]["self_advantage"]["p_holm_gee_bias_reduced"] = c
        per_judge[J]["self"]["p_holm"] = d

    hits = {J: per_judge[J]["self"]["k"] for J in MODELS}
    return {"n_judgments": len(df), "n_prompts": P,
            "per_judge": per_judge,
            "mixed_model": mixed_model(df, hits),
            "c3": pairwise(cond, C, obs, boot, w),
            "c8_gee": c8_gee(df, hits),
            "icc": icc_block(C),
            # Observed 5 x 5 accuracy table, rows = author, cols = judge.
            "cell_rates_author_by_judge": obs_R[0].tolist()}


# ------------------------------------------------------------ C3. pairwise
def pairwise(cond: str, C: np.ndarray, obs: dict, boot: dict, w: np.ndarray) -> dict:
    """Is any judge's non-self accuracy actually higher than another's?

    Two versions, because A's and B's non-self sets differ (A's excludes A's
    own text, B's excludes B's):

      full    each judge on its own 152 non-self judgments, the paper's
              comparison; prompt-cluster bootstrap CI and p, plus Fisher on
              the 152 vs 152 as the naive reference;
      common  both judges on the same 114 responses by the other three
              authors, paired within prompt; a sign-flip test swaps the A/B
              labels for all of a prompt's judgments at once (20,000 draws),
              valid because both judges saw exactly those responses.
    """
    P = C.shape[0]
    signs = rng_for(cond, 30).choice([-1.0, 1.0], size=(N_PERM, P))
    pairs = []
    for ia in range(K):
        for ib in range(ia + 1, K):
            A, B = MODELS[ia], MODELS[ib]
            k_a = int(round(obs["nonself"][ia] * P * (K - 1)))
            k_b = int(round(obs["nonself"][ib] * P * (K - 1)))
            n = P * (K - 1)
            diff_b = boot["nonself"][:, ia] - boot["nonself"][:, ib]
            common = [a for a in range(K) if a not in (ia, ib)]
            d = (C[:, common, ia] - C[:, common, ib]).sum(axis=1)
            n_c = P * len(common)
            obs_c = d.sum() / n_c
            null = signs @ d / n_c
            p_flip = float((np.sum(np.abs(null) >= abs(obs_c) - 1e-12) + 1)
                           / (N_PERM + 1))
            pairs.append({
                "a": A, "b": B,
                "nonself_a": float(obs["nonself"][ia]),
                "nonself_b": float(obs["nonself"][ib]),
                "diff": float(obs["nonself"][ia] - obs["nonself"][ib]),
                "ci_cluster": pct_ci(diff_b), "p_bootstrap": boot_p(diff_b),
                "p_fisher_naive": float(stats.fisher_exact(
                    [[k_a, n - k_a], [k_b, n - k_b]])[1]),
                "common_n": n_c,
                "common_acc_a": float(C[:, common, ia].mean()),
                "common_acc_b": float(C[:, common, ib].mean()),
                "common_diff": float(obs_c), "p_signflip": p_flip,
            })
    for pr, h in zip(pairs, holm([p["p_signflip"] for p in pairs])):
        pr["p_signflip_holm"] = h
    for pr, h in zip(pairs, holm([p["p_bootstrap"] for p in pairs])):
        pr["p_bootstrap_holm"] = h

    order = np.argsort(-obs["nonself"])
    top, second = MODELS[order[0]], MODELS[order[1]]
    ia_ = [MODELS.index(m) for m in NO_ADV]
    ib_ = [MODELS.index(m) for m in ADV]
    g_obs = obs["nonself"][ia_].mean() - obs["nonself"][ib_].mean()
    g_boot = boot["nonself"][:, ia_].mean(axis=1) - boot["nonself"][:, ib_].mean(axis=1)
    return {
        "ranking": [MODELS[i] for i in order],
        "top": top, "runner_up": second,
        "pairs": pairs,
        "group_contrast": {"no_adv": NO_ADV, "adv": ADV, "diff": float(g_obs),
                           "ci_cluster": pct_ci(g_boot),
                           "p_bootstrap": boot_p(g_boot)},
    }


# ------------------------------------------------------------ C8(i). GEE
def c8_gee(df: pd.DataFrame, hits: dict[str, int]) -> dict:
    """correct ~ judge + author + per-judge self term, all 950, GEE by prompt.

    The judge effect absorbs each judge's general attribution skill and the
    author effect how identifiable each author is to everyone, so self_J is
    J's advantage on its own text net of both: the logit-scale analogue of
    the difference in differences in weighted_adv. A judge with 0 (or 38)
    self hits has an infinite self term; its 38 self rows are dropped, which
    is exactly the limit of the MLE with that dummy in the model, and it is
    reported as not estimable.
    """
    d = df.copy()
    est = [m for m in MODELS if not degenerate(hits[m], 38)]
    drop = [m for m in MODELS if m not in est]
    d = d[~((d["is_self"] == 1) & d["judge"].isin(drop))].copy()
    for m in est:
        d[f"self_{m}"] = d["is_self"] * (d["judge"] == m)
    terms = " + ".join(f"self_{m}" for m in est)
    per = {}
    try:
        rob, red = gee(f"correct ~ C(judge) + C(true_author) + {terms}", d)
        for m in est:
            per[m] = {"estimable": True, **gee_term(rob, red, f"self_{m}")}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    for m in drop:
        per[m] = {"estimable": False,
                  "note": f"{hits[m]}/38 self hits: infinite self term"}
    # One pooled self term on all 950 (no rows dropped), for the average.
    try:
        rob, red = gee("correct ~ C(judge) + C(true_author) + is_self", df)
        pooled = gee_term(rob, red, "is_self")
    except Exception as exc:  # noqa: BLE001
        pooled = {"error": f"{type(exc).__name__}: {exc}"}
    return {"per_judge": per, "pooled_is_self": pooled}


# ------------------------------------------------------------ ICC
def anova_icc(Y: np.ndarray) -> float:
    """One-way random-effects ICC(1) for equal cluster sizes; Y is (groups, m)."""
    g, m = Y.shape
    gm = Y.mean(axis=1)
    msb = m * ((gm - Y.mean()) ** 2).sum() / (g - 1)
    msw = ((Y - gm[:, None]) ** 2).sum() / (g * (m - 1))
    den = msb + (m - 1) * msw
    return float((msb - msw) / den) if den > 0 else float("nan")


def icc_block(C: np.ndarray) -> dict:
    """How correlated is correctness within a prompt and within a response?

    Raw ICCs include the design's fixed contrasts (Claude's text is easier for
    everyone, GPT is a better judge), so each is also given on residuals after
    removing the 25 judge x author cell means -- the shared difficulty of a
    prompt or response beyond who wrote it and who judged it.

    For the peer baseline, the clusters are J's 38 responses with 4 peer
    judgments each; the analytic design effect is 1 + 3 * ICC, pooled over J
    with J-specific means.
    """
    P = C.shape[0]
    resid = C - C.mean(axis=0, keepdims=True)
    out = {
        "prompt_raw": anova_icc(C.reshape(P, -1)),
        "prompt_resid": anova_icc(resid.reshape(P, -1)),
        "response_raw": anova_icc(C.reshape(P * K, K)),
        "response_resid": anova_icc(resid.reshape(P * K, K)),
    }
    per, ssb, ssw = {}, 0.0, 0.0
    for j, J in enumerate(MODELS):
        Y = np.delete(C[:, j, :], j, axis=1)          # 38 x 4 peer judgments
        per[J] = anova_icc(Y)
        gm = Y.mean(axis=1)
        ssb += (K - 1) * ((gm - Y.mean()) ** 2).sum()
        ssw += ((Y - gm[:, None]) ** 2).sum()
    m = K - 1
    msb, msw = ssb / (K * (P - 1)), ssw / (K * P * (m - 1))
    pooled = (msb - msw) / (msb + (m - 1) * msw)
    out["peer_icc_per_judge"] = per
    out["peer_icc_pooled"] = float(pooled)
    out["peer_deff"] = float(1 + (m - 1) * pooled)
    out["peer_n_eff"] = float(P * m / (1 + (m - 1) * pooled))
    return out


# ---------------------------------------------------------------- family 3
def family3(res: dict) -> dict:
    """C2: the ten non-self cells against the floor, Holm across all ten."""
    cells = [(c, J) for c in CONDITIONS for J in MODELS]
    keys = ["p_exact_two_sided", "p_exact_one_sided",
            "p_cluster_t_two_sided", "p_cluster_t_one_sided",
            "p_cluster_gee_two_sided", "p_shuffle_one_sided",
            "p_shuffle_two_sided"]
    counts = {}
    for key in keys:
        raw = [res[c]["per_judge"][J]["nonself"][key] for c, J in cells]
        adj = holm(raw)
        for (c, J), a in zip(cells, adj):
            res[c]["per_judge"][J]["nonself"][key + "_holm"] = a
        # "Clears the floor" = above the floor (20%, or the shuffle floor for
        # the shuffle test) and significant; a cell significantly BELOW its
        # floor does not count.
        floor = [res[c]["per_judge"][J]["nonself"]["shuffle_floor"]
                 if "shuffle" in key else CHANCE for c, J in cells]
        above = [res[c]["per_judge"][J]["nonself"]["rate"] > f
                 for (c, J), f in zip(cells, floor)]
        clear_raw = [p < ALPHA and u for p, u in zip(raw, above)]
        clear_adj = [p < ALPHA and u for p, u in zip(adj, above)]
        counts[key] = {
            "raw": int(sum(clear_raw)), "holm": int(sum(clear_adj)),
            "failing_raw": [f"{J} {c}" for (c, J), ok in zip(cells, clear_raw) if not ok],
            "failing_holm": [f"{J} {c}" for (c, J), ok in zip(cells, clear_adj) if not ok],
        }
    return counts


def pattern(adv: float, p: float) -> str:
    if adv is None or not np.isfinite(adv):
        return "n/a"
    if not np.isfinite(p) or p >= ALPHA:
        return "0"
    return "+" if adv > 0 else "-"


def c8_summary(res: dict) -> dict:
    """Does the paper's qualitative self-advantage pattern survive C8?"""
    out = {}
    for c in CONDITIONS:
        rows = {}
        for J in MODELS:
            pj = res[c]["per_judge"][J]
            sa, wa = pj["self_advantage"], pj["weighted"]
            g = res[c]["c8_gee"]["per_judge"][J]
            rows[J] = {
                "paper": PAPER_PATTERN[c][J],
                "unadjusted_randomization": pattern(sa["adv"], sa["randomization"]["p_exact"]),
                "weighted_additive": pattern(wa["additive"]["adv"], wa["additive"]["p_bootstrap"]),
                "weighted_ratio": pattern(wa["ratio"]["adv"], wa["ratio"]["p_bootstrap"]),
                "gee_judge_author": (pattern(g["log_or"], g["p_robust"])
                                     if g.get("estimable") else "n/a"),
            }
        out[c] = rows
    return out


# ---------------------------------------------------------------- JSON / MD
def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def f_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def f_pct(x) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"


def f_ci(ci, pp: bool = True) -> str:
    if ci is None or not all(np.isfinite(ci)):
        return "--"
    return f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if pp else \
        f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def f_pp(x) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}"


def sig(p) -> str:
    return "significant" if p is not None and np.isfinite(p) and p < ALPHA \
        else "not significant"


def write_md(res: dict, runtime: float) -> str:
    L = []
    add = L.append
    add("# Statistics for the TACL revision (C2, C3, C6, C7, C8)\n")
    add("**Generated file.** `python src/stats_revision.py` writes this and "
        "`results/stats_revision.json`; do not edit by hand. "
        f"Seed {SEED}; {N_PERM:,} permutations; {N_BOOT:,} prompt-bootstrap "
        f"resamples; runtime {runtime:.0f} s. Percentages are percent, "
        "differences are percentage points (pp). All tests two-sided unless "
        "marked one-sided.\n")
    add("Design: 38 prompts x 5 authors = 190 responses; every response is "
        "judged once by each of the 5 judges, in each condition. For judge J: "
        "self rate = J on its 38 responses; peer baseline = the other four "
        "judges on the same 38 (152 judgments); self-advantage = self - peer; "
        "false alarm = J naming itself on the 152 responses it did not write; "
        "non-self accuracy = J's accuracy on those 152.\n")

    # ------------------------------------------------------------ C6
    add("## C6. Clustered inference for self-advantage\n")
    add("**Why the old test is wrong.** Fisher's exact test treats the 152 "
        "peer judgments as independent Bernoulli draws. They come four to a "
        "response, and a response that is easy for one peer is easy for the "
        "others (see the ICC section), so the peer rate varies more than "
        "binomial and Fisher's p is too small.\n")
    add("**(a) Randomization test, and why the labels are exchangeable.** "
        "Each of J's 38 responses is judged by all five judges, one of whom "
        "is J. H0 (sharp form): on any given response, the five judgments are "
        "exchangeable with respect to which judge wrote the text -- the "
        "author's own judgment is no more likely to be correct than a peer's. "
        "Under H0, relabelling a uniformly chosen one of the five as \"self\" "
        "leaves the joint distribution unchanged. Relabelling is done "
        "independently per response; since J wrote exactly one response per "
        "prompt, that is also per prompt, so prompts (and lineup sessions, "
        "which never span prompts) are the independent units. Holding each "
        "response's number of correct judgments fixed conditions away how "
        "identifiable that particular response is and keeps its four peer "
        "judgments together, which is the clustering Fisher ignored. The "
        "statistic, self - peer, is monotone in the number of \"self\" hits, "
        "a sum of independent Bernoulli(T_p/5); so the permutation "
        "distribution is Poisson-binomial and the p-value is exact. The "
        f"{N_PERM:,}-draw Monte Carlo p is reported as a check. Caveat: the "
        "sharp null also assumes the judges are equally skilled; a judge that "
        "is simply a better attributor overall would show a positive "
        "\"self-advantage\" here. C8 removes that.\n")
    add("**(b)** Prompt-level cluster bootstrap: resample the 38 prompts with "
        "replacement, recompute everything, percentile 95% CI.  "
        "**(c)** GEE logistic, correct ~ is_self on the 190 judgments of J's "
        "text, binomial, exchangeable working correlation within prompt; the "
        "robust (sandwich) p and the Mancl-DeRouen bias-reduced p (38 "
        "clusters is few).  Holm is applied across the five judges within "
        "each condition (family 1), to the randomization p.\n")
    for c in CONDITIONS:
        r = res[c]
        add(f"### {c}\n")
        add("| Judge | self | peer [cluster CI] | self-adv pp [cluster CI] | "
            "Fisher p (paper) | randomization p exact (MC) | GEE OR, p robust "
            "(bias-red.) | Holm p (rand.) | weaker than paper? |")
        add("|---|---|---|---|---|---|---|---|---|")
        for J in MODELS:
            pj = r["per_judge"][J]
            sa = pj["self_advantage"]
            rt = sa["randomization"]
            g = sa["gee"]
            gtxt = (f"{g['or']:.2f}, {f_p(g['p_robust'])} ({f_p(g['p_bias_reduced'])})"
                    if g.get("estimable") else "not estimable")
            weaker = "yes" if rt["p_exact"] > sa["p_fisher_paper"] else "no"
            add(f"| {J} | {f_pct(pj['self']['rate'])} | "
                f"{f_pct(pj['peer']['rate'])} {f_ci(pj['peer']['ci_cluster'], False)} | "
                f"{100 * sa['adv']:+.1f} {f_ci(sa['ci_cluster'])} | "
                f"{f_p(sa['p_fisher_paper'])} | {f_p(rt['p_exact'])} ({f_p(rt['p_mc'])}) | "
                f"{gtxt} | {f_p(sa['p_holm_randomization'])} | {weaker} |")
        add("")
        for J in MODELS:
            sa = r["per_judge"][J]["self_advantage"]
            rt = sa["randomization"]
            old_s, new_s = sig(sa["p_fisher_paper"]), sig(sa["p_holm_randomization"])
            change = ("verdict unchanged" if old_s == new_s
                      else f"verdict changes: was {old_s}, now {new_s} after Holm")
            add(f"- **{J}**: {100 * sa['adv']:+.1f} pp; exact randomization "
                f"p = {f_p(rt['p_exact'])} vs Fisher {f_p(sa['p_fisher_paper'])} "
                f"({'weaker' if rt['p_exact'] > sa['p_fisher_paper'] else 'not weaker'}); "
                f"Holm {f_p(sa['p_holm_randomization'])}, {change}.")
        add("")
        mm = r["mixed_model"]
        add("**(d) Crossed random effects** (all 950 judgments; correct ~ "
            "author + per-judge self term, random intercepts for prompt and "
            "judge; variational Bayes, N(0, 2^2) prior on fixed effects). "
            "Because the judge intercept absorbs general skill, this is "
            "closer to C8 than to (a)-(c).")
        if "error" in mm:
            add(f"Fit failed: {mm['error']}. Skipped.\n")
        else:
            add(f"Posterior SD of prompt intercepts {mm['sd_prompt']:.2f}, "
                f"judge intercepts {mm['sd_judge']:.2f}; VB and MAP "
                f"{'agree within 0.5 logit' if mm['stable'] else 'DISAGREE by more than 0.5 logit: treat as unstable'}.\n")
            add("| Judge | self log-OR (post. SD) | OR | approx p | MAP log-OR | note |")
            add("|---|---|---|---|---|---|")
            for J in MODELS:
                m = mm["per_judge"][J]
                add(f"| {J} | {m['log_or']:+.2f} ({m['post_sd']:.2f}) | "
                    f"{m['or']:.2f} | {f_p(m['p_normal_approx'])} | "
                    f"{m['map_log_or']:+.2f} | {m['note'] or ''} |")
            add("")

    # ------------------------------------------------------------ C7
    add("## C7. Intervals on every rate, and Holm families\n")
    add("Self rates: 38 judgments, one per prompt, so no clustering; exact "
        "Clopper-Pearson and Wilson. False alarms, peer baselines and non-self "
        "accuracy: 152 judgments, four per prompt, so a prompt-cluster "
        "bootstrap CI (naive Clopper-Pearson in the JSON for comparison). "
        "Family 1 (self-advantage, 5 per condition) is in C6; family 2 (self "
        "rate vs 20%, 5 per condition) and family 3 (non-self vs 20%, all 10 "
        "cells) are below.\n")
    for c in CONDITIONS:
        add(f"### {c}\n")
        add("| Judge | self k/n | self CP 95% | self Wilson 95% | self vs 20% p "
            "(Holm) | false alarm [cluster CI] | peer [cluster CI] | "
            "non-self [cluster CI] |")
        add("|---|---|---|---|---|---|---|---|")
        for J in MODELS:
            pj = res[c]["per_judge"][J]
            s = pj["self"]
            add(f"| {J} | {s['k']}/{s['n']} | {f_ci(s['ci_clopper_pearson'], False)} | "
                f"{f_ci(s['ci_wilson'], False)} | {f_p(s['p_vs_chance'])} "
                f"({f_p(s['p_holm'])}) | {f_pct(pj['false_alarm']['rate'])} "
                f"{f_ci(pj['false_alarm']['ci_cluster'], False)} | "
                f"{f_pct(pj['peer']['rate'])} {f_ci(pj['peer']['ci_cluster'], False)} | "
                f"{f_pct(pj['nonself']['rate'])} {f_ci(pj['nonself']['ci_cluster'], False)} |")
        add("")
        sr = {J: res[c]["per_judge"][J]["self"] for J in MODELS}
        n_sig = sum(v["p_holm"] < ALPHA for v in sr.values())
        flips = [f"{J} ({f_p(v['p_vs_chance'])} raw -> {f_p(v['p_holm'])} Holm)"
                 for J, v in sr.items()
                 if (v["p_vs_chance"] < ALPHA) != (v["p_holm"] < ALPHA)]
        add(f"- Family 2 ({c}): {n_sig} of 5 self rates differ from 20% after "
            "Holm. " + ("Cells that lose significance under Holm: "
                        + "; ".join(flips) + "." if flips else
                        "Holm changes no verdict.") + "\n")

    # ------------------------------------------------------------ C2
    f3 = res["family3"]
    add("## C2. Does non-self accuracy clear the 20% floor in all ten cells?\n")
    add("Per cell: exact binomial on the judge's 152 non-self judgments, and "
        "two prompt-clustered versions (one-sample t on the 38 per-prompt "
        "non-self accuracies, which have equal size 4; and an intercept-only "
        "GEE with offset logit(0.2)). Holm across the ten cells (family 3). A "
        "cell clears the floor if it is above 20% and p < .05.\n")
    add("A fourth test asks whether the judge beats its OWN guessing habits "
        "rather than 20%: within each prompt, J's four labels on the four "
        "responses it did not write are shuffled among those responses (exact "
        "null by convolution over prompts; in the lineup this keeps each "
        "session's label multiset). Its mean is the \"shuffle floor\". This "
        "matters because 20% is not the floor for a judge that rarely names "
        "itself: always naming one of the other four scores 25% on non-self "
        "text.\n")
    add("| Cell | non-self [cluster CI] | exact 2-sided (Holm) | exact 1-sided "
        "(Holm) | cluster t 2-sided (Holm) | cluster t 1-sided (Holm) | "
        "GEE 2-sided (Holm) | shuffle floor | shuffle 1-sided (Holm) |")
    add("|---|---|---|---|---|---|---|---|---|")
    for c in CONDITIONS:
        for J in MODELS:
            n = res[c]["per_judge"][J]["nonself"]
            cells = " | ".join(f"{f_p(n[k])} ({f_p(n[k + '_holm'])})" for k in (
                "p_exact_two_sided", "p_exact_one_sided", "p_cluster_t_two_sided",
                "p_cluster_t_one_sided", "p_cluster_gee_two_sided"))
            add(f"| {J} {c} | {f_pct(n['rate'])} {f_ci(n['ci_cluster'], False)} | "
                f"{cells} | {f_pct(n['shuffle_floor'])} | "
                f"{f_p(n['p_shuffle_one_sided'])} ({f_p(n['p_shuffle_one_sided_holm'])}) |")
    add("")
    add("| Test | clear raw | clear Holm | fail raw | fail Holm |")
    add("|---|---|---|---|---|")
    for k, v in f3.items():
        add(f"| {k} | {v['raw']}/10 | {v['holm']}/10 | "
            f"{', '.join(v['failing_raw']) or '--'} | {', '.join(v['failing_holm']) or '--'} |")
    add("")
    add(f"- **Answer.** The paper's \"all ten\" is wrong. {res['c2_answer']}\n")

    # ------------------------------------------------------------ C3
    add("## C3. Are the judges without self-advantage the most accurate on others' text?\n")
    add("\"full\" compares each judge's own 152 non-self judgments (different "
        "text sets), with a prompt-cluster bootstrap CI and p; Fisher on 152 "
        "vs 152 is the naive reference. \"common\" compares the two judges on "
        "the same 114 responses by the other three authors, paired within "
        f"prompt, with a {N_PERM:,}-draw sign-flip test that swaps the two "
        "judges' labels for a whole prompt at a time; Holm over the 10 pairs "
        "within a condition.\n")
    for c in CONDITIONS:
        c3 = res[c]["c3"]
        add(f"### {c}  (ranking: {' > '.join(c3['ranking'])})\n")
        add("| A vs B | non-self A / B | diff pp [cluster CI] | boot p (Holm) | "
            "Fisher naive p | common-text A / B | common diff pp | sign-flip p (Holm) |")
        add("|---|---|---|---|---|---|---|---|")
        for pr in c3["pairs"]:
            add(f"| {pr['a']} vs {pr['b']} | {f_pct(pr['nonself_a'])} / "
                f"{f_pct(pr['nonself_b'])} | {100 * pr['diff']:+.1f} "
                f"{f_ci(pr['ci_cluster'])} | {f_p(pr['p_bootstrap'])} "
                f"({f_p(pr['p_bootstrap_holm'])}) | {f_p(pr['p_fisher_naive'])} | "
                f"{f_pct(pr['common_acc_a'])} / {f_pct(pr['common_acc_b'])} | "
                f"{100 * pr['common_diff']:+.1f} | {f_p(pr['p_signflip'])} "
                f"({f_p(pr['p_signflip_holm'])}) |")
        g = c3["group_contrast"]
        add("")
        add(f"- Group contrast, mean non-self of {'/'.join(NO_ADV)} minus "
            f"{'/'.join(ADV)}: {100 * g['diff']:+.1f} pp {f_ci(g['ci_cluster'])}, "
            f"bootstrap p = {f_p(g['p_bootstrap'])}.")
        add(f"- Top vs runner-up: {c3['top']} vs {c3['runner_up']}.\n")
    add(f"- **Answer.** {res['c3_answer']}\n")

    # ------------------------------------------------------------ C8
    add("## C8. Self-advantage net of judge skill\n")
    add("A model judged mostly by strong attributors gets an inflated peer "
        "baseline, so its self-advantage is understated (and a strong judge's "
        "own advantage overstated). Two adjustments:\n")
    add("- **(i) GEE, all 950 judgments**: correct ~ judge + author + a "
        "self term per judge, exchangeable within prompt. The judge effect "
        "is general skill, the author effect is how identifiable each author "
        "is to everyone; the self term is the log-OR of J being right on its "
        "own text beyond both. Judges with 0/38 self hits have an infinite "
        "self term (not estimable).")
    add("- **(ii) Difference in differences**: for each peer K, J and K are "
        "both scored on the same 114 responses by the three other authors "
        "(third-party text). Adjusted self-advantage = mean over K of "
        "[(J on own text - J on third-party) - (K on J's text - K on "
        "third-party)], in pp; equivalently each peer's hit rate on J's text "
        "is shifted by J's minus K's third-party accuracy. The ratio variant "
        "rescales K's hit rate by J/K third-party accuracy instead. "
        "Prompt-cluster bootstrap CI and p.\n")
    for c in CONDITIONS:
        add(f"### {c}\n")
        add("| Judge | self-adv pp (unadj.) | DiD adj. peer | DiD self-adv pp "
            "[CI], p | ratio self-adv pp [CI], p | GEE judge+author self OR, p "
            "robust (bias-red.) |")
        add("|---|---|---|---|---|---|")
        for J in MODELS:
            pj = res[c]["per_judge"][J]
            wa, wr = pj["weighted"]["additive"], pj["weighted"]["ratio"]
            g = res[c]["c8_gee"]["per_judge"][J]
            gtxt = (f"{g['or']:.2f}, {f_p(g['p_robust'])} ({f_p(g['p_bias_reduced'])})"
                    if g.get("estimable") else "not estimable")
            add(f"| {J} | {f_pp(pj['self_advantage']['adv'])} | "
                f"{f_pct(wa['adjusted_peer'])} | {f_pp(wa['adv'])} "
                f"{f_ci(wa['ci_cluster'])}, {f_p(wa['p_bootstrap'])} | "
                f"{f_pp(wr['adv'])} {f_ci(wr['ci_cluster'])}, "
                f"{f_p(wr['p_bootstrap'])} | {gtxt} |")
        if c == "single":
            add("\n*Caveat:* under single-text elicitation a judge's third-party "
                "\"accuracy\" is almost entirely which model it habitually names "
                "(99% of guesses are GPT or Claude), so it measures blame "
                "bias, not skill; GPT's third-party accuracy on text by "
                "Gemini/Grok/DeepSeek is near zero, which is why the ratio "
                "is undefined for Claude and unstable for GPT. The paper "
                "already reads self-advantage from the lineup for this reason; "
                "the single-text rows here should not be used to revise it.")
        pool = res[c]["c8_gee"]["pooled_is_self"]
        if "error" not in pool:
            add(f"\nPooled self term (one is_self for all judges, net of judge "
                f"and author): OR {pool['or']:.2f}, robust p {f_p(pool['p_robust'])}.\n")
        add("Qualitative pattern (+ / - = significant at .05 in that direction, "
            "0 = not significant):\n")
        add("| Judge | paper | unadjusted (rand.) | DiD | ratio | GEE judge+author |")
        add("|---|---|---|---|---|---|")
        for J, row in res["c8_pattern"][c].items():
            add(f"| {J} | {row['paper']} | {row['unadjusted_randomization']} | "
                f"{row['weighted_additive']} | {row['weighted_ratio']} | "
                f"{row['gee_judge_author']} |")
        add("")
    add(f"- **Answer.** {res['c8_answer']}\n")

    # ------------------------------------------------------------ ICC
    add("## Effective sample size: ICC and design effect\n")
    add("One-way ANOVA ICC(1) of correctness. \"raw\" is on 0/1 correctness; "
        "\"resid\" is after removing the 25 judge x author cell means, i.e. the "
        "shared difficulty of a prompt or response beyond who wrote and who "
        "judged it. Peer baseline: clusters are J's 38 responses, 4 peer "
        "judgments each; design effect = 1 + 3 x ICC, effective n = 152 / "
        "deff. The bootstrap design effect is the cluster-bootstrap variance "
        "of the peer baseline over the binomial variance.\n")
    add("| Condition | ICC prompt raw / resid | ICC response raw / resid | "
        "peer ICC pooled | deff | effective n of 152 |")
    add("|---|---|---|---|---|---|")
    for c in CONDITIONS:
        i = res[c]["icc"]
        add(f"| {c} | {i['prompt_raw']:.3f} / {i['prompt_resid']:.3f} | "
            f"{i['response_raw']:.3f} / {i['response_resid']:.3f} | "
            f"{i['peer_icc_pooled']:.3f} | {i['peer_deff']:.2f} | {i['peer_n_eff']:.0f} |")
    add("")
    add("| Condition | " + " | ".join(f"{J} peer ICC / bootstrap deff" for J in MODELS) + " |")
    add("|---|" + "---|" * K)
    for c in CONDITIONS:
        i = res[c]["icc"]
        add(f"| {c} | " + " | ".join(
            f"{i['peer_icc_per_judge'][J]:.2f} / "
            f"{res[c]['per_judge'][J]['peer']['deff_bootstrap']:.2f}"
            for J in MODELS) + " |")
    add("")
    add(f"- {res['icc_answer']}\n")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------ verdict text
def c2_answer(res: dict) -> str:
    f3 = res["family3"]
    e1, e2 = f3["p_exact_one_sided"], f3["p_exact_two_sided"]
    t1, t2 = f3["p_cluster_t_one_sided"], f3["p_cluster_t_two_sided"]
    cl = res["lineup"]["per_judge"]["Claude"]["nonself"]
    gs = res["single"]["per_judge"]["GPT"]["nonself"]
    return (f"Exact binomial, one-sided: {e1['raw']}/10 clear raw, {e1['holm']}/10 "
            f"after Holm (fail: {', '.join(e1['failing_holm'])}). Two-sided: "
            f"{e2['raw']}/10 raw, {e2['holm']}/10 Holm. Prompt-clustered t: "
            f"{t1['raw']}/10 one-sided, {t2['raw']}/10 two-sided raw; "
            f"{t1['holm']}/10 and {t2['holm']}/10 after Holm. The plan's "
            f"p = 0.053 (Claude lineup) and p = 0.11 (GPT single) are the "
            f"ONE-SIDED exact binomial values "
            f"({f_p(cl['p_exact_one_sided'])} and {f_p(gs['p_exact_one_sided'])}); "
            f"two-sided they are {f_p(cl['p_exact_two_sided'])} and "
            f"{f_p(gs['p_exact_two_sided'])}. So \"eight of ten\" holds only "
            f"for the one-sided exact test before correction, and which two "
            f"cells fail depends on the test: the clustered tests fail "
            f"{' and '.join(t2['failing_raw'])}, while GPT single passes them "
            f"easily (its per-prompt non-self accuracy barely varies: it is "
            f"right on Claude's text and almost nowhere else). Against each "
            f"judge's own guessing habits (shuffle test) "
            f"{f3['p_shuffle_one_sided']['raw']}/10 clear "
            f"(fail: {', '.join(f3['p_shuffle_one_sided']['failing_raw'])}). "
            f"Robust statement: non-self accuracy is above chance in most "
            f"cells, but not demonstrably for "
            f"{' or '.join(sorted(set(e2['failing_raw']) & set(t2['failing_raw'])))} "
            f"(both fail the two-sided exact test and every prompt-clustered "
            f"test).")


def c3_answer(res: dict) -> str:
    bits = []
    for c in CONDITIONS:
        c3 = res[c]["c3"]
        pr = next(p for p in c3["pairs"]
                  if {p["a"], p["b"]} == {c3["top"], c3["runner_up"]})
        bits.append(f"{c}: {c3['top']} vs {c3['runner_up']} (top vs runner-up) "
                    f"{100 * abs(pr['diff']):.1f} pp, Fisher p = "
                    f"{f_p(pr['p_fisher_naive'])}, cluster bootstrap p = "
                    f"{f_p(pr['p_bootstrap'])}, common-text sign-flip p = "
                    f"{f_p(pr['p_signflip'])}")
        full = [f"{p['a']}-{p['b']}" for p in c3["pairs"]
                if p["p_bootstrap_holm"] < ALPHA]
        comm = [f"{p['a']}-{p['b']}" for p in c3["pairs"]
                if p["p_signflip_holm"] < ALPHA]
        bits.append(f"{c}: pairs that differ after Holm on the full non-self "
                    f"sets: {', '.join(full) or 'none'}; on common text: "
                    f"{', '.join(comm) or 'none'}")
        g = c3["group_contrast"]
        bits.append(f"{c}: no-advantage group minus advantage group "
                    f"{100 * g['diff']:+.1f} pp, p = {f_p(g['p_bootstrap'])}")
    return ("The plan's p = 0.91 and p = 1.00 are Fisher tests of the TOP judge "
            "against the RUNNER-UP in each condition (not top vs bottom). "
            + "; ".join(bits) + ". \"The most accurate\" is therefore a tie at "
            "the top in both conditions. Where a full-set gap is significant "
            "but the common-text gap is not, the gap comes from WHICH texts "
            "are in each judge's non-self set (a judge whose set includes "
            "Claude's widely recognized text looks better), not from skill.")


def c8_answer(res: dict) -> str:
    out = []
    for c in CONDITIONS:
        changes = []
        for J, row in res["c8_pattern"][c].items():
            for k in ("unadjusted_randomization", "weighted_additive",
                      "weighted_ratio", "gee_judge_author"):
                if row[k] not in ("n/a", row["paper"]):
                    changes.append(f"{J} {k}: {row['paper']} -> {row[k]}")
        label = ("lineup (the condition the paper reads self-advantage from)"
                 if c == "lineup" else
                 "single-text (adjustment not interpretable; see caveat)")
        out.append(f"{label}: " + ("; ".join(changes) if changes
                                   else "no departure under any method"))
    return "Departures from the paper's pattern. " + ". ".join(out) + "."


def icc_answer(res: dict) -> str:
    li, si = res["lineup"]["icc"], res["single"]["icc"]
    return (f"Peer-baseline design effect {li['peer_deff']:.2f} in the lineup "
            f"(152 peer judgments carry about {li['peer_n_eff']:.0f} "
            f"independent ones) and {si['peer_deff']:.2f} in single-text "
            f"(about {si['peer_n_eff']:.0f}).")


# ---------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    old = json.loads(OLD_JSON.read_text(encoding="utf-8"))
    res = {c: analyse(c, old) for c in CONDITIONS}
    res["family3"] = family3(res)
    res["c8_pattern"] = c8_summary(res)
    res["c2_answer"] = c2_answer(res)
    res["c3_answer"] = c3_answer(res)
    res["c8_answer"] = c8_answer(res)
    res["icc_answer"] = icc_answer(res)
    runtime = time.time() - t0
    res["meta"] = {"seed": SEED, "n_perm": N_PERM, "n_boot": N_BOOT,
                   "alpha": ALPHA, "chance": CHANCE, "runtime_s": runtime,
                   "models": MODELS}

    OUT_JSON.write_text(json.dumps(clean(res), indent=2) + "\n", encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(write_md(res, runtime), encoding="utf-8")
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}\nruntime {runtime:.1f} s")


if __name__ == "__main__":
    main()
