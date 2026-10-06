"""CompLLM: Engine B: judge-side analysis of both query conditions.

    python src/engine_b.py                      # full analysis, both conditions
    python src/engine_b.py --tex paper/numbers.tex   # also emit LaTeX macros
    python src/engine_b.py --condition lineup   # one condition only

No API key and no network required -- everything runs off results/judgments*.csv
and data/responses_v1.csv.

What this computes:

  1. Self-recognition with EXACT (Clopper-Pearson) binomial CIs against the
     20% floor. Not Wald -- at 2/38 the normal approximation puts the lower
     bound below zero.
  2. Self-advantage: a model's self-rate vs how often OTHER judges identify
     that same model's text. Fisher exact per model. This is the test that
     separates "Claude recognises itself" from "Claude's writing is simply
     the most identifiable" -- without it the diagonal is uninterpretable.
  3. Leave-one-judge-out on every headline number. With 5 judges this is 5
     points; it is a sensitivity check, NOT proof that the effect generalises.
  4. Chi-square on JUDGE-AVERAGED proportions, not pooled judgments. 950
     judgments cluster in 5 judges and 38 prompts, so a claim about judges as
     a class has n = 5. Pooled chi-square on this design is significant by
     construction; it is reported alongside for comparison only.
  5. Mixed-effects logistic on P(correct) with crossed random intercepts for
     prompt_id and judge.
  6. The mechanism test: RF *and* LogReg predicting the BLAME TARGET from the
     same 18 surface features, on wrong guesses only, against the majority
     baseline. If neither finds it, the stylometric null is robust. Reporting
     one model would leave "you used the wrong classifier" open.
  7. Sensitivity: drop the 6 Sonnet-4.6 rows from the Claude class and re-run
     self-recognition, since for those the "self" judgment is cross-model.
  8. Signal detection (signal_detection.py): the false-alarm rate beside every
     hit rate, and the leak that combines them. A self rate on its own cannot
     tell a judge that discriminates from one that just names itself a lot.
  9. Cross-attribution (cross_attribution.py): non-self accuracy and the error
     flows, folded in here so one run produces every number the paper cites.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupKFold, cross_val_predict, permutation_test_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    EXCLUDE_PROMPTS, JUDGMENTS_CSV, JUDGMENTS_SINGLE_CSV, MODELS,
    RESPONSES_CSV, RESULTS_DIR,
)
from features import FAMILIES, FEATURES, LENGTH_PROXIES, frame  # noqa: E402
from cross_attribution import analyse as cross_attribution  # noqa: E402
from engine_a import wilson  # noqa: E402
from signal_detection import (  # noqa: E402
    compare_leaks, degenerate_illustration, format_table, signal_detection,
)

CHANCE = 0.20
SEED = 2026

# The 6 Claude rows collected as Sonnet 4.6 rather than Opus 4.7.
# For these, a "self" judgment by the Opus-4.7 judge is cross-model.
SONNET_PROMPTS = ["E1", "E2", "M1", "M2", "H1", "H2"]

OUT_JSON = RESULTS_DIR / "engine_b_results.json"


# --------------------------------------------------------------------- utils
def exact_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson. Exact, and correct at k=0 and k=n where Wald is not."""
    if n == 0:
        return (float("nan"), float("nan"))
    return tuple(stats.binomtest(k, n).proportion_ci(confidence_level=conf,
                                                     method="exact"))


def binom_p(k: int, n: int) -> float:
    """Two-sided exact binomial test against the chance floor."""
    if n == 0:
        return float("nan")
    return float(stats.binomtest(k, n, CHANCE, alternative="two-sided").pvalue)


def binom_p_one_sided(k: int, n: int) -> float:
    """One-sided exact binomial test, in whichever direction the rate points.

    Reported because a below-floor rate is a directional claim and the
    two-sided p is the conservative reading of it. Grok is the case that
    matters: 2/38 gives 0.011 one-sided against 0.023 two-sided, and the paper
    quotes both so the reader can see the test was not chosen after the fact.
    """
    if n == 0:
        return float("nan")
    side = "less" if k / n < CHANCE else "greater"
    return float(stats.binomtest(k, n, CHANCE, alternative=side).pvalue)


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def load(condition: str) -> pd.DataFrame:
    path = JUDGMENTS_CSV if condition == "lineup" else JUDGMENTS_SINGLE_CSV
    df = pd.read_csv(path)
    df = df[~df["prompt_id"].isin(EXCLUDE_PROMPTS)].copy()
    if "needs_review" in df:
        df = df[df["needs_review"] != 1].copy()
    df = df[df["guessed_model"].notna() & (df["guessed_model"] != "")].copy()
    return df


# ------------------------------------------------------------ 1. self-recog
def self_recognition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for judge in MODELS:
        own = df[(df["judge"] == judge) & (df["true_author"] == judge)]
        n, k = len(own), int((own["guessed_model"] == judge).sum())
        lo, hi = exact_ci(k, n)
        p = binom_p(k, n)
        rows.append({
            "judge": judge, "k": k, "n": n,
            "rate": k / n if n else np.nan,
            "ci_lo": lo, "ci_hi": hi, "p_vs_chance": p, "sig": stars(p),
            "p_one_sided": binom_p_one_sided(k, n),
            "direction": ("above" if n and k / n > CHANCE else
                          "below" if n and k / n < CHANCE else "at"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------- 2. self-advantage
def self_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """Is a model better at spotting ITS OWN text than others are at spotting it?

    2x2 Fisher: rows = {judged by self, judged by others}, cols = {named that
    model, did not}. Without this, a high diagonal is ambiguous between
    self-recognition and the model simply writing identifiable prose.
    """
    rows = []
    for model in MODELS:
        on_model = df[df["true_author"] == model]
        s = on_model[on_model["judge"] == model]
        o = on_model[on_model["judge"] != model]
        s_hit, s_n = int((s["guessed_model"] == model).sum()), len(s)
        o_hit, o_n = int((o["guessed_model"] == model).sum()), len(o)
        table = [[s_hit, s_n - s_hit], [o_hit, o_n - o_hit]]
        odds, p = stats.fisher_exact(table) if s_n and o_n else (np.nan, np.nan)
        rows.append({
            "model": model,
            "self_rate": s_hit / s_n if s_n else np.nan, "self_n": s_n,
            "others_rate": o_hit / o_n if o_n else np.nan, "others_n": o_n,
            "advantage_pp": (s_hit / s_n - o_hit / o_n) * 100
                            if s_n and o_n else np.nan,
            "odds_ratio": odds, "fisher_p": p, "sig": stars(p),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. LOJO
def leave_one_judge_out(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute every headline number that can move, dropping each judge.

    Which numbers can move is worth stating:

      CANNOT move: a judge's own self-recognition diagonal. Dropping judge X
      removes rows where judge == X and cannot touch rows where judge == Y,
      so judge Y's diagonal is invariant by construction.

      CAN move, and are recomputed below:
        - overall accuracy
        - the blame distribution, and top-2 concentration
        - RECALL per authored model -- dropping a judge removes one of the
          five raters of every response, so how often a model's text is
          identified by anyone genuinely changes
        - SELF-ADVANTAGE per model -- the "others" rate is computed over the
          other four judges, so dropping one of them moves it

    With five judges this is five points. It is a sensitivity check on
    robustness to any single judge, NOT evidence that the effect generalises
    to judges outside this set.
    """
    rows = []
    for dropped in MODELS:
        sub = df[df["judge"] != dropped]
        shares = sub["guessed_model"].value_counts(normalize=True)
        row = {
            "dropped_judge": dropped,
            "n": len(sub),
            "accuracy": float((sub["correct"] == 1).mean()),
            "max_blame_share": float(shares.max()),
            "max_blame_model": shares.idxmax(),
            "top2_concentration": float(shares.nlargest(2).sum()),
        }
        # Recall and self-advantage, per authored model, without this judge.
        for m in MODELS:
            on_m = sub[sub["true_author"] == m]
            row[f"recall_{m}"] = (float((on_m["guessed_model"] == m).mean())
                                  if len(on_m) else np.nan)
            s = on_m[on_m["judge"] == m]
            o = on_m[on_m["judge"] != m]
            row[f"selfadv_{m}"] = (
                float((s["guessed_model"] == m).mean()
                      - (o["guessed_model"] == m).mean()) * 100
                if len(s) and len(o) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def lojo_stability(lojo: pd.DataFrame) -> dict:
    """How far does any single judge move each headline number?

    Reports the full range across the five leave-one-out fits. A small range
    means no single judge is carrying the result.
    """
    out = {}
    for col in lojo.columns:
        if col in ("dropped_judge", "n", "max_blame_model"):
            continue
        v = pd.to_numeric(lojo[col], errors="coerce").dropna()
        if v.empty:
            continue
        out[col] = {
            "min": float(v.min()), "max": float(v.max()),
            "range": float(v.max() - v.min()),
            "argmin_judge": str(lojo.loc[v.idxmin(), "dropped_judge"]),
            "argmax_judge": str(lojo.loc[v.idxmax(), "dropped_judge"]),
        }
    return out


# ------------------------------------------------- 3b. recall, per authored model
def recall_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """How often is each model's text correctly identified BY ANYONE?

    The self-recognition diagonal answers "can a model spot itself". This
    answers the complementary question: is that model's writing identifiable
    at all? A model can be unidentifiable to everyone (low recall) or
    identifiable to everyone but itself. Reporting only the diagonal conflates
    the two, and the interpretation of Grok's below-chance self-rate depends on
    which it is.
    """
    rows = []
    for model in MODELS:
        on_model = df[df["true_author"] == model]
        n = len(on_model)
        k = int((on_model["guessed_model"] == model).sum())
        lo, hi = exact_ci(k, n)
        p = binom_p(k, n)
        rows.append({
            "model": model, "k": k, "n": n,
            "recall": k / n if n else np.nan,
            "ci_lo": lo, "ci_hi": hi, "p_vs_chance": p, "sig": stars(p),
        })
    return pd.DataFrame(rows)


# ------------------------------------------- 3c. confidence calibration
def confidence_calibration(df: pd.DataFrame) -> dict:
    """Is a judge's stated confidence informative about whether it is right?

    Judges emit a 1-5 confidence with every judgment. If confidence carried
    information, accuracy would rise with it. This tests that directly rather
    than reporting only the two group means, because two means can differ while
    the relationship is non-monotonic and therefore useless for calibration.
    """
    if "confidence" not in df.columns:
        return {"error": "no confidence column"}
    d = df[["confidence", "correct", "judge"]].dropna().copy()
    d["confidence"] = pd.to_numeric(d["confidence"], errors="coerce")
    d = d.dropna(subset=["confidence"])
    if d.empty:
        return {"error": "confidence column is empty"}

    right = d[d["correct"] == 1]["confidence"]
    wrong = d[d["correct"] != 1]["confidence"]
    gap = float(right.mean() - wrong.mean()) if len(right) and len(wrong) else np.nan
    u_p = (float(stats.mannwhitneyu(right, wrong, alternative="two-sided").pvalue)
           if len(right) and len(wrong) else np.nan)

    # Accuracy at each stated confidence level -- the actual calibration curve.
    by_level = {}
    for lvl, grp in d.groupby("confidence"):
        k, n = int((grp["correct"] == 1).sum()), len(grp)
        lo, hi = exact_ci(k, n)
        by_level[str(int(lvl))] = {
            "n": n, "accuracy": k / n, "ci_lo": lo, "ci_hi": hi,
        }

    # Does accuracy actually increase with confidence?
    rho, rho_p = stats.spearmanr(d["confidence"], (d["correct"] == 1).astype(int))

    per_judge = {}
    for j, grp in d.groupby("judge"):
        r = grp[grp["correct"] == 1]["confidence"]
        w = grp[grp["correct"] != 1]["confidence"]
        per_judge[str(j)] = {
            "mean_conf": float(grp["confidence"].mean()),
            "mean_when_correct": float(r.mean()) if len(r) else np.nan,
            "mean_when_wrong": float(w.mean()) if len(w) else np.nan,
            "gap": float(r.mean() - w.mean()) if len(r) and len(w) else np.nan,
        }

    return {
        "n": int(len(d)),
        "mean_overall": float(d["confidence"].mean()),
        "mean_when_correct": float(right.mean()) if len(right) else np.nan,
        "mean_when_wrong": float(wrong.mean()) if len(wrong) else np.nan,
        "gap": gap,
        "mannwhitney_p": u_p,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "informative": bool(np.isfinite(rho_p) and rho_p < 0.05 and rho > 0),
        "accuracy_by_level": by_level,
        "per_judge": per_judge,
    }


# ------------------------------------------------- 3d. slot / position bias
def slot_bias(df: pd.DataFrame) -> dict:
    """Do judges prefer a position, independent of what sits there?

    The Latin square makes slot orthogonal to model identity, so any deviation
    from uniform here is a judge-side position preference and not a confound
    with the models. Only meaningful in the lineup condition, where slots exist.
    """
    if "slot" not in df.columns or df["slot"].isna().all():
        return {"error": "no slot column (expected for single-text)"}
    d = df[df["slot"].notna() & (df["slot"] != "")]
    if d.empty:
        return {"error": "slot column is empty"}
    counts = d["slot"].value_counts().sort_index()
    chi2, p = stats.chisquare(counts.values)[:2]
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "shares": {str(k): float(v / counts.sum()) for k, v in counts.items()},
        "chi2": float(chi2), "p": float(p), "df": int(len(counts) - 1),
        "uniform": bool(p >= 0.05),
    }


def distinct_name_counts(df: pd.DataFrame) -> dict:
    """How many distinct authors each lineup session named.

    Repeats are permitted by design -- forcing five distinct names would make
    the blame distribution uniform by construction. This reports how often the
    judges used that freedom. They mostly did not, which is why the flat lineup
    blame distribution says little about their attribution priors.
    """
    per_session = df.groupby("session_id")["guessed_model"].nunique()
    counts = per_session.value_counts().sort_index(ascending=False)
    n = len(per_session)
    return {
        "n_sessions": int(n),
        "counts": {int(k): int(v) for k, v in counts.items()},
        "all_distinct_share": float(counts.get(len(MODELS), 0) / n) if n else 0.0,
    }


# ------------------------------------------- 4. blame, judge-averaged vs pooled
def blame_tests(df: pd.DataFrame) -> dict:
    pooled = df["guessed_model"].value_counts().reindex(MODELS).fillna(0)
    chi2_pooled, p_pooled = stats.chisquare(pooled.values)[:2]

    # Unit of analysis = judge (n = 5), which is what a claim about judges needs.
    per_judge = (df.groupby("judge")["guessed_model"]
                   .value_counts(normalize=True).unstack().reindex(columns=MODELS)
                   .fillna(0))
    max_share = per_judge.max(axis=1)
    t, p_t = stats.ttest_1samp(max_share.values, CHANCE)

    # Chi-square on judge-averaged proportions, scaled to one judge's worth of
    # observations so the statistic is not inflated by pooling.
    avg = per_judge.mean(axis=0)
    n_per_judge = len(df) / df["judge"].nunique()
    chi2_avg, p_avg = stats.chisquare((avg * n_per_judge).values)[:2]

    return {
        "pooled_counts": pooled.astype(int).to_dict(),
        "pooled_shares": (pooled / pooled.sum()).to_dict(),
        "chi2_pooled": float(chi2_pooled), "p_pooled": float(p_pooled),
        "judge_averaged_shares": avg.to_dict(),
        "chi2_judge_averaged": float(chi2_avg), "p_judge_averaged": float(p_avg),
        "max_share_per_judge": max_share.to_dict(),
        "t_max_share_vs_chance": float(t), "p_max_share": float(p_t),
        "concentration_top2": float((pooled / pooled.sum())
                                    .nlargest(2).sum()),
    }


# --------------------------------------------------- 5. mixed-effects logistic
def mixed_effects(df: pd.DataFrame) -> dict:
    """P(correct) with crossed random intercepts for prompt_id and judge."""
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        return {"error": "statsmodels not available"}

    d = df[["correct", "prompt_id", "judge"]].dropna().copy()
    d["correct"] = d["correct"].astype(int)
    vc = {"prompt": "0 + C(prompt_id)", "judge": "0 + C(judge)"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = BinomialBayesMixedGLM.from_formula(
                "correct ~ 1", vc_formulas=vc, data=d)
            res = model.fit_vb(verbose=False)
        intercept = float(res.fe_mean[0])
        return {
            "intercept_logit": intercept,
            "intercept_prob": float(1 / (1 + np.exp(-intercept))),
            "sd_prompt": float(np.exp(res.vcp_mean[0])),
            "sd_judge": float(np.exp(res.vcp_mean[1])),
            "n": int(len(d)),
            "note": ("variational Bayes fit (fit_vb); crossed random "
                     "intercepts for prompt_id and judge"),
        }
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the run
        return {"error": f"{type(exc).__name__}: {exc}"}


# -------------------------------------------------------- 6. mechanism test
def mechanism_test(df: pd.DataFrame, n_perm: int = 200) -> dict:
    """Do the 18 surface features predict WHO GETS BLAMED, on wrong guesses?

    This is the paper's second (null) contribution. Bai et al. conjecture that
    blame follows "familiar stylistic patterns" and never test it. If neither a
    random forest nor logistic regression beats the majority baseline, the
    conjecture is unsupported on this corpus.
    """
    resp = pd.read_csv(RESPONSES_CSV)
    resp = resp[~resp["prompt_id"].isin(EXCLUDE_PROMPTS)]
    key = {(r.prompt_id, r.model): r.text for r in resp.itertuples()}

    wrong = df[df["correct"] != 1].copy()
    wrong["text"] = [key.get((p, a)) for p, a in
                     zip(wrong["prompt_id"], wrong["true_author"])]
    wrong = wrong[wrong["text"].notna()].copy()
    if len(wrong) < 40:
        return {"error": f"only {len(wrong)} wrong guesses; too few to fit"}

    X = frame(wrong["text"].tolist(), guard=False)[FEATURES].to_numpy()
    y = wrong["guessed_model"].to_numpy()
    groups = wrong["prompt_id"].to_numpy()

    majority = float(pd.Series(y).value_counts(normalize=True).iloc[0])
    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)

    # THE CONTROL THAT DECIDES THIS TEST.
    #
    # Beating the majority baseline, or beating shuffled labels, does not show
    # that *style* drives blame. The 18 features identify the author at 86%
    # (Engine A), and blame is conditioned on authorship, so a feature
    # classifier can score above chance purely as a noisy proxy for identity
    # without carrying any information about blame.
    #
    # So we ask the question that separates the two: does knowing the true
    # author ALONE -- no features at all -- predict blame as well? Modal
    # wrong-guess per author, learned on train folds only. If this baseline
    # matches or beats the feature models, the features add nothing beyond
    # identity and the stylometric null holds.
    ta = wrong["true_author"].to_numpy()
    ident_pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(y, y, groups=groups):
        modal = {a: pd.Series(y[tr][ta[tr] == a]).mode() for a in np.unique(ta[tr])}
        fallback = pd.Series(y[tr]).mode()[0]
        for i in te:
            m = modal.get(ta[i])
            ident_pred[i] = m[0] if m is not None and len(m) else fallback
    identity_acc = float((ident_pred == y).mean())

    out = {"n_wrong": int(len(wrong)), "n_classes": int(len(np.unique(y))),
           "majority_baseline": majority, "identity_only_baseline": identity_acc,
           "cv_folds": n_splits, "n_permutations": n_perm}
    for name, clf in (
        ("random_forest", RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1)),
        ("logistic_regression", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=SEED))),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = cross_val_predict(clf, X, y, cv=cv, groups=groups)
            acc = float((pred == y).mean())
            # Beating the majority baseline is NOT evidence on its own: the
            # baseline ignores the group structure, and a classifier that only
            # learns "which model wrote this" can score above it because blame
            # is not independent of authorship even among wrong guesses. The
            # permutation test asks the question that actually matters -- is
            # this accuracy reachable by shuffling the labels?
            _, perm_scores, p_perm = permutation_test_score(
                clf, X, y, groups=groups, cv=cv,
                n_permutations=n_perm, random_state=SEED, n_jobs=-1)
        out[name] = {
            "accuracy": acc,
            "vs_majority_pp": (acc - majority) * 100,
            "beats_majority": bool(acc > majority),
            "perm_null_mean": float(np.mean(perm_scores)),
            "perm_null_sd": float(np.std(perm_scores)),
            "perm_p": float(p_perm),
            "significant": bool(p_perm < 0.05),
        }
    # The null we claim is "features do not predict the blame target". It
    # survives only if NEITHER model is significantly better than shuffled
    # labels. Majority-baseline comparison alone is too weak to settle it.
    # The null holds unless a feature model beats BOTH shuffled labels and
    # the identity-only baseline. Beating shuffled labels alone only shows the
    # features encode authorship, which Engine A already established.
    beats_identity = any(out[k]["accuracy"] > identity_acc
                         for k in ("random_forest", "logistic_regression"))
    sig = any(out[k]["significant"]
              for k in ("random_forest", "logistic_regression"))
    out["beats_identity_baseline"] = bool(beats_identity)
    out["null_holds"] = bool(not (sig and beats_identity))
    return out


# ------------------------------------------------------------ 7. sensitivity
def sonnet_sensitivity(df: pd.DataFrame) -> dict:
    """Self-recognition with the 6 Sonnet-4.6 Claude rows dropped."""
    keep = ~((df["true_author"] == "Claude") & df["prompt_id"].isin(SONNET_PROMPTS))
    sub = df[keep]
    full = self_recognition(df).set_index("judge")
    drop = self_recognition(sub).set_index("judge")
    return {
        j: {
            "full": {"k": int(full.loc[j, "k"]), "n": int(full.loc[j, "n"]),
                     "rate": float(full.loc[j, "rate"])},
            "sonnet_dropped": {"k": int(drop.loc[j, "k"]),
                               "n": int(drop.loc[j, "n"]),
                               "rate": float(drop.loc[j, "rate"])},
            "shift_pp": float((drop.loc[j, "rate"] - full.loc[j, "rate"]) * 100),
        } for j in MODELS
    }


# ------------------------------------------------------------------ reporting
def report(condition: str, df: pd.DataFrame, n_perm: int = 200) -> dict:
    bar = "=" * 74
    print(f"\n{bar}\nENGINE B - {condition.upper()}\n{bar}")
    print(f"judgments {len(df)} | prompts {df['prompt_id'].nunique()} | "
          f"judges {df['judge'].nunique()} | accuracy "
          f"{(df['correct'] == 1).mean():.1%} (chance {CHANCE:.0%})")

    sr = self_recognition(df)
    print(f"\n-- SELF-RECOGNITION (exact 95% CI vs {CHANCE:.0%} floor) --")
    print(f"  {'judge':10s}{'k/n':>9}{'rate':>8}{'95% CI':>18}{'p':>11}  sig")
    for r in sr.itertuples():
        print(f"  {r.judge:10s}{f'{r.k}/{r.n}':>9}{r.rate:8.1%}"
              f"{f'[{r.ci_lo:.1%}, {r.ci_hi:.1%}]':>18}{r.p_vs_chance:11.2e}"
              f"  {r.sig} {r.direction}")

    sa = self_advantage(df)
    print("\n-- SELF-ADVANTAGE (self vs others on the same model's text) --")
    print(f"  {'model':10s}{'self':>9}{'others':>9}{'delta pp':>10}"
          f"{'Fisher p':>11}  sig")
    for r in sa.itertuples():
        print(f"  {r.model:10s}{r.self_rate:9.1%}{r.others_rate:9.1%}"
              f"{r.advantage_pp:10.1f}{r.fisher_p:11.2e}  {r.sig}")

    bl = blame_tests(df)
    print("\n-- BLAME (judge-averaged is the honest test; n = 5 judges) --")
    for m in MODELS:
        print(f"  {m:10s}{bl['pooled_shares'][m]:8.1%} pooled   "
              f"{bl['judge_averaged_shares'][m]:8.1%} judge-averaged")
    print(f"  top-2 concentration        : {bl['concentration_top2']:.1%}")
    print(f"  chi2 pooled (inflated)     : {bl['chi2_pooled']:.1f}, "
          f"p = {bl['p_pooled']:.3g}")
    print(f"  chi2 judge-averaged        : {bl['chi2_judge_averaged']:.1f}, "
          f"p = {bl['p_judge_averaged']:.3g}")
    print(f"  max-share vs chance (n=5)  : t = {bl['t_max_share_vs_chance']:.2f}, "
          f"p = {bl['p_max_share']:.4g}")

    lojo = leave_one_judge_out(df)
    stab = lojo_stability(lojo)
    print("\n-- LEAVE-ONE-JUDGE-OUT (sensitivity, n=5 - not proof) --")
    print(f"  {'dropped':10s}{'n':>6}{'accuracy':>10}{'top-2 blame':>13}"
          f"{'max blame':>11}  on")
    for r in lojo.itertuples():
        print(f"  {r.dropped_judge:10s}{r.n:6d}{r.accuracy:10.1%}"
              f"{r.top2_concentration:13.1%}{r.max_blame_share:11.1%}"
              f"  {r.max_blame_model}")
    print("\n  range across the five fits (small = no single judge carries it):")
    for key in ("accuracy", "top2_concentration"):
        if key in stab:
            v = stab[key]
            print(f"    {key:20s}{v['min']:8.1%} .. {v['max']:8.1%}"
                  f"   range {v['range']:.1%}")
    print(f"    {'recall, per model':20s}"
          + "  ".join(f"{m} {stab['recall_' + m]['range']:.1%}"
                      for m in MODELS if f"recall_{m}" in stab))
    print(f"    {'self-adv, per model':20s}"
          + "  ".join(f"{m} {stab['selfadv_' + m]['range']:.1f}pp"
                      for m in MODELS if f"selfadv_{m}" in stab))
    print("    (a judge's OWN diagonal cannot move when another judge is "
          "dropped -- omitted deliberately)")

    rc = recall_by_model(df)
    print("\n-- RECALL: is each model's text identifiable BY ANYONE? --")
    print(f"  {'model':10s}{'k/n':>10}{'recall':>9}{'95% CI':>18}{'p':>11}  sig")
    for r in rc.itertuples():
        print(f"  {r.model:10s}{f'{r.k}/{r.n}':>10}{r.recall:9.1%}"
              f"{f'[{r.ci_lo:.1%}, {r.ci_hi:.1%}]':>18}{r.p_vs_chance:11.2e}"
              f"  {r.sig}")

    cc = confidence_calibration(df)
    print("\n-- CONFIDENCE CALIBRATION --")
    if "error" in cc:
        print(f"  skipped: {cc['error']}")
    else:
        print(f"  mean {cc['mean_overall']:.2f} overall | "
              f"{cc['mean_when_correct']:.2f} when correct vs "
              f"{cc['mean_when_wrong']:.2f} when wrong "
              f"(gap {cc['gap']:+.2f}, Mann-Whitney p = {cc['mannwhitney_p']:.3g})")
        print(f"  Spearman rho = {cc['spearman_rho']:+.3f}, "
              f"p = {cc['spearman_p']:.3g}  -> confidence is "
              f"{'INFORMATIVE' if cc['informative'] else 'UNINFORMATIVE'}")
        print(f"  {'stated':>8}{'n':>7}{'accuracy':>11}{'95% CI':>18}")
        for lvl in sorted(cc["accuracy_by_level"], key=float):
            v = cc["accuracy_by_level"][lvl]
            ci = f"[{v['ci_lo']:.1%}, {v['ci_hi']:.1%}]"
            print(f"  {lvl:>8}{v['n']:7d}{v['accuracy']:11.1%}{ci:>18}")

    sb = slot_bias(df)
    print("\n-- POSITION BIAS (slot) --")
    if "error" in sb:
        print(f"  n/a: {sb['error']}")
    else:
        print("  " + "  ".join(f"{k} {v:.1%}" for k, v in sb["shares"].items()))
        print(f"  chi2 = {sb['chi2']:.2f}, df = {sb['df']}, p = {sb['p']:.3g}"
              f"  -> {'uniform' if sb['uniform'] else 'NON-UNIFORM, investigate'}")

    me = mixed_effects(df)
    print("\n-- MIXED-EFFECTS LOGISTIC on P(correct) --")
    if "error" in me:
        print(f"  FAILED: {me['error']}")
    else:
        print(f"  intercept {me['intercept_logit']:+.3f} logit "
              f"({me['intercept_prob']:.1%})   "
              f"sd(prompt) {me['sd_prompt']:.3f}   sd(judge) {me['sd_judge']:.3f}")

    mech = mechanism_test(df, n_perm=n_perm)
    print("\n-- MECHANISM TEST: do features predict the BLAME TARGET? --")
    if "error" in mech:
        print(f"  SKIPPED: {mech['error']}")
    else:
        print(f"  wrong guesses {mech['n_wrong']}, {mech['n_classes']} classes")
        print(f"  {'majority baseline':30s}{mech['majority_baseline']:8.1%}")
        print(f"  {'TRUE-AUTHOR-ONLY baseline':30s}"
              f"{mech['identity_only_baseline']:8.1%}   <- the control that matters")
        for k in ("random_forest", "logistic_regression"):
            m = mech[k]
            print(f"  {k:30s}{m['accuracy']:8.1%}   "
                  f"perm null {m['perm_null_mean']:.1%}+-{m['perm_null_sd']:.1%}, "
                  f"p={m['perm_p']:.4f} {'sig' if m['significant'] else 'n.s.'}")
        if mech["null_holds"]:
            print("  NULL HOLDS: features beat shuffled labels only by encoding")
            print("  authorship -- they do NOT beat identity alone, so they add no")
            print("  information about the blame target beyond who wrote the text.")
        else:
            print("  NULL DOES NOT HOLD: a feature model beats BOTH shuffled labels")
            print("  and the identity-only baseline. Style carries blame signal.")

    sens = sonnet_sensitivity(df)
    print("\n-- SENSITIVITY: 6 Sonnet-4.6 Claude rows dropped --")
    for j in MODELS:
        s = sens[j]
        print(f"  {j:10s}{s['full']['k']}/{s['full']['n']} "
              f"({s['full']['rate']:.1%})  ->  "
              f"{s['sonnet_dropped']['k']}/{s['sonnet_dropped']['n']} "
              f"({s['sonnet_dropped']['rate']:.1%})   "
              f"{s['shift_pp']:+.1f} pp")

    sd = signal_detection(df, MODELS)
    print("\n-- SIGNAL DETECTION (hit rate, false alarm, leak) --")
    print(format_table(sd))
    if any(r["degenerate"] for r in sd):
        d = degenerate_illustration()
        print(f"  degenerate cells are reported as '--': at h = f = 0 the "
              f"Haldane correction would return {d['leak']:+.2f} "
              f"(OR {d['odds_ratio']:.2f}, p = {d['p']:.2f}), which is an "
              f"artefact of the correction and not evidence.")

    xa = cross_attribution(df)
    print("\n-- ATTRIBUTION OF OTHERS' TEXT --")
    print(f"  pooled non-self accuracy {xa['pooled_non_self_accuracy']:.1%} "
          f"(n = {xa['n_pooled_non_self']})")
    for j in MODELS:
        pj = xa["per_judge"][j]
        print(f"  {j:10s}non-self {pj['non_self_accuracy']:6.1%}   "
              f"overall {pj['overall_accuracy']:6.1%}   "
              f"sends {pj['top_target_share']:.1%} of its guesses to "
              f"{pj['top_target']}")
    print("  largest flows (share of the author's own row):")
    for f in xa["top_flows"][:4]:
        print(f"    {f['true_author']:>9} -> {f['guessed_model']:<9}"
              f"{f['n']:>5}  {f['share_of_author_row']:6.1%} of row  "
              f"{f['share_of_all_errors']:6.1%} of errors")

    dn = distinct_name_counts(df) if condition == "lineup" else None
    if dn:
        print("\n-- DISTINCT NAMES PER SESSION --")
        print("  " + "  ".join(f"{k} -> {v}" for k, v in dn["counts"].items())
              + f"   ({dn['all_distinct_share']:.1%} used all "
                f"{len(MODELS)} names)")

    return {
        "condition": condition,
        "n_judgments": int(len(df)),
        "n_prompts": int(df["prompt_id"].nunique()),
        "accuracy": float((df["correct"] == 1).mean()),
        "self_recognition": sr.to_dict("records"),
        "self_advantage": sa.to_dict("records"),
        "recall": rc.to_dict("records"),
        "confidence_calibration": cc,
        "slot_bias": sb,
        "blame": bl,
        "leave_one_judge_out": lojo.to_dict("records"),
        "lojo_stability": stab,
        "mixed_effects": me,
        "mechanism_test": mech,
        "sonnet_sensitivity": sens,
        "signal_detection": sd,
        "cross_attribution": xa,
        "distinct_names": dn,
    }


# ----------------------------------------------------------------- LaTeX out
def _engine_a() -> dict:
    """Engine A's saved results. It is a separate script -- never re-run here."""
    path = RESULTS_DIR / "engine_a_results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _table(caption: str, label: str, spec: str, header: str,
           body: list[str]) -> list[str]:
    return ["% GENERATED by src/engine_b.py -- do not edit by hand.",
            "\\begin{table}[t]", "\\centering",
            f"\\caption{{{caption}}}", f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header,
            "\\midrule", *body,
            "\\bottomrule", "\\end{tabular}", "\\end{table}"]


def emit_tex(res: dict, path: Path) -> None:
    """Write the paper's macros and table fragments. Never hand-edit the output.

    Every number the prose quotes comes through here, so a claim in the paper
    cannot drift from the run that produced it. If you find yourself typing a
    digit into the paper, add a macro instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lin, sng, ea = res.get("lineup"), res.get("single"), _engine_a()

    def pct(x: float) -> str:
        return f"{x * 100:.1f}\\%"

    def mac(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}"

    out = ["% GENERATED by src/engine_b.py -- do not edit by hand.",
           "% Regenerate: python src/engine_b.py --tex paper/numbers.tex", "",
           mac("Chance", pct(CHANCE)),
           mac("NFeatures", str(len(FEATURES)))]

    # ---- Engine A: the stylometric control.
    if ea:
        acc = ea.get("accuracy", {})
        if "mean" in acc:
            n_a = int(acc.get("n", 190))
            lo, hi = wilson(int(round(acc["mean"] * n_a)), n_a)
            out += [mac("EngineAAcc", pct(acc["mean"])),
                    mac("EngineACI", f"[{pct(lo)}, {pct(hi)}]")]
        if perm := ea.get("permutation"):
            # A bound, not a point estimate: the p-value sits on the 1/(n+1)
            # resolution floor of the test.
            out.append(mac("EngineAPerm",
                           f"$p < 0.001$ (0 of {perm['n']} permutations "
                           f"reached the observed accuracy)"))
        grid = ea.get("headline", {}).get("UNTRUNCATED", {})
        if full := grid.get("FULL"):
            out += [mac("EngineAFoldMean", f"{full['rf_mean'] * 100:.1f}"),
                    mac("EngineAFoldSD", f"{full['rf_sd'] * 100:.1f}")]
        if lf := grid.get("LENGTH-FREE"):
            out += [mac("EngineALengthFreeRF", pct(lf["rf_mean"])),
                    mac("EngineALengthFreeLR", pct(lf["logreg_mean"]))]
        if lk := ea.get("leakage"):
            out.append(mac("EngineAUngrouped", f"{lk['ungrouped'] * 100:.1f}"))
        if "length_share_of_signal" in ea:
            out.append(mac("LengthShareOfSignal",
                           f"{ea['length_share_of_signal'] * 100:.0f}\\%"))
        if "length_proxy_importance_share" in ea:
            out.append(mac("LengthImportanceShare",
                           pct(ea["length_proxy_importance_share"])))

    # ---- Per-condition. Same macro shape for both, suffixed by condition.
    for tag, r in (("Lineup", lin), ("Single", sng)):
        if not r:
            continue
        if tag == "Lineup":
            out += [mac("NPrompts", str(r["n_prompts"])),
                    mac("NResponses", str(r["n_prompts"] * len(MODELS)))]
        out.append(mac(f"{tag}Acc", pct(r["accuracy"])))
        out.append(mac(f"{tag}Concentration",
                       pct(r["blame"]["concentration_top2"])))
        out += [mac(f"{tag}BlameChi",
                    f"{r['blame']['chi2_judge_averaged']:.1f}"),
                mac(f"{tag}BlameP", f"{r['blame']['p_judge_averaged']:.3g}"),
                mac(f"{tag}ConfRho",
                    f"{r['confidence_calibration']['spearman_rho']:+.3f}"),
                mac(f"{tag}ConfP",
                    f"{r['confidence_calibration']['spearman_p']:.3g}")]

        mech = r["mechanism_test"]
        out += [mac(f"Mech{tag}RF", pct(mech["random_forest"]["accuracy"])),
                mac(f"Mech{tag}LR",
                    pct(mech["logistic_regression"]["accuracy"])),
                mac(f"Mech{tag}Identity", pct(mech["identity_only_baseline"]))]

        by_sr = {x["judge"]: x for x in r["self_recognition"]}
        by_sa = {x["model"]: x for x in r["self_advantage"]}
        by_sd = {x["judge"]: x for x in r["signal_detection"]}
        xa = r["cross_attribution"]
        for m in MODELS:
            out += [mac(f"{m}Self{tag}", pct(by_sr[m]["rate"])),
                    mac(f"{m}Peer{tag}", pct(by_sa[m]["others_rate"])),
                    mac(f"{m}Adv{tag}", f"{by_sa[m]['advantage_pp']:+.1f}"),
                    mac(f"{m}FA{tag}", pct(by_sd[m]["false_alarm_rate"])),
                    mac(f"{m}NonSelf{tag}",
                        pct(xa["per_judge"][m]["non_self_accuracy"])),
                    mac(f"{m}Overall{tag}",
                        pct(xa["per_judge"][m]["overall_accuracy"])),
                    mac(f"{m}Blame{tag}",
                        pct(r["blame"]["pooled_shares"][m]))]
        out.append(mac(f"PooledNonSelf{tag}",
                       pct(xa["pooled_non_self_accuracy"])))

    if lin:
        out += [mac("SlotChi", f"{lin['slot_bias']['chi2']:.2f}"),
                mac("AllDistinctShare",
                    pct(lin["distinct_names"]["all_distinct_share"]))]
        shift = lin["sonnet_sensitivity"]["Claude"]
        out += [mac("SonnetShift", f"{shift['shift_pp']:+.1f}"),
                mac("SonnetDropped", pct(shift["sonnet_dropped"]["rate"]))]

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\nwrote {path} ({len(out) - 3} macros)")

    _emit_tables(res, ea, path.parent)


def _emit_tables(res: dict, ea: dict, out_dir: Path) -> None:
    lin, sng = res.get("lineup"), res.get("single")
    written = []

    def write(name: str, lines: list[str]) -> None:
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(name)

    if lin and sng:
        lb = {r["judge"]: r for r in lin["self_recognition"]}
        sb = {r["judge"]: r for r in sng["self_recognition"]}
        write("table_selfrecog.tex", _table(
            "Self-recognition by query format, same responses. Exact "
            "(Clopper--Pearson) 95\\% CIs against a 20\\% floor.",
            "tab:selfrecog", "lrrrr",
            "Judge & Lineup & 95\\% CI & Single-text & 95\\% CI \\\\",
            [f"{j} & {lb[j]['rate'] * 100:.1f} & "
             f"[{lb[j]['ci_lo'] * 100:.1f}, {lb[j]['ci_hi'] * 100:.1f}] & "
             f"{sb[j]['rate'] * 100:.1f} & "
             f"[{sb[j]['ci_lo'] * 100:.1f}, {sb[j]['ci_hi'] * 100:.1f}] \\\\"
             for j in MODELS]))

        write("table_blame.tex", _table(
            "Blame distribution by query format. The same responses produce a "
            "flat distribution under the lineup and collapse onto two families "
            "under single-text querying.",
            "tab:blame", "lrr",
            "Guessed & Lineup (\\%) & Single-text (\\%) \\\\",
            [f"{j} & {lin['blame']['pooled_shares'][j] * 100:.1f} & "
             f"{sng['blame']['pooled_shares'][j] * 100:.1f} \\\\"
             for j in MODELS]))

        write("table_nonself.tex", _table(
            "Accuracy with each judge's own text removed (152 non-self "
            "judgments per judge per format; chance 20\\%).",
            "tab:nonself", "lrrrr",
            "& \\multicolumn{2}{c}{Lineup} & \\multicolumn{2}{c}{Single-text} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Judge & Non-self & Overall & Non-self & Overall \\\\",
            [f"{j} & "
             f"{lin['cross_attribution']['per_judge'][j]['non_self_accuracy'] * 100:.1f} & "
             f"{lin['cross_attribution']['per_judge'][j]['overall_accuracy'] * 100:.1f} & "
             f"{sng['cross_attribution']['per_judge'][j]['non_self_accuracy'] * 100:.1f} & "
             f"{sng['cross_attribution']['per_judge'][j]['overall_accuracy'] * 100:.1f} \\\\"
             for j in MODELS]))

    # Self-advantage against the peer baseline, one table per condition.
    for tag, r, lbl in (("lineup", lin, "tab:selfadv"),
                        ("single-text", sng, "tab:selfadv-single")):
        if not r:
            continue
        sa = {x["model"]: x for x in r["self_advantage"]}
        sd = {x["judge"]: x for x in r["signal_detection"]}
        order = sorted(MODELS, key=lambda m: -sa[m]["advantage_pp"])
        name = ("table_selfadv.tex" if tag == "lineup"
                else "table_selfadv_single.tex")
        if tag == "lineup":
            caption = ("Self-recognition in the five-way lineup; the chance "
                       "floor is \\Chance{}. For reference, the stylometric "
                       "classifier reaches \\EngineAAcc{} on these same "
                       "responses; only Claude's self rate matches it. "
                       "Fisher's exact test compares each judge's 38 "
                       "self-judgments against its peers' 152 judgments on "
                       "the same responses.")
        else:
            caption = ("Self-advantage in the single-text condition; lineup "
                       "values appear in Table~\\ref{tab:selfadv}. Fisher's "
                       "exact test compares 38 self-judgments against 152 "
                       "peer judgments on the same texts.")
        write(name, _table(caption, lbl, "lrrrrr",
            "Judge & Self rate & False alarm & Peers on this text & "
            "Self-adv. & Fisher $p$ \\\\",
            [f"{m} & {sa[m]['self_rate'] * 100:.1f}\\% "
             f"({int(round(sa[m]['self_rate'] * sa[m]['self_n']))}/{sa[m]['self_n']}) & "
             f"{sd[m]['false_alarm_rate'] * 100:.1f}\\% & "
             f"{sa[m]['others_rate'] * 100:.1f}\\% & "
             f"{sa[m]['advantage_pp']:+.1f} & "
             f"{sa[m]['fisher_p']:.2g} \\\\" for m in order]))

    # Signal-detection table: both conditions stacked, degenerate cells dashed.
    if lin and sng:
        body = []
        for tag, r in (("Lineup", lin), ("Single-text", sng)):
            body.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{tag}}}}} \\\\")
            for x in sorted(r["signal_detection"],
                            key=lambda v: -(v["leak"] if not v["degenerate"]
                                            else -99)):
                if x["degenerate"]:
                    body.append(
                        f"\\quad {x['judge']} & {x['hit_rate'] * 100:.1f}\\% & "
                        f"{x['false_alarm_rate'] * 100:.1f}\\% & -- & -- & -- & "
                        f"degenerate \\\\")
                else:
                    body.append(
                        f"\\quad {x['judge']} & {x['hit_rate'] * 100:.1f}\\% & "
                        f"{x['false_alarm_rate'] * 100:.1f}\\% & "
                        f"{x['leak']:+.2f} & {x['odds_ratio']:.1f} & "
                        f"[{x['ci_lo']:.1f}, {x['ci_hi']:.1f}] & "
                        f"{x['p']:.3g} \\\\")
        write("table_leak.tex", _table(
            "Signal-detection view of self-recognition. Hits are over the "
            "judge's own 38 responses, false alarms over the 152 it did not "
            "write. Leak is the Haldane-corrected log odds ratio with a Wald "
            "test against zero. Dashes mark degenerate cells, where one row of "
            "the $2\\times2$ is empty and the leak is undefined.",
            "tab:leak", "llrrrrr",
            "Judge & Hit & FA & Leak & OR & 95\\% CI & $p$ \\\\", body))

    # Feature table and ablation grid come from Engine A.
    if imp := ea.get("feature_importance"):
        initial = {"length": "L", "punctuation": "P",
                   "lexical": "X", "readability": "R"}
        ranked = sorted(imp.items(), key=lambda kv: -kv[1])
        half = (len(ranked) + 1) // 2
        rows = []
        for left, right in zip(ranked[:half], ranked[half:] + [None] * half):
            cells = []
            for item in (left, right):
                if item is None:
                    cells.append(" & & ")
                    continue
                f, v = item
                star = "*" if f in LENGTH_PROXIES else ""
                name = f.replace("_", "\\_")
                cells.append(f"\\texttt{{{name}}}{star} & "
                             f"{initial[FAMILIES[f]]} & {v:.4f}")
            rows.append(" & ".join(cells) + " \\\\")
        write("table_features.tex", _table(
            "The 18 surface features with out-of-fold random-forest "
            "importance. Fam: L = length, P = punctuation/formatting, "
            "X = lexical, R = readability. * marks the repository's "
            "length-proxy set.",
            "tab:features", "lcrlcr",
            "Feature & Fam & Imp. & Feature & Fam & Imp. \\\\", rows))

    if grid := ea.get("headline"):
        pretty = {"UNTRUNCATED": "Untruncated",
                  "TRUNCATED-250": "Truncated 250w",
                  "TRUNCATED-TO-GROUP-MIN": "Group-min truncated"}
        rows = []
        for corpus in ("UNTRUNCATED", "TRUNCATED-TO-GROUP-MIN"):
            for fs in ("FULL", "LENGTH-FREE", "LENGTH-ONLY"):
                c = grid.get(corpus, {}).get(fs)
                if not c:
                    continue
                rf = f"{c['rf_mean'] * 100:.1f} $\\pm$ {c['rf_sd'] * 100:.1f}"
                # The control cell: length alone, after length is equalised.
                if corpus == "TRUNCATED-TO-GROUP-MIN" and fs == "LENGTH-ONLY":
                    rf = f"\\textbf{{{rf}}}"
                rows.append(
                    f"{pretty[corpus]} & {fs} & {c['k']} & {rf} & "
                    f"{c['logreg_mean'] * 100:.1f} $\\pm$ "
                    f"{c['logreg_sd'] * 100:.1f} \\\\")
        write("table_ablation.tex", _table(
            "Full ablation grid. FULL = all 18 features; LENGTH-FREE = 9 "
            "features with length proxies removed; LENGTH-ONLY = the 2 pure "
            "length features. Cells are fold means $\\pm$ s.d.; chance is "
            "20.0\\% throughout. A fixed 250-word truncation gives "
            "intermediate values (released). The bolded cell is the control: "
            "under group-min truncation a length-only classifier falls to "
            "chance, so the \\EngineAAcc{} reads as signal rather than "
            "leakage.",
            "tab:ablation", "llrrr",
            "Corpus & Features & $k$ & Random forest & Logistic regression "
            "\\\\", rows))

    for name in written:
        print(f"wrote {out_dir / name}")


def _leak_comparison(res: dict) -> list[dict]:
    """Did the format change what a judge could tell apart, or only what it said?

    The hit rate and the leak come apart here, which is why both are reported.
    GPT's self rate jumps 37 points between formats and reads as a gain in
    ability; its false-alarm rate jumps 41, and the leak does not move.
    """
    lin = {r["judge"]: r for r in res["lineup"]["signal_detection"]}
    sng = {r["judge"]: r for r in res["single"]["signal_detection"]}
    out = [compare_leaks(lin[j], sng[j]) for j in MODELS]

    print("\n-- LEAK ACROSS FORMATS (lineup -> single-text) --")
    for r in out:
        if not r["comparable"]:
            print(f"  {r['judge']:10s}not comparable (a degenerate cell)")
            continue
        print(f"  {r['judge']:10s}leak {r['leak_from']:+.2f} -> {r['leak_to']:+.2f}"
              f"   z = {r['z']:+.2f}, p = {r['p']:.3f}"
              f"   hit {r['hit_rate_delta_pp']:+.1f} pp,"
              f" false alarm {r['fa_rate_delta_pp']:+.1f} pp")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", choices=("lineup", "single", "both"),
                    default="both")
    ap.add_argument("--permutations", type=int, default=200,
                    help="permutations for the mechanism test (default 200)")
    ap.add_argument("--tex", type=Path, default=None,
                    help="also emit LaTeX macros + tables at this path")
    ap.add_argument("--from-json", action="store_true",
                    help="re-emit LaTeX from the saved results without "
                         "re-running the analysis (needs --tex)")
    args = ap.parse_args()

    # Editing a caption or adding a macro should not cost a rerun of the
    # mechanism test, which permutes labels 200 times per condition.
    if args.from_json:
        if not args.tex:
            ap.error("--from-json only makes sense with --tex")
        if not OUT_JSON.exists():
            ap.error(f"{OUT_JSON} not found -- run without --from-json first")
        emit_tex(json.loads(OUT_JSON.read_text(encoding="utf-8")), args.tex)
        return

    conditions = (["lineup", "single"] if args.condition == "both"
                  else [args.condition])
    res: dict = {}
    for c in conditions:
        res[c] = report(c, load(c), n_perm=args.permutations)

    if "lineup" in res and "single" in res:
        res["leak_comparison"] = _leak_comparison(res)

    RESULTS_DIR.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2, default=float) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")
    if args.tex:
        emit_tex(res, args.tex)


if __name__ == "__main__":
    main()
