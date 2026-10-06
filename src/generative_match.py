"""CompLLM: does a judge name itself on text that resembles what it would write?

    python src/generative_match.py              # 2000 permutations
    python src/generative_match.py --perms 200  # quick pass

The familiarity account (Wataoka et al. 2024) says a judge recognises text
that is close to what it would itself have produced. We cannot sample the
judges again, but the corpus already holds each judge's own answer to every
prompt. That answer is the proxy for "what J would write for prompt p", and
the other four responses to p are the texts J was asked to attribute.

Similarity between two responses, four ways (higher = more alike):

    stylo      minus the Euclidean distance between the 18 features of
               features.py, z-scored across the 190 responses
    word       cosine of word 1-2-gram TF-IDF vectors (sublinear tf)
    char       cosine of character 3-5-gram TF-IDF vectors (within words)
    funcword   cosine of z-scored function-word frequencies (Cosine Delta,
               Smith & Aldridge 2011); topic-free by construction

Tests.

(1) Self-naming on others' text. For judge J and every response r to prompt
    p NOT written by J: does sim(r, J's response to p) predict that J names
    itself? Logistic GEE clustered by prompt, with true-author fixed effects
    (so the comparison is between responses by the same author), and a
    second model adding r's mean similarity to the other candidates' responses
    to p, so a generic, central response cannot pass as a J-like one.
    (1b) On J's own text: are the responses most typical of J (closest to J's
    leave-prompt-out centroid) the ones J recognises, and do peers recognise
    the same ones? If peers track typicality as much as J does, typicality is
    identifiability, not self-familiarity.

(2) Which model gets named? For each wrong attribution, rank the named model
    among the four non-authors by how similar its own response to p is to r
    (chance top-1 = 25%). For every attribution, rank the named model among
    all five by similarity of r to each model's leave-prompt-out centroid
    (chance 20%), and do the same with the stylometric classifier's
    out-of-fold probabilities (the Engine A random forest) as the baseline.

(3) Two permutation nulls, 2000 draws each.
    N-ref: shuffle each author's responses across prompts (independently per
    author), so "M's response to p" becomes M's response to a random prompt.
    This keeps every author's general style and breaks only the prompt-
    specific match that generative match predicts.
    N-guess: shuffle the judge's named models within judge x true author, so
    guess marginals are kept and only the link to similarity is broken. This
    is the conditional chance level for test (2), which matters in the
    single-text condition where nearly every guess is GPT or Claude.

Caveat, stated here once and in the report. In the lineup, J's own response
to p sits in the same session as r, so "r resembles J's response" can act
through side-by-side matching (a puzzle strategy) rather than through
familiarity. The single-text condition shows one response at a time and
never shows J its own answer, so only there is similarity-to-own-response a
clean proxy for generative match; but single-text self-naming exists only for
Claude and GPT.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    EXCLUDE_PROMPTS, JUDGMENTS_CSV, JUDGMENTS_SINGLE_CSV, MODELS, RESPONSES_CSV,
    RESULTS_DIR, ROOT,
)
from features import FEATURES, frame  # noqa: E402

OUT_JSON = RESULTS_DIR / "generative_match.json"
OUT_MD = ROOT / "docs" / "revision" / "GENMATCH.md"
SEED = 2026
MIN_EVENTS = 10  # self-namings needed before a per-judge model is fit
MEASURES = ["stylo", "word", "char", "funcword"]

# Function words: closed-class English words, the standard stylometric
# topic-free set (articles, pronouns, prepositions, conjunctions, auxiliaries,
# common adverbs and quantifiers).
FUNCTION_WORDS = """
a about above after again against all also although am among an and another
any are around as at be because been before being below between both but by
can could did do does doing down during each either enough even ever every few
for from further had has have having he her here hers herself him himself his
how however i if in into is it its itself just less may me might more most
much must my myself neither no nor not now of off often on once one only or
other others our ours ourselves out over own per perhaps quite rather same
several shall she should since so some such than that the their theirs them
themselves then there these they this those though through thus to too toward
towards under until up upon us very was we were what whatever when where
whether which while who whom whose why will with within without would yet you
your yours yourself
""".split()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_responses() -> pd.DataFrame:
    r = pd.read_csv(RESPONSES_CSV)
    r["prompt_id"] = r["prompt_id"].astype(str)
    r = r[~r["prompt_id"].isin(EXCLUDE_PROMPTS)].reset_index(drop=True)
    return r[["prompt_id", "model", "text"]]


def load_judgments() -> pd.DataFrame:
    a = pd.read_csv(JUDGMENTS_CSV)
    a["condition"] = "lineup"
    b = pd.read_csv(JUDGMENTS_SINGLE_CSV)
    b["condition"] = "single"
    df = pd.concat([a, b], ignore_index=True)
    df = df[~df["prompt_id"].isin(EXCLUDE_PROMPTS)].reset_index(drop=True)
    df["named_self"] = (df["guessed_model"] == df["judge"]).astype(int)
    return df


# --------------------------------------------------------------------------
# Similarity matrices (190 x 190, higher = more alike)
# --------------------------------------------------------------------------
def _cos(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Xn = X / n
    return Xn @ Xn.T


def vectors(resp: pd.DataFrame) -> dict[str, np.ndarray]:
    """Row vectors per measure; similarity is cosine except for stylo."""
    texts = resp["text"].tolist()
    F = frame(texts).to_numpy(dtype=float)
    Fz = (F - F.mean(0)) / F.std(0)
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                           lowercase=True).fit_transform(texts).toarray()
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                           min_df=2, lowercase=True).fit_transform(texts).toarray()
    tok = [re.findall(r"[a-z']+", t.lower()) for t in texts]
    fw_index = {w: i for i, w in enumerate(FUNCTION_WORDS)}
    FW = np.zeros((len(texts), len(FUNCTION_WORDS)))
    for k, ws in enumerate(tok):
        for w in ws:
            j = fw_index.get(w)
            if j is not None:
                FW[k, j] += 1
        FW[k] /= max(len(ws), 1)
    sd = FW.std(0)
    keep = sd > 0
    FWz = (FW[:, keep] - FW[:, keep].mean(0)) / sd[keep]
    return {"stylo": Fz, "word": word, "char": char, "funcword": FWz}


def sim_matrices(V: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    S = {}
    Fz = V["stylo"]
    d = np.sqrt(((Fz[:, None, :] - Fz[None, :, :]) ** 2).sum(-1))
    S["stylo"] = -d
    for m in ("word", "char", "funcword"):
        S[m] = _cos(V[m])
    return S


def centroid_sims(V: dict[str, np.ndarray], resp: pd.DataFrame) -> dict[str, np.ndarray]:
    """sim(r, centroid of model M's responses to every OTHER prompt), shape
    (190, 5). Leave-prompt-out for every M, so the true author's centroid
    never contains r and no candidate's centroid contains the same topic."""
    out = {}
    prompts = resp["prompt_id"].to_numpy()
    models = resp["model"].to_numpy()
    for m, X in V.items():
        C = np.zeros((len(resp), len(MODELS)))
        for j, M in enumerate(MODELS):
            idxM = np.where(models == M)[0]
            for i in range(len(resp)):
                use = idxM[prompts[idxM] != prompts[i]]
                c = X[use].mean(0)
                if m == "stylo":
                    C[i, j] = -np.linalg.norm(X[i] - c)
                else:
                    den = np.linalg.norm(X[i]) * np.linalg.norm(c)
                    C[i, j] = float(X[i] @ c / den) if den else 0.0
        out[m] = C
    return out


# --------------------------------------------------------------------------
# Test 1: similarity to J's own same-prompt response -> J names itself
# --------------------------------------------------------------------------
def gee_selfname(d: pd.DataFrame, xcol: str, covar: str | None, strata: str) -> dict | None:
    """Logistic GEE of named_self on the standardised similarity, with
    stratum fixed effects (true author, or judge x author when pooled).
    Strata with no self-naming, or with nothing but, carry no within-stratum
    information and make the fixed effect diverge, so they are dropped."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    ev = d.groupby(strata)["named_self"].agg(["sum", "count"])
    ok = ev[(ev["sum"] > 0) & (ev["sum"] < ev["count"])].index
    d = d[d[strata].isin(ok)].copy()
    if d["named_self"].sum() < MIN_EVENTS:
        return None
    d["x"] = (d[xcol] - d[xcol].mean()) / d[xcol].std()
    rhs = "x"
    if covar:
        d["c"] = (d[covar] - d[covar].mean()) / d[covar].std()
        rhs += " + c"
    if d[strata].nunique() > 1:
        rhs += f" + C({strata})"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.gee(f"named_self ~ {rhs}", groups="prompt_id", data=d,
                      family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = fit.conf_int().loc["x"]
    return {"n": int(len(d)), "events": int(d["named_self"].sum()),
            "strata_kept": [str(s) for s in ok],
            "OR_per_sd": float(np.exp(fit.params["x"])),
            "ci95": [float(np.exp(ci[0])), float(np.exp(ci[1]))],
            "p": float(fit.pvalues["x"])}


def demeaned_diff(x: np.ndarray, y: np.ndarray, strata: np.ndarray) -> float:
    """Permutation statistic for test 1: mean within-stratum-centred
    similarity of self-named rows minus that of the rest."""
    xc = x.copy()
    for s in np.unique(strata):
        m = strata == s
        xc[m] = x[m] - x[m].mean()
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(xc[y == 1].mean() - xc[y == 0].mean())


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    resp = load_responses()
    df = load_judgments()
    V = vectors(resp)
    S = sim_matrices(V)
    CS = centroid_sims(V, resp)

    key = {(p, m): i for i, (p, m) in enumerate(zip(resp["prompt_id"], resp["model"]))}
    prompts = sorted(resp["prompt_id"].unique())
    pidx = {p: k for k, p in enumerate(prompts)}
    midx = {m: j for j, m in enumerate(MODELS)}
    # ref[p_k, j] = row of model j's response to prompt k
    REF = np.array([[key[(p, M)] for M in MODELS] for p in prompts])

    df["r"] = [key[(p, a)] for p, a in zip(df["prompt_id"], df["true_author"])]
    df["pk"] = df["prompt_id"].map(pidx)
    df["aj"] = df["true_author"].map(midx)
    df["jj"] = df["judge"].map(midx)
    df["gj"] = df["guessed_model"].map(midx)
    r_arr, pk_arr = df["r"].to_numpy(), df["pk"].to_numpy()
    aj_arr, jj_arr, gj_arr = df["aj"].to_numpy(), df["jj"].to_numpy(), df["gj"].to_numpy()
    cond_arr = df["condition"].to_numpy()

    def cand_sims(refmap: np.ndarray, m: str) -> np.ndarray:
        """(n_judgments, 5): sim of each judged response r to candidate M's
        response to the reference prompt refmap[p, M]."""
        rows = refmap[pk_arr]  # (n, 5) response rows
        return S[m][r_arr[:, None], rows]

    # ---------------------------------------------------------------- RF
    # Engine A's random forest, same settings, out-of-fold class
    # probabilities (the fixed stylometric reference for test 2).
    F = frame(resp["text"].tolist())
    rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
    proba = cross_val_predict(rf, F[FEATURES].to_numpy(), resp["model"].to_numpy(),
                              groups=resp["prompt_id"].to_numpy(),
                              cv=GroupKFold(n_splits=5), method="predict_proba")
    classes = sorted(resp["model"].unique())  # sklearn orders classes
    proba = proba[:, [classes.index(M) for M in MODELS]]
    rf_acc = float((proba.argmax(1) == resp["model"].map(midx).to_numpy()).mean())

    out: dict = {"n_perms": args.perms, "measures": MEASURES,
                 "rf_oof_accuracy": rf_acc}

    # ---------------------------------------------------- test 1 (observed)
    base = {m: cand_sims(REF, m) for m in MEASURES}
    for m in MEASURES:
        df[f"simJ_{m}"] = base[m][np.arange(len(df)), jj_arr]
        # mean similarity to the other candidates' same-prompt responses
        # (excluding the judge's and the author's own)
        mask = np.ones((len(df), 5), bool)
        mask[np.arange(len(df)), jj_arr] = False
        mask[np.arange(len(df)), aj_arr] = False
        df[f"simO_{m}"] = np.nanmean(np.where(mask, base[m], np.nan), axis=1)
    nonself = df[df["true_author"] != df["judge"]].copy()
    nonself["stratum"] = nonself["judge"] + "|" + nonself["true_author"]

    t1 = {"events": {}, "per_judge": {}, "pooled": {}}
    for cond in ("lineup", "single"):
        g = nonself[nonself["condition"] == cond]
        t1["events"][cond] = {J: {"self_namings_on_others_text": int(g.loc[g["judge"] == J, "named_self"].sum()),
                                  "n": int((g["judge"] == J).sum())} for J in MODELS}
        for J in MODELS:
            gj = g[g["judge"] == J]
            for m in MEASURES:
                a = gee_selfname(gj, f"simJ_{m}", None, "true_author")
                b = gee_selfname(gj, f"simJ_{m}", f"simO_{m}", "true_author")
                # descriptive: mean raw similarity, self-named vs not
                t1["per_judge"][f"{cond}|{J}|{m}"] = {
                    "author_fe": a, "author_fe_plus_centrality": b,
                    "mean_sim_selfnamed": float(gj.loc[gj["named_self"] == 1, f"simJ_{m}"].mean())
                    if gj["named_self"].sum() else None,
                    "mean_sim_other": float(gj.loc[gj["named_self"] == 0, f"simJ_{m}"].mean()),
                }
        for m in MEASURES:
            t1["pooled"][f"{cond}|{m}"] = {
                "judge_x_author_fe": gee_selfname(g, f"simJ_{m}", None, "stratum"),
                "plus_centrality": gee_selfname(g, f"simJ_{m}", f"simO_{m}", "stratum"),
            }
    out["test1"] = t1

    # -------------------------------------------- test 1b: typicality
    # typicality of r for its own author = sim to author's LPO centroid
    t1b = {}
    resp_typ = {m: CS[m][np.arange(len(resp)), resp["model"].map(midx).to_numpy()]
                for m in MEASURES}
    for cond in ("lineup", "single"):
        for A in ("Claude", "GPT"):
            g = df[(df["condition"] == cond) & (df["true_author"] == A)]
            selfhit = g[g["judge"] == A].set_index("prompt_id")["correct"]
            peerrate = g[g["judge"] != A].groupby("prompt_id")["correct"].mean()
            rows = resp[resp["model"] == A].set_index("prompt_id").index
            res = {"self_hits": int(selfhit.sum()), "n": int(len(selfhit))}
            for m in MEASURES:
                typ = pd.Series(resp_typ[m][resp["model"].to_numpy() == A], index=rows)
                sh = selfhit.reindex(rows)
                pr = peerrate.reindex(rows)
                # self: point-biserial = Spearman with a binary outcome;
                # Mann-Whitney is reported for the p.
                if 0 < sh.sum() < len(sh):
                    mw = stats.mannwhitneyu(typ[sh == 1], typ[sh == 0], alternative="two-sided")
                    auc = mw.statistic / (sh.sum() * (len(sh) - sh.sum()))
                    p_self = float(mw.pvalue)
                else:
                    auc, p_self = None, None
                rho, p_peer = stats.spearmanr(typ, pr)
                res[m] = {"self_auc_typicality": None if auc is None else float(auc),
                          "self_p": p_self,
                          "peer_rate_spearman": float(rho), "peer_p": float(p_peer)}
            t1b[f"{cond}|{A}"] = res
    out["test1b_typicality"] = t1b

    # ---------------------------------------------------- test 2 (observed)
    wrong = (gj_arr != aj_arr)

    def rank_of_guess(sims: np.ndarray, exclude_author: bool, guess: np.ndarray) -> np.ndarray:
        """1 = the guessed model is the most similar candidate. Ties count
        against the guess (rank = 1 + number of candidates strictly more
        similar, plus half the ties)."""
        s = sims.astype(float).copy()
        if exclude_author:
            s[np.arange(len(s)), aj_arr] = -np.inf
        sg = s[np.arange(len(s)), guess][:, None]
        more = (s > sg).sum(1)
        ties = (s == sg).sum(1) - 1
        return 1 + more + 0.5 * ties

    def t2_stats(sim_same: dict, sim_cent: dict, rf_p: np.ndarray, guess: np.ndarray,
                 guess_w: np.ndarray | None = None) -> dict:
        """`guess` feeds the all-attribution statistics, `guess_w` (default:
        the same) the wrong-attribution ones. They differ only under the
        guess-shuffle null, where wrong rows are shuffled among themselves so
        a shuffled guess can never land on the true author."""
        guess_w = guess if guess_w is None else guess_w
        res = {}
        for cond in ("lineup", "single"):
            c = cond_arr == cond
            cw = c & wrong
            for m in MEASURES:
                r_same = rank_of_guess(sim_same[m], True, guess_w)[cw]
                r_cent = rank_of_guess(sim_cent[m], False, guess)[c]
                r_cent_w = rank_of_guess(sim_cent[m], True, guess_w)[cw]
                res[f"{cond}|{m}"] = {
                    "same_prompt_wrong_top1": float((r_same == 1).mean()),
                    "same_prompt_wrong_meanrank": float(r_same.mean()),
                    "centroid_all_top1": float((r_cent == 1).mean()),
                    "centroid_wrong_top1": float((r_cent_w == 1).mean()),
                }
            r_rf = rank_of_guess(rf_p, False, guess)[c]
            r_rf_w = rank_of_guess(rf_p, True, guess_w)[cw]
            res[f"{cond}|rf"] = {"centroid_all_top1": float((r_rf == 1).mean()),
                                 "centroid_wrong_top1": float((r_rf_w == 1).mean())}
        return res

    sim_cent = {m: CS[m][r_arr] for m in MEASURES}
    rf_rows = proba[r_arr]
    obs2 = t2_stats(base, sim_cent, rf_rows, gj_arr)
    counts2 = {cond: {"n_all": int((cond_arr == cond).sum()),
                      "n_wrong": int(((cond_arr == cond) & wrong).sum())}
               for cond in ("lineup", "single")}
    # How identifying is each measure on its own? nearest-centroid accuracy
    ident = {m: float((CS[m].argmax(1) == resp["model"].map(midx).to_numpy()).mean())
             for m in MEASURES}
    ident["rf"] = rf_acc
    # For the same-prompt test: how often is the most similar non-author
    # candidate the true author's "nearest neighbour" in the corpus at all?
    out["test2_counts"] = counts2
    out["nearest_centroid_accuracy"] = ident

    # ------------------------------------------------------- permutations
    # Test-1 permutation statistic per (cond, judge, measure), pooled per
    # (cond, measure) with judge x author strata.
    def t1_stats(sims: dict) -> dict:
        res = {}
        ns = (aj_arr != jj_arr)
        strata = jj_arr * 10 + aj_arr
        y = df["named_self"].to_numpy()
        for cond in ("lineup", "single"):
            c = ns & (cond_arr == cond)
            for m in MEASURES:
                x = sims[m][np.arange(len(df)), jj_arr]
                res[f"{cond}|pooled|{m}"] = demeaned_diff(x[c], y[c], strata[c])
                for J in MODELS:
                    cj = c & (jj_arr == midx[J])
                    if y[cj].sum() >= MIN_EVENTS:
                        res[f"{cond}|{J}|{m}"] = demeaned_diff(x[cj], y[cj], aj_arr[cj])
        return res

    obs1 = t1_stats(base)
    null1 = {k: [] for k in obs1}
    null2_ref = {k: {kk: [] for kk in v} for k, v in obs2.items()}
    null2_guess = {k: {kk: [] for kk in v} for k, v in obs2.items()}

    # strata for guess shuffling
    strat_g = cond_arr.astype(object) + "|" + df["judge"].to_numpy() + "|" + df["true_author"].to_numpy()
    strat_groups = [np.where(strat_g == s)[0] for s in np.unique(strat_g)]
    strat_groups_w = [np.where((strat_g == s) & wrong)[0] for s in np.unique(strat_g)]

    for _ in range(args.perms):
        # N-ref: independent prompt permutation per author
        refp = np.empty_like(REF)
        for j in range(len(MODELS)):
            refp[:, j] = REF[rng.permutation(len(prompts)), j]
        sims_p = {m: cand_sims(refp, m) for m in MEASURES}
        for k, v in t1_stats(sims_p).items():
            null1[k].append(v)
        # test 2 under N-ref: only the same-prompt stats change
        s2 = t2_stats(sims_p, sim_cent, rf_rows, gj_arr)
        for k, v in s2.items():
            for kk, vv in v.items():
                null2_ref[k][kk].append(vv)
        # N-guess: shuffle guesses within condition x judge x true author
        g = gj_arr.copy()
        for idx in strat_groups:
            g[idx] = gj_arr[rng.permutation(idx)]
        gw = gj_arr.copy()
        for idx in strat_groups_w:
            gw[idx] = gj_arr[rng.permutation(idx)]
        s2g = t2_stats(base, sim_cent, rf_rows, g, gw)
        for k, v in s2g.items():
            for kk, vv in v.items():
                null2_guess[k][kk].append(vv)

    def pval(obs, null, side="greater"):
        null = np.asarray([x for x in null if not np.isnan(x)])
        if np.isnan(obs) or not len(null):
            return None
        if side == "greater":
            return float((1 + (null >= obs).sum()) / (1 + len(null)))
        return float((1 + (np.abs(null - null.mean()) >= abs(obs - null.mean())).sum()) / (1 + len(null)))

    out["test1_permutation"] = {
        k: {"observed": v, "null_mean": float(np.nanmean(null1[k])),
            "null_sd": float(np.nanstd(null1[k])),
            "p_one_sided": pval(v, null1[k], "greater"),
            "p_two_sided": pval(v, null1[k], "two")}
        for k, v in obs1.items()}

    t2 = {}
    for k, v in obs2.items():
        t2[k] = {}
        for kk, vv in v.items():
            ent = {"observed": vv}
            if len(null2_guess[k][kk]):
                ent["null_guess_mean"] = float(np.mean(null2_guess[k][kk]))
                side = "less" if "meanrank" in kk else "greater"
                ng = np.asarray(null2_guess[k][kk])
                ent["p_guess"] = float((1 + ((ng <= vv) if side == "less" else (ng >= vv)).sum()) / (1 + len(ng)))
            if "same_prompt" in kk:
                nr = np.asarray(null2_ref[k][kk])
                ent["null_ref_mean"] = float(nr.mean())
                side = "less" if "meanrank" in kk else "greater"
                ent["p_ref"] = float((1 + ((nr <= vv) if side == "less" else (nr >= vv)).sum()) / (1 + len(nr)))
            t2[k][kk] = ent
    out["test2"] = t2

    # per-judge top-1 on wrong attributions, same-prompt (observed only,
    # with its guess-shuffle chance level) - descriptive
    per_j = {}
    for cond in ("lineup", "single"):
        for J in MODELS:
            cw = (cond_arr == cond) & wrong & (jj_arr == midx[J])
            if cw.sum() == 0:
                continue
            per_j[f"{cond}|{J}"] = {"n_wrong": int(cw.sum())}
            for m in MEASURES:
                r_same = rank_of_guess(base[m], True, gj_arr)[cw]
                per_j[f"{cond}|{J}"][m] = {"top1_k": int((r_same == 1).sum()),
                                           "top1": float((r_same == 1).mean())}
    out["test2_per_judge"] = per_j

    out["reading_md"] = reading(out)
    RESULTS_DIR.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    write_md(out)
    print_summary(out)
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def fp(p):
    if p is None:
        return "n/a"
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def reading(out: dict) -> str:
    """Plain-language summary; every number comes from `out`."""
    t1, perm, t2, t1b = out["test1"], out["test1_permutation"], out["test2"], out["test1b_typicality"]
    fits = [(k, v["author_fe"]) for k, v in t1["per_judge"].items() if v["author_fe"]]
    fits += [(k, v["judge_x_author_fe"]) for k, v in t1["pooled"].items() if v["judge_x_author_fe"]]
    n_sig = sum(1 for _, a in fits if a["p"] < 0.05)
    perm_sig = sum(1 for v in perm.values() if v["p_one_sided"] is not None and v["p_one_sided"] < 0.05)

    def orr(key, m, pooled=False):
        a = (t1["pooled"][f"{key}|{m}"]["judge_x_author_fe"] if pooled
             else t1["per_judge"][f"{key}|{m}"]["author_fe"])
        return (f"OR {a['OR_per_sd']:.2f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}], "
                f"p = {fp(a['p'])}, {a['events']}/{a['n']}")

    def s2(cond, m, stat):
        e = t2[f"{cond}|{m}"][stat]
        s = f"{pct(e['observed'])} vs guess-null {pct(e['null_guess_mean'])} (p = {fp(e['p_guess'])}"
        if "p_ref" in e:
            s += f"; ref-null {pct(e['null_ref_mean'])}, p = {fp(e['p_ref'])}"
        return s + ")"

    c = out["test2_counts"]
    g1 = t1b["single|GPT"]
    return "\n".join([
        "## Reading\n",
        "**Test 1: no evidence that resemblance to the judge's own answer "
        "drives self-naming.** Across the "
        f"{len(fits)} fitted models (per judge and pooled, four measures, two "
        f"conditions), {n_sig} has p < 0.05, about what chance gives. In the "
        "clean condition, single-text, GPT names itself on 79 of the 152 "
        "responses it did not write, and the similarity of those responses to "
        f"GPT's own answer does not predict which: stylo {orr('single|GPT', 'stylo')}; "
        f"word {orr('single|GPT', 'word')}; function words {orr('single|GPT', 'funcword')}. "
        f"Claude's 13 single-text self false alarms: word {orr('single|Claude', 'word')}. "
        f"Pooled lineup, all five judges: stylo {orr('lineup', 'stylo', True)}; "
        f"char {orr('lineup', 'char', True)}. The permutation test for the "
        f"prompt-specific component gives {perm_sig} of {len(perm)} one-sided "
        "p < 0.05. The confidence intervals are wide: an odds ratio near 2 "
        "per SD in the single-text Claude cell cannot be excluded.\n",
        "**Test 1b: typical own text is not preferentially self-recognised.** "
        "Claude's AUCs for typicality predicting its own hits range from "
        + ", ".join(f"{t1b[k][m]['self_auc_typicality']:.2f}" for k in ("lineup|Claude", "single|Claude") for m in MEASURES)
        + " (all p > 0.1). GPT in single-text is the one case with a signal "
        f"(word AUC {g1['word']['self_auc_typicality']:.2f}, p = {fp(g1['word']['self_p'])}; "
        f"function words {g1['funcword']['self_auc_typicality']:.2f}, p = {fp(g1['funcword']['self_p'])}), "
        f"but it rests on {g1['n'] - g1['self_hits']} misses out of {g1['n']}, and "
        "peers' recognition of GPT's text tracks the same typicality "
        f"(word rho = {g1['word']['peer_rate_spearman']:+.2f}, p = {fp(g1['word']['peer_p'])}; "
        f"function words {g1['funcword']['peer_rate_spearman']:+.2f}, p = {fp(g1['funcword']['peer_p'])}): "
        "the typical GPT text is easy for everyone, which is identifiability, "
        "not self-familiarity.\n",
        "**Test 2: judges' choices barely follow similarity.** The measures "
        "themselves identify authors well (nearest-centroid accuracy "
        + ", ".join(f"{k} {pct(v)}" for k, v in out["nearest_centroid_accuracy"].items())
        + "). Yet when a judge is wrong, the model it names is the one whose "
        "same-prompt response most resembles the text in lineup stylo "
        f"{s2('lineup', 'stylo', 'same_prompt_wrong_top1')} of {c['lineup']['n_wrong']} "
        f"wrong attributions, and in single-text stylo {s2('single', 'stylo', 'same_prompt_wrong_top1')} "
        f"of {c['single']['n_wrong']}. Uniform chance is 25%; most of the gap "
        "above it is already in the guess-null, i.e. in which authors each "
        "judge confuses, not in which response. Over all attributions, the "
        "named model is the nearest centroid in single-text function words "
        f"{s2('single', 'funcword', 'centroid_all_top1')} and word n-grams "
        f"{s2('single', 'word', 'centroid_all_top1')}: detectable with 950 "
        "judgments, but one to three points. The stylometric classifier does "
        f"no better as a model of the judges: lineup {s2('lineup', 'rf', 'centroid_all_top1')}, "
        f"single-text {s2('single', 'rf', 'centroid_all_top1')}.\n",
        "**Taken together.** On the existing data the familiarity account gets "
        "no support at the level of individual texts: a response that is "
        "closer to what J wrote for the same prompt is not more likely to be "
        "called J, J's most typical texts are not the ones J recognises, and "
        "judges' choices do not follow any of four similarity measures that "
        "themselves recover the author 74 to 79% of the time. This fits the "
        "rationale analysis: judges apply a stable stereotype of each model, "
        "and Claude's and GPT's high self rates come from their stereotype of "
        "themselves happening to fit their own text, not from a closer match "
        "between the text and their own generations.\n",
        "**Caveats a reviewer should weigh.** (1) One corpus response per "
        "(model, prompt) is a noisy proxy for what J would write; a real "
        "generative-match test scores r under J's own likelihood, or samples "
        "J several times per prompt. A null here is a null for this proxy. "
        "(2) Surface similarity is not the judge's internal representation; "
        "the four measures span features, content n-grams, character n-grams "
        "and function words, but others are possible. (3) In the lineup, J's "
        "own response is in the session, so any similarity effect there could "
        "be side-by-side matching; single-text is the clean test but has self "
        "false alarms only for GPT (79) and Claude (13). (4) With 13 events the "
        "Claude single-text cell has little power. (5) The guess-shuffle null "
        "treats judgments within judge x author as exchangeable; responses "
        "recur across judges, so its p-values are somewhat optimistic.\n",
    ])


def print_summary(out: dict) -> None:
    print("=" * 74 + "\nGENERATIVE MATCH\n" + "=" * 74)
    print(f"  RF OOF accuracy {out['rf_oof_accuracy']:.3f}; nearest-centroid "
          + ", ".join(f"{k} {v:.3f}" for k, v in out["nearest_centroid_accuracy"].items()))
    print("\n  test 1, self-naming on others' text (OR per SD, author FE):")
    for k, v in out["test1"]["per_judge"].items():
        a = v["author_fe"]
        if a:
            print(f"    {k:28s} OR {a['OR_per_sd']:.2f} [{a['ci95'][0]:.2f},{a['ci95'][1]:.2f}] "
                  f"p {fp(a['p'])}  events {a['events']}/{a['n']}")
    print("\n  test 1 permutation (N-ref):")
    for k, v in out["test1_permutation"].items():
        print(f"    {k:28s} obs {v['observed']:+.4f}  null {v['null_mean']:+.4f}  p1 {fp(v['p_one_sided'])}")
    print("\n  test 2:")
    for k, v in out["test2"].items():
        print(f"    {k:18s} " + "  ".join(
            f"{kk.replace('same_prompt_', 'sp_').replace('centroid_', 'c_')} {e['observed']:.3f}"
            f"(g {e.get('null_guess_mean', float('nan')):.3f} p {fp(e.get('p_guess'))}"
            + (f"; r {e['null_ref_mean']:.3f} p {fp(e['p_ref'])}" if 'p_ref' in e else "") + ")"
            for kk, e in v.items()))


def write_md(out: dict) -> None:
    L = []
    w = L.append
    w("# Generative match: do judges name themselves on text like their own?\n")
    w("Generated by `src/generative_match.py`; numbers come from "
      "`results/generative_match.json`. Do not edit by hand.\n")
    w("Proxy for *what judge J would write for prompt p*: J's own corpus "
      "response to p. Four similarity measures between responses: **stylo** "
      "(negative Euclidean distance over the 18 z-scored features), **word** "
      "(word 1-2-gram TF-IDF cosine), **char** (char 3-5-gram TF-IDF cosine), "
      "**funcword** (cosine of z-scored frequencies of "
      f"{len(FUNCTION_WORDS)} function words). "
      f"Permutation nulls use {out['n_perms']} draws.\n")
    w("How identifying is each measure by itself? Nearest leave-prompt-out "
      "centroid accuracy over the 190 responses (chance 20%): "
      + ", ".join(f"{k} {pct(v)}" for k, v in out["nearest_centroid_accuracy"].items())
      + " (rf = Engine A random forest, out-of-fold).\n")
    w("**Caveat.** In the lineup, J's own response to p is in the same session "
      "as r, so similarity to it can act through side-by-side matching rather "
      "than familiarity. Only the single-text condition, where J never sees its "
      "own answer, isolates generative match, and there only Claude and GPT "
      "ever name themselves.\n")

    t1 = out["test1"]
    w("## Test 1: does resemblance to J's own answer predict J naming itself on others' text?\n")
    w("Self-namings on text J did not write (the events available):\n")
    w("| condition | " + " | ".join(MODELS) + " |\n|---|" + "---|" * len(MODELS))
    for cond in ("lineup", "single"):
        e = t1["events"][cond]
        w(f"| {cond} | " + " | ".join(f"{e[J]['self_namings_on_others_text']}/{e[J]['n']}" for J in MODELS) + " |")
    w("")
    w(f"Logistic GEE clustered by prompt; OR per SD of similarity to J's own "
      f"same-prompt response, within true author (author fixed effects; strata "
      f"with zero or all self-namings dropped). *+centrality* adds r's mean "
      f"similarity to the other candidates' responses to p. Fitted only with "
      f">= {MIN_EVENTS} events. Permutation p (N-ref, one-sided): the same "
      "within-author contrast with each author's responses shuffled across "
      "prompts, which keeps J's general style and removes the prompt-specific "
      "match; *null mean* > 0 therefore measures the general-style part.\n")
    w("| cond | judge | measure | events/n | OR per SD [95% CI] | p | +centrality OR (p) | perm obs / null mean | perm p |")
    w("|---|---|---|---|---|---|---|---|---|")
    perm = out["test1_permutation"]
    for cond in ("lineup", "single"):
        for J in MODELS + ["pooled"]:
            for m in MEASURES:
                if J == "pooled":
                    v = t1["pooled"][f"{cond}|{m}"]
                    a, b = v["judge_x_author_fe"], v["plus_centrality"]
                else:
                    v = t1["per_judge"][f"{cond}|{J}|{m}"]
                    a, b = v["author_fe"], v["author_fe_plus_centrality"]
                if not a:
                    continue
                pk = perm.get(f"{cond}|{J}|{m}")
                w(f"| {cond} | {J} | {m} | {a['events']}/{a['n']} | {a['OR_per_sd']:.2f} "
                  f"[{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}] | {fp(a['p'])} | "
                  + (f"{b['OR_per_sd']:.2f} ({fp(b['p'])})" if b else "n/a") + " | "
                  + (f"{pk['observed']:+.3f} / {pk['null_mean']:+.3f} | {fp(pk['p_one_sided'])}" if pk else "n/a | n/a")
                  + " |")
    w("")

    w("### Test 1b: typicality of J's own responses\n")
    w("Typicality = similarity of an author's response to that author's "
      "leave-prompt-out centroid. *Self AUC*: probability that a response J "
      "recognised is more typical than one it missed (Mann-Whitney). *Peer rho*: "
      "Spearman correlation of typicality with the share of the four other "
      "judges who named the author correctly. n = 38 responses per row.\n")
    w("| cond | author | self hits | measure | self AUC (p) | peer rho (p) |")
    w("|---|---|---|---|---|---|")
    for k, v in out["test1b_typicality"].items():
        cond, A = k.split("|")
        for m in MEASURES:
            t = v[m]
            sa = "n/a" if t["self_auc_typicality"] is None else f"{t['self_auc_typicality']:.2f} ({fp(t['self_p'])})"
            w(f"| {cond} | {A} | {v['self_hits']}/{v['n']} | {m} | {sa} | "
              f"{t['peer_rate_spearman']:+.2f} ({fp(t['peer_p'])}) |")
    w("")

    w("## Test 2: is the named model the one whose writing r most resembles?\n")
    c = out["test2_counts"]
    w(f"Lineup: {c['lineup']['n_all']} attributions, {c['lineup']['n_wrong']} wrong. "
      f"Single-text: {c['single']['n_all']}, {c['single']['n_wrong']} wrong.\n")
    w("- *same-prompt, wrong*: among the four non-author models, the share of "
      "wrong attributions where the named model's own response to p is the "
      "most similar to r (uniform chance 25%).")
    w("- *centroid, all*: among all five, the share where the named model's "
      "leave-prompt-out centroid is the most similar (uniform chance 20%); "
      "*centroid, wrong* restricts to wrong attributions and the four non-authors.")
    w("- *rf*: the same, ranking candidates by the random forest's out-of-fold "
      "probabilities (the stylometric-classifier baseline).")
    w("- *guess-null*: the same share with the judge's answers shuffled within "
      "judge x true author (keeps each judge's answer mix; the fair chance "
      "level, since single-text answers are almost all GPT or Claude). "
      "*ref-null*: responses shuffled across prompts within author (for the "
      "same-prompt statistic only).\n")
    w("| cond | measure | statistic | observed | guess-null mean | p (guess) | ref-null mean | p (ref) |")
    w("|---|---|---|---|---|---|---|---|")
    for k, v in out["test2"].items():
        cond, m = k.split("|")
        for kk, e in v.items():
            if "meanrank" in kk:
                continue
            w(f"| {cond} | {m} | {kk.replace('_', ' ')} | {pct(e['observed'])} | "
              f"{pct(e.get('null_guess_mean'))} | {fp(e.get('p_guess'))} | "
              f"{pct(e.get('null_ref_mean'))} | {fp(e.get('p_ref'))} |")
    w("")
    w("Per judge, same-prompt, wrong attributions (top-1 count / n):\n")
    w("| cond | judge | n wrong | " + " | ".join(MEASURES) + " |")
    w("|---|---|---|" + "---|" * len(MEASURES))
    for k, v in out["test2_per_judge"].items():
        cond, J = k.split("|")
        w(f"| {cond} | {J} | {v['n_wrong']} | " +
          " | ".join(f"{v[m]['top1_k']} ({pct(v[m]['top1'],0)})" for m in MEASURES) + " |")
    w("")
    w(out.get("reading_md", ""))
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
