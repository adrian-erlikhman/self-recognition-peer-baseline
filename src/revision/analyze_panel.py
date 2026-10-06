"""CompLLM revision: score every condition a panel has run.

    python src/revision/analyze_panel.py --panel openweight
    python src/revision/analyze_panel.py --panel frontier

Reads results/revision/<panel>/*.jsonl (and likelihood.csv for the
open-weight panel) and writes results/revision/<panel>/summary.json plus a
readable docs/revision/PANEL_<panel>.md. Every rate is reported with its
count. Uncertainty is prompt-clustered throughout: bootstrap intervals
resample prompts, and the self-advantage test permutes which of the five
judges on a response is called "self", which is exchangeable within a
response under the null of no self-advantage.

Measures
--------
attribution (lineup, single, shuffle_lineup)
  self rate       J names J on J's own texts
  peer baseline   the other judges name J on J's texts
  self-advantage  self rate - peer baseline
  false alarms    J names J on texts it did not write
  non-self acc.   J's accuracy on texts it did not write
yes/no (binary, shuffle_binary, onpolicy, onpolicy_resample)
  hit rate, false-alarm rate, and AUC of P(yes) for own versus others' text,
  which is free of the judge's bias towards answering yes.
likelihood (open-weight only)
  the accuracy of naming as "self" the text the judge finds most likely,
  relative to the rest of the panel; and whether that relative likelihood
  predicts the judge's verbal self-naming (the familiarity test).
quality
  each judge's rating of its own text against its peers' ratings of the same
  text (self-preference), and each text's peer-rated quality as a covariate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT  # noqa: E402
from panels import PANELS  # noqa: E402

RNG_SEED = 20260929
N_BOOT = 4000
N_PERM = 4000
N_PERM_AUC = 2000


def holm(ps: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values for one family (here: the judges of a panel)."""
    keys = sorted(ps, key=lambda k: ps[k])
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


TAG = ""  # set by --tag: read <cond>_<tag>.jsonl, write summary_<tag>.json


def load(panel, cond: str) -> pd.DataFrame | None:
    p = panel.results_dir / f"{cond}{'_' + TAG if TAG else ''}.jsonl"
    if not p.exists():
        return None
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if r.get("ok")]
    # Keep the last record per item (a resumed run may have re-run one).
    return pd.DataFrame({r["item_id"]: r for r in recs}.values()) if recs else None


def attribution_rows(df: pd.DataFrame, cond: str) -> pd.DataFrame:
    """One row per judgment: prompt, judge, true author, predicted."""
    rows = []
    for r in df.itertuples():
        if cond == "single":
            pred = (r.parsed or {}).get("model") if isinstance(r.parsed, dict) else None
            rows.append((r.prompt_id, r.genre, r.judge, r.true_author, pred))
        else:
            parsed = r.parsed if isinstance(r.parsed, dict) else {}
            for slot, author in r.slot_to_author.items():
                pred = (parsed.get(slot) or {}).get("model")
                rows.append((r.prompt_id, r.genre, r.judge, author, pred))
    out = pd.DataFrame(rows, columns=["prompt_id", "genre", "judge", "author", "pred"])
    out["correct"] = (out["pred"] == out["author"]).astype(int)
    return out


# -- clustered uncertainty ---------------------------------------------------
def boot_ci(values_by_prompt: dict[str, tuple[float, float]], stat, rng) -> list[float]:
    """Percentile CI of stat(num, den sums) resampling prompts."""
    keys = list(values_by_prompt)
    arr = np.array([values_by_prompt[k] for k in keys], dtype=float)
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    sims = stat(arr[idx])
    return [float(np.nanpercentile(sims, 2.5)), float(np.nanpercentile(sims, 97.5))]


def self_advantage(a: pd.DataFrame, judge: str, names: list[str], rng) -> dict:
    """Self rate, peer baseline and advantage for one judge, with a prompt
    bootstrap CI and a within-response permutation p-value."""
    own = a[a["author"] == judge]
    # Matrix: one row per J-authored response (prompt), one column per judge.
    piv = own.pivot_table(index="prompt_id", columns="judge", values="correct", aggfunc="mean")
    piv = piv.dropna()
    if piv.empty or judge not in piv:
        return {}
    j = list(piv.columns).index(judge)
    M = piv.to_numpy()
    others = [k for k in range(M.shape[1]) if k != j]
    self_rate = M[:, j].mean()
    peer = M[:, others].mean()
    obs = self_rate - peer
    # Permutation: under H0 the five judgments on a response are exchangeable
    # with respect to which judge is "self".
    k = M.shape[1]
    perm_idx = rng.integers(0, k, size=(N_PERM, M.shape[0]))
    rows = np.arange(M.shape[0])
    s = M[rows[None, :], perm_idx]                       # (N_PERM, n)
    tot = M.sum(axis=1)[None, :]
    sims = s.mean(axis=1) - ((tot - s).sum(axis=1) / (M.shape[0] * (k - 1)))
    p = float((np.abs(sims) >= abs(obs) - 1e-12).mean())
    # Bootstrap over prompts.
    idx = rng.integers(0, M.shape[0], size=(N_BOOT, M.shape[0]))
    bs = M[idx][:, :, j].mean(axis=1) - M[idx][:, :, others].mean(axis=(1, 2))
    return {"n_self": int(M.shape[0]), "self_rate": float(self_rate),
            "self_hits": int(M[:, j].sum()), "peer_baseline": float(peer),
            "self_advantage": float(obs),
            "adv_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "perm_p": p}


def one_own(df: pd.DataFrame, value: np.ndarray, own: np.ndarray):
    """(G x k) matrix of `value` by prompt and the column of the judge's own
    text, when every prompt has k texts of which exactly one is the judge's
    (true of every judge in the open-weight and frontier panels). The within-
    prompt permutation is then a random choice of column per prompt."""
    d = df.reset_index(drop=True)
    groups = list(d.groupby("prompt_id").indices.values())
    k = len(groups[0])
    if any(len(g) != k for g in groups) or any(own[g].sum() != 1 for g in groups):
        raise ValueError("unbalanced prompts")
    idx = np.array([np.asarray(g) for g in groups])
    V = value[idx]
    r = own[idx].argmax(axis=1)
    return V, r


def discrimination(mine: pd.DataFrame, judge: str, rng) -> dict:
    """Does J name itself more on its own text than on others' (hit rate
    minus false-alarm rate)? Self-advantage alone cannot say: a judge that
    names itself on half of everything beats peers who rarely name it, while
    telling its own text from others' no better than chance. Within-prompt
    permutation of which text is J's own (one lineup, or the panel's texts
    for one prompt in the single format); also the Haldane-corrected log
    odds ratio of signal detection, as in Study 1."""
    m = mine.reset_index(drop=True)
    if m.empty:
        return {}
    y = (m["pred"] == judge).to_numpy()
    own = (m["author"] == judge).to_numpy()
    if own.sum() == 0 or (~own).sum() == 0:
        return {}
    obs = y[own].mean() - y[~own].mean()
    V, r = one_own(m, y.astype(float), own)
    G, k = V.shape
    rows = np.arange(G)
    R = rng.integers(0, k, size=(N_PERM, G))
    picked = V[rows[None, :], R]
    sims = picked.mean(axis=1) - (V.sum() - picked.sum(axis=1)) / (G * (k - 1))
    h, fa = int(y[own].sum()), int(y[~own].sum())
    n1, n0 = int(own.sum()), int((~own).sum())
    lor = float(np.log(((h + .5) / (n1 - h + .5)) / ((fa + .5) / (n0 - fa + .5))))
    return {"hit_minus_fa": float(obs),
            "disc_perm_p": float((np.abs(sims) >= abs(obs) - 1e-12).mean()),
            "log_or": lor}


def attribution_metrics(a: pd.DataFrame, names: list[str], rng) -> dict:
    out = {"n_judgments": int(len(a)), "parsed_share": float(a["pred"].notna().mean()),
           "accuracy": float(a["correct"].mean()), "judges": {}}
    blame = a["pred"].value_counts(normalize=True).reindex(names).fillna(0)
    out["blame_share"] = {k: float(v) for k, v in blame.items()}
    for j in names:
        mine = a[a["judge"] == j]
        non = mine[mine["author"] != j]
        d = {"accuracy": float(mine["correct"].mean()),
             "nonself_acc": float(non["correct"].mean()),
             "nonself_n": int(len(non)),
             "false_alarm": float((non["pred"] == j).mean()),
             "false_alarm_n": int((non["pred"] == j).sum()),
             "names_self_share": float((mine["pred"] == j).mean())}
        d.update(self_advantage(a, j, names, rng))
        d.update(discrimination(mine, j, rng))
        out["judges"][j] = d
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def yesno_metrics(df: pd.DataFrame, names: list[str], rng) -> dict:
    out = {"n": int(len(df)), "judges": {}}
    for j in names:
        mine = df[df["judge"] == j]
        if mine.empty:  # a partial run
            continue
        own, oth = mine[mine["true_author"] == j], mine[mine["true_author"] != j]
        a = auc(own["p_yes"].to_numpy(float), oth["p_yes"].to_numpy(float))
        # Prompt bootstrap and within-prompt permutation of the AUC: under
        # H0 the judge's P(yes) does not depend on which of a prompt's texts
        # is its own.
        mine = mine.reset_index(drop=True)
        V, r = one_own(mine, mine["p_yes"].to_numpy(float),
                       (mine["true_author"] == j).to_numpy())
        G, k = V.shape
        rows = np.arange(G)
        n1, n0 = G, G * (k - 1)
        sims = []
        for _ in range(1000):
            b = rng.integers(0, G, size=G)
            Vb = V[b]
            rk = rankdata(Vb.ravel()).reshape(Vb.shape)
            sims.append((rk[rows, r[b]].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        rk = rankdata(V.ravel()).reshape(V.shape)
        R = rng.integers(0, k, size=(N_PERM_AUC, G))
        perm_auc = (rk[rows[None, :], R].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
        p_auc = float((np.abs(perm_auc - 0.5) >= abs(a - 0.5) - 1e-12).mean())
        out["judges"][j] = {
            "auc_perm_p": p_auc,
            "hit_rate": float(own["says_yes"].mean()), "hits": int(own["says_yes"].sum()),
            "n_own": int(len(own)),
            "false_alarm": float(oth["says_yes"].mean()), "fa": int(oth["says_yes"].sum()),
            "n_other": int(len(oth)),
            "auc": a, "auc_ci95": [float(np.nanpercentile(sims, 2.5)),
                                   float(np.nanpercentile(sims, 97.5))],
            "mean_p_yes_own": float(own["p_yes"].mean()),
            "mean_p_yes_other": float(oth["p_yes"].mean()),
            "format_mass": float(mine["mass"].mean()) if "mass" in mine else None,
        }
    return out


def likelihood_metrics(panel, rng) -> dict | None:
    p = panel.results_dir / "likelihood.csv"
    if not p.exists():
        return None
    L = pd.read_csv(p, dtype={"prompt_id": str})
    out = {"judges": {}}
    for col in ("mean_lp_cond", "mean_lp_uncond"):
        W = L.pivot_table(index=["prompt_id", "author"], columns="scorer", values=col)
        W = W.dropna()
        res = {}
        for j in panel.names:
            if j not in W:
                continue
            others = [k for k in W.columns if k != j]
            # Relative likelihood: how much more likely J finds the text than
            # its peers do. Raw likelihood alone favours whichever author
            # writes the most predictable text for every scorer.
            rel = W[j] - W[others].mean(axis=1)
            df = rel.rename("rel").reset_index()
            raw = W[j].rename("raw").reset_index()
            pick_rel = df.loc[df.groupby("prompt_id")["rel"].idxmax()]
            pick_raw = raw.loc[raw.groupby("prompt_id")["raw"].idxmax()]
            res[j] = {"rel_argmax_acc": float((pick_rel["author"] == j).mean()),
                      "raw_argmax_acc": float((pick_raw["author"] == j).mean()),
                      "n_prompts": int(df["prompt_id"].nunique()),
                      "auc_rel": auc(df.loc[df.author == j, "rel"].to_numpy(),
                                     df.loc[df.author != j, "rel"].to_numpy())}
        out["judges"][col] = res
    return out


def familiarity(panel, cond_df: pd.DataFrame | None, cond: str) -> dict | None:
    """Does the judge's relative likelihood of a NON-self text predict it
    naming itself? Difference in mean standardised relative likelihood
    between texts it claimed and texts it did not, with a within-prompt
    permutation test (GEE was anticonservative with ~20 events)."""
    p = panel.results_dir / "likelihood.csv"
    if cond_df is None or not p.exists():
        return None
    L = pd.read_csv(p, dtype={"prompt_id": str})
    W = L.pivot_table(index=["prompt_id", "author"], columns="scorer", values="mean_lp_cond")
    out = {}
    for j in panel.names:
        if j not in W:
            continue
        rel = (W[j] - W[[k for k in W.columns if k != j]].mean(axis=1)).rename("rel").reset_index()
        if cond in ("binary", "onpolicy", "onpolicy_resample", "shuffle_binary"):
            d = cond_df[(cond_df.judge == j) & (cond_df.true_author != j)][
                ["prompt_id", "true_author", "says_yes"]].rename(
                columns={"true_author": "author", "says_yes": "y"})
        else:
            d = cond_df[(cond_df.judge == j) & (cond_df.author != j)][
                ["prompt_id", "author", "pred"]].copy()
            d["y"] = (d["pred"] == j).astype(int)
        d = d.merge(rel, on=["prompt_id", "author"]).dropna().reset_index(drop=True)
        d["y"] = d["y"].astype(int)
        if d["y"].sum() < 5 or d["y"].sum() > len(d) - 5:
            out[j] = {"n": int(len(d)), "events": int(d["y"].sum()), "note": "too few events"}
            continue
        # Relative likelihood standardised within author, so that "Qwen finds
        # Llama's text likely" is compared only with other Llama texts.
        d["z"] = d.groupby("author")["rel"].transform(lambda v: (v - v.mean()) / (v.std() or 1))
        obs = d.loc[d.y == 1, "z"].mean() - d.loc[d.y == 0, "z"].mean()
        # Permutation within prompt: which of a prompt's texts the judge
        # named as its own is shuffled; prompts stay the unit.
        rng = np.random.default_rng(RNG_SEED)
        groups = [g.index.to_numpy() for _, g in d.groupby("prompt_id")]
        y = d["y"].to_numpy()
        z = d["z"].to_numpy()
        sims = np.empty(N_PERM)
        for k in range(N_PERM):
            yp = y.copy()
            for idx in groups:
                yp[idx] = rng.permutation(y[idx])
            sims[k] = z[yp == 1].mean() - z[yp == 0].mean()
        out[j] = {"n": int(len(d)), "events": int(d["y"].sum()),
                  "z_diff": float(obs),
                  "perm_p": float((np.abs(sims) >= abs(obs) - 1e-12).mean())}
    return out


def quality_metrics(panel, df: pd.DataFrame | None, rng=None) -> dict | None:
    """Self-preference: how much more J rates its own text than its peers
    rate it, minus how much more J rates others' text than its peers rate
    that, so a generous rater is not counted as self-preferring. With a
    prompt bootstrap interval."""
    if df is None:
        return None
    rng = rng or np.random.default_rng(RNG_SEED)
    # (prompt, author, judge) -> rating, as a dense array.
    W = df.pivot_table(index=["prompt_id", "true_author"], columns="judge",
                       values="expected").dropna()
    names = [n for n in panel.names if n in W.columns]
    pids = W.index.get_level_values(0).unique()
    A = np.stack([W.xs(p, level=0).reindex(names)[names].to_numpy() for p in pids])
    # A[prompt, author, judge]
    out = {}
    k = len(names)

    def pref(X, j):
        oth = [i for i in range(k) if i != j]
        by_self = X[:, j, j].mean()
        by_peers = X[:, j, oth].mean()
        j_on_others = X[:, oth, j].mean()
        peers_on_others = np.mean([X[:, a, b] for a in oth for b in oth if b != a])
        return (by_self - by_peers) - (j_on_others - peers_on_others), by_self, by_peers

    for j, name in enumerate(names):
        sp, bs, bp = pref(A, j)
        boots = []
        for _ in range(2000):
            boots.append(pref(A[rng.integers(0, len(A), size=len(A))], j)[0])
        out[name] = {"self_rating_of_own": float(bs), "peer_rating_of_own": float(bp),
                     "self_preference": float(sp),
                     "self_preference_ci95": [float(np.percentile(boots, 2.5)),
                                              float(np.percentile(boots, 97.5))]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=sorted(PANELS))
    ap.add_argument("--tag", default="", help="e.g. api120 for the expanded frontier corpus")
    args = ap.parse_args()
    global TAG
    TAG = args.tag
    panel = PANELS[args.panel]
    rng = np.random.default_rng(RNG_SEED)
    summary: dict = {"panel": panel.key, "names": panel.names}

    frames = {}
    for cond in ("lineup", "single", "shuffle_lineup"):
        df = load(panel, cond)
        if df is None:
            continue
        a = attribution_rows(df, cond)
        frames[cond] = a
        summary[cond] = attribution_metrics(a, panel.names, rng)
        if a["genre"].nunique() > 1:
            summary[cond]["by_genre"] = {
                g: attribution_metrics(ag, panel.names, rng) for g, ag in a.groupby("genre")}
    for cond in ("binary", "shuffle_binary", "onpolicy", "onpolicy_resample"):
        df = load(panel, cond)
        if df is None:
            continue
        frames[cond] = df
        summary[cond] = yesno_metrics(df, panel.names, rng)
        if df["genre"].nunique() > 1:
            summary[cond]["by_genre"] = {
                g: yesno_metrics(dg, panel.names, rng) for g, dg in df.groupby("genre")}
    for cond, S in list(summary.items()):
        if not isinstance(S, dict) or "judges" not in S:
            continue
        for key, hkey in (("perm_p", "perm_p_holm"), ("auc_perm_p", "auc_p_holm"),
                          ("disc_perm_p", "disc_p_holm")):
            ps = {j: g[key] for j, g in S["judges"].items() if key in g}
            if ps:
                for j, v in holm(ps).items():
                    S["judges"][j][hkey] = v
    lk = likelihood_metrics(panel, rng)
    if lk:
        summary["likelihood"] = lk
        summary["familiarity"] = {c: familiarity(panel, frames.get(c), c)
                                  for c in ("lineup", "single", "binary") if c in frames}
    q = quality_metrics(panel, load(panel, "quality"), rng)
    if q:
        summary["quality"] = q

    out = panel.results_dir / f"summary{'_' + TAG if TAG else ''}.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({k: (v if k in ("panel",) else "...") for k, v in summary.items()}))


if __name__ == "__main__":
    main()
