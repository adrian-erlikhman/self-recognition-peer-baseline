"""CompLLM: Engine A: author identification (the positive control).

Random forest is PRIMARY, logistic regression is the robustness check.
Rationale: this feature set has to serve as the positive control for Engine
B's null, and a null is only as strong as the most flexible model that failed
to find the signal. Using the more capable learner on both targets is what
makes "the features work on author, not on blame" a real claim rather than an
artifact of model choice.

    python src/engine_a.py                  # full analysis, 1000 permutations
    python src/engine_a.py --permutations 200   # quick pass
    python src/engine_a.py --no-permutation     # skip it entirely

Writes results/engine_a_results.json and results/engine_a_features.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EXCLUDE_PROMPTS, MODELS, RESPONSES_CSV, RESULTS_DIR  # noqa: E402
from features import (  # noqa: E402
    FEATURES,
    LENGTH_FREE,
    LENGTH_ONLY,
    frame,
    truncate_to_group_min,
    truncate_words,
)

SEED = 2026
N_FOLDS = 5
TRUNCATE_TO = 250


def rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500, random_state=SEED, n_jobs=-1, min_samples_leaf=1
    )


def logreg() -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()),
         ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=SEED))]
    )


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - m) / d, (c + m) / d)


def cv_score(X, y, groups, model, folds: int = N_FOLDS) -> tuple[float, float, np.ndarray]:
    cv = GroupKFold(n_splits=folds)
    s = cross_val_score(model, X, y, groups=groups, cv=cv, n_jobs=1)
    return float(s.mean()), float(s.std()), s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--no-permutation", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"seed": SEED, "n_folds": N_FOLDS, "truncate_to": TRUNCATE_TO}

    # ---------------------------------------------------------------- data
    df = pd.read_csv(RESPONSES_CSV)
    df["prompt_id"] = df["prompt_id"].astype(str)
    n_all = len(df)
    df = df[~df["prompt_id"].isin(EXCLUDE_PROMPTS)].reset_index(drop=True)

    print("=" * 74)
    print("ENGINE A - AUTHOR IDENTIFICATION (positive control)")
    print("=" * 74)
    print(f"rows loaded          : {n_all}")
    print(f"excluded prompts     : {EXCLUDE_PROMPTS}  -> {len(df)} rows")
    print(f"prompts              : {df['prompt_id'].nunique()}")
    print(f"classes              : {sorted(df['model'].unique())}  (chance {1/len(MODELS):.1%})")
    print(f"per class            : {df['model'].value_counts().to_dict()}")

    y = df["model"].to_numpy()
    groups = df["prompt_id"].to_numpy()

    print("\nextracting features (3 corpora) ...")
    F_full = frame(df["text"])
    F_trunc = frame([truncate_words(t, TRUNCATE_TO) for t in df["text"]])
    F_gmin = frame(truncate_to_group_min(df["text"], groups))
    F_full.insert(0, "prompt_id", df["prompt_id"].values)
    F_full.insert(1, "model", y)
    F_full.to_csv(RESULTS_DIR / "engine_a_features.csv", index=False)
    F_full = F_full[FEATURES]

    wc_by_model = df.assign(wc=F_full["word_count"]).groupby("model")["wc"]
    by_model = wc_by_model.mean().sort_values(ascending=False)
    ratio = by_model.max() / by_model.min()
    H, p_kw = stats.kruskal(*[g.to_numpy() for _, g in wc_by_model])
    print(f"\nmean word count by model: {by_model.round(1).to_dict()}")
    print(f"length ratio {ratio:.2f}x   Kruskal-Wallis H={H:.2f}, p={p_kw:.2e}")
    out["length_confound"] = {
        "mean_word_count": by_model.round(1).to_dict(),
        "ratio": float(ratio), "kruskal_H": float(H), "kruskal_p": float(p_kw),
    }

    tmeans = df.assign(wc=F_trunc["word_count"]).groupby("model")["wc"].mean()
    tratio = tmeans.max() / tmeans.min()
    print(f"after truncation to {TRUNCATE_TO} words: ratio {tratio:.2f}x")
    out["length_confound"]["ratio_truncated"] = float(tratio)

    # -------------------------------------------------------- headline grid
    conditions = [
        ("FULL", FEATURES),
        ("LENGTH-FREE", LENGTH_FREE),
        ("LENGTH-ONLY", LENGTH_ONLY),
    ]
    out["headline"] = {}
    for corpus_name, F in [
        ("UNTRUNCATED", F_full),
        (f"TRUNCATED-{TRUNCATE_TO}", F_trunc),
        ("TRUNCATED-TO-GROUP-MIN", F_gmin),
    ]:
        print(f"\n{'-' * 74}\n{corpus_name}   (GroupKFold by prompt_id, {N_FOLDS} folds)\n{'-' * 74}")
        print(f"  {'condition':<14} {'k':>3}  {'RandomForest (primary)':>26}  {'LogReg (robustness)':>22}")
        out["headline"][corpus_name] = {}
        for cond, cols in conditions:
            m_rf, s_rf, _ = cv_score(F[cols].to_numpy(), y, groups, rf())
            m_lr, s_lr, _ = cv_score(F[cols].to_numpy(), y, groups, logreg())
            print(f"  {cond:<14} {len(cols):>3}  {m_rf:>18.1%} +/- {s_rf:.1%}  "
                  f"{m_lr:>14.1%} +/- {s_lr:.1%}")
            out["headline"][corpus_name][cond] = {
                "k": len(cols), "rf_mean": m_rf, "rf_sd": s_rf,
                "logreg_mean": m_lr, "logreg_sd": s_lr,
            }

    gmeans = df.assign(wc=F_gmin["word_count"]).groupby("model")["wc"].mean()
    gratio = gmeans.max() / gmeans.min()
    print(f"after per-prompt truncation to group min: ratio {gratio:.2f}x")
    out["length_confound"]["ratio_group_min"] = float(gratio)

    lo_t = out["headline"]["TRUNCATED-TO-GROUP-MIN"]["LENGTH-ONLY"]["rf_mean"]
    print(f"\n  Control check: LENGTH-ONLY at group-min truncation = {lo_t:.1%} "
          f"(chance {1/len(MODELS):.1%}). {'PASS' if lo_t < 0.32 else 'INVESTIGATE'}")
    print("  That collapse is the control working, not a finding.")
    print(f"  The fixed {TRUNCATE_TO}-word cap leaves already-short responses untouched,")
    print("  so it under-neutralises length. Group-min is the honest control.")

    # --------------------------------------------- how much of it is length
    full = out["headline"]["UNTRUNCATED"]["FULL"]["rf_mean"]
    free = out["headline"]["UNTRUNCATED"]["LENGTH-FREE"]["rf_mean"]
    chance = 1 / len(MODELS)
    share = (full - free) / (full - chance)
    print(f"\n  Above-chance signal carried by length features: "
          f"{(full-free)*100:.1f} of {(full-chance)*100:.1f} points = {share:.0%}")
    out["length_share_of_signal"] = float(share)

    # ------------------------------------------------- leakage: grouped vs not
    print(f"\n{'-' * 74}\nCONTENT LEAKAGE: GroupKFold vs naive StratifiedKFold\n{'-' * 74}")
    Xf = F_full[FEATURES].to_numpy()
    # Same estimator, same folds, same columns as UNTRUNCATED/FULL above, so
    # reuse that score rather than paying for an identical refit.
    g_mean = out["headline"]["UNTRUNCATED"]["FULL"]["rf_mean"]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    u_mean = float(cross_val_score(rf(), Xf, y, cv=skf, n_jobs=1).mean())
    print(f"  GroupKFold by prompt_id : {g_mean:.1%}   <- honest")
    print(f"  StratifiedKFold (naive) : {u_mean:.1%}   <- leaks prompt content")
    print(f"  leakage inflation       : {(u_mean - g_mean)*100:+.1f} points")
    out["leakage"] = {"grouped": g_mean, "ungrouped": u_mean, "inflation": u_mean - g_mean}

    # ------------------------------------------------------ permutation test
    if not args.no_permutation:
        n_perm = args.permutations
        print(f"\n{'-' * 74}\nPERMUTATION TEST ({n_perm} permutations, labels shuffled within design)\n{'-' * 74}")
        print("  running ... this is the slow part")
        # Same estimator as the headline grid - otherwise the "observed"
        # score here disagrees with the number reported above.
        score, perm_scores, pval = permutation_test_score(
            RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=1),
            Xf, y, groups=groups, cv=GroupKFold(n_splits=N_FOLDS),
            n_permutations=n_perm, random_state=SEED, n_jobs=-1,
        )
        print(f"  observed {score:.1%}   null {perm_scores.mean():.1%} "
              f"+/- {perm_scores.std():.1%}   p = {pval:.4f}")
        print(f"  resolution floor at {n_perm} permutations: p >= {1/(n_perm+1):.4f}")
        out["permutation"] = {
            "n": n_perm, "observed": float(score), "null_mean": float(perm_scores.mean()),
            "null_sd": float(perm_scores.std()), "p": float(pval),
        }

    # ----------------------------------------------------- per-class detail
    print(f"\n{'-' * 74}\nPER-CLASS (RandomForest, FULL, out-of-fold predictions)\n{'-' * 74}")
    yhat = cross_val_predict(rf(), Xf, y, groups=groups,
                             cv=GroupKFold(n_splits=N_FOLDS), n_jobs=1)
    acc = float((yhat == y).mean())
    lo, hi = wilson(int((yhat == y).sum()), len(y))
    print(f"  accuracy {acc:.1%}   95% CI [{lo:.1%}, {hi:.1%}]   n={len(y)}")
    print(f"  effective n for generalisation is {df['prompt_id'].nunique()} prompt groups, "
          f"not {len(y)} responses.\n")
    rep = classification_report(y, yhat, labels=MODELS, output_dict=True, zero_division=0)
    print(classification_report(y, yhat, labels=MODELS, digits=3, zero_division=0))
    cm = confusion_matrix(y, yhat, labels=MODELS)
    print("confusion matrix (rows = true, cols = predicted)")
    print(pd.DataFrame(cm, index=MODELS, columns=MODELS).to_string())
    out["per_class"] = rep
    out["accuracy"] = {"mean": acc, "ci95": [lo, hi], "n": len(y),
                       "n_groups": int(df["prompt_id"].nunique())}
    out["confusion_matrix"] = {"labels": MODELS, "matrix": cm.tolist()}

    # ------------------------------------------------------------ importance
    print(f"\n{'-' * 74}\nFEATURE IMPORTANCE (RandomForest, fit on all data)\n{'-' * 74}")
    model = rf().fit(Xf, y)
    imp = (pd.Series(model.feature_importances_, index=FEATURES)
           .sort_values(ascending=False))
    for name, v in imp.items():
        flag = "  <- length proxy" if name not in LENGTH_FREE else ""
        print(f"  {name:<24} {v:.4f}{flag}")
    out["feature_importance"] = imp.round(5).to_dict()
    out["length_proxy_importance_share"] = float(
        imp[[f for f in FEATURES if f not in LENGTH_FREE]].sum()
    )
    print(f"\n  length proxies account for "
          f"{out['length_proxy_importance_share']:.1%} of total importance")

    path = RESULTS_DIR / "engine_a_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    print(f"wrote {RESULTS_DIR / 'engine_a_features.csv'}  "
          f"(feature table - Engine B reuses this)")
    print("=" * 74)


if __name__ == "__main__":
    main()
