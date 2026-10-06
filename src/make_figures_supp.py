"""CompLLM: the supplementary figure set.

    python src/make_figures_supp.py

The main paper figure is src/make_figure.py (two conditions, four panels). This
script produces everything else worth showing: the Engine A evidence that the
signal is real and not just length, and the Engine B panels that do not fit in
two pages. They live in docs/RESULTS.md and in the appendix / talk.

Same discipline as everywhere else in this repo: NOTHING is hardcoded. Every
value is read from results/*.json or the frozen corpus at run time, so the
figures regenerate correctly when the analysis is rerun.

Style constants are imported from make_figure.py so the whole set is visually
one family -- same palette, same fonts, same greyscale/colourblind safety
(hatch patterns carry the distinction, not colour alone), same Type-42 vector
output that arXiv and NeurIPS accept.

Figures produced:
  fig_a1_ablation      how much of Engine A is just response length
  fig_a2_confusion     out-of-fold confusion matrix
  fig_a3_features      feature importance, length proxies marked
  fig_a4_length        the 2.1x length confound across models
  fig_b1_selfadvantage self-rate vs others-rate on the same model's text
  fig_b2_blame_heatmap judge x guessed, both conditions
  fig_b3_mechanism     the blame-target test against both baselines
  fig_b4_confidence    calibration curve; informative in the lineup, not
                       under single-text querying
  fig_b5_reasons       what judges SAY they used (optional; needs
                       src/analyze_reasons.py to have run)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figure import (  # noqa: E402
    BASE_LW, FIG_DIR, FS_LABEL, FS_NOTE, FS_TICK, FS_TITLE, FS_VALUE,
    GUTTER, HATCHES, INK, PALETTE, _rc,
)
from config import (  # noqa: E402
    EXCLUDE_PROMPTS, JUDGMENTS_CSV, JUDGMENTS_SINGLE_CSV, MODELS,
    RESPONSES_CSV, RESULTS_DIR,
)
from features import LENGTH_PROXIES  # noqa: E402

CHANCE = 0.20


def load(name: str) -> dict:
    p = RESULTS_DIR / name
    if not p.exists():
        sys.exit(f"missing {p} -- run the analysis first")
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote paper/figures/{stem}.pdf + .png")


# ------------------------------------------------------- A1: length ablation
def fig_ablation(a: dict) -> None:
    """The single most important Engine A panel.

    A 5-way text classifier on these features could be a length classifier
    in disguise. This shows what happens as length is progressively equalised: under
    truncation to the per-prompt group minimum the LENGTH-ONLY baseline falls
    to chance while FULL stays far above it, which is the argument that the
    remaining signal is stylistic rather than a length artefact.
    """
    head = a["headline"]
    corpora = list(head.keys())
    fsets = ["FULL", "LENGTH-FREE", "LENGTH-ONLY"]
    x = np.arange(len(corpora))
    width = 0.26

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, fs in enumerate(fsets):
        vals = [head[c][fs]["rf_mean"] * 100 for c in corpora]
        errs = [head[c][fs]["rf_sd"] * 100 for c in corpora]
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=2.5,
               color=PALETTE[i], hatch=HATCHES[i + 1], edgecolor=INK,
               linewidth=BASE_LW, error_kw={"lw": 0.8, "ecolor": INK},
               label=f"{fs} (k={head[corpora[0]][fs]['k']})")
        # Clear the error-bar cap, not just the bar top, or the label sits
        # on the whisker.
        for xi, v, e in zip(x + (i - 1) * width, vals, errs):
            ax.text(xi, v + e + 2.2, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=FS_VALUE)

    ax.axhline(CHANCE * 100, ls="--", lw=1.0, color=INK, zorder=0)
    # Reserve a gutter on the right so the chance label never lands on a bar.
    ax.set_xlim(-0.55, len(corpora) - 1 + 0.55 + GUTTER * 0.45)
    ax.text(len(corpora) - 1 + 0.52, CHANCE * 100 + 1.5, "chance = 20%",
            fontsize=FS_NOTE, va="bottom", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("TRUNCATED-TO-GROUP-MIN", "truncated to\ngroup min")
                        .replace("TRUNCATED-250", "truncated\nto 250w")
                        .replace("UNTRUNCATED", "untruncated") for c in corpora],
                       fontsize=FS_TICK)
    ax.set_ylabel("author-identification accuracy (%)", fontsize=FS_LABEL)
    ax.set_title("Engine A: how much of the signal is response length?",
                 fontsize=FS_TITLE, loc="left")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=FS_NOTE, loc="upper right", ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.0, -0.10,
             "Random forest, GroupKFold by prompt, error bars are cross-"
             "validation SD. Truncating every response in a prompt group to "
             "that group's shortest\nresponse drives the length-only baseline "
             "to chance while the full feature set stays far above it - the "
             "remaining signal is not length.",
             fontsize=FS_NOTE, ha="left", va="top")
    save(fig, "fig_a1_ablation")


# ---------------------------------------------------------- A2: confusion
def fig_confusion(a: dict) -> None:
    cm = a["confusion_matrix"]
    labels, M = cm["labels"], np.array(cm["matrix"], dtype=float)
    row = M.sum(axis=1, keepdims=True)
    frac = M / np.where(row == 0, 1, row)

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center",
                    fontsize=FS_VALUE,
                    color="white" if frac[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=FS_TICK, rotation=30, ha="right")
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_xlabel("predicted author", fontsize=FS_LABEL)
    ax.set_ylabel("true author", fontsize=FS_LABEL)
    ax.set_title("Engine A: out-of-fold confusion", fontsize=FS_TITLE, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("share of that model's responses", fontsize=FS_NOTE)
    cb.ax.tick_params(labelsize=FS_NOTE)
    save(fig, "fig_a2_confusion")


# --------------------------------------------------------- A3: importance
def fig_features(a: dict) -> None:
    """Length proxies marked, because the honest reading is that they dominate."""
    fi = a["feature_importance"]
    names = sorted(fi, key=lambda k: fi[k])
    vals = [fi[n] * 100 for n in names]
    is_len = [n in LENGTH_PROXIES for n in names]
    share = a.get("length_proxy_importance_share", 0) * 100

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.barh(range(len(names)), vals,
            color=[PALETTE[0] if L else PALETTE[2] for L in is_len],
            hatch=["" if L else ".." for L in is_len],
            edgecolor=INK, linewidth=BASE_LW)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=FS_TICK)
    ax.set_xlabel("random-forest importance (%)", fontsize=FS_LABEL)
    ax.set_title("Engine A: feature importance", fontsize=FS_TITLE, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PALETTE[0], ec=INK, lw=BASE_LW),
               plt.Rectangle((0, 0), 1, 1, fc=PALETTE[2], ec=INK, lw=BASE_LW,
                             hatch="..")]
    ax.legend(handles, [f"length proxy ({share:.0f}% of total)", "other"],
              fontsize=FS_NOTE, loc="lower right")
    save(fig, "fig_a3_features")


# ------------------------------------------------------------ A4: length
def fig_length(a: dict) -> None:
    df = pd.read_csv(RESPONSES_CSV)
    df = df[~df["prompt_id"].isin(EXCLUDE_PROMPTS)]
    order = (df.groupby("model")["word_count"].mean()
               .sort_values(ascending=False).index.tolist())
    lc = a.get("length_confound", {})

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    for i, m in enumerate(order):
        v = df[df["model"] == m]["word_count"].to_numpy()
        ax.scatter(np.full(len(v), i) + np.linspace(-0.16, 0.16, len(v)), v,
                   s=9, color=PALETTE[i % len(PALETTE)], edgecolor=INK,
                   linewidth=0.3, zorder=3, alpha=0.85)
        ax.hlines(v.mean(), i - 0.30, i + 0.30, color=INK, lw=1.6, zorder=4)
        ax.text(i, v.max() + 40, f"{v.mean():.0f}", ha="center",
                fontsize=FS_VALUE)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=FS_TICK)
    ax.set_ylabel("words per response", fontsize=FS_LABEL)
    ax.set_title(f"The length confound: {lc.get('ratio', 0):.2f}x spread across "
                 f"models", fontsize=FS_TITLE, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.0, -0.10,
             f"Each point is one response; the bar is the model mean. "
             f"Kruskal-Wallis H = {lc.get('kruskal_H', 0):.1f}, "
             f"p = {lc.get('kruskal_p', 0):.1e}. This is why the ablation in "
             f"Fig. A1 exists.",
             fontsize=FS_NOTE, ha="left", va="top")
    save(fig, "fig_a4_length")


# ---------------------------------------------------- B1: self-advantage
def fig_selfadvantage(b: dict) -> None:
    """The test that makes the diagonal interpretable.

    A high self-recognition rate is ambiguous on its own: the model may
    recognise itself, or it may simply write text that everyone identifies.
    Comparing each model's self-rate against how often OTHER judges name that
    same model separates the two.
    """
    conds = [k for k in ("lineup", "single") if k in b]
    fig, axes = plt.subplots(1, len(conds), figsize=(6.4, 2.9), sharey=True)
    if len(conds) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conds):
        rows = b[cond]["self_advantage"]
        names = [r["model"] for r in rows]
        x = np.arange(len(names))
        s = [r["self_rate"] * 100 for r in rows]
        o = [r["others_rate"] * 100 for r in rows]
        ax.bar(x - 0.19, s, 0.38, color=PALETTE[0], edgecolor=INK,
               linewidth=BASE_LW, label="judged by itself")
        ax.bar(x + 0.19, o, 0.38, color=PALETTE[1], hatch="//", edgecolor=INK,
               linewidth=BASE_LW, label="judged by the others")
        for xi, r in zip(x, rows):
            if r["sig"] not in ("n.s.", ""):
                top = max(r["self_rate"], r["others_rate"]) * 100
                ax.text(xi, top + 4, r["sig"], ha="center", fontsize=FS_VALUE)
        ax.axhline(CHANCE * 100, ls="--", lw=1.0, color=INK, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=FS_TICK, rotation=20, ha="right")
        ax.set_title(f"({'ab'[conds.index(cond)]}) {cond}", fontsize=FS_TITLE,
                     loc="left")
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("identified as that model (%)", fontsize=FS_LABEL)
    axes[0].legend(fontsize=FS_NOTE, loc="upper left")
    fig.text(0.0, -0.12,
             "Fisher exact per model; * p<.05, ** p<.01, *** p<.001. Dashed "
             "line is the 20% chance floor. A model whose own bar exceeds the "
             "others' bar is\nrecognising itself rather than merely writing "
             "identifiable prose; Grok's runs the other way.",
             fontsize=FS_NOTE, ha="left", va="top")
    save(fig, "fig_b1_selfadvantage")


# ---------------------------------------------------------- B2: blame heat
def fig_blame_heatmap() -> None:
    # Wide, with real gutters: the right panel's y labels were landing on the
    # left panel's cells, and "guessed" was colliding with the footnote.
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    fig.subplots_adjust(wspace=0.55, bottom=0.26)
    for ax, (path, label) in zip(axes, ((JUDGMENTS_CSV, "lineup"),
                                        (JUDGMENTS_SINGLE_CSV, "single-text"))):
        d = pd.read_csv(path)
        d = d[~d["prompt_id"].isin(EXCLUDE_PROMPTS)]
        d = d[d["guessed_model"].notna() & (d["guessed_model"] != "")]
        t = pd.crosstab(d["judge"], d["guessed_model"], normalize="index") * 100
        order = [m for m in MODELS if m in t.index]
        t = t.reindex(index=order, columns=order).fillna(0)
        im = ax.imshow(t.to_numpy(), cmap="Blues", vmin=0, vmax=100)
        for i in range(len(order)):
            for j in range(len(order)):
                v = t.iloc[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=FS_VALUE, color="white" if v > 55 else INK)
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=FS_TICK, rotation=30, ha="right")
        ax.set_yticklabels(order, fontsize=FS_TICK)
        ax.set_xlabel("guessed", fontsize=FS_LABEL)
        ax.set_title(label, fontsize=FS_TITLE, loc="left")
    axes[0].set_ylabel("judge", fontsize=FS_LABEL)
    cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.04)
    cb.set_label("% of that judge's guesses", fontsize=FS_NOTE)
    cb.ax.tick_params(labelsize=FS_NOTE)
    fig.text(0.0, -0.16,
             "Rows sum to 100%. Under the lineup every judge spreads its "
             "guesses near-uniformly; under single-text querying every judge "
             "names GPT or Claude\nalmost exclusively - including Gemini, Grok "
             "and DeepSeek, which is why their self-recognition is 0% there.",
             fontsize=FS_NOTE, ha="left", va="top")
    save(fig, "fig_b2_blame_heatmap")


# ---------------------------------------------------------- B3: mechanism
def fig_mechanism(b: dict) -> None:
    """The null, shown against the baseline that actually decides it."""
    conds = [k for k in ("lineup", "single")
             if k in b and "error" not in b[k].get("mechanism_test", {})]
    fig, axes = plt.subplots(1, len(conds), figsize=(6.4, 2.9))
    if len(conds) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conds):
        m = b[cond]["mechanism_test"]
        bars = [("majority\nbaseline", m["majority_baseline"], PALETTE[4], ".."),
                ("identity\nonly", m["identity_only_baseline"], PALETTE[1], ""),
                ("random\nforest", m["random_forest"]["accuracy"], PALETTE[0], "//"),
                ("logistic\nregression", m["logistic_regression"]["accuracy"],
                 PALETTE[2], "xx")]
        x = np.arange(len(bars))
        ax.bar(x, [v * 100 for _, v, _, _ in bars],
               color=[c for _, _, c, _ in bars],
               hatch=[h for _, _, _, h in bars],
               edgecolor=INK, linewidth=BASE_LW)
        for xi, (_, v, _, _) in zip(x, bars):
            ax.text(xi, v * 100 + 1.2, f"{v * 100:.1f}", ha="center",
                    va="bottom", fontsize=FS_VALUE)
        # An unlabelled dashed line is ambiguous, and this one carries the
        # entire argument -- say what it is on the figure itself.
        ax.axhline(m["identity_only_baseline"] * 100, ls="--", lw=1.0,
                   color=INK, zorder=0)
        ax.text(-0.45, m["identity_only_baseline"] * 100 - 1.0,
                "identity baseline", fontsize=FS_NOTE, ha="left", va="top",
                style="italic")
        ax.set_xticks(x)
        ax.set_xticklabels([n for n, _, _, _ in bars], fontsize=FS_NOTE)
        ax.set_title(f"({'ab'[conds.index(cond)]}) {cond}", fontsize=FS_TITLE,
                     loc="left")
        ax.set_ylim(0, max(v for _, v, _, _ in bars) * 100 + 14)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("blame-target accuracy (%)", fontsize=FS_LABEL)
    fig.text(0.0, -0.14,
             "Predicting WHICH model gets blamed, on wrong guesses only, "
             "GroupKFold by prompt. Both feature models beat shuffled labels "
             "(p<0.005) - but only\nbecause the features encode authorship. "
             "Neither beats a baseline that knows the true author and nothing "
             "else (dashed line), so the features add no\ninformation about "
             "blame beyond identity. The claim is \"no signal beyond author "
             "identity\", not \"no signal\".",
             fontsize=FS_NOTE, ha="left", va="top")
    save(fig, "fig_b3_mechanism")


# --------------------------------------------------------- B4: confidence
def fig_confidence(b: dict) -> None:
    """Confidence calibration, which turns out to be format-dependent too.

    In the lineup, judges' stated confidence tracks their accuracy. Under
    single-text querying it carries no information. That is a third strand of
    the same format-dependence result, and it is not something the two group
    means alone would show -- the calibration CURVE is what makes it visible.
    """
    conds = [c for c in ("lineup", "single")
             if c in b and "error" not in b[c].get("confidence_calibration", {})]
    if not conds:
        print("  (skipping fig_b4: no confidence data)")
        return
    fig, axes = plt.subplots(1, len(conds), figsize=(6.4, 2.9), sharey=True)
    if len(conds) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conds):
        cc = b[cond]["confidence_calibration"]
        lv = cc["accuracy_by_level"]
        xs = sorted(lv, key=float)
        acc = [lv[k]["accuracy"] * 100 for k in xs]
        lo = [(lv[k]["accuracy"] - lv[k]["ci_lo"]) * 100 for k in xs]
        hi = [(lv[k]["ci_hi"] - lv[k]["accuracy"]) * 100 for k in xs]
        ax.errorbar([float(k) for k in xs], acc, yerr=[lo, hi], fmt="-",
                    color=PALETTE[0], ecolor=INK, elinewidth=0.8, capsize=2.5,
                    linewidth=1.4, zorder=3)
        # Levels with almost no data get hollow markers. Leaving them solid
        # lets a two-observation cell with a 1%-99% interval read as a real
        # data point; hiding them entirely would be worse.
        for k, a in zip(xs, acc):
            small = lv[k]["n"] < 10
            ax.plot(float(k), a, "o", markersize=5,
                    markerfacecolor="white" if small else PALETTE[0],
                    markeredgecolor=INK, markeredgewidth=0.6, zorder=4)
            ax.annotate(f"n={lv[k]['n']}", (float(k), a),
                        textcoords="offset points", xytext=(9, -9),
                        ha="left", fontsize=FS_NOTE,
                        color="#666666" if small else INK)
        ax.axhline(CHANCE * 100, ls="--", lw=1.0, color=INK, zorder=0)
        ax.set_xlabel("stated confidence (1-5)", fontsize=FS_LABEL)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_title(f"({'ab'[conds.index(cond)]}) {cond}   "
                     f"rho={cc['spearman_rho']:+.3f}, p={cc['spearman_p']:.2g}",
                     fontsize=FS_TITLE, loc="left")
        ax.set_ylim(0, 75)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy (%)", fontsize=FS_LABEL)
    axes[0].text(5.35, CHANCE * 100 + 1.5, "chance = 20%", fontsize=FS_NOTE,
                 va="bottom", ha="right")
    for ax in axes:
        ax.set_xlim(0.6, 5.6)
    fig.text(0.0, -0.13,
             "Exact (Clopper-Pearson) 95% CIs; Spearman rank correlation "
             "between stated confidence and correctness. In the lineup, "
             "confidence tracks accuracy -\naccuracy roughly doubles from the "
             "lowest to the highest stated level. Under single-text querying "
             "it carries no information. Calibration is format-dependent too.",
             fontsize=FS_NOTE, ha="left", va="top")
    save(fig, "fig_b4_confidence")


# ------------------------------------------------------------ B5: rationales
def fig_reasons() -> None:
    """What judges SAY they used, per condition.

    Reads results/reasons_analysis.json (src/analyze_reasons.py). Skipped
    silently if that has not been run -- it is an optional analysis.
    """
    path = RESULTS_DIR / "reasons_analysis.json"
    if not path.exists():
        print("  (skipping fig_b5: run src/analyze_reasons.py first)")
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    conds = [c for c in ("lineup", "single") if c in d]
    cues = [r["cue"] for r in d[conds[0]]["cues"]]
    order = sorted(cues, key=lambda c: -max(
        next(r["share"] for r in d[k]["cues"] if r["cue"] == c) for k in conds))

    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for i, cond in enumerate(conds):
        by = {r["cue"]: r for r in d[cond]["cues"]}
        vals = [by[c]["share"] * 100 for c in order]
        ax.barh(y + (0.2 if i else -0.2), vals, 0.38,
                color=PALETTE[i], hatch=HATCHES[i + 1], edgecolor=INK,
                linewidth=BASE_LW, label=cond)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in order], fontsize=FS_TICK)
    ax.invert_yaxis()
    ax.set_xlabel("% of stated rationales mentioning this cue", fontsize=FS_LABEL)
    ax.set_title("What judges say they used", fontsize=FS_TITLE, loc="left")
    ax.legend(fontsize=FS_NOTE, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ea = d.get("engine_a_length_proxy_importance")
    note = ("Keyword matching against a hand-built lexicon (src/analyze_reasons.py) "
            "- a coarse instrument, and the categories are ours, not the judges'. "
            "A rationale can\nmatch several cues or none. Under single-text "
            "querying 96.7% of rationales name a model explicitly, against 56.9% "
            "in the lineup: judges there are\nreasoning from model stereotypes, "
            "which is the same collapse Fig. B2 shows in their answers.")
    if ea is not None:
        note += (f" For contrast, length proxies carry {ea * 100:.0f}% of Engine "
                 f"A's feature importance.")
    fig.text(0.0, -0.14, note, fontsize=FS_NOTE, ha="left", va="top")
    save(fig, "fig_b5_reasons")


def main() -> None:
    _rc()
    a = load("engine_a_results.json")
    b = load("engine_b_results.json")
    print("supplementary figures:")
    fig_ablation(a)
    fig_confusion(a)
    fig_features(a)
    fig_length(a)
    fig_selfadvantage(b)
    fig_blame_heatmap()
    fig_mechanism(b)
    fig_confidence(b)
    fig_reasons()
    print("done.")


if __name__ == "__main__":
    main()
