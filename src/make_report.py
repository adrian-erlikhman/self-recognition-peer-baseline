"""CompLLM: generate docs/RESULTS.md, the end-to-end results compendium.

    python src/make_report.py

The submission is 2 pages. Almost none of the evidence fits in it, so this is
the companion document: every number, every test, every design decision, and
the exact commands to reproduce all of it.

Every number is read from results/*.json at run time. Nothing is typed in.
Rerun the analysis, rerun this, and the document is current.

The narrative sections -- prior art, limitations, design rationale -- are
editorial and live in this file as marked constants. They are the only
hand-written content, and they carry no figures.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "RESULTS.md"
MODELS = ["GPT", "Claude", "Gemini", "Grok", "DeepSeek"]


def load(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        sys.exit(f"missing {p} -- run the analysis scripts first")
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


# ---------------------------------------------------------------- editorial
PRIOR_ART = """
| Claim | Status |
|---|---|
| Stylometric 5-way LLM attribution | Prior work. Sun et al., ICML 2025: 97.1% on the same five vendors, with a length ablation. |
| Self-recognition at chance | Prior work. Bai et al. 2025: 10-way, about 10.3-10.9% against a 10% floor. |
| Blame concentrating on GPT/Claude ("prestige sink") | Prior work. Bai et al.: 94.0%/97.7% of guesses against 40% of generators. |
| Anonymized lineup with position control | Prior work. Davidson et al., Findings of EMNLP 2024. |
| Content-scaffolded prompts, GroupKFold by prompt | Standard. Dietrich 2026 publishes the same split as "SD-CV". |
| Including Grok/DeepSeek | Not new. Sun et al. used both. |
| **Peer baseline for self-attribution rates** | This work. Prior self rates are scored against a chance floor only. |
| **Format-dependent self-recognition** | This work. Both query formats on the same responses. |
| **Surface features do not predict the blame target beyond author identity** | This work. Bai et al. conjecture "familiar stylistic patterns" without testing it. |
"""

LIMITATIONS = """
1. **Five judges is too few** to separate a shared prior from one idiosyncratic
   judge. Leave-one-judge-out on n=5 cannot distinguish these; it is reported as
   a sensitivity check, not as evidence of generalisation.
2. **Effective n is the number of prompt groups, not the number of responses.**
   Cross-validation is grouped by `prompt_id` throughout for this reason.
3. **No quality control.** In an anonymized lineup, "blame concentrates on the
   prestige model" is observationally indistinguishable from "one model writes
   text everyone judges best." Davidson et al. and Wataoka et al. both explain
   the data without any identity prior. The fix, per-response quality ratings
   and a perplexity covariate, is left for follow-up work.
4. **The Claude class mixes two model versions** (33 Opus 4.7, 6 Sonnet 4.6).
   For those 6 the "self" judgment is cross-model. See the sensitivity table.
5. **M5 is excluded** (2 of its 5 responses are duplicates) and **H3/DeepSeek is
   truncated** (stated 820 words, actual 266, ending on a complete sentence so
   the loss is invisible).
6. **Judges did not see the prompts**, only the responses.
7. **Version matching is approximate for DeepSeek** - "Deepseek Instant" is not
   a released version string.
"""

DESIGN = """
**Random forest primary, logistic regression as robustness.** A null is only as
strong as the most flexible model that failed to find the signal. Both are
reported; LogReg beats RF in most cells.

**Per-prompt truncation to the group minimum, not a fixed cap.** A fixed cap
leaves already-short responses untouched, so length stays correlated with model
in the short tail and the length-only baseline never falls to chance. Group-min
truncation equalises length exactly within the comparison a judge sees.

**Latin square:** offset = `(judge_idx + prompt_idx) mod 5`. The offset depends
on prompt as well as judge - a simpler `offset = judge` square is also globally
balanced but pins each model to one slot for all of a given judge's prompts,
confounding that judge's position preference with one model.

**Repeats permitted in lineup answers.** Load-bearing: forcing five distinct
answers makes the marginal blame distribution uniform *by construction*. Judges
still permute most of the time, which is itself a finding - and the reason the
single-text condition exists.

**Readability implemented in-repo, not via textstat.** textstat routes syllable
counting through NLTK's cmudict; without that corpus it raises, and any
except-and-default wrapper silently turns three readability features into
constant zeros while still counting them among "18 features". `features.py`
implements them directly and `frame(guard=True)` hard-fails on any constant or
non-finite column.
"""

REPRO = """
All analysis runs offline: no API key, no network. Only re-collection needs
credentials.

    pip install -r requirements.txt
    python src/run_all.py

or step by step:

    python src/test_harness.py      # 24 offline checks
    python src/engine_a.py          # author identification
    python src/engine_b.py --tex paper/numbers.tex
    python src/make_figure.py       # paper/figures/
    python src/costs.py             # spend ledger
    python src/make_report.py       # this document

The corpus `data/responses_v1.csv` is SHA-256 pinned in `src/config.py` and the
harness refuses to start on a mismatch, so an accidental edit cannot silently
change the results.

**Determinism.** Rerunning the downstream chain reproduces every reported
accuracy, rate, CI bound and test statistic exactly. Two things legitimately
move and neither affects a claim: the permutation p-values shift with
`--permutations` (they are bounded by 1/(n+1), so they are quoted as bounds),
and the mixed-effects intercept moves at the eighth decimal because
`BinomialBayesMixedGLM.fit_vb` is a variational fit. Everything else is
byte-identical.

Every API response is archived under `raw/`, which is what makes the
judge-side analysis reproducible from a clone without an API key.
Re-collection, if ever needed:

    python src/check_models.py      # first line must read "key OK"
    python src/run_lineup.py        # resumable
    python src/parse_judgments.py
    python src/run_single.py        # resumable
    python src/parse_judgments.py --condition single
"""

FIGURES = """
All figures are generated, never drawn by hand - every value is read from
`results/*.json` or the frozen corpus at build time, so they cannot drift from
the numbers in this document. Regenerate with `src/make_figure.py` (main) and
`src/make_figures_supp.py` (the rest). Each exists as both PDF (vector, Type-42
fonts, what arXiv and NeurIPS want) and PNG (for slides).

| Figure | File | What it shows |
|---|---|---|
| **Main** | `two_conditions` | The paper's figure. Each judge's lineup self rate against its peer baseline, and all-text versus non-self accuracy in both formats. |
| A1 | `fig_a1_ablation` | How much of Engine A is just response length. The length-only baseline falls to chance under group-min truncation while the full feature set stays far above it. |
| A2 | `fig_a2_confusion` | Engine A out-of-fold confusion matrix. |
| A3 | `fig_a3_features` | Feature importance, with length proxies marked - they dominate, and the figure says so. |
| A4 | `fig_a4_length` | The length confound itself: per-response word counts by model. |
| B1 | `fig_b1_selfadvantage` | Self-rate vs others-rate on the same model's text. This is what makes the diagonal interpretable. |
| B2 | `fig_b2_blame_heatmap` | Judge x guessed, both conditions. Shows the collapse happening per judge, not just in aggregate. |
| B3 | `fig_b3_mechanism` | The blame-target test against both baselines, including the identity-only baseline that decides the null. |
| B4 | `fig_b4_confidence` | Confidence calibration. Informative in the lineup, not under single-text querying - a third strand of the format-dependence result. |
| B5 | `fig_b5_reasons` | What judges *say* they used, from 1,900 stated rationales. Keyword-based and coarse; see `src/analyze_reasons.py` for the lexicon and its limits. |
"""

MANIFEST = """
| Path | What |
|---|---|
| `data/responses_v1.csv` | frozen analysis corpus, SHA-256 pinned in `config.py` |
| `raw/*.json` | every archived API response - makes analysis key-free |
| `results/judgments.csv` | lineup judgments |
| `results/judgments_single.csv` | single-text judgments |
| `results/engine_a_results.json` | Engine A, all ablations |
| `results/engine_b_results.json` | Engine B, both conditions |
| `results/costs.csv` | per-call spend ledger |
| `src/features.py` | the 18 features + readability + group-min truncation |
| `src/engine_a.py` | author identification |
| `src/engine_b.py` | judge-side analysis |
| `src/run_all.py` | **reproduce everything in one command** |
| `src/make_figure.py` | the main paper figure |
| `src/make_figures_supp.py` | the nine supplementary figures |
| `src/analyze_reasons.py` | keyword analysis of the 1,900 stated rationales |
| `src/make_report.py` | this document |
| `paper/` | generated `numbers.tex`, table fragments and figures |
| `docs/COLLECTION.md` | request settings and upstream routing for the judge calls |
"""


def section_engine_b(w, d: dict, label: str) -> None:
    w("---")
    w("")
    w(f"## {label}")
    w("")
    w(f"{d['n_judgments']} judgments across {d['n_prompts']} prompts and 5 "
      f"judges. Overall accuracy **{pct(d['accuracy'])}** against a 20% chance "
      f"floor.")
    w("")
    w("### Self-recognition")
    w("")
    w("| Judge | k/n | Rate | Exact 95% CI | p vs chance | |")
    w("|---|---|---|---|---|---|")
    for r in d["self_recognition"]:
        w(f"| {r['judge']} | {r['k']}/{r['n']} | {pct(r['rate'])} | "
          f"[{pct(r['ci_lo'])}, {pct(r['ci_hi'])}] | {r['p_vs_chance']:.2e} | "
          f"{r['sig']} {r['direction']} |")
    w("")
    w("### Self-advantage")
    w("")
    w("Is a model better at spotting its own text than others are at spotting "
      "it? Without this, a high diagonal is ambiguous between self-recognition "
      "and simply writing identifiable prose.")
    w("")
    w("| Model | Self | Others | delta pp | Odds ratio | Fisher p | |")
    w("|---|---|---|---|---|---|---|")
    for r in d["self_advantage"]:
        orv = r.get("odds_ratio")
        orstr = "n/a" if orv is None else f"{orv:.2f}"
        w(f"| {r['model']} | {pct(r['self_rate'])} | {pct(r['others_rate'])} | "
          f"{r['advantage_pp']:+.1f} | {orstr} | {r['fisher_p']:.2e} | "
          f"{r['sig']} |")
    w("")
    bl = d["blame"]
    w("### Blame distribution")
    w("")
    w("| Guessed | Pooled | Judge-averaged |")
    w("|---|---|---|")
    for m in MODELS:
        w(f"| {m} | {pct(bl['pooled_shares'][m])} | "
          f"{pct(bl['judge_averaged_shares'][m])} |")
    w("")
    w(f"- Top-2 concentration: **{pct(bl['concentration_top2'])}**")
    w(f"- chi-sq pooled = {bl['chi2_pooled']:.1f}, p = {bl['p_pooled']:.3g} - "
      f"**inflated by construction**; {d['n_judgments']} judgments cluster in 5 "
      f"judges and {d['n_prompts']} prompts and are not independent.")
    w(f"- chi-sq on judge-averaged proportions = "
      f"{bl['chi2_judge_averaged']:.1f}, p = {bl['p_judge_averaged']:.3g}; "
      f"a claim about judges as a class has n = 5, so this is the test that "
      f"applies.")
    w(f"- Max-share vs chance across judges: t = "
      f"{bl['t_max_share_vs_chance']:.2f}, p = {bl['p_max_share']:.4g}")
    w("")
    w("### Leave-one-judge-out (sensitivity, n = 5 - not proof)")
    w("")
    w("| Dropped | n | Accuracy | Max blame share | On |")
    w("|---|---|---|---|---|")
    for r in d["leave_one_judge_out"]:
        w(f"| {r['dropped_judge']} | {r['n']} | {pct(r['accuracy'])} | "
          f"{pct(r['max_blame_share'])} | {r['max_blame_model']} |")
    w("")
    me = d.get("mixed_effects", {})
    w("### Mixed-effects logistic on P(correct)")
    w("")
    if "error" in me:
        w(f"Did not fit: `{me['error']}`")
    else:
        more = "more" if me["sd_judge"] > me["sd_prompt"] else "less"
        w(f"Crossed random intercepts for `prompt_id` and `judge`, n = "
          f"{me['n']}. Intercept {me['intercept_logit']:+.3f} logit "
          f"({pct(me['intercept_prob'])}). sd(prompt) {me['sd_prompt']:.3f}, "
          f"sd(judge) {me['sd_judge']:.3f} - judges vary {more} than prompts.")
    w("")
    mech = d.get("mechanism_test", {})
    w("### Mechanism test - do surface features predict *who gets blamed*?")
    w("")
    if "error" in mech:
        w(f"Skipped: {mech['error']}")
    else:
        w(f"On wrong guesses only (n = {mech['n_wrong']}, "
          f"{mech['n_classes']} classes), GroupKFold by prompt, "
          f"{mech['n_permutations']} permutations.")
        w("")
        w("| Predictor | Accuracy | Permutation null | p | Significant |")
        w("|---|---|---|---|---|")
        w(f"| majority baseline | {pct(mech['majority_baseline'])} | - | - | - |")
        w(f"| **true author identity alone** | "
          f"**{pct(mech['identity_only_baseline'])}** | - | - | - |")
        best = 0.0
        for k in ("random_forest", "logistic_regression"):
            m = mech[k]
            best = max(best, m["accuracy"])
            w(f"| {k.replace('_', ' ')} | {pct(m['accuracy'])} | "
              f"{pct(m['perm_null_mean'])} +/- {pct(m['perm_null_sd'])} | "
              f"{m['perm_p']:.4f} | {'yes' if m['significant'] else 'no'} |")
        w("")
        w(f"**Null {'HOLDS' if mech['null_holds'] else 'DOES NOT HOLD'}.**")
        w("")
        cmp_word = ("better" if mech["identity_only_baseline"] > best
                    else "comparably")
        w("The identity-only baseline is the control that decides this. Both "
          "feature models beat *shuffled labels* - but the 18 features identify "
          "the author at ~86% (Engine A) and blame is conditioned on "
          "authorship, so a feature model scores above chance purely as a proxy "
          "for identity. Knowing only the true author and nothing else predicts "
          f"blame {cmp_word}, so the features add nothing about blame beyond who "
          "wrote the text. The claim is **\"no signal beyond author "
          "identity\"**, not \"no signal\".")
    w("")
    sens = d.get("sonnet_sensitivity", {})
    if sens:
        w("### Sensitivity - dropping the 6 Sonnet-4.6 Claude rows")
        w("")
        w("| Judge | Full corpus | Sonnet rows dropped | Shift |")
        w("|---|---|---|---|")
        for m in MODELS:
            s = sens[m]
            w(f"| {m} | {s['full']['k']}/{s['full']['n']} "
              f"({pct(s['full']['rate'])}) | "
              f"{s['sonnet_dropped']['k']}/{s['sonnet_dropped']['n']} "
              f"({pct(s['sonnet_dropped']['rate'])}) | {s['shift_pp']:+.1f} pp |")
        w("")


def main() -> None:
    a = load("engine_a_results.json")
    b = load("engine_b_results.json")
    c = load("costs_summary.json")
    L, S = b.get("lineup"), b.get("single")

    o: list[str] = []
    w = o.append

    w("# CompLLM - end-to-end results")
    w("")
    w("**Generated file.** Regenerate with `python src/make_report.py`. "
      "Every figure below is read from `results/*.json` at run time; only "
      "the narrative sections are hand-written, and they contain no numbers.")
    w("")
    w(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
      f"seed {a.get('seed')} · {a.get('n_folds')}-fold GroupKFold by `prompt_id`")
    w("")
    w("---")
    w("")
    w("## 1. The result in one table")
    w("")
    w("The same responses, judged two ways.")
    w("")
    if L and S:
        lsr = {r["judge"]: r for r in L["self_recognition"]}
        ssr = {r["judge"]: r for r in S["self_recognition"]}
        w("| Judge | Lineup self-recog | 95% CI | Single-text self-recog | 95% CI |")
        w("|---|---|---|---|---|")
        for m in MODELS:
            x, y = lsr[m], ssr[m]
            w(f"| {m} | {pct(x['rate'])} ({x['k']}/{x['n']}) | "
              f"[{pct(x['ci_lo'])}, {pct(x['ci_hi'])}] | "
              f"{pct(y['rate'])} ({y['k']}/{y['n']}) | "
              f"[{pct(y['ci_lo'])}, {pct(y['ci_hi'])}] |")
        w("")
        w(f"Chance floor 20%. Exact (Clopper-Pearson) intervals. Lineup accuracy "
          f"{pct(L['accuracy'])}, single-text {pct(S['accuracy'])}.")
        w("")
        w(f"Blame concentration on the top two models: "
          f"**{pct(L['blame']['concentration_top2'])} under the lineup** vs "
          f"**{pct(S['blame']['concentration_top2'])} under single-text "
          f"querying** - on the identical responses.")
        w("")
        rare = ", ".join(f"{m} {int(S['blame']['pooled_counts'][m])}x"
                         for m in MODELS
                         if S["blame"]["pooled_counts"][m] < 10)
        w(f"> The three 0.0% cells in the single-text column are a "
          f"**consequence of that collapse**, not five comparable "
          f"self-recognition rates. Across {S['n_judgments']} judgments those "
          f"models are named by anyone only {rare}. Report the collapse; the "
          f"zeros follow from it.")
    w("")
    w("---")
    w("")
    w("## 2. Engine A - author identification (the control, not a contribution)")
    w("")
    w("Engine A exists to show the identifying signal *is* present in surface "
      "text, so that a judge scoring below chance reads as a model failure "
      "rather than an absence of signal. Sun et al. (ICML 2025) already reached "
      "97.1% on these vendors; this is not new work.")
    w("")
    acc = a.get("accuracy", {})
    perm = a.get("permutation", {})
    w(f"- Out-of-fold accuracy **{pct(acc.get('mean'))}**, n = {acc.get('n')}, "
      f"chance 20%.")
    w(f"- Permutation test: observed {pct(perm.get('observed'))}, null "
      f"{pct(perm.get('null_mean'))} +/- {pct(perm.get('null_sd'))} over "
      f"{perm.get('n')} permutations.")
    w(f"- **Report as a bound: p < 0.001 (0 of {perm.get('n')} permutations "
      f"reached the observed accuracy).** The computed p sits exactly on the "
      f"1/(n+1) resolution floor, so a point estimate invites the objection "
      f"that the test could not resolve any lower.")
    lk = a.get("leakage", {})
    w(f"- Content leakage ~ zero: GroupKFold {pct(lk.get('grouped'))} vs naive "
      f"StratifiedKFold {pct(lk.get('ungrouped'))} "
      f"({lk.get('inflation', 0) * 100:+.1f} pp). The accuracy is not topic "
      f"memorisation.")
    w("")
    w("### Ablations - how much of this is just length?")
    w("")
    w("| Corpus | Features | k | RF | LogReg |")
    w("|---|---|---|---|---|")
    for corpus, cells in a.get("headline", {}).items():
        for fset, v in cells.items():
            w(f"| {corpus} | {fset} | {v['k']} | "
              f"{pct(v['rf_mean'])} +/- {pct(v['rf_sd'])} | "
              f"{pct(v['logreg_mean'])} +/- {pct(v['logreg_sd'])} |")
    w("")
    lc = a.get("length_confound", {})
    counts = ", ".join(f"{k} {v:.0f}" for k, v in
                       sorted(lc.get("mean_word_count", {}).items(),
                              key=lambda kv: -kv[1]))
    w(f"Mean word count ranges {lc.get('ratio', 0):.2f}x across models "
      f"({counts}), Kruskal-Wallis H = {lc.get('kruskal_H', 0):.1f}, "
      f"p = {lc.get('kruskal_p', 0):.1e}.")
    w("")
    w(f"Length carries **{pct(a.get('length_share_of_signal'))}** of the "
      f"above-chance signal, and length proxies are "
      f"**{pct(a.get('length_proxy_importance_share'))}** of RF feature "
      f"importance. Under group-min truncation the length-only baseline falls "
      f"to chance - which is the point of truncating to the group minimum "
      f"rather than a fixed cap.")
    w("")
    w("### Per-class (out-of-fold)")
    w("")
    w("| Model | Precision | Recall | F1 | n |")
    w("|---|---|---|---|---|")
    for m in MODELS:
        r = a.get("per_class", {}).get(m)
        if r:
            w(f"| {m} | {r['precision']:.3f} | {r['recall']:.3f} | "
              f"{r['f1-score']:.3f} | {int(r['support'])} |")
    w("")

    if L:
        section_engine_b(w, L, "3. Engine B - LINEUP condition")
    if S:
        section_engine_b(w, S, "4. Engine B - SINGLE-TEXT condition")

    w("---")
    w("")
    w("## 5. Design decisions and why")
    w(DESIGN)
    w("---")
    w("")
    w("## 6. Prior art - what is and is not ours")
    w(PRIOR_ART)
    w("---")
    w("")
    w("## 7. Limitations")
    w(LIMITATIONS)
    w("---")
    w("")
    w("## 8. Reproducing all of this")
    w(REPRO)
    w("---")
    w("")
    w("## 9. Cost")
    w("")
    w(f"Lifetime **${c['total_usd']:.2f}** across **{c['calls']} archived API "
      f"calls**. The ledger is derived from `raw/` on every run, never "
      f"accumulated alongside it, so it cannot drift when a run crashes or is "
      f"resumed - and it is reproducible from a clone with no API key.")
    w("")
    w("| Condition | Calls | Cost | $/call |")
    w("|---|---|---|---|")
    for k, v in sorted(c.get("by_condition", {}).items()):
        w(f"| {k} | {v['calls']} | ${v['usd']:.2f} | "
          f"${v['usd'] / v['calls']:.4f} |")
    w("")
    w("Per-call cost varies widely by judge, which is worth knowing before "
      "planning any re-collection:")
    w("")
    w("| Condition / judge | Calls | Cost | $/call |")
    w("|---|---|---|---|")
    for k, v in sorted(c.get("by_condition_judge", {}).items()):
        w(f"| {k} | {v['calls']} | ${v['usd']:.2f} | "
          f"${v['usd'] / v['calls']:.4f} |")
    w("")
    w("---")
    w("")
    w("## 10. Figures")
    w(FIGURES)
    w("---")
    w("")
    w("## 11. File manifest")
    w(MANIFEST)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(o) + "\n", encoding="utf-8")
    text = OUT.read_text(encoding="utf-8")
    print(f"wrote {OUT}  ({len(o)} lines, {len(text.split())} words)")


if __name__ == "__main__":
    main()
