# CompLLM - end-to-end results

**Generated file.** Regenerate with `python src/make_report.py`. Every figure below is read from `results/*.json` at run time; only the narrative sections are hand-written, and they contain no numbers.

Generated 2026-09-10 07:34 UTC · seed 2026 · 5-fold GroupKFold by `prompt_id`

---

## 1. The result in one table

The same responses, judged two ways.

| Judge | Lineup self-recog | 95% CI | Single-text self-recog | 95% CI |
|---|---|---|---|---|
| GPT | 57.9% (22/38) | [40.8%, 73.7%] | 94.7% (36/38) | [82.3%, 99.4%] |
| Claude | 86.8% (33/38) | [71.9%, 95.6%] | 86.8% (33/38) | [71.9%, 95.6%] |
| Gemini | 18.4% (7/38) | [7.7%, 34.3%] | 0.0% (0/38) | [0.0%, 9.3%] |
| Grok | 5.3% (2/38) | [0.6%, 17.7%] | 0.0% (0/38) | [0.0%, 9.3%] |
| DeepSeek | 21.1% (8/38) | [9.6%, 37.3%] | 0.0% (0/38) | [0.0%, 9.3%] |

Chance floor 20%. Exact (Clopper-Pearson) intervals. Lineup accuracy 33.7%, single-text 30.6%.

Blame concentration on the top two models: **43.8% under the lineup** vs **99.3% under single-text querying** - on the identical responses.

> The three 0.0% cells in the single-text column are a **consequence of that collapse**, not five comparable self-recognition rates. Across 950 judgments those models are named by anyone only Gemini 3x, Grok 1x, DeepSeek 3x. Report the collapse; the zeros follow from it.

---

## 2. Engine A - author identification (the control, not a contribution)

Engine A exists to show the identifying signal *is* present in surface text, so that a judge scoring below chance reads as a model failure rather than an absence of signal. Sun et al. (ICML 2025) already reached 97.1% on these vendors; this is not new work.

- Out-of-fold accuracy **86.3%**, n = 190, chance 20%.
- Permutation test: observed 85.8%, null 20.2% +/- 3.6% over 1000 permutations.
- **Report as a bound: p < 0.001 (0 of 1000 permutations reached the observed accuracy).** The computed p sits exactly on the 1/(n+1) resolution floor, so a point estimate invites the objection that the test could not resolve any lower.
- Content leakage ~ zero: GroupKFold 85.8% vs naive StratifiedKFold 85.3% (-0.5 pp). The accuracy is not topic memorisation.

### Ablations - how much of this is just length?

| Corpus | Features | k | RF | LogReg |
|---|---|---|---|---|
| UNTRUNCATED | FULL | 18 | 85.8% +/- 9.2% | 87.0% +/- 6.3% |
| UNTRUNCATED | LENGTH-FREE | 9 | 63.9% +/- 9.6% | 71.7% +/- 7.4% |
| UNTRUNCATED | LENGTH-ONLY | 2 | 60.4% +/- 5.4% | 62.9% +/- 5.7% |
| TRUNCATED-250 | FULL | 18 | 71.9% +/- 6.7% | 75.1% +/- 5.4% |
| TRUNCATED-250 | LENGTH-FREE | 9 | 54.1% +/- 9.6% | 65.5% +/- 8.7% |
| TRUNCATED-250 | LENGTH-ONLY | 2 | 35.9% +/- 1.0% | 34.6% +/- 1.9% |
| TRUNCATED-TO-GROUP-MIN | FULL | 18 | 67.0% +/- 8.7% | 74.8% +/- 8.1% |
| TRUNCATED-TO-GROUP-MIN | LENGTH-FREE | 9 | 60.2% +/- 8.9% | 66.6% +/- 9.4% |
| TRUNCATED-TO-GROUP-MIN | LENGTH-ONLY | 2 | 22.1% +/- 1.9% | 25.2% +/- 1.5% |

Mean word count ranges 2.12x across models (DeepSeek 689, Claude 664, GPT 446, Gemini 347, Grok 324), Kruskal-Wallis H = 102.3, p = 3.2e-21.

Length carries **33.2%** of the above-chance signal, and length proxies are **63.1%** of RF feature importance. Under group-min truncation the length-only baseline falls to chance - which is the point of truncating to the group minimum rather than a fixed cap.

### Per-class (out-of-fold)

| Model | Precision | Recall | F1 | n |
|---|---|---|---|---|
| GPT | 0.842 | 0.842 | 0.842 | 38 |
| Claude | 0.879 | 0.763 | 0.817 | 38 |
| Gemini | 0.805 | 0.868 | 0.835 | 38 |
| Grok | 0.923 | 0.947 | 0.935 | 38 |
| DeepSeek | 0.872 | 0.895 | 0.883 | 38 |

---

## 3. Engine B - LINEUP condition

950 judgments across 38 prompts and 5 judges. Overall accuracy **33.7%** against a 20% chance floor.

### Self-recognition

| Judge | k/n | Rate | Exact 95% CI | p vs chance | |
|---|---|---|---|---|---|
| GPT | 22/38 | 57.9% | [40.8%, 73.7%] | 3.17e-07 | *** above |
| Claude | 33/38 | 86.8% | [71.9%, 95.6%] | 1.47e-18 | *** above |
| Gemini | 7/38 | 18.4% | [7.7%, 34.3%] | 1.00e+00 | n.s. below |
| Grok | 2/38 | 5.3% | [0.6%, 17.7%] | 2.33e-02 | * below |
| DeepSeek | 8/38 | 21.1% | [9.6%, 37.3%] | 8.40e-01 | n.s. above |

### Self-advantage

Is a model better at spotting its own text than others are at spotting it? Without this, a high diagonal is ambiguous between self-recognition and simply writing identifiable prose.

| Model | Self | Others | delta pp | Odds ratio | Fisher p | |
|---|---|---|---|---|---|---|
| GPT | 57.9% | 30.3% | +27.6 | 3.17 | 2.27e-03 | ** |
| Claude | 86.8% | 53.9% | +32.9 | 5.63 | 1.59e-04 | *** |
| Gemini | 18.4% | 28.3% | -9.9 | 0.57 | 3.03e-01 | n.s. |
| Grok | 5.3% | 28.9% | -23.7 | 0.14 | 1.33e-03 | ** |
| DeepSeek | 21.1% | 21.7% | -0.7 | 0.96 | 1.00e+00 | n.s. |

### Blame distribution

| Guessed | Pooled | Judge-averaged |
|---|---|---|
| GPT | 21.5% | 21.5% |
| Claude | 22.3% | 22.3% |
| Gemini | 18.8% | 18.8% |
| Grok | 17.8% | 17.8% |
| DeepSeek | 19.6% | 19.6% |

- Top-2 concentration: **43.8%**
- chi-sq pooled = 6.6, p = 0.157 - **inflated by construction**; 950 judgments cluster in 5 judges and 38 prompts and are not independent.
- chi-sq on judge-averaged proportions = 1.3, p = 0.857; a claim about judges as a class has n = 5, so this is the test that applies.
- Max-share vs chance across judges: t = 2.56, p = 0.06293

### Leave-one-judge-out (sensitivity, n = 5 - not proof)

| Dropped | n | Accuracy | Max blame share | On |
|---|---|---|---|---|
| GPT | 760 | 31.3% | 22.8% | Claude |
| Claude | 760 | 32.6% | 21.7% | Claude |
| Gemini | 760 | 33.0% | 22.9% | Claude |
| Grok | 760 | 35.7% | 22.1% | Claude |
| DeepSeek | 760 | 35.8% | 22.1% | Claude |

### Mixed-effects logistic on P(correct)

Crossed random intercepts for `prompt_id` and `judge`, n = 950. Intercept -0.699 logit (33.2%). sd(prompt) 0.286, sd(judge) 0.379 - judges vary more than prompts.

### Mechanism test - do surface features predict *who gets blamed*?

On wrong guesses only (n = 630, 5 classes), GroupKFold by prompt, 200 permutations.

| Predictor | Accuracy | Permutation null | p | Significant |
|---|---|---|---|---|
| majority baseline | 23.0% | - | - | - |
| **true author identity alone** | **38.9%** | - | - | - |
| random forest | 29.7% | 21.1% +/- 1.8% | 0.0050 | yes |
| logistic regression | 31.9% | 20.9% +/- 1.7% | 0.0050 | yes |

**Null HOLDS.**

The identity-only baseline is the control that decides this. Both feature models beat *shuffled labels* - but the 18 features identify the author at ~86% (Engine A) and blame is conditioned on authorship, so a feature model scores above chance purely as a proxy for identity. Knowing only the true author and nothing else predicts blame better, so the features add nothing about blame beyond who wrote the text. The claim is **"no signal beyond author identity"**, not "no signal".

### Sensitivity - dropping the 6 Sonnet-4.6 Claude rows

| Judge | Full corpus | Sonnet rows dropped | Shift |
|---|---|---|---|
| GPT | 22/38 (57.9%) | 22/38 (57.9%) | +0.0 pp |
| Claude | 33/38 (86.8%) | 27/32 (84.4%) | -2.5 pp |
| Gemini | 7/38 (18.4%) | 7/38 (18.4%) | +0.0 pp |
| Grok | 2/38 (5.3%) | 2/38 (5.3%) | +0.0 pp |
| DeepSeek | 8/38 (21.1%) | 8/38 (21.1%) | +0.0 pp |

---

## 4. Engine B - SINGLE-TEXT condition

950 judgments across 38 prompts and 5 judges. Overall accuracy **30.6%** against a 20% chance floor.

### Self-recognition

| Judge | k/n | Rate | Exact 95% CI | p vs chance | |
|---|---|---|---|---|---|
| GPT | 36/38 | 94.7% | [82.3%, 99.4%] | 3.13e-23 | *** above |
| Claude | 33/38 | 86.8% | [71.9%, 95.6%] | 1.47e-18 | *** above |
| Gemini | 0/38 | 0.0% | [0.0%, 9.3%] | 3.44e-04 | *** below |
| Grok | 0/38 | 0.0% | [0.0%, 9.3%] | 3.44e-04 | *** below |
| DeepSeek | 0/38 | 0.0% | [0.0%, 9.3%] | 3.44e-04 | *** below |

### Self-advantage

Is a model better at spotting its own text than others are at spotting it? Without this, a high diagonal is ambiguous between self-recognition and simply writing identifiable prose.

| Model | Self | Others | delta pp | Odds ratio | Fisher p | |
|---|---|---|---|---|---|---|
| GPT | 94.7% | 61.2% | +33.6 | 11.42 | 2.62e-05 | *** |
| Claude | 86.8% | 82.2% | +4.6 | 1.43 | 6.31e-01 | n.s. |
| Gemini | 0.0% | 1.3% | -1.3 | 0.00 | 1.00e+00 | n.s. |
| Grok | 0.0% | 0.7% | -0.7 | 0.00 | 1.00e+00 | n.s. |
| DeepSeek | 0.0% | 0.7% | -0.7 | 0.00 | 1.00e+00 | n.s. |

### Blame distribution

| Guessed | Pooled | Judge-averaged |
|---|---|---|
| GPT | 45.9% | 45.9% |
| Claude | 53.4% | 53.4% |
| Gemini | 0.3% | 0.3% |
| Grok | 0.1% | 0.1% |
| DeepSeek | 0.3% | 0.3% |

- Top-2 concentration: **99.3%**
- chi-sq pooled = 1403.5, p = 1.21e-302 - **inflated by construction**; 950 judgments cluster in 5 judges and 38 prompts and are not independent.
- chi-sq on judge-averaged proportions = 280.7, p = 1.58e-59; a claim about judges as a class has n = 5, so this is the test that applies.
- Max-share vs chance across judges: t = 10.74, p = 0.0004263

### Leave-one-judge-out (sensitivity, n = 5 - not proof)

| Dropped | n | Accuracy | Max blame share | On |
|---|---|---|---|---|
| GPT | 760 | 28.7% | 56.8% | Claude |
| Claude | 760 | 28.6% | 60.7% | Claude |
| Gemini | 760 | 31.7% | 53.0% | GPT |
| Grok | 760 | 31.6% | 54.7% | GPT |
| DeepSeek | 760 | 32.6% | 58.8% | Claude |

### Mixed-effects logistic on P(correct)

Crossed random intercepts for `prompt_id` and `judge`, n = 950. Intercept -0.831 logit (30.3%). sd(prompt) 0.123, sd(judge) 0.371 - judges vary more than prompts.

### Mechanism test - do surface features predict *who gets blamed*?

On wrong guesses only (n = 659, 4 classes), GroupKFold by prompt, 200 permutations.

| Predictor | Accuracy | Permutation null | p | Significant |
|---|---|---|---|---|
| majority baseline | 53.0% | - | - | - |
| **true author identity alone** | **64.9%** | - | - | - |
| random forest | 63.0% | 52.3% +/- 2.2% | 0.0050 | yes |
| logistic regression | 61.0% | 51.4% +/- 2.2% | 0.0050 | yes |

**Null HOLDS.**

The identity-only baseline is the control that decides this. Both feature models beat *shuffled labels* - but the 18 features identify the author at ~86% (Engine A) and blame is conditioned on authorship, so a feature model scores above chance purely as a proxy for identity. Knowing only the true author and nothing else predicts blame better, so the features add nothing about blame beyond who wrote the text. The claim is **"no signal beyond author identity"**, not "no signal".

### Sensitivity - dropping the 6 Sonnet-4.6 Claude rows

| Judge | Full corpus | Sonnet rows dropped | Shift |
|---|---|---|---|
| GPT | 36/38 (94.7%) | 36/38 (94.7%) | +0.0 pp |
| Claude | 33/38 (86.8%) | 27/32 (84.4%) | -2.5 pp |
| Gemini | 0/38 (0.0%) | 0/38 (0.0%) | +0.0 pp |
| Grok | 0/38 (0.0%) | 0/38 (0.0%) | +0.0 pp |
| DeepSeek | 0/38 (0.0%) | 0/38 (0.0%) | +0.0 pp |

---

## 5. Design decisions and why

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

---

## 6. Prior art - what is and is not ours

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

---

## 7. Limitations

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

---

## 8. Reproducing all of this

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

---

## 9. Cost

Lifetime **$10.67** across **1140 archived API calls**. The ledger is derived from `raw/` on every run, never accumulated alongside it, so it cannot drift when a run crashes or is resumed - and it is reproducible from a clone with no API key.

| Condition | Calls | Cost | $/call |
|---|---|---|---|
| lineup | 190 | $5.06 | $0.0266 |
| single | 950 | $5.61 | $0.0059 |

Per-call cost varies widely by judge, which is worth knowing before planning any re-collection:

| Condition / judge | Calls | Cost | $/call |
|---|---|---|---|
| lineup/Claude | 38 | $1.34 | $0.0353 |
| lineup/DeepSeek | 38 | $0.05 | $0.0012 |
| lineup/GPT | 38 | $2.05 | $0.0540 |
| lineup/Gemini | 38 | $1.39 | $0.0366 |
| lineup/Grok | 38 | $0.23 | $0.0060 |
| single/Claude | 190 | $1.48 | $0.0078 |
| single/DeepSeek | 190 | $0.05 | $0.0003 |
| single/GPT | 190 | $2.19 | $0.0115 |
| single/Gemini | 190 | $1.47 | $0.0078 |
| single/Grok | 190 | $0.42 | $0.0022 |

---

## 10. Figures

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

---

## 11. File manifest

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

