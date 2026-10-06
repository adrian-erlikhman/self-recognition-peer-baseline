# Statistics for the TACL revision (C2, C3, C6, C7, C8)

**Generated file.** `python src/stats_revision.py` writes this and `results/stats_revision.json`; do not edit by hand. Seed 2026; 20,000 permutations; 10,000 prompt-bootstrap resamples; runtime 4 s. Percentages are percent, differences are percentage points (pp). All tests two-sided unless marked one-sided.

Design: 38 prompts x 5 authors = 190 responses; every response is judged once by each of the 5 judges, in each condition. For judge J: self rate = J on its 38 responses; peer baseline = the other four judges on the same 38 (152 judgments); self-advantage = self - peer; false alarm = J naming itself on the 152 responses it did not write; non-self accuracy = J's accuracy on those 152.

## C6. Clustered inference for self-advantage

**Why the old test is wrong.** Fisher's exact test treats the 152 peer judgments as independent Bernoulli draws. They come four to a response, and a response that is easy for one peer is easy for the others (see the ICC section), so the peer rate varies more than binomial and Fisher's p is too small.

**(a) Randomization test, and why the labels are exchangeable.** Each of J's 38 responses is judged by all five judges, one of whom is J. H0 (sharp form): on any given response, the five judgments are exchangeable with respect to which judge wrote the text -- the author's own judgment is no more likely to be correct than a peer's. Under H0, relabelling a uniformly chosen one of the five as "self" leaves the joint distribution unchanged. Relabelling is done independently per response; since J wrote exactly one response per prompt, that is also per prompt, so prompts (and lineup sessions, which never span prompts) are the independent units. Holding each response's number of correct judgments fixed conditions away how identifiable that particular response is and keeps its four peer judgments together, which is the clustering Fisher ignored. The statistic, self - peer, is monotone in the number of "self" hits, a sum of independent Bernoulli(T_p/5); so the permutation distribution is Poisson-binomial and the p-value is exact. The 20,000-draw Monte Carlo p is reported as a check. Caveat: the sharp null also assumes the judges are equally skilled; a judge that is simply a better attributor overall would show a positive "self-advantage" here. C8 removes that.

**(b)** Prompt-level cluster bootstrap: resample the 38 prompts with replacement, recompute everything, percentile 95% CI.  **(c)** GEE logistic, correct ~ is_self on the 190 judgments of J's text, binomial, exchangeable working correlation within prompt; the robust (sandwich) p and the Mancl-DeRouen bias-reduced p (38 clusters is few).  Holm is applied across the five judges within each condition (family 1), to the randomization p.

### lineup

| Judge | self | peer [cluster CI] | self-adv pp [cluster CI] | Fisher p (paper) | randomization p exact (MC) | GEE OR, p robust (bias-red.) | Holm p (rand.) | weaker than paper? |
|---|---|---|---|---|---|---|---|---|
| GPT | 57.9 | 30.3 [23.0, 38.2] | +27.6 [+10.5, +44.7] | 0.002 | 0.002 (0.002) | 3.17, 0.002 (0.002) | 0.008 | no |
| Claude | 86.8 | 53.9 [46.1, 61.8] | +32.9 [+21.1, +44.1] | 0.00016 | 0.0003 (0.00055) | 5.63, 0.00025 (0.00036) | 0.002 | yes |
| Gemini | 18.4 | 28.3 [21.7, 34.9] | -9.9 [-23.0, +4.6] | 0.303 | 0.309 (0.308) | 0.57, 0.214 (0.227) | 0.619 | yes |
| Grok | 5.3 | 28.9 [21.7, 36.8] | -23.7 [-32.9, -13.2] | 0.001 | 0.002 (0.002) | 0.14, 0.007 (0.008) | 0.008 | yes |
| DeepSeek | 21.1 | 21.7 [15.1, 28.9] | -0.7 [-15.1, +15.1] | 1.000 | 1.000 (1.000) | 0.96, 0.932 (0.934) | 1.000 | no |

- **GPT**: +27.6 pp; exact randomization p = 0.002 vs Fisher 0.002 (not weaker); Holm 0.008, verdict unchanged.
- **Claude**: +32.9 pp; exact randomization p = 0.0003 vs Fisher 0.00016 (weaker); Holm 0.002, verdict unchanged.
- **Gemini**: -9.9 pp; exact randomization p = 0.309 vs Fisher 0.303 (weaker); Holm 0.619, verdict unchanged.
- **Grok**: -23.7 pp; exact randomization p = 0.002 vs Fisher 0.001 (weaker); Holm 0.008, verdict unchanged.
- **DeepSeek**: -0.7 pp; exact randomization p = 1.000 vs Fisher 1.000 (not weaker); Holm 1.000, verdict unchanged.

**(d) Crossed random effects** (all 950 judgments; correct ~ author + per-judge self term, random intercepts for prompt and judge; variational Bayes, N(0, 2^2) prior on fixed effects). Because the judge intercept absorbs general skill, this is closer to C8 than to (a)-(c).
Posterior SD of prompt intercepts 0.36, judge intercepts 0.35; VB and MAP agree within 0.5 logit.

| Judge | self log-OR (post. SD) | OR | approx p | MAP log-OR | note |
|---|---|---|---|---|---|
| GPT | +0.87 (0.33) | 2.38 | 0.009 | +0.90 |  |
| Claude | +1.81 (0.46) | 6.09 | 0.0001 | +1.67 |  |
| Gemini | -0.90 (0.41) | 0.41 | 0.029 | -0.76 |  |
| Grok | -1.82 (0.64) | 0.16 | 0.004 | -1.67 |  |
| DeepSeek | +0.32 (0.40) | 1.38 | 0.420 | +0.25 |  |

### single

| Judge | self | peer [cluster CI] | self-adv pp [cluster CI] | Fisher p (paper) | randomization p exact (MC) | GEE OR, p robust (bias-red.) | Holm p (rand.) | weaker than paper? |
|---|---|---|---|---|---|---|---|---|
| GPT | 94.7 | 61.2 [53.3, 69.1] | +33.6 [+25.0, +42.1] | 2.6e-05 | 4.1e-05 (0.0002) | 11.42, 0.00039 (0.00056) | 0.00021 | yes |
| Claude | 86.8 | 82.2 [78.9, 86.2] | +4.6 [-7.2, +14.5] | 0.631 | 0.651 (0.655) | 1.43, 0.466 (0.478) | 1.000 | yes |
| Gemini | 0.0 | 1.3 [0.0, 3.3] | -1.3 [-3.3, +0.0] | 1.000 | 1.000 (1.000) | not estimable | 1.000 | no |
| Grok | 0.0 | 0.7 [0.0, 2.0] | -0.7 [-2.0, +0.0] | 1.000 | 1.000 (1.000) | not estimable | 1.000 | no |
| DeepSeek | 0.0 | 0.7 [0.0, 2.0] | -0.7 [-2.0, +0.0] | 1.000 | 1.000 (1.000) | not estimable | 1.000 | no |

- **GPT**: +33.6 pp; exact randomization p = 4.1e-05 vs Fisher 2.6e-05 (weaker); Holm 0.00021, verdict unchanged.
- **Claude**: +4.6 pp; exact randomization p = 0.651 vs Fisher 0.631 (weaker); Holm 1.000, verdict unchanged.
- **Gemini**: -1.3 pp; exact randomization p = 1.000 vs Fisher 1.000 (not weaker); Holm 1.000, verdict unchanged.
- **Grok**: -0.7 pp; exact randomization p = 1.000 vs Fisher 1.000 (not weaker); Holm 1.000, verdict unchanged.
- **DeepSeek**: -0.7 pp; exact randomization p = 1.000 vs Fisher 1.000 (not weaker); Holm 1.000, verdict unchanged.

**(d) Crossed random effects** (all 950 judgments; correct ~ author + per-judge self term, random intercepts for prompt and judge; variational Bayes, N(0, 2^2) prior on fixed effects). Because the judge intercept absorbs general skill, this is closer to C8 than to (a)-(c).
Posterior SD of prompt intercepts 0.28, judge intercepts 1.49; VB and MAP agree within 0.5 logit.

| Judge | self log-OR (post. SD) | OR | approx p | MAP log-OR | note |
|---|---|---|---|---|---|
| GPT | +1.40 (0.66) | 4.04 | 0.033 | +1.24 |  |
| Claude | -1.82 (0.50) | 0.16 | 0.0002 | -1.77 |  |
| Gemini | -0.80 (1.51) | 0.45 | 0.595 | -0.45 | 0 or 38 self hits: estimate is the prior, not the data |
| Grok | -0.70 (1.55) | 0.50 | 0.654 | -0.36 | 0 or 38 self hits: estimate is the prior, not the data |
| DeepSeek | -0.55 (1.62) | 0.58 | 0.734 | -0.25 | 0 or 38 self hits: estimate is the prior, not the data |

## C7. Intervals on every rate, and Holm families

Self rates: 38 judgments, one per prompt, so no clustering; exact Clopper-Pearson and Wilson. False alarms, peer baselines and non-self accuracy: 152 judgments, four per prompt, so a prompt-cluster bootstrap CI (naive Clopper-Pearson in the JSON for comparison). Family 1 (self-advantage, 5 per condition) is in C6; family 2 (self rate vs 20%, 5 per condition) and family 3 (non-self vs 20%, all 10 cells) are below.

### lineup

| Judge | self k/n | self CP 95% | self Wilson 95% | self vs 20% p (Holm) | false alarm [cluster CI] | peer [cluster CI] | non-self [cluster CI] |
|---|---|---|---|---|---|---|---|
| GPT | 22/38 | [40.8, 73.7] | [42.2, 72.1] | 3.2e-07 (1.3e-06) | 10.5 [6.6, 14.5] | 30.3 [23.0, 38.2] | 39.5 [32.2, 46.7] |
| Claude | 33/38 | [71.9, 95.6] | [72.7, 94.2] | 1.5e-18 (7.3e-18) | 9.2 [5.3, 13.2] | 53.9 [46.1, 61.8] | 25.7 [18.4, 32.9] |
| Gemini | 7/38 | [7.7, 34.3] | [9.2, 33.4] | 1.000 (1.000) | 20.4 [17.1, 23.0] | 28.3 [21.7, 34.9] | 40.8 [30.9, 51.3] |
| Grok | 2/38 | [0.6, 17.7] | [1.5, 17.3] | 0.023 (0.070) | 20.4 [17.1, 23.0] | 28.9 [21.7, 36.8] | 30.9 [22.4, 39.5] |
| DeepSeek | 8/38 | [9.6, 37.3] | [11.1, 36.3] | 0.840 (1.000) | 17.8 [13.8, 21.1] | 21.7 [15.1, 28.9] | 26.3 [17.1, 36.2] |

- Family 2 (lineup): 2 of 5 self rates differ from 20% after Holm. Cells that lose significance under Holm: Grok (0.023 raw -> 0.070 Holm).

### single

| Judge | self k/n | self CP 95% | self Wilson 95% | self vs 20% p (Holm) | false alarm [cluster CI] | peer [cluster CI] | non-self [cluster CI] |
|---|---|---|---|---|---|---|---|
| GPT | 36/38 | [82.3, 99.4] | [82.7, 98.5] | 3.1e-23 (1.6e-22) | 52.0 [45.4, 58.6] | 61.2 [53.3, 69.1] | 24.3 [23.0, 25.0] |
| Claude | 33/38 | [71.9, 95.6] | [72.7, 94.2] | 1.5e-18 (5.9e-18) | 8.6 [4.6, 13.2] | 82.2 [78.9, 86.2] | 27.0 [25.0, 29.6] |
| Gemini | 0/38 | [0.0, 9.3] | [0.0, 9.2] | 0.00034 (0.001) | 0.0 [0.0, 0.0] | 1.3 [0.0, 3.3] | 32.9 [29.6, 36.8] |
| Grok | 0/38 | [0.0, 9.3] | [0.0, 9.2] | 0.00034 (0.001) | 0.0 [0.0, 0.0] | 0.7 [0.0, 2.0] | 33.6 [29.6, 37.5] |
| DeepSeek | 0/38 | [0.0, 9.3] | [0.0, 9.2] | 0.00034 (0.001) | 0.0 [0.0, 0.0] | 0.7 [0.0, 2.0] | 28.3 [23.7, 32.9] |

- Family 2 (single): 5 of 5 self rates differ from 20% after Holm. Holm changes no verdict.

## C2. Does non-self accuracy clear the 20% floor in all ten cells?

Per cell: exact binomial on the judge's 152 non-self judgments, and two prompt-clustered versions (one-sample t on the 38 per-prompt non-self accuracies, which have equal size 4; and an intercept-only GEE with offset logit(0.2)). Holm across the ten cells (family 3). A cell clears the floor if it is above 20% and p < .05.

A fourth test asks whether the judge beats its OWN guessing habits rather than 20%: within each prompt, J's four labels on the four responses it did not write are shuffled among those responses (exact null by convolution over prompts; in the lineup this keeps each session's label multiset). Its mean is the "shuffle floor". This matters because 20% is not the floor for a judge that rarely names itself: always naming one of the other four scores 25% on non-self text.

| Cell | non-self [cluster CI] | exact 2-sided (Holm) | exact 1-sided (Holm) | cluster t 2-sided (Holm) | cluster t 1-sided (Holm) | GEE 2-sided (Holm) | shuffle floor | shuffle 1-sided (Holm) |
|---|---|---|---|---|---|---|---|---|
| GPT lineup | 39.5 [32.2, 46.7] | 3.7e-08 (3.3e-07) | 2.8e-08 (2.5e-07) | 7.1e-06 (4.2e-05) | 3.5e-06 (2.1e-05) | 4.7e-10 (2.8e-09) | 22.4 | 2.3e-05 (0.00016) |
| Claude lineup | 25.7 [18.4, 32.9] | 0.085 (0.170) | 0.053 (0.107) | 0.148 (0.296) | 0.074 (0.148) | 0.103 (0.207) | 22.7 | 0.239 (0.239) |
| Gemini lineup | 40.8 [30.9, 51.3] | 5.1e-09 (5.1e-08) | 3.8e-09 (3.8e-08) | 0.00029 (0.001) | 0.00015 (0.00073) | 1.8e-06 (9.2e-06) | 19.9 | 6e-08 (5.4e-07) |
| Grok lineup | 30.9 [22.4, 39.5] | 0.002 (0.009) | 0.00094 (0.006) | 0.015 (0.044) | 0.007 (0.022) | 0.003 (0.009) | 19.9 | 0.002 (0.008) |
| DeepSeek lineup | 26.3 [17.1, 36.2] | 0.054 (0.167) | 0.036 (0.107) | 0.221 (0.296) | 0.110 (0.148) | 0.167 (0.207) | 20.6 | 0.064 (0.193) |
| GPT single | 24.3 [23.0, 25.0] | 0.187 (0.187) | 0.110 (0.110) | 9.7e-08 (7.8e-07) | 4.9e-08 (3.9e-07) | 8.2e-13 (6.6e-12) | 12.0 | 1.3e-12 (1.3e-11) |
| Claude single | 27.0 [25.0, 29.6] | 0.042 (0.167) | 0.023 (0.093) | 2.5e-07 (1.8e-06) | 1.3e-07 (8.8e-07) | 2.1e-12 (1.4e-11) | 22.9 | 0.003 (0.011) |
| Gemini single | 32.9 [29.6, 36.8] | 0.00022 (0.002) | 0.00013 (0.0009) | 6.1e-08 (5.5e-07) | 3.1e-08 (2.8e-07) | 3.2e-15 (2.8e-14) | 25.0 | 4.7e-05 (0.00028) |
| Grok single | 33.6 [29.6, 37.5] | 9.6e-05 (0.00077) | 6.3e-05 (0.0005) | 3.3e-08 (3.3e-07) | 1.6e-08 (1.6e-07) | 3.8e-16 (3.8e-15) | 25.0 | 1.8e-07 (1.4e-06) |
| DeepSeek single | 28.3 [23.7, 32.9] | 0.015 (0.073) | 0.009 (0.045) | 0.001 (0.004) | 0.00055 (0.002) | 6.2e-05 (0.00025) | 25.0 | 0.107 (0.215) |

| Test | clear raw | clear Holm | fail raw | fail Holm |
|---|---|---|---|---|
| p_exact_two_sided | 7/10 | 5/10 | Claude lineup, DeepSeek lineup, GPT single | Claude lineup, DeepSeek lineup, GPT single, Claude single, DeepSeek single |
| p_exact_one_sided | 8/10 | 6/10 | Claude lineup, GPT single | Claude lineup, DeepSeek lineup, GPT single, Claude single |
| p_cluster_t_two_sided | 8/10 | 8/10 | Claude lineup, DeepSeek lineup | Claude lineup, DeepSeek lineup |
| p_cluster_t_one_sided | 8/10 | 8/10 | Claude lineup, DeepSeek lineup | Claude lineup, DeepSeek lineup |
| p_cluster_gee_two_sided | 8/10 | 8/10 | Claude lineup, DeepSeek lineup | Claude lineup, DeepSeek lineup |
| p_shuffle_one_sided | 7/10 | 7/10 | Claude lineup, DeepSeek lineup, DeepSeek single | Claude lineup, DeepSeek lineup, DeepSeek single |
| p_shuffle_two_sided | 7/10 | 7/10 | Claude lineup, DeepSeek lineup, DeepSeek single | Claude lineup, DeepSeek lineup, DeepSeek single |

- **Answer.** The paper's "all ten" is wrong. Exact binomial, one-sided: 8/10 clear raw, 6/10 after Holm (fail: Claude lineup, DeepSeek lineup, GPT single, Claude single). Two-sided: 7/10 raw, 5/10 Holm. Prompt-clustered t: 8/10 one-sided, 8/10 two-sided raw; 8/10 and 8/10 after Holm. The plan's p = 0.053 (Claude lineup) and p = 0.11 (GPT single) are the ONE-SIDED exact binomial values (0.053 and 0.110); two-sided they are 0.085 and 0.187. So "eight of ten" holds only for the one-sided exact test before correction, and which two cells fail depends on the test: the clustered tests fail Claude lineup and DeepSeek lineup, while GPT single passes them easily (its per-prompt non-self accuracy barely varies: it is right on Claude's text and almost nowhere else). Against each judge's own guessing habits (shuffle test) 7/10 clear (fail: Claude lineup, DeepSeek lineup, DeepSeek single). Robust statement: non-self accuracy is above chance in most cells, but not demonstrably for Claude lineup or DeepSeek lineup (both fail the two-sided exact test and every prompt-clustered test).

## C3. Are the judges without self-advantage the most accurate on others' text?

"full" compares each judge's own 152 non-self judgments (different text sets), with a prompt-cluster bootstrap CI and p; Fisher on 152 vs 152 is the naive reference. "common" compares the two judges on the same 114 responses by the other three authors, paired within prompt, with a 20,000-draw sign-flip test that swaps the two judges' labels for a whole prompt at a time; Holm over the 10 pairs within a condition.

### lineup  (ranking: Gemini > GPT > Grok > DeepSeek > Claude)

| A vs B | non-self A / B | diff pp [cluster CI] | boot p (Holm) | Fisher naive p | common-text A / B | common diff pp | sign-flip p (Holm) |
|---|---|---|---|---|---|---|---|
| GPT vs Claude | 39.5 / 25.7 | +13.8 [+3.9, +23.0] | 0.006 (0.060) | 0.014 | 24.6 / 23.7 | +0.9 | 1.000 (1.000) |
| GPT vs Gemini | 39.5 / 40.8 | -1.3 [-13.8, +11.2] | 0.844 (1.000) | 0.907 | 40.4 / 43.9 | -3.5 | 0.704 (1.000) |
| GPT vs Grok | 39.5 / 30.9 | +8.6 [-2.0, +19.7] | 0.132 (0.722) | 0.149 | 45.6 / 29.8 | +15.8 | 0.020 (0.179) |
| GPT vs DeepSeek | 39.5 / 26.3 | +13.2 [+0.7, +25.7] | 0.046 (0.366) | 0.020 | 47.4 / 27.2 | +20.2 | 0.016 (0.164) |
| Claude vs Gemini | 25.7 / 40.8 | -15.1 [-27.0, -3.3] | 0.009 (0.083) | 0.007 | 28.9 / 33.3 | -4.4 | 0.590 (1.000) |
| Claude vs Grok | 25.7 / 30.9 | -5.3 [-17.1, +6.6] | 0.411 (1.000) | 0.373 | 19.3 / 29.8 | -10.5 | 0.144 (0.865) |
| Claude vs DeepSeek | 25.7 / 26.3 | -0.7 [-13.8, +12.5] | 0.938 (1.000) | 1.000 | 30.7 / 23.7 | +7.0 | 0.426 (1.000) |
| Gemini vs Grok | 40.8 / 30.9 | +9.9 [-2.0, +22.4] | 0.120 (0.722) | 0.094 | 43.9 / 30.7 | +13.2 | 0.067 (0.472) |
| Gemini vs DeepSeek | 40.8 / 26.3 | +14.5 [-0.7, +28.9] | 0.062 (0.433) | 0.011 | 42.1 / 25.4 | +16.7 | 0.041 (0.326) |
| Grok vs DeepSeek | 30.9 / 26.3 | +4.6 [-7.9, +16.4] | 0.474 (1.000) | 0.447 | 33.3 / 28.9 | +4.4 | 0.649 (1.000) |

- Group contrast, mean non-self of Gemini/Grok/DeepSeek minus GPT/Claude: +0.1 pp [-8.4, +8.6], bootstrap p = 0.974.
- Top vs runner-up: Gemini vs GPT.

### single  (ranking: Grok > Gemini > DeepSeek > Claude > GPT)

| A vs B | non-self A / B | diff pp [cluster CI] | boot p (Holm) | Fisher naive p | common-text A / B | common diff pp | sign-flip p (Holm) |
|---|---|---|---|---|---|---|---|
| GPT vs Claude | 24.3 / 27.0 | -2.6 [-5.3, -0.7] | 0.021 (0.124) | 0.694 | 0.0 / 3.5 | -3.5 | 0.120 (0.719) |
| GPT vs Gemini | 24.3 / 32.9 | -8.6 [-12.5, -4.6] | 0.0002 (0.002) | 0.128 | 32.5 / 33.3 | -0.9 | 1.000 (1.000) |
| GPT vs Grok | 24.3 / 33.6 | -9.2 [-13.2, -5.3] | 0.0002 (0.002) | 0.100 | 32.5 / 33.3 | -0.9 | 1.000 (1.000) |
| GPT vs DeepSeek | 24.3 / 28.3 | -3.9 [-9.2, +0.7] | 0.132 (0.530) | 0.515 | 32.5 / 10.5 | +21.9 | 5e-05 (0.0005) |
| Claude vs Gemini | 27.0 / 32.9 | -5.9 [-9.9, -2.0] | 0.005 (0.034) | 0.316 | 34.2 / 10.5 | +23.7 | 5e-05 (0.0005) |
| Claude vs Grok | 27.0 / 33.6 | -6.6 [-11.2, -2.0] | 0.004 (0.032) | 0.261 | 35.1 / 11.4 | +23.7 | 5e-05 (0.0005) |
| Claude vs DeepSeek | 27.0 / 28.3 | -1.3 [-6.6, +3.9] | 0.665 (1.000) | 0.898 | 35.1 / 27.2 | +7.9 | 0.031 (0.219) |
| Gemini vs Grok | 32.9 / 33.6 | -0.7 [-3.3, +2.0] | 0.826 (1.000) | 1.000 | 43.9 / 44.7 | -0.9 | 1.000 (1.000) |
| Gemini vs DeepSeek | 32.9 / 28.3 | +4.6 [-1.3, +10.5] | 0.153 (0.530) | 0.455 | 43.9 / 37.7 | +6.1 | 0.193 (0.771) |
| Grok vs DeepSeek | 33.6 / 28.3 | +5.3 [-0.7, +11.2] | 0.101 (0.505) | 0.385 | 44.7 / 37.7 | +7.0 | 0.136 (0.719) |

- Group contrast, mean non-self of Gemini/Grok/DeepSeek minus GPT/Claude: +5.9 pp [+2.9, +9.0], bootstrap p = 0.0002.
- Top vs runner-up: Grok vs Gemini.

- **Answer.** The plan's p = 0.91 and p = 1.00 are Fisher tests of the TOP judge against the RUNNER-UP in each condition (not top vs bottom). lineup: Gemini vs GPT (top vs runner-up) 1.3 pp, Fisher p = 0.907, cluster bootstrap p = 0.844, common-text sign-flip p = 0.704; lineup: pairs that differ after Holm on the full non-self sets: none; on common text: none; lineup: no-advantage group minus advantage group +0.1 pp, p = 0.974; single: Grok vs Gemini (top vs runner-up) 0.7 pp, Fisher p = 1.000, cluster bootstrap p = 0.826, common-text sign-flip p = 1.000; single: pairs that differ after Holm on the full non-self sets: GPT-Gemini, GPT-Grok, Claude-Gemini, Claude-Grok; on common text: GPT-DeepSeek, Claude-Gemini, Claude-Grok; single: no-advantage group minus advantage group +5.9 pp, p = 0.0002. "The most accurate" is therefore a tie at the top in both conditions. Where a full-set gap is significant but the common-text gap is not, the gap comes from WHICH texts are in each judge's non-self set (a judge whose set includes Claude's widely recognized text looks better), not from skill.

## C8. Self-advantage net of judge skill

A model judged mostly by strong attributors gets an inflated peer baseline, so its self-advantage is understated (and a strong judge's own advantage overstated). Two adjustments:

- **(i) GEE, all 950 judgments**: correct ~ judge + author + a self term per judge, exchangeable within prompt. The judge effect is general skill, the author effect is how identifiable each author is to everyone; the self term is the log-OR of J being right on its own text beyond both. Judges with 0/38 self hits have an infinite self term (not estimable).
- **(ii) Difference in differences**: for each peer K, J and K are both scored on the same 114 responses by the three other authors (third-party text). Adjusted self-advantage = mean over K of [(J on own text - J on third-party) - (K on J's text - K on third-party)], in pp; equivalently each peer's hit rate on J's text is shifted by J's minus K's third-party accuracy. The ratio variant rescales K's hit rate by J/K third-party accuracy instead. Prompt-cluster bootstrap CI and p.

### lineup

| Judge | self-adv pp (unadj.) | DiD adj. peer | DiD self-adv pp [CI], p | ratio self-adv pp [CI], p | GEE judge+author self OR, p robust (bias-red.) |
|---|---|---|---|---|---|
| GPT | +27.6 | 38.6 | +19.3 [+4.4, +34.2], 0.011 | +19.0 [+3.0, +34.5], 0.019 | 2.16, 0.020 (0.023) |
| Claude | +32.9 | 51.8 | +35.1 [+21.3, +48.2], 0.0002 | +36.2 [+14.0, +53.8], 0.002 | 6.20, 0.00039 (0.00055) |
| Gemini | -9.9 | 37.7 | -19.3 [-32.2, -5.9], 0.006 | -19.4 [-34.8, -5.2], 0.008 | 0.37, 0.014 (0.017) |
| Grok | -23.7 | 25.4 | -20.2 [-30.5, -9.6], 0.0004 | -26.3 [-41.0, -15.0], 0.0002 | 0.16, 0.011 (0.013) |
| DeepSeek | -0.7 | 9.6 | +11.4 [-7.7, +31.4], 0.244 | +6.0 [-9.8, +22.3], 0.464 | 1.74, 0.332 (0.345) |

Pooled self term (one is_self for all judges, net of judge and author): OR 1.24, robust p 0.154.

Qualitative pattern (+ / - = significant at .05 in that direction, 0 = not significant):

| Judge | paper | unadjusted (rand.) | DiD | ratio | GEE judge+author |
|---|---|---|---|---|---|
| GPT | + | + | + | + | + |
| Claude | + | + | + | + | + |
| Gemini | 0 | 0 | - | - | - |
| Grok | - | - | - | - | - |
| DeepSeek | 0 | 0 | 0 | 0 | 0 |

### single

| Judge | self-adv pp (unadj.) | DiD adj. peer | DiD self-adv pp [CI], p | ratio self-adv pp [CI], p | GEE judge+author self OR, p robust (bias-red.) |
|---|---|---|---|---|---|
| GPT | +33.6 | 65.4 | +29.4 [+20.2, +38.6], 0.0002 | +15.8 [-36.6, +40.2], 0.421 | 2.49, 0.373 (0.386) |
| Claude | +4.6 | 96.9 | -10.1 [-21.7, +0.0], 0.051 | -- --, -- | 0.03, 0.001 (0.002) |
| Gemini | -1.3 | -3.1 | +3.1 [-0.7, +6.6], 0.108 | -0.4 [-1.0, +0.0], 0.265 | not estimable |
| Grok | -0.7 | -3.1 | +3.1 [-0.7, +6.6], 0.105 | -0.2 [-0.8, +0.0], 0.714 | not estimable |
| DeepSeek | -0.7 | -10.1 | +10.1 [+3.9, +15.8], 0.002 | -0.5 [-1.8, +0.0], 0.722 | not estimable |

*Caveat:* under single-text elicitation a judge's third-party "accuracy" is almost entirely which model it habitually names (99% of guesses are GPT or Claude), so it measures blame bias, not skill; GPT's third-party accuracy on text by Gemini/Grok/DeepSeek is near zero, which is why the ratio is undefined for Claude and unstable for GPT. The paper already reads self-advantage from the lineup for this reason; the single-text rows here should not be used to revise it.

Pooled self term (one is_self for all judges, net of judge and author): OR 0.22, robust p 0.117.

Qualitative pattern (+ / - = significant at .05 in that direction, 0 = not significant):

| Judge | paper | unadjusted (rand.) | DiD | ratio | GEE judge+author |
|---|---|---|---|---|---|
| GPT | + | + | + | 0 | 0 |
| Claude | 0 | 0 | 0 | n/a | - |
| Gemini | 0 | 0 | 0 | 0 | n/a |
| Grok | 0 | 0 | 0 | 0 | n/a |
| DeepSeek | 0 | 0 | + | 0 | n/a |

- **Answer.** Departures from the paper's pattern. lineup (the condition the paper reads self-advantage from): Gemini weighted_additive: 0 -> -; Gemini weighted_ratio: 0 -> -; Gemini gee_judge_author: 0 -> -. single-text (adjustment not interpretable; see caveat): GPT weighted_ratio: + -> 0; GPT gee_judge_author: + -> 0; Claude gee_judge_author: 0 -> -; DeepSeek weighted_additive: 0 -> +.

## Effective sample size: ICC and design effect

One-way ANOVA ICC(1) of correctness. "raw" is on 0/1 correctness; "resid" is after removing the 25 judge x author cell means, i.e. the shared difficulty of a prompt or response beyond who wrote and who judged it. Peer baseline: clusters are J's 38 responses, 4 peer judgments each; design effect = 1 + 3 x ICC, effective n = 152 / deff. The bootstrap design effect is the cluster-bootstrap variance of the peer baseline over the binomial variance.

| Condition | ICC prompt raw / resid | ICC response raw / resid | peer ICC pooled | deff | effective n of 152 |
|---|---|---|---|---|---|
| lineup | 0.011 / 0.023 | 0.097 / 0.033 | 0.014 | 1.04 | 146 |
| single | -0.031 / 0.009 | 0.632 / 0.114 | -0.065 | 0.80 | 189 |

| Condition | GPT peer ICC / bootstrap deff | Claude peer ICC / bootstrap deff | Gemini peer ICC / bootstrap deff | Grok peer ICC / bootstrap deff | DeepSeek peer ICC / bootstrap deff |
|---|---|---|---|---|---|
| lineup | 0.05 / 1.13 | 0.00 / 0.99 | -0.04 / 0.85 | 0.03 / 1.07 | 0.04 / 1.07 |
| single | 0.02 / 1.03 | -0.21 / 0.35 | -0.01 / 0.95 | 0.00 / 0.99 | 0.00 / 0.99 |

- Peer-baseline design effect 1.04 in the lineup (152 peer judgments carry about 146 independent ones) and 0.80 in single-text (about 189).

