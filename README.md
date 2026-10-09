# Above Chance Is Not Enough: Auditing Self-Recognition in LLM Judges

Data and code for the paper by Adrian Erlikhman, Michael Tarekegn and Philo
Juang (arXiv, October 2026).

LLM judges are anonymised on the assumption that they cannot tell who wrote a
text. The simplest check, the self-naming test, asks whether a judge names
itself as the author of its own text more often than chance. This repository
holds everything needed to score that check against two controls from the same
experiment: the judge's **false-alarm rate** (how often it names itself on text
it did not write) and the **peer baseline** (how often the *other* judges name
it on its text). The self-naming test flags 15 of 37 cases; 8 of them
discriminate, and 3 of those also exceed their peer baseline (plus Claude in
the yes/no format, which has no peer baseline).

Every number in the paper can be recomputed from the files here with the
released code. Most are LaTeX macros written by the scripts below; the
revision's verdict counts, shared-prompt comparison and robustness checks come
from `paper_arxiv/code_council/`, and the few values entered by hand in
`paper_arxiv/numbers_c_extra.tex` name the check that produced each one. No API
key is needed: every API response was archived when it was collected.

## What is here

| | |
|---|---|
| **Study 1, chat-app corpus** | 190 texts (38 prompts x 5 frontier models, written in the consumer chat apps, Jun-Jul 2026); 1,900 judgments (lineup and one text at a time); 950 yes/no and 190 word-shuffled lineup judgments (Oct 2026) |
| **Study 1, API corpus** | 600 texts (120 prompts in four genres x 5 models, OpenRouter, 1 Oct 2026); 600 lineup and 3,000 single-text judgments |
| **Study 2, open-weight panel** | 480 texts plus a second sample each (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3, Phi-4-mini); lineup, single-text, yes/no, on-policy and resampled on-policy judgments; every judge's log-likelihood of every text; quality ratings |
| **Rationales** | all 1,900 one-sentence reasons from Study 1, coded into eleven cue types, with the codebook and the 150-reason validation sample |

## Reproduce the paper

```bash
pip install -r requirements.txt          # Python 3.12
python src/run_all.py                    # Study 1 (chat-app corpus): classifier, judges, statistics
python src/stats_revision.py             # clustered statistics for Study 1
python src/revision/analyze_panel.py --panel openweight
python src/revision/mark_refusals.py --panel frontier --condition shuffle_lineup  # idempotent
python src/revision/analyze_panel.py --panel frontier              # yes/no and shuffled lineup
python src/revision/bow_shuffle.py                                 # bag-of-words on shuffled text
python src/revision/standard_vs_proper.py --arxiv                  # rate test vs ours, 37 settings
python src/revision/analyze_panel.py --panel frontier --tag api120 # API corpus
python paper_tacl/make_numbers.py        # shared numbers and tables
python paper_arxiv/code_council/build_verdicts.py   # verdict table, Figure 2, the 37 cases
python paper_arxiv/code_council/build_shared.py     # the 38 shared prompts, Figure 4
python paper_arxiv/code_council/build_robust.py     # who names whom (Figure 3), name order
python paper_arxiv/code_council/peer_discrimination.py  # like-for-like peer comparison
python paper_arxiv/build.py              # arXiv-only numbers and tables, then compile (needs Tectonic)
```

The four `code_council/` scripts rewrite their tables and macro files in
place; the verdict and shared-prompt captions in the paper carry a sentence or
two added by hand, so compare rather than overwrite if you rerun them.

`paper_arxiv/` is the source of the arXiv version. `paper_tacl/` holds the
generators for the numbers and tables the two versions share (and the
anonymous journal version); `paper/` holds Study 1's generated tables.

## Re-running the experiments

Collecting new judgments needs an OpenRouter key in `.env` (see
`.env.example`); the open-weight study needs a CUDA GPU and
`requirements-openweight.txt`.

```bash
python src/run_lineup.py ; python src/run_single.py                 # Study 1, original formats
python src/revision/run_condition.py --panel frontier --condition binary
python src/revision/run_condition.py --panel frontier --condition shuffle_lineup
python src/revision/collect_frontier.py --expand                    # API corpus
python src/revision/run_openweight.py                               # all of Study 2
```

Every runner is resumable and archives each raw response (`raw/`,
`results/revision/*/raw/`). Model identifiers, dates and request settings are
in `docs/COLLECTION.md` and Appendix A of the paper.

## Layout

```
data/                     prompts (prompts_bank.csv, prompts_v2.csv) and all corpora
  responses_v1.csv        chat-app corpus (SHA-256 pinned in src/config.py)
  responses_api120.csv    API corpus
  openweight/             open-weight corpus, two samples per prompt
src/                      Study 1: features, classifier (engine_a), judges (engine_b),
                          statistics, rationale coding, generative-match tests
src/revision/             panels, judge conditions, open-weight backend (int4),
                          likelihood scoring, analysis of every condition
results/                  judgments, derived JSON; results/revision/<panel>/ per panel
raw/                      one file per Study 1 API call
docs/                     generated reports (RESULTS.md, revision/*.md) and collection notes
annotate/                 the page used to label the rationale validation sample
paper_arxiv/, paper_tacl/, paper/   paper sources and generators
tools/                    helpers that keep long GPU runs alive on a laptop
```

## Notes

- Six of Claude's 38 chat-app texts were written by Claude Sonnet 4.6, not
  Opus 4.7; the paper reports the effect of dropping them.
- Prompt M5 is excluded from the chat-app corpus (two corrupted answers).
- The open-weight models run as NF4 checkpoints re-packed to 4-bit groups of
  32 (`src/revision/int4.py`); the same weights write and judge.
- The rationale validation labels come from two LLM annotators (instances of
  Claude); no human annotated them.

## Licence

Code: MIT. Data (prompts, corpora, judgments, annotations): CC BY 4.0. The
model-written texts are outputs of the listed models, released for research.
