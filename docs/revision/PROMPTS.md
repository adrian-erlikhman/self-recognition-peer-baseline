# Prompts: the original bank and the expanded v2 set

Two files, both UTF-8 CSV:

| File | Rows | Columns | What it is |
|---|---|---|---|
| `data/prompts_bank.csv` | 40 | `prompt_id, tier, prompt_text, expected_signal, in_corpus` | The study's original prompt bank, verbatim |
| `data/prompts_v2.csv` | 120 | `prompt_id, genre, tier, prompt_text, source, answer` | The expanded set for the revision |

Neither file is named `data/prompts.csv` on purpose. `src/run_lineup.py` shows
the prompt to the judge when that file exists, and the published results come
from judges that saw responses only. Don't rename either file to that name
unless you want to run the prompt-visible condition.

## prompts_bank.csv

**Source.** The Google Doc "Prompt Bank" (Drive id
`1qXkQfAyZ7ZqMKj9RbcTXB1Fc8ygBISRUrv98hTi2a94`), read on 29 Sept 2026. Its three
tables (Tier 1 Easy, Tier 2 Medium, Tier 3 Hard) give each prompt's ID, the text
to submit verbatim, and an "expected stylometric signal" column. The prompt text
here is the full submitted string. That includes the closing prose-only
instruction, which the doc sets in bold blue, but the bold is not part of the
text.

**Fidelity.** The prompt text is copied exactly as it appears in the doc. The
only non-ASCII character in it is the em-dash (U+2014), which the Hard prompts
use for parentheticals. Quotation marks and apostrophes are plain ASCII in the
doc's text export, and they stay that way here. Medium and Hard prompts end
with "Do not address counterarguments." before the prose instruction. Easy
prompts don't have that sentence.

**Checks run when the files were built:**

- There are 40 IDs, all unique: E1-E14, M1-M13 and H1-H13.
- Every prompt ends with the exact sentence "Write in continuous prose
  paragraphs only. Do not use bullet points, numbered lists, headers, bold text,
  or any other formatting."
- E1 matches the prompt quoted in `local/paper/main.tex` (Appendix, "Example
  prompt (E1, verbatim)") character for character once LaTeX line breaks are
  collapsed.
- Every prompt in the corpus has the same tier in `data/responses_v1.csv`.
- `in_corpus` is 1 for the 39 IDs present in `data/responses_v1.csv` and 0 for
  E14.

**Bank vs. corpus:**

| | Bank | Corpus (`responses_v1.csv`) |
|---|---|---|
| Easy | E1-E14 (14) | E1-E13 (13) |
| Medium | M1-M13 (13) | M1-M13 (13) |
| Hard | H1-H13 (13) | H1-H13 (13) |
| Total | 40 prompts, 200 responses planned | 39 prompts x 5 models = 195 responses |

- E14 (volunteering) is in the bank but was never collected. Nothing in the
  repo or the paper mentions it; the paper's "One of 39 prompts" counts from
  the collected set.
- No corpus ID is missing from the bank.
- M5 is collected, but the analysis excludes it (Gemini's M5 copies its E13;
  see `EXCLUDE_PROMPTS` in `config.py`), leaving 38 prompts.
- The doc's overview predates the study. It says "40 prompts x 5 models = 200
  responses". It lists the models as "GPT-5.5, Claude Opus 4.7, Gemini , Grok,
  DeepSeek" with Gemini's version blank. It also carries an early headline
  (models fail "roughly 65%" of the time, and wrong guesses land on "GPT-4o or
  Claude") that doesn't match the paper. Treat only the tables as the source.
- The doc sets a 150-word minimum per response. Every response in
  `responses_v1.csv` meets it.

## prompts_v2.csv

**Composition (120 prompts):**

| genre | source | IDs | Easy | Medium | Hard | Total |
|---|---|---|---|---|---|---|
| prose | bank | E1-E14, M1-M13, H1-H13 | 14 | 13 | 13 | 40 |
| prose | new | E15-E21, M14-M20, H14-H19 | 7 | 7 | 6 | 20 |
| code | new | C1-C20 | 7 | 8 | 5 | 20 |
| short | new | S1-S20 | 10 | 8 | 2 | 20 |
| math | new | Q1-Q20 | 7 | 9 | 4 | 20 |
| **all** | | | **45** | **45** | **30** | **120** |

For prose, the tier is the bank's Flesch Reading Ease target: Easy above 60,
Medium 30-60, Hard below 30. For code, short and math, the tier is a rough
content difficulty that we assigned by hand. Use it to balance samples, not as
an analysis variable.

The `source=bank` rows are byte-identical to `prompts_bank.csv`. E14 is in v2
too, so a fresh collection gets all 40 original prompts. New prompts are plain
ASCII.

### The four genres

**Prose (60).** The 20 new prompts copy the bank's scaffold for each tier:

- Easy prompts open with "Argue / Make the case / Convince someone that...",
  followed by three everyday reasons listed inline. There's no counterargument
  sentence.
- Medium prompts open with "Defend the position / Make the case / Argue that...".
  Then comes "Address/Cover exactly these three points: (1) ...; (2) ...;
  (3) ...", then "Do not address counterarguments."
- Hard prompts open with "Defend the thesis / Argue that...". Then comes
  "Develop exactly these three claims:" with three dense, named-theory claims,
  then "Do not address counterarguments."

All of them end with the bank's prose-only sentence. None of the topics repeats
a bank topic:

- Easy: learning an instrument, walking or biking short trips, first aid, home
  gardening, a daily planner, board games, learning to swim.
- Medium: preventive maintenance, open-source contribution, the shipping
  container, citizen science, code review, checklists in aviation and surgery,
  weather forecasting.
- Hard: interventionist causation, etiological function, testimony,
  idealized models, knowing-how, biodiversity and stability.

Politically or socially contested topics were left out.

**Code (20).** Each prompt asks for one self-contained Python function and gives
its exact signature and behaviour. Edge cases are pinned down, and where it
helps a worked example is included, such as `merge_intervals`, `rotate_matrix`
and `edit_distance("kitten", "sitting") == 3`. Tie-breaks are specified
(`top_k_words`, `topological_order`), so every correct answer computes the same
function. Every code prompt ends with the same sentence: "Use only the Python
standard library. Include a docstring and brief inline comments. Return only
the code in a single Python code block, with no explanation before or after."
Because the code is fixed, what can vary is the author's habits: naming,
docstring and comment style, type hints, loop and comprehension idioms, and
error handling. The difficulty mix runs from `is_palindrome` and `fizzbuzz`
through Kadane's algorithm and duration parsing to Dijkstra, a deterministic
topological sort and N-queens. Every example quoted in a prompt was checked
against a reference implementation.

**Short answer (20).** These are factual or explanatory questions that each
end with "Answer in two to four sentences of plain prose.", for example
"Explain why the sky appears blue during the day." They cover everyday
physics, biology, earth science and two neutral definitions (algorithm,
inflation). The answers are well settled, so the content is about the same for
every model and the length is capped.

**Math (20).** These are word problems with a single numeric answer. Each ends
with "Solve the problem, showing your reasoning step by step in plain prose,
and end with a final line of the form 'Answer: <number>'." The `answer` column
holds the correct value. Each one was computed in code (exact fractions, or
brute-force enumeration where possible) and agreed with a hand solution:

| | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 |
|---|---|---|---|---|---|---|---|---|---|---|
| answer | 19.5 | 315 | 126 | 9368 | 32.4 | 0.1389 | 5624.32 | 2.4 | 120 | 50 |

| | Q11 | Q12 | Q13 | Q14 | Q15 | Q16 | Q17 | Q18 | Q19 | Q20 |
|---|---|---|---|---|---|---|---|---|---|---|
| answer | 21 | 103 | 17340 | 24 | 8 | 10 | 180 | 9600 | 130 | 271 |

Answers that are money or probabilities state their rounding in the prompt: Q6
is rounded to four decimals and Q7 to the cent. The `answer` column is blank
for non-math rows. It lets the pipeline check correctness separately from
style. That matters because a wrong final answer changes the content, and a
difference in content could otherwise pass for a difference in style.

## Design rationale

**Why content scaffolding.** The study asks whether a model can recognize its
own writing *style*. If the prompts left the content open, each model would
pick its own arguments, examples and structure. A judge could then recognize
itself, or be fooled, by what was said rather than how. The bank closes that
gap: each prompt fixes the claim, the three points to cover and, at Medium and
Hard, forbids counterarguments. The five responses then say the same thing, and
what is left to tell them apart is sentence length, hedging, connectives,
punctuation and the rest of the stylometric signal listed in each prompt's
`expected_signal`. The fixed prose-only sentence removes formatting as a
giveaway, so any formatting that survives is the model's own choice. The new
prompts keep this discipline in every genre. Code prompts fix the function,
short-answer prompts fix a settled fact and a length, and math prompts fix a
problem with one right answer.

**Why code, short answers and math.** A reviewer said that 38 argumentative
prose prompts may not show the effect transfers to other kinds of output, and
named code, math and short answers. The three new genres test that directly,
and each one stresses the self-recognition claim in a different way:

- Code has almost no room for rhetorical style. The signal left is
  programming idiom, which tests whether self-recognition relies on prose-style
  features at all.
- Short answers are two to four sentences, far below the bank's 150-word floor
  for prose. This tests whether the effect survives when there is little text
  to go on.
- Math reasoning follows a fixed logical path to a checkable answer. The
  differences left are in how steps are narrated and laid out, and the
  `answer` column lets correctness be separated from style.

The 20 new prose prompts, together with the uncollected E14, grow the original
genre from 38 analysed prompts to 60. The prose results can then be re-run on
a set that is half new, as a check against overfitting to the original bank.
