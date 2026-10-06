"""CompLLM: offline self-test. No API calls, no money spent.

    python src/test_harness.py

Checks the parts that would otherwise only fail after you have already paid:
the Latin square, the completion parser against messy real-world judge
output, and the sessions -> judgments pipeline end to end on synthetic data.
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from design import build_design, verify_balance  # noqa: E402
from run_lineup import canon_model, extract_json, parse_completion  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  - ' + detail if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
print("\n[1] Latin square")
prompts = [f"P{i}" for i in range(39)]
design = build_design(prompts, config.MODELS)
verify_balance(design, n_prompts=39)
check("975 judgments", len(design) == 975, str(len(design)))
check("195 sessions", len({r["session_id"] for r in design}) == 195)

from collections import Counter  # noqa: E402

pairs = Counter((r["true_author"], r["slot"]) for r in design)
check("every (model, slot) pair occurs exactly 39x", set(pairs.values()) == {39})

# within-judge variation: a judge must NOT see one model in a fixed slot
per_judge = Counter(
    (r["judge"], r["true_author"], r["slot"]) for r in design
)
check(
    "no judge sees a model in the same slot on all 39 prompts",
    max(per_judge.values()) < 39,
    f"max={max(per_judge.values())}",
)

# each session is a permutation of the 5 slots
sess: dict[str, list] = {}
for r in design:
    sess.setdefault(r["session_id"], []).append(r["slot"])
check("every session fills all 5 slots exactly once",
      all(sorted(v) == sorted(config.SLOTS) for v in sess.values()))

# ---------------------------------------------------------------------------
print("\n[2] Model-name canonicalisation")
cases = {
    "Claude": "Claude", "claude": "Claude", "Claude 3.5 Sonnet": "Claude",
    "Anthropic Claude": "Claude", "opus": "Claude",
    "GPT": "GPT", "GPT-4o": "GPT", "ChatGPT": "GPT", "gpt-5.5": "GPT",
    "OpenAI GPT-4": "GPT",
    "Gemini": "Gemini", "gemeni": "Gemini", "Google Gemini Pro": "Gemini",
    "Grok": "Grok", "grok-4": "Grok", "xAI Grok": "Grok",
    "DeepSeek": "DeepSeek", "deepseek": "DeepSeek", "Deep Seek": "DeepSeek",
    "deepseek-v3": "DeepSeek",
    "": None, "Mistral": None, "unknown": None,
}
bad = {k: (canon_model(k), v) for k, v in cases.items() if canon_model(k) != v}
check(f"{len(cases)} name variants canonicalise correctly", not bad, str(bad))

# ---------------------------------------------------------------------------
print("\n[3] Completion parser against messy judge output")

clean = json.dumps({s: {"model": m, "confidence": 4, "reason": "x"}
                    for s, m in zip(config.SLOTS, config.MODELS)})
fenced = f"Here is my analysis.\n```json\n{clean}\n```\nHope that helps!"
flat = json.dumps(dict(zip(config.SLOTS, config.MODELS)))
prose = """A: Claude
B: GPT-4o
C: Gemini
D - Grok
E. DeepSeek"""
markdown = """**A**: Claude
**B**: GPT
**C**: Gemini
**D**: Grok
**E**: DeepSeek"""
repeats = json.dumps({s: {"model": "Claude", "confidence": 2} for s in config.SLOTS})
partial = json.dumps({"A": {"model": "Claude"}, "B": {"model": "GPT"}})
refusal = "I'm not able to determine which model wrote these responses."

for label, text, want_n, want_review in [
    ("clean JSON", clean, 5, False),
    ("JSON in a code fence with prose", fenced, 5, False),
    ("flat JSON (no nested dict)", flat, 5, False),
    ("plain-text lines", prose, 5, False),
    ("markdown-bolded lines", markdown, 5, False),
    ("all five answers identical (repeats allowed)", repeats, 5, False),
    ("partial answer -> needs_review", partial, 2, True),
    ("refusal -> needs_review", refusal, 0, True),
]:
    parsed, review = parse_completion(text)
    check(f"{label}: {want_n} slots, needs_review={want_review}",
          len(parsed) == want_n and review == want_review,
          f"got {len(parsed)} slots, review={review}")

check("repeats are preserved, not deduplicated",
      [v["model"] for v in parse_completion(repeats)[0].values()] == ["Claude"] * 5)
check("extract_json ignores trailing prose",
      extract_json(fenced) is not None)

# ---------------------------------------------------------------------------
print("\n[4] End-to-end: synthetic sessions -> judgments.csv")

rng = random.Random(7)
tmp = Path(tempfile.mkdtemp())
orig_sessions, orig_judgments = config.SESSIONS_JSONL, config.JUDGMENTS_CSV
config.SESSIONS_JSONL = tmp / "sessions.jsonl"
config.JUDGMENTS_CSV = tmp / "judgments.csv"

import parse_judgments  # noqa: E402

parse_judgments.SESSIONS_JSONL = config.SESSIONS_JSONL
parse_judgments.JUDGMENTS_CSV = config.JUDGMENTS_CSV

by_session: dict[str, dict] = {}
for r in design:
    s = by_session.setdefault(
        r["session_id"],
        {"session_id": r["session_id"], "prompt_id": r["prompt_id"],
         "judge": r["judge"], "slot_to_author": {}},
    )
    s["slot_to_author"][r["slot"]] = r["true_author"]

# Simulate a prestige sink: 70% of guesses go to Claude, rest uniform.
with config.SESSIONS_JSONL.open("w", encoding="utf-8") as fh:
    for s in by_session.values():
        parsed = {}
        for slot in config.SLOTS:
            guess = "Claude" if rng.random() < 0.70 else rng.choice(config.MODELS)
            parsed[slot] = {"model": guess, "confidence": rng.randint(1, 5), "reason": "t"}
        fh.write(json.dumps({**s, "parsed": parsed, "needs_review": False,
                             "raw_text": "", "ok": True}) + "\n")

df = parse_judgments.to_frame(parse_judgments.load_sessions())
check("975 judgment rows", len(df) == 975, str(len(df)))
check("no missing true_author", df["true_author"].notna().all())
check("no missing guessed_model", df["guessed_model"].notna().all())
check("correct is 0/1", set(df["correct"].unique()) <= {0, 1})
check("self_slot count = 195 (one per session)", int(df["is_self_slot"].sum()) == 195)
check("balance survives into the analysis table",
      len(set(df.groupby(["true_author", "slot"]).size())) == 1)
check("simulated blame concentration is recovered",
      0.60 < (df["guessed_model"] == "Claude").mean() < 0.85,
      f"{(df['guessed_model'] == 'Claude').mean():.1%}")

config.SESSIONS_JSONL, config.JUDGMENTS_CSV = orig_sessions, orig_judgments

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED - harness is safe to run against the API.")
print("=" * 60)
