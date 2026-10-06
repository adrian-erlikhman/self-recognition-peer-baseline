"""CompLLM: shared configuration.

Everything needed to re-run the study lives in this file. Nothing here is read from the environment except the
API key.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RAW_DIR = ROOT / "raw"

RESPONSES_CSV = DATA_DIR / "responses_v1.csv"
# SHA-256 of responses_v1.csv, checked at run time. If this assertion fails,
# the corpus changed and every number in the paper must be regenerated.
RESPONSES_SHA256 = "522ad6f45999680ecc601486f041f7efb80f116da291660403f0c5d0d2e9d16b"

DESIGN_CSV = RESULTS_DIR / "lineup_design.csv"

# Condition 1 - LINEUP: judge sees all five responses to a prompt at once.
SESSIONS_JSONL = RESULTS_DIR / "sessions.jsonl"
JUDGMENTS_CSV = RESULTS_DIR / "judgments.csv"

# Condition 2 - SINGLE: judge sees one response at a time (Bai et al. 2025
# format). Required because in the lineup condition judges spontaneously
# assign a one-to-one permutation, which forces the marginal blame
# distribution to be uniform and makes concentration unmeasurable. Only
# independent per-response judgments can show a prestige sink.
SESSIONS_SINGLE_JSONL = RESULTS_DIR / "sessions_single.jsonl"
JUDGMENTS_SINGLE_CSV = RESULTS_DIR / "judgments_single.csv"

# --------------------------------------------------------------------------
# Author / judge classes
# --------------------------------------------------------------------------
# Canonical order. The index of a model in this list is its identity in the
# Latin square, so DO NOT reorder this list after collection has started.
MODELS: list[str] = ["GPT", "Claude", "Gemini", "Grok", "DeepSeek"]

SLOTS: list[str] = ["A", "B", "C", "D", "E"]

# Prompts excluded from the lineup, with the reason. Excluding here rather
# than editing the CSV keeps RESPONSES_SHA256 valid and keeps the exclusion
# visible and reproducible.
#
#   M5 - two of its five responses are corrupt. M5/DeepSeek holds a verbatim
#        copy of M5/Grok (stated 771 words, actual 386 = Grok's count) and
#        M5/Gemini holds a verbatim copy of E13/Gemini. Only 3 of 5 responses
#        are genuine, so the lineup would have false ground truth.
#        Recollect M5 and remove it from this list to restore 39 prompts.
EXCLUDE_PROMPTS: list[str] = ["M5"]

# --------------------------------------------------------------------------
# OpenRouter
# --------------------------------------------------------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from .env into os.environ (no dependency needed).

    Real environment variables always win, so CI can override the file.
    """
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_dotenv()


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set.\n"
            "  export OPENROUTER_API_KEY=sk-or-v1-...\n"
            "or put it in a .env file next to this repo (see .env.example)."
        )
    return key


# Judge models, checked against the live OpenRouter catalogue at collection
# time (results/openrouter_catalogue.json).
#
# Each judge is version-matched to the model that AUTHORED that class's
# responses, taken from the corpus `model_version` column. This matters: if
# the judge is a different version from the author, the diagonal measures
# cross-version recognition rather than self-recognition. Bai et al. (2025)
# used the same model as author and judge.
#
#   class      corpus author string        judge slug                  match
#   GPT        "Limited GPT-5.5 Instant"   openai/gpt-5.5              exact
#   Claude     "Claude Opus 4.7" (33/39)   anthropic/claude-opus-4.7   exact
#   Gemini     "Gemini 3.5 Flash"          google/gemini-3.5-flash     exact
#   Grok       "Fast Grok 4.3"             x-ai/grok-4.3               exact
#   DeepSeek   "Deepseek Instant"          deepseek/deepseek-chat      approximate
#
# Six of the 39 Claude rows were written by Sonnet 4.6, not Opus 4.7, so for
# those six a "self" judgment is cross-model. engine_b.py reports a
# sensitivity check with them dropped.
# "Deepseek Instant" is not a released version string; deepseek-chat is the
# general chat endpoint and the closest available match.
JUDGE_SLUGS: dict[str, str] = {
    "GPT":      "openai/gpt-5.5",             # $5.00 / $30.00 per M
    "Claude":   "anthropic/claude-opus-4.7",  # $5.00 / $25.00 per M
    "Gemini":   "google/gemini-3.5-flash",    # $1.50 /  $9.00 per M
    "Grok":     "x-ai/grok-4.3",              # $1.25 /  $2.50 per M
    "DeepSeek": "deepseek/deepseek-chat",     # $0.26 /  $1.03 per M
}

# Sampling. Temperature 0 for reproducibility; providers do not guarantee
# determinism even at 0, which is why every raw completion is archived to
# raw/ so the analysis is reproducible from stored data regardless.
TEMPERATURE = 0.0
# Far above the size of the answer itself. Reasoning models (Gemini 3.5
# Flash, and the GPT-5.x and Grok-4.x reasoning tiers) spend completion
# budget on reasoning tokens before emitting any answer; at a few hundred
# tokens they return mid-sentence fragments and nothing parseable.
MAX_TOKENS = 8000
REQUEST_TIMEOUT = 300

# Per-judge overrides. Gemini 3.5 Flash spends far more of its budget on
# reasoning than the others: at 4000 tokens it still failed to emit JSON in
# 14 of 38 lineup sessions. A session that returned but did not parse is not
# treated as done, so re-running picks those up.
MAX_TOKENS_BY_JUDGE: dict[str, int] = {
    "Gemini": 16000,
}
MAX_RETRIES = 4
RETRY_BASE_DELAY = 4.0

# Politeness / rate limiting
CONCURRENCY = int(__import__("os").environ.get("COMPLLM_CONCURRENCY", "4"))
