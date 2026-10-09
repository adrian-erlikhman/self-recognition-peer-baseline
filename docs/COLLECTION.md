# How the judgments were collected

Judge queries went through OpenRouter, a commercial gateway that exposes
models from several vendors behind one OpenAI-compatible endpoint. It was used
for the judge calls only. The response corpus itself was collected earlier
through the vendors' own chat interfaces.

## Judges

From `src/config.py`, version-matched to the model that authored each class:

| Class | Slug | List price, in / out per 1M tokens |
|---|---|---|
| GPT | `openai/gpt-5.5` | $5.00 / $30.00 |
| Claude | `anthropic/claude-opus-4.7` | $5.00 / $25.00 |
| Gemini | `google/gemini-3.5-flash` | $1.50 / $9.00 |
| Grok | `x-ai/grok-4.3` | $1.25 / $2.50 |
| DeepSeek | `deepseek/deepseek-chat` | $0.26 / $1.03 |

Judge calls were made on 16 and 17 August 2026. Requests: `temperature = 0`, `max_tokens = 8000` (16000 for Gemini, which
spends most of its budget on reasoning tokens), 300 s timeout, concurrency 4.
Every response is archived under `raw/`, including the `provider` field and
OpenRouter's per-call `usage.cost`, which is where `src/costs.py` gets the
spend ledger.

## Routing

OpenRouter picks an upstream host per request for models with several hosts.
Across all 1,140 archived calls:

| Slug | Upstream hosts |
|---|---|
| `openai/gpt-5.5` | OpenAI 228 |
| `google/gemini-3.5-flash` | Google 228 |
| `x-ai/grok-4.3` | xAI 228 |
| `anthropic/claude-opus-4.7` | Anthropic 227, Google 1 |
| `deepseek/deepseek-chat` | DeepInfra 122, StreamLake 69, Novita 37 |

Four judges were served by one host throughout. DeepSeek was split across
three, which may differ in quantization or sampling. Its lineup accuracy by
host:

| Host | Correct | Total | Accuracy |
|---|---|---|---|
| DeepInfra | 33 | 135 | 24.4% |
| Novita | 6 | 30 | 20.0% |
| StreamLake | 9 | 25 | 36.0% |

Chi-square 2.01, df 2, p = 0.37. Disclosed as a design note, not corrected for.

## Version matching

Two of five judge slugs match the chat-app corpus author string exactly
(Claude Opus 4.7, Gemini 3.5 Flash). The other three rows record the product
label shown in each app: "Limited GPT-5.5 Instant", "Fast Grok 4.3" and
"Deepseek Instant". We do not know whether these variants share weights and
settings with `openai/gpt-5.5`, `x-ai/grok-4.3` and `deepseek/deepseek-chat`,
so for those three judges "self" on the chat-app essays is an approximate
match. Six of Claude's 39 rows record Sonnet 4.6 rather than Opus 4.7. The
API texts were written by the judge slugs themselves, so there the match is
exact.
