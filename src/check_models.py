"""CompLLM: verify OpenRouter model slugs and prices before spending money.

    python src/check_models.py            # candidates for the five families
    python src/check_models.py --all gpt  # every slug matching "gpt"

Prices are per 1M tokens, read live from OpenRouter's catalogue.
Paste the slugs you choose into JUDGE_SLUGS in src/config.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    EXCLUDE_PROMPTS, JUDGE_SLUGS, OPENROUTER_MODELS_URL, RESPONSES_CSV, api_key,
)

FAMILY_PREFIX = {
    "GPT": "openai/",
    "Claude": "anthropic/",
    "Gemini": "google/",
    "Grok": "x-ai/",
    "DeepSeek": "deepseek/",
}

# Slugs matching these substrings are hidden from the shortlist - they are
# small/cheap/legacy variants we would not use as a frontier judge.
NOISE = ("mini", "nano", "lite", "flash-8b", "instruct:free", ":free", "vision",
         "embed", "tts", "whisper", "image", "search", "codex", "distill")


def fetch_models() -> list[dict]:
    r = requests.get(
        OPENROUTER_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key()}"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("data", [])


def verify_key() -> None:
    """Fail loudly on a dead key.

    The catalogue endpoint is PUBLIC -- OpenRouter serves it regardless of
    whether the bearer token is valid, so fetch_models() succeeds and prints a
    full price table even when the key is revoked. This function hits an
    endpoint that actually authenticates, so a bad key is caught here instead
    of as a wall of HTTP 401s partway through a collection run.
    """
    r = requests.get(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {api_key()}"},
        timeout=30,
    )
    if r.status_code == 401:
        sys.exit(
            "AUTH FAILED (HTTP 401) -- the key in .env is not valid.\n"
            "  Create one at https://openrouter.ai -> Keys and check that\n"
            "  .env holds the current key."
        )
    r.raise_for_status()
    d = r.json().get("data", {})
    limit, usage = d.get("limit"), d.get("usage")
    budget = "unlimited" if limit is None else f"${limit:.2f} cap"
    print(f"key OK  |  usage ${usage or 0:.2f}  |  {budget}")
    if limit is not None and usage is not None and limit - usage < 3:
        print(f"  WARNING: only ${limit - usage:.2f} left under this key's cap.")


def price(m: dict, kind: str) -> float:
    try:
        return float(m.get("pricing", {}).get(kind, 0) or 0) * 1_000_000
    except (TypeError, ValueError):
        return 0.0


def show(models: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'slug':<46} {'$/M in':>9} {'$/M out':>9}  {'ctx':>9}")
    print("  " + "-" * 78)
    for m in models:
        print(
            f"  {m['id']:<46} {price(m,'prompt'):>9.2f} {price(m,'completion'):>9.2f}"
            f"  {m.get('context_length', 0):>9,}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", default="", help="show every slug containing this string")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    verify_key()
    models = fetch_models()
    print(f"OpenRouter catalogue: {len(models)} models")
    by_id = {m["id"]: m for m in models}

    if args.all:
        hits = [m for m in models if args.all.lower() in m["id"].lower()]
        show(sorted(hits, key=lambda m: m["id"]), f'ALL SLUGS MATCHING "{args.all}"')
        return

    for family, prefix in FAMILY_PREFIX.items():
        hits = [
            m for m in models
            if m["id"].startswith(prefix)
            and not any(n in m["id"].lower() for n in NOISE)
        ]
        hits.sort(key=lambda m: (-price(m, "prompt"), m["id"]))
        show(hits[: args.top], f"{family}  (prefix {prefix})")

    print("\n" + "=" * 80)
    print("CONFIGURED IN config.py:")
    total_in = total_out = 0.0
    all_ok = True
    for family, slug in JUDGE_SLUGS.items():
        m = by_id.get(slug)
        if m:
            pin, pout = price(m, "prompt"), price(m, "completion")
            total_in += pin
            total_out += pout
            print(f"  OK       {family:<10} {slug:<40} ${pin:.2f}/M in  ${pout:.2f}/M out")
        else:
            all_ok = False
            print(f"  MISSING  {family:<10} {slug:<40} <-- not in catalogue, fix this")

    if all_ok:
        # One lineup session per (prompt, judge): ~3.5k tokens in, ~350 out.
        ids = pd.read_csv(RESPONSES_CSV)["prompt_id"].astype(str)
        n_prompts = int(ids[~ids.isin(EXCLUDE_PROMPTS)].nunique())
        in_tok = n_prompts * 3500 / 1e6
        out_tok = n_prompts * 350 / 1e6
        est = in_tok * total_in + out_tok * total_out
        print(f"\n  Estimated lineup cost ({n_prompts} prompts x "
              f"{len(JUDGE_SLUGS)} judges): ${est:,.2f}")
        print("  Budget 2-3x that for retries and the single-text condition.")
    print("=" * 80)


if __name__ == "__main__":
    main()
