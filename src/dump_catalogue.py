"""CompLLM: dump the full OpenRouter catalogue to a file.

    python src/dump_catalogue.py

Writes results/openrouter_catalogue.json with every model id, price, and
context length. One command instead of running check_models.py --all
separately for each family, and it leaves an artifact that can be inspected
(or diffed later, when slugs inevitably change again).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OPENROUTER_MODELS_URL, RESULTS_DIR, api_key  # noqa: E402

FAMILIES = {
    "GPT": ("openai/",),
    "Claude": ("anthropic/",),
    "Gemini": ("google/gemini", "google/gemma"),
    "Grok": ("x-ai/",),
    "DeepSeek": ("deepseek/",),
}


def main() -> None:
    r = requests.get(
        OPENROUTER_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key()}"},
        timeout=60,
    )
    r.raise_for_status()
    models = r.json().get("data", [])

    def price(m: dict, kind: str) -> float:
        try:
            return float(m.get("pricing", {}).get(kind, 0) or 0) * 1_000_000
        except (TypeError, ValueError):
            return 0.0

    slim = sorted(
        (
            {
                "id": m["id"],
                "name": m.get("name", ""),
                "in_per_m": round(price(m, "prompt"), 4),
                "out_per_m": round(price(m, "completion"), 4),
                "context": m.get("context_length", 0),
            }
            for m in models
        ),
        key=lambda d: d["id"],
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "openrouter_catalogue.json"
    path.write_text(json.dumps(slim, indent=1), encoding="utf-8")

    print(f"{len(slim)} models -> {path}\n")
    for family, prefixes in FAMILIES.items():
        hits = [m for m in slim if any(m["id"].startswith(p) for p in prefixes)]
        print(f"{family}: {len(hits)} slugs")
        for m in hits:
            print(f"   {m['id']:<48} {m['in_per_m']:>8.2f} in  {m['out_per_m']:>8.2f} out")
        print()


if __name__ == "__main__":
    main()
