"""CompLLM revision: the two judge panels.

A panel is a set of models that both write the corpus and judge it, so that
"self" always means judge and author are the same model.

  frontier   the original five, served through OpenRouter (the submitted
             study). Collection needs OPENROUTER_API_KEY; nothing here runs
             without it.
  openweight five open-weight instruction models from five labs, run locally
             in 4-bit on one consumer GPU. Weights are public, so there is no
             serving variance, no vendor watermark, and the judge's own
             likelihood of any text can be read directly.

Names in `name` are what the judge is shown as the candidate list, in the
panel's canonical order. As in config.MODELS, the index of a model in this
list is its identity in the Latin square: do not reorder after collection.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import JUDGE_SLUGS, MODELS, ROOT  # noqa: E402


@dataclass(frozen=True)
class Panel:
    key: str
    backend: str                      # "openrouter" or "hf"
    names: list[str]                  # candidate names shown to judges
    model_ids: dict[str, str]         # name -> OpenRouter slug or HF repo
    aliases: dict[str, str] = field(default_factory=dict)  # lowercased -> name
    results_dir: Path = ROOT / "results"
    corpus_csv: Path = ROOT / "data" / "responses_v1.csv"


FRONTIER = Panel(
    key="frontier",
    backend="openrouter",
    names=list(MODELS),
    model_ids=dict(JUDGE_SLUGS),
    aliases={
        "gpt-4o": "GPT", "chatgpt": "GPT", "openai": "GPT", "gpt-5": "GPT",
        "gpt5": "GPT", "claude sonnet": "Claude", "claude opus": "Claude",
        "sonnet": "Claude", "opus": "Claude", "anthropic": "Claude",
        "google": "Gemini", "bard": "Gemini", "xai": "Grok", "x-ai": "Grok",
        "deep seek": "DeepSeek",
    },
    results_dir=ROOT / "results" / "revision" / "frontier",
    corpus_csv=ROOT / "data" / "responses_v1.csv",
)

# All five are ungated on the Hugging Face Hub. Each is Unsloth's bitsandbytes
# NF4 export of the lab's own instruction model (Alibaba, Meta, Mistral AI,
# Google, Microsoft), about 3-6 GB apiece, so one fits an 8 GB GPU with room
# for a lineup's context. BASE_IDS (below) records the unquantised sources.
OPENWEIGHT = Panel(
    key="openweight",
    backend="hf",
    # Gemma-2-9B was planned as a fifth judge and dropped on 1 Oct 2026: its
    # 256k-entry vocabulary does not fit beside the other weights on the
    # collection laptop (8 GB GPU, 32 GB shared RAM) without exhausting memory.
    names=["Qwen", "Llama", "Mistral", "Phi"],
    model_ids={
        "Qwen":    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "Llama":   "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
        "Mistral": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "Gemma":   "unsloth/gemma-2-9b-it-bnb-4bit",
        "Phi":     "unsloth/Phi-4-mini-instruct-bnb-4bit",
    },
    aliases={
        "qwen2.5": "Qwen", "alibaba": "Qwen", "tongyi": "Qwen",
        "llama 3": "Llama", "llama-3": "Llama", "meta": "Llama",
        "mixtral": "Mistral", "mistral ai": "Mistral",
        "phi-4": "Phi", "phi-3": "Phi", "microsoft": "Phi",
    },
    results_dir=ROOT / "results" / "revision" / "openweight",
    corpus_csv=ROOT / "data" / "openweight" / "responses_ow.csv",
)

PANELS = {p.key: p for p in (FRONTIER, OPENWEIGHT)}


def canon(panel: Panel, raw) -> str | None:
    """Map a judge's free-text model name onto the panel's candidate list."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    for name in panel.names:
        if s == name.lower():
            return name
    if s in panel.aliases:
        return panel.aliases[s]
    # Longest match first, so "gemini" cannot shadow a longer alias.
    keys = sorted([n.lower() for n in panel.names] + list(panel.aliases),
                  key=len, reverse=True)
    for k in keys:
        if k in s:
            return next((n for n in panel.names if n.lower() == k),
                        panel.aliases.get(k))
    return None


# Unquantised sources of the open-weight panel, for the paper's model table.
BASE_IDS = {
    "Qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "Llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "Gemma":   "google/gemma-2-9b-it",
    "Phi":     "microsoft/Phi-4-mini-instruct",
}
