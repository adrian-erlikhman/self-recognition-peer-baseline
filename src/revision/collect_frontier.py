"""CompLLM revision: frontier-panel collection (A6, A7, and A4's resample).

NOT RUN YET: the OpenRouter key on file was revoked on or before 29 Sept 2026.
Put a working key in .env (OPENROUTER_API_KEY=...) and run the steps below in
order. Every call is archived under results/revision/frontier/raw/, and
--dry-run prints what would be sent and an input-token estimate.

  --repair     A7. Re-collect the defective rows of responses_v1.csv:
               M5 for all five models (two of its five were corrupt copies),
               the six Sonnet 4.6 rows of the Claude class with Opus 4.7
               (E1, E2, M1, M2, H1, H2), and H3/DeepSeek (truncated). Writes
               data/responses_v2_repair.csv. merge_corpus() then builds
               data/responses_v2.csv = v1 with those rows replaced, plus M5.
  --expand     A6. Every model answers every prompt in data/prompts_v2.csv
               that v1 lacks (E14, the 20 new prose prompts, code, short,
               math), one sample each: data/responses_v2_expand.csv.
  --resample   A4. A second, independent answer from each model to every
               prompt in the corpus it will judge, used as the judge's own
               turn in `run_condition.py --condition onpolicy_resample`:
               results/revision/frontier/resample_s2.csv.

A methods caveat the paper must state: the v1 corpus was collected by hand in
each vendor's consumer chat app at default settings, while everything here
goes through the API at temperature 0.7 with no system prompt. The chat apps
add their own system prompts. Keep the two sources distinguishable in the
analysis (the `source` column) and report the self-naming rates separately
for v1 rows and API rows before pooling them.

DeepSeek: pin a dated snapshot rather than the moving `deepseek/deepseek-chat`
alias. Run src/check_models.py, pick the dated DeepSeek slug that is live on
the day, and set it in DEEPSEEK_PINNED below and in panels.FRONTIER before
collecting anything, so the author and the judge are the same snapshot.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import OpenRouterBackend  # noqa: E402
from config import ROOT  # noqa: E402
from panels import FRONTIER  # noqa: E402

DEEPSEEK_PINNED = ""  # e.g. "deepseek/deepseek-chat-v3.2-2026xxxx"; see docstring
REPAIR_ROWS = ([("M5", m) for m in FRONTIER.names]
               + [(p, "Claude") for p in ("E1", "E2", "M1", "M2", "H1", "H2")]
               + [("H3", "DeepSeek")])
MAX_NEW = {"prose": 4000, "code": 3000, "short": 800, "math": 3000}
TEMPERATURE = 0.7
FIELDS = ["prompt_id", "genre", "tier", "model", "model_version", "source",
          "word_count", "text"]


def prompts() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data" / "prompts_v2.csv", dtype=str)
    return p.set_index("prompt_id")


def slug(model: str) -> str:
    if model == "DeepSeek" and DEEPSEEK_PINNED:
        return DEEPSEEK_PINNED
    return FRONTIER.model_ids[model]


MODELS_ONLY: list[str] = []


def collect(pairs: list[tuple[str, str]], out: Path, source: str, dry: bool) -> None:
    if MODELS_ONLY:
        # One file per model process, so parallel runs never share a file;
        # combine_parts() concatenates them before --merge.
        pairs = [(p, m) for p, m in pairs if m in MODELS_ONLY]
        out = out.with_name(f"{out.stem}__{'_'.join(MODELS_ONLY)}{out.suffix}")
    P = prompts()
    done = set()
    if out.exists():
        d = pd.read_csv(out, dtype=str)
        done = set(zip(d["prompt_id"], d["model"]))
    todo = [(p, m) for p, m in pairs if (p, m) not in done]
    print(f"{out.name}: {len(pairs)} rows, {len(todo)} to collect")
    if dry:
        for p, m in todo[:3]:
            print(f"  {p} / {m} via {slug(m)}: {P.loc[p, 'prompt_text'][:160]}...")
        chars = sum(len(P.loc[p, "prompt_text"]) for p, _ in todo)
        print(f"  est. input tokens {chars / 4:,.0f}; output ~{len(todo) * 700:,} tokens")
        if not DEEPSEEK_PINNED:
            print("  WARNING: DEEPSEEK_PINNED is empty; DeepSeek rows would use the moving alias.")
        return
    new = not out.exists()
    for m in FRONTIER.names:
        mine = [(p, mm) for p, mm in todo if mm == m]
        if not mine:
            continue
        be = OpenRouterBackend(slug(m), FRONTIER.results_dir / "raw" / "collect")
        for genre in sorted({P.loc[p, "genre"] for p, _ in mine}):
            chunk = [(p, mm) for p, mm in mine if P.loc[p, "genre"] == genre]
            convs = [[{"role": "user", "content": P.loc[p, "prompt_text"]}] for p, _ in chunk]
            ids = [f"{source}__{p}__{m}" for p, _ in chunk]
            texts = be.chat(convs, MAX_NEW[genre], temperature=TEMPERATURE, ids=ids)
            with out.open("a", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                    new = False
                for (p, _), t in zip(chunk, texts):
                    w.writerow({"prompt_id": p, "genre": genre, "tier": P.loc[p, "tier"],
                                "model": m, "model_version": slug(m), "source": source,
                                "word_count": len(t.split()), "text": t})


def combine_parts(base: Path) -> None:
    """Concatenate per-model part files (base__Model.csv) into base."""
    parts = sorted(base.parent.glob(f"{base.stem}__*{base.suffix}"))
    if not parts:
        return
    frames = [pd.read_csv(base, dtype=str, keep_default_na=False)] if base.exists() else []
    frames += [pd.read_csv(f, dtype=str, keep_default_na=False) for f in parts]
    d = pd.concat(frames).drop_duplicates(["prompt_id", "model"], keep="last")
    d.to_csv(base, index=False)
    print(f"combined {len(parts)} part files into {base.name}: {len(d)} rows")


def merge_corpus() -> Path:
    """v1 with the repaired rows swapped in and M5 restored -> responses_v2.csv."""
    v1 = pd.read_csv(ROOT / "data" / "responses_v1.csv", dtype=str)
    v1["source"] = "chat_app_v1"
    v1["genre"] = "prose"
    rep = pd.read_csv(ROOT / "data" / "responses_v2_repair.csv", dtype=str)
    key = set(zip(rep["prompt_id"], rep["model"]))
    keep = v1[[k not in key for k in zip(v1["prompt_id"], v1["model"])]]
    parts = [keep, rep]
    exp = ROOT / "data" / "responses_v2_expand.csv"
    if exp.exists():
        parts.append(pd.read_csv(exp, dtype=str))
    v2 = pd.concat(parts, ignore_index=True).sort_values(["prompt_id", "model"])
    out = ROOT / "data" / "responses_v2.csv"
    v2.to_csv(out, index=False)
    print(f"wrote {out}: {len(v2)} rows, {v2['prompt_id'].nunique()} prompts")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--expand", action="store_true")
    ap.add_argument("--resample", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", default="", help="comma list; default all five")
    ap.add_argument("--api-v1", action="store_true",
                    help="API answers to the 39 v1 prompts, for an all-API corpus")
    ap.add_argument("--build-api120", action="store_true",
                    help="data/responses_api120.csv: every prompt, every model, API only")
    args = ap.parse_args()
    MODELS_ONLY[:] = [m for m in args.models.split(",") if m]

    if args.repair:
        collect(REPAIR_ROWS, ROOT / "data" / "responses_v2_repair.csv", "api_repair", args.dry_run)
    if args.expand:
        P = prompts()
        v1_ids = set(pd.read_csv(ROOT / "data" / "responses_v1.csv", dtype=str)["prompt_id"])
        pairs = [(p, m) for p in P.index if p not in v1_ids for m in FRONTIER.names]
        collect(pairs, ROOT / "data" / "responses_v2_expand.csv", "api_expand", args.dry_run)
    if args.api_v1:
        v1_ids = sorted(pd.read_csv(ROOT / "data" / "responses_v1.csv", dtype=str)["prompt_id"].unique())
        pairs = [(p, m) for p in v1_ids for m in FRONTIER.names]
        collect(pairs, ROOT / "data" / "responses_v2_apiv1.csv", "api_v1", args.dry_run)
    if args.build_api120:
        frames = []
        for base in ("responses_v2_apiv1.csv", "responses_v2_expand.csv"):
            combine_parts(ROOT / "data" / base)
            frames.append(pd.read_csv(ROOT / "data" / base, dtype=str, keep_default_na=False))
        d = pd.concat(frames).drop_duplicates(["prompt_id", "model"], keep="last")
        d = d.sort_values(["prompt_id", "model"])
        out = ROOT / "data" / "responses_api120.csv"
        d.to_csv(out, index=False)
        full = d.groupby("prompt_id")["model"].nunique()
        print(f"wrote {out}: {len(d)} rows, {int((full == len(FRONTIER.names)).sum())} complete prompts")
    if args.merge:
        combine_parts(ROOT / "data" / "responses_v2_repair.csv")
        combine_parts(ROOT / "data" / "responses_v2_expand.csv")
        merge_corpus()
    if args.resample:
        corpus = ROOT / "data" / "responses_v2.csv"
        if not corpus.exists():
            corpus = ROOT / "data" / "responses_v1.csv"
        c = pd.read_csv(corpus, dtype=str)
        pairs = sorted(set(zip(c["prompt_id"], c["model"])))
        collect(pairs, FRONTIER.results_dir / "resample_s2.csv", "api_resample", args.dry_run)


if __name__ == "__main__":
    main()
