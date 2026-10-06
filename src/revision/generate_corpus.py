"""CompLLM revision: write the open-weight corpus.

Each open-weight model answers every prompt in data/prompts_v2.csv once per
sample index. Sample 1 is the corpus the judges attribute; sample 2 is a
second, independent answer to the same prompt, used for the within-model
variance estimate and as the judge's own turn in `onpolicy_resample`.

Decoding: temperature 0.7, top-p 0.95, a fixed seed per batch. The frontier
corpus came from the vendors' chat apps at their default settings, which also
sample; greedy decoding would make every model's prose unnaturally repetitive.

    python src/revision/generate_corpus.py                  # all models, sample 1
    python src/revision/generate_corpus.py --sample 2
    python src/revision/generate_corpus.py --models Qwen --limit 5

Resumable: (prompt, model, sample) rows already in the CSV are skipped. The
CSV gets a SHA-256 in data/openweight/SHA256SUMS once complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import make_backend  # noqa: E402
from config import ROOT  # noqa: E402
from panels import OPENWEIGHT  # noqa: E402

PROMPTS_CSV = ROOT / "data" / "prompts_v2.csv"
OUT_DIR = ROOT / "data" / "openweight"
MAX_NEW = {"prose": 1100, "code": 900, "short": 300, "math": 900}
TEMPERATURE, TOP_P = 0.7, 0.95
FIELDS = ["prompt_id", "genre", "tier", "model", "model_id", "sample", "seed",
          "n_tokens", "hit_limit", "word_count", "text"]


def out_path(sample: int) -> Path:
    return OUT_DIR / ("responses_ow.csv" if sample == 1 else f"responses_ow_s{sample}.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--models", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only")
    ap.add_argument("--genres", default="", help="e.g. prose,code")
    args = ap.parse_args()

    prompts = pd.read_csv(PROMPTS_CSV, dtype=str)
    if args.genres:
        prompts = prompts[prompts["genre"].isin(args.genres.split(","))]
    if args.limit:
        prompts = prompts.head(args.limit)
    models = [m for m in (args.models.split(",") if args.models else OPENWEIGHT.names)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = out_path(args.sample)
    done = set()
    if path.exists():
        prev = pd.read_csv(path, dtype=str)
        done = set(zip(prev["prompt_id"], prev["model"]))
    new_file = not path.exists()

    for name in models:
        todo = prompts[[(p, name) not in done for p in prompts["prompt_id"]]]
        if todo.empty:
            print(f"{name}: nothing to do")
            continue
        print(f"{name}: loading {OPENWEIGHT.model_ids[name]}", flush=True)
        t0 = time.time()
        be = make_backend(OPENWEIGHT, name)
        print(f"{name}: loaded in {time.time() - t0:.0f}s; {len(todo)} prompts", flush=True)
        for genre, grp in todo.groupby("genre"):
            convs = [[{"role": "user", "content": t}] for t in grp["prompt_text"]]
            seed = 1000 * args.sample + OPENWEIGHT.names.index(name)
            texts = be.chat(convs, MAX_NEW[genre], temperature=TEMPERATURE,
                            top_p=TOP_P, seed=seed, progress=f"{name}/{genre}")
            with path.open("a", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if new_file:
                    w.writeheader()
                    new_file = False
                for row, text in zip(grp.itertuples(), texts):
                    n_tok = len(be.tok(text, add_special_tokens=False)["input_ids"])
                    w.writerow({
                        "prompt_id": row.prompt_id, "genre": genre, "tier": row.tier,
                        "model": name, "model_id": OPENWEIGHT.model_ids[name],
                        "sample": args.sample, "seed": seed, "n_tokens": n_tok,
                        "hit_limit": int(n_tok >= MAX_NEW[genre] - 5),
                        "word_count": len(text.split()), "text": text,
                    })
        be.close()
        del be
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT_DIR / f"SHA256SUMS_s{args.sample}").write_text(f"{digest}  {path.name}\n")
    print(f"wrote {path} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
