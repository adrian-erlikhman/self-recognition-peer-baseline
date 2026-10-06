"""CompLLM revision: run one judging condition on one panel.

    python src/revision/run_condition.py --panel openweight --condition lineup
    python src/revision/run_condition.py --panel openweight --condition binary
    python src/revision/run_condition.py --panel frontier --condition binary --judges Claude,GPT
    python src/revision/run_condition.py --panel frontier --condition shuffle_lineup --dry-run

Conditions are described in conditions.py. Output is one JSON line per item
in results/revision/<panel>/<condition>.jsonl; re-running skips items already
there, so a run can be stopped and resumed. The frontier panel needs a working
OPENROUTER_API_KEY in .env and archives every raw response under
results/revision/frontier/raw/.

Which texts are judged
----------------------
openweight: data/openweight/responses_ow.csv (all genres; --genres narrows).
frontier:   data/responses_v1.csv minus EXCLUDE_PROMPTS, or --corpus for a
            re-collected corpus in the same format.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import conditions as C  # noqa: E402
from backends import make_backend  # noqa: E402
from config import EXCLUDE_PROMPTS, ROOT  # noqa: E402
from panels import PANELS  # noqa: E402

CONDITIONS = ["lineup", "single", "binary", "onpolicy", "onpolicy_resample",
              "shuffle_lineup", "shuffle_binary", "quality"]
CHUNK = 40
GEN_TOKENS = {"lineup": 500, "shuffle_lineup": 500, "single": 160}


def load_corpus(panel, corpus: str | None, genres: str) -> pd.DataFrame:
    path = Path(corpus) if corpus else panel.corpus_csv
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if path.name == "responses_v1.csv":
        df = df[~df["prompt_id"].isin(EXCLUDE_PROMPTS)]
    if "genre" not in df:
        df["genre"] = "prose"
    if genres:
        df = df[df["genre"].isin(genres.split(","))]
    # Keep only prompts every author answered, so each lineup is full.
    full = df.groupby("prompt_id")["model"].nunique()
    df = df[df["prompt_id"].isin(full[full == df["model"].nunique()].index)]
    return df


def task_texts() -> dict[str, str]:
    """Prompt text by id, for conditions that show the task (on-policy,
    quality). Frontier judges never saw the task in the submitted study; the
    original lineup and single conditions still do not show it."""
    out = {}
    for f in ("prompts_v2.csv", "prompts_bank.csv"):
        p = ROOT / "data" / f
        if p.exists():
            d = pd.read_csv(p, dtype=str)
            out.update(dict(zip(d["prompt_id"], d["prompt_text"])))
    return out


def build_items(panel, cond: str, df: pd.DataFrame, judges: list[str],
                second: pd.DataFrame | None) -> list[dict]:
    """Every (judge, stimulus) the condition needs, with its prompt."""
    text = {(r.prompt_id, r.model): r.text for r in df.itertuples()}
    genre = {r.prompt_id: r.genre for r in df.itertuples()}
    prompts = sorted(df["prompt_id"].unique())
    tasks = task_texts() if cond in ("onpolicy", "onpolicy_resample", "quality") else {}
    items = []

    if cond in ("lineup", "shuffle_lineup"):
        for row in C.lineup_design(prompts, panel.names):
            if row["judge"] not in judges:
                continue
            st = {}
            for s, a in row["slot_to_author"].items():
                t = text[(row["prompt_id"], a)]
                if cond == "shuffle_lineup":
                    t = C.shuffle_words(t, C.stable_seed(row["prompt_id"], a))
                st[s] = t
            items.append({**row, "genre": genre[row["prompt_id"]],
                          "conv": [{"role": "user", "content": C.lineup_prompt(panel.names, st)}]})
        return items

    for (pid, author), t in sorted(text.items()):
        for judge in judges:
            base = {"prompt_id": pid, "genre": genre[pid], "judge": judge,
                    "true_author": author}
            if cond == "single":
                conv = [{"role": "user", "content": C.single_prompt(panel.names, t)}]
                items.append({**base, "item_id": f"{pid}__{author}__{judge}", "conv": conv})
            elif cond in ("binary", "shuffle_binary"):
                if cond == "shuffle_binary":
                    t2 = C.shuffle_words(t, C.stable_seed(pid, author))
                else:
                    t2 = t
                conv = [{"role": "user", "content": C.binary_prompt(t2)}]
                items.append({**base, "item_id": f"{pid}__{author}__{judge}", "conv": conv})
            elif cond == "quality":
                conv = [{"role": "user", "content": C.quality_prompt(tasks[pid], t)}]
                items.append({**base, "item_id": f"{pid}__{author}__{judge}", "conv": conv})
            elif cond == "onpolicy":
                # The judge's own turn is its corpus answer, verbatim; when the
                # candidate is also the judge's, the two are identical.
                own = text[(pid, judge)]
                conv = C.onpolicy_conv(tasks[pid], own, t)
                items.append({**base, "item_id": f"{pid}__{author}__{judge}", "conv": conv})
            elif cond == "onpolicy_resample":
                if second is None or (pid, judge) not in second:
                    continue
                conv = C.onpolicy_conv(tasks[pid], second[(pid, judge)], t)
                items.append({**base, "item_id": f"{pid}__{author}__{judge}", "conv": conv})
    return items


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("ok"):
            out.add(r["item_id"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=sorted(PANELS))
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--judges", default="")
    ap.add_argument("--genres", default="")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file, e.g. frontiercorpus when the "
                         "open-weight judges rate the frontier corpus")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    panel = PANELS[args.panel]
    judges = [j for j in args.judges.split(",") if j] or list(panel.names)
    df = load_corpus(panel, args.corpus or None, args.genres)
    if args.limit:
        keep = sorted(df["prompt_id"].unique())[: args.limit]
        df = df[df["prompt_id"].isin(keep)]

    second = None
    if args.condition == "onpolicy_resample":
        if panel.key == "openweight":
            p2 = ROOT / "data" / "openweight" / "responses_ow_s2.csv"
        else:
            p2 = panel.results_dir / "resample_s2.csv"
        if p2.exists():
            d2 = pd.read_csv(p2, dtype=str, keep_default_na=False)
            second = {(r.prompt_id, r.model): r.text for r in d2.itertuples()}
        else:
            raise SystemExit(f"{p2} is missing. For the open-weight panel run "
                             "generate_corpus.py --sample 2; for the frontier panel "
                             "run collect_frontier.py --resample first.")

    items = build_items(panel, args.condition, df, judges, second)
    out = panel.results_dir / f"{args.condition}{'_' + args.tag if args.tag else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(out)
    todo = [i for i in items if i["item_id"] not in finished]
    print(f"{panel.key}/{args.condition}: {len(items)} items, {len(finished)} done, "
          f"{len(todo)} to run", flush=True)
    if args.dry_run:
        if items:
            ex = items[0]
            print(json.dumps({k: v for k, v in ex.items() if k != "conv"}, indent=1))
            for m in ex["conv"]:
                print(f"--- {m['role']} ---\n{m['content'][:1800]}")
        chars = sum(sum(len(m["content"]) for m in i["conv"]) for i in todo)
        print(f"est. input tokens {chars / 4:,.0f}")
        return

    for judge in judges:
        mine = [i for i in todo if i["judge"] == judge]
        if not mine:
            continue
        t0 = time.time()
        be = make_backend(panel, judge)
        print(f"  {judge}: model ready in {time.time() - t0:.0f}s, {len(mine)} items", flush=True)
        # Written in chunks, so a crash mid-judge loses at most one chunk.
        for c0 in range(0, len(mine), CHUNK):
            run_chunk(panel, args.condition, be, judge, mine[c0:c0 + CHUNK], out)
        be.close()
        del be
        print(f"  {judge}: done in {time.time() - t0:.0f}s", flush=True)


def run_chunk(panel, cond: str, be, judge: str, mine: list[dict], out: Path) -> None:
        convs = [i["conv"] for i in mine]
        ids = [f"{cond}__{i['item_id']}" for i in mine]
        kw = {"ids": ids} if panel.backend == "openrouter" else {}
        recs = []
        if cond in ("lineup", "shuffle_lineup", "single"):
            texts = be.chat(convs, GEN_TOKENS[cond], temperature=0.0,
                            progress=f"{judge}/{cond}", **kw)
            for it, t in zip(mine, texts):
                if cond == "single":
                    parsed, review = C.parse_single(panel, t)
                else:
                    parsed, review = C.parse_lineup(panel, t)
                recs.append({**{k: v for k, v in it.items() if k != "conv"},
                             "parsed": parsed, "needs_review": review,
                             "raw_text": t, "ok": True})
        elif cond == "quality":
            opts = [str(d) for d in range(1, 10)]
            probs = be.choice_probs(convs, opts, **kw)
            for it, p in zip(mine, probs):
                ev = sum(int(o) * p[o] for o in opts)
                recs.append({**{k: v for k, v in it.items() if k != "conv"},
                             "probs": {o: p[o] for o in opts}, "mass": p["_mass"],
                             "stepped": p.get("_stepped"),
                             "expected": ev, "argmax": max(opts, key=lambda o: p[o]),
                             "ok": True})
        else:  # binary family
            probs = be.choice_probs(convs, ["Yes", "No"], **kw)
            for it, p in zip(mine, probs):
                recs.append({**{k: v for k, v in it.items() if k != "conv"},
                             "p_yes": p["Yes"], "mass": p["_mass"],
                             "says_yes": bool(p["Yes"] > 0.5),
                             "raw_text": p.get("_text"), "ok": True})
        with out.open("a", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
