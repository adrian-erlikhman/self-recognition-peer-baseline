"""CompLLM revision: each open-weight judge's likelihood of every text (A1).

Wataoka et al. (2024) propose that self-preference follows familiarity: a
judge favours text it finds likely. For open-weight judges that can be read
off directly. For every (scorer, response) pair we record the mean per-token
log-probability of the response as the scorer's reply to the task
("conditional"), and of the response with no task before it
("unconditional").

Two uses downstream (analyze_openweight.py):
  1. A likelihood attributor: for each lineup, name as "self" the response
     the judge finds most likely, relative to how likely the other judges
     find it. This is the ceiling a judge could reach from its own
     probabilities, which the verbal judgment can be compared with.
  2. The familiarity test: does a judge name itself more often on text it
     finds more likely, holding the true author fixed?

    python src/revision/score_likelihood.py
    python src/revision/score_likelihood.py --judges Qwen --genres prose
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import make_backend  # noqa: E402
from panels import OPENWEIGHT  # noqa: E402
from run_condition import load_corpus, task_texts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="")
    ap.add_argument("--genres", default="")
    ap.add_argument("--corpus", default="")
    args = ap.parse_args()

    panel = OPENWEIGHT
    df = load_corpus(panel, args.corpus or None, args.genres)
    tasks = task_texts()
    out = panel.results_dir / "likelihood.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    have = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["scorer"])
    judges = [j for j in args.judges.split(",") if j] or panel.names

    for scorer in judges:
        if scorer in set(have["scorer"]):
            print(f"{scorer}: already scored")
            continue
        t0 = time.time()
        be = make_backend(panel, scorer)
        rows = list(df.itertuples())
        prefixes = [[{"role": "user", "content": tasks[r.prompt_id]}] for r in rows]
        cond = be.score(prefixes, [r.text for r in rows])
        unc = be.score([[] for _ in rows], [r.text for r in rows])
        recs = []
        for r, lc, lu in zip(rows, cond, unc):
            recs.append({"prompt_id": r.prompt_id, "genre": r.genre, "author": r.model,
                         "scorer": scorer, "n_tokens": len(lc),
                         "mean_lp_cond": sum(lc) / max(len(lc), 1),
                         "mean_lp_uncond": sum(lu) / max(len(lu), 1),
                         "sum_lp_cond": sum(lc)})
        be.close()
        del be
        part = pd.DataFrame(recs)
        part.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"{scorer}: {len(recs)} texts in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
