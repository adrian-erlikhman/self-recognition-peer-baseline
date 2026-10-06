"""CompLLM revision: the whole open-weight study, in order.

    .venv/Scripts/python src/revision/run_openweight.py            # everything
    .venv/Scripts/python src/revision/run_openweight.py --list
    .venv/Scripts/python src/revision/run_openweight.py --from lineup

Each step is resumable, so re-running after an interruption continues where
it stopped. On one RTX 5060 Laptop (8 GB) the full run takes several hours;
the corpus and the lineups dominate.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

# The word-shuffle conditions (conditions.shuffle_words) were dropped on
# 1 Oct 2026: shuffling tests whether a judge's self-recognition survives the
# loss of word order, and no open-weight judge recognises itself to begin with.
STEPS = [
    ("corpus",            ["generate_corpus.py", "--sample", "1"]),
    ("binary",            ["run_condition.py", "--panel", "openweight", "--condition", "binary"]),
    ("lineup",            ["run_condition.py", "--panel", "openweight", "--condition", "lineup"]),
    ("single",            ["run_condition.py", "--panel", "openweight", "--condition", "single"]),
    ("likelihood",        ["score_likelihood.py"]),
    ("onpolicy",          ["run_condition.py", "--panel", "openweight", "--condition", "onpolicy"]),
    ("quality",           ["run_condition.py", "--panel", "openweight", "--condition", "quality"]),
    # The open-weight panel as an independent quality panel for the frontier
    # corpus (A1's quality covariate for the submitted study).
    ("quality_frontier",  ["run_condition.py", "--panel", "openweight", "--condition", "quality",
                           "--corpus", str(HERE.parent.parent / "data" / "responses_v1.csv"),
                           "--tag", "frontiercorpus"]),
    ("quality_fr_stats",  ["quality_frontier.py"]),
    ("corpus_s2",         ["generate_corpus.py", "--sample", "2"]),
    ("onpolicy_resample", ["run_condition.py", "--panel", "openweight", "--condition", "onpolicy_resample"]),
    ("analyze",           ["analyze_panel.py", "--panel", "openweight"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--from", dest="start", default="")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    if args.list:
        for name, cmd in STEPS:
            print(f"{name:18s} {' '.join(cmd)}")
        return
    names = [n for n, _ in STEPS]
    todo = STEPS[names.index(args.start):] if args.start else STEPS
    if args.only:
        todo = [s for s in STEPS if s[0] in args.only.split(",")]
    for name, cmd in todo:
        t0 = time.time()
        print(f"=== {name}: {' '.join(cmd)}", flush=True)
        r = subprocess.run([PY, str(HERE / cmd[0]), *cmd[1:]])
        print(f"=== {name}: exit {r.returncode} after {time.time() - t0:.0f}s", flush=True)
        if r.returncode:
            raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
