"""CompLLM: reproduce every result from the frozen corpus, in one command.

    python src/run_all.py            # everything
    python src/run_all.py --quick    # fewer permutations, as a smoke test
    python src/run_all.py --list     # show the pipeline without running

No API key and no network are required. Every judge response ever collected is
archived under raw/, and the corpus is SHA-256 pinned in config.py, so this
runs end to end from a fresh clone.

It does not re-collect data; that costs money and needs credentials (see
README.md). This only re-derives results from data already in the repo.

The full run is dominated by permutation testing (Engine A at 1000
permutations, Engine B at 200) and takes tens of minutes. --quick drops both
to 50 as a wiring check; verify_paper is skipped there because it asserts the
permutation count the paper reports.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def steps(quick: bool) -> list[tuple[str, list[str], str]]:
    a_perm = "50" if quick else "1000"
    b_perm = "50" if quick else "200"
    plan: list[tuple[str, list[str], str]] = [
        ("test_harness",
         ["src/test_harness.py"],
         "24 offline checks - corpus hash, Latin-square balance, feature guards"),
        ("engine_a",
         ["src/engine_a.py", "--permutations", a_perm],
         f"author identification + ablations + permutation test (n={a_perm})"),
        ("engine_b",
         ["src/engine_b.py", "--permutations", b_perm,
          "--tex", "paper/numbers.tex"],
         f"judge-side analysis, both conditions (n={b_perm}); emits paper macros"),
        ("cross_attribution",
         ["src/cross_attribution.py", "--json"],
         "who gets mistaken for whom -> results/cross_attribution.json"),
        ("analyze_reasons",
         ["src/analyze_reasons.py"],
         "what judges SAY they used -- keyword analysis of 1,900 rationales"),
        ("make_figure",
         ["src/make_figure.py"],
         "the main two-condition figure"),
        ("make_figures_supp",
         ["src/make_figures_supp.py"],
         "the nine supplementary figures"),
        ("costs",
         ["src/costs.py"],
         "spend ledger, rebuilt from raw/"),
        ("make_report",
         ["src/make_report.py"],
         "docs/RESULTS.md, the end-to-end compendium"),
        # TACL revision. These read archived data only; the open-weight panel's
        # model runs are src/revision/run_openweight.py (GPU, hours) and are
        # not repeated here.
        ("stats_revision",
         ["src/stats_revision.py"],
         "prompt-clustered tests, CIs, Holm families, skill-adjusted baseline"),
        ("rationale_codes",
         ["src/rationale_codes.py"],
         "coded cue types in the 1,900 rationales, and whether cited cues are true"),
        ("generative_match",
         ["src/generative_match.py"],
         "does similarity to the judge's own answer predict it naming itself"),
        ("analyze_openweight",
         ["src/revision/analyze_panel.py", "--panel", "openweight"],
         "every open-weight condition that has been run"),
        ("paper_numbers",
         ["paper_tacl/make_numbers.py"],
         "numbers_rev.tex and the tables of the TACL draft"),
    ]
    # verify_paper asserts the permutation count the paper reports, among 180
    # other figures, so under --quick it would fail on the one thing --quick
    # deliberately changes.
    if not quick:
        plan.append(
            ("verify_paper",
             ["src/verify_paper.py"],
             "every number the paper states, against what was just regenerated"))
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="50 permutations everywhere - smoke test only")
    ap.add_argument("--list", action="store_true",
                    help="print the pipeline and exit")
    args = ap.parse_args()

    plan = steps(args.quick)

    if args.list:
        print("CompLLM pipeline:\n")
        for i, (name, cmd, why) in enumerate(plan, 1):
            print(f"  {i:2d}. {name:20s} {why}")
            print(f"     {' '.join(cmd)}")
        return 0

    if args.quick:
        print("!! --quick: 50 permutations. Wiring check only; "
              "verify_paper is skipped.\n")

    bar = "=" * 74
    results: list[tuple[str, bool, float]] = []
    t_all = time.monotonic()

    for i, (name, cmd, why) in enumerate(plan, 1):
        print(f"\n{bar}\n[{i}/{len(plan)}] {name} - {why}\n{bar}")
        t0 = time.monotonic()
        proc = subprocess.run([PY, *cmd], cwd=ROOT)
        dt = time.monotonic() - t0
        ok = proc.returncode == 0
        results.append((name, ok, dt))
        if not ok:
            print(f"\n!! {name} FAILED (exit {proc.returncode}) after {dt:.1f}s")
            print("!! Stopping - later steps read this one's output.")
            break

    print(f"\n{bar}\nSUMMARY\n{bar}")
    for name, ok, dt in results:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:20s} {dt:7.1f}s")
    total = time.monotonic() - t_all
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n  total {total:.1f}s")

    if failed:
        print(f"\n  FAILED: {', '.join(failed)}")
        return 1

    if len(results) < len(plan):
        print("\n  incomplete - not all steps ran")
        return 1

    print("""
  All results regenerated from the frozen corpus.

  Outputs:
    results/engine_a_results.json    results/engine_b_results.json
    results/cross_attribution.json   results/engine_a_features.csv
    results/costs.csv                docs/RESULTS.md
    paper/numbers.tex                paper/table_*.tex
    paper/figures/*.pdf + *.png""")

    if args.quick:
        print("\n  verify_paper did not run (--quick); re-run without it before\n"
              "  trusting any number above.")
    else:
        print("\n  verify_paper checked every figure the paper states against\n"
              "  the JSON it was just regenerated from.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
