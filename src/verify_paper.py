#!/usr/bin/env python3
"""Check every number the paper states against the results it came from.

The paper quotes roughly a hundred figures across six tables, eight appendices
and a figure caption. Most reach LaTeX as macros from engine_b.py --tex, so
they cannot drift. The rest -- anything quoted in running prose, and every
number in a table typed by hand -- can, and a wrong digit in a submitted paper
is not recoverable.

So the claims are written out here, each against the path in the results JSON
it must equal, and this script re-reads the JSON and compares. Run it before
every build:

    python src/verify_paper.py           # 0 if the paper is sound
    python src/verify_paper.py --verbose # print every check, not just failures

Tolerances are one unit in the last place the paper prints, so 86.8 passes on
0.868421 and fails on 0.8600. Where the paper truncates rather than rounds the
tolerance is widened to 0.1 and the entry says so.

A value that rounds to exactly what the paper prints sits on the tolerance
boundary, where binary floats cannot be trusted: 0.03895 - 0.0389 evaluates to
5.000000000000143e-05, which is greater than 5e-05. So the comparison carries a
relative slack, and a value is also accepted whenever it formats to the same
string the paper shows.

Adding a claim is one line. If a number is in the paper and not in this file,
nobody is checking it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FAMILIES  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODELS = ["GPT", "Claude", "Gemini", "Grok", "DeepSeek"]

# Appendix B quotes these as sums over Table 3, so they are checked against the
# importances rather than stored -- if a family sum drifts, one of the 18
# per-feature checks above will have caught the cause.
FAMILY_SUMS = {"length": 0.5525, "lexical": 0.2157,
               "readability": 0.1263, "punctuation": 0.1054}


def dig(blob: dict, path: str):
    """Walk a dotted path. `a.b[2].c` and list-of-dicts keyed by name both work."""
    cur = blob
    for part in path.split("."):
        if part.endswith("]") and "[" in part:
            part, _, idx = part.partition("[")
            cur = cur[part] if part else cur
            idx = idx.rstrip("]")
            cur = cur[int(idx)] if idx.lstrip("-").isdigit() else _by_name(cur, idx)
        else:
            cur = cur[part]
    return cur


def _matches(got: float, want: float, tol: float) -> bool:
    """Within tolerance, or printing identically to what the paper shows."""
    if abs(got - want) <= tol * (1 + 1e-9) + 1e-12:
        return True
    places = len(str(want).partition(".")[2])
    return f"{got:.{places}f}" == f"{want:.{places}f}"


def _by_name(rows: list[dict], name: str):
    for r in rows:
        if name in (r.get("judge"), r.get("model")):
            return r
    raise KeyError(f"no row named {name!r}")


# (section, claim as printed, results path, expected, scale, tolerance)
# scale 100 means the JSON holds a proportion and the paper prints a percentage.
def claims() -> list[tuple]:
    C: list[tuple] = []
    A, B = "engine_a", "engine_b"

    # ---- Abstract and 3.1: the stylometric control.
    C += [
        ("3.1", "classifier accuracy 86.3%", A, "accuracy.mean", 86.3, 100, 0.05),
        ("3.1", "fold mean 85.8", A, "headline.UNTRUNCATED.FULL.rf_mean", 85.8, 100, 0.05),
        ("3.1", "fold s.d. 9.2", A, "headline.UNTRUNCATED.FULL.rf_sd", 9.2, 100, 0.05),
        ("3.1", "ungrouped fold mean 85.3", A, "leakage.ungrouped", 85.3, 100, 0.05),
        ("3.1", "length share of margin 33%", A, "length_share_of_signal", 33.0, 100, 0.5),
        ("3.1", "length importance share 63.1%", A, "length_proxy_importance_share", 63.1, 100, 0.05),
        ("3.1", "length-free RF 63.9%", A, "headline.UNTRUNCATED.LENGTH-FREE.rf_mean", 63.9, 100, 0.05),
        ("3.1", "length-free LR 71.7%", A, "headline.UNTRUNCATED.LENGTH-FREE.logreg_mean", 71.7, 100, 0.05),
        ("3.1", "permutations run", A, "permutation.n", 1000, 1, 0),
        ("C", "permutation null mean 20.2", A, "permutation.null_mean", 20.2, 100, 0.05),
        ("C", "permutation null s.d. 3.6", A, "permutation.null_sd", 3.6, 100, 0.05),
        ("A", "mean word count, DeepSeek 688.6", A, "length_confound.mean_word_count.DeepSeek", 688.6, 1, 0.05),
        ("A", "mean word count, Grok 324.0", A, "length_confound.mean_word_count.Grok", 324.0, 1, 0.05),
    ]
    for m, f1 in [("Grok", .935), ("DeepSeek", .883), ("GPT", .842),
                  ("Gemini", .835), ("Claude", .817)]:
        C.append(("3.1", f"per-class F1 {m} {f1}", A, f"per_class.{m}.f1-score", f1, 1, 0.0005))

    # ---- Table 4, the ablation grid.
    for corpus, feats, rf, lr in [
        ("UNTRUNCATED", "FULL", 85.8, 87.0),
        ("UNTRUNCATED", "LENGTH-FREE", 63.9, 71.7),
        ("UNTRUNCATED", "LENGTH-ONLY", 60.4, 62.9),
        ("TRUNCATED-TO-GROUP-MIN", "FULL", 67.0, 74.8),
        ("TRUNCATED-TO-GROUP-MIN", "LENGTH-FREE", 60.2, 66.6),
        ("TRUNCATED-TO-GROUP-MIN", "LENGTH-ONLY", 22.1, 25.2),
    ]:
        C.append(("T4", f"{corpus}/{feats} RF", A, f"headline.{corpus}.{feats}.rf_mean", rf, 100, 0.05))
        C.append(("T4", f"{corpus}/{feats} LR", A, f"headline.{corpus}.{feats}.logreg_mean", lr, 100, 0.05))

    # ---- Table 3, feature importances.
    for f, imp in [("sent_count", .1267), ("avg_para_len", .0927),
                   ("avg_sent_len", .0894), ("word_count", .0881),
                   ("sent_len_sd", .0785), ("para_count", .0771),
                   ("hapax_rate", .0599), ("ttr", .0589),
                   ("avg_word_len", .0483), ("comma_per_100w", .0458),
                   ("gunning_fog", .0439), ("flesch_kincaid_grade", .0435),
                   ("flesch_reading_ease", .0389), ("semicolon_per_100w", .0316),
                   ("hedge_per_100w", .0264), ("passive_rate", .0223),
                   ("colon_per_100w", .0215), ("emdash_per_100w", .0065)]:
        C.append(("T3", f"importance {f}", A, f"feature_importance.{f}", imp, 1, 0.00005))

    # ---- Headline judge-side rates.
    C += [
        ("3.2", "lineup overall accuracy 33.7%", B, "lineup.accuracy", 33.7, 100, 0.05),
        ("3.3", "single overall accuracy 30.6%", B, "single.accuracy", 30.6, 100, 0.05),
        ("3.3", "single top-2 concentration 99.3%", B, "single.blame.concentration_top2", 99.3, 100, 0.05),
        ("A", "lineup slot chi2 = 0.00", B, "lineup.slot_bias.chi2", 0.0, 1, 0.005),
        ("3.2", "sessions naming 5 distinct 85.3%", B, "lineup.distinct_names.all_distinct_share", 85.3, 100, 0.05),
        ("F", "lineup blame chi2 1.3", B, "lineup.blame.chi2_judge_averaged", 1.3, 1, 0.05),
        ("F", "single blame chi2 280.7", B, "single.blame.chi2_judge_averaged", 280.7, 1, 0.05),
        ("F", "lineup confidence rho +0.146", B, "lineup.confidence_calibration.spearman_rho", 0.146, 1, 0.0005),
        ("F", "single confidence rho +0.019", B, "single.confidence_calibration.spearman_rho", 0.019, 1, 0.0005),
        ("F", "pooled non-self, lineup 32.6%", B, "lineup.cross_attribution.pooled_non_self_accuracy", 32.6, 100, 0.05),
        ("F", "pooled non-self, single 29.2%", B, "single.cross_attribution.pooled_non_self_accuracy", 29.2, 100, 0.05),
        ("F", "single errors n = 659", B, "single.cross_attribution.n_wrong", 659, 1, 0),
        ("3.4", "mech RF lineup 29.7%", B, "lineup.mechanism_test.random_forest.accuracy", 29.7, 100, 0.05),
        ("3.4", "mech LR lineup 31.9%", B, "lineup.mechanism_test.logistic_regression.accuracy", 31.9, 100, 0.05),
        ("3.4", "identity baseline lineup 38.9%", B, "lineup.mechanism_test.identity_only_baseline", 38.9, 100, 0.05),
        ("3.4", "mech RF single 63.0%", B, "single.mechanism_test.random_forest.accuracy", 63.0, 100, 0.05),
        ("3.4", "mech LR single 61.0%", B, "single.mechanism_test.logistic_regression.accuracy", 61.0, 100, 0.05),
        ("3.4", "identity baseline single 64.9%", B, "single.mechanism_test.identity_only_baseline", 64.9, 100, 0.05),
        ("G", "Sonnet-dropped Claude self 84.4%", B, "lineup.sonnet_sensitivity.Claude.sonnet_dropped.rate", 84.4, 100, 0.05),
    ]

    # ---- Table 1 and Table 6: self rate, false alarm, peer baseline, advantage.
    #      Table 6's +33.5 for GPT is truncated from 33.55, hence the wider band.
    for cond, rows in [
        ("lineup", [("Claude", 86.8, 9.2, 53.9, 32.9, .05),
                    ("GPT", 57.9, 10.5, 30.3, 27.6, .05),
                    ("DeepSeek", 21.1, 17.8, 21.7, -0.7, .05),
                    ("Gemini", 18.4, 20.4, 28.3, -9.9, .05),
                    ("Grok", 5.3, 20.4, 28.9, -23.7, .05)]),
        ("single", [("GPT", 94.7, 52.0, 61.2, 33.5, .1),
                    ("Claude", 86.8, 8.6, 82.2, 4.6, .05),
                    ("Gemini", 0.0, 0.0, 1.3, -1.3, .05),
                    ("Grok", 0.0, 0.0, 0.7, -0.7, .05),
                    ("DeepSeek", 0.0, 0.0, 0.7, -0.7, .05)]),
    ]:
        tab = "T1" if cond == "lineup" else "T6"
        for m, self_r, fa, peer, adv, tol in rows:
            C += [
                (tab, f"{cond} {m} self rate", B, f"{cond}.self_recognition[{m}].rate", self_r, 100, 0.05),
                (tab, f"{cond} {m} false alarm", B, f"{cond}.signal_detection[{m}].false_alarm_rate", fa, 100, 0.05),
                (tab, f"{cond} {m} peer baseline", B, f"{cond}.self_advantage[{m}].others_rate", peer, 100, 0.05),
                (tab, f"{cond} {m} self-advantage", B, f"{cond}.self_advantage[{m}].advantage_pp", adv, 1, tol),
            ]

    # ---- Table 5, the leak. Degenerate cells are asserted degenerate.
    for cond, rows in [
        ("lineup", [("Claude", 4.06, 58.2), ("GPT", 2.42, 11.3),
                    ("DeepSeek", 0.24, 1.27), ("Gemini", -0.09, 0.92),
                    ("Grok", -1.33, 0.26)]),
        ("single", [("Claude", 4.14, 62.9), ("GPT", 2.60, 13.5)]),
    ]:
        for m, leak, odds in rows:
            C += [
                ("T5", f"{cond} {m} leak", B, f"{cond}.signal_detection[{m}].leak", leak, 1, 0.005),
                ("T5", f"{cond} {m} odds ratio", B, f"{cond}.signal_detection[{m}].odds_ratio", odds, 1, 0.05),
            ]

    # ---- Table 2, non-self accuracy.
    for cond, rows in [
        ("lineup", [("GPT", 39.5, 43.2), ("Claude", 25.7, 37.9),
                    ("Gemini", 40.8, 36.3), ("Grok", 30.9, 25.8),
                    ("DeepSeek", 26.3, 25.3)]),
        ("single", [("GPT", 24.3, 38.4), ("Claude", 27.0, 38.9),
                    ("Gemini", 32.9, 26.3), ("Grok", 33.6, 26.8),
                    ("DeepSeek", 28.3, 22.6)]),
    ]:
        for m, ns, ov in rows:
            base = f"{cond}.cross_attribution.per_judge.{m}"
            C += [("T2", f"{cond} {m} non-self", B, f"{base}.non_self_accuracy", ns, 100, 0.05),
                  ("T2", f"{cond} {m} overall", B, f"{base}.overall_accuracy", ov, 100, 0.05)]

    # ---- App F: error flows, quoted as a share of the author's own row.
    for cond, author, named, share in [
        ("lineup", "Grok", "DeepSeek", 32.6), ("lineup", "DeepSeek", "Claude", 28.4),
        ("lineup", "Gemini", "GPT", 27.9),
        ("single", "DeepSeek", "Claude", 65.3), ("single", "Gemini", "GPT", 58.9),
    ]:
        C.append(("F", f"{cond} flow {author}->{named}", B,
                  f"{cond}.cross_attribution.row_rates.{author}.{named}",
                  share, 100, 0.05))

    # The single-text flows the paper quotes as a share of all errors instead.
    for author, named, share in [("DeepSeek", "Claude", 18.8), ("Gemini", "GPT", 17.0),
                                 ("Grok", "GPT", 15.0), ("Grok", "Claude", 13.4)]:
        C.append(("F", f"single flow {author}->{named}, share of errors", B,
                  f"single.cross_attribution.matrix.{author}.{named}",
                  share, 100 / 659, 0.05))

    # ---- App F: where each judge sends its single-text guesses.
    for judge, target, share in [("GPT", "GPT", 60.5), ("Claude", "GPT", 72.6),
                                 ("DeepSeek", "GPT", 68.4),
                                 ("Gemini", "Claude", 82.1), ("Grok", "Claude", 89.5)]:
        C.append(("F", f"single {judge} sends to {target}", B,
                  f"single.cross_attribution.per_judge.{judge}.guess_share.{target}",
                  share, 100, 0.05))

    # ---- App D: the leak barely moves while the hit rate swings.
    C += [
        ("D", "GPT leak lineup->single p = 0.821", B, "leak_comparison[GPT].p", 0.821, 1, 0.0005),
        ("D", "Claude leak lineup->single p = 0.918", B, "leak_comparison[Claude].p", 0.918, 1, 0.0005),
        ("D", "GPT hit rate swing +36.8 pp", B, "leak_comparison[GPT].hit_rate_delta_pp", 36.8, 1, 0.05),
        # The paper's +41.5 is the difference of the two rounded rates; the exact
        # difference is 41.45, so this one is checked to a tenth either way.
        ("D", "GPT false-alarm swing +41.5 pp", B, "leak_comparison[GPT].fa_rate_delta_pp", 41.5, 1, 0.1),
        ("D", "Grok leak p = 0.052", B, "lineup.signal_detection[Grok].p", 0.052, 1, 0.0005),
        ("D", "Gemini self rate vs floor p = 1.00", B, "lineup.self_recognition[Gemini].p_vs_chance", 1.00, 1, 0.005),
        ("D", "DeepSeek self rate vs floor p = 0.840", B, "lineup.self_recognition[DeepSeek].p_vs_chance", 0.840, 1, 0.0005),
        ("D", "Grok self rate vs floor p = 0.023", B, "lineup.self_recognition[Grok].p_vs_chance", 0.023, 1, 0.0005),
        ("D", "Grok self rate vs floor, one-sided p = 0.011", B, "lineup.self_recognition[Grok].p_one_sided", 0.011, 1, 0.0005),
    ]
    return C


def family_sums(ea: dict) -> list[tuple[str, bool, str]]:
    imp = ea.get("feature_importance", {})
    got: dict[str, float] = {}
    for feat, value in imp.items():
        got[FAMILIES[feat]] = got.get(FAMILIES[feat], 0.0) + value
    out = []
    for fam, want in FAMILY_SUMS.items():
        have = got.get(fam, 0.0)
        out.append((f"App B: {fam} importance sums to {want}",
                    abs(have - want) <= 0.0001, f"summed to {have:.4f}"))
    out.append(("App B: the 18 importances sum to 1",
                abs(sum(imp.values()) - 1.0) <= 0.001,
                f"summed to {sum(imp.values()):.4f}"))
    return out


def structural(eb: dict) -> list[tuple[str, bool, str]]:
    """Claims that are not a number: degeneracy, counts, verdicts."""
    out = []
    for m in ("Gemini", "Grok", "DeepSeek"):
        row = _by_name(eb["single"]["signal_detection"], m)
        out.append((f"App D: single-text {m} cell is degenerate",
                    bool(row["degenerate"]), "paper reports it as a dash"))
    for cond in ("lineup", "single"):
        out.append((f"3.4: {cond} stylometric null holds",
                    bool(eb[cond]["mechanism_test"]["null_holds"]),
                    "features must not beat the identity-only baseline"))
        out.append((f"Methods: {cond} parsed 950/950",
                    eb[cond]["n_judgments"] == 950, "no unparsed judgments"))
        out.append((f"Methods: {cond} has 38 prompts",
                    eb[cond]["n_prompts"] == 38, "M5 excluded"))
    top2 = [r["top2_concentration"] for r in eb["single"]["leave_one_judge_out"]]
    out.append(("App F: single-text LOJO top-2 stays in [99.1%, 99.9%]",
                all(0.990 <= v <= 0.9995 for v in top2),
                f"observed {min(top2):.3%} to {max(top2):.3%}"))

    lc = {r["judge"]: r for r in eb["leak_comparison"]}
    out.append(("App D: three single-text leaks are not comparable",
                sum(not r["comparable"] for r in lc.values()) == 3,
                "Gemini, Grok and DeepSeek have degenerate single-text cells"))

    dn = eb["lineup"]["distinct_names"]["counts"]
    out.append(("App A: distinct-name spread is 5->162, 4->21, 3->6, 2->1",
                {str(k): v for k, v in dn.items()} ==
                {"5": 162, "4": 21, "3": 6, "2": 1}, str(dn)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    blobs = {}
    for key, name in (("engine_a", "engine_a_results.json"),
                      ("engine_b", "engine_b_results.json")):
        path = RESULTS / name
        if not path.exists():
            print(f"verify_paper: {path} missing -- run the engines first.")
            return 2
        blobs[key] = json.loads(path.read_text(encoding="utf-8"))

    bad = []
    for section, label, blob, path, want, scale, tol in claims():
        try:
            got = float(dig(blobs[blob], path)) * scale
        except (KeyError, IndexError, TypeError) as exc:
            bad.append((section, label, f"missing: {path} ({exc})"))
            continue
        if not _matches(got, want, tol):
            bad.append((section, label, f"paper {want}, results {got:.6g}"))
        elif args.verbose:
            print(f"  ok  [{section:>4}] {label:<44} {got:.4f}")

    checks = structural(blobs["engine_b"]) + family_sums(blobs["engine_a"])
    for label, ok, note in checks:
        if not ok:
            bad.append(("struct", label, note))
        elif args.verbose:
            print(f"  ok  [stru] {label}")

    total = len(claims()) + len(checks)
    if bad:
        print(f"\n{len(bad)} of {total} claims do not match the results:\n")
        for section, label, why in bad:
            print(f"  [{section:>6}] {label}\n           {why}")
        return 1
    print(f"all {total} paper claims match the results on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
