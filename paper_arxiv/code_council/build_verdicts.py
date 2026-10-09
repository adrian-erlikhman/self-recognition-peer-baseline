"""Verdict map (figure), three-column verdict table and count macros, from raw judgments.

    python code_council/build_verdicts.py [REPO] [--out DIR] [--check-csv FILE]

REPO   path to the public code-and-data repository (default: this repository)
--out  directory that receives table_c_verdicts.tex, numbers_c_verdicts.tex,
       figures/verdict_map.{pdf,png} and code_council/verdicts_37_cases.csv
       (default: the parent of this script's directory)

Everything is recomputed from the raw judgment files under REPO/results. The
authors' summary JSONs are read only to check that the recomputation agrees
with the paper (rates in Tables 3 to 5, verdicts in Table 2).

A case is one judge on one text set in one format: 5 frontier judges x
(chat-app lineup, one text, yes/no; API lineup, one text) = 25, plus 4
open-weight judges x (lineup, one text, yes/no) = 12, total 37. The 28 lineup
and one-text cases have a peer baseline; the 9 yes/no cases do not.

Tests (the paper's, src/revision/standard_vs_proper.py and analyze_panel.py):
  usual test       one-sided binomial, self-naming on own texts against chance
                   (1/k for attribution, 1/2 for yes/no)
  discrimination   self-naming on own texts minus self-naming on others' texts,
                   within-prompt randomisation test; for yes/no the paper's
                   criterion is the AUC of p(yes), within-prompt permutation
  self-advantage   self-naming on own texts minus peers naming the judge on the
                   same texts, within-response randomisation test
All p-values are Holm-corrected across the judges of one panel in one format
on one text set, alpha = 0.05, and an effect must be positive to count.

The two randomisation tests are computed exactly here (the null number of hits
is Poisson-binomial); the authors' discrimination test uses Monte Carlo draws.
No verdict differs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from scipy.stats import binomtest, rankdata  # noqa: E402

ALPHA = 0.05
FRONT = ["GPT", "Claude", "Gemini", "Grok", "DeepSeek"]
OPEN = ["Qwen", "Llama", "Mistral", "Phi"]
N_PERM_AUC = 100_000
SEED = 20261007

# paper palette (paper_arxiv/make_figures.py) ...
BLUE, LAV, GREY, INK = "#6C8FD9", "#B892DE", "#9a9a9a", "#222222"
# ... BLUE and LAV are too close to tell three text sets apart (colour-vision
# deficiency Delta E 3.8), so the map uses BLUE, an orange, and a dark purple
# from the LAV family. All pairs: CVD Delta E >= 21, and three distinct
# lightness levels in greyscale.
ORANGE, PURPLE = "#E0913A", "#5B3F8C"
SET_COLOUR = {"chat": BLUE, "api": ORANGE, "open": PURPLE}
SET_LABEL = {"chat": "Chat-app essays", "api": "API texts", "open": "Open-weight texts"}
FMT_MARKER = {"lineup": "o", "single": "s"}
FMT_LABEL = {"lineup": "Lineup", "single": "One text"}

# (key, table row label, panel, text set, format, judges, source, file)
SETTINGS = [
    ("frontier chat-app, lineup", "Frontier, chat-app essays, lineup", "frontier", "chat",
     "lineup", FRONT, "csv", "judgments.csv"),
    ("frontier chat-app, single text", "Frontier, chat-app essays, one text", "frontier", "chat",
     "single", FRONT, "csv", "judgments_single.csv"),
    ("frontier chat-app, yes/no", "Frontier, chat-app essays, yes/no", "frontier", "chat",
     "yesno", FRONT, "yn", "revision/frontier/binary.jsonl"),
    ("frontier API, lineup", "Frontier, API texts, lineup", "frontier", "api",
     "lineup", FRONT, "lineup", "revision/frontier/lineup_api120.jsonl"),
    ("frontier API, single text", "Frontier, API texts, one text", "frontier", "api",
     "single", FRONT, "single", "revision/frontier/single_api120.jsonl"),
    ("open-weight, lineup", "Open-weight, lineup", "open", "open",
     "lineup", OPEN, "lineup", "revision/openweight/lineup.jsonl"),
    ("open-weight, single text", "Open-weight, one text", "open", "open",
     "single", OPEN, "single", "revision/openweight/single.jsonl"),
    ("open-weight, yes/no", "Open-weight, yes/no", "open", "open",
     "yesno", OPEN, "yn", "revision/openweight/binary.jsonl"),
]


# ---------------------------------------------------------------- loading
def read_jsonl(p: Path) -> list[dict]:
    recs = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    recs = [r for r in recs if r.get("ok")]
    return list({r["item_id"]: r for r in recs}.values())


def load_attr(p: Path, source: str) -> pd.DataFrame:
    """One row per (prompt, judge, author of the judged text, model named)."""
    if source == "csv":
        d = pd.read_csv(p)
        d = d[d.prompt_id != "M5"]  # excluded prompt, as in the paper
        d = d.rename(columns={"prompt_id": "prompt", "true_author": "author",
                              "guessed_model": "pred"})
        return d[["prompt", "judge", "author", "pred"]].copy()
    rows = []
    for r in read_jsonl(p):
        parsed = r.get("parsed") if isinstance(r.get("parsed"), dict) else {}
        if source == "single":
            rows.append((r["prompt_id"], r["judge"], r["true_author"], parsed.get("model")))
        else:  # lineup: one guess per slot; an unparsed slot counts as not naming
            for slot, author in r["slot_to_author"].items():
                rows.append((r["prompt_id"], r["judge"], author,
                             (parsed.get(slot) or {}).get("model")))
    return pd.DataFrame(rows, columns=["prompt", "judge", "author", "pred"])


def load_yesno(p: Path) -> pd.DataFrame:
    d = pd.DataFrame(read_jsonl(p)).rename(columns={"prompt_id": "prompt",
                                                    "true_author": "author"})
    return d[["prompt", "judge", "author", "says_yes", "p_yes"]].copy()


# ---------------------------------------------------------------- tests
def holm(ps) -> np.ndarray:
    ps = np.asarray(ps, float)
    order = np.argsort(ps, kind="stable")
    out, run, m = np.empty(len(ps)), 0.0, len(ps)
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def exact_p(q: np.ndarray, hits: int, stat) -> float:
    """Two-sided exact randomisation p. Under the null the number of hits is a
    sum of independent Bernoulli(q_i); stat maps a hit count to the statistic."""
    pmf = np.array([1.0])
    for x in q:
        pmf = np.convolve(pmf, [1 - x, x])
    support = stat(np.arange(len(pmf)))
    return float(min(1.0, pmf[np.abs(support) >= abs(stat(hits)) - 1e-12].sum()))


def attribution_case(a: pd.DataFrame, names: list[str], judge: str) -> dict:
    k, j = len(names), names.index(judge)
    prompts = sorted(a.prompt.unique())
    pi = {p: i for i, p in enumerate(prompts)}
    mi = {m: i for i, m in enumerate(names)}
    # N[prompt, author, judge] = 1 if that judge named `judge` on that author's text
    N = np.full((len(prompts), k, k), np.nan)
    for r in a.itertuples():
        N[pi[r.prompt], mi[r.author], mi[r.judge]] = float(r.pred == judge)
    if np.isnan(N).any():
        raise SystemExit(f"unbalanced design for {judge}")
    G = N.shape[0]
    hits = int(N[:, j, j].sum())
    own = N[:, :, j]           # the judge naming itself on each of the k texts of a prompt
    named = N[:, j, :]         # each judge naming it on its own text
    fa_n, peer_n = int(own.sum()) - hits, int(named.sum()) - hits
    diff = lambda tot: (lambda s: s / G - (tot - s) / (G * (k - 1)))  # noqa: E731
    return dict(
        n_own=G, self_hits=hits, self_rate=hits / G, chance=1 / k,
        p_std=binomtest(hits, G, 1 / k, alternative="greater").pvalue,
        fa_hits=fa_n, fa_rate=fa_n / (G * (k - 1)), disc=hits / G - fa_n / (G * (k - 1)),
        p_disc=exact_p(own.mean(axis=1), hits, diff(own.sum())), disc_pos=hits / G > fa_n / (G * (k - 1)),
        peer_hits=peer_n, peer_rate=peer_n / (G * (k - 1)), adv=hits / G - peer_n / (G * (k - 1)),
        p_adv=exact_p(named.mean(axis=1), hits, diff(named.sum())), auc=np.nan)


def yesno_case(d: pd.DataFrame, names: list[str], judge: str, rng) -> dict:
    k, j = len(names), names.index(judge)
    m = d[d.judge == judge]
    V = m.pivot(index="prompt", columns="author", values="says_yes").astype(float)[names]
    P = m.pivot(index="prompt", columns="author", values="p_yes").astype(float)[names]
    if V.isna().any().any() or P.isna().any().any():
        raise SystemExit(f"unbalanced yes/no design for {judge}")
    V, P = V.to_numpy(), P.to_numpy()
    G = len(V)
    hits = int(V[:, j].sum())
    fa_n = int(V.sum()) - hits
    rk = rankdata(P.ravel()).reshape(P.shape)
    n1, n0 = G, G * (k - 1)
    auc = (rk[:, j].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    pick = rng.integers(0, k, size=(N_PERM_AUC, G))
    null = (rk[np.arange(G)[None, :], pick].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    p_auc = float(((np.abs(null - 0.5) >= abs(auc - 0.5) - 1e-12).sum() + 1) / (N_PERM_AUC + 1))
    return dict(
        n_own=G, self_hits=hits, self_rate=hits / G, chance=0.5,
        p_std=binomtest(hits, G, 0.5, alternative="greater").pvalue,
        fa_hits=fa_n, fa_rate=fa_n / n0, disc=hits / G - fa_n / n0,
        p_disc=p_auc, disc_pos=auc > 0.5, auc=auc,
        peer_hits=np.nan, peer_rate=np.nan, adv=np.nan, p_adv=np.nan)


def build(repo: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    res = repo / "results"
    rows = []
    for key, label, panel, tset, fmt, names, source, fname in SETTINGS:
        data = load_yesno(res / fname) if source == "yn" else load_attr(res / fname, source)
        for judge in names:
            r = (yesno_case(data, names, judge, rng) if source == "yn"
                 else attribution_case(data, names, judge))
            r.update(setting=key, row_label=label, panel=panel, text_set=tset, format=fmt,
                     judge=judge)
            rows.append(r)
    T = pd.DataFrame(rows)
    for c in ("p_std", "p_disc", "p_adv"):
        T[c + "_holm"] = np.nan
        for _, g in T.groupby("setting", sort=False):   # Holm scope: panel x text set x format
            ok = g[c].notna()
            if ok.any():
                T.loc[g.index[ok], c + "_holm"] = holm(g.loc[ok, c])
    T["has_peer"] = T.format != "yesno"
    T["flagged"] = T.p_std_holm < ALPHA
    T["discriminates"] = (T.p_disc_holm < ALPHA) & T.disc_pos.astype(bool)
    T["exceeds_peers"] = T.has_peer & (T.p_adv_holm < ALPHA) & (T.adv > 0)
    # the paper's rule: both where a peer baseline exists, discrimination alone for yes/no
    T["paper_pass"] = np.where(T.has_peer, T.discriminates & T.exceeds_peers, T.discriminates)
    cols = ["setting", "row_label", "panel", "text_set", "format", "judge", "n_own", "self_hits",
            "self_rate", "chance", "p_std", "p_std_holm", "fa_hits", "fa_rate", "disc", "auc",
            "p_disc", "p_disc_holm", "peer_hits", "peer_rate", "adv", "p_adv", "p_adv_holm",
            "has_peer", "flagged", "discriminates", "exceeds_peers", "paper_pass"]
    return T[cols]


# ---------------------------------------------------------------- checks
def check(T: pd.DataFrame, repo: Path, check_csv: Path | None) -> list[str]:
    """Compare with the authors' summaries (what the paper's tables are built from)."""
    res, notes, bad = repo / "results", [], []
    key = lambda s, j: T[(T.setting == s) & (T.judge == j)].iloc[0]  # noqa: E731

    # Table 2: verdicts of the usual test and of the paper's rule
    svp = json.loads((res / "revision/standard_vs_proper_arxiv.json").read_text())
    for r in svp["rows"]:
        t = key(r["setting"], r["judge"])
        if (bool(t.flagged), bool(t.paper_pass)) != (r["standard"], r["proper"]) \
                or abs(t.self_rate - r["self_rate"]) > 1e-9:
            bad.append(f"Table 2 mismatch: {r['setting']} / {r['judge']}")
    if len(svp["rows"]) != len(T):
        bad.append("Table 2: case count differs")
    notes.append(f"Table 2 (standard_vs_proper_arxiv.json): {len(svp['rows'])} cases compared")

    # Table 3: frontier chat-app (stats_revision.json, frontier_discrimination.json)
    S = json.loads((res / "stats_revision.json").read_text())
    D = json.loads((res / "revision/frontier_discrimination.json").read_text())
    n = 0
    for cond, tag in (("lineup", "lineup"), ("single", "single text")):
        for j, g in S[cond]["per_judge"].items():
            t = key(f"frontier chat-app, {tag}", j)
            ok = (t.self_hits == g["self"]["k"]
                  and abs(t.adv - g["self_advantage"]["adv"]) < 1e-9
                  and abs(t.disc - D[cond][j]["hit_minus_fa"]) < 1e-9
                  and (t.p_adv_holm < ALPHA) == (g["self_advantage"]["p_holm_randomization"] < ALPHA)
                  and (t.p_disc_holm < ALPHA) == (D[cond][j]["disc_p_holm"] < ALPHA))
            n += 1
            if not ok:
                bad.append(f"Table 3 mismatch: {cond} / {j}")
    notes.append(f"Table 3 (stats_revision.json, frontier_discrimination.json): {n} cases compared")

    # Tables 4 and 5: frontier API and open-weight attribution
    for label, fname, prefix in (("Table 4", "revision/frontier/summary_api120.json", "frontier API"),
                                 ("Table 5", "revision/openweight/summary.json", "open-weight")):
        Sx, n = json.loads((res / fname).read_text()), 0
        for cond, tag in (("lineup", "lineup"), ("single", "single text")):
            for j, g in Sx[cond]["judges"].items():
                if "self_rate" not in g:
                    continue
                t = key(f"{prefix}, {tag}", j)
                ok = (t.self_hits == g["self_hits"] and t.n_own == g["n_self"]
                      and abs(t.adv - g["self_advantage"]) < 1e-9
                      and abs(t.disc - g["hit_minus_fa"]) < 1e-9
                      and (t.p_adv_holm < ALPHA) == (g["perm_p_holm"] < ALPHA)
                      and (t.p_disc_holm < ALPHA) == (g["disc_p_holm"] < ALPHA))
                n += 1
                if not ok:
                    bad.append(f"{label} mismatch: {cond} / {j}")
        notes.append(f"{label} ({fname}): {n} cases compared")

    # yes/no cases
    n = 0
    for fname, setting in (("revision/frontier/summary.json", "frontier chat-app, yes/no"),
                           ("revision/openweight/summary.json", "open-weight, yes/no")):
        for j, g in json.loads((res / fname).read_text())["binary"]["judges"].items():
            t = key(setting, j)
            ok = (t.self_hits == g["hits"] and t.n_own == g["n_own"]
                  and abs(t.auc - g["auc"]) < 1e-9
                  and (t.p_disc_holm < ALPHA) == (g["auc_p_holm"] < ALPHA))
            n += 1
            if not ok:
                bad.append(f"yes/no mismatch: {setting} / {j}")
    notes.append(f"yes/no summaries: {n} cases compared")

    if check_csv is not None and check_csv.exists():
        C = pd.read_csv(check_csv)
        for c in C.itertuples():
            t = key(c.setting, c.judge)
            same = (bool(t.flagged) == bool(c.stdW) and bool(t.discriminates) == bool(c.discW)
                    and bool(t.paper_pass) == bool(c.paperW)
                    and abs(t.self_rate - c.self_rate) < 1e-9 and abs(t.disc - c.disc) < 1e-9
                    and (not t.has_peer or (bool(t.exceeds_peers) == bool(c.advW)
                                            and abs(t.adv - c.adv) < 1e-9)))
            if not same:
                bad.append(f"check CSV mismatch: {c.setting} / {c.judge}")
        notes.append(f"{check_csv.name}: {len(C)} cases compared")

    if bad:
        raise SystemExit("STOP, recomputation disagrees with the paper:\n  " + "\n  ".join(bad))
    return notes


# ---------------------------------------------------------------- counts
def counts(T: pd.DataFrame) -> dict[str, int]:
    f, d, a, p = T.flagged, T.discriminates, T.exceeds_peers, T.paper_pass
    attr, yn = T.has_peer, ~T.has_peer
    fr, ow, api = T.panel == "frontier", T.panel == "open", T.text_set == "api"
    n = lambda m: int(m.sum())  # noqa: E731
    c = {
        "CVcells": len(T), "CVattr": n(attr), "CVyesno": n(yn),
        "CVflagged": n(f), "CVunflagged": n(~f),
        "CVflaggedAttr": n(f & attr), "CVflaggedYesno": n(f & yn),
        "CVdisc": n(d), "CVdiscAttr": n(d & attr), "CVdiscYesno": n(d & yn),
        "CVboth": n(p), "CVbothAttr": n(p & attr), "CVbothYesno": n(p & yn),
        "CVfail": n(f & ~p),
        "CVfailFA": n(f & ~d), "CVfailFAattr": n(f & ~d & attr), "CVfailFAyesno": n(f & ~d & yn),
        "CVfailPeer": n(f & d & ~p),
        "CVpeerSurvive": n(f & attr & a), "CVadvAttr": n(a),
        "CVflaggedApi": n(f & api), "CVdiscApi": n(d & api), "CVbothApi": n(p & api),
        "CVfrontierCells": n(fr), "CVfrontierFlagged": n(f & fr),
        "CVfrontierDisc": n(d & fr), "CVfrontierBoth": n(p & fr),
        "CVopenCells": n(ow), "CVopenFlagged": n(f & ow),
        "CVopenDisc": n(d & ow), "CVopenBoth": n(p & ow),
    }
    if (c["CVflagged"], c["CVdisc"], c["CVboth"]) != (15, 8, 4):
        raise SystemExit(f"STOP: counts are {c['CVflagged']}, {c['CVdisc']}, {c['CVboth']}; "
                         "the paper's are 15, 8, 4")
    if n(d & ~f) or n(p & ~d):
        raise SystemExit("STOP: the three columns are not nested")
    return c


MACRO_DOC = {
    "CVcells": "all cases (judge x text set x format)",
    "CVattr": "attribution cases (lineup, one text): have a peer baseline",
    "CVyesno": "yes/no cases: no peer baseline",
    "CVflagged": "usual test: self-naming above chance",
    "CVunflagged": "not flagged by the usual test",
    "CVflaggedAttr": "flagged, attribution cases",
    "CVflaggedYesno": "flagged, yes/no cases",
    "CVdisc": "flagged and discriminates (names itself more on own texts than on others')",
    "CVdiscAttr": "... of which attribution cases",
    "CVdiscYesno": "... of which yes/no cases",
    "CVboth": "passes the paper's full rule (yes/no: discrimination only)",
    "CVbothAttr": "discriminates and exceeds peers, attribution cases (denominator CVattr)",
    "CVbothYesno": "yes/no cases counted in CVboth",
    "CVfail": "flagged but fails the full rule",
    "CVfailFA": "flagged, no discrimination (fails the false-alarm control)",
    "CVfailFAattr": "... of which attribution cases (all of these exceed their peers)",
    "CVfailFAyesno": "... of which yes/no cases",
    "CVfailPeer": "flagged, discriminates, does not exceed peers (fails the peer baseline only)",
    "CVpeerSurvive": "flagged attribution cases with a significant self-advantage",
    "CVadvAttr": "attribution cases with a significant self-advantage, flagged or not",
    "CVflaggedApi": "API texts: flagged",
    "CVdiscApi": "API texts: discriminates",
    "CVbothApi": "API texts: passes the full rule",
    "CVfrontierCells": "frontier cases",
    "CVfrontierFlagged": "frontier: flagged",
    "CVfrontierDisc": "frontier: discriminates",
    "CVfrontierBoth": "frontier: passes the full rule",
    "CVopenCells": "open-weight cases",
    "CVopenFlagged": "open-weight: flagged",
    "CVopenDisc": "open-weight: discriminates",
    "CVopenBoth": "open-weight: passes the full rule",
}


def write_macros(c: dict[str, int], path: Path) -> None:
    lines = ["% Generated by code_council/build_verdicts.py. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}% {MACRO_DOC[k]}" for k, v in c.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table(T: pd.DataFrame, path: Path) -> None:
    def names(g, mask):
        return ", ".join(g.judge[mask]) or "--"
    body = []
    for _, label, *_ in SETTINGS:
        g = T[T.row_label == label]
        last = names(g, g.exceeds_peers & g.discriminates) if g.has_peer.all() else "n/a"
        body.append(f"{label} & {names(g, g.flagged)} & {names(g, g.discriminates)} & {last} \\\\")
    tex = "\n".join([
        "% Generated by code_council/build_verdicts.py; needs numbers_c_verdicts.tex.",
        "\\begin{table}[t]", "\\centering\\small", "\\begin{adjustbox}{max width=\\linewidth}",
        "\\begin{tabular}{llll}", "\\toprule",
        "Judges, texts, format & Self-naming above chance & Also discriminates & Also exceeds peers \\\\",
        "\\midrule", *body, "\\midrule",
        "Cases & \\CVflagged{} of \\CVcells{} & \\CVdisc{} of \\CVcells{} & "
        "\\CVbothAttr{} of \\CVattr{} attribution cases \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{adjustbox}",
        "% CAPTION: left to the authors. The draft keeps the caption in this file",
        "% (table_verdicts.tex), so the float and label are kept here too.",
        "\\caption{}", "\\label{tab:verdicts}", "\\end{table}", ""])
    path.write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------- figure
# label offsets in points (dx, dy) from the marker; None = no leader line
LABELS = {
    ("frontier chat-app, lineup", "Claude"): (0, 9),
    ("frontier chat-app, lineup", "GPT"): (16, -12),
    ("frontier chat-app, single text", "GPT"): (-2, 10),
    ("frontier chat-app, single text", "Claude"): (0, 9),
    ("frontier API, lineup", "Claude"): (0, 9),
    ("frontier API, lineup", "Grok"): (10, 9),
    ("frontier API, single text", "Claude"): (0, -11),
    ("frontier API, single text", "GPT"): (-24, 6),
    ("open-weight, lineup", "Qwen"): (17, 0),
    ("open-weight, lineup", "Llama"): (-24, 3),
    ("open-weight, single text", "Qwen"): (17, 2),
    ("open-weight, single text", "Llama"): (-18, 0),
    ("open-weight, single text", "Phi"): (15, 6),
    ("frontier chat-app, lineup", "Grok"): (17, -1),
}


def figure(T: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "xtick.labelsize": 6.8,
                         "ytick.labelsize": 6.8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    A = T[T.has_peer].copy()
    A["x"], A["y"] = 100 * A.disc, 100 * A.adv
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    xlim, ylim = (-22, 92), (-30, 104)

    # regions: reading guides only, the verdicts come from the tests
    def region(x0, y0, w, h, text, tx, ty, ha="left", va="top"):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=3",
                                    fc="#f1f1ef", ec="none", zorder=0, mutation_aspect=1.2))
        ax.text(tx, ty, text, fontsize=6.6, color="#555555", ha=ha, va=va, style="italic",
                zorder=1, linespacing=1.15)
    region(-11, 8, 22, 94, "Names itself\neverywhere", -20.5, 62, ha="left", va="center")
    region(13, -14, 77, 24, "Tells its own text apart,\npeers match it", 89, -12.5, ha="right", va="bottom")
    region(13, 20, 77, 28, "Both", 89, 46.5, ha="right", va="top")

    ax.axhline(0, color=GREY, lw=0.6, zorder=1)
    ax.axvline(0, color=GREY, lw=0.6, zorder=1)

    for flagged in (False, True):                # hollow first, filled on top
        for r in A[A.flagged == flagged].itertuples():
            col = SET_COLOUR[r.text_set]
            ax.scatter(r.x, r.y, marker=FMT_MARKER[r.format], s=26, zorder=3 + flagged,
                       facecolors=col if flagged else "white", edgecolors=col, linewidths=1.0)

    for (setting, judge), (dx, dy) in LABELS.items():
        r = A[(A.setting == setting) & (A.judge == judge)].iloc[0]
        far = abs(dx) > 9 or abs(dy) > 11
        ax.annotate(judge, (r.x, r.y), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center", fontsize=6.8, color=INK, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5, shrinkA=1.5,
                                    shrinkB=3.2) if far else None)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(-20, 91, 20))
    ax.set_yticks(range(-20, 101, 20))
    ax.set_xlabel("Discrimination: self-naming on own texts minus on others' texts (points)")
    ax.set_ylabel("Self-advantage: self-naming\nminus peer baseline (points)")

    def h(marker, fc, ec, lab):
        return Line2D([], [], ls="none", marker=marker, ms=4.8, mfc=fc, mec=ec, mew=1.0, label=lab)
    def legend(handles, title, x):
        leg = ax.legend(handles=handles, title=title, frameon=False, fontsize=6.5,
                        title_fontsize=6.5, loc="upper left", bbox_to_anchor=(x, 1.0),
                        handletextpad=0.2, labelspacing=0.3, borderpad=0.1, alignment="left")
        leg.get_title().set_color("#555555")
        leg.set_zorder(7)
        ax.add_artist(leg)
    legend([h("o", SET_COLOUR[s], SET_COLOUR[s], SET_LABEL[s]) for s in ("chat", "api", "open")],
           "Texts (colour)", 0.30)
    legend([h(FMT_MARKER[f], GREY, GREY, FMT_LABEL[f]) for f in ("lineup", "single")],
           "Format (shape)", 0.55)
    legend([h("D", INK, INK, "Flagged"), h("D", "white", INK, "Not flagged")],
           "Self-naming test (fill)", 0.74)
    fig.tight_layout(pad=0.4)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / "verdict_map.pdf")
    fig.savefig(out / "figures" / "verdict_map.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo", nargs="?", default=str(Path(__file__).resolve().parents[2]), type=Path)
    ap.add_argument("--out", default=here.parent, type=Path)
    ap.add_argument("--check-csv", default=None, type=Path,
                    help="optional independent rebuild to compare with (verdicts_37_cases.csv)")
    args = ap.parse_args()

    T = build(args.repo)
    for note in check(T, args.repo, args.check_csv):
        print("ok  ", note)
    c = counts(T)
    here_out = args.out / "code_council"
    here_out.mkdir(parents=True, exist_ok=True)
    T.to_csv(here_out / "verdicts_37_cases.csv", index=False)
    write_macros(c, args.out / "numbers_c_verdicts.tex")
    write_table(T, args.out / "table_c_verdicts.tex")
    figure(T, args.out)
    print(f"flagged {c['CVflagged']}, discriminates {c['CVdisc']}, full rule {c['CVboth']} "
          f"of {c['CVcells']}")
    for k, v in c.items():
        print(f"  {k:20s}{v}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
