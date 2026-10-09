"""Robustness outputs for the lineup: who names whom, and the answer-order design.

    python code_council/build_robust.py [--repo REPO] [--out <revised dir>]
                                        [--check-csv verify-I04-naming.csv]

Everything is recomputed from the archived judgments of the public repository:

  chat-app lineup    raw/<prompt>__<judge>.json  (archived responses) joined to
                     results/lineup_design.csv   (slot -> author)
  chat-app one text  raw/single__<prompt>__<author>__<judge>.json
  API lineup         results/revision/frontier/lineup_api120.jsonl (raw_text)

Writes, under --out:

  figures/naming_matrix.pdf / .png   judge-by-author naming rates, chat-app essays
  table_c_anchor.tex                 lineup design robustness (tab:anchor)
  table_c_naming.tex                 hit and false-alarm rates per judge (tab:naming)
  numbers_c_robust.tex               \\CR... macros for the text

Before writing anything the script reproduces the counts of the paper's
Table 3 (table_frontier.tex) and Table 4 lineup rates (table_frontier120.tex),
checks its parse against the repository's parsed files, and, if given,
compares every hit and false-alarm count with an independent CSV. Any
mismatch stops the run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

M = ["GPT", "Claude", "Gemini", "Grok", "DeepSeek"]
SLOTS = list("ABCDE")
N_BOOT = 4000
SEED = 20261007


# ----------------------------------------------------------------- loading
def _parse(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    i, j = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[i:j + 1])
    except Exception:
        return {}


def _name(x) -> str | None:
    if isinstance(x, dict):
        x = x.get("model")
    if not isinstance(x, str):
        return None
    low = x.lower()
    for m in M:
        if m.lower() in low:
            return m
    if "chatgpt" in low or "openai" in low:
        return "GPT"
    return None


def _content(path: Path) -> str:
    r = json.loads(path.read_text(encoding="utf-8"))["response"]
    try:
        return r["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


COLS = ["lineup", "prompt", "judge", "slot", "author", "pred"]


def load_chat_lineup(repo: Path) -> pd.DataFrame:
    des = pd.read_csv(repo / "results" / "lineup_design.csv")
    rows = []
    for sid, g in des.groupby("session_id"):
        p = _parse(_content(repo / "raw" / f"{sid}.json"))
        for r in g.itertuples():
            rows.append((sid, r.prompt_id, r.judge, r.slot, r.true_author, _name(p.get(r.slot))))
    d = pd.DataFrame(rows, columns=COLS)
    return d[d.prompt != "M5"].reset_index(drop=True)   # M5 is excluded in the paper


def load_chat_single(repo: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((repo / "raw").glob("single__*.json")):
        _, pid, author, judge = f.stem.split("__")
        rows.append((f.stem, pid, judge, None, author, _name(_parse(_content(f)))))
    d = pd.DataFrame(rows, columns=COLS)
    return d[d.prompt != "M5"].reset_index(drop=True)


def load_api_lineup(repo: Path) -> pd.DataFrame:
    seen = {}
    with open(repo / "results" / "revision" / "frontier" / "lineup_api120.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("ok", True) is not False:
                    seen[r["item_id"]] = r
    rows = []
    for r in seen.values():
        p = _parse(r.get("raw_text"))
        for s, a in r["slot_to_author"].items():
            rows.append((r["item_id"], r["prompt_id"], r["judge"], s, a, _name(p.get(s))))
    return pd.DataFrame(rows, columns=COLS)


# ----------------------------------------------------------------- measures
def naming(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (named author, judge): hits on the author's texts, false alarms on the rest."""
    out = []
    for a in M:
        for j in M:
            dj = d[d.judge == j]
            own, oth = dj[dj.author == a], dj[dj.author != a]
            out.append(dict(named_author=a, judge=j, hits=int((own.pred == a).sum()), n_author_texts=len(own),
                            false_alarms=int((oth.pred == a).sum()), n_other_texts=len(oth)))
    t = pd.DataFrame(out)
    t["hit_rate"] = 100 * t.hits / t.n_author_texts
    t["fa_rate"] = 100 * t.false_alarms / t.n_other_texts
    return t


def self_adv(d: pd.DataFrame, j: str) -> tuple[int, int, int, int, float]:
    own = d[(d.author == j) & (d.judge == j)]
    peer = d[(d.author == j) & (d.judge != j)]
    k, n, pk, pn = int((own.pred == j).sum()), len(own), int((peer.pred == j).sum()), len(peer)
    return k, n, pk, pn, 100 * (k / n - pk / pn)


def self_adv_stratified(d: pd.DataFrame, j: str) -> float:
    """Self minus peers within each position of the judge's text, weighted by own texts per position."""
    t = d[d.author == j]
    diffs, w = [], []
    for s in SLOTS:
        ts = t[t.slot == s]
        own, peer = ts[ts.judge == j], ts[ts.judge != j]
        if len(own) and len(peer):
            diffs.append((own.pred == j).mean() - (peer.pred == j).mean())
            w.append(len(own))
    return 100 * float(np.average(diffs, weights=w))


def add_order(d: pd.DataFrame) -> pd.DataFrame:
    """rot = (position of text - position of its author in the instruction's name list) mod 5."""
    d = d.copy()
    d["rot"] = (d.slot.map(SLOTS.index) - d.author.map(M.index)) % 5
    if d.groupby("lineup").rot.nunique().max() != 1:
        sys.exit("STOP: a lineup is not a rotation of the name list")
    d["listed"] = d.rot == 0
    d["correct"] = d.pred == d.author
    return d


def design(d: pd.DataFrame) -> dict:
    d = add_order(d)
    one = d.groupby("lineup").pred.apply(lambda s: s.nunique() == 5)
    li, ot = d[d.listed], d[~d.listed]
    r = dict(orders=int(d.rot.nunique()), lineups=int(len(one)), one_k=int(one.sum()),
             listed_k=int(li.correct.sum()), listed_n=len(li), other_k=int(ot.correct.sum()), other_n=len(ot))
    for j in M:
        r[j] = dict(printed=self_adv(d, j)[4], strat=self_adv_stratified(d, j), removed=self_adv(ot, j)[4],
                    removed_counts=self_adv(ot, j)[:4])
    return r


def self_minus_peer(d: pd.DataFrame, a: str, peer: str) -> tuple[float, float, float, float, float, float]:
    """Self-naming minus one named peer on a's texts, prompt bootstrap; also against the best peer of each resample."""
    t = d[d.author == a].assign(hit=lambda x: (x.pred == a).astype(float))
    piv = t.pivot(index="prompt", columns="judge", values="hit")[M].to_numpy()
    ai, pi = M.index(a), M.index(peer)
    others = [i for i in range(5) if i != ai]
    obs = 100 * (piv[:, ai].mean() - piv[:, pi].mean())
    idx = np.random.default_rng(SEED).integers(0, len(piv), (N_BOOT, len(piv)))
    b = piv[idx].mean(1)                                  # resample x judge
    lo, hi = np.percentile(100 * (b[:, ai] - b[:, pi]), [2.5, 97.5])
    lo2, hi2 = np.percentile(100 * (b[:, ai] - b[:, others].max(1)), [2.5, 97.5])
    obs2 = 100 * (piv[:, ai].mean() - piv[:, others].mean(0).max())
    return obs, lo, hi, obs2, lo2, hi2


# ----------------------------------------------------------------- checks
T3_LINEUP = {"GPT": (22, 16, 46), "Claude": (33, 14, 82), "Gemini": (7, 31, 43), "Grok": (2, 31, 44), "DeepSeek": (8, 27, 33)}
T3_SINGLE = {"GPT": (36, 79, 93), "Claude": (33, 13, 125), "Gemini": (0, 0, 2), "Grok": (0, 0, 1), "DeepSeek": (0, 0, 1)}
T4_LINEUP = {"GPT": (19.2, 15.0, 20.4, 4.2), "Claude": (56.7, 50.4, 12.7, 6.2), "Gemini": (26.7, 31.9, 18.3, -5.2),
             "Grok": (32.5, 30.2, 16.2, 2.3), "DeepSeek": (21.7, 18.8, 17.7, 2.9)}
T3_ADV = {"GPT": 27.6, "Claude": 32.9}


def checks(CL, CS, AL, NT, repo: Path, check_csv: Path | None) -> list[str]:
    log, bad = [], []

    def ok(cond: bool, msg: str) -> None:
        log.append(("PASS  " if cond else "FAIL  ") + msg)
        if not cond:
            bad.append(msg)

    ok(len(CL) == 950 and CL.prompt.nunique() == 38 and CL.lineup.nunique() == 190, f"chat-app lineup: {len(CL)} judgments, {CL.prompt.nunique()} prompts, {CL.lineup.nunique()} lineups")
    ok(len(CS) == 950 and CS.prompt.nunique() == 38, f"chat-app one text: {len(CS)} judgments, {CS.prompt.nunique()} prompts")
    ok(len(AL) == 3000 and AL.prompt.nunique() == 120 and AL.lineup.nunique() == 600, f"API lineup: {len(AL)} judgments, {AL.prompt.nunique()} prompts, {AL.lineup.nunique()} lineups")
    for nm, d in (("chat-app lineup", CL), ("chat-app one text", CS), ("API lineup", AL)):
        ok(int(d.pred.isna().sum()) == 0, f"{nm}: no unparsed answers")

    # my parse of the archived responses against the repository's parsed files
    for nm, d, f in (("chat-app lineup", CL, "judgments.csv"), ("chat-app one text", CS, "judgments_single.csv")):
        J = pd.read_csv(repo / "results" / f)
        m = d.merge(J, left_on=["prompt", "judge", "author"], right_on=["prompt_id", "judge", "true_author"])
        ok(len(m) == len(d) and int((m.pred != m.guessed_model).sum()) == 0, f"{nm}: parse of raw responses equals results/{f} ({len(m)} rows)")
    mism = 0
    with open(repo / "results" / "revision" / "frontier" / "lineup_api120.jsonl", encoding="utf-8") as fh:
        parsed = {json.loads(l)["item_id"]: json.loads(l).get("parsed") or {} for l in fh if l.strip()}
    for r in AL.itertuples():
        mism += _name(parsed[r.lineup].get(r.slot)) != r.pred
    ok(mism == 0, "API lineup: parse of raw_text equals the file's parsed field (3000 rows)")

    # the paper's Table 3 (counts) and Table 4 lineup block (rates)
    for nm, key, ref in (("lineup", "chat-app lineup", T3_LINEUP), ("one text", "chat-app one-text", T3_SINGLE)):
        t = NT[key]
        for j in M:
            s = t[(t.named_author == j) & (t.judge == j)].iloc[0]
            p = t[(t.named_author == j) & (t.judge != j)]
            got = (int(s.hits), int(s.false_alarms), int(p.hits.sum()))
            ok(got == ref[j] and s.n_author_texts == 38 and s.n_other_texts == 152 and p.n_author_texts.sum() == 152,
               f"Table 3 {nm} {j}: self {got[0]}/38, false alarms {got[1]}/152, peers {got[2]}/152 (paper {ref[j][0]}, {ref[j][1]}, {ref[j][2]})")
    t = NT["API lineup"]
    for j in M:
        s = t[(t.named_author == j) & (t.judge == j)].iloc[0]
        p = t[(t.named_author == j) & (t.judge != j)]
        got = (round(s.hit_rate, 1), round(100 * p.hits.sum() / p.n_author_texts.sum(), 1), round(s.fa_rate, 1),
               round(s.hit_rate - 100 * p.hits.sum() / p.n_author_texts.sum(), 1))
        ok(np.allclose(got, T4_LINEUP[j], atol=0.051), f"Table 4 lineup {j}: self, peer, FA, self-adv = {got} (paper {T4_LINEUP[j]})")
    for j, v in T3_ADV.items():
        ok(round(self_adv(CL, j)[4], 1) == v, f"Table 3 self-advantage {j}: {self_adv(CL, j)[4]:+.1f} (paper +{v})")

    if check_csv is not None and check_csv.exists():
        V = pd.read_csv(check_csv)
        n = 0
        for key, t in NT.items():
            m = t.merge(V[V.setting == key], on=["named_author", "judge"], suffixes=("", "_v"))
            same = ((m.hits == m.hits_v) & (m.false_alarms == m.false_alarms_v) & (m.n_author_texts == m.n_author_texts_v)
                    & (m.n_other_texts == m.n_other_texts_v) & (m.hit_rate.round(1) == m.hit_rate_v) & (m.fa_rate.round(1) == m.fa_rate_v))
            n += len(m)
            ok(len(m) == 25 and bool(same.all()), f"{key}: 25 of 25 cells equal {check_csv.name} (hits, false alarms, denominators, rates)")
    else:
        log.append("SKIP  no independent naming CSV given")
    if bad:
        print("\n".join(log))
        sys.exit("STOP: " + str(len(bad)) + " check(s) failed; nothing written")
    return log


# ----------------------------------------------------------------- outputs
def pct(x: float) -> str:
    return f"{x:.1f}\\%"


def sgn(x: float) -> str:
    return (f"+{x:.1f}" if x >= 0 else f"$-${abs(x):.1f}")


def write_anchor(D: dict, path: Path) -> None:
    c, a = D["chat"], D["api"]

    def kn(k, n):
        return f"{k:,}/{n:,} ({100 * k / n:.1f}\\%)".replace(",", "{,}")

    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength\tabcolsep{5pt}", r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{l rr}", r"\toprule", r" & Chat-app essays & API texts \\", r"\midrule",
         f"Lineups that use each name once & {kn(c['one_k'], c['lineups'])} & {kn(a['one_k'], a['lineups'])} \\\\",
         r"\midrule", r"\multicolumn{3}{l}{\emph{Attribution accuracy, all five judges}} \\",
         f"\\quad Answers in the listed order of the names & {kn(c['listed_k'], c['listed_n'])} & {kn(a['listed_k'], a['listed_n'])} \\\\",
         f"\\quad Answers in the other four orders & {kn(c['other_k'], c['other_n'])} & {kn(a['other_k'], a['other_n'])} \\\\"]
    for j in ("Claude", "GPT"):
        L += [r"\midrule", f"\\multicolumn{{3}}{{l}}{{\\emph{{Self-advantage of {j} (points)}}}} \\\\",
              f"\\quad As printed, all lineups & {sgn(c[j]['printed'])} & {sgn(a[j]['printed'])} \\\\",
              f"\\quad Stratified by position of the text & {sgn(c[j]['strat'])} & {sgn(a[j]['strat'])} \\\\",
              f"\\quad Listed-order lineups removed & {sgn(c[j]['removed'])} & {sgn(a[j]['removed'])} \\\\"]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
          r"\caption{Lineup design and the frontier results. The answers to a prompt appear in one of five orders, the "
          r"rotations of the order in which the instructions list the candidate names, so in one fifth of the lineups "
          r"the answers follow the listed order. Judges tend to assign the names in that order, and accuracy is about "
          r"twice as high in those lineups. Every judge sees each model's answer in each position equally often, so the "
          r"effect raises the self-naming rate and the peer baseline together. The self-advantages of Claude and GPT "
          r"on the chat-app essays change by about 3 points or less when they are computed within each position of the "
          r"text and averaged, or when the listed-order lineups are removed. On the API texts positions are fully "
          r"balanced within each judge, so stratifying leaves the self-advantage unchanged. First row: lineups in "
          r"which the judge assigns each of the five names to one answer, although the instructions allow repeats.}",
          r"\label{tab:anchor}", r"\end{table}", ""]
    path.write_text("\n".join(L), encoding="utf-8")


SETTINGS = [("chat-app lineup", "Chat-app, lineup"), ("chat-app one-text", "Chat-app, one text"), ("API lineup", "API, lineup")]


def write_naming(NT: dict, path: Path) -> None:
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength\tabcolsep{4pt}", r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{l rr rr rr}", r"\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in SETTINGS) + r" \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Judge & Hit & False alarm & Hit & False alarm & Hit & False alarm \\"]
    for a in ("Claude", "GPT"):
        L += [r"\midrule", f"\\multicolumn{{7}}{{l}}{{\\emph{{Naming {a}}}}} \\\\"]
        for j in M:
            cells = []
            for key, _ in SETTINGS:
                r = NT[key][(NT[key].named_author == a) & (NT[key].judge == j)].iloc[0]
                cells += [f"{r.hit_rate:.1f}", f"{r.fa_rate:.1f}"]
            if j == a:
                cells = [f"\\textbf{{{x}}}" for x in cells]
            L.append(f"{j} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
          r"\caption{How often each frontier judge names Claude and names GPT (\%). Hit: on the texts that model wrote "
          r"(38 chat-app essays, 120 API texts). False alarm: on the texts of the other four models (152 and 480). "
          r"Bold rows are the judge's own texts and repeat the self-naming and false-alarm rates of "
          r"Tables~\ref{tab:frontier} and~\ref{tab:frontier120}. The mean of the four other hit rates in a block is "
          r"the peer baseline. The chat-app hit rates are the Claude and GPT columns of Figure~\ref{fig:naming}.}",
          r"\label{tab:naming}", r"\end{table}", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def macros(NT: dict, D: dict, best: tuple) -> list[tuple[str, str, str]]:
    def cell(key, a, j):
        return NT[key][(NT[key].named_author == a) & (NT[key].judge == j)].iloc[0]

    c, a = D["chat"], D["api"]
    cl, cs, al = "chat-app lineup", "chat-app one-text", "API lineup"
    num = lambda n: f"{n:,}".replace(",", "{,}")
    m = [("OneToOneChat", pct(100 * c["one_k"] / c["lineups"]), "chat-app lineups that use each of the five names once"),
         ("OneToOneChatK", str(c["one_k"]), "... count"), ("OneToOneChatN", str(c["lineups"]), "... of lineups"),
         ("OneToOneApi", pct(100 * a["one_k"] / a["lineups"]), "API lineups that use each name once"),
         ("OneToOneApiK", str(a["one_k"]), "... count"), ("OneToOneApiN", str(a["lineups"]), "... of lineups"),
         ("AccListedChat", pct(100 * c["listed_k"] / c["listed_n"]), "chat-app lineup accuracy, answers in the listed order of the names"),
         ("AccListedChatK", str(c["listed_k"]), "... correct"), ("AccListedChatN", str(c["listed_n"]), "... judgments"),
         ("AccOtherChat", pct(100 * c["other_k"] / c["other_n"]), "chat-app lineup accuracy, other four orders"),
         ("AccOtherChatK", str(c["other_k"]), "... correct"), ("AccOtherChatN", str(c["other_n"]), "... judgments"),
         ("AccListedApi", pct(100 * a["listed_k"] / a["listed_n"]), "API lineup accuracy, listed order"),
         ("AccListedApiK", str(a["listed_k"]), "... correct"), ("AccListedApiN", str(a["listed_n"]), "... judgments"),
         ("AccOtherApi", pct(100 * a["other_k"] / a["other_n"]), "API lineup accuracy, other four orders"),
         ("AccOtherApiK", str(a["other_k"]), "... correct"), ("AccOtherApiN", num(a["other_n"]), "... judgments"),
         ("AdvStratClaude", sgn(c["Claude"]["strat"]), "Claude self-advantage, chat-app lineup, stratified by position"),
         ("AdvStratGPT", sgn(c["GPT"]["strat"]), "GPT, same"),
         ("AdvNoListedClaude", sgn(c["Claude"]["removed"]), "Claude self-advantage, chat-app lineup, listed-order lineups removed"),
         ("AdvNoListedGPT", sgn(c["GPT"]["removed"]), "GPT, same"),
         ("AdvStratClaudeApi", sgn(a["Claude"]["strat"]), "Claude self-advantage, API lineup, stratified by position"),
         ("AdvStratGPTApi", sgn(a["GPT"]["strat"]), "GPT, same"),
         ("AdvNoListedClaudeApi", sgn(a["Claude"]["removed"]), "Claude self-advantage, API lineup, listed-order lineups removed"),
         ("AdvNoListedGPTApi", sgn(a["GPT"]["removed"]), "GPT, same")]
    for j in ("GPT", "Claude"):
        r = cell(cl, "Claude", j)
        m += [(f"{j}OnClaudeHit", pct(r.hit_rate), f"{j} names Claude on Claude's chat-app essays, lineup"),
              (f"{j}OnClaudeHitK", str(r.hits), "... count of 38"),
              (f"{j}OnClaudeFA", pct(r.fa_rate), f"{j} names Claude on the 152 essays Claude did not write, lineup"),
              (f"{j}OnClaudeFAK", str(r.false_alarms), "... count of 152")]
    for j in ("Gemini", "Grok", "DeepSeek"):
        m.append((f"{j}OnClaudeHit", pct(cell(cl, "Claude", j).hit_rate), f"{j} names Claude on Claude's chat-app essays, lineup"))
    for j in ("Gemini", "Claude"):
        m.append((f"{j}OnClaudeApiHit", pct(cell(al, "Claude", j).hit_rate), f"{j} names Claude on Claude's API texts, lineup"))
    for j in ("Gemini", "Grok"):
        r = cell(cs, "Claude", j)
        m += [(f"{j}OnClaudeSingleHit", pct(r.hit_rate), f"{j} names Claude on Claude's chat-app essays, one text"),
              (f"{j}OnClaudeSingleFA", pct(r.fa_rate), f"{j} names Claude on essays Claude did not write, one text")]
    lo_hi = [x for g in ("GPT",) for x in [cell(cl, "GPT", j).hit_rate for j in M if j != g]]
    m += [("PeersOnGPTMin", pct(min(lo_hi)), "lowest single-peer hit rate on GPT's chat-app essays, lineup"),
          ("PeersOnGPTMax", pct(max(lo_hi)), "highest single-peer hit rate on GPT's chat-app essays, lineup"),
          ("ClaudeVsBestPeer", sgn(best[0]), "Claude self-naming minus its best single peer (GPT), chat-app lineup, points"),
          ("ClaudeVsBestPeerCI", f"[{sgn(best[1])}, {sgn(best[2])}]", f"... 95% prompt-bootstrap interval, {N_BOOT} resamples"),
          ("ClaudeVsBestPeerLo", sgn(best[1]), "... lower bound"), ("ClaudeVsBestPeerHi", sgn(best[2]), "... upper bound")]
    return m


def write_macros(m: list, path: Path, others: list[Path]) -> None:
    names = [n for n, _, _ in m]
    assert len(set(names)) == len(names) and all(re.fullmatch(r"[A-Za-z]+", n) for n in names)
    defined = set()
    for f in others:
        defined |= set(re.findall(r"\\(?:re)?newcommand\{?\\([A-Za-z]+)", f.read_text(encoding="utf-8")))
    clash = sorted(defined & {"CR" + n for n in names})
    if clash:
        sys.exit("STOP: macro names already defined elsewhere: " + ", ".join(clash))
    L = ["% Generated by code_council/build_robust.py. Do not edit by hand."]
    L += [f"\\newcommand{{\\CR{n}}}{{{v}}}% {c}" for n, v, c in m]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figure(NT: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    ink = "#222222"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    # one hue, light to dark, lightness falls monotonically so it survives greyscale printing
    cmap = LinearSegmentedColormap.from_list("blue", ["#F4F7FD", "#B9CAEE", "#6C8FD9", "#34539F", "#16264F"])
    fig = plt.figure(figsize=(5.5, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], left=0.105, right=0.925, bottom=0.17, top=0.91, wspace=0.12)
    panels = [("chat-app lineup", "(a) Lineup"), ("chat-app one-text", "(b) One text at a time")]
    im = None
    for k, (key, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        # rows = judge, columns = author; cell = % of the author's essays on which the judge names the author
        Z = NT[key].pivot(index="judge", columns="named_author", values="hit_rate").loc[M, M].to_numpy()
        im = ax.imshow(Z, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(5):
            for j in range(5):
                rgb = cmap(Z[i, j] / 100)[:3]
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                ax.text(j, i, f"{Z[i, j]:.0f}" if Z[i, j] in (0, 100) else f"{Z[i, j]:.1f}", ha="center", va="center",
                        fontsize=7, color="white" if lum < 0.5 else ink, fontweight="bold" if i == j else "normal")
        ax.set_xticks(np.arange(-0.5, 5), minor=True)
        ax.set_yticks(np.arange(-0.5, 5), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="both", length=0)
        for i in range(5):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=ink, lw=1.3, zorder=5, clip_on=False))
        ax.set_xticks(range(5))
        ax.set_xticklabels(M, fontsize=6.3)
        ax.set_yticks(range(5))
        ax.set_yticklabels(M if k == 0 else [], fontsize=6.8)
        ax.set_xlabel("Author of the essays")
        if k == 0:
            ax.set_ylabel("Judge")
        ax.set_title(title, fontsize=8, loc="left", pad=4)
        for s in ax.spines.values():
            s.set_visible(False)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 25, 50, 75, 100])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.8, length=2, width=0.5)
    cb.set_label("Judge names the author (%)", fontsize=7)
    out.mkdir(exist_ok=True)
    fig.savefig(out / "naming_matrix.pdf")
    fig.savefig(out / "naming_matrix.png", dpi=300)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=Path, default=here.parent)
    ap.add_argument("--check-csv", type=Path, default=None)
    ap.add_argument("--macro-files", type=Path, nargs="*", default=None,
                    help="other macro files to test for name clashes (default: numbers_*.tex next to the output)")
    args = ap.parse_args()

    CL, CS, AL = load_chat_lineup(args.repo), load_chat_single(args.repo), load_api_lineup(args.repo)
    NT = {"chat-app lineup": naming(CL), "chat-app one-text": naming(CS), "API lineup": naming(AL)}
    log = checks(CL, CS, AL, NT, args.repo, args.check_csv)
    print("\n".join(log))

    D = {"chat": design(CL), "api": design(AL)}
    best = self_minus_peer(CL, "Claude", "GPT")
    hits = NT["chat-app lineup"]
    peers = hits[(hits.named_author == "Claude") & (hits.judge != "Claude")]
    assert peers.loc[peers.hit_rate.idxmax(), "judge"] == "GPT"      # GPT is Claude's best single peer

    others = args.macro_files
    if others is None:
        others = [f for f in sorted(args.out.glob("numbers_*.tex")) if f.name != "numbers_c_robust.tex"]
    mac = macros(NT, D, best)
    write_macros(mac, args.out / "numbers_c_robust.tex", others)
    write_anchor(D, args.out / "table_c_anchor.tex")
    write_naming(NT, args.out / "table_c_naming.tex")
    figure(NT, args.out / "figures")

    print("\n== naming rates (%), rows = judge, columns = author named; hit then false alarm")
    for key, t in NT.items():
        print("--", key)
        print(t.pivot(index="judge", columns="named_author", values="hit_rate").loc[M, M].round(1).to_string())
        print(t.pivot(index="judge", columns="named_author", values="fa_rate").loc[M, M].round(1).to_string())
    print("\n== design")
    for nm, d in D.items():
        print(nm, {k: v for k, v in d.items() if k not in M})
        for j in M:
            print(f"   {j:9s} printed {d[j]['printed']:+.1f}  stratified {d[j]['strat']:+.1f}  listed removed {d[j]['removed']:+.1f} {d[j]['removed_counts']}")
    print(f"\nClaude minus GPT on Claude's essays: {best[0]:+.1f} [{best[1]:+.1f}, {best[2]:+.1f}];"
          f" minus best peer of each resample: {best[3]:+.1f} [{best[4]:+.1f}, {best[5]:+.1f}]")
    print("\n== macros")
    for n, v, c in mac:
        print(f"\\CR{n} = {v}   % {c}")


if __name__ == "__main__":
    main()
