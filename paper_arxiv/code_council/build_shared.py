"""Chat-app essays against API texts on the 38 shared prompts (lineup format).

    python code_council/build_shared.py [REPO] [--out DIR] [--seed N] [--draws N]

REPO   path to the public code-and-data repository (default: this repository)
--out  directory that receives table_c_shared.tex, numbers_c_shared.tex,
       figures/shared38.{pdf,png} and code_council/shared38_stats.csv
       (default: the parent of this script's directory)

Everything is recomputed from two raw files:
  REPO/results/judgments.csv                           chat-app essays, lineup
  REPO/results/revision/frontier/lineup_api120.jsonl   API texts, lineup
The 38 shared prompts are the prompt ids of the chat-app file.

For each judge J and each text set, over the 38 prompts:
  self-naming   J names itself on its own texts                     (k of 38)
  false alarms  J names itself on texts it did not write            (k of 152)
  peer baseline the other four judges name J on J's texts           (k of 152)
  discrimination  = self-naming - false alarms
  self-advantage  = self-naming - peer baseline
All rates are pooled ratios. Intervals are 95% percentile intervals from a
prompt bootstrap: the 38 prompts are resampled with replacement, and the SAME
resampled prompts are applied to both text sets, so the chat-minus-API
difference is a paired difference over prompts.

Before writing anything the script checks its rates against the paper
(Table 3 lineup rows; Table 11 "38 shared prompts" columns) and stops if any
value differs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

JUDGES = ["GPT", "Claude", "Gemini", "Grok", "DeepSeek"]
QUANT = ["self", "fa", "peer", "disc", "adv"]
QNAME = {"self": "self-naming", "fa": "false alarms", "peer": "peer baseline",
         "disc": "discrimination", "adv": "self-advantage"}

# Paper values used as checks (draft table_frontier.tex and numbers_rev.tex).
TABLE3 = {  # judge: (self k, false-alarm k, peer k, self-advantage in points)
    "GPT": (22, 16, 46, 27.6), "Claude": (33, 14, 82, 32.9), "Gemini": (7, 31, 43, -9.9),
    "Grok": (2, 31, 44, -23.7), "DeepSeek": (8, 27, 33, -0.7)}
TABLE11 = {  # judge: (self %, peer %, self-advantage, discrimination), 38 shared prompts
    "GPT": (28.9, 16.4, 12.5, 11.2), "Claude": (81.6, 63.8, 17.8, 73.7),
    "Gemini": (34.2, 30.9, 3.3, 17.8), "Grok": (21.1, 25.0, -3.9, 3.3),
    "DeepSeek": (15.8, 17.8, -2.0, -2.0)}


def load_chat(repo: Path) -> pd.DataFrame:
    d = pd.read_csv(repo / "results" / "judgments.csv")
    d = d[["prompt_id", "judge", "true_author", "guessed_model"]]
    return d.rename(columns={"guessed_model": "guess"})


def load_api(repo: Path) -> pd.DataFrame:
    rows = []
    path = repo / "results" / "revision" / "frontier" / "lineup_api120.jsonl"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            parsed = r.get("parsed") or {}
            for slot, author in r["slot_to_author"].items():
                guess = (parsed.get(slot) or {}).get("model")
                rows.append((r["prompt_id"], r["judge"], author, guess))
    return pd.DataFrame(rows, columns=["prompt_id", "judge", "true_author", "guess"])


def per_prompt(d: pd.DataFrame, j: str, pids: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-prompt numerators and denominators for the three rates of judge j."""
    def nd(sub: pd.DataFrame):
        g = sub.assign(x=(sub.guess == j).astype(float)).groupby("prompt_id").x.agg(["sum", "count"])
        g = g.reindex(pids).fillna(0.0)
        return g["sum"].to_numpy(), g["count"].to_numpy()
    return {"self": nd(d[(d.judge == j) & (d.true_author == j)]),
            "fa": nd(d[(d.judge == j) & (d.true_author != j)]),
            "peer": nd(d[(d.judge != j) & (d.true_author == j)])}


def compute(chat: pd.DataFrame, api: pd.DataFrame, seed: int, draws: int) -> dict:
    pids = sorted(set(chat.prompt_id))
    api = api[api.prompt_id.isin(pids)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pids), (draws, len(pids)))
    out = {}
    for j in JUDGES:
        pt, bs, cnt = {}, {}, {}
        for lab, d in (("chat", chat), ("api", api)):
            pp = per_prompt(d, j, pids)
            for q in ("self", "fa", "peer"):
                n, dn = pp[q]
                pt[lab, q] = 100 * n.sum() / dn.sum()
                bs[lab, q] = 100 * n[idx].sum(1) / dn[idx].sum(1)
                cnt[lab, q] = (int(n.sum()), int(dn.sum()))
            for q, other in (("disc", "fa"), ("adv", "peer")):
                pt[lab, q] = pt[lab, "self"] - pt[lab, other]
                bs[lab, q] = bs[lab, "self"] - bs[lab, other]
        for q in QUANT:
            pt["diff", q] = pt["chat", q] - pt["api", q]
            bs["diff", q] = bs["chat", q] - bs["api", q]
        res = {}
        for key in pt:
            lo, hi = np.percentile(bs[key], [2.5, 97.5])
            res[key] = {"est": float(pt[key]), "lo": float(lo), "hi": float(hi),
                        "k": cnt.get(key, (None, None))[0], "n": cnt.get(key, (None, None))[1]}
        out[j] = res
    return out, pids, len(api)


def check(res: dict) -> list[str]:
    bad = []
    for j in JUDGES:
        r = res[j]
        got3 = (r["chat", "self"]["k"], r["chat", "fa"]["k"], r["chat", "peer"]["k"],
                round(r["chat", "adv"]["est"], 1))
        if got3 != TABLE3[j]:
            bad.append(f"Table 3 {j}: paper {TABLE3[j]}, recomputed {got3}")
        got11 = tuple(round(r["api", q]["est"], 1) for q in ("self", "peer", "adv", "disc"))
        if got11 != TABLE11[j]:
            bad.append(f"Table 11 {j}: paper {TABLE11[j]}, recomputed {got11}")
    return bad


# ---------------------------------------------------------------- formatting
def sgn(x: float) -> str:
    """Signed number in the paper's LaTeX style: +12.5, $-$3.9."""
    x = round(x, 1) + 0.0  # no negative zero
    s = f"{x:+.1f}"
    return s.replace("-", "$-$")


def ci(c: dict) -> str:
    return f"[{sgn(c['lo'])}, {sgn(c['hi'])}]"


def star(c: dict) -> str:
    return "$^{*}$" if (c["lo"] > 0 or c["hi"] < 0) else ""


def write_table(res: dict, path: Path, draws: int) -> None:
    L = ["% Generated by code_council/build_shared.py. Do not edit by hand.",
         r"\begin{table}[t]",
         r"\centering\footnotesize",
         r"\setlength\tabcolsep{3pt}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{l lll rrl}",
         r"\toprule",
         r" & \multicolumn{3}{c}{Self-advantage} & \multicolumn{3}{c}{Discrimination} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Judge & Chat-app & API & Difference & Chat-app & API & Difference \\",
         r"\midrule"]
    for j in JUDGES:
        r = res[j]
        cells = [j]
        for lab in ("chat", "api", "diff"):
            c = r[lab, "adv"]
            cells.append(f"{sgn(c['est'])} {ci(c)}{star(c)}")
        cells.append(sgn(r["chat", "disc"]["est"]))
        cells.append(sgn(r["api", "disc"]["est"]))
        c = r["diff", "disc"]
        cells.append(f"{sgn(c['est'])} {ci(c)}{star(c)}")
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
          r"\caption{Frontier judges in the lineup on the 38 prompts that the chat-app essays and "
          r"the API texts share, in points. Self-advantage is the self-naming rate minus the peer "
          r"baseline, and discrimination is the self-naming rate minus the false-alarm rate. "
          r"Difference is the chat-app value minus the API value. Brackets are 95\% bootstrap "
          r"intervals over prompts (" + f"{draws:,}".replace(",", "{,}") + " resamples" +
          r", with the same resampled prompts applied to "
          r"both text sets), and an asterisk marks an interval that excludes zero. The intervals "
          r"are not corrected for multiple comparisons. For Claude and GPT the two "
          r"self-advantages differ by about 15 points and both intervals for the difference "
          r"include zero. GPT's discrimination is lower on the API texts, and Claude's is "
          r"almost the same on both sets.}",
          r"\label{tab:shared}", r"\end{table}", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def write_macros(res: dict, path: Path, seed: int, draws: int) -> list[tuple[str, str, str]]:
    M: list[tuple[str, str, str]] = []

    def add(name, val, note):
        assert name.isalpha(), name
        M.append((name, val, note))

    pct = lambda x: f"{x:.1f}\\%"
    setname = {"chat": "Chat", "api": "API"}
    setlong = {"chat": "chat-app essays", "api": "API texts, 38 shared prompts"}
    add("CSprompts", "38", "prompts shared by the chat-app essays and the API texts")
    add("CSdraws", f"{draws:,}".replace(",", "{,}"), "bootstrap resamples of the prompts")
    for j in ("Claude", "GPT"):
        r = res[j]
        for lab in ("chat", "api"):
            S = setname[lab]
            add(f"CS{j}{S}SelfK", str(r[lab, "self"]["k"]), f"{j} names itself, of 38, {setlong[lab]}")
            add(f"CS{j}{S}Self", pct(r[lab, "self"]["est"]), "... as a rate")
            add(f"CS{j}{S}FAK", str(r[lab, "fa"]["k"]), f"{j} false alarms, of 152, {setlong[lab]}")
            add(f"CS{j}{S}FA", pct(r[lab, "fa"]["est"]), "... as a rate")
            add(f"CS{j}{S}PeerK", str(r[lab, "peer"]["k"]), f"peers name {j} on its texts, of 152, {setlong[lab]}")
            add(f"CS{j}{S}Peer", pct(r[lab, "peer"]["est"]), "... as a rate")
            for q, Q in (("disc", "Disc"), ("adv", "Adv")):
                c = r[lab, q]
                add(f"CS{j}{S}{Q}", sgn(c["est"]), f"{j} {QNAME[q]}, {setlong[lab]}")
                add(f"CS{j}{S}{Q}Lo", sgn(c["lo"]), "... 95% interval, lower")
                add(f"CS{j}{S}{Q}Hi", sgn(c["hi"]), "... upper")
                add(f"CS{j}{S}{Q}CI", ci(c), "... interval")
        for q, Q in (("self", "Self"), ("fa", "FA"), ("peer", "Peer"), ("disc", "Disc"), ("adv", "Adv")):
            c = r["diff", q]
            add(f"CS{j}{Q}Diff", sgn(c["est"]), f"{j} {QNAME[q]}, chat-app minus API")
            add(f"CS{j}{Q}DiffLo", sgn(c["lo"]), "... 95% interval, lower")
            add(f"CS{j}{Q}DiffHi", sgn(c["hi"]), "... upper")
            add(f"CS{j}{Q}DiffCI", ci(c), "... interval")
    lines = ["% Generated by code_council/build_shared.py. Do not edit by hand.",
             f"% Lineup, 38 shared prompts; prompt bootstrap, {draws} draws, seed {seed}."]
    lines += [f"\\newcommand{{\\{n}}}{{{v}}}% {note}" for n, v, note in M]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return M


def write_csv(res: dict, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["judge", "set", "quantity", "k", "n", "estimate", "ci_lo", "ci_hi"])
        for j in JUDGES:
            for lab in ("chat", "api", "diff"):
                for q in QUANT:
                    c = res[j][lab, q]
                    w.writerow([j, lab, q, c["k"] if c["k"] is not None else "",
                                c["n"] if c["n"] is not None else "",
                                f"{c['est']:.2f}", f"{c['lo']:.2f}", f"{c['hi']:.2f}"])


# -------------------------------------------------------------------- figure
def make_figure(res: dict, figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    BLUE, ORANGE, INK = "#6C8FD9", "#E0913A", "#222222"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "xtick.labelsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "pdf.fonttype": 42})
    quants = [("self", "Self-naming\nrate"), ("peer", "Peer\nbaseline"), ("fa", "False-alarm\nrate")]
    sets = [("chat", "Chat-app essays", BLUE, "o", -0.14), ("api", "API texts", ORANGE, "s", +0.14)]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True)
    for ax, (j, title) in zip(axes, (("Claude", "(a) Claude"), ("GPT", "(b) GPT"))):
        ax.axhline(20, color="#9a9a9a", lw=0.7, ls=(0, (4, 3)), zorder=1)
        # label heights: value labels sit beside their marks; where the API label of one
        # quantity would run into the chat-app label of the next, move the two apart
        ylab = {(q, lab): res[j][lab, q]["est"] for q, _ in quants for lab, *_ in sets}
        for (qa, _), (qb, _) in zip(quants[:-1], quants[1:]):
            a, b = ylab[qa, "api"], ylab[qb, "chat"]
            if abs(a - b) < 6.0:
                mid, up = (a + b) / 2, (1 if a >= b else -1)
                ylab[qa, "api"], ylab[qb, "chat"] = mid + up * 3.0, mid - up * 3.0
        for x, (q, _) in enumerate(quants):
            for lab, _, col, mk, dx in sets:
                c = res[j][lab, q]
                ax.plot([x + dx, x + dx], [c["lo"], c["hi"]], color=col, lw=1.5,
                        solid_capstyle="round", zorder=2)
                ax.plot(x + dx, c["est"], mk, color=col, ms=5.5, mec="white", mew=0.8, zorder=3)
                ha, off = ("right", -0.10) if dx < 0 else ("left", 0.10)
                ax.text(x + dx + off, ylab[q, lab], f"{c['est']:.1f}", ha=ha, va="center",
                        fontsize=6.3, color=INK, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
        ax.set_xticks(range(len(quants)))
        ax.set_xticklabels([n for _, n in quants])
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="x", length=0)
        ax.set_title(title, fontsize=8, loc="left", color=INK, pad=4)
    axes[0].set_ylabel("% of texts")
    axes[0].text(-0.56, 21.5, "chance", ha="left", va="bottom", fontsize=6.3, color="#6f6f6f")
    handles = [Line2D([0], [0], color=col, marker=mk, ms=5.5, mec="white", mew=0.8, lw=1.5, label=name)
               for _, name, col, mk, _ in sets]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 1.0), ncol=2,
               frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.2, borderaxespad=0)
    fig.tight_layout(pad=0.4, w_pad=1.2, rect=(0, 0, 1, 0.93))
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "shared38.pdf")
    fig.savefig(figdir / "shared38.png", dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo", nargs="?", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--draws", type=int, default=10000)
    a = ap.parse_args()
    repo, out = Path(a.repo), Path(a.out)

    chat, api = load_chat(repo), load_api(repo)
    res, pids, n_api = compute(chat, api, a.seed, a.draws)
    print(f"shared prompts: {len(pids)}; chat-app rows {len(chat)}; API rows on shared prompts {n_api}; "
          f"missing guesses chat {int(chat.guess.isna().sum())}, API {int(api.guess.isna().sum())}")
    if len(pids) != 38 or len(chat) != 950 or n_api != 950:
        print("STOP: unexpected number of prompts or rows"); return 1
    bad = check(res)
    if bad:
        print("STOP: recomputation does not match the paper"); print("\n".join(bad)); return 1
    print("check passed: Table 3 lineup rows (counts and self-advantage) and Table 11 shared-prompt "
          "columns (self, peer, adv., disc.) match for all five judges")

    for j in JUDGES:
        print(f"\n{j}")
        for q in QUANT:
            r = res[j]
            f = lambda c: f"{c['est']:+6.1f} [{c['lo']:+6.1f}, {c['hi']:+6.1f}]"
            kn = lambda c: f"{c['k']}/{c['n']}" if c["k"] is not None else ""
            print(f"  {QNAME[q]:15s} chat {kn(r['chat', q]):>7s} {f(r['chat', q])} | "
                  f"API {kn(r['api', q]):>7s} {f(r['api', q])} | diff {f(r['diff', q])}")

    (out / "code_council").mkdir(parents=True, exist_ok=True)
    write_csv(res, out / "code_council" / "shared38_stats.csv")
    write_table(res, out / "table_c_shared.tex", a.draws)
    macros = write_macros(res, out / "numbers_c_shared.tex", a.seed, a.draws)
    make_figure(res, out / "figures")
    print(f"\nwrote table_c_shared.tex, numbers_c_shared.tex ({len(macros)} macros), "
          f"figures/shared38.pdf/.png, code_council/shared38_stats.csv under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
