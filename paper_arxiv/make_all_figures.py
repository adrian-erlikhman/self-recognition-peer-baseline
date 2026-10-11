"""Every figure in the paper, in one style (figstyle.py).

    python paper_arxiv/make_all_figures.py

Reads only computed results: results/*.json, code_council/verdicts_37_cases.csv
and strongest_peer.csv, and the shared-prompt and naming computations of
code_council/build_shared.py and build_robust.py (imported, not rerun as
scripts, so their tables and macros are left alone). The figures carry no
value labels or notes; the numbers are in the tables and captions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import figstyle as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
OUT = HERE / "figures"
CC = HERE / "code_council"
sys.path.insert(0, str(CC))
FRONTIER = ["GPT", "Claude", "Gemini", "Grok", "DeepSeek"]
OPEN = ["Qwen", "Llama", "Mistral", "Phi"]
plt = S.plt


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def pct_axis(ax, top=100, ylabel="% of texts"):
    ax.set_ylim(0, top * 1.04)
    ax.set_yticks(range(0, top + 1, 20 if top >= 60 else 10))
    ax.set_ylabel(ylabel)
    S.ygrid(ax)


def top_legend(ax, handles=None, ncol=None, **kw):
    h, l = ax.get_legend_handles_labels() if handles is None else (handles, [x.get_label() for x in handles])
    ax.legend(h, l, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol or len(l),
              borderaxespad=0.1, **kw)


# ------------------------------------------------------------ Figure 1

def funnel() -> None:
    d = pd.read_csv(CC / "verdicts_37_cases.csv")
    sp = pd.read_csv(CC / "strongest_peer.csv")
    beat = {(r.setting, r.judge) for r in sp.itertuples() if r.diff > 0 and r.p_holm < 0.05}
    setting = {("frontier", "chat", "lineup"): "frontier chat-app, lineup",
               ("frontier", "chat", "single"): "frontier chat-app, single text",
               ("frontier", "api", "lineup"): "frontier API, lineup",
               ("frontier", "api", "single"): "frontier API, single text"}
    flagged = d[d.flagged]
    disc = flagged[flagged.discriminates]
    both = disc[disc.has_peer & disc.exceeds_peers]
    nopeer = disc[~disc.has_peer]
    best = both[[(setting.get((r.panel, r.text_set, r.format)), r.judge) in beat
                 for r in both.itertuples()]]
    fr = lambda t: int((t.panel == "frontier").sum())
    ow = lambda t: int((t.panel != "frontier").sum())
    stages = [("Flagged by the\nself-naming test", fr(flagged), ow(flagged), 0),
              ("Distinguish\ntheir own text", fr(disc), ow(disc), 0),
              ("Beat the\npooled peers", fr(both), ow(both), len(nopeer)),
              ("Beat the\nstrongest peer", fr(best), ow(best), 0)]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.TEXT, 2.1),
                                 gridspec_kw={"width_ratios": [1.1, 1]})
    ys = [3, 2, 1, 0]
    for y, (lab, nf, no, nyn) in zip(ys, stages):
        if nf:
            ax.barh(y, nf, color=S.NAVY, height=0.56, zorder=2)
        if no:
            ax.barh(y, no, left=nf, color=S.OPEN, height=0.56, zorder=2)
        if nyn:
            ax.barh(y, nyn, left=nf + no, color="white", edgecolor=S.NAVY, hatch="////",
                    linewidth=0.6, height=0.56, zorder=2)
        ax.text(nf + no + nyn + 0.35, y, str(nf + no + nyn), va="center", ha="left",
                fontsize=S.FS_TICK, color=S.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in stages])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 16.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Cases (of 37)")
    ax.xaxis.grid(True, color=S.GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    top_legend(ax, [Patch(color=S.NAVY, label="Frontier"), Patch(color=S.OPEN, label="Open-weight"),
                    Patch(facecolor="white", edgecolor=S.NAVY, hatch="////", lw=0.6, label="Yes/no")],
               fontsize=S.FS_NOTE)
    ax.set_ylim(-0.6, 3.5)

    P = load(RES / "stats_revision.json")["lineup"]["per_judge"]
    spl = sp[sp.setting == "frontier chat-app, lineup"].set_index("judge")
    judges = ["Claude", "GPT"]
    S.bars(bx, judges, [
        ("Self-naming", [P[j]["self"]["rate"] * 100 for j in judges], S.SELF),
        ("Strongest peer", [spl.loc[j, "best_rate"] * 100 for j in judges], S.BEST),
        ("Pooled peers", [P[j]["peer"]["rate"] * 100 for j in judges], S.PEER),
        ("False alarms", [P[j]["false_alarm"]["rate"] * 100 for j in judges], S.FA)],
        labels=False, width=0.8)
    pct_axis(bx)
    S.chance(bx, label=False)
    top_legend(bx, ncol=2, fontsize=S.FS_NOTE)
    fig.text(0.005, 0.99, "(a)", ha="left", va="top", fontsize=S.FS_TITLE)
    fig.text(0.53, 0.99, "(b)", ha="left", va="top", fontsize=S.FS_TITLE)
    S.save(fig, OUT / "funnel")


# ------------------------------------------------------------ Figure 2

def verdict_map() -> None:
    import build_verdicts as BV
    T = pd.read_csv(CC / "verdicts_37_cases.csv")
    A = T[T.has_peer].copy()
    A["x"], A["y"] = 100 * A.disc, 100 * A.adv
    colour = {"chat": S.CHAT, "api": S.API, "open": S.OPEN}
    fig, ax = plt.subplots(figsize=(S.TEXT, 2.9))
    ax.axhline(0, color="#b0b0b0", lw=0.6, zorder=1)
    ax.axvline(0, color="#b0b0b0", lw=0.6, zorder=1)
    for flagged in (False, True):
        for r in A[A.flagged == flagged].itertuples():
            ax.scatter(r.x, r.y, marker=BV.FMT_MARKER[r.format], s=26, zorder=3 + flagged,
                       facecolors=colour[r.text_set] if flagged else "white",
                       edgecolors=colour[r.text_set], linewidths=0.9)
    # name only the frontier judges that distinguish their own text
    for (setting, judge), (dx, dy) in BV.LABELS.items():
        r = A[(A.setting == setting) & (A.judge == judge)].iloc[0]
        if r.x < 10:
            continue
        ax.annotate(judge, (r.x, r.y), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center", fontsize=S.FS_NOTE, color=S.INK, zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#b0b0b0", lw=0.5, shrinkA=1.5,
                                    shrinkB=3.2) if (abs(dx) > 9 or abs(dy) > 11) else None)
    ax.set_xlim(-22, 92)
    ax.set_ylim(-30, 104)
    ax.set_xticks(range(-20, 91, 20))
    ax.set_yticks(range(-20, 101, 20))
    ax.set_xlabel("Discrimination (points)")
    ax.set_ylabel("Advantage over pooled peers (points)")
    mk = lambda m, fc, ec, lab: Line2D([], [], ls="none", marker=m, ms=4.8, mfc=fc, mec=ec,
                                       mew=0.9, label=lab)
    top_legend(ax, [mk("o", S.CHAT, S.CHAT, "Chat-app essays"), mk("o", S.API, S.API, "API texts"),
                    mk("o", S.OPEN, S.OPEN, "Open-weight texts"),
                    mk("o", "#777777", "#777777", "Lineup"), mk("s", "#777777", "#777777", "One text"),
                    mk("o", "white", "#777777", "Not flagged")], fontsize=S.FS_NOTE,
               columnspacing=1.0)
    S.save(fig, OUT / "verdict_map")


# ------------------------------------------------------------ Figure 3

def lead() -> None:
    St = load(RES / "stats_revision.json")
    for cond, h, leg in (("lineup", 1.75, True), ("single", 1.5, False)):
        P = St[cond]["per_judge"]
        fig, ax = plt.subplots(figsize=(S.TEXT, h))
        S.bars(ax, FRONTIER, [
            ("Self-naming rate", [P[j]["self"]["rate"] * 100 for j in FRONTIER], S.SELF),
            ("Peer baseline", [P[j]["peer"]["rate"] * 100 for j in FRONTIER], S.PEER),
            ("False-alarm rate", [P[j]["false_alarm"]["rate"] * 100 for j in FRONTIER], S.FA)],
            labels=False, width=0.72)
        pct_axis(ax)
        S.chance(ax, label=False)
        if leg:
            top_legend(ax)
        S.save(fig, OUT / f"lead_{cond}")


# ------------------------------------------------------------ Figure 4

def naming_matrix() -> None:
    import build_robust as BR
    NT = {"lineup": BR.naming(BR.load_chat_lineup(ROOT)),
          "single": BR.naming(BR.load_chat_single(ROOT))}
    M = BR.M
    fig = plt.figure(figsize=(S.TEXT, 2.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
    im = None
    for k, (key, title) in enumerate([("lineup", "(a) Lineup"), ("single", "(b) One text at a time")]):
        ax = fig.add_subplot(gs[0, k])
        Z = NT[key].pivot(index="judge", columns="named_author", values="hit_rate").loc[M, M].to_numpy()
        im = ax.imshow(Z, cmap=S.SEQ, vmin=0, vmax=100, aspect="auto")
        for i in range(5):
            for j in range(5):
                rgb = S.SEQ(Z[i, j] / 100)[:3]
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                ax.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center", fontsize=S.FS_NOTE,
                        color="white" if lum < 0.45 else S.INK)
        ax.set_xticks(np.arange(-0.5, 5), minor=True)
        ax.set_yticks(np.arange(-0.5, 5), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="both", length=0)
        for i in range(5):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=S.INK, lw=0.9,
                                   zorder=5, clip_on=False))
        ax.set_xticks(range(5))
        ax.set_xticklabels(M, fontsize=S.FS_NOTE)
        ax.set_yticks(range(5))
        ax.set_yticklabels(M if k == 0 else [])
        ax.set_xlabel("Author")
        if k == 0:
            ax.set_ylabel("Judge")
        ax.set_title(title)
        for s in ax.spines.values():
            s.set_visible(False)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 50, 100])
    cb.outline.set_linewidth(0)
    cb.ax.tick_params(labelsize=S.FS_NOTE, length=0)
    cb.set_label("% of the author's essays", fontsize=S.FS_NOTE)
    S.save(fig, OUT / "naming_matrix")


# ------------------------------------------------------------ Figure 5

def shared38() -> None:
    import build_shared as BS
    res, _, _ = BS.compute(BS.load_chat(ROOT), BS.load_api(ROOT), 20261007, 10000)
    quants = [("self", "Self-naming"), ("peer", "Peer baseline"), ("fa", "False alarms")]
    sets = [("chat", "Chat-app essays", "o", -0.12, S.CHAT), ("api", "API texts", "s", +0.12, S.API)]
    fig, axes = plt.subplots(1, 2, figsize=(S.TEXT, 2.05), sharey=True)
    for ax, (j, title) in zip(axes, (("Claude", "(a) Claude"), ("GPT", "(b) GPT"))):
        for x, (q, _) in enumerate(quants):
            for lab, _, mk, dx, col in sets:
                c = res[j][lab, q]
                ax.plot([x + dx, x + dx], [c["lo"], c["hi"]], color=col, lw=1.3,
                        solid_capstyle="round", zorder=2)
                ax.plot(x + dx, c["est"], mk, color=col, ms=5, mec="white", mew=0.7, zorder=3)
        ax.set_xticks(range(len(quants)))
        ax.set_xticklabels([n for _, n in quants])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, 2.5)
        S.ygrid(ax)
        ax.set_ylim(0, 102)
        ax.set_yticks(range(0, 101, 20))
        S.chance(ax, label=False)
        ax.set_title(title)
    axes[0].set_ylabel("% of texts")
    handles = [Line2D([0], [0], color=col, marker=mk, ms=5, mec="white", mew=0.7, ls="none",
                      label=name) for _, name, mk, _, col in sets]
    fig.legend(handles=handles, loc="outside upper right", ncol=2)
    S.save(fig, OUT / "shared38")


# ------------------------------------------------------------ appendix

def appendix() -> None:
    E = load(RES / "engine_b_results.json")
    for cond in ("lineup", "single"):
        sh = E[cond]["blame"]["pooled_shares"]
        fig, ax = plt.subplots(figsize=(S.HALF, 1.6))
        S.bars(ax, FRONTIER, [("Share of guesses", [sh[j] * 100 for j in FRONTIER], S.NAVY)],
               labels=False, width=0.56)
        ax.set_xticklabels(FRONTIER, fontsize=S.FS_NOTE)
        pct_axis(ax, 60, "% of guesses")
        S.chance(ax, label=False)
        S.save(fig, OUT / f"guess_{cond}")

    F = load(RES / "revision" / "frontier" / "summary.json")["binary"]["judges"]
    O = load(RES / "revision" / "openweight" / "summary.json")["binary"]["judges"]
    for name, names, D, leg in (("yes_frontier", FRONTIER, F, True), ("yes_open", OPEN, O, True)):
        fig, ax = plt.subplots(figsize=(S.HALF, 1.8))
        S.bars(ax, names, [("Own texts", [D[j]["hit_rate"] * 100 for j in names], S.SELF),
                           ("Others' texts", [D[j]["false_alarm"] * 100 for j in names], S.FA)],
               labels=False, width=0.66)
        ax.set_xticklabels(names, fontsize=S.FS_NOTE)
        pct_axis(ax, 100, '% answered "yes"')
        if leg:
            top_legend(ax, fontsize=S.FS_NOTE)
        S.save(fig, OUT / name)

    Ej = E["lineup"]["cross_attribution"]["per_judge"]
    Sh = load(RES / "revision" / "frontier" / "summary.json")["shuffle_lineup"]["judges"]
    B = load(RES / "revision" / "frontier" / "bow_shuffle.json")
    judges = [j for j in FRONTIER if j != "Claude"]
    cats = judges + ["Bag-of-words"]
    orig = [Ej[j]["overall_accuracy"] * 100 for j in judges] + [B["original"]["accuracy"] * 100]
    shuf = [Sh[j]["accuracy"] * 100 for j in judges] + [B["shuffled"]["accuracy"] * 100]
    fig, ax = plt.subplots(figsize=(0.8 * S.TEXT, 1.8))
    S.bars(ax, cats, [("Original essays", orig, S.NAVY), ("Words shuffled", shuf, S.LIGHT)],
           labels=False, width=0.66)
    pct_axis(ax, 100, "Accuracy (%)")
    S.chance(ax, label=False)
    top_legend(ax, fontsize=S.FS_NOTE)
    S.save(fig, OUT / "shuffle")


if __name__ == "__main__":
    S.apply()
    OUT.mkdir(exist_ok=True)
    for f in (funnel, verdict_map, lead, naming_matrix, shared38, appendix):
        f()
        print("drew", f.__name__)
