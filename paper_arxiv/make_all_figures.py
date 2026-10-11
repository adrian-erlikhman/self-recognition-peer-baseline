"""Every figure in the paper, in one style (figstyle.py).

    python paper_arxiv/make_all_figures.py

Reads only computed results: results/*.json, code_council/verdicts_37_cases.csv
and strongest_peer.csv, and the shared-prompt and naming computations of
code_council/build_shared.py and build_robust.py (imported, not rerun as
scripts, so their tables and macros are left alone).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

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
    ax.set_ylim(0, top * 1.14)
    ax.set_yticks(range(0, top + 1, 20 if top >= 60 else 10))
    ax.set_ylabel(ylabel)
    S.ygrid(ax)


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
    notes = [f"−{len(flagged) - len(disc)} false-alarm control",
             f"−{len(disc[disc.has_peer & ~disc.exceeds_peers])} pooled peer baseline",
             f"−{len(both) - len(best)} strongest single peer"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.TEXT, 2.35),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    ys = [3, 2, 1, 0]
    for y, (lab, nf, no, nyn) in zip(ys, stages):
        if nf:
            ax.barh(y, nf, color=S.SLATE, height=0.58, zorder=2)
        if no:
            ax.barh(y, no, left=nf, color=S.OPEN, height=0.58, zorder=2)
        if nyn:
            ax.barh(y, nyn, left=nf + no, color="white", edgecolor=S.SLATE, hatch="////",
                    linewidth=0.6, height=0.58, zorder=2)
        tot = nf + no + nyn
        ax.text(tot + 0.4, y, f"{nf + no}" + (f" + {nyn} yes/no" if nyn else ""),
                va="center", ha="left", fontsize=S.FS_TICK, color=S.INK)
    for y, t in zip([2.5, 1.5, 0.5], notes):
        ax.text(16.5, y, t, va="center", ha="right", fontsize=S.FS_NOTE, color=S.MUTED,
                style="italic")
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in stages])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 17)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Cases (of 37)")
    ax.xaxis.grid(True, color=S.GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=S.SLATE, label="Frontier"),
                       Patch(color=S.OPEN, label="Open-weight"),
                       Patch(facecolor="white", edgecolor=S.SLATE, hatch="////", lw=0.6,
                             label="Yes/no (no peer baseline)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=S.FS_NOTE,
              columnspacing=0.8)
    ax.set_title("(a) The audit", pad=16)

    P = load(RES / "stats_revision.json")["lineup"]["per_judge"]
    spl = sp[sp.setting == "frontier chat-app, lineup"].set_index("judge")
    judges = ["Claude", "GPT"]
    series = [("Self-naming rate", [P[j]["self"]["rate"] * 100 for j in judges], S.SELF),
              ("Peer baseline", [P[j]["peer"]["rate"] * 100 for j in judges], S.PEER),
              ("Strongest peer", [spl.loc[j, "best_rate"] * 100 for j in judges], S.BEST),
              ("False-alarm rate", [P[j]["false_alarm"]["rate"] * 100 for j in judges], S.FA)]
    S.bars(bx, judges, series, labels=False, width=0.84)
    w = 0.84 / 4
    for k, (_, vals, _) in enumerate(series):
        for i, v in enumerate(vals):
            t = S.fmt(v) if k != 2 else f"{S.fmt(v)}\n{spl.loc[judges[i], 'best_peer']}"
            bx.text(i + (k - 1.5) * w, v + 1.5, t, ha="center", va="bottom",
                    fontsize=S.FS_NOTE, color=S.INK, linespacing=0.95)
    pct_axis(bx)
    S.chance(bx, label=False)
    S.legend_top(bx, ncol=3, with_chance=True, fontsize=S.FS_NOTE, columnspacing=0.8)
    bx.set_title("(b) Claude and GPT, chat-app lineup", pad=24)
    S.save(fig, OUT / "funnel")


# ------------------------------------------------------------ Figure 2

def verdict_map() -> None:
    import build_verdicts as BV
    T = pd.read_csv(CC / "verdicts_37_cases.csv")
    A = T[T.has_peer].copy()
    A["x"], A["y"] = 100 * A.disc, 100 * A.adv
    colour = {"chat": S.CHAT, "api": S.API, "open": S.OPEN}
    edge = {"chat": "#0f7a54", "api": "#8a5b00", "open": "#b84d78"}
    fig, ax = plt.subplots(figsize=(S.TEXT, 3.2))

    def region(x0, y0, w, h, text, tx, ty, ha="left", va="top"):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=3",
                                    fc="#f3f3f1", ec="none", zorder=0, mutation_aspect=1.2))
        ax.text(tx, ty, text, fontsize=S.FS_NOTE, color=S.MUTED, ha=ha, va=va,
                style="italic", zorder=1, linespacing=1.15)
    region(-11, 8, 22, 94, "Names itself\neverywhere", -20.5, 62, va="center")
    region(13, -14, 77, 24, "Tells its own text apart,\npeers match it", 89, -12.5,
           ha="right", va="bottom")
    region(13, 20, 77, 28, "Both", 89, 46.5, ha="right")
    ax.axhline(0, color="#9a9a9a", lw=0.6, zorder=1)
    ax.axvline(0, color="#9a9a9a", lw=0.6, zorder=1)
    for flagged in (False, True):
        for r in A[A.flagged == flagged].itertuples():
            ax.scatter(r.x, r.y, marker=BV.FMT_MARKER[r.format], s=30, zorder=3 + flagged,
                       facecolors=colour[r.text_set] if flagged else "white",
                       edgecolors=edge[r.text_set], linewidths=0.9)
    for (setting, judge), (dx, dy) in BV.LABELS.items():
        r = A[(A.setting == setting) & (A.judge == judge)].iloc[0]
        far = abs(dx) > 9 or abs(dy) > 11
        ax.annotate(judge, (r.x, r.y), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center", fontsize=S.FS_NOTE, color=S.INK, zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.5, shrinkA=1.5,
                                    shrinkB=3.2) if far else None)
    ax.set_xlim(-22, 92)
    ax.set_ylim(-30, 104)
    ax.set_xticks(range(-20, 91, 20))
    ax.set_yticks(range(-20, 101, 20))
    ax.set_xlabel("Discrimination: self-naming on own texts minus on others' texts (points)")
    ax.set_ylabel("Self-advantage over the pooled peers (points)")

    def h(marker, fc, ec, lab):
        return Line2D([], [], ls="none", marker=marker, ms=5, mfc=fc, mec=ec, mew=0.9, label=lab)

    def legend(handles, title, x):
        leg = ax.legend(handles=handles, title=title, fontsize=S.FS_NOTE,
                        title_fontsize=S.FS_NOTE, loc="upper left", bbox_to_anchor=(x, 1.0),
                        handletextpad=0.2, labelspacing=0.3, borderpad=0.1, alignment="left")
        leg.get_title().set_color(S.MUTED)
        leg.set_zorder(7)
        ax.add_artist(leg)
    legend([h("o", colour[s], edge[s], BV.SET_LABEL[s]) for s in ("chat", "api", "open")],
           "Texts (colour)", 0.30)
    legend([h(BV.FMT_MARKER[f], "#9a9a9a", "#666666", BV.FMT_LABEL[f]) for f in ("lineup", "single")],
           "Format (shape)", 0.55)
    legend([h("D", S.SLATE, S.SLATE, "Flagged"), h("D", "white", S.SLATE, "Not flagged")],
           "Self-naming test (fill)", 0.74)
    S.save(fig, OUT / "verdict_map")


# ------------------------------------------------------------ Figure 3

def lead() -> None:
    St = load(RES / "stats_revision.json")
    for cond, h, leg in (("lineup", 1.85, True), ("single", 1.6, False)):
        P = St[cond]["per_judge"]
        fig, ax = plt.subplots(figsize=(S.TEXT, h))
        S.bars(ax, FRONTIER, [
            ("Self-naming rate", [P[j]["self"]["rate"] * 100 for j in FRONTIER], S.SELF),
            ("Peer baseline", [P[j]["peer"]["rate"] * 100 for j in FRONTIER], S.PEER),
            ("False-alarm rate", [P[j]["false_alarm"]["rate"] * 100 for j in FRONTIER], S.FA)])
        pct_axis(ax)
        S.chance(ax, label=False)
        if leg:
            S.legend_top(ax, with_chance=True)
        S.save(fig, OUT / f"lead_{cond}")


# ------------------------------------------------------------ Figure 4

def naming_matrix() -> None:
    import build_robust as BR
    NT = {"lineup": BR.naming(BR.load_chat_lineup(ROOT)),
          "single": BR.naming(BR.load_chat_single(ROOT))}
    M = BR.M
    fig = plt.figure(figsize=(S.TEXT, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045])
    im = None
    for k, (key, title) in enumerate([("lineup", "(a) Lineup"), ("single", "(b) One text at a time")]):
        ax = fig.add_subplot(gs[0, k])
        Z = NT[key].pivot(index="judge", columns="named_author", values="hit_rate").loc[M, M].to_numpy()
        im = ax.imshow(Z, cmap=S.SEQ, vmin=0, vmax=100, aspect="auto")
        for i in range(5):
            for j in range(5):
                rgb = S.SEQ(Z[i, j] / 100)[:3]
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                ax.text(j, i, S.fmt(Z[i, j]), ha="center", va="center", fontsize=S.FS_NOTE,
                        color="white" if lum < 0.45 else S.INK,
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(np.arange(-0.5, 5), minor=True)
        ax.set_yticks(np.arange(-0.5, 5), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="both", length=0)
        for i in range(5):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=S.INK, lw=1.1,
                                   zorder=5, clip_on=False))
        ax.set_xticks(range(5))
        ax.set_xticklabels(M, fontsize=S.FS_NOTE)
        ax.set_yticks(range(5))
        ax.set_yticklabels(M if k == 0 else [])
        ax.set_xlabel("Author of the essays")
        if k == 0:
            ax.set_ylabel("Judge")
        ax.set_title(title)
        for s in ax.spines.values():
            s.set_visible(False)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 25, 50, 75, 100])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=S.FS_NOTE, length=2, width=0.5)
    cb.set_label("Judge names the author (%)", fontsize=S.FS_TICK)
    S.save(fig, OUT / "naming_matrix")


# ------------------------------------------------------------ Figure 5

def shared38() -> None:
    import build_shared as BS
    res, _, _ = BS.compute(BS.load_chat(ROOT), BS.load_api(ROOT), 20261007, 10000)
    quants = [("self", "Self-naming\nrate", S.SELF), ("peer", "Peer\nbaseline", S.PEER),
              ("fa", "False-alarm\nrate", S.FA)]
    sets = [("chat", "Chat-app essays", "o", -0.15, S.CHAT), ("api", "API texts", "s", +0.15, S.API)]
    fig, axes = plt.subplots(1, 2, figsize=(S.TEXT, 2.3), sharey=True)
    for ax, (j, title) in zip(axes, (("Claude", "(a) Claude"), ("GPT", "(b) GPT"))):
        for x, (q, _, _) in enumerate(quants):
            for lab, _, mk, dx, col in sets:
                c = res[j][lab, q]
                ax.plot([x + dx, x + dx], [c["lo"], c["hi"]], color=col, lw=1.4,
                        solid_capstyle="round", zorder=2)
                ax.plot(x + dx, c["est"], mk, color=col, ms=5.5, mec="white", mew=0.8, zorder=3)
                ha, off = ("right", -0.07) if dx < 0 else ("left", 0.07)
                ax.text(x + dx + off, c["est"], S.fmt(c["est"]), ha=ha, va="center",
                        fontsize=S.FS_NOTE, color=S.INK, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.06", fc="white", ec="none"))
        ax.set_xticks(range(len(quants)))
        ax.set_xticklabels([n for _, n, _ in quants])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, 2.6)
        S.ygrid(ax)
        ax.set_ylim(0, 105)
        ax.set_yticks(range(0, 101, 20))
        S.chance(ax, label=False)
        ax.set_title(title)
    axes[0].set_ylabel("% of texts")
    handles = [Line2D([0], [0], color=col, marker=mk, ms=5.5, mec="white", mew=0.8, lw=1.4,
                      label=name) for _, name, mk, _, col in sets] + [S.chance_handle()]
    fig.legend(handles=handles, loc="outside upper right", ncol=3, fontsize=S.FS_TICK)
    S.save(fig, OUT / "shared38")


# ------------------------------------------------------------ appendix

def appendix() -> None:
    E = load(RES / "engine_b_results.json")
    for cond in ("lineup", "single"):
        sh = E[cond]["blame"]["pooled_shares"]
        fig, ax = plt.subplots(figsize=(S.HALF, 1.75))
        S.bars(ax, FRONTIER, [("Share of guesses", [sh[j] * 100 for j in FRONTIER], S.SLATE)],
               width=0.6)
        ax.set_xticklabels(FRONTIER, fontsize=S.FS_NOTE)
        pct_axis(ax, 60, "% of all guesses")
        S.chance(ax, label=False)
        S.save(fig, OUT / f"guess_{cond}")

    F = load(RES / "revision" / "frontier" / "summary.json")["binary"]["judges"]
    O = load(RES / "revision" / "openweight" / "summary.json")["binary"]["judges"]
    for name, names, D, leg in (("yes_frontier", FRONTIER, F, True), ("yes_open", OPEN, O, False)):
        fig, ax = plt.subplots(figsize=(S.HALF, 1.95 if leg else 1.75))
        S.bars(ax, names, [("Own texts", [D[j]["hit_rate"] * 100 for j in names], S.SELF),
                           ("Others' texts", [D[j]["false_alarm"] * 100 for j in names], S.FA)],
               label_size=5.8)
        ax.set_xticklabels(names, fontsize=S.FS_NOTE)
        pct_axis(ax, 100, '% answered "yes"')
        if leg:
            S.legend_top(ax)
        S.save(fig, OUT / name)

    Ej = E["lineup"]["cross_attribution"]["per_judge"]
    Sh = load(RES / "revision" / "frontier" / "summary.json")["shuffle_lineup"]["judges"]
    B = load(RES / "revision" / "frontier" / "bow_shuffle.json")
    judges = [j for j in FRONTIER if j != "Claude"]
    cats = judges + ["Bag-of-words"]
    orig = [Ej[j]["overall_accuracy"] * 100 for j in judges] + [B["original"]["accuracy"] * 100]
    shuf = [Sh[j]["accuracy"] * 100 for j in judges] + [B["shuffled"]["accuracy"] * 100]
    fig, ax = plt.subplots(figsize=(0.8 * S.TEXT, 2.0))
    S.bars(ax, cats, [("Original essays", orig, S.SLATE), ("Words shuffled", shuf, S.SLATE_LIGHT)])
    pct_axis(ax, 100, "Attribution accuracy (%)")
    S.chance(ax, label=False)
    S.legend_top(ax, with_chance=True)
    S.save(fig, OUT / "shuffle")


if __name__ == "__main__":
    S.apply()
    OUT.mkdir(exist_ok=True)
    for f in (funnel, verdict_map, lead, naming_matrix, shared38, appendix):
        f()
        print("drew", f.__name__)
