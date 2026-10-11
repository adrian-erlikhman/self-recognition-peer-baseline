"""Figure 1: what the two controls do to the cases the self-naming test flags,
next to the lineup rates of the two judges that pass both.

    python make_funnel_figure.py [--repo REPO] [--out ../figures]

Left panel: counts from code_council/verdicts_37_cases.csv (flagged, then
discriminates, then exceeds peers), split by panel. The yes/no cases have no
peer baseline, so the one that discriminates (Claude) is drawn hatched at the
last stage. Right panel: self-naming rate, peer baseline and false-alarm rate
of Claude and GPT in the chat-app lineup, from results/stats_revision.json
(the same values as the full figure in Section 4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, LAV, GREY, INK, PALE = "#6C8FD9", "#B892DE", "#9a9a9a", "#222222", "#C9D6F2"
SERIES = [("Self-naming rate", "self", BLUE, None),
          ("Peer baseline", "peer", LAV, "//////"),
          ("False-alarm rate", "false_alarm", GREY, "......")]

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "hatch.linewidth": 0.5, "hatch.color": "white",
                     "pdf.fonttype": 42})


SETTING = {("frontier", "chat", "lineup"): "frontier chat-app, lineup",
           ("frontier", "chat", "single"): "frontier chat-app, single text",
           ("frontier", "api", "lineup"): "frontier API, lineup",
           ("frontier", "api", "single"): "frontier API, single text",
           ("open", "open", "lineup"): "open-weight, lineup",
           ("open", "open", "single"): "open-weight, single text"}


def counts(csv: Path, strongest: Path) -> dict:
    d = pd.read_csv(csv)
    s = pd.read_csv(strongest)
    beat = {(r.setting, r.judge) for r in s.itertuples() if r.diff > 0 and r.p_holm < 0.05}
    front = d.panel == "frontier"
    flagged = d[d.flagged]
    disc = flagged[flagged.discriminates]
    both = disc[disc.has_peer & disc.exceeds_peers]
    nopeer = disc[~disc.has_peer]
    key = lambda r: (SETTING.get((r.panel, r.text_set, r.format)), r.judge)
    best = both[[key(r) in beat for r in both.itertuples()]] if len(both) else both
    return {
        "cells": len(d),
        "stages": [
            ("Flagged by the\nself-naming test", (flagged.panel == "frontier").sum(), (flagged.panel != "frontier").sum(), 0),
            ("Distinguish\ntheir own text", (disc.panel == "frontier").sum(), (disc.panel != "frontier").sum(), 0),
            ("Beat the\npooled peers", (both.panel == "frontier").sum(), (both.panel != "frontier").sum(), len(nopeer)),
            ("Beat the\nstrongest peer", (best.panel == "frontier").sum(), (best.panel != "frontier").sum(), 0),
        ],
        "fa_removed": len(flagged) - len(disc),
        "fa_open": (flagged[~flagged.discriminates].panel != "frontier").sum(),
        "peer_removed": len(disc[disc.has_peer & ~disc.exceeds_peers]),
        "best_removed": len(both) - len(best),
        "front_cells": int(front.sum()),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default=str(here.parent / "figures"))
    a = ap.parse_args()
    C = counts(here / "verdicts_37_cases.csv", here / "strongest_peer.csv")
    stats = json.loads((Path(a.repo) / "results" / "stats_revision.json").read_text(encoding="utf-8"))
    P = stats["lineup"]["per_judge"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.5), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    # Left: the funnel, one horizontal bar per stage, top to bottom.
    ys = [3, 2, 1, 0]
    for y, (lab, nf, no, nyn) in zip(ys, C["stages"]):
        ax.barh(y, nf, color=BLUE, height=0.56, edgecolor="white", linewidth=0)
        ax.barh(y, no, left=nf, color=PALE, height=0.56, edgecolor="white", linewidth=0)
        if nyn:
            ax.barh(y, nyn, left=nf + no, color=BLUE, height=0.56, edgecolor="white",
                    linewidth=0, hatch="//////")
        total = nf + no + nyn
        txt = f"{nf + no}" + (f" + {nyn} yes/no" if nyn else "")
        ax.text(total + 0.35, y, txt, va="center", ha="left", fontsize=8, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in C["stages"]], fontsize=7.5)
    ax.set_xlim(0, 19.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel(f"Cases (of {C['cells']})", fontsize=7.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    note = dict(fontsize=7, color="#555555", ha="left", va="center")
    ax.text(9.6, 2.5, f"$-${C['fa_removed']} by the false-alarm\ncontrol ({C['fa_open']} open-weight)", **note)
    ax.text(8.6, 1.5, f"$-${C['peer_removed']} by the pooled\npeer baseline", **note)
    ax.text(4.6, 0.5, f"$-${C['best_removed']} by the strongest\nsingle peer", **note)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                       plt.Rectangle((0, 0), 1, 1, color=PALE),
                       plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, hatch="//////",
                                     edgecolor="white", linewidth=0)],
              labels=["Frontier", "Open-weight", "Yes/no (no peer baseline)"],
              frameon=False, fontsize=7, ncol=3, loc="upper center",
              bbox_to_anchor=(0.42, -0.24), handlelength=1.2, columnspacing=1.0)
    ax.set_title("(a) The audit", fontsize=8, loc="left")

    # Right: the two judges that beat the pooled peers, chat-app lineup, with
    # the single peer that names each of them most often on its own essays.
    judges = ["Claude", "GPT"]
    sp = pd.read_csv(here / "strongest_peer.csv")
    sp = sp[sp.setting == "frontier chat-app, lineup"].set_index("judge")
    series = SERIES[:2] + [("Strongest single peer", "strongest", "#7E57C2", None)] + SERIES[2:]
    w = 0.2
    for k, (lab, key, col, hatch) in enumerate(series):
        xs = [i + (k - 1.5) * w for i in range(len(judges))]
        if key == "strongest":
            vals = [sp.loc[j, "best_rate"] * 100 for j in judges]
            lab = "Strongest single peer"
        else:
            vals = [P[j][key]["rate"] * 100 for j in judges]
        bars = bx.bar(xs, vals, width=w * 0.92, color=col, label=lab, hatch=hatch,
                      edgecolor="white", linewidth=0)
        for jj, b, v in zip(judges, bars, vals):
            t = f"{v:.0f}\n{sp.loc[jj, 'best_peer']}" if key == "strongest" else f"{v:.0f}"
            bx.text(b.get_x() + b.get_width() / 2, v + 1.5, t, ha="center",
                    va="bottom", fontsize=6.5 if key == "strongest" else 7, color=INK,
                    linespacing=0.9)
    bx.axhline(20, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=0)
    bx.set_xticks(range(len(judges)))
    bx.set_xticklabels(judges)
    bx.set_ylim(0, 112)
    bx.set_yticks([0, 20, 40, 60, 80, 100])
    bx.set_ylabel("% of texts", fontsize=7.5)
    bx.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.45, -0.12), handlelength=1.4, columnspacing=0.8)
    bx.set_title("(b) Claude and GPT, lineup", fontsize=8, loc="left")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "funnel.pdf")
    fig.savefig(out / "funnel.png", dpi=300)
    plt.close(fig)
    print("wrote funnel (pdf, png) to", out, C["stages"], C["fa_removed"], C["peer_removed"])


if __name__ == "__main__":
    main()
