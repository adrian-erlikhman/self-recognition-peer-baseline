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


def counts(csv: Path) -> dict:
    d = pd.read_csv(csv)
    front = d.panel == "frontier"
    flagged = d[d.flagged]
    disc = flagged[flagged.discriminates]
    both = disc[disc.has_peer & disc.exceeds_peers]
    nopeer = disc[~disc.has_peer]
    return {
        "cells": len(d),
        "stages": [
            ("Flagged by the\nself-naming test", (flagged.panel == "frontier").sum(), (flagged.panel != "frontier").sum(), 0),
            ("Also\ndiscriminate", (disc.panel == "frontier").sum(), (disc.panel != "frontier").sum(), 0),
            ("Also exceed\ntheir peers", (both.panel == "frontier").sum(), (both.panel != "frontier").sum(), len(nopeer)),
        ],
        "fa_removed": len(flagged) - len(disc),
        "fa_open": (flagged[~flagged.discriminates].panel != "frontier").sum(),
        "peer_removed": len(disc[disc.has_peer & ~disc.exceeds_peers]),
        "front_cells": int(front.sum()),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default=str(here.parent / "figures"))
    a = ap.parse_args()
    C = counts(here / "verdicts_37_cases.csv")
    stats = json.loads((Path(a.repo) / "results" / "stats_revision.json").read_text(encoding="utf-8"))
    P = stats["lineup"]["per_judge"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.15), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    # Left: the funnel, one horizontal bar per stage, top to bottom.
    ys = [2, 1, 0]
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
    ax.text(9.6, 1.5, f"$-${C['fa_removed']} by the false-alarm\ncontrol ({C['fa_open']} open-weight)", **note)
    ax.text(5.2, 0.5, f"$-${C['peer_removed']} by the peer baseline\n(all frontier)", **note)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                       plt.Rectangle((0, 0), 1, 1, color=PALE),
                       plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, hatch="//////",
                                     edgecolor="white", linewidth=0)],
              labels=["Frontier", "Open-weight", "Yes/no (no peer baseline)"],
              frameon=False, fontsize=7, ncol=3, loc="upper center",
              bbox_to_anchor=(0.42, -0.24), handlelength=1.2, columnspacing=1.0)
    ax.set_title("(a) The two controls", fontsize=8, loc="left")

    # Right: the two judges that pass both controls, chat-app lineup.
    judges = ["Claude", "GPT"]
    w = 0.26
    for k, (lab, key, col, hatch) in enumerate(SERIES):
        xs = [i + (k - 1) * w for i in range(len(judges))]
        vals = [P[j][key]["rate"] * 100 for j in judges]
        bars = bx.bar(xs, vals, width=w * 0.92, color=col, label=lab, hatch=hatch,
                      edgecolor="white", linewidth=0)
        for b, v in zip(bars, vals):
            bx.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=7, color=INK)
    bx.axhline(20, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=0)
    bx.set_xticks(range(len(judges)))
    bx.set_xticklabels(judges)
    bx.set_ylim(0, 105)
    bx.set_yticks([0, 20, 40, 60, 80, 100])
    bx.set_ylabel("% of texts", fontsize=7.5)
    bx.legend(frameon=False, fontsize=6.8, loc="upper right", handlelength=1.4,
              borderaxespad=0.1)
    bx.set_title("(b) Claude and GPT, lineup", fontsize=8, loc="left")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "funnel.pdf")
    fig.savefig(out / "funnel.png", dpi=300)
    plt.close(fig)
    print("wrote funnel (pdf, png) to", out, C["stages"], C["fa_removed"], C["peer_removed"])


if __name__ == "__main__":
    main()
