"""Figures for the TACL draft, from the results JSON.

    python paper_tacl/make_figures.py

Figure 1 (lead): for each frontier judge, its self-naming rate, its peer
baseline and its false-alarm rate, in both formats. Encoded by marker shape
and ink, not hue, so it survives black-and-white printing (TACL discourages
colour figures).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
INK, MID, LIGHT = "#1a1a1a", "#6b6b6b", "#bdbdbd"


def lead() -> None:
    S = json.loads((ROOT / "results" / "stats_revision.json").read_text())
    order = ["Claude", "GPT", "DeepSeek", "Gemini", "Grok"]
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.0), sharey=True)
    for ax, cond, title in zip(axes, ("lineup", "single"), ("Lineup", "One text at a time")):
        for y, j in enumerate(order):
            g = S[cond]["per_judge"][j]
            s, p, fa = g["self"]["rate"], g["peer"]["rate"], g["false_alarm"]["rate"]
            ax.plot([p * 100, s * 100], [y, y], color=LIGHT, lw=2.5, solid_capstyle="round", zorder=1)
            ax.scatter(fa * 100, y, marker="x", s=22, color=MID, lw=1.1, zorder=2,
                       label="False alarms (names itself on others' text)" if y == 0 else None)
            ax.scatter(p * 100, y, s=26, facecolor="white", edgecolor=INK, lw=1.1, zorder=3,
                       label="Peer baseline (others name it on its text)" if y == 0 else None)
            ax.scatter(s * 100, y, s=26, color=INK, zorder=4,
                       label="Self-naming (names itself on its own text)" if y == 0 else None)
            adv = (s - p) * 100
            ax.text(101, y, f"{adv:+.0f}", va="center", ha="left", fontsize=7, color=INK)
        ax.axvline(20, color=LIGHT, lw=0.8, ls=(0, (2, 2)), zorder=0)
        ax.set_xlim(-2, 100)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("% of texts")
        ax.set_title(title, fontsize=8, loc="left")
        ax.tick_params(length=2)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    axes[1].text(101, -0.9, "self − peer", fontsize=6.5, color=MID, ha="left")
    axes[0].text(101, -0.9, "self − peer", fontsize=6.5, color=MID, ha="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", ncol=3, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, -0.08), handletextpad=0.3, columnspacing=1.2)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.3, top=0.9, wspace=0.18)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "lead.pdf", bbox_inches="tight")
    fig.savefig(OUT / "lead.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    lead()
    print(f"wrote {OUT / 'lead.pdf'}")
