"""Figure 1 of the arXiv version, from the results JSON.

    python paper_arxiv/make_figures.py

For each frontier judge on the chat-app essays: its self-naming rate, its
peer baseline and its false-alarm rate, in the lineup and one text at a time,
with the self-advantage in points on the right of each panel.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
INK, BLUE, MID, LIGHT = "#1a1a1a", "#1f4e79", "#7a7a7a", "#c9d3de"


def lead() -> None:
    S = json.loads((ROOT / "results" / "stats_revision.json").read_text())
    order = ["Claude", "GPT", "DeepSeek", "Gemini", "Grok"]
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.2), sharey=True)
    for ax, cond, title in zip(axes, ("lineup", "single"), ("(a) Lineup", "(b) One text at a time")):
        for y, j in enumerate(order):
            g = S[cond]["per_judge"][j]
            s, p, fa = g["self"]["rate"], g["peer"]["rate"], g["false_alarm"]["rate"]
            ax.plot([p * 100, s * 100], [y, y], color=LIGHT, lw=3, solid_capstyle="round", zorder=1)
            ax.scatter(fa * 100, y, marker="x", s=24, color=MID, lw=1.1, zorder=2,
                       label="False-alarm rate" if y == 0 else None)
            ax.scatter(p * 100, y, s=30, facecolor="white", edgecolor=BLUE, lw=1.2, zorder=3,
                       label="Peer baseline" if y == 0 else None)
            ax.scatter(s * 100, y, s=30, color=BLUE, zorder=4,
                       label="Self-naming rate" if y == 0 else None)
            adv = (s - p) * 100
            ax.text(104, y, f"{adv:+.0f}".replace("-", "−"), va="center", ha="left", fontsize=8,
                    color=INK, fontweight="bold" if abs(adv) >= 20 else "normal")
        ax.text(104, -0.85, "Adv.", fontsize=7.5, color=MID, ha="left", va="center")
        ax.axvline(20, color=MID, lw=0.7, ls=(0, (2, 2)), zorder=0)
        ax.set_xlim(-2, 100)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("% of texts")
        ax.set_title(title, fontsize=9, loc="left", pad=10)
        ax.tick_params(length=2)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.06), handletextpad=0.3, columnspacing=1.6)
    fig.subplots_adjust(left=0.1, right=0.93, bottom=0.3, top=0.86, wspace=0.32)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "lead.pdf", bbox_inches="tight")
    fig.savefig(OUT / "lead.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    lead()
    print(f"wrote {OUT / 'lead.pdf'}")
