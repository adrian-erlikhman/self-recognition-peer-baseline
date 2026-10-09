"""Figures of the arXiv version, from the results JSON.

    python paper_arxiv/make_figures.py

Bar charts with the value printed on each bar, two colours, one PDF per
panel so the LaTeX can lay panels out as (a)/(b) subfigures.

  lead_lineup / lead_single   self-naming rate, peer baseline and false-alarm
                              rate per frontier judge (chat-app essays)
  yes_frontier / yes_open     share of "yes" to "Did you write this text?",
                              own texts against others'
  guess_lineup / guess_single share of all guesses each frontier model receives
  shuffle                     lineup accuracy on original and word-shuffled
                              essays, with the style classifiers for reference
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"
BLUE, LAV, GREY, INK = "#6C8FD9", "#B892DE", "#9a9a9a", "#222222"
FRONTIER = ["GPT", "Claude", "Gemini", "Grok", "DeepSeek"]
OPEN = ["Qwen", "Llama", "Mistral", "Phi"]

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "xtick.labelsize": 6.8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6})


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def label(ax, bars, fmt="{:.1f}", size=6.5):
    for b in bars:
        h = b.get_height()
        if h is None or (isinstance(h, float) and math.isnan(h)):
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + 1.2, fmt.format(h), ha="center",
                va="bottom", fontsize=size, color=INK)


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def grouped(name, cats, series, ylabel, chance=None, legend=True, ylim=100, w=3.0, h=1.9):
    fig, ax = plt.subplots(figsize=(w, h))
    n = len(series)
    width = 0.8 / n
    for k, (lab, vals, col) in enumerate(series):
        xs = [i + (k - (n - 1) / 2) * width for i in range(len(cats))]
        bars = ax.bar(xs, vals, width=width * 0.95, color=col, label=lab)
        label(ax, bars, size=6.5 if n < 3 else 5.2)
    if chance is not None:
        ax.axhline(chance, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_ylim(0, ylim * 1.12)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=False, fontsize=6.5, loc="lower center", ncol=n,
                  bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0)
    save(fig, name)


def lead() -> None:
    S = load(RES / "stats_revision.json")
    for cond in ("lineup", "single"):
        P = S[cond]["per_judge"]
        grouped(f"lead_{cond}", FRONTIER, [
            ("Self-naming rate", [P[j]["self"]["rate"] * 100 for j in FRONTIER], BLUE),
            ("Peer baseline", [P[j]["peer"]["rate"] * 100 for j in FRONTIER], LAV),
            ("False-alarm rate", [P[j]["false_alarm"]["rate"] * 100 for j in FRONTIER], GREY),
        ], "% of texts", chance=20, legend=True, w=3.4)


def yes() -> None:
    F = load(RES / "revision" / "frontier" / "summary.json")["binary"]["judges"]
    grouped("yes_frontier", FRONTIER, [
        ("Own texts", [F[j]["hit_rate"] * 100 for j in FRONTIER], BLUE),
        ("Others' texts", [F[j]["false_alarm"] * 100 for j in FRONTIER], LAV),
    ], "% answered \"yes\"")
    O = load(RES / "revision" / "openweight" / "summary.json")["binary"]["judges"]
    grouped("yes_open", OPEN, [
        ("Own texts", [O[j]["hit_rate"] * 100 for j in OPEN], BLUE),
        ("Others' texts", [O[j]["false_alarm"] * 100 for j in OPEN], LAV),
    ], "% answered \"yes\"", legend=False)


def guesses() -> None:
    E = load(RES / "engine_b_results.json")
    for cond in ("lineup", "single"):
        sh = E[cond]["blame"]["pooled_shares"]
        fig, ax = plt.subplots(figsize=(3.0, 1.9))
        bars = ax.bar(FRONTIER, [sh[j] * 100 for j in FRONTIER], color=BLUE, width=0.6)
        label(ax, bars)
        ax.axhline(20, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.set_ylim(0, 65)
        ax.set_ylabel("% of all guesses")
        save(fig, f"guess_{cond}")


def shuffle() -> None:
    E = load(RES / "engine_b_results.json")["lineup"]["cross_attribution"]["per_judge"]
    S = load(RES / "revision" / "frontier" / "summary.json")["shuffle_lineup"]["judges"]
    judges = [j for j in FRONTIER if j != "Claude"]
    B = load(RES / "revision" / "frontier" / "bow_shuffle.json")
    cats = judges + ["Bag-of-words"]
    orig = [E[j]["overall_accuracy"] * 100 for j in judges] + [B["original"]["accuracy"] * 100]
    shuf = [S[j]["accuracy"] * 100 for j in judges] + [B["shuffled"]["accuracy"] * 100]
    grouped("shuffle", cats, [("Original essays", orig, BLUE), ("Words shuffled", shuf, LAV)],
            "Attribution accuracy (%)", chance=20, w=4.6, h=2.0)


if __name__ == "__main__":
    # Figure 1 is drawn by code_council/make_lead_figures.py: stacked panels at
    # full text width, so the bar labels print at 7 pt (lead() above is the
    # earlier side-by-side version).
    import subprocess, sys
    for script in ("make_lead_figures.py", "make_funnel_figure.py"):
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "code_council" / script)],
                       check=True)
    yes()
    guesses()
    shuffle()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
