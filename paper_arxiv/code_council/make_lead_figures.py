"""Figure 1 panels (lead_lineup, lead_single), from the results JSON.

    python make_lead_figures.py [--repo REPO] [--out ../figures]

Self-naming rate, peer baseline and false-alarm rate per frontier judge on the
chat-app essays, one PDF (and a 300 dpi PNG) per panel. Reduced from
paper_arxiv/make_figures.py: same data, palette and fonts. Each panel is drawn
at the full 5.5 in text width with no tight-bbox rescaling, so that included
at width=\\linewidth every font prints at its nominal size (7 pt or larger).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, LAV, GREY, INK = "#6C8FD9", "#B892DE", "#9a9a9a", "#222222"
FRONTIER = ["GPT", "Claude", "Gemini", "Grok", "DeepSeek"]
TEXT_WIDTH_IN = 5.5
PANEL_HEIGHT_IN = 1.75
LABEL_PT = 7.0
# (legend label, key in per_judge, fill colour, hatch). The hatches keep the
# three series apart in greyscale, where the three fills have similar lightness.
SERIES = [("Self-naming rate", "self", BLUE, None),
          ("Peer baseline", "peer", LAV, "//////"),
          ("False-alarm rate", "false_alarm", GREY, "......")]

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "hatch.linewidth": 0.5, "hatch.color": "white",
                     "pdf.fonttype": 42})


def fmt(v: float) -> str:
    return "0" if v == 0 else f"{v:.1f}"


def values(stats: dict, cond: str) -> dict:
    P = stats[cond]["per_judge"]
    return {key: [P[j][key]["rate"] * 100 for j in FRONTIER] for _, key, _, _ in SERIES}


def panel(name: str, vals: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, PANEL_HEIGHT_IN), layout="constrained")
    n = len(SERIES)
    width = 0.8 / n
    for k, (lab, key, col, hatch) in enumerate(SERIES):
        xs = [i + (k - (n - 1) / 2) * width for i in range(len(FRONTIER))]
        bars = ax.bar(xs, vals[key], width=width * 0.92, color=col, label=lab,
                      hatch=hatch, edgecolor="white", linewidth=0)
        for b, v in zip(bars, vals[key]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, fmt(v), ha="center",
                    va="bottom", fontsize=LABEL_PT, color=INK, zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.axhline(20, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(range(len(FRONTIER)))
    ax.set_xticklabels(FRONTIER)
    ax.set_xlim(-0.55, len(FRONTIER) - 0.45)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("% of texts")
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", ncol=n,
              bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.6)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "figures"))
    a = ap.parse_args()
    stats = json.loads((Path(a.repo) / "results" / "stats_revision.json")
                       .read_text(encoding="utf-8"))
    for cond in ("lineup", "single"):
        panel(f"lead_{cond}", values(stats, cond), Path(a.out))
    print("wrote lead_lineup, lead_single (pdf, png) to", a.out)


if __name__ == "__main__":
    main()
