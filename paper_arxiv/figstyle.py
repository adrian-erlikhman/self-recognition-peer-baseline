"""One visual style for every figure in the paper.

A restrained palette of navy, two lighter blues and a light grey, separated
by lightness so every figure also reads in greyscale; the figures carry no
value labels or notes (the numbers are in the tables and captions).

  measures    self-naming rate  navy      strongest peer    mid blue
              peer baseline     light blue  false-alarm rate light grey
  text sets   chat-app essays   navy      API texts         mid blue
              open-weight       grey      (also the open-weight panel)
  neutral     navy, with light blue for a second condition

Each figure is drawn at the width it is printed at (TEXT = the 5.5 in text
block, HALF = one of two side-by-side subfigures), so every font prints at
its nominal size.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

TEXT, HALF = 5.5, 2.7

# A restrained palette: navy, two lighter blues and a light grey, separated by
# lightness so every figure also reads in greyscale.
NAVY, MID, LIGHT, GREYL = "#1d3f6e", "#4f7fc0", "#a9c4e6", "#cfd3d9"
SELF, BEST, PEER, FA = NAVY, MID, LIGHT, GREYL
CHAT, API, OPEN = NAVY, MID, "#9aa3ad"
SLATE, SLATE_LIGHT = NAVY, LIGHT
INK, MUTED, GRID, CHANCE = "#1f1f1f", "#666666", "#e6e6e6", "#8c8c8c"

SEQ = LinearSegmentedColormap.from_list(
    "paper_blue", ["#f4f8fd", "#c4daf3", "#6fa3e3", SELF, "#164a8a", "#0b2a52"])

FS_TICK, FS_LABEL, FS_NOTE, FS_TITLE = 7, 7.5, 6.5, 8


def apply() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": FS_LABEL, "axes.labelsize": FS_LABEL, "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TICK,
        "axes.edgecolor": "#444444", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": "#444444", "ytick.color": "#444444",
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titlepad": 4,
        "legend.frameon": False, "legend.handlelength": 1.0, "legend.handleheight": 0.8,
        "legend.columnspacing": 1.2, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
    })


def ygrid(ax) -> None:
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def chance(ax, y=20, label=True, x=None) -> None:
    ax.axhline(y, color=CHANCE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    if label:
        x0 = ax.get_xlim()[1] if x is None else x
        ax.text(x0, y + 1.2, "chance", ha="right", va="bottom", fontsize=FS_NOTE, color=MUTED)


def fmt(v: float) -> str:
    return "0" if round(v, 1) == 0 else ("100" if round(v, 1) == 100 else f"{v:.1f}")


def bars(ax, cats, series, labels=True, width=0.8, label_size=FS_NOTE):
    """Grouped bars. series = [(name, values, colour)]."""
    n = len(series)
    w = width / n
    out = []
    for k, (name, vals, col) in enumerate(series):
        xs = [i + (k - (n - 1) / 2) * w for i in range(len(cats))]
        b = ax.bar(xs, vals, width=w, color=col, label=name, edgecolor="white",
                   linewidth=0.6, zorder=2)
        out.append(b)
        if labels:
            for r, v in zip(b, vals):
                ax.text(r.get_x() + r.get_width() / 2, v + 1.5, fmt(v), ha="center",
                        va="bottom", fontsize=label_size, color=INK, zorder=3)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(cats) - 0.45)
    return out


def chance_handle(y=20):
    from matplotlib.lines import Line2D
    return Line2D([0], [0], color=CHANCE, lw=0.7, ls=(0, (3, 2)), label=f"Chance ({y}%)")


def legend_top(ax, ncol=None, with_chance=False, **kw):
    h, l = ax.get_legend_handles_labels()
    if with_chance:
        c = chance_handle()
        h, l = h + [c], l + [c.get_label()]
    return ax.legend(h, l, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol or len(l),
                     borderaxespad=0.1, **kw)


def save(fig, path_no_ext) -> None:
    fig.savefig(f"{path_no_ext}.pdf")
    fig.savefig(f"{path_no_ext}.png", dpi=300)
    plt.close(fig)
