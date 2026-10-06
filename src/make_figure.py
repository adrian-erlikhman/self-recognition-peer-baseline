r"""CompLLM: the paper's main figure.

Builds `paper/figures/two_conditions.pdf` (vector) and `.png` (slides) from
`results/engine_b_results.json` and `results/engine_a_results.json`.

    python src/make_figure.py

Three panels:

    (a) lineup: each judge's self rate (with exact 95% CI) against its peer
        baseline, the rate at which the other four judges identify that same
        text. The gap is the self-advantage, annotated at the right. Chance
        and the stylometric classifier are drawn as reference lines.
    (b) lineup and (c) single-text: each judge's accuracy on all text against
        its accuracy on non-self text. Judges with a self-advantage fall when
        their own text is removed; judges without one rise.

Nothing is hardcoded. Every rate, interval, advantage and significance mark is
read from the results at run time, as is the model order. Legible in
greyscale: identity is carried by marker shape and the tick labels, colour is
layered on top. Palette is Okabe-Ito.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = ROOT / "results" / "engine_b_results.json"
ENGINE_A_JSON = ROOT / "results" / "engine_a_results.json"
FIG_DIR = ROOT / "paper" / "figures"
STEM = "two_conditions"

# --------------------------------------------------------------------------
# Cosmetics. Nothing below encodes a result. The style constants are shared
# with make_figures_supp.py so the whole set reads as one family.
# --------------------------------------------------------------------------
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
HATCHES = ["", "//", "..", "xx", "\\\\"]
MARKERS = ["s", "o", "^", "D", "v"]

INK = "#111111"
GREY = "#888888"
BASE_LW = 0.7
GUTTER = 1.15

FIG_W, FIG_H = 6.4, 2.75
FS_TITLE = 8.5
FS_LABEL = 8.0
FS_TICK = 7.4
FS_VALUE = 7.2
FS_NOTE = 6.6


def _rc() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": FS_LABEL,
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        # Type 42 (TrueType) fonts, which arXiv and NeurIPS accept.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path}: run the analysis first, this script only draws.")
    return json.loads(path.read_text(encoding="utf-8"))


def model_order(data: dict) -> list[str]:
    """Canonical model order, taken from the JSON rather than assumed."""
    order = list(data["lineup"]["blame"]["pooled_shares"].keys())
    for cond in ("lineup", "single"):
        other = list(data[cond]["blame"]["pooled_shares"].keys())
        if other != order:
            raise SystemExit(f"model order disagrees between conditions: {order} vs {other}")
    return order


def style(order: list[str]) -> dict[str, tuple[str, str]]:
    """Colour and marker per model, assigned by position in the canonical order."""
    return {m: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)])
            for i, m in enumerate(order)}


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
def panel_selfadv(ax, lin: dict, classifier: float, sty: dict) -> None:
    sr = {r["judge"]: r for r in lin["self_recognition"]}
    sa = {r["model"]: r for r in lin["self_advantage"]}
    chance = 100.0 / len(sr)
    rows = sorted(sa, key=lambda m: -sa[m]["advantage_pp"])
    ys = list(range(len(rows)))[::-1]

    for m, y in zip(rows, ys):
        colour, marker = sty[m]
        self_r = sr[m]["rate"] * 100
        peer = sa[m]["others_rate"] * 100
        lo = (sr[m]["rate"] - sr[m]["ci_lo"]) * 100
        hi = (sr[m]["ci_hi"] - sr[m]["rate"]) * 100
        ax.plot([peer, self_r], [y, y], color=colour, lw=1.6, zorder=2)
        ax.errorbar(self_r, y, xerr=[[lo], [hi]], fmt="none", ecolor=GREY,
                    elinewidth=0.8, capsize=2, zorder=3)
        ax.plot(peer, y, "o", ms=5, mfc="white", mec=INK, mew=0.8, zorder=4)
        ax.plot(self_r, y, marker, ms=5.5, mfc=colour, mec=INK, mew=0.5, zorder=5)
        sig = sa[m]["sig"] not in ("n.s.", "")
        ax.text(112, y, f"{sa[m]['advantage_pp']:+.1f}", ha="left", va="center",
                fontsize=FS_VALUE, color=INK if sig else GREY)

    ax.axvline(chance, ls="--", lw=0.8, color=GREY, zorder=1)
    ax.axvline(classifier, ls=":", lw=0.9, color=INK, zorder=1)
    ax.text(chance, -0.85, "chance", ha="center", va="top", fontsize=FS_NOTE, color=GREY)
    ax.text(classifier, -0.85, "classifier", ha="center", va="top", fontsize=FS_NOTE)
    ax.text(112, len(rows) - 0.4, "self-advantage", ha="left", va="bottom",
            fontsize=FS_NOTE, color=GREY)

    ax.set_yticks(ys)
    ax.set_yticklabels(rows, fontsize=FS_TICK)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-1.2, len(rows) - 0.3)
    ax.set_xlabel("accuracy on this model's text (%)", fontsize=FS_LABEL)
    ax.set_title("(a) self rate vs. peer baseline (lineup)", fontsize=FS_TITLE, loc="left")
    ax.tick_params(labelsize=FS_TICK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def panel_nonself(ax, cond: dict, title: str, sty: dict, chance: float,
                  ylim: tuple[float, float]) -> None:
    pj = cond["cross_attribution"]["per_judge"]
    for m, (colour, marker) in sty.items():
        ys = [pj[m]["overall_accuracy"] * 100, pj[m]["non_self_accuracy"] * 100]
        ax.plot([0, 1], ys, marker=marker, ms=5, color=colour, mec=INK, mew=0.5,
                lw=1.4, zorder=3)
    ax.axhline(chance, ls="--", lw=0.8, color=GREY, zorder=1)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["all text", "non-self"], fontsize=FS_TICK)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=FS_TITLE, loc="left")
    ax.tick_params(labelsize=FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)


def build(data: dict, ea: dict):
    order = model_order(data)
    sty = style(order)
    chance = 100.0 / len(order)
    classifier = ea["accuracy"]["mean"] * 100
    lin, sng = data["lineup"], data["single"]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(FIG_W, FIG_H), gridspec_kw=dict(width_ratios=[1.7, 1, 1]))
    fig.subplots_adjust(left=0.10, right=0.995, top=0.90, bottom=0.30, wspace=0.42)

    panel_selfadv(ax_a, lin, classifier, sty)

    accs = [v[k] * 100 for c in (lin, sng)
            for v in c["cross_attribution"]["per_judge"].values()
            for k in ("overall_accuracy", "non_self_accuracy")]
    ylim = (chance - 3, max(accs) + 4)
    panel_nonself(ax_b, lin, "(b) lineup", sty, chance, ylim)
    panel_nonself(ax_c, sng, "(c) single-text", sty, chance, ylim)
    ax_b.set_ylabel("accuracy (%)", fontsize=FS_LABEL)

    handles = [Line2D([0], [0], color=c, marker=mk, ms=5, mec=INK, mew=0.5, lw=1.4,
                      label=m) for m, (c, mk) in sty.items()]
    handles.append(Line2D([0], [0], ls="none", marker="o", ms=5, mfc="white",
                          mec=INK, mew=0.8, label="peer baseline"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=FS_NOTE, handlelength=1.6, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.0))
    return fig, order, sty


def main() -> int:
    _rc()
    # The main figure sits in body text, so it uses the serif face; the
    # supplementary set keeps the shared sans-serif style.
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})
    data = load(RESULTS_JSON)
    ea = load(ENGINE_A_JSON)
    fig, order, _ = build(data, ea)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{STEM}.pdf"
    png = FIG_DIR / f"{STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=400)
    plt.close(fig)

    print(f"wrote {pdf.relative_to(ROOT)}")
    print(f"wrote {png.relative_to(ROOT)}")
    sa = {r["model"]: r for r in data["lineup"]["self_advantage"]}
    sr = {r["judge"]: r for r in data["lineup"]["self_recognition"]}
    print("\nplotted values (lineup):")
    print(f"  {'model':<9} {'self':>7} {'peers':>7} {'adv':>7}")
    for m in order:
        print(f"  {m:<9} {sr[m]['rate'] * 100:>6.1f}% {sa[m]['others_rate'] * 100:>6.1f}% "
              f"{sa[m]['advantage_pp']:>+7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
