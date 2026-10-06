"""Signal-detection view of self-recognition: hits, false alarms, and the leak.

A self-attribution rate on its own is a hit rate. A judge that names itself
freely posts a high one without discriminating at all, so the rate is only
interpretable next to the false-alarm rate -- how often the judge names itself
on text it did not write. The full description is the 2x2:

                        named self   named someone else
    own text (38)            h            38 - h
    others' text (152)       f           152 - f

We summarise it with a Haldane-corrected log odds ratio, called the leak here
to keep it distinct from self-advantage, which controls for a different thing
(how identifiable the text is, rather than how freely the judge names itself).
Neither statistic subsumes the other.

The Haldane correction is what makes an empty cell survive arithmetic, and that
is exactly the trap: at h = f = 0 it returns a confident-looking +1.38 for what
is really no evidence at all. Hence `degenerate` -- if either row of the 2x2 is
empty, the table says nothing about discrimination and the leak is undefined.
Three of the five single-text cells land there, which is the whole argument for
why a single-text protocol cannot measure this capability.
"""
from __future__ import annotations

import math

import pandas as pd
from scipy import stats

HALDANE = 0.5


def leak_table(df: pd.DataFrame, judge: str) -> dict:
    """Hits, false alarms, and the leak for one judge in one condition."""
    seen = df[df["judge"] == judge]
    own = seen[seen["true_author"] == judge]
    other = seen[seen["true_author"] != judge]

    h = int((own["guessed_model"] == judge).sum())
    f = int((other["guessed_model"] == judge).sum())
    n_own, n_other = len(own), len(other)

    out = {
        "judge": judge,
        "hits": h, "n_own": n_own,
        "hit_rate": h / n_own if n_own else float("nan"),
        "false_alarms": f, "n_other": n_other,
        "false_alarm_rate": f / n_other if n_other else float("nan"),
    }

    # An empty row carries no information about discrimination. Report the
    # cells and stop -- do not let the correction manufacture an odds ratio.
    if not n_own or not n_other or (h == 0 and f == 0) or (
            h == n_own and f == n_other):
        out.update(degenerate=True, leak=None, se=None, odds_ratio=None,
                   ci_lo=None, ci_hi=None, z=None, p=None)
        return out

    cells = (h, n_own - h, f, n_other - f)
    leak = math.log(((h + HALDANE) * (n_other - f + HALDANE))
                    / ((n_own - h + HALDANE) * (f + HALDANE)))
    se = math.sqrt(sum(1.0 / (c + HALDANE) for c in cells))
    z = leak / se
    out.update(
        degenerate=False,
        leak=leak, se=se, odds_ratio=math.exp(leak),
        ci_lo=math.exp(leak - 1.96 * se), ci_hi=math.exp(leak + 1.96 * se),
        z=z, p=2 * (1 - stats.norm.cdf(abs(z))),
    )
    return out


def signal_detection(df: pd.DataFrame, models: list[str]) -> list[dict]:
    return [leak_table(df, j) for j in models]


def compare_leaks(a: dict, b: dict) -> dict:
    """Test whether one judge's leak moved between two conditions.

    The point of this is the dissociation: GPT's hit rate swings 37 points
    across formats while its leak barely moves, so the format changed how
    willing it was to name itself and not what it could tell apart.
    """
    if a.get("degenerate") or b.get("degenerate"):
        return {"judge": a["judge"], "comparable": False}
    d = a["leak"] - b["leak"]
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    z = d / se
    return {
        "judge": a["judge"], "comparable": True,
        "leak_from": a["leak"], "leak_to": b["leak"], "delta": d,
        "hit_rate_from": a["hit_rate"], "hit_rate_to": b["hit_rate"],
        "hit_rate_delta_pp": (b["hit_rate"] - a["hit_rate"]) * 100,
        "fa_rate_from": a["false_alarm_rate"], "fa_rate_to": b["false_alarm_rate"],
        "fa_rate_delta_pp": (b["false_alarm_rate"] - a["false_alarm_rate"]) * 100,
        "z": z, "p": 2 * (1 - stats.norm.cdf(abs(z))),
    }


def degenerate_illustration(n_own: int = 38, n_other: int = 152) -> dict:
    """What the correction returns at h = f = 0, which is why the rule exists."""
    cells = (0, n_own, 0, n_other)
    leak = math.log(((0 + HALDANE) * (n_other + HALDANE))
                    / ((n_own + HALDANE) * (0 + HALDANE)))
    se = math.sqrt(sum(1.0 / (c + HALDANE) for c in cells))
    z = leak / se
    return {"leak": leak, "odds_ratio": math.exp(leak), "z": z,
            "p": 2 * (1 - stats.norm.cdf(abs(z)))}


def format_table(rows: list[dict]) -> str:
    out = [f"  {'judge':<10}{'hit':>8}{'FA':>8}{'leak':>8}{'OR':>9}"
           f"{'95% CI':>20}{'p':>11}"]
    for r in rows:
        if r["degenerate"]:
            out.append(f"  {r['judge']:<10}{r['hit_rate']:>7.1%}"
                       f"{r['false_alarm_rate']:>8.1%}{'--':>8}{'--':>9}"
                       f"{'--':>20}{'degenerate':>11}")
            continue
        ci = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"
        out.append(f"  {r['judge']:<10}{r['hit_rate']:>7.1%}"
                   f"{r['false_alarm_rate']:>8.1%}{r['leak']:>+8.2f}"
                   f"{r['odds_ratio']:>9.2f}{ci:>20}{r['p']:>11.3g}")
    return "\n".join(out)
