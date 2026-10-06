"""Macros and Table 2 for the open-weight panel (Study 2).

Called by make_numbers.py with results/revision/openweight/summary.json.
Writes paper_tacl/table_openweight.tex as a side effect.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parent / "src" / "revision"))
sys.path.insert(0, str(HERE.parent / "src"))
from panels import OPENWEIGHT  # noqa: E402

NAMES = list(OPENWEIGHT.names)
TAG = {"lineup": "Lineup", "single": "Single", "shuffle_lineup": "ShufLineup",
       "binary": "Binary", "shuffle_binary": "ShufBinary", "onpolicy": "OnPol",
       "onpolicy_resample": "OnPolRes"}


def _pct(x):
    return "--" if x is None else f"{100 * x:.1f}\\%"


def _pp(x):
    return f"{100 * x:+.1f}".replace("-", "$-$")


def _ci(v, pp=True):
    f = (lambda a: f"{100 * a:+.1f}") if pp else (lambda a: f"{a:.2f}")
    return f"[{f(v[0])}, {f(v[1])}]".replace("-", "$-$")


def _p(p):
    if p < 0.001:
        return "$<0.001$"
    return f"${p:.3f}$"


def macros(S: dict) -> dict[str, str]:
    m: dict[str, str] = {}
    first = next((S[c] for c in ("lineup", "binary", "single") if c in S), None)
    for cond, tag in TAG.items():
        if cond not in S:
            continue
        for j, g in S[cond]["judges"].items():
            if "self_rate" in g:  # attribution formats
                m[f"OW{j}{tag}Self"] = _pct(g["self_rate"])
                m[f"OW{j}{tag}SelfK"] = f"{g['self_hits']}/{g['n_self']}"
                m[f"OW{j}{tag}Peer"] = _pct(g["peer_baseline"])
                m[f"OW{j}{tag}Adv"] = _pp(g["self_advantage"])
                m[f"OW{j}{tag}AdvCI"] = _ci(g["adv_ci95"])
                m[f"OW{j}{tag}P"] = _p(g["perm_p"])
                if "perm_p_holm" in g:
                    m[f"OW{j}{tag}Holm"] = _p(g["perm_p_holm"])
                m[f"OW{j}{tag}FA"] = _pct(g["false_alarm"])
                m[f"OW{j}{tag}FAK"] = f"{g['false_alarm_n']}/{g['nonself_n']}"
                if "hit_minus_fa" in g:
                    m[f"OW{j}{tag}Disc"] = _pp(g["hit_minus_fa"])
                    m[f"OW{j}{tag}DiscHolm"] = _p(g["disc_p_holm"])
                    m[f"OW{j}{tag}DiscLOR"] = f"{g['log_or']:.2f}".replace("-", "$-$")
                m[f"OW{j}{tag}NamesSelf"] = _pct(g["names_self_share"])
                m[f"OW{j}{tag}NonSelf"] = _pct(g["nonself_acc"])
            elif "auc" in g:  # yes/no formats
                m[f"OW{j}{tag}AUC"] = f"{g['auc']:.2f}"
                m[f"OW{j}{tag}AUCCI"] = _ci(g["auc_ci95"], pp=False)
                if "auc_p_holm" in g:
                    m[f"OW{j}{tag}AUCHolm"] = _p(g["auc_p_holm"])
                m[f"OW{j}{tag}Hit"] = _pct(g["hit_rate"])
                m[f"OW{j}{tag}FA"] = _pct(g["false_alarm"])
        if "accuracy" in S[cond]:
            m[f"OW{tag}Acc"] = _pct(S[cond]["accuracy"])
        if "blame_share" in S[cond]:
            b = S[cond]["blame_share"]
            for j, v in b.items():
                m[f"OW{tag}Blame{j}"] = _pct(v)
            top2 = sorted(b.values(), reverse=True)[:2]
            m[f"OW{tag}TopTwo"] = _pct(sum(top2))
        # Range of per-genre values, for "between x and y in every genre".
        bg = S[cond].get("by_genre", {})
        for gname, G in bg.items():
            for j, x in G.get("judges", {}).items():
                if "auc" in x:
                    m[f"OW{j}{tag}{gname.capitalize()}AUC"] = f"{x['auc']:.2f}"
        cells = [(G["judges"][j]["auc"], j, gname, G["judges"][j]["auc_ci95"])
                 for gname, G in bg.items() for j in G.get("judges", {})
                 if "auc" in G["judges"][j]]
        if cells:
            a, j, gname, ci = max(cells)
            m[f"OW{tag}AUCGenreMax"] = f"{a:.2f}"
            m[f"OW{tag}AUCGenreMaxCI"] = _ci(ci, pp=False)
            m[f"OW{tag}AUCGenreMaxWho"] = f"{j}, {gname}"
        for j in NAMES:
            vals = [G["judges"][j]["auc"] for G in bg.values()
                    if j in G.get("judges", {}) and "auc" in G["judges"][j]]
            if vals:
                m[f"OW{j}{tag}AUCGenreLo"] = f"{min(vals):.2f}"
                m[f"OW{j}{tag}AUCGenreHi"] = f"{max(vals):.2f}"
    if "likelihood" in S:
        for j, g in S["likelihood"]["judges"]["mean_lp_cond"].items():
            m[f"OW{j}LikAcc"] = _pct(g["rel_argmax_acc"])
            m[f"OW{j}LikRawAcc"] = _pct(g["raw_argmax_acc"])
            m[f"OW{j}LikAUC"] = f"{g['auc_rel']:.2f}"
    if "likelihood" in S:
        LK = S["likelihood"]["judges"]
        for key, tag in (("mean_lp_cond", "Cond"), ("mean_lp_uncond", "Unc")):
            accs = [g["rel_argmax_acc"] for g in LK[key].values()]
            aucs = [g["auc_rel"] for g in LK[key].values()]
            m[f"OWLik{tag}AccMin"] = _pct(min(accs))
            m[f"OWLik{tag}AccMax"] = _pct(max(accs))
            m[f"OWLik{tag}AUCMin"] = f"{min(aucs):.2f}"
            rawaccs = [g["raw_argmax_acc"] for g in LK[key].values()]
            m[f"OWLik{tag}RawMin"] = _pct(min(rawaccs))
            m[f"OWLik{tag}RawMax"] = _pct(max(rawaccs))
    fam = [(g["perm_p"], j, c) for c, F in (S.get("familiarity") or {}).items()
           for j, g in (F or {}).items() if "perm_p" in g]
    if fam:
        p, j, c = min(fam)
        m["OWFamTests"] = str(len(fam))
        m["OWFamMinP"] = _p(p)
        m["OWFamMinHolm"] = _p(min(1.0, p * len(fam)))
        m["OWFamMinWho"] = f"{j}, {c.replace('binary', 'yes/no').replace('single', 'single text')}"
        m["OWFamUntestable"] = str(sum(1 for F in S["familiarity"].values()
                                       for g in (F or {}).values() if "perm_p" not in g))
    for cond, F in (S.get("familiarity") or {}).items():
        for j, g in (F or {}).items():
            if "z_diff" in g:
                m[f"OW{j}Fam{TAG.get(cond, cond)}"] = f"{g['z_diff']:+.2f}".replace("-", "$-$")
                m[f"OW{j}Fam{TAG.get(cond, cond)}P"] = _p(g["perm_p"])
    if "quality" in S:
        for j, g in S["quality"].items():
            m[f"OW{j}QualSelfPref"] = f"{g['self_preference']:+.2f}".replace("-", "$-$")
            if "self_preference_ci95" in g:
                lo, hi = g["self_preference_ci95"]
                m[f"OW{j}QualSelfPrefCI"] = f"[{lo:+.2f}, {hi:+.2f}]".replace("-", "$-$")
        sp = [g["self_preference"] for g in S["quality"].values()]
        m["OWQualSelfPrefMin"] = f"{min(sp):+.2f}".replace("-", "$-$")
        m["OWQualSelfPrefMax"] = f"{max(sp):+.2f}".replace("-", "$-$")
    if first is not None:
        n = first.get("n_judgments") or first.get("n")
        m["OWNJudgments"] = f"{n:,}".replace(",", "{,}")
    for c in ("binary", "single"):
        if c in S:
            k = len(NAMES)
            m["OWNResponses"] = str(S[c]["n"] // k if "n" in S[c] else S[c]["n_judgments"] // k)
            break
    m["OWNPrompts"] = "120"
    table(S)
    genre_table(S)
    return m


def table(S: dict) -> None:
    def get(cond, j):
        g = S.get(cond, {}).get("judges", {}).get(j, {})
        return g if ("self_rate" in g or "auc" in g) else {}

    def disc(g):
        if "hit_minus_fa" not in g:
            return "--"
        star = "$^{*}$" if g.get("disc_p_holm", 1) < 0.05 else ""
        return _pp(g["hit_minus_fa"]) + star

    def adv(g):
        star = "$^{*}$" if g.get("perm_p_holm", 1) < 0.05 else ""
        return f"{_pp(g['self_advantage'])}{star} {_ci(g['adv_ci95'])}"

    rows = [r"\begin{table*}[t]", r"\centering\small", r"\setlength\tabcolsep{3.5pt}",
            r"\begin{tabular}{l rrrrr rrr rr}", r"\toprule",
            r" & \multicolumn{5}{c}{Lineup} & \multicolumn{3}{c}{Yes/no, fresh context}"
            r" & \multicolumn{2}{c}{Own likelihood} \\",
            r"\cmidrule(lr){2-6}\cmidrule(lr){7-9}\cmidrule(lr){10-11}",
            r"Judge & Self & Peer & FA & Self-adv. [95\% CI] & Hit$-$FA & Hit & FA & AUC"
            r" & Pick-self & AUC \\", r"\midrule"]
    for j in NAMES:
        L, B = get("lineup", j), get("binary", j)
        K = S.get("likelihood", {}).get("judges", {}).get("mean_lp_cond", {}).get(j, {})
        bstar = "$^{*}$" if B.get("auc_p_holm", 1) < 0.05 else ""
        cells = [j,
                 _pct(L["self_rate"]) if L else "--",
                 _pct(L["peer_baseline"]) if L else "--",
                 _pct(L["false_alarm"]) if L else "--",
                 adv(L) if L else "--",
                 disc(L) if L else "--",
                 _pct(B["hit_rate"]) if B else "--",
                 _pct(B["false_alarm"]) if B else "--",
                 f"{B['auc']:.2f}{bstar}" if B else "--",
                 _pct(K["rel_argmax_acc"]) if K else "--",
                 f"{K['auc_rel']:.2f}" if K else "--"]
        rows.append(" & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Study~2, open-weight judges, all four genres pooled (120 prompts). "
             r"Lineup: self-naming, peer-baseline and false-alarm (FA) rates, the "
             r"self-advantage with a prompt-bootstrap interval, and the hit minus "
             r"false-alarm rate, which is positive only if a judge names itself more on its "
             r"own text than on others'. Yes/no: ``Did you write this text?'' in a fresh "
             r"context; hit and false-alarm rates at $P(\text{yes})>0.5$ and the AUC of "
             r"$P(\text{yes})$ for own versus others' text (0.5 is chance). Own likelihood: "
             r"the share of prompts on which the text the judge finds most likely, relative "
             r"to the other judges, is its own (chance 25\%), and the AUC of that relative "
             r"likelihood. $^{*}$Holm-corrected within-prompt permutation $p<0.05$ across "
             r"the four judges.}",
             r"\label{tab:openweight}", r"\end{table*}"]
    (HERE / "table_openweight.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def genre_table(S: dict) -> None:
    """Appendix table: every judge in every genre, for the lineup (self-
    advantage, hit minus false-alarm rate) and the three yes/no formats (AUC)."""
    genres = ["prose", "code", "short", "math"]
    label = {"prose": "Prose", "code": "Code", "short": "Short", "math": "Math"}

    def cell(cond, g, j, key, pp):
        x = S.get(cond, {}).get("by_genre", {}).get(g, {}).get("judges", {}).get(j, {})
        if key not in x:
            return "--"
        return _pp(x[key]) if pp else f"{x[key]:.2f}"

    rows = [r"\begin{table}[t]", r"\centering\small", r"\setlength\tabcolsep{3pt}",
            r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{ll rr rrr}", r"\toprule",
            r" & & \multicolumn{2}{c}{Lineup} & \multicolumn{3}{c}{Yes/no AUC} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
            r"Genre & Judge & Self-adv. & Hit$-$FA & Fresh & On-pol. & Resamp. \\",
            r"\midrule"]
    for gi, g in enumerate(genres):
        for ji, j in enumerate(NAMES):
            rows.append(" & ".join([label[g] if ji == 0 else "", j,
                                    cell("lineup", g, j, "self_advantage", True),
                                    cell("lineup", g, j, "hit_minus_fa", True),
                                    cell("binary", g, j, "auc", False),
                                    cell("onpolicy", g, j, "auc", False),
                                    cell("onpolicy_resample", g, j, "auc", False)]) + r" \\")
        if gi < len(genres) - 1:
            rows.append(r"\midrule")
    rows += [r"\bottomrule", r"\end{tabular}}",
             r"\caption{Study~2 by genre (per judge: 60 lineups and 240 yes/no "
             r"judgments for prose, 20 and 80 for each other genre). Lineup: "
             r"self-advantage and hit minus "
             r"false-alarm rate, in points. Yes/no: AUC of $P(\text{yes})$ for own versus "
             r"others' text in a fresh context, with the judge's own answer verbatim in "
             r"the conversation (on-policy), and with a second sample of its own as its "
             r"turn (resampled); 0.5 is chance.}",
             r"\label{tab:genre}", r"\end{table}"]
    (HERE / "table_openweight_genre.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
