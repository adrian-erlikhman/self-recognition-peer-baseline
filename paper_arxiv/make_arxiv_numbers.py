"""Numbers and tables that only the arXiv version prints.

    python paper_arxiv/make_arxiv_numbers.py

Reads the results the TACL submission computed but had no room for, plus the
two frontier conditions run for this version (yes/no and word-shuffled
lineup on the chat-app corpus, 5 Oct 2026), and writes numbers_arxiv.tex and
the generated table_g_*.tex files. Run src/revision/analyze_panel.py
--panel frontier first so results/revision/frontier/summary.json is current.
Hand-written tables (table_x_*.tex) take their values from the macros.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
FRONTIER = ["GPT", "Claude", "Gemini", "Grok", "DeepSeek"]
OPEN = ["Qwen", "Llama", "Mistral", "Phi"]


def neg(s: str) -> str:
    return s.replace("-", "$-$")


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}\\%"


def pp(x: float) -> str:
    return neg(f"{x * 100:+.1f}")


def ci_pp(ci) -> str:
    return neg(f"[{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]")


def ci_auc(ci) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def pval(p: float) -> str:
    if p < 0.001:
        return "$<0.001$"
    return f"${p:.3f}$"


def mac(name: str, val: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{val}}}"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------------------
def tally(m: list[str]) -> None:
    """The rate test against ours over every judge-setting, now including the
    frontier yes/no setting (src/revision/standard_vs_proper.py --arxiv).
    Redefines the TACL submission's SVP macros."""
    T = load(RES / "revision" / "standard_vs_proper_arxiv.json")
    if not T:
        return
    def ren(name, val):
        m.append(f"\\renewcommand{{\\{name}}}{{{val}}}")
    ren("SVPCells", str(T["cells"]))
    ren("SVPStd", str(T["standard_positive"]))
    ren("SVPProper", str(T["proper_positive"]))
    ren("SVPFalse", str(T["standard_false_positive"]))
    ren("SVPMissed", str(T["standard_missed"]))
    ren("SVPFalsePct", f"{100 * T['standard_false_positive'] / T['standard_positive']:.0f}\\%")


def frontier_new(m: list[str]) -> None:
    S = load(RES / "revision" / "frontier" / "summary.json")
    if not S:
        return
    if "binary" in S:
        B = S["binary"]["judges"]
        for j, g in B.items():
            m += [mac(f"FB{j}Hit", pct(g["hit_rate"])),
                  mac(f"FB{j}HitK", f"{g['hits']}/{g['n_own']}"),
                  mac(f"FB{j}FA", pct(g["false_alarm"])),
                  mac(f"FB{j}FAK", f"{g['fa']}/{g['n_other']}"),
                  mac(f"FB{j}Disc", pp(g["hit_rate"] - g["false_alarm"])),
                  mac(f"FB{j}AUC", f"{g['auc']:.2f}"),
                  mac(f"FB{j}AUCCI", ci_auc(g["auc_ci95"])),
                  mac(f"FB{j}Holm", pval(g.get("auc_p_holm", g["auc_perm_p"]))),
                  mac(f"FB{j}YesAll", pct((g["hits"] + g["fa"]) / (g["n_own"] + g["n_other"])))]
        rows = []
        for j in FRONTIER:
            g = B.get(j)
            if not g:
                continue
            star = "$^{*}$" if g.get("auc_p_holm", 1) < 0.05 else ""
            rows.append(f"{j} & {g['hits']}/{g['n_own']} & {g['fa']}/{g['n_other']} & "
                        f"{pp(g['hit_rate'] - g['false_alarm'])}{star} & "
                        f"{g['auc']:.2f} {ci_auc(g['auc_ci95'])} & "
                        f"{pval(g.get('auc_p_holm', g['auc_perm_p']))} \\\\")
        (HERE / "table_g_frontier_yesno.tex").write_text("\n".join([
            r"\begin{table}[t]", r"\centering\small", r"\setlength\tabcolsep{3pt}",
            r"\begin{adjustbox}{max width=\linewidth}",
            r"\begin{tabular}{l rr r l r}", r"\toprule",
            r"Judge & Hits & False alarms & Hit$-$FA & AUC [95\% CI] & $p$ (Holm) \\",
            r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
            r"\caption{Frontier judges asked ``Did you write this text?'' about each of "
            r"the 190 chat-app essays in a fresh conversation (38 own and 152 others' "
            r"per judge). Hits: yes to its own essays; false alarms: yes to others'. "
            r"No provider returned probabilities, so the AUC is computed from the "
            r"yes/no answers. $p$ is corrected for the five judges; "
            r"$^{*}p<0.05$.}",
            r"\label{tab:fyesno}", r"\end{table}"]) + "\n", encoding="utf-8")
    if "shuffle_lineup" in S:
        A = S["shuffle_lineup"]
        m.append(mac("FSAcc", pct(A["accuracy"])))
        m.append(mac("FSNJudgments", f"{A['n_judgments']}"))
        recs = [json.loads(l) for l in (RES / "revision" / "frontier" / "shuffle_lineup.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        m.append(mac("FSClaudeRefused", str(sum(1 for r in recs if r.get("refused") and r["judge"] == "Claude"))))
        m.append(mac("FSGeminiMissing", str(sum(1 for r in recs if not r.get("ok") and r["judge"] == "Gemini"))))
        okr = [r for r in recs if r.get("ok")]
        m.append(mac("FSNLineups", str(len(okr))))
        m.append(mac("FSAllDistinct", pct(sum(len({(v or {}).get("model") for v in r["parsed"].values()}) == 5
                                              for r in okr) / len(okr), 0)))
        cl = [(r["parsed"].get(s) or {}).get("model") == "Claude"
              for r in okr for s, a in r["slot_to_author"].items() if a == "Claude"]
        m += [mac("FSClaudePeer", pct(sum(cl) / len(cl), 0)),
              mac("FSClaudePeerK", f"{sum(cl)}/{len(cl)}"),
              mac("FSNPromptsAdv", str(A["judges"]["GPT"]["n_self"]))]
        bow = load(RES / "revision" / "frontier" / "bow_shuffle.json")
        if bow:
            b = bow["shuffled"]
            m += [mac("BowShufAcc", pct(b["accuracy"])),
                  mac("BowShufK", f"{b['correct']}/{b['n']}"),
                  mac("BowShufCI", f"[{b['ci95'][0] * 100:.1f}, {b['ci95'][1] * 100:.1f}]")]
        for j, g in A["judges"].items():
            if "self_rate" not in g:
                continue
            m += [mac(f"FS{j}Self", pct(g["self_rate"])),
                  mac(f"FS{j}SelfK", f"{g['self_hits']}/{g['n_self']}"),
                  mac(f"FS{j}Peer", pct(g["peer_baseline"])),
                  mac(f"FS{j}FA", pct(g["false_alarm"])),
                  mac(f"FS{j}Adv", pp(g["self_advantage"])),
                  mac(f"FS{j}AdvCI", ci_pp(g["adv_ci95"])),
                  mac(f"FS{j}AdvHolm", pval(g["perm_p_holm"])),
                  mac(f"FS{j}Disc", pp(g["hit_minus_fa"])),
                  mac(f"FS{j}DiscHolm", pval(g["disc_p_holm"])),
                  mac(f"FS{j}Overall", pct(g["accuracy"])),
                  mac(f"FS{j}Blame", pct(A["blame_share"][j]))]
        rows = []
        for j in FRONTIER:
            g = A["judges"].get(j)
            if not g or "self_rate" not in g:
                rows.append(f"{j} & \\multicolumn{{5}}{{c}}{{\\emph{{refused every lineup}}}} & "
                            f"\\{j}AdvLineup \\\\")
                continue
            sa = "$^{*}$" if g["perm_p_holm"] < 0.05 else ""
            sd = "$^{*}$" if g["disc_p_holm"] < 0.05 else ""
            rows.append(f"{j} & {pct(g['self_rate'])} & {pct(g['peer_baseline'])} & "
                        f"{pct(g['false_alarm'])} & {pp(g['self_advantage'])}{sa} "
                        f"{ci_pp(g['adv_ci95'])} & {pp(g['hit_minus_fa'])}{sd} & "
                        f"\\{j}AdvLineup \\\\")
        # The summary mixes samples (Self and Peer on the 24 prompts every
        # answering judge completed, false alarms on all 38). The table in the
        # paper is the corrected one on 24 prompts throughout, written by hand
        # from shuffle_lineup.jsonl; keep it rather than regenerate this one.
        out = HERE / "table_g_frontier_shuffle.tex"
        if out.exists() and "SAMPLE RULE" in out.read_text(encoding="utf-8"):
            return
        out.write_text("\n".join([
            r"\begin{table}[t]", r"\centering\small", r"\setlength\tabcolsep{2.5pt}",
            r"\begin{adjustbox}{max width=\linewidth}",
            r"\begin{tabular}{l rrr l r r}", r"\toprule",
            r"Judge & Self & Peer & FA & Self-adv. [95\% CI] & Hit$-$FA & Original \\",
            r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
            r"\caption{The chat-app lineup repeated with every essay's words shuffled. "
            r"Columns as in Table~\ref{tab:frontier}; Original is the self-advantage on "
            r"the unshuffled essays. Claude's provider refused all of Claude's shuffled "
            r"lineups, so Claude has no judgments here, but its essays were still "
            r"judged by the other four. No value is significant after correction.}",
            r"\label{tab:fshuffle}", r"\end{table}"]) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def openweight_extra(m: list[str]) -> None:
    S = load(RES / "revision" / "openweight" / "summary.json")
    if not S:
        return
    LK = S["likelihood"]["judges"]
    F = S.get("familiarity", {})
    Q = S.get("quality", {})
    rows = []
    for j in OPEN:
        c, u = LK["mean_lp_cond"][j], LK["mean_lp_uncond"][j]
        fam = []
        for cond in ("lineup", "single", "binary"):
            g = (F.get(cond) or {}).get(j, {})
            fam.append(neg(f"{g['z_diff']:+.2f}") + f" ({g['perm_p']:.2f})" if "z_diff" in g else "--")
        q = Q.get(j, {})
        qs = (neg(f"{q['self_preference']:+.2f}") if q else "--")
        rows.append(f"{j} & {pct(c['rel_argmax_acc'])} & {pct(c['raw_argmax_acc'])} & "
                    f"{pct(u['rel_argmax_acc'])} & {pct(u['raw_argmax_acc'])} & "
                    + " & ".join(fam) + f" & {qs} \\\\")
        m += [mac(f"OW{j}LikUncAcc", pct(u["rel_argmax_acc"])),
              mac(f"OW{j}LikUncRawAcc", pct(u["raw_argmax_acc"]))]
        if q:
            m += [mac(f"OW{j}QualSelfOwn", f"{q['self_rating_of_own']:.2f}"),
                  mac(f"OW{j}QualPeerOwn", f"{q['peer_rating_of_own']:.2f}")]
    (HERE / "table_g_ow_likelihood.tex").write_text("\n".join([
        r"\begin{table}[t]", r"\centering\small", r"\setlength\tabcolsep{4pt}",
        r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{l rr rr ccc r}", r"\toprule",
        r" & \multicolumn{2}{c}{With prompt} & \multicolumn{2}{c}{Without prompt}"
        r" & \multicolumn{3}{c}{Claims follow likelihood? $z$ ($p$)} & Self-pref. \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
        r"Judge & Relative & Raw & Relative & Raw & Lineup & Single & Yes/no & (1--9) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
        r"\caption{What the open-weight judges' probabilities contain. Left: the "
        r"share of the 120 prompts on which the text a judge finds most likely is "
        r"its own (chance 25\%), measured relative to the other judges or on its "
        r"own, with and without the prompt. Middle: whether the other models' texts "
        r"a judge claims are more likely under it than those it does not claim "
        r"(standardised difference, with uncorrected $p$; -- where the judge claims "
        r"all or almost none). Right: self-preference in quality ratings, in points "
        r"on the 1--9 scale.}",
        r"\label{tab:owlik}", r"\end{table}"]) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
NICE = {"length": "Length", "format_punct": "Punctuation, formatting",
        "structure": "Structure", "syntax": "Sentence construction",
        "lexical": "Word choice", "tone": "Tone, register",
        "hedging": "Hedging, balance", "rhetoric": "Rhetorical flourish",
        "brand_prior": "A model's typical style", "self_reference": "The judge itself",
        "content": "Content, quality"}


def rationales(m: list[str]) -> None:
    R = load(RES / "rationale_codes.json")
    if not R:
        return
    rows = []
    for cat in R["categories"]:
        cells = []
        for cond in ("lineup", "single"):
            D = R["distribution"][cond]
            cells += [f"{D[j][cat]['share'] * 100:.0f}" for j in FRONTIER]
        rows.append(f"{NICE[cat]} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(["GPT", "Cla.", "Gem.", "Grok", "DS"] * 2)
    (HERE / "table_g_cues.tex").write_text("\n".join([
        r"\begin{table}[t]", r"\centering\small", r"\setlength\tabcolsep{4pt}",
        r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{l rrrrr rrrrr}", r"\toprule",
        r" & \multicolumn{5}{c}{Lineup} & \multicolumn{5}{c}{One text at a time} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        f"Cue cited & {head} \\\\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
        r"\caption{Cues the frontier judges cite in their one-sentence reasons, as a "
        r"percentage of each judge's 190 reasons per format (multi-label; "
        f"{R['distribution']['lineup']['_mean_cues_per_reason']:.1f} codes per lineup "
        f"reason and {R['distribution']['single']['_mean_cues_per_reason']:.1f} per "
        r"single-text reason). No reason in either format refers to the judge "
        r"itself.}",
        r"\label{tab:cues}", r"\end{table}"]) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def sci(p: float) -> str:
    if p < 0.001:
        e = int(math.floor(math.log10(p)))
        return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"
    return f"${p:.3f}$"


def stats_tables() -> None:
    S = load(RES / "stats_revision.json")
    if not S:
        return
    rows = []
    for cond, lab in (("lineup", "Lineup"), ("single", "Single")):
        P, MM = S[cond]["per_judge"], S[cond]["mixed_model"]["per_judge"]
        for i, j in enumerate(FRONTIER):
            sa = P[j]["self_advantage"]
            g, r = sa.get("gee", {}), MM[j]
            gee = (f"{g['or']:.2f} ({sci(g['p_bias_reduced'])})" if g.get("estimable") else "--")
            rem = (f"{r['or']:.2f} ({sci(r['p_normal_approx'])})" if r.get("estimable") and not r.get("note") else "--")
            rows.append(f"{lab if i == 0 else ''} & {j} & {pp(sa['adv'])} & "
                        f"{sci(sa['p_fisher_paper'])} & {sci(sa['randomization']['p_exact'])} & "
                        f"{sci(sa['p_bootstrap'])} & {gee} & {rem} \\\\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    (HERE / "table_g_clustered.tex").write_text("\n".join([
        r"\begin{table}[t]", r"\centering\small", r"\setlength\tabcolsep{4pt}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{ll r rrr rr}", r"\toprule",
        r"Format & Judge & Self-adv. & Fisher & Randomisation & Bootstrap & GEE OR ($p$) & Random-effects OR ($p$) \\",
        r"\midrule", *rows, r"\end{tabular}", r"\end{adjustbox}",
        r"\caption{Frontier judges, chat-app essays: the self-advantage test under five "
        r"analyses (uncorrected two-sided $p$). Fisher's exact test treats the 152 peer "
        r"judgments as independent and is shown only for comparison; the exact "
        r"within-response randomisation test is the one the paper reports. GEE: "
        r"logistic regression of correctness on a self indicator over the 190 "
        r"judgments of the judge's texts, exchangeable working correlation within "
        r"prompt, bias-reduced sandwich $p$. Random effects: all 950 judgments, author "
        r"fixed effects, a self term per judge, crossed random intercepts for prompt "
        r"and judge; it absorbs each judge's general skill. -- where a judge never "
        r"names itself and the estimate is the prior.}",
        r"\label{tab:clustered}", r"\end{table}"]) + "\n", encoding="utf-8")



# ---------------------------------------------------------------------------
TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}"}


def tex(s: str) -> str:
    """Escape prompt text for LaTeX; `code` becomes typewriter and straight
    quotes become typographic ones."""
    import re
    # arXiv compiles with pdfLaTeX: keep the source ASCII.
    for a, b in (("—", "---"), ("–", "--"), ("‘", "'"), ("’", "'"),
                 ("“", '"'), ("”", '"'), ("…", "...")):
        s = s.replace(a, b)
    if any(ord(c) > 127 for c in s):
        raise SystemExit(f"non-ASCII character left in prompt text: {s!r}")
    parts = re.split(r"(`[^`]*`)", s)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(r"\texttt{" + "".join(TEX_ESC.get(c, c) for c in p[1:-1]) + "}")
            continue
        p = "".join(TEX_ESC.get(c, c) for c in p)
        p = re.sub(r"(^|[\s(\[])'", r"\1`", p)
        p = re.sub(r'(^|[\s(\[])"', r"\1``", p).replace('"', "''")
        out.append(p)
    return "".join(out)


def appendix_prompts() -> None:
    """Every judge instruction, verbatim, for the appendix (the 120 prompts
    themselves are in data/prompts_v2.csv)."""
    import sys
    sys.path.insert(0, str(ROOT / "src" / "revision"))
    sys.path.insert(0, str(ROOT / "src"))
    import conditions as C  # noqa: E402

    def lst(title: str, body: str) -> list[str]:
        return [r"\needspace{10\baselineskip}", rf"\paragraph{{{title}}}\mbox{{}}",
                r"\begin{lstlisting}", body.strip(), r"\end{lstlisting}"]

    names = FRONTIER
    tmpl = []
    tmpl += lst("Lineup",
                C.lineup_prompt(names, {s: f"<response {s}>" for s in C.slots(5)}))
    tmpl += lst("One text at a time", C.single_prompt(names, "<text>"))
    tmpl += lst("Did you write this text? (fresh conversation)", C.binary_prompt("<text>"))
    conv = C.onpolicy_conv("<task prompt>", "<the judge's own answer>", "<candidate>")
    tmpl += lst("Did you write this text? (after answering the prompt; three turns)",
                "\n\n".join(f"[{t['role']}]\n{t['content']}" for t in conv))
    tmpl += lst("Quality rating (open-weight judges)", C.quality_prompt("<task prompt>", "<text>"))

    out = ["% GENERATED by paper_arxiv/make_arxiv_numbers.py -- do not edit by hand.", *tmpl]
    (HERE / "sections" / "gen_instructions.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def costs(m: list[str]) -> None:
    c = load(RES / "costs_summary.json")
    if c:
        m += [mac("CostStudyOne", f"\\${c['total_usd']:.2f}"),
              mac("CallsStudyOne", f"{c['calls']:,}".replace(",", "{,}"))]


def main() -> None:
    m = ["% GENERATED by paper_arxiv/make_arxiv_numbers.py -- do not edit by hand."]
    tally(m)
    frontier_new(m)
    openweight_extra(m)
    rationales(m)
    costs(m)
    stats_tables()
    appendix_prompts()
    (HERE / "numbers_arxiv.tex").write_text("\n".join(m) + "\n", encoding="utf-8")
    print(f"wrote numbers_arxiv.tex ({len(m) - 1} macros)")


if __name__ == "__main__":
    main()
