#!/usr/bin/env python
"""Generates tab_families.tex, tab_families_full.tex and counts.tex from ALL probe sources.

The census was assembled from four rounds that store their results in two different shapes,
and for a while the tables in the paper were maintained by hand while make_breadth.py still
produced a stale version from two of the four. This script is the single source of truth:

  s30_crossmodel   nested {models: {key: {cells, agg: {conv: {ratio_median}}}}}
  s36_models       same shape, sharded over part_*.json
  s49_2025models   flat {"<name>|<ds>|<mech>|<conv>": {ratio, ...}}   (TiRex, FlowState, Timer-XL)
  s46_moirai2      flat {"<ds>|<mech>|<model>|<conv>": {ratio, ...}}  (Moirai 2.0)

rho is kept from each source's stored aggregate, so printed values never drift; the full
table additionally reports the numerator (perm) and denominator (redraw) of rho, recomputed
as medians over the stored per-cell values, which every source carries.

Checkpoints with no reproducible rho are dropped from the census; Timer-XL's "poisons" entry
is a measurement.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))

_s30 = json.load(open(os.path.join(EXP, "s30_crossmodel", "s30_results.json")))["models"]
_s36 = {}
for _f in sorted(glob.glob(os.path.join(EXP, "s36_models", "part_*.json"))):
    _s36.update(json.load(open(_f)).get("models", {}))
_f25 = json.load(open(os.path.join(EXP, "s49_2025models", "s49_results.json")))["probe"]
_f46 = json.load(open(os.path.join(EXP, "s46_moirai2", "s46_results.json")))["probe"]
_f62 = json.load(open(os.path.join(EXP, "s62_tempopfn", "s62_results.json")))["probe"]


def _nested(src, key, conv):
    a = src.get(key, {}).get("agg", {}).get(conv)
    return a["ratio_median"] if a else None


def _nested_cells(src, key, conv):
    return [v for k, v in src.get(key, {}).get("cells", {}).items()
            if k.rsplit("|", 1)[-1] == conv]


def _flat(d, name, conv, pos):
    """pos=0: name is the first field (s49); pos=2: the third (s46)."""
    out = []
    for k, v in d.items():
        p = k.split("|")
        if len(p) == 4 and p[pos] == name and p[3] == conv:
            r = v.get("ratio")
            if r is not None and np.isfinite(r):
                out.append(r)
    return float(np.median(out)) if out else None


def _flat_cells(d, name, conv, pos):
    return [v for k, v in d.items()
            if len(k.split("|")) == 4 and k.split("|")[pos] == name
            and k.split("|")[3] == conv]


def _med(cells, field):
    vals = [c.get(field) for c in cells
            if c.get(field) is not None and np.isfinite(c.get(field))]
    return float(np.median(vals)) if vals else None


# display name, family, params, declared-path label, behaviour,
# (plain, declared) rho getter, (plain, declared) cell-list getter
ROWS = [
    (r"\bolt{} tiny", "Chronos-Bolt", "8.7M", r"mask/\texttt{NaN}", "discards",
     lambda: (_nested(_s36, "bolt_tiny", "plain"), _nested(_s36, "bolt_tiny", "mask")),
     lambda: (_nested_cells(_s36, "bolt_tiny", "plain"), _nested_cells(_s36, "bolt_tiny", "mask"))),
    (r"\bolt{} mini", "Chronos-Bolt", "21M", r"mask/\texttt{NaN}", "discards",
     lambda: (_nested(_s36, "bolt_mini", "plain"), _nested(_s36, "bolt_mini", "mask")),
     lambda: (_nested_cells(_s36, "bolt_mini", "plain"), _nested_cells(_s36, "bolt_mini", "mask"))),
    (r"\bolt{} small", "Chronos-Bolt", "48M", r"mask/\texttt{NaN}", "discards",
     lambda: (_nested(_s36, "bolt_small", "plain"), _nested(_s36, "bolt_small", "mask")),
     lambda: (_nested_cells(_s36, "bolt_small", "plain"), _nested_cells(_s36, "bolt_small", "mask"))),
    (r"\bolt{} base", "Chronos-Bolt", "205M", r"mask/\texttt{NaN}", "discards",
     lambda: (_nested(_s30, "bolt", "plain"), _nested(_s30, "bolt", "mask")),
     lambda: (_nested_cells(_s30, "bolt", "plain"), _nested_cells(_s30, "bolt", "mask"))),
    (r"\ctwo{}", "Chronos-2", "120M", r"\texttt{NaN}", "discards",
     lambda: (_nested(_s30, "chronos2", "plain"), _nested(_s30, "chronos2", "nan")),
     lambda: (_nested_cells(_s30, "chronos2", "plain"), _nested_cells(_s30, "chronos2", "nan"))),
    ("Chronos-T5 small", "Chronos-T5", "46M", r"\texttt{NaN}", "discards",
     lambda: (_nested(_s30, "t5", "plain"), _nested(_s30, "t5", "nan")),
     lambda: (_nested_cells(_s30, "t5", "plain"), _nested_cells(_s30, "t5", "nan"))),
    ("Chronos-T5 base", "Chronos-T5", "200M", r"\texttt{NaN}", "discards",
     lambda: (_nested(_s36, "t5_base", "plain"), _nested(_s36, "t5_base", "nan")),
     lambda: (_nested_cells(_s36, "t5_base", "plain"), _nested_cells(_s36, "t5_base", "nan"))),
    ("TiRex 1.1", "TiRex", "35M", r"\texttt{NaN}", "discards",
     lambda: (_flat(_f25, "tirex-1.1", "plain", 0), _flat(_f25, "tirex-1.1", "nan", 0)),
     lambda: (_flat_cells(_f25, "tirex-1.1", "plain", 0), _flat_cells(_f25, "tirex-1.1", "nan", 0))),
    ("FlowState r1", "FlowState", "9.1M", r"\texttt{NaN}", "discards",
     lambda: (_flat(_f25, "flowstate-r1", "plain", 0), _flat(_f25, "flowstate-r1", "nan", 0)),
     lambda: (_flat_cells(_f25, "flowstate-r1", "plain", 0), _flat_cells(_f25, "flowstate-r1", "nan", 0))),
    ("TempoPFN", "TempoPFN", "38M", r"\texttt{NaN} token", "discards",
     lambda: (_flat(_f62, "tempopfn-38m", "plain", 0), _flat(_f62, "tempopfn-38m", "nan", 0)),
     lambda: (_flat_cells(_f62, "tempopfn-38m", "plain", 0), _flat_cells(_f62, "tempopfn-38m", "nan", 0))),
    (r"\timesfm{} 2.5", "TimesFM", "200M", r"\texttt{NaN}", "overwrites",
     lambda: (_nested(_s30, "timesfm", "plain"), _nested(_s30, "timesfm", "nan")),
     lambda: (_nested_cells(_s30, "timesfm", "plain"), _nested_cells(_s30, "timesfm", "nan"))),
    (r"\moirai{} 2.0 small", "Moirai 2.0", "11M", "obs. mask", "retains",
     lambda: (_flat(_f46, "moirai2", "plain", 2), _flat(_f46, "moirai2", "nan", 2)),
     lambda: (_flat_cells(_f46, "moirai2", "plain", 2), _flat_cells(_f46, "moirai2", "nan", 2))),
    (r"\moirai{} 1.1 small$^{\dagger}$", "Moirai 1.1", "14M", "obs. mask", "retains",
     lambda: (_nested(_s36, "moirai_s", "plain"), _nested(_s36, "moirai_s", "nan")),
     lambda: (_nested_cells(_s36, "moirai_s", "plain"), _nested_cells(_s36, "moirai_s", "nan"))),
    (r"\moirai{} 1.1 base$^{\dagger}$", "Moirai 1.1", "91M", "obs. mask", "retains",
     lambda: (_nested(_s30, "moirai", "plain"), _nested(_s30, "moirai", "nan")),
     lambda: (_nested_cells(_s30, "moirai", "plain"), _nested_cells(_s30, "moirai", "nan"))),
    (r"\moirai{} 1.1 large$^{\dagger}$", "Moirai 1.1", "311M", "obs. mask", "retains",
     lambda: (_nested(_s36, "moirai_l", "plain"), _nested(_s36, "moirai_l", "nan")),
     lambda: (_nested_cells(_s36, "moirai_l", "plain"), _nested_cells(_s36, "moirai_l", "nan"))),
    (r"\moirai{}-MoE small$^{\dagger}$", "Moirai-MoE", "117M", "obs. mask", "retains",
     lambda: (_nested(_s36, "moe_s", "plain"), _nested(_s36, "moe_s", "nan")),
     lambda: (_nested_cells(_s36, "moe_s", "plain"), _nested_cells(_s36, "moe_s", "nan"))),
    (r"\moirai{}-MoE base$^{\dagger}$", "Moirai-MoE", "935M", "obs. mask", "retains",
     lambda: (_nested(_s36, "moe_b", "plain"), _nested(_s36, "moe_b", "nan")),
     lambda: (_nested_cells(_s36, "moe_b", "plain"), _nested_cells(_s36, "moe_b", "nan"))),
    ("Timer-XL base", "Timer-XL", "84M", "none", "no declared path",
     lambda: (_flat(_f25, "timer-base-84m", "plain", 0), "poisons"),
     lambda: (_flat_cells(_f25, "timer-base-84m", "plain", 0), [])),
]
# Checkpoints with no reproducible rho are not kept as rows: the fourth convention is
# established by Timer-XL, which runs.

GROUPS = [("discards", "discards the fill content once a hole is declared"),
          ("overwrites", "overwrites the fill with its own interpolation"),
          ("retains", "retains the fill content after a declaration"),
          ("no declared path", "ships no way to declare a hole at all")]

DAGGER = (r"$^{\dagger}$Sampled forecaster: its $\rho$ is reported against a measured noise "
          r"floor of $\rho_{\text{self}} = 0.45$--$0.69$, which is what a checkpoint that "
          r"discarded the fill would report given the same stochasticity "
          r"(App.~\ref{app:rhonoise}); the deterministic families sit at $\rho_{\text{self}} "
          r"= 0.0000$.")

PATHDEF = (r"\emph{plain} feeds the filled context with nothing declared; \emph{declared} "
           r"goes through the checkpoint's own declaration path.")

# relMSE on the declared path (linear fill, 70% missing), quoted from the leaderboard table
# (Table~\ref{tab:leaderboard}, built by make_tab_leaderboard_full.py from s57/s64_bench3):
# the "med." column of that table's linear-fill panel. The probe JSONs this script reads do
# not carry accuracy numbers, so the values are hardcoded here, matching the .tex; families
# the leaderboard does not measure print "--".
RELMSE = {"Chronos-Bolt": "$1.57$", "Chronos-2": "$1.52$", "TiRex": "$1.59$",
          "FlowState": "$1.30$", "TimesFM": "$1.55$", "Moirai 2.0": "$1.60$"}


def fmt(v):
    if v is None:
        return "--"
    if isinstance(v, str):
        return r"\hi{" + v + "}"
    return r"\hi{0.00}" if abs(v) < 5e-3 else f"{v:.2f}"


def fmt3(v):
    return "--" if v is None else f"{v:.3f}"


def counts():
    n_meas = sum(1 for *_, g, _c in ROWS if g()[0] is not None)
    fams = {r[1] for r in ROWS if r[5]()[0] is not None}
    return n_meas, len(fams), len(ROWS)


def full_table():
    L = [r"\begin{table}[H]", r"\centering",
         r"\caption{\textbf{The trichotomy of Corollary~\ref{prop:reach} across \numallmodels{} "
         r"checkpoints from \numfamilies{} families.} $\rho$ of \eqref{eq:rho}, median over three "
         r"datasets $\times$ three mechanisms $\times$ 60 windows at rate $0.3$, $L=512$, $H=64$; "
         + PATHDEF + r" \emph{perm.} and \emph{redraw} are its numerator and denominator (RMS "
         r"forecast change under a fill permutation and under an independent redraw), median "
         r"over the same cells. The two zero patterns separate the conventions: a "
         r"\texttt{NaN}-declared path answers even a redraw with exactly nothing, while "
         r"\bolt{}'s mask path answers the redraw but not the permutation, the rank-two "
         r"leak of Corollary~\ref{prop:reach} made visible. In the declared columns, \hi{red} "
         r"marks every path that throws the fill's content away; the unmarked values are the "
         r"families that keep it, which is the contrast Section~\ref{sec:causal} turns into a "
         r"natural experiment. Timer-XL's declared entry is a measurement, not a $\rho$: a raw "
         r"\texttt{NaN} renders its forecast non-finite. "
         + DAGGER + r"}",
         r"\label{tab:familiesfull}", r"\vspace{2pt}", r"\footnotesize",
         r"\begin{tabular}{lrlcccccc}", r"\toprule",
         r"& & & \multicolumn{3}{c}{plain} & \multicolumn{3}{c}{declared} \\",
         r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
         r"Checkpoint & Params & Declared path & $\rho$ & perm. & redraw"
         r" & $\rho$ & perm. & redraw \\",
         r"\midrule"]
    # a NaN-only declared path is rank 0; Bolt is the one family whose mask path leaks
    # the rank-two (loc, scale) channel, so its cell carries both possibilities
    RANK = {"discards": "$0$", "overwrites": "$0$", "retains": r"$|\Mset|$",
            "no declared path": r"$|\Mset|$ only"}
    first = True
    for beh, title in GROUPS:
        if not first:
            L.append(r"\midrule")
        first = False
        L.append(r"\multicolumn{9}{l}{\emph{" + title + r"}} \\[1pt]")
        prev_fam = None
        for disp, fam, par, path, b, get, getc in ROWS:
            if b != beh:
                continue
            if prev_fam is not None and fam != prev_fam:
                L.append(r"\addlinespace[2pt]")
            prev_fam = fam
            pl, dc = get()
            cpl, cdc = getc()
            pp, rp = _med(cpl, "perm_rms_rel"), _med(cpl, "redraw_rms_rel")
            if isinstance(dc, str):
                pd, rd = None, None
            else:
                pd, rd = _med(cdc, "perm_rms_rel"), _med(cdc, "redraw_rms_rel")
            rank = "$2$ or $0$" if path.startswith("mask") else RANK[beh]
            L.append(f"{disp} & {par} & {path} & {fmt(pl)} & {fmt3(pp)} & {fmt3(rp)} "
                     f"& {fmt(dc)} & {fmt3(pd)} & {fmt3(rd)} \\\\")
    # the rank column is subsumed by the path labels and the prose; kept out of the table
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def family_table():
    order, seen = [], {}
    for disp, fam, par, path, beh, get, _getc in ROWS:
        fam = "Moirai 1.1/MoE" if fam in ("Moirai 1.1", "Moirai-MoE") else fam
        pl, dc = get()
        if fam not in seen:
            seen[fam] = {"n": 0, "pl": [], "dc": [], "path": path, "beh": beh,
                         "dag": r"$^{\dagger}$" in disp}
            order.append(fam)
        e = seen[fam]
        e["n"] += 1
        e["pl"].append(pl)
        e["dc"].append(dc)

    def rng(vals):
        v = [x for x in vals if isinstance(x, (int, float))]
        if not v:
            # a non-numeric entry is a measured failure mode, e.g. Timer-XL's "poisons"
            txt = [x for x in vals if isinstance(x, str)]
            return fmt(txt[0]) if txt else "--"
        if max(v) - min(v) < 5e-3:
            return fmt(v[0])
        lo, hi = fmt(min(v)), fmt(max(v))
        return f"{lo}--{hi}"

    L = [r"\begin{table}[tb]", r"\centering",
         r"\caption{\textbf{Four conventions behind one description.} $\rho$ of \eqref{eq:rho}, "
         r"median over three datasets $\times$ three mechanisms $\times$ 60 windows at rate "
         r"$0.3$; ranges span each family's checkpoints. " + PATHDEF + r" \hi{Red} marks every "
         r"declared path that throws the fill's content away. relMSE is the median over nine "
         r"benchmarks at $70\%$ missing under a linear fill on the declared path "
         r"(Tables~\ref{tab:leaderboard} and~\ref{tab:leaderboardfull}), shown for the families those tables measure. "
         r"$^{\dagger}$Sampled forecaster: "
         r"$\rho$ against a measured noise floor (App.~\ref{app:rhonoise}). "
         r"\emph{Poisons}: a raw \texttt{NaN} makes Timer-XL's forecast non-finite. "
         r"Table~\ref{tab:familiesfull} lists every checkpoint, with $\rho$'s two terms.}",
         r"\label{tab:families}", r"\vspace{2pt}", r"\footnotesize",
         r"\begin{tabular}{llrlccc}", r"\toprule",
         r"Family & Declared path & $n$ & Behaviour & $\rho$ plain & $\rho$ declared "
         r"& relMSE decl. \\",
         r"\midrule"]
    prev_beh = None
    for fam in order:
        e = seen[fam]
        if prev_beh is not None and e["beh"] != prev_beh:
            L.append(r"\addlinespace[2pt]")
        prev_beh = e["beh"]
        name = fam + (r"$^{\dagger}$" if e["dag"] else "")
        L.append(f"{name} & {e['path']} & {e['n']} & {e['beh']} & {rng(e['pl'])} "
                 f"& {rng(e['dc'])} & {RELMSE.get(fam, '--')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


if __name__ == "__main__":
    open(os.path.join(HERE, "tab_families.tex"), "w").write(family_table() + "\n")
    open(os.path.join(HERE, "tab_families_full.tex"), "w").write(full_table() + "\n")
    n_meas, n_fam, n_all = counts()
    words = {1: "one", 2: "two", 3: "three", 6: "six", 7: "seven", 8: "eight", 9: "nine",
             10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
             15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
             20: "twenty", 21: "twenty-one", 22: "twenty-two"}
    open(os.path.join(HERE, "counts.tex"), "w").write(
        "% generated by make_families.py -- do not edit\n"
        f"\\newcommand{{\\nummodels}}{{{words.get(n_meas, n_meas)}}}\n"
        f"\\newcommand{{\\numfamilies}}{{{words.get(n_fam, n_fam)}}}\n"
        f"\\newcommand{{\\numallmodels}}{{{words.get(n_all, n_all)}}}\n")
    print(f"wrote tab_families.tex, tab_families_full.tex, counts.tex "
          f"({n_meas} measured checkpoints, {n_fam} measured families, {n_all} rows total)")
