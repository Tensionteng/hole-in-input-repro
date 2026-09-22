#!/usr/bin/env python
"""Builds the two breadth tables and the breadth heatmap from S35 (datasets) and S36+S30
(models)."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
import glob
S35 = json.load(open(os.path.join(EXP, "s35_breadth", "s35_results.json")))
# The breadth table quotes S41, which measured the same grid and probe at 300 windows and
# recorded MAE alongside MSE from the same forecasts.
S41 = {}
for _f in sorted(glob.glob(os.path.join(EXP, "s41_mae", "part_*.json"))):
    _d = json.load(open(_f))
    for _k in ("grid", "probe"):
        S41.setdefault(_k, {}).update(_d.get(_k, {}))
COUNTS = json.load(open(os.path.join(HERE, "dataset_counts.json")))

DS = [("ETTh1", "ETTh1"), ("ETTh2", "ETTh2"), ("ETTm1", "ETTm1"), ("ETTm2", "ETTm2"),
      ("weather", "weather"), ("electricity", "electricity"), ("traffic", "traffic"),
      ("exchange", "exchange"), ("illness", "illness")]
MECH = [("mcar", "scattered"), ("block", "block"), ("mnar_high", "censoring"),
        ("mnar_extreme", "extreme")]


def fmt(v, d=2):
    return "--" if v is None else (f"{v:.{d}f}" if v < 100 else f"{v:.0f}")


def dataset_table():
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{\textbf{Proposition~\ref{prop:reach} and the mechanism ordering across "
         r"nine benchmark datasets, under both metrics.} Context $L=512$, horizon $H=64$, "
         r"$300$ windows; \bolt{} base. $\rho$ of \eqref{eq:rho} at rate $0.3$. Each damage "
         r"cell is the paired per-window median at rate $0.7$ under a linear fill, written "
         r"relMSE\,/\,relMAE from the same forecasts. The declared column is the larger of "
         r"the mask and \texttt{NaN} paths: \texttt{NaN} is $0$ exactly everywhere and mask "
         r"is $\le 10^{-5}$, so both print as $0.00$. \hi{Red} marks the worst mechanism "
         r"in each row: value censoring on eight of the nine, and extreme censoring on "
         r"\texttt{exchange}, the one dataset that reverses the ordering.}",
         r"\label{tab:breadth}", r"\vspace{2pt}", r"\small",
         r"\setlength{\tabcolsep}{3.6pt}",
         r"\begin{tabular}{lrccrrrr}", r"\toprule",
         r"& & \multicolumn{2}{c}{$\rho$ (rate $0.3$)} & "
         r"\multicolumn{4}{c}{relMSE\,/\,relMAE at rate $0.7$} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}",
         r"dataset & series & plain & declared & " +
         " & ".join(n for _, n in MECH) + r" \\", r"\midrule"]
    GROUP = {"ETTh1": "ETT (electricity transformer)", "ETTh2": None, "ETTm1": None,
             "ETTm2": None, "weather": "other sources", "electricity": None,
             "traffic": None, "exchange": None, "illness": None}
    for k, name in DS:
        g = GROUP.get(k)
        if g:
            if k != DS[0][0]:
                L.append(r"\midrule")
            L.append(r"\multicolumn{8}{l}{\emph{" + g + r"}} \\[1pt]")
        pr = S41["probe"][k]
        n = COUNTS[k]["n_series"]
        dec = max(pr["mask"]["rms"], pr["nan"]["rms"])
        mse = [S41["grid"][f"{k}|{m}|0.7"]["mse"]["median"] for m, _ in MECH]
        mae = [S41["grid"][f"{k}|{m}|0.7"]["mae"]["median"] for m, _ in MECH]
        best = max(mse)
        row = " & ".join(
            (r"\hi{" + fmt(a) + "}\\,/\\,\\hi{" + fmt(b) + "}") if a == best
            else fmt(a) + "\\,/\\," + fmt(b) for a, b in zip(mse, mae))
        L.append(f"{name} & {n:,}".replace(",", "{,}")
                 + f" & {pr['plain']['rms']:.2f} & "
                 + ("0.00" if dec < 5e-3 else f"{dec:.2f}") + f" & {row} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def model_rows():
    """The checkpoint list, shared by the full table and the per-family summary."""
    s30 = json.load(open(os.path.join(EXP, "s30_crossmodel", "s30_results.json")))["models"]
    parts = {}
    d = os.path.join(EXP, "s36_models")
    for f in sorted(os.listdir(d)):
        if f.startswith("part_") and f.endswith(".json"):
            parts.update(json.load(open(os.path.join(d, f))).get("models", {}))
    return [
        ("bolt_tiny", parts, r"\bolt{} tiny", "Chronos-Bolt", "8.7M", r"mask/\texttt{NaN}"),
        ("bolt_mini", parts, r"\bolt{} mini", "Chronos-Bolt", "21M", r"mask/\texttt{NaN}"),
        ("bolt_small", parts, r"\bolt{} small", "Chronos-Bolt", "48M", r"mask/\texttt{NaN}"),
        ("bolt", s30, r"\bolt{} base", "Chronos-Bolt", "205M", r"mask/\texttt{NaN}"),
        ("chronos2", s30, r"\ctwo{}", "Chronos-2", "120M", r"\texttt{NaN}"),
        ("t5", s30, "Chronos-T5 small", "Chronos-T5", "46M", r"\texttt{NaN}"),
        ("t5_base", parts, "Chronos-T5 base", "Chronos-T5", "200M", r"\texttt{NaN}"),
        ("timesfm", s30, r"\timesfm{} 2.5", "TimesFM", "200M", r"\texttt{NaN}"),
        ("moirai_s", parts, r"\moirai{} small", "Moirai 1.1", "14M", "obs. mask"),
        ("moirai", s30, r"\moirai{} base", "Moirai 1.1", "91M", "obs. mask"),
        ("moirai_l", parts, r"\moirai{} large", "Moirai 1.1", "311M", "obs. mask"),
        ("moe_s", parts, "Moirai-MoE small", "Moirai-MoE", "117M", "obs. mask"),
        ("moe_b", parts, "Moirai-MoE base", "Moirai-MoE", "935M", "obs. mask"),
    ]


def model_table():
    """S36 + S30 aggregates. Written by hand-checked extraction, not by trusting a report."""
    s30 = json.load(open(os.path.join(EXP, "s30_crossmodel", "s30_results.json")))["models"]
    parts = {}
    d = os.path.join(EXP, "s36_models")
    for f in sorted(os.listdir(d)):
        if f.startswith("part_") and f.endswith(".json"):
            parts.update(json.load(open(os.path.join(d, f))).get("models", {}))
    rows = [
        # key, source, display name, family, params, declared-path description
        ("bolt_tiny", parts, r"\bolt{} tiny", "Chronos-Bolt", "8.7M", r"mask/\texttt{NaN}"),
        ("bolt_mini", parts, r"\bolt{} mini", "Chronos-Bolt", "21M", r"mask/\texttt{NaN}"),
        ("bolt_small", parts, r"\bolt{} small", "Chronos-Bolt", "48M", r"mask/\texttt{NaN}"),
        ("bolt", s30, r"\bolt{} base", "Chronos-Bolt", "205M", r"mask/\texttt{NaN}"),
        ("chronos2", s30, r"\ctwo{}", "Chronos-2", "120M", r"\texttt{NaN}"),
        ("t5", s30, "Chronos-T5 small", "Chronos-T5", "46M", r"\texttt{NaN}"),
        ("t5_base", parts, "Chronos-T5 base", "Chronos-T5", "200M", r"\texttt{NaN}"),
        ("timesfm", s30, r"\timesfm{} 2.5", "TimesFM", "200M", r"\texttt{NaN}"),
        ("moirai_s", parts, r"\moirai{} small", "Moirai 1.1", "14M", "obs. mask"),
        ("moirai", s30, r"\moirai{} base", "Moirai 1.1", "91M", "obs. mask"),
        ("moirai_l", parts, r"\moirai{} large", "Moirai 1.1", "311M", "obs. mask"),
        ("moe_s", parts, "Moirai-MoE small", "Moirai-MoE", "117M", "obs. mask"),
        ("moe_b", parts, "Moirai-MoE base", "Moirai-MoE", "935M", "obs. mask"),
    ]
    # TimeMoE is listed without numbers on purpose: three decoder-only checkpoints could not
    # be made numerically reliable under the pinned transformers version (see s36_notes.md),
    # and the claim the row makes -- that the family ships no way to declare a hole -- is a
    # property of its API, read off modeling_time_moe.py rather than measured.
    EXTRA = [("TimeMoE 50M/200M", "TimeMoE", "50--200M", "none", None, None,
              r"$|\Mset|$ only"),
             ("Sundial base", "Sundial", "128M", "none", None, None, r"$|\Mset|$ only")]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{\textbf{The trichotomy of Proposition~\ref{prop:reach} across "
         r"\numfam{} families and \nummod{} checkpoints.} $\rho$ of \eqref{eq:rho}, median "
         r"over three datasets $\times$ three mechanisms $\times$ 60 windows at rate $0.3$, "
         r"with $L=512$ and $H=64$ throughout. "
         r"Rank is read off $\rho$ and the declared path's own definition. In the declared "
         r"column, \hi{red} marks every path that throws the fill's content away; the "
         r"unmarked values below are the families that keep it, which is the contrast "
         r"Section~\ref{sec:causal} turns into a natural experiment. The last two rows "
         r"carry no $\rho$: those checkpoints are not numerically reliable under our pinned "
         r"library version (App.~\ref{app:excluded}), and their entry here is read off their "
         r"API, which has no argument for declaring a hole.}",
         r"\label{tab:families}", r"\vspace{2pt}", r"\footnotesize",
         r"\begin{tabular}{lrlccc}", r"\toprule",
         r"Checkpoint & Params & Declared path & $\rho$ plain & $\rho$ declared "
         r"& rank$\,\Jm$ \\", r"\midrule"]
    BEHAVIOUR = {"Chronos-Bolt": "discards the fill content once a hole is declared",
                 "Chronos-2": "discards the fill content once a hole is declared",
                 "Chronos-T5": "discards the fill content once a hole is declared",
                 "TimesFM": "overwrites the fill with its own interpolation",
                 "Moirai 1.1": "retains the fill content after a declaration",
                 "Moirai-MoE": "retains the fill content after a declaration"}
    prev = prev_beh = None
    for key, src, disp, fam, par, path in rows:
        if key not in src or "agg" not in src[key]:
            continue
        agg = src[key]["agg"]
        pl = agg.get("plain", {}).get("ratio_median")
        dec_key = "mask" if agg.get("mask", {}).get("ratio_median") is not None else "nan"
        dc = agg.get(dec_key, {}).get("ratio_median")
        unsup = agg.get("nan", {}).get("n_unsupported", 0) > 0 and dc is None
        beh = BEHAVIOUR.get(fam)
        if beh != prev_beh:
            if prev_beh is not None:
                L.append(r"\midrule")
            L.append(r"\multicolumn{6}{l}{\emph{" + beh + r"}} \\[1pt]")
        elif fam != prev and prev is not None:
            L.append(r"\addlinespace[2pt]")
        # The family repeats down consecutive rows; write it once and span it.
        prev, prev_beh = fam, beh
        if unsup:
            dcs, rank = r"n/a", r"$|\Mset|$ only"
        elif dc is not None and dc < 5e-3:
            # rank 2 is reachable only where the family offers a mask alongside NaN; a
            # NaN-only path never lets the fill into the computation at all.
            dcs = r"\hi{0.00}"
            rank = r"$2$ or $0$" if "mask" in path else r"$0$"
        else:
            dcs, rank = f"{dc:.2f}", r"$|\Mset|$"
        L.append(f"{disp} & {par} & {path} & {pl:.2f} & {dcs} & {rank} \\\\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{6}{l}{\emph{ships no way to declare a hole at all}} \\[1pt]")
    for disp, fam, par, path, _, _, rank in EXTRA:
        L.append(f"{disp} & {par} & {path} & --- & --- & {rank} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    nmod = sum(1 for r in rows if r[0] in r[1] and "agg" in r[1][r[0]]
               and r[1][r[0]]["agg"].get("plain", {}).get("ratio_median") is not None)
    nfam = len(set(r[3] for r in rows if r[0] in r[1] and "agg" in r[1][r[0]]
                   and r[1][r[0]]["agg"].get("plain", {}).get("ratio_median") is not None))
    return "\n".join(L), nfam, nmod


def family_table(rows, src_of):
    """One row per family: the range of rho over its checkpoints, for the main text."""
    order, seen = [], {}
    for key, src, disp, fam, par, path in rows:
        if key not in src or "agg" not in src[key]:
            continue
        a = src[key]["agg"]
        pl = a.get("plain", {}).get("ratio_median")
        if pl is None:
            continue
        dec_key = "mask" if a.get("mask", {}).get("ratio_median") is not None else "nan"
        dc = a.get(dec_key, {}).get("ratio_median")
        if fam not in seen:
            seen[fam] = {"n": 0, "pl": [], "dc": [], "path": path, "par": []}
            order.append(fam)
        e = seen[fam]
        e["n"] += 1
        e["pl"].append(pl)
        e["dc"].append(dc)
        e["par"].append(par)
    BEH = {"Chronos-Bolt": "discards", "Chronos-2": "discards", "Chronos-T5": "discards",
           "TimesFM": "overwrites", "Moirai 1.1": "retains", "Moirai-MoE": "retains",
           "Moirai 2.0": "retains", "TiRex": "discards", "FlowState": "discards",
           "Timer-XL": "no declared path"}
    # Timer-XL is measured on the plain path but has no declared path at all: a raw NaN
    # renders its forecast non-finite, so its declared cell is a failure mode, not a rho.
    DECL_OVERRIDE = {"Timer-XL": r"\hi{poisons}"}
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{\textbf{Four conventions behind one description.} $\rho$ of "
         r"\eqref{eq:rho}, median over three datasets $\times$ three mechanisms $\times$ 60 "
         r"windows at rate $0.3$, $L=512$, $H=64$; ranges span the checkpoints of each family. "
         r"\hi{Red} marks every declared path that throws the fill's content away. "
         r"The $n$ column counts all $15$ checkpoints; \nummodels{} of them, from "
         r"\numfamilies{} families, admit a $\rho$ measurement --- the three decoder-only "
         r"checkpoints that do not are named in Appendix~\ref{app:excluded}. "
         r"\emph{Poisons}: a raw \texttt{NaN} renders Timer-XL's forecast non-finite, the "
         r"convention's failure mode rather than an interface. "
         r"Table~\ref{tab:familiesfull} in Appendix~\ref{app:families} lists every "
         r"checkpoint individually.}",
         r"\label{tab:families}", r"\vspace{2pt}", r"\small",
         r"\begin{tabular}{llrlcc}", r"\toprule",
         r"Family & Declared path & $n$ & Behaviour & $\rho$ plain & $\rho$ declared \\",
         r"\midrule"]
    def rng(v, lo=None):
        v = [x for x in v if x is not None]
        if not v:
            return "---"
        return f"{min(v):.2f}" if max(v) - min(v) < 5e-3 else f"{min(v):.2f}--{max(v):.2f}"
    for fam in order:
        e = seen[fam]
        d = rng(e["dc"])
        dcell = DECL_OVERRIDE.get(fam, r"\hi{0.00}" if d == "0.00" else d)
        L.append(f"{fam} & {e['path']} & {e['n']} & {BEH.get(fam,'')} & {rng(e['pl'])} "
                 f"& {dcell} \\\\")
    L += [r"\addlinespace[2pt]",
          r"TimeMoE, Sundial & none & 3 & no declared path & --- & --- \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


if __name__ == "__main__":
    # NOTE: the family tables and counts.tex moved to make_families.py, which reads all four
    # probe sources (s30, s36, s49_2025models, s46). The model_rows()/model_table()/
    # family_table() helpers below are dead and kept only for reference.
    open(os.path.join(HERE, "tab_breadth.tex"), "w").write(dataset_table() + "\n")
    print(f"wrote tab_breadth.tex ({len(DS)} datasets); "
          f"run make_families.py for tab_families*.tex and counts.tex")
