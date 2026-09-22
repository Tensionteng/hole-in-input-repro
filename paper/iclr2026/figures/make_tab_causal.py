#!/usr/bin/env python
"""The natural-experiment table, generated from S46 (Moirai 2.0) so its cells cannot drift."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
R = json.load(open(os.path.join(EXP, "s46_moirai2", "s46_results.json")))
S = R["sweep"]
DS = ("ETTh1", "ETTm1", "weather")
MECH = [("mcar", "scattered dropout"), ("block", "block outage"),
        ("mnar_high", "value censoring")]
RATES = (0.3, 0.7)


def cell(mech, rate, a, model):
    return float(np.mean([S[f"{d}|{mech}|{rate}|{a}|{model}|declared"]["median"]
                          for d in DS]))


L = [r"\begin{table}[!tb]", r"\centering",
     r"\caption{\textbf{The same sweep on two interfaces.} Both declared paths receive an "
     r"identical fill on identical windows; Moirai 2.0 retains its content and \bolt{} "
     r"does not. At $\alphaq{=}1$ the fill \emph{is} the ground truth, so \hi{red} marks the "
     r"four cells where a perfect imputation should show up if it can: it removes most of "
     r"Moirai 2.0's damage (though not all of it, a point Section~\ref{sec:causal} "
     r"returns to) and leaves \bolt{} where it was. $L=512$, $H=64$, $150$ windows per "
     r"cell, averaged over three datasets.}",
     r"\label{tab:causal}", r"\vspace{2pt}", r"\small",
     r"\begin{tabular}{lcccccc}", r"\toprule",
     r"& & \multicolumn{2}{c}{\bolt{} (declared)} & \multicolumn{2}{c}{Moirai 2.0 (declared)}"
     r" \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
     r"Mechanism & rate & $\alphaq\!=\!0$ & $\alphaq\!=\!1$ & $\alphaq\!=\!0$ "
     r"& $\alphaq\!=\!1$ \\", r"\midrule"]
for mi, (m, label) in enumerate(MECH):
    if mi:
        L.append(r"\addlinespace[2pt]")
    for ri, r in enumerate(RATES):
        v = [cell(m, r, 0.0, "bolt"), cell(m, r, 1.0, "bolt"),
             cell(m, r, 0.0, "moirai2"), cell(m, r, 1.0, "moirai2")]
        f = [f"{x:.3f}" for x in v]
        # the alpha=1 column: where the two families part company
        if m != "mnar_high":
            f[3] = r"\hi{" + f[3] + "}"
        row = (label if ri == 0 else "") + f" & {r} & {f[0]} & {f[1]} & {f[2]} & {f[3]} \\\\"
        L.append(row)
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
out = os.path.join(HERE, "tab_causal.tex")
open(out, "w").write("\n".join(L) + "\n")
print("wrote", out)
