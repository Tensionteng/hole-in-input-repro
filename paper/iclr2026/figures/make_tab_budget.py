#!/usr/bin/env python
"""The error-budget table, generated from S25's paired aggregate.

Written as a generator rather than maintained by hand so that no number in it can drift from
the run that produced it. The marked cell in each row is the largest of the three attributed
terms, chosen by the data rather than by the author.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
A = json.load(open(os.path.join(EXP, "s25_twofloor", "s25_paired_agg.json")))

MECH = [("mcar", "scattered dropout"), ("block", "block outage"),
        ("mnar_high", "value censoring"), ("mnar_extreme", "extreme censoring")]
RATES = (0.3, 0.7)


def num(v):
    return (f"$-${abs(v):.3f}" if v < 0 else f"{v:.3f}")


L = [r"\begin{table}[t]", r"\centering",
     r"\caption{\textbf{Where the damage lives.} The decomposition of \eqref{eq:decomp}. "
     r"\hi{Red} marks the largest of the three attributed terms in each row. Under scattered "
     r"and block missingness the fill term is negative and the damage is information or "
     r"architecture; only under heavy censoring is the fill the place to intervene. \bolt{} "
     r"base, $L=512$, $H=64$.}",
     r"\label{tab:budget}", r"\vspace{2pt}", r"\small",
     r"\begin{tabular}{llcccccc}", r"\toprule",
     r"& & \multicolumn{3}{c}{achievable relMSE} & \multicolumn{3}{c}{excess attributed to} \\",
     r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
     r"Mechanism & rate & best fixed & learned fill & direct & (i) info & (ii) arch "
     r"& (iii) fill \\", r"\midrule"]
for mi, (m, label) in enumerate(MECH):
    if mi:
        L.append(r"\addlinespace[2pt]")
    for ri, r in enumerate(RATES):
        c = A[f"{m}:{r}"]
        terms = [c["term1_information"], c["term2_architecture"], c["term3_fill"]]
        k = max(range(3), key=lambda i: terms[i])
        cells = [num(v) for v in terms]
        cells[k] = r"\hi{" + cells[k] + "}"
        head = (r"\multirow{2}{*}{" + label + "}") if ri == 0 else ""
        L.append(f"{head} & {r} & {c['best_fixed']:.3f} & {c['fillnet']:.3f} & "
                 f"{c['direct']:.3f} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(HERE, "tab_budget.tex"), "w").write("\n".join(L) + "\n")

# figures the prose quotes, recomputed here so they cannot drift from the table
tot = {f"{m}:{r}": sum([A[f'{m}:{r}']['term1_information'],
                        A[f'{m}:{r}']['term2_architecture'],
                        A[f'{m}:{r}']['term3_fill']]) for m, _ in MECH for r in RATES}
print("wrote tab_budget.tex")
print("  total excess at rate 0.7: " + "  ".join(
    f"{m}={tot[f'{m}:0.7']:.4f}" for m, _ in MECH))
print("  mnar_high:0.7 winrate = %.4f" % A["mnar_high:0.7"]["winrate"])
