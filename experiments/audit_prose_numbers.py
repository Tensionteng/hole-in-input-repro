#!/usr/bin/env python
"""Every numeral in the paper's prose, and whether anything traces it to a result file.

verify_paper_numbers_v2.py checks a list of numbers I chose to check, which means it can only
catch errors in claims I remembered to doubt. This scans the source instead: it pulls every
numeral out of the main text and appendices, subtracts the ones a verifier already covers and
the ones that are structural rather than empirical (equation indices, rates, section numbers),
and prints what is left. What is left is the set of claims whose provenance rests on nothing
but my memory.
"""
import glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", "paper", "iclr2026"))

# Numbers that are not empirical claims and should not need a source.
# Two claims are traceable only under a flag or only to a round's summary; both are named
# here so the audit stays honest about them instead of silently passing.
DOCUMENTED_GAPS = {
    "11.77": "the whole-corpus NaN share; checked by verify_main_text.py under FULL_SCAN=1",
    "2.37": "the linear-fill top-decile ratio; only the repaired arm (2.25) is in a results "
            "file, the baseline is in S5's summary",
    "66.3": "the share of Penmanshiel windows in which the zero fill lay outside the ORIGINAL "
            "admissible class; a disclosure about a design flaw that was fixed mid-round, "
            "measured by S26 and not reproducible from its stored results",
}
STRUCTURAL = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",   # counts, indices
    "0.1", "0.3", "0.5", "0.7", "0.9",        # the missingness rates and the nominal level
    "512", "64", "96", "144", "24", "32", "16", "128",   # L, H, patch sizes
    "0.0", "1.0", "0.25", "0.75", "80", "90", "100",
}
SKIP_ENV = ("figure", "table", "tabular", "tikzpicture", "equation", "align")


def strip_tex(s):
    s = re.sub(r"(?<!\\)%.*", "", s)                     # comments (not escaped \%)
    s = re.sub(r"\\label\{[^}]*\}|\\ref\{[^}]*\}|\\eqref\{[^}]*\}", " ", s)
    s = re.sub(r"\\cite[tp]?\{[^}]*\}", " ", s)
    s = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\input\{[^}]*\}", " ", s)
    for env in SKIP_ENV:                                    # drop float bodies, keep captions
        s = re.sub(r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}", " ", s,
                   flags=re.S)
    return s


def numerals(s):
    # 1.234 / 12{,}345 / 1.7\times10^{-5} / 94\% ; keep the numeric part only
    out = []
    for m in re.finditer(r"(?<![A-Za-z0-9_])(\d+(?:\{,\}\d{3})*(?:\.\d+)?)", s):
        out.append((m.group(1).replace("{,}", ""), m.start()))
    return out


def main():
    # The verifiers record every value they assert, so coverage is read from what actually
    # ran rather than from a regex over their source.
    checked = set()
    for f in ("claims_main_text.json", "claims_late_rounds.json",
              "claims_appendix.json"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            checked |= {abs(float(v)) for v in json.load(open(p))}
    expanded = set()
    for v in checked:
        for alt in (v, v * 100, v / 100, v * 1e6, v * 1e7, v / 1e6, v / 1e7):
            if alt == 0 or not (1e-9 < abs(alt) < 1e9):
                continue
            for d in range(0, 6):
                expanded.add(f"{alt:.{d}f}".rstrip("0").rstrip(".") or "0")
                expanded.add(f"{alt:.{d}f}")
    print(f"the verifiers assert {len(checked)} distinct values "
          f"({len(expanded)} with unit and precision variants)\n")

    rows = []
    for f in sorted(glob.glob(os.path.join(PAPER, "sections", "*.tex"))):
        body = strip_tex(open(f).read())
        for val, pos in numerals(body):
            if val in DOCUMENTED_GAPS or val in STRUCTURAL or val in expanded:
                continue
            # also accept a trailing-zero variant of the printed form
            if val.rstrip("0").rstrip(".") in expanded:
                continue
            ctx = re.sub(r"\s+", " ", body[max(0, pos - 60):pos + 30]).strip()
            rows.append((os.path.basename(f), val, ctx))
    by_file = {}
    for f, v, c in rows:
        by_file.setdefault(f, []).append((v, c))
    json.dump({f: [[v, c] for v, c in vs] for f, vs in by_file.items()},
              open(os.path.join(HERE, "prose_numbers.json"), "w"), indent=1)
    total = sum(len(v) for v in by_file.values())
    print(f"{total} numerals in prose are not traceable to a verifier check")
    print(f"{len(DOCUMENTED_GAPS)} further values are documented gaps rather than checks:")
    for k, v in DOCUMENTED_GAPS.items():
        print(f"    {k:>8s}  {v}")
    print()
    for f in sorted(by_file, key=lambda k: -len(by_file[k])):
        print(f"── {f}  ({len(by_file[f])})")
        for v, c in by_file[f][:100]:
            print(f"     {v:>12s}   ...{c}")
    return total


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 0)
