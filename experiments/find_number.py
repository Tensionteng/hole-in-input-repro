#!/usr/bin/env python
"""Where does a number in the paper come from?

Walks every results JSON in every round and reports the paths whose value matches, so a claim
in the prose can be traced to the file that produced it instead of to memory.
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield path, float(o)


def main(targets, rel=2e-3, only=None, scalars_only=True):
    files = sorted(glob.glob(os.path.join(HERE, "*", "*.json")))
    if only:
        files = [f for f in files if only in f]
    cache = []
    for f in files:
        try:
            cache.append((os.path.relpath(f, HERE), json.load(open(f))))
        except Exception:
            pass
    for t in targets:
        tv = float(t)
        hits = []
        for name, d in cache:
            for path, v in walk(d):
                # per-window arrays contain thousands of near-matches by chance; a claim in
                # the prose always quotes a summary, so only scalars can be its source.
                if scalars_only and "[" in path:
                    continue
                for scale, lab in ((1, ""), (100, " x100"), (0.01, " /100")):
                    if v == 0:
                        continue
                    if abs(v * scale - tv) <= rel * max(abs(tv), 1e-9):
                        hits.append((name, path, v, lab))
                        break
        print(f"\n=== {t}  ({len(hits)} hits)")
        for name, path, v, lab in hits[:10]:
            print(f"    {name:44s} {path}  = {v:g}{lab}")
        if len(hits) > 10:
            print(f"    ... and {len(hits)-10} more")


if __name__ == "__main__":
    a = sys.argv[1:]
    only = None
    if a and a[0].startswith("--only="):
        only = a.pop(0).split("=", 1)[1]
    main(a, only=only)
