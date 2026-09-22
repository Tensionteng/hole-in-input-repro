#!/usr/bin/env python
"""Triage: extract the tiny Status members from each SCADA zip via HTTP range
and tally status messages/IEC categories, to map where curtailment lives."""
import io
import re
import sys
import zlib
import struct
from collections import Counter

import pandas as pd

from zip_remote_extract import find_central_dir, content_length, http_range

ZIPS = {
    "kelmarsh": {
        y: f"https://zenodo.org/api/records/16807551/files/Kelmarsh_SCADA_{y}_{i}.zip/content"
        for y, i in [(2018, 3084), (2019, 3085), (2020, 3086), (2021, 4456),
                     (2022, 4457), (2023, 5961), (2024, 5962)]
    },
    "penmanshiel": {
        y: f"https://zenodo.org/api/records/16807304/files/Penmanshiel_SCADA_{y}_WT01-10_{i}.zip/content"
        for y, i in [(2016, 3107), (2017, 3114), (2018, 3113), (2019, 3112),
                     (2020, 3109), (2021, 4460), (2022, 4462)]
    },
}
# 2023 penmanshiel is split by quarters
ZIPS["penmanshiel"][2023] = "https://zenodo.org/api/records/16807304/files/Penmanshiel_SCADA_2023_02_WT_01-10_5982.zip/content"

KEYWORDS = re.compile(r"curtail|restrict|grid|constraint|export|limit|instruct|derat|redispatch|stop.*extern|external", re.I)


def fetch_member(url, e):
    lh = http_range(url, e["lho"], e["lho"] + 30 + 2048)
    nlen, elen = struct.unpack_from("<HH", lh, 26)
    data_off = e["lho"] + 30 + nlen + elen
    blob = http_range(url, data_off, data_off + e["csize"] - 1)
    if e["method"] == 8:
        return zlib.decompressobj(-15).decompress(blob)
    return blob


def main():
    for farm, years in ZIPS.items():
        for y, url in sorted(years.items()):
            try:
                es = find_central_dir(url, content_length(url))
            except Exception as ex:
                print(f"{farm} {y}: CD failed {ex}")
                continue
            msgs = Counter()
            cats = Counter()
            for e in es:
                if not e["name"].startswith("Status_"):
                    continue
                try:
                    raw = fetch_member(url, e)
                    df = pd.read_csv(io.BytesIO(raw), comment="#", dtype=str)
                    df.columns = [c.strip() for c in df.columns]
                    for _, r in df.iterrows():
                        m = str(r.get("Message", ""))
                        c = str(r.get("IEC category", ""))
                        s = str(r.get("Status", ""))
                        if KEYWORDS.search(m) or KEYWORDS.search(c) or KEYWORDS.search(s):
                            msgs[f"{s}|{r.get('Code','')}|{m}|{c}"] += 1
                        cats[c] += 1
                except Exception as ex:
                    print(f"  {e['name']}: {ex}")
            print(f"\n### {farm} {y}: keyword hits={sum(msgs.values())}")
            for k, v in msgs.most_common(12):
                print(f"   {v:5d}  {k}")
            print("   IEC cats:", dict(cats.most_common(10)))


if __name__ == "__main__":
    main()
