#!/usr/bin/env python
"""Quick curtailment scan of a probed Kelmarsh/Penmanshiel turbine CSV."""
import sys

import numpy as np
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path, skiprows=9, low_memory=False)
df = df.rename(columns={df.columns[0]: "time"})
print(f"== {path}: {len(df)} rows, cols={len(df.columns)}")
for c in df.columns:
    if "Curtail" in c or "setpoint" in c.lower():
        s = pd.to_numeric(df[c], errors="coerce")
        nn = s.notna().mean()
        pos = (s.fillna(0) > 0).mean()
        extra = f" frac>0={pos:.4f} sum={s.sum():.0f}" if "Curtail" in c else \
                f" frac<2049={(s < 2049).mean():.4f} min={s.min()}"
        print(f"  {c!r}: nonnull={nn:.3f} {extra}")
p = pd.to_numeric(df.get("Power (kW)"), errors="coerce")
w = pd.to_numeric(df.get("Wind speed (m/s)"), errors="coerce")
print(f"  power nonnull={p.notna().mean():.3f} wind nonnull={w.notna().mean():.3f}")
