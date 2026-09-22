#!/usr/bin/env python
"""Fit the S16 mechanism detector once and cache it.

S38 and S39 each need the same frozen classifier. Fitting it inside every process put four
400-iteration HistGradientBoosting fits on the machine at once and starved the GPU rounds.
"""
import os, pickle, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "s05_characterization"))
sys.path.insert(0, os.path.join(HERE, "s16_mechgate"))
import run_s16_mechgate as s16

OUT = os.path.join(HERE, "s16_mechgate", "s16_ckpt", "gbdt_frozen.pkl")
F, y, _ = s16.build_train_features(n_win=220)
print("features", F.shape)
clf = s16.fit_clf("gbdt", F, y)
pickle.dump(clf, open(OUT, "wb"))
print("wrote", OUT)
