#!/usr/bin/env python
"""s69b G0b tuner: search the harmonic period bank / energy share until the headroom
gate clears (median MSE*(0,0)/MSE*(1,1) >= 2.0 at tailblock 0.7, >= 1.5 at block 0.7).
Uses the REAL v2 generator and conditioning path (gen_s69_corpus / bayes_s69), so the
numbers here are exactly what the gate will compute. Scratch tool for parameter
selection; the chosen parameters go into gen_s69_corpus.HARM_BANK/HARM_SHARE and are
recorded in s69_notes.md."""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_s69_corpus as g69
import bayes_s69 as b69

L, H = g69.L, g69.H
N = 64


def headroom(bank, share, n=N):
    x, c, phi, rho, h, per = g69.gen_pool_v2(n, g69.EVAL_SEED_V2, bank, share)
    ratios = {"tailblock": [], "block": []}
    floors = []
    for i in range(n):
        S_obs, S_fo, S_ff = b69.series_covariances_v2(float(phi[i]), float(rho[i]),
                                                      float(h[i]), per[i])
        _, cl = b69.condition(S_obs, S_fo, S_ff, np.ones(2 * L), np.zeros(2 * L))
        floors.append(cl)
        for mech in ratios:
            m = g69.eval_target_mask(i, mech, 0.7)
            blend, nv = b69.cell_obs_vectors(m, 0.0, 0.0)
            _, m0 = b69.condition(S_obs, S_fo, S_ff, blend, nv)
            ratios[mech].append(m0 / cl)
    return (float(np.median(ratios["tailblock"])), float(np.median(ratios["block"])),
            float(np.median(floors)))


CANDS = [((16, 32, 64), (0.3, 0.6)),
         ((32, 64, 128), (0.3, 0.6)),
         ((64, 128, 256), (0.3, 0.6)),
         ((96, 192, 384), (0.3, 0.6)),
         ((128, 256, 512), (0.3, 0.6)),
         ((128, 256, 512), (0.5, 0.8)),
         ((192, 384, 768), (0.5, 0.8))]

CANDS2 = [((192, 384, 768), (0.6, 0.85)),
          ((256, 512, 768), (0.5, 0.8)),
          ((256, 512, 1024), (0.5, 0.8)),
          ((256, 512, 1024), (0.6, 0.85)),
          ((384, 768, 1536), (0.5, 0.8)),
          ((384, 768, 1536), (0.6, 0.9)),
          ((512, 1024, 2048), (0.6, 0.9))]

if __name__ == "__main__":
    cands = CANDS2 if len(sys.argv) > 1 and sys.argv[1] == "round2" else CANDS
    for bank, share in cands:
        rt, rb, fl = headroom(bank, share)
        ok = rt >= b69.G0B_TAIL and rb >= b69.G0B_BLOCK
        print(f"bank={bank} share={share}: tail0.7={rt:.2f} (>=2.0) "
              f"block0.7={rb:.2f} (>=1.5) floor={fl:.3f}  {'CLEARS' if ok else 'short'}",
              flush=True)
