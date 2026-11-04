"""Z7: construction B (chance-corrected tables) on the 3DM graph of the split hardness proof, caps 2 and 3, G = M.
Pad: smallest even n' with c*n*b < n' (the condition the proof uses). ok=True means OPT >= 2M exactly for 3DM yes-instances.
Cap 3 with q=2 takes about 30 minutes per instance."""
import sys, time, itertools, random
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_zero_marginal import *
# 3DM graphs, construction B, caps 2,3
cases = [(1, [(0,0,0)]), (2, [(0,0,0),(1,1,1)]), (2, [(0,0,0),(1,1,0)]), (2, [(0,1,0),(1,0,1),(1,1,1)]), (2, [(0,0,1),(1,0,0),(1,1,1)])]
for c in (2, 3):
    for q, tr in cases:
        t0 = time.time()
        L, Rr, adj = build_H(q, tr); n = len(L); adjA = H_to_pair(L, Rr, adj)
        npad = pm_pad_size(n, adjA, c); W = pad_pm(n, adjA, npad); M = len(W)
        assert zero_margins(W)
        core, x, y, nb, vin, vout = agg_from_W(W, n, 2)
        opt = opt_agg(core, x, y, 2, nb, vin, vout, c, M)
        dm = dm_exists(q, tr)
        print(f"B c={c} q={q} m={len(tr)} dm={dm} n={n} M={M} OPT={float(opt):.3f} 2M={2*M} ok={(opt >= 2*M) == dm}  {time.time()-t0:.0f}s", flush=True)
