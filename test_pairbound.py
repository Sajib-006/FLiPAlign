"""Does the tighter pair bound turn refusals into proofs? DLPFC 151507 vs 151508, old vs new
bound, same settings: gap, rotation eps, and eps by far-shift radius (spots)."""
import sys, time; sys.path.insert(0, '.')
import numpy as np, run_step2 as S
from smoothalign import SmoothAlign
from smoothpar import solve_par, far_par, far_par_ext
A, la = S.dlpfc(151507); B, lb = S.dlpfc(151508); lb = np.random.default_rng(0).permutation(7)[lb]
for pb in (False, True):
    al = SmoothAlign(A, la, B, lb, h=0.25, sigma=1.0, tpad=8.0, contrast=True, pair_bound=pb)
    t0 = time.time(); o = solve_par(al, n_proc=6); th, t, fl = o['transform']
    u = far_par(al, o['best'], th, fl, n_proc=6)
    rads = {}
    for rad in (1.0, 2.0, 3.0, 5.0, 8.0):
        ue = far_par_ext(al, o['best'], th, fl, t, shift_radius=rad, n_proc=6); rads[rad] = round(ue['eps'], 4)
    print("pair_bound=%s: gap %.4f  eps_rotation %.4f  eps by shift radius (spots) %s  evals %d  total %.0fs"
          % (pb, o['gap'], u['eps'], rads, o['evals'], time.time() - t0), flush=True)
print("DONE", flush=True)
