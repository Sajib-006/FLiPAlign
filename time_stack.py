"""Run time of the stack labelling methods on each of the 6 stacks (clean): tables, chain, forest d=2 + ascent, split search."""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
import stack_jlp as J
import stack_cap as SC
p, secs = J.sections(); out = []
for i in range(6):
    S = [J.load_section(p, sec, secs.index(sec)) for sec in secs[i * J.KS:(i + 1) * J.KS]]
    t0 = time.time(); C = {(s, u): J.table(S[s], S[u]) for s in range(J.KS) for u in range(s + 1, min(s + 3, J.KS))}; tt = time.time() - t0
    t0 = time.time(); J.chain(C, J.KS); tc = time.time() - t0
    t0 = time.time(); sig, _ = J.forest(C, list(C), J.KS, 2); tf = time.time() - t0
    z0 = np.concatenate([sig[s] for s in range(J.KS)]); E = [(s, u) for (s, u) in C if u - s <= 2]; W = SC.weights(C, E)
    t0 = time.time(); SC.capped_search(W, z0, J.N, cap=2); ts = time.time() - t0
    out.append(dict(stack=i, cells=int(sum(len(s["xy"]) for s in S)), tables_s=round(tt, 1), chain_s=round(tc, 3), forest_s=round(tf, 2), split_s=round(ts, 1)))
    print(out[-1], flush=True)
json.dump(out, open("results/stack_timing.json", "w"), indent=1)
