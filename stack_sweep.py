"""Cap and class-count sweep (2026-11-03, answering review finding 6: is cap 2 a lucky pick?).
Same stacks, versions, tables, weights w and start (forest under C) as stack_cc.py. Runs capped_search for caps 1-4
and class counts G = N (8) and G = 12, and reports NMI, ARI, classes used and the value of w.
Usage: python3 stack_sweep.py STACK -> results/stacksweep_{i}.json"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import stack_jlp as J
import stack_cap as SC
from stack_ctrl import variants

KS, N = J.KS, J.N

if __name__ == "__main__":
    i = int(sys.argv[1]); p, secs = J.sections()
    S = [J.load_section(p, sec, secs.index(sec)) for sec in secs[i * KS:(i + 1) * KS]]; del p
    res = {"stack": i, "runs": {}}
    for name, V in variants(S).items():
        C = {(s, u): J.table(V[s], V[u]) for s in range(KS) for u in range(s + 1, min(s + 3, KS))}
        E = list(C); W = SC.weights(C, E)
        sig, _ = J.forest(C, E, KS, 2); z0 = np.concatenate([sig[s] for s in range(KS)])
        out = {"start": dict(SC.score(V, z0), value=round(SC.value(W, z0), 4))}
        for G in (N, 12):
            for cap in (1, 2, 3, 4):
                zc = SC.capped_search(W, z0, G, cap=cap)
                out[f"cap{cap}_G{G}"] = dict(SC.score(V, zc), value=round(SC.value(W, zc), 4))
        res["runs"][name] = out
        print(name, json.dumps({k: v["nmi"] for k, v in out.items()}), flush=True)
        json.dump(res, open(f"results/stacksweep_{i}.json", "w"), indent=1)
    print("DONE", flush=True)
