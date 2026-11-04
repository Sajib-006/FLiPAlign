"""Cap-1 control (2026-11-02, answering review finding 6).
The split method optimises the chance-corrected table w, the one-to-one methods optimise C. To see whether the gain
comes from allowing splits or from the change of score, compare under the SAME score w and the SAME local search:
  cap1: capped_search with cap=1 (only swaps of two labels of one section: one-to-one under w)
  cap2: capped_search with cap=2 (the split method of the paper)
each from two starts: the forest answer under C (z0, as in the paper) and the forest answer under w (forest_w).
Same 6 stacks x 5 versions as stack_cc.py. Usage: python3 stack_ctrl.py STACK -> results/stackctrl_{i}.json"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import stack_jlp as J
import stack_cap as SC

KS, N = J.KS, J.N


def variants(S):
    mid = 3; out = {"clean": S}
    s1 = [dict(s) for s in S]; s1[mid] = dict(S[mid], lab=S[mid]["lab_s1"]); out["S1_recluster"] = s1
    s2 = [dict(s) for s in S]; xs = S[mid]["xy"][:, 0]; keep = xs < np.median(xs)
    s2[mid] = {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(xs) else v) for k, v in S[mid].items()}; out["S2_crop50"] = s2
    def shuffled(s, frac, seed):
        r = np.random.default_rng(seed); lab = s["lab"].copy(); j = r.random(len(lab)) < frac; lab[j] = r.integers(0, N, j.sum())
        return dict(s, lab=lab)
    s3 = [dict(s) for s in S]; s3[mid] = shuffled(S[mid], 0.3, 1); out["S3_noise30"] = s3
    s4 = [dict(s) for s in S]; s4[3] = shuffled(S[3], 0.3, 2); s4[5] = shuffled(S[5], 0.3, 3); out["S4_two_noisy"] = s4
    return out


if __name__ == "__main__":
    i = int(sys.argv[1]); p, secs = J.sections()
    S = [J.load_section(p, sec, secs.index(sec)) for sec in secs[i * KS:(i + 1) * KS]]; del p
    res = {"stack": i, "runs": {}}
    for name, V in variants(S).items():
        C = {(s, u): J.table(V[s], V[u]) for s in range(KS) for u in range(s + 1, min(s + 3, KS))}
        E = list(C); W = SC.weights(C, E)
        sig, _ = J.forest(C, E, KS, 2); z0 = np.concatenate([sig[s] for s in range(KS)])
        Cw = {e: SC.contrast(C[e]) for e in E}
        sw, _ = J.forest(Cw, E, KS, 2); zw = np.concatenate([sw[s] for s in range(KS)])
        out = {}
        for start, z in (("fromC", z0), ("fromW", zw)):
            out[f"start_{start}"] = dict(SC.score(V, z), value=round(SC.value(W, z), 4))
            for cap in (1, 2):
                zc = SC.capped_search(W, z, N, cap=cap)
                out[f"cap{cap}_{start}"] = dict(SC.score(V, zc), value=round(SC.value(W, zc), 4))
        res["runs"][name] = out
        print(name, json.dumps({k: v["nmi"] for k, v in out.items()}), flush=True)
        json.dump(res, open(f"results/stackctrl_{i}.json", "w"), indent=1)
    print("DONE", flush=True)
