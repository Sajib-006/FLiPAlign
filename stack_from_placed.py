"""Two-step pipeline on the six stacks (2026-10-22): FLiPAlign placements (results/stackplace_{k}.json) are chained
along each stack, then the stack is labelled at the placed positions and scored against Allen divisions.
Chaining: pair j placed section j+1 onto section j at its true position; FLiPAlign's score is unchanged by a rigid
move of both sections, so placing j+1 onto section j at its placed position F_j(true_j) gives F_{j+1} = F_j o E_j,
where E_j (true -> placed, fitted rigidly to the answer) is stored with each pair. Section 0 keeps its position.
Reports the drift (median distance of each section from its true position) and NMI/ARI of forest and split labelling,
next to the same methods at the Allen positions. Usage: python3 stack_from_placed.py -> results/stack_placed.json"""
import json, sys
import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
sys.path.insert(0, ".")
import stack_jlp as J
import stack_cap as SC

KS, N = J.KS, J.N
import os
PRE = os.environ.get("PRE", "stackplace")


def score(V, z):
    lab = np.concatenate([z[s * N + V[s]["lab"]] for s in range(KS)]); div = np.concatenate([v["div"] for v in V])
    return dict(nmi=round(NMI(div, lab), 4), ari=round(ARI(div, lab), 4))


def label(V):
    C = {(s, u): J.table(V[s], V[u]) for s in range(KS) for u in range(s + 1, min(s + 3, KS))}
    sig, _ = J.forest(C, list(C), KS, 2); z0 = np.concatenate([sig[s] for s in range(KS)])
    E = [(s, u) for (s, u) in C if u - s <= 2]; W = SC.weights(C, E)
    return {"forest": score(V, z0), "split": score(V, SC.capped_search(W, z0, N, cap=2))}


p, secs = J.sections(); out = []
for st in range(6):
    S = [J.load_section(p, sec, secs.index(sec)) for sec in secs[st * KS:(st + 1) * KS]]
    pr = [json.load(open(f"results/{PRE}_{st * 7 + j}.json"))[0] for j in range(7)]
    R, t = np.eye(2), np.zeros(2); V = [dict(S[0])]; drift = [0.0]
    for j in range(7):
        Ej, ej = np.array(pr[j]["E_R"]), np.array(pr[j]["E_t"]) * J.UNIT            # stored in units; positions are um
        R, t = R @ Ej, R @ ej + t                                                   # F_{j+1} = F_j o E_j
        xy = S[j + 1]["xy"] @ R.T + t
        drift.append(round(float(np.median(np.linalg.norm(xy - S[j + 1]["xy"], axis=1))), 1))
        V.append(dict(S[j + 1], xy=xy))
    r = {"stack": st, "pair_err_um": [q["err_um"] for q in pr], "certified": [bool(q["certified"]) for q in pr],
         "drift_um": drift, "placed": label(V), "allen": label(S)}
    out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(f"results/{PRE}_summary.json", "w"), indent=1)
