"""Refusal-aware two-step pipeline: the stack is cut at links FLiPAlign refused.
stack_from_placed.py chains every FLiPAlign placement, refused ones included. Here a refused link cuts the stack:
the sections are split into pieces joined only by certified links. Each piece keeps its first section at its true
position (pieces are not placed relative to each other, since no certified link joins them). Inside a piece the
placements are chained as before (F_{j+1} = F_j o E_j), and the piece is labelled on its own (forest and split, sections
up to two apart, same settings as the paper). Reported per stack: the pieces, the share of sections in pieces of two or
more, the drift of every section (median distance from its true position), and the NMI of each piece with atlas
divisions, plus the cell-weighted mean over pieces of two or more. The all-links chaining is reported next to it.
Usage: PRE=stackplacefk python3 runs_stacks/stack_refuse.py -> results/{PRE}_refuse.json"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, os, sys
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as NMI
sys.path.insert(0, ".")
import stack_jlp as J
import stack_cap as SC

KS, N = J.KS, J.N
PRE = os.environ.get("PRE", "stackplacefk")


def weights(C, edges, K):
    W = np.zeros((K * N, K * N))
    for s, u in edges:
        B = SC.contrast(C[(s, u)]); W[s * N:(s + 1) * N, u * N:(u + 1) * N] = B; W[u * N:(u + 1) * N, s * N:(s + 1) * N] = B.T
    return W


def label(V):
    """forest and split labelling of one piece; returns NMI of each and the cell count"""
    K = len(V); div = np.concatenate([v["div"] for v in V])
    if K == 1:
        return {"forest": round(NMI(div, V[0]["lab"]), 4), "split": round(NMI(div, V[0]["lab"]), 4), "cells": len(div)}
    C = {(s, u): J.table(V[s], V[u]) for s in range(K) for u in range(s + 1, min(s + 3, K))}
    sig, _ = J.forest(C, list(C), K, 2); z0 = np.concatenate([sig[s] for s in range(K)])
    zc = SC.capped_search(weights(C, list(C), K), z0, N, cap=2)
    sc = lambda z: round(NMI(div, np.concatenate([z[s * N + V[s]["lab"]] for s in range(K)])), 4)
    return {"forest": sc(z0), "split": sc(zc), "cells": len(div)}


p, secs = J.sections(); out = []
for st in range(6):
    S = [J.load_section(p, sec, secs.index(sec)) for sec in secs[st * KS:(st + 1) * KS]]
    pr = [json.load(open(f"results/{PRE}_{st * 7 + j}.json"))[0] for j in range(7)]
    cert = [bool(q["certified"]) for q in pr]
    pieces, cur = [], [0]
    for j in range(7):
        if cert[j]: cur.append(j + 1)
        else: pieces.append(cur); cur = [j + 1]
    pieces.append(cur)
    V, drift = [None] * KS, [0.0] * KS
    for pc in pieces:
        R, t = np.eye(2), np.zeros(2); V[pc[0]] = dict(S[pc[0]])
        for a, b in zip(pc[:-1], pc[1:]):
            Ej, ej = np.array(pr[a]["E_R"]), np.array(pr[a]["E_t"]) * J.UNIT
            R, t = R @ Ej, R @ ej + t
            xy = S[b]["xy"] @ R.T + t
            drift[b] = round(float(np.median(np.linalg.norm(xy - S[b]["xy"], axis=1))), 1)
            V[b] = dict(S[b], xy=xy)
    lab = [dict(label([V[k] for k in pc]), sections=pc) for pc in pieces]
    big = [r for r in lab if len(r["sections"]) > 1]
    wmean = lambda key: round(sum(r[key] * r["cells"] for r in big) / max(sum(r["cells"] for r in big), 1), 4) if big else None
    r = {"stack": st, "certified": cert, "pieces": pieces, "share_in_pieces_ge2": round(sum(len(pc) for pc in pieces if len(pc) > 1) / KS, 3),
         "drift_um": drift, "max_drift_um": max(drift), "piece_labels": lab, "nmi_forest_pieces": wmean("forest"), "nmi_split_pieces": wmean("split")}
    out.append(r); print(json.dumps(r), flush=True)
    json.dump(out, open(f"results/{PRE}_refuse.json", "w"), indent=1)
print("DONE", flush=True)
