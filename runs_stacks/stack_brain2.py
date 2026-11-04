"""Second mouse brain for the stack labelling experiment (2026-11-04).
Brain 2 = Allen Brain Cell Atlas, Zhuang-ABCA-1 (MERFISH, 1,122 genes, coronal, one hemisphere, 147 sections).
Same recipe as stack_jlp.py, reusing its functions unchanged: each section is clustered on its own (12 clusters, then 8
regions by neighbourhood mix), 6 stacks x 8 sections, scored against the Allen CCF division of each cell.
  python3 runs_stacks/stack_brain2.py prep      -> data/zhuang/stackcache2_{idx}.npz + data/zhuang/stacks2.json  (sbatch, ~20 min)
  python3 runs_stacks/stack_brain2.py STACK     -> results/stackb2_{i}.json
Also holds run_stack() and consistency(), reused by stack_consistency.py for brain 1.
Atlas-free consistency: for every pair (s, s+1), each cell of s is matched to its nearest cell of s+1 (KD-tree on the
placed coordinates, within 50 um, no match -> skipped); NMI between the class of the cell and the class of its match;
mean over the 7 pairs."""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, os, sys, time
import numpy as np, pandas as pd, h5py
from scipy.spatial import cKDTree
from sklearn.metrics import normalized_mutual_info_score as NMI
sys.path.insert(0, ".")
import stack_jlp as J
import stack_cap as SC
import stack_ctrl as CT
from bench_gt import UNIT, cluster
from bench_domains import neighbourhood_domains

KS, N = J.KS, J.N
Z = "data/zhuang"; MINCELLS = 14000; STRIDE = 2


def consistency(V, z, maxd=50.0):
    """z: class of each (section, label) node, length KS*N. Returns mean NMI over neighbouring pairs and class count."""
    cls = [z[s * N + V[s]["lab"]] for s in range(KS)]; vals = []; cov = []; agr = []; kap = []
    for s in range(KS - 1):
        t = cKDTree(V[s + 1]["xy"]); d, j = t.query(V[s]["xy"], distance_upper_bound=maxd); ok = np.isfinite(d)
        if ok.sum() < 100: continue
        a, b = cls[s][ok], cls[s + 1][j[ok]]
        vals.append(NMI(a, b)); cov.append(ok.mean()); ag = (a == b).mean(); agr.append(ag)
        K = max(a.max(), b.max()) + 1; pe = float((np.bincount(a, minlength=K) / len(a) * np.bincount(b, minlength=K) / len(b)).sum())
        kap.append((ag - pe) / (1 - pe))
    # NOTE: NMI is unchanged by renaming the classes of one section, so for one-to-one labellings it does not depend on the
    # matching. same-class rate and kappa (chance-corrected) do depend on it, so they are reported as well.
    return dict(cons=round(float(np.mean(vals)), 4), same=round(float(np.mean(agr)), 4), kappa=round(float(np.mean(kap)), 4), n_classes=int(len(np.unique(np.concatenate(cls)))), match_frac=round(float(np.mean(cov)), 3))


def zof(sig):
    return np.concatenate([sig[s] for s in range(KS)])


def run_stack(V, mid=3):
    """five-version-ready: methods on one version V; returns {method: {nmi, ari, n_classes, cons...}}"""
    C = {(s, u): J.table(V[s], V[u]) for s in range(KS) for u in range(s + 1, min(s + 4, KS))}
    C.update({(min(s, mid), max(s, mid)): J.table(V[min(s, mid)], V[max(s, mid)]) for s in range(KS)
              if s != mid and (min(s, mid), max(s, mid)) not in C})
    E = [(s, u) for (s, u) in C if u - s <= 2]; W = SC.weights(C, E)
    sigs = {"chain": J.chain(C, KS)}
    sigs["forest_d2"], _ = J.forest(C, list(C), KS, 2)
    sigs["star"] = J.star(C, KS, mid)
    zs = {k: zof(v) for k, v in sigs.items()}
    z0 = zs["forest_d2"]
    Cw = {e: SC.contrast(C[e]) for e in E}; sw, _ = J.forest(Cw, E, KS, 2); zw = zof(sw)
    for start, z in (("fromC", z0), ("fromW", zw)):
        for cap in (1, 2):
            zs[f"cap{cap}_{start}"] = SC.capped_search(W, z, N, cap=cap)
    out = {}
    for k, z in zs.items():
        lab = np.concatenate([z[s * N + V[s]["lab"]] for s in range(KS)])
        r = SC.score(V, z); r.update(consistency(V, z)); out[k] = r
    return out


def sections2():
    return json.load(open(f"{Z}/stacks2.json"))


def load2(sec, idx):
    zf = np.load(f"{Z}/stackcache2_{sec}.npz", allow_pickle=True); return {k: zf[k] for k in zf.files}


def prep():
    t0 = time.time(); lg = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    m = pd.read_csv(f"{Z}/cell_metadata.csv", usecols=["cell_label", "brain_section_label", "x", "y"], dtype={"cell_label": str})
    c = pd.read_csv(f"{Z}/ccf_coordinates.csv", dtype={"cell_label": str}).rename(columns={"x": "cx", "y": "cy", "z": "cz"})
    t = pd.read_csv(f"{Z}/parcellation_to_parcellation_term_membership.csv")
    dv = t[t.parcellation_term_set_name == "division"].drop_duplicates("parcellation_index").set_index("parcellation_index").parcellation_term_name
    m = m.merge(c, on="cell_label"); m = m[m.parcellation_index.isin(dv.index)].copy()
    m["div"] = dv.loc[m.parcellation_index].values
    zs = pd.read_csv(f"{Z}/cell_metadata.csv", usecols=["brain_section_label", "z"]).groupby("brain_section_label").z.first()
    cnt = m.brain_section_label.value_counts(); E = [s for s in zs.sort_values().index if cnt.get(s, 0) >= MINCELLS and zs[s] > 0.2]
    lg(f"cells with CCF division {len(m)}; eligible sections {len(E)}")
    # coordinate check: section frame (x,y) vs CCF (z=ML, y=DV)
    for nm, cols in (("section_xy", ["x", "y"]), ("ccf_ml_dv", ["cz", "cy"])):
        d = []
        for a, b in zip(E[10:60:7], E[12:62:7]):
            A = m[m.brain_section_label == a][cols].values; B = m[m.brain_section_label == b][cols].values
            d.append(np.median(cKDTree(B).query(A)[0]) * 1000)
        lg(f"alignment check {nm}: median nearest-cell distance between sections 2 apart (um) = {np.mean(d):.1f}")
    cols = ["x", "y"]
    starts = [int(round(i * (len(E) - 1 - STRIDE * (KS - 1)) / 5)) for i in range(6)]
    stacks = [[E[st + STRIDE * j] for j in range(KS)] for st in starts]
    use = sorted({s for st in stacks for s in st}); idx = {s: i for i, s in enumerate(E)}
    lg(f"stack starts {starts}; sections used {len(use)}")
    json.dump(dict(stacks=stacks, idx=idx, z={s: float(zs[s]) for s in use}), open(f"{Z}/stacks2.json", "w"))
    m = m[m.brain_section_label.isin(use)].set_index("cell_label")
    buf = {s: [] for s in use}
    with h5py.File(f"{Z}/Zhuang-ABCA-1-raw.h5ad") as h:
        lab = h["obs"]["cell_label"][:].astype(str); X = h["X"]; B = 16281 * 2
        pos = m.index.get_indexer(lab); sec = np.where(pos >= 0, m.brain_section_label.values[np.maximum(pos, 0)], "")
        for r0 in range(0, len(lab), B):
            r1 = min(r0 + B, len(lab)); sel = np.nonzero(pos[r0:r1] >= 0)[0]
            if len(sel) == 0: continue
            a = X[r0:r1][sel]
            for s in np.unique(sec[r0:r1][sel]):
                k = sec[r0:r1][sel] == s; buf[s].append((lab[r0:r1][sel][k], a[k].astype(np.float16)))
            if (r0 // B) % 20 == 0: lg(f"read rows {r0}/{len(lab)}")
    for s in use:
        cells = np.concatenate([b[0] for b in buf[s]]); Xs = np.vstack([b[1] for b in buf[s]]); buf[s] = None
        ok = Xs.astype(np.float32).sum(1) >= 10; cells, Xs = cells[ok], Xs[ok]
        q = m.loc[cells]; xy = q[cols].values * 1000.0; i = idx[s]
        Xf = Xs.astype(np.float32); cl = cluster(Xf, 12, i); lab12 = neighbourhood_domains(xy / UNIT, cl, K=N, seed=i)
        cl6 = cluster(Xf, 6, i + 100); lab6 = neighbourhood_domains(xy / UNIT, cl6, K=N, seed=i + 100)
        np.savez(f"{Z}/stackcache2_{s}.npz", X=Xs, xy=xy, lab=lab12, lab_s1=lab6, cl=cl, div=q["div"].astype(str).values,
                 struct=q.parcellation_index.astype(str).values)
        lg(f"{s}: {len(cells)} cells, {len(np.unique(q['div']))} divisions")
    lg("PREP DONE")


if __name__ == "__main__":
    if sys.argv[1] == "prep": prep(); sys.exit()
    i = int(sys.argv[1]); t0 = time.time(); info = sections2(); stack = info["stacks"][i]
    V0 = [load2(s, info["idx"][s]) for s in stack]
    res = {"stack": i, "sections": stack, "z_mm": [info["z"][s] for s in stack], "n_cells": [int(len(v["xy"])) for v in V0], "runs": {}}
    for name, V in CT.variants(V0).items():
        res["runs"][name] = run_stack(V)
        print(name, json.dumps({k: (v["nmi"], v["cons"]) for k, v in res["runs"][name].items()}), f"{time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(f"results/stackb2_{i}.json", "w"), indent=1)
    res["seconds"] = round(time.time() - t0); json.dump(res, open(f"results/stackb2_{i}.json", "w"), indent=1)
    print("DONE", flush=True)
