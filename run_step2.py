"""Step 2: redo the older small-data experiments with the NEW method (SmoothAlign, contrast
score, proven-uniqueness test), so the paper presents one method.

  k1     DLPFC Visium, 5 pairs as in run_k1.py (adjacent, 300 um apart, different donors,
         adjacent + known 73 deg + flip, adjacent with L2+L3 merged in B). Layer names in B
         scrambled; report angle, flip, layer matching recovered (x/7), uniqueness.
  amb    DLPFC ambiguity survey, all 66 pairs (as in run_ambiguity.py). "Ambiguous" now
         means the uniqueness proof FAILS (eps <= 0): some alignment >20 deg away (or
         flipped) could score within the bound of the best.
  tonsil CytAssist RNA vs protein, same design as run_blockC.py (independent clusterings,
         different 70% of spots each, known rotation + flip), 6 settings.
  far    Visium vs Xenium Rep1, regions, h=2, sigma 20, with a 3 h uniqueness check
         (the 1 h check ran out of time).

Units: DLPFC and tonsil in spot spacings (100 um), Xenium in cell spacings.
Usage: python3 run_step2.py k1|amb|tonsil|far
"""
import itertools, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from smoothalign import SmoothAlign, far_upper_bound, rot, FLIP
from scipy.optimize import linear_sum_assignment

LAY = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]
DONOR = {151507: 1, 151508: 1, 151509: 1, 151510: 1,
         151669: 2, 151670: 2, 151671: 2, 151672: 2,
         151673: 3, 151674: 3, 151675: 3, 151676: 3}
ADJACENT = {(151507, 151508), (151509, 151510), (151669, 151670),
            (151671, 151672), (151673, 151674), (151675, 151676)}
SIG, H, TPAD = 1.0, 0.25, 8.0          # small-data settings (spot units)


def save(out, r):
    res = json.load(open(out)) if os.path.exists(out) else []
    res.append(r); json.dump(res, open(out, "w"), indent=1); print(json.dumps(r), flush=True)


def angdiff(a, b):
    return float(abs(np.degrees(np.angle(np.exp(1j * (a - b))))))


_c = {}
def dlpfc(s):
    if s in _c: return _c[s]
    lm = pd.read_csv("data/dlpfc/barcode_level_layer_map.tsv", sep="\t", header=None,
                     names=["barcode", "sample", "layer"])
    p = pd.read_csv(f"data/dlpfc/pos_{s}.txt", header=None,
                    names=["barcode", "in_tissue", "row", "col", "py", "px"])
    m = p[p.in_tissue == 1].merge(lm[lm["sample"] == s], on="barcode")
    m = m[m.layer.isin(LAY)]
    xy = np.c_[m.px.values, m.py.values] / 137.0
    _c[s] = (xy - xy.mean(0), np.array([LAY.index(l) for l in m.layer]))
    return _c[s]


def matching(al, th, fl, t):
    P = al._points(th, fl) + t
    x = (P - al.lo) / al.h; i0 = np.floor(x).astype(int); f = x - i0
    Gs = np.zeros((al.n, al.KB))
    for (di, dj, w) in ((0, 0, (1 - f[:, 0]) * (1 - f[:, 1])), (1, 0, f[:, 0] * (1 - f[:, 1])),
                        (0, 1, (1 - f[:, 0]) * f[:, 1]), (1, 1, f[:, 0] * f[:, 1])):
        Gs += w[:, None] * al.G[:, i0[:, 0] + di, i0[:, 1] + dj].T
    C = np.zeros((al.KA, al.KB)); np.add.at(C, al.la, Gs)
    _, r_, c_ = al._assign(C)
    return {int(k): int(l) for k, l in zip(r_, c_) if l < al.KB}


def align(A, la, B, lb, sigma=SIG, h=H, tpad=TPAD, far_limit=1800):
    al = SmoothAlign(A, la, B, lb, h=h, sigma=sigma, tpad=tpad, contrast=True)
    out = al.solve(time_limit=3600)
    th, t, fl = out["transform"]
    r = {"best": round(out["best"], 4), "gap": round(out["gap"], 4),
         "certified": out["certified"], "angle_deg": round(float(np.degrees(th)) % 360, 2),
         "flip": int(fl), "shift": [round(float(v), 2) for v in t],
         "solve_s": round(out["secs"], 1)}
    if out["certified"]:
        u = far_upper_bound(al, out["best"], th, fl, time_limit=far_limit)
        r.update({"far_exact": round(u["far_exact"], 4), "eps_unique": round(u["eps"], 4),
                  "unique_PROVEN": bool(u["eps"] > 0), "far_complete": u["complete"]})
    return al, (th, t, fl), r


def k1():
    rng = np.random.default_rng(0)
    cases = [(151507, 151508, "adjacent 10 um, same donor", None, False),
             (151507, 151510, "same donor, 300 um apart", None, False),
             (151507, 151673, "different donors (negative control)", None, False),
             (151507, 151508, "adjacent + known 73 deg and mirror flip",
              (np.radians(73.0), np.array([6.0, -4.0]), 1), False),
             (151507, 151508, "adjacent, B merges L2+L3", None, True)]
    for a, b, note, extra, merge in cases:
        A, la = dlpfc(a); B, lb = dlpfc(b)
        perm = rng.permutation(7); lb2 = perm[lb]
        if merge:
            lb2 = np.where(lb2 == perm[2], perm[1], lb2); _, lb2 = np.unique(lb2, return_inverse=True)
        if extra is not None:
            B = (rot(extra[0]) @ (FLIP if extra[2] else np.eye(2)) @ B.T).T + extra[1]
        t0 = time.time()
        al, (th, t, fl), r = align(A, la, B, lb2)
        pi = matching(al, th, fl, t)
        right = sum(1 for k in range(7) if pi.get(k, -1) == perm[k]) if not merge else None
        r = {"pair": f"{a} vs {b}", "note": note, **r,
             "layers matched right": f"{right}/7" if right is not None else "n/a (merged)",
             "total_s": round(time.time() - t0, 1)}
        if extra is not None:
            r["angle_err_vs_known"] = round(angdiff(th, extra[0]), 2)
            r["flip_ok"] = int(fl == extra[2])
        save("results_step2_k1.json", r)


def amb():
    for i, (a, b) in enumerate(itertools.combinations(sorted(DONOR), 2), 1):
        grp = ("adjacent" if (a, b) in ADJACENT else
               "same donor" if DONOR[a] == DONOR[b] else "different donor")
        A, la = dlpfc(a); B, lb = dlpfc(b)
        perm = np.random.default_rng(i).permutation(7)
        _, _, r = align(A, la, B, perm[lb], far_limit=900)
        save("results_step2_ambiguity.json", {"i": i, "a": a, "b": b, "group": grp, **r})


def tonsil():
    import run_blockC as BC                              # reuses its loader (clusterings)
    xy, lr, lp, sp_ = BC.load()
    KA, KB = lr.max() + 1, lp.max() + 1
    truth, agree = BC.true_correspondence(lr, lp, KA, KB)
    rng = np.random.default_rng(0)
    for ang, fl in [(0.0, 0), (73.0, 1), (211.0, 0), (137.0, 1), (290.0, 0), (30.0, 1)]:
        n = len(xy)
        iA = rng.choice(n, int(0.7 * n), replace=False)
        iB = rng.choice(n, int(0.7 * n), replace=False)
        th0 = np.radians(ang); t = np.array([5.0, -3.0])
        A, la = xy[iA], lr[iA]
        B = (rot(th0) @ (FLIP if fl else np.eye(2)) @ xy[iB].T).T + t
        al, (th, tt, f), r = align(A, la, B, lp[iB])
        pi = matching(al, th, f, tt)
        shared = [k for k in truth if k in pi]
        r = {"applied": [ang, fl], "domains RNA/protein": [int(KA), int(KB)], **r,
             "angle_err": round(angdiff(th, th0), 2), "flip_ok": int(f == fl),
             "domain correspondence": f"{sum(pi[k] == truth[k] for k in shared)}/{len(shared)}",
             "ceiling (same-spot agreement)": round(float(agree), 3)}
        save("results_step2_tonsil.json", r)


def far():
    import run_visium as RV
    from scipy.spatial import cKDTree
    from bench_domains import neighbourhood_domains
    xy_um, lab = RV.xenium("Rep1")
    d, _ = cKDTree(xy_um).query(xy_um[::20], k=2); RV.SP["um"] = float(np.median(d[:, 1]))
    xyX = xy_um / RV.SP["um"]
    reg = neighbourhood_domains(xyX, lab, K=6, seed=1)
    A = RV.xyV_um / RV.SP["um"]
    al = SmoothAlign(A, RV.labV, xyX, reg, h=2.0, sigma=20.0, tpad=300.0, contrast=True)
    out = al.solve(time_limit=7200)
    th, t, fl = out["transform"]
    u = far_upper_bound(al, out["best"], th, fl, time_limit=10800)
    save("results_step2_far.json", {"test": "Visium vs Xenium Rep1, regions, h=2, sigma 20, 3 h far check",
         "angle_deg": round(float(np.degrees(th)) % 360, 2), "flip": int(fl),
         "best": round(out["best"], 4), "gap": round(out["gap"], 4), "certified": out["certified"],
         "far_exact": round(u["far_exact"], 4), "far_ub": round(u["far_ub"], 4),
         "eps_unique": round(u["eps"], 4), "unique_PROVEN": bool(u["eps"] > 0),
         "far_complete": u["complete"], "far_s": round(u["far_secs"], 1)})


if __name__ == "__main__":
    {"k1": k1, "amb": amb, "tonsil": tonsil, "far": far}[sys.argv[1]]()
    print("DONE", flush=True)
