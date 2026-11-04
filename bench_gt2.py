"""Known-answer benchmark, round 2 (reviewer fixes, 2026-09-30).

Why a second set: the first 24 cases came from the two Xenium breast slides that were also used to
pick the settings. Here the SAME case generator (bench_gt.py: disjoint cell halves, disjoint gene
halves, own clusters per side, pseudo-Visium spots on half the cases, same-direction crops, random
rotation + 50% flip + shift) runs on tissue never used for tuning:
  * mouse brain, Xenium, 3 coronal sections (10x "FF Mouse Brain MultiSection 1-3"): left and
    right halves are near mirror images, so this is a hard test of refusing;
  * human colon, Visium HD (GSE280318, P2 cancer, P3 and P5 normal), 8 um bins pooled to 16 um;
  * human brain DLPFC, Visium (12 sections), spots on both sides.
72 cases (24 per tissue). Settings are the fixed rule; nothing is tuned on these cases.

Modes
  make    build data/gt2/case_XX.npz (+ manifest.json). Baseline genes (300 most variable, same
          on both sides) are stored as AX/BX/genes so bench_baselines.py runs unchanged.
  ours    our method under the fixed rule, then a ZONE LADDER: the smallest zone in which the
          answer is proven unique (reviewers: state the zone). Works on any case folder.
  local   local search on the same score, 50 random starts, every start saved (for the
          refusal baseline: refuse when a far start scores almost as high).
Env: GT_DIR (default data/gt2), GT_OUT, GT_H (grid step, default 2), N_PROC, GT_TL (seconds).
Usage: python3 bench_gt2.py make | ours FIRST LAST | local FIRST LAST
"""
import json, os, sys, time
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from smoothalign import SmoothAlign, rot, FLIP
from smoothpar import solve_par, far_par, far_par_ext
from bench_domains import neighbourhood_domains
from bench_gt import UNIT, cluster, pseudo_visium, crop_pair, place, tre

D = os.environ.get("GT_DIR", "data/gt2")
H = float(os.environ.get("GT_H", "2"))
NP = int(os.environ.get("N_PROC", "14"))
TL = float(os.environ.get("GT_TL", "10800"))
# per-overlap score (smoothalign min_overlap); None = the original score
MINOV = float(os.environ["GT_MINOV"]) if os.environ.get("GT_MINOV") else None
# GT_FLIP=known: the mirror flip is given from outside (mounting protocol, slide image); search and
# certify with the flip fixed. Default: both flips are searched.
FLIP_KNOWN = os.environ.get("GT_FLIP") == "known"
# GT_TPAD: half-width of the shift window in units (default 300 = 2.8 mm; follow-up runs on very large sections)
TPAD = float(os.environ.get("GT_TPAD", "300"))
# zone ladder, from the easiest far set to the hardest: (rotation zone in deg, shift radius in
# units of 9.44 um; None = any shift, i.e. rotation-only check). A larger far set can only lower
# eps, so we stop at the first level that fails.
LADDER = [(20.0, None), (20.0, 60.0), (20.0, 30.0), (10.0, 20.0), (5.0, 10.0), (3.0, 5.0)]


# ---------------------------------------------------------------- data sources
def lognorm_hvg(X, n):
    """indices of the n most variable genes after library-size log-normalisation"""
    import scipy.sparse as sp
    lib = np.asarray(X.sum(1)).ravel()
    Xn = sp.diags(1e3 / np.maximum(lib, 1)) @ sp.csr_matrix(X)
    Xn.data = np.log1p(Xn.data)
    v = np.asarray(Xn.multiply(Xn).mean(0)).ravel() - np.asarray(Xn.mean(0)).ravel() ** 2
    return np.argsort(v)[::-1][:n]


def xenium(prefix):
    """any 10x Xenium output given its cell_feature_matrix.h5 + cells.csv.gz (prefix_...)"""
    return mbrain(None, prefix)


def mbrain(s, prefix=None):
    import pandas as pd, h5py, scipy.sparse as sp
    prefix = prefix or f"data/mbrain/S{s}"
    with h5py.File(f"{prefix}_cell_feature_matrix.h5") as f:
        g = f["matrix"]
        M = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=g["shape"][:]).T.tocsr()
        bc = g["barcodes"][:].astype(str); names = g["features/name"][:].astype(str)
        ftype = g["features/feature_type"][:].astype(str)
    keep_g = ftype == "Gene Expression"
    c = pd.read_csv(f"{prefix}_cells.csv.gz")
    c.index = c.cell_id.astype(str)
    X = M[:, keep_g]
    xy = c.loc[bc, ["x_centroid", "y_centroid"]].values.astype(float)
    ok = np.asarray(X.sum(1)).ravel() >= 10
    X, xy, names = X[ok], xy[ok], names[keep_g]
    if X.shape[1] > 2000:                                  # 5,000-gene panels: keep 2,000 most variable
        top = lognorm_hvg(X, 2000); X, names = X[:, top], names[top]
    return xy, np.asarray(X.todense(), np.float32), names


def colon(s, bin_um=16.0):
    import scipy.sparse as sp
    import run_colon as RC
    xy, X, genes = RC.load_hd(s)                       # 8 um bins, um coordinates
    key = np.floor(xy / bin_um).astype(np.int64)
    key = key[:, 0] * 1_000_000 + key[:, 1]
    u, inv = np.unique(key, return_inverse=True)
    P = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(u), len(inv)))
    Xa = (P @ X).tocsr()
    cnt = np.asarray(P.sum(1)).ravel()
    xya = (P @ xy) / cnt[:, None]
    ok = np.asarray(Xa.sum(1)).ravel() >= 20
    Xa, xya = Xa[ok], xya[ok]
    top = lognorm_hvg(Xa, 2000)
    return xya, np.asarray(Xa[:, top].todense(), np.float32), np.asarray(genes)[top]


def dlpfc(s):
    import check_dlpfc_baselines as C
    xy, _, X, genes = C.load(s)                         # xy in spot-grid units (100 um), centred
    top = lognorm_hvg(X, 2000)
    return xy * 100.0, X[:, top].astype(np.float32), np.asarray(genes)[top]


def sources():
    out = [("mouse brain", "mbrain", s) for s in (1, 2, 3)] * 8
    out = sorted(out, key=lambda z: z[2])
    col = sorted([("colon", "colon", s) for s in ("P2CRC", "P3NAT", "P5NAT")] * 8, key=lambda z: z[2])
    dl = sorted([("DLPFC", "dlpfc", s) for s in (151507, 151508, 151509, 151510, 151669, 151670,
                                                  151671, 151672, 151673, 151674, 151675, 151676)] * 2,
                key=lambda z: z[2])
    return out + col + dl


def sources3():
    """GT3 = held-out test set, made after the per-overlap score was fixed (2026-09-30): six
    datasets never used before; 8 cases each; seeds 9000 + i (GT1 used 1000 + i, GT2 5000 + i)."""
    ds = [("mouse brain (Prime 5K)", "brainprime"), ("mouse brain (V1)", "brainv1"),
          ("human kidney", "kidney"), ("human lymph node", "lymphnode"), ("human liver", "liver"),
          ("human heart", "heart")]
    return [(t, "xenium", f"data/xen3/{k}") for t, k in ds for _ in range(8)]


def sources4():
    """GT4 = second held-out set (review 5): six more public Xenium datasets never touched before, made
    after every part of the method was fixed (FROZEN_2026-10-05.md); 8 cases each; seeds 16000 + i."""
    ds = [("human tonsil", "tonsil"), ("human pancreas", "pancreas"), ("human lung cancer", "lung"),
          ("human skin", "skin"), ("mouse pup", "pup"), ("human brain", "hbrain")]
    return [(t, "xenium", f"data/xen4/{k}") for t, k in ds for _ in range(8)]


def sources_ov():
    """overlap sweep (2026-10-01): the five mouse brain datasets, 4 cases each, A's share on B fixed"""
    ds = [("mouse brain (MS1)", "mbrain", 1), ("mouse brain (MS2)", "mbrain", 2), ("mouse brain (MS3)", "mbrain", 3),
          ("mouse brain (Prime 5K)", "xenium", "data/xen3/brainprime"), ("mouse brain (V1)", "xenium", "data/xen3/brainv1")]
    return [d for d in ds for _ in range(4)]


OV_LEVEL = {"gt_ov100": 1.0, "gt_ov80": 0.8, "gt_ov60": 0.6}


def crop_fixed(xyA, xyB, rng, ov):
    """same-direction crops with a FIXED share of A on B: A keeps p <= q(0.9), B keeps p >= q(qB)
    with qB = 0.9 (1 - ov); ov = 1 keeps both whole"""
    if ov >= 1.0:
        return np.ones(len(xyA), bool), np.ones(len(xyB), bool)
    u = rng.normal(size=2); u /= np.linalg.norm(u)
    pA, pB = xyA @ u, xyB @ u
    return pA <= np.quantile(pA, 0.9), pB >= np.quantile(pB, 0.9 * (1 - ov))


# ---------------------------------------------------------------- make
def make():
    os.makedirs(D, exist_ok=True)
    cache, manifest = {}, []
    loaders = {"mbrain": mbrain, "colon": colon, "dlpfc": dlpfc, "xenium": xenium}
    third = os.path.basename(D) == "gt3"; fourth = os.path.basename(D) == "gt4"
    ovl = OV_LEVEL.get(os.path.basename(D))
    src = sources_ov() if ovl is not None else sources4() if fourth else sources3() if third else sources()
    for i, (tissue, kind_src, s) in enumerate(src):
        if (kind_src, s) not in cache:
            cache.clear()
            xy, X, genes = loaders[kind_src](s)
            cache[(kind_src, s)] = (xy, X, genes, lognorm_hvg(X, 300))
        xy, X, genes, bg = cache[(kind_src, s)]
        rng = np.random.default_rng((12000 if ovl is not None else 16000 if fourth else 9000 if third else 5000) + i)
        both_spots = kind_src == "dlpfc"
        kind = "spot" if (both_spots or i % 2) else "cell"
        perm = rng.permutation(len(xy)); hA, hB = perm[: len(xy) // 2], perm[len(xy) // 2:]
        g = rng.permutation(X.shape[1]); gA, gB = g[: len(g) // 2], g[len(g) // 2:]
        mA, mB = crop_fixed(xy[hA], xy[hB], rng, ovl) if ovl is not None else crop_pair(xy[hA], xy[hB], rng)
        Bxy, BX = xy[hB][mB], X[hB][mB]
        if both_spots:
            lbB = cluster(BX[:, gB], 12, i)
        else:
            lbB = neighbourhood_domains(Bxy / UNIT, cluster(BX[:, gB], 12, i), K=6, seed=i)
        Axy, AX = xy[hA][mA], X[hA][mA]
        if both_spots:
            laA = cluster(AX[:, gA], 12, i + 7)
        elif kind == "spot":
            Axy, AX = pseudo_visium(Axy, AX, rng)
            laA = cluster(AX[:, gA], 12, i + 7)
        else:
            laA = neighbourhood_domains(Axy / UNIT, cluster(AX[:, gA], 12, i + 7), K=6, seed=i + 7)
        th0 = rng.uniform(0, 2 * np.pi); fl0 = int(rng.integers(2))
        s0 = rng.uniform(-150, 150, 2) * UNIT
        M0 = rot(th0) @ (FLIP if fl0 else np.eye(2))
        cA = Axy.mean(0)
        Aobs = (Axy - cA) @ M0.T + cA + s0
        ov = float(np.mean(cKDTree(Bxy).query(Axy)[0] < 30))
        np.savez_compressed(f"{D}/case_{i:02d}.npz", Aobs=Aobs, Atrue=Axy, la=laA, B=Bxy, lb=lbB,
                            AX=AX[:, bg], BX=BX[:, bg], genes=np.asarray(genes)[bg],
                            th0=th0, fl0=fl0, s0=s0)
        m = {"case": i, "tissue": tissue, "section": str(s), "kind": kind, "nA": int(len(Axy)),
             "nB": int(len(Bxy)), "applied_deg": round(float(np.degrees(th0)), 2),
             "applied_flip": fl0, "overlap": round(ov, 3)}
        manifest.append(m); print(json.dumps(m), flush=True)
        json.dump(manifest, open(f"{D}/manifest.json", "w"), indent=1)


# ---------------------------------------------------------------- ours + zone ladder
def truth_deg(z):
    W = np.linalg.inv(rot(float(z["th0"])) @ (FLIP if int(z["fl0"]) else np.eye(2)))
    return float(np.degrees(np.arctan2(W[1, 0], W[0, 0])) % 360)


def zone_move_um(Aobs, deg, radius):
    """median distance a point can move inside the zone (rotation up to deg, shift up to radius)"""
    r = np.linalg.norm(Aobs - Aobs.mean(0), axis=1)
    return float(np.median(2 * r * np.sin(np.radians(deg) / 2)) + (radius or 0) * UNIT)


def relabel(z, kind, spec):
    """GT_RELABEL='seed:K:Kreg' (reviewers: dependence on the clustering). Re-make both label sets from the
    stored expression (the 300 stored genes, split into two disjoint halves so the sides still share no
    gene): k-means with K clusters and a new seed, then, for cell sides, Kreg tissue regions."""
    seed, K, Kreg = (int(v) for v in spec.split(":"))
    rng = np.random.default_rng(777 + seed)
    g = rng.permutation(z["AX"].shape[1]); gA, gB = g[: len(g) // 2], g[len(g) // 2:]
    ca = cluster(np.asarray(z["AX"], float)[:, gA], K, 100 + seed)
    cb = cluster(np.asarray(z["BX"], float)[:, gB], K, 200 + seed)
    la = ca if kind == "spot" else neighbourhood_domains(z["Aobs"] / UNIT, ca, K=Kreg, seed=100 + seed)
    lb = neighbourhood_domains(z["B"] / UNIT, cb, K=Kreg, seed=200 + seed)
    return la, lb


def ours(first, last):
    out = os.environ.get("GT_OUT", f"results_{os.path.basename(D)}_ours_h{H:g}.json")
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {r["case"] for r in res}
    for i in range(first, last + 1):
        f = f"{D}/case_{i:02d}.npz"
        if i in done or not os.path.exists(f):
            continue
        z = np.load(f)
        A, B = z["Aobs"] / UNIT, z["B"] / UNIT
        la_, lb_ = z["la"], z["lb"]
        if os.environ.get("GT_RELABEL"):
            kind = {m["case"]: m["kind"] for m in json.load(open(f"{D}/manifest.json"))}[i]
            la_, lb_ = relabel(z, kind, os.environ["GT_RELABEL"])
        t0 = time.time()
        al = SmoothAlign(A, la_, B, lb_, h=H, sigma=20.0, tpad=TPAD, contrast=True,
                         min_overlap=MINOV)
        flips = (int(z["fl0"]),) if FLIP_KNOWN else (0, 1)
        o = solve_par(al, n_proc=NP, time_limit=TL, flips=flips)
        th, t, fl = o["transform"]
        P = place(z["Aobs"], th, fl, np.asarray(t) * UNIT, z["B"].mean(0))
        # is the TRUE placement inside each certified zone? (truth frame: see bench_gt.make)
        Wt = np.linalg.inv(rot(float(z["th0"])) @ (FLIP if int(z["fl0"]) else np.eye(2)))
        t_true = (z["Atrue"].mean(0) - z["B"].mean(0)) / UNIT
        r = {"case": i, "h": H, "tpad": TPAD, "min_overlap": MINOV, "flip_known": FLIP_KNOWN, "relabel": os.environ.get("GT_RELABEL"), "tre_um": round(tre(P, z["Atrue"]), 1),
             "true_shift_units": [round(float(v), 3) for v in t_true],
             "angle": round(float(np.degrees(th)) % 360, 4), "flip": int(fl),
             "shift_units": [round(float(v), 3) for v in t],
             "truth": [round(truth_deg(z), 2), int(z["fl0"])], "flip_ok": int(fl == int(z["fl0"])),
             "best": round(o["best"], 4), "gap": round(o["gap"], 4), "certified_search": o["certified"],
             "solve_s": round(o["secs"], 1), "ladder": []}
        if o["certified"]:
            for deg_z, rad in LADDER:
                if rad is None:
                    u = far_par(al, o["best"], th, fl, exclude_deg=deg_z, n_proc=NP, time_limit=TL / 2, flips=flips)
                else:
                    u = far_par_ext(al, o["best"], th, fl, t, exclude_deg=deg_z, shift_radius=rad,
                                    n_proc=NP, time_limit=TL / 2, flips=flips)
                step = {"zone_deg": deg_z, "zone_shift_units": rad, "eps": round(u["eps"], 4),
                        "complete": u["complete"], "far_s": round(u["far_secs"], 1),
                        "far_exact": round(u["far_exact"], 4),
                        "zone_move_um": round(zone_move_um(z["Aobs"], deg_z, rad), 1)}
                if rad is None and u.get("best_far_placement"):
                    d2, f2, t2 = u["best_far_placement"]
                    step["rival"] = [round(float(d2), 2), int(f2)]
                    step["rival_tre_um"] = round(tre(place(z["Aobs"], np.radians(d2), f2,
                                                           np.asarray(t2) * UNIT, z["B"].mean(0)), z["Atrue"]), 1)
                    step["rival_is_mirror"] = int(f2 != int(fl))
                r["ladder"].append(step)
                if u["eps"] <= 0:
                    break
        ok = [s for s in r["ladder"] if s["eps"] > 0]
        r["certified"] = bool(ok)
        r["finest_zone"] = [ok[-1]["zone_deg"], ok[-1]["zone_shift_units"]] if ok else None
        r["finest_zone_move_um"] = ok[-1]["zone_move_um"] if ok else None
        r["total_s"] = round(time.time() - t0, 1)
        res.append(r); json.dump(res, open(out, "w"), indent=1); print(json.dumps(r), flush=True)


def neg(first, last):
    """Negative controls (review 5): pairs with NO true alignment. A comes from case i, B from the case
    8 further on in the same set (the next tissue; sets have 8 cases per dataset). Full default method
    (flip searched): search, zone ladder, and the mirror self-check quantities. A sound "align or
    refuse" should refuse here, or at least score far lower than on true pairs."""
    out = os.environ.get("GT_OUT", f"results_{os.path.basename(D)}_neg.json")
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {r["case"] for r in res}
    man = {m["case"]: m for m in json.load(open(f"{D}/manifest.json"))}
    for i in range(first, last + 1):
        j = (i + 8) % len(man)
        if i in done or not os.path.exists(f"{D}/case_{i:02d}.npz"):
            continue
        za, zb = np.load(f"{D}/case_{i:02d}.npz"), np.load(f"{D}/case_{j:02d}.npz")
        A, B = za["Aobs"] / UNIT, zb["B"] / UNIT
        t0 = time.time()
        al = SmoothAlign(A, za["la"], B, zb["lb"], h=H, sigma=20.0, tpad=300.0, contrast=True)
        o = solve_par(al, n_proc=NP, time_limit=TL)
        th, t, fl = o["transform"]
        r = {"case": i, "A_tissue": man[i]["tissue"], "B_case": j, "B_tissue": man[j]["tissue"],
             "angle": round(float(np.degrees(th)) % 360, 4), "flip": int(fl),
             "shift_units": [round(float(v), 3) for v in t], "best": round(o["best"], 4),
             "gap": round(o["gap"], 4), "solve_s": round(o["secs"], 1), "ladder": []}
        if o["certified"]:
            for deg_z, rad in LADDER:
                if rad is None:
                    u = far_par(al, o["best"], th, fl, exclude_deg=deg_z, n_proc=NP, time_limit=TL / 2)
                else:
                    u = far_par_ext(al, o["best"], th, fl, t, exclude_deg=deg_z, shift_radius=rad,
                                    n_proc=NP, time_limit=TL / 2)
                r["ladder"].append({"zone_deg": deg_z, "zone_shift_units": rad, "eps": round(u["eps"], 4),
                                    "complete": u["complete"]})
                if u["eps"] <= 0:
                    break
        ok = [s_ for s_ in r["ladder"] if s_["eps"] > 0]
        r["certified"] = bool(ok)
        r["finest_zone"] = [ok[-1]["zone_deg"], ok[-1]["zone_shift_units"]] if ok else None
        if ok:                                             # mirror self-check quantities
            Mk = tissue_mask(al, B)
            o2 = solve_par(al, n_proc=NP, time_limit=TL, flips=(1 - fl,))
            th2, t2, fl2 = o2["transform"]
            r.update({"S_answer": r["best"], "O_answer": round(mask_mean(al, Mk, th, fl, t), 4),
                      "S_other": round(o2["best"], 4), "O_other": round(mask_mean(al, Mk, th2, fl2, t2), 4)})
        r["total_s"] = round(time.time() - t0, 1)
        res.append(r); json.dump(res, open(out, "w"), indent=1); print(json.dumps(r), flush=True)


def tight(first, last):
    """Decision with the TIGHT bound (review 5): the far check with rotation AND shift splitting
    (smoothpar.far_split), so the bound slack shrinks from the grid term (~0.015) towards 0.001. Zone
    (20 degrees, any shift) around the stored answer (GT_COARSE). eps_tight > 0 = certified under the
    tight bound; the run stops as soon as that is proven, or at the time limit (then: not proven)."""
    import glob
    from smoothpar import far_split
    coarse = {}
    for f in glob.glob(os.environ["GT_COARSE"]):
        for r in json.load(open(f)):
            coarse[r["case"]] = r
    out = os.environ["GT_OUT"]
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {r["case"] for r in res}
    for i in range(first, last + 1):
        c = coarse.get(i)
        if i in done or c is None:
            continue
        z = np.load(f"{D}/case_{i:02d}.npz")
        al = SmoothAlign(z["Aobs"] / UNIT, z["la"], z["B"] / UNIT, z["lb"], h=H, sigma=20.0, tpad=300.0, contrast=True)
        th, fl = np.radians(c["angle"]), c["flip"]
        t = np.asarray(c["shift_units"]) if "shift_units" in c else al.rotation_eval(th, fl, 1e-6)[1]
        u = far_split(al, c["best"], th, fl, t, exclude_deg=20.0, shift_radius=None, n_proc=NP, time_limit=TL,
                      min_cell=al.h / np.sqrt(2) / 8, stop_below=c["best"])
        r = {"case": i, "best": c["best"], "tre_um": c["tre_um"], "certified_loose": c["certified"],
             "eps_loose": c["ladder"][0]["eps"] if c["ladder"] else None,
             "eps_tight": round(u["eps"], 5), "far_exact": round(u["far_exact"], 5), "complete": u["complete"],
             "far_evals": u["far_evals"], "far_s": round(u["far_secs"], 1)}
        res.append(r); json.dump(res, open(out, "w"), indent=1); print(json.dumps(r), flush=True)


# ---------------------------------------------------------------- local search (refusal baseline)
def local(first, last, n_starts=None, maxfev=300):
    n_starts = int(os.environ.get("LOCAL_STARTS", "50")) if n_starts is None else n_starts
    from scipy.optimize import minimize
    out = os.environ.get("GT_OUT", f"results_{os.path.basename(D)}_local.json")
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {r["case"] for r in res}
    for i in range(first, last + 1):
        f = f"{D}/case_{i:02d}.npz"
        if i in done or not os.path.exists(f):
            continue
        z = np.load(f)
        A, B = z["Aobs"] / UNIT, z["B"] / UNIT
        al = SmoothAlign(A, z["la"], B, z["lb"], h=H, sigma=20.0, tpad=300.0, contrast=True, workers=1)
        rng = np.random.default_rng(1000 + i); runs = []; t0 = time.time()
        for _ in range(n_starts):
            fl = int(rng.integers(2))
            x0 = np.array([rng.uniform(0, 360), *rng.uniform(-300, 300, 2)])
            fn = lambda x: -al.exact_score(np.radians(x[0]), fl, np.clip(x[1:], -300, 300))
            o = minimize(fn, x0, method="Nelder-Mead", options={
                "initial_simplex": np.array([x0, x0 + [15, 0, 0], x0 + [0, 40, 0], x0 + [0, 0, 40]]),
                "maxfev": maxfev, "xatol": 0.5, "fatol": 1e-4})
            d, tt = float(o.x[0] % 360), np.clip(o.x[1:], -300, 300)
            P = place(z["Aobs"], np.radians(d), fl, tt * UNIT, z["B"].mean(0))
            runs.append({"score": round(-o.fun, 5), "angle": round(d, 2), "flip": fl,
                         "shift": [round(float(v), 2) for v in tt], "tre_um": round(tre(P, z["Atrue"]), 1)})
        r = {"case": i, "h": H, "starts": runs, "sec": round(time.time() - t0, 1)}
        res.append(r); json.dump(res, open(out, "w"), indent=1)
        best = max(runs, key=lambda q: q["score"])
        print(json.dumps({"case": i, "best": best, "sec": r["sec"]}), flush=True)


def extend(first, last):
    """answers proven only at the rotation level (zone (20, any)): try far-shift radii 150 and 100
    units so the fine stage gets a shift bound. Reads GT_COARSE, writes GT_OUT."""
    import glob
    coarse = {}
    for f in glob.glob(os.environ["GT_COARSE"]):
        for r in json.load(open(f)):
            coarse[r["case"]] = r
    out = os.environ["GT_OUT"]
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {r["case"] for r in res}
    for i in range(first, last + 1):
        c = coarse.get(i)
        if i in done or c is None or not c["certified"] or c["finest_zone"][1] is not None:
            continue
        z = np.load(f"{D}/case_{i:02d}.npz")
        A, B = z["Aobs"] / UNIT, z["B"] / UNIT
        al = SmoothAlign(A, z["la"], B, z["lb"], h=H, sigma=20.0, tpad=300.0, contrast=True)
        th, fl = np.radians(c["angle"]), c["flip"]
        t = np.asarray(c["shift_units"]) if "shift_units" in c else al.rotation_eval(th, fl, 1e-6)[1]
        r = dict(c); r["ladder"] = list(c["ladder"])
        for rad in (150.0, 100.0):
            u = far_par_ext(al, c["best"], th, fl, t, exclude_deg=20.0, shift_radius=rad, n_proc=NP,
                            time_limit=TL / 2)
            r["ladder"].append({"zone_deg": 20.0, "zone_shift_units": rad, "eps": round(u["eps"], 4),
                                "complete": u["complete"], "far_s": round(u["far_secs"], 1),
                                "zone_move_um": round(zone_move_um(z["Aobs"], 20.0, rad), 1), "extended": True})
            if u["eps"] <= 0:
                break
            r["finest_zone"] = [20.0, rad]; r["finest_zone_move_um"] = r["ladder"][-1]["zone_move_um"]
        r["shift_units"] = [round(float(v), 3) for v in t]
        res.append(r); json.dump(res, open(out, "w"), indent=1); print(json.dumps(r), flush=True)


def fine(first, last):
    """stage 2 of the certificate inside each coarse zone (finecert.py). Needs the coarse results:
    GT_COARSE = glob of result files from the 'ours' mode."""
    import glob
    from finecert import fine_stage
    coarse = {}
    for f in glob.glob(os.environ["GT_COARSE"]):
        for r in json.load(open(f)):
            coarse[r["case"]] = r
    out = os.environ.get("GT_OUT", f"results_{os.path.basename(D)}_fine.json")
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {r["case"] for r in res}
    for i in range(first, last + 1):
        c = coarse.get(i)
        if i in done or c is None or not c["certified"] or c["finest_zone"][1] is None:
            continue
        z = np.load(f"{D}/case_{i:02d}.npz")
        A, B = z["Aobs"] / UNIT, z["B"] / UNIT
        # coarse answer: recompute its shift (not stored) with one exact evaluation on the coarse
        # score at the stored angle and flip
        al1 = SmoothAlign(A, z["la"], B, z["lb"], h=H, sigma=20.0, tpad=300.0, contrast=True, workers=NP)
        th1, fl1 = np.radians(c["angle"]), c["flip"]
        t1 = (np.asarray(c["shift_units"], float) if "shift_units" in c
              else al1.rotation_eval(th1, fl1, 1e-6)[1])
        del al1
        a1, r1 = c["finest_zone"]
        t0 = time.time()
        o, (th2, t2) = fine_stage(A, z["la"], B, z["lb"], th1, fl1, t1, a1, r1, n_proc=NP, time_limit=TL)
        P = place(z["Aobs"], th2, fl1, np.asarray(t2) * UNIT, z["B"].mean(0))
        o.update({"case": i, "coarse_zone": [a1, r1], "coarse_tre_um": c["tre_um"],
                  "tre2_um": round(tre(P, z["Atrue"]), 1),
                  "fine_zone_move_um": (min(q["zone_move_um"] for q in o["ladder2"] if q["eps"] > 0)
                                        if o["fine_zone"] else None),
                  "total_s": round(time.time() - t0, 1)})
        res.append(o); json.dump(res, open(out, "w"), indent=1); print(json.dumps(o), flush=True)


def count(first, last):
    """search effort (reviewers): ranges visited by branch-and-bound against a full grid at leaf width"""
    out = os.environ.get("GT_OUT", f"results_{os.path.basename(D)}_count.json")
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {r["case"] for r in res}
    for i in range(first, last + 1):
        f = f"{D}/case_{i:02d}.npz"
        if i in done or not os.path.exists(f):
            continue
        z = np.load(f)
        al = SmoothAlign(z["Aobs"] / UNIT, z["la"], z["B"] / UNIT, z["lb"], h=H, sigma=20.0, tpad=300.0, contrast=True)
        o = solve_par(al, n_proc=NP, time_limit=TL)
        leaf = (al.h / np.sqrt(2)) / (2 * al.rmax)                 # half-width at which splitting stops
        r = {"case": i, "evals": o["evals"], "solve_s": round(o["secs"], 1), "gap": round(o["gap"], 4),
             "best": round(o["best"], 4), "leaf_halfwidth_deg": round(float(np.degrees(leaf)), 4),
             "full_grid_at_leaf": int(2 * np.ceil(np.pi / leaf)), "nA": int(al.n), "grid_nodes": int(np.prod(al.shape)),
             "KA": int(al.KA), "KB": int(al.KB)}
        res.append(r); json.dump(res, open(out, "w"), indent=1); print(json.dumps(r), flush=True)


def grid(first, last, step_deg=1.0):
    """Baseline asked by reviewers: a plain angle grid (every step_deg, both flips) with FFT over all
    shifts, on the SAME score, no bound and no proof. Saves the best score and shift per angle so that
    coarser grids can be read off afterwards."""
    from smoothpar import _pool, _eval
    out = os.environ.get("GT_OUT", f"results_{os.path.basename(D)}_grid.json")
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {r["case"] for r in res}
    for i in range(first, last + 1):
        f = f"{D}/case_{i:02d}.npz"
        if i in done or not os.path.exists(f):
            continue
        z = np.load(f)
        A, B = z["Aobs"] / UNIT, z["B"] / UNIT
        t0 = time.time()
        al = SmoothAlign(A, z["la"], B, z["lb"], h=H, sigma=20.0, tpad=300.0, contrast=True)
        degs = np.arange(0, 360, step_deg)
        pool = _pool(al, NP)
        rows = pool.map(_eval, [(np.radians(d), fl, 1e-6) for fl in (0, 1) for d in degs])
        pool.close(); pool.join()
        per = []
        for th, fl, d, sc, t, ub in rows:
            P = place(z["Aobs"], th, fl, np.asarray(t) * UNIT, z["B"].mean(0))
            per.append([round(float(np.degrees(th)), 2), int(fl), round(float(sc), 5),
                        round(float(t[0]), 2), round(float(t[1]), 2), round(tre(P, z["Atrue"]), 1)])
        b = max(per, key=lambda q: q[2])
        r = {"case": i, "h": H, "step_deg": step_deg, "best": b[2], "angle": b[0], "flip": b[1],
             "shift_units": [b[3], b[4]], "tre_um": b[5], "per_angle": per, "total_s": round(time.time() - t0, 1),
             "evals": len(per)}
        res.append(r); json.dump(res, open(out, "w")); print(json.dumps({k: v for k, v in r.items() if k != "per_angle"}), flush=True)


def tissue_mask(al, B):
    """B's tissue mask M = S / (S + c) on al's grid (same construction as in SmoothAlign)"""
    from scipy.ndimage import gaussian_filter
    S = np.zeros(al.shape)
    ij = np.floor((B - al.lo) / al.h + 0.5).astype(int)
    keep = (ij >= 0).all(1) & (ij[:, 0] < al.shape[0]) & (ij[:, 1] < al.shape[1])
    np.add.at(S, (ij[keep, 0], ij[keep, 1]), 1.0)
    tot = gaussian_filter(S, al.sigma / al.h, mode="constant")
    return tot / (tot + al.c)


def mask_mean(al, Mk, th, fl, t):
    P = al._points(th, fl) + np.asarray(t, float) - al.t0
    x = (P - al.lo) / al.h; i0 = np.floor(x).astype(int); f = x - i0
    v = sum(w * Mk[i0[:, 0] + di, i0[:, 1] + dj] for di, dj, w in (
        (0, 0, (1 - f[:, 0]) * (1 - f[:, 1])), (1, 0, f[:, 0] * (1 - f[:, 1])),
        (0, 1, (1 - f[:, 0]) * f[:, 1]), (1, 1, f[:, 0] * f[:, 1])))
    return float(v.mean())


def flipcheck(first, last):
    """Mirror self-check (reviewers): for the answer (flip f) find the best placement with the OTHER flip
    and compare. Saves score S and overlap O (share of A on B's tissue) for both, so a rule such as
    'refuse the flip if the other flip's agreement per overlapping point is within a factor of the
    answer's' can be tested. Reads GT_COARSE (flip-searched results)."""
    import glob
    coarse = {}
    for f in glob.glob(os.environ["GT_COARSE"]):
        for r in json.load(open(f)):
            coarse[r["case"]] = r
    out = os.environ["GT_OUT"]
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {r["case"] for r in res}
    for i in range(first, last + 1):
        c = coarse.get(i)
        if i in done or c is None:
            continue
        z = np.load(f"{D}/case_{i:02d}.npz")
        A, B = z["Aobs"] / UNIT, z["B"] / UNIT
        al = SmoothAlign(A, z["la"], B, z["lb"], h=H, sigma=20.0, tpad=TPAD, contrast=True)
        Mk = tissue_mask(al, B)
        th, fl = np.radians(c["angle"]), c["flip"]
        t = np.asarray(c["shift_units"]) if "shift_units" in c else al.rotation_eval(th, fl, 1e-6)[1]
        o = solve_par(al, n_proc=NP, time_limit=TL, flips=(1 - fl,))
        th2, t2, fl2 = o["transform"]
        P2 = place(z["Aobs"], th2, fl2, np.asarray(t2) * UNIT, z["B"].mean(0))
        r = {"case": i, "answer_flip": int(fl), "true_flip": int(z["fl0"]), "answer_tre_um": c["tre_um"],
             "certified": c["certified"], "S_answer": c["best"], "O_answer": round(mask_mean(al, Mk, th, fl, t), 4),
             "S_other": round(o["best"], 4), "O_other": round(mask_mean(al, Mk, th2, fl2, t2), 4),
             "other_tre_um": round(tre(P2, z["Atrue"]), 1), "other_gap": round(o["gap"], 4),
             "other_s": round(o["secs"], 1)}
        res.append(r); json.dump(res, open(out, "w"), indent=1); print(json.dumps(r), flush=True)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "make":
        make()
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        {"ours": ours, "local": local, "fine": fine, "extend": extend, "grid": grid, "flipcheck": flipcheck, "count": count,
         "neg": neg, "tight": tight}[m](a, b)
    print("DONE", flush=True)
