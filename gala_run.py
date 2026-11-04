"""GALA (Ding and Zeng 2026) on one real pair, as given, scored on the clicked landmarks.
Usage: python3 gala_run.py K [settings...]   (env gala_env, GPU)  Input: results_gala/pairK_in.npz (from gala_prep.py)
Output: results_gala/lm_given_pairK_gala.json, rows with landmark_err_median_um.
Rows per setting s and map: method gala_<s>_aff (GALA global affine only) and gala_<s>_full (affine + LDDMM), fit "plain" and "trimmed"
(rigid fit with reflection allowed to the moved A points, as for CAST/STalign) and landmark_err_own_median_um (landmarks sent through
GALA's own map, no rigid refit).
Settings: s1 = tutorial-5 settings (dx 30 um); s2 = dx 60 um; s3 = dx 30 um, other seed."""
import json, os, sys, time
import numpy as np, torch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from GALA.gala import gala
from GALA.utils import rasterize_channel, transform_points_source_to_target

k = int(sys.argv[1]); wanted = sys.argv[2:] or ["s1", "s2", "s3", "s4", "s5"]
out = f"results_gala/lm_given_pair{k}_gala.json"
NPTS = 20000
SET = {"s1": dict(dx=30.0, seed=42), "s2": dict(dx=60.0, seed=42), "s3": dict(dx=30.0, seed=7),
       "s4": dict(dx=30.0, seed=42, scale=True), "s5": dict(dx=60.0, seed=42, scale=True)}
# s4, s5: scale of both axes limited to 0.8..1.25 (the sections have the same scale; the rigid tools have scale 1), run once as a proper
# rotation and once with one axis mirrored, keep the run with the lower final GALA energy (no truth used)
d = np.load(f"results_gala/pair{k}_in.npz"); A, B, XA, XB, LA, LB = (d[x] for x in ("A", "B", "XA", "XB", "LA", "LB"))
floor = float(d["floor"])
if os.environ.get("SELF"):      # self test: B is A turned by 70 deg, mirrored, shifted (same expression); GALA must find it
    th = np.radians(70.0); Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([1.0, -1.0])
    mv = lambda P: (P - A.mean(0)) @ Rm.T + A.mean(0) + np.array([300.0, -200.0])
    B, XB, LB = mv(A), XA, mv(LA); floor = 0.0; out = f"results_gala/selftest_pair{k}_gala.json"
rng = np.random.default_rng(1)
ia = rng.choice(len(A), min(NPTS, len(A)), replace=False); ib = rng.choice(len(B), min(NPTS, len(B)), replace=False)


def procrustes(src, dst):
    cs, cd = src.mean(0), dst.mean(0); U, S, Vt = np.linalg.svd((src - cs).T @ (dst - cd)); R = Vt.T @ U.T
    return R, cd - R @ cs


def fit_rigid(src, dst, trimmed=False, keep=0.5, iters=10):      # same as baselines.fit_rigid (reflection allowed)
    R, tv = procrustes(src, dst)
    if trimmed:
        for _ in range(iters):
            r = np.linalg.norm(src @ R.T + tv - dst, axis=1); order = np.argsort(r); sel = order[: int(keep * len(src))]
            if len(sel) < 10: break
            R, tv = procrustes(src[sel], dst[sel])
    return R, tv


def top3(xy, X):                                                   # GALA's own gene choice (utils.top_p_genes), on the source
    Y = (X - X.mean(0)) / (X.std(0) + 1e-9)
    pr = KNeighborsRegressor(n_neighbors=10).fit(xy, Y).predict(xy)
    r2 = r2_score(Y, pr, multioutput="raw_values")
    r2[(X.std(0) < 1e-6) | ((X > 0).mean(0) < 0.05)] = -np.inf     # constant or almost empty genes break GALA's channel normalisation
    return np.argsort(-r2)[:3]


res = json.load(open(out)) if os.path.exists(out) else []
done = {r["method"] for r in res}
g = top3(A[ia], XA[ia])
for s in wanted:
    if f"gala_{s}_aff" in done: continue
    cfg = SET[s]; dx = cfg["dx"]; t0 = time.time()
    try:
        sI = np.concatenate([XA[ia][:, g], XA[ia].sum(1, keepdims=True)], 1)
        sJ = np.concatenate([XB[ib][:, g], XB[ib].sum(1, keepdims=True)], 1)
        XI, YI, I = rasterize_channel(A[ia, 0], A[ia, 1], sI, dx=dx)
        XJ, YJ, J = rasterize_channel(B[ib, 0], B[ib, 1], sJ, dx=dx)
        xI, xJ = [YI, XI], [YJ, XJ]
        rIy, rIx = YI.max() - YI.min(), XI.max() - XI.min(); rJy, rJx = YJ.max() - YJ.min(), XJ.max() - XJ.min()
        sx_range = rIx / rJx if rIx > rJx else rJx / rIx; sy_range = rIy / rJy if rIy > rJy else rJy / rIy
        # as tutorial 5, except the angle and the scale signs are free (default search space): reflection allowed
        gene_space = [{}, {}, {}, {}, {}]
        params = dict(num_generations=1000, num_iterations=5000, num_repeats=2, epV=10.0, a=300, sigmaR=2e5,
                      sigmaM=float(I.std()), sigmaB=0.1, random_seed=cfg["seed"])
        if cfg.get("scale"):
            cands = []
            for sgn in (1.0, -1.0):
                gs = [{}, {"low": 0.8, "high": 1.25} if sgn > 0 else {"low": -1.25, "high": -0.8}, {"low": 0.8, "high": 1.25}, {}, {}]
                rr = gala(xI, I, xJ, J, gene_space=gs, **params); cands.append((float(rr["E"][-1][0]), rr))
                print("variant", sgn, "final energy", cands[-1][0], flush=True)
            resu = min(cands, key=lambda c: c[0])[1]
        else:
            resu = gala(xI, I, xJ, J, gene_space=gene_space, **params)
        sec = round(time.time() - t0, 1)
        Am = resu["A"].cpu().numpy(); v, xv = resu["v"], resu["xv"]
        def maps(P):                                              # points in um (x, y) -> (affine only, affine + LDDMM), um (x, y)
            p = torch.as_tensor(P[:, [1, 0]].copy(), dtype=torch.float64, device=resu["v"].device)
            aff = (resu["A"][:2, :2] @ p.T + resu["A"][:2, -1][:, None]).T.cpu().numpy()[:, [1, 0]]
            full = transform_points_source_to_target(xv, v, resu["A"], p.clone()).cpu().numpy()[:, [1, 0]]
            return aff, full
        Pa, Pf = maps(A[ia]); La, Lf = maps(LA)
        det = float(np.linalg.det(Am[:2, :2]))
        for tag, P, L in (("aff", Pa, La), ("full", Pf, Lf)):
            own = float(np.median(np.linalg.norm(L - LB, axis=1)))
            for fit in ("plain", "trimmed"):
                R, tv = fit_rigid(A[ia], P, trimmed=(fit == "trimmed"))
                err = float(np.median(np.linalg.norm(LA @ R.T + tv - LB, axis=1)))
                res.append({"case": k, "method": f"gala_{s}_{tag}", "fit": fit, "angle": round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360), 2),
                            "flip": int(np.linalg.det(R) < 0), "landmark_err_median_um": round(err, 1),
                            "landmark_err_own_median_um": round(own, 1), "floor_um": round(floor, 1), "pair": k, "arm": "given",
                            "sec": sec, "gala_affine_det": round(det, 3), "gala_aff_params": [round(float(x), 3) for x in resu["Aparams"][-1]],
                            "dx_um": dx, "n_pts": [len(ia), len(ib)], "genes_idx": [int(x) for x in g]})
        json.dump(res, open(out, "w"), indent=1)
        print(json.dumps([{x: r[x] for x in ("method", "fit", "landmark_err_median_um", "landmark_err_own_median_um", "flip")} for r in res[-4:]]), "floor", floor, "sec", sec, flush=True)
    except Exception as e:
        import traceback
        res.append({"case": k, "method": f"gala_{s}_aff", "error": repr(e)[:300], "where": traceback.format_exc()[-600:]})
        json.dump(res, open(out, "w"), indent=1); print("ERR", res[-1], flush=True)
print("DONE", flush=True)
