"""Plain rigid image registration of the two tissue photos of each real pair, scored on the clicked landmarks like the
other tools. Usage: python3 runs_baselines/image_reg.py PAIR [PAIR ...]   (run from the repository root)

Method (standard, no tuning per pair):
 1. Each photo (the one the landmarks were clicked on) is turned into a tissue image (H&E darkness above the
    background level, or a binary tissue mask) and resampled into the analysis frame of its section (microns),
    using the same photo-to-micron maps as landmarks/landmarks_to_um.py, at one pixel size for both photos.
 2. Global search: every rotation in 1 degree steps and both mirror flips; for each, the best shift over all integer
    shifts by FFT, scored by overlap-normalised cross-correlation (NCC, Padfield 2012) or by phase correlation
    (Reddy and Chatterji 1996). The top 5 distinct angles are refined in 0.1 degree steps with a sub-pixel peak.
 3. Optional: the top 3 refined answers are refined again by mutual information (SimpleITK, Mattes MI, rigid),
    and the one with the best MI is kept.
 4. The answer P(a) = R(deg) F^flip a + t maps A's analysis frame to B's, so it is scored directly with
    landmark_common.lm_error (centres set to 0, t absolute), against floor + 100 um.
Settings: image {intensity, mask} x pixel size {30, 50} um x score {ncc, phase} x {no MI, MI}. Every setting is saved;
as for the other tools, the summary reports the best setting chosen with the truth and the median over settings.
Output: results/placement/lm_given_pair{k}_imagereg.json (one row per setting), and sanity-check PNGs in WORK.
Env: PHOTOS (folder with calibration.json and pairN_*/left_A.png, right_B.png; default <repo>/../../landmark_work),
     WORK (folder for overlays; default <repo>/../../image_baseline_work)."""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import glob, json, time
import numpy as np
from PIL import Image
from scipy import fft as sfft
from scipy.ndimage import gaussian_filter, map_coordinates, affine_transform, binary_fill_holes, binary_opening, label
from landmark_common import UNIT, NAMES, landmarks, rigid_fit, lm_error, rot, FLIP

ROOT = repo_paths.ROOT
PHOTOS = os.environ.get("PHOTOS", os.path.normpath(f"{ROOT}/../../landmark_work"))
WORK = os.environ.get("WORK", os.path.normpath(f"{ROOT}/../../image_baseline_work"))
CAL = json.load(open(f"{PHOTOS}/calibration.json")); PAIRS = json.load(open(f"{ROOT}/landmarks/pairs.json"))
PIX_UM = (30.0, 50.0)
WORKERS = int(os.environ.get("WORKERS", "2"))


# ------------------------------------------------------------------ photo pixels <-> analysis frame (microns)
def to_um(key, xy):
    """same map as landmarks/landmarks_to_um.py, for an (N,2) array of photo pixels"""
    c = CAL[key]; xy = np.asarray(xy, float)
    if "xenium_px_from_he_px" in c:
        M = np.array(c["xenium_px_from_he_px"]); full = xy * [c["full_per_small_x"], c["full_per_small_y"]]
        return (full @ M[:2, :2].T + M[:2, 2]) * c["um_per_xenium_px"]
    return xy * (c["fullres_per_picture_px"] * c["um_per_fullres_px"])


def to_px(key, uv):
    c = CAL[key]; uv = np.asarray(uv, float)
    if "xenium_px_from_he_px" in c:
        M = np.array(c["xenium_px_from_he_px"]); full = (uv / c["um_per_xenium_px"] - M[:2, 2]) @ np.linalg.inv(M[:2, :2]).T
        return full / [c["full_per_small_x"], c["full_per_small_y"]]
    return uv / (c["fullres_per_picture_px"] * c["um_per_fullres_px"])


def um_per_photo_px(key):
    p = to_um(key, [[0, 0], [1, 0], [0, 1]]); return float(np.sqrt(abs(np.linalg.det(np.stack([p[1] - p[0], p[2] - p[0]])))))


# ------------------------------------------------------------------ tissue images
def tissue_maps(rgb):
    """darkness above the background level (0 on glass), and a binary tissue mask (Otsu on colour saturation)"""
    x = rgb.astype(float) / 255.0; gray = x.mean(2)
    edge = np.concatenate([gray[:10].ravel(), gray[-10:].ravel(), gray[:, :10].ravel(), gray[:, -10:].ravel()])
    dark = np.clip(np.median(edge) - gray, 0, None); dark /= max(np.percentile(dark, 99.5), 1e-6)
    sat = x.max(2) - x.min(2); sat = gaussian_filter(sat, 1.0)
    h, e = np.histogram(sat, 256); cen = (e[:-1] + e[1:]) / 2; w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * cen) / np.maximum(w0, 1); m1 = (np.sum(h * cen) - np.cumsum(h * cen)) / np.maximum(w1, 1)
    thr = cen[np.argmax(w0 * w1 * (m0 - m1) ** 2)]                                   # Otsu threshold
    m = binary_opening(sat > thr, iterations=2); lab, n = label(m)
    if n:
        sz = np.bincount(lab.ravel()); sz[0] = 0; m = np.isin(lab, np.nonzero(sz >= 0.002 * m.size)[0])
    return np.clip(dark, 0, 1), binary_fill_holes(m).astype(float)


def render(key, img, s):
    """resample a photo-pixel image into the analysis frame at s um per pixel. Pixel (r, c) <-> um (x0 + c s, y0 + r s)"""
    h, w = img.shape; cor = to_um(key, [[0, 0], [w, 0], [0, h], [w, h]]); lo, hi = cor.min(0), cor.max(0)
    xs, ys = np.arange(lo[0], hi[0], s), np.arange(lo[1], hi[1], s); X, Y = np.meshgrid(xs, ys)
    sig = 0.5 * s / um_per_photo_px(key)
    src = gaussian_filter(img, sig) if sig > 0.5 else img
    p = to_px(key, np.stack([X.ravel(), Y.ravel()], 1)) - 0.5                       # pixel centres
    val = map_coordinates(src, [p[:, 1], p[:, 0]], order=1, cval=0.0).reshape(X.shape)
    sup = map_coordinates(np.ones_like(img), [p[:, 1], p[:, 0]], order=0, cval=0.0).reshape(X.shape)
    return val * sup, sup, lo.copy()


def warp(img, origin, s, M, c, out_shape, out_origin):
    """value at output um point b of the image moved by b = M a + c (linear interpolation, 0 outside)"""
    Mi = np.linalg.inv(M); off = (Mi @ (np.asarray(out_origin, float) - c) - origin) / s      # source pixel (x, y) = Mi p + off
    P = np.array([[0, 1], [1, 0]])
    return affine_transform(img, P @ Mi @ P, offset=P @ off, output_shape=tuple(out_shape), order=1, cval=0.0)


# ------------------------------------------------------------------ FFT search over shifts
class Fixed:
    def __init__(s, f, mf, size):
        s.size, s.f, s.mf = size, f, mf
        s.F, s.MF, s.FF = (sfft.rfft2(z.astype(np.float32), size, workers=WORKERS) for z in (f, mf, f * f))

    def corr(s, X, G):                                       # C[k] = sum_p x[p] g[p + k]
        return sfft.irfft2(np.conj(X) * G, s.size, workers=WORKERS)

    def scores(s, g, mg, kind, min_frac):
        G = sfft.rfft2(g.astype(np.float32), s.size, workers=WORKERS)
        if kind == "phase":
            Rr = np.conj(s.F) * G; return sfft.irfft2(Rr / np.maximum(np.abs(Rr), 1e-9), s.size, workers=WORKERS)
        MG, GG = (sfft.rfft2(z.astype(np.float32), s.size, workers=WORKERS) for z in (mg, g * g))
        N = s.corr(s.MF, MG); Sf = s.corr(s.F, MG); Sg = s.corr(s.MF, G); Sff = s.corr(s.FF, MG); Sgg = s.corr(s.MF, GG); Sfg = s.corr(s.F, G)
        N, Sf, Sg, Sff, Sgg, Sfg = (z.astype(np.float64) for z in (N, Sf, Sg, Sff, Sgg, Sfg))
        Nc = np.maximum(N, 1.0); num = Sfg - Sf * Sg / Nc; den = (Sff - Sf ** 2 / Nc) * (Sgg - Sg ** 2 / Nc)
        out = num / np.sqrt(np.maximum(den, 1e-12))
        out[(N < min_frac) | (den <= 1e-9)] = -1.0
        return out


def peak(C, size, D):
    """best integer shift k (x, y), sub-pixel by a parabola on each axis, and the peak value. Valid k: -(len B - 1) .. D - 1"""
    i = np.unravel_index(np.argmax(C), C.shape); v = C[i]; sub = []
    for ax in (1, 0):
        j = list(i); a = C[tuple(j)]; j[ax] = (i[ax] - 1) % C.shape[ax]; l = C[tuple(j)]; j[ax] = (i[ax] + 1) % C.shape[ax]; r = C[tuple(j)]
        d = l - 2 * a + r; sub.append(0.5 * (l - r) / d if d < 0 else 0.0)
    k = np.array([i[1], i[0]], float)
    k = np.where(k > D - 1, k - np.array(size[::-1]), k) + np.clip(sub, -0.5, 0.5)
    return k, float(v)


def eval_pose(fx, A, mA, oA, B, oB, s, deg, fl, kind, min_frac):
    """rotate/flip A about its grid centre into a square canvas; FFT over all shifts; return (score, M, c)"""
    M = rot(np.radians(deg)) @ (FLIP if fl else np.eye(2))
    h, w = A.shape; cA = oA + s * np.array([w - 1, h - 1]) / 2; D = int(np.ceil(np.hypot(h, w))) + 2
    q = M @ cA; oC = q - s * (D - 1) / 2                     # canvas um origin; canvas holds A moved by M (c = 0)
    g = warp(A, oA, s, M, np.zeros(2), (D, D), oC); mg = (warp(mA, oA, s, M, np.zeros(2), (D, D), oC) > 0.5).astype(float)
    g *= mg
    C = fx.scores(g, mg, kind, min_frac); k, v = peak(C, fx.size, D)
    # canvas pixel = B pixel + k  ->  c = oB - oC - s k
    return v, M, oB - oC - s * k


def search(A, mA, oA, B, mB, oB, s, kind, min_frac_area):
    h, w = A.shape; D = int(np.ceil(np.hypot(h, w))) + 2
    size = tuple(sfft.next_fast_len(n) for n in (B.shape[0] + D, B.shape[1] + D))
    fx = Fixed(B * mB, mB, size); min_frac = min_frac_area * min(mA.sum(), mB.sum())
    coarse = [(eval_pose(fx, A, mA, oA, B, oB, s, d, fl, kind, min_frac)[0], d, fl) for fl in (0, 1) for d in range(360)]
    coarse.sort(key=lambda z: -z[0]); seeds = []
    for v, d, fl in coarse:                                   # top 5 distinct (more than 3 degrees apart)
        if all(fl != f2 or min(abs(d - d2), 360 - abs(d - d2)) > 3 for _, d2, f2 in seeds):
            seeds.append((v, d, fl))
        if len(seeds) == 5:
            break
    fine = []
    for _, d0, fl in seeds:
        best = max((eval_pose(fx, A, mA, oA, B, oB, s, d0 + dd, fl, kind, min_frac) + (d0 + dd,)
                    for dd in np.round(np.arange(-1.0, 1.0001, 0.1), 2)), key=lambda z: z[0])
        fine.append(dict(score=best[0], deg=float(best[3] % 360), flip=fl, c=best[2]))
    fine.sort(key=lambda z: -z["score"])
    return fine


# ------------------------------------------------------------------ mutual-information refinement (optional)
def mi_refine(A, oA, B, oB, s, cand):
    import SimpleITK as sitk
    fl = cand["flip"]
    Ai = A[::-1] if fl else A; oAi = np.array([oA[0], -oA[1] - (A.shape[0] - 1) * s]) if fl else oA   # a' = F a
    fix = sitk.GetImageFromArray(B.astype(np.float32)); fix.SetSpacing((s, s)); fix.SetOrigin(tuple(map(float, oB)))
    mov = sitk.GetImageFromArray(np.ascontiguousarray(Ai).astype(np.float32)); mov.SetSpacing((s, s)); mov.SetOrigin(tuple(map(float, oAi)))
    th = np.radians(cand["deg"]); Rm = rot(-th)                 # a' = R(-th) (b - c)
    T = sitk.Euler2DTransform(); T.SetCenter((0.0, 0.0)); T.SetAngle(-th); T.SetTranslation(tuple(map(float, -Rm @ cand["c"])))
    reg = sitk.ImageRegistrationMethod(); reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.REGULAR); reg.SetMetricSamplingPercentage(0.5, 1); reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-3, numberOfIterations=200, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift(); reg.SetShrinkFactorsPerLevel([2, 1]); reg.SetSmoothingSigmasPerLevel([1, 0])
    reg.SetInitialTransform(T, inPlace=True)
    reg.Execute(fix, mov)
    ph = T.GetAngle(); t = np.array(T.GetTranslation()); deg = float(np.degrees(-ph) % 360)
    c = -rot(np.radians(deg)) @ t                             # a' = R(ph) b + t  ->  b = R(-ph) a' - R(-ph) t
    return dict(score=float(reg.GetMetricValue()), deg=deg, flip=fl, c=c)


# ------------------------------------------------------------------ scoring
def score_row(k, LA, LB, deg, fl, c_um, floor):
    e = lm_error(LA, LB, np.zeros(2), np.zeros(2), deg, fl, np.asarray(c_um) / UNIT)
    med = float(np.median(e)); return round(med, 1), bool(med < floor + 100)


def overlay(path, B, A_on_B):
    def n(z): return z / max(np.percentile(z, 99.5), 1e-6)
    rgb = np.stack([n(A_on_B), n(B), n(A_on_B)], 2); rgb = 1 - np.clip(rgb, 0, 1)      # B dark green-ish, A magenta
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


SETTINGS = [(im, s, kind) for s in PIX_UM for im in ("intensity", "mask") for kind in ("ncc", "phase")]


def prepare(k):
    """photos, landmarks, truth and the checks that need no search"""
    pname = next(p for p in PAIRS if p.startswith(f"pair{k}_")); ka, kb = PAIRS[pname]["left"], PAIRS[pname]["right"]
    LA, LB = landmarks(k, ROOT); R, tau = rigid_fit(LA, LB)
    floor = float(np.median(np.linalg.norm(LA @ R.T + tau - LB, axis=1)) * UNIT)
    tdeg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360)
    rgbA, rgbB = (np.asarray(Image.open(f"{PHOTOS}/{pname}/{f}").convert("RGB")) for f in ("left_A.png", "right_B.png"))
    return dict(k=k, ka=ka, kb=kb, LA=LA, LB=LB, floor=floor, tdeg=tdeg, tc=tau * UNIT,
                start=round(float(np.median(np.linalg.norm(LA - LB, axis=1)) * UNIT), 1), mapsA=tissue_maps(rgbA), mapsB=tissue_maps(rgbB))


def images(P, im, s):
    i = 0 if im == "intensity" else 1
    return render(P["ka"], P["mapsA"][i], s) + render(P["kb"], P["mapsB"][i], s)


def check_pair(k):
    """0: photo pixels -> um reproduces the landmark file; 1: the truth through our scoring gives the floor;
    2: the truth moves photo A onto photo B (overlay PNG and NCC at the truth)"""
    import pandas as pd
    P = prepare(k); px = pd.read_csv(f"{ROOT}/landmarks/pair{k}_landmarks.csv")
    da = np.abs(to_um(P["ka"], px[["left_x_px", "left_y_px"]].values) - P["LA"] * UNIT).max()
    db = np.abs(to_um(P["kb"], px[["right_x_px", "right_y_px"]].values) - P["LB"] * UNIT).max()
    e_truth, _ = score_row(k, P["LA"], P["LB"], P["tdeg"], 0, P["tc"], P["floor"])
    e_id, _ = score_row(k, P["LA"], P["LB"], 0.0, 0, np.zeros(2), P["floor"])
    out = dict(pair=k, max_um_diff_landmarks_A=round(float(da), 2), max_um_diff_landmarks_B=round(float(db), 2), floor_um=round(P["floor"], 1),
               truth_err_via_scoring_um=e_truth, identity_err_um=e_id, start_err_um=P["start"], truth_angle=round(P["tdeg"], 2))
    for im in ("intensity", "mask"):
        A, supA, oA, B, supB, oB = images(P, im, PIX_UM[0]); Mt = rot(np.radians(P["tdeg"]))
        AonB = warp(A, oA, PIX_UM[0], Mt, P["tc"], B.shape, oB); sAonB = warp(supA, oA, PIX_UM[0], Mt, P["tc"], B.shape, oB) > 0.5
        overlay(f"{WORK}/overlay_truth_pair{k}_{im}.png", B, AonB)
        out[f"ncc_at_truth_{im}"] = round(float(ncc_at(B, supB, AonB, sAonB)), 3)
    return out


def run_setting(P, idx, do_mi=True):
    im, s, kind = SETTINGS[idx]; k = P["k"]; t0 = time.time()
    A, supA, oA, B, supB, oB = images(P, im, s)
    fine = search(A, supA, oA, B, supB, oB, s, kind, 0.3); sec = time.time() - t0; rows = []
    for mi in ((False, True) if do_mi else (False,)):
        t1 = time.time()
        if mi:
            try:
                ref = [mi_refine(A, oA, B, oB, s, f) for f in fine[:3]]
            except ImportError:
                continue
            best = min(ref, key=lambda z: z["score"])
        else:
            best = fine[0]
        err, ok = score_row(k, P["LA"], P["LB"], best["deg"], best["flip"], best["c"], P["floor"])
        rows.append(dict(case=k, method=f"imagereg_{kind}{'_mi' if mi else ''}", fit=f"{im}_{int(s)}um", angle=round(best["deg"], 2),
                         flip=int(best["flip"]), t_abs_units=[round(float(v), 2) for v in np.asarray(best["c"]) / UNIT],
                         score=round(best["score"], 4), sec=round(sec + time.time() - t1, 1), arm="given", start_err_um=P["start"],
                         landmark_err_median_um=err, floor_um=round(P["floor"], 1), pair=k, right_within_floor_plus_100=ok,
                         candidates=[dict(deg=round(f["deg"], 1), flip=f["flip"], score=round(f["score"], 4),
                                          err_um=score_row(k, P["LA"], P["LB"], f["deg"], f["flip"], f["c"], P["floor"])[0]) for f in fine],
                         note="P(a) = R(angle) F^flip a + t_abs_units, analysis-frame units (9.44 um), no centring; "
                              "candidates = top 5 search answers with their landmark errors (for diagnosis only)"))
        print(json.dumps({x: rows[-1][x] for x in ("pair", "method", "fit", "angle", "flip", "landmark_err_median_um", "sec")}), flush=True)
        if im == "intensity" and s == PIX_UM[0] and kind == "ncc" and not mi:
            M = rot(np.radians(best["deg"])) @ (FLIP if best["flip"] else np.eye(2))
            overlay(f"{WORK}/overlay_found_pair{k}.png", B, warp(A, oA, s, M, best["c"], B.shape, oB))
    return rows


def ncc_at(B, supB, AonB, supAonB):
    m = (supB > 0.5) & supAonB; x, y = B[m], AonB[m]
    return np.corrcoef(x, y)[0, 1] if m.sum() > 10 else np.nan


def self_test():
    """known transform: move photo B of pair 3 by a known rotation, flip and shift and check the search finds it"""
    pname = next(p for p in PAIRS if p.startswith("pair3_")); kb = PAIRS[pname]["right"]
    dB, _ = tissue_maps(np.asarray(Image.open(f"{PHOTOS}/{pname}/right_B.png").convert("RGB"))); s = 50.0
    B, supB, oB = render(kb, dB, s); M = rot(np.radians(137.0)) @ FLIP; c = np.array([1234.0, -2345.0])
    h, w = B.shape; D = int(np.hypot(h, w)) + 4; Minv = np.linalg.inv(M); cB = oB + s * np.array([w, h]) / 2
    oA = Minv @ (cB - c) - s * D / 2                         # A = B moved by the inverse transform: B(b) = A(M^-1 (b - c))
    A = warp(B, oB, s, Minv, -Minv @ c, (D, D), oA); supA = warp(supB, oB, s, Minv, -Minv @ c, (D, D), oA) > 0.5
    out = {}
    for kind in ("ncc", "phase"):
        f = search(A, supA.astype(float), oA, B, supB, oB, s, kind, 0.3)[0]
        out[kind] = dict(deg=round(f["deg"], 2), flip=f["flip"], c=[round(v, 1) for v in f["c"]])
    out["truth"] = dict(deg=137.0, flip=1, c=c.tolist())
    return out


if __name__ == "__main__":
    os.makedirs(f"{WORK}/parts", exist_ok=True); cmd = sys.argv[1]
    if cmd == "selftest":                                    # python3 image_reg.py selftest
        print(json.dumps(self_test()))
    elif cmd == "check":                                     # python3 image_reg.py check 1 2 ...
        for k in map(int, sys.argv[2:]):
            c = check_pair(k); json.dump(c, open(f"{WORK}/checks_pair{k}.json", "w"), indent=1); print(json.dumps(c), flush=True)
    elif cmd == "part":                                      # python3 image_reg.py part PAIR SETTING   (0..7)
        k, i = int(sys.argv[2]), int(sys.argv[3]); rows = run_setting(prepare(k), i)
        json.dump(rows, open(f"{WORK}/parts/pair{k}_set{i}.json", "w"), indent=1)
    elif cmd == "merge":                                     # python3 image_reg.py merge  -> results/placement/lm_given_pair{k}_imagereg.json
        for k in range(1, 8):
            fs = sorted(glob.glob(f"{WORK}/parts/pair{k}_set*.json"))
            if not fs:
                continue
            rows = [r for f in fs for r in json.load(open(f))]
            json.dump(rows, open(f"{ROOT}/results/placement/lm_given_pair{k}_imagereg.json", "w"), indent=1)
            e = [r["landmark_err_median_um"] for r in rows]
            print(f"pair{k} {NAMES[k]:12s} settings {len(rows)} best {min(e)} median {np.median(e)} floor {rows[0]['floor_um']}")
    else:                                                    # python3 image_reg.py 1 2 ...   (everything, one pair after another)
        for k in map(int, sys.argv[1:]):
            P = prepare(k); rows = [r for i in range(len(SETTINGS)) for r in run_setting(P, i)]
            json.dump(rows, open(f"{ROOT}/results/placement/lm_given_pair{k}_imagereg.json", "w"), indent=1)
