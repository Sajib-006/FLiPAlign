"""Fine stage of the certificate (reviewer fix: the coarse certificate only pins the answer to about
20 deg and 280 um, because the maps are blurred at sigma = 20 units).

Stage 1 (coarse, already run): score S1 (sigma 20, h 2) over ALL placements; every placement
outside the zone Z1 = (alpha1, r1) around the answer T1 scores at least eps1 below it.
Stage 2 (this file): a sharper score S2 (sigma2, h2) searched ONLY inside Z1 (rotation within
alpha1 of T1, same flip, shifts in a square window of half-width r1 + 2 around t1, which contains the
disc of Z1). Branch-and-bound gives the S2-best placement T2 in there with a proven gap, then the
same far search, restricted to Z1, proves every placement in Z1 outside a small zone Z2 around T2
scores at least eps2 lower under S2.
Together: outside Z1 nothing competes under S1; inside Z1 the sharp score has one peak, pinned to Z2.
Both steps use the same bound (Theorem "valid bound"); only the search region changes.
"""
import heapq, time
import numpy as np
from smoothalign import SmoothAlign
from smoothpar import _pool, _eval


def _cover(lo, hi, n0):
    d = (hi - lo) / (2 * n0)
    return [(lo + (2 * k + 1) * d, d) for k in range(n0)]


def solve_range(al, lo, hi, flip, n0=40, tol=1e-3, min_delta=None, n_proc=16, time_limit=None):
    """branch-and-bound over rotations in [lo, hi] (radians), one flip"""
    t0 = time.time()
    if min_delta is None:
        min_delta = (al.h / np.sqrt(2)) / (2 * al.rmax)
    pool = _pool(al, n_proc)
    best, bT, heap, cnt, floor_gap, capped, evals = -np.inf, None, [], 0, 0.0, False, 0
    for th, fl, d, s, t, ub in pool.map(_eval, [(c, flip, d) for c, d in _cover(lo, hi, n0)]):
        evals += 1
        if s > best: best, bT = s, (th, t, fl)
        heapq.heappush(heap, (-ub, cnt, th, d, fl)); cnt += 1
    while heap:
        batch = []
        while heap and len(batch) < 2 * n_proc:
            nu, c_, th, d, fl = heapq.heappop(heap)
            if -nu <= best + tol:
                heapq.heappush(heap, (nu, c_, th, d, fl)); break
            if d <= min_delta:
                floor_gap = max(floor_gap, -nu - best); continue
            batch.append((th, d, fl))
        if not batch:
            break
        if time_limit and time.time() - t0 > time_limit:
            for th, d, fl in batch:
                heapq.heappush(heap, (-np.inf, cnt, th, d, fl)); cnt += 1
            capped = True; break
        for th, fl, d, s, t, ub in pool.map(_eval, [(c, fl, d / 2) for th, d, fl in batch
                                                     for c in (th - d / 2, th + d / 2)]):
            evals += 1
            if s > best: best, bT = s, (th, t, fl)
            if ub > best + tol:
                heapq.heappush(heap, (-ub, cnt, th, d, fl)); cnt += 1
    pool.close(); pool.join()
    open_gap = max([0.0] + [(-h_[0] if np.isfinite(h_[0]) else np.inf) - best for h_ in heap])
    return {"best": float(best), "transform": bT, "gap": max(floor_gap, open_gap, tol),
            "certified": not capped, "evals": evals, "secs": time.time() - t0}


def far_range(al, best, lo, hi, flip, th2, t2, alpha2, r2, n0=40, min_delta=None, n_proc=16,
              time_limit=None):
    """upper bound on S2 over placements with rotation in [lo, hi], this flip, outside the zone
    (|theta - th2| <= alpha2 and |t - t2| <= r2). Ranges wholly inside the rotation part of the
    zone are checked only at shifts farther than r2 - h/sqrt2 (as in smoothpar.far_par_ext)."""
    t0 = time.time()
    if min_delta is None:
        min_delta = (al.h / np.sqrt(2)) / (2 * al.rmax)
    mask = (np.asarray(t2, float), float(r2))
    dist = lambda th: abs(np.angle(np.exp(1j * (th - th2))))
    job = lambda th, d: (th, flip, d, mask) if dist(th) + d <= alpha2 else (th, flip, d)
    pool = _pool(al, n_proc)
    heap, cnt, floor_ub, capped, evals, far_exact = [], 0, -np.inf, False, 0, -np.inf
    for th, fl, d, s, t, ub in pool.map(_eval, [job(c, d) for c, d in _cover(lo, hi, n0)]):
        evals += 1
        if np.isfinite(ub):
            far_exact = max(far_exact, s); heapq.heappush(heap, (-ub, cnt, th, d)); cnt += 1
    while heap:
        batch = []
        while heap and len(batch) < 2 * n_proc:
            nu, c_, th, d = heapq.heappop(heap)
            if d <= min_delta:
                floor_ub = max(floor_ub, -nu); continue
            if -nu <= floor_ub:
                heapq.heappush(heap, (nu, c_, th, d)); break
            batch.append((th, d, -nu))
        if not batch:
            break
        if time_limit and time.time() - t0 > time_limit:
            for th, d, u in batch:
                heapq.heappush(heap, (-u, cnt, th, d)); cnt += 1
            capped = True; break
        for th, fl, d, s, t, ub in pool.map(_eval, [job(c, d / 2) for th, d, _ in batch
                                                     for c in (th - d / 2, th + d / 2)]):
            evals += 1
            if np.isfinite(ub):
                far_exact = max(far_exact, s); heapq.heappush(heap, (-ub, cnt, th, d)); cnt += 1
    pool.close(); pool.join()
    far_ub = max([floor_ub] + [-h_[0] for h_ in heap])
    return {"eps": float(best - far_ub), "far_ub": float(far_ub), "far_exact": float(far_exact),
            "complete": not capped, "evals": evals, "secs": time.time() - t0}


# zone shapes tried (rotation deg, shift units); they are not nested, so every one is checked and
# the answer is reported pinned to the certified zone that lets points move least
FINE_LADDER = [(5.0, 10.0), (3.0, 10.0), (2.0, 10.0), (1.0, 10.0), (3.0, 5.0), (2.0, 5.0), (1.0, 5.0),
               (1.0, 3.0), (0.5, 3.0)]


def fine_stage(A, la, B, lb, th1, fl1, t1, alpha1_deg, r1, sigma2=5.0, h2=0.5, n_proc=16,
               time_limit=7200, unit_um=9.44):
    """run stage 2 inside the coarse zone (alpha1_deg, r1) around (th1, fl1, t1); units as A, B"""
    rad = np.linalg.norm(A - A.mean(0), axis=1)
    zone_move = lambda deg, sh: float(np.median(2 * rad * np.sin(np.radians(deg) / 2)) + sh) * unit_um
    a1 = np.radians(alpha1_deg)
    al = SmoothAlign(A, la, B, lb, h=h2, sigma=sigma2, tpad=r1 + 2.0, contrast=True, center=t1)
    o = solve_range(al, th1 - a1, th1 + a1, fl1, n_proc=n_proc, time_limit=time_limit)
    th2, t2, _ = o["transform"]
    out = {"sigma2": sigma2, "h2": h2, "best2": round(o["best"], 4), "gap2": round(o["gap"], 4),
           "search2_complete": o["certified"], "search2_s": round(o["secs"], 1),
           "theta2_deg": round(float(np.degrees(th2)) % 360, 3), "t2": [round(float(v), 2) for v in t2],
           "inside_Z1": bool(abs(np.angle(np.exp(1j * (th2 - th1)))) <= a1
                             and np.hypot(*(np.asarray(t2) - np.asarray(t1))) <= r1),
           "ladder2": []}
    if o["certified"]:
        for al2, r2 in FINE_LADDER:
            u = far_range(al, o["best"], th1 - a1, th1 + a1, fl1, th2, t2, np.radians(al2), r2,
                          n_proc=n_proc, time_limit=time_limit / 2)
            out["ladder2"].append({"zone_deg": al2, "zone_shift_units": r2, "eps": round(u["eps"], 4),
                                   "complete": u["complete"], "far_s": round(u["secs"], 1),
                                   "zone_move_um": round(zone_move(al2, r2), 1)})
    ok = [s for s in out["ladder2"] if s["eps"] > 0]
    best_zone = min(ok, key=lambda s: s["zone_move_um"]) if ok else None
    out["fine_zone"] = [best_zone["zone_deg"], best_zone["zone_shift_units"]] if ok else None
    return out, (th2, t2)
