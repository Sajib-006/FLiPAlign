"""Parallel versions of SmoothAlign.solve and far_upper_bound.

Same branch-and-bound logic as the serial code; the only change is that the rotation
ranges waiting to be split are handled in batches, one range per worker process. Each
worker runs its FFTs single-threaded, so 16 workers use 16 cores.

The aligner is shared with the workers by fork (copy-on-write), so it is built once.
Linux only (the server); on macOS use the serial versions.
"""
import heapq, time
import multiprocessing as mp
import numpy as np

_AL = None


def _eval(job):
    th, flip, d = job[:3]
    mask = job[3] if len(job) > 3 else None
    s, t, ub = _AL.rotation_eval(th, flip, d, far_shift=mask)
    return th, flip, d, s, t, ub


def _eval2(job):
    """node = (th, flip, d, off, cell, mask): rotation range and a shift sub-cell"""
    th, flip, d, off, cell, mask = job
    s, t, ub = _AL.rotation_eval(th, flip, d, far_shift=mask, offset=off, cell=cell)
    return job, s, t, ub


def _pool(al, n_proc):
    global _AL
    _AL = al
    al.workers = 1                        # one core per worker
    return mp.get_context("fork").Pool(n_proc)


def solve_par(al, n0=90, tol=1e-3, min_delta=None, n_proc=16, time_limit=None, flips=(0, 1)):
    """Parallel 1-D branch-and-bound over rotation x flip. Same outputs as al.solve()."""
    t0 = time.time()
    if min_delta is None:
        min_delta = (al.h / np.sqrt(2)) / (2 * al.rmax)
    d0 = np.pi / n0
    pool = _pool(al, n_proc)
    best, bT, profile, heap, cnt, floor_gap, capped, evals = -1.0, None, [], [], 0, 0.0, False, 0
    for th, fl, d, s, t, ub in pool.map(_eval, [((2 * k + 1) * d0, f, d0) for f in flips for k in range(n0)]):
        evals += 1; profile.append((th, fl, s))
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
        jobs = [(c, fl, d / 2) for th, d, fl in batch for c in (th - d / 2, th + d / 2)]
        for th, fl, d, s, t, ub in pool.map(_eval, jobs):
            evals += 1
            if s > best: best, bT = s, (th % (2 * np.pi), t, fl)
            if ub > best + tol:
                heapq.heappush(heap, (-ub, cnt, th, d, fl)); cnt += 1
    pool.close(); pool.join()
    open_gap = max([0.0] + [(-h_[0] if np.isfinite(h_[0]) else np.inf) - best for h_ in heap])
    return {"best": best, "transform": bT, "gap": max(floor_gap, open_gap, 0.0),
            "certified": not capped, "capped": capped, "evals": evals,
            "secs": time.time() - t0, "profile": profile}


def far_par(al, best, th_best, flip_best, exclude_deg=20.0, n0=90, min_delta=None,
            n_proc=16, time_limit=None, flips=(0, 1)):
    """Parallel far_upper_bound: same definition and output fields as the serial one."""
    t0 = time.time()
    if min_delta is None:
        min_delta = (al.h / np.sqrt(2)) / (2 * al.rmax)
    ex = np.radians(exclude_deg)
    dist = lambda th, fl: np.inf if fl != flip_best else abs(np.angle(np.exp(1j * (th - th_best))))
    d0 = np.pi / n0
    pool = _pool(al, n_proc)
    jobs = [((2 * k + 1) * d0, f, d0) for f in flips for k in range(n0)
            if dist((2 * k + 1) * d0, f) + d0 > ex]
    heap, cnt, far_exact, floor_ub, capped, evals = [], 0, -np.inf, -np.inf, False, 0
    far_at = None
    for th, fl, d, s, t, ub in pool.map(_eval, jobs):
        evals += 1
        if dist(th, fl) > ex and s > far_exact:
            far_exact, far_at = s, (float(np.degrees(th)) % 360, int(fl), [float(v) for v in t])
        heapq.heappush(heap, (-ub, cnt, th, d, fl)); cnt += 1
    while heap:
        batch = []
        while heap and len(batch) < 2 * n_proc:
            nu, c_, th, d, fl = heapq.heappop(heap)
            if d <= min_delta:
                floor_ub = max(floor_ub, -nu); continue
            if -nu <= floor_ub:
                heapq.heappush(heap, (nu, c_, th, d, fl)); break
            batch.append((th, d, fl, -nu))
        if not batch:
            break
        if time_limit and time.time() - t0 > time_limit:
            for th, d, fl, u in batch:
                heapq.heappush(heap, (-u, cnt, th, d, fl)); cnt += 1
            capped = True; break
        jobs = [(c, fl, d / 2) for th, d, fl, _ in batch for c in (th - d / 2, th + d / 2)
                if dist(c, fl) + d / 2 > ex]
        for th, fl, d, s, t, ub in pool.map(_eval, jobs):
            evals += 1
            if dist(th, fl) > ex and s > far_exact:
                far_exact, far_at = s, (float(np.degrees(th)) % 360, int(fl), [float(v) for v in t])
            heapq.heappush(heap, (-ub, cnt, th, d, fl)); cnt += 1
    pool.close(); pool.join()
    far_ub = max([floor_ub] + [-h_[0] for h_ in heap])
    return {"far_ub": float(far_ub), "far_exact": float(far_exact), "eps": float(best - far_ub),
            "complete": not capped, "far_evals": evals, "far_secs": time.time() - t0,
            "best_far_placement": far_at}


def far_par_ext(al, best, th_best, flip_best, t_best, exclude_deg=20.0, shift_radius=30.0,
                n0=90, min_delta=None, n_proc=16, time_limit=None, flips=(0, 1)):
    """Uniqueness check that also covers FAR SHIFTS at near rotations.
    Far set = (rotation more than exclude_deg from the best, or flipped; any shift)
            + (rotation within exclude_deg, same flip; shift more than shift_radius cells
               from t_best).
    Intervals that straddle the rotation boundary are checked with ALL shifts (conservative).
    Same outputs as far_par, plus which part of the far set holds the highest bound."""
    t0 = time.time()
    if min_delta is None:
        min_delta = (al.h / np.sqrt(2)) / (2 * al.rmax)
    ex = np.radians(exclude_deg)
    mask = (np.asarray(t_best, float), float(shift_radius))
    dist = lambda th, fl: np.inf if fl != flip_best else abs(np.angle(np.exp(1j * (th - th_best))))

    def job(th, fl, d):
        near = dist(th, fl) + d <= ex                   # wholly inside the near-rotation zone
        return (th, fl, d, mask) if near else (th, fl, d)

    d0 = np.pi / n0
    pool = _pool(al, n_proc)
    heap, cnt, far_exact, floor_ub, capped, evals = [], 0, -np.inf, -np.inf, False, 0
    where = {"rotation": -np.inf, "shift": -np.inf}
    res = pool.map(_eval, [job((2 * k + 1) * d0, f, d0) for f in flips for k in range(n0)])
    for th, fl, d, s, t, ub in res:
        evals += 1
        if not np.isfinite(ub):
            continue
        far_exact = max(far_exact, s)
        heapq.heappush(heap, (-ub, cnt, th, d, fl)); cnt += 1
    while heap:
        batch = []
        while heap and len(batch) < 2 * n_proc:
            nu, c_, th, d, fl = heapq.heappop(heap)
            if d <= min_delta:
                floor_ub = max(floor_ub, -nu)
                k = "shift" if dist(th, fl) + d <= ex else "rotation"
                where[k] = max(where[k], -nu)
                continue
            if -nu <= floor_ub:
                heapq.heappush(heap, (nu, c_, th, d, fl)); break
            batch.append((th, d, fl, -nu))
        if not batch:
            break
        if time_limit and time.time() - t0 > time_limit:
            for th, d, fl, u in batch:
                heapq.heappush(heap, (-u, cnt, th, d, fl)); cnt += 1
            capped = True; break
        jobs = [job(c, fl, d / 2) for th, d, fl, _ in batch for c in (th - d / 2, th + d / 2)]
        for th, fl, d, s, t, ub in pool.map(_eval, jobs):
            evals += 1
            if not np.isfinite(ub):
                continue
            far_exact = max(far_exact, s)
            heapq.heappush(heap, (-ub, cnt, th, d, fl)); cnt += 1
    pool.close(); pool.join()
    far_ub = max([floor_ub] + [-h_[0] for h_ in heap])
    return {"far_ub": float(far_ub), "far_exact": float(far_exact), "eps": float(best - far_ub),
            "complete": not capped, "far_evals": evals, "far_secs": time.time() - t0,
            "highest_bound_in": max(where, key=where.get) if np.isfinite(max(where.values())) else "open"}


# ---------------------------------------------------------------------------------------------
# Rotation AND shift splitting (review round 2): when a rotation range is already narrow (its
# movement term 2 rmax sin(d/2) is no larger than the shift cell), split the shift cell into four
# sub-cells at sub-grid offsets instead. Same score (same maps, read at finer offsets); the bound
# slack then keeps shrinking, so the proven gap is not stuck at the grid term.
def _children(th, fl, d, off, cell, rmax):
    if 2 * rmax * np.sin(d / 2) > cell:
        return [(th - d / 2, fl, d / 2, off, cell), (th + d / 2, fl, d / 2, off, cell)]
    q = cell / 2                                 # sub-square centres: +-side/4 per axis = +-cell/(2 sqrt2)
    a = cell / (2 * np.sqrt(2))
    return [(th, fl, d, (off[0] + sx * a, off[1] + sy * a), q) for sx in (-1, 1) for sy in (-1, 1)]


def solve_split(al, n0=90, tol=1e-3, min_cell=None, n_proc=16, time_limit=None):
    """branch-and-bound over rotation x flip x shift sub-cells; stops when the gap <= tol or the
    shift cell reaches min_cell (default h / 16)"""
    t0 = time.time()
    min_cell = al.h / np.sqrt(2) / 16 if min_cell is None else min_cell
    cell0 = al.h / np.sqrt(2); d0 = np.pi / n0
    pool = _pool(al, n_proc)
    best, bT, heap, cnt, floor_gap, capped, evals = -np.inf, None, [], 0, 0.0, False, 0
    jobs = [((2 * k + 1) * d0, f, d0, (0.0, 0.0), cell0, None) for f in (0, 1) for k in range(n0)]
    while True:
        for job, s, t, ub in pool.map(_eval2, jobs):
            evals += 1
            th, fl, d, off, cell, _ = job
            if s > best: best, bT = s, (th % (2 * np.pi), t, fl)
            heapq.heappush(heap, (-ub, cnt, th, fl, d, off, cell)); cnt += 1
        jobs = []
        while heap and len(jobs) < 2 * n_proc:
            nu, c_, th, fl, d, off, cell = heapq.heappop(heap)
            if -nu <= best + tol:
                heapq.heappush(heap, (nu, c_, th, fl, d, off, cell)); break
            if cell <= min_cell and 2 * al.rmax * np.sin(d / 2) <= cell:
                floor_gap = max(floor_gap, -nu - best); continue
            jobs += [ch + (None,) for ch in _children(th, fl, d, off, cell, al.rmax)]
        if not jobs:
            break
        if time_limit and time.time() - t0 > time_limit:
            capped = True; break
    pool.close(); pool.join()
    open_gap = max([0.0] + [-h_[0] - best for h_ in heap])
    return {"best": float(best), "transform": bT, "gap": max(floor_gap, open_gap, tol),
            "certified": not capped, "evals": evals, "secs": time.time() - t0}


def far_split(al, best, th_best, flip_best, t_best, exclude_deg=20.0, shift_radius=None, n0=90,
              min_cell=None, n_proc=16, time_limit=None, stop_below=None):
    """far check with rotation and shift splitting. Far set: other flip, rotation more than
    exclude_deg away, or (shift_radius given) shift more than shift_radius from t_best at near
    rotations. stop_below: stop early once the largest far bound is below this value (eps proven)."""
    t0 = time.time()
    min_cell = al.h / np.sqrt(2) / 16 if min_cell is None else min_cell
    cell0 = al.h / np.sqrt(2); d0 = np.pi / n0; ex = np.radians(exclude_deg)
    dist = lambda th, fl: np.inf if fl != flip_best else abs(np.angle(np.exp(1j * (th - th_best))))
    def mk(th, fl, d, off, cell):
        if dist(th, fl) + d <= ex:                       # wholly inside the near-rotation zone
            if shift_radius is None:
                return None                              # rotation-only check: not far
            return (th, fl, d, off, cell, (np.asarray(t_best, float), float(shift_radius)))
        return (th, fl, d, off, cell, None)
    pool = _pool(al, n_proc)
    heap, cnt, floor_ub, capped, evals, far_exact = [], 0, -np.inf, False, 0, -np.inf
    jobs = [j for j in (mk((2 * k + 1) * d0, f, d0, (0.0, 0.0), cell0) for f in (0, 1) for k in range(n0)) if j]
    while True:
        for job, s, t, ub in pool.map(_eval2, jobs):
            evals += 1
            th, fl, d, off, cell, mask = job
            if not np.isfinite(ub):
                continue
            if dist(th, fl) > ex or mask is not None:
                far_exact = max(far_exact, s)
            heapq.heappush(heap, (-ub, cnt, th, fl, d, off, cell)); cnt += 1
        jobs = []
        top = -heap[0][0] if heap else -np.inf
        if stop_below is not None and max(top, floor_ub) < stop_below:
            break
        while heap and len(jobs) < 2 * n_proc:
            nu, c_, th, fl, d, off, cell = heapq.heappop(heap)
            if -nu <= floor_ub:
                heapq.heappush(heap, (nu, c_, th, fl, d, off, cell)); break
            if cell <= min_cell and 2 * al.rmax * np.sin(d / 2) <= cell:
                floor_ub = max(floor_ub, -nu); continue
            jobs += [j for j in (mk(*ch) for ch in _children(th, fl, d, off, cell, al.rmax)) if j]
        if not jobs:
            break
        if time_limit and time.time() - t0 > time_limit:
            capped = True; break
    pool.close(); pool.join()
    far_ub = max([floor_ub] + [-h_[0] for h_ in heap])
    return {"far_ub": float(far_ub), "far_exact": float(far_exact), "eps": float(best - far_ub),
            "complete": not capped, "far_evals": evals, "far_secs": time.time() - t0}
