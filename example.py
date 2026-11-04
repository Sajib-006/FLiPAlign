"""Small FLiPAlign example on a synthetic pair (runs in about a minute on a laptop, no data download).

Section B: a disc of cells with six irregular regions. Section A: 70% of B, turned by 37 degrees, mirrored and shifted, with its
region labels renamed at random (so the two label sets share no names). FLiPAlign searches every turn, flip and shift,
then tests the zone around its answer: if every placement outside the zone scores lower by a proven margin eps > 0,
the answer is certified; otherwise it refuses.
Usage: python3 example.py"""
import numpy as np
from smoothalign import SmoothAlign, far_upper_bound, rot

rng = np.random.default_rng(0)
g = np.stack(np.meshgrid(np.arange(-60, 61), np.arange(-60, 61)), -1).reshape(-1, 2).astype(float)
B = g[(g ** 2).sum(1) < 60 ** 2]
seeds = rng.uniform(-60, 60, (14, 2))                                                    # irregular regions: nearest seed
lb = np.argmin(((B[:, None] - seeds[None]) ** 2).sum(-1), 1) % 6

keep = B[:, 0] < np.quantile(B[:, 0], 0.7)                                               # A covers 70% of B
th_true, flip_true, t_true = np.radians(37.0), 1, np.array([12.0, -7.0])
F = np.diag([1.0, -1.0]) if flip_true else np.eye(2)
A = (B[keep] - t_true) @ rot(th_true)                                                     # B = R F A + t, so A = F R^T (B - t)
A = A @ F.T
la = rng.permutation(6)[lb[keep]]                                                         # renamed labels

al = SmoothAlign(A, la, B, lb, h=2.0, sigma=8.0, tpad=40.0, contrast=True)
o = al.solve()
th, t, fl = o["transform"]
print(f"best score {o['best']:.3f}, proven search gap {o['gap']:.3f}")
print(f"answer: turn {np.degrees(th) % 360:.1f} deg, flip {fl}, shift ({t[0]:.1f}, {t[1]:.1f}) units")
u = far_upper_bound(al, o["best"], th, fl, exclude_deg=20.0)
print("certified, margin eps = %.3f" % u["eps"] if u["eps"] > 0 else "refused (no proven margin)")
err = np.median(np.linalg.norm(al._points(th, fl) + t - B[keep], axis=1))       # placed cells against their true spots
print(f"median error of the placed cells: {err:.2f} units")
