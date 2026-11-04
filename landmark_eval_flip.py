"""FLiPAlign (frozen answers, realfine_t1..t5) against the clicked landmarks. Usage: python3 landmark_eval_flip.py K (K=1..5)
Env: ROOT (project folder on ARC)"""
import json, os, sys
import numpy as np
from landmark_common import *
k = int(sys.argv[1]); root = os.environ.get("ROOT", HOME)
c = load_pair(k); A, B = c["A"], c["B"]; T = truth(k, A, B, root); LA, LB = landmarks(k, root)
r = json.load(open(f"{root}/results/realfine_t{k}.json" if k <= 5 else f"{root}/results/lungfine{k}_t0.json"))[0]
out = {"pair": k, "name": NAMES[k], "n_landmarks": T["n"], "floor_um": round(T["floor_um"], 1), "truth_angle": round(T["angle_deg"], 2),
       "flipalign_certified": r["certified"], "flipalign_zone": r["finest_zone"]}
for tag, (deg, sh) in (("coarse", (r["angle"], r["shift_units"])), ("fine", (r.get("fine_angle"), r.get("fine_shift_units")))):
    if deg is None:
        continue
    e = lm_error(LA, LB, A.mean(0), B.mean(0), deg, r["flip"], sh)
    out[tag] = {"angle": round(deg, 3), "flip": r["flip"], "shift": sh, "median_um": round(float(np.median(e)), 1), "max_um": round(float(e.max()), 1),
                "angle_error_deg": round(float(abs((deg - T["angle_deg"] + 180) % 360 - 180)), 2)}
out["right_within_floor_plus_100"] = {t: out[t]["median_um"] < out["floor_um"] + 100 for t in ("coarse", "fine") if t in out}
os.makedirs(f"{root}/results", exist_ok=True); json.dump(out, open(f"{root}/results/lm_flip_pair{k}.json", "w"), indent=1); print(json.dumps(out))
