"""Table of GALA vs floor+100 and the other tools. Usage: python3 gala_summary.py"""
import json, glob, os
import numpy as np
names = {1: "breast Rep1", 2: "breast Rep2", 3: "colon P2CRC", 4: "colon P3NAT", 5: "colon P5NAT", 6: "lung HD1", 7: "lung HD2"}
S = "server_results/new/"
for k in range(1, 8):
    p = f"{S}lm_given_pair{k}_gala.json"
    if not os.path.exists(p): print(k, "missing"); continue
    rows = [r for r in json.load(open(p)) if "landmark_err_median_um" in r]
    floor = rows[0]["floor_um"]
    def best(f): return min(rows, key=f)
    b_rig = best(lambda r: r["landmark_err_median_um"]); b_own = best(lambda r: r["landmark_err_own_median_um"])
    ok = lambda e: "right" if e < floor + 100 else "wrong"
    print(f"pair{k} {names[k]:12s} floor {floor:6.1f} | rigid-fit best {b_rig['landmark_err_median_um']:8.1f} ({b_rig['method']},{b_rig['fit']}) {ok(b_rig['landmark_err_median_um'])}"
          f" | own-map best {b_own['landmark_err_own_median_um']:8.1f} ({b_own['method']}) {ok(b_own['landmark_err_own_median_um'])} | sec/setting {rows[0]['sec']}")
