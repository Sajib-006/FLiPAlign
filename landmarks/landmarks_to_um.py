"""Turn clicked landmark files into microns, in the same frames the analysis uses.
Usage: python3 landmarks_to_um.py pairN_name_landmarks.csv   (the pair folder name must start the file name)
Left picture = section A (Visium), right = section B (Xenium or Visium HD).
Writes <file>_um.csv with columns point, ax_um, ay_um, bx_um, by_um."""
import csv, json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
cal = json.load(open(f"{H}/calibration.json")); pairs = json.load(open(f"{H}/pairs.json"))

def to_um(key, x, y):
    c = cal[key]
    if "xenium_px_from_he_px" in c:                                   # Xenium H&E: picture px -> full H&E px -> Xenium px -> um
        M = np.array(c["xenium_px_from_he_px"]); full = np.array([x * c["full_per_small_x"], y * c["full_per_small_y"], 1.0])
        return tuple((M @ full)[:2] * c["um_per_xenium_px"])
    f = c["fullres_per_picture_px"] * c["um_per_fullres_px"]          # Visium or Visium HD: picture px -> full-res px -> um
    return x * f, y * f

if __name__ == "__main__":
    fn = sys.argv[1]; pair = next(p for p in pairs if os.path.basename(fn).startswith(p.split("_")[0] + "_"))
    ka, kb = pairs[pair]["left"], pairs[pair]["right"]
    rows = list(csv.DictReader(open(fn))); out = []
    for r in rows:
        a = to_um(ka, float(r["left_x_px"]), float(r["left_y_px"])); b = to_um(kb, float(r["right_x_px"]), float(r["right_y_px"]))
        out.append([r["point"], *[f"{v:.1f}" for v in (*a, *b)]])
    with open(fn.replace(".csv", "_um.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["point", "ax_um", "ay_um", "bx_um", "by_um"]); w.writerows(out)
    print(pair, len(out), "points written")
