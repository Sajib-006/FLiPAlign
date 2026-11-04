"""Numbers and figure for the reviewer-round experiments (2026-09-30). Reads ../server_results/new/.

  * zone ladder on the known-answer sets (gt1 = first 24 cases, gt2 = 72 cases on new tissues):
    right / wrong (200 um), certified at each zone, wrong answers certified, how far the zone lets
    points move;
  * refusal baseline: local search with 50 starts refuses when a far start (other flip, rotation
    more than 20 deg away, or shift more than 30 units away) scores within tau of its best;
    share answered vs share wrong among answered, against FLiPAlign ranked by its eps;
  * fair DLPFC on all 66 pairs (own gene clusters): layer score of certified vs refused.
Writes tables/new_numbers.tex (macros) and figures/refusal.pdf."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "../server_results/new"
RIGHT_UM = 200
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def merged(prefix):
    out = {}
    for f in sorted(glob.glob(f"{R}/{prefix}_t*.json")):
        for r in json.load(open(f)):
            out[r.get("case", r.get("pair"))] = r
    return out


def angdiff(a, b):
    return abs((a - b + 180) % 360 - 180)


def ladder_stats(ours, tag):
    n = len(ours)
    right = [c for c, r in ours.items() if r["tre_um"] < RIGHT_UM]
    cert = [c for c, r in ours.items() if r["certified"]]
    wrong_cert = [c for c in cert if ours[c]["tre_um"] >= RIGHT_UM]
    fine = [c for c in cert if ours[c]["finest_zone"] and ours[c]["finest_zone"][1] is not None]
    moves = [ours[c]["finest_zone_move_um"] for c in cert]
    return {f"{tag}n": n, f"{tag}right": len(right), f"{tag}cert": len(cert), f"{tag}wrongcert": len(wrong_cert),
            f"{tag}wrong": n - len(right), f"{tag}wrongrefused": sum(1 for c in ours if ours[c]["tre_um"] >= RIGHT_UM and c not in cert),
            f"{tag}rightrefused": sum(1 for c in right if c not in cert),
            f"{tag}median": f"{np.median([r['tre_um'] for r in ours.values()]):.0f}",
            f"{tag}certshift": len(fine),
            f"{tag}zonemove": f"{np.median(moves):.0f}" if moves else "--",
            f"{tag}medmin": f"{np.median([r['total_s'] for r in ours.values()]) / 60:.0f}"}


def far_margin(starts, fl_best, deg_best, t_best, zone_deg=20.0, zone_shift=30.0):
    """best score among starts that land outside the zone around the best start"""
    far = [s["score"] for s in starts
           if s["flip"] != fl_best or angdiff(s["angle"], deg_best) > zone_deg
           or np.hypot(*(np.asarray(s["shift"]) - t_best)) > zone_shift]
    return max(far) if far else -np.inf


def refusal_curves(ours, local):
    """(share answered, share wrong among answered) as the refusal threshold moves"""
    cases = sorted(set(ours) & set(local))
    L = []
    for c in cases:
        st = local[c]["starts"]; b = max(st, key=lambda s: s["score"])
        m = b["score"] - far_margin(st, b["flip"], b["angle"], np.asarray(b["shift"]))
        L.append((m, b["tre_um"] >= RIGHT_UM))
    O = []
    for c in cases:
        r = ours[c]
        e = r["ladder"][0]["eps"] if r["ladder"] else -np.inf
        O.append((e, r["tre_um"] >= RIGHT_UM))

    def curve(pairs):
        pairs = sorted(pairs, key=lambda p: -p[0]); xs, ys = [], []
        for k in range(1, len(pairs) + 1):
            xs.append(k / len(pairs)); ys.append(np.mean([w for _, w in pairs[:k]]))
        return np.array(xs), np.array(ys), pairs
    return curve(L), curve(O), cases


def main():
    M = {}
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 1.9), gridspec_kw={"wspace": 0.35})
    for ax, tag, name in zip(axs, ("GTa", "GTb"), ("gt1", "gt2")):
        ours, local = merged(f"{name}h2"), merged(f"{name}local")
        if not ours:
            continue
        M.update(ladder_stats(ours, tag))
        if local:
            (lx, ly, lp), (ox, oy, op), cases = refusal_curves(ours, local)
            ax.plot(lx * 100, ly * 100, color=ORANGE, lw=1.6, label="local search + far check")
            ax.plot(ox * 100, oy * 100, color=BLUE, lw=1.6, label="FLiPAlign, ranked by $\\varepsilon$")
            k = sum(1 for e, _ in op if e > 0)
            if k:
                ax.plot([ox[k - 1] * 100], [oy[k - 1] * 100], "o", color=BLUE, ms=5, mec=SURF)
            # local search at the same share answered as FLiPAlign's certificate
            if k:
                M[f"{tag}localwrongatcov"] = f"{ly[k - 1] * 100:.0f}"
                M[f"{tag}localwrongnum"] = int(round(ly[k - 1] * k))
            M[f"{tag}localwrongall"] = int(round(ly[-1] * len(ly)))
            M[f"{tag}localcases"] = len(cases)
            ax.set_title("first 24 cases" if name == "gt1" else "72 new-tissue cases", fontsize=8, loc="left")
            ax.set_xlabel("cases answered (%)"); ax.set_xlim(0, 100); ax.set_ylim(bottom=-2)
            ax.grid(axis="y", color=GRID, lw=0.6)
    axs[0].set_ylabel("wrong among answered (%)")
    axs[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.savefig("figures/refusal.pdf", bbox_inches="tight"); fig.savefig("figures/refusal.png", dpi=220, bbox_inches="tight")
    h4 = merged("gt1h4")
    if h4:
        M.update(ladder_stats(h4, "GTf"))
    dl = merged("dlpfc66")
    if dl:
        cert = [r["layer_acc"] for r in dl.values() if r["certified"]]
        ref = [r["layer_acc"] for r in dl.values() if not r["certified"]]
        M.update({"DLFn": len(dl), "DLFcert": len(cert), "DLFcertMed": f"{np.median(cert):.2f}" if cert else "--",
                  "DLFcertMin": f"{min(cert):.2f}" if cert else "--", "DLFcertBad": sum(v < 0.4 for v in cert),
                  "DLFref": len(ref), "DLFrefBad": sum(v < 0.4 for v in ref),
                  "DLFrefMed": f"{np.median(ref):.2f}" if ref else "--"})
    open("tables/new_numbers.tex", "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
    return M


if __name__ == "__main__":
    print(json.dumps(main(), indent=0))
