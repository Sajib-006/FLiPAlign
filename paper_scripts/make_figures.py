"""Paper figures, drawn straight from the result files (../server_results).
Palette: validated categorical order (blue, orange, aqua, yellow, magenta, green); ours is blue.
Every method is named on the axis (identity is never color alone); certified vs refused uses
filled vs hollow markers as well as color."""
import collections, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "../server_results"; F = "figures"; os.makedirs(F, exist_ok=True)
load = lambda f: json.load(open(os.path.join(R, f)))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
rng = np.random.default_rng(0)


def gt_data():
    ours = {r["case"]: r for r in load("results_gt_ours.json")}
    for r in load("results_gt_ours_recap.json"):
        ours[r["case"]] = r
    cert = {c: bool(r.get("unique_shift30") or r.get("unique_shift60") or r.get("unique_rotation")) for c, r in ours.items()}
    tab = collections.defaultdict(dict)
    for r in load("results_gt_baselines.json"):
        if "error" not in r:
            tab[(r["method"], r["fit"])][r["case"]] = r["tre_um"]
    names = {"local_50": "Local search\n(our score, 50 starts)", "paste2_s0.5": "PASTE2", "paste2_s0.7": "PASTE2",
             "paste_a0.1": "PASTE", "slat_k20": "SLAT", "cpd_8": "CPD"}
    best = {}
    for (m, f), d in tab.items():
        n = names.get(m, m); med = np.median(list(d.values()))
        if n not in best or med < best[n][0]:
            best[n] = (med, d)
    return ours, cert, best


def fig_benchmark():
    ours, cert, best = gt_data()
    rows = [("FLiPAlign", {c: r["tre_um"] for c, r in ours.items()})] + \
           [(n, d) for n, (m, d) in sorted(best.items(), key=lambda kv: kv[1][0])]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for y, (name, d) in enumerate(rows):
        col = PAL[y % len(PAL)]
        v = np.array([d[c] for c in sorted(d)]); jit = rng.uniform(-0.18, 0.18, len(v))
        if name == "FLiPAlign":
            cs = sorted(d); filled = np.array([cert[c] for c in cs])
            ax.scatter(v[filled], y + jit[filled], s=26, color=col, lw=0, zorder=3, label="certified unique")
            ax.scatter(v[~filled], y + jit[~filled], s=26, facecolors="none", edgecolors=col, lw=1.2, zorder=3,
                       label="refused")
        else:
            ax.scatter(v, y + jit, s=18, color=col, lw=0, alpha=0.85, zorder=3)
        ax.plot([np.median(v)] * 2, [y - 0.32, y + 0.32], color=INK, lw=1.6, zorder=4)
    ax.axvline(100, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(100, len(rows) - 0.45, " 100 $\\mu$m", color=INK2, fontsize=8, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(8, 2e4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("Error per case ($\\mu$m, log scale)")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.legend(loc="upper right", frameon=False, fontsize=8, title="FLiPAlign:", title_fontsize=8,
              bbox_to_anchor=(1.0, 1.02))
    fig.tight_layout(); fig.savefig(f"{F}/benchmark.pdf"); fig.savefig(f"{F}/benchmark.png", dpi=200)


def fig_certificate():
    d = load("results_dlpfc_decide.json")
    ours, cert, _ = gt_data()
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.7))
    groups = [("certified\nunique", PAL[0], True), ("refused", PAL[0], False)]   # both are ours: blue; fill = status
    ax = axs[0]
    for x, (lab, col, is_c) in enumerate(groups):
        v = np.array([r["layer_acc"] for r in d if (r.get("eps_rotation", -1) > 0) == is_c and r.get("certified")])
        ax.scatter(x + rng.uniform(-0.2, 0.2, len(v)), v, s=20, lw=1.1,
                   **({"color": col} if is_c else {"facecolors": "none", "edgecolors": col}), zorder=3)
        ax.plot([x - 0.28, x + 0.28], [np.median(v)] * 2, color=INK, lw=1.6, zorder=4)
        ax.text(x, 1.04, f"n = {len(v)}", ha="center", fontsize=8, color=INK2)
    ax.axhline(0.4, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylim(0, 1.1)
    ax.set_ylabel("Layer score (hand-drawn layers)"); ax.set_title("DLPFC, 66 pairs", fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax = axs[1]
    for x, (lab, col, is_c) in enumerate(groups):
        v = np.array([r["tre_um"] for c, r in ours.items() if cert[c] == is_c])
        ax.scatter(x + rng.uniform(-0.2, 0.2, len(v)), v, s=20, lw=1.1,
                   **({"color": col} if is_c else {"facecolors": "none", "edgecolors": col}), zorder=3)
        ax.plot([x - 0.28, x + 0.28], [np.median(v)] * 2, color=INK, lw=1.6, zorder=4)
        ax.text(x, 3000, f"n = {len(v)}", ha="center", fontsize=8, color=INK2)
    ax.axhline(200, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_yscale("log"); ax.set_ylim(8, 5000)
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Error ($\\mu$m, log scale)"); ax.set_title("Known-truth cases, 24", fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    fig.tight_layout(); fig.savefig(f"{F}/certificate.pdf"); fig.savefig(f"{F}/certificate.png", dpi=200)


if __name__ == "__main__":
    fig_benchmark(); fig_certificate()
    print(sorted(os.listdir(F)))
