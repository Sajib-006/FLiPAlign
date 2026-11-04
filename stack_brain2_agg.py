import json, sys, numpy as np
R = "server_results/stack/"
def load(prefix):
    return [json.load(open(f"{R}{prefix}_{i}.json")) for i in range(6)]
def rows(res):
    out = []
    for r in res:
        for v, m in r["runs"].items():
            m = dict(m); one = max(("chain", "forest_d2", "star"), key=lambda k: m[k]["nmi"])
            m["best1"] = m[one]; m["split"] = m["cap2_fromC"]; m["cap1"] = m["cap1_fromC"]; out.append((r["stack"], v, m))
    return out
ms = lambda a: f"{np.mean(a):.3f} ± {np.std(a, ddof=1) / np.sqrt(len(a)):.3f}"
def report(name, res):
    rs = rows(res); L = [f"## {name}  (n = {len(rs)} stack-versions)", ""]
    L += ["| method | NMI | ARI | classes | consistency NMI | same-class rate | kappa |", "|---|---|---|---|---|---|---|"]
    for k in ("chain", "forest_d2", "star", "best1", "cap1", "split", "cap1_fromW", "cap2_fromW"):
        g = lambda f: [m[k][f] for _, _, m in rs]
        L.append(f"| {k} | {ms(g('nmi'))} | {ms(g('ari'))} | {np.mean(g('n_classes')):.1f} | {ms(g('cons'))} | {ms(g('same'))} | {ms(g('kappa'))} |")
    sp = np.array([m["split"]["nmi"] for _, _, m in rs])
    for ref in ("best1", "cap1"):
        b = np.array([m[ref]["nmi"] for _, _, m in rs])
        L.append(f"\nsplit vs {ref}: wins {int((sp > b + 1e-9).sum())}/{len(rs)}, ties {int((abs(sp - b) <= 1e-9).sum())}, median relative NMI gain {np.median(sp / b - 1) * 100:.1f}%")
    L += ["", "per version (mean NMI over 6 stacks): " + "; ".join(
        f"{v}: best1 {np.mean([m['best1']['nmi'] for _, vv, m in rs if vv == v]):.3f}, cap1 {np.mean([m['cap1']['nmi'] for _, vv, m in rs if vv == v]):.3f}, split {np.mean([m['split']['nmi'] for _, vv, m in rs if vv == v]):.3f}" for v in res[0]["runs"])]
    L += ["", "per stack (mean NMI over 5 versions): " + "; ".join(
        f"s{s}: best1 {np.mean([m['best1']['nmi'] for ss, _, m in rs if ss == s]):.3f}, split {np.mean([m['split']['nmi'] for ss, _, m in rs if ss == s]):.3f}" for s in range(6))]
    return "\n".join(L)
if __name__ == "__main__":
    print(report("Brain 2 (Zhuang-ABCA-1)", load("stackb2")))
    print(); print(report("Brain 1 (Allen 638850), recomputed with consistency", load("stackcons")))
