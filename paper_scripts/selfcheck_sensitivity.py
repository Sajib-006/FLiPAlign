"""Mirror self-check threshold sensitivity and how many wrong certificates hold (supplement table tab:selfsens).
Reads the saved held-out results (results/placement). Usage: python3 selfcheck_sensitivity.py"""
import io, contextlib, sys
sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    from make_results import load, manifest, RIGHT
    import make_results2 as M2
from make_results2 import holds
def ratio_fn(prefix):
    F = load(prefix)
    return lambda c: None if F.get(c) is None else (F[c]["S_other"]/max(F[c]["O_other"],1e-9))/(F[c]["S_answer"]/max(F[c]["O_answer"],1e-9))
tot = {}
for sn in ("gt3","gt4"):
    O, K = load(f"{sn}h2"), load(f"fk_{sn}"); rf = ratio_fn(f"fc_{sn}"); cs = sorted(O)
    print(sn, "cases", len(cs))
    for th in (None, 0.90, 0.95, 0.98, 1.00):
        cert=[c for c in cs if O[c]["certified"] and (th is None or rf(c) is None or rf(c) < th)]
        right=sum(O[c]["tre_um"] < RIGHT for c in cert)
        print(f"  thresh {th}: certified {len(cert)}, right {right}, wrong {len(cert)-right}")
        tot.setdefault(th,[0,0]); tot[th][0]+=len(cert); tot[th][1]+=right
    for nm, D in (("searched, no check", O), ("known", K)):
        wrong=[c for c in cs if D[c]["certified"] and D[c]["tre_um"] >= RIGHT]
        h=[holds(D[c], K[c]) for c in wrong]
        print(f"  {nm}: wrong certificates {len(wrong)}, of which hold {sum(1 for x in h if x)}, not hold {sum(1 for x in h if x is False)}, unknown {sum(1 for x in h if x is None)}")
print("both sets", {k:(v[0],v[1]) for k,v in tot.items()})
