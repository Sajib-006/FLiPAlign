"""Atlas-free consistency score for brain 1 (Allen 638850), same labellings as stack_ctrl.py (variants, tables, forest, caps).
Usage: python3 runs_stacks/stack_consistency.py STACK -> results/stackcons_{i}.json  (also holds NMI/ARI vs atlas for cross-check)"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, sys, time
sys.path.insert(0, ".")
import stack_jlp as J
import stack_ctrl as CT
import stack_brain2 as B2

if __name__ == "__main__":
    i = int(sys.argv[1]); t0 = time.time(); p, secs = J.sections()
    S = [J.load_section(p, sec, secs.index(sec)) for sec in secs[i * J.KS:(i + 1) * J.KS]]; del p
    res = {"stack": i, "runs": {}}
    for name, V in CT.variants(S).items():
        res["runs"][name] = B2.run_stack(V)
        print(name, json.dumps({k: (v["nmi"], v["cons"]) for k, v in res["runs"][name].items()}), f"{time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(f"results/stackcons_{i}.json", "w"), indent=1)
    res["seconds"] = round(time.time() - t0); json.dump(res, open(f"results/stackcons_{i}.json", "w"), indent=1)
    print("DONE", flush=True)
