"""Section-level bootstrap of the held-out placement counts (2026-11-03, review finding 8).
Cases from one source section are not independent, so resample source sections (6 per set, 8 cases each) with
replacement, 10,000 times. For each resample: FLiPAlign right answers (flip searched, nothing handed over) and the
best tool's right answers, with the best tool and start (as given or flip, + angle, + position) re-chosen in each resample (favours the tools).
Writes ../server_results/boot_sections.json and tables/boot_numbers.tex."""
import json
import numpy as np
from make_results import load, manifest
from make_results3 import tool_best, GOOD

rng = np.random.default_rng(0); B = 10000; out = {}
TB = {s: [[f"{s}base", f"{s}base3", f"{s}slat", f"{s}spateo", f"{s}cast", f"{s}stalign", f"rc_{s}stalign"],
          [f"rs_{s}base", f"rs_{s}spateo", f"rs_{s}cast", f"rs_{s}stalign", f"rc_rs_{s}cpd", f"rc_rs_{s}stalign"],
          [f"rp_{s}base", f"rp_{s}spateo", f"rp_{s}cast", f"rc_rp_{s}cpd", f"rc_rp_{s}stalign"]] for s in ("gt3", "gt4")}
per = {}
for s, op in (("gt3", "gt3h2"), ("gt4", "gt4h2")):
    man = manifest(s); O = load(op); cs = sorted(c for c in man if c in O)
    sec = {c: man[c]["section"] for c in cs}
    ours = {c: int(GOOD(c, O[c]["tre_um"])) for c in cs}
    tools = {}
    for arm, (pats, extra) in enumerate(((TB[s][0], [f"{s}fo_base", f"{s}fo_spateo", f"{s}fo_cast"]), (TB[s][1], None), (TB[s][2], None))):
        for t, v in tool_best(pats, cs, GOOD, extra).items():     # arms: flip handed; + angle; + position
            if v:
                tools[(t, arm)] = {c: int(GOOD(c, e)) for c, e in v[2].items()}
    per[s] = (cs, sec, ours, tools)
def stat(pick):
    o = b = 0
    for s, secs_ in pick.items():
        cs, sec, ours, tools = per[s]
        cases = [c for x in secs_ for c in cs if sec[c] == x]
        o += sum(ours[c] for c in cases)
        b += max(sum(d.get(c, 0) for c in cases) for d in tools.values())
    return o, b
full = {s: sorted(set(per[s][1].values())) for s in per}
o0, b0 = stat(full); out["observed"] = {"flipalign": o0, "best_tool": b0, "ratio": round(o0 / b0, 3)}
for name, sets in (("both", ("gt3", "gt4")), ("second", ("gt4",))):
    R = []
    for _ in range(B):
        pick = {s: list(rng.choice(full[s], len(full[s]), replace=True)) for s in sets}
        o, b = stat(pick); R.append((o, b))
    R = np.array(R); ratio = R[:, 0] / np.maximum(R[:, 1], 1)
    o_, b_ = stat({s: full[s] for s in sets})
    out[name] = {"flipalign": o_, "best_tool": b_, "ratio": round(o_ / b_, 3),
                 "ratio_ci": [round(float(np.percentile(ratio, 2.5)), 2), round(float(np.percentile(ratio, 97.5)), 2)],
                 "p_flip_not_better": round(float(np.mean(R[:, 0] <= R[:, 1])), 4)}
json.dump(out, open("../server_results/boot_sections.json", "w"), indent=1); print(json.dumps(out, indent=1))
