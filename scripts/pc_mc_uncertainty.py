"""Monte Carlo uncertainty of exact-policy per-pair critical masses: recompute p_c with 8 independent seed sets."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.emulator import simulate, t50
CMS = list(range(1, 12))
def pc(P, seed):
    c = [np.mean([t50(simulate(P, 24, cm, max_rounds=150, seed=seed + 1000 * cm + s)["f"]) is not None for s in range(300)]) for cm in CMS]
    for i in range(len(c) - 1):
        if c[i] < 0.5 <= c[i + 1]: return (CMS[i] + (0.5 - c[i]) / (c[i + 1] - c[i] + 1e-12)) / 24
    return np.nan
out = {}
for m in ("qwen3-8b", "gemma3-12b", "mistral-7b"):
    js = json.load(open(f"results/policies_local/{m}.json")); P = np.array(js["P"]).mean(1)
    out[m] = {}
    for j, pr in enumerate(js["pairs"]):
        v = [pc(P[j], 200_000 * (r + 1)) for r in range(8)]
        out[m]["".join(pr)] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)))
    print(m, {k: (round(v["mean"], 3), round(v["sd"], 4)) for k, v in out[m].items()}, flush=True)
Path("results/sensitivity/pc_mc_uncertainty.json").write_text(json.dumps(out, indent=1))
