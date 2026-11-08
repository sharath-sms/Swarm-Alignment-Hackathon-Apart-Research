"""Confirmatory v2 analysis (preregistered addendum in research/prereg.md): token-conditional emulator test.
Live runs: results/runs/live_token/deepseek-v4-flash__N24__cm{cm}__s{seed}__w{A}{B}.json, directions MQ (push Q) and
QM (push M). Predictions: results/predictions/deepseek-v4-flash__{AB}__N24__mr150.json.
H5 (primary): pooled over cm in {3,4}, flips(push M) > flips(push Q); one-sided Fisher exact, alpha 0.05.
H1': each (direction, cm) live flip count inside its 95% predictive interval.
H2': median-T50 live/emulator ratio per cell with >=3 flips (bootstrap CI) vs equivalence margin [0.67, 1.5].
"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.emulator import load_policy, simulate, t50

M, N = "deepseek-v4-flash", 24
out = {"cells": {}}
live = defaultdict(list)
for f in glob.glob(f"results/runs/live_token/{M}__N{N}__cm*__s*__w*.json"):
    r = json.load(open(f)); A, B = r["config"]["words"]; cm = r["config"]["cm"]
    t = t50(np.array([x["f"] for x in r["rounds"]]))
    live[(A + B, cm)].append(None if t is None else (t + 1) * N / 2)
for d in ("MQ", "QM"):
    pr = {r["cm"]: r for r in json.load(open(f"results/predictions/{M}__{d}__N24__mr150.json"))["rows"]}
    P = load_policy(json.load(open(f"results/policies/{M}__{d}.json")))
    for cm in sorted(pr):
        L = live.get((d, cm), []); k = sum(t is not None for t in L); lo, hi = pr[cm]["k_flips_95PI"]
        cell = dict(direction=d, cm=cm, n=len(L), k=k, PI=[lo, hi], H1=bool(lo <= k <= hi) if L else None,
                    flip_pred=pr[cm]["flip_point"])
        Tl = [t for t in L if t is not None]
        if len(Tl) >= 3:
            em = [t50(simulate(P, N, cm, max_rounds=150, seed=888_000 + s)["f"]) for s in range(1000)]
            Te = [(t + 1) * N / 2 for t in em if t is not None]
            rng = np.random.default_rng(0)
            rat = [np.median(rng.choice(Tl, len(Tl))) / np.median(rng.choice(Te, len(Te))) for _ in range(2000)]
            cell["T_ratio"] = [float(np.median(Tl) / np.median(Te)), float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5))]
        out["cells"][f"{d}_cm{cm}"] = cell
        print(cell)
km = sum(t is not None for cm in (3, 4) for t in live.get(("QM", cm), [])); nm = sum(len(live.get(("QM", cm), [])) for cm in (3, 4))
kq = sum(t is not None for cm in (3, 4) for t in live.get(("MQ", cm), [])); nq = sum(len(live.get(("MQ", cm), [])) for cm in (3, 4))
if nm and nq:
    pv = fisher_exact([[km, nm - km], [kq, nq - kq]], alternative="greater")[1]
    out["H5"] = dict(push_M=f"{km}/{nm}", push_Q=f"{kq}/{nq}", p_one_sided=float(pv), pass_=bool(pv < 0.05))
    print("H5:", out["H5"])
out["H1_all"] = all(c["H1"] for c in out["cells"].values() if c["H1"] is not None)
print("H1' all cells inside PI:", out["H1_all"])
Path("results/live_analysis/live_token__deepseek-v4-flash.json").write_text(json.dumps(out, indent=1))
