"""Preregistered Test 3 analysis (frozen before data; see research/prereg_test3.md, published to GitHub before runs).
Runs: results/runs/live_test3/deepseek-v4-flash__N24__cm{cm}__s{seed}__w{A}{B}.json (new targets X/Y, both directions).
Predictions: results/predictions/test3_deepseek-v4-flash.json (MLE parametric bootstrap PIs; horizon 100 rounds).
H6 (primary, directional): pooled over the shared cm levels, the direction the emulator predicts to flip more often
    (higher mean predicted flip probability) flips more often live; one-sided Fisher exact, alpha = 0.05.
H7 (numerical): Brier score of the frozen point predictions is lower than that of a leave-one-out constant reference;
    cell-stratified paired bootstrap (5,000) 95% CI of (Brier_ref - Brier_pred) excludes 0.
H8 (interval): every (direction, cm) cell's live flip count lies inside its 95% predictive interval (reported jointly).
"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.emulator import t50

M = "deepseek-v4-flash"; rng = np.random.default_rng(0)
pred = json.loads(Path(f"results/predictions/test3_{M}.json").read_text()); MR = pred["max_rounds"]
P = {(d, r["cm"]): r for d, v in pred["dirs"].items() for r in v["rows"]}
runs = []
for f in glob.glob(f"results/runs/live_test3/{M}__N24__cm*__s*__w*.json"):
    r = json.load(open(f)); d = "".join(r["config"]["words"]); fs = np.array([x["f"] for x in r["rounds"]])
    t = t50(fs[:MR]); runs.append(dict(d=d, cm=r["config"]["cm"], y=int(t is not None)))
res = {"n_runs": len(runs), "cells": {}}
by = defaultdict(list)
for r in runs: by[(r["d"], r["cm"])].append(r["y"])
for k, v in sorted(by.items()):
    lo, hi = P[k]["k_flips_95PI"]; res["cells"][f"{k[0]}_cm{k[1]}"] = dict(k=int(sum(v)), n=len(v), pred=P[k]["flip_point"], PI=[lo, hi], inside=bool(lo <= sum(v) <= hi))
dirs = list(pred["dirs"]); easy = max(dirs, key=lambda d: np.mean([r["flip_point"] for r in pred["dirs"][d]["rows"]])); hard = [d for d in dirs if d != easy][0]
ke = sum(r["y"] for r in runs if r["d"] == easy); ne = sum(1 for r in runs if r["d"] == easy)
kh = sum(r["y"] for r in runs if r["d"] == hard); nh = sum(1 for r in runs if r["d"] == hard)
res["H6"] = dict(predicted_easier=easy, easier=f"{ke}/{ne}", harder=f"{kh}/{nh}",
                 p_one_sided=float(fisher_exact([[ke, ne - ke], [kh, nh - kh]], alternative="greater")[1]))
res["H6"]["pass"] = res["H6"]["p_one_sided"] < 0.05
y = np.array([r["y"] for r in runs], float); p = np.array([P[(r["d"], r["cm"])]["flip_point"] for r in runs])
loo = np.array([(y.sum() - y[i]) / (len(y) - 1) for i in range(len(y))])
br = lambda yy, pp: float(np.mean((pp - yy) ** 2))
cells = defaultdict(list)
for i, r in enumerate(runs): cells[(r["d"], r["cm"])].append(i)
diffs = []
for _ in range(5000):
    idx = np.concatenate([rng.choice(v, len(v)) for v in cells.values()]); diffs.append(br(y[idx], loo[idx]) - br(y[idx], p[idx]))
res["H7"] = dict(brier_pred=br(y, p), brier_loo_ref=br(y, loo), diff=br(y, loo) - br(y, p),
                 ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
res["H7"]["pass"] = res["H7"]["ci"][0] > 0
res["H8"] = dict(all_inside=all(c["inside"] for c in res["cells"].values()))
print(json.dumps(res, indent=1))
Path("results/live_analysis/test3.json").write_text(json.dumps(res, indent=1))
