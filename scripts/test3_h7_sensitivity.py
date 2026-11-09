"""POST-HOC sensitivity for Test 3 H7 (review #4): the preregistered H7 result is unchanged; this adds
(a) a direction-only comparator (push Y always flips, push X never), (b) a bootstrap that RECOMPUTES the
leave-one-out reference inside each replicate, and (c) a parametric bootstrap that draws each cell's flip probability
from Beta(k+1/2, n-k+1/2) before drawing outcomes, so cells with identical outcomes still carry uncertainty."""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from swarm_ews.emulator import t50
rng = np.random.default_rng(1)
pred = json.load(open("results/predictions/test3_deepseek-v4-flash.json")); MR = pred["max_rounds"]
P = {(d, r["cm"]): r["flip_point"] for d, v in pred["dirs"].items() for r in v["rows"]}
runs = []
for f in glob.glob("results/runs/live_test3/*.json"):
    r = json.load(open(f)); d = "".join(r["config"]["words"])
    runs.append((d, r["config"]["cm"], int(t50(np.array([x["f"] for x in r["rounds"]])[:MR]) is not None)))
y = np.array([r[2] for r in runs], float); p = np.array([P[(r[0], r[1])] for r in runs])
direction = np.array([1.0 if r[0] == "XY" else 0.0 for r in runs])
br = lambda yy, pp: float(np.mean((pp - yy) ** 2))
loo = lambda yy: (yy.sum() - yy) / (len(yy) - 1)
cells = defaultdict(list)
for i, r in enumerate(runs): cells[(r[0], r[1])].append(i)
res = {"brier_emulator": br(y, p), "brier_direction_only": br(y, direction), "brier_loo_reference": br(y, loo(y))}
# (b) nonparametric, reference recomputed inside each replicate
d_ref, d_dir = [], []
for _ in range(5000):
    idx = np.concatenate([rng.choice(v, len(v)) for v in cells.values()]); yy = y[idx]
    d_ref.append(br(yy, loo(yy)) - br(yy, p[idx])); d_dir.append(br(yy, direction[idx]) - br(yy, p[idx]))
# (c) parametric with Beta-smoothed cell probabilities (boundary cells keep uncertainty)
e_ref, e_dir = [], []
for _ in range(5000):
    yy = y.copy()
    for k_, v in cells.items():
        th = rng.beta(y[v].sum() + .5, len(v) - y[v].sum() + .5); yy[v] = (rng.random(len(v)) < th).astype(float)
    e_ref.append(br(yy, loo(yy)) - br(yy, p)); e_dir.append(br(yy, direction) - br(yy, p))
ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
res.update(ref_minus_emulator_recomputed_ci=ci(d_ref), ref_minus_emulator_parametric_ci=ci(e_ref),
           direction_minus_emulator_recomputed_ci=ci(d_dir), direction_minus_emulator_parametric_ci=ci(e_dir))
print(json.dumps(res, indent=1)); Path("results/live_analysis/test3_h7_sensitivity.json").write_text(json.dumps(res, indent=1))
