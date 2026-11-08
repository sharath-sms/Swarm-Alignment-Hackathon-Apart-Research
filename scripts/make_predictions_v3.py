"""Frozen predictions for preregistered Test 3 (new model, new targets).
Policy uncertainty: parametric bootstrap from the unsmoothed per-state MLE (binomial(n_s, p̂_s)/n_s), which the estimator
study found far better calibrated than Jeffreys posterior draws. Monte Carlo: 1,000 emulator runs per point prediction,
200 per bootstrap draw. Also chooses cm levels near each direction's predicted threshold if --choose is given.
Usage: python scripts/make_predictions_v3.py gemma-4-31b-it XY YX --choose 3 --runs-per-cell 6
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binom
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.emulator import simulate, t50

def load(model, d):
    js = json.loads(Path(f"results/policies/{model}__{d}.json").read_text())
    keys = ["".join(str((i // 4 ** (4 - k)) % 4) for k in range(5)) for i in range(1024)]
    p = np.array([js["policy"][k]["P_B"] for k in keys], float); n = np.array([js["policy"][k]["n"] for k in keys], int)
    return p, n

MR = 150
def flip(P, cm, R, seed):
    T = [t50(simulate(P, 24, cm, max_rounds=MR, seed=seed + s)["f"]) for s in range(R)]
    tip = [(t + 1) * 12 for t in T if t is not None]
    return len(tip) / R, (float(np.median(tip)) if tip else None)

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("dirs", nargs="+")
ap.add_argument("--choose", type=int, default=3); ap.add_argument("--runs-per-cell", type=int, default=6)
ap.add_argument("--draws", type=int, default=40); ap.add_argument("--max-rounds", type=int, default=150)
ap.add_argument("--shared", type=int, default=0, help="choose this many cm levels shared by all directions, maximising the spread of predictions")
a = ap.parse_args(); MR = a.max_rounds
rng = np.random.default_rng(11); out = {"model": a.model, "N": 24, "max_rounds": MR, "method": "MLE parametric bootstrap", "dirs": {}}
curves = {d: {cm: flip(load(a.model, d)[0], cm, 400, 50_000 + 1000 * cm)[0] for cm in range(1, 13)} for d in a.dirs}
shared = None
if a.shared:
    spread = {cm: max(curves[d][cm] for d in a.dirs) - min(curves[d][cm] for d in a.dirs) for cm in range(1, 13)}
    shared = sorted(sorted(spread, key=lambda c: -spread[c])[:a.shared]); out["shared_levels"] = shared; out["spread"] = spread
    print("spread by cm", {c: round(v, 2) for c, v in spread.items()}, "-> shared", shared, flush=True)
for d in a.dirs:
    p, n = load(a.model, d)
    curve = curves[d]
    # choose levels with predicted flip probability spread across (0.1, 0.95)
    targets = np.linspace(0.15, 0.9, a.choose)
    cands = sorted(curve, key=lambda c: curve[c])
    chosen = shared if shared else sorted({min(curve, key=lambda c: abs(curve[c] - t)) for t in targets})
    draws = [np.where(n > 0, rng.binomial(np.maximum(n, 1), p) / np.maximum(n, 1), p) for _ in range(a.draws)]
    rows = []
    for cm in chosen:
        f0, m0 = flip(p, cm, 1000, 70_000 + 1000 * cm)
        fd = [flip(P, cm, 200, 90_000 + 1000 * cm + 7 * i)[0] for i, P in enumerate(draws)]
        k = a.runs_per_cell; pk = np.mean([binom.pmf(np.arange(k + 1), k, f) for f in fd], axis=0); cdf = np.cumsum(pk)
        rows.append(dict(cm=cm, p=cm / 24, n_live=k, flip_point=f0, flip_PI=[float(np.percentile(fd, 2.5)), float(np.percentile(fd, 97.5))],
                         k_flips_95PI=[int(np.searchsorted(cdf, 0.025)), int(np.searchsorted(cdf, 0.975))], median_T_point=m0))
        print(d, rows[-1], flush=True)
    pc = None
    for c in range(1, 12):
        if curve[c] < 0.5 <= curve[c + 1]: pc = (c + (0.5 - curve[c]) / (curve[c + 1] - curve[c])) / 24; break
    out["dirs"][d] = dict(curve=curve, pc_point=pc, rows=rows)
    print(d, "curve", {c: round(v, 2) for c, v in curve.items()}, "p_c", pc, flush=True)
Path("results/predictions").mkdir(parents=True, exist_ok=True)
Path(f"results/predictions/test3_{a.model}.json").write_text(json.dumps(out, indent=1))
