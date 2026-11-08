"""Emulator predictions for the preregistered live test (exact live settings), with policy-posterior uncertainty
and the predictive distribution of the number of flips out of n live runs."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binom
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, "scripts")
from swarm_ews.emulator import load_policy, simulate, t50
from emulator_study import policy_draws
model, N, MR = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); plan = json.loads(sys.argv[4])  # {"cm": n_runs}
js = json.loads(Path(f"results/policies/{model}.json").read_text()); P0 = load_policy(js)
draws = policy_draws(js, 40, np.random.default_rng(1))
rows = []
for cm, n in plan.items():
    cm = int(cm)
    def stats(P, R, seed):
        T = [t50(simulate(P, N, cm, max_rounds=MR, seed=seed + s)["f"]) for s in range(R)]
        tip = [(t + 1) * N / 2 for t in T if t is not None]
        return len(tip) / R, (float(np.median(tip)) if tip else None)
    f0, m0 = stats(P0, 1000, 123_000)
    fl, md = zip(*[stats(P, 200, 456_000) for P in draws])
    # predictive distribution of k flips out of n: mixture over policy draws
    pk = np.mean([binom.pmf(np.arange(n + 1), n, f) for f in fl], axis=0)
    cdf = np.cumsum(pk); lo = int(np.searchsorted(cdf, 0.025)); hi = int(np.searchsorted(cdf, 0.975))
    mdv = [m for m in md if m is not None]
    rows.append(dict(cm=cm, p=cm / N, n_live=n, flip_point=f0, flip_PI=[float(np.percentile(fl, 2.5)), float(np.percentile(fl, 97.5))],
                     k_flips_95PI=[lo, hi], median_T_point=m0,
                     median_T_PI=[float(np.percentile(mdv, 2.5)), float(np.percentile(mdv, 97.5))] if mdv else None))
    print(rows[-1], flush=True)
Path("results/predictions").mkdir(parents=True, exist_ok=True)
Path(f"results/predictions/{model}__N{N}__mr{MR}.json").write_text(json.dumps(dict(model=model, N=N, max_rounds=MR, rows=rows), indent=1))
