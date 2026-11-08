"""R3 (independent review, issue 5): estimator study with a FIXED target mixture.

Fixes the confound in policy_sensitivity.py (sample k cycled through 6 pairs, so K=4/8 changed the pair weights as
well as precision). Here K in {6, 12, 24, 48, 120} (multiples of 6): every pair gets exactly K/6 samples per state,
so the pooled target (equal pair weights; option order randomised per sample as in live play) is fixed across K.
Truth: emulator critical mass of the exact pooled policy (mean over pairs and orders).
For each K, `reps` independent extraction replicates; estimators:
  MLE (n_B/n), Jeffreys posterior mean ((n_B+.5)/(n+1)), Laplace ((n_B+1)/(n+2)).
Reported: bias, RMSE, and (at K in {12, 48}) coverage of two 90% intervals for p_c:
  (i) MLE parametric bootstrap (resample counts from MLE probabilities) — cannot represent unseen transitions;
  (ii) Jeffreys posterior draws.
p_c is a finite-horizon quantity: smallest committed fraction with P(T50 <= 150 rounds) >= 0.5, N=24.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.emulator import simulate, t50  # noqa: E402

N, MR, CMS = 24, 150, list(range(1, 12))


def flip_curve(P, R, seed):
    return np.array([np.mean([t50(simulate(P, N, cm, max_rounds=MR, seed=seed + 1000 * cm + s)["f"]) is not None
                              for s in range(R)]) for cm in CMS])


def pc_of(c):
    for i in range(len(c) - 1):
        if c[i] < 0.5 <= c[i + 1]:
            return (CMS[i] + (0.5 - c[i]) / (c[i + 1] - c[i] + 1e-12)) / N
    return CMS[0] / N if c[0] >= 0.5 else np.nan


def main(name, reps=30, draws=20):
    js = json.loads(Path(f"results/policies_local/{name}.json").read_text()); Pjo = np.array(js["P"])   # (6,2,1024)
    truth_P = Pjo.mean(axis=(0, 1)); truth = pc_of(flip_curve(truth_P, 400, 1))
    rng = np.random.default_rng(42); res = {"name": name, "truth_pc": truth, "K": {}}
    print(f"{name}: truth p_c = {truth:.4f}", flush=True)
    for K in (6, 12, 24, 48, 120):
        est = {"mle": [], "jeffreys": [], "laplace": []}; cov = {"mle_boot": [], "jeffreys_post": []}
        for r in range(reps):
            nb = np.zeros(1024)
            for j in range(6):
                for _ in range(K // 6):
                    o = rng.integers(0, 2, 1024)
                    nb += rng.random(1024) < Pjo[j, o, np.arange(1024)]
            Pm, Pj, Pl = nb / K, (nb + .5) / (K + 1), (nb + 1) / (K + 2)
            for k, P in (("mle", Pm), ("jeffreys", Pj), ("laplace", Pl)):
                est[k].append(pc_of(flip_curve(P, 150, 500 + r)))
            if K in (12, 48) and r < 15:
                bm = [pc_of(flip_curve(rng.binomial(K, Pm) / K, 100, 900 + d)) for d in range(draws)]
                bj = [pc_of(flip_curve(rng.beta(nb + .5, K - nb + .5), 100, 950 + d)) for d in range(draws)]
                for key, v in (("mle_boot", bm), ("jeffreys_post", bj)):
                    v = np.array(v, float); v = v[~np.isnan(v)]
                    lo, hi = np.percentile(v, [5, 95]) if len(v) else (np.nan, np.nan)
                    cov[key].append(bool(lo <= truth <= hi))
        d = {}
        for k, v in est.items():
            v = np.array(v, float); v = v[~np.isnan(v)]
            d[k] = dict(mean=float(v.mean()), bias=float(v.mean() - truth), rmse=float(np.sqrt(np.mean((v - truth) ** 2))),
                        rel_bias=float((v.mean() - truth) / truth), n=len(v))
        if cov["mle_boot"]:
            d["coverage90"] = {k: float(np.mean(v)) for k, v in cov.items()}
        res["K"][K] = d
        print(f"{name} K={K:3d}: " + " | ".join(f"{k} bias {v['bias']:+.4f} rmse {v['rmse']:.4f}" for k, v in d.items() if k != "coverage90")
              + (f" | coverage90 {d['coverage90']}" if "coverage90" in d else ""), flush=True)
    Path("results/sensitivity").mkdir(parents=True, exist_ok=True)
    Path(f"results/sensitivity/v2_{name}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
