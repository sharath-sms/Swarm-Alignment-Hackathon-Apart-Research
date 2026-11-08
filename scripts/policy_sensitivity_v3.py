"""Review #3 item 4: interval coverage for the critical mass under the BALANCED design, with design-preserving
(stratified) uncertainty methods. K in {12, 48}; K/6 samples per (pair, state); option order random per sample.
Methods (nominal 90% percentile intervals, `draws` draws each):
  pooled_mle_boot   : binomial(K, pooled MLE) per state  (the approximation flagged by the reviewer)
  strat_mle_boot    : per-pair binomial(K/6, pair MLE), averaged over pairs  (parametric bootstrap preserving design)
  strat_jeffreys    : per-pair Beta(k+.5, K/6-k+.5) draws, averaged over pairs
Coverage of the exact pooled critical mass over `reps` independent extraction replicates, with Clopper-Pearson CIs.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import beta
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.emulator import simulate, t50


def main(name, reps=40, draws=20):
    js = json.loads(Path(f"results/policies_local/{name}.json").read_text()); Pjo = np.array(js["P"])
    truth_P = Pjo.mean(axis=(0, 1))
    def flips(P, cms, R, seed):
        return np.array([np.mean([t50(simulate(P, 24, cm, max_rounds=150, seed=seed + 1000 * cm + s)["f"]) is not None for s in range(R)]) for cm in cms])
    def pc_of(c, cms):
        for i in range(len(c) - 1):
            if c[i] < 0.5 <= c[i + 1]: return (cms[i] + (0.5 - c[i]) / (c[i + 1] - c[i] + 1e-12)) / 24
        return cms[0] / 24 if c[0] >= 0.5 else cms[-1] / 24
    full = list(range(1, 13)); truth = pc_of(flips(truth_P, full, 400, 1), full)
    c0 = int(round(truth * 24)); cms = list(range(max(1, c0 - 4), min(13, c0 + 5)))   # window around threshold
    rng = np.random.default_rng(7); res = {"name": name, "truth_pc": truth, "window_cms": cms, "K": {}}
    print(f"{name}: truth {truth:.4f}, window {cms}", flush=True)
    for K in (12, 48):
        m = K // 6; cov = {"pooled_mle_boot": [], "strat_mle_boot": [], "strat_jeffreys": []}
        for r in range(reps):
            kb = np.zeros((6, 1024))
            for j in range(6):
                for _ in range(m):
                    o = rng.integers(0, 2, 1024); kb[j] += rng.random(1024) < Pjo[j, o, np.arange(1024)]
            pooled = kb.sum(0) / K; pj = kb / m
            gens = {"pooled_mle_boot": lambda: rng.binomial(K, pooled) / K,
                    "strat_mle_boot": lambda: (rng.binomial(m, pj) / m).mean(0),
                    "strat_jeffreys": lambda: rng.beta(kb + .5, m - kb + .5).mean(0)}
            for meth, g in gens.items():
                v = np.array([pc_of(flips(g(), cms, 80, 3000 + 37 * d + r), cms) for d in range(draws)])
                lo, hi = np.percentile(v, [5, 95]); cov[meth].append(bool(lo <= truth <= hi))
        res["K"][K] = {}
        for meth, v in cov.items():
            k, n = int(np.sum(v)), len(v)
            res["K"][K][meth] = dict(covered=f"{k}/{n}", rate=k / n, cp95=[float(beta.ppf(.025, k, n - k + 1)) if k else 0.0,
                                                                         float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.0])
        print(f"{name} K={K}: " + " | ".join(f"{m_} {d['covered']} {np.round(d['cp95'], 2).tolist()}" for m_, d in res["K"][K].items()), flush=True)
    Path(f"results/sensitivity/v3_coverage_{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
