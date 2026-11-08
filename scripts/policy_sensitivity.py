"""How reliable are emulator-based critical-mass estimates? (exact local policies as ground truth)

Ground truth: exact per-(pair, order) policies from logprobs. Live protocol randomises option order per call, so the
per-pair effective policy is P_j(s) = mean_o P[j, o, s].
  1. True flip curve per word pair j, and the counterbalanced MIXTURE flip curve (what a live experiment that
     counterbalances pairs measures): flip_mix(cm) = mean_j flip_j(cm).
  2. POOLED emulator: P_pool(s) = mean_j P_j(s) -> flip_pool(cm). Pooling bias = flip_pool - flip_mix.
  3. SAMPLING: re-run our extraction procedure in silico (MLE and Jeffreys-smoothed estimators) — K samples per state cycling pairs (pooled) or per pair —
     B replicates -> distribution of estimated p_c (flip=0.5) vs truth, for K in {4, 8, 16, 48, 128}.
Output: results/sensitivity/<name>.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.emulator import simulate, t50  # noqa: E402

N, MR, R = 24, 150, 300
CMS = list(range(1, 11))


def flip_curve(P, R=R, seed=0):
    return np.array([np.mean([t50(simulate(P, N, cm, max_rounds=MR, seed=seed + 1000 * cm + s)["f"]) is not None
                              for s in range(R)]) for cm in CMS])


def pc_of(curve):
    for i in range(len(curve) - 1):
        if curve[i] < 0.5 <= curve[i + 1]:
            return (CMS[i] + (0.5 - curve[i]) / (curve[i + 1] - curve[i] + 1e-12)) / N
    return CMS[0] / N if curve[0] >= 0.5 else np.nan


def main(path, B=20):
    js = json.loads(Path(path).read_text()); name = js["name"]
    Pjo = np.array(js["P"]); Pj = Pjo.mean(axis=1)          # (6, 1024)
    res = {"name": name, "pairs": js["pairs"], "per_pair": {}}
    curves = []
    for j, pr in enumerate(js["pairs"]):
        c = flip_curve(Pj[j], seed=10 * j); curves.append(c)
        res["per_pair"]["".join(pr)] = dict(flip=c.tolist(), p_c=float(pc_of(c)),
                                             P_switch_after_k_B=[float(Pj[j][int("0" * (5 - k) + "1" * k, 4)]) for k in range(1, 6)])
        print(f"{name} pair {''.join(pr)}: p_c={pc_of(c):.3f} flips={np.round(c, 2).tolist()}", flush=True)
    mix = np.mean(curves, axis=0); pool = flip_curve(Pj.mean(axis=0), seed=99)
    res["mixture"] = dict(flip=mix.tolist(), p_c=float(pc_of(mix)))
    res["pooled"] = dict(flip=pool.tolist(), p_c=float(pc_of(pool)),
                         max_abs_flip_error_vs_mixture=float(np.max(np.abs(pool - mix))))
    print(f"MIXTURE (live counterbalanced) p_c={pc_of(mix):.3f} flips={np.round(mix, 2).tolist()}")
    print(f"POOLED emulator p_c={pc_of(pool):.3f} flips={np.round(pool, 2).tolist()}  max|err|={res['pooled']['max_abs_flip_error_vs_mixture']:.2f}")
    rng = np.random.default_rng(0); res["sampling"] = {}
    for K in (4, 8, 16, 48, 128):
        est_pool, est_pair, est_pool_mle, est_pair_mle = [], [], [], []
        for b in range(B):
            # pooled extraction: sample k cycles through pairs (as in extract_policy.py)
            cnt = np.zeros(1024); n = np.zeros(1024)
            for k in range(K):
                j = k % 6; o = rng.integers(0, 2); x = rng.random(1024) < Pjo[j, o]; cnt += x; n += 1
            est_pool.append(pc_of(flip_curve((cnt + 0.5) / (n + 1), R=150, seed=500 + b)))
            est_pool_mle.append(pc_of(flip_curve(cnt / n, R=150, seed=500 + b)))
            # per-pair extraction for pair 0, compared to pair-0 truth
            cnt = np.zeros(1024)
            for k in range(K):
                o = rng.integers(0, 2); cnt += rng.random(1024) < Pjo[0, o]
            est_pair.append(pc_of(flip_curve((cnt + 0.5) / (K + 1), R=150, seed=700 + b)))
            est_pair_mle.append(pc_of(flip_curve(cnt / K, R=150, seed=700 + b)))
        q = lambda v: [float(np.nanpercentile(v, x)) for x in (2.5, 50, 97.5)]
        res["sampling"][K] = dict(pooled_pc_jeffreys=q(est_pool), pair0_pc_jeffreys=q(est_pair),
                                  pooled_pc_mle=q(est_pool_mle), pair0_pc_mle=q(est_pair_mle),
                                  truth_pooled=res["pooled"]["p_c"], truth_pair0=res["per_pair"]["".join(js["pairs"][0])]["p_c"])
        print(f"K={K:3d}: pooled MLE {np.round(q(est_pool_mle), 3)} Jeffreys {np.round(q(est_pool), 3)} (truth {res['pooled']['p_c']:.3f}); "
              f"per-pair MLE {np.round(q(est_pair_mle), 3)} Jeffreys {np.round(q(est_pair), 3)} (truth {res['sampling'][K]['truth_pair0']:.3f})", flush=True)
    Path("results/sensitivity").mkdir(parents=True, exist_ok=True)
    Path(f"results/sensitivity/{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], B=int(sys.argv[2]) if len(sys.argv) > 2 else 20)
