"""F1 verification: null behaviour of the EWS indicators on pure independent binomial noise (no EWS by construction).

(1) Reproduce the critique's test: B=12 observations per point, 50 points, 300 reps, p flat 0.03 / rising 0.01->0.3 /
    flat 0.2 -> mean Kendall tau of the binomial-normalised variance (and AC1) indicator.
(2) Same settings for the monitoring regime (n_obs=48, 200 points; p flat 0.02/0.1/0.3, rising 0.01->0.5).
(3) Surrogate-calibrated scores (swarm_ews.ews_calibrated, M=200): should be mean ~0, sd ~1, 5% two-sided tails.
Output: results/emulator/verify_f1.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews import ews_calibrated as ec  # noqa: E402

M = 200
SETTINGS = [
    ("critique: B=12, T=50, flat p=0.03", 12, lambda T: np.full(T, 0.03), 50),
    ("critique: B=12, T=50, p rising 0.01->0.30", 12, lambda T: np.linspace(0.01, 0.30, T), 50),
    ("critique: B=12, T=50, flat p=0.20", 12, lambda T: np.full(T, 0.20), 50),
    ("monitor: n=48, T=200, flat p=0.02", 48, lambda T: np.full(T, 0.02), 200),
    ("monitor: n=48, T=200, flat p=0.10", 48, lambda T: np.full(T, 0.10), 200),
    ("monitor: n=48, T=200, flat p=0.30", 48, lambda T: np.full(T, 0.30), 200),
    ("monitor: n=48, T=200, p rising 0.01->0.50", 48, lambda T: np.linspace(0.01, 0.50, T), 200),
    ("monitor: n=48, T=200, p logistic rise 0.02->0.6 (mid 0.75T, scale 0.08T)", 48,
     lambda T: 0.02 + 0.58 / (1 + np.exp(-(np.arange(T) - 0.75 * T) / (0.08 * T))), 200),
    ("monitor: n=48, T=200, sharp late rise 0.02->0.42 (mid 0.85T, scale 0.04T)", 48,
     lambda T: 0.02 + 0.40 / (1 + np.exp(-(np.arange(T) - 0.85 * T) / (0.04 * T))), 200),
]


def ashery_variant_tau(f, B):
    """Indicator as in scripts/reanalysis_ashery.py: rolling variance / pointwise right-edge trend p(1-p)/B."""
    from swarm_ews import ews
    ind = ews.rolling_indicators(f, 0.5, 0.2); _, tr = ews.detrend(f, 0.2); p = np.clip(tr, 1e-3, 1 - 1e-3)
    return ews.trend_tau(ind["variance"] / (p * (1 - p) / B))


def summ(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return dict(n=int(len(v)), mean=round(float(v.mean()), 3), sd=round(float(v.std(ddof=1)), 3),
                frac_abs_gt_1_96=round(float(np.mean(np.abs(v) > 1.96)), 3), frac_gt_1_645=round(float(np.mean(v > 1.645)), 3))


def main():
    reps = 300
    out = {}
    for si, (label, n, pfun, T) in enumerate(SETTINGS):
        rng = np.random.default_rng(1000 + si)
        p = pfun(T)
        rows = []
        for r in range(reps):
            f = rng.binomial(n, p) / n
            d = ec.calibrated_scores(f, n, M=M, seed=r)
            d05 = ec.calibrated_scores(f, n, M=M, seed=r, sur_bandwidth=0.05)
            d.update(var_z05=d05["var_z"], ac1_z05=d05["ac1_z"], ash_tau=ashery_variant_tau(f, n))
            rows.append(d)
        res = {"ashery_variant_var_tau_mean": round(float(np.nanmean([x["ash_tau"] for x in rows])), 3)}
        for k in ("var", "ac1"):
            tau = np.array([x[f"{k}_tau"] for x in rows]); z = np.array([x[f"{k}_z"] for x in rows])
            res[k] = dict(raw_tau_mean=round(float(np.nanmean(tau)), 3), raw_tau_sd=round(float(np.nanstd(tau)), 3),
                          raw_tau_nan=int(np.isnan(tau).sum()), calibrated_z=summ(z),
                          calibrated_z_surbw005=summ([x[f"{k}_z05"] for x in rows]))
        out[label] = res
        print(f"{label:52s} VAR tau {res['var']['raw_tau_mean']:+.3f} -> z mean {res['var']['calibrated_z']['mean']:+.3f} "
              f"sd {res['var']['calibrated_z']['sd']:.3f} tail {res['var']['calibrated_z']['frac_abs_gt_1_96']:.3f} | "
              f"AC1 tau {res['ac1']['raw_tau_mean']:+.3f} (nan {res['ac1']['raw_tau_nan']}) -> z mean "
              f"{res['ac1']['calibrated_z']['mean']:+.3f} sd {res['ac1']['calibrated_z']['sd']:.3f} "
              f"tail {res['ac1']['calibrated_z']['frac_abs_gt_1_96']:.3f} (n {res['ac1']['calibrated_z']['n']}) | "
              f"Ashery-variant tau {res['ashery_variant_var_tau_mean']:+.3f} | z(surbw .05) var "
              f"{res['var']['calibrated_z_surbw005']['mean']:+.3f} ac1 {res['ac1']['calibrated_z_surbw005']['mean']:+.3f}", flush=True)
    Path("results/emulator").mkdir(parents=True, exist_ok=True)
    Path("results/emulator/verify_f1.json").write_text(json.dumps(dict(reps=reps, M=M, results=out), indent=1))


if __name__ == "__main__":
    main()
