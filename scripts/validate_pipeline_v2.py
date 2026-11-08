"""Study 0 (v2): pipeline validation with the controls demanded by the methodology review (04, F1/F2).

Conditions (all synthetic, no LLM):
  TIP   harvested logistic, r ramped DOWN through the fold -> tips (true CSD)
  AWAY  harvested logistic, r ramped UP by the same amount (forced, moves away from fold)
  MEAN  proportional-harvest logistic with K ramped down: mean declines, recovery rate constant (no CSD)
  STAT  harvested logistic at constant r (stationary)
  AR    AR(1) vs AR(1) (pure noise sanity check)
Monitors: EWS variance / AC1 (Kendall tau of rolling indicator) vs LEVEL baseline (-tau of the raw level,
i.e. "the state is falling") and RESID-free trend baseline.

Pre-registered pass criteria:
  P1 TIP vs STAT     AUROC(var, ac1) >= 0.8          (sensitivity, as before)
  P2 TIP vs MEAN     AUROC(ac1) >= 0.8                (EWS distinguish CSD from mere mean drift)
  P3 MEAN vs STAT    AUROC(ac1) CI includes 0.5      (EWS do not fire on mean drift alone)
  P4 TIP vs AWAY     AUROC(var, ac1) >= 0.8          (EWS track approach to threshold, not mere forcing)
  P5 LEVEL baseline  TIP vs MEAN AUROC CI includes 0.5 or below (baseline cannot tell them apart -> EWS add value)
  P6 AR vs AR        CI includes 0.5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews import ews  # noqa: E402
from swarm_ews.controls import ar1, harvested_logistic, proportional_logistic_rampK  # noqa: E402

OUT = Path("results/validation"); OUT.mkdir(parents=True, exist_ok=True)
N, T, BURN, BUFFER = 60, 400, 20, 20      # larger tip buffer (was 5) to avoid collapse-onset leakage
WF, BW = 0.5, 0.2
rng = np.random.default_rng(7)

def level_score(x):
    tau, _ = kendalltau(np.arange(len(x)), x)
    return -float(tau)  # high = state falling

# TIP
tip = []
for s in range(3 * N):
    x, _, t = harvested_logistic(T=T, ramp=True, seed=s)
    if t is not None and t - BUFFER - BURN >= 80:
        tip.append(x[BURN:t - BUFFER])
    if len(tip) == N:
        break
L = [len(x) for x in tip]
def cut(x):
    l = int(rng.choice(L)); return x[BURN:BURN + l]
# AWAY: same |ramp| but upward (r from 2x r* to 3.5x r*), segment length matched
away = [cut(harvested_logistic(T=T, ramp=True, r_start_mult=2.0, r_end_mult=3.5, seed=20_000 + s)[0]) for s in range(N)]
stat = [cut(harvested_logistic(T=T, ramp=False, seed=10_000 + s)[0]) for s in range(N)]
# MEAN: K ramp chosen so the mean declines over the segment by a similar fraction to TIP segments
mean = [cut(proportional_logistic_rampK(T=T, seed=30_000 + s)[0]) for s in range(N)]
ara = [ar1(T=250, seed=s) for s in range(N)]; arb = [ar1(T=250, seed=5000 + s) for s in range(N)]

def taus(xs): return [ews.ews_taus(x, WF, BW) for x in xs]
TT, AW, ST, ME, A1, A2 = map(taus, (tip, away, stat, mean, ara, arb))

def comp(P, Q, keys=ews.PRIMARY):
    return {k: dict(zip(("auroc", "lo", "hi"), ews.auroc_ci([p[k] for p in P], [q[k] for q in Q]))) for k in keys}

res = {
  "n": N, "buffer": BUFFER,
  "mean_drop_frac": {"TIP": float(np.median([(x[:20].mean()-x[-20:].mean())/x[:20].mean() for x in tip])),
                     "MEAN": float(np.median([(x[:20].mean()-x[-20:].mean())/x[:20].mean() for x in mean]))},
  "TIP_vs_STAT": comp(TT, ST), "TIP_vs_MEAN": comp(TT, ME), "MEAN_vs_STAT": comp(ME, ST),
  "TIP_vs_AWAY": comp(TT, AW), "AR_vs_AR": comp(A1, A2),
  "LEVEL_baseline": {
     "TIP_vs_STAT": dict(zip(("auroc","lo","hi"), ews.auroc_ci([level_score(x) for x in tip], [level_score(x) for x in stat]))),
     "TIP_vs_MEAN": dict(zip(("auroc","lo","hi"), ews.auroc_ci([level_score(x) for x in tip], [level_score(x) for x in mean]))),
  },
}
inc = lambda d: d["lo"] <= 0.5 <= d["hi"]
res["criteria"] = {
  "P1": all(res["TIP_vs_STAT"][k]["auroc"] >= 0.8 for k in ews.PRIMARY),
  "P2": res["TIP_vs_MEAN"]["ac1"]["auroc"] >= 0.8,
  "P3": inc(res["MEAN_vs_STAT"]["ac1"]),
  "P4": all(res["TIP_vs_AWAY"][k]["auroc"] >= 0.8 for k in ews.PRIMARY),
  "P5": res["LEVEL_baseline"]["TIP_vs_MEAN"]["lo"] <= 0.5,
  "P6": all(inc(res["AR_vs_AR"][k]) for k in ews.PRIMARY),
}
res["PASS"] = all(res["criteria"].values())

# online monitor with min indicator points >= 20 (fix for inflated thresholds), calibrated on MEAN+STAT nulls
res["online"] = {}
for k in ews.PRIMARY:
    min_len = int(np.ceil(20 / (1 - WF))) + 1   # window = WF*len -> >=20 indicator points
    nulls = [ews.online_tau(x, k, min_len=min_len, step=2, window_frac=WF, bandwidth=BW) for x in mean + stat]
    thr = ews.calibrate_threshold(nulls, far=0.05)
    det, leads = 0, []
    for x in tip:
        a = ews.first_alarm(ews.online_tau(x, k, min_len=min_len, step=2, window_frac=WF, bandwidth=BW), thr)
        if a is not None:
            det += 1; leads.append(len(x) + BUFFER - a)
    res["online"][k] = dict(min_len=min_len, threshold=thr, detection_rate=det / len(tip),
                            median_lead=float(np.median(leads)) if leads else None)

(OUT / "validation_v2.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: res[k] for k in ("mean_drop_frac", "criteria", "PASS", "online")}, indent=2))
for c in ("TIP_vs_STAT", "TIP_vs_MEAN", "MEAN_vs_STAT", "TIP_vs_AWAY", "AR_vs_AR"):
    print(c, {k: round(v["auroc"], 3) for k, v in res[c].items()}, {k: [round(v["lo"],2), round(v["hi"],2)] for k, v in res[c].items()})
print("LEVEL", {k: (round(v["auroc"],3), round(v["lo"],2), round(v["hi"],2)) for k, v in res["LEVEL_baseline"].items()})
