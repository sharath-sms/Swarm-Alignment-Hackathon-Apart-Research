"""Step 1: validate the EWS pipeline on synthetic controls before touching any LLM data.

Pass criteria (pre-registered):
- Positive control (harvested logistic, ramp vs null): AUROC >= 0.8 for variance and AC1.
- Negative control (AR(1) vs AR(1)): AUROC CI includes 0.5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews import ews  # noqa: E402
from swarm_ews.controls import ar1, harvested_logistic  # noqa: E402

OUT = Path("results/validation")
OUT.mkdir(parents=True, exist_ok=True)
N_RUNS, T, BURN, BUFFER = 40, 400, 20, 5
WF, BW = 0.5, 0.2

rng = np.random.default_rng(123)

# ---- positive control ----
tip_segments, tip_full, tip_idx = [], [], []
for s in range(N_RUNS):
    x, r, tip = harvested_logistic(T=T, ramp=True, seed=s)
    if tip is None or tip - BUFFER - BURN < 60:
        continue
    tip_segments.append(x[BURN : tip - BUFFER])
    tip_full.append(x)
    tip_idx.append(tip)
lengths = [len(s) for s in tip_segments]
null_segments, null_full = [], []
for s in range(N_RUNS):
    x, r, _ = harvested_logistic(T=T, ramp=False, seed=10_000 + s)
    L = int(rng.choice(lengths))
    null_segments.append(x[BURN : BURN + L])
    null_full.append(x)

pos = [ews.ews_taus(x, WF, BW) for x in tip_segments]
neg = [ews.ews_taus(x, WF, BW) for x in null_segments]
results = {"positive_control": {}, "negative_control": {}, "n_tip": len(pos), "n_null": len(neg)}
for k in ews.INDICATORS:
    auc, lo, hi = ews.auroc_ci([p[k] for p in pos], [q[k] for q in neg])
    results["positive_control"][k] = dict(auroc=auc, ci=[lo, hi])

# ---- negative control: AR(1) vs AR(1) ----
a = [ews.ews_taus(ar1(T=250, seed=s), WF, BW) for s in range(N_RUNS)]
b = [ews.ews_taus(ar1(T=250, seed=5000 + s), WF, BW) for s in range(N_RUNS)]
for k in ews.INDICATORS:
    auc, lo, hi = ews.auroc_ci([p[k] for p in a], [q[k] for q in b])
    results["negative_control"][k] = dict(auroc=auc, ci=[lo, hi])

# ---- online monitor on positive control (AC1 and variance) ----
results["online"] = {}
for k in ews.PRIMARY:
    null_traj = [ews.online_tau(x, k, min_len=40, step=2, window_frac=WF, bandwidth=BW) for x in null_segments]
    thr = ews.calibrate_threshold(null_traj, far=0.05)
    leads, detected = [], 0
    for x in tip_segments:
        tr = ews.online_tau(x, k, min_len=40, step=2, window_frac=WF, bandwidth=BW)
        a_t = ews.first_alarm(tr, thr)
        if a_t is not None:
            detected += 1
            leads.append(len(x) + BUFFER - a_t)  # rounds before tip
    results["online"][k] = dict(threshold=thr, detection_rate=detected / len(tip_segments),
                                median_lead=float(np.median(leads)) if leads else None)

# ---- surrogate p-values for a few tipping runs ----
results["surrogate_p_ac1_first5"] = [ews.surrogate_pvalues(x, n=99)["ac1"] for x in tip_segments[:5]]

ok_pos = all(results["positive_control"][k]["auroc"] >= 0.8 for k in ews.PRIMARY)
ok_neg = all(results["negative_control"][k]["ci"][0] <= 0.5 <= results["negative_control"][k]["ci"][1]
             for k in ews.PRIMARY)
results["PASS"] = bool(ok_pos and ok_neg)
(OUT / "validation.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))

# ---- figure ----
fig, ax = plt.subplots(3, 2, figsize=(10, 8), sharex="col")
for col, (x, title) in enumerate([(tip_segments[0], "Ramp → tip"), (null_segments[0], "Null (no ramp)")]):
    ind = ews.rolling_indicators(x, WF, BW)
    _, trend = ews.detrend(x, BW)
    ax[0, col].plot(x, lw=0.8); ax[0, col].plot(trend, lw=1.5); ax[0, col].set_title(title)
    ax[1, col].plot(ind["variance"]); ax[1, col].set_ylabel("variance")
    ax[2, col].plot(ind["ac1"]); ax[2, col].set_ylabel("AC1"); ax[2, col].set_xlabel("time (pre-tip segment)")
fig.tight_layout()
fig.savefig(OUT / "positive_control_example.png", dpi=150)
print("saved", OUT)
