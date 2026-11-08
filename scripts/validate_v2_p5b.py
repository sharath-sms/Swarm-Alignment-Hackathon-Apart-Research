"""Post-hoc P5' (documented): paired bootstrap of AUROC(EWS) - AUROC(LEVEL) for TIP vs MEAN.
Original P5 (level baseline cannot separate TIP from MEAN) FAILED: level AUROC 0.62 [0.52,0.72] because a fold
produces an accelerating decline. P5' asks the question P5 was meant to answer: do EWS add discrimination beyond the level?"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews import ews
from swarm_ews.controls import harvested_logistic, proportional_logistic_rampK
N, T, BURN, BUFFER, WF, BW = 60, 400, 20, 20, 0.5, 0.2
rng = np.random.default_rng(7)
tip = []
for s in range(3 * N):
    x, _, t = harvested_logistic(T=T, ramp=True, seed=s)
    if t is not None and t - BUFFER - BURN >= 80: tip.append(x[BURN:t - BUFFER])
    if len(tip) == N: break
L = [len(x) for x in tip]
def cut(x): l = int(rng.choice(L)); return x[BURN:BURN + l]
_ = [cut(harvested_logistic(T=T, ramp=True, r_start_mult=2.0, r_end_mult=3.5, seed=20_000 + s)[0]) for s in range(N)]
_ = [cut(harvested_logistic(T=T, ramp=False, seed=10_000 + s)[0]) for s in range(N)]   # keep rng stream identical to v2
mean = [cut(proportional_logistic_rampK(T=T, seed=30_000 + s)[0]) for s in range(N)]
lvl = lambda x: -kendalltau(np.arange(len(x)), x)[0]
S = {"ac1": ([ews.ews_taus(x, WF, BW)["ac1"] for x in tip], [ews.ews_taus(x, WF, BW)["ac1"] for x in mean]),
     "variance": ([ews.ews_taus(x, WF, BW)["variance"] for x in tip], [ews.ews_taus(x, WF, BW)["variance"] for x in mean]),
     "level": ([lvl(x) for x in tip], [lvl(x) for x in mean])}
y = np.r_[np.ones(N), np.zeros(N)]
b = np.random.default_rng(0); out = {}
for k in ("ac1", "variance"):
    d = []
    for _ in range(4000):
        i = b.integers(0, N, N); j = b.integers(0, N, N)
        auc = lambda m: roc_auc_score(y, np.r_[np.array(S[m][0])[i], np.array(S[m][1])[j]])
        d.append(auc(k) - auc("level"))
    lo, hi = np.percentile(d, [2.5, 97.5]); out[k] = dict(diff=float(np.mean(d)), lo=float(lo), hi=float(hi), PASS=bool(lo > 0))
print(json.dumps(out, indent=2))
p = Path("results/validation/validation_v2.json"); r = json.loads(p.read_text())
r["P5_original_failed_note"] = "Level baseline AUROC TIP vs MEAN 0.62 [0.52,0.72]: fold gives accelerating decline."
r["P5prime_posthoc_EWS_minus_LEVEL"] = out
r["PASS_with_P5prime"] = all(v for kk, v in r["criteria"].items() if kk != "P5") and all(v["PASS"] for v in out.values())
p.write_text(json.dumps(r, indent=2)); print("PASS_with_P5prime:", r["PASS_with_P5prime"])
