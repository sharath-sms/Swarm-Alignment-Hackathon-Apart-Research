"""R5/R6: proper scores with bootstrap CIs and baselines (tests 1 and 2); event-definition and horizon sensitivity;
spontaneous-flip rate bounds; Monte Carlo error of emulator predictions."""
import glob, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
sys.path.insert(0, '.')
from swarm_ews.emulator import load_policy, simulate, t50
rng = np.random.default_rng(0); e = 1e-3
ll = lambda y, p: -np.mean(y * np.log(np.clip(p, e, 1 - e)) + (1 - y) * np.log(np.clip(1 - p, e, 1 - e)))
br = lambda y, p: np.mean((p - y) ** 2)
def scores(y, preds, B=5000):
    out = {}
    for k, p in preds.items():
        bb = [br(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B))]
        out[k] = dict(brier=round(br(y, p), 3), brier_ci=[round(float(np.percentile(bb, q)), 3) for q in (2.5, 97.5)], logloss=round(ll(y, p), 3))
    ks = list(preds); base = ks[-1]
    for k in ks[:-1]:
        d = [br(y[i], preds[k][i]) - br(y[i], preds[base][i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B))]
        out[k][f"brier_minus_{base}_ci"] = [round(float(np.percentile(d, q)), 3) for q in (2.5, 97.5)]
    return out
def runs(pattern):
    R = []
    for f in glob.glob(pattern):
        r = json.load(open(f)); fs = np.array([x["f"] for x in r["rounds"]])
        sustained = bool(np.any(np.convolve(fs >= 0.9, np.ones(5), "valid") >= 5))
        R.append(dict(cm=r["config"]["cm"], words="".join(r["config"]["words"]), f=fs, t50=t50(fs), sustained=sustained))
    return R
res = {}
# ---- test 2
pred2 = {}
for d in ("MQ", "QM"):
    for r in json.load(open(f"results/predictions/deepseek-v4-flash__{d}__N24__mr150.json"))["rows"]: pred2[(d, r["cm"])] = r["flip_point"]
pool = load_policy(json.load(open("results/policies/deepseek-v4-flash.json")))
pool_flip = {cm: float(np.mean([t50(simulate(pool, 24, cm, max_rounds=150, seed=70_000 + s)["f"]) is not None for s in range(1000)])) for cm in range(2, 9)}
R2 = runs("results/runs/live_token/*.json"); y2 = np.array([int(r["t50"] is not None) for r in R2], float)
res["test2_scores"] = scores(y2, {"token_conditional": np.array([pred2[(r["words"], r["cm"])] for r in R2]),
                                  "pooled_policy": np.array([pool_flip[r["cm"]] for r in R2]),
                                  "base_rate": np.full(len(y2), y2.mean())})
# ---- test 1 (pooled = the preregistered prediction; per-run own-pair MLE from m4 analysis is post hoc)
R1 = runs("results/runs/live_quench/*.json"); y1 = np.array([int(r["t50"] is not None) for r in R1], float)
res["test1_scores"] = scores(y1, {"pooled_prereg": np.array([pool_flip.get(r["cm"], 1.0) for r in R1]), "base_rate": np.full(len(y1), y1.mean())})
# ---- event definition & horizon sensitivity (test 2 H5)
def h5(R, flip):
    km = sum(flip(r) for r in R if r["words"] == "QM" and r["cm"] in (3, 4)); kq = sum(flip(r) for r in R if r["words"] == "MQ" and r["cm"] in (3, 4))
    return dict(push_M=f"{km}/10", push_Q=f"{kq}/10", p=float(fisher_exact([[km, 10 - km], [kq, 10 - kq]], alternative="greater")[1]))
res["H5_sensitivity"] = {
  "T50_150 (prereg)": h5(R2, lambda r: r["t50"] is not None),
  "T50_100": h5(R2, lambda r: r["t50"] is not None and r["t50"] < 100),
  "sustained_f>=0.9_5rounds_150": h5(R2, lambda r: r["sustained"])}
res["T50_vs_sustained_agreement"] = {k: float(np.mean([(r["t50"] is not None) == r["sustained"] for r in R])) for k, R in (("test1", R1), ("test2", R2))}
# ---- spontaneous-flip upper bounds (rule of three / exact binomial 95%)
res["spontaneous_flip_bounds"] = {"exact-policy models (6 pairs x 200 runs each, 0 flips)": round(1 - 0.05 ** (1 / 1200), 5),
                                  "per pair (200 runs, 0 flips)": round(1 - 0.05 ** (1 / 200), 4),
                                  "Llama-3-70B archive (4 runs, 0 flips)": round(1 - 0.05 ** (1 / 4), 3)}
res["MC_se_emulator_flip_1000_runs_max"] = round(float(np.sqrt(0.25 / 1000)), 4)
print(json.dumps(res, indent=1, default=float))
Path("results/live_analysis/r5_r6_scores.json").write_text(json.dumps(res, indent=1, default=float))
