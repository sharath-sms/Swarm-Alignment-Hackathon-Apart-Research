"""Review #3 item 5: proper scores with correct provenance.
- Test 1: FROZEN preregistered predictions (results/predictions/deepseek-v4-flash__N24__mr150.json, flip_point per cm).
- Test 2: FROZEN token-conditional predictions; pooled comparator = frozen Test 1 pooled predictions at the same cm
  (cm 6 not frozen in test 1 -> emulator value from the same frozen pooled policy, flagged).
- Base rate: leave-one-out mean of the other runs (out-of-sample constant reference).
- Bootstrap: stratified by design cell (direction x cm), preserving the fixed allocation; paired differences.
"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from swarm_ews.emulator import load_policy, simulate, t50
rng = np.random.default_rng(0); e = 1e-3
br = lambda y, p: float(np.mean((p - y) ** 2))
ll = lambda y, p: float(-np.mean(y * np.log(np.clip(p, e, 1 - e)) + (1 - y) * np.log(np.clip(1 - p, e, 1 - e))))
def runs(pattern):
    out = []
    for f in glob.glob(pattern):
        r = json.load(open(f)); out.append(dict(cm=r["config"]["cm"], words="".join(r["config"]["words"]),
                                               y=int(t50(np.array([x["f"] for x in r["rounds"]])) is not None)))
    return out
def evaluate(R, preds, cell_key):
    y = np.array([r["y"] for r in R], float)
    loo = np.array([(y.sum() - y[i]) / (len(y) - 1) for i in range(len(y))]); preds = {**preds, "base_rate_LOO": loo}
    cells = defaultdict(list)
    for i, r in enumerate(R): cells[cell_key(r)].append(i)
    out = {k: dict(brier=round(br(y, p), 3), logloss=round(ll(y, p), 3)) for k, p in preds.items()}
    names = list(preds); boots = {k: [] for k in names}
    for _ in range(5000):
        idx = np.concatenate([rng.choice(v, len(v)) for v in cells.values()])
        for k in names: boots[k].append(br(y[idx], preds[k][idx]))
    for k in names:
        out[k]["brier_ci"] = [round(float(np.percentile(boots[k], q)), 3) for q in (2.5, 97.5)]
    for a in names:
        for b in names:
            if a < b:
                d = np.array(boots[a]) - np.array(boots[b])
                out[f"diff:{a}-{b}"] = dict(est=round(br(y, preds[a]) - br(y, preds[b]), 3), ci=[round(float(np.percentile(d, q)), 3) for q in (2.5, 97.5)])
    return out
frozen1 = {r["cm"]: r["flip_point"] for r in json.load(open("results/predictions/deepseek-v4-flash__N24__mr150.json"))["rows"]}
pool = load_policy(json.load(open("results/policies/deepseek-v4-flash.json")))
cm6 = float(np.mean([t50(simulate(pool, 24, 6, max_rounds=150, seed=123_000 + s)["f"]) is not None for s in range(1000)]))
frozen2 = {}
for d in ("MQ", "QM"):
    for r in json.load(open(f"results/predictions/deepseek-v4-flash__{d}__N24__mr150.json"))["rows"]: frozen2[(d, r["cm"])] = r["flip_point"]
R1 = runs("results/runs/live_quench/*.json"); R2 = runs("results/runs/live_token/*.json")
res = {"note": "frozen predictions; LOO base rate; cell-stratified paired bootstrap",
       "test1": evaluate(R1, {"pooled_frozen": np.array([frozen1[r["cm"]] for r in R1])}, lambda r: r["cm"]),
       "test2": evaluate(R2, {"token_conditional_frozen": np.array([frozen2[(r["words"], r["cm"])] for r in R2]),
                              "pooled_frozen": np.array([frozen1.get(r["cm"], cm6) for r in R2])}, lambda r: (r["words"], r["cm"])),
       "test2_pooled_cm6_not_frozen_value": cm6}
print(json.dumps(res, indent=1))
Path("results/live_analysis/r5_v2_scores.json").write_text(json.dumps(res, indent=1))
