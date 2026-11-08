"""M4 (external critique): do LIVE decisions follow the same P(B | state, pair) as the EXTRACTION samples?

Live: every cached call from the COMPLETED experiment `live_quench` (v1; v2 `live_token` deliberately excluded while
its preregistered runs are in progress). From each request we reconstruct (pair A/B, option order, memory state)
from the prompt text; the response gives the choice (first attempt only, try0).
Extraction: all cached `policy/deepseek-v4-flash/<state>/k<k>` samples; pair = WORD_PAIRS[k % 6].
Tests per (pair, state) cell with ≥ n_min samples on both sides: two-proportion difference; aggregate likelihood-ratio
(G) test of homogeneity across all such cells; calibration plot data. Also M3/M5: predict each live run with ITS OWN
pair's MLE policy (pooled MLE fallback where a pair-state has no samples) and score vs live outcomes.
"""
from __future__ import annotations
import glob, json, re, sqlite3, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.envs.naming_game import parse_value  # noqa: E402
from swarm_ews.emulator import simulate, t50  # noqa: E402

MODEL = "deepseek-v4-flash"
WORD_PAIRS = [("Q", "M"), ("M", "Q"), ("X", "Y"), ("Y", "X"), ("F", "J"), ("J", "F")]
REC = re.compile(r"- You chose (\w), partner chose (\w)")
con = sqlite3.connect("data/cache/llm_cache.sqlite")


def state_code(prompt, A, B):
    recs = REC.findall(prompt)
    if len(recs) != 5: return None
    return int("".join(str(2 * (m == B) + (p == B)) for m, p in recs), 4)


# --- live decisions (v1 only) ---
live = defaultdict(lambda: [0, 0])          # (pair, state) -> [nB, n]
live_by_run_words = {}
for req, text in con.execute("select request, text from cache where request like ?", ('%"tag": "live_quench/%',)):
    q = json.loads(req)
    if "/try0" not in q["tag"]: continue
    m = re.search(r"/cm(\d+)/s(\d+)/t(\d+)/a(\d+)/try0", q["tag"])
    opts = re.search(r"from: \[(\w), (\w)\]", q["instructions"] or "")
    if not m or not opts: continue
    seed = int(m.group(2)); A, B = WORD_PAIRS[seed % 6]
    s = state_code(q["input"], A, B)
    v = parse_value(text, (A, B))
    if s is None or v is None: continue
    c = live[(A + B, s)]; c[0] += v == B; c[1] += 1
# --- extraction samples ---
ext = defaultdict(lambda: [0, 0])
for req, text in con.execute("select request, text from cache where request like ?", (f'%"tag": "policy/{MODEL}/%',)):
    q = json.loads(req); m = re.match(rf"policy/{re.escape(MODEL)}/(\d{{5}})/k(\d+)", q["tag"])
    if not m: continue
    A, B = WORD_PAIRS[int(m.group(2)) % 6]; v = parse_value(text, (A, B))
    if v is None: continue
    c = ext[(A + B, int(m.group(1), 4))]; c[0] += v == B; c[1] += 1

n_live = sum(v[1] for v in live.values()); print(f"live decisions reconstructed: {n_live} in {len(live)} (pair,state) cells")
res = {"n_live_decisions": n_live, "by_nmin": {}}
for n_min in (5, 20):
    cells = [k for k in live if live[k][1] >= n_min and ext.get(k, [0, 0])[1] >= n_min]
    G = 0.0; df = 0; diffs = []; wts = []
    for k in cells:
        a, n1 = live[k]; b, n2 = ext[k]; p = (a + b) / (n1 + n2)
        for x, n in ((a, n1), (b, n2)):
            for obs, exp in ((x, n * p), (n - x, n * (1 - p))):
                if obs > 0 and exp > 0: G += 2 * obs * np.log(obs / exp)
        df += 1; diffs.append(a / n1 - b / n2); wts.append(n1)
    pv = float(chi2.sf(G, df)) if df else None
    w = np.array(wts) / np.sum(wts) if wts else None
    res["by_nmin"][n_min] = dict(cells=len(cells), live_decisions_covered=int(sum(live[k][1] for k in cells)), G=G, df=df,
                                 p=pv, weighted_mean_abs_diff=float(np.sum(w * np.abs(diffs))) if wts else None,
                                 weighted_mean_diff_live_minus_ext=float(np.sum(w * np.array(diffs))) if wts else None)
    print(f"n_min={n_min}: cells={len(cells)} covering {res['by_nmin'][n_min]['live_decisions_covered']} live decisions; "
          f"G={G:.1f} df={df} p={pv:.3g}; weighted mean |P_live−P_ext|={res['by_nmin'][n_min]['weighted_mean_abs_diff']:.3f}; "
          f"signed={res['by_nmin'][n_min]['weighted_mean_diff_live_minus_ext']:+.3f}")
# top discrepant heavily-visited cells
top = sorted([k for k in live if live[k][1] >= 20 and ext.get(k, [0, 0])[1] >= 8],
             key=lambda k: -live[k][1] * abs(live[k][0] / live[k][1] - ext[k][0] / ext[k][1]))[:10]
res["top_discrepant"] = [dict(pair=k[0], state=np.base_repr(k[1], 4).zfill(5), live=f"{live[k][0]}/{live[k][1]}",
                              ext=f"{ext[k][0]}/{ext[k][1]}") for k in top]
for t in res["top_discrepant"]: print("  ", t)

# --- M3/M5: per-run prediction with own pair's MLE policy (pooled MLE fallback) ---
pooled = np.zeros((1024, 2))
for (pr, s), (b, n) in ext.items(): pooled[s] += [b, n]
Ppool = np.where(pooled[:, 1] > 0, pooled[:, 0] / np.maximum(pooled[:, 1], 1), 0.5)
def pair_policy(pr):
    P = Ppool.copy()
    for s in range(1024):
        b, n = ext.get((pr, s), [0, 0])
        if n > 0: P[s] = b / n
    return P
y, p_pool, p_pair = [], [], []
for f in glob.glob(f"results/runs/live_quench/{MODEL}__N24__cm*__s*.json"):
    r = json.load(open(f)); cm = r["config"]["cm"]; A, B = r["config"]["words"]
    y.append(int(t50(np.array([x["f"] for x in r["rounds"]])) is not None))
    for P, store in ((Ppool, p_pool), (pair_policy(A + B), p_pair)):
        store.append(np.mean([t50(simulate(P, 24, cm, max_rounds=150, seed=31_000 + s)["f"]) is not None for s in range(300)]))
y, p_pool, p_pair = map(np.array, (y, p_pool, p_pair)); e = 1e-3
ll = lambda p: float(-np.mean(y * np.log(np.clip(p, e, 1 - e)) + (1 - y) * np.log(np.clip(1 - p, e, 1 - e))))
res["per_run_prediction_MLE"] = dict(n=len(y), brier_pooled=float(np.mean((p_pool - y) ** 2)), brier_pair=float(np.mean((p_pair - y) ** 2)),
                                     logloss_pooled=ll(p_pool), logloss_pair=ll(p_pair))
print("per-run prediction (MLE):", res["per_run_prediction_MLE"])
Path("results/live_analysis/m4_live_vs_extraction.json").write_text(json.dumps(res, indent=1))
