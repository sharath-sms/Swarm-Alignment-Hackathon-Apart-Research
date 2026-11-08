"""R1 (independent review, issues 1-2): live-vs-extraction agreement, done properly.

Data: every cached first-attempt decision from live_quench (v1) and live_token (v2), and every DeepSeek extraction
sample. For each record we parse (word pair A/B, option order, memory state, choice) from the prompt text.
Cells: (pair, state) [order-marginalised: matches the live protocol, which randomises order per call] and
(pair, order, state) [order-conditional]. Cells with < n_min decisions on either side are dropped; cells whose pooled
proportion is 0 or 1 carry no information about a difference and are reported but excluded from tests.

1. Calibrated homogeneity test: G statistic with a parametric-bootstrap null (both sides drawn from the pooled cell
   proportion), not the chi-square approximation.
2. Equivalence: decision-weighted mean |p_live - p_ext| minus its expectation under the null (sampling-noise
   floor) = "excess discrepancy"; percentile CI from a nonparametric bootstrap that resamples decisions within cells
   on both sides. Prespecified tolerance delta = 0.05 (absolute probability), motivated by step 4.
3. Power: perturb the extraction-side cell probabilities on the logit scale so that the mean |shift| is 0.03/0.05/0.10,
   regenerate both sides from (perturbed, unperturbed) probabilities, and record how often the calibrated test rejects.
4. Consequence: shift the near-threshold DeepSeek pooled policy by +/-0.05 (absolute, clipped) in switching
   probabilities and report the change in emulator critical mass (N=24, horizon 150).
5. Stratified: order-conditional cells; time blocks (live: early vs late rounds; v1 vs v2; extraction: first pass
   k<8 vs top-up k>=8).
"""
from __future__ import annotations
import json, re, sqlite3, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.envs.naming_game import parse_value  # noqa: E402
from swarm_ews.emulator import load_policy, simulate, t50  # noqa: E402

MODEL = "deepseek-v4-flash"; WP = [("Q", "M"), ("M", "Q"), ("X", "Y"), ("Y", "X"), ("F", "J"), ("J", "F")]
REC = re.compile(r"- You chose (\w), partner chose (\w)"); OPT = re.compile(r"from: \[(\w), (\w)\]")
rng = np.random.default_rng(0)
con = sqlite3.connect("data/cache/llm_cache.sqlite") if Path("data/cache/llm_cache.sqlite").exists() else None


def parse(q, text, A, B):
    recs = REC.findall(q["input"]); o = OPT.search(q["instructions"] or "")
    if len(recs) != 5 or not o: return None
    v = parse_value(text, (A, B))
    if v is None: return None
    s = int("".join(str(2 * (m == B) + (p == B)) for m, p in recs), 4)
    return s, int(o.group(1) == B), int(v == B)          # order = 1 if B listed first


live, ext = [], []
CSV = Path("results/decisions/deepseek_decisions.csv.gz")
if "--csv" in sys.argv or not Path("data/cache/llm_cache.sqlite").exists():
    import csv, gzip
    with gzip.open(CSV, "rt") as fh:
        for r in csv.DictReader(fh):
            s_, o_, c_ = int(r["state"]), int(r["option_order"]), int(r["choice_B"])
            if r["source"] == "extraction": ext.append((r["pair"], s_, o_, c_, int(r["k"])))
            else: live.append((r["pair"], s_, o_, c_, int(r["round"]), int(r["source"] == "live_v2")))
    rows_iter = []
else:
    rows_iter = None
for req, text in ([] if rows_iter is not None else con.execute("select request, text from cache where request like '%\"tag\": \"live_quench/%' or request like '%\"tag\": \"live_token/%'")):
    q = json.loads(req); tag = q["tag"]
    if "/try0" not in tag: continue
    m = re.search(r"/cm(\d+)/(?:w(\w)(\w)/)?s(\d+)/t(\d+)/a\d+/try0", tag)
    if not m: continue
    A, B = (m.group(2), m.group(3)) if m.group(2) else WP[int(m.group(4)) % 6]
    r = parse(q, text, A, B)
    if r: live.append((A + B, *r, int(m.group(5)), 1 if "live_token" in tag else 0))
for req, text in ([] if rows_iter is not None else con.execute("select request, text from cache where request like ?", (f'%"tag": "policy/{MODEL}/%',))):
    q = json.loads(req); m = re.match(rf"policy/{re.escape(MODEL)}/(\d{{5}})/k(\d+)", q["tag"])
    if not m: continue
    A, B = WP[int(m.group(2)) % 6]; r = parse(q, text, A, B)
    if r: ext.append((A + B, *r, int(m.group(2))))
live = np.array(live, dtype=object); ext = np.array(ext, dtype=object)
print(f"live decisions {len(live)}, extraction samples {len(ext)}")


def cells(Lrows, Erows, with_order, n_min):
    key = (lambda r: (r[0], r[2], r[1])) if with_order else (lambda r: (r[0], r[1]))
    L = defaultdict(list); E = defaultdict(list)
    for r in Lrows: L[key(r)].append(r[3])
    for r in Erows: E[key(r)].append(r[3])
    ks = [k for k in L if len(L[k]) >= n_min and len(E.get(k, [])) >= n_min]
    return [(np.array(L[k]), np.array(E[k])) for k in ks]


def G(cs):
    g = 0.0
    for l, e in cs:
        n1, n2, a, b = len(l), len(e), l.sum(), e.sum(); p = (a + b) / (n1 + n2)
        for x, n in ((a, n1), (b, n2)):
            for o, ex in ((x, n * p), (n - x, n * (1 - p))):
                if o > 0 and ex > 0: g += 2 * o * np.log(o / ex)
    return g


def D(cs):
    w = np.array([len(l) for l, _ in cs], float); w /= w.sum()
    return float(np.sum(w * np.array([abs(l.mean() - e.mean()) for l, e in cs])))


def analyse(cs_all, label, B=500, power=True):
    var = [(l, e) for l, e in cs_all if 0 < (l.sum() + e.sum()) < (len(l) + len(e))]
    out = dict(cells=len(cs_all), informative_cells=len(var), live_decisions=int(sum(len(l) for l, _ in cs_all)),
               live_decisions_informative=int(sum(len(l) for l, _ in var)))
    g_obs, d_obs = G(var), D(var)
    gn, dn = [], []
    for _ in range(B):
        sim = []
        for l, e in var:
            p = (l.sum() + e.sum()) / (len(l) + len(e))
            sim.append((rng.random(len(l)) < p, rng.random(len(e)) < p))
        gn.append(G([(a.astype(int), b.astype(int)) for a, b in sim])); dn.append(D(sim))
    d_floor = float(np.mean(dn))
    # unbiased per-cell estimate of the squared true difference: (l̄-ē)^2 - v_l - v_e, v = p̂(1-p̂)/(n-1)
    w = np.array([len(l) for l, _ in var], float); w /= w.sum()
    sq = np.array([(l.mean() - e.mean()) ** 2 - l.var(ddof=1) / len(l) - e.var(ddof=1) / len(e) for l, e in var])
    ms = float(np.sum(w * sq)); bms = []
    for _ in range(2000):
        i = rng.integers(0, len(var), len(var)); ww = w[i] / w[i].sum(); bms.append(float(np.sum(ww * sq[i])))
    rms = lambda x: float(np.sqrt(max(x, 0.0)))
    out.update(G=round(g_obs, 1), p_calibrated=float((1 + np.sum(np.array(gn) >= g_obs)) / (1 + B)),
               mean_abs_diff=round(d_obs, 4), null_floor_mean_abs_diff=round(d_floor, 4),
               rms_true_diff=round(rms(ms), 4),
               rms_true_diff_ci=[round(rms(np.percentile(bms, 2.5)), 4), round(rms(np.percentile(bms, 97.5)), 4)],
               signed_mean_diff=round(float(np.sum(w * np.array([l.mean() - e.mean() for l, e in var]))), 4))
    if power:
        out["power"] = {}
        for target in (0.03, 0.05, 0.10):
            rej = 0
            for _ in range(100):
                sim = []
                for l, e in var:
                    p = np.clip((l.sum() + e.sum()) / (len(l) + len(e)), 0.02, 0.98)
                    lo = np.log(p / (1 - p)) + rng.choice([-1, 1]) * target / (p * (1 - p))  # delta-method logit shift
                    q = 1 / (1 + np.exp(-lo))
                    sim.append(((rng.random(len(l)) < p).astype(int), (rng.random(len(e)) < q).astype(int)))
                rej += G(sim) >= np.percentile(gn, 95)
            out["power"][f"shift_{target}"] = rej / 100
    print(f"[{label}] {out}", flush=True)
    return out


res = {"n_live": len(live), "n_ext": len(ext)}
res["marginal_order_n5"] = analyse(cells(live, ext, False, 5), "pair,state n>=5")
res["marginal_order_n20"] = analyse(cells(live, ext, False, 20), "pair,state n>=20", power=False)
res["order_conditional_n5"] = analyse(cells(live, ext, True, 5), "pair,order,state n>=5", power=False)
tmed = np.median([r[4] for r in live])
res["live_early_rounds"] = analyse(cells([r for r in live if r[4] < tmed], ext, False, 5), "live early rounds", power=False)
res["live_late_rounds"] = analyse(cells([r for r in live if r[4] >= tmed], ext, False, 5), "live late rounds", power=False)
res["live_v1_only"] = analyse(cells([r for r in live if r[5] == 0], ext, False, 5), "live v1", power=False)
res["live_v2_only"] = analyse(cells([r for r in live if r[5] == 1], ext, False, 5), "live v2", power=False)
# extraction first pass vs top-ups, at state level (pairs pooled; per-pair cells are too small in the first pass)
st = lambda rows: [("ALL",) + tuple(r[1:]) for r in rows]
res["ext_first_pass_vs_topup"] = analyse(cells(st([r for r in ext if r[4] < 8]), st([r for r in ext if r[4] >= 8]), False, 5),
                                         "extraction first pass vs top-up (state level)", power=False)
# 4. consequence of a 0.05 policy shift on critical mass
js = json.loads(Path(f"results/policies/{MODEL}.json").read_text()); P = load_policy(js)
def pc(Pol):
    fl = [np.mean([t50(simulate(Pol, 24, cm, max_rounds=150, seed=600 + s)["f"]) is not None for s in range(300)]) for cm in range(1, 9)]
    for i in range(7):
        if fl[i] < 0.5 <= fl[i + 1]: return (i + 1 + (0.5 - fl[i]) / (fl[i + 1] - fl[i] + 1e-12)) / 24
    return None
res["pc_shift"] = {"base": pc(P), "plus_0.05": pc(np.clip(P + 0.05, 0, 1)), "minus_0.05": pc(np.clip(P - 0.05, 0, 1))}
print("p_c under +/-0.05 policy shift:", res["pc_shift"])
def counts(rows, by_pair):
    c = defaultdict(lambda: np.zeros(2))
    for r in rows: c[(r[0] if by_pair else "ALL", r[1])] += [r[3], 1]
    return c
cl, ce = counts(live, False), counts(ext, False)
P_ext = np.array([ce[("ALL", s_)][0] / ce[("ALL", s_)][1] if ce[("ALL", s_)][1] else 0.5 for s_ in range(1024)])
P_live = np.array([cl[("ALL", s_)][0] / cl[("ALL", s_)][1] if cl[("ALL", s_)][1] >= 5 else P_ext[s_] for s_ in range(1024)])
res["pc_live_vs_ext_policy_pooled"] = {"ext": pc(P_ext), "live_where_n>=5": pc(P_live),
                                       "states_replaced": int(sum(cl[("ALL", s_)][1] >= 5 for s_ in range(1024)))}
clp, cep = counts(live, True), counts(ext, True); res["pc_live_vs_ext_policy_by_pair"] = {}
for A, B in [("M", "Q"), ("Q", "M")]:
    k = A + B
    Pe = np.array([cep[(k, s_)][0] / cep[(k, s_)][1] if cep[(k, s_)][1] else P_ext[s_] for s_ in range(1024)])
    Pl = np.array([clp[(k, s_)][0] / clp[(k, s_)][1] if clp[(k, s_)][1] >= 5 else Pe[s_] for s_ in range(1024)])
    res["pc_live_vs_ext_policy_by_pair"][k] = {"ext": pc(Pe), "live_where_n>=5": pc(Pl)}
print("p_c live-estimated vs extraction-estimated policy:", res["pc_live_vs_ext_policy_pooled"], res["pc_live_vs_ext_policy_by_pair"])
Path("results/live_analysis/m4_v3_equivalence.json").write_text(json.dumps(res, indent=1, default=float))
