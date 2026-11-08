"""Monitoring v2 (fix for external critique F2, with F1-calibrated EWS): level-matched incremental value of EWS.

For one policy (extracted LLM policy or best-response reference), cm=0, N=48:
 (a) q_c exactly as in scripts/emulator_study.py (30 constant q x 100 runs, horizon 400, flip prob 0.5 crossing).
 (b) TOWARD ramps (q: 0 -> 1.6 q_c over T_ramp=300 rounds, then hold) and a large NULL pool of runs that never tip within
     the 480-round horizon: STAT (constant q on a grid in [0.2, 0.95] q_c), STOPSHORT (ramp at the TOWARD slope or half
     of it, holding at plateaus 0.3..0.95 q_c), AWAY (from 1.0/0.9/0.8/0.6 q_c down to 0 over T_ramp), plus the original
     emulator_study nulls (AWAY 1.0 q_c->0, STOPSHORT 0.6 q_c, STAT 0.8 q_c).
 (c) For each TOWARD segment f[burn : tip-L] (L in 10/25/50/100), candidate null segments f[burn : e] from every
     null run and end e (step 10 rounds) with |len - len_TOWARD| <= LEN_CAL and e <= horizon - max(L, 50) (the null is
     observed not to tip for >= L rounds after the segment). 1-NN matching with replacement on [level, trend]
     standardised by the TOWARD SDs at that lead, caliper CAL (in SD units); unmatched TOWARD segments are dropped.
     level = mean of last 10 rounds; trend = Kendall tau of f over the segment (same features as emulator_study).
 (d) On the matched sample: single-feature AUROC (level, trend, raw EWS taus, calibrated EWS z, oracle q) and repeated
     5-fold (grouped by run) CV logistic AUROC for level+trend vs level+trend+EWS; paired bootstrap (resampling
     matched pairs) 95% CIs for Delta AUROC on the out-of-fold predictions.
 (e) Prereg H4 (Delta CV-AUROC <= 0.02 at leads 25 and 50, pooled nulls, Holm over 2 indicators x 2 leads), both on
     the ORIGINAL unmatched design (emulator_study replication) and the level-matched design, for the preregistered raw
     indicators and the F1-calibrated z indicators.
Calibrated indicators: swarm_ews.ews_calibrated.calibrated_scores (M binomial surrogates around the series' own
bandwidth-0.2 trend; n_obs = N for cm=0). Sensitivity: surrogate trend bandwidth 0.05 (z05).
Usage: python scripts/monitor_v2.py --policy results/policies/deepseek-v4-flash.json
       python scripts/monitor_v2.py --best-response
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews import ews_calibrated as ec  # noqa: E402
from swarm_ews.emulator import best_response_policy, load_policy, simulate, t50  # noqa: E402

BURN = 10
LEADS = (10, 25, 50, 100)
H4_LEADS = (25, 50)
MARGIN = 0.02
EWS_RAW = ("var_tau", "ac1_tau")
EWS_CAL = ("var_z", "ac1_z")
EWS_CAL05 = ("var_z05", "ac1_z05")


# ---------------------------------------------------------------- simulation
def compute_qc(P, N, max_rounds=400):
    qs = np.round(np.linspace(0.02, 0.6, 30), 3); flips = []
    for q in qs:
        flips.append(float(np.mean([t50(simulate(P, N, 0, max_rounds=max_rounds, q_schedule=(lambda t, q=q: q),
                                                 seed=50_000 + s)["f"]) is not None for s in range(100)])))
    for i in range(len(qs) - 1):
        if flips[i] < 0.5 <= flips[i + 1]:
            return float(qs[i] + (0.5 - flips[i]) * (qs[i + 1] - qs[i]) / (flips[i + 1] - flips[i])), qs.tolist(), flips
    return None, qs.tolist(), flips


def null_configs(qc, TR, R_pool, R_orig):
    s1 = 1.6 * qc / TR
    cf = []
    for m in list(np.round(np.linspace(0.2, 0.95, 16), 3)) + [1.0]:
        cf.append(("STAT", f"STAT_{m}", (lambda t, a=m * qc: a), R_pool))
    for slope_mult in (1.0, 0.5):
        for m in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.1):
            cf.append(("STOPSHORT", f"STOPSHORT_{m}_s{slope_mult}",
                       (lambda t, a=m * qc, s=s1 * slope_mult: min(a, s * t)), R_pool))
    for m in (1.0, 0.9, 0.8, 0.6):
        cf.append(("AWAY", f"AWAY_{m}", (lambda t, a=m * qc: max(0.0, a * (1 - t / TR))), R_pool))
    # original emulator_study nulls (for the unmatched replication; also part of the pool)
    cf.append(("AWAY", "ORIG_AWAY", (lambda t: max(0.0, qc * (1 - t / TR))), R_orig))
    cf.append(("STOPSHORT", "ORIG_STOPSHORT", (lambda t: min(0.6 * qc, s1 * t)), R_orig))
    cf.append(("STAT", "ORIG_STAT", (lambda t: 0.8 * qc), R_orig))
    return cf


# ---------------------------------------------------------------- features
def cheap_feats(x):
    """level, trend; trend of a constant segment (Kendall tau undefined -> NaN, e.g. f == 0 throughout for exact
    policies with P(B|all-A) ~ 0) is set to 0."""
    tau = kendalltau(np.arange(len(x)), x)[0]
    return float(np.mean(x[-10:])), (0.0 if np.isnan(tau) else float(tau))


def path_feats(x):
    """Extra mean-path descriptors for the stronger baseline: OLS slopes over the last 20 / 50 rounds."""
    out = {}
    for w in (20, 50):
        y = np.asarray(x[-w:], float); t = np.arange(len(y))
        out[f"slope{w}"] = float(np.polyfit(t, y, 1)[0]) if len(y) >= 3 else 0.0
    return out


def _calib_job(args):
    key, x, n_obs, M, seed = args
    d = ec.calibrated_scores(x, n_obs, M=M, seed=seed)
    d05 = ec.calibrated_scores(x, n_obs, M=M, seed=seed, sur_bandwidth=0.05)
    return key, dict(var_tau=d["var_tau"], ac1_tau=d["ac1_tau"], var_z=d["var_z"], ac1_z=d["ac1_z"],
                     var_z05=d05["var_z"], ac1_z05=d05["ac1_z"])


# ---------------------------------------------------------------- evaluation
def auc(y, s):
    s = np.nan_to_num(np.asarray(s, float))
    return float(roc_auc_score(y, s))


def oof_pred(X, y, groups, reps=5):
    X = np.nan_to_num(np.asarray(X, float))
    pr = np.zeros(len(y))
    for r in range(reps):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=r)
        for tr, te in cv.split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
            pr[te] += m.predict_proba(X[te])[:, 1]
    return pr / reps


def boot_index(n_pos, n_neg, paired, B, rng):
    """Yield index arrays into the stacked [pos; neg] sample. Paired: resample pair ids (pos i <-> neg i)."""
    for _ in range(B):
        if paired:
            i = rng.integers(0, n_pos, n_pos)
            yield np.r_[i, n_pos + i]
        else:
            yield np.r_[rng.integers(0, n_pos, n_pos), n_pos + rng.integers(0, n_neg, n_neg)]


def evaluate(F_pos, F_neg, g_pos, g_neg, paired, B=2000, seed=0):
    """F_*: list of feature dicts. Returns single-feature AUROCs (CI), CV AUROCs (CI) and Delta AUROC (CI, p)."""
    keys = ("level", "trend", "slope20", "slope50", "oracle") + EWS_RAW + EWS_CAL + EWS_CAL05
    y = np.r_[np.ones(len(F_pos)), np.zeros(len(F_neg))]
    col = {k: np.array([f[k] for f in F_pos] + [f[k] for f in F_neg], float) for k in keys}
    groups = np.r_[np.asarray(g_pos), np.asarray(g_neg)]
    LT, LTP = ("level", "trend"), ("level", "trend", "slope20", "slope50")
    models = {"LT": LT, "LTP": LTP}
    for tag, (v, a) in (("raw", EWS_RAW), ("cal", EWS_CAL), ("cal05", EWS_CAL05)):
        models[f"LT+var_{tag}"] = LT + (v,)
        models[f"LT+ac1_{tag}"] = LT + (a,)
        models[f"LT+ews_{tag}"] = LT + (v, a)
        models[f"LTP+var_{tag}"] = LTP + (v,)
        models[f"LTP+ac1_{tag}"] = LTP + (a,)
        models[f"LTP+ews_{tag}"] = LTP + (v, a)
        models[f"ews_only_{tag}"] = (v, a)
    preds = {m: oof_pred(np.column_stack([col[k] for k in ks]), y, groups) for m, ks in models.items()}
    scores = {**{f"feat:{k}": np.nan_to_num(col[k]) for k in keys}, **{f"cv:{m}": p for m, p in preds.items()}}
    point = {k: auc(y, s) for k, s in scores.items()}
    deltas = {m[3:]: (f"cv:{m}", "cv:LT") for m in models if m.startswith("LT+")}
    deltas.update({"P_" + m[4:]: (f"cv:{m}", "cv:LTP") for m in models if m.startswith("LTP+")})
    rng = np.random.default_rng(seed)
    bs = {k: [] for k in scores}; bd = {d: [] for d in deltas}
    for idx in boot_index(len(F_pos), len(F_neg), paired, B, rng):
        yy = y[idx]
        a = {k: roc_auc_score(yy, s[idx]) for k, s in scores.items()}
        for k in scores: bs[k].append(a[k])
        for d, (m1, m0) in deltas.items(): bd[d].append(a[m1] - a[m0])
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    out = {"n_pos": len(F_pos), "n_neg": len(F_neg),
           "auroc": {k: dict(est=round(point[k], 3), ci=ci(bs[k])) for k in scores}, "delta": {}}
    for d, (m1, m0) in deltas.items():
        v = np.array(bd[d]); est = point[m1] - point[m0]
        out["delta"][d] = dict(est=round(est, 4), ci=[round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
                               # H0_sup: Delta <= margin (rejection => EWS raise AUROC by > margin => H4 falsified)
                               p_sup=float((1 + np.sum(v <= MARGIN)) / (1 + len(v))),
                               # H0_eq: Delta >= margin (rejection => Delta < margin established => H4 positively supported)
                               p_eq=float((1 + np.sum(v >= MARGIN)) / (1 + len(v))))
    return out


def pair_groups(mneg):
    """CV groups for a matched sample: matched pairs stay in the same fold, and pairs sharing a null run are merged
    (connected components). Splitting exactly balanced pairs across folds makes CV AUROC of uninformative features
    systematically < 0.5 (training-fold imbalance is the mirror image of the test fold's)."""
    parent = list(range(len(mneg)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    first = {}
    for i, n in enumerate(mneg):
        if n["id"] in first:
            parent[find(i)] = find(first[n["id"]])
        else:
            first[n["id"]] = i
    return [f"G{find(i)}" for i in range(len(mneg))]


def holm(pvals):
    k = len(pvals); order = np.argsort(pvals); adj = np.empty(k); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (k - r) * pvals[i])); adj[i] = run
    return adj


def h4_decision(res_by_lead, tag):
    tests = [(L, ind, res_by_lead[f"lead{L}"]["delta"][f"{ind}_{tag}"]) for L in H4_LEADS for ind in ("var", "ac1")
             if f"lead{L}" in res_by_lead]
    if len(tests) < 4:
        return dict(decision="NOT EVALUABLE (missing leads)", tests=len(tests))
    ps = holm(np.array([t[2]["p_sup"] for t in tests])); pe = holm(np.array([t[2]["p_eq"] for t in tests]))
    rows = [dict(lead=L, indicator=ind, delta=d["est"], ci=d["ci"], p_sup_holm=round(float(a), 4), p_eq_holm=round(float(b), 4))
            for (L, ind, d), a, b in zip(tests, ps, pe)]
    if any(r["p_sup_holm"] < 0.05 for r in rows):
        dec = "FAIL (EWS raise CV-AUROC by > 0.02, Holm-significant)"
    elif all(r["p_eq_holm"] < 0.05 for r in rows):
        dec = "PASS (all Delta < 0.02 established, Holm)"
    else:
        dec = "PASS-NOT-FALSIFIED / equivalence not established (no Delta significantly > 0.02, but CI upper >= 0.02)"
    return dict(decision=dec, tests=rows)


# ---------------------------------------------------------------- main
def main(a):
    t0 = time.time()
    if a.best_response:
        name, P = "best_response", best_response_policy(5, 0.02)
    else:
        js = json.loads(Path(a.policy).read_text()); name, P = js["model"], load_policy(js)
    N, TR, H = a.N, a.T_ramp, int(1.6 * a.T_ramp)
    qc, qs, flips = compute_qc(P, N)
    print(f"[{name}] q_c = {qc}  ({time.time() - t0:.0f}s)", flush=True)
    res = dict(policy=name, settings=vars(a), qcrit=dict(q=qs, flip=flips, q_c=qc))
    if qc is None:
        Path(a.out_dir, f"monitor_v2_{name}.json").write_text(json.dumps(res, indent=1)); return

    # ---- TOWARD
    sched_tw = lambda t: min(1.6 * qc, 1.6 * qc * t / TR)
    toward = []
    for s in range(a.R_toward):
        f = simulate(P, N, 0, max_rounds=H, q_schedule=sched_tw, seed=300_000 + s, stop_on_tip=True)["f"]
        tip = t50(f)
        if tip is not None:
            toward.append(dict(id=f"T{s}", f=f, tip=tip))
    tips = [r["tip"] for r in toward]
    res["toward"] = dict(n_runs=a.R_toward, n_tipped=len(toward),
                         tip_round_pct10_50_90=[float(np.percentile(tips, q)) for q in (10, 50, 90)])
    print(f"[{name}] TOWARD tipped {len(toward)}/{a.R_toward}, tip rounds {res['toward']['tip_round_pct10_50_90']}", flush=True)

    # ---- NULL pool
    nulls = {}; pool_stats = {}
    for ci_, (cls, lab, sched, R) in enumerate(null_configs(qc, TR, a.R_pool, a.R_orig)):
        n_tip = 0
        for s in range(R):
            f = simulate(P, N, 0, max_rounds=H, q_schedule=sched, seed=400_000 + 1000 * ci_ + s, stop_on_tip=False)["f"]
            tip = t50(f)
            rid = f"{lab}#{s}"
            nulls[rid] = dict(cls=cls, lab=lab, f=f, tip=tip, q=np.array([sched(t) for t in range(H)]))
            n_tip += tip is not None
        pool_stats[lab] = dict(R=R, tipped=n_tip)
    res["null_pool"] = pool_stats
    print(f"[{name}] null pool simulated: {len(nulls)} runs ({time.time() - t0:.0f}s)", flush=True)

    # candidate null segments (never-tipping runs only)
    cand = []   # (rid, end, len, level, trend)
    for rid, r in nulls.items():
        if r["tip"] is not None:
            continue
        for e in range(BURN + 40, H + 1, 10):
            x = r["f"][BURN:e]; lv, tr = cheap_feats(x)
            cand.append((rid, e, e - BURN, lv, tr))
    C = np.array([(c[2], c[3], c[4]) for c in cand], float)
    print(f"[{name}] candidate null segments: {len(cand)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- build segments for each lead (matched + original unmatched design)
    rng = np.random.default_rng(0)
    seg_jobs = {}   # key -> series
    design = {}
    for L in LEADS:
        tw = [r for r in toward if r["tip"] - L - BURN >= 40]
        if len(tw) < 20:
            continue
        pos = []
        for r in tw:
            x = r["f"][BURN:r["tip"] - L]; lv, tr = cheap_feats(x)
            pos.append(dict(id=r["id"], key=f"{r['id']}|{r['tip'] - L}", len=len(x), level=lv, trend=tr,
                            oracle=float(sched_tw(r["tip"] - L - 1)), **path_feats(x)))
            seg_jobs[pos[-1]["key"]] = x
        Pf = np.array([(p["level"], p["trend"]) for p in pos])
        sd = Pf.std(axis=0, ddof=1); sd[sd == 0] = 1
        okc = np.array([c[1] <= H - max(L, 50) for c in cand])
        matched, dists = [], []
        for p in pos:
            m = okc & (np.abs(C[:, 0] - p["len"]) <= a.len_caliper)
            if not m.any():
                matched.append(None); dists.append(np.inf); continue
            idx = np.where(m)[0]
            d = np.sqrt(((C[idx, 1] - p["level"]) / sd[0]) ** 2 + ((C[idx, 2] - p["trend"]) / sd[1]) ** 2)
            j = idx[np.argmin(d)]; matched.append(j); dists.append(float(d.min()))
        keep = [i for i, d in enumerate(dists) if d <= a.caliper]
        mpos = [pos[i] for i in keep]; mneg = []
        for i in keep:
            rid, e, ln, lv, tr = cand[matched[i]]
            key = f"{rid}|{e}"
            mneg.append(dict(id=rid, key=key, len=ln, level=lv, trend=tr, oracle=float(nulls[rid]["q"][e - 1]),
                             cls=nulls[rid]["cls"], lab=nulls[rid]["lab"], **path_feats(nulls[rid]["f"][BURN:e])))
            seg_jobs[key] = nulls[rid]["f"][BURN:e]
        # original unmatched design (emulator_study replication: time-matched random ends, ORIG nulls)
        ends = [r["tip"] - L for r in tw]
        oneg = []
        for lab in ("ORIG_AWAY", "ORIG_STOPSHORT", "ORIG_STAT"):
            for rid, r in nulls.items():
                if r["lab"] != lab or not (r["tip"] is None or r["tip"] > max(ends)):
                    continue
                e = int(rng.choice(ends)); x = r["f"][BURN:e]; lv, tr = cheap_feats(x)
                key = f"{rid}|{e}"
                oneg.append(dict(id=rid, key=key, len=len(x), level=lv, trend=tr, oracle=float(r["q"][e - 1]), cls=r["cls"],
                                 **path_feats(x)))
                seg_jobs[key] = x
        # match quality
        def smd(a_, b_):
            a_, b_ = np.asarray(a_), np.asarray(b_)
            return round(float((a_.mean() - b_.mean()) / np.sqrt((a_.var(ddof=1) + b_.var(ddof=1)) / 2 + 1e-12)), 3)
        mq = dict(n_toward=len(pos), n_matched=len(keep), frac_matched=round(len(keep) / len(pos), 3),
                  median_dist=round(float(np.median([dists[i] for i in keep])), 3) if keep else None,
                  unique_null_segments=len({n["key"] for n in mneg}), unique_null_runs=len({n["id"] for n in mneg}),
                  null_class_mix={c: sum(n["cls"] == c for n in mneg) for c in ("STAT", "STOPSHORT", "AWAY")},
                  mean_abs_dlevel=round(float(np.mean([abs(p["level"] - n["level"]) for p, n in zip(mpos, mneg)])), 4) if keep else None,
                  mean_abs_dtrend=round(float(np.mean([abs(p["trend"] - n["trend"]) for p, n in zip(mpos, mneg)])), 4) if keep else None,
                  smd_unmatched_orig=dict(level=smd([p["level"] for p in pos], [n["level"] for n in oneg]),
                                          trend=smd([p["trend"] for p in pos], [n["trend"] for n in oneg])),
                  smd_matched=dict(level=smd([p["level"] for p in mpos], [n["level"] for n in mneg]),
                                   trend=smd([p["trend"] for p in mpos], [n["trend"] for n in mneg]),
                                   length=smd([p["len"] for p in mpos], [n["len"] for n in mneg])) if len(keep) > 2 else None,
                  unmatched_toward_level_range=[round(float(min(pos[i]["level"] for i in range(len(pos)) if i not in keep)), 3),
                                                round(float(max(pos[i]["level"] for i in range(len(pos)) if i not in keep)), 3)]
                  if len(keep) < len(pos) else None,
                  toward_level_median=round(float(np.median(Pf[:, 0])), 3), toward_trend_median=round(float(np.median(Pf[:, 1])), 3))
        design[L] = dict(pos=pos, mpos=mpos, mneg=mneg, oneg=oneg, mq=mq)
        print(f"[{name}] lead {L}: {mq}", flush=True)

    # ---- calibrated EWS for all needed segments (parallel)
    keys = list(seg_jobs)
    jobs = [(k, seg_jobs[k], N, a.M, i) for i, k in enumerate(keys)]
    feats = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for k, d in ex.map(_calib_job, jobs, chunksize=16):
            feats[k] = d
    print(f"[{name}] calibrated EWS on {len(keys)} segments ({time.time() - t0:.0f}s)", flush=True)

    # ---- evaluate
    res["matched"], res["unmatched_orig"], res["match_quality"] = {}, {}, {}
    for L, D in design.items():
        add = lambda lst: [{**s, **feats[s["key"]]} for s in lst]
        pos, mpos, mneg, oneg = add(D["pos"]), add(D["mpos"]), add(D["mneg"]), add(D["oneg"])
        res["match_quality"][f"lead{L}"] = D["mq"]
        res["unmatched_orig"][f"lead{L}"] = evaluate(pos, oneg, [p["id"] for p in pos], [n["id"] for n in oneg],
                                                     paired=False, B=a.B, seed=L)
        if len(mpos) >= 30:
            g = pair_groups(mneg)
            res["matched"][f"lead{L}"] = evaluate(mpos, mneg, g, g, paired=True, B=a.B, seed=L)
            res["match_quality"][f"lead{L}"]["n_cv_groups"] = len(set(g))
        for lab, R in (("unmatched_orig", res["unmatched_orig"]), ("matched", res["matched"])):
            if f"lead{L}" not in R:
                continue
            r = R[f"lead{L}"]; A = r["auroc"]; Dd = r["delta"]
            print(f"[{name}] lead {L:3d} {lab:15s} CV LTP {A['cv:LTP']['est']:.3f} +ews_cal {A['cv:LTP+ews_cal']['est']:.3f} | n={r['n_pos']}/{r['n_neg']} level {A['feat:level']['est']:.3f} trend "
                  f"{A['feat:trend']['est']:.3f} varz {A['feat:var_z']['est']:.3f} ac1z {A['feat:ac1_z']['est']:.3f} oracle "
                  f"{A['feat:oracle']['est']:.3f} | CV LT {A['cv:LT']['est']:.3f} +ews_cal {A['cv:LT+ews_cal']['est']:.3f} "
                  f"+ews_raw {A['cv:LT+ews_raw']['est']:.3f} ewsonly_cal {A['cv:ews_only_cal']['est']:.3f} | dVar_cal "
                  f"{Dd['var_cal']['est']:+.3f} {Dd['var_cal']['ci']} dAC1_cal {Dd['ac1_cal']['est']:+.3f} {Dd['ac1_cal']['ci']}", flush=True)
    res["H4"] = {f"{d}_{t}": h4_decision(res[d], t) for d in ("unmatched_orig", "matched") for t in ("raw", "cal", "cal05")}
    for k, v in res["H4"].items():
        print(f"[{name}] H4 {k:22s}: {v['decision']}", flush=True)
    res["runtime_s"] = round(time.time() - t0, 1)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    Path(a.out_dir, f"monitor_v2_{name}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"[{name}] done in {res['runtime_s']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy"); ap.add_argument("--best-response", action="store_true")
    ap.add_argument("--N", type=int, default=48); ap.add_argument("--T-ramp", type=int, default=300)
    ap.add_argument("--R-toward", type=int, default=300); ap.add_argument("--R-pool", type=int, default=100)
    ap.add_argument("--R-orig", type=int, default=200); ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--caliper", type=float, default=0.25); ap.add_argument("--len-caliper", type=int, default=20)
    ap.add_argument("--B", type=int, default=1000); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", default="results/emulator")
    main(ap.parse_args())
