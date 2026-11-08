"""M4 v4 (review #3): live-vs-extraction discrepancy with explicit estimand, population-cluster bootstrap, exact-shift
power analysis, and coverage validated by simulating whole experiments. Input: results/decisions/deepseek_decisions.csv.gz.

Estimand. For (pair, state) cells c with >= 2 live decisions and >= 2 extraction samples (eligibility depends only on
counts, never on outcomes), MS = sum_c w_c (pi_live,c - pi_ext,c)^2 with w_c = n_live,c / sum n_live (weights conditional
on the realised visitation). Estimator: per-cell unbiased (dbar^2 - s2_l/n_l - s2_e/n_e); report RMS = sqrt(max(MS,0)).
Uncertainty. Jackknife: delete-one-population (live) + delete-one-group (extraction, 20 random groups); Wald CI on MS.
A population bootstrap with replacement was tried first and FAILED coverage (0/40 at true MS = 0): duplicated
decisions bias the noise-corrected estimator upward.
Reference test. Homogeneity G statistic vs an independent-Bernoulli parametric bootstrap (cell sizes fixed). This is a
model-based reference that does not reproduce adaptive visitation or stopping; reported as exploratory.
Power. Exact absolute shifts: q_c = p_c +/- delta (sign random, flipped at the boundary), realised weighted |shift|
reported; null threshold from unperturbed simulations of the same cells; delta = 0 gives the false-positive rate.
Coverage. Simulated experiments: N=24 emulator populations with the real design (cm values, run counts, 150-round
horizon, stop rule) driven by a known live policy P_live = clip(P_ext + perturbation of known RMS); extraction samples
drawn with the real per-state counts from P_ext; the same estimator + population bootstrap; coverage of the true MS.
Plug-in sensitivity. Emulator flip curve with extraction-only vs hybrid (live where n_live >= 5) policies: delta p_c and
max |delta phi| over c = 1..8 (post hoc; not a bound on forecast error).
"""
from __future__ import annotations
import csv, gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.emulator import simulate, t50  # noqa: E402

rng = np.random.default_rng(0)
CSV = Path("results/decisions/deepseek_decisions.csv.gz")
live_runs = defaultdict(list)        # run_id -> list of (pair, state, choice)
ext = defaultdict(list)              # (pair, state) -> list of choices
with gzip.open(CSV, "rt") as fh:
    for r in csv.DictReader(fh):
        c = (r["pair"], int(r["state"]))
        if r["source"] == "extraction":
            ext[c].append(int(r["choice_B"]))
        else:
            live_runs[(r["source"], r["pair"], r["cm"], r["seed"])].append((r["pair"], int(r["state"]), int(r["choice_B"])))
run_ids = list(live_runs); ext = {k: np.array(v) for k, v in ext.items()}
print(f"live populations {len(run_ids)}, live decisions {sum(len(v) for v in live_runs.values())}, extraction cells {len(ext)}")


def live_cells(ids):
    L = defaultdict(list)
    for rid in ids:
        for p, s, ch in live_runs[rid]:
            L[(p, s)].append(ch)
    return {k: np.array(v) for k, v in L.items()}


def ms_estimate(L, E):
    ks = [k for k in L if len(L[k]) >= 2 and len(E.get(k, [])) >= 2]
    nl = np.array([len(L[k]) for k in ks], float); w = nl / nl.sum()
    sq = np.array([(L[k].mean() - E[k].mean()) ** 2 - L[k].var(ddof=1) / len(L[k]) - E[k].var(ddof=1) / len(E[k]) for k in ks])
    signed = np.array([L[k].mean() - E[k].mean() for k in ks])
    return float(np.sum(w * sq)), float(np.sum(w * signed)), len(ks), int(nl.sum())


def G(pairs):
    g = 0.0
    for l, e in pairs:
        n1, n2, a, b = len(l), len(e), l.sum(), e.sum(); p = (a + b) / (n1 + n2)
        for x, n in ((a, n1), (b, n2)):
            for o, ex in ((x, n * p), (n - x, n * (1 - p))):
                if o > 0 and ex > 0: g += 2 * o * np.log(o / ex)
    return g


L_all = live_cells(run_ids)
ms, signed, ncell, ndec = ms_estimate(L_all, ext)
def jackknife_se(Lfun, ids, E, G=20, seed=0):
    """Delete-one-population jackknife (live side) + delete-one-group jackknife over extraction samples (G groups);
    no resampling with replacement (duplicated decisions bias the noise-corrected estimator). Variances added."""
    n = len(ids); th_l = []
    for i in range(n):
        th_l.append(ms_estimate(Lfun([x for j, x in enumerate(ids) if j != i]), E)[0])
    th_l = np.array(th_l); v_l = (n - 1) / n * np.sum((th_l - th_l.mean()) ** 2)
    r = np.random.default_rng(seed); grp = {k: r.integers(0, G, len(v)) for k, v in E.items()}; th_e = []
    Lall = Lfun(ids)
    for g in range(G):
        Eg = {k: v[grp[k] != g] for k, v in E.items()}
        th_e.append(ms_estimate(Lall, {k: v for k, v in Eg.items() if len(v)})[0])
    th_e = np.array(th_e); v_e = (G - 1) / G * np.sum((th_e - th_e.mean()) ** 2)
    return float(np.sqrt(v_l + v_e))
se = jackknife_se(live_cells, run_ids, ext)
rms = lambda x: float(np.sqrt(max(x, 0)))
res = {"estimand": "decision-weighted mean squared difference over (pair,state) cells with >=2 obs per side",
       "cells": ncell, "live_decisions": ndec, "MS": ms, "MS_se_jackknife": se, "RMS": rms(ms), "signed_mean_diff": signed,
       "MS_ci95": [ms - 1.96 * se, ms + 1.96 * se], "RMS_upper95": rms(ms + 1.96 * se)}
print("estimate:", {k: res[k] for k in ("cells", "live_decisions", "MS", "MS_se_jackknife", "RMS", "RMS_upper95", "signed_mean_diff")}, flush=True)

# --- reference test + exact-shift power (cells with >= 5 per side, all cells incl. zero-variation ones)
cells5 = [(L_all[k], ext[k]) for k in L_all if len(L_all[k]) >= 5 and len(ext.get(k, [])) >= 5]
pool = [((l.sum() + e.sum()) / (len(l) + len(e)), len(l), len(e)) for l, e in cells5]
def sim_null():
    return [((rng.random(nl) < p).astype(int), (rng.random(ne) < p).astype(int)) for p, nl, ne in pool]
gnull = np.array([G(sim_null()) for _ in range(500)]); thr = np.percentile(gnull, 95)
res["reference_test"] = dict(cells=len(cells5), G=G(cells5), p_independent_bernoulli=float((1 + np.sum(gnull >= G(cells5))) / 501))
res["power_exact_shift"] = {}
wl = np.array([nl for _, nl, _ in pool], float); wl /= wl.sum()
for delta in (0.0, 0.01, 0.02, 0.03, 0.05):
    rej, real = 0, []
    for _ in range(200):
        sim = []; shifts = []
        for p, nl, ne in pool:
            sgn = rng.choice([-1, 1]); q = p + sgn * delta
            if q < 0 or q > 1: q = p - sgn * delta
            q = min(max(q, 0.0), 1.0); shifts.append(abs(q - p))
            sim.append(((rng.random(nl) < q).astype(int), (rng.random(ne) < p).astype(int)))
        rej += G(sim) >= thr; real.append(float(np.sum(wl * np.array(shifts))))
    res["power_exact_shift"][str(delta)] = dict(reject_rate=rej / 200, realised_weighted_abs_shift=float(np.mean(real)))
print("reference test:", res["reference_test"]); print("power (delta 0 = false-positive rate):", res["power_exact_shift"])

# --- coverage by simulating whole experiments (state level, pairs pooled, real design)
pol = json.loads(Path("results/policies/deepseek-v4-flash.json").read_text())["policy"]
P_ext = np.array([pol["".join(str((i // 4 ** (4 - d)) % 4) for d in range(5))]["P_B"] or 0.0 for i in range(1024)])
ext_n = np.zeros(1024, int)
for (p, s), v in ext.items(): ext_n[s] += len(v)
design = [(int(cm), 1) for (src, pr, cm, seed) in run_ids]
def sim_experiment(P_live, seed):
    r = np.random.default_rng(seed); Lr = {}
    for i, (cm, _) in enumerate(design):
        com = np.zeros(24, bool); com[r.choice(24, cm, replace=False)] = True; st = np.zeros(24, np.int64); run = 0; recs = []
        for t in range(150):
            pB = P_live[st]; ch = np.where(com, 1, (r.random(24) < pB).astype(int))
            for a_ in np.where(~com)[0]: recs.append((int(st[a_]), int(ch[a_])))
            perm = r.permutation(24); pt = np.empty(24, int); pt[perm[0::2]] = perm[1::2]; pt[perm[1::2]] = perm[0::2]
            st = (st * 4 + 2 * ch + ch[pt]) % 1024
            run = run + 1 if ch[~com].mean() >= 0.9 else 0
            if run >= 5: break
        Lr[i] = recs
    E = {("ALL", s): (r.random(ext_n[s]) < P_ext[s]).astype(int) for s in range(1024) if ext_n[s] > 0}
    return Lr, E
def cells_from(Lr, ids):
    D = defaultdict(list)
    for i in ids:
        for s, ch in Lr[i]: D[("ALL", s)].append(ch)
    return {k: np.array(v) for k, v in D.items()}
res["coverage"] = {}
for target_rms in (0.0, 0.02):
    cover = []
    for e_ in range(60):
        pert = rng.normal(0, 1, 1024); P_live = np.clip(P_ext + target_rms * pert / np.sqrt(np.mean(pert ** 2)), 0, 1)
        Lr, E = sim_experiment(P_live, 10_000 + e_)
        Lc = cells_from(Lr, range(len(design)))
        ks = [k for k in Lc if len(Lc[k]) >= 2 and len(E.get(k, [])) >= 2]
        w = np.array([len(Lc[k]) for k in ks], float); w /= w.sum()
        true_ms = float(np.sum(w * np.array([(P_live[k[1]] - P_ext[k[1]]) ** 2 for k in ks])))
        est = ms_estimate(Lc, E)[0]
        se_ = jackknife_se(lambda ids: cells_from(Lr, ids), list(range(len(design))), E, G=20, seed=e_)
        cover.append(bool(est - 1.96 * se_ <= true_ms <= est + 1.96 * se_))
    k = int(np.sum(cover)); n = len(cover)
    from scipy.stats import beta
    res["coverage"][str(target_rms)] = dict(nominal=0.95, covered=f"{k}/{n}", rate=k / n,
                                            clopper_pearson_95=[float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0])
    print(f"coverage at true RMS {target_rms}:", res["coverage"][str(target_rms)], flush=True)

# --- plug-in sensitivity (full curve)
def curve(P, pair_policy):
    return np.array([np.mean([t50(simulate(pair_policy, 24, cm, max_rounds=150, seed=800 + s)["f"]) is not None for s in range(400)]) for cm in range(1, 9)])
def pc_of(c):
    for i in range(7):
        if c[i] < 0.5 <= c[i + 1]: return (i + 1 + (0.5 - c[i]) / (c[i + 1] - c[i] + 1e-12)) / 24
    return None
res["plugin_sensitivity"] = {}
for label, pair_keys in (("pooled", None), ("push_Q (MQ)", "MQ"), ("push_M (QM)", "QM")):
    def pol_from(cells_side):
        P = P_ext.copy(); cnt = defaultdict(lambda: [0, 0])
        for (p, s), v in cells_side.items():
            if pair_keys and p != pair_keys: continue
            cnt[s][0] += v.sum(); cnt[s][1] += len(v)
        return P, cnt
    Pe, ce = pol_from(ext); Pe = Pe.copy()
    for s, (b, n) in ce.items():
        if n: Pe[s] = b / n
    _, cl = pol_from(L_all); Pl = Pe.copy()
    for s, (b, n) in cl.items():
        if n >= 5: Pl[s] = b / n
    ce_ = curve(None, Pe); cl_ = curve(None, Pl)
    res["plugin_sensitivity"][label] = dict(pc_extraction=pc_of(ce_), pc_hybrid_live=pc_of(cl_),
                                            max_abs_dphi=float(np.max(np.abs(ce_ - cl_))), states_replaced=int(sum(n >= 5 for b, n in cl.values())))
print("plug-in sensitivity:", res["plugin_sensitivity"])
Path("results/live_analysis/m4_v4.json").write_text(json.dumps(res, indent=1, default=float))
