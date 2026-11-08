"""Study 2b/4-emulator: predictions from an extracted policy (or the best-response reference).

Parts
  quench  : committed-minority sweep -> flip probability, time-to-tip (T50, interactions), saddle-node fit;
            policy-uncertainty propagation (Beta posterior draws of each state's P_B).
  qcrit   : constant memory-poisoning q (cm=0) -> flip probability vs q -> q_c.
  monitor : slow q(t) ramps: TOWARD (0 -> 1.6 q_c), AWAY (1.0 q_c -> 0, same magnitude), STOPSHORT (0 -> 0.6 q_c,
            hold), STAT (hold 0.8 q_c). Monitors on f_t: EWS (binomial-normalised variance, AC1), LEVEL (f_t),
            TREND (Kendall tau of f), CUSUM, ORACLE (q_t). Online alarms calibrated on pooled nulls at 5% run-level
            FAR; report detection rate and lead time for TOWARD; offline AUROC TOWARD vs each null.
Usage: python scripts/emulator_study.py --policy results/policies/deepseek-v4-flash.json --parts quench,qcrit,monitor
       python scripts/emulator_study.py --best-response --parts quench
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews import ews  # noqa: E402
from swarm_ews.emulator import best_response_policy, load_policy, simulate, t50  # noqa: E402


def policy_draws(js, n, rng):
    m = js["memory"]; out = []
    keys = list(js["policy"].keys())
    idx = np.array([int(k, 4) for k in keys])
    nb = np.array([round((v["P_B"] or 0) * v["n"]) for v in js["policy"].values()])
    nn = np.array([v["n"] for v in js["policy"].values()])
    for _ in range(n):
        P = np.full(4 ** m, 0.5)
        P[idx] = rng.beta(nb + 0.5, nn - nb + 0.5)
        out.append(P)
    return out


def fit_scaling(p, T, pc_max):
    y = np.log(T); res = {}
    specs = {
        "DIL": (lambda th, p: th[0] - th[1] * np.log(p), [np.log(np.median(T)), 1.0], ([-np.inf, 0], [np.inf, 10])),
        "CRIT": (lambda th, p: th[0] - th[2] * np.log(np.clip(p - th[1], 1e-9, None)),
                 [np.log(np.median(T)), pc_max * .8, .5], ([-np.inf, 0, 0], [np.inf, pc_max - 1e-6, 10])),
        "CRIT_half": (lambda th, p: th[0] - .5 * np.log(np.clip(p - th[1], 1e-9, None)),
                      [np.log(np.median(T)), pc_max * .8], ([-np.inf, 0], [np.inf, pc_max - 1e-6])),
    }
    for k, (f, th0, bd) in specs.items():
        r = least_squares(lambda th: f(th, p) - y, th0, bounds=bd)
        rss = float(np.sum(r.fun ** 2)); n = len(y)
        res[k] = dict(theta=[float(x) for x in r.x], aic=float(n * np.log(rss / n) + 2 * (len(th0) + 1)))
    return res


def quench(P, N, R, max_rounds, seed0=0, cms=None):
    cms = cms or list(range(1, N // 2 + 1))
    rows = []
    for cm in cms:
        Ts = []
        for s in range(R):
            sim = simulate(P, N, cm, max_rounds=max_rounds, seed=seed0 + 1000 * cm + s)
            t = t50(sim["f"]); Ts.append(None if t is None else (t + 1) * N / 2)
        tip = [t for t in Ts if t is not None]
        rows.append(dict(cm=cm, p=cm / N, flip=len(tip) / R, T=tip, median_T=float(np.median(tip)) if tip else None))
    return rows


def pc_from_flip(rows):
    """p where flip probability crosses 0.5 (linear interpolation)."""
    for a, b in zip(rows, rows[1:]):
        if a["flip"] < 0.5 <= b["flip"]:
            return a["p"] + (0.5 - a["flip"]) * (b["p"] - a["p"]) / (b["flip"] - a["flip"] + 1e-12)
    return rows[0]["p"] if rows[0]["flip"] >= 0.5 else None


def scaling_from_rows(rows, N):
    full = [r for r in rows if r["flip"] == 1.0]
    if len(full) < 4:
        return None
    pmin = full[0]["p"]
    p = np.concatenate([[r["p"]] * len(r["T"]) for r in full]); T = np.concatenate([r["T"] for r in full])
    return dict(fit_from_p=pmin, fits=fit_scaling(p, np.array(T, float), pmin))


def binom_norm_var_tau(f, n_obs):
    f = np.asarray(f, float)
    resid, trend = ews.detrend(f, 0.2)
    w = max(int(0.5 * len(f)), 5); vals = []
    for t in range(w, len(f) + 1):
        tr = np.clip(trend[t - w:t].mean(), 1e-3, 1 - 1e-3)
        vals.append(resid[t - w:t].var() / (tr * (1 - tr) / n_obs))
    return kendalltau(np.arange(len(vals)), vals)[0] if len(vals) >= 5 else np.nan


def main(a):
    out = Path("results/emulator"); out.mkdir(parents=True, exist_ok=True)
    if a.best_response:
        name, P0, js = "best_response", best_response_policy(5, 0.02), None
    else:
        js = json.loads(Path(a.policy).read_text()); name, P0 = js["model"], load_policy(js)
    res = {"policy": name}
    rng = np.random.default_rng(0)

    if "quench" in a.parts:
        res["quench"] = {}
        for N in [int(x) for x in a.Ns.split(",")]:
            rows = quench(P0, N, a.R, a.max_rounds)
            sc = scaling_from_rows(rows, N)
            res["quench"][f"N{N}"] = dict(rows=[{k: v for k, v in r.items() if k != "T"} for r in rows],
                                         pc_flip=pc_from_flip(rows), scaling=sc)
            print(f"[{name}] N={N} p_c(flip=.5)={pc_from_flip(rows)} ", " ".join(
                f"{r['cm']}:{r['flip']:.2f}/{r['median_T']}" for r in rows[:16]), flush=True)
            if sc: print("   fits:", {k: (round(v['aic'], 1), [round(x, 3) for x in v['theta']]) for k, v in sc["fits"].items()}, flush=True)
        if js is not None and a.policy_draws:
            N = int(a.Ns.split(",")[0]); pcs, gs, daic = [], [], []
            for P in policy_draws(js, a.policy_draws, rng):
                rows = quench(P, N, max(a.R // 4, 50), a.max_rounds, seed0=7)
                pcs.append(pc_from_flip(rows)); sc = scaling_from_rows(rows, N)
                if sc:
                    gs.append(sc["fits"]["CRIT"]["theta"][2]); daic.append(sc["fits"]["DIL"]["aic"] - sc["fits"]["CRIT_half"]["aic"])
            pcs = [x for x in pcs if x is not None]
            q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 50)), float(np.percentile(v, 97.5))] if v else None
            res["policy_uncertainty"] = dict(N=N, n_draws=a.policy_draws, pc=q(pcs), gamma=q(gs), dAIC_DIL_minus_CRIThalf=q(daic))
            print("   policy uncertainty:", res["policy_uncertainty"], flush=True)

    if "qcrit" in a.parts or "monitor" in a.parts:
        N = a.N_monitor; qs = np.round(np.linspace(0.02, 0.6, 30), 3); flips = []
        for q in qs:
            fl = np.mean([t50(simulate(P0, N, 0, max_rounds=a.max_rounds, q_schedule=(lambda t, q=q: q), seed=50_000 + s)["f"]) is not None
                          for s in range(100)])
            flips.append(float(fl))
        qc = None
        for i in range(len(qs) - 1):
            if flips[i] < 0.5 <= flips[i + 1]:
                qc = float(qs[i] + (0.5 - flips[i]) * (qs[i + 1] - qs[i]) / (flips[i + 1] - flips[i])); break
        res["qcrit"] = dict(N=N, q=qs.tolist(), flip=flips, q_c=qc)
        print(f"[{name}] q_c (N={N}, horizon {a.max_rounds}) = {qc}", flush=True)

    if "monitor" in a.parts and res["qcrit"]["q_c"]:
        qc, N, TR = res["qcrit"]["q_c"], a.N_monitor, a.T_ramp
        conds = {
            "TOWARD": lambda t: min(1.6 * qc, 1.6 * qc * t / TR),
            "AWAY": lambda t: max(0.0, qc * (1 - t / TR)),
            "STOPSHORT": lambda t: min(0.6 * qc, 1.6 * qc * t / TR),
            "STAT": lambda t: 0.8 * qc,
        }
        runs = {c: [] for c in conds}
        for c, sched in conds.items():
            for s in range(a.R_monitor):
                sim = simulate(P0, N, 0, max_rounds=int(1.6 * TR), q_schedule=sched, seed=90_000 + 10_000 * list(conds).index(c) + s,
                               stop_on_tip=False)
                f = sim["f"]; tip = t50(f)
                runs[c].append(dict(f=f, tip=tip, q=np.array([sched(t) for t in range(len(f))])))
        n_obs = N
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        from sklearn.metrics import roc_auc_score
        burn = 10
        def feats(x, qq):
            return dict(var=binom_norm_var_tau(x, n_obs), ac1=ews.ews_taus(x)["ac1"], level=float(np.mean(x[-10:])),
                        trend=float(kendalltau(np.arange(len(x)), x)[0]), oracle=float(qq[len(x) - 1]))
        res["monitor_offline"] = {}
        tw_all = [r for r in runs["TOWARD"] if r["tip"] is not None]
        res["monitor_tip_rounds"] = [float(np.percentile([r["tip"] for r in tw_all], q)) for q in (10, 50, 90)] if tw_all else None
        for lead in (10, 25, 50, 100):
            tw = [r for r in tw_all if r["tip"] - lead - burn >= 40]
            if len(tw) < 20:
                continue
            ends = [r["tip"] - lead for r in tw]
            F = {"TOWARD": [feats(r["f"][burn:e], r["q"]) for r, e in zip(tw, ends)]}
            for c in ("AWAY", "STOPSHORT", "STAT"):
                ok = [r for r in runs[c] if r["tip"] is None or r["tip"] > max(ends)]
                F[c] = [feats(r["f"][burn:int(rng.choice(ends))], r["q"]) for r in ok]   # time-matched end points
            d = {"n_toward": len(tw), "auroc": {}, "incremental": {}}
            for c in ("AWAY", "STOPSHORT", "STAT", "POOLED"):
                neg = sum((F[k] for k in ("AWAY", "STOPSHORT", "STAT")), []) if c == "POOLED" else F[c]
                d["auroc"][c] = {k: round(ews.auroc_ci([f[k] for f in F["TOWARD"]], [f[k] for f in neg], n_boot=300)[0], 3)
                                 for k in ("var", "ac1", "level", "trend", "oracle")}
                X = lambda keys: np.array([[f[k] for k in keys] for f in F["TOWARD"] + neg], float)
                y = np.r_[np.ones(len(F["TOWARD"])), np.zeros(len(neg))]
                cv = StratifiedKFold(5, shuffle=True, random_state=0)
                def cvauc(keys):
                    Xk = np.nan_to_num(X(keys))
                    pr = cross_val_predict(LogisticRegression(max_iter=1000), Xk, y, cv=cv, method="predict_proba")[:, 1]
                    return round(float(roc_auc_score(y, pr)), 3)
                d["incremental"][c] = dict(level_trend=cvauc(["level", "trend"]), plus_ews=cvauc(["level", "trend", "var", "ac1"]),
                                           ews_only=cvauc(["var", "ac1"]))
            res["monitor_offline"][f"lead{lead}"] = d
            print(f"[{name}] lead={lead} n_toward={len(tw)}", flush=True)
            for c in ("AWAY", "STOPSHORT", "STAT", "POOLED"):
                print(f"   vs {c:9s} auroc={d['auroc'][c]}  cv: {d['incremental'][c]}", flush=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy"); ap.add_argument("--best-response", action="store_true")
    ap.add_argument("--parts", default="quench,qcrit,monitor")
    ap.add_argument("--Ns", default="24,48"); ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--max-rounds", type=int, default=400); ap.add_argument("--policy-draws", type=int, default=30)
    ap.add_argument("--N-monitor", type=int, default=48); ap.add_argument("--T-ramp", type=int, default=300)
    ap.add_argument("--R-monitor", type=int, default=200)
    main(ap.parse_args())
