"""Study 1 (EXPLORATORY): does time-to-tip in LLM naming games follow saddle-node "ghost" scaling?

Data: Ashery et al. 2025 public runs (committed agents swapped in at t=0; fixed minority per run).
Time-to-tip T50 = first interaction at which the committed word reaches >= 50% of the last W non-committed
answers (W = 2N). Runs never reaching it are right-censored at their length.
Models fit to individual uncensored runs with cm >= cm_min (all such runs tip), log-normal errors:
  CRIT : T = A (p - p_c)^-gamma       (saddle-node ghost; diverges at p_c > 0)
  CRIT½: same, gamma fixed at 0.5     (mean-field saddle-node exponent)
  DIL  : T = A p^-gamma               (no threshold: bigger minority just pushes faster)
Model comparison by AIC; bootstrap (resample runs within cm) for p_c, gamma CIs.
Exploratory only: results here set hypotheses for confirmatory live runs, not conclusions.
"""
from __future__ import annotations
import glob, json, pickle, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RAW, OUT = Path("data/raw/ashery"), Path("results/study1_time_to_tip"); OUT.mkdir(parents=True, exist_ok=True)

class Safe(pickle.Unpickler):
    def find_class(self, m, n): raise pickle.UnpicklingError(f"blocked {m}.{n}")

def load(model_glob):
    rows = []
    for f in sorted(glob.glob(str(RAW / model_glob))):
        m = re.search(r"swap_0_(\d+)cmtd_(\w)_(\w)_.*_(\d+)mem_.*_(\d+)ps", Path(f).name)
        cm, w0, w1, mem, N = int(m[1]), m[2], m[3], int(m[4]), int(m[5])
        d = Safe(open(f, "rb")).load()
        for k, r in d.items():
            tr, word = r["tracker"], r["convergence"]["committed_to"]
            com = {p for p, v in r["simulation"].items() if v["committed_tag"]}
            ans, idx = [], []
            for t, ((p1, p2), (a1, a2)) in enumerate(zip(tr["players"], tr["answers"])):
                for p, a in ((p1, a1), (p2, a2)):
                    if p not in com: ans.append(a == word); idx.append(t)
            ans = np.array(ans, float); W = 2 * N
            roll = np.convolve(ans, np.ones(W) / W, mode="valid")
            hit = np.where(roll >= 0.5)[0]
            T, event = (idx[hit[0] + W - 1] + 1, 1) if len(hit) else (len(tr["outcome"]), 0)
            rows.append(dict(cm=cm, N=N, mem=mem, p=cm / N, order=f"{w0}{w1}", run=k, T=T, event=event))
    return rows

def km_median(T, E):
    T, E = np.asarray(T), np.asarray(E); order = np.argsort(T); S = 1.0; n = len(T)
    for i, j in enumerate(order):
        if E[j]: S *= 1 - 1 / (n - i)
        if S <= 0.5: return float(T[j])
    return float("inf")

def fit(p, T, kind, pc_max):
    y = np.log(T)
    if kind == "DIL":
        f = lambda th: th[0] - th[1] * np.log(p); th0 = [np.log(np.median(T)), 1.0]; lb, ub = [-np.inf, 0], [np.inf, 10]
    elif kind == "CRIT":
        f = lambda th: th[0] - th[2] * np.log(np.clip(p - th[1], 1e-9, None)); th0 = [np.log(np.median(T)), pc_max * 0.7, 0.5]
        lb, ub = [-np.inf, 0, 0], [np.inf, pc_max - 1e-6, 10]
    else:  # CRIT_half
        f = lambda th: th[0] - 0.5 * np.log(np.clip(p - th[1], 1e-9, None)); th0 = [np.log(np.median(T)), pc_max * 0.7]
        lb, ub = [-np.inf, 0], [np.inf, pc_max - 1e-6]
    r = least_squares(lambda th: f(th) - y, th0, bounds=(lb, ub))
    rss = float(np.sum(r.fun ** 2)); n, k = len(y), len(th0) + 1
    return dict(theta=r.x.tolist(), rss=rss, aic=n * np.log(rss / n) + 2 * k, pred=f)

def analyse(name, rows, cm_min):
    by = defaultdict(list)
    for r in rows: by[r["cm"]].append(r)
    table = {cm: dict(p=v[0]["p"], n=len(v), n_tip=sum(r["event"] for r in v),
                      km_median=km_median([r["T"] for r in v], [r["event"] for r in v])) for cm, v in sorted(by.items())}
    fitrows = [r for r in rows if r["cm"] >= cm_min and r["event"]]
    p = np.array([r["p"] for r in fitrows]); T = np.array([r["T"] for r in fitrows], float)
    pc_max = cm_min / rows[0]["N"]
    fits = {k: fit(p, T, k, pc_max) for k in ("CRIT", "CRIT_half", "DIL")}
    rng = np.random.default_rng(0); boot = {"pc": [], "gamma": [], "dAIC_DIL_minus_CRIThalf": []}
    groups = defaultdict(list)
    for r in fitrows: groups[r["cm"]].append(r)
    for _ in range(1000):
        s = [g[i] for g in groups.values() for i in rng.integers(0, len(g), len(g))]
        bp, bT = np.array([r["p"] for r in s]), np.array([r["T"] for r in s], float)
        c = fit(bp, bT, "CRIT", pc_max); boot["pc"].append(c["theta"][1]); boot["gamma"].append(c["theta"][2])
        boot["dAIC_DIL_minus_CRIThalf"].append(fit(bp, bT, "DIL", pc_max)["aic"] - fit(bp, bT, "CRIT_half", pc_max)["aic"])
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out = dict(model=name, cm_min=cm_min, n_fit_runs=len(fitrows), table=table,
               fits={k: dict(theta=v["theta"], aic=v["aic"]) for k, v in fits.items()},
               boot=dict(pc_ci=ci(boot["pc"]), gamma_ci=ci(boot["gamma"]),
                         dAIC_DIL_minus_CRIThalf_ci=ci(boot["dAIC_DIL_minus_CRIThalf"]),
                         frac_boot_CRIThalf_better=float(np.mean(np.array(boot["dAIC_DIL_minus_CRIThalf"]) > 0))))
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for r in rows:
        ax[0].scatter(r["p"], r["T"], s=14, alpha=.5, c="C0" if r["event"] else "C3", marker="o" if r["event"] else "^")
    pg = np.linspace(pc_max * 1.001, max(p) * 1.02, 200)
    for k, c in (("CRIT", "C1"), ("CRIT_half", "C2"), ("DIL", "k")):
        ax[0].plot(pg, np.exp(fits[k]["pred"](fits[k]["theta"]) if False else 0) if False else None) if False else None
        th = fits[k]["theta"]
        yy = (th[0] - th[1] * np.log(pg)) if k == "DIL" else (th[0] - (th[2] if k == "CRIT" else 0.5) * np.log(np.clip(pg - th[1], 1e-9, None)))
        ax[0].plot(pg, np.exp(yy), c=c, label=f"{k} AIC={fits[k]['aic']:.1f}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("committed fraction p"); ax[0].set_ylabel("time to tip T50 (interactions)")
    ax[0].set_title(f"{name}: red ^ = censored"); ax[0].legend(fontsize=7)
    th = fits["CRIT"]["theta"]
    ax[1].scatter(np.log(p - th[1]), np.log(T), s=12, alpha=.5)
    ax[1].set_xlabel("log(p - p_c_hat)"); ax[1].set_ylabel("log T50"); ax[1].set_title(f"slope = -gamma_hat = {-th[2]:.2f}")
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=150); plt.close(fig)
    return out

if __name__ == "__main__":
    res = {}
    claude = load("claude_data/*swap*24ps*.pkl")
    res["claude_N24"] = analyse("claude_N24", claude, cm_min=4)
    l31 = load("llama31_data/*swap*.pkl")
    for N in (24, 48):
        rows = [r for r in l31 if r["N"] == N]
        if rows:
            # cm_min = smallest cm where every run tips
            cms = sorted({r["cm"] for r in rows})
            ok = [c for c in cms if all(r["event"] for r in rows if r["cm"] == c)]
            if ok: res[f"llama31_N{N}"] = analyse(f"llama31_N{N}", rows, cm_min=min(ok))
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    for k, v in res.items():
        print(f"\n== {k}  (fit on cm>={v['cm_min']}, n={v['n_fit_runs']})")
        for cm, t in v["table"].items(): print(f"  cm={cm:2d} p={t['p']:.3f} n={t['n']:2d} tipped={t['n_tip']:2d} KM-median T={t['km_median']}")
        for m, f in v["fits"].items(): print(f"  {m:9s} AIC={f['aic']:8.2f} theta={np.round(f['theta'],3).tolist()}")
        print("  boot:", {kk: (np.round(vv, 3).tolist() if isinstance(vv, list) else round(vv, 3)) for kk, vv in v["boot"].items()})
